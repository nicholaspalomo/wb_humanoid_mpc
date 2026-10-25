# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Unit tests for Angular Center of Mass (aCOM) JAX pipeline."""

import inspect
import json
import os
import re
import tempfile
import unittest
from xml.etree import ElementTree

import google.protobuf
from humanoid_mpc_config import task_file_pb2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax

from humanoid_learning.acom import dataset_generator
from humanoid_learning.acom import export_acom
from humanoid_learning.acom import models
from humanoid_learning.acom import train_acom
from humanoid_learning.acom import train_main
import nproto_textproto
from tools.hooks import pointer_nullability

# The notebook behind `make train-acom-jupyter`, run end to end by TestTrainingNotebook.
_NOTEBOOK_PATH = "notebooks/train_acom_siren.ipynb"


def _find_robot_model(relative_path: str) -> str | None:
    """Resolve a robot model path from workspace or Bazel runfiles."""
    resolved = dataset_generator.resolve_xml_path(relative_path)
    if os.path.exists(resolved):
        return resolved
    return None


def _load_task_file(robot: str) -> task_file_pb2.TaskFile:
    """The robot's centroidal MPC task.textproto, parsed strictly."""
    task_path = dataset_generator.resolve_xml_path(
        train_main._ROBOT_CONFIGS[robot]["task"]
    )
    return nproto_textproto.load_textproto(task_path, task_file_pb2.TaskFile)


def _initial_joint_state(
    task: task_file_pb2.TaskFile, joint_names: list[str]
) -> np.ndarray:
    """The joints of the task file's initial_state, i.e. the MPC's nominal posture, in the order of `joint_names`."""
    positions = {
        entry.joint: entry.value for entry in task.initial_state.joint_positions
    }
    return np.array([float(positions[name]) for name in joint_names])


def _urdf_joint_limits(urdf_path: str) -> dict:
    """Joint name -> (lower, upper) for every URDF joint that declares limits."""
    limits = {}
    for joint in ElementTree.parse(urdf_path).getroot().findall("joint"):
        limit = joint.find("limit")
        if (
            limit is not None
            and "lower" in limit.attrib
            and "upper" in limit.attrib
        ):
            limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    return limits


def _header_joint_names(content: str) -> list:
    """The joint_names[] a generated weight header records, in order."""
    match = re.search(r"joint_names\[\d+\] = \{([^}]*)\}", content)
    return re.findall(r'"([^"]+)"', match.group(1)) if match else []


def _quaternion_rate_matrix(xi: jnp.ndarray) -> jnp.ndarray:
    """G(xi), with xi (x) (omega, 0) = G(xi) omega for quaternion coefficients (x, y, z, w)."""
    return jnp.array(
        [
            [xi[3], -xi[2], xi[1]],
            [xi[2], xi[3], -xi[0]],
            [-xi[1], xi[0], xi[3]],
            [-xi[0], -xi[1], -xi[2]],
        ]
    )


def _quaternion_conjugate(xi: jnp.ndarray) -> jnp.ndarray:
    """The conjugate of quaternion coefficients (x, y, z, w), the inverse of a unit quaternion."""
    return jnp.concatenate([-xi[:3], xi[3:]])


def _world_rate_map(
    attitude_derivative: jnp.ndarray, xi: jnp.ndarray
) -> jnp.ndarray:
    """The world angular velocity per unit rate of each column of d(xi)/dq, from omega_W = 2 vec(xi_dot (x) xi^-1).

    Args:
        attitude_derivative: The derivative of the unit quaternion xi, shape (4, n).
        xi: The unit quaternion, coefficients (x, y, z, w).

    Returns:
        The (3, n) map from the n rates to the world angular velocity.
    """
    inverse = _quaternion_conjugate(xi)
    return jax.vmap(
        lambda column: 2.0 * models.quaternion_product(column, inverse)[:3],
        in_axes=1,
        out_axes=1,
    )(attitude_derivative)


def _with_large_offsets(params: models.SirenParams) -> models.SirenParams:
    """`params` with the readout scaled up and biased, so Delta_theta is about a radian, as on the shipped robots."""
    w_out, b_out = params[-1]
    return params[:-1] + [(20.0 * w_out, b_out + jnp.array([0.5, -0.7, 0.4]))]


def _relative_rate_error(
    model: models.SirenACOM,
    params: models.SirenParams,
    q_samples: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Mean ||J_l(Delta_theta) J_Delta_theta - Abar||_F / ||Abar||_F over the samples, as the C++ acceptance test grades."""
    rate_jacobians = jax.vmap(
        model.acom_joint_rate_jacobian, in_axes=(None, 0)
    )(params, jnp.asarray(q_samples))
    errors = np.linalg.norm(np.asarray(rate_jacobians) - targets, axis=(1, 2))
    return float(np.mean(errors / np.linalg.norm(targets, axis=(1, 2))))


def _run_notebook(path: str, overrides: dict) -> dict:
    """Executes a notebook's code cells in order in one namespace, the way Run All would.

    Every cell must be plain Python: an IPython magic does not compile, which is
    the point - the notebook stays runnable outside Jupyter.

    Args:
        path: The notebook file.
        overrides: Values that replace the ones the cell starting with
            "# Parameters" sets, right after it runs.

    Returns:
        The namespace after the last cell.
    """
    with open(path, "r", encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(  # pylint: disable=exec-used  # Runs the shipped notebook's cells, as Jupyter's Run All would.
            compile(source, f"{path} cell {index}", "exec"), namespace
        )
        if source.lstrip().startswith("# Parameters"):
            namespace.update(overrides)
    return namespace


class TestTheProtobufRuntime(unittest.TestCase):
    def test_it_is_the_one_the_config_gencode_was_made_for(self):
        # The task file is read through the configuration schemas' generated modules, linked against the protobuf of the
        # operator tools, while this package's hub brings its own: both must be the same release, or
        # the first on the path would refuse the other's gencode.
        source = inspect.getsource(task_file_pb2)
        match = re.search(
            r"ValidateProtobufRuntimeVersion\(\s*_runtime_version\.Domain\.PUBLIC,\s*(\d+),\s*(\d+),\s*(\d+)",
            source,
        )
        assert match is not None, "the gencode names no runtime version"
        self.assertEqual(google.protobuf.__version__, ".".join(match.groups()))


class TestAcomPipeline(unittest.TestCase):
    """Tests SIREN model forward pass, Jacobians, training convergence, and export."""

    # pylint: disable=invalid-name  # A_bar: the notation of Chen et al., IROS 2023 (humanoid_learning/acom/README.md).

    def setUp(self):
        self.in_dim = 6
        self.model = models.SirenACOM(
            in_dim=self.in_dim,
            hidden_dim=32,
            num_layers=2,
            out_dim=3,
            omega_0=30.0,
        )
        self.key = jax.random.PRNGKey(123)
        self.params = self.model.init_params(self.key)

    def test_forward_and_jacobian_shapes(self):
        """Verifies forward pass and Jacobian output dimensions."""
        q_j = jnp.zeros((self.in_dim,))
        delta_theta = self.model.forward(self.params, q_j)
        self.assertEqual(delta_theta.shape, (3,))

        jac = self.model.jacobian_qj(self.params, q_j)
        self.assertEqual(jac.shape, (3, self.in_dim))

        # The ACoM attitude: a unit quaternion (x, y, z, w).
        xi = models.quaternion_exp(jnp.array([0.2, -0.4, 1.1]))
        attitude = self.model.acom_attitude(self.params, xi, q_j)
        self.assertEqual(attitude.shape, (4,))
        np.testing.assert_allclose(jnp.linalg.norm(attitude), 1.0, atol=1e-6)

        # Its tangent Jacobian: columns (body rotation of the base, joints).
        tangent = self.model.acom_attitude_tangent_jacobian(self.params, q_j)
        self.assertEqual(tangent.shape, (3, 3 + self.in_dim))

    def test_acom_attitude_composes_the_base_with_the_joint_rotation(self):
        """xi_a = xi_hat (x) Exp(Delta_theta), so a rotation of the whole robot rotates the ACoM one for one."""
        q_j = jnp.linspace(-0.8, 0.9, self.in_dim)
        delta_theta = self.model.forward(self.params, q_j)
        identity = jnp.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            self.model.acom_attitude(self.params, identity, q_j),
            models.quaternion_exp(delta_theta),
            atol=1e-6,
        )
        xi = models.quaternion_exp(jnp.array([0.3, 0.1, -0.7]))
        turn = models.quaternion_exp(jnp.array([-0.5, 0.2, 2.4]))
        np.testing.assert_allclose(
            self.model.acom_attitude(
                self.params, models.quaternion_product(turn, xi), q_j
            ),
            models.quaternion_product(
                turn, self.model.acom_attitude(self.params, xi, q_j)
            ),
            atol=1e-6,
        )
        # The scale of xi does not matter.
        np.testing.assert_allclose(
            self.model.acom_attitude(self.params, 3.0 * xi, q_j),
            self.model.acom_attitude(self.params, xi, q_j),
            atol=1e-6,
        )

    def test_acom_attitude_tangent_jacobian_is_the_derivative_of_the_attitude(
        self,
    ):
        """The two blocks are R(Delta_theta)^T and J_r(Delta_theta) J_Delta_theta, the derivative of the attitude.

        A body perturbation dphi_a of the attitude moves its coefficients by
        1/2 G(xi_a) dphi_a, so jax.jacobian of the attitude with respect to a
        body rotation of the base, and with respect to the joints, is
        1/2 G(xi_a) times each block. The base block is a rotation, not the
        identity of the additive Euler rule.
        """
        q_j = jnp.linspace(-0.8, 0.9, self.in_dim)
        xi = models.quaternion_exp(jnp.array([0.3, 0.1, -0.7]))
        attitude = self.model.acom_attitude(self.params, xi, q_j)
        tangent = self.model.acom_attitude_tangent_jacobian(self.params, q_j)
        rate_matrix = 0.5 * _quaternion_rate_matrix(attitude)

        def attitude_of_rotated_base(dphi: jnp.ndarray) -> jnp.ndarray:
            rotated = models.quaternion_product(xi, models.quaternion_exp(dphi))
            return self.model.acom_attitude(self.params, rotated, q_j)

        base_derivative = jax.jacobian(attitude_of_rotated_base)(
            jnp.zeros((3,))
        )
        np.testing.assert_allclose(
            base_derivative, jnp.dot(rate_matrix, tangent[:, :3]), atol=1e-5
        )
        joint_derivative = jax.jacobian(
            lambda joints: self.model.acom_attitude(self.params, xi, joints)
        )(q_j)
        np.testing.assert_allclose(
            joint_derivative, jnp.dot(rate_matrix, tangent[:, 3:]), atol=1e-4
        )

        delta_theta = self.model.forward(self.params, q_j)
        np.testing.assert_allclose(
            tangent[:, :3],
            models.quaternion_rotation_matrix(
                models.quaternion_exp(delta_theta)
            ).T,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            jnp.dot(tangent[:, :3].T, tangent[:, :3]), np.eye(3), atol=1e-5
        )
        self.assertGreater(
            float(jnp.max(jnp.abs(tangent[:, :3] - jnp.eye(3)))), 1e-4
        )

    def test_so3_right_jacobian_matches_the_exponential(self):
        """Exp(phi + d) = Exp(phi) (x) Exp(J_r(phi) d) to first order, on both sides of the series threshold."""
        for phi in (
            jnp.array([0.02, -0.03, 0.01]),
            jnp.array([0.4, -1.1, 0.7]),
        ):
            rate_matrix = 0.5 * _quaternion_rate_matrix(
                models.quaternion_exp(phi)
            )
            derivative = jax.jacobian(models.quaternion_exp)(phi)
            np.testing.assert_allclose(
                derivative,
                jnp.dot(rate_matrix, models.so3_right_jacobian(phi)),
                atol=1e-5,
            )

    def test_so3_left_jacobian_matches_the_exponential(self):
        """Exp(phi + d) = Exp(J_l(phi) d) (x) Exp(phi) to first order, and J_l(phi) = R(Exp(phi)) J_r(phi)."""
        for phi in (
            jnp.array([0.02, -0.03, 0.01]),
            jnp.array([0.4, -1.1, 0.7]),
        ):
            exp_phi = models.quaternion_exp(phi)

            def left_increment(d, phi=phi, exp_phi=exp_phi):
                # 2 vec(Exp(phi + d) (x) Exp(phi)^-1), whose derivative at d = 0 is the left Jacobian.
                return (
                    2.0
                    * models.quaternion_product(
                        models.quaternion_exp(phi + d),
                        _quaternion_conjugate(exp_phi),
                    )[:3]
                )

            left = models.so3_left_jacobian(phi)
            np.testing.assert_allclose(
                jax.jacobian(left_increment)(jnp.zeros((3,))), left, atol=1e-5
            )
            np.testing.assert_allclose(
                left,
                jnp.dot(
                    models.quaternion_rotation_matrix(exp_phi),
                    models.so3_right_jacobian(phi),
                ),
                atol=1e-6,
            )

    def test_joint_rate_jacobian_is_the_world_rate_of_the_attitude(self):
        """R(xi) J_l(Delta_theta) J_Delta_theta is the world angular velocity of xi (x) Exp(Delta_theta) per joint rate.

        Differentiated straight from the attitude, independently of the
        tangent Jacobian, and at offsets of about a radian, where J_l is far
        from the identity: the quantity training fits is the one the runtime's
        composition turns joint velocities into, and J_Delta_theta alone is not.
        """
        params = _with_large_offsets(self.params)
        q_j = jnp.linspace(-0.8, 0.9, self.in_dim)
        xi = models.quaternion_exp(jnp.array([0.3, 0.1, -0.7]))
        attitude = self.model.acom_attitude(params, xi, q_j)
        attitude_derivative = jax.jacobian(
            lambda joints: self.model.acom_attitude(params, xi, joints)
        )(q_j)
        world_rate = _world_rate_map(attitude_derivative, attitude)

        rate_jacobian = self.model.acom_joint_rate_jacobian(params, q_j)
        base_rotation = models.quaternion_rotation_matrix(xi)
        np.testing.assert_allclose(
            world_rate, jnp.dot(base_rotation, rate_jacobian), atol=1e-4
        )
        # The tangent Jacobian's joint block, carried from the attitude's axes into the world's, is the same map.
        tangent = self.model.acom_attitude_tangent_jacobian(params, q_j)
        np.testing.assert_allclose(
            jnp.dot(
                models.quaternion_rotation_matrix(attitude), tangent[:, 3:]
            ),
            jnp.dot(base_rotation, rate_jacobian),
            atol=1e-4,
        )
        # Not vacuous: at these offsets J_Delta_theta alone is a different map.
        offset_angle = float(jnp.linalg.norm(self.model.forward(params, q_j)))
        self.assertGreater(offset_angle, 0.5)
        jacobian = self.model.jacobian_qj(params, q_j)
        self.assertGreater(
            float(
                jnp.linalg.norm(rate_jacobian - jacobian)
                / jnp.linalg.norm(jacobian)
            ),
            0.1,
        )

    def test_training_loss_is_the_composed_rate_map_error(self):
        """The Frobenius term train_acom minimizes is ||J_l(Delta_theta) J_Delta_theta - Abar||_F^2, not that of J alone."""
        params = _with_large_offsets(self.params)
        rng = np.random.default_rng(5)
        q_batch = jnp.asarray(rng.uniform(-1.0, 1.0, size=(16, self.in_dim)))
        targets = jnp.asarray(rng.standard_normal((16, 3, self.in_dim)))
        _, eval_step = train_acom.create_train_step(
            self.model, optax.adam(1e-3), reg_weight=0.0
        )
        _, aux = eval_step(params, q_batch, targets)

        rate_jacobians = jax.vmap(
            self.model.acom_joint_rate_jacobian, in_axes=(None, 0)
        )(params, q_batch)
        jacobians = jax.vmap(self.model.jacobian_qj, in_axes=(None, 0))(
            params, q_batch
        )
        np.testing.assert_allclose(
            aux["frob_loss"],
            jnp.mean(
                jnp.sum(jnp.square(rate_jacobians - targets), axis=(1, 2))
            ),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            aux["jacobian_frob"],
            jnp.mean(jnp.sum(jnp.square(jacobians - targets), axis=(1, 2))),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            aux["relative_errors"],
            jnp.linalg.norm(rate_jacobians - targets, axis=(1, 2))
            / jnp.linalg.norm(targets, axis=(1, 2)),
            rtol=1e-5,
        )

    def test_training_convergence_on_synthetic_data(self):
        """Verifies that JAX training converges on a synthetic linear/sinusoidal CMM."""
        num_samples = 200
        rng = np.random.default_rng(42)
        q_samples = rng.uniform(
            -1.0, 1.0, size=(num_samples, self.in_dim)
        ).astype(np.float32)

        # Synthetic target Jacobian: J(q) = W_true + 0.1 * sin(q)
        w_true = rng.standard_normal((3, self.in_dim)).astype(np.float32)
        A_bar_samples = np.stack(
            [w_true + 0.1 * np.sin(q) for q in q_samples], axis=0
        )

        dataset = {
            "q_joints": q_samples,
            "A_bar_omega": A_bar_samples,
        }

        _, _, history = train_acom.train_acom(
            dataset=dataset,
            in_dim=self.in_dim,
            hidden_dim=32,
            num_layers=2,
            num_epochs=15,
            batch_size=64,
            learning_rate=5e-3,
            verbose=False,
        )

        # Loss should decrease across epochs
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])

    def test_tensorboard_logging(self):
        """Verifies that TensorBoard SummaryWriter logs scalars, histograms, and figures."""
        num_samples = 100
        rng = np.random.default_rng(42)
        q_samples = rng.uniform(
            -1.0, 1.0, size=(num_samples, self.in_dim)
        ).astype(np.float32)
        w_true = rng.standard_normal((3, self.in_dim)).astype(np.float32)
        A_bar_samples = np.stack(
            [w_true + 0.1 * np.sin(q) for q in q_samples], axis=0
        )
        dataset = {
            "q_joints": q_samples,
            "A_bar_omega": A_bar_samples,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "tb_logs")
            _, _, history = train_acom.train_acom(
                dataset=dataset,
                in_dim=self.in_dim,
                hidden_dim=16,
                num_layers=2,
                num_epochs=3,
                batch_size=32,
                learning_rate=1e-3,
                verbose=False,
                log_dir=log_dir,
                log_histograms=True,
                histogram_freq=1,
                log_figures=True,
            )

            self.assertTrue(os.path.exists(log_dir))
            # Verify event file was generated
            event_files = [
                f for f in os.listdir(log_dir) if "events.out.tfevents" in f
            ]
            self.assertGreater(len(event_files), 0)
            self.assertIn("val_rmse", history)
            self.assertIn("train_frob", history)
            self.assertIn("grad_norm", history)
            self.assertGreater(history["grad_norm"][0], 0.0)

    def test_export_utilities(self):
        """Verifies JSON and C++ header generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_acom.json")
            cpp_path = os.path.join(tmpdir, "TestAcomWeights.h")

            export_acom.export_to_json(self.params, json_path)
            export_acom.export_to_cpp_header(
                self.params, cpp_path, class_name="TestWeights"
            )

            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(cpp_path))

            with open(cpp_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("struct TestWeights", content)
                self.assertIn("static constexpr size_t input_dim = 6;", content)
            # A header without joint names has no pointer, and so needs no nullability include either.
            self.assertEqual(
                pointer_nullability.check_source(content, cpp_path), []
            )
            self.assertNotIn("absl/base/nullability.h", content)

    def test_export_round_trip_parity(self):
        """Verifies that exported C++ header weights match the original JAX parameters.

        Parses the generated header back to verify that:
        1. The structural constants (input_dim, output_dim, num_layers, omega_0) are correct.
        2. The flattened weight arrays have the expected number of elements.
        3. The serialized weight values match the original JAX params to 10 decimal places.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "TestParity.h")
            export_acom.export_to_cpp_header(
                self.params, cpp_path, class_name="TestParity", omega_0=30.0
            )

            with open(cpp_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verify structural metadata
            self.assertIn(
                f"static constexpr size_t input_dim = {self.in_dim};", content
            )
            self.assertIn("static constexpr size_t output_dim = 3;", content)
            num_layers_total = len(self.params)  # 2 hidden + 1 output = 3
            self.assertIn(
                f"static constexpr size_t num_layers = {num_layers_total};",
                content,
            )
            self.assertIn("static constexpr double omega_0 = 30.0;", content)

            # Verify each layer's weight array length
            for idx, (w, b) in enumerate(self.params):
                w_np = np.array(w)
                b_np = np.array(b)
                out_dim, in_dim = w_np.shape

                self.assertIn(
                    f"static constexpr size_t W{idx}_rows = {out_dim};", content
                )
                self.assertIn(
                    f"static constexpr size_t W{idx}_cols = {in_dim};", content
                )

                # Extract the weight array from the header and verify element count
                w_pattern = rf"W{idx}\[{out_dim * in_dim}\] = \{{([^}}]+)\}}"
                w_match = re.search(w_pattern, content)
                if w_match is None:
                    self.fail(f"Could not find W{idx} array in header")
                w_values = [
                    float(v.strip()) for v in w_match.group(1).split(",")
                ]
                self.assertEqual(len(w_values), out_dim * in_dim)

                # Verify numerical parity of weights (10 decimal places in header)
                np.testing.assert_allclose(
                    w_values,
                    w_np.flatten(),
                    atol=1e-9,
                    err_msg=f"Weight mismatch in layer {idx}",
                )

                # Verify bias values
                b_pattern = rf"b{idx}\[{out_dim}\] = \{{([^}}]+)\}}"
                b_match = re.search(b_pattern, content)
                if b_match is None:
                    self.fail(f"Could not find b{idx} array in header")
                b_values = [
                    float(v.strip()) for v in b_match.group(1).split(",")
                ]
                np.testing.assert_allclose(
                    b_values,
                    b_np.flatten(),
                    atol=1e-9,
                    err_msg=f"Bias mismatch in layer {idx}",
                )

    def test_exported_num_layers_matches_cpp_loader(self):
        """Regression: the exported header must satisfy the C++ static_assert.

        AngularCenterOfMass::createFromStaticWeights hard-codes three parameter
        tuples, two sinusoidal layers plus a linear readout. SirenACOM counts only
        the sinusoidal layers, so an off-by-one here produces a header that fails
        to compile rather than one that misbehaves at runtime.
        """
        model = models.SirenACOM(
            in_dim=self.in_dim,
            hidden_dim=8,
            num_layers=train_main._CPP_SUPPORTED_NUM_LAYERS,
            out_dim=3,
        )
        params = model.init_params(jax.random.PRNGKey(0))
        self.assertEqual(len(params), train_main._CPP_SUPPORTED_NUM_LAYERS + 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "TestLayers.h")
            export_acom.export_to_cpp_header(
                params, cpp_path, class_name="TestLayers"
            )
            with open(cpp_path, "r", encoding="utf-8") as f:
                content = f.read()
        self.assertIn("static constexpr size_t num_layers = 3;", content)

    def test_library_defaults_build_a_loadable_network(self):
        """Direct users of the library get the C++ evaluator's layer count unless they ask otherwise.

        A default of 3 sine layers is how the old training notebook produced a
        4-layer header that the C++ static_assert rejects.
        """
        self.assertEqual(
            models.SirenACOM(in_dim=3).num_layers,
            train_main._CPP_SUPPORTED_NUM_LAYERS,
        )
        self.assertEqual(
            inspect.signature(train_acom.train_acom)
            .parameters["num_layers"]
            .default,
            train_main._CPP_SUPPORTED_NUM_LAYERS,
        )

    def test_export_records_joint_names(self):
        """The exported header must record the joint ordering it was trained on."""
        joint_names = [f"joint_{i}" for i in range(self.in_dim)]
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "TestNames.h")
            export_acom.export_to_cpp_header(
                self.params,
                cpp_path,
                class_name="TestNames",
                joint_names=joint_names,
            )
            with open(cpp_path, "r", encoding="utf-8") as f:
                content = f.read()

        self.assertIn(f"joint_names[{self.in_dim}]", content)
        for name in joint_names:
            self.assertIn(f'"{name}"', content)
        # The generated header is held to the repository's pointer rules through its generator (AGENTS.md): the
        # lint does not read generated files.
        self.assertEqual(
            pointer_nullability.check_source(content, cpp_path), []
        )

        # A joint list that does not describe the network must be rejected.
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                export_acom.export_to_cpp_header(
                    self.params,
                    os.path.join(tmpdir, "Bad.h"),
                    joint_names=joint_names[:-1],
                )


class TestComposedObjective(unittest.TestCase):
    """Training fits the rate map of the composition, on a target with large offsets where that matters."""

    # pylint: disable=invalid-name  # A_bar: the notation of Chen et al., IROS 2023 (humanoid_learning/acom/README.md).

    # An exactly integrable target: the connection of the attitude Exp(M q), whose rate map is J_l(M q) M. Over the
    # unit box its offsets average about 50 degrees, which is the regime of the shipped networks (20 to 35 degrees on
    # average, up to 80) and more.
    _OFFSET_MAP = np.array(
        [[0.9, -0.3, 0.4], [0.2, 0.8, -0.5], [-0.4, 0.3, 0.7]]
    )
    # The mean relative rate error a network that fits the composition reaches here (about 0.01 in practice).
    _RATE_TOLERANCE = 0.05
    _NUM_SAMPLES = 2000
    _EPOCHS = 60
    _HIDDEN_DIM = 32
    _BATCH_SIZE = 64
    _LEARNING_RATE = 2e-3

    def _dataset(self, seed: int, num_samples: int) -> dict[str, np.ndarray]:
        """Samples the synthetic target over the unit box."""
        rng = np.random.default_rng(seed)
        q_samples = rng.uniform(-1.0, 1.0, size=(num_samples, 3))
        targets = np.stack(
            [
                np.asarray(
                    models.so3_left_jacobian(jnp.asarray(self._OFFSET_MAP @ q))
                )
                @ self._OFFSET_MAP
                for q in q_samples
            ]
        )
        return {"q_joints": q_samples, "A_bar_omega": targets}

    def _train_on_the_jacobian_alone(
        self, model: models.SirenACOM, dataset: dict[str, np.ndarray]
    ) -> models.SirenParams:
        """Trains `model` with the objective J_Delta_theta ~ Abar, on train_acom's architecture and schedule."""
        q_train = jnp.asarray(dataset["q_joints"])
        A_train = jnp.asarray(dataset["A_bar_omega"])
        steps_per_epoch = q_train.shape[0] // self._BATCH_SIZE
        optimizer = optax.adamw(
            optax.cosine_decay_schedule(
                self._LEARNING_RATE, self._EPOCHS * steps_per_epoch, alpha=0.01
            ),
            weight_decay=1e-4,
        )
        params = model.init_params(jax.random.PRNGKey(42))
        opt_state = optimizer.init(params)

        def loss(params, q_batch, A_batch):
            jacobians = jax.vmap(model.jacobian_qj, in_axes=(None, 0))(
                params, q_batch
            )
            offsets = jax.vmap(model.forward, in_axes=(None, 0))(
                params, q_batch
            )
            return jnp.mean(
                jnp.sum(jnp.square(jacobians - A_batch), axis=(1, 2))
            ) + 1e-4 * jnp.mean(jnp.sum(jnp.square(offsets), axis=1))

        @jax.jit
        def step(params, opt_state, q_batch, A_batch):
            grads = jax.grad(loss)(params, q_batch, A_batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state

        key = jax.random.PRNGKey(1)
        for _ in range(self._EPOCHS):
            key, permutation_key = jax.random.split(key)
            order = jax.random.permutation(permutation_key, q_train.shape[0])
            for batch in range(steps_per_epoch):
                indices = order[
                    batch * self._BATCH_SIZE : (batch + 1) * self._BATCH_SIZE
                ]
                params, opt_state = step(
                    params, opt_state, q_train[indices], A_train[indices]
                )
        return params

    def test_the_composed_loss_reaches_the_rate_tolerance_where_the_plain_loss_does_not(
        self,
    ):
        validation = self._dataset(seed=11, num_samples=300)
        offsets_deg = np.degrees(
            np.linalg.norm(validation["q_joints"] @ self._OFFSET_MAP.T, axis=1)
        )
        self.assertGreater(
            float(np.mean(offsets_deg)), 40.0, "the offsets must be large"
        )

        dataset = self._dataset(seed=1, num_samples=self._NUM_SAMPLES)
        model, params, history = train_acom.train_acom(
            dataset=dataset,
            in_dim=3,
            hidden_dim=self._HIDDEN_DIM,
            num_epochs=self._EPOCHS,
            batch_size=self._BATCH_SIZE,
            learning_rate=self._LEARNING_RATE,
            verbose=False,
        )
        composed_error = _relative_rate_error(
            model, params, validation["q_joints"], validation["A_bar_omega"]
        )
        self.assertLess(composed_error, self._RATE_TOLERANCE)
        # The history reports the same quantity, on train_acom's own validation split.
        best_epoch = int(history["best_epoch"][0])
        self.assertLess(
            history["val_mean_relative_error"][best_epoch], self._RATE_TOLERANCE
        )

        plain_params = self._train_on_the_jacobian_alone(model, dataset)
        plain_error = _relative_rate_error(
            model,
            plain_params,
            validation["q_joints"],
            validation["A_bar_omega"],
        )
        self.assertGreater(plain_error, 2.0 * self._RATE_TOLERANCE)


class TestDatasetGenerator(unittest.TestCase):
    """Tests for the MuJoCo-based dataset generator."""

    # pylint: disable=invalid-name  # I_G, A_bar: the notation of Chen et al., IROS 2023 (humanoid_learning/acom/README.md).

    atlas_xml: str | None
    atlas_urdf: str | None
    g1_xml: str | None

    @classmethod
    def setUpClass(cls):
        """Find robot model files, skip if not available."""
        cls.atlas_xml = _find_robot_model(
            "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml"
        )
        cls.atlas_urdf = _find_robot_model(
            "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"
        )
        cls.g1_xml = _find_robot_model(
            "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"
        )

    def _require_atlas(self) -> str:
        """The Atlas MJCF; skips the test without it."""
        if self.atlas_xml is None:
            self.skipTest("Atlas XML model not found")
        return self.atlas_xml

    def _require_atlas_urdf(self) -> tuple[str, str]:
        """The Atlas MJCF and URDF; skips the test without them."""
        if self.atlas_xml is None or self.atlas_urdf is None:
            self.skipTest("Atlas XML or URDF model not found")
        return self.atlas_xml, self.atlas_urdf

    def _require_g1(self) -> str:
        """The G1 MJCF; skips the test without it."""
        if self.g1_xml is None:
            self.skipTest("G1 XML model not found")
        return self.g1_xml

    def test_generator_init_floating_base(self):
        """Verifies generator initializes correctly for a floating-base robot."""
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        self.assertEqual(gen.num_mj_joints, gen.nv - 6)
        self.assertGreater(gen.num_mj_joints, 0)
        self.assertEqual(len(gen.joint_limits_lower), gen.num_mj_joints)
        self.assertEqual(len(gen.joint_limits_upper), gen.num_mj_joints)

    def test_centroidal_matrices_nonzero(self):
        """Regression: Verifies that A_bar_omega is NOT all zeros.

        Previously, mj_subtreeVel() was missing, causing subtree_angmom to remain
        zero. The resulting A_bar_omega was identically zero, making the trained
        network output zero for all inputs.
        """
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        q_zero = np.zeros(gen.num_mj_joints)
        I_G, A_omega_j, A_bar = gen.compute_centroidal_matrices(q_zero)

        # I_G should be a positive-definite 3x3 matrix
        self.assertEqual(I_G.shape, (3, 3))
        eigvals = np.linalg.eigvalsh(I_G)
        self.assertTrue(
            np.all(eigvals > 0), f"I_G not positive definite: {eigvals}"
        )

        # A_omega_j and A_bar should NOT be all zeros
        self.assertEqual(A_omega_j.shape, (3, gen.num_mj_joints))
        self.assertEqual(A_bar.shape, (3, gen.num_mj_joints))
        self.assertGreater(
            np.linalg.norm(A_bar),
            1e-10,
            "A_bar_omega is all zeros! mj_subtreeVel() likely missing.",
        )

    def test_locked_inertia_is_symmetric_positive_definite(self):
        """Verifies the extracted locked inertia is a valid inertia matrix.

        I_G is read out of the base angular velocity columns of the centroidal
        momentum matrix, so a frame or index mistake there shows up as a loss of
        symmetry or positive definiteness long before it shows up in training loss.
        """
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = rng.uniform(-0.5, 0.5, size=gen.num_mj_joints)
            I_G, _, A_bar = gen.compute_centroidal_matrices(q)
            np.testing.assert_allclose(I_G, I_G.T, atol=1e-9)
            self.assertGreater(np.min(np.linalg.eigvalsh(I_G)), 0.0)
            # A_bar is the solve I_G^-1 @ A_omega_j, so it must reproduce it.
            _, A_omega_j, _ = gen.compute_centroidal_matrices(q)
            np.testing.assert_allclose(I_G @ A_bar, A_omega_j, atol=1e-8)

    def test_base_translation_columns_are_zero(self):
        """Angular momentum about the CoM is invariant to base translation."""
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        gen.data.qpos[:] = 0.0
        gen.data.qpos[2] = 0.8
        gen.data.qpos[3] = 1.0
        gen.data.qvel[:] = 0.0
        mujoco.mj_forward(gen.model, gen.data)
        for col in range(3):
            gen.data.qvel[:] = 0.0
            gen.data.qvel[col] = 1.0
            mujoco.mj_comVel(gen.model, gen.data)
            mujoco.mj_subtreeVel(gen.model, gen.data)
            np.testing.assert_allclose(
                gen.data.subtree_angmom[1], 0.0, atol=1e-9
            )

    def test_centroidal_matrices_configuration_dependent(self):
        """Verifies that A_bar_omega changes with joint configuration."""
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        q_zero = np.zeros(gen.num_mj_joints)
        q_perturbed = np.zeros(gen.num_mj_joints)
        q_perturbed[0] = 0.3  # perturb first joint

        _, _, A_bar_zero = gen.compute_centroidal_matrices(q_zero)
        _, _, A_bar_perturbed = gen.compute_centroidal_matrices(q_perturbed)

        self.assertFalse(
            np.allclose(A_bar_zero, A_bar_perturbed, atol=1e-10),
            "A_bar_omega should change with configuration.",
        )

    def test_generate_dataset_shapes(self):
        """Verifies dataset output shapes and types."""
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        num_samples = 10
        dataset = gen.generate_dataset(num_samples=num_samples, seed=42)

        self.assertIn("q_joints", dataset)
        self.assertIn("A_bar_omega", dataset)
        self.assertIn("I_G", dataset)

        self.assertEqual(
            dataset["q_joints"].shape, (num_samples, gen.num_mj_joints)
        )
        self.assertEqual(
            dataset["A_bar_omega"].shape, (num_samples, 3, gen.num_mj_joints)
        )
        self.assertEqual(dataset["I_G"].shape, (num_samples, 3, 3))

        self.assertEqual(dataset["q_joints"].dtype, np.float32)
        self.assertEqual(dataset["A_bar_omega"].dtype, np.float32)

    def test_dataset_a_bar_nonzero_all_samples(self):
        """Regression: Every A_bar_omega sample must be non-zero."""
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        dataset = gen.generate_dataset(num_samples=20, seed=123)

        for i in range(dataset["A_bar_omega"].shape[0]):
            norm = np.linalg.norm(dataset["A_bar_omega"][i])
            self.assertGreater(
                norm, 1e-10, f"Sample {i}: A_bar_omega is zero (norm={norm})"
            )

    # LINT.IfChange(joint_permutation_test)
    def test_joint_order_matches_pinocchio_kinematic_tree(self):
        """Regression: the dataset joint axis must be in Pinocchio's ordering.

        Pinocchio numbers joints by a depth-first walk of the kinematic tree, not
        by the order in which <joint> elements appear in the URDF. The Atlas URDF
        lists its joints alphabetically, so a document-order permutation scrambles
        the joint vector relative to the C++ MPC, which indexes joints via
        Pinocchio. The scrambling is invisible downstream because the joint count
        is unchanged, so it is asserted explicitly here. How siblings are ordered
        is test_sibling_joints_are_ordered_by_name's business.
        """
        atlas_xml, atlas_urdf = self._require_atlas_urdf()
        tree_order = dataset_generator._parse_pinocchio_joint_order(atlas_urdf)

        document_order = [
            j.attrib["name"]
            for j in ElementTree.parse(atlas_urdf).findall(".//joint")
            if j.attrib.get("type") not in ("fixed", "floating")
        ]
        self.assertEqual(set(tree_order), set(document_order))
        self.assertNotEqual(
            tree_order,
            document_order,
            "Atlas is the regression fixture precisely because its URDF"
            " document order differs from its kinematic tree order.",
        )

        # A parent joint must always precede its children in a tree walk.
        self.assertLess(
            tree_order.index("back_bkz"), tree_order.index("back_bkx")
        )
        self.assertLess(
            tree_order.index("l_arm_shz"), tree_order.index("l_arm_elx")
        )
        self.assertLess(
            tree_order.index("l_leg_hpz"), tree_order.index("l_leg_akx")
        )

        gen = dataset_generator.AcomDatasetGenerator(
            atlas_xml, urdf_path=atlas_urdf
        )
        self.assertEqual(gen.pinocchio_joint_names, tree_order)

    def test_joint_permutation_atlas(self):
        """Verifies that the URDF/MuJoCo joint permutation is applied correctly.

        The reordering path is exercised here on Atlas; that every shipped
        robot's permutation is the identity is asserted, not assumed, in
        TestShippedRobots.test_joint_permutation_is_the_identity.
        """
        atlas_xml, atlas_urdf = self._require_atlas_urdf()
        gen_with_urdf = dataset_generator.AcomDatasetGenerator(
            atlas_xml, urdf_path=atlas_urdf
        )
        gen_without_urdf = dataset_generator.AcomDatasetGenerator(atlas_xml)

        perm = gen_with_urdf.joint_perm
        self.assertIsNotNone(perm)
        self.assertEqual(len(perm), gen_with_urdf.num_mj_joints)
        np.testing.assert_array_equal(
            np.sort(perm),
            np.arange(gen_with_urdf.num_mj_joints),
            "joint_perm is not a valid permutation",
        )

        # perm[i] = MuJoCo index of Pinocchio's i-th joint.
        for i, name in enumerate(gen_with_urdf.pinocchio_joint_names):
            self.assertEqual(gen_with_urdf.mj_joint_names[perm[i]], name)

        ds_pin = gen_with_urdf.generate_dataset(num_samples=5, seed=42)
        ds_mj = gen_without_urdf.generate_dataset(num_samples=5, seed=42)
        np.testing.assert_allclose(
            ds_pin["q_joints"], ds_mj["q_joints"][:, perm], atol=1e-6
        )
        np.testing.assert_allclose(
            ds_pin["A_bar_omega"], ds_mj["A_bar_omega"][:, :, perm], atol=1e-6
        )

    # LINT.ThenChange(//humanoid_learning/acom/dataset_generator.py:joint_permutation)

    def test_sibling_joints_are_ordered_by_name(self):
        """Pinocchio visits the children of a link sorted by joint NAME.

        urdfdom fills each link's child list by iterating a std::map of joints,
        so siblings come out alphabetically whatever order the URDF lists them in.
        The fixture lists its three siblings out of order and gives the first
        one a child, so a document-order walk and a name-order walk disagree.
        testAcomAngularVelocityConsistency.cpp parses the same fixture with
        Pinocchio itself and asserts the same order, which is what ties this
        rule to the parser the MPC runs.
        """
        # LINT.IfChange(sibling_order_fixture)
        urdf = """<robot name="sibling_order">
  <link name="base"/>
  <link name="z_link"/>
  <link name="z_child_link"/>
  <link name="a_link"/>
  <link name="m_link"/>
  <joint name="zeta_joint" type="revolute"><parent link="base"/><child link="z_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="alpha_joint" type="revolute"><parent link="base"/><child link="a_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="mid_joint" type="revolute"><parent link="base"/><child link="m_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="zeta_child_joint" type="revolute"><parent link="z_link"/><child link="z_child_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""
        expected_order = [
            "alpha_joint",
            "mid_joint",
            "zeta_joint",
            "zeta_child_joint",
        ]
        # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/testAcomAngularVelocityConsistency.cpp:sibling_order_fixture)
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = os.path.join(tmpdir, "sibling_order.urdf")
            with open(urdf_path, "w", encoding="utf-8") as urdf_file:
                urdf_file.write(urdf)
            order = dataset_generator._parse_pinocchio_joint_order(urdf_path)
        self.assertEqual(order, expected_order)
        # The fixture only proves anything because its document order is NOT the answer.
        self.assertNotEqual(
            order,
            ["zeta_joint", "zeta_child_joint", "alpha_joint", "mid_joint"],
        )

    def test_joint_without_a_parent_or_child_link_is_rejected(self):
        """A joint missing its <parent> or <child> is named in the error, not an AttributeError."""
        for missing in ("parent", "child"):
            with self.subTest(missing=missing):
                elements = {
                    "parent": '<parent link="base"/>',
                    "child": '<child link="leg"/>',
                }
                del elements[missing]
                urdf = (
                    '<robot name="broken"><link name="base"/><link name="leg"/>'
                    '<joint name="hip"'
                    f' type="revolute">{"".join(elements.values())}</joint>'
                    "</robot>"
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    urdf_path = os.path.join(tmpdir, "broken.urdf")
                    with open(urdf_path, "w", encoding="utf-8") as urdf_file:
                        urdf_file.write(urdf)
                    with self.assertRaisesRegex(
                        ValueError, f"joint 'hip' has no <{missing}> element"
                    ):
                        dataset_generator._parse_pinocchio_joint_order(
                            urdf_path
                        )

    def test_joint_permutation_none_without_urdf(self):
        """Without URDF, joint_perm should be None."""
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        self.assertIsNone(gen.joint_perm)

    def test_generator_deterministic_seed(self):
        """Same seed should produce identical datasets."""
        atlas_xml = self._require_atlas()
        gen = dataset_generator.AcomDatasetGenerator(atlas_xml)
        ds1 = gen.generate_dataset(num_samples=10, seed=42)
        ds2 = gen.generate_dataset(num_samples=10, seed=42)
        np.testing.assert_array_equal(ds1["q_joints"], ds2["q_joints"])
        np.testing.assert_array_equal(ds1["A_bar_omega"], ds2["A_bar_omega"])

    def test_g1_generator(self):
        """Verifies generator works with G1 model."""
        g1_xml = self._require_g1()
        gen = dataset_generator.AcomDatasetGenerator(g1_xml)
        self.assertGreater(gen.num_mj_joints, 0)

        q_zero = np.zeros(gen.num_mj_joints)
        _, _, A_bar = gen.compute_centroidal_matrices(q_zero)
        self.assertGreater(np.linalg.norm(A_bar), 1e-10)


class TestShippedRobots(unittest.TestCase):
    """Properties every robot train_main.py can train must have, checked for each."""

    def test_every_robot_has_its_model_files(self):
        """Without these the rest of this class would have nothing to check."""
        self.assertGreater(len(train_main.robot_names()), 0)
        for robot in train_main.robot_names():
            for key in ("xml", "urdf", "task"):
                path = train_main._ROBOT_CONFIGS[robot][key]
                self.assertIsNotNone(
                    _find_robot_model(path), f"{robot}: {path}"
                )

    def test_the_network_sees_the_joints_the_task_file_leaves_active(self):
        """The fixed joints and the robot name come from the MPC's own task file.

        The network's inputs are the MPC model's joints one for one: Pinocchio's
        order minus model_settings.fixed_joint_names. The task file's initial state
        names every active joint once, which cross-checks them against a block of
        the task file the generator does not read.
        """
        for robot in train_main.robot_names():
            with self.subTest(robot=robot):
                task = _load_task_file(robot)
                fixed_joints = list(task.model_settings.fixed_joint_names)
                self.assertEqual(
                    train_main.load_mpc_model_settings(robot),
                    (task.model_settings.robot_name, fixed_joints),
                )

                gen = train_main.make_generator(robot)
                self.assertEqual(gen.fixed_joints, set(fixed_joints))
                self.assertEqual(
                    gen.active_joint_names,
                    [
                        n
                        for n in gen.pinocchio_joint_names
                        if n not in fixed_joints
                    ],
                )
                # The initial state names every joint of the network's input, and only those.
                self.assertEqual(
                    sorted(
                        entry.joint
                        for entry in task.initial_state.joint_positions
                    ),
                    sorted(gen.active_joint_names),
                )
                self.assertEqual(
                    len(task.initial_state.joint_positions),
                    gen.num_active_joints,
                )

    def test_joint_permutation_is_the_identity(self):
        """Asserted, not assumed: MuJoCo and Pinocchio agree on every shipped robot.

        Each MJCF declares its joints in Pinocchio's order, so the permutation the
        generator applies is the identity. A non-identity one is not an error -
        the reordering exists to handle it - but it would mean the two model
        files have drifted apart, which deserves a look.
        """
        for robot in train_main.robot_names():
            with self.subTest(robot=robot):
                gen = train_main.make_generator(robot)
                np.testing.assert_array_equal(
                    gen.joint_perm, np.arange(gen.num_mj_joints)
                )

    def test_sampling_box_is_the_urdf_joint_limit_box(self):
        """The generator trains on the box the C++ acceptance test grades on.

        The generator samples MuJoCo's joint ranges; the C++ test samples the
        URDF limits through Pinocchio. They are the same box only if the MJCF and
        the URDF agree and nothing is trimmed off either end.
        """
        for robot in train_main.robot_names():
            with self.subTest(robot=robot):
                gen = train_main.make_generator(robot)
                limits = _urdf_joint_limits(
                    dataset_generator.resolve_xml_path(
                        train_main._ROBOT_CONFIGS[robot]["urdf"]
                    )
                )
                low, high = gen.sampling_bounds()
                self.assertEqual(len(low), gen.num_active_joints)
                for joint, name in enumerate(gen.active_joint_names):
                    self.assertIn(name, limits)
                    self.assertAlmostEqual(
                        low[joint], limits[name][0], places=5
                    )
                    self.assertAlmostEqual(
                        high[joint], limits[name][1], places=5
                    )

    def test_nominal_pose_is_inside_the_sampling_box(self):
        """The posture the MPC stands in is one the network was trained on.

        A 10 % margin trimmed off each joint range used to put G1's nominal knee
        angle (0.1 rad) outside the training box, so the network was evaluated in
        stance on configurations it had never seen.
        """
        for robot in train_main.robot_names():
            with self.subTest(robot=robot):
                gen = train_main.make_generator(robot)
                nominal = _initial_joint_state(
                    _load_task_file(robot), gen.active_joint_names
                )
                low, high = gen.sampling_bounds()
                outside = [
                    f"{name}={value} not in [{lo:.4f}, {hi:.4f}]"
                    for name, value, lo, hi in zip(
                        gen.active_joint_names, nominal, low, high
                    )
                    if not lo <= value <= hi
                ]
                self.assertEqual(
                    outside,
                    [],
                    f"{robot}'s initial_state is outside the sampling box",
                )

    def test_generated_samples_fill_the_sampling_box(self):
        """Positive control for the box tests: the samples come from that box, all of it.

        The two tests above read sampling_bounds(), so a margin trimmed off the
        box inside generate_dataset alone would slip past them. Every joint must
        therefore have samples in the outer tenth of its range at both ends: of
        200 uniform draws, the chance that none lands there is 0.9**200, about
        1e-9, while the 10 % margin the generator used to trim leaves it empty.
        """
        for robot in train_main.robot_names():
            with self.subTest(robot=robot):
                gen = train_main.make_generator(robot)
                q_joints = gen.generate_dataset(num_samples=200, seed=5)[
                    "q_joints"
                ]
                low, high = gen.sampling_bounds()
                self.assertTrue(np.all(q_joints >= low.astype(np.float32)))
                self.assertTrue(np.all(q_joints <= high.astype(np.float32)))
                band = 0.1 * (high - low)
                np.testing.assert_array_less(q_joints.min(axis=0), low + band)
                np.testing.assert_array_less(high - band, q_joints.max(axis=0))


class TestFixedJoints(unittest.TestCase):
    """The fixed-joint path every shipped header with wrists depends on."""

    # pylint: disable=invalid-name  # I_G, A_bar: the notation of Chen et al., IROS 2023 (humanoid_learning/acom/README.md).

    def setUp(self):
        self.gen = train_main.make_generator("atlas")
        self.fixed = sorted(self.gen.fixed_joints)
        self.assertGreater(
            len(self.fixed),
            0,
            "these tests need a robot whose task file fixes joints",
        )

    def _full_configuration(
        self, q_active: np.ndarray, fixed_value: float = 0.0
    ):
        """Scatters a dataset row back into a full MuJoCo-order configuration."""
        q_full = np.full(self.gen.num_mj_joints, fixed_value)
        for value, name in zip(q_active, self.gen.active_joint_names):
            q_full[self.gen.mj_joint_names.index(name)] = value
        return q_full

    def test_fixed_joints_are_held_at_zero_and_dropped(self):
        dataset = self.gen.generate_dataset(num_samples=6, seed=3)
        n_active = self.gen.num_mj_joints - len(self.fixed)
        self.assertEqual(self.gen.num_active_joints, n_active)
        self.assertEqual(dataset["q_joints"].shape, (6, n_active))
        self.assertEqual(dataset["A_bar_omega"].shape, (6, 3, n_active))
        for name in self.fixed:
            self.assertNotIn(name, self.gen.active_joint_names)

        active_mj = [
            self.gen.mj_joint_names.index(n)
            for n in self.gen.active_joint_names
        ]
        for sample in range(6):
            q_full = self._full_configuration(
                dataset["q_joints"][sample].astype(np.float64)
            )
            I_G, _, A_bar = self.gen.compute_centroidal_matrices(q_full)
            # Every row was computed with the wrists at zero, and its columns are the active joints' own.
            np.testing.assert_allclose(
                dataset["I_G"][sample], I_G, rtol=1e-4, atol=1e-6
            )
            np.testing.assert_allclose(
                dataset["A_bar_omega"][sample],
                A_bar[:, active_mj],
                rtol=1e-4,
                atol=1e-5,
            )

        # Positive control: with the wrists anywhere but zero the locked inertia is measurably different, so the
        # comparison above would notice a sampler that forgot to zero them.
        q_full = self._full_configuration(
            dataset["q_joints"][0].astype(np.float64), fixed_value=1.0
        )
        I_G_moved, _, _ = self.gen.compute_centroidal_matrices(q_full)
        self.assertFalse(
            np.allclose(dataset["I_G"][0], I_G_moved, rtol=1e-4, atol=1e-6)
        )

    def test_unknown_fixed_joint_is_rejected(self):
        """A misspelt fixed joint must fail here, not as an input_dim mismatch in the C++ MPC."""
        config = train_main._ROBOT_CONFIGS["atlas"]
        with self.assertRaisesRegex(ValueError, "l_arm_wyr"):
            dataset_generator.AcomDatasetGenerator(
                config["xml"],
                urdf_path=config["urdf"],
                fixed_joints=["l_arm_wyr"],
            )

    def test_sa01_generator_fixes_nothing(self):
        """SA01 is legs only: its task file fixes no joint, and all twelve reach the network."""
        gen = train_main.make_generator("sa01")
        self.assertEqual(gen.fixed_joints, set())
        self.assertEqual(gen.num_active_joints, 12)
        _, _, A_bar = gen.compute_centroidal_matrices(
            np.zeros(gen.num_mj_joints)
        )
        self.assertGreater(np.linalg.norm(A_bar), 1e-10)


class TestTrainMainExport(unittest.TestCase):
    """The header train_main.py (and the notebook) writes, and how it is installed."""

    def _params(self, in_dim: int, hidden_dim: int = 8):
        model = models.SirenACOM(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=train_main._CPP_SUPPORTED_NUM_LAYERS,
        )
        return model.init_params(jax.random.PRNGKey(1))

    def test_exported_header_is_what_the_cpp_loader_expects(self):
        gen = train_main.make_generator("atlas")
        with tempfile.TemporaryDirectory() as tmpdir:
            _, cpp_path = train_main.export_network(
                self._params(gen.num_active_joints),
                gen,
                "atlas",
                tmpdir,
                recipe={"num_samples": 123, "epochs": 4},
            )
            self.assertEqual(
                os.path.basename(cpp_path), "AcomSirenWeightsAtlas.h"
            )
            with open(cpp_path, "r", encoding="utf-8") as header:
                content = header.read()
        # The static_asserts of createFromStaticWeights, and the joint list AngularCenterOfMass::Create checks.
        self.assertIn("struct AcomSirenWeightsAtlas {", content)
        self.assertIn(
            "static constexpr size_t num_layers ="
            f" {train_main._CPP_SUPPORTED_NUM_LAYERS + 1};",
            content,
        )
        self.assertIn("static constexpr size_t output_dim = 3;", content)
        self.assertIn(
            f"static constexpr size_t input_dim = {gen.num_active_joints};",
            content,
        )
        self.assertEqual(_header_joint_names(content), gen.active_joint_names)
        # And the header records how it was made.
        self.assertIn(" *   model_settings.robot_name: atlas", content)
        self.assertIn(" *   num_samples: 123", content)
        self.assertIn(" *   epochs: 4", content)
        self.assertIn(
            " *   sampling: uniform over the full joint-limit box", content
        )

    def test_provenance_stays_inside_the_banner_comment(self):
        """A recorded value cannot close the banner comment and leak into the code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "Banner.h")
            export_acom.export_to_cpp_header(
                self._params(4),
                cpp_path,
                class_name="Banner",
                provenance={"note": "ends */ here", "epochs": 3},
            )
            with open(cpp_path, "r", encoding="utf-8") as header:
                content = header.read()
        banner = content.split("#pragma once")[0]
        self.assertEqual(banner.count("*/"), 1, banner)
        self.assertIn(" *   epochs: 3", banner)
        self.assertIn("ends * / here", banner)

    def test_install_refuses_a_different_width(self):
        gen = train_main.make_generator("sa01")
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as out:
            _, cpp_path = train_main.export_network(
                self._params(gen.num_active_joints, hidden_dim=8),
                gen,
                "sa01",
                out,
                recipe={},
            )
            installed = os.path.join(workspace, train_main._header_path("sa01"))
            os.makedirs(os.path.dirname(installed))
            with open(installed, "w", encoding="utf-8") as header:
                header.write("  static constexpr size_t W0_rows = 64;\n")

            with self.assertRaisesRegex(ValueError, "64-wide"):
                train_main.install_header(cpp_path, "sa01", 8, workspace)
            with open(installed, "r", encoding="utf-8") as header:
                self.assertIn(
                    "W0_rows = 64;",
                    header.read(),
                    "a refused install must not write",
                )

            # Same width, or an explicit override, installs.
            self.assertEqual(
                train_main.install_header(
                    cpp_path, "sa01", 8, workspace, force_architecture=True
                ),
                installed,
            )
            with open(installed, "r", encoding="utf-8") as header:
                self.assertIn("W0_rows = 8;", header.read())
            train_main.install_header(cpp_path, "sa01", 8, workspace)


class TestTrainingNotebook(unittest.TestCase):
    """The notebook behind `make train-acom-jupyter` runs, and exports a header the C++ build accepts."""

    def setUp(self):
        notebook = _find_robot_model(_NOTEBOOK_PATH)
        if notebook is None:
            self.fail(f"{_NOTEBOOK_PATH} is missing from the runfiles")
        self.notebook = notebook

    def test_shipped_parameters_are_safe_defaults(self):
        with open(self.notebook, "r", encoding="utf-8") as notebook_file:
            cells = [
                "".join(c["source"])
                for c in json.load(notebook_file)["cells"]
                if c["cell_type"] == "code"
            ]
        parameters = [c for c in cells if c.lstrip().startswith("# Parameters")]
        self.assertEqual(
            len(parameters), 1, "exactly one cell starts with '# Parameters'"
        )
        namespace = {"os": os}
        exec(  # pylint: disable=exec-used  # Runs the shipped notebook's parameter cell.
            parameters[0], namespace
        )
        self.assertIn(namespace["ROBOT"], train_main.robot_names())
        self.assertEqual(
            namespace["HIDDEN_DIM"], train_main._SHIPPED_HIDDEN_DIM
        )
        self.assertFalse(
            namespace["INSTALL_HEADER"],
            "Run All must not overwrite a shipped header",
        )

    def test_notebook_runs_and_exports_a_loadable_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            namespace = _run_notebook(
                self.notebook,
                {
                    "ROBOT": "atlas",
                    "NUM_SAMPLES": 48,
                    "EPOCHS": 2,
                    "HIDDEN_DIM": 8,
                    "OUTPUT_DIR": tmpdir,
                    "TENSORBOARD": False,
                    "INSTALL_HEADER": False,
                },
            )
            cpp_path = namespace["cpp_path"]
            self.assertEqual(os.path.dirname(cpp_path), tmpdir)
            with open(cpp_path, "r", encoding="utf-8") as header:
                content = header.read()
        gen = namespace["generator"]
        self.assertIn(
            f"struct {train_main.header_class_name('atlas')} {{", content
        )
        self.assertIn(
            "static constexpr size_t num_layers ="
            f" {train_main._CPP_SUPPORTED_NUM_LAYERS + 1};",
            content,
        )
        self.assertIn("static constexpr size_t output_dim = 3;", content)
        self.assertIn(
            f"static constexpr size_t input_dim = {gen.num_active_joints};",
            content,
        )
        self.assertEqual(_header_joint_names(content), gen.active_joint_names)
        self.assertIn(" *   driver: notebooks/train_acom_siren.ipynb", content)


if __name__ == "__main__":
    unittest.main()
