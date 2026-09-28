"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""Unit tests for Angular Center of Mass (aCOM) JAX pipeline."""

import inspect
import json
import os
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

from humanoid_learning.acom import train_main
from humanoid_learning.acom.models import SirenACOM
from humanoid_learning.acom.train_main import _CPP_SUPPORTED_NUM_LAYERS
from humanoid_learning.acom.train_acom import train_acom
from humanoid_learning.acom.export_acom import export_to_json, export_to_cpp_header
from humanoid_learning.acom.dataset_generator import (
    AcomDatasetGenerator,
    _parse_pinocchio_joint_order,
    resolve_xml_path,
)

# The notebook behind `make train-acom-jupyter`, run end to end by TestTrainingNotebook.
_NOTEBOOK_PATH = "notebooks/train_acom_siren.ipynb"

# Index of the first joint in the centroidal state (and in task.yaml's initialState):
# six normalized momenta, three base positions and three base Euler angles come first.
_STATE_JOINT_OFFSET = 12


def _find_robot_model(relative_path: str) -> str:
    """Resolve a robot model path from workspace or Bazel runfiles."""
    resolved = resolve_xml_path(relative_path)
    if os.path.exists(resolved):
        return resolved
    return None


def _load_task_file(robot: str) -> dict:
    """The robot's centroidal MPC task.yaml, parsed."""
    task_path = resolve_xml_path(train_main._ROBOT_CONFIGS[robot]["task"])
    with open(task_path, "r", encoding="utf-8") as task_file:
        return yaml.safe_load(task_file)


def _initial_joint_state(task: dict, num_joints: int) -> np.ndarray:
    """The joint block of task.yaml's initialState, i.e. the MPC's nominal posture."""
    initial_state = task["initialState"]
    return np.array(
        [
            float(initial_state[f"({_STATE_JOINT_OFFSET + joint},0)"])
            for joint in range(num_joints)
        ]
    )


def _urdf_joint_limits(urdf_path: str) -> dict:
    """Joint name -> (lower, upper) for every URDF joint that declares limits."""
    limits = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )
    return limits


def _header_joint_names(content: str) -> list:
    """The joint_names[] a generated weight header records, in order."""
    match = re.search(r"joint_names\[\d+\] = \{([^}]*)\}", content)
    return re.findall(r'"([^"]+)"', match.group(1)) if match else []


def _run_notebook(path: str, overrides: dict) -> dict:
    """Executes a notebook's code cells in order in one namespace, the way Run All would.

    `overrides` replace the values the cell starting with "# Parameters" sets,
    right after it runs. Every cell must be plain Python: an IPython magic does
    not compile, which is the point - the notebook stays runnable outside Jupyter.

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
        exec(compile(source, f"{path} cell {index}", "exec"), namespace)
        if source.lstrip().startswith("# Parameters"):
            namespace.update(overrides)
    return namespace


class TestAcomPipeline(unittest.TestCase):
    """Tests SIREN model forward pass, Jacobians, training convergence, and export."""

    def setUp(self):
        self.in_dim = 6
        self.model = SirenACOM(
            in_dim=self.in_dim, hidden_dim=32, num_layers=2, out_dim=3, omega_0=30.0
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

        # Full aCOM pose: [pos(3), rpy(3), q_j(6)]
        q_full = jnp.zeros((3 + 3 + self.in_dim,))
        acom_pose = self.model.full_acom_pose(self.params, q_full)
        self.assertEqual(acom_pose.shape, (3,))

        acom_jac = self.model.full_acom_jacobian(self.params, q_full)
        self.assertEqual(acom_jac.shape, (3, 6 + self.in_dim))
        # Check base linear velocity block is zero and angular velocity block is identity
        np.testing.assert_allclose(acom_jac[:, :3], np.zeros((3, 3)))
        np.testing.assert_allclose(acom_jac[:, 3:6], np.eye(3))

    def test_training_convergence_on_synthetic_data(self):
        """Verifies that JAX training converges on a synthetic linear/sinusoidal CMM."""
        num_samples = 200
        rng = np.random.default_rng(42)
        q_samples = rng.uniform(-1.0, 1.0, size=(num_samples, self.in_dim)).astype(
            np.float32
        )

        # Synthetic target Jacobian: J(q) = W_true + 0.1 * sin(q)
        w_true = rng.standard_normal((3, self.in_dim)).astype(np.float32)
        A_bar_samples = np.stack([w_true + 0.1 * np.sin(q) for q in q_samples], axis=0)

        dataset = {
            "q_joints": q_samples,
            "A_bar_omega": A_bar_samples,
        }

        _, trained_params, history = train_acom(
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
        q_samples = rng.uniform(-1.0, 1.0, size=(num_samples, self.in_dim)).astype(
            np.float32
        )
        w_true = rng.standard_normal((3, self.in_dim)).astype(np.float32)
        A_bar_samples = np.stack([w_true + 0.1 * np.sin(q) for q in q_samples], axis=0)
        dataset = {
            "q_joints": q_samples,
            "A_bar_omega": A_bar_samples,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "tb_logs")
            _, _, history = train_acom(
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
            event_files = [f for f in os.listdir(log_dir) if "events.out.tfevents" in f]
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

            export_to_json(self.params, json_path)
            export_to_cpp_header(self.params, cpp_path, class_name="TestWeights")

            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(cpp_path))

            with open(cpp_path, "r") as f:
                content = f.read()
                self.assertIn("struct TestWeights", content)
                self.assertIn("static constexpr std::size_t input_dim = 6;", content)

    def test_export_round_trip_parity(self):
        """Verifies that exported C++ header weights match the original JAX parameters.

        Parses the generated header back to verify that:
        1. The structural constants (input_dim, output_dim, num_layers, omega_0) are correct.
        2. The flattened weight arrays have the expected number of elements.
        3. The serialized weight values match the original JAX params to 10 decimal places.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "TestParity.h")
            export_to_cpp_header(
                self.params, cpp_path, class_name="TestParity", omega_0=30.0
            )

            with open(cpp_path, "r") as f:
                content = f.read()

            # Verify structural metadata
            self.assertIn(
                f"static constexpr std::size_t input_dim = {self.in_dim};", content
            )
            self.assertIn("static constexpr std::size_t output_dim = 3;", content)
            num_layers_total = len(self.params)  # 2 hidden + 1 output = 3
            self.assertIn(
                f"static constexpr std::size_t num_layers = {num_layers_total};",
                content,
            )
            self.assertIn("static constexpr double omega_0 = 30.0;", content)

            # Verify each layer's weight array length
            for idx, (w, b) in enumerate(self.params):
                w_np = np.array(w)
                b_np = np.array(b)
                out_dim, in_dim = w_np.shape

                self.assertIn(
                    f"static constexpr std::size_t W{idx}_rows = {out_dim};", content
                )
                self.assertIn(
                    f"static constexpr std::size_t W{idx}_cols = {in_dim};", content
                )

                # Extract the weight array from the header and verify element count
                w_pattern = rf"W{idx}\[{out_dim * in_dim}\] = \{{([^}}]+)\}}"
                w_match = re.search(w_pattern, content)
                self.assertIsNotNone(w_match, f"Could not find W{idx} array in header")
                w_values = [float(v.strip()) for v in w_match.group(1).split(",")]
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
                self.assertIsNotNone(b_match, f"Could not find b{idx} array in header")
                b_values = [float(v.strip()) for v in b_match.group(1).split(",")]
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
        model = SirenACOM(
            in_dim=self.in_dim,
            hidden_dim=8,
            num_layers=_CPP_SUPPORTED_NUM_LAYERS,
            out_dim=3,
        )
        params = model.init_params(jax.random.PRNGKey(0))
        self.assertEqual(len(params), _CPP_SUPPORTED_NUM_LAYERS + 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "TestLayers.h")
            export_to_cpp_header(params, cpp_path, class_name="TestLayers")
            with open(cpp_path, "r") as f:
                content = f.read()
        self.assertIn("static constexpr std::size_t num_layers = 3;", content)

    def test_library_defaults_build_a_loadable_network(self):
        """Direct users of the library get the C++ evaluator's layer count unless they ask otherwise.

        A default of 3 sine layers is how the old training notebook produced a
        4-layer header that the C++ static_assert rejects.
        """
        self.assertEqual(SirenACOM(in_dim=3).num_layers, _CPP_SUPPORTED_NUM_LAYERS)
        self.assertEqual(
            inspect.signature(train_acom).parameters["num_layers"].default,
            _CPP_SUPPORTED_NUM_LAYERS,
        )

    def test_export_records_joint_names(self):
        """The exported header must record the joint ordering it was trained on."""
        joint_names = [f"joint_{i}" for i in range(self.in_dim)]
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "TestNames.h")
            export_to_cpp_header(
                self.params, cpp_path, class_name="TestNames", joint_names=joint_names
            )
            with open(cpp_path, "r") as f:
                content = f.read()

        self.assertIn(f"joint_names[{self.in_dim}]", content)
        for name in joint_names:
            self.assertIn(f'"{name}"', content)

        # A joint list that does not describe the network must be rejected.
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                export_to_cpp_header(
                    self.params,
                    os.path.join(tmpdir, "Bad.h"),
                    joint_names=joint_names[:-1],
                )


class TestDatasetGenerator(unittest.TestCase):
    """Tests for the MuJoCo-based dataset generator."""

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

    def _require_atlas(self):
        if self.atlas_xml is None:
            self.skipTest("Atlas XML model not found")

    def _require_atlas_urdf(self):
        if self.atlas_xml is None or self.atlas_urdf is None:
            self.skipTest("Atlas XML or URDF model not found")

    def _require_g1(self):
        if self.g1_xml is None:
            self.skipTest("G1 XML model not found")

    def test_generator_init_floating_base(self):
        """Verifies generator initializes correctly for a floating-base robot."""
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
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
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
        q_zero = np.zeros(gen.num_mj_joints)
        I_G, A_omega_j, A_bar = gen.compute_centroidal_matrices(q_zero)

        # I_G should be a positive-definite 3x3 matrix
        self.assertEqual(I_G.shape, (3, 3))
        eigvals = np.linalg.eigvalsh(I_G)
        self.assertTrue(np.all(eigvals > 0), f"I_G not positive definite: {eigvals}")

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
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
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
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
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
            np.testing.assert_allclose(gen.data.subtree_angmom[1], 0.0, atol=1e-9)

    def test_centroidal_matrices_configuration_dependent(self):
        """Verifies that A_bar_omega changes with joint configuration."""
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
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
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
        num_samples = 10
        dataset = gen.generate_dataset(num_samples=num_samples, seed=42)

        self.assertIn("q_joints", dataset)
        self.assertIn("A_bar_omega", dataset)
        self.assertIn("I_G", dataset)

        self.assertEqual(dataset["q_joints"].shape, (num_samples, gen.num_mj_joints))
        self.assertEqual(
            dataset["A_bar_omega"].shape, (num_samples, 3, gen.num_mj_joints)
        )
        self.assertEqual(dataset["I_G"].shape, (num_samples, 3, 3))

        self.assertEqual(dataset["q_joints"].dtype, np.float32)
        self.assertEqual(dataset["A_bar_omega"].dtype, np.float32)

    def test_dataset_a_bar_nonzero_all_samples(self):
        """Regression: Every A_bar_omega sample must be non-zero."""
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
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
        self._require_atlas_urdf()
        tree_order = _parse_pinocchio_joint_order(self.atlas_urdf)

        document_order = [
            j.attrib["name"]
            for j in ET.parse(self.atlas_urdf).findall(".//joint")
            if j.attrib.get("type") not in ("fixed", "floating")
        ]
        self.assertEqual(set(tree_order), set(document_order))
        self.assertNotEqual(
            tree_order,
            document_order,
            "Atlas is the regression fixture precisely because its URDF document "
            "order differs from its kinematic tree order.",
        )

        # A parent joint must always precede its children in a tree walk.
        self.assertLess(tree_order.index("back_bkz"), tree_order.index("back_bkx"))
        self.assertLess(tree_order.index("l_arm_shz"), tree_order.index("l_arm_elx"))
        self.assertLess(tree_order.index("l_leg_hpz"), tree_order.index("l_leg_akx"))

        gen = AcomDatasetGenerator(self.atlas_xml, urdf_path=self.atlas_urdf)
        self.assertEqual(gen.pinocchio_joint_names, tree_order)

    def test_joint_permutation_atlas(self):
        """Verifies that the URDF/MuJoCo joint permutation is applied correctly.

        The reordering path is exercised here on Atlas; that every shipped
        robot's permutation is the identity is asserted, not assumed, in
        TestShippedRobots.test_joint_permutation_is_the_identity.
        """
        self._require_atlas_urdf()
        gen_with_urdf = AcomDatasetGenerator(self.atlas_xml, urdf_path=self.atlas_urdf)
        gen_without_urdf = AcomDatasetGenerator(self.atlas_xml)

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
        expected_order = ["alpha_joint", "mid_joint", "zeta_joint", "zeta_child_joint"]
        # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/testAcomAngularVelocityConsistency.cpp:sibling_order_fixture)
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = os.path.join(tmpdir, "sibling_order.urdf")
            with open(urdf_path, "w", encoding="utf-8") as urdf_file:
                urdf_file.write(urdf)
            order = _parse_pinocchio_joint_order(urdf_path)
        self.assertEqual(order, expected_order)
        # The fixture only proves anything because its document order is NOT the answer.
        self.assertNotEqual(
            order, ["zeta_joint", "zeta_child_joint", "alpha_joint", "mid_joint"]
        )

    def test_joint_permutation_none_without_urdf(self):
        """Without URDF, joint_perm should be None."""
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
        self.assertIsNone(gen.joint_perm)

    def test_generator_deterministic_seed(self):
        """Same seed should produce identical datasets."""
        self._require_atlas()
        gen = AcomDatasetGenerator(self.atlas_xml)
        ds1 = gen.generate_dataset(num_samples=10, seed=42)
        ds2 = gen.generate_dataset(num_samples=10, seed=42)
        np.testing.assert_array_equal(ds1["q_joints"], ds2["q_joints"])
        np.testing.assert_array_equal(ds1["A_bar_omega"], ds2["A_bar_omega"])

    def test_g1_generator(self):
        """Verifies generator works with G1 model."""
        self._require_g1()
        gen = AcomDatasetGenerator(self.g1_xml)
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
                self.assertIsNotNone(_find_robot_model(path), f"{robot}: {path}")

    def test_the_network_sees_the_joints_the_task_file_leaves_active(self):
        """The fixed joints and the robot name come from the MPC's own task file.

        The network's inputs are the MPC model's joints one for one: Pinocchio's
        order minus model_settings.fixedJointNames. The task file's initial state
        has one entry per active joint, which cross-checks the count against a
        block of the task file the generator does not read.
        """
        for robot in train_main.robot_names():
            with self.subTest(robot=robot):
                task = _load_task_file(robot)
                model_settings = task["model_settings"]
                fixed_joints = model_settings["fixedJointNames"] or []
                self.assertEqual(
                    train_main.load_mpc_model_settings(robot),
                    (model_settings["robotName"], list(fixed_joints)),
                )

                gen = train_main.make_generator(robot)
                self.assertEqual(gen.fixed_joints, set(fixed_joints))
                self.assertEqual(
                    gen.active_joint_names,
                    [n for n in gen.pinocchio_joint_names if n not in fixed_joints],
                )
                joint_entries = [
                    key
                    for key in task["initialState"]
                    if int(key.strip("()").split(",")[0]) >= _STATE_JOINT_OFFSET
                ]
                self.assertEqual(len(joint_entries), gen.num_active_joints)

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
                    resolve_xml_path(train_main._ROBOT_CONFIGS[robot]["urdf"])
                )
                low, high = gen.sampling_bounds()
                self.assertEqual(len(low), gen.num_active_joints)
                for joint, name in enumerate(gen.active_joint_names):
                    self.assertIn(name, limits)
                    self.assertAlmostEqual(low[joint], limits[name][0], places=5)
                    self.assertAlmostEqual(high[joint], limits[name][1], places=5)

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
                    _load_task_file(robot), gen.num_active_joints
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
                    outside, [], f"{robot}'s initialState is outside the sampling box"
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
                q_joints = gen.generate_dataset(num_samples=200, seed=5)["q_joints"]
                low, high = gen.sampling_bounds()
                self.assertTrue(np.all(q_joints >= low.astype(np.float32)))
                self.assertTrue(np.all(q_joints <= high.astype(np.float32)))
                band = 0.1 * (high - low)
                np.testing.assert_array_less(q_joints.min(axis=0), low + band)
                np.testing.assert_array_less(high - band, q_joints.max(axis=0))


class TestFixedJoints(unittest.TestCase):
    """The fixed-joint path every shipped header with wrists depends on."""

    def setUp(self):
        self.gen = train_main.make_generator("atlas")
        self.fixed = sorted(self.gen.fixed_joints)
        self.assertGreater(
            len(self.fixed), 0, "these tests need a robot whose task file fixes joints"
        )

    def _full_configuration(self, q_active: np.ndarray, fixed_value: float = 0.0):
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
            self.gen.mj_joint_names.index(n) for n in self.gen.active_joint_names
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
            AcomDatasetGenerator(
                config["xml"], urdf_path=config["urdf"], fixed_joints=["l_arm_wyr"]
            )

    def test_sa01_generator_fixes_nothing(self):
        """SA01 is legs only: its task file fixes no joint, and all twelve reach the network."""
        gen = train_main.make_generator("sa01")
        self.assertEqual(gen.fixed_joints, set())
        self.assertEqual(gen.num_active_joints, 12)
        _, _, A_bar = gen.compute_centroidal_matrices(np.zeros(gen.num_mj_joints))
        self.assertGreater(np.linalg.norm(A_bar), 1e-10)


class TestTrainMainExport(unittest.TestCase):
    """The header train_main.py (and the notebook) writes, and how it is installed."""

    def _params(self, in_dim: int, hidden_dim: int = 8):
        model = SirenACOM(
            in_dim=in_dim, hidden_dim=hidden_dim, num_layers=_CPP_SUPPORTED_NUM_LAYERS
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
            self.assertEqual(os.path.basename(cpp_path), "AcomSirenWeightsAtlas.h")
            with open(cpp_path, "r", encoding="utf-8") as header:
                content = header.read()
        # The static_asserts of createFromStaticWeights, and the joint list AngularCenterOfMass::Create checks.
        self.assertIn("struct AcomSirenWeightsAtlas {", content)
        self.assertIn(
            f"static constexpr std::size_t num_layers = {_CPP_SUPPORTED_NUM_LAYERS + 1};",
            content,
        )
        self.assertIn("static constexpr std::size_t output_dim = 3;", content)
        self.assertIn(
            f"static constexpr std::size_t input_dim = {gen.num_active_joints};",
            content,
        )
        self.assertEqual(_header_joint_names(content), gen.active_joint_names)
        # And the header records how it was made.
        self.assertIn(" *   model_settings.robotName: atlas", content)
        self.assertIn(" *   num_samples: 123", content)
        self.assertIn(" *   epochs: 4", content)
        self.assertIn(" *   sampling: uniform over the full joint-limit box", content)

    def test_provenance_stays_inside_the_banner_comment(self):
        """A recorded value cannot close the banner comment and leak into the code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cpp_path = os.path.join(tmpdir, "Banner.h")
            export_to_cpp_header(
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
                header.write("  static constexpr std::size_t W0_rows = 64;\n")

            with self.assertRaisesRegex(ValueError, "64-wide"):
                train_main.install_header(cpp_path, "sa01", 8, workspace)
            with open(installed, "r", encoding="utf-8") as header:
                self.assertIn(
                    "W0_rows = 64;", header.read(), "a refused install must not write"
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
        self.notebook = _find_robot_model(_NOTEBOOK_PATH)
        self.assertIsNotNone(
            self.notebook, f"{_NOTEBOOK_PATH} is missing from the runfiles"
        )

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
        exec(parameters[0], namespace)
        self.assertIn(namespace["ROBOT"], train_main.robot_names())
        self.assertEqual(namespace["HIDDEN_DIM"], train_main._SHIPPED_HIDDEN_DIM)
        self.assertFalse(
            namespace["INSTALL_HEADER"], "Run All must not overwrite a shipped header"
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
        self.assertIn(f"struct {train_main.header_class_name('atlas')} {{", content)
        self.assertIn(
            f"static constexpr std::size_t num_layers = {_CPP_SUPPORTED_NUM_LAYERS + 1};",
            content,
        )
        self.assertIn("static constexpr std::size_t output_dim = 3;", content)
        self.assertIn(
            f"static constexpr std::size_t input_dim = {gen.num_active_joints};",
            content,
        )
        self.assertEqual(_header_joint_names(content), gen.active_joint_names)
        self.assertIn(" *   driver: notebooks/train_acom_siren.ipynb", content)


if __name__ == "__main__":
    unittest.main()
