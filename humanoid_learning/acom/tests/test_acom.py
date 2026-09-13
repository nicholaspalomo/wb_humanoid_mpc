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

import os
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from humanoid_learning.acom.models import SirenACOM
from humanoid_learning.acom.train_main import _CPP_SUPPORTED_NUM_LAYERS
from humanoid_learning.acom.train_acom import train_acom
from humanoid_learning.acom.export_acom import export_to_json, export_to_cpp_header
from humanoid_learning.acom.dataset_generator import (
    AcomDatasetGenerator,
    _parse_pinocchio_joint_order,
    resolve_xml_path,
)


def _find_robot_model(relative_path: str) -> str:
    """Resolve a robot model path from workspace or Bazel runfiles."""
    resolved = resolve_xml_path(relative_path)
    if os.path.exists(resolved):
        return resolved
    return None


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
        is unchanged, so it is asserted explicitly here.
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

        Both robot models happen to declare their MuJoCo joints in kinematic tree
        order, so the permutation is the identity. That is asserted rather than
        assumed: a non-identity permutation here would mean MuJoCo and Pinocchio
        disagree, and the reordering path below is what keeps the dataset aligned.
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


if __name__ == "__main__":
    unittest.main()
