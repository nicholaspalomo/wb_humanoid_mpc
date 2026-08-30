"""Unit tests for Angular Center of Mass (aCOM) JAX pipeline."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp

from humanoid_learning.acom.models import SirenACOM
from humanoid_learning.acom.train_acom import train_acom
from humanoid_learning.acom.export_acom import export_to_json, export_to_cpp_header


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

    def test_export_utilities(self):
        """Verifies JSON and C++ header generation."""
        import tempfile
        import os

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
                self.assertIn("static constexpr size_t input_dim = 6;", content)


if __name__ == "__main__":
    unittest.main()
