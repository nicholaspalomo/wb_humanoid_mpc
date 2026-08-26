"""Unit and integration tests for JAX Whole-Body Controller, Residual Environment, and GRPO."""

import unittest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from humanoid_learning.envs.humanoid_residual_wbc_env import (
    HumanoidResidualWBCConfig,
    HumanoidResidualWBCEnv,
)
from humanoid_learning.training.offline_dataset import OfflineTrajectoryDataset
from humanoid_learning.training.policy_network import HumanoidResidualPolicy
from humanoid_learning.training.train_grpo import (
    HumanoidGRPOConfig,
    HumanoidGRPOTrainer,
)
from humanoid_learning.wbc.jax_wbc import JaxWBCConfig, JaxWholeBodyController


class TestJaxWholeBodyController(unittest.TestCase):
    """Test suite for JAX Whole-Body Controller."""

    def setUp(self):
        self.wbc = JaxWholeBodyController(
            nv=18,
            n_act=12,
            n_contacts=2,
            config=JaxWBCConfig(tau_max=80.0, friction_coef=0.6),
        )

    def test_wbc_solve_outputs(self):
        """Test basic WBC torque and acceleration solving."""
        q = jnp.zeros((19,), dtype=jnp.float32).at[2].set(0.85).at[3].set(1.0)
        qd = jnp.zeros((18,), dtype=jnp.float32)
        q_des = jnp.full((12,), 0.1, dtype=jnp.float32)

        out = self.wbc.solve(q=q, qd=qd, q_des=q_des)

        # Check shapes
        self.assertEqual(out.torques.shape, (12,))
        self.assertEqual(out.qddot.shape, (18,))
        self.assertEqual(out.contact_forces.shape, (6,))
        self.assertEqual(out.posture_error.shape, (12,))

        # Torques should respect bounds
        self.assertTrue(jnp.all(jnp.abs(out.torques) <= 80.0 + 1e-5))

        # Normal contact forces should be positive
        self.assertTrue(out.contact_forces[2] > 0.0)
        self.assertTrue(out.contact_forces[5] > 0.0)

    def test_wbc_jit_compilation(self):
        """Test JIT compilation of the WBC solver."""
        solve_jit = jax.jit(self.wbc.solve)
        q = jnp.zeros((19,), dtype=jnp.float32).at[2].set(0.85).at[3].set(1.0)
        qd = jnp.zeros((18,), dtype=jnp.float32)
        q_des = jnp.zeros((12,), dtype=jnp.float32)

        out = solve_jit(q=q, qd=qd, q_des=q_des)
        self.assertEqual(out.torques.shape, (12,))


class TestHumanoidResidualEnvironment(unittest.TestCase):
    """Test suite for Humanoid Residual WBC Environment."""

    def setUp(self):
        self.env = HumanoidResidualWBCEnv(
            config=HumanoidResidualWBCConfig(action_scale=0.25)
        )

    def test_env_reset(self):
        """Test environment reset and observation dimension."""
        rng = jax.random.PRNGKey(42)
        state = self.env.reset(rng)

        self.assertEqual(self.env.observation_size, state.obs.shape[0])
        self.assertEqual(self.env.action_size, self.env.n_act)
        self.assertEqual(state.obs.shape, (self.env.observation_size,))
        self.assertEqual(float(state.done), 0.0)

    def test_env_step_with_residuals(self):
        """Test environment step with residual actions and JIT compilation."""
        step_fn = jax.jit(self.env.step)
        reset_fn = jax.jit(self.env.reset)

        rng = jax.random.PRNGKey(42)
        state = reset_fn(rng)

        action = jnp.full((self.env.action_size,), 0.05, dtype=jnp.float32)
        next_state = step_fn(state, action)

        self.assertEqual(next_state.obs.shape, (self.env.observation_size,))
        self.assertTrue(jnp.isfinite(next_state.reward))
        self.assertIn("forward_vel", next_state.metrics)
        self.assertIn("tracking_reward", next_state.metrics)


class TestPolicyAndDataset(unittest.TestCase):
    """Test suite for Humanoid Residual Policy and Offline Dataset."""

    def setUp(self):
        self.obs_dim = 40
        self.act_dim = 12
        self.rng = jax.random.PRNGKey(123)

    def test_offline_dataset_sampling(self):
        """Test offline dataset creation and minibatch sampling."""
        dataset = OfflineTrajectoryDataset(
            filepath=None,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            synthetic_samples=500,
        )
        batch = dataset.sample_batch(self.rng, batch_size=32)

        self.assertEqual(batch.observations.shape, (32, self.obs_dim))
        self.assertEqual(batch.actions.shape, (32, self.act_dim))
        self.assertEqual(batch.rewards.shape, (32,))

    def test_policy_sampling_and_kl(self):
        """Test policy forward pass, stochastic sampling, and analytical KL divergence."""
        rng1, rng2 = jax.random.split(self.rng)
        policy_rngs = nnx.Rngs(params=rng1)
        policy = HumanoidResidualPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            rngs=policy_rngs,
        )

        ref_rngs = nnx.Rngs(params=rng2)
        ref_policy = HumanoidResidualPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            rngs=ref_rngs,
        )

        dummy_obs = jnp.zeros((4, self.obs_dim), dtype=jnp.float32)
        action, log_prob = policy.sample_action(rng1, dummy_obs)

        self.assertEqual(action.shape, (4, self.act_dim))
        self.assertEqual(log_prob.shape, (4,))
        self.assertTrue(jnp.all(action >= -1.0) and jnp.all(action <= 1.0))

        # Check KL divergence evaluation
        kl = policy.evaluate_kl(dummy_obs, ref_policy)
        self.assertEqual(kl.shape, (4,))
        self.assertTrue(jnp.all(kl >= 0.0))


class TestHumanoidGRPOTrainer(unittest.TestCase):
    """Test suite for GRPO Trainer on Humanoid WBC."""

    def test_grpo_advantage_and_train_step(self):
        """Test group relative rollout collection and gradient update."""
        env = HumanoidResidualWBCEnv()
        rng = jax.random.PRNGKey(42)
        rng1, rng2 = jax.random.split(rng)

        policy = HumanoidResidualPolicy(
            obs_dim=env.observation_size,
            act_dim=env.action_size,
            rngs=nnx.Rngs(params=rng1),
        )
        ref_policy = HumanoidResidualPolicy(
            obs_dim=env.observation_size,
            act_dim=env.action_size,
            rngs=nnx.Rngs(params=rng2),
        )
        nnx.update(ref_policy, nnx.state(policy))

        config = HumanoidGRPOConfig(
            group_size=4,
            num_envs=2,
            rollout_horizon=4,
            learning_rate=1e-3,
        )
        trainer = HumanoidGRPOTrainer(
            env=env,
            policy=policy,
            ref_policy=ref_policy,
            config=config,
        )

        # 1. Collect group rollouts
        batch, _ = trainer.sample_group_rollouts(rng)
        self.assertEqual(batch.states.shape, (2, env.observation_size))
        self.assertEqual(batch.actions.shape, (2, 4, env.action_size))
        self.assertEqual(batch.returns.shape, (2, 4))
        self.assertEqual(batch.advantages.shape, (2, 4))

        # Group advantages must have near-zero mean across each group
        group_means = jnp.mean(batch.advantages, axis=-1)
        np.testing.assert_allclose(
            np.array(group_means), np.zeros(2), atol=1e-4
        )

        # 2. Perform optimization step
        loss, kl_div, clip_frac = trainer.train_step(batch)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(kl_div))
        self.assertTrue(0.0 <= clip_frac <= 1.0)


class TestRobotModelLoader(unittest.TestCase):
    """Test suite for RobotModelLoader across G1 and Atlas models."""

    def test_load_g1_spec(self):
        """Test loading Unitree G1 robot specification."""
        from humanoid_learning.wbc.robot_model_loader import load_robot_spec

        spec = load_robot_spec("g1")
        self.assertIn("g1", spec.name.lower())
        self.assertEqual(spec.n_act, 29)
        self.assertEqual(spec.nq, 36)
        self.assertEqual(spec.nv, 35)
        self.assertGreater(spec.total_mass, 30.0)
        self.assertLess(spec.total_mass, 45.0)
        self.assertEqual(len(spec.joint_limits_lower), 29)
        self.assertEqual(len(spec.joint_limits_upper), 29)
        self.assertEqual(len(spec.torque_limits), 29)

    def test_load_atlas_spec(self):
        """Test loading normal DRC Atlas robot specification."""
        from humanoid_learning.wbc.robot_model_loader import load_robot_spec

        spec = load_robot_spec("atlas")
        self.assertEqual(spec.name, "atlas")
        self.assertNotIn("gantry", spec.name.lower())
        self.assertEqual(spec.n_act, 28)
        self.assertEqual(spec.nq, 35)
        self.assertEqual(spec.nv, 34)
        self.assertGreater(spec.total_mass, 140.0)
        self.assertLess(spec.total_mass, 210.0)

    def test_wbc_with_g1_spec(self):
        """Test initializing and stepping WBC with G1 robot spec."""
        from humanoid_learning.wbc.robot_model_loader import load_robot_spec

        spec = load_robot_spec("g1")
        wbc = JaxWholeBodyController(spec=spec)
        self.assertEqual(wbc.n_act, 29)
        self.assertEqual(wbc.nv, 35)

        q = (
            jnp.zeros((spec.nq,), dtype=jnp.float32)
            .at[2]
            .set(spec.default_standing_height)
            .at[3]
            .set(1.0)
        )
        qd = jnp.zeros((spec.nv,), dtype=jnp.float32)
        q_des = spec.get_q_nominal_jax()

        out = wbc.solve(q=q, qd=qd, q_des=q_des)
        self.assertEqual(out.torques.shape, (29,))
        self.assertEqual(out.qddot.shape, (35,))

    def test_quadruped_wbc_with_four_contacts(self):
        """Test WBC with a 4-contact quadruped robot model (e.g. ANYmal / Unitree Go2)."""
        from humanoid_learning.wbc.robot_model_loader import RobotModelSpec

        quad_spec = RobotModelSpec(
            name="quadruped_12dof",
            nq=19,  # 3 pos + 4 quat + 12 joints
            nv=18,  # 6 base + 12 joints
            n_act=12,
            joint_names=[
                "FL_hip", "FL_thigh", "FL_calf",
                "FR_hip", "FR_thigh", "FR_calf",
                "RL_hip", "RL_thigh", "RL_calf",
                "RR_hip", "RR_thigh", "RR_calf",
            ],
            actuator_names=[f"act_{i}" for i in range(12)],
            q_nominal=[0.0, 0.6, -1.2] * 4,
            joint_limits_lower=[-2.0] * 12,
            joint_limits_upper=[2.0] * 12,
            torque_limits=[80.0] * 12,
            total_mass=25.0,
            body_masses={"base": 15.0, "FL_foot": 2.5, "FR_foot": 2.5, "RL_foot": 2.5, "RR_foot": 2.5},
            default_standing_height=0.45,
            contact_body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            limb_joint_indices={
                "FL": [0, 1, 2],
                "FR": [3, 4, 5],
                "RL": [6, 7, 8],
                "RR": [9, 10, 11],
            },
        )
        self.assertEqual(quad_spec.num_contact_points, 4)

        wbc = JaxWholeBodyController(spec=quad_spec)
        self.assertEqual(wbc.n_contacts, 4)
        self.assertEqual(wbc.n_act, 12)
        self.assertEqual(wbc.nv, 18)

        q = jnp.zeros((19,), dtype=jnp.float32).at[2].set(0.45).at[3].set(1.0)
        qd = jnp.zeros((18,), dtype=jnp.float32)
        q_des = quad_spec.get_q_nominal_jax()

        out = wbc.solve(q=q, qd=qd, q_des=q_des)
        self.assertEqual(out.torques.shape, (12,))
        self.assertEqual(out.qddot.shape, (18,))
        self.assertEqual(out.contact_forces.shape, (12,))  # 4 contacts * 3 = 12 forces
        self.assertTrue(jnp.all(out.contact_forces[2::3] > 0))  # Positive normal contact forces

    def test_drone_zero_contacts_wbc(self):
        """Test WBC with a 0-contact aerial robot/drone (quadrotor/hexarotor)."""
        from humanoid_learning.wbc.robot_model_loader import RobotModelSpec

        drone_spec = RobotModelSpec(
            name="quadrotor_drone",
            nq=11,  # 3 pos + 4 quat + 4 rotor speeds/servos
            nv=10,  # 6 base + 4 actuators
            n_act=4,
            joint_names=["rotor_1", "rotor_2", "rotor_3", "rotor_4"],
            actuator_names=["motor_1", "motor_2", "motor_3", "motor_4"],
            q_nominal=[0.0] * 4,
            joint_limits_lower=[-100.0] * 4,
            joint_limits_upper=[100.0] * 4,
            torque_limits=[25.0] * 4,
            total_mass=1.5,
            body_masses={"base": 1.5},
            default_standing_height=1.0,
            contact_body_names=[],  # 0 CONTACT POINTS
            limb_joint_indices={"rotors": [0, 1, 2, 3]},
        )
        self.assertEqual(drone_spec.num_contact_points, 0)

        wbc = JaxWholeBodyController(spec=drone_spec)
        self.assertEqual(wbc.n_contacts, 0)
        self.assertEqual(wbc.n_act, 4)
        self.assertEqual(wbc.nv, 10)

        q = jnp.zeros((11,), dtype=jnp.float32).at[2].set(1.0).at[3].set(1.0)
        qd = jnp.zeros((10,), dtype=jnp.float32)
        q_des = drone_spec.get_q_nominal_jax()

        out = wbc.solve(q=q, qd=qd, q_des=q_des)
        self.assertEqual(out.torques.shape, (4,))
        self.assertEqual(out.qddot.shape, (10,))
        self.assertEqual(out.contact_forces.shape, (0,))  # 0 contact forces


if __name__ == "__main__":
    unittest.main()


