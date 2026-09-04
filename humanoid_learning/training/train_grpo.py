"""Group Relative Policy Optimization (GRPO) for Humanoid Residual Whole-Body Control."""

import argparse
from dataclasses import dataclass
import os
import time
from typing import Dict, List, NamedTuple, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from humanoid_learning.envs.humanoid_residual_wbc_env import (
    HumanoidResidualWBCConfig,
    HumanoidResidualWBCEnv,
)
from humanoid_learning.training.bootstrap_bc import bootstrap_policy
from humanoid_learning.training.policy_network import HumanoidResidualPolicy


# ==============================================================================
# GRPO Training Hyperparameter Defaults
# ==============================================================================
DEFAULT_GRPO_GROUP_SIZE: int = 4  # Number of candidate actions G per prompt/state
DEFAULT_GRPO_NUM_ENVS: int = 4  # Batch size B of parallel environments
DEFAULT_GRPO_ROLLOUT_HORIZON: int = 8  # Horizon H for evaluating each candidate rollout
DEFAULT_GRPO_NUM_ITERATIONS: int = 20  # Number of outer GRPO iterations
DEFAULT_GRPO_NUM_EPOCHS_PER_ITER: int = 2  # Inner optimization epochs per iteration
DEFAULT_GRPO_LEARNING_RATE: float = 3e-4  # AdamW learning rate
DEFAULT_GRPO_CLIP_EPS: float = 0.2  # PPO surrogate clipping epsilon
DEFAULT_GRPO_BETA_KL: float = 0.05  # Reference policy KL penalty weight
DEFAULT_GRPO_GAMMA: float = 0.99  # Discount factor
DEFAULT_GRPO_ACTION_SCALE: float = 0.25  # Joint residual scale (rad)
DEFAULT_GRPO_BOOTSTRAP_EPOCHS: int = 5  # Offline BC warmstart epochs
DEFAULT_GRADIENT_CLIP_NORM: float = 1.0  # Max gradient norm for AdamW
ADVANTAGE_NORMALIZATION_EPS: float = 1e-6  # Numerical epsilon for group advantage std


@dataclass
class HumanoidGRPOConfig:
    """Hyperparameters for Humanoid Residual WBC GRPO training."""

    group_size: int = DEFAULT_GRPO_GROUP_SIZE
    num_envs: int = DEFAULT_GRPO_NUM_ENVS
    rollout_horizon: int = DEFAULT_GRPO_ROLLOUT_HORIZON
    num_iterations: int = DEFAULT_GRPO_NUM_ITERATIONS
    num_epochs_per_iter: int = DEFAULT_GRPO_NUM_EPOCHS_PER_ITER
    learning_rate: float = DEFAULT_GRPO_LEARNING_RATE
    clip_eps: float = DEFAULT_GRPO_CLIP_EPS
    beta_kl: float = DEFAULT_GRPO_BETA_KL
    gamma: float = DEFAULT_GRPO_GAMMA
    action_scale: float = DEFAULT_GRPO_ACTION_SCALE


class GRPORolloutBatch(NamedTuple):
    """Container for group-sampled rollouts."""

    states: jax.Array  # [B, obs_dim]
    actions: jax.Array  # [B, G, act_dim]
    old_log_probs: jax.Array  # [B, G]
    advantages: jax.Array  # [B, G]
    returns: jax.Array  # [B, G]
    mean_vel: float
    mean_height: float
    mean_torque: float


class HumanoidGRPOTrainer:
    """Group Relative Policy Optimization (GRPO) Trainer for Humanoid WBC."""

    def __init__(
        self,
        env: HumanoidResidualWBCEnv,
        policy: HumanoidResidualPolicy,
        ref_policy: HumanoidResidualPolicy,
        config: HumanoidGRPOConfig | None = None,
    ):
        self.env = env
        self.policy = policy
        self.ref_policy = ref_policy
        self.config = config or HumanoidGRPOConfig()

        self.optimizer = nnx.Optimizer(
            self.policy,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(
                    learning_rate=self.config.learning_rate, weight_decay=1e-4
                ),
            ),
            wrt=nnx.Param,
        )

        self._step_fn = jax.jit(self.env.step)
        self._reset_fn = jax.jit(self.env.reset)

    def sample_group_rollouts(
        self, rng: jax.Array
    ) -> Tuple[GRPORolloutBatch, jax.Array]:
        """Sample G candidate residual rollouts for each parallel environment state."""
        B = self.config.num_envs
        G = self.config.group_size
        H = self.config.rollout_horizon

        # 1. Reset environments to obtain batch of initial states
        obs_list = []
        state_list = []
        for i in range(B):
            rng, reset_rng = jax.random.split(rng)
            st = self._reset_fn(reset_rng)
            obs_list.append(st.obs)
            state_list.append(st)

        batch_obs = jnp.stack(obs_list, axis=0)  # [B, obs_dim]

        # 2. For each environment state, sample G candidate actions from policy
        actions_bg = []
        old_log_probs_bg = []
        returns_bg = []

        all_vels = []
        all_heights = []
        all_torques = []

        for b in range(B):
            obs_b = batch_obs[b]  # [obs_dim]
            actions_g = []
            log_probs_g = []
            returns_g = []

            for g in range(G):
                rng, act_rng = jax.random.split(rng)
                act, lp = self.policy.sample_action(act_rng, obs_b)
                actions_g.append(act)
                log_probs_g.append(lp)

                # Rollout candidate action through WBC and environment over horizon H
                sim_state = state_list[b]
                cum_return = 0.0
                gamma_pow = 1.0

                for h in range(H):
                    sim_state = self._step_fn(sim_state, act)
                    r = float(sim_state.reward)
                    cum_return += gamma_pow * r
                    gamma_pow *= self.config.gamma

                    all_vels.append(
                        float(sim_state.metrics.get("forward_vel", 0.0))
                    )
                    all_heights.append(
                        float(sim_state.metrics.get("torso_height", 0.85))
                    )
                    all_torques.append(
                        float(sim_state.metrics.get("mean_torque", 0.0))
                    )

                    if float(sim_state.done) > 0.5:
                        break

                returns_g.append(cum_return)

            actions_bg.append(jnp.stack(actions_g, axis=0))
            old_log_probs_bg.append(jnp.array(log_probs_g))
            returns_bg.append(jnp.array(returns_g))

        actions = jnp.stack(actions_bg, axis=0)  # [B, G, act_dim]
        old_log_probs = jnp.stack(old_log_probs_bg, axis=0)  # [B, G]
        returns = jnp.stack(returns_bg, axis=0)  # [B, G]

        # 3. Compute Group Relative Advantages: A_i^(g) = (R_i^(g) - mean(R_i)) / (std(R_i) + eps)
        mean_R = jnp.mean(returns, axis=-1, keepdims=True)  # [B, 1]
        std_R = jnp.std(returns, axis=-1, keepdims=True)  # [B, 1]
        advantages = (returns - mean_R) / (std_R + 1e-6)  # [B, G]

        batch = GRPORolloutBatch(
            states=batch_obs,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            mean_vel=float(jnp.mean(jnp.array(all_vels))),
            mean_height=float(jnp.mean(jnp.array(all_heights))),
            mean_torque=float(jnp.mean(jnp.array(all_torques))),
        )
        return batch, rng

    def train_step(
        self, batch: GRPORolloutBatch
    ) -> Tuple[float, float, float]:
        """Perform one GRPO optimization step on the sampled group rollout batch."""
        B, G, act_dim = batch.actions.shape
        flat_obs = jnp.repeat(batch.states, G, axis=0)  # [B*G, obs_dim]
        flat_actions = batch.actions.reshape(B * G, act_dim)  # [B*G, act_dim]
        flat_old_lp = batch.old_log_probs.reshape(B * G)  # [B*G]
        flat_adv = batch.advantages.reshape(B * G)  # [B*G]

        clip_eps = self.config.clip_eps
        beta_kl = self.config.beta_kl

        def loss_fn(model: HumanoidResidualPolicy):
            # 1. Evaluate log prob under updated policy
            new_log_prob = model.evaluate_log_prob(flat_obs, flat_actions)

            # 2. Importance sampling ratio: rho = exp(log pi_new - log pi_old)
            log_ratio = jnp.clip(new_log_prob - flat_old_lp, -10.0, 10.0)
            ratio = jnp.exp(log_ratio)

            # 3. Clipped surrogate loss
            surr1 = ratio * flat_adv
            surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * flat_adv
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # 4. KL divergence penalty against bootstrapped reference policy
            kl_div = jnp.mean(model.evaluate_kl(flat_obs, self.ref_policy))
            total_loss = policy_loss + beta_kl * kl_div

            # Approximate clip fraction
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
            )
            return total_loss, (policy_loss, kl_div, clip_fraction)

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (p_loss, kl_val, clip_frac)), grads = grad_fn(self.policy)
        self.optimizer.update(self.policy, grads)

        return float(total_loss), float(kl_val), float(clip_frac)


def train_humanoid_grpo(
    config: HumanoidGRPOConfig | None = None,
    output_dir: str = "checkpoints/humanoid_grpo",
    bootstrap_epochs: int = 5,
    demos_path: str = "",
    robot: str | None = None,
) -> Tuple[HumanoidResidualPolicy, List[dict]]:
    """Run full GRPO training loop for Humanoid Residual WBC."""
    config = config or HumanoidGRPOConfig()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(
        "🚀 Humanoid Residual WBC Training with Group Relative Policy Optimization (GRPO)"
    )
    if robot:
        print(f"   Robot Model:              {robot}")
    print(f"   Candidate Group Size (G): {config.group_size}")
    print(f"   Parallel Envs (B):        {config.num_envs}")
    print(f"   Rollout Horizon (H):      {config.rollout_horizon}")
    print(f"   Total Iterations:         {config.num_iterations}")
    print(f"   KL Penalty Weight:        {config.beta_kl}")
    print(f"   Output Directory:         {os.path.abspath(output_dir)}")
    print("=" * 80)

    # 1. Environment and Policy Setup
    env_config = HumanoidResidualWBCConfig(
        robot=robot, action_scale=config.action_scale
    )
    env = HumanoidResidualWBCEnv(config=env_config)

    # 2. Offline Bootstrapping (Warmstart via Behavioral Cloning)
    bc_ckpt = os.path.join(output_dir, "bootstrapped_policy.npz")
    policy, _ = bootstrap_policy(
        demos_path=demos_path,
        epochs=bootstrap_epochs,
        output_path=bc_ckpt,
        obs_dim=env.observation_size,
        act_dim=env.action_size,
    )

    # Frozen Reference Policy for KL divergence penalty
    ref_rng = nnx.Rngs(params=jax.random.PRNGKey(999))
    ref_policy = HumanoidResidualPolicy(
        obs_dim=env.observation_size,
        act_dim=env.action_size,
        rngs=ref_rng,
    )
    # Copy initial bootstrapped weights to reference policy
    nnx.update(ref_policy, nnx.state(policy))

    trainer = HumanoidGRPOTrainer(
        env=env,
        policy=policy,
        ref_policy=ref_policy,
        config=config,
    )

    # 3. GRPO Training Loop
    rng = jax.random.PRNGKey(101)
    history = []
    t_start = time.time()

    print("\n" + "=" * 80)
    print("=== Starting Humanoid GRPO Training Loop ===")
    print(
        f"{'Iter':>5} | {'Mean Ret':>9} | {'Max Ret':>9} | {'Fwd Vel':>8} | {'Height':>7} | {'Torque':>7} | {'Loss':>8} | {'KL Div':>8} | {'Clip%':>6} | {'Time':>6}"
    )
    print("-" * 80)

    for it in range(1, config.num_iterations + 1):
        t_it = time.time()

        # Step A: Collect Group Rollouts
        rollout_batch, rng = trainer.sample_group_rollouts(rng)

        # Step B: Multi-Epoch Optimization
        loss_val, kl_val, clip_frac = 0.0, 0.0, 0.0
        for _ in range(config.num_epochs_per_iter):
            loss_val, kl_val, clip_frac = trainer.train_step(rollout_batch)

        mean_ret = float(jnp.mean(rollout_batch.returns))
        max_ret = float(jnp.max(rollout_batch.returns))
        dt = time.time() - t_it

        metrics = {
            "iteration": it,
            "mean_return": mean_ret,
            "max_return": max_ret,
            "forward_vel": rollout_batch.mean_vel,
            "torso_height": rollout_batch.mean_height,
            "mean_torque": rollout_batch.mean_torque,
            "loss": loss_val,
            "kl_div": kl_val,
            "clip_frac": clip_frac,
            "time": dt,
        }
        history.append(metrics)

        print(
            f"{it:>5d} | {mean_ret:>9.2f} | {max_ret:>9.2f} | {rollout_batch.mean_vel:>8.3f} | {rollout_batch.mean_height:>7.3f} | {rollout_batch.mean_torque:>7.2f} | {loss_val:>8.4f} | {kl_val:>8.4f} | {clip_frac * 100:>5.1f}% | {dt:>5.1f}s"
        )

    total_time = time.time() - t_start
    print("-" * 80)
    print(
        f"✓ GRPO Training Completed in {total_time:.2f}s ({total_time / config.num_iterations:.2f}s / iter)"
    )

    # Save final model weights
    final_ckpt = os.path.join(output_dir, "final_grpo_policy.npz")
    flat_state = {
        str(k): np.array(v)
        for k, v in dict(nnx.state(policy).flat_state()).items()
    }
    np.savez(final_ckpt, **flat_state)
    print(f"✓ Final trained policy saved to '{final_ckpt}'.")

    return policy, history


def main():
    parser = argparse.ArgumentParser(
        description="Train Humanoid Residual WBC with GRPO"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of GRPO iterations",
    )
    parser.add_argument(
        "--group_size", type=int, default=4, help="Candidate group size G"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4,
        help="Parallel batch size B of environments",
    )
    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=8,
        help="Horizon H for candidate rollouts",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--beta_kl", type=float, default=0.05, help="KL penalty weight"
    )
    parser.add_argument(
        "--bootstrap_epochs",
        type=int,
        default=5,
        help="Offline BC warmstart epochs",
    )
    parser.add_argument(
        "--demos_path",
        type=str,
        default="",
        help="Path to HDF5 demonstration dataset",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="",
        help="Robot name or path to XML/URDF/YAML definition (e.g. 'g1', 'atlas', 'unitree_g1')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/humanoid_grpo",
        help="Checkpoint directory",
    )
    args = parser.parse_args()

    config = HumanoidGRPOConfig(
        group_size=args.group_size,
        num_envs=args.num_envs,
        rollout_horizon=args.rollout_horizon,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        beta_kl=args.beta_kl,
    )

    train_humanoid_grpo(
        config=config,
        output_dir=args.output_dir,
        bootstrap_epochs=args.bootstrap_epochs,
        demos_path=args.demos_path,
        robot=args.robot if args.robot else None,
    )


if __name__ == "__main__":
    import numpy as np

    main()
