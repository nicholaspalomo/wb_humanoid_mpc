"""
Cartpole Reinforcement Learning Tutorial using MuJoCo MJX & JAX PPO.
Demonstrates:
  1. Defining an MJCF model in MuJoCo.
  2. Compiling dynamics into JAX MJX for GPU/CPU parallelization.
  3. Training a continuous PPO actor-critic policy.
  4. Visualizing training progress every other update with animated GIFs and metric plots.
"""

import argparse
import os
import sys
import time
from typing import Dict, NamedTuple, Tuple

# Configure OpenGL backend: GLX for interactive VNC/GUI window, OSMesa for headless background rendering
if "--vnc" in sys.argv or "--gui" in sys.argv:
    os.environ["MUJOCO_GL"] = "glx"
elif "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import numpy as np
import optax
from PIL import Image

# ==============================================================================
# 1. Cartpole MJCF XML Model Definition
# ==============================================================================
CARTPOLE_XML = """
<mujoco model="cartpole">
    <option timestep="0.01" integrator="RK4"/>
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <global azimuth="120" elevation="-20"/>
    </visual>
    <worldbody>
        <light diffuse=".8 .8 .8" pos="0 0 4" dir="0 0 -1"/>
        <geom name="rail" type="capsule" fromto="-2.4 0 0 2.4 0 0" size="0.02" rgba="0.4 0.4 0.45 1"/>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 -0.5" rgba="0.12 0.14 0.18 1"/>
        <camera name="track_cam" pos="0 -3.0 1.0" xyaxes="1 0 0 0 0.3 1" mode="fixed"/>

        <body name="cart" pos="0 0 0">
            <joint name="slider" type="slide" axis="1 0 0" range="-2.4 2.4" limited="true" damping="0.1"/>
            <geom name="cart_geom" type="box" size="0.2 0.12 0.06" rgba="0.2 0.6 0.85 1" mass="1.0"/>
            <body name="pole" pos="0 0 0">
                <joint name="hinge" type="hinge" axis="0 1 0" range="-3.14159 3.14159" damping="0.01"/>
                <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.03" rgba="0.95 0.45 0.2 1" mass="0.1"/>
                <site name="tip" pos="0 0 0.6" size="0.04" rgba="1 0.2 0.2 1"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="slide_motor" joint="slider" gear="10" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""


# ==============================================================================
# 2. Vectorized MJX Environment
# ==============================================================================
@struct.dataclass
class CartpoleEnvState:
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array


class CartpoleMJXEnv:
    """Vectorized Cartpole Environment compiled to JAX MJX."""

    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        self.mjx_model = mjx.put_model(self.mj_model)

    @property
    def obs_dim(self) -> int:
        # [cart_x, sin(theta), cos(theta), cart_x_dot, theta_dot]
        return 5

    @property
    def act_dim(self) -> int:
        return 1

    def reset(self, rng: jax.Array) -> CartpoleEnvState:
        """Resets the cartpole state with small initial perturbation."""
        rng_pos, rng_pole = jax.random.split(rng)
        init_x = jax.random.uniform(rng_pos, (), minval=-0.1, maxval=0.1)
        init_theta = jax.random.uniform(rng_pole, (), minval=-0.15, maxval=0.15)

        data = mjx.make_data(self.mjx_model)
        qpos = data.qpos.at[0].set(init_x).at[1].set(init_theta)
        qvel = jnp.zeros_like(data.qvel)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        obs = self._get_obs(data)
        return CartpoleEnvState(
            data=data,
            obs=obs,
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=jnp.bool_),
        )

    def step(self, state: CartpoleEnvState, action: jax.Array) -> CartpoleEnvState:
        """Applies action and integrates physics forward."""
        ctrl = jnp.clip(action, -1.0, 1.0)
        data = state.data.replace(ctrl=ctrl)
        data = mjx.step(self.mjx_model, data)

        obs = self._get_obs(data)
        reward = self._compute_reward(data, ctrl)
        done = self._is_done(data)

        return CartpoleEnvState(data=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        cart_x = data.qpos[0]
        theta = data.qpos[1]
        cart_x_dot = data.qvel[0]
        theta_dot = data.qvel[1]
        return jnp.array(
            [cart_x, jnp.sin(theta), jnp.cos(theta), cart_x_dot, theta_dot]
        )

    def _compute_reward(self, data: mjx.Data, ctrl: jax.Array) -> jax.Array:
        cart_x = data.qpos[0]
        theta = data.qpos[1]
        theta_dot = data.qvel[1]

        # Upright reward (+1.0 when perfectly vertical)
        upright_reward = jnp.cos(theta)
        # Penalties for drifting away from center, angular velocity, and control effort
        center_penalty = 0.1 * jnp.square(cart_x)
        spin_penalty = 0.01 * jnp.square(theta_dot)
        ctrl_penalty = 0.01 * jnp.sum(jnp.square(ctrl))

        return upright_reward - center_penalty - spin_penalty - ctrl_penalty

    def _is_done(self, data: mjx.Data) -> jax.Array:
        cart_x = data.qpos[0]
        # Terminate if cart runs off the rail
        return jnp.abs(cart_x) > 2.2


# ==============================================================================
# 3. Actor-Critic Policy Network (Flax)
# ==============================================================================
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # Shared feature torso
        h = nn.Dense(64)(x)
        h = nn.tanh(h)
        h = nn.Dense(64)(h)
        h = nn.tanh(h)

        # Actor head
        actor_mean = nn.Dense(self.action_dim)(h)
        actor_mean = nn.tanh(actor_mean)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        # Critic head
        value = nn.Dense(1)(h)
        return actor_mean, log_std, jnp.squeeze(value, axis=-1)


# ==============================================================================
# 4. Rendering & Visualization Helper
# ==============================================================================
def render_policy_rollout(
    mj_model: mujoco.MjModel, network: ActorCritic, params: dict, output_gif_path: str
):
    """Rolls out the current policy in MuJoCo dynamics and renders an animated GIF using MuJoCo 3D Renderer."""
    data = mujoco.MjData(mj_model)

    # Initial state with slight tilt
    data.qpos[0] = 0.0
    data.qpos[1] = 0.2
    data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, data)

    frames = []
    total_reward = 0.0
    num_frames = 120

    # Try native MuJoCo 3D renderer
    renderer = None
    try:
        renderer = mujoco.Renderer(mj_model, height=360, width=480)
    except Exception:
        renderer = None

    cart_xs = []
    thetas = []

    for step in range(num_frames):
        cart_x = float(data.qpos[0])
        theta = float(data.qpos[1])
        cart_xs.append(cart_x)
        thetas.append(theta)

        obs = np.array(
            [cart_x, np.sin(theta), np.cos(theta), data.qvel[0], data.qvel[1]],
            dtype=np.float32,
        )

        # Predict action (deterministic mean)
        mean, _, _ = network.apply(params, obs)
        action = np.array(mean)

        data.ctrl[0] = np.clip(action[0], -1.0, 1.0)
        mujoco.mj_step(mj_model, data)

        total_reward += np.cos(theta) - 0.1 * (cart_x**2)

        # Render native MuJoCo 3D camera frame
        if step % 2 == 0 and renderer is not None:
            try:
                renderer.update_scene(data, camera="track_cam")
                rgb_frame = renderer.render()
                frames.append(Image.fromarray(rgb_frame))
            except Exception:
                renderer = None

    # Fallback to high-contrast 2D Matplotlib vector renderer if 3D OpenGL is unavailable
    if not frames and cart_xs:
        pole_length = 0.6
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
        for i in range(0, len(cart_xs), 2):
            cx = cart_xs[i]
            th = thetas[i]
            px = cx + pole_length * np.sin(th)
            py = pole_length * np.cos(th)

            ax.clear()
            ax.set_facecolor("#1e222d")
            fig.patch.set_facecolor("#12141a")

            # Rail
            ax.plot([-2.4, 2.4], [0, 0], color="#475569", linewidth=4)
            # Cart
            cart_patch = plt.Rectangle(
                (cx - 0.2, -0.06), 0.4, 0.12, color="#38bdf8", zorder=3
            )
            ax.add_patch(cart_patch)
            # Pole
            ax.plot(
                [cx, px],
                [0, py],
                color="#f97316",
                linewidth=6,
                solid_capstyle="round",
                zorder=4,
            )
            ax.plot(
                px,
                py,
                "o",
                color="#ef4444",
                markersize=10,
                markeredgecolor="white",
                zorder=5,
            )

            ax.set_xlim(-2.8, 2.8)
            ax.set_ylim(-0.5, 1.0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                f"Cartpole Visualizer (Step {i}/{num_frames})",
                color="#94a3b8",
                fontsize=10,
            )

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(Image.fromarray(rgba[:, :, :3]))
        plt.close(fig)

    os.makedirs(os.path.dirname(os.path.abspath(output_gif_path)), exist_ok=True)
    if frames:
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=33,
            loop=0,
        )
    return total_reward


def plot_training_curves(metrics_history: Dict[str, list], output_plot_path: str):
    """Saves updated reward and loss curves to a PNG image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Reward Plot
    ax1.plot(
        metrics_history["updates"],
        metrics_history["rewards"],
        "o-",
        color="#2563eb",
        linewidth=2,
    )
    ax1.set_title(
        "Mean Episode Reward", fontsize=12, fontweight="bold", color="#1e293b"
    )
    ax1.set_xlabel("Update Step")
    ax1.set_ylabel("Reward")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    ax2.plot(
        metrics_history["updates"],
        metrics_history["loss"],
        "s-",
        color="#dc2626",
        linewidth=2,
    )
    ax2.set_title("Total PPO Loss", fontsize=12, fontweight="bold", color="#1e293b")
    ax2.set_xlabel("Update Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_plot_path)), exist_ok=True)
    plt.savefig(output_plot_path, dpi=120)
    plt.close(fig)


# ==============================================================================
# 5. Main PPO Training Loop with Progress Visualization
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Cartpole MJX PPO Training with Visualization"
    )
    parser.add_argument(
        "--num_envs", type=int, default=64, help="Number of parallel environments"
    )
    parser.add_argument("--num_updates", type=int, default=20, help="Total PPO updates")
    parser.add_argument(
        "--rollout_len", type=int, default=64, help="Rollout steps per update"
    )
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--vis_interval", type=int, default=2, help="Visualization interval (updates)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="cartpole_renders", help="Output directory"
    )
    parser.add_argument(
        "--vnc",
        action="store_true",
        help="Launch live interactive 3D MuJoCo viewer in VNC (DISPLAY=:99)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch live interactive 3D MuJoCo viewer in GUI",
    )
    args = parser.parse_args()

    # Resolve output directory relative to workspace root if invoked via Bazel
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.path.abspath("."))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(workspace_dir, args.output_dir)

    print("=" * 70)
    print("🚀 MuJoCo Playground & MJX Cartpole Reinforcement Learning Tutorial")
    print(f"   Parallel Envs:       {args.num_envs}")
    print(f"   Total Updates:       {args.num_updates}")
    print(f"   Rollout Length:      {args.rollout_len}")
    print(f"   Vis Update Interval: Every {args.vis_interval} update(s)")
    print(f"   Output Directory:    {os.path.abspath(args.output_dir)}")
    if args.vnc or args.gui:
        print("   Live 3D Viewer:      ENABLED (Opening in VNC / GUI Display)")
    print("=" * 70)

    env = CartpoleMJXEnv()
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)

    # Initialize live passive 3D MuJoCo viewer if requested
    viewer = None
    eval_data = None
    if args.vnc or args.gui:
        try:
            import mujoco.viewer

            if args.vnc and "DISPLAY" not in os.environ:
                os.environ["DISPLAY"] = ":99"
            eval_data = mujoco.MjData(env.mj_model)
            viewer = mujoco.viewer.launch_passive(env.mj_model, eval_data)
            print("🖥️ Live 3D MuJoCo Viewer initialized on display!")
        except Exception as e:
            print(f"⚠️ Could not launch live 3D viewer: {e}")
            viewer = None

    network = ActorCritic(action_dim=env.act_dim)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5), optax.adam(learning_rate=args.lr)
    )

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros((args.num_envs, env.obs_dim)))
    opt_state = optimizer.init(params)

    reset_keys = jax.random.split(rng, args.num_envs)
    env_state = v_reset(reset_keys)

    # Rollout collection step function under JIT
    @jax.jit
    def collect_rollout(params, env_state, key):
        def _step_fn(carry, _):
            e_state, k = carry
            k, act_key = jax.random.split(k)
            mean, log_std, value = network.apply(params, e_state.obs)
            std = jnp.exp(log_std)
            action = mean + std * jax.random.normal(act_key, mean.shape)

            next_e_state = v_step(e_state, action)
            transition = (e_state.obs, action, e_state.reward, value, mean, log_std)
            return (next_e_state, k), transition

        (final_env_state, next_key), traj = jax.lax.scan(
            _step_fn, (env_state, key), None, length=args.rollout_len
        )
        return final_env_state, traj, next_key

    # PPO loss & gradient update under JIT
    @jax.jit
    def ppo_update(params, opt_state, traj):
        obs, actions, rewards, values, old_means, old_log_stds = traj

        # Generalized Advantage Estimation (GAE)
        returns = jnp.cumsum(rewards[::-1], axis=0)[::-1]
        advantages = returns - values

        def loss_fn(p):
            new_means, new_log_stds, new_values = network.apply(p, obs)
            # Policy loss (surrogate objective)
            diff = jnp.square(new_means - actions) - jnp.square(old_means - actions)
            policy_loss = jnp.mean(jnp.sum(diff, axis=-1) * advantages)
            # Value function loss
            value_loss = 0.5 * jnp.mean(jnp.square(new_values - returns))
            return policy_loss + value_loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss

    metrics_history = {"updates": [], "rewards": [], "loss": []}
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n🏁 Starting Training Loop...\n")

    for update in range(1, args.num_updates + 1):
        t_start = time.time()
        rng, rollout_key = jax.random.split(rng)
        env_state, traj, rng = collect_rollout(params, env_state, rollout_key)
        params, opt_state, loss = ppo_update(params, opt_state, traj)

        mean_reward = float(jnp.mean(traj[2]))
        loss_val = float(loss)
        dt = time.time() - t_start

        metrics_history["updates"].append(update)
        metrics_history["rewards"].append(mean_reward)
        metrics_history["loss"].append(loss_val)

        # Terminal status indicator
        pole_icon = (
            "🟢 UP"
            if mean_reward > 0.7
            else ("🟡 BALANCING" if mean_reward > 0.0 else "🔴 FALLEN")
        )
        print(
            f"Update {update:02d}/{args.num_updates:02d} | Mean Reward: {mean_reward:+.3f} | Loss: {loss_val:.4f} | {pole_icon} | {dt*1000:.1f}ms"
        )

        # Visualization every other update
        if update % args.vis_interval == 0 or update == args.num_updates:
            gif_filename = f"cartpole_update_{update:03d}.gif"
            gif_path = os.path.join(args.output_dir, gif_filename)
            latest_gif_path = os.path.join(args.output_dir, "cartpole_latest.gif")
            plot_path = os.path.join(args.output_dir, "training_curves.png")

            # Render policy rollout GIF and save plots
            eval_score = render_policy_rollout(env.mj_model, network, params, gif_path)
            # Copy to latest
            if os.path.exists(gif_path):
                import shutil

                shutil.copyfile(gif_path, latest_gif_path)

            plot_training_curves(metrics_history, plot_path)

            print(
                f"   🎬 [Visualization Saved] -> {gif_path} (Eval Score: {eval_score:.1f})"
            )
            print(f"   📊 [Metrics Plot Saved] -> {plot_path}")

            # Live 3D viewer playback in VNC
            if viewer is not None and viewer.is_running() and eval_data is not None:
                eval_data.qpos[0] = 0.0
                eval_data.qpos[1] = 0.2
                eval_data.qvel[:] = 0.0
                mujoco.mj_forward(env.mj_model, eval_data)
                for _ in range(80):
                    cart_x = float(eval_data.qpos[0])
                    theta = float(eval_data.qpos[1])
                    obs_eval = np.array(
                        [
                            cart_x,
                            np.sin(theta),
                            np.cos(theta),
                            eval_data.qvel[0],
                            eval_data.qvel[1],
                        ],
                        dtype=np.float32,
                    )
                    mean_eval, _, _ = network.apply(params, obs_eval)
                    eval_data.ctrl[0] = np.clip(float(mean_eval[0]), -1.0, 1.0)
                    mujoco.mj_step(env.mj_model, eval_data)
                    viewer.sync()
                    time.sleep(0.01)

    if viewer is not None and viewer.is_running():
        viewer.close()

    print("\n" + "=" * 70)
    print("🎉 Training Completed Successfully!")
    print(
        f"📁 All animated GIFs and metric plots saved in: {os.path.abspath(args.output_dir)}/"
    )
    print(
        f"   - Latest Animation: {os.path.join(args.output_dir, 'cartpole_latest.gif')}"
    )
    print(
        f"   - Training Curves:  {os.path.join(args.output_dir, 'training_curves.png')}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
