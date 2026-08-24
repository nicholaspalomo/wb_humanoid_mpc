"""
Cartpole Reinforcement Learning Tutorial using Google Brax PPO & MuJoCo MJX.
Demonstrates:
  1. Defining an MJCF model in MuJoCo.
  2. Subclassing brax.envs.base.PipelineEnv for GPU-accelerated MJX physics.
  3. Training a continuous PPO actor-critic policy via brax.training.agents.ppo.
  4. Visualizing training progress with high-contrast animated GIFs and live 3D VNC playback.
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional

# Configure OpenGL backend: GLX for interactive VNC/GUI window, OSMesa for headless background rendering
if "--vnc" in sys.argv or "--gui" in sys.argv:
    os.environ["MUJOCO_GL"] = "glx"
elif "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

from brax.envs.base import PipelineEnv, State
from brax.training.agents.ppo import train as ppo
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx
import numpy as np
from PIL import Image

import humanoid_learning.examples

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
# 2. Brax Pipeline Environment for Cartpole
# ==============================================================================
class CartpoleBraxEnv(PipelineEnv):
    """Cartpole Environment inheriting from Brax PipelineEnv for GPU-accelerated MJX."""

    def __init__(self, **kwargs):
        self.mj_model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        mjx_model = mjx.put_model(self.mj_model)
        super().__init__(sys=mjx_model, n_frames=1, **kwargs)

    @property
    def action_size(self) -> int:
        return self.sys.nu

    @property
    def observation_size(self) -> int:
        return 5

    def reset(self, rng: jax.Array) -> State:
        """Resets the cartpole state with initial tilt perturbation."""
        rng_pos, rng_pole = jax.random.split(rng)
        init_x = jax.random.uniform(rng_pos, (), minval=-0.2, maxval=0.2)
        init_theta = jax.random.uniform(rng_pole, (), minval=-0.4, maxval=0.4)

        qpos = jnp.array([init_x, init_theta])
        qvel = jnp.zeros(2)
        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward = jnp.zeros(())
        done = jnp.zeros(())
        metrics = {
            "reward": reward,
            "upright": jnp.clip(jnp.cos(init_theta), 0.0, 1.0),
            "cart_x": init_x,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Applies action and integrates physics forward using MJX."""
        ctrl = jnp.clip(action, -1.0, 1.0)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        obs = self._get_obs(pipeline_state)
        done = self._is_done(pipeline_state)
        reward = self._compute_reward(pipeline_state, ctrl, done)

        metrics = {
            "reward": reward,
            "upright": jnp.clip(jnp.cos(pipeline_state.qpos[1]), 0.0, 1.0),
            "cart_x": pipeline_state.qpos[0],
        }
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    def _get_obs(self, pipeline_state: mjx.Data) -> jax.Array:
        cart_x = pipeline_state.qpos[0]
        theta = pipeline_state.qpos[1]
        cart_x_dot = pipeline_state.qvel[0]
        theta_dot = pipeline_state.qvel[1]
        return jnp.array(
            [cart_x, jnp.sin(theta), jnp.cos(theta), cart_x_dot, theta_dot]
        )

    def _compute_reward(
        self, pipeline_state: mjx.Data, ctrl: jax.Array, done: jax.Array
    ) -> jax.Array:
        cart_x = pipeline_state.qpos[0]
        theta = pipeline_state.qpos[1]
        theta_dot = pipeline_state.qvel[1]

        upright_reward = jnp.clip(jnp.cos(theta), 0.0, 1.0)
        center_penalty = 0.05 * jnp.square(cart_x)
        spin_penalty = 0.005 * jnp.square(theta_dot)
        ctrl_penalty = 0.005 * jnp.sum(jnp.square(ctrl))
        fall_penalty = jnp.where(done > 0.5, -1.0, 0.0)

        return (
            upright_reward
            - center_penalty
            - spin_penalty
            - ctrl_penalty
            + fall_penalty
        )

    def _is_done(self, pipeline_state: mjx.Data) -> jax.Array:
        cart_x = pipeline_state.qpos[0]
        theta = pipeline_state.qpos[1]
        out_of_bounds = jnp.abs(cart_x) > 2.2
        fallen = jnp.cos(theta) < 0.2
        return jnp.where(out_of_bounds | fallen, 1.0, 0.0)


# ==============================================================================
# 3. Rendering & Visualization Helper
# ==============================================================================
def render_policy_rollout(
    mj_model: mujoco.MjModel, inference_fn, output_gif_path: str
) -> float:
    """Rolls out the trained policy in MuJoCo dynamics and renders an animated GIF."""
    data = mujoco.MjData(mj_model)

    # Initial state with perturbation tilt
    data.qpos[0] = 0.0
    data.qpos[1] = 0.35
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
    rng = jax.random.PRNGKey(0)

    for step in range(num_frames):
        cart_x = float(data.qpos[0])
        theta = float(data.qpos[1])
        cart_xs.append(cart_x)
        thetas.append(theta)

        obs = np.array(
            [cart_x, np.sin(theta), np.cos(theta), data.qvel[0], data.qvel[1]],
            dtype=np.float32,
        )

        rng, act_rng = jax.random.split(rng)
        action, _ = inference_fn(obs, act_rng)
        act_scalar = float(np.array(action)[0])

        data.ctrl[0] = np.clip(act_scalar, -1.0, 1.0)
        mujoco.mj_step(mj_model, data)

        total_reward += np.cos(theta) - 0.05 * (cart_x**2)

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

            ax.plot([-2.4, 2.4], [0, 0], color="#475569", linewidth=4)
            cart_patch = plt.Rectangle(
                (cx - 0.2, -0.06), 0.4, 0.12, color="#38bdf8", zorder=3
            )
            ax.add_patch(cart_patch)
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

    os.makedirs(
        os.path.dirname(os.path.abspath(output_gif_path)), exist_ok=True
    )
    if frames:
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=33,
            loop=0,
        )
    return total_reward


def plot_training_curves(
    metrics_history: Dict[str, list], output_plot_path: str
):
    """Saves updated reward and loss curves to a PNG image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Evaluation Return / Score Plot
    eval_steps = metrics_history.get("eval_steps", [])
    eval_scores = metrics_history.get("eval_scores", [])
    if eval_steps and eval_scores:
        ax1.plot(
            eval_steps,
            eval_scores,
            "o-",
            color="#2563eb",
            linewidth=2,
        )
    ax1.set_title(
        "Evaluation Episode Return",
        fontsize=12,
        fontweight="bold",
        color="#1e293b",
    )
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    loss_steps = metrics_history.get("loss_steps", [])
    loss_vals = metrics_history.get("loss", [])
    if loss_steps and loss_vals:
        ax2.plot(
            loss_steps,
            loss_vals,
            "s-",
            color="#dc2626",
            linewidth=2,
        )
    ax2.set_title(
        "PPO Total Loss", fontsize=12, fontweight="bold", color="#1e293b"
    )
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(
        os.path.dirname(os.path.abspath(output_plot_path)), exist_ok=True
    )
    plt.savefig(output_plot_path, dpi=120)
    plt.close(fig)


# ==============================================================================
# 4. Main Brax PPO Training Loop with Visualization Callback
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Cartpole Brax MJX PPO Training with Visualization"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=64,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=50_000,
        help="Total environment steps",
    )
    parser.add_argument(
        "--num_evals",
        type=int,
        default=10,
        help="Number of evaluation checkpoints",
    )
    parser.add_argument(
        "--episode_length", type=int, default=200, help="Episode length"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cartpole_renders",
        help="Output directory",
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
    workspace_dir = os.environ.get(
        "BUILD_WORKSPACE_DIRECTORY", os.path.abspath(".")
    )
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(workspace_dir, args.output_dir)

    print("=" * 70)
    print("🚀 Google Brax & MuJoCo MJX Cartpole PPO Training")
    print(f"   Parallel Envs:       {args.num_envs}")
    print(f"   Total Timesteps:     {args.total_timesteps:,}")
    print(f"   Evaluation Epochs:   {args.num_evals}")
    print(f"   Episode Length:      {args.episode_length}")
    print(f"   Output Directory:    {os.path.abspath(args.output_dir)}")
    if args.vnc or args.gui:
        print("   Live 3D Viewer:      ENABLED (Opening in VNC / GUI Display)")
    print("=" * 70)

    env = CartpoleBraxEnv()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize live passive 3D MuJoCo viewer if requested
    viewer = None
    eval_data = None
    if args.vnc or args.gui:
        try:
            import mujoco.viewer

            if args.vnc and "DISPLAY" not in os.environ:
                os.environ["DISPLAY"] = ":99"
            eval_data = mujoco.MjData(env.mj_model)
            eval_data.qpos[0] = 0.0
            eval_data.qpos[1] = 0.35
            eval_data.qvel[:] = 0.0
            mujoco.mj_forward(env.mj_model, eval_data)
            viewer = mujoco.viewer.launch_passive(env.mj_model, eval_data)
            viewer.sync()
            print("🖥️ Live 3D MuJoCo Viewer initialized on display :99!")
        except Exception as e:
            print(f"⚠️ Could not launch live 3D viewer: {e}")
            viewer = None

    metrics_history = {
        "eval_steps": [],
        "eval_scores": [],
        "loss_steps": [],
        "loss": [],
    }

    step_count = 0
    t_last = time.time()

    def progress_callback(num_steps: int, metrics: Dict[str, float]):
        nonlocal step_count, t_last
        step_count += 1
        dt = time.time() - t_last
        t_last = time.time()

        eval_reward = float(
            metrics.get("eval/episode_reward", metrics.get("eval/reward", 0.0))
        )
        loss_val = float(metrics.get("training/total_loss", 0.0))

        metrics_history["eval_steps"].append(num_steps)
        metrics_history["eval_scores"].append(eval_reward)
        if loss_val != 0.0:
            metrics_history["loss_steps"].append(num_steps)
            metrics_history["loss"].append(loss_val)

        pole_icon = (
            "🟢 UP"
            if eval_reward > 100.0
            else ("🟡 BALANCING" if eval_reward > 20.0 else "🔴 FALLEN")
        )
        print(
            f"Step {num_steps:>6d} | Eval Reward: {eval_reward:>+7.2f} | Loss: {loss_val:.4f} | {pole_icon} | {dt*1000:.1f}ms"
        )

        plot_path = os.path.join(args.output_dir, "training_curves.png")
        plot_training_curves(metrics_history, plot_path)

    print("\n🏁 Launching Brax PPO Training Loop...\n")

    # Run Brax PPO training
    make_inference_fn, params, final_metrics = ppo.train(
        environment=env,
        num_timesteps=args.total_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=args.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=16,
        num_updates_per_batch=4,
        discounting=args.gamma,
        learning_rate=args.lr,
        entropy_cost=1e-2,
        num_envs=args.num_envs,
        batch_size=32,
        seed=42,
        progress_fn=progress_callback,
    )

    inference_fn = make_inference_fn(params)

    # Render final policy rollout
    final_gif_path = os.path.join(args.output_dir, "cartpole_latest.gif")
    eval_score = render_policy_rollout(
        env.mj_model, inference_fn, final_gif_path
    )
    print(
        f"\n🎬 [Final Visualization Saved] -> {final_gif_path} (Score: {eval_score:.1f})"
    )

    # Live 3D playback in VNC if viewer is active
    if viewer is not None and viewer.is_running() and eval_data is not None:
        print("🖥️ Running live playback in 3D MuJoCo viewer...")
        with viewer.lock():
            eval_data.qpos[0] = 0.0
            eval_data.qpos[1] = 0.35
            eval_data.qvel[:] = 0.0
            mujoco.mj_forward(env.mj_model, eval_data)
        viewer.sync()

        rng_eval = jax.random.PRNGKey(123)
        for _ in range(150):
            if not viewer.is_running():
                break
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
            rng_eval, act_rng = jax.random.split(rng_eval)
            action, _ = inference_fn(obs_eval, act_rng)
            with viewer.lock():
                eval_data.ctrl[0] = np.clip(
                    float(np.array(action)[0]), -1.0, 1.0
                )
                mujoco.mj_step(env.mj_model, eval_data)
            viewer.sync()
            time.sleep(0.015)

    if viewer is not None:
        try:
            if viewer.is_running():
                viewer.close()
        except Exception:
            pass
        viewer = None

    print("\n" + "=" * 70)
    print("🎉 Brax PPO Training Completed Successfully!")
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

    # Clean exit for VNC/GUI to prevent GLFW/Mesa background thread teardown crash
    if args.vnc or args.gui:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
