# Whole-Body Humanoid MPC

This repository contains a Whole-Body Nonlinear Model Predictive Controller (NMPC) for humanoid loco-manipulation control. This approach directly optimizes through the **full-order torque-level dynamics in real time** to generate dynamic humanoid behaviors, building upon an extended and updated version of [OCS2](https://github.com/leggedrobotics/ocs2) integrated natively into a **Bazel monorepo**.

**Interactive Velocity and Base Height Control via Joystick:**

![vokoscreenNG-2025-12-21_20-35-31-ezgif com-optimize](https://github.com/user-attachments/assets/daf374ba-fe82-469d-9270-63d18a51bb53)

---

## 🤖 MPC Formulations

### Centroidal Dynamics MPC
The centroidal MPC optimizes over the **whole-body kinematics** and the center of mass dynamics, with a choice to use either a Single Rigid Body Dynamics (SRBD) model or the full centroidal dynamics. This approach extends the OCS2 centroidal formulation by generalizing costs and constraints to 6-DoF contacts and arbitrary end-effectors. For theoretical background, see [Sleiman et al., *A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation*](https://arxiv.org/abs/2103.00946).

### Whole-Body Dynamics MPC
The **whole-body dynamics** MPC optimizes directly over contact forces, joint accelerations, and joint torques across the planning horizon. For details on the optimization and dynamic consistency formulation, see [Galliker et al., *Bipedal Locomotion with Nonlinear Model Predictive Control: Online Gait Generation using Whole-Body Dynamics*](http://ames.caltech.edu/galliker2022bipedal.pdf).

---

## 🦾 Supported Robot Models

| Robot Platform | Centroidal NMPC | Whole-Body NMPC | MuJoCo Physics Sim | RViz Dummy Sim |
|---|:---:|:---:|:---:|:---:|
| **Unitree G1** | ✅ | ✅ | ✅ | ✅ |
| **Unitree R1** | ✅ | — | ✅ | ✅ |
| **DRC Atlas** | ✅ | — | ✅ | ✅ |
| **1X Neo** | *Coming Soon* | *Coming Soon* | *Coming Soon* | *Coming Soon* |

![Screencast2024-12-16180254-ezgif com-optimize(3)](https://github.com/user-attachments/assets/d4b1f0da-39ca-4ce1-b53c-e1d040abe1be)

---

## 🚀 Getting Started

### 1. Repository Setup

```bash
git clone https://github.com/1x-technologies/wb-humanoid-mpc.git
cd wb-humanoid-mpc
```

> **Note:** The repository uses **Bazel 9.x** with `bzlmod` for hermetic dependency management. ROS 2 packages and dependencies are built directly within the Bazel workspace.

### 2. Environment Setup

The recommended way to develop and run the simulation is using the provided Docker container.

<details>
<summary><b>Option A: VS Code Dev Containers (Recommended)</b></summary>

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension in VS Code.
2. Open the repository in VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS), and select **Dev Containers: Reopen in Container**.
3. Once the container builds, the environment is automatically set up.

</details>

<details>
<summary><b>Option B: Docker Compose / Remote SSH (Antigravity / Cursor / Terminal)</b></summary>

1. Start the container in detached mode:
   ```bash
   docker compose up -d --build
   ```
2. Attach a terminal into the container:
   ```bash
   docker compose exec app bash
   ```
3. (Optional) Run `./docker/image_build.bash` and `./docker/launch_wb_mpc.bash` helper scripts.

</details>

<details>
<summary><b>Option C: Local Installation (Ubuntu 24.04 / ROS 2 Jazzy)</b></summary>

Ensure **ROS 2 Jazzy** is installed on your machine. Then install system packages and Bazelisk:
```bash
envsubst < dependencies.txt | xargs sudo apt-get install -y --no-install-recommends
curl -sSL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
make install-hooks
```

</details>

---

## 🛠️ Build & Test Commands

The repository includes a top-level `Makefile` for streamlined building and testing:

```bash
# Build all Bazel targets across the monorepo
make build-all

# Run all unit and integration tests
make test-all

# Run code formatters (Clang-Format, Black, whitespace)
make format

# Run linter and verify IFTTT directives
make lint

# Clean build artifacts
make clean        # Incremental clean
make clean-all    # Deep clean including external caches
```

---

## 🖥️ Launching Simulations

### Visualization Options
- **Local Linux:** GUI windows (MuJoCo / RViz / Controller GUI) render via X11 forwarding.
- **macOS / Remote SSH:** Use the `-vnc` targets to stream the desktop directly to your browser. Navigate to **`http://localhost:6080/vnc.html`** and click **Connect**. See the [Visualization Guide](.devcontainer/VISUALIZATION.md) for full details.

### Launch Targets

#### 1. Unitree G1
```bash
# Centroidal MPC
make launch-g1-sim-vnc          # MuJoCo Physics Sim (Browser / macOS / Remote)
make launch-g1-dummy-sim-vnc    # RViz Dummy Sim (Browser / macOS / Remote)
make launch-g1-sim              # MuJoCo Physics Sim (Native X11)
make launch-g1-dummy-sim        # RViz Dummy Sim (Native X11)

# Whole-Body Dynamics MPC
make launch-wb-g1-sim-vnc       # MuJoCo Physics Sim (Browser / macOS / Remote)
make launch-wb-g1-dummy-sim-vnc # RViz Dummy Sim (Browser / macOS / Remote)
make launch-wb-g1-sim           # MuJoCo Physics Sim (Native X11)
make launch-wb-g1-dummy-sim     # RViz Dummy Sim (Native X11)
```

#### 2. Unitree R1
```bash
make launch-r1-sim-vnc         # MuJoCo Physics Sim (Browser / macOS / Remote)
make launch-r1-dummy-sim-vnc   # RViz Dummy Sim (Browser / macOS / Remote)
make launch-r1-sim             # MuJoCo Physics Sim (Native X11)
make launch-r1-dummy-sim       # RViz Dummy Sim (Native X11)
make launch-r1-sandbox-vnc     # Interactive URDF Model Viewer
```

#### 3. DRC Atlas
```bash
make launch-drc-atlas-sim-vnc       # MuJoCo Physics Sim (Browser / macOS / Remote)
make launch-drc-atlas-dummy-sim-vnc # RViz Dummy Sim (Browser / macOS / Remote)
make launch-drc-atlas-sim           # MuJoCo Physics Sim (Native X11)
make launch-drc-atlas-dummy-sim     # RViz Dummy Sim (Native X11)
make launch-drc-atlas-sandbox-vnc   # Interactive URDF Model Viewer
```

> **Cleanup Tip:** Run `make kill-sims` at any time to clean up any orphaned simulation, publisher, or ROS 2 background processes.

---

## 🕹️ Interactive Controls & Simulation Lifecycle

### Supervisory Finite State Machine (FSM)
Simulations launch in a safe **Zero-Torque Mode** suspended on a virtual gantry so the robot settles safely while the MPC solver initializes:

| FSM State / Mode | Description |
|---|---|
| `ZERO_TORQUE` | Passive spawn state; solver warms up without commanding torques. |
| `JOINT_PD` | Joint-space proportional-derivative posture control tracking nominal stance. |
| `WB_MPC` / `MPC_ACTIVE` | Active Whole-Body / Centroidal MPC solver closed-loop control. |
| `LOCK_GANTRY` / `UNLOCK_GANTRY` | Suspends or releases the virtual gantry holding the floating base. |

State transitions are managed natively over ROS 2 topics:
- **Command Topic:** `/humanoid/fsm_command` (`std_msgs/msg/String`)
- **State Topic:** `/humanoid/fsm_state` (`std_msgs/msg/String`, Transient Local QoS)

### Teleoperation & Root Height Control
- Use the **Robot Base Controller GUI** or connect an **Xbox Controller** to command velocity vectors ($v_x, v_y, \omega_z$).
- The **Height Slider** controls the **Virtual Gantry Height** when locked (allowing you to lift and lower the robot above the ground) and sets the **Desired Pelvis Height** when walking.

![robot_remote_control](https://github.com/user-attachments/assets/779be1da-97a1-4d0c-8f9b-b9d2df88384f)

### MuJoCo 3D Viewer Hotkeys
When focused in the MuJoCo simulation viewport, use these keyboard shortcuts:

| Key | Action |
|:---:|---|
| **`1`** | Toggle **Visual Meshes** on/off (hides STL shells to inspect collision primitives) |
| **`2`** | Toggle **Collision Primitives** on/off |
| **`0`** | Toggle **Floor / Ground Plane** on/off |
| **`t`** | Toggle **Model Transparency** (sets 30% alpha for x-ray inspection) |
| **`c`** | Toggle **Contact Points** visualization |
| **`f`** | Toggle **Contact Force** 3D vectors |
| **`m`** | Toggle **Center of Mass (CoM)** indicator |
| **`i`** | Toggle **Link Inertia Ellipsoids** |
| **`h`** | Toggle **Convex Hulls** |
| **`p`** | Print hotkey cheatsheet to the console |

---

## 🎮 Interactive Jupyter Control Dashboard

Launch the unified browser-based teleoperation and diagnostics dashboard:

```bash
make jupyter
```

Open **`http://localhost:8888`** and load [`notebooks/humanoid_control_dashboard.ipynb`](notebooks/humanoid_control_dashboard.ipynb).

### Dashboard Highlights:
1. **Simulation Process Manager:** One-click startup, monitoring, and shutdown of any robot model and solver backend with live terminal logs.
2. **Virtual Joystick:** Directional D-pad and continuous analog velocity sliders streaming commands at 25 Hz.
3. **Angular Center of Mass (aCOM) Studio:** Train JAX/SIREN networks on Centroidal Momentum Matrices and export static C++ headers (`AngularCenterOfMassWeights.h`).
4. **Live Telemetry:** Real-time strip charts plotting base Euler angles, aCOM decoupling metrics, and ground reaction forces.

---

## 🧠 Reinforcement Learning with MuJoCo Playground (`humanoid_learning`)

The repository includes a GPU-accelerated RL and imitation learning pipeline built on **Google DeepMind's [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)**, **MJX**, **JAX**, and **Brax**.

<details>
<summary><b>GPU Training Setup & Hardware Prerequisites</b></summary>

### 1. Host Machine Prerequisites
Ensure your host machine has an NVIDIA driver and the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** installed.

Verify passthrough from the host:
```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

### 2. Starting with GPU Passthrough
```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
docker compose exec app bash
```

*(Non-GPU hosts automatically fall back to multi-threaded CPU execution).*

</details>

<details>
<summary><b>Training Commands & Trajectory Export</b></summary>

```bash
# Run RL unit and smoke tests
make test-rl

# Launch PPO Policy Training (MJX / Playground)
make train-rl
# Or with customized parameters:
bazel run //humanoid_learning/training:train_ppo -- --num_envs=4096 --total_timesteps=10000000

# Export recorded MPC rollouts to HDF5 demonstration datasets
make export-rollouts

# Behavioral Cloning (BC) imitation learning warmstart
make train-bc

# Export trained policy to ONNX format for C++ deployment
bazel run //humanoid_learning/export:export_onnx -- --output_path=models/humanoid_policy.onnx
```

</details>

---

## 📚 Citation

If you use Whole-Body Humanoid MPC in your academic research, please cite:

```bibtex
@misc{wholebodyhumanoidmpcweb,
   author = {Manuel Yves Galliker and Nicholas Palomo},
   title = {Whole-body Humanoid MPC: Realtime Physics-Based Procedural Loco-Manipulation Planning and Control},
   howpublished = {https://github.com/nicholaspalomo/wb_humanoid_mpc},
   year = {2026}
}
```

## 👥 Acknowledgements

This project was originally created by [Manuel Yves Galliker](https://github.com/manumerous) and open-sourced in collaboration with 1X Technologies.

Special thanks to the open-source robotics community:
- [OCS2](https://github.com/leggedrobotics/ocs2)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [HPIPM](https://github.com/giaf/hpipm)
- [MuJoCo & MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
