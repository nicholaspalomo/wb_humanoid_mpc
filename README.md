# Whole-Body Humanoid MPC

This repository contains a Whole-Body Nonlinear Model Predictive Controller (NMPC) for humanoid loco-manipulation control. This approach enables to directly optimize through the **full-order torque-level dynamics in realtime** to generate a wide range of humanoid behaviors building up on an extended & updated version of [OCS2](https://github.com/leggedrobotics/ocs2) integrated natively into a **Bazel monorepo**.

**Interactive Velocity and Base Height Control via Joystick:**

![vokoscreenNG-2025-12-21_20-35-31-ezgif com-optimize](https://github.com/user-attachments/assets/daf374ba-fe82-469d-9270-63d18a51bb53)


It contains the following hardware platform agnostic MPC formulations:

### Centroidal Dynamics MPC
The centroidal MPC optimizes over the **whole-body kinematics** and the center of mass dynamics, with a choice to either use a single rigid body model or the full centroidal dynamics. This specific approach builds up on the centroidal model in OCS2 by generalizing costs and constraints to a 6 DoF contact among others. A concise explanation of the OCS2 centroidal model can be found in [Sleiman et. al., A Unified MPC Framework for Whole-Body Dynamic Locomotion and Manipulation](https://arxiv.org/abs/2103.00946).

### Whole-Body Dynamics MPC
The **whole-body dynamics** MPC optimizes over the contact forces and joint accelerations with the option to compute the joint torques for each step planned across the horizon. The most relevant information on the chosen approach can currently be found in [Galliker et al., Bipedal Locomotion with Nonlinear Model Predictive Control: Online Gait Generation using Whole-Body Dynamics](http://ames.caltech.edu/galliker2022bipedal.pdf).

### Runtime-Configurable MPC Problem Definition
The optimal control problem (OCP) formulation is fully configurable at runtime from robot-specific YAML files (`task.yaml`). Costs, terminal costs, state soft constraints, soft constraints, and equality constraints can be dynamically added, disabled, or swapped per robot without code modifications. See the [MPC Problem Definition Documentation](humanoid_nmpc/humanoid_common_mpc/src/problem/README.md) for the configuration schema, supported term types, and API usage.

### Gait Switching Time Optimization & Contact Feedback
The framework includes online optimization of gait phase switching times using Hamiltonian jump sensitivities from the Pontryagin Maximum Principle, paired with early/late touchdown reactive contact adaptation. See the [Gait Optimization Documentation](humanoid_nmpc/humanoid_common_mpc/src/gait/README.md) for full mathematical derivations and architecture block diagrams.

### Robot Examples

The project supports the following robot examples:

- Unitree G1 (Centroidal MPC & Whole-Body MPC)
- DRC Atlas (Centroidal MPC with Contact Wrench Cone)
- Unitree R1 (Centroidal MPC)
- 1X Neo (Coming soon)

![Screencast2024-12-16180254-ezgif com-optimize(3)](https://github.com/user-attachments/assets/d4b1f0da-39ca-4ce1-b53c-e1d040abe1be)

## Get Started

### Repository Setup

Clone the repository:

```bash
git clone https://github.com/1x-technologies/wb-humanoid-mpc.git
cd wb-humanoid-mpc
```

> **Note:** The repository uses **Bazel 9.x** with `bzlmod` for native dependency resolution. Legacy ROS/colcon submodules are fully integrated into the monorepo structure.

### Install Dependencies & Workspace Setup
The project supports both Dockerized workspaces (recommended) or a local installation for developing and running the humanoid MPC.

**Platform Support:** The Docker setup is fully compatible with **Linux** and **macOS** (including Apple Silicon via Rosetta 2 x86 emulation). On macOS, GUI visualization uses VNC instead of native X11 forwarding — see the [Visualization Guide](.devcontainer/VISUALIZATION.md) for details.

<details>
<summary>Build & run Dockerized workspace in VS Code</summary>

We provide a [Dockerfile](docker/Dockerfile) to enable running and developing the project from a containerized environment. Check out [devcontainer.json](.devcontainer/devcontainer.json) for environment configuration.

For working in **Visual Studio Code**, install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension. Open the repository in VS Code, press `Ctrl + Shift + P` (or `Cmd + Shift + P`), and select `Dev Containers: Rebuild and Reopen in Container`.

Once the container starts, the Git pre-commit hooks and Bazel environment are automatically configured.

</details>

<details>
<summary>Build & run Dockerized workspace in alternative IDE (e.g. Antigravity / Cursor)</summary>

If you are not using VS Code or are connected via Remote SSH:

1. Spin up the container from the repository root (`docker-compose.yaml`):

```bash
docker compose up -d --build
```

2. Attach your IDE terminal to the container:

```bash
docker compose exec app bash
```

Alternatively, use the provided helper scripts:
```bash
./docker/image_build.bash
./docker/launch_wb_mpc.bash
```

</details>

<details>
<summary>Install Dependencies Locally</summary>

Make sure you have **ROS 2** installed on your system (e.g. ROS 2 Jazzy as specified in the [installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)).

Install system dependencies and Bazel (via Bazelisk):

```bash
envsubst < dependencies.txt | xargs sudo apt-get install -y --no-install-recommends
curl -sSL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel
```

Install Git pre-commit formatting hooks:
```bash
make install-hooks
```

</details>

### Building the MPC with Bazel

The repository uses **Bazel** for high-performance parallel compilation. Build limits are pre-configured in `.bazelrc` (`--jobs=8`) to optimize build times while preventing system RAM exhaustion.

```bash
# Build all Bazel targets across the monorepo
make build-all

# Run all unit tests
make test-all

# Auto-format C++ and Python source files
make format
```

## Running the Examples

Once you launch the NMPC, an RViz visualization window will appear. The first time you start the MPC for a robot model, CppAD auto-differentiation code generation will run (which may take a few minutes depending on your system). Subsequent runs reuse generated dynamic libraries instantly.

### Visualization Setup

- **Linux:** GUI applications render directly via X11 forwarding (`DISPLAY=:99` or host display).
- **macOS / Remote SSH:** Use the `-vnc` suffixed Makefile targets which automatically launch the built-in VNC server. Open **http://localhost:6080/vnc.html** in your browser and click **Connect**. See the [Visualization Guide](.devcontainer/VISUALIZATION.md) for complete details.

### Launch Commands

For **Centroidal Dynamics MPC**:

```bash
# G1 Robot Dummy Simulation
make launch-g1-dummy-sim          # X11 Forwarding (Linux)
make launch-g1-dummy-sim-vnc      # VNC Browser Display (macOS / Remote)

# DRC Atlas Robot Simulation
make launch-drc-atlas-dummy-sim     # Dummy simulation (Linux)
make launch-drc-atlas-dummy-sim-vnc # Dummy simulation (macOS / VNC)
make launch-drc-atlas-sim           # MuJoCo simulation (Linux)
make launch-drc-atlas-sim-vnc       # MuJoCo simulation (macOS / VNC)
```

For **Whole-Body Dynamics MPC**:

```bash
# G1 Robot Whole-Body Simulation
make launch-wb-g1-dummy-sim       # X11 Forwarding (Linux)
make launch-wb-g1-dummy-sim-vnc   # VNC Browser Display (macOS / Remote)
```

#### Interactive Robot Control
Command a desired base velocity and root link height via the **Robot Base Controller GUI** and an **Xbox Controller Joystick**. For the joystick, connect via USB or Bluetooth. The GUI automatically detects connected joysticks and provides interactive velocity sliders.

![robot_remote_control](https://github.com/user-attachments/assets/779be1da-97a1-4d0c-8f9b-b9d2df88384f)

## Reinforcement Learning with MuJoCo Playground (`humanoid_learning`)

The repository includes a GPU-accelerated Reinforcement Learning pipeline built on **Google DeepMind's [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)**, **MuJoCo MJX**, **JAX**, and **Brax**.

<details>
<summary><b>GPU Training Setup & Hardware Prerequisites</b></summary>

### 1. Host Machine GPU Prerequisites
To train RL policies on an **NVIDIA GPU** inside the container, ensure your host has:
1. An **NVIDIA GPU Driver** installed (`nvidia-smi` works on host).
2. The **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** installed.

**Verify Host GPU Docker Passthrough:**
Run this verification command on your **host terminal**:
```bash
docker run --rm --gpus all ubuntu nvidia-smi
```
If this prints your GPU details, Docker has full access to your GPU.

### 2. Starting the Container with GPU Acceleration
Start the container using the GPU compose override:
```bash
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
docker compose exec app bash
```

> **CPU Fallback:** Users without an NVIDIA GPU (e.g. macOS Apple Silicon or CPU Linux) can still run the default container (`docker compose up -d`). JAX will automatically fall back to CPU execution.

</details>

<details>
<summary><b>Running RL Training & Imitation Learning</b></summary>

Once inside the container (or Dev Container), use the following commands:

### Run RL Unit & Smoke Tests
Validates JAX JIT compilation, MuJoCo MJX simulation stepping, and device backend discovery:
```bash
make test-rl
```

### Launch PPO Policy Training (MJX / Playground)
Trains a velocity-tracking policy using parallelized MJX physics simulation:
```bash
# Default quick start
make train-rl

# Or configure environment batch size and total training steps:
bazel run //humanoid_learning/training:train_ppo -- --num_envs=4096 --total_timesteps=10000000
```

### Export MPC Trajectories to RL Demos
Convert recorded observations from `mpc_observation_logger` into HDF5 demonstration datasets:
```bash
make export-rollouts
# Or specify custom input/output paths:
python3 humanoid_nmpc/humanoid_common_mpc_pyutils/humanoid_common_mpc_pyutils/export_rollouts.py \
    --input_path=. \
    --output_path=data/r1_mpc_demos.h5
```

### Behavioral Cloning (BC) Warmstart
Pretrain an actor policy using supervised imitation learning on MPC demonstration rollouts before fine-tuning with PPO:
```bash
make train-bc
# Or with specific demo dataset:
bazel run //humanoid_learning/training:bc_warmstart -- --demos_path=data/r1_mpc_demos.h5 --epochs=20
```

### Export Trained Policy to ONNX
Serialize trained JAX/Flax actor weights to ONNX for C++ runtime deployment:
```bash
bazel run //humanoid_learning/export:export_onnx -- --output_path=models/humanoid_policy.onnx
```

</details>

<details>
<summary><b>Managing Python Dependencies (rules_python)</b></summary>

The RL pipeline uses **Bazel `rules_python`** with an isolated, hermetic Python 3.11 toolchain:
- Edit dependencies in [`humanoid_learning/requirements.txt`](humanoid_learning/requirements.txt).
- Recompile and lock dependencies reproducibly across platforms:
  ```bash
  make lock-rl-deps
  ```

</details>

## Citing Whole-Body Humanoid MPC
To cite Whole-Body Humanoid MPC in your academic research, please use the following BibTeX entry:

```bibtex
@misc{wholebodyhumanoidmpcweb,
   author = {Manuel Yves Galliker},
   title = {Whole-body Humanoid MPC: Realtime Physics-Based Procedural Loco-Manipulation Planning and Control},
   howpublished = {https://github.com/1x-technologies/wb_humanoid_mpc},
   year = {2024}
}
```

## Acknowledgements
Created and actively maintained by [Manuel Yves Galliker](https://github.com/manumerous).

Special thanks to [Nicholas Palomo](https://github.com/nicholaspalomo) for implementing the Dockerization, Bazel monorepo migration, and CI automation.

This project is founded on the great work of many open-source contributors:
- [ocs2](https://github.com/leggedrobotics/ocs2)
- [pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [hpipm](https://github.com/giaf/hpipm)

Part of this work was developed during my time at [1X Technologies](https://www.1x.tech/). I would like to kindly thank Eric Jang and Bernt Børnich for supporting the open sourcing of this project.

Further I would like to thank Michael Purcell, Jesper Smith, Simon Zimmermann, Joel Filho, Paal Arthur Schjelderup Thorseth, Varit (Ohm) Vichathorn, Sjur Grønnevik Wroldsen, Armin Nurkanovic, Charles Khazoom and Farbod Farshidian for the many fruitful discussions, insights, contributions and support.
