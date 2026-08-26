# Policy-Guided Whole-Body Control: Offline Trajectory Bootstrapped GRPO

Implementation of **Group Relative Policy Optimization (GRPO)** for **Residual Whole-Body Control (WBC)** of general robots in **JAX / MuJoCo MJX / Brax**, bootstrapped from **offline whole-body MPC trajectories**.

---

## 🌟 System Architecture & Block Diagram

```
+─────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
│                                           System Architecture & Dataflow                                        │
│                                                                                                                 │
│   [Offline Demonstration Dataset]                                                                               │
│          │ (HDF5/NPZ Whole-Body MPC / MoCap Rollouts: s, q_ref(t), tau_ref)                                     │
│          ▼                                                                                                      │
│   [1. Behavioral Cloning (BC) Bootstrapping] ──► Initializes Policy θ_0 & Frozen Reference Prior π_ref          │
│                                                                                                                 │
│   [2. Online GRPO Sampling Loop (JAX / Brax / MJX)]:                                                            │
│      Offline Reference Trajectory q_ref(t) + Current State s_t ∈ R^{obs_dim}                                    │
│          │                                                                                                      │
│          ├─► Stochastic Gaussian Policy π_θ(Δq | s_t) ──► Samples Group of G Candidates {Δq^(1), ..., Δq^(G)}   │
│          │                                                                                                      │
│          ├─► Direct Trajectory Target Fusion: q_des^(g)(t) = q_ref(t) + action_scale * Δq^(g)                    │
│          │                                                                                                      │
│          ├─► [3. JAX-Native Quadratic Program (QP) Whole-Body Controller (WBC)]:                                │
│          │   min_{qddot, f_c, tau}  0.5 * ( ||J_p qddot - qddot_des||_Wp^2 + ||J_c qddot + Jdot_c qd||_Wc^2 +   │
│          │                                 w_f ||f_c||^2 + w_tau ||tau||^2 )                                    │
│          │   s.t.  M(q) qddot + h(q, qd) = S^T tau + J_c^T f_c   (Equations of Motion)                          │
│          │         |f_x| <= mu * f_z, |f_y| <= mu * f_z, f_z >= f_z_min  (Friction Cones for n_c contacts)        │
│          │         -tau_max <= tau <= tau_max                     (Actuator Saturation)                         │
│          │   Output: Motor Torques τ^(g) ∈ R^{n_act}                                                            │
│          │                                                                                                      │
│          ├─► [4. Physics Simulation (MuJoCo MJX / Brax)]:                                                       │
│          │   Step parallel environments ──► Trajectory returns R^(1), ..., R^(G)                                │
│          │                                                                                                      │
│          └─► [5. Critic-Free GRPO Optimization]:                                                                │
│              Adv^(g) = (R^(g) - mean({R^(j)})) / (std({R^(j)}) + eps)                                           │
│              L_GRPO = -min(ratio * Adv, clip(ratio, 1-eps, 1+eps) * Adv) + beta_KL * D_KL(pi_theta || pi_ref)   │
│              Optax AdamW Updates on Policy Parameters θ                                                         │
+─────────────────────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Detailed Execution Pipeline Diagram

```mermaid
flowchart TD
    subgraph OfflinePhase ["1. Offline Bootstrapping Phase"]
        DEMO["Offline Trajectory Dataset<br/>(HDF5/NPZ MPC / MoCap Demonstrations)"]
        BC["Behavioral Cloning (BC) Pretraining<br/>L_BC = E[ ||μ_θ(s) - a*||² - 0.1 log π_θ(a*|s) ]"]
        INIT_P["Bootstrapped Policy θ₀"]
        REF_P["Frozen Reference Prior π_ref"]
        DEMO --> BC
        BC --> INIT_P
        BC --> REF_P
    end

    subgraph OnlineSampling ["2. Group Rollout Sampling (GRPO in JAX)"]
        TRAJ["Offline Trajectory Slice<br/>q_ref(t) at simulation step t"]
        ST["State Observation s_t<br/>• Relative Joint Positions (q - q_nominal)<br/>• Joint Velocities q̇<br/>• Base Pos/Quat & Linear/Angular Vel<br/>• Trajectory Phase [cos(2πt/T), sin(2πt/T)]<br/>• Command Velocity"]
        POLICY["Gaussian Residual Policy π_θ<br/>Δq ~ N(μ_θ(s_t), diag(σ²))<br/>Sample Group of G Candidates: {Δq⁽¹⁾, ..., Δq⁽ᴳ⁾}"]
        FUSION["Direct Target Fusion<br/>q_des(t) = q_ref(t) + action_scale · Δq"]
        TRAJ --> FUSION
        ST --> POLICY --> FUSION
    end

    subgraph WBCSolver ["3. Universal JAX-Native QP Whole-Body Controller"]
        QP["Quadratic Program / Inverse Dynamics<br/>min 0.5 ( ||J_p q̈ - q̈_des||²_Wp + ||J_c q̈ + J̇_c q̇||²_Wc + w_f ||f_c||² + w_τ ||τ||² )<br/>s.t. M(q) q̈ + h(q, q̇) = Sᵀ τ + J_cᵀ f_c<br/>Friction Cones (for n_c ≥ 1) & Torque Saturation [-τ_max, τ_max]"]
        TAU["Joint Torques τ ∈ R^{n_act}"]
        FUSION --> QP --> TAU
    end

    subgraph PhysicsSim ["4. Physics Simulation (MJX / Brax)"]
        ENV["Multi-Step Physics Rollout<br/>Step parallel environments with joint torques τ"]
        RET["Trajectory Returns R⁽¹⁾, ..., R⁽ᴳ⁾<br/>• Direct Trajectory Imitation: exp(-α ||q - q_ref(t)||²)<br/>• Velocity & Upright Tracking<br/>• Base Height Maintenance<br/>• Residual & Torque Penalties"]
        TAU --> ENV --> RET
    end

    subgraph GRPOOpt ["5. Critic-Free GRPO Optimization"]
        ADV["Group Relative Advantage Normalization<br/>A⁽ᵍ⁾ = (R⁽ᵍ⁾ - μ_R) / (σ_R + ε)"]
        RATIO["Importance Sampling Ratio<br/>ρ⁽ᵍ⁾(θ) = exp(log π_θ(Δq⁽ᵍ⁾|s) - log π_old(Δq⁽ᵍ⁾|s))"]
        KL["Analytical Reference KL Divergence<br/>D_KL(π_θ || π_ref)"]
        LOSS["GRPO Clipped Loss<br/>L_GRPO = -min(ρ A, clip(ρ, 1-ε, 1+ε) A) + β_KL D_KL"]
        OPT["Optax AdamW Gradient Update on θ"]

        RET --> ADV
        POLICY -.-> RATIO
        INIT_P -.-> POLICY
        REF_P -.-> KL
        ADV --> LOSS
        RATIO --> LOSS
        KL --> LOSS
        LOSS --> OPT
        OPT -.-> POLICY
    end
```

---

## 🦿 Mathematical Formulations

### 1. Model-Based Whole-Body Controller (WBC) QP

At each control step $t$, the controller solves a Quadratic Program (QP) to compute optimal generalized accelerations $\ddot{q} \in \mathbb{R}^{n_v}$, ground reaction contact forces $f_c \in \mathbb{R}^{3n_c}$, and actuator motor torques $\tau \in \mathbb{R}^{n_{\text{act}}}$:

$$\min_{\ddot{q}, f_c, \tau} \frac{1}{2} \left[ \|\ddot{q}_j - \ddot{q}_j^{\text{des}}\|_{W_p}^2 + \|J_c(q) \ddot{q} + \dot{J}_c(q, \dot{q}) \dot{q}\|_{W_c}^2 + w_f \|f_c - f_c^{\text{des}}\|^2 + w_\tau \|\tau\|^2 \right]$$

#### Posture / Residual Acceleration Target
$$\ddot{q}_j^{\text{des}} = k_p (q_{\text{des}}(t) - q_j) + k_d (\dot{q}_{\text{des}}(t) - \dot{q}_j) = k_p \left( q_{\text{ref\_offline}}(t) + \text{action\_scale} \cdot \Delta q_t - q_j \right) - k_d \dot{q}_j$$

#### Equations of Motion (Equality Constraint)
$$M(q) \ddot{q} + h(q, \dot{q}) = S^T \tau + J_c(q)^T f_c$$
where:
- $M(q) \in \mathbb{R}^{n_v \times n_v}$ is the generalized mass/inertia matrix.
- $h(q, \dot{q}) \in \mathbb{R}^{n_v}$ is the nonlinear bias vector (Coriolis, centrifugal, and gravity).
- $S = \begin{bmatrix} 0_{n_{\text{act}} \times (n_v - n_{\text{act}})} & I_{n_{\text{act}}} \end{bmatrix} \in \mathbb{R}^{n_{\text{act}} \times n_v}$ is the actuation selection matrix.
- $J_c(q) \in \mathbb{R}^{3n_c \times n_v}$ is the stance contact Jacobian for $n_c$ contact points ($n_c = 0$ for drones, $n_c = 2$ for bipeds, $n_c = 4$ for quadrupeds).

#### Contact Friction Cones & Normal Bounds (for $n_c \ge 1$)
$$|f_{c, x}^{(i)}| \le \mu f_{c, z}^{(i)}, \quad |f_{c, y}^{(i)}| \le \mu f_{c, z}^{(i)}, \quad f_{z, \min} \le f_{c, z}^{(i)} \le f_{z, \max}, \quad \forall i \in \{1, \dots, n_c\}$$

#### Actuator Torque Limits
$$-\tau_{\max} \le \tau \le \tau_{\max}$$

---

### 2. The RL Policy & Residual Formulation

The policy $\pi_\theta(\Delta q | s_t)$ outputs bounded residual joint position corrections $\Delta q \in [-1, 1]^{n_{\text{act}}}$:
$$\mu_\theta(s_t) = \tanh\left(\text{MLP}_\theta(s_t)\right), \quad \sigma = \exp(\text{clamp}(\log \sigma, -5.0, 1.0))$$
$$\Delta q \sim \mathcal{N}\left(\mu_\theta(s_t), \text{diag}(\sigma^2)\right)$$

#### Behavioral Cloning (BC) Bootstrapping Loss
Given an offline dataset of state-action demonstration transitions $\mathcal{D} = \{(s_i, a_i^*)\}_{i=1}^N$:
$$\mathcal{L}_{\text{BC}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ \|\mu_\theta(s_i) - a_i^*\|^2 - 0.1 \log \pi_\theta(a_i^* | s_i) \right]$$
Pretraining on $\mathcal{D}$ initializes the policy parameters $\theta_0$ and creates a frozen reference policy $\pi_{\text{ref}} = \pi_{\theta_0}$.

#### Analytical KL Divergence
Between the current policy $\pi_\theta = \mathcal{N}(\mu_p, \sigma_p^2)$ and reference prior $\pi_{\text{ref}} = \mathcal{N}(\mu_q, \sigma_q^2)$:
$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \sum_{j=1}^{n_{\text{act}}} \left[ \log \frac{\sigma_{q, j}}{\sigma_{p, j}} + \frac{\sigma_{p, j}^2 + (\mu_{p, j} - \mu_{q, j})^2}{2 \sigma_{q, j}^2} - \frac{1}{2} \right]$$

---

### 3. Group Relative Policy Optimization (GRPO)

GRPO samples a group of $G$ candidate residual actions $\{\Delta q_i^{(1)}, \dots, \Delta q_i^{(G)}\}$ for each state $s_i$ across $B$ parallel environments.

#### Critic-Free Group Advantage Normalization
$$A_i^{(g)} = \frac{R_i^{(g)} - \bar{R}_i}{\sigma_{R, i} + \epsilon_{\text{adv}}}$$
where $\bar{R}_i = \frac{1}{G} \sum_{j=1}^G R_i^{(j)}$ and $\sigma_{R, i} = \sqrt{\frac{1}{G} \sum_{j=1}^G (R_i^{(j)} - \bar{R}_i)^2}$.

#### Clipped Surrogate Policy Objective with Reference KL Regularization
$$r_i^{(g)}(\theta) = \exp\left( \log \pi_\theta(\Delta q_i^{(g)} | s_i) - \log \pi_{\theta_{\text{old}}}(\Delta q_i^{(g)} | s_i) \right)$$
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{B \cdot G} \sum_{i=1}^B \sum_{g=1}^G \min\left( r_i^{(g)}(\theta) A_i^{(g)}, \text{clip}(r_i^{(g)}(\theta), 1-\epsilon, 1+\epsilon) A_i^{(g)} \right) + \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

---

## 🤖 Multi-Robot Support & Config Schema

All robot models, kinematic parameters, and WBC tuning weights are externalized in `humanoid_learning/configs/robots/`:

```yaml
name: atlas
nq: 35
nv: 34
n_act: 28
total_mass: 192.76
default_standing_height: 0.93
contact_body_names:
  - l_foot
  - r_foot
limb_joint_indices:
  left_arm: [3, 4, 5, 6, 7, 8]
  torso: [9]
  right_arm: [10, 11, 12, 13, 14, 15]
  left_leg: [16, 17, 18, 19, 20, 21]
  right_leg: [22, 23, 24, 25, 26, 27]
wbc_config:
  w_base_acc: 100.0
  w_posture: 10.0
  w_contact_acc: 1000.0
  w_force_reg: 0.0001
  w_torque_reg: 0.0001
  kp_posture: 100.0
  kd_posture: 10.0
  friction_coef: 0.6
  f_z_min: 5.0
  f_z_max: 4727.54
  tau_max: 100.0
```

---

## 🛠️ Codebase Structure

```text
humanoid_learning/
├── configs/
│   ├── grpo_residual_wbc.yaml       # Centralized hyperparameters & WBC weights
│   └── robots/
│       ├── g1_29dof.yaml            # Unitree G1 robot definition (29 DOFs)
│       └── atlas.yaml               # Standard DRC Atlas definition (28 DOFs)
├── envs/
│   ├── base_env.py                  # Base MJX Humanoid environment
│   └── humanoid_residual_wbc_env.py # Offline trajectory residual WBC environment
├── wbc/
│   ├── __init__.py                  # WBC package exports
│   ├── robot_model_loader.py        # Dynamic MuJoCo / Pinocchio / YAML model parser
│   └── jax_wbc.py                   # Batched JAX QP Whole-Body Controller
├── training/
│   ├── generate_robot_spec.py       # CLI robot spec generator utility
│   ├── offline_dataset.py           # Offline HDF5/NPZ demonstration dataset loader
│   ├── policy_network.py            # Flax NNX Gaussian Residual Policy Network
│   ├── bootstrap_bc.py              # Behavioral Cloning pretraining pipeline
│   ├── train_grpo.py                # Critic-Free GRPO training loop
│   └── train_ppo.py                 # Baseline Brax PPO pipeline
└── tests/
    ├── test_jax_wbc.py              # Unit tests for WBC, Multi-Robot, Env, Policy, & GRPO
    ├── test_cartpole.py             # Cartpole benchmark test
    └── test_rl_imports.py           # JAX/MJX environment smoke tests
```

---

## 🚀 Quickstart Guide

### 1. Run Unit Test Suite
```bash
# Run WBC and GRPO unit tests (12 test cases)
PYTHONPATH=. python -m unittest humanoid_learning/tests/test_jax_wbc.py
```

### 2. Generate / Inspect Robot Specification
```bash
# Generate specification for Unitree G1
PYTHONPATH=. python humanoid_learning/training/generate_robot_spec.py --input g1

# Generate specification for normal DRC Atlas
PYTHONPATH=. python humanoid_learning/training/generate_robot_spec.py --input atlas
```

### 3. Bootstrap Policy from Offline Demonstrations
```bash
PYTHONPATH=. python humanoid_learning/training/bootstrap_bc.py \
    --epochs 15 \
    --batch_size 128 \
    --lr 1e-3 \
    --output_path checkpoints/bootstrapped_policy.npz
```

### 4. Launch GRPO Training for Residual WBC
```bash
PYTHONPATH=. python humanoid_learning/training/train_grpo.py \
    --robot g1 \
    --num_iterations 30 \
    --group_size 4 \
    --num_envs 4 \
    --rollout_horizon 8 \
    --learning_rate 3e-4 \
    --beta_kl 0.05 \
    --bootstrap_epochs 5 \
    --output_dir checkpoints/humanoid_grpo
```
