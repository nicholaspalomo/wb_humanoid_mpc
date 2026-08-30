# Angular Center of Mass (aCOM) for Humanoid Robots

This package implements the **Angular Center of Mass (aCOM)** representation for multibody humanoid robots, based on the research paper:

> **"Integrable Whole-body Orientation Coordinates for Legged Robots"**
> *Yu-Ming Chen, Gabriel Nelson, Robert Griffin, Michael Posa, and Jerry Pratt*
> *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.*

---

## 1. Problem Formulation & Theoretical Background

### 1.1 The Holonomic vs. Non-Holonomic Divide in Centroidal Dynamics
In legged locomotion:
- **Translational Center of Mass ($\mathbf{r}_{\text{CoM}}$)** is an exact, holonomic function of the robot configuration $\mathbf{q} \in SE(3) \times \mathbb{R}^{n_j}$:
  $$\mathbf{r}_{\text{CoM}}(\mathbf{q}) = \frac{1}{m_{\text{total}}} \sum_{i=1}^n m_i \mathbf{r}_i(\mathbf{q}), \quad \frac{d}{dt}\mathbf{r}_{\text{CoM}}(\mathbf{q}) = \mathbf{v}_{\text{CoM}}$$
  Consequently, translational motion planning and MPC can track an absolute position target $\mathbf{r}_{\text{CoM}} \in \mathbb{R}^3$ without integration drift.

- **Centroidal Angular Momentum ($\mathbf{L}_G$)** is fundamentally **non-holonomic**:
  $$\mathbf{L}_G = \mathbf{A}_\omega(\mathbf{q}) \dot{\mathbf{q}} = \mathbf{I}_G(\mathbf{q}) \boldsymbol{\omega}_b + \mathbf{A}_{\omega, j}(\mathbf{q}) \dot{\mathbf{q}}_j$$
  where $\mathbf{A}_\omega(\mathbf{q})$ is the angular block of the Centroidal Momentum Matrix (CMM), and $\mathbf{I}_G(\mathbf{q})$ is the whole-body locked rotational inertia at the CoM.

Dividing by locked inertia yields the **instantaneous locked angular velocity** (connection 1-form $\bar{\mathbf{A}}_\omega$):
$$\boldsymbol{\omega}_{\text{locked}} = \mathbf{I}_G^{-1}(\mathbf{q})\mathbf{L}_G = \boldsymbol{\omega}_b + \bar{\mathbf{A}}_{\omega, j}(\mathbf{q})\dot{\mathbf{q}}_j, \quad \text{where } \bar{\mathbf{A}}_{\omega, j}(\mathbf{q}) = \mathbf{I}_G^{-1}(\mathbf{q}) \mathbf{A}_{\omega, j}(\mathbf{q})$$

Because the differential connection $\bar{\mathbf{A}}_{\omega, j}(\mathbf{q})$ has non-zero curvature ($d\bar{\mathbf{A}}_\omega + \frac{1}{2}[\bar{\mathbf{A}}_\omega, \bar{\mathbf{A}}_\omega] \neq 0$), integrating $\boldsymbol{\omega}_{\text{locked}}$ along a closed trajectory in joint space ($\oint \dot{\mathbf{q}}_j dt = 0$) results in a non-zero geometric phase (the "falling cat" rotation). **No exact holonomic whole-body orientation coordinate $\mathbf{R}(\mathbf{q}) \in SO(3)$ exists.**

---

### 1.2 The aCOM Coordinate Definition
The core idea of Pratt et al. is to construct an **integrable configuration-dependent approximation**:
$$\boldsymbol{\theta}_{\text{aCOM}}(\mathbf{q}): \mathcal{Q} \to \mathbb{R}^3$$
such that its time derivative $\dot{\boldsymbol{\theta}}_{\text{aCOM}} = \mathbf{J}_{\text{aCOM}}(\mathbf{q})\dot{\mathbf{q}}$ best approximates $\boldsymbol{\omega}_{\text{locked}}$.

#### Optimization Objective (Frobenius Loss on Jacobians):
$$\min_{\boldsymbol{\theta}_{\text{aCOM}}} \mathbb{E}_{\mathbf{q} \sim \mathcal{D}} \left\| \frac{\partial \boldsymbol{\theta}_{\text{aCOM}}}{\partial \mathbf{q}}(\mathbf{q}) - \begin{bmatrix} \mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \mathbf{I}_G^{-1}(\mathbf{q})\mathbf{A}_{\omega, j}(\mathbf{q}) \end{bmatrix} \right\|_F^2$$

#### $SE(3)$ Floating-Base Equivariance:
To guarantee that pure floating-base rotations shift the whole-body orientation rigidly while translations have zero effect:
$$\boldsymbol{\theta}_{\text{aCOM}}(\mathbf{q}) = \boldsymbol{\theta}_{\text{base}} + \Delta \boldsymbol{\theta}(\mathbf{q}_j)$$
$$\mathbf{J}_{\text{aCOM}}(\mathbf{q}) = \begin{bmatrix} \mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \frac{\partial \Delta \boldsymbol{\theta}}{\partial \mathbf{q}_j}(\mathbf{q}_j) \end{bmatrix}$$

where $\Delta \boldsymbol{\theta}(\mathbf{q}_j): \mathbb{R}^{n_j} \to \mathbb{R}^3$ represents the internal joint contribution to whole-body orientation.

---

## 2. System Architecture & Pipeline

```mermaid
flowchart TD
    subgraph DataGeneration [1. Ground Truth CMM Sampling]
        URDF[Robot URDF / MJCF] --> MjModel[MuJoCo Dynamics Model]
        Sampler[Uniform Joint Sampler q_j] --> MjModel
        MjModel --> CMM[Compute CMM A_w & Locked Inertia I_G]
        CMM --> Target[Normalized Connection A_bar_w = I_G^-1 * A_w_j]
    end

    subgraph JAXTraining [2. JAX SIREN Neural Optimization]
        Target --> LossFn[Frobenius Loss: ||J_theta - A_bar_w||_F^2]
        SIREN[SIREN Network: sin_w0_Wx_b] --> Autograd[jax.jacobian / jax.vmap]
        Autograd --> LossFn
        LossFn --> Optax[Optax AdamW + Cosine Decay]
        Optax --> TrainParams[Optimized SIREN Parameters]
    end

    subgraph RuntimeInference [3. Real-Time C++ Deployment]
        TrainParams --> Exporter[export_acom.py]
        Exporter --> JSON[JSON Weights]
        Exporter --> Header[AngularCenterOfMassWeights.h]
        Header --> CppClass[AngularCenterOfMass C++ Class]
        CppClass --> MPC[Centroidal / Whole-Body MPC Tracking]
    end
```

---

## 3. Sinusoidal Representation Network (SIREN)

Standard MLPs with ReLU or GELU activations struggle to represent smooth differential forms and their Jacobians. We employ **Sinusoidal Representation Networks (SIREN)** with periodic activation functions:

$$\mathbf{h}_0 = \sin\left(\omega_0 (\mathbf{W}_0 \mathbf{q}_j + \mathbf{b}_0)\right)$$
$$\mathbf{h}_l = \sin\left(\omega_0 (\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l)\right), \quad l = 1, \dots, L-1$$
$$\Delta \boldsymbol{\theta}(\mathbf{q}_j) = \mathbf{W}_{\text{out}} \mathbf{h}_{L-1} + \mathbf{b}_{\text{out}}$$

### Exact Analytical Jacobian via Chain Rule:
Let $\mathbf{z}_l = \omega_0 (\mathbf{W}_l \mathbf{h}_{l-1} + \mathbf{b}_l)$. The exact layer derivatives are:
$$\frac{\partial \mathbf{h}_0}{\partial \mathbf{q}_j} = \omega_0 \operatorname{diag}\left(\cos(\mathbf{z}_0)\right) \mathbf{W}_0$$
$$\frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}} = \omega_0 \operatorname{diag}\left(\cos(\mathbf{z}_l)\right) \mathbf{W}_l$$
$$\mathbf{J}_{\Delta \theta}(\mathbf{q}_j) = \frac{\partial \Delta \boldsymbol{\theta}}{\partial \mathbf{q}_j} = \mathbf{W}_{\text{out}} \left( \prod_{l=L-1}^1 \frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}} \right) \frac{\partial \mathbf{h}_0}{\partial \mathbf{q}_j}$$

---

## 4. Usage Guide

### 4.1 Running Tests
```bash
# Run JAX model, training, and export unit tests
bazel test //humanoid_learning/acom/tests:test_acom

# Run C++ analytical Jacobian and equivariance unit tests
bazel test //humanoid_nmpc/humanoid_centroidal_mpc_test:test_angular_center_of_mass
```

### 4.2 Training an aCOM Model in JAX
```bash
# Train on Unitree G1 (29-DoF)
bazel run //humanoid_learning/acom:train_main -- --robot g1 --num_samples 10000 --epochs 50 --output_dir /tmp/acom_g1

# Train on DRC Atlas
bazel run //humanoid_learning/acom:train_main -- --robot atlas --num_samples 10000 --epochs 50 --output_dir /tmp/acom_atlas
```

### 4.3 C++ Real-Time Integration
Include the zero-dependency C++ header and evaluate whole-body aCOM in microseconds:

```cpp
#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"

// Instantiate aCOM evaluator
ocs2::humanoid::AngularCenterOfMass acom(n_joints, hidden_dim, num_layers, 30.0);
acom.setWeights(loaded_layers);

// Evaluate orientation and Jacobian
ocs2::vector3_t theta_acom = acom.computeAcomOrientation(q);
ocs2::matrix_t J_acom = acom.computeAcomJacobian(q);
```

---

## 5. Benefits for Legged Control & MPC

1. **Drift-Free Holonomic Tracking**:
   Replaces velocity-level momentum bounds with a direct configuration error penalty in the MPC cost function:
   $$\mathcal{L}_{\text{rot}} = w_{\text{aCOM}} \left\| \boldsymbol{\theta}_{\text{aCOM}}(\mathbf{q}) - \boldsymbol{\theta}_{\text{ref}}(t) \right\|^2$$
2. **Dynamic Decoupling**:
   Allows the robot to swing its arms or twist its torso vigorously while maintaining the true whole-body orientation aligned with the locomotion heading.
3. **Zero Runtime Overhead**:
   The small SIREN forward pass + analytical chain-rule Jacobian evaluates in $< 5\,\mu\text{s}$ in C++, making it suitable for 500 Hz NMPC and WBC loops.
