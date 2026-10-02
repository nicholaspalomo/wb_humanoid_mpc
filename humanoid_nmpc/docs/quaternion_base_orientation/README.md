> **Status: design, implementation in progress.** This is the reviewed design for switching the humanoid NMPC's base orientation from Euler angles to quaternions. It becomes this formulation's README (section layout of Appendix C) once the switch lands; until then line references point at the code before the switch.

# Quaternion base orientation for the humanoid NMPC: final design

**Scope and conventions**

- **Worktree:** `/home/nico-palomo/workspace/wb_humanoid_mpc_quat` at HEAD `c331ddd`.
- **Line numbers** refer to that worktree. The main checkout has uncommitted edits to several of these files, so re-locate anchors before editing.
- **Path abbreviations** (relative to the worktree root):
  - `common/` = `humanoid_nmpc/humanoid_common_mpc`
  - `cmpc/` = `humanoid_nmpc/humanoid_centroidal_mpc`
  - `wbmpc/` = `humanoid_nmpc/humanoid_wb_mpc`
  - `ros2c/` = `humanoid_nmpc/humanoid_common_mpc_ros2`
  - `cros2/` = `humanoid_nmpc/humanoid_centroidal_mpc_ros2`
  - `wbros2/` = `humanoid_nmpc/humanoid_wb_mpc_ros2`
  - `ocs2/` = `lib/ocs2`
  - `cm/` = `lib/ocs2/pinocchio/centroidal_model`
- **On the main line** (after the ROS removal) the three `*ros2/` packages are gone. Their robot-side loop is the robot
  process of `humanoid_nmpc/humanoid_common_mpc_app/robot` (`RobotProcess`, `SimFsmBridge`, `SimFallRecovery`,
  `InitialSimState`), their MPC side the MPC nodes of `humanoid_common_mpc_app/node` and `humanoid_{centroidal,wb}_mpc_app`,
  and their visualization `humanoid_common_mpc_app/visualization`. The harness of Step 1 lives in
  `humanoid_nmpc/humanoid_mpc_validation`, not in `ros2c/test/support`, and solves through
  `InProcessMpcLink::runSolverIteration()` (`Execution::kCaller`), not through hooks of the MRT joint controllers. Read
  the `ros2c/`, `cros2/` and `wbros2/` anchors below as pointers into those packages.
- **Rules every implementing agent follows** (repository standing rules):
  - Do not use `auto` except when the initializer is `std::make_unique` or `std::make_shared`. This includes range-for loops, structured bindings and lambdas.
  - Give every non-self-evident literal argument a `/*paramName=*/` comment, using the callee's parameter name exactly.
  - Prefer Abseil throughout: `absl::Status`/`StatusOr` with the status macros, `absl::StrCat`, `absl::flat_hash_map`, `absl::log` `CHECK`/`LOG`.
  - Follow Google C++ and Python style, with one class per header for every new cost, constraint, manifold or controller class.
  - Use American spelling.
  - Every step ships its tests in the same change.
  - Every new cross-file coupling gets `LINT.IfChange`/`LINT.ThenChange` (Appendix A).
  - Run `make format` and `make lint` (`python3 -m tools.hooks.format_code` and `python3 -m tools.hooks.lint_code`, in the dev container) before a step is done.
  - Run one `bazel` command at a time, never pass a larger `--jobs`, and never run a compiler outside Bazel while a build runs. CppAD library generation in tests is itself a compiler run, so tests that regenerate libraries run one at a time.

---

## 1. Decision summary

Both MPCs will carry the base attitude as a unit quaternion in the state, and the vendored OCS2 SQP becomes a native manifold (tangent-space) solver.

**Storage versus tangent.**
- Trajectories, observations, targets and ROS messages store the quaternion: centroidal `nx = 13+nj`, whole-body `nx = 13+2nj`.
- The QP, Riccati gains, line search, shooting gaps, initial-state gap and feedback policy work in the 3-D rotation tangent: `ndx = 12+nj` and `12+2nj`. These are exactly today's Euler state sizes, so HPIPM sizes and gain shapes are unchanged.

**Root joint.** The Pinocchio root joint of both MPCs becomes `Composite(Translation, Spherical)`, which gives `v_base = [ṗ_W, ω_B]`. This choice:
- keeps `A_b(0,0) = m`, so the CppAD-safe closed-form momentum-matrix inverse and the whole-body 3×3 Schur solve stay as they are;
- removes the pitch ±90° singularity;
- makes every Pinocchio velocity Jacobian the exact tangent Jacobian of the state.

**Costs and constraints.**
- Every term keeps OCS2's contract: it differentiates with respect to the stored (ambient) state, and the transcription pulls the derivatives back once.
- Residuals that are already rotation-based keep their exact form, so their weights keep their meaning. These are: end-effector orientation and plane distance, foot yaw, twist/zero-acceleration, wrench cone, DCM/ICP, collisions and torques.
- The Euler-weighted rows of `Q`, `Q_final` and `Q_acom` act on a heading–tilt residual laid out exactly like today's (yaw, pitch, roll) rows. So every `task.yaml`, GUI slider, FSM parser and Python reader keeps its keys, indices and meaning.
- Euler angles survive only at named human-facing converters: the `initialState` loader, the target calculators, the heuristic seam, the planned-heading override, pose commands, telemetry and display.

**Why this design.** It is the faithful-manifold design, which the three judges rated best on correctness, robustness and verifiability.
- The re-anchored chart design is rejected because its state stores a heading angle plus MRPs rather than a quaternion. That contradicts the user's decision to "completely switch to a quaternion representation".
- The embedded-ℝ⁴ design is rejected because its radial mode decouples only up to the shooting defects. It also needs hemisphere alignment at every consumer, and its radial-decoupling test bound would fail on real-time-iteration warm starts.

**The judges' fatal flaws are fixed:**
1. OCS2's `QuadraticStateInputCost`/`QuadraticStateCost` have `final` methods, so they are not subclassed. New humanoid `Tuning*` cost classes replace them, the hot-reload casts are migrated, and `InputQuadraticCost` becomes input-only. This removes the `zeroQ` sizing hazard.
2. The tilt is evaluated with `n = ‖s_xy‖` plus a series expansion, so it cannot be 0/0 at a level attitude.
3. The ACoM Jacobian carries the `Exp(Δθ)` adjoint.
4. YAML relabels keep the first comment token `theta_base_*`/`omega_base_*` that `humanoid_finite_state_machine.py:296-308` filters on.
5. The Pinocchio Spherical-joint spike tapes under both `-c opt` and `-c fastbuild`, and fastbuild-with-asserts is never used as a mitigation.

**Staging.** The migration is staged so that an intermediate step runs the quaternion-formulated costs on the old Euler coordinates. That separates formulation effects from coordinate effects. Each stage is gated against golden data recorded from the Euler formulation before anything is removed.

### 1.1 Choices the user made (2026-10-01)

| # | Choice | Evidence | Decision |
|---|---|---|---|
| Q1 | **Behavior-changing fixes of live reference defects.** D1/D1b: both target calculators average the measured roll-rate slot (index 5) into the intermediate yaw knot. The centroidal one also uses `Ab_inv·h̄`, which is `v_b/m`, so its measured term is about 0 (`cmpc/src/command/CentroidalMpcTargetTrajectoriesCalculator.cpp:143-148`; `wbmpc/src/command/WBMpcTargetTrajectoriesCalculator.cpp:100-105`). D2: `ProceduralMpcMotionManager.cpp:182` reads `baseVelocity(3)`, which is `L_x/m` on the centroidal model, and `:198` tests the command where the measurement was meant. D3: `SwitchedModelReferenceManager.cpp:337, 346` read centroidal slots on the whole-body model; this path is latent. | **Fixed DURING the switch** (the user's choice over the recommendation to defer them): D1/D1b/D2/D3 are fixed in Step 7, together with the formulation change on Euler coordinates, so G2/M2 include them. The Step 7 gate against G1/M1 therefore widens for the reference-related quantities (intermediate yaw knots, the motion manager's measured-velocity inputs, yaw-rate tracking); the expected differences are recorded per defect in the golden README, and the formulation's own E2 tolerances still apply to every other term. Former Step 11 is folded into Step 7; Step 2's item 6 (making the old `0.5·command` explicit) is dropped. |
| Q2 | **G1 whole-body foot-cost weights.** `wbmpc/src/cost/EndEffectorDynamicsCostHelpers.cpp:109-110` overwrite the velocity weights with the acceleration values. The configured values (lin vel 50/50/0, ang vel 100/100/100, lin acc 5/5/0, ang acc 2/2/2) were therefore never applied. The effective values are (5,5,0), (2,2,2), (0.01,0.01,0.01) and (0.01,0.01,0.01). | **Fix the loader and set the YAML to the currently effective values** (the user chose the recommendation), so behavior is bit-identical (Step 2). Restoring the originally configured values is a re-tuning the user can do afterwards in MuJoCo. |

Everything else is settled by the evidence and decided below.

### 1.2 Decisions and reasons

| # | Decision | Reason / rejected alternative |
|---|---|---|
| D1 | Root joint `JointModelComposite{JointModelTranslation, JointModelSpherical}` in `common/src/pinocchio_model/createPinocchioModel.cpp:69-76` (shared by both MPCs). The unused OCS2 copy at `cm/src/FactoryFunctions.cpp:46-52, 74-79` changes to match. | World-frame `ṗ_W` keeps the whole-body rows 29..31, `v.head<3>()` readers and the `Ab(0,0)=m` structure. FreeFlyer (SE(3)-local `[v_B, ω_B]`) breaks all three. FreeFlyer is the fallback only if Step 0 fails under `-c opt` (§7, R1). |
| D2 | Native tangent-space SQP in the vendored OCS2. A `StateManifold` is set on the `OptimalControlProblem`. A `nullptr` manifold keeps every existing code path bitwise identical. | No radial mode, no sign alignment, QP sizes unchanged. HPIPM is untouched because its sizes come from `dfdx.cols()` (`ocs2/oc/src/oc_problem/OcpSize.cpp:55-59`). |
| D3 | Discretization: OCS2's existing RK4 on the ambient vector field, then normalization (`F = Π∘Φ_RK4`). The defect is expressed on the manifold (§2.7). | The vector field is tangent to S³ (`Gᵀ(ξ)ξ = 0`), so RK4 keeps fourth order. Measured norm drift is about 1.1e-10 per step at dt 0.02 and |ω| = 5. A Lie-group RK would mean rewriting `SensitivityIntegratorImpl.cpp`. |
| D4 | Every cost and constraint returns derivatives with respect to the **ambient** state. The transcription pulls them back once (`J ← J·E`, `H ← EᵀHE`). Analytic terms that naturally have tangent derivatives lift them (`J_a = J_t·E⁺`). | The roughly 30 CppAD terms need no derivative plumbing. |
| D5 | Homogeneity rule: every function of the state sees the quaternion only through `ξ̂ = ξ/‖ξ‖`, computed safely (§2.6). The single exception is the kinematic row `ξ̇ = ½G(ξ)ω_B`, which uses the raw `ξ`. | Makes the retraction-curvature term vanish (`∇_ξℓ·ξ = 0`). Makes RK4 stages and the CppAD tape point (`x = ones`, so `ξ̂ = ½(1,1,1,1)`) valid configurations. Maps a zero state to the identity rotation instead of NaN. |
| D6 | Tuning layout = tangent size: `n_t = ndx` = today's Euler `nx`, index for index. YAML `Q`, `Q_final`, `R`, `initialState` are unchanged in size, keys and indices. Costs evaluate a tuning residual `r(x, x_ref)` with its own Jacobian. | `loadEigenMatrix` would silently shift every index ≥ 9 for a quaternion-sized Q. The GUI (`remote_control/tk_app/yaml_param_tree.py`, `mpc_params_tab.py:618-664`) needs no code change. |
| D7 | The Euler-weighted orientation rows use the heading–tilt split (§2.8.1): wrapped twist yaw, plus tilt rotation vector components (y, x) in the heading frame. | Tilt is exactly independent of the yaw error. `Log(R_refᵀR)` leaks tilt between axes under a yaw error: at a 0.35 rad yaw error it reports (0.108, 0.032) for a true (0.1, 0.05). Every robot ships `Q_yaw = 0`, so yaw errors are not controlled by Q, and SA01/R1 roll/pitch weights (85/5) would change meaning. |
| D8 | Whole-body angular-velocity rows: `e_ω = P₃(ω_B − R(ξ̂)ᵀR(ξ̂_r)ω_{B,r})`, i.e. a body-frame error with the reference transported, ordered (z, y, x). | Exactly zero for correct tracking of a world yaw-rate command at any tilt (as the Euler-rate difference is). The Jacobian is linear plus one skew term. Equal to the Euler-rate rows at a level attitude (`T_B(0) = P₃`). |
| D9 | One heading definition for reads: the twist `ψ_h(ξ) = 2·atan2(ξ_z, ξ_w)`, read in the hemisphere `ξ_w ≥ 0` (so `ψ_h ∈ (−π, π]` and `ψ_h(−ξ) = ψ_h(ξ)` exactly; implemented with signed zeros made positive, §2.8.1). Edits of level references at the Euler boundary (heuristic offsets, planned-heading override) use ZYX decomposition, add or replace, then recompose. | The twist is singular only upside down; ZYX yaw is singular at pitch ±90°. The two differ by `−2·atan(tan(θ/2)tan(φ/2)) ≈ −θφ/2` (0.0113 rad at θ = φ = 0.15). For level references (every shipped reference) ZYX yaw equals twist yaw exactly, so the edits are exact Euler arithmetic. |
| D10 | Targets are stored as quaternions in the solver layout and interpolated with shortest-path slerp through the robot model. Pose commands are subdivided so consecutive knots are ≤ π/2 apart in yaw. | Slerp between level knots is exactly linear in yaw, so it reproduces today's linear Euler interpolation. Subdivision keeps "+270° turns +270°". nlerp would be 2.4 mrad off at 0.84 rad spacing. |
| D11 | New `ManifoldLinearController`: `u = u*_k + K_k·(x ⊖ x̄_k)`, blended between nodes. Adds `ControllerType::MANIFOLD_LINEAR`, appended to the OCS2 enum (value 5, after `ONNX` and `BEHAVIORAL`; the value is not a wire format). The ROS IDL constant is dropped with ROS; the protobuf wire code (`humanoid_mpc_msgs/controller_type.proto`, which has 0-2 today) is added by the protobuf policy encoder, which maps the two enums explicitly (`ControllerType.h` carries an IFTTT link to the proto). | `LinearController` computes `uff + K·x` (`ocs2/core/src/control/LinearController.cpp:79-87`), which is undefined with a `nu×ndx` K and `x ∈ ℝ^nx`. `LinearController::unFlatten` sizes K from the state message (`ocs2/ros2_interfaces/src/mrt/MRT_ROS_Interface.cpp:178`). Atlas and SA01 run `useFeedbackPolicy: true`. |
| D12 | No hemisphere alignment anywhere. | Every consumer is double-cover invariant: shortest-path Log in `⊖`, shortest slerp, the heading–tilt residual, the rotation-based residuals, and the equivariant flow. Verified by `−ξ` tests on every term and on the MRT (§5c). |
| D13 | New cost classes `TuningQuadraticStateInputCost` and `TuningQuadraticStateCost` (humanoid) derive from `ocs2::StateInputCost`/`StateCost` with their own gain setters. `StateInputQuadraticCost`, `StateQuadraticCost` and `BasePoseShapedQuadraticStateCost` re-base onto them. `InputQuadraticCost` becomes input-only. The updater is migrated. | `getValue`/`getQuadraticApproximation` are `final` (`ocs2/core/include/ocs2_core/cost/QuadraticStateInputCost.h:54, 59`; `QuadraticStateCost.h:49, 55`). `Collection::get` is a `dynamic_cast` (`ocs2/core/include/ocs2_core/misc/Collection.h:182`), so an unmigrated `get<QuadraticStateInputCost>` would throw `bad_cast`. The updater would catch it and only `LOG(WARNING)` (`cmpc/src/mrt/MpcParameterUpdaterModule.cpp:986-993`), silently disabling hot reload. |
| D14 | CppAD libraries go in a layout-tagged folder `cppad_code_gen/cppad_<mpc><robot>/<kStateLayoutTag>` (`common/src/common/ModelSettings.cpp:189`). `CppAdInterface` checks `Domain()`/`Range()` on load and throws with the folder name on mismatch; a fresh interface (no earlier `createModels()`, not a copy) finds the expected range by one off-tape evaluation of the function at the tape point, so a changed output size with unchanged inputs is caught too. A library with the same dimensions but other arithmetic cannot be told apart: that is what the layout tag is for. | Every task file ships `recompileLibrariesCppAd: false` (Atlas `task.yaml:705`, SA01 `:607`, G1 `:376`, R1 `:363`, G1-WB `:147`). Today only the `.so` file's existence is checked (`ocs2/core/src/automatic_differentation/CppAdInterface.cpp:140-155, 305`). |
| D15 | Pre-existing defects that do not change behavior are fixed before the goldens (Step 2). Behavior-changing ones are user-gated after the switch (Q1, Step 11). | The switch is validated against an unchanged plant. Each behavior change is validated alone. |
| D16 | A staged ladder: layout-agnostic API on Euler (gate: bitwise or 1e-12 equal to G1), then the formulation on Euler coordinates (gate: stated first-order tolerances against G1; records G2), then the coordinate switch (gate: exact to round-off against G2). | Separates "formulation changed the behavior" from "coordinates/solver changed the behavior". |
| D17 | The SRBD branch (`cm/src/ModelHelperFunctions.cpp:61-79`) is ported, not refused. | It is a model option, not an Euler path. All robots ship `centroidalModelType: 0`, so the port is tested against Full at the nominal posture. |
| D18 | The whole-body `initialState` rows `9+nj..11+nj` (G1: 32..34) are reinterpreted as body angular velocity in (z, y, x) order, matching their `omega_base_z/y/x` labels and the Q rows. | Shipped values are 0, so this is identical. The label becomes true (today they are Euler rates). |

---

## 2. Formulation

### 2.1 Notation and quaternion algebra

- **Quaternion storage.** `ξ = (ξx, ξy, ξz, ξw)` in Eigen `coeffs()` and Pinocchio order, with the Hamilton product. `R(ξ)` maps base to world. This is the same convention as `RobotState::getRootRotationLocalToWorldFrame()` (`robot_runtime/robot_model/include/robot_model/RobotState.h:22-26`).
- **Construction trap.** Build quaternions from coefficient vectors only. Eigen's 4-scalar constructor takes `(w, x, y, z)`; that trap already corrupted `wbmpc/src/cost/EndEffectorDynamicsQuadraticCost.cpp:117-118`.
- **Rate matrix.** `ξ ⊗ (ω, 0) = G(ξ)·ω`, with

```math
G(\xi)=\begin{bmatrix} \xi_w&-\xi_z&\xi_y\\ \xi_z&\xi_w&-\xi_x\\ -\xi_y&\xi_x&\xi_w\\ -\xi_x&-\xi_y&-\xi_z\end{bmatrix},\qquad G^\top G=\|\xi\|^2 I_3,\qquad G^\top\xi=0 .
```

- **Tangent maps.** `E(ξ) = ½G(ξ)` (4×3) maps a body right perturbation to `δξ`. `E⁺(ξ) = 2Gᵀ(ξ̂)/‖ξ‖` (3×4) is its left inverse; on the unit sphere it equals `2Gᵀ(ξ)`, and `E⁺E = I₃`.
- **Exp and Log.**
  - `Exp(φ) = (sin(|φ|/2)·φ/|φ|, cos(|φ|/2))`, with a series for |φ| < 1e-6.
  - `Log(ξ) = 2·atan2(‖v‖, w)·v/‖v‖` after `ξ ← −ξ` when `w < 0`, so ‖Log‖ ≤ π (shortest path); series for ‖v‖ < 1e-6.
- **Right Jacobians.**
  - `Jr(φ) = I − (1−cos α)/α²·[φ]× + (α − sin α)/α³·[φ]×²`
  - `Jr⁻¹(φ) = I + ½[φ]× + (1/α² − (1+cos α)/(2α sin α))·[φ]×²`
  - α = |φ|. The `Jr`/`Jr⁻¹` coefficients `(α − sin α)/α³` and `1/α² − (1+cos α)/(2α sin α)` cancel in double precision as α → 0, so they use their Taylor series below α = 0.1 (`kJacobianSeriesThreshold`, `UnitQuaternionMath.h`), truncated where the series is exact to round-off; the `(1−cos α)/α²` term and `Exp`/`Log` switch at 1e-6 only to avoid 0/0.
- **Permutation.** `P₃ = antidiag(1,1,1)` maps (x, y, z) to (z, y, x).
- **Euler.** `Θ = (ψ, θ, φ)` is ZYX with yaw first, `R = R_z(ψ)R_y(θ)R_x(φ)`. `T_B(Θ)` maps Euler rates to body angular velocity, `ω_B = T_B(Θ)Θ̇`, and `T_B(0) = P₃`. It exists as `getLocalAngularVelocityFromEulerAnglesZyxDerivatives` in `ocs2/robotic_tools/.../RotationDerivativesTransforms.h:141-175` and is used only at the boundary and in tests.

### 2.2 Root joint and Pinocchio vectors

With `Composite(Translation, Spherical)`:

| | Configuration `q` | Velocity `v` | Acceleration `a` |
|---|---|---|---|
| Base | `[p_W(3), ξ(4)]` | `[ṗ_W(3), ω_B(3)]` | `[p̈_W, ω̇_B]` |
| Joints | `q_j` | `q̇_j` | `q̈_j` |
| Sizes | `nq = 7+nj` | `nv = 6+nj` | `nv` |

**Key identity (§1.3 of the faithful design; checked by `testConfigurationTangent`).** With the right-trivialized tangent (translation in the world frame, rotation as a body right perturbation), the configuration part of the state tangent equals Pinocchio's velocity coordinates. Consequences:
- Every Pinocchio `v`-space Jacobian (frame, CoM, centroidal map) is the tangent Jacobian with respect to the configuration.
- `pinocchio::integrate(q, v)` equals `q ⊕ v`.

Unchanged by the root-joint swap:
- `checkPinocchioJointNaming` (`common/src/pinocchio_model/pinocchioUtils.cpp:56-60`)
- `readPinocchioJointLimits` (`:102-103`, which uses `.tail`)
- `ContactPlanningModelParameters.cpp:88-98` (`pinocchio::neutral` is the identity quaternion)
- `createDefaultPinocchioInterface`'s consumer `ModelSettings.cpp:208-215` (names only)
- the visualizer's root link (`ros2c/src/visualization/HumanoidVisualizer.cpp:75-77`)

### 2.3 State, tangent, tuning and input layouts

**Centroidal** (`cmpc/include/humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h:52-60, 76-196`):

| Block | Ambient `x` (storage) | Tangent `δx` (QP, gains) | Tuning rows (YAML) |
|---|---|---|---|
| `h̄ = [ċ, L/m]` (world, normalized momentum) | 0..5 | 0..5 | 0..5 |
| `p_W` | 6..8 | 6..8 | 6..8 |
| base orientation | 9..12 = ξ (xyzw) | 9..11 = δφ (body right perturbation) | 9 = yaw, 10 = pitch, 11 = roll (heading–tilt residual) |
| `q_j` | 13..12+nj | 12..11+nj | 12..11+nj |
| size | `nx = 13+nj` | `ndx = 12+nj` | `n_t = 12+nj` |

- **Input:** `u = [W_l(6), W_r(6), q̇_j]` in the world frame, or the basis scalings λ followed by `q̇_j`. Unchanged.
- **Pinocchio vectors:**
  - `q = [p_W, ξ̂, q_j]`.
  - `v = [v_b, q̇_j]` with `v_b = [ṗ_W, ω_B] = A_b⁻¹(q)·(m·h̄ − A_j(q)·q̇_j)`. This is the existing formula (`cm/src/CentroidalModelPinocchioMapping.cpp:84-107`), with the vector at `:102` sized `nv`.
- **Start indices:** `getJointStartindex` 12 → 13 (`CentroidalMpcRobotModel.h:95-96`; `cm/include/.../implementation/AccessHelperFunctionsImpl.h:147, 155`). The joint-velocity start index is an input index and is unchanged. `adaptBasePoseHeight` (state index 8) is unchanged.

**Whole-body** (`wbmpc/include/humanoid_wb_mpc/common/WBAccelMpcRobotModel.h:47-212`):

| Block | Ambient | Tangent | Tuning rows |
|---|---|---|---|
| `p_W` | 0..2 | 0..2 | 0..2 |
| orientation | 3..6 = ξ | 3..5 = δφ | 3 yaw, 4 pitch, 5 roll |
| `q_j` | 7..6+nj | 6..5+nj | 6..5+nj |
| `ṗ_W` (world) | 7+nj..9+nj | 6+nj..8+nj | 6+nj..8+nj |
| `ω_B` (body, x,y,z) | 10+nj..12+nj | 9+nj..11+nj | 9+nj = ω_z, 10+nj = ω_y, 11+nj = ω_x |
| `q̇_j` | 13+nj.. | 12+nj.. | 12+nj.. |
| size | `nx = 13+2nj` | `ndx = 12+2nj` | `n_t = 12+2nj` |

- `q = [p_W, ξ̂, q_j]` and `v = state.tail(nv)`, both Pinocchio-native.
- Start indices: joints 6 → 7; joint velocities `12+nj` → `13+nj` (`WBAccelMpcRobotModel.h:85-89`).
- The input `[W_l, W_r, q̈_j]` is unchanged.

**Per-robot sizes** (taken from today's `initialState` sizes):

| Robot / MPC | nj | `nx` Euler → quaternion | `ndx = n_t` | `nq` / `nv` | N (dt) | rate | feedback |
|---|---|---|---|---|---|---|---|
| DRC Atlas, centroidal | 24 | 36 → 37 | 36 | 31 / 30 | 50 (0.02) | 50 Hz | on |
| EngineAI SA01, centroidal | 12 | 24 → 25 | 24 | 19 / 18 | 50 (0.02) | 80 Hz | on |
| Unitree G1, centroidal | 23 | 35 → 36 | 35 | 30 / 29 | 60 (0.02) | 80 Hz | off |
| Unitree R1, centroidal | 22 | 34 → 35 | 34 | 29 / 28 | 60 (0.02) | 80 Hz | off |
| Unitree G1, whole-body | 23 | 58 → 59 | 58 | 30 / 29 | 32 (0.035) | 60 Hz | off |

### 2.4 The state manifold

The state lives on a product manifold `M = ∏ segments`, where each segment is either Euclidean(n) or UnitQuaternion (ambient 4, tangent 3):
- centroidal: `[E(9), Q, E(nj)]`
- whole-body: `[E(3), Q, E(nj+6+nj)]`

Operations, given for the quaternion segment (Euclidean segments use the obvious counterparts):

| Operation | Definition |
|---|---|
| `retract(x, δ, α)` | `ξ ⊗ Exp(α·δφ)`, then renormalize (removes round-off) |
| `difference(x0, x1)` | `δ ∈ T_{x0}` with `x0 ⊕ δ = x1`: `Log(ξ0⁻¹ ⊗ ξ1)`, shortest path. This is Pinocchio's convention. |
| `interpolate(x0, x1, α)` | slerp, shortest path; Euclidean lerp elsewhere |
| `project(x)` | `ξ ← ξ/‖ξ‖` |
| `E(x)`, `E⁺(x)` | `blockdiag(I, ½G(ξ), I)`, `blockdiag(I, E⁺(ξ), I)` |
| `getMaximumRotationAngle(δ)` | max over quaternion segments of ‖δφ‖ (used by the warm-start guard) |

Derivative plumbing is applied in place on the 4 quaternion columns/rows only, never as a dense E. This costs O(rows·12) per term per node.

### 2.5 Continuous dynamics

**Centroidal** (`cm/src/PinocchioCentroidalDynamicsAD.cpp:75-94`; `cmpc/src/dynamics/CentroidalDynamicsBasisInputsAD.cpp:197-224`).

The momentum rows are unchanged code (`cm/src/ModelHelperFunctions.cpp:166-194`). The orientation enters only through Pinocchio forward kinematics.

```math
\dot{\bar h}=\begin{bmatrix}g+\tfrac1m\sum_i f_i\\ \tfrac1m\sum_i\big((r_i-c)\times f_i+\tau_i\big)\end{bmatrix},\qquad
\begin{bmatrix}\dot p_W\\ \omega_B\end{bmatrix}=A_b^{-1}(q)\big(m\bar h-A_j(q)\dot q_j\big)
```

```math
A_b=\begin{bmatrix} mI & -m[c-p]_\times R(\xi)\\ 0 & I_G^W R(\xi)\end{bmatrix}
\;\Rightarrow\;
A_b^{-1}=\begin{bmatrix} I/m & [c-p]_\times\, R\,(I_G^W R)^{-1}\\ 0 & (I_G^W R)^{-1}\end{bmatrix}
```

- **Inverse.** `Ab(0,0) = m` and `Ab.block<3,3>(3,0) = 0` still hold, so the closed-form inverse in `cm/include/.../implementation/ModelHelperFunctionsImpl.h:39-47` stays valid as written. `(I_G^W R)⁻¹ = Rᵀ I_G⁻¹` uses 3×3 cofactor inverses only. It is never singular, because `I_G ≻ 0`. Today `I_G^W T_W(Θ)` has `det T_W = −cos θ`.
- **Do not replace it with a 6×6 LU:** `common/src/pinocchio_model/DynamicsHelperFunctions.cpp:231-240` documents why CppADCG cannot record it.

Ambient flow map. The quaternion row uses the **raw** `ξ`; everything else uses `q` built from `ξ̂`.

```math
\dot x=f_a(x,u)=\big[\dot{\bar h};\ \dot p_W;\ \tfrac12 G(\xi)\,\omega_B;\ \dot q_j\big]
```

This replaces the single write `getGeneralizedCoordinates(stateDerivative) = v` at `PinocchioCentroidalDynamicsAD.cpp:91` and `CentroidalDynamicsBasisInputsAD.cpp:221`. Tangent velocity (used by tests): `f_t = E⁺f_a = [ḣ̄; ṗ_W; ω_B; q̇_j]`.

**SRBD branch** (`cm/src/ModelHelperFunctions.cpp:61-79`), ported:
- `r̄ = comToBasePositionNominal` and `Ī = centroidalInertiaNominal` (base frame, from Pinocchio at the nominal posture).
- `com = p − R r̄`, `Ab_TR = m·[R r̄]×·R`, `Ab_BR = R·Ī`.
- `data.Ag` is sized `(6, nv)`.

**Whole-body** (`wbmpc/src/dynamics/DynamicsHelperFunctions.cpp:124-146`):

```math
\dot x=\big[\dot p_W;\ \tfrac12 G(\xi)\,\omega_B;\ \dot q_j;\ a_b(q,v,u);\ \ddot q_j\big],\qquad a_b=[\ddot p_W;\dot\omega_B]
```

- `a_b` comes from the unchanged CppAD-safe Schur solve (`common/src/pinocchio_model/DynamicsHelperFunctions.cpp:221-264`). The block `A = M_bb[0:3,0:3] = mI` still holds. The rotational Schur complement becomes the body-frame CoM inertia, positive definite at every attitude (today `SᵀIS` in Euler rates, singular at |θ| = 90°). Fix the comment at `:237-238`.
- Sizes `genDim` → `nv` at `:68-69, :106, :131-136`.
- Inverse dynamics `computeJointTorques` and `computeBaseHeldJointTorques` (common `:285-360`, whole-body `:152-194`) are generic given `q` (nq) and `v` (nv). `a_b = 0` still means "held" (zero twist rate).

### 2.6 Taping rules (CppAD / CppADCG, `-ffast-math` at `ocs2/core/include/ocs2_core/automatic_differentiation/CppAdInterface.h:71, 83`)

1. **Safe normalization.**
   - AD: guard the **squared** norm, `n_safe = sqrt(CppAD::CondExpGt(ξᵀξ, ε², ξᵀξ, 1))`, `ξ̂ = ξ/n_safe`, ε = 1e-6 (`quaternionSafeNormalize`, `UnitQuaternionMath.h`). The square root is taken of the selected branch, so neither branch has an unbounded derivative.
   - Not `n = sqrt(ξᵀξ)`, `n_safe = CondExpGt(n, ε, n, 1)`, as first designed: its values are the same, but at the zero quaternion the reverse sweep of `sqrt` multiplies the unselected branch's zero partial by `1/(2n) = ∞`, which CppADCG generates as a plain product, so `0·∞ = NaN` in every reverse-mode (cost) Jacobian. Steps 0, 4 and 5 each showed it (`testQuaternionRootJointCppAd.cpp` `SquaredNormGuardIsDifferentiableAtAZeroQuaternion`, `testBaseOrientationCppAd.cpp`); a forward-mode probe of it is finite, so only a reverse-mode test proves the guard.
   - Double: the same rule with an `if`.
   - A zero quaternion therefore gives `R = I` (Eigen's polynomial `toRotationMatrix`).
   - Never call `Eigen::normalized()`/`normalize()` on `ad_scalar_t`: they record a comparison between variables (`DynamicsHelperFunctions.cpp:231-240`).
2. **Tape point.** `CppAdInterface` tapes at `xp.setOnes()` (`CppAdInterface.cpp:75-77`), so `ξ̂ = ½(1,1,1,1)` is a valid unit quaternion.
3. **No plain `if`, no `Eigen::Quaternion(Matrix3)`, no `AngleAxis`, no log map or atan2 at zero error inside any tape.**
   - The base-orientation residuals (§2.8.1-2.8.3) are analytic double-precision code and are not taped.
   - The taped end-effector residuals keep their existing CppAD-safe forms: `matrixToQuaternion` for the AD scalar, `quaternionDistance`, `rotationMatrixDistanceToPlane`, and the half-angle `footYawError`.
4. **Pinocchio asserts.** Pinocchio's spherical/free-flyer `calc` may assert unit norm. Under `-c opt` (`.bazelrc:31`, the repository default) asserts are compiled out. Step 0 records the `-c fastbuild` outcome. CppAD-taping tests run in the default `opt` mode, and no plan step relies on fastbuild-with-asserts.

### 2.7 Transcription and SQP on the manifold

Here `Φ(x, u)` is OCS2's RK4 node map, with ambient sensitivities `A_a = ∂Φ/∂x` (nx×nx) and `B_a = ∂Φ/∂u`.

**Intermediate node** (`ocs2/oc/src/multiple_shooting/Transcription.cpp:40-94`):

```math
\hat F=\Pi(\Phi),\quad b_k=\mathrm{difference}(x_{k+1},\hat F)
```

```math
A_k=\mathcal J(b_k)\,E^+(\hat F)\,\Pi'(\Phi)\,A_a\,E(x_k),\qquad
B_k=\mathcal J(b_k)\,E^+(\hat F)\,\Pi'(\Phi)\,B_a
```

- `𝒥(b) = blockdiag(I, Jr⁻¹(b_φ), I)`.
- On the quaternion rows, `E⁺(F̂)Π' = 2Gᵀ(ξ̂_F)/‖ξ_Φ‖`, because `Gᵀξ̂ = 0`.
- OCS2's dynamics struct receives `f = b_k`, `dfdx = A_k`, `dfdu = B_k`.
- This is the exact Newton linearization of `x_{k+1} ⊕ δx_{k+1} = F(x_k ⊕ δx_k, u+δu)`, since `Log(Exp(b)Exp(η)) ≈ b + Jr⁻¹(b)η` (finite-difference check 3.4e-11 in the faithful design). The coefficient on `δx_{k+1}` stays −I, as HPIPM expects.

**Other nodes and steps:**
- **Event nodes** (`Transcription.cpp:156-192`): identity jump map, so `b = difference(x_next, x)`, `A = 𝒥(b)`; `dfdu.setZero(ndx, 0)` at `:172`.
- **Costs** (before the projection at `:96-123`): `g_t = Eᵀg_a`, `H_t = EᵀH_aE`, `H_ux,t = H_ux,a·E`. The retraction-curvature term `−¼(∇_ξℓ·ξ)I₃` vanishes by D5. **Constraints:** `C_t = C_a·E`.
- **Initial state** (`ocs2/sqp/sqp/src/SqpSolver.cpp:236`): `δx₀ = difference(x[0], x_init)`. The same applies to the initial violation at `:422-424, :474-476`.
- **Step** (`:519-520`, `ocs2/oc/include/ocs2_oc/multiple_shooting/Helpers.h:52-66`): `x_new = x ⊕ αδx`; inputs keep `incrementTrajectory`.
- **Metrics** (`ocs2/oc/src/multiple_shooting/MetricsComputation.cpp:99-100, 127-128`): `dynamicsViolation = difference(x_next, Π(Φ))`. Violation norms, the Armijo metric and the filter (`FilterLinesearch.cpp`) are unchanged formulas on tangent vectors, in the same units (radians) as today.
- **Warm start** (`ocs2/oc/src/multiple_shooting/Initialization.cpp:54-58`, `Initialization.h:65-68`): manifold interpolation of the previous solution.
  - Guard: if `getMaximumRotationAngle(difference(x[0], x_init)) > π/2`, discard the previous solution and initialize from `x_init`. This is impossible in nominal real-time iteration and protects against the Log cut locus after a reset or fall.
- **No re-anchoring exists or is needed.** Every iterate lies on M and every difference is intrinsic.
- **Policy** (`Helpers.cpp:82-118`): with a non-flat manifold, build a `ManifoldLinearController(time, x̄, u*, K, manifold)`.
  - The `uff -= K·x` at `:95` is not used.
  - Pre-event nodes copy anchor, input and gain from the previous node, as `:84-90` copies `uff`/`K` today.

```math
u(t,x)=(1-\alpha)\big[u^*_k+K_k\,\mathrm{difference}(\bar x_k,x)\big]+\alpha\big[u^*_{k+1}+K_{k+1}\,\mathrm{difference}(\bar x_{k+1},x)\big]
```

- **Unchanged:** HPIPM (sizes from `OcpSize.cpp:55-59`; `δx₀` eliminated), the projection (`Transcription.cpp:96-123`, `Helpers.cpp:38-58`), `remapProjectedGain` (`Helpers.cpp:50-56`), `PerformanceIndexComputation`, and `FilterLinesearch`. State-only equality constraints never reach the QP (`SqpSolver.cpp:394-400`), and none is added: there is no norm constraint.
- **Guards:**
  - `createValueFunction` throws with a non-flat manifold (`SqpSolver.cpp:172-191, 321-329`; off by default per `SqpSettings.h:58`).
  - `GaussNewtonDDP`'s constructor throws with a non-flat manifold.
  - IPM and SLP are not Bazel targets (`ocs2/BUILD.bazel` defines core, oc, robotic_tools, mpc, qp_solver, ddp, hpipm, sqp, pinocchio and ros2), so they need no guard.

### 2.8 Costs and constraints

#### 2.8.1 Heading–tilt residual (base orientation, ACoM)

For a unit `ξ` (the cost always uses `ξ̂`):

```math
\rho=\sqrt{\xi_w^2+\xi_z^2},\qquad \psi_h(\xi)=2\,\mathrm{atan2}(\sigma\xi_z,|\xi_w|),\ \sigma=\operatorname{sign}\text{ that brings }(\xi_z,\xi_w)\text{ to }\xi_w\ge0,\qquad
q_t(\xi)=\tfrac1\rho(0,0,\xi_z,\xi_w)
```

```math
s(\xi)=q_t^{-1}\otimes\xi=\Big(\tfrac{\xi_w\xi_x+\xi_z\xi_y}{\rho},\ \tfrac{\xi_w\xi_y-\xi_z\xi_x}{\rho},\ 0,\ \rho\Big),\qquad \xi=q_t\otimes s
```

```math
n=\big\|s_{xy}\big\|,\qquad
\kappa(n,\rho)=\begin{cases}2\,\mathrm{atan2}(n,\rho)/n & n\ge 10^{-6}\\ \tfrac{2}{\rho}\big(1-\tfrac{n^2}{3\rho^2}\big) & n<10^{-6}\end{cases},\qquad \tau(\xi)=\kappa\,s_{xy}
```

**Judge fix:** `n` is `‖s_xy‖`, never `sqrt(1−ρ²)`. Because `s_w = ρ ≥ 0`, `τ = Log(s)_xy` needs no sign flip and |τ| ≤ π. In double code, clamp `ρ ≥ 1e-9` (upside-down guard).

The residual, rows in the YAML order (yaw, pitch, roll):

```math
e_{HT}(\xi,\xi_r)=\big[\ \mathrm{wrap}_\pi(\psi_h(\xi)-\psi_h(\xi_r)),\ \ \tau_y(\xi)-\tau_y(\xi_r),\ \ \tau_x(\xi)-\tau_x(\xi_r)\ \big]
```

Jacobian with respect to the body right perturbation of `ξ`: `J_HT(ξ) = [∇_ξψ_h; ∇_ξτ_y; ∇_ξτ_x]·½G(ξ)`. The closed-form ingredients (coefficient order x, y, z, w):
- `∇ψ_h = (2/ρ²)(0, 0, w, −z)`
- `N = (wx+zy, wy−zx)`, `∂N_x = (w, z, y, x)`, `∂N_y = (−z, w, −x, y)`, `∂ρ = (0, 0, z, w)/ρ`, `s_xy = N/ρ`
- `∂κ/∂n = (2ρ/(n²+ρ²) − κ)/n` (series `−4n/(3ρ³)`), `∂κ/∂ρ = −2/(n²+ρ²)`
- `∂τ = κ·∂s_xy + s_xy·(∂κ/∂n·ŝ_xyᵀ∂s_xy + ∂κ/∂ρ·∂ρ)`

All of these are pinned by finite-difference tests.

Properties (verified numerically in the design phase; re-checked by tests):

| Property | Value |
|---|---|
| `J_HT` at any level attitude `q_z(ψ)` | `P₃`; hence `J_HT·T_B(0) = I`: the Euler-coordinate Hessians are equal at a level **state** (any reference). At a tilted state the Euler-chart Jacobian `J_HT(φ(Θ))·T_B(Θ)` deviates from `I` by about `\|tilt\|/2` (the yaw row picks up `−tan(θ/2)`, `−tan(φ/2)`), even against a level reference, so `J_eᵀQJ_e − Q = O(\|tilt\|·\|Q\|)` (`testBaseOrientation.cpp` `EulerChartJacobianIsTheIdentityOnlyAtALevelState`) |
| Single-axis tilt | `τ = (φ, θ)` exactly |
| Mixed tilt, relative error against (φ, θ) | 0.066 % at 0.05 rad, 0.26 % at 0.1, 1.06 % at 0.2 |
| Yaw row against Euler | `e_yaw − (ψ−ψ_r) = −2atan(tan(θ/2)tan(φ/2)) + 2atan(tan(θ_r/2)tan(φ_r/2))` (≈ −½(θφ − θ_rφ_r)) |
| Yaw-error independence | tilt rows exactly independent of the yaw error |
| Pitch 90° | finite, full rank (singular values about 1.95, 1.00, 0.80) |
| Double cover | `s(−ξ) = s(ξ)`; `ψ_h(−ξ) = ψ_h(ξ)` exactly (hemisphere-read heading), so the residual is invariant bit for bit; the raw `2·atan2(ξ_z, ξ_w)` would shift by 2π, and an exactly upside-down `−ξ` with negative zeros would read `±2π` |
| Singularities | tilt π only (upside down); yaw row discontinuous at a yaw error of exactly ±π (kept away by re-seeded references, §2.9) |
| SA01 weights (0/5/85) at mixed tilt | cost −0.15 % at 0.05 rad, −0.61 % at 0.1, −2.4 % at 0.2 |

#### 2.8.2 Tuning deviation and the quadratic state costs

These replace `StateInputQuadraticCost` (`common/src/cost/StateInputQuadraticCost.cpp:60-88`), `StateQuadraticCost` (`StateQuadraticCost.cpp:42-48`) and `BasePoseShapedQuadraticStateCost` (`.cpp:37-41`).

**Centroidal residual:**

```math
r=\big[\bar h-\bar h_r;\ p-p_r;\ e_{HT}(\hat\xi,\hat\xi_r);\ q_j-q_{j,r}\big],\qquad J_t=\mathrm{blockdiag}(I_9,\ J_{HT}(\hat\xi),\ I_{n_j})
```

**Whole-body residual:**

```math
r=\big[p-p_r;\ e_{HT};\ q_j-q_{j,r};\ \dot p-\dot p_r;\ e_\omega;\ \dot q_j-\dot q_{j,r}\big],\qquad
e_\omega=P_3\big(\omega_B-R(\hat\xi)^\top R(\hat\xi_r)\,\omega_{B,r}\big)
```

`∂e_ω/∂ω_B = P₃` and `∂e_ω/∂δφ = −P₃[R(ξ̂)ᵀR(ξ̂_r)ω_{B,r}]×`.

**Cost and Gauss–Newton derivatives:**

```math
L=\tfrac12 r^\top Q r+\tfrac12\delta u^\top R\,\delta u,\qquad \delta u=u-u_r
```

```math
\nabla_x L=J^\top Q r,\quad \nabla^2_{xx}L=J^\top Q J,\quad \nabla_u L=R\,\delta u,\quad \nabla^2_{uu}L=R,\quad \nabla^2_{ux}L=0,\qquad J=J_t\,P(x)
```

- `P(x)` is the robot model's physical-tangent Jacobian (§3.2): in the quaternion layout it is `E⁺` on the orientation block. So the cost returns ambient derivatives (D4) and the pull-back recovers `J_tᵀQr` and `J_tᵀQJ_t` exactly.
- The cross term `P` of OCS2's class is unused on every shipped robot and is dropped.
- `J` is assembled block-wise (index copy plus a 3×4 block).

**References:**
- running: `x_r = referenceManager.getDesiredState(targets, x, t)`, which slerps internally (§2.9);
- `u_r = referenceManager.getDesiredInput(...)`;
- terminal: `x_r = shapeBasePose(t, robotModel.getDesiredState(targets, t))`, scaled by `terminalCostScaling`.

`zeroBasePoseWeights` (`common/src/cost/ComAndAcomTrackingCost.cpp:236-238`, rows 6..11) keeps its tuning indices.

#### 2.8.3 `ComAndAcomTrackingCost` (Atlas)

Files: `common/src/cost/ComAndAcomTrackingCost.cpp:30-43, 61-65, 135-238`; `common/src/acom/AngularCenterOfMass.cpp:49-55, 285-305`; `common/include/humanoid_common_mpc/acom/AngularCenterOfMass.h:44-64, 146-166`.

```math
\xi_a(x)=\hat\xi\otimes \mathrm{Exp}\big(\Delta\theta(q_j)\big),\qquad e_{acom}=e_{HT}\big(\xi_a(x),\xi_a(x_r)\big),\qquad e_{com}=c(q)-c(q_r)
```

```math
\frac{\partial e_{acom}}{\partial\delta\varphi}=J_{HT}(\xi_a)\,R(\Delta\theta)^\top,\qquad
\frac{\partial e_{acom}}{\partial\delta q_j}=J_{HT}(\xi_a)\,J_r(\Delta\theta)\,J_{\Delta\theta}(q_j),\qquad
\frac{\partial e_{com}}{\partial(p,\delta\varphi,\delta q_j)}=J_{com}
```

- **Judge fix:** the `R(Δθ)ᵀ` adjoint and `J_r(Δθ)` are included.
- `Δθ` is the network's base-frame rotation offset. The network is fitted at the identity base (`humanoid_learning/acom/dataset_generator.py:367-387`), so no retraining is needed, and the permutation `acomXyzToZyx` (`AngularCenterOfMass.h:44-64`) disappears.
- `J_com` is Pinocchio's 3×nv CoM Jacobian, placed in the configuration tangent columns through `getPinocchioTangentJacobian` (§3.2).
- The `nq == 6+nj` check at `:61` and `kGeneralizedBaseDim = 6` are replaced by robot-model dimensions.
- `Q_acom` rows stay (yaw, pitch, roll).
- Agreement with today's additive rule: first order, O(|Δθ|·|tilt| + |Δθ|²).

#### 2.8.4 Analytic terms that use Pinocchio Jacobians

- **`ContactWrenchConeConstraint`** (`common/src/constraint/ContactWrenchConeConstraint.cpp`):
  - Delete the zero-state probe (`:83-96`) and `CHECK_EQ(nq, nv)` (`:151-156`).
  - `dfdx = (A_f[f_l]× + A_τ[τ_l]×)·J_ang,LOCAL·V(x)`, where `V(x) = getPinocchioTangentJacobian(x)` (nv×nx) replaces the probed selection matrix at `:428-445`. `J_ang,LOCAL` is already the body tangent. Exact (E1).
- **`ExternalTorqueQuadraticCostAD`** (`common/src/cost/ExternalTorqueQuadraticCostAD.cpp:124-130`): size `J_ee` as (6, **nv**). Exact.

#### 2.8.5 Terms whose residual is unchanged

These read `q`, `v` and `a` through accessors; only start indices and library folders change.

| Term | File(s) | Note |
|---|---|---|
| Torso `EndEffectorKinematicsQuadraticCost` | `common/src/cost/EndEffectorKinematicsQuadraticCost.cpp:88-147`; `EndEffectorKinematicCostHelpers.cpp:116-124` | Keep `quaternionDistance` (world-frame half-angle sine) so `orientation_x/y/z` weights keep their meaning. Do not switch to a log map: that changes the scale by about 2. Reference from FK of the slerped, shaped `x_ref`. |
| `CentroidalMpcEndEffectorFootCost` | `cmpc/src/cost/CentroidalMpcEndEffectorFootCost.cpp:106-201`; `.h:94-109` | Unchanged plane distance and half-angle foot yaw. Plane-normal yaw from `getBaseHeading(x_ref)` (`SwitchedModelReferenceManager.cpp:150-153`). Delete the dead `xRef`/`uRef` at `:154-155`. |
| `EndEffectorDynamicsFootCost` (whole-body) | `wbmpc/src/cost/EndEffectorDynamicsFootCost.cpp:93-154` | Unchanged. Accelerations come from the new ω-state flow. Delete the dead `xRef`/`uRef` at `:136-137`. |
| `ZeroVelocityConstraintCppAd` → `EndEffectorKinematicsTwistConstraint` | `cmpc/src/constraint/ZeroVelocityConstraintCppAd.cpp:75-98`; `common/src/constraint/EndEffectorKinematicsTwistConstraint.cpp:96-199` | Unchanged (`rotationMatrixDistanceToPlane`, `M(R, ω)`). |
| `ZeroAccelerationConstraintCppAd` → `EndEffectorDynamicsAccelerationsConstraint` | `wbmpc/src/constraint/...:84-146` | Unchanged. Fix the stale comment at `:118-119`. |
| Normal-velocity, swing-vertical, `ContactMomentXY`, `FrictionForceCone`, `ForceWeightedSlip`, `ContactComplementarity`, `GroundPenetration`, `FootCollision`, `DcmTerminalCost`, `ICPCost`, `JointTorqueCostCppAd` | as in the costs map | E1 (§4.4). `DcmTerminalCost.cpp:241` reads the target through the model interpolation. |
| `InputQuadraticCost`, `ZeroWrench`, `BasisScalingNonNegativity`, `JointLimitsSoftConstraint`, both `JointMimic*` | — | E0. Indices through `getJointStartindex()`/`getJointVelocitiesStartindex()`. |
| Contact-planner SO(2) terms (`common/src/contact_planning/{cost,constraint}/*`) | — | Unchanged. Only the boundary reads change (§2.9). |

### 2.9 References, heading and the Euler boundary

**Human-facing layer.** `TargetTrajectoriesCalculatorBase` (`common/src/command/TargetTrajectoriesCalculatorBase.cpp`) stays Euler: pose `[x, y, z, ψ, θ, φ]`, unbounded scalar ψ, Δψ in degrees.
- `getCurrentBasePoseTarget` (`:150-157`) becomes `[getBasePosition(x), getBaseHeading(x), 0, 0]`.
- `integrateTargetBasePose` (`:180-195`) and `getDeltaBaseTarget` (`:119-144`) are unchanged.
- `estimateTimeToTarget` (`:201-209`) uses the commanded scalar Δψ.

**Knot assembly.** Each calculator writes a knot through `setBasePoseEulerZyx(state, pose6)`, which in the quaternion layout writes `ξ = q_z(ψ)⊗q_y(θ)⊗q_x(φ)`.
- Centroidal: `cmpc/src/command/CentroidalMpcTargetTrajectoriesCalculator.cpp:84-86, 192-195`.
- Whole-body: `wbmpc/src/command/WBMpcTargetTrajectoriesCalculator.cpp:65-66, 127-130`.
- The Eigen comma initializers are deleted.
- **Whole-body velocity knot:** `ṗ_W,r = (v_x, v_y, 0)`, and `setBaseAngularVelocityFromWorldYawRate(state, ψ̇_cmd)` writes `ω_{B,r} = R(ξ̂_r)ᵀ(0, 0, ψ̇_cmd)`, which is `(0,0,ψ̇)` for level references. This replaces the comma-initialized Euler-rate slot at `:93-94`.
- **Centroidal momentum target** (`:132-141`, `L/m = I_G(:,2)ψ̇/m`) is unchanged; it is the channel that actually drives yaw, since `Q_yaw = 0` everywhere.

**Pose-command subdivision.** `n = max(1, ⌈|Δψ|/(π/2)⌉)` equal segments in time, yaw and position, interpolated linearly. Slerp about z between subdivided level knots is exactly linear in yaw, so the result equals today's two-knot linear Euler interpolation. The velocity-command knot spacing is at most `1 rad/s × 0.84 s = 0.84 rad < π/2`.

**Interpolation.** OCS2's `TargetTrajectories::getDesiredState` stays linear for flat systems (`ocs2/core/src/reference/TargetTrajectories.cpp:77-83`). The humanoid interpolates every state trajectory through `MpcRobotModelBase::getDesiredState(targets, t)`, which slerps the orientation block. Sites:
- `StateInputQuadraticCost.cpp:79`, `StateQuadraticCost.cpp:46`, `BasePoseShapedQuadraticStateCost.cpp:40`
- `ComAndAcomTrackingCost.cpp:158, 181`
- `EndEffectorKinematicsQuadraticCost.cpp:92`
- `CentroidalMpcEndEffectorFootCost.cpp:154`, `DcmTerminalCost.cpp:241`
- `SwitchedModelReferenceManager.cpp:150, 334, 344, 386`
- `ContactPlanningReferenceManager.cpp:225, 598-631` (the resampler stores interpolated knots)
- `EndEffectorDynamicsFootCost.cpp:136`
- `HumanoidVisualizer.cpp:349, 389`
- `HumanoidTelemetryPublisher.cpp:168`, `PinocchioTelemetryPublisher.cpp:431, 555`

A lint rule bans single-argument `getDesiredState(` and `LinearInterpolation::interpolate(` on state trajectories under `humanoid_nmpc/` outside the robot-model helper (Appendix B).

**Reads of the robot's heading use the twist `ψ_h`** (via `getBaseHeading`). Every consumer is wrap-safe (cos/sin or `moduloAngleWithReference`):
- swing-plane normal (`SwitchedModelReferenceManager.cpp:150-153`)
- `measuredBaseYaw_` (`:294`)
- arm swing (`:396-398`)
- heuristic-context yaw (`:427`)
- contact-planner input (`ContactPlanningReferenceManager.cpp:800`)
- planner heading (`:176-181`), which uses `ψ_h(ξ_a)` with the ACoM, else `ψ_h(ξ)`
- current-pose target (`TargetTrajectoriesCalculatorBase.cpp:150-157`)
- reset targets

The foot yaws at `:183-191` (`atan2(R10, R00)` of the contact frame) are unchanged.

**Edits of level references at the Euler boundary:**
- `shapeBasePose` (`SwitchedModelReferenceManager.cpp:413-450`, seam `:437-447`): `(ψ, θ, φ) = getBaseOrientationEulerZyx(x)`, add the offsets with the `maximumTilt` clamp as today, then `setBaseOrientationEulerZyx`. This is exactly today's Euler-additive arithmetic for any reference, and the IFTTT `base_pose_heuristic_seam` stays.
- `PlannedHeadingOverride` (`common/src/contact_planning/execution/PlannedHeadingOverride.cpp:40-59`): keep "replace ZYX yaw (re-branched with `moduloAngleWithReference`), keep θ and φ" through the same accessors. This is exact for the level knots it edits.

**Yaw range.** The state never stores an angle, so no wrap exists in the solver. The yaw residual is a wrapped relative twist. References are re-seeded from the measured heading at every command and are densified, so per-node yaw errors stay far from π.

### 2.10 Euler tuning mapping (exact)

The index map from the tuning layout to the state is §2.3.

| Tuning item | New meaning | Exactness |
|---|---|---|
| `Q`, `Q_final` rows other than orientation and whole-body ω; off-diagonals; `R`; `terminalCostScaling` | same quantities | identical |
| `Q`/`Q_final` yaw row ("theta_base_z") | `wrap(ψ_h − ψ_h,r)`² | ≈ −½(θφ − θ_rφ_r) shift; weight 0 on every shipped robot (Atlas `task.yaml:883`, SA01 `:755`, G1c `:513`, R1 `:494`, G1-WB `:297`) |
| `Q`/`Q_final` pitch, roll rows ("theta_base_y/x") | `(τ_y − τ_y,r)²`, `(τ_x − τ_x,r)²` | exact for single-axis tilt; mixed tilt per §2.8.1; independent of the yaw error; Hessians equal at any level state (any reference), first order in the tilt otherwise |
| Whole-body rows 32..34 ("omega_base_z/y/x", Euler rates today) | `e_ω` rows (z, y, x) | equal at level; O(tilt·rate) otherwise; zero error for correct world-yaw-rate tracking at any tilt; shipped weights isotropic (3/3/3) |
| `Q_acom` (yaw, pitch, roll) | heading–tilt of `ξ̂⊗Exp(Δθ)` | first order |
| `Q_com`, task-space torso and foot weights, `foot_constraint.*`, `contactWrenchConeSoftConstraint.*`, `swingPitchAngle`, `reference.yaml` rates, `contact_planning.yaml` | unchanged | exact (heading reads ≤ ½\|θφ\| for measured headings) |
| `initialState` (centroidal) | Euler rows 9..11 → `ξ = q_z q_y q_x` | exact rotation |
| `initialState` (whole-body) | rows 3..5 → ξ; rows 32..34 = ω_B (z, y, x) (D18) | shipped zeros, so identical |
| `locomotion_heuristics.orientation_compensation.*`, `periodic_orientation.*`, `maximumTilt` | Euler offsets at the seam | exact (all `base_pose` lists empty as shipped) |
| Velocity and pose commands | Euler/yaw scalar, converted per knot | exact for level targets |
| Reset heading | `q_t(ξ_obs)` | twist rather than ZYX yaw, ≤ ½\|θφ\| |
| G1-WB `task_space_foot_cost_weights` | loader fixed, YAML set to effective values (Q2) | identical |

**YAML comments** become GUI slider labels. They are reworded **keeping the first token**, which `humanoid_finite_state_machine.py:296-308` filters on. For example:

```yaml
"(10,10)": 5.0  # theta_base_y: pitch, tilt about the heading-frame y axis [rad]
"(32,32)": 3.0  # omega_base_z: base angular velocity about body z [rad/s]
```

### 2.11 MRT, reset, policy, rollout, visualization, telemetry

**Observation builders** move into robot-model helpers, so the MRTs are layout-agnostic from Step 6:
- `getGeneralizedCoordinatesFromMeasurement(p_W, ξ_meas, q_j)`
- `getGeneralizedVelocitiesFromMeasurement(R, v_B, ω_B, q̇_j)`

**Centroidal** `updateMpcState` (`cmpc/src/mrt/CentroidalMpcMrtJointController.cpp:227-248`):
- `ξ = robotState.getRootRotationLocalToWorldFrame().coeffs()`.
- `CHECK(std::abs(ξ.norm() − 1) < 1e-3)`, then normalize.
- `q = [p, ξ, q_j]`, `v = [R·v_B, ω_B, q̇_j]` (sized nv; fixes `:237`).
- `h̄ = A_g(q)·v/m`; write through setters.
- `quaternionToEulerZYX` (`:230`) and the cos θ division (`:239-240`) disappear. No sign alignment is needed (D12).

**Whole-body** `updateMpcState` (`wbmpc/src/mrt/WBMpcMrtJointController.cpp:201-216`): set `p`, `ξ`, `q_j`, `ṗ_W = R·v_B`, `ω_B`, `q̇_j` directly.

**Reset targets:**
- Centroidal (`:899-931`): `targetState(10) = targetState(11) = 0` (`:906-908`) becomes `setBaseOrientation(x, q_t(ξ_obs))`.
- Whole-body (`:653-673`): `segment<2>(4)` (`:660`) becomes the same heading projection, and velocities are zeroed with `tail(nv)` (`:657`).

**Other MRT details:**
- Gravity compensation (centroidal `:936-955`, whole-body `:307-315`): measured quaternion; zero velocity sized nv (`:949`).
- The "Base pitch" diagnostic (`:498-501`) becomes `getBaseOrientationEulerZyx(x)(1)` (clamped `asin`).
- `evaluatePolicy` (centroidal `:414`, whole-body `:516`): unchanged call. `ManifoldLinearController` and the MRT's manifold interpolation of the plan state handle it.
- Whole-body inverse dynamics on the planned state (`:527-529`): the slerped plan is unit.

**Whole-body `getBaseComVelocity`** returns `[ṗ_W; P₃ω_B]` ("tuning-order base velocity"). Index 3 is ω_z ≈ ψ̇ and index 5 is ω_x ≈ φ̇, so `ProceduralMpcMotionManager.cpp:175-234` and `WBMpcTargetTrajectoriesCalculator.cpp:100-105` keep today's meaning to first order (Q1 defers their fixes). Centroidal `getBaseComVelocity` (h̄) is unchanged.

**Rollouts and dummy sims.** `RolloutBase::setStateManifold`; `TimeTriggeredRollout` (`ocs2/oc/src/rollout/TimeTriggeredRollout.cpp:89`) projects every output state and its `clone()` keeps the manifold. `StateTriggeredRollout` and `InitializerRollout` do neither, so they refuse a non-null manifold (`supportsStateManifold()`, a `CHECK`) instead of ignoring it; the humanoid uses `TimeTriggeredRollout` only. The dummy loops (`ocs2/ros2_interfaces/src/mrt/MRT_ROS_Dummy_Loop.cpp:182-195`) feed back projected states, so no humanoid dummy-loop subclass is needed.

**Visualization:**
- `publishBaseTransform(position, quaternion)` (`ros2c/src/visualization/HumanoidVisualizer.cpp:126-175`).
- The zero state at `:81` becomes `getNeutralState()`.
- `SimFsmBridge.cpp:234-250` sets the `RobotState` rotation from `getBaseOrientation(initState)`.

**Telemetry:**
- `common/include/humanoid_common_mpc/common/Types.h:126-150` splits `FLOATING_BASE_DIM`/`JOINT_COORDINATE_OFFSET` into:
  - `BASE_CONFIGURATION_DIM = 7` (names `base_x, base_y, base_z, base_qx, base_qy, base_qz, base_qw`)
  - `BASE_VELOCITY_DIM = 6` (`base_vx, base_vy, base_vz` world; `base_wx, base_wy, base_wz` body)
  - with separate configuration and velocity joint offsets.
- `PinocchioTelemetryPublisher.cpp:369-403, 454` builds `q`/`v` through the measurement helpers. The singular `θ̇(ω_W)` at `:386-387` is removed.
- The Euler topics (`/robot/base_euler`, `/mpc/target_base_euler`, `:541-575`; `HumanoidTelemetryPublisher.cpp:60-187`) are kept, derived through `EulerBoundary` with a clamped `asin`.

### 2.12 Exact identities for the equivalence tests

Chart `Φ`: identity on `h̄`, `p`, `q_j` (and `ṗ`, `q̇_j`), `φ(Θ) = q_z(ψ)q_y(θ)q_x(φ)` on angles, and whole-body `ω_B = T_B(Θ)Θ̇`. Let `P_e(x)` be the physical-tangent Jacobian in Euler coordinates (§3.2).

```math
D\varphi(\Theta)=\tfrac12\,G(\varphi(\Theta))\,T_B(\Theta),\qquad
f_t(\Phi(x_e),u)=C(x_e)\,f_e(x_e,u),\qquad C(x_e)=\text{Euler}\to\text{physical tangent}
```

```math
J_e=J_t\,P_e(x_e),\qquad H_e=P_e^\top H_t P_e\ \ (\text{Gauss–Newton}),\qquad \partial f_t/\partial u=C\,\partial f_e/\partial u
```

```math
(\partial f_t/\partial\delta)\,C=C\,A_e+(\partial C/\partial x_e)[f_e]
```

These hold at any attitude with cos θ ≠ 0, not only for small errors.

---

## 3. File-by-file changes

### 3.1 Vendored OCS2 (`lib/ocs2`)

**New:**
- `core/include/ocs2_core/manifold/StateManifold.h` (abstract), `EuclideanStateManifold.h`, `ProductStateManifold.h`, `UnitQuaternionMath.h`, plus `core/src/manifold/*.cpp`.
  - `UnitQuaternionMath.h` holds the templated `quaternionRateMatrix<SCALAR_T>`, Exp/Log, Jr/Jr⁻¹ and slerp, and is used by the humanoid flow maps.
  - `StateManifold` API: `getAmbientDim`, `getTangentDim`, `retract(x, dx, alpha, xNew)`, `difference(x0, x1)`, `interpolate(x0, x1, alpha)`, `project(x&)`, `getMaximumRotationAngle(dx)`, `pullBackStateColumns(x, J&)`, `pullBackGradient(x, g&)`, `pullBackHessian(x, H&)`, `pushForwardDynamics(xNext, phi, f&, A&, B&)`, `pushForwardJump(xNext, x, f&, A&)`, `liftStateColumns(x, J&)`.
  - Held as `std::shared_ptr<const StateManifold>` (immutable, shared across worker threads, no mutable caches).
  - The new sources are picked up by the existing globs in `ocs2/BUILD.bazel:68-90`.
- `core/include/ocs2_core/control/ManifoldLinearController.h` + `core/src/control/ManifoldLinearController.cpp`:
  - `computeInput` as in §2.7;
  - `flatten` per time as `[u*; x̄; vec(K)]`, with `x̄(t)` the slerped anchor, `K(t)` the interpolated gain and `u*(t) = computeInput(t, x̄(t))` (not the interpolated nominal input), so the flattened law is the node's at the nodes, `computeInput()`'s at every time on a flat manifold, and off by a term bilinear in the anchor distance and the deviation otherwise: an encoder may sample on its own grid;
  - `unFlatten(stateDim, inputDim, tangentDim, time, data, manifold)`.
- `oc/include/ocs2_oc/multiple_shooting/ManifoldProjection.h` + `.cpp`: the node, event, cost and constraint pull-backs of §2.7 as free functions.
- `oc/include/ocs2_oc/oc_data/StateTrajectoryInterpolation.h`: manifold-aware interpolation of a state trajectory (linear when the manifold is `nullptr`).

**Changed:**
- `core/include/ocs2_core/control/ControllerType.h:37`: add `MANIFOLD_LINEAR`.
- `core/src/automatic_differentation/CppAdInterface.cpp:140-155, 299-305`: after `loadModels`/`loadModelsIfAvailable`, check `model_->Domain() == variableDim_ + parameterDim_` and `Range() == rangeDim_` (`rangeDim_` from an earlier `createModels()` or the copied interface, else one off-tape evaluation of the function). Throw `std::runtime_error` naming the library folder and telling the user to delete it or set `recompileLibrariesCppAd`. Add `getTapeOperationCount()` (`fun.size_op()`), used by the benchmark.
- `oc/include/ocs2_oc/oc_problem/OptimalControlProblem.h:48-138` (+ copy/assign/swap in the `.cpp`): `std::shared_ptr<const StateManifold> stateManifoldPtr;` default `nullptr`.
- `oc/src/multiple_shooting/Transcription.cpp:54-55, 62, 66-91, 125-154, 156-192`: with a manifold, call `ManifoldProjection` after discretization and before `projectTranscription` (`:96-123`); `dfdu.setZero(ndx, 0)` at `:172`.
- `oc/src/multiple_shooting/MetricsComputation.cpp:99-100, 127-128`: manifold violation.
- `oc/src/multiple_shooting/Initialization.cpp:54-58, 73` and `Initialization.h:65-68`: manifold interpolation.
- `oc/include/ocs2_oc/multiple_shooting/Helpers.h:52-66`: `retractTrajectory(manifold, x, dx, alpha, xNew)`.
- `oc/src/multiple_shooting/Helpers.cpp:82-118`: build a `ManifoldLinearController` when a manifold is set.
- `oc/src/rollout/TimeTriggeredRollout.cpp:40, 89` + `RolloutBase` setter: project outputs.
- `sqp/sqp/src/SqpSolver.cpp`:
  - `:236`, `:422-424`, `:474-476`: difference;
  - `:379-382`, `:412`: empty blocks sized ndx;
  - `:519-520`: retract;
  - `:331-344`: controller;
  - `:216-217`: warm-start rejection guard;
  - `:172-191, 321-329` + constructor: throw for `createValueFunction` with a non-flat manifold.
- `mpc/include/ocs2_mpc/MRT_BASE.h` + `mpc/src/MRT_BASE.cpp:122-134`: `setStateManifold`; plan-state slerp in `evaluatePolicy`.
- ~~`ros2_msgs/msg/MpcFlattenedController.idl:12-14`: `CONTROLLER_MANIFOLD_LINEAR = 3`.~~ Dropped with ROS; the protobuf encoder adds the wire code to `humanoid_mpc_msgs/controller_type.proto`.
- `ros2_interfaces/src/mpc/MPC_ROS_Interface.cpp:112-119`: new case.
- `ros2_interfaces/src/mrt/MRT_ROS_Interface.cpp:170-180` (+ `.h` `setStateManifold`): new case.
- `ddp/src/GaussNewtonDDP.cpp` constructor: throw on a non-flat manifold.
- `cm/include/ocs2_centroidal_model/CentroidalModelInfo.h:68-73`, `cm/src/CentroidalModelInfo.cpp:135-154`:
  - `generalizedCoordinatesNum` → `configurationDim` (nq) and `velocityDim` (nv);
  - `actuatedDofNum = nv − 6`, `stateDim = 6 + nq`, new `stateTangentDim = 6 + nv`;
  - copy all of them in `toCppAd()`.
- `cm/src/FactoryFunctions.cpp:46-52, 74-79, 91-103, 115-123`: root joint; check `model.nv == nj + 6`; `qPinocchioNominal << 0,0,0, 0,0,0,1, joints`; `vPinocchioNominal` sized nv.
- `cm/include/ocs2_centroidal_model/AccessHelperFunctions.h:96-103` + `implementation/AccessHelperFunctionsImpl.h:126-174`:
  - `getBasePose` is split into `getBasePosition` (3 at 6) and `getBaseQuaternion` (4 at 9);
  - `getJointAngles` starts at 13;
  - `getGeneralizedCoordinates` is the raw `(6, nq)` slice;
  - new `configurationDerivative(stateDerivative, info)` (nq rows at 6).
- `cm/include/.../CentroidalModelPinocchioMapping.h:44-60` + `cm/src/CentroidalModelPinocchioMapping.cpp:75-167`:
  - `getPinocchioJointPosition` uses the safe `ξ̂` (§2.6);
  - velocity sized nv (`:102`);
  - header comments rewritten;
  - `getOcs2Jacobian` (`:112-167`) throws "differentiated by CppAD only", as `WBAccelPinocchioStateInputMapping.h:58-61` does today.
- `cm/src/ModelHelperFunctions.cpp:61-79, 90-127, 150-161`: SRBD port; `Ag` sized (6, nv); delete `updateCentroidalDynamicsDerivatives` and `getTranslationalJacobianComToContactPointInWorldFrame` (no caller outside the deleted analytic path).
- `cm/include/.../implementation/ModelHelperFunctionsImpl.h:52-193`: delete `getMappingZyxGradient`, `getRotationMatrixZyxGradient`, `getCentroidalMomentumZyxGradient`.
- `cm/src/PinocchioCentroidalDynamicsAD.cpp:57, 91`: flow map §2.5.
- `pinocchio/pinocchio_interface/src/PinocchioEndEffectorKinematics.cpp:135, 310, 348`: Jacobians sized `model.nv`.

**Deleted** (after a `grep` shows no caller in `lib/ocs2` targets or `humanoid_nmpc`):
- `cm/src/PinocchioCentroidalDynamics.cpp` + header. This also removes the latent `3*inputIdx` bug at `:128-129`.
- `cm/src/CentroidalModelRbdConversions.cpp` + header. This also removes the latent twist misuse at `:184-187, 225`.
- `cm/test/testAnymalCentroidalModel.cpp` (not a Bazel target).

**New `cc_test`s in `ocs2/BUILD.bazel`:** see §5c.

### 3.2 `humanoid_common_mpc`

**New:**
- `include/humanoid_common_mpc/orientation/BaseOrientation.h` + `src/orientation/BaseOrientation.cpp`:
  - `twistHeading`, `headingQuaternion` (`q_t`), `swing`, `tiltVector`;
  - `headingTiltError(ξ, ξ_r)` returning `{residual, jacobianBodyTangent}`;
  - safe normalization helpers;
  - `configurationTangentLift` (E⁺).
  - Uses `ocs2_core/manifold/UnitQuaternionMath.h`.
- `include/humanoid_common_mpc/orientation/EulerBoundary.h` + `.cpp`, the only humanoid home of Euler conversions:
  - `quaternionFromEulerZyx`;
  - `eulerZyxFromQuaternion` (clamped `asin`; replaces `quaternionToEulerZYX` from `pinocchio_model/DynamicsHelperFunctions.h:436-450`);
  - `localAngularVelocityFromEulerZyxRates` (T_B);
  - `stateFromTuningLayout`/`tuningLayoutFromState` helpers, returning `absl::StatusOr<vector_t>` that rejects a wrong size and names the expected layout.
- `include/humanoid_common_mpc/common/StateLayout.h`: `constexpr absl::string_view kStateLayoutTag = "translation_spherical_quaternion_v1";`, under `LINT.IfChange(cppad_layout_tag)`.
- `include/humanoid_common_mpc/cost/TuningDeviation.h`: abstract, double-only, one class.
  - `size_t getDim() const`
  - `vector_t evaluate(const vector_t& state, const vector_t& stateRef) const`
  - `TuningDeviationApproximation linearize(const vector_t& state, const vector_t& stateRef) const` returning the residual and an `n_t × nx` Jacobian in state coordinates.
- `include/humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h` + `.cpp`:
  - derives `ocs2::StateInputCost`;
  - holds `Q_`, `R_` and a `std::shared_ptr<const TuningDeviation>`;
  - `setStateWeights(matrix_t)`, `setInputWeights(matrix_t)`, `getStateWeights()`, `getInputWeights()`;
  - final `getValue`/`getQuadraticApproximation`;
  - protected pure virtual `getReferences(time, state, input, targets) -> std::pair<vector_t, vector_t>`.
- `include/humanoid_common_mpc/cost/TuningQuadraticStateCost.h` + `.cpp`: the terminal counterpart deriving `ocs2::StateCost`.

**Changed:**
- `src/pinocchio_model/createPinocchioModel.cpp:69-76`: root joint (Step 8).
- `include/humanoid_common_mpc/common/MpcRobotModelBase.h:53-127, 252-257`:
  - **Removed:** `getGenCoordinatesDim`, `gen_coordinates_dim`, `base_dim`. The compiler then lists every ambiguous nq/nv site; there are 9 files using `getGenCoordinatesDim` and 12 using `generalizedCoordinatesNum`.
  - **Added:** `getConfigurationDim()`, `getVelocityDim()`, `getStateTangentDim()`, `getBaseOrientationStartIndex()`; `getBasePosition`/`setBasePosition`; `getBaseOrientation` (normalized quaternion) / `setBaseOrientation`; `getBaseHeading` (twist; returns the stored ZYX yaw in the Euler layout until Step 7); `getDesiredState(targets, t)` (layout interpolation); `getNeutralState()`; `getGeneralizedCoordinatesFromMeasurement`, `getGeneralizedVelocitiesFromMeasurement`; `getPhysicalTangentJacobian(state)` (ndx × nx); `getPinocchioTangentJacobian(state)` (nv × nx).
  - **Renamed, Euler boundary only** (lint-banned in formulation directories): `getBasePose` → `getBasePoseEulerZyx`, `setBasePose` → `setBasePoseEulerZyx`, `getBaseOrientationEulerZYX` → `getBaseOrientationEulerZyx`, `setBaseOrientationEulerZYX` → `setBaseOrientationEulerZyx`. In the quaternion layout the getter returns a wrapped yaw and a clamped pitch.
  - `getGeneralizedCoordinates(state)` returns `q` with the safe `ξ̂`.
- `include/humanoid_common_mpc/common/BasisInputsModelDecorator.h:168-215` and `BasisInputsMappingDecorator.h:63-90`: forward the new API.
- `include/humanoid_common_mpc/cost/StateInputQuadraticCost.h:39`, `src/cost/StateInputQuadraticCost.cpp:60-88`; `StateQuadraticCost.{h:40, cpp:42-48}`: re-based on `TuningQuadraticStateInputCost`. `StateQuadraticCost` has an empty input block.
- `include/.../cost/BasePoseShapedQuadraticStateCost.h:53`, `src/cost/BasePoseShapedQuadraticStateCost.cpp:37-41`: re-based on `TuningQuadraticStateCost`; the reference is `shapeBasePose(t, model.getDesiredState(targets, t))`.
- `src/cost/InputQuadraticCost.cpp:36-65` + header: derives `ocs2::StateInputCost` directly with `setInputWeights(R)`. No state block exists, which fixes the `zeroQ` hazard (`MpcParameterUpdaterModule.cpp:974, 1009`).
- `src/HumanoidCostConstraintFactory.cpp:231-256, 262-282, 330-343, 497-523`: `Q`/`Q_final` sized `getStateTangentDim()` (`:232-233, 263-264, 502-503`); construct the new classes with the formulation's `TuningDeviation`.
- `src/cost/ComAndAcomTrackingCost.cpp`, `src/acom/AngularCenterOfMass.cpp`, `include/.../acom/AngularCenterOfMass.h`: §2.8.3. Keep `computeJointOrientationOffset(q_j)` and `computeJointOffsetJacobian(q_j)`; add `acomAttitude(ξ_base, q_j)`; delete the Euler-additive API.
- `src/constraint/ContactWrenchConeConstraint.cpp:83-96, 151-156, 411-455`: §2.8.4.
- `src/cost/ExternalTorqueQuadraticCostAD.cpp:125`: `getVelocityDim()`.
- `src/pinocchio_model/DynamicsHelperFunctions.cpp:237-238` comment; `include/.../DynamicsHelperFunctions.h:436-450` (`quaternionToEulerZYX` moved into `EulerBoundary`), `:483, :508` docs ("[base configuration (7), joint angles]"), `:524-526` warning re-examined.
- `src/reference_manager/SwitchedModelReferenceManager.cpp:136-160, 288-325, 383-450`: §2.9.
- `src/command/TargetTrajectoriesCalculatorBase.cpp:119-209` + header: §2.9 (heading read, subdivision helper `subdivideYawKnots`). The self-assignment in `setTargetRotationVelocity_` (`TargetTrajectoriesCalculatorBase.h:108`) is fixed in Step 2.
- `src/contact_planning/ContactPlanningReferenceManager.cpp:176-181, 225, 598-631, 800`: §2.9.
- `src/contact_planning/execution/PlannedHeadingOverride.cpp:53-57`: accessor rename only.
- `src/locomotion_heuristics/LocomotionHeuristicModelParameters.cpp:89`: `getBasePosition`.
- `src/contact/FootprintCornerHeights.cpp:130-133`: hash `getConfigurationDim()` and `getVelocityDim()`.
- `src/common/ModelSettings.cpp:189`: `modelFolderCppAd = absl::StrCat("cppad_code_gen/cppad_", mpcName, robotName, "/", kStateLayoutTag)`. The centroidal contact-input key (`cmpc/src/CentroidalMpcInterface.cpp:333-342`) nests below it.
- `include/humanoid_common_mpc/common/Types.h:126-150`: §2.11.

### 3.3 `humanoid_centroidal_mpc`

- `include/humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h:52-196`: layout §2.3; the new API; `state_dim = 13+nj`; `getJointStartindex = 13`; physical-tangent and Pinocchio-tangent Jacobians.
- New `include/humanoid_centroidal_mpc/cost/CentroidalTuningDeviation.h` + `src/cost/CentroidalTuningDeviation.cpp`.
- `src/dynamics/CentroidalDynamicsBasisInputsAD.cpp:197-224`: flow map §2.5 (`:221`). `CentroidalDynamicsAD.{h,cpp}` (`:38-64`) needs no change beyond the new `info`.
- `src/CentroidalMpcInterface.cpp`:
  - `:159` `loadDefaultJointState(model.nv − 6, …)`;
  - `:176-177` `initialState` via `stateFromTuningLayout`;
  - register the manifold on the problem (Step 8);
  - `:333-342` key nested under the layout tag;
  - `:617-639` callback via accessors;
  - the updater gets `getStateTangentDim()`;
  - expose `getStateManifold()`.
- `src/command/CentroidalMpcTargetTrajectoriesCalculator.cpp:84-86, 107-109, 143-148, 192-195`: §2.9; Step 2 rewrite of `:143-148` (Q1 default).
- `src/mrt/CentroidalMpcMrtJointController.cpp:90, 227-262, 414, 457-469, 498-501, 899-931, 936-955`: §2.11; `mcpMrtInterface_.setStateManifold(...)` in the constructor.
- `src/mrt/MpcParameterUpdaterModule.cpp:434-448, 656-661, 723-730, 974-1031`:
  - `stateDim_` renamed `tuningDim_`;
  - casts become `get<TuningQuadraticStateInputCost>("stateInputQuadraticCost")`, `get<TuningQuadraticStateInputCost>("stateQuadraticCost")`, `get<InputQuadraticCost>("inputQuadraticCost").setInputWeights(R)`, and `finalCostPtr->get<TuningQuadraticStateCost>(kQuadraticTerminalCostTerm)`;
  - `zeroQ` is removed.
- `src/cost/CentroidalMpcEndEffectorFootCost.cpp:154-155`, `src/cost/DcmTerminalCost.cpp:241`: model interpolation; remove dead variables.
- `test/support/ProblemFingerprint.{h,cpp}`, `AtlasReferenceStack.cpp`, `DrcAtlasContactTestModel.cpp`: layout-agnostic accessors and tangent-sized LQ data. The Transcription pull-back happens automatically.

### 3.4 `humanoid_wb_mpc`

- `include/humanoid_wb_mpc/common/WBAccelMpcRobotModel.h:47-212`:
  - layout §2.3; new API; `getBaseComVelocity` = `[ṗ_W; P₃ω_B]`;
  - `getBaseAngularVelocityLocal`/`setBaseAngularVelocityLocal`, `setBaseAngularVelocityFromWorldYawRate`;
  - remove `getBaseEulerZYXDerivatives`/`setBaseOrientationEulerZYXDerivatives`.
- New `include/humanoid_wb_mpc/cost/WBTuningDeviation.h` + `src/cost/WBTuningDeviation.cpp`.
- `src/dynamics/DynamicsHelperFunctions.cpp:51-146`: §2.5.
- `src/WBMpcInterface.cpp:114, 132-134, 233-247, 331`: `initialState` via `stateFromTuningLayout`; manifold registration; rollout manifold; `getStateManifold()`.
- `src/command/WBMpcTargetTrajectoriesCalculator.cpp:51-74, 80-138`: §2.9 (`getGenCoordinatesDim` uses at `:65-66` become accessors).
- `src/mrt/WBMpcMrtJointController.cpp:75, 83-85, 201-216, 307-315, 509-548, 636-673`: §2.11; `setStateManifold`.
- `src/cost/EndEffectorDynamicsCostHelpers.cpp:107-110`: loader fix (Step 2).
- `src/constraint/EndEffectorDynamicsAccelerationsConstraint.cpp:118-119`: comment (Step 2).
- `src/cost/EndEffectorDynamicsFootCost.cpp:136-137`: model interpolation; remove dead variables.
- **Deleted** (Step 2; dead and broken): `src/cost/EndEffectorDynamicsQuadraticCost.cpp` + header (and the `BUILD.bazel` srcs entry), `include/humanoid_wb_mpc/common/WBAccelPinocchioStateInputMapping.h` (and its include at `wbros2/src/WBMpcDummySimNode.cpp:40`).

### 3.5 ROS 2 packages

- `ros2c/src/visualization/HumanoidVisualizer.cpp:75-81, 126-175, 184, 323-391`; `ros2c/src/telemetry/PinocchioTelemetryPublisher.cpp:68-76, 194-247, 369-403, 429-454, 541-575`; `ros2c/src/telemetry/HumanoidTelemetryPublisher.cpp:60-187`; `ros2c/src/fsm/SimFsmBridge.cpp:234-250`: §2.11.
- `cros2/src/CentroidalMpcDummySimNode.cpp:33, 88`, `wbros2/src/WBMpcDummySimNode.cpp:32, 40, 72-101`: remove the analytic `PinocchioEndEffectorKinematics` include; `mrt.setStateManifold`; rollout manifold; initial observation from the converted `initialState`. Also remove the include at `testHumanoidVisualizer.cpp:40`.
- `cros2/src/CentroidalMpcRobotSim.cpp`, `wbros2/src/WBMpcRobotSim.cpp`, `cros2/src/CentroidalMpcSqpNode.cpp`, `wbros2/src/WBMpcSqpNode.cpp`: manifold wiring only. The pose-command nodes (`CentroidalMpcKeyboardPoseCommandNode.cpp:88-92`, `wbros2/src/WBMpcPoseCommand.cpp:93-95`) are unchanged (Δψ in degrees through the base calculator).
- `ros2c/src/ros_comm/MRTPolicySubscriber.cpp:53-114`: no change; its controller decoding is commented out and it reads states only. Note this in its header.

### 3.6 Python, tools, configuration, documentation

- `humanoid_nmpc/humanoid_common_mpc_pyutils/humanoid_common_mpc_pyutils/mpc_observation_logger.py:55-70`, `mpc_observation_inspector.py:49-51`: columns `qx, qy, qz, qw`; base velocity columns `vx, vy, vz` (world) and `wx, wy, wz` (body) for the whole-body model.
- `humanoid_learning/acom/models.py:142-166`: `full_acom_pose` becomes quaternion composition (`acom_attitude`), mirroring the C++, with its test.
- `tools/plotjuggler/humanoid_telemetry.xml`: new names.
- `remote_control`: no code change (the YAML layout is unchanged). `humanoid_finite_state_machine.py:296-308` relies on the kept tokens.
- Five `task.yaml` files (Atlas, SA01, G1 centroidal, R1, G1-WB): reworded orientation comments (§2.10) and `LINT.IfChange(tuning_layout)` blocks around `initialState`/`Q`/`Q_final`. G1-WB `task_space_foot_cost_weights` (`:476-494`) gets the effective values (Q2), with a comment stating the values previously listed but never applied.
- `tools/hooks/lint_code.py` + tests: Appendix B.
- `humanoid_nmpc/docs/quaternion_base_orientation/README.md` (new; Appendix C); index in `humanoid_nmpc/docs/README.md`; update `robot_models/drc_atlas/README.md:15` ("base Euler angles" block 6..11), the root `README.md` if it describes the state, and `humanoid_learning/acom/README.md`.

---

## 4. What is removed, what stays Euler, and how the switch is verified

### 4.1 Removed from the formulation

- `SphericalZYX` roots (`createPinocchioModel.cpp:73`, `cm/src/FactoryFunctions.cpp:48-50, 75-77`).
- Euler/Euler-rate state slots and their accessors, including the whole-body Euler-rate API and `getGenCoordinatesDim`.
- `q̇ = v` in all three flow maps.
- The ZYX gradients and analytic Euler code in `cm/` (§3.1).
- Euler conversions in both MRTs and in telemetry.
- The additive ACoM rule.
- Raw `x − x_ref` deviations in the quadratic costs.
- The `nq == nv` shortcuts (`ContactWrenchConeConstraint.cpp:151-156, 428-445`; `ComAndAcomTrackingCost.cpp:61, 199-217`; `AngularCenterOfMass.cpp:49-55`).
- The dead whole-body classes.

`RotationDerivativesTransforms.h` stays in `ocs2/robotic_tools` with no humanoid production caller outside `EulerBoundary`.

### 4.2 Stays Euler: the conversion points

All conversions go through `EulerBoundary` or the renamed `*EulerZyx` accessors, and nowhere else (lint-enforced):
1. the `initialState` loaders (`CentroidalMpcInterface.cpp:176-177`, `WBMpcInterface.cpp:132-134`);
2. `TargetTrajectoriesCalculatorBase` and the knot assembly in both calculators;
3. the heuristic seam `shapeBasePose`;
4. `PlannedHeadingOverride`;
5. the pose-command nodes;
6. telemetry Euler topics, logs and the visualizer's human-facing outputs;
7. YAML comments and the GUI;
8. Python YAML readers (`tools/locomotion_heuristics/derive_parameters.py:233-246`, `humanoid_learning/acom/tests/test_acom.py:76-84, 694`), which are unaffected.

### 4.3 Verification ladder

| Label | Recorded at | Content |
|---|---|---|
| **M0**, **B0** | Step 1, on the worktree with Steps 0, 2, 4 and 5 already applied (the harness needs Step 5's heading-tilt split for its metrics; see "M0/B0 provenance" below) | closed-loop metrics, all five configurations; solve-time benchmark and tape operation counts |
| **G1**, **M1** | Step 3, after the behavior-preserving fixes | Euler goldens with configuration snapshots; closed-loop metrics. The reference commit hash is recorded in the golden README (tag `euler-formulation-final` if the user agrees). |
| **G2**, **M2** | end of Step 7 | quaternion-formulated costs on Euler coordinates |
| **M3**, **B3** | Step 10 | final formulation |

| Gate | Comparison |
|---|---|
| Step 2 | Bitwise per fix, by its own tests (G1-WB foot weights: the rewritten block applies the weights the legacy loader applied, on frozen copies of both blocks). M0 already contains Step 2, so M0 ≈ M1 cannot detect a Step 2 change; it gates Step 3 only. Centroidal knots differ by ≤ \|v_b\|/(2m)·0.7T. |
| Step 4 | Flat path bitwise: `test_sqp_flat_parity` (a toy-problem golden recorded before the change, its data hash pinned in the test; on the main line, whose HPIPM is built for TARGET=GENERIC, the solver is compared with a re-recording for that build, which the main line's solver from before the merge of Step 4 reproduces bit for bit on that build and which the test holds to the original recording but for the feedback gains of the configuration HPIPM's IPM solves) and every manifold branch gated on a null manifold. M0/B0 contain Step 4 too, so they cannot detect a flat-path change; B0 includes Step 4's flat-path overhead (a null check per node), which the B3-vs-B0 gate therefore does not charge to the switch. |
| Step 6 | ProblemFingerprint and G1 reproduce to 1e-12 relative (bitwise where no conversion is involved) |
| Step 7 | G1 under E2 tolerances (§4.4); M2 vs M1 within bands |
| Step 8 | G2 under E1 tolerances (only coordinates change); G1 under E2; all of §5c |
| Step 10 | M3 vs M1/M2 within bands; B3 vs B0 within the real-time gate |

**M0/B0 provenance.** "HEAD as-is" was not achievable: the harness's metrics use Step 5's heading-tilt split, and the
worktree's Boost-free state with Steps 0, 2, 4 and 5 exists only as one uncommitted state, with no snapshot of the state
before them. M0 and B0 therefore measure that state (`data/closed_loop/M0/README.md`, `data/benchmark/B0/README.md`), and
the neutrality of Steps 2 and 4 rests on their own bitwise evidence (the Step 2 and Step 4 rows above), not on M0. A
later baseline that must catch a change of Step 2 or Step 4 would have to be recorded on a state without them.

### 4.4 Equivalence classes and tolerances

| Class / check | Relation | Tolerance | Reason |
|---|---|---|---|
| E0: InputQuadratic, ZeroWrench, FrictionForceCone, BasisScalingNonNegativity, JointLimits, JointMimic | values, Jacobians up to index relabeling | bitwise, or 1e-12 if indices move | no orientation |
| E1: every kinematic or dynamic term (§2.8.4-2.8.5), inverse dynamics, FK | value at `Φ(x_e)`; `J_e = J_t·P_e`; `H_e = P_eᵀH_tP_e` | values 1e-9 relative (absolute floor 1e-10); Jacobians 1e-8 relative (floor 1e-9) | round-off, CppADCG `-ffast-math` reassociation, Euler↔quaternion conversion |
| Dynamics value and input Jacobian | `f_t = C·f_e`, `∂f_t/∂u = C·∂f_e/∂u` | momentum rows 1e-10; tangent rows 1e-9 | same physics |
| Dynamics state Jacobian | §2.12 with `∂C/∂x` by central finite difference in the test | 1e-7 | finite-difference error |
| RK4 node map | `A_t·P_e(x_k) = P_e(x_{k+1})·A_e` | 1e-6 | RK4 is coordinate-dependent at O(dt⁵) |
| E2: Q/Q_final orientation rows (G1 vs Step 7/8) | value | yaw ≤ ½\|θφ − θ_rφ_r\| + 1e-12; tilt ≤ 3e-5 rad at \|tilt\| ≤ 0.05; cost value relative ≤ 2e-3 at 0.05 rad and ≤ 2.5e-2 at 0.2 rad | twist vs ZYX yaw; cubic tilt terms (§2.8.1) |
| E2: same rows, gradient and Hessian at a level **state** (zero roll and pitch of the state, any reference) | through `P_e` | 1e-12 | `J_HT·T_B(0) = I` |
| E2: same rows, gradient and Hessian at a tilted state | through `P_e` | `‖J_eᵀQJ_e − Q‖_max ≤ \|tilt\|·‖Q‖_max` (and `‖J_e − I‖_max ≤ 0.6\|tilt\|`) | `J_HT(φ(Θ))T_B(Θ) = I + O(\|tilt\|)`, coefficient ½ |
| E2: whole-body ω rows | value | ≤ \|tilt\|·\|Θ̇\| + 1e-12 | `T_B` vs `P₃` |
| E2: ComAndAcom | value, Jacobian | ≤ c·(\|Δθ\|·\|tilt\| + \|Δθ\|²), with c set from the measured `max |Δθ|` recorded in G1 | multiplicative vs additive composition |
| G2 vs Step 8 | every term (now E1) | as E1 | only coordinates change |
| SQP step, tier 1 (Step 8 vs G2, level attitude, zero attitude error, zero base rate) | `du`; `dx_t` vs `C·dx_e` | 1e-6 relative | RK4 coordinate dependence only |
| SQP step, tier 2 (Step 8 vs G1 at attitude perturbations ε = 0.02 and ε/2) | `du`, `C·dx` | discrepancy ratio between ε and ε/2 in [3, 5] | O(ε²) |
| Reference knots | `eulerZyx(ξ_k) = Θ_k` (yaw mod 2π) | 1e-12 | exact conversion |
| Reference node grid (slerp vs linear Euler, level knots) | Θ | 1e-12 | slerp about z is linear in yaw |
| Seam output, swing normals | — | 1e-12 | level references |
| Heading reads (measured) | twist vs ZYX | ≤ ½\|θφ\| + 1e-12 | D9 |

### 4.5 Closed-loop metrics and bands

**Scenarios** (all five configurations; lockstep runner, §5d):
- standing 10 s;
- walking at 0.5 m/s for 15 s (0.3 m/s if M0 shows Atlas limited);
- lateral 0.2 m/s;
- turning in place at +0.5 rad/s through 720° and back;
- turning at 1 rad/s for 12.6 s;
- an arc at 0.3 m/s and 0.3 rad/s.

**Metrics:**
- survival (fall flag);
- base-height mean, standard deviation and RMS error;
- tilt RMS and maximum (|τ|);
- velocity RMS error in the heading frame (the frame of the velocity command it is compared with; on a tilted base it differs from the body frame by the tilt); yaw-rate RMS error; cumulative unwrapped heading;
- stance-foot slip;
- joint torque RMS;
- maximum rotation part of ‖δx₀‖ per solve;
- maximum |‖ξ_k‖ − 1| over all nodes and solves;
- NaN count;
- solve-time p50/p99 per phase (`getBenchmarks()`);
- failures and resets.

**Bands** (against M1, and against M2 for Step 10):

| Metric | Band |
|---|---|
| survival | equal or better |
| height | within 5 mm |
| tilt | within max(20 %, 0.005 rad) |
| velocity | within max(15 %, 0.02 m/s) |
| yaw rate | within max(15 %, 0.02 rad/s) |
| slip | within max(20 %, 5 mm) |
| p99 solve time | within +10 % |
| quaternion norm | `max |‖ξ‖−1|` < 1e-9 |

**720° turn exception** (`compareClosedLoopRuns`, `MetricBands.h`). M1 records the Euler behavior at the ±π crossings: today's measured yaw is `atan2`-wrapped (`DynamicsHelperFunctions.h:443`) against an unwrapped warm start, which gives a 2π `δx₀` at `SqpSolver.cpp:236`. A run whose baseline's heading crosses an odd multiple of π (located in the baseline's archived time series; the turning scenarios' series are archived beside their documents) is compared:
- on the velocity and yaw-rate errors recomputed from both time series outside ±0.5 s windows around the baseline's crossings, within the same bands;
- on reaching the baseline's heading peak (−0.1 rad), and the commanded 4π ± 0.1 rad where the baseline reached it. As designed the run "must reach 4π ± 0.1"; in M0 no robot does (the whole-body G1 peaks at 5.3 rad, the centroidal robots at ≤ 0.5 rad), so the commanded peak is required only where the baseline reaches it;
- on no spike in ‖δx₀‖: `initial_state_gap.max_rotation_rad` (the SQP's own first-iteration `δx₀`) below π/2, which a 2π wrap exceeds. A run on Euler coordinates (M1, M2, a main-line rerun of M0: no quaternion norm) wraps where its own heading crosses the cut, as its Euler baseline does, and its gap then decays over a few solves (on the whole-body G1, from 2π to below π/2 within 0.17 s). Against a baseline whose own gap reached π/2 it is checked solve by solve, from the per-solve gaps of its time series (`solve_initial_state_rotation_gap`): a solve at or above π/2 must lie within ±0.5 s of one of the run's own crossings and below 2π + π/2, no more than a wrap of a gap below π/2. A spike elsewhere, a larger one, or one the series cannot locate fails, as does every spike of a quaternion run;
- on every other band as usual.
A baseline whose heading reaches ±π without an archived time series is a violation of its own.

### 4.6 Real-time gate

`benchmark_mpc_solve` (`testonly` `cc_binary`, centroidal and whole-body) runs N solves on states recorded by the lockstep runner while walking, with the configured `nThreads`, built at Step 1 (B0; with Steps 0, 2, 4 and 5 applied, see §4.3) and at Step 10 (B3) on the same machine, one at a time; `compareSolveBenchmarks` (`SolveBenchmarkGate.h`, `make benchmark-mpc-solve ... BASELINE=B0`) checks the gate, matching each library across the layout-tagged folder of Step 8. Accept if:
- mean solve time ≤ +5 %;
- p99 ≤ +10 %;
- LQ-approximation phase ≤ +8 %;
- each robot keeps p99 < 80 % of its MPC period (20 / 12.5 / 16.7 ms);
- the tape operation count of each CppAD library is ≤ +5 % (machine-independent).

A failure blocks Step 10 until the pull-backs are block-optimized.

---

## 5. Test plan

### (a) Golden data from the Euler formulation (recorded before anything is removed)

**Infrastructure** (Step 1, each piece with its own tests):
- `common/test/support/GoldenIo.{h,cpp}`: labeled matrices at `%.17g`, in the style of `ProblemFingerprint`/`common/test/data/contact_planning`. The header carries the commit hash and a hash of every configuration file.
- `cmpc/test/golden/recordEulerGolden.cpp` and `wbmpc/test/golden/recordEulerGolden.cpp` (`testonly` `cc_binary`), with Make targets `make record-euler-golden ROBOT=…` run one robot at a time.
- Data in `cmpc/test/data/euler_golden/<robot>/{G1,G2}/` and `wbmpc/test/data/euler_golden/g1/{G1,G2}/`, each with a `config/` snapshot of `task.yaml`, `reference.yaml` and `contact_planning.yaml`. The equivalence tests run the new code on the snapshots, so later tuning cannot invalidate them. Record selected nodes only, keeping files under 1 MB.

**Points per robot:**
- `initialState`;
- roll/pitch ±0.02, ±0.05, ±0.2 (single-axis), and combined (0.05, −0.03) and (0.2, 0.1);
- yaw 1.3, 2.5, −3.0 (with 0.05 tilt);
- pitch 80° (records `cond(A_b,22)` to document the Euler singularity; excluded from equivalence);
- 5 seeded random states (tilt ≤ 0.3, uniform yaw);
- mid-swing walking states from `setWalkingReferences` / `AtlasReferenceStack`;
- whole-body: the same, plus |Θ̇| ≤ 1.

**Quantities:**
- the Euler state, the physical state `(h̄, p, ξ, q_j [, ṗ_W, ω_B, q̇_j])` and `P_e(x)`;
- `f`, `∂f/∂x`, `∂f/∂u`; RK4 `Φ`, `A`, `B`;
- per named cost: value, gradient, Gauss–Newton Hessian (state, input, cross); per named constraint: value and Jacobians; soft-constraint penalties;
- inverse-dynamics torques (both MRT paths); SRBD dynamics on G1 (test copy with `centroidalModelType: 1`); the measured `max |Δθ|` (Atlas ACoM);
- loaded and zeroed `Q`/`Q_final`/`Q_acom`;
- references:
  - target knots for velocity commands {0, ±1 rad/s, 0.5 m/s} at initial yaw {0, 1.3, ±3.0};
  - pose commands Δψ ∈ {90°, 270°, −135°};
  - `getDesiredState` on the node grid;
  - `shapeBasePose` with a heuristic enabled (test config copy);
  - swing-plane normals (Atlas `swingPitchAngle: 0.08`);
  - nominal footholds;
  - planner inputs and `PlannedHeadingOverride` output (Atlas, contact planning enabled in a test copy);
  - reset targets;
- SQP: one real-time-iteration step from a fixed warm start through `SolverBase::run(..., const PrimalSolution&)` (`SolverBase.h:103`) at perturbations ε and ε/2, recording `dx`, `du`, α, the performance index and `K` at node 0 (Atlas, SA01). Plus the iteration log of a converged solve.

**Closed loop:** M0 and M1 as in §4.5.

### (b) Equivalence after the formulation and coordinate steps

New tests, all reading the snapshot configurations:
- `cmpc/test/testEulerGoldenDynamicsEquivalence.cpp`
- `cmpc/test/testEulerGoldenTermEquivalence.cpp`
- `cmpc/test/testEulerGoldenReferenceEquivalence.cpp`
- `cmpc/test/testEulerGoldenSqpStepEquivalence.cpp`
- the whole-body counterparts in `wbmpc/test/`

Tolerances are those of §4.4. Run against G1 at Steps 6 (1e-12), 7 (E2) and 8 (E2), and against G2 at Step 8 (E1).

### (c) Quaternion-specific tests

| Test (target) | Checks |
|---|---|
| `ocs2/core/test/manifold/testStateManifold.cpp` (`test_state_manifold`) | `difference(x, x⊕δ) = δ` for ‖δ‖ < π; ⊕/⊖ inverse; `E⁺E = I`; `Jr⁻¹` vs finite differences; retraction curvature identity `∂²(ξ⊗Exp(δ)) = −ξ/4·I`; slerp endpoints and the antipodal boundary; in-place pull-backs equal dense `E` products |
| `ocs2/oc/test/multiple_shooting/testManifoldProjection.cpp` | node/event/cost/constraint pull-backs vs finite differences of `F(x⊕δ, u+δu) ⊖ x_next` (≤ 1e-8) |
| `ocs2/core/test/control/testManifoldLinearController.cpp` | node exactness; between-node blend; pre-event copy; `flatten`/`unFlatten` round trip; resampling off the nodes (flat: exact; curved: exact at the anchor, bilinear away); `−x` invariance |
| `ocs2/ros2_interfaces/test/testManifoldControllerMessage.cpp` | `createMpcPolicyMsg` → `readPolicyMsg` round trip of `MANIFOLD_LINEAR` |
| `ocs2/sqp/sqp/test/testSqpOnSO3.cpp` | rigid-body attitude OCP (ambient 7, tangent 6) tracking a 720° heading reference with a simulated plant: monotone heading; rotation part of ‖δx₀‖ < 0.1; same `u` to 1e-12 under `−ξ_init` |
| `ocs2/sqp/sqp/test/testSqpFlatParity.cpp` | a flat toy problem's recorded solve (recorded before Step 4) is bitwise identical with `nullptr` manifold; `test_mpc_reset` unchanged |
| `ocs2/core/test/automatic_differentiation/testCppAdLibraryDimensionCheck.cpp` | a library with a stale domain throws, naming the folder |
| `common/test/testQuaternionRootJointCppAd.cpp` (Step 0, kept) | taping and code generation of `computeCentroidalMap`, `crba`, `nonLinearEffects`, `ccrba`, frame placements and Jacobians (values), the momentum inverse path (with its Jacobian), and `forwardKinematics(q, v, a)` with `getFrameVelocity`/`getFrameClassicalAcceleration` (with its Jacobian, and a scalar of it to second order in the root's quaternion, rate and acceleration) on Translation+Spherical; matches double at random attitudes including pitch 90°; recorded under `-c opt` and `-c fastbuild` (the frame-motion tests under `-c opt` only). Not covered, for Step 8: the production whole-body terms as taped, second order over all their inputs, and the derivatives of the value-only libraries |
| `common/test/testBaseOrientation.cpp` | `J_HT` vs finite differences (ambient and tangent `ξ⊗Exp(εe_i)`); level gives `P₃`; tilt at exactly level (no NaN); pitch 89.9/90/90.1°; upside-down clamp; double cover; yaw independence; §2.8.1 identities |
| `common/test/testEulerBoundary.cpp` | round trips; `asin` clamp at ±90°; `T_B`; `stateFromTuningLayout` size errors |
| `common/test/testConfigurationTangent.cpp` | finite differences of FK and CoM under `⊕` equal Pinocchio `v`-Jacobians (§2.2); `pinocchio::integrate` on the composite equals `⊕` |
| `common/test/testTuningQuadraticCost.cpp` | value, gradient, Hessian vs finite differences; Step 6 identity deviation bitwise vs the old class; empty-input variant |
| `common/test/testInputQuadraticCost.cpp` | input-only; `setInputWeights` hot reload (regression for the `zeroQ` flaw) |
| `cmpc/test/testCentroidalQuaternionDynamics.cpp`, `wbmpc/test/testWBQuaternionDynamics.cpp` | AD Jacobians vs finite differences; homogeneity `f(λξ) = [f_other, λf_ξ]` for λ ∈ {0.9, 1.1}; `f(−ξ)` flips only `ξ̇`; pitch 90° finite with `cond(A_b) = cond(I_G)(1 ± 1e-9)`; RK4 norm drift < 1e-8 over a horizon at \|ω\| = 5 |
| `cmpc/test/testSrbdQuaternion.cpp` | SRBD `A_b` equals Full at the nominal posture |
| `cmpc/test/testQuaternionInvariances.cpp` (parameterized Atlas, SA01, G1, R1) + `wbmpc/test/testWBQuaternionInvariances.cpp` | for every registered term and the dynamics: tangent finite differences (central, ε = 1e-6, 1e-5 relative); radial invariance `g(cξ) = g(ξ)`, `∇_ξg·ξ = 0` for c ∈ {0.5, 2}; `g(−ξ) = g(ξ)`; pitch 90° finite and one SQP iteration finite; yaw equivariance under `R_z(α)` of state and reference for α ∈ {π−0.01, −π+0.01, 10π+0.1} |
| `cmpc/test/testUnitNorm.cpp` | after retract ≤ 1e-15; slerped targets; rollout projection exact; 60 s dummy loop ≤ 1e-12 |
| `cmpc/test/testTurnInPlace720.cpp` | SQP + projected RK4 rollout (no MuJoCo) at 1 rad/s for 4π: rotation part of ‖δx₀‖ < 0.1; no step > 3× the median in cost or `dynamicsViolationSSE`; monotone heading; bounded feedback correction |
| `cmpc/test/testPoseCommandSubdivision.cpp` | Δψ = 270° turns +270°; knots ≤ π/2 apart; node grid equals linear Euler (1e-12) |
| `cmpc/test/testMrtObservationDoubleCover.cpp` + whole-body counterpart | `ξ` and `−ξ` observations give identical solver inputs, policy inputs and joint commands; a `w ≥ 0`-canonicalized stream crossing ±π gives continuous behavior |
| `common/test/testCppAdLayoutTag.cpp` | the folder contains `kStateLayoutTag` (pattern of `testJointTorqueCostLibraryName.cpp`) |
| `tools/hooks/test_lint_code.py` (extend) | each Appendix B rule fires on a violating snippet and not on the allow-listed boundary |
| `humanoid_nmpc/remote_control/test/test_tuning_layout.py` | every shipped `task.yaml`: Q/Q_final/initialState sizes equal `n_t`; orientation comment tokens start with `theta_base_`/`omega_base_`; the FSM's `joint_val_map` contains no base rows |
| `wbmpc/test/testFootWeightsLoader.cpp` (Step 2) | each YAML key lands in its own field; shipped G1-WB weights produce the previously effective vector |
| `cmpc/test/testIntermediateKnotEffectiveBehavior.cpp` (Step 2) | the rewritten knot differs from the old expression by ≤ \|v_b\|/(2m)·0.7T |
| `common/test/testQuaternionToEulerClamp.cpp` (Step 2) | finite output when \|2(wy − zx)\| exceeds 1 by round-off |
| Python `humanoid_learning/acom/tests` | the `acom_attitude` composition matches the C++ formula on fixtures |

**Migrated tests** (they pin Euler layouts or the composite joint):
- `cmpc/test`: `testAcomWiring`, `testBasisInputsModelDecorator`, `testDcmTerminalCost`, `testFootYawResidual`, `testMrtJointControllerReset`, `testYawCommandDynamics` (`:70`), `testNominalPendulum`, `testContactPlanningIntegration` (`:249`), `testLocomotionHeuristicIntegration` (`:869`), `testMpcParameterUpdaterModule` (`:352, 536, 551, 1379, 1426, 1453, 1476`), `testRelaxedContactConstraints`, `testShippedContactPlanningFiles`, `testMpcResetSolverStack`, `testVelocityCommandFilterWiring`, plus `support/{AtlasReferenceStack, DrcAtlasContactTestModel, ProblemFingerprint}`.
- `humanoid_nmpc/humanoid_centroidal_mpc_test/src`: `testCentroidalMpcRobotModel`, `testDynamicsHelperFunctions`, `testPinocchioFrameConversions`, `testBasisInputsFormulation` (`:92, 610, 853, 925`), `testActiveInStance`, `testAngularCenterOfMass`, `testPinocchioTelemetryPublisher`, `testContactWrenchConeConstraint`. `testCentroidalConversions` tests the deleted RBD conversions and is deleted.
- `common/test`: `testAcomAngularVelocityConsistency` (rewritten in ω terms against `dataset_generator.py:394-431`), `testAngularCenterOfMass`, `testComAndAcomTrackingCost` (`:276-292`), `testFloatingBaseDynamics`, `testJointTorqueInverseDynamics`, `testEndEffectorKinematicsTwistConstraint`.
- Whole-body: `testWBMpcMrtJointController` (`:103-133, 189-197`), `testJointTorqueStateInputOverloads` (`:100-123`, using `pinocchio::randomConfiguration`), `testWBMpcConstruction`, `testBaseHeightFollowsTerrain` (`:178, 320`), `wbros2/test/testWBMpcPoseCommand` (`:118, 143-186`), `testContactWrenchConeBarrierRange`, `testWBContactWrenchConeCreate`, `testJointTorqueCostLibraryName`.
- `ros2c/test/testSimFallRecovery` (`:151`); `robot_models/engineai_sa01/.../testPinocchioModel.cpp`; the GUI tests `remote_control/test/test_live_update_coverage.py`, `test_drc_atlas_parameter_coverage.py` (expected to pass unchanged).
- Rule for all of them: zero states become `getNeutralState()`; raw attitude indices become accessors. A grep audit for `state(9)`, `state(10)`, `state(11)`, `segment<3>(9)`, `segment(3,3)`, `head(6)`, `6 + nj` and `getGenCoordinatesDim` runs before Step 8 closes.

### (d) Closed-loop MuJoCo smoke tests on every robot

- **Harness:** `ros2c/test/support/LockstepClosedLoop.{h,cpp}`.
  - Headless `MujocoSimInterface` (`robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/MujocoSimInterface.h:103`), the MRT joint controller, and `MPC_MRT_Interface::advanceMpc` called synchronously every `k` simulation steps at the task-file rate. No ROS executor; the precedent is `testSimFallRecovery`.
  - Per-formulation drivers live in `cros2/test/closed_loop/` and `wbros2/test/closed_loop/`, extracted from `CentroidalMpcRobotSim.cpp:62-450` and `WBMpcRobotSim.cpp:97-391`.
- **Outputs:** metrics JSON at `<pkg>/test/data/closed_loop/<label>/<robot>_<scenario>.json`.
- **Targets:** `cc_test`s tagged `manual` and `exclusive`, plus `make closed-loop-metrics ROBOT=… SCENARIO=… LABEL=…`, one at a time.
- **Determinism test:** with `nThreads` overridden to 1, two runs agree to 1e-9. Production comparisons use the configured threads and the §4.5 bands.
- **Configurations:** Atlas, SA01, G1 centroidal, R1 and G1 whole-body; scenarios and bands as in §4.5.

---

## 6. Implementation plan

Each step is independently buildable and testable, lands as its own commit or commits, and closes with `format_code.py`, `lint_code.py` and the listed tests.

**Step 0: Spike (go/no-go for D1).**
- Files: `common/test/testQuaternionRootJointCppAd.cpp`, `common/BUILD.bazel`. The test builds a Translation+Spherical G1 model locally (no production change).
- Acceptance: passes under `-c opt`, and the `-c fastbuild` outcome is recorded in the test's header comment.
- If `opt` fails: switch to the FreeFlyer fallback (§7, R1) before any further step, with the same state and tangent layout.

**Step 1: Measurement infrastructure, no production change.**
- Files: `LockstepClosedLoop` + drivers; `benchmark_mpc_solve` (both formulations); `GoldenIo`; `CppAdInterface::getTapeOperationCount()` (additive accessor); Make targets.
- Record M0 and B0 on HEAD. (Done on the worktree with Steps 0, 2, 4 and 5 applied; §4.3, "M0/B0 provenance".)
- Acceptance: `GoldenIo` round trip; metrics-schema test; determinism test; the benchmark runs on all five configurations.

**Step 2: Behavior-preserving fixes, one commit each, each with its test.**
1. `asin` clamp at `common/include/.../DynamicsHelperFunctions.h:445`.
2. Delete `EndEffectorDynamicsQuadraticCost.*` and `WBAccelPinocchioStateInputMapping.h`.
3. Stale comment `EndEffectorDynamicsAccelerationsConstraint.cpp:118-119`.
4. Self-assignment `TargetTrajectoriesCalculatorBase.h:108`.
5. G1-WB foot-weight loader (`EndEffectorDynamicsCostHelpers.cpp:107-110`) plus YAML effective values (Q2).
6. Centroidal intermediate knot made explicit as `0.5·command` (`CentroidalMpcTargetTrajectoriesCalculator.cpp:143-148`; Q1 default), with a comment naming the previous expression and its value.
7. Dead `xRef`/`uRef` removed.
8. `PinocchioEndEffectorKinematics.cpp:135, 310, 348` sized `nv` (a no-op while nq = nv).

Acceptance: unit tests; ProblemFingerprint bitwise for every robot except the centroidal velocity-command knots (bounded as stated); a closed-loop rerun within the §4.5 bands of M0 (G1-WB bitwise).

**Step 3: Record G1 and M1.**
- Files: recorder binaries, data and configuration snapshots, golden README (provenance: commit, configs, the defects deferred by Q1).
- Acceptance: the recorder is deterministic (two runs identical); files committed; sizes under 1 MB each.

**Step 4: OCS2 manifold infrastructure with a flat default (no humanoid change).**
- Files: all of §3.1 except the `cm/` and `pinocchio_interface` items.
- Acceptance:
  - `test_state_manifold`, `testManifoldProjection`, `testManifoldLinearController`, `testManifoldControllerMessage`, `testSqpOnSO3`, `testSqpFlatParity` (bitwise), `testCppAdLibraryDimensionCheck`;
  - the whole humanoid test suite passes unchanged;
  - ProblemFingerprint bitwise equal to G1 (the humanoid still uses `nullptr`).

**Step 5: Orientation library.**
- Files: `orientation/BaseOrientation.*`, `orientation/EulerBoundary.*`, `common/StateLayout.h`.
- Acceptance: `testBaseOrientation`, `testEulerBoundary`, and a CppAD tape test of `quaternionRateMatrix` and safe normalization at `x = ones`.

**Step 6: Layout-agnostic API and new cost classes on Euler coordinates.**
- Files:
  - `MpcRobotModelBase.h`, `CentroidalMpcRobotModel.h`, `WBAccelMpcRobotModel.h` and the decorators: §3.2 API, implemented for the Euler layout (`getBaseHeading` = stored yaw; physical-tangent Jacobian with `T_B`; Pinocchio-tangent Jacobian = selection; `getDesiredState` linear; measurement helpers reproducing today's MRT math);
  - `CentroidalModelInfo` nq/nv split (still equal) and every sizing site (`MRT :232, :237, :949`; `ExternalTorque :125`; `FootprintCornerHeights :130-133`; whole-body dynamics `:68-69, :106`);
  - `TuningDeviation` classes with the identity implementation (`r = x − x_ref`, `J = I`);
  - the `Tuning*` cost classes; `InputQuadraticCost` input-only; factory sizes; updater migration;
  - every consumer of the renamed and new accessors (reference manager, calculators with knot subdivision, planner boundary, `PlannedHeadingOverride`, MRTs via measurement helpers, telemetry with the `Types.h` split at values 6/6, visualizer, `SimFsmBridge`, `ContactWrenchConeConstraint` via `getPinocchioTangentJacobian` with the probe deleted);
  - migrated tests.
- Acceptance:
  - ProblemFingerprint and G1 within 1e-12 relative (bitwise where no conversion is involved);
  - `testTuningQuadraticCost` (bitwise vs old), `testInputQuadraticCost`;
  - updater and GUI tests pass;
  - closed-loop rerun bitwise equal to M1 with `nThreads = 1`.

**Step 7: Formulation changes on Euler coordinates.**
- Files: `CentroidalTuningDeviation`/`WBTuningDeviation` heading–tilt and `e_ω` (§2.8.2) through the Euler physical-tangent Jacobian; `ComAndAcomTrackingCost` + `AngularCenterOfMass` (§2.8.3); `getBaseHeading` becomes the twist; `humanoid_learning/acom/models.py` mirror; tests.
- Acceptance: G1 under E2 tolerances; ACoM tolerance from the measured `max |Δθ|`; closed-loop M2 within bands of M1. Then record G2 and M2.

**Step 8: Coordinate switch (one atomic commit, because the root joint is shared).**
- Files:
  - root joint (`createPinocchioModel.cpp:69-76`, `cm/src/FactoryFunctions.cpp`);
  - `cm/` items of §3.1 (info dims nq ≠ nv, access helpers, mapping normalization, SRBD port, flow map, deletions);
  - robot-model internals switched to the quaternion layout (layout, accessors, `E⁺`-based tangent Jacobians, slerp interpolation, measurement helpers, `setBaseOrientationEulerZyx` writes ξ, `stateFromTuningLayout` converts);
  - the three flow maps;
  - manifold registration in both interfaces, the MRT joint controllers, dummy nodes and rollouts;
  - layout tag at `ModelSettings.cpp:189`;
  - `Types.h` values 7/6 and the telemetry names;
  - Python logger and inspector; PlotJuggler;
  - YAML comment rewording and `LINT.IfChange(tuning_layout)` blocks;
  - lint rules (Appendix B) enforced;
  - all migrated tests.
- Acceptance:
  - G2 under E1 tolerances; G1 under E2;
  - every §5c test; `testCentroidalQuaternionDynamics`, `testSrbdQuaternion`, `testQuaternionInvariances` (all robots), `testUnitNorm`, `testTurnInPlace720`, `testPoseCommandSubdivision`, `testMrtObservationDoubleCover`, `testCppAdLayoutTag`, `test_tuning_layout`, lint tests;
  - the full `bazel test //...` passes;
  - the production CppAD terms of both MPCs, taped on the new root, against double precision with their first- and (where generated) second-order derivatives: Step 0 spiked the kinds of computation, not these tapes.
- Note: every CppAD library regenerates on first use (memory: one Bazel command at a time; library-compiling tests one at a time).

**Step 9: Documentation.**
- Files: `humanoid_nmpc/docs/quaternion_base_orientation/README.md` (Appendix C) and its index entry; `robot_models/drc_atlas/README.md:15`; `humanoid_learning/acom/README.md`; IFTTT pairs of Appendix A completed.
- Acceptance: `ifttt-lint` and `lint_code.py` clean; README-name rule satisfied.

**Step 10: Closed-loop validation.**
- Record M3 and B3 on all five configurations.
- Acceptance: §4.5 bands against M1 and M2; §4.6 real-time gate; then the user validates in MuJoCo before hardware.

**Step 11 (user-gated, Q1): behavior-changing defect fixes, each a separate commit with a test and a MuJoCo comparison.**
- D1/D1b: intermediate knot = (measured + command)/2, using `mapping.getPinocchioJointVelocity` (mass-correct) and the world yaw rate `(R ω_B)_z`; both calculators.
- D2: `ProceduralMpcMotionManager.cpp:182, 198` use a yaw-rate accessor and the measured value.
- D3: `SwitchedModelReferenceManager.cpp:337, 346` read per-model yaw rate and velocity accessors.

---

## 7. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Pinocchio 3 `JointModelSpherical` inside a composite asserts or compares variables under CppADCG. It cannot be inspected here: Pinocchio is not on the host. | Step 0 before any other work. Safe normalization makes the tape point a valid unit quaternion. Taping tests run in the default `-c opt`; fastbuild-with-asserts is never relied on. **Fallback** (same state and tangent): FreeFlyer root. The mapping converts `v_FF = [R(ξ)ᵀṗ_W, ω_B]`; translational Jacobian columns are rotated by `Rᵀ`; `A_b⁻¹` is generalized to `[[Rᵀ/m, [r]×I⁻¹Rᵀ],[0, I⁻¹Rᵀ]]`; the whole-body Schur `A`-block changes accordingly. |
| R2 | OCS2 regression for flat users | `nullptr` fast paths; `testSqpFlatParity` bitwise; full humanoid suite after Step 4 with fingerprints bitwise; DDP and value-function guards. |
| R3 | A missed flat site crashes on dimensions or silently mixes ambient and tangent sizes | §3.1 lists every site found by the solver map. `testManifoldProjection` covers node, event and terminal. Toy SO(3) problem end to end. Removing `getGenCoordinatesDim` forces every nq/nv choice to compile-fail until decided. |
| R4 | Stale Euler CppAD libraries are loaded (`recompileLibrariesCppAd: false` everywhere) | Layout tag in the folder plus the throwing domain/range check; `testCppAdLibraryDimensionCheck`, `testCppAdLayoutTag`. |
| R5 | A Q of the wrong size shifts indices silently | Loaders size by `getStateTangentDim()`; `stateFromTuningLayout` rejects wrong sizes; `test_tuning_layout`; IFTTT `tuning_layout`. |
| R6 | Hot reload silently lost through a `bad_cast` | Updater migrated to the new classes; updater tests extended; `InputQuadraticCost` input-only removes the `zeroQ` mismatch. |
| R7 | Some function sees the raw ξ, so the retraction-curvature term no longer vanishes, or a non-unit RK4 stage biases Pinocchio | Safe-normalized `getGeneralizedCoordinates`; radial-invariance tests over every registered term and the dynamics (`∇_ξg·ξ = 0`). |
| R8 | Log cut locus at π (δx₀, defects) | Warm-start rejection when the rotation part of δx₀ exceeds π/2; per-node defects are O(dt·\|ω\|), far from π at 50-80 Hz. |
| R9 | Yaw residual discontinuity at ±π yaw error | References re-seeded from the measured heading every command; knots ≤ π/2 (velocity knots ≤ 0.84 rad); `maxAngularAcceleration` limits; `Q_yaw = 0` on every robot. |
| R10 | The tuning meaning drifts slightly (twist vs ZYX heading, cubic tilt, whole-body ω rows, ACoM composition) | Stated E2 tolerances; G2 separates formulation from coordinates; M2 vs M1; README section "what to watch when re-validating". |
| R11 | Feedback semantics on Atlas/SA01 (`K` now acts on `x ⊖ x̄`) | `ManifoldLinearController` tests; check `K_q·C ≈ K_e` to first order in the SQP-step golden; closed-loop bands. |
| R12 | Real-time regression from the pull-backs | In-place, block-structured operations; the §4.6 benchmark gate; tape operation counts. |
| R13 | Pre-existing defects contaminate goldens or get silently re-implemented | Step 2 behavior-preserving fixes before G1; behavior-changing fixes deferred to Step 11 (Q1); the golden README lists what is deferred. |
| R14 | Raw state slices or raw interpolation survive in humanoid code | Lint rules (Appendix B); grep audit before Step 8 closes; `−ξ` and radial tests over every term. |
| R15 | The FSM and GUI parse YAML comments | First token kept; `test_tuning_layout`; IFTTT to `humanoid_finite_state_machine.py`. |
| R16 | Memory pressure when every library regenerates | Closed-loop and library-compiling tests `exclusive` and run one at a time; no out-of-Bazel compiles during builds (AGENTS.md). |
| R17 | Upside-down states (falls) | Residual clamps keep values finite; dynamics stay regular (`I_G ≻ 0`); `SimFallRecovery` already computes tilt from the quaternion. |
| R18 | The ACoM network's Δθ is not a true rotation vector (non-integrable connection) | First-order agreement is tested; `testAcomAngularVelocityConsistency` rewritten in ω terms against `dataset_generator.py:394-431`; tolerance set from the measured `max |Δθ|`. |

---

## Appendix A: IFTTT pairs to add

1. **`state_layout`:** the layout comments and start-index constants in `CentroidalMpcRobotModel.h` and `WBAccelMpcRobotModel.h` ↔ `common/include/humanoid_common_mpc/common/StateLayout.h:cppad_layout_tag` ↔ the README layout tables ↔ `mpc_observation_logger.py`/`mpc_observation_inspector.py` columns.
2. **`tuning_layout`:** the row constants in `CentroidalTuningDeviation.cpp`/`WBTuningDeviation.cpp` ↔ the `initialState`/`Q`/`Q_final` blocks of the five `task.yaml` files ↔ `humanoid_finite_state_machine.py` token filter (`:296-308`) ↔ `remote_control/test/test_tuning_layout.py` ↔ the README tuning table.
3. **`controller_type`:** `ocs2/core/include/ocs2_core/control/ControllerType.h` ↔ `ocs2/ros2_msgs/msg/MpcFlattenedController.idl` ↔ the switches in `MPC_ROS_Interface.cpp` and `MRT_ROS_Interface.cpp`.
4. **`telemetry_layout`:** `Types.h` base names and offsets ↔ `tools/plotjuggler/humanoid_telemetry.xml` ↔ `testPinocchioTelemetryPublisher.cpp`.
5. **`cppad_layout_tag`:** `StateLayout.h` ↔ `ModelSettings.cpp:189` ↔ the README CppAD section.
6. **`euler_boundary`:** the lint allow-list in `tools/hooks/lint_code.py` ↔ README "Conversion points".
7. Existing `base_pose_heuristic_seam` (`SwitchedModelReferenceManager.cpp:437-447` ↔ `BasePoseHeuristic.h`): keep, with the comment updated to say the seam operates on the Euler boundary accessors.

## Appendix B: lint rules (`tools/hooks/lint_code.py`, each with a test)

1. `SphericalZYX` is banned in `humanoid_nmpc/`, `robot_runtime/` and `lib/ocs2/pinocchio/centroidal_model/`.
2. The Euler API is banned in formulation directories, i.e. `humanoid_nmpc/*/src/{dynamics,cost,constraint,end_effector,acom}/**` and the corresponding `include/**` directories, plus the robot-model headers. Banned names: `EulerAnglesZyx`, `EulerZyx`, `EulerZYX`, `ZyxEulerAngles`, `getBasePoseEulerZyx`, `setBasePoseEulerZyx`, includes of `RotationDerivativesTransforms.h` and `orientation/EulerBoundary.h`. Allowed elsewhere (the boundary of §4.2).
3. A single-argument `getDesiredState(` or `LinearInterpolation::interpolate(` on a state trajectory is banned under `humanoid_nmpc/` outside `MpcRobotModelBase` and its implementations' interpolation helpers.
4. Raw quaternion construction with four scalars (`quaternion_t(`/`Eigen::Quaternion<…>(` followed by four arguments) is banned under `humanoid_nmpc/` outside `orientation/` (the `(w,x,y,z)` trap).

## Appendix C: README skeleton (`humanoid_nmpc/docs/quaternion_base_orientation/README.md`)

1. **Why:** the Euler singularity (`det T_W = −cos θ`), the yaw wrap hazard, and the user's requirement.
2. **Block diagram:**

```
 task.yaml / GUI / commands (Euler tuning layout, n_t = ndx)        RobotState (MuJoCo / estimator)
   initialState, Q, Q_final, Q_acom, R, reference.yaml               p_W, xi (xyzw), v_B, w_B, q_j, qd_j
            |                                                                  |
            v                                                                  v
   EulerBoundary: stateFromTuningLayout          MRT observation: q = [p, xi, q_j], v = [R v_B, w_B, qd_j]
   TargetTrajectoriesCalculatorBase (psi scalar)    h = A_g(q) v / m   (no Euler, no sign alignment)
   -> knots q_z(psi_k), subdivided <= pi/2                             | x_init (ambient, unit)
            | TargetTrajectories (xi)                                  v
            v                                 +---------------- OCS2 SQP on M = R^a x S^3 x R^b ----------------+
   SwitchedModelReferenceManager              |  dx0 = x[0] (-) x_init     RK4 ambient -> project -> defect (-)  |
   getDesiredState: slerp, shapeBasePose ---->|  terms: ambient derivatives, pulled back J E, E^T H E           |
   (Euler seam), heading = twist             |  HPIPM (ndx, unchanged)    step x (+) a dx    warm start slerp    |
                                              +----------------------------------------------------------------+
   Costs/constraints (CppAD, read q = [p, xi_hat, q_j]):               | ManifoldLinearController
     Tuning deviation: heading-tilt rows (yaw,pitch,roll), w-rows      v u = u* + K (x (-) x_bar)
     ComAndAcom: xi (x) Exp(dtheta(q_j))                       MRT -> joint torques / PD targets -> robot
     EE quaternionDistance / plane / twist / wrench cone (unchanged)   telemetry: Euler topics for display
```

3. **Derivation:** §2.1, §2.5, §2.7, §2.8 and §2.12 of this document, in LaTeX.
4. **Layout tables:** §2.3.
5. **Tuning mapping:** §2.10.
6. **Conversion points:** §4.2, tied to the lint allow-list.
7. **CppAD libraries:** the layout tag and the domain check.
8. **Validation:** the ladder, tolerances and the closed-loop checklist. What to watch in MuJoCo: tilt RMS on SA01/R1 under strong turning, Atlas ACoM yaw behavior, feedback on Atlas/SA01, and the absence of a ±π transient.
9. **Test map:** §5.
