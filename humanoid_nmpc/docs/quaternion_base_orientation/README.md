# Quaternion base orientation

Both humanoid MPCs, the centroidal and the whole-body one, store the attitude of the floating base as a unit quaternion
and solve on the state manifold it lives on. The task files, the tuning GUI and every human-facing input and output keep
Euler angles, and one mapping layer converts them. This document is the formulation's reference: why quaternions, the
data flow, the derivation, the layouts, the tuning map, the conversion points, the CppAD libraries, the bus and the
runtime, the display, the tolerances, the tests and what they prove, and the risks that remain.

There is no Euler layout, no toggle and no task-file switch back: Euler angles remain only at the conversion points of
section 4.2 (the tuning layer, the references' human-facing layer, the display) and as the Euler side of the
equivalence suites, Pinocchio's own `SphericalZYX` root, which `spherical-zyx` keeps out of the rest of the C++
(Appendix B). Code comments cite the section numbers ("design section 4.2" in every Euler boundary marker, for
example), so they are kept stable.

| Section | What |
|---|---|
| 1 | why, the block diagram, the requirements, the decisions and their reasons |
| 2 | the formulation: algebra, root joint, layouts (2.3), manifold, dynamics, taping, SQP, costs, references, the tuning map (2.10), the runtime state, the identities the equivalence tests use |
| 3 | where the code lives, the CppAD libraries, the bus and the realtime path, the display, live tuning |
| 4 | what the formulation does not have, the conversion points (4.2), how it is verified (4.3), the tolerances (4.4), the closed-loop bands (4.5), the real-time gate (4.6), the reference behavior (4.7), what to watch in MuJoCo (4.8) |
| 5 | the Euler side of the tests, the equivalence suites and the test map by Bazel target |
| 6 | the risks that remain and the decisions still open |
| A, B | the IFTTT pairs and the lint rules |

**Path abbreviations:**
- `common/` = `humanoid_nmpc/humanoid_common_mpc`, `cmpc/` = `humanoid_nmpc/humanoid_centroidal_mpc`,
  `cmpc_test/` = `humanoid_nmpc/humanoid_centroidal_mpc_test`, `wbmpc/` = `humanoid_nmpc/humanoid_wb_mpc`;
- `app/` = `humanoid_nmpc/humanoid_common_mpc_app`, `capp/` = `humanoid_nmpc/humanoid_centroidal_mpc_app`,
  `wapp/` = `humanoid_nmpc/humanoid_wb_mpc_app`;
- `ipc/` = `humanoid_nmpc/humanoid_mpc_ipc`, `msgs/` = `humanoid_nmpc/humanoid_mpc_msgs`,
  `cfg/` = `humanoid_nmpc/humanoid_mpc_config`, `val/` = `humanoid_nmpc/humanoid_mpc_validation`,
  `rr/` = `humanoid_nmpc/humanoid_rerun_viewer`;
- `ocs2/` = `lib/ocs2`, `cm/` = `lib/ocs2/pinocchio/centroidal_model`;
- `common/<dir>/` names a module of `common/` by its directory: its header in `include/humanoid_common_mpc/<dir>/` and its
  source in `src/<dir>/` (`common/orientation/`, `common/cost/`, `common/common/`, ...).

Code is cited by symbol and Bazel target, not by line.

---

## 1. Why, and what was decided

### 1.1 Why quaternions

- **The Euler root is singular.** With the `Composite(Translation, SphericalZYX)` root the base velocity was
  `[ṗ_W, Θ̇]`, the rates of the ZYX angles `Θ = (ψ, θ, φ)`, and the world angular velocity `ω_W = T_W(Θ)·Θ̇` with
  `det T_W = −cos θ`. At a pitch of ±90° the map loses rank: the centroidal momentum matrix's rotational block
  `I_G^W·T_W(Θ)` and the whole-body rotational Schur complement `T_Bᵀ·I·T_B` become singular, and every Jacobian through
  them grows as `1/cos θ` before it.
- **The Euler yaw wraps.** The state stored the yaw as an angle. The measured yaw comes `atan2`-wrapped into
  (−π, π], the warm start and the references carry it unwrapped, so where the heading crosses ±π the SQP's initial
  gap `δx₀` jumps by 2π. On the whole-body Unitree G1 turning in place through 720°, the Euler formulation's gap
  reached 2π (6.285 rad) at each of the two crossings and decayed below π/2 within 0.17 s, and the tilt peaked at
  0.10 to 0.15 rad shortly after, depending on the gait phase at the crossing. A quaternion state stores no angle: the
  gap is a shortest-path logarithm, at most π, and the warm start never crosses a cut.
- **The requirement:** a quaternion representation of the base orientation throughout the dynamics, the costs and the
  constraints, with no toggle; every tuning hyperparameter stays in Euler angles, mapped in one layer.

### 1.2 Block diagram

```
 task.textproto / reference.textproto / contact_planning.textproto      tuning GUI (labels = field names), editor
 initial_state, state_weights, final_state_weights, input_weights,        | live edits of the hot fields
 acom_weights {yaw pitch roll}, base_angular_velocity {yaw pitch roll}    v
        |                                       MpcParameterUpdaterModule, pre-solve on the solver thread:
        v                                         HotFieldApplier lists kCentroidalAppliers (12), kWholeBodyAppliers (7)
 config/weights/StateInputWeightsFromConfig        QuadraticCostWeightsApplier, ComAndAcomWeightsApplier, ...
   tuningLayoutOf(StateInputLayout): n_t = ndx      -> OcpTermUpdates::updateTerm / updateTermIfPresent
   trackingStateWeightsFromConfig, terminalStateWeightsFromConfig,           |
   stateValuesFromConfig, inputWeightsFromConfig, acomWeightsFromConfig      | the same converters
        | tuning-layout Q, Q_final, R, Q_acom, x0                             | and the same setters
        +-------------------------------------+-------------------------------+
                                              v
 ============ the one mapping layer: common/orientation/ and the robot models' boundary ============
   robot model: getTuningLayout() (placeTuningLayout), stateFromTuning / tuningFromState,
                get/setBasePoseEulerZyx, get/setBaseOrientationEulerZyx, setBaseAngularVelocityFromWorldYawRate
   EulerBoundary: quaternionFromEulerZyx, eulerZyxFromQuaternion, stateFromTuningLayout, tuningLayoutFromState
   BaseOrientation: twistHeading, headingQuaternion, headingTiltError, safelyNormalizedQuaternion
   TuningLayoutDeviation: rows (yaw, pitch, roll) = e_HT, whole-body rows (z, y, x) = e_w, block Jacobians
 ==================================================================================================
        | x0 = stateFromTuning(initial_state)       | Tuning*QuadraticState*Cost.setStateWeights(Q) (TuningGaussNewton)
        |                                           | ComAndAcomTrackingCost.setWeights(Q_com, Q_acom)
        v                                           v
 velocity / pose commands -> TargetTrajectoriesCalculatorBase (yaw a scalar): knots on the arc, <= pi/2 apart
   (velocityCommandKnots, knotSubdivisionCount, subdivideYawKnots), written by setBasePoseEulerZyx
   -> SwitchedModelReferenceManager: model.getDesiredState (slerp), shapeBasePose (Euler seam), twist heading
        |
        v
 OCS2 SQP on M = E(a) x S^3 x E(b)  (stateManifoldOf(model) -> problem.stateManifoldPtr, rollout, MRT)
   dx0 = x[0] (-) x_init, RK4 on the ambient field -> project -> defect on M, term derivatives pulled back (J E, E'HE),
   HPIPM on ndx, x (+) alpha dx, warm start interpolated on M and dropped beyond pi/2
        | ManifoldLinearController: u = u* + K (x (-) xbar)
        v
 MpcServer -> MpcPolicy, CONTROLLER_TYPE_MANIFOLD_LINEAR (3), node [u*, xbar, vec K] -> bus -> RemoteMpcLink
   (Config::stateManifold; decoded on the communication thread, MRT_BASE::moveToBuffer)
        |
        v
 robot realtime thread: RealtimePolicyEvaluator (differenceInto / interpolateInto, buffers preallocated)
   -> MRT joint controller -> joint torques and PD targets -> MuJoCo or the robot
   measurement: fillGeneralizedCoordinatesFromMeasurement / fillGeneralizedVelocitiesFromMeasurement
                q = [p_W, xi, q_j], v = [R(xi) v_B, w_B, qd_j]
        | robot/state
        v
 MPC node, visualization thread: TelemetryBuilder -> Rerun tab "Base Pose & Euler":
   Euler angles, Euler rates T_B^-1 w_B (NaN at |cos(pitch)| < 1e-6), body angular velocity w_B
```

The live path and the start-up path share every step from the converters on: the same functions produce the
tuning-layout matrices and hand them to the same setters, so the cost's `TuningLayoutDeviation` is the single map from
the Euler-written rows to the quaternion residual at start-up and live alike (section 3.5).

### 1.3 Requirements

| Requirement | What it means |
|---|---|
| Representation | The dynamics, the costs and the constraints read a unit quaternion; no toggle, no task-file switch back, no Euler fallback. Tuning stays Euler and is mapped in one layer (`common/orientation/`). |
| Display | Rerun shows the Euler angles and both the body angular velocity and the Euler rates; the rates are display values computed off the realtime thread, NaN within 1e-6 of the pitch singularity (section 3.4). |
| Live tuning | Hot reload on both formulations routes every weight through the same one mapping layer as start-up, with "live update == fresh start" tests on the quaternion layout (section 3.5). |

### 1.4 Decisions and reasons

| # | Decision | Reason / rejected alternative |
|---|---|---|
| D1 | Root joint `JointModelComposite{JointModelTranslation, JointModelSpherical}` for both MPCs (`getBaseJointcomposite()`, `common/src/pinocchio_model/createPinocchioModel.cpp`), and for OCS2's centroidal model factory (`cm/src/FactoryFunctions.cpp`). | World-frame `ṗ_W` keeps the whole-body linear-velocity rows, every `v.head<3>()` reader and the `A_b(0,0) = m` structure. FreeFlyer (SE(3)-local `[v_B, ω_B]`) breaks all three. |
| D2 | Native tangent-space SQP in the vendored OCS2: a `StateManifold` on the `OptimalControlProblem`. A null manifold keeps every existing code path bit for bit (`test_sqp_flat_parity`). | No radial mode, no sign alignment, QP sizes unchanged: HPIPM takes its sizes from `dfdx.cols()`. |
| D3 | Discretization: OCS2's RK4 on the ambient vector field, then normalization (`F = Π∘Φ_RK4`), the defect expressed on the manifold (section 2.7). | The field is tangent to S³ (`Gᵀ(ξ)ξ = 0`), so RK4 keeps its fourth order; a Lie-group RK would rewrite the sensitivity integrator. |
| D4 | Every cost and constraint returns derivatives with respect to the **ambient** state; the transcription pulls them back once (`J ← J·E`, `H ← EᵀHE`). Analytic terms with natural tangent derivatives lift them (`J_a = J_t·E⁺`). | The roughly 30 CppAD terms need no derivative plumbing. |
| D5 | Homogeneity rule: every function of the state sees the quaternion only through `ξ̂ = ξ/‖ξ‖`, computed safely (section 2.6). The one exception is the kinematic row `ξ̇ = ½G(ξ)ω_B`, on the raw `ξ`. | The retraction-curvature term vanishes (`∇_ξℓ·ξ = 0`); RK4 stages and the CppAD tape point (`x = ones`, `ξ̂ = ½(1,1,1,1)`) are valid configurations; a zero state reads as the identity rotation, not NaN. |
| D6 | Tuning layout = tangent size: `n_t = ndx`, the ZYX Euler state index for index. The task file's `state_weights`, `final_state_weights`, `input_weights` and `initial_state` blocks are unchanged: `tuningLayoutOf()` places them on vectors that keep their size and indices, and the costs evaluate a tuning residual with its own Jacobian. | A quaternion-sized Q would shift every index past the orientation; the named blocks cannot, and the conversions refuse a block of the wrong layout. The GUI, rendered from the schemas, needed no change. |
| D7 | The Euler-weighted orientation rows are the heading-tilt split (section 2.8.1): the wrapped twist yaw, and the tilt rotation vector's components (y, x) in the heading frame. | The tilt is exactly independent of the yaw error. `Log(R_refᵀR)` leaks tilt between axes under a yaw error (at 0.35 rad it reports (0.108, 0.032) for a true (0.1, 0.05)). Every robot ships a yaw weight of 0, and SA01's and R1's pitch and roll weights (5 and 85) would change meaning. |
| D8 | Whole-body angular-velocity rows: `e_ω = P₃(ω_B − R(ξ̂)ᵀR(ξ̂_r)ω_{B,r})`, a body-frame error with the reference transported, ordered (z, y, x). | Exactly zero for correct tracking of a world yaw-rate command at any tilt, as the Euler-rate difference is. For a reference whose Euler rate is a world yaw rate (every shipped one) `e_ω = P₃T_B(Θ)(Θ̇ − Θ̇_r)` exactly (section 4.4). |
| D9 | One heading for every read: the twist `ψ_h(ξ) = 2·atan2(ξ_z, ξ_w)` read in the hemisphere `ξ_w ≥ 0` (`twistHeading()`, in (−π, π], `ψ_h(−ξ) = ψ_h(ξ)` exactly). Edits of level references at the Euler boundary (heuristic offsets, the planned-heading override) decompose in ZYX, add or replace, and recompose. | The twist is singular only upside down; the ZYX yaw at a pitch of ±90°. The two differ by `o(θ, φ) = −2·atan(tan(θ/2)tan(φ/2)) ≈ −θφ/2`. For level references (every shipped one) they agree exactly. |
| D10 | Targets are stored as quaternions in the solver layout and interpolated by shortest-path slerp through the robot model (`interpolateStateTrajectory()`). Velocity and pose commands are cut so that consecutive knots are at most π/2 apart in yaw. | Slerp between level knots is linear in yaw, so it reproduces the Euler layout's linear interpolation; the subdivision keeps "+270° turns +270°". |
| D11 | `ManifoldLinearController`: `u = u*_k + K_k·(x ⊖ x̄_k)`, blended between nodes, `ControllerType::MANIFOLD_LINEAR` in OCS2 (enum value 3, not a wire format) and `CONTROLLER_TYPE_MANIFOLD_LINEAR = 3` on the bus (`msgs/controller_type.proto`; the `controller_types` IFTTT pair, `ipc/` maps the two). | `LinearController` computes `uff + K·x`, undefined with a `nu×ndx` K and `x ∈ ℝ^nx`. Atlas and SA01 run `use_feedback_policy: true`. |
| D12 | No hemisphere alignment anywhere. | Every consumer is double-cover invariant: the shortest-path `Log` in `⊖`, shortest slerp, the heading-tilt residual, the rotation-based residuals, the equivariant flow; `−ξ` tests on every term and on both MRTs (section 5(c)). Two cut loci of measure zero keep the inversion instead: `Log` at an exact half turn (section 2.1) and the tilt exactly upside down (section 2.8.1). |
| D13 | Cost classes `TuningQuadraticStateInputCost` and `TuningQuadraticStateCost` (`common/cost/`) derive from `ocs2::StateInputCost` / `StateCost` with their own weight setters; `StateInputQuadraticCost`, `StateQuadraticCost` and `BasePoseShapedQuadraticStateCost` are built on them; `InputQuadraticCost` is input-only. Every hot-reload cast moved with them. | OCS2's `QuadraticStateInputCost` / `QuadraticStateCost` have `final` approximations. `Collection::get` is a `dynamic_cast`: a stale cast would throw `bad_cast`, which `updateTermIfPresent()` (`OcpTermUpdates.h`) turns into a WARNING, silently disabling a hot field. The live-equals-fresh tests compare the weights the running costs hold. |
| D14 | CppAD libraries live in a layout-tagged folder `cppad_code_gen/cppad_<mpc><robot>/<kStateLayoutTag>`, and `CppAdInterface` refuses a loaded library whose domain or range differs, naming the folder (section 3.2). | Every task file ships `model_settings.recompile_libraries_cpp_ad: false`, and a library was reused whenever its `.so` existed. |
| D17 | The single-rigid-body branch of the centroidal model is ported, not refused. | It is a model option (`model_settings.centroidal_model: "single_rigid_body_dynamics"`), not an Euler path; every robot ships `"full_centroidal_dynamics"`, so the port is tested against the full model at the nominal posture (`testSrbdQuaternion`). |
| D18 | The whole-body `initial_state.base_angular_velocity { yaw pitch roll }` rows are the body angular velocity in (z, y, x) order, matching the weight rows. | The rows of the values and of the weights mean the same; at a level attitude they equal the ZYX rates (`state_values.proto`). The shipped values are 0. |

---

## 2. Formulation

### 2.1 Notation and quaternion algebra

- **Storage.** `ξ = (ξx, ξy, ξz, ξw)` in Eigen `coeffs()` and Pinocchio order, Hamilton product, `R(ξ)` from the base to
  the world, as `RobotState::getRootRotationLocalToWorldFrame()` (`robot_runtime/robot_model`).
- **Construction trap.** Quaternions are built from coefficient vectors only: Eigen's four-scalar constructor takes
  `(w, x, y, z)`; `quaternion-scalar-constructor` bans the four-scalar form outside `orientation/` (Appendix B).
- **Rate matrix.** `ξ ⊗ (ω, 0) = G(ξ)·ω`, with

```math
G(\xi)=\begin{bmatrix} \xi_w&-\xi_z&\xi_y\\ \xi_z&\xi_w&-\xi_x\\ -\xi_y&\xi_x&\xi_w\\ -\xi_x&-\xi_y&-\xi_z\end{bmatrix},\qquad G^\top G=\|\xi\|^2 I_3,\qquad G^\top\xi=0 .
```

- **Tangent maps.** `E(ξ) = ½G(ξ)` (4×3) maps a body right perturbation to `δξ`; its left inverse is

```math
E(\xi)=\tfrac12 G(\xi),\qquad E^+(\xi)=\frac{2\,G^\top(\hat\xi)}{\|\xi\|},\qquad E^+E=I_3\quad(\|\xi\|>\varepsilon),
```

  where at or below ε the guarded divisor is 1, so `E⁺E = ‖ξ‖²I₃` and the zero quaternion lifts to zero; on the unit
  sphere `E⁺ = 2Gᵀ(ξ)` (`configurationTangentLift()`, `getBaseRotationTangentJacobian()`).
- **Exp and Log** (`ocs2/core/include/ocs2_core/manifold/UnitQuaternionMath.h`):

```math
\mathrm{Exp}(\varphi)=\Big(\sin\tfrac{|\varphi|}{2}\,\tfrac{\varphi}{|\varphi|},\ \cos\tfrac{|\varphi|}{2}\Big),\qquad
\mathrm{Log}(\xi)=2\,\mathrm{atan2}\big(\|v\|,\,w\big)\,\frac{v}{\|v\|}\quad\text{after }\xi\leftarrow-\xi\text{ when }w<0,
```

  with `ξ = (v, w)`, a series for |φ| < 1e-6 and for ‖v‖ ≤ 1e-6·‖ξ‖, and ‖Log‖ ≤ π (shortest path).
  - Every guard of `Log` is relative to ‖ξ‖², so `Log(cξ) = Log(ξ)` for every c > 0, to round-off for
    1e-150 ≤ ‖ξ‖ ≤ 1e150, quaternions shorter than ε included. (Absolute guards took the series for every short
    quaternion whatever its angle: 1 rad about (1, 2, 3)/√14 came out as 0.984 rad at ‖ξ‖ = 2e-6.)
  - Double cover: `Log(conj ξ) = −Log(ξ)` bit for bit everywhere, and `Log(−ξ) = Log(ξ)` bit for bit for every w ≠ 0.
    At an exact half turn (w = ±0, which products of exact quaternions do produce) −ξ and conj ξ have the same
    coefficients up to the sign of the zero, so both cannot hold: `Log` returns `π·v/‖v‖` for either zero, keeping the
    inversion, so `Log(−ξ) = −Log(ξ)` there, the other logarithm of the same rotation. `difference(x0, ±x1)` and slerp
    toward ±x1 then take the two opposite shortest geodesics, and `Jr⁻¹(Log ξ)` is the derivative of the branch returned
    (`test_state_manifold`, `AtAnExactHalfTurnJrInverseIsTheDerivativeOfTheReturnedLogarithm`).
- **Right Jacobians**, with `α = |φ|`:

```math
J_r(\varphi)=I-\frac{1-\cos\alpha}{\alpha^2}[\varphi]_\times+\frac{\alpha-\sin\alpha}{\alpha^3}[\varphi]_\times^2,\qquad
J_r^{-1}(\varphi)=I+\tfrac12[\varphi]_\times+\Big(\frac1{\alpha^2}-\frac{1+\cos\alpha}{2\alpha\sin\alpha}\Big)[\varphi]_\times^2 .
```

  The `[φ]×²` coefficients cancel in double precision as α → 0, so they use their Taylor series below α = 0.1
  (`kJacobianSeriesThreshold`); `(1 − cos α)/α²` and `Exp` switch at 1e-6, only to avoid 0/0.
- **Permutation.** `P₃ = antidiag(1, 1, 1)` maps (x, y, z) to (z, y, x).
- **Euler.** `Θ = (ψ, θ, φ)` is ZYX with the yaw first, `R = R_z(ψ)R_y(θ)R_x(φ)`. `T_B(Θ)` maps Euler rates to the body
  angular velocity, `ω_B = T_B(Θ)Θ̇`, with `T_B(0) = P₃` (`eulerZyxRateToLocalAngularVelocityMatrix()`,
  `EulerBoundary.h`, built on `ocs2/robotic_tools`' `RotationDerivativesTransforms.h`); it is used only at the boundary
  and in tests.

### 2.2 Root joint and Pinocchio vectors

With `Composite(Translation, Spherical)`:

| | Configuration `q` | Velocity `v` | Acceleration `a` |
|---|---|---|---|
| Base | `[p_W(3), ξ(4)]` | `[ṗ_W(3), ω_B(3)]` | `[p̈_W, ω̇_B]` |
| Joints | `q_j` | `q̇_j` | `q̈_j` |
| Sizes | `nq = 7 + nj` | `nv = 6 + nj` | `nv` |

**Key identity.** With the right-trivialized tangent (translation in the world frame, rotation as a body right
perturbation), the configuration part of the state tangent equals Pinocchio's velocity coordinates. So every Pinocchio
`v`-space Jacobian (frame, CoM, centroidal map) is the tangent Jacobian with respect to the configuration, and
`pinocchio::integrate(q, v) = q ⊕ v`. `testConfigurationTangent` checks both on every shipped robot; the root-joint
equivalence suite (`common/test/equivalence`, section 5(b)) checks the kinematics, the centroidal quantities and the
dynamics of the two roots against each other through the chart of section 2.12.

Independent of the root joint: the joint-naming check (`checkPinocchioJointNaming()`), the joint-limit reader
(`readPinocchioJointLimits()`, which reads the tails), the contact planner's model parameters (`pinocchio::neutral` is
the identity quaternion) and the visualization's root link.

### 2.3 State, tangent, tuning and input layouts

**Centroidal** (`CentroidalMpcRobotModel`, `cmpc/include/humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h`):

| Block | Ambient `x` (storage) | Tangent `δx` (QP, gains) | Tuning rows (`tuningLayoutOf()` segment) |
|---|---|---|---|
| `h̄ = [ċ, L/m]` (world, normalized momentum) | 0..5 | 0..5 | 0..5 (`normalized_linear_momentum`, `normalized_angular_momentum`) |
| `p_W` | 6..8 (`getBaseStartindex()` = 6) | 6..8 | 6..8 (`base_position`) |
| base orientation | 9..12 = ξ (x, y, z, w) (`getBaseOrientationStartIndex()` = 9) | 9..11 = δφ (body right perturbation) | 9 = yaw, 10 = pitch, 11 = roll (`base_orientation`, `kEulerZyxOrientation`: the heading-tilt residual) |
| `q_j` | 13..12+nj (`getJointStartindex()` = 13) | 12..11+nj | 12..11+nj (`joint_positions`) |
| size | `getStateDim()`: `nx = 13 + nj` | `getStateTangentDim()`: `ndx = 12 + nj` | `n_t = ndx` |

- **Input:** `u = [W_l(6), W_r(6), q̇_j]` in the world frame, or the basis scalings λ followed by `q̇_j`
  (`contact_input_parameterization`). Unchanged; `getJointVelocitiesStartindex()` is an input index.
- **Pinocchio vectors:** `q = [p_W, ξ̂, q_j]` (`getGeneralizedCoordinates()`), and
  `v = [v_b, q̇_j]` with `v_b = [ṗ_W, ω_B] = A_b⁻¹(q)·(m·h̄ − A_j(q)·q̇_j)` (`CentroidalModelPinocchioMapping`).
  `CentroidalModelInfo` carries `configurationDim` (nq), `velocityDim` (nv), `stateDim = 6 + nq` and
  `stateTangentDim = 6 + nv`; the joints start at 13 (`cm/.../AccessHelperFunctions.h`: `getBasePosition`,
  `getBaseQuaternion`, `getJointAngles`).

**Whole body** (`WBAccelMpcRobotModel`, `wbmpc/include/humanoid_wb_mpc/common/WBAccelMpcRobotModel.h`):

| Block | Ambient | Tangent | Tuning rows |
|---|---|---|---|
| `p_W` | 0..2 | 0..2 | 0..2 (`base_position`) |
| orientation | 3..6 = ξ | 3..5 = δφ | 3 yaw, 4 pitch, 5 roll (`base_orientation`) |
| `q_j` | 7..6+nj (`getJointStartindex()` = 7) | 6..5+nj | 6..5+nj (`joint_positions`) |
| `ṗ_W` (world) | 7+nj..9+nj | 6+nj..8+nj | 6+nj..8+nj (`base_linear_velocity`) |
| `ω_B` (body, x, y, z) | 10+nj..12+nj | 9+nj..11+nj | 9+nj = ω_z, 10+nj = ω_y, 11+nj = ω_x (`base_angular_velocity`, `kZyxOrderedAngularVelocity`, D18) |
| `q̇_j` | 13+nj.. (`getJointVelocitiesStartindex()`) | 12+nj.. | 12+nj.. (`joint_velocities`) |
| size | `nx = 13 + 2nj` | `ndx = 12 + 2nj` | `n_t = ndx` |

- `q = [p_W, ξ̂, q_j]` and `v = state.tail(nv)`, both Pinocchio-native; the input `[W_l, W_r, q̈_j]` is unchanged.
- `getBaseComVelocity()` is the tuning-order base velocity `[ṗ_W; P₃ω_B]`; `getBaseAngularVelocityLocal()` reads
  `ω_B` as stored.

**The tuning layout** (`TuningLayout`, `common/orientation/EulerBoundary.h`) is built by `tuningLayoutOf()`
(`common/config/weights/StateInputLayout.h`) from the schema's field names and placed on the state by
`placeTuningLayout(layout)`: each `TuningSegment` has its `stateStart` and `stateDim`,
the segments consecutive in the state in tuning order, every one as many state rows as tuning rows but the
orientation, which has 4. The robot models hold the placed layout (`getTuningLayout()`), and `tuningDimension()` of a
`StateInputLayout` is `getStateTangentDim()`. `stateManifoldOf()` (`common/common/StateManifoldFactory.h`) builds the
product manifold from the same placement.

**Accessors** (`MpcRobotModelBase`, the layout-agnostic API every consumer uses; the hot ones unchecked, with sizes as
documented preconditions):

| Group | Accessors |
|---|---|
| Sizes | `getStateDim()` (nx), `getStateTangentDim()` (ndx = n_t), `getConfigurationDim()` (nq), `getVelocityDim()` (nv), `getBaseOrientationStartIndex()`, `getJointStartindex()`, `getJointVelocitiesStartindex()` |
| Base | `getBasePosition` / `setBasePosition`, `getBaseOrientation` (safely normalized coefficients) / `setBaseOrientation`, `getBaseHeading` (the twist, in (−π, π]) |
| Euler boundary (section 4.2) | `getBasePoseEulerZyx` / `setBasePoseEulerZyx` (`[x, y, z, yaw, pitch, roll]`), `getBaseOrientationEulerZyx` / `setBaseOrientationEulerZyx`, `setBaseAngularVelocityFromWorldYawRate`; the getters wrap the yaw into (−π, π] and clamp the pitch, the setters take any yaw |
| Whole-body velocity | `getBaseAngularVelocityLocal` / `setBaseAngularVelocityLocal`, `getBaseAngularVelocityLocalJacobian` |
| Motion and commands | `getBaseTwistInWorldFrame(state, jointVelocities)` (exact; centroidal `A_b⁻¹(m h̄ − A_j q̇_j)`), `getTargetPlanarCommand(targetState, compositeYawInertia)` |
| Tuning | `getTuningLayout()`, `stateFromTuning` / `tuningFromState` (`absl::StatusOr<vector_t>`, InvalidArgument on a wrong size, naming the layout) |
| Interpolation | `getDesiredState(targets, t)`, `interpolateStateTrajectory(t, times, states)` (the orientation block slerped the shorter way, the manifold's `interpolate()` at LinearInterpolation's weight `1 - alpha` bit for bit; every other row `LinearInterpolation`'s, bit for bit with `TargetTrajectories::getDesiredState`, which can differ from the manifold's `(1 - a) x0 + a x1` in the last bit), `getNeutralState()` |
| Tangent maps | `getBaseRotationTangentJacobian` (3×4 `E⁺(ξ)`), `getPinocchioTangentJacobian` (nv×nx) |
| Realtime measurement | `fillGeneralizedCoordinatesFromMeasurement`, `fillGeneralizedVelocitiesFromMeasurement` (into preallocated buffers; no allocation, check, log or exception) |

**Per-robot sizes:**

| Robot / MPC | nj | nx | ndx = n_t (the Euler oracle's state) | nq / nv | horizon / dt | MPC rate | feedback policy |
|---|---|---|---|---|---|---|---|
| DRC Atlas, centroidal | 24 | 37 | 36 | 31 / 30 | 1.0 s / 0.02 s | 50 Hz | on |
| EngineAI SA01, centroidal | 12 | 25 | 24 | 19 / 18 | 1.0 s / 0.02 s | 80 Hz | on |
| Unitree G1, centroidal | 23 | 36 | 35 | 30 / 29 | 1.2 s / 0.02 s | 80 Hz | off |
| Unitree R1, centroidal | 22 | 35 | 34 | 29 / 28 | 1.2 s / 0.02 s | 80 Hz | off |
| Unitree G1, whole body | 23 | 59 | 58 | 30 / 29 | 1.1 s / 0.035 s | 60 Hz | off |

### 2.4 The state manifold

The state lives on a product manifold (`ProductStateManifold`, `ocs2/core/include/ocs2_core/manifold/`) of Euclidean
segments and one unit-quaternion segment (ambient 4, tangent 3): centroidal `[E(9), Q, E(nj)]`, whole body
`[E(3), Q, E(2nj + 6)]`. `stateManifoldOf(model)` builds it from the model's placed tuning layout and refuses any other
placement; the interfaces set it on the problem (`stateManifoldPtr`), the rollout and the MRT, and expose it as
`getStateManifold()`. It is immutable and shared across the solver's worker threads as
`std::shared_ptr<const StateManifold>`, the type OCS2's APIs take.

| Operation | Quaternion segment (Euclidean segments use the obvious counterparts) |
|---|---|
| `retract(x, δ, α)` | `ξ ⊗ Exp(α·δφ)`, renormalized (removes round-off) |
| `difference(x0, x1)` | `δ ∈ T_{x0}` with `x0 ⊕ δ = x1`: `Log(ξ0⁻¹ ⊗ ξ1)`, shortest path, Pinocchio's convention; homogeneous of degree 0 in both, so an ambient flow needs no normalization; invariant under ξ1 → −ξ1 but at an exact half turn |
| `interpolate(x0, x1, α)` | slerp along the shorter way |
| `project(x)` | `ξ ← ξ/‖ξ‖` |
| `E(x)`, `E⁺(x)` | `blockdiag(I, ½G(ξ), I)`, `blockdiag(I, E⁺(ξ), I)` |
| `getMaximumRotationAngle(δ)` | the largest ‖δφ‖ over the quaternion segments (the warm-start guard, the closed-loop gap metric) |
| `differenceInto`, `interpolateInto` | the allocation-free forms into presized outputs, fixed-size per quaternion segment; the by-value forms are defined through them, bit for bit |

Derivative plumbing is applied in place on the four quaternion columns and rows, never as a dense `E`: O(rows·12) per
term and node (`pullBackStateColumns`, `pullBackGradient`, `pullBackHessian`, `pushForwardDynamics`, `pushForwardJump`,
`liftStateColumns`).

### 2.5 Continuous dynamics

**Centroidal** (`cm/src/PinocchioCentroidalDynamicsAD.cpp`, `cmpc/src/dynamics/CentroidalDynamicsBasisInputsAD.cpp`).
The momentum rows are unchanged; the orientation enters only through Pinocchio's forward kinematics:

```math
\dot{\bar h}=\begin{bmatrix}g+\tfrac1m\sum_i f_i\\ \tfrac1m\sum_i\big((r_i-c)\times f_i+\tau_i\big)\end{bmatrix},\qquad
\begin{bmatrix}\dot p_W\\ \omega_B\end{bmatrix}=A_b^{-1}(q)\big(m\bar h-A_j(q)\dot q_j\big)
```

```math
A_b=\begin{bmatrix} mI & -m[c-p]_\times R(\xi)\\ 0 & I_G^W R(\xi)\end{bmatrix}
\;\Rightarrow\;
A_b^{-1}=\begin{bmatrix} I/m & [c-p]_\times\, R\,(I_G^W R)^{-1}\\ 0 & (I_G^W R)^{-1}\end{bmatrix}
```

- `A_b(0,0) = m` and `A_b.block<3,3>(3,0) = 0` hold, so the closed-form, CppAD-safe inverse of
  `cm/.../implementation/ModelHelperFunctionsImpl.h` is valid as written. `(I_G^W R)⁻¹ = Rᵀ I_G⁻¹` uses 3×3 cofactor
  inverses only and is never singular, because `I_G ≻ 0`. On an Euler root the block is `I_G^W T_W(Θ)`, with
  `det T_W = −cos θ`.
- It is not replaced by a 6×6 LU: CppADCG cannot record the comparisons of a pivoting solve
  (`DynamicsHelperFunctions.cpp` documents why).

The ambient flow map takes the quaternion row on the **raw** `ξ` and everything else on `q` built from `ξ̂`:

```math
\dot x=f_a(x,u)=\big[\dot{\bar h};\ \dot p_W;\ \tfrac12 G(\xi)\,\omega_B;\ \dot q_j\big],\qquad
f_t=E^+f_a=\big[\dot{\bar h};\ \dot p_W;\ \omega_B;\ \dot q_j\big]
```

`computeConfigurationDerivative()` (`cm/.../ModelHelperFunctionsImpl.h`) writes `[ṗ_W, ½G(ξ)ω_B, q̇_j]` from
Pinocchio's `v`, for both centroidal flow maps.

**Single rigid body** (D17): with `r̄` the CoM-to-base position and `Ī` the centroidal inertia at the nominal posture,
both in the base frame, `com = p − R r̄`, `A_b,TR = m·[R r̄]×·R`, `A_b,BR = R·Ī`, and `A_g` is `(6, nv)`.

**Whole body** (`wbmpc/src/dynamics/DynamicsHelperFunctions.cpp`):

```math
\dot x=\big[\dot p_W;\ \tfrac12 G(\xi)\,\omega_B;\ \dot q_j;\ a_b(q,v,u);\ \ddot q_j\big],\qquad a_b=[\ddot p_W;\ \dot\omega_B]
```

- `a_b` comes from the CppAD-safe Schur solve of `common/src/pinocchio_model/DynamicsHelperFunctions.cpp`. The block
  `M_bb[0:3, 0:3] = mI` still holds; the rotational Schur complement is the body-frame CoM inertia, positive definite at
  every attitude (on an Euler root it is `SᵀIS` in Euler rates, singular at |θ| = 90°).
- The inverse dynamics `computeJointTorques()` and `computeBaseHeldJointTorques()` take `q` (nq) and `v` (nv). **"Base
  held"** is zero base acceleration `ω̇_B = 0`; on an Euler root it is `Θ̈ = 0`, a body angular acceleration
  `(d/dt T_B)·Θ̇`. The two roots' base-held joint torques therefore differ by `M_jw(q)·(d/dt T_B)·Θ̇` wherever the base
  turns (`testMrtTorqueChartEquivalence`, section 4.4).

### 2.6 Taping rules (CppAD / CppADCG, compiled with `-ffast-math` by `CppAdInterface`)

1. **Safe normalization** (`quaternionSafeNormalize`, `safelyNormalizedQuaternion()`):

```math
n_{safe}=\sqrt{\mathrm{CondExpGt}\big(\xi^\top\xi,\ \varepsilon^2,\ \xi^\top\xi,\ 1\big)},\qquad \hat\xi=\xi/n_{safe},\qquad \varepsilon=10^{-6}.
```

   The guard is on the **squared** norm, and the square root is taken of the selected branch, so neither branch has an
   unbounded derivative. Guarding the norm instead, `n = sqrt(ξᵀξ)`, `n_safe = CondExpGt(n, ε, n, 1)`, has the same values, but at
   the zero quaternion the reverse sweep of `sqrt` multiplies the unselected branch's zero partial by `1/(2n) = ∞`,
   which CppADCG generates as a plain product: `0·∞ = NaN` in every reverse-mode (cost) Jacobian. A forward-mode probe of
   it is finite, so only a reverse-mode test proves the guard (`testQuaternionRootJointCppAd`
   `SquaredNormGuardIsDifferentiableAtAZeroQuaternion`, `testBaseOrientationCppAd`). Double precision applies the same
   rule with an `if`. A quaternion with ‖ξ‖ ≤ ε is left unchanged; `R` (Eigen's polynomial `toRotationMatrix`) is then I
   to within 2‖ξ‖² per entry, exactly I at zero, and the heading-tilt functions read every such ξ as the zero quaternion
   (section 2.8.1). `Eigen::normalized()` / `normalize()` are never called on `ad_scalar_t`: they record a comparison
   between variables.
2. **Tape point.** `CppAdInterface` tapes at `x = ones`, so `ξ̂ = ½(1, 1, 1, 1)` is a valid unit quaternion.
3. **No plain `if`, no `Eigen::Quaternion(Matrix3)`, no `AngleAxis`, no log map or `atan2` at zero error inside a tape.**
   The base-orientation residuals (sections 2.8.1 to 2.8.3) are analytic double-precision code, not taped. The taped
   end-effector residuals keep their CppAD-safe forms: `matrixToQuaternion` for the AD scalar, the torso cost's rotation
   vector through `quaternionLog` (its Taylor branch is selected by `CondExpGt` at and near zero error, and the guarded
   `atan2` branch is finite there, so neither a plain `if` nor an `atan2` at zero is taped;
   `//humanoid_nmpc/humanoid_common_mpc:testEndEffectorOrientationError` checks the generated Jacobian at zero error),
   `rotationMatrixDistanceToPlane` (the constraints' half-angle tilt), the foot costs' tilt angle `tiltAngleToPlane`
   (`quaternionLog` of the shortest arc, whose series branch is selected at and near a flat foot;
   `//humanoid_nmpc/humanoid_common_mpc:testTiltAngleToPlane` checks the generated Jacobian there) and the half-angle
   form of `footYawError`.
4. **Pinocchio asserts.** Pinocchio's spherical `calc` may assert a unit norm; under `-c opt`, the repository default,
   asserts are compiled out, and no taping test relies on `-c fastbuild` with asserts.

### 2.7 Transcription and SQP on the manifold

Here `Φ(x, u)` is OCS2's RK4 node map, with ambient sensitivities `A_a = ∂Φ/∂x` (nx×nx) and `B_a = ∂Φ/∂u`.

**Intermediate node** (`projectIntermediateNodeOnManifold()`, `ocs2/oc/.../multiple_shooting/ManifoldProjection.h`):

```math
\hat F=\Pi(\Phi),\quad b_k=\mathrm{difference}(x_{k+1},\hat F),\qquad
A_k=\mathcal J(b_k)\,E^+(\hat F)\,\Pi'(\Phi)\,A_a\,E(x_k),\qquad
B_k=\mathcal J(b_k)\,E^+(\hat F)\,\Pi'(\Phi)\,B_a
```

- `𝒥(b) = blockdiag(I, Jr⁻¹(b_φ), I)`; on the quaternion rows `E⁺(F̂)Π' = 2Gᵀ(ξ̂_F)/‖ξ_Φ‖`, because `Gᵀξ̂ = 0`.
- OCS2's dynamics struct receives `f = b_k`, `dfdx = A_k`, `dfdu = B_k`. This is the exact Newton linearization of
  `x_{k+1} ⊕ δx_{k+1} = F(x_k ⊕ δx_k, u + δu)`, since `Log(Exp(b)Exp(η)) ≈ b + Jr⁻¹(b)η`; the coefficient on
  `δx_{k+1}` stays −I, as HPIPM expects (`test_manifold_projection` against finite differences).

**Other nodes and steps:**
- **Event nodes** (`projectEventNodeOnManifold()`): identity jump map, so `b = difference(x_next, x)`, `A = 𝒥(b)`.
- **Costs and constraints** (`pullBackCost()`, `pullBackConstraint()`, before the projection of the state-input
  equality constraints): `g_t = Eᵀg_a`, `H_t = EᵀH_aE`, `H_ux,t = H_ux,a·E`, `C_t = C_a·E`. The retraction-curvature
  term `−¼(∇_ξℓ·ξ)I₃` vanishes by D5.
- **Initial state** (`SqpSolver`): `δx₀ = difference(x[0], x_init)`, and the same for the initial violation. An
  initial state without a rotation, a quaternion block at or below `kQuaternionNormalizationEpsilon` (the zero
  quaternion) or NaN (`StateManifold::hasRepresentableRotations()`), is refused: the solve throws, which
  `MpcServer` reports as a failed solve. `retract()` cannot move such a block and `difference()` reads every attitude
  as no rotation away from it, so a plan from it would keep that attitude for as long as it is warm started
  (`testSqpOnSO3`).
- **Step** (`retractTrajectory()`): `x_new = x ⊕ αδx`; inputs keep `incrementTrajectory`.
- **Metrics:** the dynamics violation is `difference(x_next, Π(Φ))`; the violation norms, the Armijo metric and the
  filter line search are the flat formulas on tangent vectors, in radians on the rotation.
- **Warm start:** manifold interpolation of the previous solution. If `getMaximumRotationAngle(difference(x[0],
  x_init)) > π/2`, the previous solution is dropped (with a WARNING) and the trajectories start from `x_init`: this does
  not happen in nominal real-time iteration and keeps the gap away from the cut locus of `Log` at π after a reset or a
  fall. A warm start with a state without a rotation (as above) is dropped the same way: it reads as no rotation away
  from `x_init`, so the quarter-turn rule alone would keep it.
- **No state re-anchoring** exists or is needed: every iterate lies on M and every difference is intrinsic. The policy
  gain is the one quantity carried from the pre-step to the post-step iterate (below).
- **Solver.** The SQP is the fork's only solver (`lib/ocs2/README.md`); it builds no value function.
- **Unchanged:** HPIPM (its sizes from the tangent `dfdx.cols()`; `δx₀` eliminated), the projection of state-input
  equality constraints and `remapProjectedGain`, `PerformanceIndexComputation`, the filter line search. No state-only
  equality constraint reaches the QP, and none is added: there is no norm constraint.

**Policy** (`multiple_shooting::toPrimalSolution(..., stateManifold)`, `ocs2/oc/src/multiple_shooting/Helpers.cpp`):
with a non-flat manifold the SQP builds a `ManifoldLinearController(time, x̄, u*, K, manifold)`; pre-event nodes copy
the anchor, input and gain of the node before them, and the last node those of the one before it.

```math
u(t,x)=(1-\alpha)\big[u^*_k+K_k\,\mathrm{difference}(\bar x_k,x)\big]+\alpha\big[u^*_{k+1}+K_{k+1}\,\mathrm{difference}(\bar x_{k+1},x)\big]
```

- **Which gain at which anchor (no gain transport).** `K_k` is the last QP's gain,
  linearized at the pre-step iterate `x_old,k`; the controller applies it with unchanged body-frame components at the
  post-step anchor `x̄_k = x_old,k ⊕ αδx_k` (`getRiccatiFeedback` after the step), as the flat solver applies its
  pre-step gain at the post-step state. On a flat state the law `u_new + K(x − x_new) = u_old + αk + K(x − x_old)` is
  the QP's affine law exactly. On a quaternion block, with `a = αδx_k,φ`, `e = difference(x̄_k, x)` and `K_rot` the gain's
  columns of that block, the published law and the QP's chart law `u_old + αk + K·difference(x_old,k, x)` differ by
  exactly `K_rot(e + a − Log(Exp(a)Exp(e))) = −K_rot(Jr⁻¹(a) − I)e + O(‖e‖²)`.
  `Jr⁻¹(a) − I = ½[a]× + c(t)[a]×²` is normal, with `t = ‖a‖`, `c(t) = 1/t² − (1 + cos t)/(2t sin t)` and
  `‖Jr⁻¹(a) − I‖₂ = (t/2)·√(1 + 4c(t)²t²)`: at most 1.037·t/2 for `t ≤ π/2`, the warm-start guard's bound on the step of
  node 0. `½‖K_rot‖·t·‖e‖` is therefore the leading term of the difference, not a bound on it.
- **Neither law is the exact policy's linearization.** For a smooth row `f` of the exact policy, with body gradient
  `k(x)` and `g(d) = f(x Exp(d))` its log-chart function at `x`, `k(x Exp(a)) b = ∇g(a) Jr⁻¹(a) b`, so
  `k(x Exp(a)) = [k(x) + aᵀH] Jr⁻¹(a) + O(‖a‖²)` with `H` the (symmetric) Hessian of `g` at 0. To first order the body
  gain changes by the bilinear form `aᵀHb + ½k·(a × b)`: a symmetric part that depends on the problem and an
  antisymmetric part that `k` alone fixes. Each choice of gain at `x̄_k` assumes something about the unknown `H`, which
  a real-time iteration does not compute:

| Gain at `x̄_k` | First-order error against the exact gain, as a form in `(a, b)` | Exact when |
|---|---|---|
| `K_old` (the code: body-frame components kept) | `aᵀHb + ½k·(a × b)` | the policy is invariant along the step, as along the orbit of a world-frame rotation that turns the orientation alone and leaves the dynamics, every cost and every constraint (references fixed) unchanged: the heading of an attitude-only model whose terms see the orientation through its tilt alone. Not the robots' MPCs: their world-frame momentum, positions, contact forces and references make the policy at most equivariant, `u*(g·x) = g·u*(x)` with `g` turning every world-frame quantity, so the gain turns with `g` |
| `K_old Jr⁻¹(a)` (the QP's chart) | `aᵀHb` | the policy is affine in `difference(x_old, x)` |
| the Euler layout's flat law | the ZYX chart's Hessian form | the policy is affine in the ZYX angles; along a world-yaw step `Θ(R_z(t)R) = Θ(R) + (t, 0, 0)`, so there it agrees with the code |

- Averaged over step and deviation directions the chart transport has the smaller squared error (by `½‖k‖²` per row in
  the Frobenius norm of the forms). Direction by direction neither wins on the robots: the code's gain is exact along the
  heading of an attitude-only model with a tilt-only cost (the transport off by `½|k|t`), but with a world-frame input,
  as contact forces are, that policy is only equivariant, and along the heading the code's gain is the further off (0.32
  against the transport's 0.18 at `t` = 0.05 rad, 2.53 against 1.41 at 0.4 rad, `‖K*‖` = 6.4, in a horizon-3 attitude
  problem). The code keeps the body-frame gain for the reasons that hold whatever the problem: no
  choice is exact without `H`; either choice moves the gain by about `½‖K_rot‖·αδx_φ`, 1 to 1.5 % at the typical 0.02 to
  0.03 rad gaps and about 20 % at 0.4 rad, the order of the flat RTI's own pre-step-gain error and second order in the
  policy; and along a world-yaw step it is the Euler layout's law. `ManifoldLinearController.h` states the law, and
  `test_sqp_on_so3` pins it after one SQP iteration: `ThePublishedGainIsTheQpsOwnWithItsComponentsUnchanged`
  identifies the QP's `K_0` independently, from two solves of the same QP whose initial gaps differ, and finds the
  published gain equal to it to 1.3e-13 where the transported `K_0·Jr⁻¹(a_0)` is off by 7.0 (`‖K_0‖` = 45.7,
  `|a_0|` = 0.39 rad); `ThePolicyAppliesThePreStepGainAtThePostStepAnchorAsDocumented` pins the anchor, the quadratic
  remainder and the closed-form norm of `Jr⁻¹(a) − I`.

### 2.8 Costs and constraints

#### 2.8.1 Heading-tilt residual (base orientation, ACoM)

For a unit `ξ` (the costs always pass `ξ̂`; `common/orientation/BaseOrientation.h`):

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

`twistHeading()` is `ψ_h`, `headingQuaternion()` is `q_t`, `swingQuaternion()` is `s` and `tiltVector()` is `τ`. `n` is
`‖s_xy‖`, never `sqrt(1 − ρ²)`, so a level attitude is not 0/0. Because `s_w = ρ ≥ 0`, `τ = Log(s)_xy` needs no sign
flip and |τ| ≤ π; double code clamps `ρ ≥ 1e-9` (the upside-down guard).

**Singular sets.** The split has two, and `BaseOrientation.h` states what every function returns there:
- *Inside the normalization guard* (‖ξ‖ ≤ ε, section 2.6), where `ξ̂ = ξ` is not a unit quaternion: the zero
  quaternion's values, exactly: `ψ_h = 0`, `q_t = I`, `s = 0`, `τ = 0` (so `e_HT(ξ, ξ_r) = e_HT(I, ξ_r)`), and
  `J_HT = 0`, the exact derivative of the residual, which is constant there. Read as the non-unit ξ̂ instead, the swing
  has n and ρ of the order of ‖ξ‖, the test `n < 1e-6` takes the series at every tilt, and the tilt came out 0.9 % off
  at ZYX (0.8, 0.6, 0.7), and as −12.15 rad for a roll of 2.5 rad. Scale invariance holds for ‖ξ‖ > ε. A NaN
  coefficient is not read as the identity: it reaches every function reading it (`isInsideNormalizationGuard` is
  `|ξ|² ≤ ε²`; `BaseOrientation.AQuaternionWithANaNIsNotReadAsTheIdentity`).
- *Exactly upside down* (ρ = 0, ξ_z = ξ_w = 0), where the heading is undefined: it is read as 0 and `q_t` as the
  identity for ξ and −ξ alike, so `s(−ξ) = −s(ξ)` and `τ(−ξ) = −τ(ξ)`: `τ(ξ) = π·n` and `−π·n` are the two logarithms of
  the same half turn about the horizontal axis `n = (ξ_x, ξ_y)/‖(ξ_x, ξ_y)‖`. No rule on the coefficients' values keeps
  both `τ(−ξ) = τ(ξ)` and `τ(ξ⁻¹) = −τ(ξ)` there (they would force `τ = 0`); the split keeps the inversion, as `Log` does
  at a half turn (section 2.1). Nor can a heading-zero rule be invariant to world yaw there: `q_z(α)⊗ξ` is the half turn
  about `R_z(α/2)·n`.
- In the clamp band 0 < ρ ≤ 1e-9, `q_t` comes from the angle of `(ξ_z, ξ_w)` brought to the hemisphere `ξ_w > 0` (or
  `ξ_w = 0 < ξ_z`) with its sign restored, so −ξ gives exactly `−q_t` and the same swing, and the sign invariance holds
  bit for bit.

Neither set is reached in operation: stored and reference states are unit quaternions, and an exactly upside-down one
needs `ξ_z = ξ_w = 0` in floating point. `testBaseOrientation` pins all three
(`InsideTheNormalizationGuardEveryFunctionReadsTheZeroQuaternion`, `ExactlyUpsideDownMinusXiGivesTheOtherLogarithmOfTheHalfTurn`,
`DoubleCoverIsExactInTheUpsideDownBand`).

The residual, rows in the order of the tuning layout (yaw, pitch, roll) (`headingTiltResidual()`):

```math
e_{HT}(\xi,\xi_r)=\big[\ \mathrm{wrap}_\pi(\psi_h(\xi)-\psi_h(\xi_r)),\ \ \tau_y(\xi)-\tau_y(\xi_r),\ \ \tau_x(\xi)-\tau_x(\xi_r)\ \big]
```

Its Jacobian with respect to the body right perturbation of `ξ` (`headingTiltJacobian()`):

```math
J_{HT}(\xi)=\begin{bmatrix}\nabla_\xi\psi_h\\ \nabla_\xi\tau_y\\ \nabla_\xi\tau_x\end{bmatrix}\tfrac12 G(\xi),\qquad
\nabla\psi_h=\tfrac{2}{\rho^2}(0,0,w,-z),
```

with, in coefficient order (x, y, z, w), `N = (wx + zy, wy − zx)`, `∂N_x = (w, z, y, x)`, `∂N_y = (−z, w, −x, y)`,
`∂ρ = (0, 0, z, w)/ρ`, `s_xy = N/ρ`, `∂κ/∂n = (2ρ/(n² + ρ²) − κ)/n` (series `−4n/(3ρ³)`), `∂κ/∂ρ = −2/(n² + ρ²)`, and
`∂τ = κ·∂s_xy + s_xy·(∂κ/∂n·ŝ_xyᵀ∂s_xy + ∂κ/∂ρ·∂ρ)`. All are pinned by finite-difference tests.

| Property | Value |
|---|---|
| `J_HT` at any level attitude `q_z(ψ)` | `P₃`; hence `J_HT·T_B(0) = I`: the Euler-coordinate Hessians are equal at a level **state** (any reference). At a tilted state the Euler-chart Jacobian `J_HT(φ(Θ))·T_B(Θ)` deviates from `I` by about `\|tilt\|/2` (the yaw row picks up `−tan(θ/2)`, `−tan(φ/2)`), even against a level reference, so `J_eᵀQJ_e − Q = O(\|tilt\|·\|Q\|)` (`testBaseOrientation` `EulerChartJacobianIsTheIdentityOnlyAtALevelState`) |
| Single-axis tilt | `τ = (φ, θ)` exactly |
| Mixed tilt, relative error against (φ, θ) | at `θ = φ` = 0.05, 0.1, 0.2 rad (per axis, `\|tilt\| = √2·θ`): 0.066 %, 0.26 %, 1.06 %; at `\|tilt\| = \|(θ, φ)\|` = 0.05, 0.1, 0.2 (section 4.4's measure): 0.033 %, 0.13 %, 0.53 % on the diagonal, 0.034 %, 0.14 %, 0.55 % in the worst direction |
| Yaw row against Euler | `e_yaw − (ψ − ψ_r) = wrap(o(θ, φ) − o(θ_r, φ_r))`, `o(θ, φ) = −2·atan(tan(θ/2)tan(φ/2)) = −½θφ(1 + (θ² + φ²)/12 + …)`; `½\|θφ\|` is the leading term, not a bound |
| Yaw-error independence | the tilt rows are exactly independent of the yaw error |
| Pitch 90° | finite, full rank (singular values about 1.95, 1.00, 0.80) |
| Double cover | for ρ > 0, `s(−ξ) = s(ξ)` and `q_t(−ξ) = −q_t(ξ)` bit for bit (in the clamp band too); `ψ_h(−ξ) = ψ_h(ξ)` exactly at every ξ, so the residual is invariant bit for bit; exactly upside down `τ(−ξ) = −τ(ξ)` (above) |
| Singularities | a tilt of π only (upside down); the yaw row is discontinuous at a yaw error of exactly ±π, which re-seeded references keep away (section 2.9) |
| SA01 weights (0 / 5 / 85) at a mixed tilt, against a level reference | cost −0.15 %, −0.61 %, −2.4 % at `θ = φ` = 0.05, 0.1, 0.2 rad; −0.076 %, −0.31 %, −1.22 % at `\|tilt\|` = 0.05, 0.1, 0.2 on the diagonal |

#### 2.8.2 The tuning deviation and the quadratic state costs

`TuningLayoutDeviation` (`common/orientation/TuningLayoutDeviation.h`, compiled into `:humanoid_common_mpc` to keep
`:orientation` free of robot-model dependencies) is the one evaluation of tuning rows. It is a copyable, immutable
value built from the robot model (retained; the model outlives it), held by value in the `Tuning*` costs, and it walks
`getTuningLayout()` segment by segment, reading the state only through the model (`getBaseOrientation`, `getBaseRotationTangentJacobian`, `getBaseAngularVelocityLocal`,
`getBaseAngularVelocityLocalJacobian`), so it never branches on the layout:

| Segment kind | Residual rows | Jacobian blocks |
|---|---|---|
| `kEuclidean` | `x[s] − x_r[s]` on the segment's state slots | an identity run |
| `kEulerZyxOrientation` (`base_orientation`) | `e_HT(ξ̂, ξ̂_r)` (yaw, pitch, roll) | `J_HT(ξ̂)·getBaseRotationTangentJacobian(x)` on the 4 orientation slots |
| `kZyxOrderedAngularVelocity` (whole body `base_angular_velocity`) | `e_ω = P₃(ω_B − R(ξ̂)ᵀR(ξ̂_r)ω_{B,r})` (z, y, x) | `P₃` on the angular-velocity slots; `−P₃[R(ξ̂)ᵀR(ξ̂_r)ω_{B,r}]×·getBaseRotationTangentJacobian(x)` on the orientation slots |

So the centroidal and whole-body residuals are

```math
r_c=\big[\bar h-\bar h_r;\ p-p_r;\ e_{HT}(\hat\xi,\hat\xi_r);\ q_j-q_{j,r}\big],\qquad
r_{wb}=\big[p-p_r;\ e_{HT};\ q_j-q_{j,r};\ \dot p-\dot p_r;\ e_\omega;\ \dot q_j-\dot q_{j,r}\big].
```

`linearize()` returns a `TuningDeviationApproximation` (the residual, identity runs and dense blocks), and the costs
assemble the Gauss-Newton model block by block (`tuningGradient()`, `tuningHessian()`, `common/cost/TuningGaussNewton.h`):
`M = QJ` by copying Q's columns for identity runs and `Q(:, rows)·block` for dense blocks, then `JᵀM` likewise, O(n_t²)
plus O(n_t × 12) per block, never a dense J:

```math
L=\tfrac12 r^\top Q r+\tfrac12\delta u^\top R\,\delta u,\qquad
\nabla_x L=J^\top Q r,\quad \nabla^2_{xx}L=J^\top Q J,\quad \nabla_u L=R\,\delta u,\quad \nabla^2_{uu}L=R,\quad \nabla^2_{ux}L=0 .
```

`J` is with respect to the ambient state (D4), so the pull-back recovers `J_tᵀQr` and `J_tᵀQJ_t` in the tangent. OCS2's
cross term is unused on every shipped robot and is dropped.

- `TuningQuadraticStateInputCost` (running, `ocs2::StateInputCost`) and `TuningQuadraticStateCost` (terminal,
  `ocs2::StateCost`) hold Q (n_t square) and R; `setStateWeights()` / `setInputWeights()` refuse a wrong size,
  asymmetry or a non-finite entry (`validateQuadraticWeights()`), and their approximations are `final`.
  `StateInputQuadraticCost`, `StateQuadraticCost` and `BasePoseShapedQuadraticStateCost` derive from them;
  `InputQuadraticCost` is input-only (`setInputWeights()`), which removed the old `zeroQ` sizing hazard.
- **References:** running `x_r = referenceManager.getDesiredState(targets, x, t)`, which interpolates through the robot
  model (section 2.9); `u_r = getDesiredInput(...)`; terminal `x_r = shapeBasePose(t, model.getDesiredState(targets, t))`,
  weighted by `terminal_cost_scaling`.
- On a layout of differences only, the costs are OCS2's `QuadraticStateInputCost` arithmetic bit for bit
  (`testTuningQuadraticCost`); on the robots' layouts they equal the Euler differences at a level state.

#### 2.8.3 `ComAndAcomTrackingCost` (DRC Atlas)

`common/cost/ComAndAcomTrackingCost.h`, `common/acom/AngularCenterOfMass.h`:

```math
\xi_a(x)=\hat\xi\otimes \mathrm{Exp}\big(\Delta\theta(q_j)\big),\qquad e_{acom}=e_{HT}\big(\xi_a(x),\xi_a(x_r)\big),\qquad e_{com}=c(q)-c(q_r)
```

```math
\frac{\partial e_{acom}}{\partial\delta\varphi}=J_{HT}(\xi_a)\,R(\Delta\theta)^\top,\qquad
\frac{\partial e_{acom}}{\partial\delta q_j}=J_{HT}(\xi_a)\,J_r(\Delta\theta)\,J_{\Delta\theta}(q_j),\qquad
\frac{\partial e_{com}}{\partial(p,\delta\varphi,\delta q_j)}=J_{com}
```

- `acomAttitude(ξ, q_j)` composes the attitude and `acomAttitudeTangentJacobian(q_j)` returns `R(Δθ)ᵀ` and
  `J_r(Δθ)·J_Δθ` from one network pass; `computeJointOrientationOffset()` and its Jacobian stay. The orientation block is
  that times `getBaseRotationTangentJacobian(x)`, and `J_com` (Pinocchio's 3×nv CoM Jacobian) is placed through
  `getPinocchioTangentJacobian(x)`. `Create` checks nq, nv and the joint count against the model.
- `Δθ` is the network's base-frame rotation offset. The network is fitted at the identity base
  (`humanoid_learning/acom/dataset_generator.py`), so it is independent of the base attitude; the Python mirror is `acom_attitude` /
  `acom_attitude_tangent_jacobian` (`humanoid_learning/acom/models.py`, the `acom_attitude` IFTTT pair).
- What the network is fitted to follows from the composition. The world angular velocity of `ξ_a` is
  `R(ξ)·(ω_B + J_l(Δθ)·J_Δθ·q̇_j)`, since `R(Δθ)·J_r(Δθ) = J_l(Δθ)` (the left Jacobian of SO(3)), and the locked
  angular velocity `I_G⁻¹L_G` is `R(ξ)·(ω_B + Ā_ω,j·q̇_j)` with the connection `Ā_ω,j` in base axes. The base term
  agrees exactly, so the ACoM is the locked rotation exactly where `J_l(Δθ)·J_Δθ = Ā_ω,j`, and training minimizes
  `‖J_l(Δθ)·J_Δθ − Ā_ω,j‖²_F` (`train_acom.py`, `SirenACOM.acom_joint_rate_jacobian`). The additive rule's objective
  `‖J_Δθ − Ā_ω,j‖²_F` agrees only to first order in `|Δθ|`; the shipped offsets average 20° to 35° and reach 80°,
  where `‖J_l − I‖₂` is 0.17 to 0.30 and 0.66. `testAcomAngularVelocityConsistency` grades the same map,
  `R(Exp(Δθ))·(J_r(Δθ)·J_Δθ)` from `acomAttitude` and `acomAttitudeTangentJacobian`, against Pinocchio's `ccrba`.
- `acom_weights` rows stay (yaw, pitch, roll); `ComAndAcomTrackingCost::setWeights()` takes them at start-up and live.
- Against the additive rule `Θ_a = Θ + P₃Δθ` of the ZYX angles: the composition moves the attitude by
  O(|Δθ|·|tilt| + |Δθ|²) from it, and the heading-tilt rows of the composed attitude differ from its ZYX angles as in
  section 2.8.1: cubic in the tilt on the pitch and roll rows, and on the yaw row by the twist-against-ZYX identity
  `wrap(o(θ_a, φ_a) − o(θ_{a,r}, φ_{a,r}))`, O(|tilt|²), which does not vanish with Δθ: at Δθ = 0 and a base tilt
  (0.1, 0.1) against a level reference the yaw row is −5.008e-3 rad, where the additive rule gave 0. `state_weights`'
  yaw weight is 0 on every robot, but Atlas weighs the ACoM yaw row by 25 × scaling 100 = 2500. The Jacobian differs at
  first order, by at most `c·(|tilt| + |Δθ|)` per entry (section 4.4).
- The contact planner's heading with the ACoM is `twistHeading(acomAttitude(...))`, and `PlannedHeadingOverride` takes
  its offset from `acomHeadingOffset()`, the twist of `Exp(Δθ)` (exact for level knots).

#### 2.8.4 Analytic terms that use Pinocchio Jacobians

- **`ContactWrenchConeConstraint`:** `dfdx = (A_f[f_l]× + A_τ[τ_l]×)·J_ang,LOCAL·getPinocchioTangentJacobian(x)`;
  `J_ang,LOCAL` is already the body tangent. Exact (E1).
- **`ExternalTorqueQuadraticCostAD`:** `J_ee` is (6, nv). Exact.

#### 2.8.5 Terms whose residual is unchanged

These read `q`, `v` and `a` through the accessors, and their residuals do not read the orientation's parameterization.

| Term | Note |
|---|---|
| Torso `EndEffectorKinematicsQuadraticCost` | the rotation vector `Log(q_r ⊗ q⁻¹) = θn` in world axes (`computeTaskSpaceErrors()`), so `task_space_costs[].weights.orientation_{x,y,z}` are per rad² about the world axes. It replaced `quaternionDistance`, the half-angle sine `sin(θ/2)n`; a weight `w` on that is `w/4` on `θn` at small angles, and the shipped weights were divided by 4 (G1 and R1 100 → 25), which keeps the closed loop at small errors. The reference is the forward kinematics of the slerped, shaped `x_r`. |
| `CentroidalMpcEndEffectorFootCost` | plane distance and half-angle foot yaw; the plane-normal yaw is `getBaseHeading(x_r)`. |
| `EndEffectorDynamicsFootCost` (whole body) | unchanged; the accelerations come from the ω-state flow. |
| `EndEffectorKinematicsTwistConstraint` (the centroidal zero-velocity constraints) | unchanged (`rotationMatrixDistanceToPlane`, `M(R, ω)`). Its value is `b + A_x·[p; e] + A_v,lin·v + A_v,ang·M(x)·ω`, `ω` the foot's world-frame angular velocity and `M = ∂e/∂ω` (plus `½·n·v_nᵀ` under `position_and_orientation`, `v_n` the foot normal), and `getLinearApproximation()` holds `M` at its value at the point: its state Jacobian is the derivative less exactly `D = A_v,ang·(∂M/∂x)·ω` (along the retraction `D_i = A_v,ang·(d/ds M(x ⊕ s·e_i))·ω`; a soft form's gradient lacks `p'(f)ᵀ·D`), and its input Jacobian is exact, since `M` reads no input. `testQuaternionRootTermDerivatives` computes `D` independently (`M` differenced along the retraction, `ω` the twist's own value) and holds the reported Jacobian plus `D` to the differences of the value at every point, entry by entry within their error bound, which rejects `D` with one entry moved by 1e-6: the residual is at most 2e-12 of the Jacobian's scale on every centroidal robot, so `D` is the whole difference; the soft form, which no shipped robot lists, is held the same way through a soft wrapper of one stance foot's constraint that the test adds to its copy of the problem (`D` contracted with `p'(f)`); it also holds the tilt rows of `M·ω` to the rate of `e` along the flow map, `(∂e/∂x)·f_a`, to round-off (section 4.4). `D` is linear in `ω`, up to about 5 % of the Jacobian's scale per rad/s of the foot on the SA01, G1 and R1 (`K_d` = 1) and 0.1 % on the Atlas (`K_d` = 0.01): 0 at rest, at most 1 % at the points built from the walk (feet turning at 0.05–0.26 rad/s), and at `random_0` 0.18 % on the Atlas (2.3 rad/s), 2 % on the SA01 (0.7 rad/s), 2.8 % on the R1 (0.8 rad/s) and 64 % on the G1, whose draw of the normalized angular momentum (within 0.2 m²/s per axis) spins it at 13.5 rad/s about the vertical, its lightest axis (`I_zz/m` ≈ 0.0135 m² against ≈ 0.1 m² about the horizontal ones) - a stress point for the derivatives, not a stance the MPC visits. At a solution the stance rows hold `K_p·e + K_d·M·ω = 0`, so a flat foot (`e = 0`) has `M·ω = 0`: `ω = 0` and `D = 0` under `position_and_orientation` (Atlas), while `position_and_tilt` (SA01, G1, R1) leaves the pivot rate `ω_n` about the foot normal to the other terms, and then `D = −ω_n·A_v,ang·M·∂v_n/∂x` (from `M(x)·v_n(x) = 0`); a tilted foot turns at about `(K_p/K_d)·tilt`. The value is exact, so a converged iterate is feasible; `D` enters only the linearization, and at a fixed point it leaves the exact Lagrangian's state gradient at `Dᵀλ` instead of 0. The whole body's `ZeroAccelerationConstraintCppAd` maps `ω` through the same `M` but tapes the rates themselves, the map included (section 2.8.6): its Jacobian is held exact at every point. |
| `FrictionForceConeConstraint` | unchanged residual. Its state Hessian is a regularization, a fixed diagonal shift `−sI` in the ambient state, so its pull-back onto the quaternion block is a quarter of the Euler one (`EᵀE = ¼‖ξ‖²I₃`); the class comment says so. |
| Normal-velocity, swing-vertical, `ContactMomentXY`, `ForceWeightedSlip`, `ContactComplementarity`, `GroundPenetration`, `FootCollision`, `DcmTerminalCost`, `JointTorqueCostCppAd` | E1 (section 4.4). `DcmTerminalCost` reads its target through the model interpolation. |
| `InputQuadraticCost`, `ZeroWrench`, `BasisScalingNonNegativity`, `JointLimitsSoftConstraint`, both `JointMimic*` | E0. Indices through `getJointStartindex()` / `getJointVelocitiesStartindex()`. |
| Contact-planner SO(2) terms (`common/src/contact_planning/{cost,constraint}`) | unchanged; their heading reads go through the boundary (section 2.9). |

#### 2.8.6 The whole-body stance constraint

`ZeroAccelerationConstraintCppAd`, the whole body's stance-foot constraint (the `zero_velocity` term), evaluates an
`EndEffectorDynamicsAccelerationsConstraint`, a PD law on the foot's pose with respect to the ground plane. Unlike the
terms of section 2.8.5, its residual changed: its orientation rows used to read the foot's raw angular velocity and
acceleration.

- **Law.** `g = b + A_x·[p; e] + A_v·d/dt [p; e] + A_a·d²/dt² [p; e]`, `p` the foot's position and
  `e = −(v_n × n)/√(2(1 + v_n·n))` its orientation error to the ground plane (`v_n = R·e_z` the foot normal, `n = e_z`),
  the sine of half the foot's tilt. With the diagonal gains of `stanceFootAccelerationConstraintConfig()` the
  orientation rows are `K_p·e + K_d·ė + K_a·ë`, the PD law on the error. Before, they read the foot's raw `ω` and `α`:
  twice `ė` at a level foot, twice `ë` but for the term under **Gains**, and with no error term for the rotation
  about `n`.
- **Rates.** With `ω` and `α = dω/dt` the foot's world-frame angular velocity and acceleration (the angular parts of
  Pinocchio's `LOCAL_WORLD_ALIGNED` frame velocity and classical acceleration, which `testStanceFootOrientationRates`
  differences along the robot's motion), `v̇_n = ω × v_n`, `v̈_n = α × v_n + ω × (ω × v_n)`, `a = v_n × n`,
  `c = v_n·n` and `s = √(2(1 + c))` (so `ṡ = ċ/s`): `ė = −ȧ/s + a·ċ/s³` (`= M·ω`, the centroidal map of the twist
  constraint, section 2.8.5) and `ë = −ä/s + (2ȧ·ċ + a·c̈)/s³ − 3a·ċ²/s⁵`. `e` has no component along `n`; that row
  carries half the rotation rate about the foot normal, `½·v_n·ω`, and its exact derivative `½·v_n·α`
  (`d/dt (v_n·ω) = (ω × v_n)·ω + v_n·α`), so a stance foot can neither tilt nor pivot. The linear rows are the foot's
  linear velocity and classical acceleration, as before.
- **Guard.** For `1 + c < 1e-6` both rates fall back to `ω/2` and `α/2`, the centroidal fallback (both read the guard
  `kOrientationErrorWrtPlaneAntiParallelThreshold` of `EndEffectorKinematicsTwistConstraint.h` and the half-angle
  scaling `kQuaternionHalfAngleFactor` of `UnitQuaternionMath.h`), as a conditional expression whose other branch is evaluated at `1 + c = 1`, so that no
  division by zero is taped.
- **Gains.** The G1's `angular_velocity_error_gain` and `angular_acceleration_error_gain` doubled with the law (20 → 40,
  1 → 2): at a level foot `ė = ω/2` and `ë = α/2 + ½(n·ω)(n × ω)`, so the law is the former one but for
  `(K_a/2)·(n·ω)(n × ω)`, which vanishes at rest and on the row along `n`, while a tilted foot is held by the rates of
  its error.
- **Libraries.** The two rates are CppAD libraries of their own, `<foot>_pose_wrt_plane_rate` and
  `<foot>_pose_wrt_plane_acceleration` (`end_effector/OrientationErrorWrtPlaneRates.h`, the plane normal a parameter),
  6 rows each, the linear part included. Value and linearization read the same libraries, so the Jacobian is the
  derivative, with no frozen map. They took the place of `<foot>_twist` and `<foot>_accelerations`, and with them no
  term reads the foot's orientation quaternion, its error to a reference quaternion or its raw angular rates through
  `PinocchioEndEffectorDynamicsCppAd` any more. The whole-body `EndEffectorDynamicsFootCost` (`task_space_foot_cost`,
  shipped on the G1-WB) tapes its own kinematics and still weighs the foot's orientation to the plane, its angular
  velocity and its angular acceleration, and the centroidal foot cost and twist constraint read the foot's angular
  velocity (section 2.8.5). `PinocchioEndEffectorDynamicsCppAd` generates `<foot>_position`, `_velocity` and
  `_linear_acceleration` (the swing foot's normal-motion constraint), `_orientation_wrt_plane` and the two rates, and
  no longer `_orientation`, `_orientationError`, `_angular_velocity` or `_angular_acceleration` (section 3.2). Its taped
  functions capture the frames by value, one Pinocchio AD interface that every copy shares and a pointer to the robot
  model, which must outlive the object and every copy, so a copy never tapes through the object it was copied from.
- **Tests.** `testOrientationErrorWrtPlaneRates` and `testStanceFootOrientationRates` (section 4.4); the first-order
  equivalence suite differences the constraint, hard and in the soft wrapper it adds, at every probe point (section 5).

### 2.9 References, heading and the Euler boundary

**Human-facing layer.** `TargetTrajectoriesCalculatorBase` (`common/command/`) stays Euler: a pose
`[x, y, z, ψ, θ, φ]` with an unbounded scalar ψ, Δψ in degrees.
- `getCurrentBasePoseTarget()` is `[getBasePosition(x), getBaseHeading(x), 0, 0]`.
- `integrateTargetBasePose()` moves a knot along the arc of a constant planar twist, `Δp = T sinc(θ/2) R_z(θ/2) v₀`,
  `θ = ψ̇T`, `Δψ = θ` (`planarArcDisplacement()`; the straight chord at the starting heading is 0.065 m off the
  arc at the end of a 1.2 s horizon at 0.3 m/s and 0.3 rad/s).
- **Velocity-command knots** (`velocityCommandKnots()`, both calculators): over the first segment `[t₀, t₀ + 0.7T]` the
  base holds the mean `((v_m + v_c)/2, (ω_m + ω_c)/2)` of the measured and the commanded planar twist in its heading
  frame, then the command up to `t₀ + T` (D1, D1b).
  - The measured twist is the robot's mean motion over its last stride (`ProceduralMpcMotionManager`,
    `StrideMotionHistory`): the rate of the twist heading, `(Ψ(t) − Ψ(t − W))/W`, and the heading-frame velocity of the
    base, over the cycle `W` of the gait the schedule runs. The boxcar over one cycle removes every harmonic of the
    gait's sway exactly (its response `sinc(πfW)` vanishes at `f = k/W`), and the heading excludes the joints' motion by
    construction (the instantaneous centroidal "base twist" `A_b⁻¹(m h̄)` was the joints-locked twist
    `I_G⁻¹L`, wrong by more than the yaw rate itself on a walking robot, and even the exact pelvis twist sways past the
    gait thresholds).
  - The first segment assumes the body twist ramps linearly from the measured to the commanded over `τ = 0.7T`; holding
    the mean twist ends at the ramp's heading exactly and, on a straight line, at its position; on a turn the position
    differs by `(τ²/12)R_z(ψ₀)J(ω_m v_c − ω_c v_m) + O(τ³ω²|v|)`, `J = R_z(π/2)` (4.4 mm at τ = 0.84 s and 0.5 rad/s
    from 0.15 to 0.3 m/s; `testVelocityCommandArc`). The stride mean lags the instantaneous motion by about `W/2` and
    keeps the sway out of the measured term only: the knots are laid out from the heading of the attitude the solve
    starts from, which sways with the gait (0.093 rad from its stride mean on G1's recorded walk, 0.145 rad on R1's),
    turning the intermediate knot sideways by up to 0.039 m on G1 at 0.5 m/s.
  - When the stride history (re)starts, the motion before it is the exact base twist of the observation the solve
    starts from, `getBaseTwistInWorldFrame(state, jointVelocities)` with the observation's measured joint velocities,
    which the runners hand over before the solve (`ProceduralMpcMotionManager::observe()`, from
    `MpcServer::Hooks::preSolveObserver` in the MPC node and from `InProcessMpcLink::Config::solverObservationObserver`;
    the `observation_hand_over` IFTTT pair). In operation a history restarts at a full reset, while the robot is held or
    stands. A stall of the solves of up to 1.5 s (`StrideMotionHistory::kMaxBridgedGap`), a network stall among them,
    is bridged rather than restarting it: the heading rate exact while the window starts before it, and at most
    0.044 rad/s and 0.050 m/s off for a window starting inside a 1 s stall on the recorded walks.
  - The gait switches (D2) read the stride mean, its turn and its heading-frame velocity alike, only once its window is
    sampled densely throughout (`measuredOverTheWholeWindow()`: no restart and no bridged stall within it); until then
    only the switch out of stance, which reads no motion, is made. In the first cycle of a new gait the window still
    reaches into the gait before and reads the heading's sway as a turn of up to its peak-to-peak over `W` (0.12 and
    0.14 rad/s on the G1's and R1's recorded walks); holding the switches for that cycle too is an open decision
    (section 6).
- `estimateTimeToTarget()` uses the commanded scalar Δψ.

**Knot assembly.** Each calculator writes a knot through `setBasePoseEulerZyx(state, pose6)`, which writes
`ξ = q_z(ψ)⊗q_y(θ)⊗q_x(φ)`.
- **Whole-body velocity knot:** `ṗ_W,r = (R_z(ψ_k)(v_x, v_y), 0)`, the command in the knot's own heading frame,
  and `setBaseAngularVelocityFromWorldYawRate(state, ψ̇)` writes `ω_{B,r} = R(ξ̂_r)ᵀ(0, 0, ψ̇)`, which is `(0, 0, ψ̇)` for
  a level knot.
- **Centroidal momentum target** (`L/m = I_G(:, 2)ψ̇/m`): the channel that drives yaw, since the yaw weight is 0
  everywhere. At knot `k` it is that of the measured body turned about the world z axis to the knot's heading,
  `R_z(ψ_k − ψ₀)` applied to the linear and the angular part alike; `L_z`, and the yaw rate
  `getTargetPlanarCommand()` reads back as `m h_z / I_zz` with the composite yaw inertia the knot was built with, are
  unchanged.

**Knot spacing.** Pose commands are cut by `subdivideYawKnots()` into `n = max(1, ⌈|Δψ|/(π/2)⌉)` equal segments in
time, yaw and position; slerp about z between subdivided level knots is linear in yaw, so the node grid equals the
two-knot linear interpolation of the ZYX angles (`testPoseCommandSubdivision`). Velocity-command segments are cut by
`knotSubdivisionCount()` on their arc: the fewest equal parts for which consecutive knots are at most π/2 apart and the
linear interpolation between them stays within `ℓ|Δψ|/(8n²) ≤ 5 mm` of the arc and `|v_c|Δψ²/(8n²) ≤ 5 mm/s` of its
velocity target (`ℓ` the segment's arc length; between knots `δ = Δψ/n` apart the chord is at most `ρ(1 − cos(δ/2))` off
the arc of radius `ρ = |v|/|ω|`). The count is capped at 64 parts and a segment's turn is clamped at `64·π/2` (100.5 rad),
which only a measured spin above 239 rad/s at T = 1.2 s reaches, so consecutive velocity knots are at most π/2 apart for
every input (`testIntermediateKnotMeasuredRate` `AnyMeasuredSpinKeepsTheKnotsAQuarterTurnApart`). With three
knots alone the reference would be 0.05 to 0.21 m and as many m/s off the arc between them at the command limits of
SA01 to G1; cut into 7 to 11 knots it stays within 5 mm and 5 mm/s. A segment that only turns or only translates is cut for the
yaw spacing alone. The contact planner's dense resample (`ContactPlanningReferenceManager`) inserts knots where the base
turns by more than π/2 between plan nodes (above 63 rad/s at `dt = 0.025 s`).

**Interpolation.** OCS2's `TargetTrajectories::getDesiredState` stays linear for flat systems. The humanoid interpolates
every state trajectory through `MpcRobotModelBase::getDesiredState(targets, t)` / `interpolateStateTrajectory()`, which
slerp the orientation block: the `Tuning*` costs, the ACoM cost, the torso task-space cost, the foot costs,
`DcmTerminalCost`, `SwitchedModelReferenceManager`, `ContactPlanningReferenceManager` (its resampler and
`updatePredictedComState()`), and, for display, `PolicySnapshot` (states on the manifold's geodesic, inputs linear).
`state-interpolation` bans a one-argument `getDesiredState(t)` and a `LinearInterpolation::interpolate` of a state
trajectory under `humanoid_nmpc/` outside the robot models (Appendix B).

**Reads of the robot's heading use the twist `ψ_h`** (`getBaseHeading()`), and every consumer is wrap-safe (cos and
sin, a wrapped difference or `moduloAngleWithReference`): the swing-plane normal, the measured base yaw of the
reference manager, the arm swing, the heuristic context's yaw, the contact planner's input and heading (with the ACoM
`ψ_h(ξ_a)`, section 2.8.3), the current-pose target, and the reset targets. The foot yaws (`atan2(R10, R00)` of the
contact frame) are unchanged.

**Edits of level references at the Euler boundary:**
- `shapeBasePose()` (`SwitchedModelReferenceManager`, the `base_pose_heuristic_seam` pair): `(ψ, θ, φ) =
  getBaseOrientationEulerZyx(x)`, the heuristic offsets added with the `maximum_tilt` clamp, then
  `setBaseOrientationEulerZyx()`: Euler-additive arithmetic for any reference.
- `PlannedHeadingOverride`: replaces the ZYX yaw (re-branched with `moduloAngleWithReference`) and keeps θ and φ, through
  the same accessors; exact for the level knots it edits.

**The commanded twist and the reset heading** (section 4.7): `getCommandedVelocity()` and `getCommandedYawRate()` read a target
knot through `getTargetPlanarCommand(knot, compositeYawInertia)` (D3; the reference manager latches the composite yaw
inertia every solve, which the centroidal answer needs and the whole-body one ignores), and the reset targets write
`setBaseOrientation(x, headingQuaternion(getBaseOrientation(x)))` (D9). `predictedBaseAt()` follows the arc of a turning
command (`commandedPoseChange()`), the trapezoid bit for bit without a turn.

**Yaw range.** The state never stores an angle, so no wrap exists in the solver; the yaw residual is a wrapped relative
twist; references are re-seeded from the measured heading at every command and densified, so per-node yaw errors stay
far from π.

### 2.10 Euler tuning mapping (exact)

**The schemas name the Euler rows.** The orientation blocks of `cfg/state_weights.proto`, `cfg/acom_weights.proto` and
`cfg/state_values.proto` are `{yaw pitch roll}` rows whose comments say what each row means in the quaternion
formulation (`cfg/yaw_pitch_roll.proto` names the one conversion layer). GUI controls are labeled by field names, and no
comment of a file is parsed. The index map from the tuning layout to the state is section 2.3.

| Tuning item (schema field) | Converter | Consumer | Exactness |
|---|---|---|---|
| `state_weights.base_orientation {yaw pitch roll}`, `final_state_weights.base_orientation` | `trackingStateWeightsFromConfig()` / `terminalStateWeightsFromConfig()`: n_t × n_t Q | the `Tuning*` costs through `TuningLayoutDeviation`: rows = `e_HT` (section 2.8.1) | yaw: `wrap(ψ_h − ψ_h,r)`, shifted from the Euler difference by `wrap(o(θ, φ) − o(θ_r, φ_r))` (and by 2π where `ψ − ψ_r` leaves (−π, π]), weight 0 on every robot; pitch, roll: exact for single-axis tilt, mixed per section 2.8.1, independent of the yaw error; Hessians equal at any level state, first order in the tilt otherwise; value and gradient equal at a level state against a reference tilted about at most one axis |
| every other `state_weights` / `final_state_weights` block, `terminal_cost_scaling`, `input_weights` | the same, `inputWeightsFromConfig()` | the same rows, plain differences | identical |
| whole-body `state_weights.base_angular_velocity {yaw pitch roll}` | the same | rows (z, y, x) of `e_ω` (D8) | for a reference whose Euler rate is a world yaw rate (every shipped one) `e_ω = P₃T_B(Θ)(Θ̇ − Θ̇_r)` exactly: equal at a level state, within `\|tilt\|·‖Θ̇ − Θ̇_r‖` otherwise, zero for correct world-yaw-rate tracking at any tilt. At a level state, with `v = Θ̇ − Θ̇_r`, the orientation block is `[[0, 0, −v_θ], [0, 0, v_ψ], [0, −v_ψ, 0]]`, so the Gauss-Newton Hessian is Q only where the yaw and pitch rate errors vanish (the roll-rate error does not enter it), and the gradient is the Euler cost's only where `(0, −q_x v_ψ v_φ, (q_y − q_z) v_ψ v_θ)` vanishes; with G1-WB's weights (3 / 3 / 3) at rest against a reference turning at 0.5 rad/s, `JᵀQJ − Q` is +0.75 on the pitch and roll diagonal and ±1.5 between the pitch angle and the roll rate and between the roll angle and the pitch rate |
| `acom_weights {yaw pitch roll}` | `acomWeightsFromConfig()` 3×3 | `ComAndAcomTrackingCost`: `e_HT(ξ ⊗ Exp(Δθ(q_j)), ...)` | tilt rows first order; the yaw row also by the twist-against-ZYX identity of the ACoM attitude, O(\|tilt\|²), which Atlas's ACoM yaw weight (2500) sees; Jacobian first order (section 2.8.3) |
| `initial_state` (every block) | `stateValuesFromConfig()`: n_t vector | `model.stateFromTuning()` in both interfaces' `Create`, start-up only (`RELOAD_START_UP`) | exact rotation: rows (yaw, pitch, roll) become `ξ = q_z q_y q_x` |
| whole-body `initial_state.base_angular_velocity` | the same | **D18:** the body angular velocity in (z, y, x), values and weights alike | shipped zeros: identical; the equivalence chart applies `ω_B = T_B(Θ)Θ̇` itself (section 2.12) |
| `com_weights`, `task_space_foot_cost`, `model_settings.foot_constraint`, `contacts.*`, `swing_trajectory_config.swing_pitch_angle` | unchanged | unchanged residuals | identical |
| `locomotion_heuristics.{orientation_compensation, periodic_orientation}.*`, `maximum_tilt` | `locomotionHeuristicConfigFromConfig()` | the `shapeBasePose()` seam through the Euler accessors | exact (every `base_pose` list ships empty) |
| `reference_file` (yaw rates, `default_joint_state` by joint name), velocity and pose commands | `ReferenceFromConfig` | knots through `setBasePoseEulerZyx()` and `setBaseAngularVelocityFromWorldYawRate()` | exact for level targets |
| `contact_planning_file` heading terms | `ContactPlanningFromConfig` | heading reads through `getBaseHeading()`, or `twistHeading(acomAttitude(...))` with the ACoM | `\|o(θ, φ)\|` exactly, `o = −2 atan(tan(θ/2) tan(φ/2))`; `½\|θφ\|` is its leading term, not a bound |
| the reset heading | none | `headingQuaternion(ξ_obs)` (D9) | the twist instead of the ZYX yaw, by `\|o(θ, φ)\|` |

A task file reads, for example (the comments say what the rows mean):

```textproto
state_weights {
  base_orientation { yaw: 0.0 pitch: 5.0 roll: 85.0 }  # rows of the heading-tilt residual: yaw, then the tilt [rad]
  base_angular_velocity { yaw: 3.0 pitch: 3.0 roll: 3.0 }  # whole body: the body angular velocity error about z, y, x [rad/s]
}
```

### 2.11 Measurement, reset, policy, rollout

**Observation.** The MRT joint controllers build the observation through the robot model's realtime measurement
helpers, into buffers sized at construction (`updateMpcState()`): `fillGeneralizedCoordinatesFromMeasurement(p_W, ξ,
q_j, q)` writes `q = [p_W, ξ̂, q_j]` (the measured quaternion read through its safe normalization) and
`fillGeneralizedVelocitiesFromMeasurement(ξ, v_B, ω_B, q̇_j, v)` writes `v = [R(ξ̂)v_B, ω_B, q̇_j]`. The centroidal
controller then writes `h̄ = A_g(q)·v/m` and `q` into the state; the whole-body one writes `q` and `v` as they are. There
is no Euler conversion, no `cos θ` division, no sign alignment (D12) and no per-tick CHECK or LOG: a gross norm error is
handled by the safe normalization, never by an assertion on the realtime thread. A quaternion with no rotation at all
(at or below `kQuaternionNormalizationEpsilon`) is the one error the MPC side refuses: the SQP throws on it as an initial
state (section 2.7), so `MpcServer` publishes no policy from it and requests a reset, as for any failed solve.

**Reset targets** (`CentroidalMpcResetTarget`, `WBMpcResetTarget`): the measured attitude projected on its heading,
`setBaseOrientation(x, headingQuaternion(getBaseOrientation(x)))` (D9), the whole body's velocities zeroed over
`tail(nv)`. Gravity compensation reads the measured quaternion, with zero velocity sized nv.

**Policy evaluation.** Both MRT controllers keep their `std::shared_ptr<const StateManifold>` (from `stateManifoldOf()`)
as a member declared before the `RealtimePolicyEvaluator`, which they construct with `.get()`. `MRT_BASE::setStateManifold`
slerps the plan state; `InProcessMpcLink` gives its MRT the MPC problem's own manifold, so every in-process caller gets
it unasked. The whole-body inverse dynamics on the planned state read a slerped, unit plan.

**Rollouts and dummy simulators.** `RolloutBase::setStateManifold`; `TimeTriggeredRollout` projects every output state
and its `clone()` keeps the manifold; it is the fork's only rollout. The dummy
simulators (`*DummySimMain.cpp`, `app/node/src/DummySimLoop.cpp`) set `config.link.stateManifold`, so their MRT and
evaluator decode and execute the manifold policy and feed back projected states.

### 2.12 Exact identities for the equivalence tests

The chart `Φ` (`val/include/humanoid_mpc_validation/chart/EulerChart.h`): the identity on `h̄`, `p`, `q_j` (and `ṗ`,
`q̇_j`), `φ(Θ) = q_z(ψ)q_y(θ)q_x(φ)` on the angles, and on the whole body `ω_B = T_B(Θ)Θ̇`. `P_e(x)` is the
physical-tangent Jacobian in Euler coordinates (the `½GT_B` blocks), and `C(x_e)` the Euler-to-physical-tangent map:

```math
D\varphi(\Theta)=\tfrac12\,G(\varphi(\Theta))\,T_B(\Theta),\qquad
f_t(\Phi(x_e),u)=C(x_e)\,f_e(x_e,u),\qquad
J_e=J_t\,P_e(x_e),\qquad H_e=P_e^\top H_t P_e\ \ (\text{Gauss–Newton}),\qquad \partial f_t/\partial u=C\,\partial f_e/\partial u
```

```math
(\partial f_t/\partial\delta)\,C=C\,A_e+(\partial C/\partial x_e)[f_e]
```

These hold at any attitude with cos θ ≠ 0, not only for small errors. Three more relations (section 4.4):
- **Base-held torques:** `τ_e = τ_t + M_jw(q)·(d/dt T_B)·Θ̇` (section 2.5).
- **Friction-cone Hessian:** a fixed shift `c·I_a` of the ambient state, whose chart image `c·dΦᵀdΦ` is `c·I_e` but on
  the orientation's blocks, its angle block a quarter (section 2.8.5).
- **RK4 node map:** RK4 is coordinate-dependent at O(dt⁵): the Euler oracle's RK4 step and the SQP's manifold step agree
  through the chart to that order, and the coordinate dependence shrinks by at least 8 when dt is halved (third order or
  faster), where a difference of the flow maps would only halve.

---

## 3. Runtime: where it lives, the CppAD libraries, the bus, the display, live tuning

### 3.1 Where the formulation lives

| Package | What |
|---|---|
| `ocs2/core` | `StateManifold`, `EuclideanStateManifold`, `ProductStateManifold`, `StateManifoldSegment`, `UnitQuaternionMath.h` (`quaternionRateMatrix`, Exp, Log, Jr, Jr⁻¹, slerp, `quaternionSafeNormalize`); `ManifoldLinearController` (`getTimeStamps()`, `getAnchorStates()`, `getNominalInputs()`, `getFeedbackGains()`, `keepFirst()`); `ControllerType::MANIFOLD_LINEAR`; the `CppAdInterface` domain and range check and `getTapeOperationCount()` |
| `ocs2/oc`, `ocs2/sqp`, `ocs2/mpc` | `OptimalControlProblem::stateManifoldPtr`; `ManifoldProjection` (section 2.7); `StateTrajectoryInterpolation.h`; the manifold `toPrimalSolution()`; the SQP's initial gap, warm-start guard and retraction; `MRT_BASE::setStateManifold`; `RolloutBase::setStateManifold` |
| `cm/` | the quaternion root, `CentroidalModelInfo` with nq ≠ nv, `getBasePosition` / `getBaseQuaternion`, `computeConfigurationDerivative()`, the safely normalized mapping, the single-rigid-body port |
| `common/orientation` | `BaseOrientation` and `EulerBoundary` (Bazel `:orientation`), `TuningLayoutDeviation` (in `:humanoid_common_mpc`) |
| `common/common`, `common/config/weights` | `MpcRobotModelBase` and the decorators, `StateManifoldFactory` (`stateManifoldOf()`), `StateLayout.h` (`kStateLayoutTag`, `:state_layout`), `Types.h` (`kBaseConfigurationDim = 7`, `kBaseVelocityDim = 6`, `getBaseDofNames()`, `getBaseVelocityDofNames()`); `StateInputLayout` (`tuningLayoutOf()`, `tuningDimension()`), `StateInputWeightsFromConfig` |
| `common/cost`, `common/acom`, `common/parameter_update` | the `Tuning*` costs and `TuningGaussNewton`; `AngularCenterOfMass`, `ComAndAcomTrackingCost`; the shared updater and appliers (section 3.5) |
| `common/command`, `common/reference_manager`, `common/contact_planning` | the knots, the stride-mean measured motion and the heading reads of section 2.9 |
| `cmpc/`, `wbmpc/` | the robot models, the flow maps, both interfaces' manifold wiring, the MRT controllers and reset targets |
| `ipc/`, `msgs/` | the manifold policy's wire format, its decoder and the realtime evaluator (section 3.3) |
| `app/visualization`, `rr/` | the Euler display (section 3.4) |
| `val/`, `common/test/equivalence` | the Euler chart, the closed-loop harness and the benchmark; the equivalence suites and their Euler oracle (sections 4 and 5) |

### 3.2 CppAD libraries

- **The tag.** `kStateLayoutTag = "translation_spherical_quaternion_v1"` (`common/common/StateLayout.h`, Bazel
  `:state_layout`) names the layout the libraries are taped for: the root `Composite(Translation, Spherical)`, the base
  orientation stored as a unit quaternion (x, y, z, w), the base angular velocity in the body frame. It changes whenever
  the layout of the state, the input or the Pinocchio configuration a library reads changes.
- **The folder.** `ModelSettings` builds `modelFolderCppAd = cppad_code_gen/cppad_<mpc><robot>/<kStateLayoutTag>`
  (`ModelSettingsFromConfig.cpp`, the `cppad_model_folder` / `cppad_layout_tag` IFTTT pair); the centroidal MPC's
  contact-input key nests below it (`CentroidalMpcInterface::keyCppAdModelFolder()`: `wrench_inputs` or
  `basis<n_b>_<hash>`). Libraries of an untagged layout sit one folder up and are never read.
- **The refusal.** On load `CppAdInterface` checks the library's `Domain()` against its variables plus parameters and
  its `Range()` against the expected output size (from an earlier `createModels()`, the copied interface, or one
  off-tape evaluation at the tape point), and throws naming the folder: "It is stale (for example from before a change
  of the state layout): delete the folder, or set the task file's model_settings.recompile_libraries_cpp_ad: true, to
  regenerate it." (`test_cppad_library_dimension_check`). The interfaces' set-up converts it to an InvalidArgument
  Status at their one exception boundary (`catchingModelBuilderExceptions()`, `catchingLibraryExceptions()`). A library with
  the same dimensions and other arithmetic cannot be told apart; that is what the tag is for.
- **Regenerating.** Every shipped task file sets `model_settings.recompile_libraries_cpp_ad: false`, so a library is
  generated once, on first use, and reused while its `.so` and its generator stamp (`kCppAdGeneratorTag`,
  `test_cppad_generator_stamp`) match. The first start of each robot compiles all its libraries into
  the tagged folder (minutes; one robot at a time, AGENTS.md); to force it, delete the folder or set the field to `true`
  for one start. A library that is added or dropped needs no new tag: the whole body's `<foot>_pose_wrt_plane_rate` and
  `<foot>_pose_wrt_plane_acceleration` (section 2.8.6) are generated on first use into a folder whose other libraries
  load as they are, their generated sources byte for byte those of before, and `<foot>_twist` and
  `<foot>_accelerations`, which they replaced, are no longer generated, nor are `<foot>_orientation`,
  `<foot>_orientationError`, `<foot>_angular_velocity` and `<foot>_angular_acceleration`, which no term evaluated (a
  folder generated before keeps their files, unread). Per foot of the G1 the tape operations of the two (502 and
  8,624) exceed those of the two they replaced (433 and 8,508) by 185; the real-time gate's total (section 4.6) counts
  the libraries an MPC builds, so it drops by the operations of the four that are no longer generated. The centroidal
  MPC's `PinocchioEndEffectorKinematicsCppAd` (the contact frames' and the task-space links') no longer generates
  `<frame>_orientationError`, which no term evaluated either: per robot 28 libraries instead of 31 (Atlas, G1, R1) and
  21 instead of 23 (SA01), the other ones' generated sources byte for byte those of before. Code generation is
  deterministic (`test_cppad_codegen_determinism`), and a copy of a term tapes itself, never the term it was copied from
  (`test_cppad_term_copies`, lib/ocs2/README.md). `testCppAdLayoutTag` pins that
  every shipped configuration's folder ends in the tag; `testWBQuaternionDynamics` and the invariance tests tape the
  production flow maps and terms on the quaternion root.

### 3.3 Bus and runtime

- **Wire format** (`msgs/controller_type.proto`, `msgs/mpc_policy.proto` field `controller_data`, the `controller_data` /
  `manifold_node_writer` IFTTT pair): `CONTROLLER_TYPE_MANIFOLD_LINEAR = 3`. A `ManifoldLinearController` travels only as
  its own node arrays, node i = `[u*_i (nu), x̄_i (nx), vec K_i (nu × ndx, column-major)]`, from `getNominalInputs()`,
  `getAnchorStates()` and `getFeedbackGains()`, `nu + nx + nu·ndx` doubles per node.
- **Why the anchors travel.** They are not `state_trajectory`: `toPrimalSolution()` gives every pre-event node the
  anchor, input and gain of the node before it, and the last node those of the one before it, while the state trajectory
  keeps each node's own state. The encoder (`policyToProto()`) refuses a manifold controller whose time stamps are not
  the time trajectory (the SQP always builds it there); the encoder needs no manifold.
- **Decoding** (`policyFromProto(message, stateManifold, ...)`, on the robot's communication thread): the receiver's
  manifold gives ndx and must have the states' ambient dimension. A receiver on a manifold refuses
  `CONTROLLER_TYPE_LINEAR`, whose gains would act on the stored state; a flat receiver refuses
  `CONTROLLER_TYPE_MANIFOLD_LINEAR`. `RemoteMpcLink::Config::stateManifold` sets the manifold on the link's MRT, its
  decoder and its evaluator, and `Create` checks the ambient size. The RobotMains (`CentroidalMpcRobotMain.cpp`,
  `WBMpcRobotMain.cpp`, the `robot_mpc_link` IFTTT pair) set it from `stateManifoldOf(model)`.
- **The realtime path** (`RealtimePolicyEvaluator`, `ipc/`): it computes what `MRT_BASE::evaluatePolicy()` does, the same
  doubles, with `differenceInto()` / `interpolateInto()` and the buffers `gain_` (nu × ndx), `delta_` (ndx), `nodeInput_`
  (nu) and `planState_` (nx) allocated once in its constructor. A policy of a type it does not evaluate on its state - a
  LINEAR one with a manifold, a MANIFOLD_LINEAR one without - is `kUnsupportedController`, never left to the allocating
  `MRT_BASE` fallback. Nothing on the realtime thread allocates, logs, locks, throws or CHECKs per tick: the evaluator
  compares sizes and returns an outcome, and nothing aborts (`test_realtime_policy_evaluator`,
  `test_remote_mpc_link_allocations`, `test_mpc_message_conversions_allocations`,
  `test_centroidal_mpc_mrt_joint_controller_allocations`, `testWBMpcMrtJointControllerAllocations`,
  `test_robot_realtime_allocations`).
- **Bandwidth.** About 2 % more than a linear policy of an Euler state of the same robot: one more state entry per node
  and the anchor (`docs/distributed_runtime/README.md`, "Bandwidth").
- **End to end.** `test_centroidal_mpc_manifold_policy_over_bus` solves the shipped Atlas MPC (feedback on) over the bus
  and in process: the decoded policy matches the in-process one node for node, and the robot's evaluator matches
  `MRT_BASE::evaluatePolicy()` bit for bit at 0.001 s, around the contact event at 0.3 s and at mid-horizon. The node
  tests assert `CONTROLLER_TYPE_MANIFOLD_LINEAR` on Atlas and `CONTROLLER_TYPE_FEEDFORWARD` on the whole-body G1.

### 3.4 Display

The display requirement of section 1.3, implemented in `app/visualization` (`TelemetryBuilder`, `RobotStateDecoder`) on the
MPC node's visualization thread, never in the robot process, into preallocated or fixed-size buffers:

| Series (`rr/` telemetry contract) | Value |
|---|---|
| `base_pose/{roll, pitch, yaw}` | the ZYX Euler angles of the quaternion (`eulerZyxFromQuaternion()`, yaw and roll in (−π, π], pitch clamped), measured and reference |
| `base_pose/{roll_rate, pitch_rate, yaw_rate}` | the Euler rates `T_B(Θ)⁻¹ω_B` (`eulerZyxRatesForDisplay()`), all three NaN, a gap in the plot, where `\|cos(pitch)\| < kDisplayEulerRateMinCosPitch = 1e-6` or the pitch is NaN; the reference's 0 where the target carries no velocity |
| `base_pose/angular_velocity_{x, y, z}` | the body angular velocity `ω_B`: measured, the sample's `base_angular_velocity_local`; reference, `v[3:6]` of the reference state |
| `base_twist/angular_{x, y, z}` | the world-frame angular velocity `R(ξ)ω_B`, its panels titled "(world)" |
| `generalized_base/{position_z, pitch, roll}`, `dofs/position/*` | a display buffer `[p, (yaw, pitch, roll), q_j]` (`getBaseDofNames()`) in place of Pinocchio's q |
| `generalized_base/{pitch_rate, roll_rate}` | the same values as the Euler-rate rows |
| `dofs/velocity/*`, `dofs/force/*` | Pinocchio's v under `getBaseVelocityDofNames()` (`base_vx, base_vy, base_vz` world, `base_wx, base_wy, base_wz` body) |
| `mpc_observation/state` | generic `x0..x{nx−1}`, the quaternion's four entries |

The Euler-rate rows (row 3) and the body angular velocity (row 4) are in the "Base Pose & Euler" tab, the tab the
default `rr/config/plot_config.textproto` selects, so it shows the angles, the Euler rates and the body rates without an
edit. The groups are the `telemetry_groups` (`TelemetryBuilder.cpp`) ↔ `panel_groups`, `complete_groups`, `panel_tabs`
(`rr/python/humanoid_rerun_viewer/telemetry_contract.py`) pairs; `rr/README.md` lists every series ("Telemetry"), and
`app/visualization/README.md` the builder. `VisualizationModel` refuses a robot model whose nq is not `7 + nj` or whose
nv is not `6 + nj`, and the nodes give it the interface's manifold for `PolicySnapshot`. Tests: `test_robot_state_decoder`
(a level turn; tilted bases, where the rates equal the world-angular-velocity route to 1e-12 and a central
difference of the Euler angles of `ξ ⊗ Exp(tω_B)` to 1e-8; `cos(pitch)` from 1e-2 down to 1e-5; NaN at ±90° and at
`cos = 5e-7`, finite at 2e-6), `test_telemetry_builder`, `test_contracts`, `test_plot_config`, `test_bridge`.

### 3.5 Live tuning through the one layer

The shared parameter updater (`MpcParameterUpdaterModule`, `common/parameter_update/`) runs before every solve on the
solver thread. Its appliers are listed per formulation as `HotFieldApplierEntry` arrays: `kCentroidalAppliers` (12,
`cmpc/src/mrt/CentroidalMpcParameterUpdater.cpp`) and `kWholeBodyAppliers` (7,
`wbmpc/src/parameter_update/WholeBodyHotFieldAppliers.cpp`). The hot fields that carry orientation:

| Field | Applier | Path |
|---|---|---|
| `state_weights`, `final_state_weights`, `terminal_cost_scaling` (both formulations: `base_orientation`, and on the whole body `base_angular_velocity`) | `QuadraticCostWeightsApplier` | `trackingStateWeightsFromConfig()` / `terminalStateWeightsFromConfig()`, the start-up converters, then `TuningQuadraticStateInputCost::setStateWeights()` / `TuningQuadraticStateCost::setStateWeights()` through `OcpTermUpdates::updateTerm()`; the costs' `TuningLayoutDeviation` maps the Euler rows. Under the CoM + ACoM cost both paths zero the base-pose block (`BasePoseWeights::kZeroedForComAndAcomTracking`) |
| `acom_weights` (centroidal) | `ComAndAcomWeightsApplier` | `acomWeightsFromConfig()` → `ComAndAcomTrackingCost::setWeights()`, as at start-up |
| `locomotion_heuristics.*` (centroidal) | `LocomotionHeuristicsApplier` | the Euler seam `shapeBasePose()` |
| `swing_trajectory_config.swing_pitch_angle` | `ReferenceManagerApplier` | unchanged residual |

`initial_state` is `RELOAD_START_UP`: the updater warns and never applies it; it reaches the solver only through
`stateFromTuning()` at start-up. `HotUpdateTarget::tuningDim()` is `tuningDimension()` of the layout, and the size check
compares it with the running problem's tangent dimension. No applier, updater, GUI or Python tool converts Euler angles
(`euler-boundary` covers `parameter_update/`). **Live update == fresh start on the quaternion layout:**
`testLiveUpdateEqualsFreshStart` (`cmpc/`, 8 cases: `WithEveryAcomRowWeighedOnTheShippedAtlas` with the multiplicative
ACoM, `WithEveryBaseOrientationRowWeighedOnTheAtlasWithoutTheAcomCost`) and `test_wb_live_tuning` (`wbmpc/`,
`AnUpdateOfEveryOrientationRowSolvesAsAFreshStartOnTheEditedFile`): every case weighs every orientation row nonzero
(the shipped yaw weight is 0, which a scaled leaf would keep at 0), checks that the problem and every solver worker are
on a manifold with ambient = tangent + 1, and requires the reloaded MPC to solve as the fresh one with bitwise-equal
weights in the running costs (all 24 whole-body `policy_difference` values 0; every planned quaternion unit to 1.1e-16).

---

## 4. What stays Euler, and how the formulation is verified

### 4.1 What the formulation does not have

- No `SphericalZYX` root (`common/src/pinocchio_model/createPinocchioModel.cpp`, `cm/src/FactoryFunctions.cpp`):
  `spherical-zyx` keeps it to the equivalence suites' oracle.
- No Euler or Euler-rate state slot and no accessor of one, no whole-body Euler-rate API, and no `nq == nv` shortcut:
  every consumer sizes its configuration nq and its velocity nv.
- No `q̇ = v` in a flow map, no ZYX gradient, and no analytic Euler code in `cm/`.
- No Euler conversion in the MRTs; the telemetry's Euler angles and rates are display values (section 3.4).
- No additive ACoM rule: the ACoM attitude is the composition of section 2.8.3, in C++ and in its Python twin.
- No raw `x − x_ref` of a stored state in the quadratic state costs: they evaluate the tuning deviation (section 2.8.2).

`RotationDerivativesTransforms.h` stays in `ocs2/robotic_tools` with no humanoid production caller outside
`EulerBoundary`.

### 4.2 Stays Euler: the conversion points

All conversions go through `EulerBoundary` or the robot models' `*EulerZyx` accessors, and nowhere else. The
`euler-boundary` check (`tools/hooks/euler_boundary.py`) keeps the Euler API out of the formulation directories and out
of the robot-model headers but at their marked boundary accessors:
<!-- LINT.IfChange(euler_boundary) -->
- **Formulation directories** (`FORMULATION_DIRS`, below a package's `src/` or `include/<package>/` in
  `humanoid_nmpc/`): `dynamics`, `cost`, `constraint`, `end_effector`, `acom`, `parameter_update`, and the helpers the
  CppAD tapes are recorded through, `pinocchio_model` (the kinematics and dynamics helpers of the flow maps and the
  wrench-cone, contact-moment and DCM terms) and `contact` (the foot corner heights). No Euler API there.
- **Formulation files and trees** (`FORMULATION_FILES`, `FORMULATION_TREES`): `HumanoidPreComputation.{h,cpp}` and its
  whole-body subclass `WBMpcPreComputation.{h,cpp}` (the kinematics and swing-foot constraint configs the constraints
  read), the heading-tilt residual `orientation/BaseOrientation.{h,cpp}`, the end-effector kinematics the centroidal
  MPC tapes, `lib/ocs2/pinocchio/pinocchio_interface/`'s `PinocchioEndEffectorKinematicsCppAd.{h,cpp}` (that tree's
  other files are not taped and not read), and `lib/ocs2/pinocchio/centroidal_model/`, whose flow map, mapping and
  kinematics the centroidal MPC tapes (the check reads `lib/ocs2` for these files and that tree).
- **Robot-model headers** (`ROBOT_MODEL_HEADERS`, in scope by name): `MpcRobotModelBase.h`,
  `BasisInputsModelDecorator.h`, `BasisInputsMappingDecorator.h`, `CentroidalMpcRobotModel.h`,
  `WBAccelMpcRobotModel.h`. Their boundary lines carry `NOLINT(euler-boundary): Euler boundary accessor (design section
  4.2)` where they name the Euler API.

The conversion points:
1. the robot models' boundary accessors (`MpcRobotModelBase`, `CentroidalMpcRobotModel`, `WBAccelMpcRobotModel` and
   the decorators that forward them): `get/setBasePoseEulerZyx`, `get/setBaseOrientationEulerZyx`, `stateFromTuning` /
   `tuningFromState` and `setBaseAngularVelocityFromWorldYawRate`;
2. the tuning layout, the one mapping of the Euler-written weights and initial state into the quaternion layout:
   `EulerBoundary`'s `TuningLayout` (`stateFromTuningLayout` / `tuningLayoutFromState`), its construction
   (`tuningLayoutOf()`, `StateInputLayout`), `TuningLayoutDeviation` (section 2.8.2) and `StateManifoldFactory`, which
   reads the placement;
3. the `initial_state` conversion (`stateValuesFromConfig()` through `stateFromTuning`, in both interfaces' `Create`);
4. `TargetTrajectoriesCalculatorBase` and the knot assembly of both calculators, pose commands included;
5. the heuristic seam `shapeBasePose()`;
6. `PlannedHeadingOverride`;
7. logs (the whole-body MRT's reset-target log line), and the display: `RobotStateDecoder` and `TelemetryBuilder`
   (Euler angles and the display-only Euler rates, section 3.4);
8. the Euler chart (`val/chart/EulerChart`) and the equivalence suites' Euler side (`common/test/equivalence`:
   `EulerMpcOracle`, `EulerChartStateManifold`, `RootJointModelPair`);
9. the schemas' named orientation blocks and the GUI;
10. the Python readers of the task file (`tools/locomotion_heuristics/derive_parameters.py`,
    `humanoid_learning/acom/tests/test_acom.py`), which read the blocks by name.
<!-- LINT.ThenChange(//tools/hooks/euler_boundary.py:euler_boundary) -->

### 4.3 How the formulation is verified

On its mathematics, by tests that build everything from the shipped configuration files and read no recorded data of
another formulation (section 5):
- **against an independent Euler formulation:** the root-joint suite holds the production root's kinematics,
  centroidal quantities and dynamics to Pinocchio's own `Composite(Translation, SphericalZYX)` root through the chart of
  section 2.12, and the MPC-level suite holds each shipped MPC's flow maps, shooting intervals, rollouts, orientation
  costs and real-time SQP iteration to an Euler oracle written from the definitions on that root (section 5(b));
- **against itself:** every term of both MPCs and the flow map against finite differences of their double-precision
  values on the quaternion root (`testQuaternionRootTermDerivatives_<robot>`); every cost's and soft constraint's
  Hessian, pulled back as the solver pulls it back, against the Gauss-Newton or penalty model `J'φ''J` its class
  prescribes, built from its verified first order (with the friction cone's `c·I_a` shift), and against its exact
  Hessian minus the curvature that model drops (`testQuaternionRootTermHessians.cpp`, the same binaries); the radial,
  double-cover and gimbal-lock invariances of every term (`testQuaternionInvariances`, `testWBQuaternionInvariances`),
  the production tapes, live update equal to a fresh start (section 3.5), the manifold policy over the bus (section
  3.3) and the allocation tests of the realtime path.

A closed-loop run or a solve benchmark is held to a baseline label recorded under `val/data/` by the rules of sections
4.5 and 4.6.

### 4.4 Equivalence classes and tolerances

| Class / check | Relation | Tolerance | Reason |
|---|---|---|---|
| E0: InputQuadratic, ZeroWrench, BasisScalingNonNegativity, JointLimits, JointMimic | values and Jacobians up to index relabeling | bitwise, or 1e-12 where indices move | no orientation. The Euler terms no longer exist, so these classes are not compared term by term: they are held by the term derivative suite and the root-joint suite (section 5) |
| E1: every kinematic or dynamic term (sections 2.8.4, 2.8.5), inverse dynamics, forward kinematics | value at `Φ(x_e)`; `J_e = J_t·P_e`; `H_e = P_eᵀH_tP_e` | values 1e-9 relative (floor 1e-10); Jacobians 1e-8 relative (floor 1e-9) | round-off, CppADCG `-ffast-math` reassociation, the conversion. With no Euler terms left these are not compared term by term; the root-joint suite holds the kinematics, the centroidal map and RNEA through the chart, `testMrtTorqueChartEquivalence` the base-held torques, `testQuaternionRootTermDerivatives` every term's first order, and `H_e = P_eᵀH_tP_e` is the Gauss-Newton identity |
| Every term and the flow map on the quaternion root | reported derivatives, pulled back into the tangent, against Richardson-extrapolated central differences along the retraction and the inputs; the input Hessians of a `ConstraintOrder::Quadratic` constraint against differences of its input Jacobian | 1e-6 relative to max(1, the largest entry) | the extrapolation's O(h⁴) truncation and 3ε\|f\|/h round-off at h = 1e-4, and `-ffast-math` |
| Every cost's and soft constraint's Hessian on the quaternion root | pulled back as the solver pulls it back, against `J̃ᵀφ''J̃ + Σp'H_c` of the term's verified inner Jacobian (`testQuaternionRootTermHessians.cpp`); against the exact Hessian (differences of the verified gradient) minus the dropped curvature `Σφ'_i∇²v_i` | assembly: 1e-12 of the magnitude of the summands; differences: the Richardson estimate \|R(h/2) − R(h)\| plus 6δ/h, δ = 1e-12 of each row's magnitude (floor 1); each comparison rejects one entry moved by 1e-6 | the round-off of two assemblies of the same products; the extrapolation's truncation and round-off |
| Friction-cone state Hessian | not a Gauss-Newton term: a fixed shift `c·I_a` of the ambient state, `c = −s·Σp'(h) ≥ 0`, whose pull-back `c·EᵀE` is `c` on every tangent row but `c/4` on the rotation's three, and whose chart image `c·dΦᵀdΦ` is `c·I_e` but on the orientation's blocks (the angles' `(c/4)·T_BᵀT_B`; on the whole body also `c·T_BᵀT_B` on the rates and their coupling with the angles) | 1e-12 of the summands' magnitude (`testQuaternionRootTermHessians.cpp`) | a fixed ambient diagonal shift (section 2.8.5); no other term has an exact orientation Hessian |
| Twist constraint's state Jacobian | at every point, the reported `J·E` plus the dropped term `D` (`M` differenced along the retraction, times the twist's own `ω`) against the differences of the value, at rest (`ω = 0`, `D = 0`) `J·E` alone; the tilt rows of `M·ω` against `(∂e/∂x)·f_a` | as every term's; and entry by entry within both differences' Richardson estimates plus 6δ/h at h = 1e-3 (δ = 1e-12 of each row's magnitude, floor 1) plus 1e-12 of the Jacobian row's magnitude, which must reject `D` with its largest entry moved by 1e-6 of max(1, itself) (it does by a factor of 12 at least; the residual is under a thousandth of the bound); the rate: 1e-12 of the summands' magnitude | its rate map is held at the point (section 2.8.5): the reported Jacobian is the derivative less exactly `D`, which vanishes where the foot does not turn |
| Whole-body stance rates (`testOrientationErrorWrtPlaneRates`, `testStanceFootOrientationRates`) | the taped `ė` and `ë` (section 2.8.6) against Richardson differences in `t` of the production `e(R(t))`, `R(t) = Exp(ωt + αt²/2)·R₀`, and of `½·v_n(t)·ω(t)`, `ω(t) = J_l(φ)·φ̇`, at level attitudes, tilts to 80°, steep ones to `1 + c` = 1e-2 and `1 + c` from 1e-2 to 1e-5, and there the taped Jacobian against differences, entry by entry (from `1 + c` = 1e-5 to 1.1e-6 the tape against its double instantiation and a finite Jacobian; inside the guard the fallback, its Jacobian); the G1 constraint against `K_p·e + K_d·ė + K_a·ë` and its linear rows assembled from double-precision Pinocchio, whose `LOCAL_WORLD_ALIGNED` rates are the derivatives along `q ⊕ (v·t + a·t²/2)`; a level foot against the former law with half the angular gains; the Jacobian along the manifold, entry by entry | rates 1e-8 of `max(1, r)` and `max(1, r²)`, `r = max(‖ω‖, √(ψ‖α‖))/ψ`, `ψ = min(1, s)`: the oracle, extrapolated twice from `h = 4e-2/max(1, r)`, errs by at most 6e-12 at a tilt, 3e-10 between 80° and `1 + c` = 1e-2 and 7e-10 near the anti-parallel normal (it measures 3.6e-10 there); the taped Jacobian within 1e-7 of the smaller of its block's natural scale (`r^k/ψ²` by the attitude, `1/ψ` and `r/ψ` by `ω`, `1/ψ` by `α`) and `max(1, ‖J‖_max)`, at most 0.07 of it, which rejects a 1e-6 change of the largest entry of the maps (the rate's by `ω`, the acceleration's by `α`) by a factor of 4 at least; the level foot's former law but for `(K_a/2)·(n·ω)(n × ω)`, 1e-11 of the angular rows' scale; the constraint's Jacobian within the differences' bound (as the twist constraint's), which rejects a 1e-6 change of its largest entry and of the orientation rows' largest (by a factor of 249 at least) and the frozen-map form's state Jacobian (by 1.6e8), whose input Jacobian is the exact one | the differences' O(h⁶) and O(h⁴) truncation and the round-off `ε/ψ` of `e` near the anti-parallel normal; the libraries' round-off |
| Dynamics value and input Jacobian | `f_t = C·f_e`, `∂f_t/∂u = C·∂f_e/∂u` | momentum rows 1e-10; the other rows 1e-9; each block of rows (momenta, base position, orientation, joints, velocities) relative to its own largest entry, floor 1 | same physics |
| Dynamics state Jacobian | section 2.12 with `∂C/∂x` by central differences | the Richardson error bound of the differences (`h` = 1e-4, their round-off included) plus 1e-8 of the largest entry (floor 1) | difference-quotient error |
| RK4 node map | the SQP's manifold step against the Euler oracle's RK4 map through the chart, `A_t·P_e(x_k) = P_e(x_{k+1})·A_e` | twice (`kStepHalvingSafety`) each map's distance from its 4-substep (quarter-step) interval times 256/255, the estimate of its local RK4 error, plus the finite-difference error and 1e-8; rollouts: the coordinate dependence divided by at least 8 when dt halves | RK4 is coordinate-dependent at O(dt⁵), above 1e-6 at fast points (up to 1.6e-2 on G1-WB) |
| Base-held joint torques | `τ_e = τ_t + M_jw·(d/dt T_B)·Θ̇` | 1e-9 relative | "held" is `Θ̈ = 0` on the Euler root, `ω̇_B = 0` on the quaternion root (section 2.5) |
| E2: Q / Q_final orientation rows (the heading-tilt residual against the Euler difference of the same rows) | value | yaw `\|wrap(o(θ, φ) − o(θ_r, φ_r))\|` + 1e-12, the exact identity (its leading term ½\|θφ − θ_rφ_r\| is no bound: at θ = φ = 0.05, \|o\| = 1.25052e-3); tilt ≤ 3e-5 rad at \|tilt\| ≤ 0.05, scaled by (\|tilt\|/0.05)³ above; cost value relative ≤ 2e-3 at \|tilt\| ≤ 0.05 rad and ≤ 2.5e-2 at ≤ 0.2 rad **against a level reference with the yaw row weighted 0** and pitch / roll weights within a ratio of 1e3 (every shipped robot): a yaw weight breaks it at first order in the tilt (2.8e-3 at 0.05 with a yaw weight of 1 against SA01's 5 / 85, 3.7e-2 at 0.2 with 10), a tilted reference too (3.7e-3 at 0.05), and a roll-only weighting reaches 2.66e-2 at 0.2 | twist against ZYX yaw; cubic tilt terms (section 2.8.1) |
| E2: same rows at a level **state** (any reference); value and gradient against a reference tilted about at most one axis | through `P_e` | 1e-12 | `J_HT·T_B(0) = I` |
| E2: same rows, gradient and Hessian at a tilted state | through `P_e` | `‖J_eᵀQJ_e − Q‖_max ≤ \|tilt\|·‖Q‖_max` (and `‖J_e − I‖_max ≤ 0.6\|tilt\|`) | `J_HT(φ(Θ))T_B(Θ) = I + O(\|tilt\|)`, coefficient ½ |
| E2: whole-body ω rows | value | world-yaw-rate reference: `‖e_ω − (Θ̇ − Θ̇_r)‖ ≤ \|tilt\|·‖Θ̇ − Θ̇_r‖` + 1e-12 (`e_ω = P₃T_B(Θ)(Θ̇ − Θ̇_r)` exactly); any reference: `\|tilt\|·‖Θ̇‖ + (\|tilt_r\| + √3·(\|Δψ\| + \|Δθ\| + \|Δφ\|))·‖Θ̇_r‖` per row; a bound in ‖Θ̇‖ alone fails (0.0499 at a pitch of 0.1 at rest against a level reference turning at 0.5 rad/s) | `T_B` against `P₃`; the reference rate transported |
| E2: ComAndAcom | value, Jacobian | rows: the base attitudes' own heading-tilt change plus `c·(\|Δθ\|·\|tilt\| + \|Δθ\|²)` for the state and the reference; Jacobian `c·(\|tilt\| + \|Δθ\|)` per entry; c = 1 + max \|Δθ\| + max \|tilt\| (`testComAndAcomTrackingCost` `ComAndAcomEulerTolerance`, at \|tilt\|, \|Δθ\| ≤ 0.2: the rows reach 0.75 of the bound and the Jacobian 0.96 over its draws (a denser probe: 0.81 and 0.98)) | multiplicative against additive composition; Atlas's ACoM yaw weight is 2500 |
| E2: heading reads, reset targets, planner heading | twist against ZYX | `\|wrap(ψ_h − ψ)\| = \|o(θ, φ)\|` to 1e-12; the D9 reset target to `\|o\|`; the ACoM planner heading to `\|o\| + c(\|Δθ\|\|tilt\| + \|Δθ\|²)` (`ComAndAcomEulerTolerance.ThePlannerHeadingStaysWithinTheStatedBound`) | D9 |
| SQP step, tier 1 (S1: the walking reference, a resting warm start whose input exerts no moment on the base, level or tilted) | the quaternion manifold's real-time iteration (dx, du, K0) against the same iteration on the Euler chart (`EulerChartStateManifold`, `d = P_e·dx_e`) | 1e-10 relative (the largest recorded, 3.7e-12 on G1-WB tilted) | without rotation (and, on the whole body, without angular velocity) in the shooting gaps the two QPs are one up to round-off: both integrate the same RK4 flow, and the chart maps gap-free sensitivities by the identity. RK4's O(dt⁵) coordinate dependence separates the chart SQP from a true Euler-RK4 formulation, which the RK4 node map and rollout rows bound; it is not in this comparison |
| SQP step, tier 2 (S1: an initial state off the warm start by an attitude ε) | `Log(ξ_w⁻¹ξ₀)` against `P_e(x₀,e − x_warm,e)` | ratio between ε and ε/2 in [3, 5], or both within tier 1 | O(ε²) |
| SQP step with gaps that carry a rotation b or, on the whole body, an angular velocity (S2, S1w) | the two charts' iterations | their difference falls by a factor of at least 1.5 when b (or the angular-velocity kick) halves, and vanishes with it | the gaps' sensitivity, `Jr⁻¹(b)` against `T_B(Θ_next)T_B(Θ_F)⁻¹`, differs at first order; on the whole body the chart's sensitivity has the coupling block `C(Θ, Θ̇_next − Θ̇_F)·T_B⁻¹` |
| SQP fixed points (S3) | the two charts' gap linearizations at the resting warm start, at the quaternion SQP's iterate from it, and along that iterate with its gaps scaled to none (the SQP need not converge, and on Atlas and the whole-body G1 it cycles, on Atlas the same cycle in both charts (`MpcSqpConvergence.OnAtlasTheEulerChartCyclesAsTheQuaternionManifoldDoes`): its QP has the costs' Gauss-Newton Hessian without the hard constraints' curvature and its line search accepts any merit decrease while the violation stays below `g_max`; section 6, risk 12) | at most 100 times the largest gap, round-off without gaps | the same discrete problem in two charts of its defects: the same fixed points |
| Reference knots, node grid | knots written by `setBasePoseEulerZyx()` read back by `getBasePoseEulerZyx()` (yaw mod 2π) | 1e-12 | exact conversion; slerp about z between level knots is linear in yaw, so a subdivided pose command's node grid is the two-knot linear interpolation's (`testPoseCommandSubdivision`) |

### 4.5 Closed-loop metrics and bands

**Scenarios** (all five configurations; the lockstep runner of `val/`, section 5(d)): standing 10 s; walking at 0.5 m/s
for 15 s (0.3 m/s for SA01 and R1, `RobotConfiguration::walkingScenario`); lateral 0.2 m/s; turning in place at
+0.5 rad/s through 720° and back; turning at 1 rad/s for 12.6 s; an arc at 0.3 m/s and 0.3 rad/s.

**Metrics** (`ClosedLoopMetrics`): survival; base-height mean, standard deviation and RMS error; tilt RMS and maximum
(|τ|); velocity RMS error in the heading frame; yaw-rate RMS error; cumulative unwrapped heading; stance-foot slip; joint
torque RMS; the largest rotation part of `δx₀` per solve (`initial_state_gap.max_rotation_rad`); the largest
`|‖ξ_k‖ − 1|` over all nodes and solves (`quaternion_norm.max_deviation`); NaN count; solve-time p50 / p99 per phase;
the SQP's step per solve (`sqp_step`: full, reduced and no steps, full steps refused above `g_max`, the rule that
accepted each, the longest run of solves with a growing violation, and the statistics of the violation and of the
swing-foot normal-velocity residual after the step; section 6, risk 12); failures and resets.

**Bands** (`closedLoopMetricBands()`, `MetricBands.h`, the `metric_bands` IFTTT pair with `val/README.md`): every error
metric is `BandDirection::kNoWorse` - it fails only when worse than the baseline by more than its band, and a candidate
better by more than the band is reported as an improvement, not a violation - and the base height's mean is
`kTwoSided`:

| Metric | Band |
|---|---|
| survival | equal or better |
| height | mean within 5 mm either way; RMS error no more than 5 mm worse |
| tilt (RMS, maximum) | no more than max(20 %, 0.005 rad) worse |
| velocity | no more than max(15 %, 0.02 m/s) worse |
| yaw rate | no more than max(15 %, 0.02 rad/s) worse |
| slip | no more than max(20 %, 5 mm) worse |
| p99 solve time | at most +10 % |
| quaternion norm | `max \|‖ξ‖ − 1\|` < 1e-9 |

### 4.6 Real-time gate

`benchmark_mpc_solve` (`testonly` `cc_binary`, both formulations) replays the walking states of
`val/data/benchmark/states/` with the configured `nThreads`, one robot at a time; `compareSolveBenchmarks()`
(`SolveBenchmarkGate.h`, `make benchmark-mpc-solve ROBOT=<robot> LABEL=<label> BASELINE=<label>`) accepts if:
- mean solve time ≤ +5 %;
- p99 ≤ +10 %;
- LQ-approximation phase ≤ +8 %;
- each robot keeps p99 below 80 % of its MPC period (20 / 12.5 / 16.7 ms);
- every CppAD library's tape operation count, and their total, ≤ +5 % (machine-independent).

**Toolchain requirement.** The timing comparisons hold only between documents of one machine and one toolchain
(`provenance.machine`, `provenance.toolchain`: the compiler, the HPIPM build target - `GENERIC`, from
`bazel/system_libs.bzl` - and the container image); `compareSolveBenchmarks()` refuses them otherwise and still checks
the p99 against the period and the tape counts. A timing baseline is recorded on an idle machine; a container rebuild
changes the toolchain, so a later candidate needs a baseline re-recorded on the new image.

### 4.7 Reference behavior: the measured motion, the commanded twist, the reset heading

| Item | Behavior | Tests |
|---|---|---|
| D1 (whole body), D1b (centroidal) | the first segment of a velocity command's knots holds the mean of the command and the robot's **stride-mean** measured motion, in the heading frame (section 2.9), the knots on the arc and at most π/2 apart; a restarted history is seeded with the exact base twist `getBaseTwistInWorldFrame()` (centroidal `A_b⁻¹(m h̄ − A_j q̇_j)`) | `testWBIntermediateKnotMeasuredRate`, `testIntermediateKnotMeasuredRate`, `testVelocityCommandArc`, `testCentroidalRobotModelBoundary` |
| D2 | the gait switches read the stride mean of the twist-heading rate and of the heading-frame velocity, once the window is measured throughout; the Unitree G1 at 0.5 m/s stays in `slow_walk` (slow_walk's mean speed peaks at 0.250 m/s, the raise needs more than 0.25) | `testProceduralGaitTransitionsUseMeasuredYawRate`, `testStrideMotionHistory`, `testProceduralMpcMotionManagerCreate`, `test_stride_motion_of_recorded_walks`, `test_mpc_node_runtime` |
| D3 | the reference manager reads a knot's commanded planar twist through `getTargetPlanarCommand(knot, compositeYawInertia)` | `testWBReferenceManagerCommandedTwist` |
| D9 | the reset target keeps the twist heading of the measured attitude, `headingQuaternion()`: the reset yaw differs from the ZYX yaw by `\|o(θ, φ)\|` | `testCentroidalMpcResetTarget`, `testWBMpcResetTarget` |

### 4.8 What to watch in MuJoCo

- **The 720° turn** (`turn_in_place_720` on the whole-body G1, the shipped tuning that turns through ±π): no tilt spike
  at the ±π crossings and a small `initial_state_gap.max_rotation_rad`. No test runs the whole-body G1's turn; on the
  centroidal EngineAI SA01, at 1 rad/s one way with the yaw weighed 40 in the test only, `testTurnInPlace720` keeps the
  rotation gap below 0.1 rad over 4π in 13.0 s, largest 0.0067 rad.
- **Tilt under turning** on SA01 and R1 (pitch and roll weights 5 and 85): the rows are heading-frame tilts, cubic in a
  mixed tilt (section 2.8.1).
- **Atlas's ACoM yaw** (weight 2500): the multiplicative composition's yaw row differs from the additive rule's at
  O(|tilt|²).
- **Feedback over the bus** on Atlas and SA01: the manifold-linear policy (`mpc/policy` carries
  `CONTROLLER_TYPE_MANIFOLD_LINEAR`; `make ipc-hz TOPIC=mpc/policy`).
- **Gait switching:** the Unitree G1 at 0.5 m/s stays in `slow_walk` (section 4.7), and a new gait's first cycle can
  read the sway as a turn of up to 0.12 / 0.14 rad/s (G1 / R1).
- **The display:** the Euler-rate plots have gaps near a pitch of ±90° by design; the body angular velocity and the
  world-frame twist plots do not.
- **The SQP's cycle** on Atlas and the whole-body G1 (section 6, risk 12): a run's `sqp_step.full_steps_refused_above_g_max`,
  `sqp_step.longest_violation_growth` and `sqp_step.swing_normal_velocity_residual`, and the `solve_sqp_step` and
  `solve_swing_normal_velocity_residual` rows of its time series, which locate it in the gait.
- **The first start** of each robot compiles its CppAD libraries into the tagged folder (section 3.2).

---

## 5. Tests

### (a) The Euler side

- **The chart** (`val/chart/EulerChart`, `test_euler_chart`): the Euler state of a tuning layout (`EulerStateLayout`,
  the task file's `initial_state` row for row), `Φ`, `Φ⁻¹` (the angles wrapped), `dΦ/dx_e` and `P_e` (section 2.12),
  through `orientation/EulerBoundary.h` only.
- **The Euler roots** (`common/test/equivalence`): `RootJointModelPair` builds each shipped robot's reduced URDF model
  twice, on `Composite(Translation, Spherical)` and on Pinocchio's `Composite(Translation, SphericalZYX)`, with the
  chart's velocity maps; `EulerMpcOracle` writes both MPCs' flow maps (O1 to O4) in the Euler coordinates on the
  Euler-root twin of the MPC's own Pinocchio model, from their definitions, calling nothing of the production flow maps;
  `EulerChartStateManifold` is the quaternion state space seen through the chart, on which OCS2's SQP performs the
  Euler-coordinate formulation's iteration with the MPC's own terms.
- **The inputs** (`MpcEquivalenceFixture`): every MPC, oracle and reference is built from the configuration's shipped
  files; states and inputs are seeded random draws or built from the task file's `initial_state`, and the SQP and term
  tests run under a walking reference (`walkingReference()`: a double support, then four single supports of 0.35 s; a
  target 1 m ahead in 2 s) from a resting warm start (`restingWarmStart()`: the initial state with the input that
  carries the weight on both feet). A retuned task file or URDF is tested as it is: no recorded data, no hash.

### (b) Equivalence suites

- **The root-joint suite** (`common/test/equivalence`, `RootJointModelPair`): each shipped robot's reduced URDF model
  built twice, on `Composite(Translation, Spherical)` and on `Composite(Translation, SphericalZYX)`, the chart between them
  `EulerChart`'s own `Φ` and `P_e`, mappings C1 to C7 derived in its header. `testRootJointKinematicsEquivalence` (frame
  placements, Jacobians `J_e = J·J_v`, twists, accelerations, the CoM, `integrate` / `difference`),
  `testRootJointCentroidalEquivalence` (`A_G,e = A_G·J_v`, `h_G`, `I_G`, `dh_G/dt`, the chain rule of `dA_G/dt`, the base
  velocity recovery), `testRootJointDynamicsEquivalence` (`M_e = J_vᵀMJ_v`, the nonlinear effects, gravity, RNEA with and
  without external wrenches, ABA with a backward-error check, energies), `testQuaternionRootOnlyCases` (pitch exactly
  ±90° as level dynamics under turned gravity, the Euler root's determinant ratio `−cos(pitch)`, paths through pitch 90°
  and yaw ±π, `−ξ` bit for bit). Margins: 3e-15 relative against 1e-12; `dA_G/dt` 1.4e-10 against 1e-8; removing
  `dT_B/dt` from the chart fails all 7 acceleration-dependent tests on all 5 robots. The two roots share the
  `Translation` sub-joint, so `testRootJointFreeFlyerEquivalence` also holds the production root to Pinocchio's own
  `FreeFlyer` through `J_m = blockdiag(Rᵀ, I, I)` (placements, LOCAL and world-aligned twists, classical accelerations,
  RNEA, CRBA, the nonlinear effects with `J̇_m·v`, ABA, the centroidal map) and pins the base block of `A_G` that the
  closed-form inverse assumes, `[[m I, −m[c − p]ₓR], [0, I_G R]]`, on every robot at pitch ±90°, yaw π and `−ξ`; and
  `testMrtTorqueChartEquivalence` holds the MRTs' feedforward torques (`computeJointTorques`,
  `computeBaseHeldJointTorques`) to the Euler root's equations of motion: equal for the free base, `τ_e − τ_t =
  M_jω·(dT_B/dt)·Θ̇` for the held one.
- **The MPC-level suite** (`testMpcEquivalence_<robot>`, one exclusive binary per robot): an Euler side built from the MPC's
  own Pinocchio model with the root swapped and the flow maps written from their definitions (`EulerMpcOracle`, O1 to
  O4). `testMpcDynamicsEquivalence` (the flow map at 1e-9 through the chart, its Jacobians by
  the chain rule, the SQP's tangent `A_t`, `B_t` against the Euler RK4 map within each map's RK4 error, pitch ±90°, yaw
  ±π), `testMpcRolloutEquivalence` (full-horizon rollouts within the summed Richardson error, halving dt shrinks the gap
  by at least 8; largest node difference 4e-12 to 6e-8, 2e-11 to 2e-8 from one state), `testMpcOrientationCostEquivalence` (the orientation rows
  against closed forms in the Euler angles at 1e-12, third-order cost difference, the Gauss-Newton model at a level state
  and within `|t|·‖Q‖` at a tilt), `testMpcSqpStepEquivalence` (S1 and S2 of section 4.4; S1w, the whole body's
  angular-velocity gap without rotation: the chart's gap sensitivity is the identity but for `C(Θ, Θ̇_next − Θ̇_F)·T_B⁻¹`,
  and the iterations differ in proportion to it; S3, the same discrete problem and so the same fixed points: at the
  resting warm start and at the quaternion SQP's iterate from it the two charts' linearizations of the flow and jump
  gaps (an event node's identity jump included), the only data in which the QPs differ, differ at most in proportion to the gaps, and closing the iterate's gaps by construction they
  approach each other with the gaps and coincide without them), and `testMpcSqpConvergence` (the SQP on the shipped
  tuning, read from its own records of its iterations: iterated from the resting warm start it converges by its own test
  within 100 iterations, the G1 in 17, R1 in 24, SA01 in 40; in real time, one iteration and one node per solve
  without a plant, every solve of the second half (after the first 20 solves have left the iterate of (C1)) takes the
  full step - EngineAI SA01 refuses one full step in the first half, at t = 0.12 s, above `g_max` - leaves a violation
  below `g_max` in most solves and holds the swing
  rows, in their own units (m/s on the centroidal MPCs, m/s² on the whole body); on Atlas and the whole-body G1 the test
  holds the known cycle of section 6, risk 12, instead, under which Atlas's real-time solves hold the swing rows all
  the same).
  `testMpcDynamicsEquivalence` also holds the SQP interval with a gap rotation of up to 1.5 rad to the
  derivatives of `difference(x_next, Π(F(x ⊕ d, u + δu)))` on the real dynamics (`Jr⁻¹(b)`).
- **Every term on the quaternion root** (`testQuaternionRootTermDerivatives_<robot>`, one exclusive binary per robot):
  every cost, soft constraint and constraint of the MPC, intermediate and final, and its flow map, differentiated by
  Richardson-extrapolated central differences of their double-precision values along the manifold's retraction and the
  inputs, against the derivatives they report pulled back into the tangent, at the initial state, single- and two-axis
  tilts, headings of 2.5 and −3.0 rad, a pitch of 80°, a seeded random state, the middle of both single supports of
  the walk, and at rest at a level attitude and at pitch ±90°; the input Hessians of every
  `ConstraintOrder::Quadratic` constraint against differences of its input Jacobian. No production library generates a
  second order: every CppAD library of the MPCs is first order. The piecewise polynomial barriers, which those points
  leave in their zero region, are held at two more points inside their bands (`barrierBandPoints()`): the joint limits
  and the basis scalings at the initial point with the joints and the scalings moved into the bands, the foot
  collision at the right foot's mid-swing point with the joint angles moved until the closest collision points of the
  feet or the knees are inside its band. Both tests of the binary fail a term of their collections, the pre-jump ones
  included, that no point checks (`coverageFindings()`), and a term checked only where what they hold is zero - the
  reported first derivative, the reported Hessian - so that its comparisons were of zeros (`nonZeroFindings()`; a term
  no point can reach is exempted by name with the reason). The first-order test adds
  the soft form of a stance foot's constraint, which no shipped robot lists, to its copy
  (`addSoftStanceFootConstraint()`), so that the soft branch of the dropped-term check of section 2.8.5 runs on the
  centroidal MPC and the whole body's soft stance gradient is differenced at every point (section 2.8.6). At the same
  points (`TermProbePoints.h`),
  `testQuaternionRootTermHessians.cpp` enumerates every cost and soft constraint by name, gives each the rule its
  class documents (`L = φ(v(x, u))`: a tracking residual under `½vᵀWv` - the tuning deviation, the input deviation,
  every CppAD cost vector, the DCM error, the CoM and ACoM residual - or a constraint under its penalty; a term of no
  known class fails by name) and holds, in the tangent: the value to `φ(v)`, the radial gradient `ξᵀ∂L/∂ξ` to zero (so
  `EᵀHE` needs no retraction curvature), `J` and a penalty's `p'`, `p''` to differences (one-sided across a branch
  switch of the rotation-to-quaternion conversion, which flips a quaternion residual's sign and leaves the cost
  unchanged), the reported Hessian to `J̃ᵀφ''J̃ + Σp'H_c` within the round-off of the two assemblies (symmetric,
  positive semi-definite), a `ConstraintOrder::Quadratic` constraint's second order to its exact one minus its
  documented shift (the friction cone's state Hessian exactly `−s·I_a`, so its soft constraint's is `c·I_a`,
  `c = −sΣp' ≥ 0`, and `c·EᵀE`, a quarter on the rotation rows, in the tangent), and the reported Hessian to the exact
  Hessian (differences of the verified gradient) minus the dropped curvature `Σφ'_i∇²v_i`: equal to the exact Hessian
  where `φ'(v) = 0` or `v` is linear, different from it by exactly that curvature where neither holds (both sides are
  counted on every robot). Every Hessian comparison rejects one entry moved by 1e-6.

### (c) Test map by Bazel target

| Target | Checks |
|---|---|
| `//lib/ocs2:test_state_manifold` | `difference(x, x⊕δ) = δ` for ‖δ‖ < π; ⊕ / ⊖ inverse; `E⁺E = I`; `Jr⁻¹` against finite differences; the retraction-curvature identity; slerp endpoints and the antipodal boundary; `Log(cξ) = Log(ξ)` from 1e-150 to 1e150 and the half-turn branch, on the tape too; in-place pull-backs equal dense products; the `Into` forms bit for bit and allocation-free; the `pushForwardDynamics` precondition |
| `//lib/ocs2:test_manifold_projection` | node, event, cost and constraint pull-backs against finite differences of `F(x⊕δ, u+δu) ⊖ x_next` (≤ 1e-8); the manifold `toPrimalSolution` anchors |
| `//lib/ocs2:test_manifold_linear_controller` | node exactness; between-node blend; pre-event copy; `flatten` / `unFlatten`; resampling off the nodes; `−x` invariance; `keepFirst` |
| `//lib/ocs2:test_sqp_on_so3` | a rigid-body attitude OCP (ambient 7, tangent 6) tracking a 720° heading: monotone heading, rotation part of `δx₀` < 0.1, the same `u` to 1e-12 under `−ξ_init`; the gain law of section 2.7 ("Which gain at which anchor"): the QP's gain at the post-step anchor, its components unchanged |
| `//lib/ocs2:test_sqp_flat_parity` | a flat toy problem's recorded solve bit for bit with a null manifold |
| `//lib/ocs2:test_mrt_state_manifold`, `:test_time_triggered_rollout_manifold`, `:test_cppad_library_dimension_check` | the MRT's slerped plan; the rollout's projection; the stale-library refusal naming the folder |
| `common:testBaseOrientation`, `:testBaseOrientationCppAd` | `J_HT` against finite differences; level gives `P₃`; tilt at exactly level; pitch 89.9 / 90 / 90.1°; the upside-down clamp; the double cover bit for bit in the clamp band; the two singular sets; a NaN quaternion; the section 2.8.1 identities; the squared-norm guard and the rate matrix on a tape at `x = ones` |
| `common:testEulerBoundary` | round trips; the `asin` clamp at ±90° (with its round-off cases) and parity with an independent ZYX conversion; `T_B`; `stateFromTuningLayout` sizes and refusals; the placement |
| `common:testConfigurationTangent` | `pinocchio::integrate` / `difference` equal the manifold's; finite differences of frames and CoM equal Pinocchio's Jacobians; `computeConfigurationDerivative()` |
| `common:testStateManifoldFactory`, `:testStateLayout`, `:testCppAdLayoutTag` | the product manifold of both layouts; a layout that does not describe the state refused (a state of another size, an unplaced layout, an orientation on 3 or 5 rows: "as a quaternion, on 4 rows"); the tag; every shipped folder ends in it |
| `common:testTuningLayoutDeviation`, `:testTuningQuadraticCost`, `:testInputQuadraticCost` | finite differences of every block along the retraction; radial invariance; at a level state the tangent model equals the Euler differences of the tuning layout; the tilt bounds through the chart; `−ξ` bit for bit for the residual and the cost; block against dense assembly; OCS2 parity on a layout of differences; input-only hot reload |
| `common:testComAndAcomTrackingCost`, `:testAngularCenterOfMass`, `:testPlannedHeadingOverrideAcom`, `:testAcomAngularVelocityConsistency` | the ACoM rows and their adjoint against finite differences at a tilted base with bent joints; the E2 ComAndAcom row, Jacobian and planner-heading bounds of section 4.4 against the additive Euler rule (`ComAndAcomEulerTolerance`); `acomAttitude`; the planner heading and offset; the network's rates against the centroidal connection |
| `common:config_weights_test` (`test/config/weights/testStateInputWeightsFromConfig.cpp`) | every shipped task file's `state_weights`, `final_state_weights` and `initial_state` convert to n_t rows, each orientation block on its rows |
| `common:parameter_update_test` | the shared updater on the quaternion layout (nx = n_t + 1, Q n_t square) |
| `common:testFloatingBaseDynamics`, `:testJointTorqueInverseDynamics`, `:testPinocchioJointNaming`, `:testQuaternionRootJointCppAd` | the quaternion root and its joint offsets; inverse dynamics on normalized samples; taping of the root's Pinocchio passes, matched to double at random attitudes including pitch 90° |
| `common:testSubdivideYawKnots`, `:testVelocityCommandArc`, `:testStrideMotionHistory`, `:testCommandedPoseChange`, `:testProceduralGaitTransitionsUseMeasuredYawRate`, `:testReferenceConsumersFollowTheModelLayout`, `:testInProcessMpcLink` | the knots and their spacing; the stride mean; the arc; the gait switches; every reference consumer through the model (the resample bound above 63 rad/s); the in-process link's MRT on the problem's manifold |
| `common/test/equivalence:testRootJoint{Kinematics,Centroidal,Dynamics,FreeFlyer}Equivalence`, `:testQuaternionRootOnlyCases`, `:testMrtTorqueChartEquivalence`, `:testMpcEquivalence_<robot>`, `:testQuaternionRootTermDerivatives_<robot>` | section 5(b) |
| `cmpc:testCentroidalRobotModelBoundary` | sizes, the manifold (ambient = tangent + 1), `get(set(pose))` = pose mod 2π to 1e-12, the slerp midpoint heading wrap(3.25), tangent maps against `pinocchio::difference`, the exact base twist against Pinocchio at joint rates up to 3 rad/s (1e-9), allocation-free measurement helpers |
| `cmpc:testCentroidalCommandYawEquivariance` | the knots of a velocity and of a pose command, and the slerped reference between them, turn with the robot across a heading of ±π: position, quaternion as a rotation, normalized momentum, to 1e-12, on the four centroidal robots |
| `cmpc:testCentroidalQuaternionDynamics`, `:testSrbdQuaternion` | AD Jacobians against finite differences; homogeneity `f(λξ) = [f_other, λf_ξ]`; `f(−ξ)` flips only `ξ̇`; the angular block `I_G R` of `A_b` with `I_G`'s condition number at every attitude, pitch 90° included; RK4 norm drift below 1e-8 for a spin about a principal axis, fourth order (a few 1e-7) for a tumbling body; SRBD `A_b` equals the full model's at the nominal posture, and so do its flow map (the function the tape records, in double) and its derivatives in the momentum, the base position and the quaternion with the joints still, at every attitude |
| `cmpc:testQuaternionInvariances` (Atlas, SA01, G1, R1) | for every registered term and the dynamics: tangent finite differences; radial invariance `g(cξ) = g(ξ)` for c ∈ {0.5, 2, −1, −2}; `g(−ξ) = g(ξ)`; pitch 90° finite; that a turn's representation (α against α + 2πk) changes no term and the dynamics turn with the state (the costs are not yaw-equivariant on either layout: the shipped x and y weights differ) |
| `cmpc:testUnitNorm`, `:testTurnInPlace720`, `:testMrtObservationDoubleCover`, `:testPoseCommandSubdivision` | unit norm after retraction, slerp and rollout (largest error 2.2e-16 over 4,800 solves); 4π in place (the yaw weighed 40 in the test, which no shipped tuning does); `ξ` and `−ξ` observations give identical solver inputs and joint commands, a `w ≥ 0` stream across ±π is continuous; Δψ = 270° turns +270° |
| `cmpc:testLiveUpdateEqualsFreshStart`, `cmpc_test:test_mpc_parameter_updater_module`, `cmpc:testMpcParameterUpdaterModule` | section 3.5 |
| `cmpc:testMrtJointControllerActionParity` | the MRT joint controllers' actions against their golden `cmpc/test/golden/mrt_joint_controller_actions.txt`, recorded on the quaternion root ("base held" is `ω̇_B = 0`, section 2.5) |
| `cmpc:testIntermediateKnotMeasuredRate`, `:testCentroidalMpcResetTarget`, `:testProceduralMpcMotionManagerCreate` | D1b, D9, D2 (section 4.7) |
| `cmpc_test:test_centroidal_mpc_robot_model`, `:test_pinocchio_frame_conversions`, `:test_dynamics_helper_functions`, `:test_centroidal_conversions`, `:test_centroidal_mpc_mrt_joint_controller_allocations` | the robot model and conversions on the quaternion root; one Pinocchio allocation per centroidal map and per nonlinear-effects pass |
| `cmpc_test:test_centroidal_mpc_mrt_policy_execution`, `wbmpc:testWBMpcMrtPolicyExecution` | a policy the realtime evaluator refuses (a `LinearController` on the quaternion state, a `ManifoldLinearController` with ambient gains) is held with the hold's action, never handed to `MRT_BASE::evaluatePolicy()`, allocates what the hold does and posts one `kPolicyRefused`; the whole body's action of a manifold-linear policy at pitch ±90° and yaw π is finite and the same under `−ξ` to round-off (1e-12) |
| `wbmpc:testWBRobotModelBoundary`, `:testWBQuaternionDynamics`, `:testWBQuaternionInvariances`, `:testWBMrtObservationDoubleCover` | as the centroidal ones; RK4 drift 3.9e-9 at the rollout's 0.015 s step and, for one 0.035 s SQP node, a truncation error below 1e-7 that halving the step cuts more than 16-fold (each node is retracted, so it does not add up); 46 term evaluations over 4 points, finite differences to 8.7e-10, all bit for bit under `−ξ` |
| `wbmpc:testOrientationErrorWrtPlaneRates`, `:testStanceFootOrientationRates`, `:testEndEffectorDynamicsAccelerationsConstraint` | the stance rates of section 2.8.6: their math on the taped functions; the G1 constraint against the PD law assembled from double-precision kinematics, the rescaled gains at a level foot, the linearization along the manifold; the constraint's composition on an affine stand-in |
| `wbmpc:test_wb_live_tuning`, `:testWBIntermediateKnotMeasuredRate`, `:testWBReferenceManagerCommandedTwist`, `:testWBMpcResetTarget`, `:testTargetTrajectoriesCalculatorSetters`, `:testWBMpcMrtJointControllerAllocations` | section 3.5; D1, D3, D9; the command references turn with the robot across ±π; a base-pose heuristic's tilt keeps the reference's world angular velocity (`shapeBasePose` carries `w_B,r` into the tilted body frame); pose turns of 90°, 270° and −135° through subdivided knots; the realtime path's allocations |
| `ipc:test_mpc_message_conversions`, `:test_realtime_policy_evaluator`, `:test_remote_mpc_link_parity`, `:test_remote_mpc_link_allocations`, `:test_mpc_message_conversions_allocations` | the manifold policy's round trip from `toPrimalSolution` with an event; refusals; evaluator parity with `MRT_BASE::evaluatePolicy` across an event and under `−x`; remote against in-process; zero allocations |
| `capp:test_centroidal_mpc_manifold_policy_over_bus`, `wapp:test_wb_mpc_manifold_policy_over_bus`, `capp:test_centroidal_mpc_node`, `wapp:test_wb_mpc_node`, `app/node:test_mpc_node_runtime` | section 3.3 (the manifold policy over the bus on both layouts, the quaternion at ambient offset 9 and 3); the node tests' decoded controller types and the dummy plant's unit quaternion (1e-9) |
| `app/robot:test_robot_realtime_allocations`, `:test_sim_fall_recovery`, `:test_initial_sim_state` | the robot process allocates nothing; q from the quaternion; the stored quaternion bit for bit |
| `app/visualization:test_robot_state_decoder`, `:test_telemetry_builder`, `rr:test_contracts`, `:test_plot_config`, `:test_bridge` | section 3.4 |
| `val:test_euler_chart`, `val:test_initial_state_gap`, `val:test_closed_loop_metrics`, `val:test_sqp_step_diagnostics`, `val:test_solve_benchmark_gate` | the chart of section 5(a); the gap through the manifold and `maxQuaternionNormDeviationOf()`; the band rules and the SQP-step metrics; the SQP step a solve records (the swing rows found by name, their largest residual); the toolchain refusal |
| `tools/hooks:test_spherical_zyx`, `:test_euler_boundary`, `:test_state_interpolation`, `:test_quaternion_scalar_constructor`, `:test_cpp_source` | each rule fires on a violating snippet and not on the boundary (Appendix B) |
| `//humanoid_learning/acom/tests:test_acom` | `acom_attitude` and its tangent Jacobian match the C++ formulas |
| `robot_models/*/*/test/testPinocchioModel.cpp` (playgrounds) | configurations from `pinocchio::neutral` and the root's `idx_q()`, so the printed base heights stay right |

### (d) Closed-loop harness

`val/` (`LockstepClosedLoop`, `ClosedLoopDriver` and its centroidal and whole-body drivers): the headless MuJoCo
simulator, the formulation's MPC as its node builds it and its MRT joint controller as its robot binary builds it, with
an `InProcessMpcLink` whose solver iterations the runner calls (`Execution::kCaller`); one thread, no bus. Targets are
`manual` and `exclusive`; `make closed-loop-metrics ROBOT=<robot> SCENARIO=<scenario> LABEL=<label> [BASELINE=<label>]`
runs one robot at a time and writes `val/data/closed_loop/<label>/<robot>_<scenario>.json`; with `nThreads` 1 two runs
agree to 1e-9 (`test_lockstep_determinism`). Never set `RECORD_STATES=1` on a comparison run: it overwrites the
benchmark's replay states. See `val/README.md`.

---

## 6. Remaining risks and open decisions

| # | Risk or decision | State |
|---|---|---|
| 1 | **Gait-switch speed clause:** the faster switch reads the stride-mean speed, so the Unitree G1 stays in `slow_walk` at 0.5 m/s | open for the user: keep it and lower the raise threshold out of `slow_walk`, or read the instantaneous speed for raises only |
| 2 | **First cycle of a new gait:** the stride window still reaches into the previous gait and reads up to 0.12 / 0.14 rad/s of sway as a turn (G1 / R1) | open for the user: hold the switches for that cycle too (exact, but stance to trot then takes at least 8 s) |
| 3 | **Arm swing:** `SwitchedModelReferenceManager::getDesiredState`'s arm swing reads `getBaseHeading()` of the node's state, so the reference depends on the state while the cost Jacobians omit that dependence: `StateInputQuadraticCost`'s state Jacobian lacks `−Σ_j Q_jj (q_j − ref_j)·(∂ref_j/∂h)·(∂h/∂x)`, `∂ref_j/∂h = ∓0.15·sin(2π(phase − 0.15))·v_lat`, `v_lat = −sin h·v_x + cos h·v_y`, on the four arm joints. The term derivative suite's walking reference commands no velocity, so it leaves this term unexercised (`testQuaternionRootTermDerivatives.cpp` says so), as `testQuaternionInvariances` does by switching the arm swing off; the Euler formulation did the same | **fixed**: the arm swing reads the heading of the reference (`getBaseHeading(xNominal)`), which is a constant of the step, and `testQuaternionInvariances` now checks the state costs with the arm swing on (humanoid_nmpc/docs/locomotion_heuristics/README.md, the note on the procedural arm swing) |
| 4 | Shipped tunings weigh no yaw angle in Q (Atlas weighs its ACoM yaw at 2500 in `ComAndAcomTrackingCost`; the whole-body G1 weighs the body yaw rate at 3 in Q). The centroidal robots, Atlas included, do not turn in place: in the archived `turn_in_place_720` runs, recorded before HEAD cf69a5e3 and not yet re-recorded on the current tree, `heading.cumulative_max_rad` is about 0.03 to 0.27 rad against the commanded 4π (the scenario turns back to its start heading, so its end heading cannot tell a turn from none); the whole-body G1 turns part way, 4.4 to 5.2 rad | the yaw is driven by the momentum target; `testTurnInPlace720` weighs it in the test only |
| 5 | Tuning meaning drifts slightly: twist against ZYX heading, cubic tilt, the whole-body ω rows, the ACoM composition | stated E2 tolerances (section 4.4) |
| 6 | The yaw residual is discontinuous at a yaw error of ±π; the `Log` cut locus at π | references re-seeded every command; knots at most π/2 apart for every input; the warm-start guard at π/2 |
| 7 | A future change re-introduces raw state slices, a linear state interpolation or an Euler conversion in the formulation | the four lint rules (Appendix B); radial and `−ξ` tests over every term |
| 8 | Stale CppAD libraries | the tag in the folder and the domain / range refusal (section 3.2) |
| 9 | Upside-down states (falls) | the residual clamps keep values finite; the dynamics stay regular (`I_G ≻ 0`); `SimFallRecovery` computes the tilt from the quaternion |
| 10 | The ACoM network's Δθ is not a true rotation vector (a non-integrable connection) | first-order agreement tested; `testAcomAngularVelocityConsistency` against `dataset_generator.py` |
| 11 | A container rebuild before a later benchmark | the timing baseline must be re-recorded on the new image (section 4.6) |
| 12 | **The SQP cycles on DRC Atlas and on the whole-body Unitree G1** instead of converging (`testMpcSqpConvergence`, the shipped tuning, from the resting warm start toward the walking reference). Its QP's Hessian is the costs' Gauss-Newton Hessian without the hard constraints' curvature `λᵀ∇²g` (whether the projection removes them or not), its merit is the cost alone (this fork's Lagrangian sums are zero), and `FilterLinesearch::acceptStep()` takes a step between `g_min` and `g_max` for a decrease of the merit or of the violation. So each full step lowers the merit and multiplies the violation of the hard rows - on Atlas a swing row, `foot_r_contact_normalVelocity` at nodes 12 to 14 (0.22 to 0.26 s, just after the right lift-off at 0.2 s), by 2.5 per iteration on average, 2.32 on the whole-body G1 - until a full step's violation (1.0e-2 to 1.8e-2 on Atlas) passes `g_max` = 1e-2; the CONSTRAINT rule refuses it, the half step is taken below `g_max`, and the full step after it drops the violation by an order of magnitude while it raises the merit. The cycle repeats every 4 to 5 iterations on Atlas with its shipped `touch_down_height_offset` 0.04 and the swing apex above the higher end of the swing (every 4 to 5 on the whole-body G1) for as long as it runs (13 resets in iterations 50 to 99 on each), in the Euler chart too (`OnAtlasTheEulerChartCyclesAsTheQuaternionManifoldDoes`: the chart's SQP cycles with at least 3 resets in the second half, 13 of them in iterations 50 to 99, and its merits agree with the quaternion manifold's over the first 20 iterations to second order in the step, within 0.05 `φ₁²` relative, `φ₁` the largest base rotation of the first step: at most 1.25e-4 for `φ₁` = 0.111 rad, 0.010 `φ₁²`, the retractions' difference `R Exp(φ)` against `R(Θ + T_B⁻¹φ)`, `½|φ|²` at most, the problem being the same on both; its violations agree only to about 1 %); within a period no iterate is dominated by one before it. On the whole-body G1 the violation that cycles is carried by several curved hard rows: about two thirds by the left mid-swing row `foot_l_contact_normalVelocity` (node 27, at 0.825 s, 0.175 s after the 0.65 s lift-off) and about a third by the stance feet's height (row 2, nodes 33 to 35, after the 1.0 s touchdown) and pivot (row 5, nodes 27 and 28) rows; the tilt rows carry 1 to 2 %. That cycle predates the whole-body stance orientation-rate law of section 2.8.6: under the former law (the raw `ω` and `α`, gains 20 and 1) the iterated SQP gave 11 to 12 resets, a growth of 2.262 per iteration and a period of 5 (measured with the swing apex above the lower end of the swing), and removing any one family of curved hard rows moves the cycle to another. The whole body's swing row is not a velocity: it is `k_v (v_z − ż_ref) + k_a (a_z − z̈_ref) + k_p (z − z_ref)` in m/s² (10, 1 and 100 on the G1, `WBMpcPreComputation::normalAccelerationConstraintConfig()`), where the centroidal row is `v_z − ż_ref + k_p (z − z_ref)` in m/s, so its residuals are not the centroidal ones (`testMpcSqpConvergence` holds it to `k_v` times the centroidal bound). In real time (one iteration and one node per solve without a plant: Atlas's 50 Hz at `dt` 0.02) the warm start is the last plan node for node, but for the newly initialized end of the horizon, whose violation each shift adds (0.004 to 0.13, a median of 0.08, on the converging configurations). That holds since the fix of the SQP's trajectory spreading (`lib/ocs2/README.md`, "What this fork changes"): before it the warm start handed each post-event interval the next node's input and the node after it about the midpoint of the next two states (on the whole-body G1 an input off by `|Δu|` of about 800 to 1000 and a state by `|Δx|` of 5 to 7 there, a dynamics gap of 22 to 27 against 0.3 to 0.6 elsewhere), every full step was judged above `g_max`, where the CONSTRAINT rule accepts any decrease, and the violation climbed 1.6 to 1.8 times per solve to a plateau (a median of 0.12 over the last 20 of 40 solves on Atlas with a swing residual of 0.59 m/s, 2.9 and 8.3 m/s² on the whole-body G1). With the warm start fixed, the converging configurations leave a violation with a median of 3.6e-3 to 3.9e-3 (at most 2.3e-2) and a swing residual with a median of 0.0092 to 0.029 m/s (at most 0.13), every step of the second half full. Atlas now holds its swing rows in real time too (a median of 0.056 m/s, at most 0.27) and leaves a violation below `g_max` in most solves (a median of 8.0e-3, at most 3.8e-2), but the cycle still shows: the violation climbs above `g_max` for runs of solves, and a full step that raises it there is refused and the reduced step lands below `g_max`, a reset (2 in the last 20 solves). The whole-body G1 stays above `g_max` in each of the last 20 solves (baselines of 0.43 to 4.0, a median of 0.55 after the step, at least 0.12), refuses 6 of 20 full steps for raising the violation, and does not hold its swing rows (a median of 2.3 m/s², at most 12.6). Unitree G1, Unitree R1 and EngineAI SA01 converge by the solver's own test in 17, 24 and 40 iterations | open for the user. Remedies, each behind a default-off switch if implemented: an exact (ℓ1) penalty of the hard violation in the merit, or a cap on its growth in the acceptance rule (a step taken for its merit decrease only while the violation does not grow past a bound); a second-order correction of the step (the QP re-solved with the post-step residual of the hard rows before the line search judges it); the curvature of the projected hard rows in the QP's Hessian (`λᵀ∇²g` with the projection's multipliers); `normal_velocity` as a soft constraint on Atlas (the contact-implicit formulation's priced row). A closed-loop run shows the cycle in `sqp_step` (`full_steps_refused_above_g_max`, `longest_violation_growth`, the violation and the swing residual; `val/README.md`) and in the `solve_sqp_step` rows of its time series |

---

## Appendix A: IFTTT pairs of the formulation

| Label(s) | Ties |
|---|---|
| `cppad_layout_tag` ↔ `cppad_model_folder` | `StateLayout.h` ↔ `ModelSettingsFromConfig.cpp` (and this README, section 3.2) |
| `controller_types` | `ocs2/core/.../ControllerType.h` ↔ `msgs/controller_type.proto` (↔ `ipc/src/MpcMessageConversions.cpp`) |
| `controller_data` → `manifold_node_writer` | `msgs/mpc_policy.proto` ↔ the encoder of `MpcMessageConversions.cpp` |
| `euler_boundary` | section 4.2 ↔ `tools/hooks/euler_boundary.py` |
| `orientation_rules` | `AGENTS.md` ↔ `tools/hooks/checks.py:registry` |
| `input_offsets` | `StateInputWeightsFromConfig.cpp` ↔ the robot models' input indices and `StateInputLayout.cpp` (the tuning layout itself has one description, `tuningLayoutOf()`, and no pair) |
| `robot_mpc_link`, `mpc_wiring` | the two RobotMains; the two MPC nodes |
| `observation_hand_over` | `MpcNodeRuntime.cpp` ↔ `ClosedLoopDriver.cpp` |
| `acom_attitude` | `humanoid_learning/acom/models.py` ↔ `AngularCenterOfMass.cpp` |
| `base_pose_heuristic_seam` | `SwitchedModelReferenceManager.cpp` ↔ the heuristic layer |
| `telemetry_groups` ↔ `panel_groups`, `complete_groups`, `panel_tabs` | `TelemetryBuilder.cpp` ↔ `rr/.../telemetry_contract.py` ↔ `rr/README.md` |
| `metric_bands` | `MetricBands.h` ↔ `val/README.md` and this README |
| `real_time_gate` | `SolveBenchmarkGate.h` ↔ `val/README.md` and this README (section 4.6) |
| `configuration_tangent_data` ↔ `configuration_tangent_robots`, `equivalence_robots`, `mpc_equivalence_robots`, `shipped_robots` | the shipped robots ↔ the tests that must cover each one |

## Appendix B: lint rules

Registry checks of `tools/hooks/checks.py`, each with its module and `test_<module>.py`, listed in `AGENTS.md`
(`orientation_rules`) and run by `make lint` (`python3 -m tools.hooks.lint_code --only <check> --paths <dir>` narrows a
run). A line where a rule is wrong carries `// NOLINT(<check>): <reason>`.

| Check | Rule | Scope | Escape |
|---|---|---|---|
| `spherical-zyx` (`spherical_zyx.py`) | no `SphericalZYX` root joint | the C++ of `humanoid_nmpc/`, `robot_runtime/`, `lib/ocs2/pinocchio/centroidal_model/` | the equivalence suites' Euler oracle roots, with a reason |
| `euler-boundary` (`euler_boundary.py`) | no identifier containing `EulerZyx`, `EulerAnglesZyx` or `ZyxEulerAngles` (any capitalization), no `#include` of `RotationDerivativesTransforms.h` or `orientation/EulerBoundary.h` | the formulation directories, files and trees and the robot-model headers (section 4.2) | `NOLINT(euler-boundary): Euler boundary accessor (design section 4.2)` on the boundary accessors |
| `state-interpolation` (`state_interpolation.py`) | no one-argument `getDesiredState(t)` and no `LinearInterpolation::interpolate` of anything but inputs | `humanoid_nmpc/` outside the robot models | deliberate linear interpolations in tests, with a reason |
| `quaternion-scalar-constructor` (`quaternion_scalar_constructor.py`) | no quaternion built from four scalars (Eigen takes `(w, x, y, z)`, the state stores `(x, y, z, w)`) | `humanoid_nmpc/` outside `orientation/` | — |
