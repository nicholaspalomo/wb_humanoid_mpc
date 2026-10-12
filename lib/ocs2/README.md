# OCS2 Toolbox
A fork of the Optimal Control for Switched Systems (OCS2) library, built with Bazel and free of ROS and Boost.

## Summary
OCS2 is a C++ toolbox tailored for Optimal Control for Switched Systems (OCS2). Upstream, the toolbox provides an efficient implementation of the following algorithms:

* SLQ: Continuous-time domain DDP
* iLQR: Discrete-time domain DDP
* SQP: Multiple-shooting algorithm based on HPIPM
* PISOC: Path integral stochastic optimal control

This fork keeps only the SQP, the solver the humanoid NMPC runs (see "What this fork leaves out").

![legged-robot](https://leggedrobotics.github.io/ocs2/_static/gif/legged_robot.gif)

OCS2 handles general path constraints through Augmented Lagrangian or relaxed barrier methods. To facilitate the application of OCS2 in robotic tasks, it provides the user with additional tools to set up the system dynamics (such as kinematic or dynamic models) and cost/constraints (such as self-collision avoidance and end-effector tracking) from a URDF model. The library also provides an automatic differentiation tool to calculate derivatives of the system dynamics, constraints, and cost. Upstream OCS2 also ships ROS interfaces for deployment on robots; this fork has none, and the humanoid NMPC talks over its own IPC bus instead (`humanoid_nmpc/docs/distributed_runtime/README.md`). The toolbox’s efficient and numerically stable implementations in conjunction with its user-friendly interface have paved the way for employing it on numerous robotic applications with limited onboard computation power.

For more information refer to the project's [Documentation Page](https://leggedrobotics.github.io/ocs2/) 

## What this fork leaves out

The fork carries only the packages this repository builds (`lib/ocs2/BUILD.bazel`), and it does not depend on Boost:
the integrators are native ports of the odeint loops OCS2 used. It reads no configuration files: the settings structs
(`mpc::Settings`, `rollout::Settings`, `sqp::Settings`, ...) are filled by the caller, which in this repository converts
them from the typed textprotos of `humanoid_nmpc/humanoid_mpc_config`, and the upstream `LoadData` loaders, the
`loadSettings` functions and the property tree they read into are removed. The upstream code that was never built here,
or that nothing here uses, is removed rather than ported:

- loopshaping, throughout `core`, `oc`, `mpc` and `robotic_tools`;
- every ROS package: the ROS 1 `msgs` and `ros_interfaces` and their ROS 2 ports (`MPC_ROS_Interface`,
  `MRT_ROS_Interface`, `MRT_ROS_Dummy_Loop`, `DummyObserver`, the visualization helpers and the messages), which the
  IPC bus replaces (`humanoid_nmpc/humanoid_mpc_ipc`), and every package of `robotic_examples`, `raisim`, `mpcnet` and
  `perceptive`, including their ROS launch files;
- the solvers `frank_wolfe`, `ipm` and `slp`, `pinocchio/self_collision`, `self_collision_visualization` and
  `sphere_approximation`, and `python_interface`;
- the Boost INFO configuration format and `misc/Log.h`;
- the DDP solvers, `ddp` (`GaussNewtonDDP` with `SLQ` and `ILQR`, `GaussNewtonDDP_MPC`, `ContinuousTimeLqr`, the
  Riccati equations and the search strategies; `ddp/unsupported` and `DDP_DataCollector` went first), which nothing
  here ran: the humanoid NMPC runs the SQP. With them went the dense reference QP solver of `test_tools/qp_solver`,
  which only they and their tests used (the SQP depended on it without including it), their two tests (the hybrid
  bouncing-mass SLQ test and the state-manifold guard of `GaussNewtonDDP`), `core/initialization/OperatingPoints.h`,
  which only the bouncing-mass test included, and `core/model_data/ModelDataLinearInterpolation.h`, the `ModelData`
  field accessors for `LinearInterpolation` that only the DDP interpolated with;
- the upstream unit tests that no target here builds, with the test problems and fixtures only they used, and
  `core/misc/BugReportTest.cpp`: the tests that remain are the `cc_test` targets of `lib/ocs2/BUILD.bazel`;
- the parts of `core` and `oc` that nothing here calls: `ControllerAdjustmentBase`, `OdeFunc`, `misc/CommandLine.h`,
  `misc/LTI_Equations.h`, `misc/LinearFunction.h`, `StateInputSoftBoxConstraint`, `thread_support/ExecuteAndSleep.h`,
  the augmented Lagrangian terms `StateAugmentedLagrangian` and `StateInputAugmentedLagrangian` with their factories
  (`augmented_lagrangian/AugmentedLagrangian.h`), the CppAD costs `StateCostCppAd` and `StateInputCostCppAd`,
  `SystemDynamicsLinearizer` with the finite differences only it used (`FiniteDifferenceMethods.h`),
  `TransferFunctionBase`, `multiple_shooting/LagrangianEvaluation.h`, `ReferenceManagerDecorator`, `OcpToKkt`, the
  `Ruzi` preconditioner and `PerformanceIndicesRollout`;
- the state-triggered rollout, `StateTriggeredRollout` with `RootFinder`, `StateTriggeredEventHandler` and
  `StateBasedLinearController`, and with it the root-finder settings of `rollout::Settings` and of the task file's
  `rollout` block (retired), the guard surfaces of the dynamics (`computeGuardSurfaces`, the
  `guardSurfaces*` approximations and the `<model>_guard_surfaces` CppAD library, which a model folder generated before
  keeps unused), the time derivatives of the flow and jump maps, the flow- and jump-map parameter hooks of
  `SystemDynamicsBaseAD` with the always empty parameter argument of its `systemFlowMap` and `systemJumpMap` (the
  libraries are taped and generated as before), the event handler of the integrators (`SystemEventHandler`), whose
  events only a test raised, the rollout abort hooks (`abortRollout`, `reactivateRollout`, `resetRollout`) and
  `InitializerRollout`: both MPCs roll out time-triggered;
- the rest of the augmented Lagrangian, whose collections no problem here filled: `StateAugmentedLagrangianCollection`
  and `StateInputAugmentedLagrangianCollection` with their interfaces and the eight `*LagrangianPtr` collections of
  `OptimalControlProblem`, the augmented penalties (`augmented::QuadraticPenalty`, `SmoothAbsolutePenalty`,
  `ModifiedRelaxedBarrierPenalty`, `SlacknessSquaredHingePenalty`) and their base `AugmentedPenaltyBase`, through
  which `MultidimensionalPenalty` handed the soft constraints' penalties a multiplier that was always zero (it holds the
  `PenaltyBase` penalties itself now), the multipliers (`model_data/Multiplier.h`,
  `oc_data/DualSolution.h`), the Lagrangian metrics of `Metrics`, `OptimalControlProblemHelperFunction.h` and
  `SolverObserver`; `PerformanceIndex::equalityLagrangian` and `inequalityLagrangian` stay, and are always 0;
- the solver queries nothing asked: the value function (`SqpSolver::getValueFunction`,
  `sqp::Settings::createValueFunction`, retired from the task file, `HpipmInterface::getRiccatiCostToGo`), `getHamiltonian`,
  `getStateInputEqualityConstraintLagrangian` and `getIntermediateDualSolution` of `SolverBase`,
  `MPC_MRT_Interface::getLinearFeedbackGain`, `HpipmInterface::getRiccatiFeedforward`, `MrtObserver`, the
  trajectory spreading of linear controllers and dual solutions, the status and the debug print of
  `TrajectorySpreading`, and the multiplier of the projected state-input equality constraints
  (`ProjectionMultiplierCoefficients`, which the SQP computed only with `sqp::Settings::extractProjectionMultiplier`
  on, which no shipped task file set, and never read), with that setting, retired from the task file, and the QR
  projection `LinearAlgebra::qrConstraintProjection` it switched to: the projection is the LU one, as it was by default;
- smaller helpers nothing called: the `ModelData` checkers with the `checkSize` and `checkBeingPSD` overloads of
  `Types.h`, `approximatePreJumpLQ`, the multiple-shooting `compute{Intermediate,Event,Terminal}Performance`, the
  soft `DoubleSidedPenalty` and `SmoothAbsolutePenalty`, the analytical `PinocchioEndEffectorKinematics` (the CppAD one
  stays) with `AngularVelocityMapping.h` and `quaternionDistanceJacobian`, the rotation-vector and Euler-uniqueness
  helpers of `RotationTransforms.h`, the global-frame Euler-rate maps of `RotationDerivativesTransforms.h`,
  `createPinocchioInterface`, `getPinocchioInterfaceFromUrdfString` and the URDF tree `PinocchioInterface` copied, three
  lookups of `Lookup.h`, `numerics::almost_le` and `almost_ge`, `ControllerType::ONNX` and `BEHAVIORAL`, the
  relaxed-barrier parameters `inequalityConstraintMu` and `inequalityConstraintDelta` of `sqp::Settings` (retired from
  the task file before), the linear algebra that only tests and the deleted checkers used (`makePsdGershgorin`,
  `makePsdCholesky`, `computeInverseMatrixUUT`, `computeConstraintProjection`, `rank`, `symmetricEigenvalues`, the
  pseudo-inverse of `luConstraintProjection` and `misc/randomMatrices.h`) or nothing used (`makePsdEigenvalue`,
  `eigenvalues`),
  `getNumberOfPrecedingEvents` and `findIntersectionToExtendableInterval` of `ModeSchedule.h`, `concatenate` and
  `controllerEventTimes` of the controllers, the `setController` of `LinearController` and `FeedforwardController`,
  `FeedforwardController`'s constructor from a rollout of another controller, `LinearController::deltaBiasArray_`,
  which nothing filled, the `stateDim`, `inputDim` and `time` that `ModelData` carried for nobody,
  `TimeTriggeredRollout::systemDynamicsPtr`, `SolverBase::printString` and `RequestSet::containsAll`;
- `PinocchioCentroidalDynamics` and `CentroidalModelRbdConversions` of `pinocchio/centroidal_model`;
- the catkin and CMake scaffolding: the `ocs2` metapackage, `sqp/blasfeo_catkin`, `core/cmake`, `doc` and the Jenkins
  pipeline.

## What this fork adds: states on a manifold

The humanoid MPCs store the base orientation as a unit quaternion, so the fork's SQP solves in the tangent of a state
manifold (`humanoid_nmpc/docs/quaternion_base_orientation/README.md`, sections 2.4 and 2.7). A null manifold, the
default, leaves every flat code path bit for bit as it was (`test_sqp_flat_parity`).

- `core/.../manifold/`: `StateManifold` (`retract`, `difference`, `interpolate`, `project`, the allocation-free
  `differenceInto` / `interpolateInto`, the in-place derivative pull-backs), `EuclideanStateManifold`,
  `ProductStateManifold` of Euclidean and unit-quaternion segments, and `UnitQuaternionMath.h` (the rate matrix, Exp,
  Log, the right Jacobians, slerp, the CppAD-safe normalization); `test_state_manifold`.
- `core/.../control/ManifoldLinearController.h`, `u = u* + K (x (-) xbar)`, `ControllerType::MANIFOLD_LINEAR`;
  `test_manifold_linear_controller`.
- `OptimalControlProblem::stateManifoldPtr`; the multiple-shooting projection of nodes, events, costs and constraints
  onto the tangent (`oc/.../multiple_shooting/ManifoldProjection.h`), the manifold `toPrimalSolution()`, the SQP's
  initial gap, retraction and warm-start guard (beyond pi / 2); `test_manifold_projection`, `test_sqp_on_so3`.
- `MRT_BASE::setStateManifold` (slerped plan states), `RolloutBase::setStateManifold` (`TimeTriggeredRollout` projects
  its states); `test_mrt_state_manifold`, `test_time_triggered_rollout_manifold`.
- `CppAdInterface` refuses a cached library whose domain or range differs, naming its folder;
  `test_cppad_library_dimension_check`.
- `pinocchio/centroidal_model` on the root `Composite(Translation, Spherical)`: nq = nv + 1 in `CentroidalModelInfo`,
  the quaternion at state slots 9 to 12, `computeConfigurationDerivative()`.

## Local changes to the vendored CppAD / CppADCodeGen

`thirdparty/include/cppad` vendors CppAD 20190200.5 (`Version.txt`) and CppADCodeGen f3680f1 (`cg/Version.txt`). They
are left as they come but for one change, which makes code generation deterministic:

- **`hash_code()` for `CG`, by value** (declared in `cg/declare_cg.hpp`, defined in `cg/identical.hpp`). CppAD's
  recorder files every constant of a tape under a hash code and gives a new constant the parameter index of the one
  constant last filed under the same code, if `IdenticalEqualCon()` says they are equal (`local/recorder.hpp`,
  `put_con_par()`). CppADCodeGen defined no hash code for `CG`, so CppAD's default applied (`local/hash_code.hpp`),
  which sums the bytes of the object. A `CG` keeps its value behind a pointer (`cg/cg.hpp`), so equal constants were
  filed apart or together by where the heap put their values, and with them the deduplication of the tape's
  constants, the common subexpressions `optimize()` finds by parameter index (`local/optimize/match_op.hpp`), the
  constant terms of its cumulative sums, and so the generated C sources: two generations of one function gave
  different code (reordered sums, merged or unmerged subexpressions, `0 - v` against `-0 - v`), and so libraries
  whose results differed in the last bits from one generation to the next. The overload hashes a constant's value, as
  CppAD hashes a `double`, so that `+0` and `-0` are filed apart and keep their signs, and gives a variable, which
  `IdenticalEqualCon()` never matches, a fixed code. The test is `test_cppad_codegen_determinism`.
- **The generator stamp** (`core/.../CppAdInterface.h`, `kCppAdGeneratorTag`), which goes with it. `createModels()`
  writes `<model>_lib.generator` next to the library, holding the tag, and `loadModelsIfAvailable()` regenerates a
  library whose stamp is missing or names another tag instead of loading it. A library generated before the change
  is therefore regenerated once on its next use, wherever it is cached (the checkout's `cppad_code_gen/`, a runfiles
  tree). Bump the tag with any change to the vendored CppAD or CppADCodeGen, to `optimize()`'s options or to the C
  printer that changes the sources generated from the same tape. The test is `test_cppad_generator_stamp`.

The change also moves the tape operation counts of the solve benchmark (`getTapeOperationCount()`): which constants
share a parameter no longer depends on the heap, and `+0` and `-0` never do. The counts of a benchmark recorded before
it are those of one heap layout, and its gate fails only on an increase of more than 5 %.
