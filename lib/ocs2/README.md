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
  bouncing-mass SLQ test and the state-manifold guard of `GaussNewtonDDP`), and `core/initialization/OperatingPoints.h`,
  which only the bouncing-mass test included;
- the upstream unit tests that no target here builds, with the test problems and fixtures only they used, and
  `core/misc/BugReportTest.cpp`: the tests that remain are the `cc_test` targets of `lib/ocs2/BUILD.bazel`;
- the parts of `core` and `oc` that nothing here calls: `ControllerAdjustmentBase`, `OdeFunc`, `misc/CommandLine.h`,
  `misc/LTI_Equations.h`, `misc/LinearFunction.h`, `StateInputSoftBoxConstraint`, `thread_support/ExecuteAndSleep.h`,
  the augmented Lagrangian terms `StateAugmentedLagrangian` and `StateInputAugmentedLagrangian` with their factories
  (`augmented_lagrangian/AugmentedLagrangian.h`), the CppAD costs `StateCostCppAd` and `StateInputCostCppAd`,
  `SystemDynamicsLinearizer` with the finite differences only it used (`FiniteDifferenceMethods.h`),
  `TransferFunctionBase`, `multiple_shooting/LagrangianEvaluation.h`, `ReferenceManagerDecorator`, `OcpToKkt`, the
  `Ruzi` preconditioner and `PerformanceIndicesRollout`;
- `PinocchioCentroidalDynamics` and `CentroidalModelRbdConversions` of `pinocchio/centroidal_model`;
- the catkin and CMake scaffolding: the `ocs2` metapackage, `sqp/blasfeo_catkin`, `core/cmake`, `doc` and the Jenkins
  pipeline.

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
it, such as B0, are those of one heap layout, and its gate fails only on an increase of more than 5 %.
