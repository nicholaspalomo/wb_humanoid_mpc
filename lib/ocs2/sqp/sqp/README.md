# OCS2 SQP
This package contains a multiple-shooting, sequential-quadratic-programming solver for problems defined with the OCS2 toolbox.

## Dependencies
HPIPM is used as solver for the QP subproblems. HPIPM and Blasfeo are provided by the `@hpipm` and `@blasfeo` Bazel repositories
(bazel/system_libs.bzl); the hpipm_catkin package wraps the HPIPM interface for OCS2.

## Read-only diagnostics
The solver records what it did and lets nothing in the solve read it back: `getInitialStateGap()` is `delta_x0` of the
last solve's first iteration, and `getIterationRecords()` every iteration of the last solve - the performance the line
search started from, the full step's verdict (`StepInfo::fullStepType`, `fullStepConstraintViolation`: a refused
`CONSTRAINT` full step was above `g_max`), the step taken and the convergence test after it. The closed-loop metrics of
`humanoid_nmpc/humanoid_mpc_validation` sample a solve's step from them, and the equivalence suites' convergence test
reads the iterations of a solve (`test_sqp_live_settings` holds the records to the iterations log).
