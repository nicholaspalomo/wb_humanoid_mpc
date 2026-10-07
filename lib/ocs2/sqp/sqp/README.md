# OCS2 SQP
This package contains a multiple-shooting, sequential-quadratic-programming solver for problems defined with the OCS2 toolbox.

## Dependencies
HPIPM is used as solver for the QP subproblems. HPIPM and Blasfeo are provided by the `@hpipm` and `@blasfeo` Bazel repositories
(bazel/system_libs.bzl); the hpipm_catkin package wraps the HPIPM interface for OCS2.