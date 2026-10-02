"""humanoid_common_mpc_pyutils: Python tools around the MPC.

- mpc_observation_logger: records robot/mpc_observation from the IPC bus into a CSV file (a Bazel binary);
- export_rollouts: turns those CSV files into the demonstration dataset of humanoid_learning's BC warm start;
- mpc_observation_inspector: crops a recording interactively.

The package imports none of them, so that each runs with only its own dependencies.
"""
