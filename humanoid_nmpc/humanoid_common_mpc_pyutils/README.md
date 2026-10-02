# humanoid_common_mpc_pyutils

Python tools around the MPC.

## Recording MPC observations

`mpc_observation_logger` subscribes to `robot/mpc_observation` on the IPC bus
([`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md)) and writes every
observation, in the order it arrived, to `mpc_observation_<YYYYmmdd_HHMMSS>.csv`:

```bash
bazel run //humanoid_nmpc/humanoid_common_mpc_pyutils:mpc_observation_logger -- \
    --network_config config/ipc/network.textproto --output_dir . --duration 30   # 0: until Ctrl-C
```

The columns are `time, mode, x0, x1, ..., u0, u1, ...`: the MPC state and input as the robot sends them. The names are
generic because the layout depends on the formulation and the robot; the task file's `ModelSettings` say which
component is which. The logger only subscribes, so it runs on any machine of the network file.

## Exporting demonstrations

`make export-rollouts` (`humanoid_common_mpc_pyutils/export_rollouts.py`, system Python) turns the CSV files of the
start directory into `data/mpc_demonstrations.h5`, the dataset `make train-bc` warm-starts from: the state columns are
the observations and the input columns the actions by default (`--obs_cols`, `--act_cols` pick others). Without h5py
it writes an `.npz` file; without any log it writes a synthetic dataset.

## Tests

| Target | What it checks |
|---|---|
| `:test_mpc_observation_logger` | the CSV columns and file name; every observation of a loopback bus reaches the file in order; observations of other dimensions are skipped; `export_rollouts.py` reads the file |
| `:test_export_rollouts` | the default columns of the logger's files and of the ROS-era logger's; explicit columns; every row of every file |

`test/test_xbox_controller.py` is a manual check of an Xbox controller (it needs the xpad driver,
https://github.com/paroj/xpad, and a controller), not a Bazel test.
