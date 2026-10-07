# Robot models

Every robot is a directory of Bazel packages: a description package (`*_description`: the URDF and MuJoCo models in
`urdf/`, the meshes, and the model sandbox `launch/sandbox.textproto`) and one package per MPC that runs it
(`*_centroidal_mpc`, and for the Unitree G1 also `g1_wb_mpc`), which holds the MPC's configuration in `config/`, its
launch files in `launch/` and, for the centroidal MPCs, a Pinocchio playground in `test/testPinocchioModel.cpp`.
`tests/` cross-checks them: the URDF and MJCF inertias agree, the launch files start built binaries with files that
exist, and the filegroups ship no side files.

| Robot | Packages | README |
| --- | --- | --- |
| DRC Atlas | `drc_atlas_description`, `drc_atlas_centroidal_mpc` | [drc_atlas](drc_atlas/README.md) |
| EngineAI SA01 | `engineai_sa01_description`, `engineai_sa01_centroidal_mpc` | [engineai_sa01](engineai_sa01/README.md) |
| Unitree G1 | `g1_description`, `g1_centroidal_mpc`, `g1_wb_mpc` | [unitree_g1](unitree_g1/README.md) |
| Unitree R1 | `unitree_r1_description`, `unitree_r1_centroidal_mpc` | - |

## Configuration files

The hyperparameters of an MPC package are typed textprotos in its `config/`, each one message of
`humanoid_nmpc/humanoid_mpc_config`. That package's README lists the schemas and their conventions, and its `.proto`
files document every field, its unit and its default:

| File | Message | What it sets |
| --- | --- | --- |
| `config/mpc/task.textproto` | `TaskFile` | the formulation, the model, the weights, the constraints and the solver; the settings of the robot process, the visualization and the simulator |
| `config/mpc/contact_planning.textproto` | `ContactPlanningFile` | the online contact planner (Atlas and SA01; it plans while the task file's `contact_schedule_source` is `contact_planner`) |
| `config/command/reference.textproto` | `ReferenceFile` | the command limits and filter, the default base height and the default joint state |
| `config/controller/joint_pd_gains.textproto` | `JointPdGainsFile` | the PD gains and torque limits of the robot's joint controller |

Every robot shares the gait patterns, `humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto` (`GaitFile`).
Each file starts with the schema it is written in (`# proto-file:` and `# proto-message:`), which editors read and
`make lint` checks. The launch files pass the task, reference and gait files to the binaries (`--task_file`,
`--reference_file`, `--gait_file`); the contact planner's file and the PD gains file are found from the task file's
path, in the `config/` layout above.

## Tuning

Edit a file and save it, or move the sliders of the tuning GUI (`humanoid_nmpc/remote_control/README.md`, "Tuning the
configuration").

- **Strict.** A file is parsed strictly: an unknown or misspelled field, a value of the wrong type or a field given
  twice is an error naming the file, line and column. At start-up it stops the process; a running stack refuses the
  edit, logs why, and keeps the values it runs with.
- **By name.** Weights and states name what they weigh, never an index: `{x y z}` blocks
  (`state_weights { base_position { x: 0 y: 0 z: 0 } }`), the base orientation as Euler angles
  (`base_orientation { yaw: 0 pitch: 0 roll: 85 }`), and every joint of the MPC model exactly once, in any order
  (`joint_positions { joint: "back_bkz" value: 5 }`). A block that names a joint the model does not have, or misses
  one, is refused with the joint's name.
- **What applies when.** The schemas mark each field `RELOAD_HOT` or `RELOAD_START_UP`
  (`humanoid_nmpc/humanoid_mpc_config/tuning_options.proto`). A running MPC, centroidal or whole-body, polls its task
  file, its contact planner's file and its reference file about once a second, applies the hot fields, and logs every
  changed start-up field of the task file that it reads as taking effect at its next start. The robot process reloads
  `joint_pd_gains.textproto` and the task file's `contact_estimator` and `contact_wrench_gate` while it runs, from its
  own stored copies (`tools/deploy/README.md`, "The stored configuration"): the bundled files seed them, a deploy
  that changes a file reaches them, and the GUI's **Save** writes them over the bus. In `make launch-<robot>-sim` they
  are seeded from the checkout at every start; an editor's save of a checkout file reaches the running robot process
  at its next start, or at once with `bazel run //humanoid_nmpc/remote_control:push_robot_config -- <file>`
  (`tools/deploy/README.md`, "Simulation").
- **The GUI.** The MPC Parameters, Joint PD Gains and Command Limits tabs are rendered from the schemas, so a field
  added to a schema (with its conversion in C++) appears on its tab without GUI code. A slider moved on the first two
  publishes the whole edited file to the running stack (`operator/mpc_parameters`, `operator/pd_gains`) without
  writing it; `enable_online_tuning: false` in the task file locks both. **Save** writes the edited values into the
  file and keeps every other byte of it - comments, blank lines, LINT directives, the spelling of the values not
  changed - replacing the file atomically, so that no watcher reads half a file; the file as first loaded is kept
  beside it as `<file>.bak`, which git ignores and the robot filegroups leave out (`bazel/robot_files.bzl`). It then
  sends the exact text it wrote of each file the robot reads to the robot process (`operator/config_save`), which
  checks it as it would at start-up, stores it and answers: the tab says whether the robot saved it, found it
  unchanged, refused it (and why), or did not answer
  (`humanoid_nmpc/docs/distributed_runtime/README.md`, "Saving the configuration").
  **Reset All** returns to the file as loaded or last saved, and publishes it.
