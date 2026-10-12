# humanoid_rerun_viewer

The Rerun bridge draws the IPC bus's visualization and status messages in [Rerun](https://rerun.io): the 3D scene of
the robot and the MPC's plan, and the plots of the telemetry. It is a Python process that only subscribes, so it runs on any machine of
the network file, usually the laptop. See also
[`humanoid_nmpc/docs/distributed_runtime/README.md`](../docs/distributed_runtime/README.md), "Visualization with
Rerun".

```text
viz/scene (VisualizationScene) ----> world/robots/<instance>/<link>   link poses, meshes logged once per instance
                                     world/markers/..., world/plan/... Arrows3D, Points3D, LineStrips3D
viz/telemetry (TelemetrySeries) ---> telemetry/<path>                 Scalars, batched every 50 ms
robot/fsm_state, mpc/status, ------> status/...                       TextLog and a few Scalars
robot/loop_timing
```

The bridge knows nothing about the robot beyond its URDF. Everything it draws is computed by the visualization
publisher on the MPC machine. The paths below are the contract between the two: `scene_contract.py`,
`telemetry_contract.py` and `status_contract.py` in `python/humanoid_rerun_viewer/` hold them, and the tables below
list them.

## Running

```bash
bazel run //humanoid_nmpc/humanoid_rerun_viewer -- --urdf robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf
bazel run //humanoid_nmpc/humanoid_rerun_viewer -- --urdf <urdf> --rerun_sink serve_web   # in the dev container
bazel run //humanoid_nmpc/humanoid_rerun_viewer -- --urdf <urdf> --rerun_sink save --rrd_path /tmp/walk.rrd --duration 30
```

Relative paths are looked up in the directory the bridge was started from (`BUILD_WORKING_DIRECTORY` under
`bazel run`), then in the repository root. Ctrl-C or SIGTERM stops it cleanly: the bus is closed, the buffered plots
are sent and the recording is flushed (a second Ctrl-C exits at once). The exit status is 0, or 2 for a usage or
configuration error. The remote_control GUI starts the built binary
(`.bazel/bin/humanoid_nmpc/humanoid_rerun_viewer/humanoid_rerun_viewer`) for its "Open Rerun viewer" button, with its
own `--network_config` and, when the GUI was given `--urdf_file` (every launch file gives it one), `--urdf`, so that
viewer draws the robot, the markers and the plots (remote_control/README.md).

The sink is chosen by name with `--rerun_sink` (`rerun_sinks.SINKS`):

<!-- LINT.IfChange(sink_names) -->
| Sink | What it does |
|---|---|
| `spawn` (default) | starts the native viewer that the rerun-sdk wheel ships (`rerun_cli/rerun`) and streams to it |
| `connect` | streams to a viewer that already runs, at `--rerun_url` |
| `serve_web` | serves the recording over gRPC (`--grpc_port`) and the web viewer over HTTP (`--web_port`); in the dev container open `http://localhost:9090` on the host |
| `save` | writes an `.rrd` file (`--rrd_path`) to open later with `rerun <file>` |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/rerun_sinks.py:sink_names) -->

Every sink sends the blueprint and makes it the active layout, even in a viewer that already shows this application.
It is sent with `send_blueprint()` after the sink is set up, never as the sink's `default_blueprint`: with one,
rerun-sdk 0.38's `serve_grpc()` failed to return in about one start in twelve, and `connect_grpc()` blocks for minutes
when no viewer is up.
When the viewer of `connect` is not there (or a spawned viewer was closed), rerun-sdk waits about 5 s for it at start
and then keeps retrying in the background. At exit the bridge waits at most 5 s for the recording to reach the viewer,
then exits without it (rerun-sdk 0.38 would wait without end).

<!-- LINT.IfChange(sink_flags) -->
| Flag | Default | Meaning |
|---|---|---|
| `--urdf` | none | the robot's URDF; without it no robot is drawn, but the markers and the plots are |
| `--package_path` | | another directory to search for `package://` packages (repeatable) |
| `--network_config` | `config/ipc/network.textproto` | the bus's network file |
| `--rerun_sink` | `spawn` | `spawn`, `connect`, `serve_web` or `save` |
| `--rerun_url` | `rerun+http://127.0.0.1:9876/proxy` | `connect`: the viewer's gRPC URL |
| `--rrd_path` | | `save`: the file to write |
| `--web_port` | `9090` | `serve_web`: the web viewer's port |
| `--grpc_port` | `9876` | `spawn`: the viewer's port; `serve_web`: the recording's port |
| `--open_browser` | off | `serve_web`: also open a browser on this machine |
| `--max_scene_frequency` | `12.5` (`serve_web`) / none | maximum scene messages to log per wall second (0 or none: unlimited) |
| `--plots` / `--no-plots` | off (`serve_web`) / on | subscribe to `viz/telemetry` and show plot tabs in blueprint |
| `--plot_config` | `humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto` | a configuration file (`.textproto` or text file) specifying topics/signals to plot ('none' to disable) |
| `--terminal_state` / `--no-terminal_state` | on | draw the end-of-trajectory robot visualization in `viz/scene` |
| `--robot_instances` | all | comma-separated robot instances to draw (e.g. `measured`, `measured,terminal_state`) |
| `--app_id` | `humanoid_nmpc` | the Rerun application id; recordings of one id share the viewer's layout |
| `--follow_robot` / `--no-follow_robot` | on | the 3D view's eye follows the measured robot's root link |
| `--flush_period` | `0.1` (`serve_web`) / `0.05` | how often the buffered plots are sent [s] |
| `--duration` | `0` | stop after this many seconds; 0 runs until Ctrl-C |
| `--log_level` | `INFO` | of the bridge's own messages |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/cli.py:sink_flags) -->

## The plot configuration

`--plot_config` names a textproto of `humanoid_rerun_viewer_proto.PlotConfig` (`proto/plot_config.proto`; the shipped,
commented `config/plot_config.textproto`), or a plain list of signals, one per line. The launch files pass the shipped one to the bridge (`mpc.textproto`, `dummy_sim.textproto`,
`sandbox.textproto`) and to the robot process (`robot.textproto`); `PLOT_CONFIG=<file>` on a launch target replaces it
for the bridge and, in `make launch-<robot>-sim`, for the robot.

| Field | Default | What |
|---|---|---|
| `signals` | none: no plot tabs | tab titles, panel titles, telemetry and status paths or wildcards to plot; the rest go to a tab of their own |
| `custom_tab_title` | `Signals` | the title of that tab |
| `robot_instances` | all | the robot instances to draw in the 3D scene (`measured`, `terminal_state`) |
| `draw_terminal_state` | true | whether to draw the end-of-trajectory robot (`terminal_state`) |
| `mujoco_viewer` | `web` | the MuJoCo viewer of the robot process, by registry name, or several tried in order: `web` (a browser tab at `http://localhost:6090`), `glfw` (a window on `DISPLAY`), `none` |

`mujoco_viewer` is the robot's, not the bridge's: the robot process reads it from its own `--plot_config` when no
`--viewer` is given (`VIEWER=`, `WB_ROBOT_VIEWER`), so `make launch-<robot>-sim` opens the MuJoCo viewer in a browser
tab by default and `VIEWER=glfw` brings back the window (humanoid_nmpc/humanoid_common_mpc_app/robot/README.md,
`--plot_config`; robot_runtime/mujoco_sim_interface/README.md, "Viewers"). The bridge checks it as the robot does
(`plot_config.check_mujoco_viewer()`: known names, none repeated, nothing after `none` or `glfw`) and refuses a file
with a bad one, which the robot would refuse at start-up too. The robot reads textprotos only, strictly
(`nproto::LoadTextprotoFile()`): it refuses a plain list of signals, unless `VIEWER=` names the viewer and it does not
read the file at all. Without `VIEWER=`, `make launch-<robot>-sim` refuses a `PLOT_CONFIG` whose leading comment block
does not declare `# proto-message: humanoid_rerun_viewer_proto.PlotConfig` before it builds or stops anything
(`tools/deploy/session.sh`), and names the file the robot then reads as `/etc/wb-humanoid-robot/plot_config.textproto`,
the path the robot's errors give.

## What it subscribes to

<!-- LINT.IfChange(bus_defaults) -->
| Topic | Message | Delivery | Drawn as |
|---|---|---|---|
| `viz/scene` | `VisualizationScene` | latest | link poses and markers under `world/` |
| `viz/telemetry` | `TelemetrySeries` | all: every sample is plotted | scalars under `telemetry/`, sent every `--flush_period` (50 ms) as one Arrow record batch |
| `robot/fsm_state` | `FsmState` | latest | a text line on every change |
| `mpc/status` | `MpcStatus` | all: every solve time is plotted | scalars, and a text line when the solver's health or resets change |
| `robot/loop_timing` | `LoopTiming` | all | scalars and one text line per report |

Every 5 s the bridge reports what went wrong since the last report (malformed parts, handler errors, messages the bus
rejected) to `status/bridge` and to its log. At shutdown the last flush waits at most 5 s for the sink.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/bus_bridge.py:bus_defaults) -->

The README of the distributed runtime gives `mpc/status` and `robot/loop_timing` "latest" delivery, for the GUI. The
bridge takes every message of these, because it plots them. ZeroMQ reconnects on its own, so the publishers may start,
stop and restart in any order.

## Time

Every entity is logged on two timelines:

- `robot_time`: the robot's clock, from the message (`VisualizationScene.time`, `TelemetrySeries.time`,
  `MpcStatus.observation_time`). `FsmState` and `LoopTiming` carry no time, so they take the latest robot time the
  bridge has seen. The time panel follows this timeline.
- `wall_time`: the bridge's wall clock when the message arrived.

Rerun's own `log_time` timeline is switched off, because `wall_time` replaces it everywhere, `send_columns` included.
When the robot's clock goes back, for example after a simulation restarts, the scene is logged in full again at the new
times.

## The 3D scene

The 3D view's origin is `world`, the fixed frame. The view coordinates are right-handed with z up, and a line grid
with 1 m cells lies in the xy plane. The eye starts behind and above the origin (`blueprint.initial_eye_position()`)
and follows the measured robot's root link (`--no-follow_robot` keeps it fixed). The blueprint hides what is clutter
most of the time (the "Shown" columns below); the blueprint panel shows it again.

### Robot instances

The bridge logs the URDF's visuals once per instance, as static data under
`world/robots/<instance>/<link>/visual_<i>`. Each visual gets a static `Transform3D` (its origin in the link frame, and
the mesh scale) and its geometry: `Asset3D` for a mesh, `Boxes3D`, `Cylinders3D` or `Ellipsoids3D` for a URDF
primitive. Each `RobotModelInstance` of a scene then logs one `Transform3D` per link at
`world/robots/<instance>/<link>`: the link's world pose, translation and quaternion. Links without visuals are skipped.
A pose equal to the one logged last for the link is skipped too.

<!-- LINT.IfChange(robot_instances) -->
| Instance | Entity | Color | Alpha | Shown |
|---|---|---|---|---|
| `measured` | `world/robots/measured` | the URDF's materials | 0.4 | yes |
| `terminal_state` | `world/robots/terminal_state` | blue tint | 0.4 | yes |
| `terminal_target` | `world/robots/terminal_target` | green tint | 0.3 | hidden |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/scene_contract.py:robot_instances) -->

An instance of another name is drawn in a dark gray tint when it first
appears, and its meshes are logged then.

`package://<package>/<path>` resolves to the nearest directory named `<package>` above the URDF, or one directly
inside a directory above it, or else a directory of that name under `--package_path` or the repository's
`robot_models/`.

<!-- LINT.IfChange(mesh_media_types) -->
Rerun's `Asset3D` draws STL, OBJ, glTF and GLB (`.stl`, `.obj`, `.gltf`, `.glb`, in any case). A mesh in another
format, such as COLLADA (`.dae`), is drawn from a file of the same name in a supported format next to it, when one
exists (Atlas ships both), and is skipped with a warning otherwise. The shipped URDFs all use `.stl` / `.STL`.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/urdf_model.py:mesh_media_types) -->

### Markers

The path of an `Arrows`, `Spheres` or `LineStrips` message is relative to `world`. A radius of 0 and an empty color
list take the marker's defaults below: OCS2's MATLAB-like palette (`ocs2::Color`), arrows with a 0.01 m shaft and a
0.02 m head, lines 0.01 m wide. Rerun draws an arrow's shaft with half its radius and the head with all of it, so the
radius is 0.01. Colors are one for all or one per element. Every scene carries every path it draws: an empty message
draws nothing, and a marker missing from a scene is cleared too.

<!-- LINT.IfChange(markers) -->
| Entity | Message | Default radius [m] | Default colors | Shown |
|---|---|---|---|---|
| `world/markers/contact_forces` | `Arrows` | 0.01 | green | yes |
| `world/markers/center_of_pressure` | `Spheres` | 0.015 | green | yes |
| `world/markers/corner_forces` | `Arrows` | 0.01 | blue | hidden |
| `world/plan/end_effectors` | `LineStrips` | 0.005 | purple, orange, blue, green, yellow per strip | yes |
| `world/plan/base` | `LineStrips` | 0.005 | red | yes |
| `world/plan/com` | `LineStrips` | 0.005 | yellow | yes |
| `world/plan/footholds` | `Spheres` | 0.015 | the contact colors | yes |
| `world/markers/collision_spheres` | `Spheres` | 0.015 | red, alpha 0.5 | hidden |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/scene_contract.py:markers) -->

`Spheres` become `Points3D` with radii, which Rerun draws as shaded spheres. A marker at a path the table does not
list is drawn in dark gray with a warning. `Color.a` is the opacity as sent: proto3 cannot tell an alpha of 0 from an
unset one, so a marker whose colors all have alpha 0 is invisible, and the bridge warns about it once.

## The plots: the telemetry contract

Each `ScalarGroup` of a `TelemetrySeries` becomes the entity `telemetry/<path>` with one series per name. Its
`SeriesLines` (the names, and the colors of the tables below) is logged statically the first time the path is seen,
and again when its number of values changes. The values of a group must match its names one to one; a group whose
names and values differ in length is skipped and counted.

The 6 panel tabs hold 35 panels; the "Base Pose & Euler" tab has a row of the Euler angle rates and a row of the body
angular velocity of the base below the Euler angles. The producer is the C++ visualization publisher of the MPC node
([`humanoid_common_mpc_app/visualization`](../humanoid_common_mpc_app/visualization/README.md)), one message per
`robot/state` sample. The values are SI (m, rad, s, N, N m) in the world frame unless a row says otherwise, and every
source is taken at the sample's time: "measured" is the sample, "reference" the MPC's target trajectory (not the
plan), "plan" the latest MPC policy's state and input, "mpc" the plan's
contact wrenches, "target" the joint action the robot applied. Before the first policy the reference and the plan are
the measured robot at rest, and the plan's wrenches are 0. Left is contact 0 of `ModelSettings::contactNames`, right is
contact 1.

The MPC's state holds the base orientation as a unit quaternion and its velocity as the body angular velocity w_B
(humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 3.4). The plots show what a person reads: the ZYX
Euler angles (roll, pitch, yaw) of the quaternion, their rates `T_B(Theta)^-1 w_B`, and w_B itself, so that the default
plot configuration, which selects the "Base Pose & Euler" tab, shows all three. The Euler angles and rates are display values,
computed by the MPC node's visualization thread and never by the robot; the rates are NaN, a gap in the plot, within
about 1e-6 rad of a pitch of +-90 degrees, where `T_B` is singular.

### Panel groups

There is one small group per panel. Its names are the panel's curves, and it is colored from `palette.py`: measured
blue against a red reference for positions, measured green against an orange reference for
angles and rates. Each panel is plotted from its group alone. The values also appear in the complete groups below. The
duplication costs a few doubles per message, and it keeps every panel independent of the robot's joint names.

<!-- LINT.IfChange(panel_groups) -->
| Path | Names | Unit | Tab / panel | Values |
|---|---|---|---|---|
| `base_pose/position_x` | measured, reference | m | Base Pose & Euler / Base Pos X [m] | measured: root position; reference: base position of the target trajectory |
| `base_pose/position_y` | measured, reference | m | Base Pose & Euler / Base Pos Y [m] | as above |
| `base_pose/position_z` | measured, reference | m | Base Pose & Euler / Base Pos Z (Height) [m] | as above |
| `base_pose/roll` | measured, reference | rad | Base Pose & Euler / Base Roll [rad] | measured: root orientation as roll, pitch, yaw; reference: the target trajectory's |
| `base_pose/pitch` | measured, reference | rad | Base Pose & Euler / Base Pitch [rad] | as above |
| `base_pose/yaw` | measured, reference | rad | Base Pose & Euler / Base Yaw [rad] | as above |
| `base_pose/roll_rate` | measured, reference | rad/s | Base Pose & Euler / Euler Roll Rate [rad/s] | the ZYX Euler angle rates of the body angular velocity, `T_B(Theta)^-1 w_B`, for display; NaN within about 1e-6 rad of a pitch of +-90 degrees. measured: of the sample; reference: of the target trajectory, 0 where it carries no velocity |
| `base_pose/pitch_rate` | measured, reference | rad/s | Base Pose & Euler / Euler Pitch Rate [rad/s] | as above |
| `base_pose/yaw_rate` | measured, reference | rad/s | Base Pose & Euler / Euler Yaw Rate [rad/s] | as above |
| `base_pose/angular_velocity_x` | measured, reference | rad/s | Base Pose & Euler / Body Angular Vel X [rad/s] | the body angular velocity w_B, in the base frame. measured: the sample's `base_angular_velocity_local`; reference: the target trajectory's, Pinocchio's `v[3:6]`, 0 where it carries no velocity |
| `base_pose/angular_velocity_y` | measured, reference | rad/s | Base Pose & Euler / Body Angular Vel Y [rad/s] | as above |
| `base_pose/angular_velocity_z` | measured, reference | rad/s | Base Pose & Euler / Body Angular Vel Z [rad/s] | as above |
| `base_twist/linear_x` | measured, reference | m/s | Base Twist (Linear & Angular) / Linear Vel X [m/s] | measured: root linear velocity; reference: base CoM velocity of the target trajectory |
| `base_twist/linear_y` | measured, reference | m/s | Base Twist (Linear & Angular) / Linear Vel Y [m/s] | as above |
| `base_twist/linear_z` | measured, reference | m/s | Base Twist (Linear & Angular) / Linear Vel Z [m/s] | as above |
| `base_twist/angular_x` | measured, reference | rad/s | Base Twist (Linear & Angular) / Angular Vel X (world) [rad/s] | the root's angular velocity in the world frame, `R(xi) w_B`. measured: of the sample; reference: of the target trajectory, 0 where it carries no velocity |
| `base_twist/angular_y` | measured, reference | rad/s | Base Twist (Linear & Angular) / Angular Vel Y (world) [rad/s] | as above |
| `base_twist/angular_z` | measured, reference | rad/s | Base Twist (Linear & Angular) / Angular Vel Z (world) [rad/s] | as above |
| `contact_forces/left_normal` | mpc, measured | N | Contact Forces (MPC vs Measured) / Left Foot Normal Force Fz [N] | mpc: force z of the plan's contact wrench; measured: the force sensor's |
| `contact_forces/right_normal` | mpc, measured | N | Contact Forces (MPC vs Measured) / Right Foot Normal Force Fz [N] | as above, right |
| `contact_forces/left_tangential` | mpc_x, measured_x, mpc_y, measured_y | N | Contact Forces (MPC vs Measured) / Left Foot Tangential Force Fx/Fy [N] | mpc_x, mpc_y: the policy's; measured_x, measured_y: the force sensor's |
| `contact_forces/right_tangential` | mpc_x, measured_x, mpc_y, measured_y | N | Contact Forces (MPC vs Measured) / Right Foot Tangential Force Fx/Fy [N] | as above, right |
| `generalized_base/position_z` | measured, reference | m | Generalized Coordinates (Pinocchio) / Base Z Height [m] | generalized coordinate base_z; reference: of the target trajectory |
| `generalized_base/pitch` | measured, reference | rad | Generalized Coordinates (Pinocchio) / Base Pitch [rad] | as above, base_pitch: the Euler angle, which the display coordinates hold in place of the quaternion |
| `generalized_base/roll` | measured, reference | rad | Generalized Coordinates (Pinocchio) / Base Roll [rad] | as above, base_roll |
| `generalized_base/velocity_z` | measured, reference | m/s | Generalized Coordinates (Pinocchio) / Base Vz [m/s] | generalized velocity base_z; reference: of the target trajectory |
| `generalized_base/pitch_rate` | measured, reference | rad/s | Generalized Coordinates (Pinocchio) / Base Pitch Rate [rad/s] | as above, base_pitch: the Euler angle rate, not the angular velocity, the values of `base_pose/pitch_rate` |
| `generalized_base/roll_rate` | measured, reference | rad/s | Generalized Coordinates (Pinocchio) / Base Roll Rate [rad/s] | as above, base_roll |
| `foot_kinematics/left_acceleration_z` | measured, reference | m/s^2 | Frame Kinematics & Acceleration / Left Foot Linear Accel Z [m/s^2] | linear z of the classical acceleration of contact 0's frame, LOCAL_WORLD_ALIGNED; reference: of the target trajectory |
| `foot_kinematics/right_acceleration_z` | measured, reference | m/s^2 | Frame Kinematics & Acceleration / Right Foot Linear Accel Z [m/s^2] | as above, contact 1 |
| `foot_kinematics/left_velocity_z` | measured, reference | m/s | Frame Kinematics & Acceleration / Left Foot Twist Z [m/s] | linear z of contact 0's frame velocity, LOCAL_WORLD_ALIGNED; reference: of the target trajectory |
| `foot_kinematics/right_velocity_z` | measured, reference | m/s | Frame Kinematics & Acceleration / Right Foot Twist Z [m/s] | as above, contact 1 |

The three panels of the "Joint Dynamics" tab plot the complete joint groups below: every joint, and the velocity panel
the velocity targets too.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/telemetry_contract.py:panel_groups, //humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/telemetry_contract.py:panel_tabs) -->

### Complete groups

These carry the rest of the telemetry, per joint, per degree of freedom, per contact and per tracked frame, the plan's
included. The orientation quaternions are not plotted: the Euler angles next to them (and the 3D scene) show them. The blueprint plots them in the tabs "Joint Dynamics", "Generalized (all DOFs)", "Frames", "Contact
Wrenches" and "MPC Observation".

<!-- LINT.IfChange(complete_groups) -->
| Path | Names | Unit | Values |
|---|---|---|---|
| `joints/position/measured` | the joints (1) | rad | measured joint position |
| `joints/position/target` | the joints (1) | rad | position target of the joint action applied, the measured position without one |
| `joints/velocity/measured` | the joints (1) | rad/s | measured joint velocity |
| `joints/velocity/target` | the joints (1) | rad/s | velocity target of the joint action applied, 0 without one |
| `joints/effort/measured` | the joints (1) | N m | effort applied: feed-forward + kp (q_des - q) + kd (qd_des - qd), 0 without an action |
| `joints/effort/target` | the joints (1) | N m | feed-forward effort of the joint action applied |
| `dofs/position/measured` | the DOFs (2) | m, rad | generalized coordinates for display: base position, the Euler ZYX angles of the base quaternion, MPC joints |
| `dofs/position/reference` | the DOFs (2) | m, rad | generalized coordinates of the target trajectory |
| `dofs/position/plan` | the DOFs (2) | m, rad | generalized coordinates of the plan |
| `dofs/velocity/measured` | the velocity DOFs (4) | m/s, rad/s | generalized velocities: base world linear velocity, base body angular velocity, joints |
| `dofs/velocity/reference` | the velocity DOFs (4) | m/s, rad/s | generalized velocities of the target trajectory |
| `dofs/velocity/plan` | the velocity DOFs (4) | m/s, rad/s | generalized velocities of the plan |
| `dofs/force/measured` | the velocity DOFs (4) | N, N m | generalized forces, dual to the velocities: 0 for the base, the effort applied for a joint |
| `dofs/force/reference` | the velocity DOFs (4) | N, N m | generalized forces: 0 for the base, the feed-forward effort for a joint |
| `contact_wrenches/left/mpc` | force_x, force_y, force_z, torque_x, torque_y, torque_z | N, N m | the plan's contact wrench |
| `contact_wrenches/left/measured` | force_x, force_y, force_z | N | the force sensor's force |
| `contact_wrenches/right/mpc` | force_x, force_y, force_z, torque_x, torque_y, torque_z | N, N m | the plan's contact wrench |
| `contact_wrenches/right/measured` | force_x, force_y, force_z | N | the force sensor's force |
| `mpc_observation/state` | `x0`, `x1`, ... (3) | | the observation's MPC state |
| `mpc_observation/input` | `u0`, `u1`, ... (3) | | the MPC policy's input at the observation's time |
| `mpc_observation/mode` | mode | | the observation's mode number |
| `frames/pose/<frame>/<source>` | x, y, z, roll, pitch, yaw | m, rad | world position and roll, pitch, yaw |
| `frames/twist/<frame>/<source>` | linear_x, linear_y, linear_z, angular_x, angular_y, angular_z | m/s, rad/s | frame velocity, LOCAL_WORLD_ALIGNED |
| `frames/acceleration/<frame>/<source>` | linear_x, linear_y, linear_z, angular_x, angular_y, angular_z | m/s^2, rad/s^2 | classical acceleration, LOCAL_WORLD_ALIGNED |
| `frames/wrench/<frame>/<source>` | force_x, force_y, force_z, torque_x, torque_y, torque_z | N, N m | measured: the force sensor's wrench for a contact frame; reference: the contact wrench of the target trajectory's input; plan: the plan's; 0 for a frame that is not a contact |

1. The robot's joints, `ModelSettings::fullJointNames` in order.
2. `base_x`, `base_y`, `base_z`, `base_yaw`, `base_pitch`, `base_roll` (`getBaseDofNames()`: the base's Euler angles in
   place of its quaternion), then `ModelSettings::mpcModelJointNames` in order.
3. One per component of the MPC state, or of the MPC input; the state's base orientation is its quaternion's four
   entries.
4. `base_vx`, `base_vy`, `base_vz`, `base_wx`, `base_wy`, `base_wz` (`getBaseVelocityDofNames()`: the world linear and
   the body angular velocity), then `ModelSettings::mpcModelJointNames` in order.

`<frame>` is each frame of the task file's `telemetry_frames` (the contact frames when it lists none), and `<source>` is
`measured`, `reference` (the target trajectory's) or `plan`. The kind comes first in the path so that one view can plot
one kind for every frame. The measured accelerations are differences of consecutive samples; the reference's and the
plan's are central differences along their own trajectories.
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/telemetry_contract.py:complete_groups) -->

With the four frames the shipped task files track, a message carries 32 panel groups and 69 complete groups, 101 in
all. `test/synthetic_messages.py` builds a full message (`full_telemetry()`), which can serve as a reference for a
producer.

## Status

<!-- LINT.IfChange(status_series) -->
| Entity | Archetype | Series | Unit | From |
|---|---|---|---|---|
| `status/robot/fsm_state` | TextLog | | | `FsmState`: one line per change of mode, gantry, MPC health or controller resets; WARN while the MPC is unhealthy |
| `status/mpc/status` | TextLog | | | `MpcStatus`: one line when the solver's health or error changes (ERROR while unhealthy), and one per reset served |
| `status/robot/loop_timing` | TextLog | | | `LoopTiming`: one line per report, DEBUG, or WARN when the loop overran, missed a period or dropped something |
| `status/bridge` | TextLog | | | the bridge: what went wrong since the last report |
| `status/mpc/solve_time` | Scalars | solve_time | ms | `MpcSolverStatus.solve_time_ms` |
| `status/mpc/consecutive_failures` | Scalars | consecutive_failures | | `MpcSolverStatus.consecutive_failures` |
| `status/mpc/observations` | Scalars | received, skipped | | `MpcStatus.observations_received`, `observations_skipped` |
| `status/robot/loop_period` | Scalars | target, mean, max | ms | `LoopTiming.target_period_s`, `mean_period_s`, `max_period_s` |
| `status/robot/compute_time` | Scalars | max | ms | `LoopTiming.max_compute_time_s` |
| `status/robot/wake_up_lateness` | Scalars | max | ms | `LoopTiming.max_lateness_s` |
| `status/robot/loop_events` | Scalars | overruns, missed_periods, telemetry_samples_dropped, stale_policies_dropped | | `LoopTiming` |
| `status/robot/policy_age` | Scalars | policy_age | s | `LoopTiming.policy_age_s`; a gap while no policy is in use |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/status_contract.py:status_series) -->

The "Events" view below the 3D scene shows the text logs; the "Status" tab plots the scalars.

## Robustness

No handler raises.

- A message of another type on a subscribed topic, one that is not three frames, or one that does not parse is
  rejected and counted by the bus.
- A part that does not fit the contract is skipped and counted in `BridgeStatistics.malformed` under its topic and a
  reason, and the rest of the message is drawn. Examples are an invalid path, names and values of different lengths, a
  non-finite coordinate, a zero quaternion, or too many colors.
- An unexpected exception is counted in `BridgeStatistics.handler_errors`.

Each kind of problem is logged at most once every 5 s.

## Cost

`test_bridge_cost` measures the bridge's CPU time per message while it writes an `.rrd` file. It counts every thread
of the process, the Rerun SDK's encoder and writer included. On the development workstation
(`bazel test --test_output=all`):

| Message | CPU per message | At the usual rate |
|---|---|---|
| `TelemetrySeries`, 95 groups (29 joints, 4 frames), 14.4 KB, flushed every 50 ms | 0.8 ms (0.4 ms on the bus thread; parsing 7 us) | 8 % of a core at 100 Hz |
| `VisualizationScene`, 3 instances of a 30-link robot and every marker, 6 KB | 4 ms | 12 % of a core at 30 Hz |

The telemetry is cheap because a flush sends every group of every buffered message as one Arrow record batch
(`scalar_batcher.FrameBatcher`): one `send_columns` call per group cost twice as much (1.5 ms per message). A longer
`--flush_period` lowers the cost further. The scene costs one `Transform3D` per link and instance, about 30 us each in
the SDK; a pose that has not changed is not logged again. A record batch of all transforms was slower, because Rerun
then builds one chunk per link from it. `rerun_scene_frequency` in the task file bounds the scene's rate.

## The model sandbox

`:model_sandbox` draws a robot's URDF without an MPC or a simulator: `make launch-<robot>-sandbox` runs it with this
bridge (`robot_models/<robot>/<robot>_description/launch/sandbox.textproto`).

```bash
bazel run //humanoid_nmpc/humanoid_rerun_viewer:model_sandbox -- --urdf robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf
bazel run //humanoid_nmpc/humanoid_rerun_viewer -- --urdf robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf
```

It computes the world pose of every link itself (`urdf_kinematics.py`: the URDF's joint tree, revolute, continuous,
prismatic, fixed and mimic joints, the root link at the world origin) and publishes them as the `measured` instance of
a `VisualizationScene` on `viz/scene`, as the node `--ipc_node` (default `mpc`: no MPC runs in the sandbox), whenever a
joint moves and again every `--republish_period` (1 s) for a bridge that starts later. The bridge draws it as any
scene. Where the joint positions come from is chosen by name:

<!-- LINT.IfChange(joint_sources) -->
| `--joint_source` | Joint positions |
|---|---|
| `sliders` (default) | a Tk window "Model Sandbox" with one slider per joint (its limits, or one turn where it has none) and a "Nominal pose" button |
| `nominal` | every joint at 0, clamped to its limits |
<!-- LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/model_sandbox.py:joint_sources) -->

The sandbox draws no TF frames, interactive markers or collision geometry.

## Tests

| Target | What it checks |
|---|---|
| `:test_contracts` | every curve of the panel tabs is a series of a group its panel plots, colored from the palette; every group is unique, valid and plotted; the "Base Pose & Euler" tab's rows of Euler angles, Euler rates and body angular velocity, the world-frame titles of the base twist's angular panels, and the names rules of the coordinates and the velocities; this README lists every path |
| `:test_urdf_model` | URDF parsing, `package://` resolution, materials, primitives, `.dae` twins and errors on a sample package; every mesh of every URDF in `robot_models/` resolves |
| `:test_robot_meshes` | the static model of every instance: one `Asset3D` per mesh, primitives, tints and alphas; the EngineAI SA01 in full |
| `:test_blueprint` | the blueprint builds; its tabs, views and hidden entities; it survives a round trip through an `.rrd` file |
| `:test_plot_config` | the plot configuration: textproto and plain-text files, the filtering of tabs, panels and status series, the robot instances, and `mujoco_viewer` (a name or a list, the default `web`, unknown names and lists the robot refuses, the shipped files' `web`); the shipped file plots the Euler angles, the Euler rates and the body angular velocity |
| `:test_scalar_batcher` | rows are sent in one `send_columns` per entity, or as one record batch per flush whose chunks equal those of `send_columns`, on both timelines, in order |
| `:test_bridge` | every handler, driven with synthetic messages and read back from an `.rrd` file: entities, components, clearing, change detection, status lines, malformed parts, a NaN value plotted as a gap |
| `:test_rerun_sinks` | the sink registry, `save`, the viewer binary of the wheel |
| `:test_serve_web_sink` | `serve_web` answering HTTP, in a process of its own |
| `:test_end_to_end` | scene, telemetry and status published with `robot_ipc` on a loopback bus and bridged into an `.rrd` file, a message of the wrong type rejected; the binary stopped with SIGINT and SIGTERM, ended by `--duration`, exiting promptly when its `connect` viewer is not there, and exiting 2 on configuration errors |
| `:test_bridge_cost` | the cost per message above, with a generous bound |
| `:test_urdf_kinematics` | the forward kinematics of a small URDF against hand-computed poses (limits, mimic joints, the bridge's rotation convention); on every shipped URDF at random joint positions, a finite unit pose for every link and rigid joints; the URDFs it refuses |
| `:test_model_sandbox` | the sandbox's positions (clamped, refused when not a joint or not finite), its scene, when it is published, the slider ranges, the binary's G1 scene over a loopback bus, and the slider window (Tk) |
