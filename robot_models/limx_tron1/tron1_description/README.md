# tron1_description

The URDF and MuJoCo model of the LimX Dynamics Tron1 wheel-foot biped (model `WF_TRON1A`), its meshes and its license.
Pinocchio loads `urdf/tron1.urdf` and the Rerun viewer draws it; the MuJoCo simulator loads `urdf/tron1.xml`.

`make launch-tron1-sandbox` (`launch/sandbox.textproto`) shows the URDF in Rerun with a slider per joint. The
centroidal MPC configuration is [`../tron1_centroidal_mpc`](../tron1_centroidal_mpc/README.md).

## Sources and license

| Files | Source | Commit |
| --- | --- | --- |
| `urdf/tron1.urdf`, `meshes/` | [limxdynamics/tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description), `pointfoot/WF_TRON1A/urdf/robot.urdf` and `pointfoot/WF_TRON1A/meshes/*.STL` | `ac417d44656d3c7494a095cecd62928e777a0f3a` (2026-11-05) |
| `urdf/tron1.xml` | the same repository, `pointfoot/WF_TRON1A/xml/robot.xml` | the same commit |

Only `pointfoot/WF_TRON1A` is vendored. The repository is released under the Apache License 2.0; `LICENSE` is its
`LICENSE` file, copied verbatim. Its `package.xml` says `<license>TODO</license>`; the `LICENSE` file governs.
Every vendored file is modified from its upstream file (Apache-2.0 section 4(b)): the URDF and the MJCF open with an
XML comment that says so and names the commit, and the changes are listed below. The meshes are decimated copies of
the upstream STL files.

## Changes to the URDF

- The mesh URIs `package://robot_description/pointfoot/WF_TRON1A/meshes/<name>.STL` are
  `package://tron1_description/meshes/<name>.stl`, the decimated meshes below.
- `wheel_L_Joint` and `wheel_R_Joint` are `continuous` instead of `revolute` with limits of -/+1e6 rad: a wheel spins
  freely. Their `<limit>` keeps the effort (40 N m) and velocity (40 rad/s) and drops `lower` and `upper`.
- A comment at the top says that the file is modified and names the upstream file and commit.
- The trailing whitespace of its lines is removed, and the file ends with a newline.

Nothing else is changed: links, joints, inertials, collisions, the Gazebo tags and the transmissions are as upstream,
and so is the robot name `wheel_biped`.

## The MuJoCo model and its changes

The upstream MJCF is the base of `tron1.xml`: its bodies, joint axes, joint ranges and the collision primitives of the
base and the legs are kept. The changes, after the conventions of
[`anymal_b.xml`](../../anybotics_anymal_b/anymal_b_description/README.md):

- Every body's inertial is the URDF's, written as `fullinertia` in the body frame, with the fixed `limx_imu` link
  (0.01 kg) lumped into `base_Link` as Pinocchio lumps it (upstream: principal moments and axes, the same masses).
  The total is 22.273 kg, which `robot_models/tests/test_urdf_mjcf_inertial_consistency.py` checks body by body.
- The base body keeps the URDF root link's name `base_Link` and is unyawed, as upstream; its free joint is named
  `floating_base_joint` (upstream: unnamed, with zero damping, stiffness and armature written out).
- The joints have the URDF's names and ranges, a damping of 2 N m s/rad on the legs and no armature (upstream: damping
  0.01 and armature 0.01 on every joint, which the URDF and Pinocchio do not have).
- The wheel hinges are unlimited (`limited="false"`, upstream: limited to -/+1e6 rad), with a damping of 0.1 N m s/rad
  and the URDF's friction of 0.01 N m as `frictionloss`. The damping is that of the bearings, small so that the wheels
  roll freely (0.4 N m at 0.5 m/s); it is not what keeps a wheel's PD stable. The wheel PD is a velocity damping only
  (kp 0), explicit at the simulator's 0.5 ms step, and stays stable for a kd below about 2 I / h = 36 N m s/rad, with
  I = 0.00915 kg m^2 the wheel's effective inertia about its axis with the base free (0.00974 kg m^2 of the wheel
  alone); `robot_models/tests/test_joint_pd_stability.py` checks the gains of the MPC package.
- Each wheel body has one collision geom, `wheel_<side>_collision`: a cylinder coaxial with the wheel joint's axis (y),
  radius 0.127 m and half-width 0.025 m, centered at the wheel center (0, +/-0.00707, 0) m in the wheel link frame, as
  the URDF's wheel collision (upstream MJCF: half-width 0.005 m at 0.008 m). It has `condim="3"`, a point contact with
  sliding friction only, and `priority="1"`. A site `wheel_<side>_center` marks the wheel center.
- The collision geoms use the contact parameters of the other robots' collision class (friction 1.0,
  `solimp="0.95 0.99 0.001"`, `solref="0.004 1.0"`; upstream: friction 0.6 and a margin of 1 mm). The knees keep their
  shin cylinder and the hub motor housing around the wheel axle (radius 0.052 m, inside the wheel).
- The actuators are one `<motor>` per joint of gear 1, named like the joint, with `ctrlrange` and `forcerange` the
  URDF's effort: -/+80 N m on the legs and -/+40 N m on the wheels (upstream: the same motors,
  without `forcerange`).
- The upstream sensors (frame orientation, gyro, accelerometer, joint positions on an `imu` site) are removed: the
  simulator reads the state itself. A site `limx_imu` marks the URDF's IMU frame, which is the base frame.
- An inactive weld `gantry` from the world to `base_Link` is the virtual gantry, which the simulator activates
  (`MujocoSimInterface`, `gantry_hold: "weld_constraint"`).
- The scene has the floor, sky, light and visual settings of `anymal_b.xml`, with the default (Euler) integrator and
  pyramidal cones; the upstream `size`, camera and light defaults and the contact visualization settings are removed.
- The visual meshes are the nine decimated STL files below.

## Geometry for the MPC

Measured from the URDF and checked in MuJoCo:

| Quantity | Value |
| --- | --- |
| root link | `base_Link` |
| joints, in the URDF's (and Pinocchio's) order | `abad_L_Joint`, `hip_L_Joint`, `knee_L_Joint`, `wheel_L_Joint`, `abad_R_Joint`, `hip_R_Joint`, `knee_R_Joint`, `wheel_R_Joint` |
| joint axes | abad x; hip y on the left, -y on the right; knee -y on the left, y on the right; wheel y |
| mass | 22.273 kg |
| wheel radius | 0.127 m, the URDF's wheel collision cylinder |
| wheel center rt the wheel joint frame | (0, 0.00707, 0) m on the left, (0, -0.00707, 0) m on the right, the center of the collision cylinder |
| wheel joint frame rt the knee frame | (0.15, +/-0.0435, -0.25981) m |
| default posture | abad 0, hip +0.35 / -0.35 rad and knee +0.8 / -0.8 rad (left / right), wheels 0 |
| base height | 0.7485 m: the wheel centers 0.6215 m below the base and 0.127 m above the ground |

The joint axes of the two sides are mirrored, so a symmetric posture has opposite hip and knee angles on the two sides.
In the default posture the robot's center of mass is 0.6 mm ahead of the wheel centers along x (-2.8 mm and -3.4 mm
in the base frame) and 0.446 m above the ground, where a two-wheeled inverted pendulum balances; at zero joint angles
it is 20 mm ahead of them. Hip angle h and knee angle 2h move the wheel straight up and down below the hip,
because the thigh and the shank have the same length (0.3 m). Started from the default posture on the ground with leg
joint PD alone and no balance control, the robot rests on its two wheel cylinders and tips over within about two
seconds, as an inverted pendulum does.

## Meshes

The upstream meshes are binary STL files of 401,552 triangles and 20 MB, in meters. `meshes/` holds one binary STL per
upstream file, 3.5 MB in all. A one-off script, outside the repository, decimated each mesh by quadric edge collapse
(fast-simplification 0.2.0, aggressiveness 4, loaded and written with trimesh 5.2.0) to its share of 70,000 triangles:

| File | Triangles upstream | Triangles shipped |
| --- | --- | --- |
| `base_Link.stl` | 124,502 | 21,702 |
| `abad_L_Link.stl`, `abad_R_Link.stl` | 33,172, 26,018 | 5,781, 4,534 |
| `hip_L_Link.stl`, `hip_R_Link.stl` | 42,432, 42,512 | 7,396, 7,410 |
| `knee_L_Link.stl`, `knee_R_Link.stl` | 45,460, 47,444 | 7,922, 8,268 |
| `wheel_L_Link.stl`, `wheel_R_Link.stl` | 20,006 | 3,486 |

The bounding box of each decimated mesh is within 0.73 mm of the upstream one. The build has no mesh-processing
dependency.
