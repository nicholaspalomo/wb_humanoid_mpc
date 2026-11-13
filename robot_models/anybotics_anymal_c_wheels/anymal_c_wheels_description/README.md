# anymal_c_wheels_description

The URDF and MuJoCo model of an ANYbotics ANYmal C with a driven wheel at the end of each leg, its meshes and its
license. Pinocchio loads `urdf/anymal_c_wheels.urdf` and the Rerun viewer draws it; the MuJoCo simulator loads
`urdf/anymal_c_wheels.xml`.

`make launch-anymal-c-wheels-sandbox` (`launch/sandbox.textproto`) shows the URDF in Rerun with a slider per joint.
Its centroidal MPC configuration is [`../anymal_c_wheels_centroidal_mpc`](../anymal_c_wheels_centroidal_mpc/README.md).

## Sources and license

| Files | Source | Commit |
| --- | --- | --- |
| `urdf/anymal_c_wheels.urdf` | [LuanDev3/anymal_with_wheels](https://github.com/LuanDev3/anymal_with_wheels), `aww_description/urdf/anymal_with_wheels.urdf` | `bcca224ff5f17017c685682e46ef808393f0d02b` (2022-01-28) |
| `meshes/` | [ANYbotics/anymal_c_simple_description](https://github.com/ANYbotics/anymal_c_simple_description), `meshes/*.dae` | `87b68511622f0f17e78d9ddca7d862f150a93fb4` (2023-05-12) |
| `urdf/anymal_c_wheels.xml` | written for this repository from `urdf/anymal_c_wheels.urdf` | - |

Only the URDF of `anymal_with_wheels` is vendored: none of its meshes (`foot.stl`, `jackal-wheel.stl`), launch files or
other packages. Its URDF is an adapted copy of ANYbotics' ANYmal C description, and `aww_description/LICENSE` is
ANYbotics' BSD-3-Clause text, copyright 2020 ANYbotics AG, while `aww_description/package.xml` says
`<license>TODO</license>`. The `LICENSE` file governs: `LICENSE` here is that file, copied verbatim.
`anymal_c_simple_description`'s `LICENSE` is the same text byte for byte, so the one file covers the URDF, the MJCF
derived from it and the meshes.

## Changes to the URDF

- A comment at the top says that the file is modified and names the upstream file and commit.
- The Gazebo tags (the wheel contact parameters, the leg materials, the `gazebo_ros_control` and controller-manager
  plugins and the Velodyne sensor plugin), the Gazebo XML namespaces of `<robot>` and the 16 ROS `<transmission>`
  elements are removed, with the comments that introduce them.
- The lidar cage and the Velodyne VLP-16 (`lidar_cage`, `velodyne_base_link` and `velodyne`, on the fixed joints
  `base_to_lidar_cage`, `velodyne_base_mount_joint` and `velodyne_base_scan_joint`) are removed. The Velodyne's visuals
  came from `velodyne_description`, which is not vendored. Their inertials are lumped into a new `<inertial>` of `base`:
  1.385 kg (0.545 + 0.83 + 0.01) at (0.364787, 0, 0.142408) m in the base frame, with the inertia about that point
  ixx 4.42720e-3, iyy 3.62949e-3, izz 2.43284e-3 and ixz -1.09083e-5 kg m^2 (ixy = iyz = 0). Upstream `base` has no
  inertial: its mass is on the fixed `base_inertia` link, which is kept.
- The mesh URIs `package://anymal_c_simple_description/meshes/<name>.dae` are
  `package://anymal_c_wheels_description/meshes/<name>.stl`, the decimated meshes below. The commented-out `handle`
  link keeps its upstream URI.
- The visual of each `<leg>_LEG` wheel adapter (`aww_description`'s `foot.stl`) is a cylinder of its adapter collision
  (radius 0.0175 m, length 0.282504 m, centered 0.168985 m above the `LEG` frame), and the visual of each wheel
  (Clearpath Jackal's `jackal-wheel.stl`) is its collision cylinder (radius 0.098 m, length 0.04 m).
- The top-level materials `black` and `orange`, which the wheel and IMU visuals name, are defined.
- The wheel joints `<leg>_wheel` stay `continuous` and get `<limit effort="80.0" velocity="7.5"/>`, the limits of
  the leg joints' drives; upstream gives them none (its Gazebo transmissions drive them by velocity).
- A stray text node (`-`) between two elements of `LH_LEG` is removed.
- The trailing whitespace of its lines is removed, and the file ends with a newline.

Nothing else is changed: links, joints, inertials, collisions and the robot name `aww` are as upstream. Two upstream
values are kept although they look like mistakes:

- `face_rear` weighs 12.73 kg, `face_front` and ANYmal C's `face_rear` 0.73 kg. With it the robot weighs 63.150 kg
  and its center of mass is 81 mm behind the base origin; with 0.73 kg it would weigh 51.150 kg.
- The inertia of `hatch` has every product equal to its moments (all 0.001 kg m^2), which is singular. It is a fixed
  child of `base`, so Pinocchio and the MJCF lump it into the base, whose inertia is positive definite.

## The MuJoCo model

Upstream has no MJCF. A one-off script, outside the repository, wrote `anymal_c_wheels.xml` from the URDF, after the
conventions of [`anymal_b.xml`](../../anybotics_anymal_b/anymal_b_description/README.md):

- One body per moving URDF link (`base`, and per leg `<leg>_HIP`, `<leg>_THIGH`, `<leg>_SHANK` and `<leg>_wheel_link`),
  at that link's frame at zero joint angles. The URDF turns its drive and link frames about the base axes (the HAA
  drives by 150 degrees about x), so the leg bodies are rotated by them and every joint turns about the x axis of
  its body (the wheels about y), with the URDF's axis signs.
- Every body's inertial is the URDF's, with every fixed-joint descendant lumped in as Pinocchio lumps them, written as
  `fullinertia` in the body frame (63.150 kg in total). `robot_models/tests/test_urdf_mjcf_inertial_consistency.py`
  checks them body by body.
- The base is unyawed and keeps the URDF root link's name `base`; its free joint is named `floating_base_joint`.
- The joints have the URDF's names and ranges (`HAA` -0.72 to 0.49 rad on the left legs and -0.49 to 0.72 rad on the
  right, `HFE` and `KFE` +-9.42 rad) and a damping of 2 N m s/rad. The wheel hinges are unlimited
  (`limited="false"`) with a damping of 0.1 N m s/rad, that of a bearing, small so that the wheels roll freely.
- The actuators are one `<motor>` per joint of gear 1, named like the joint, with `ctrlrange` and `forcerange` +-80 N m,
  the URDF's effort limit: the simulator writes each joint's torque into its ctrl.
- The collision geoms are the URDF's boxes and cylinders, with the contact parameters of the other robots' collision
  class (friction 1.0, `solimp="0.95 0.99 0.001"`, `solref="0.004 1.0"`). The URDF's foot sphere of each `LEG` link
  (radius 0.03 m, its center 22.5 mm from the wheel axis) lies inside the wheel's radius and is left out.
- Each wheel body has one collision geom, `<leg>_wheel_collision`: a cylinder coaxial with the wheel joint's axis
  (y), radius 0.098 m and half-width 0.02 m, centered at the wheel center, the origin of the wheel link. It has
  `condim="3"`, a point contact with sliding friction only, and `priority="1"`. A site `<leg>_wheel_center` marks the
  wheel center.
- Contacts between the base and each `THIGH` are excluded, as in `anymal_b.xml`; MuJoCo excludes those between a body
  and its parent.
- An inactive weld `gantry` from the world to `base` is the virtual gantry, which the simulator activates
  (`MujocoSimInterface`, `gantry_hold: "weld_constraint"`).
- The scene has the floor, sky, light and visual settings of `anymal_b.xml`, with the default (Euler) integrator and
  pyramidal cones.
- The visual meshes are the 13 decimated STL files below, each drawn in a single color.

The wheel PD is a velocity damping only (kp 0). The simulator applies it explicitly at its 0.5 ms step, where it stays
stable for a kd below about 2 I / h = 9 N m s/rad, with I = 0.00224 kg m^2 the wheel's effective inertia about its
axis with the base free (0.0024 kg m^2 of the wheel alone). Leg gains of kp 400 N m/rad and kd 10 N m s/rad, those of
ANYmal B, are stable with the scene's damping; `robot_models/tests/test_joint_pd_stability.py` checks the gains of the
MPC package.

## Geometry for the MPC

Measured from the URDF and checked in MuJoCo 4.0.1:

| Quantity | Value |
| --- | --- |
| root link | `base` |
| joints, per leg | `<leg>_HAA`, `<leg>_HFE`, `<leg>_KFE`, `<leg>_wheel` on legs `LF`, `RF`, `LH`, `RH`; the MJCF lists them in that order, and Pinocchio orders the legs by the names of their fixed `base_<leg>_HAA` joints (urdfdom sorts a link's child joints by name): `LF`, `LH`, `RF`, `RH` (Pinocchio 4.1.0 builds it so) |
| mass | 63.150 kg: base 36.988 kg (the lumped lidar included), and per leg HIP 2.781, THIGH 3.071, SHANK 0.48842 and wheel 0.2 kg |
| wheel radius | 0.098 m, the URDF's wheel collision cylinder (length 0.04 m) |
| wheel center rt the wheel joint frame | (0, 0, 0) m: the wheel link's inertial and collision origins |
| wheel joint frame rt the `<leg>_LEG` frame | (0, 0.01, 0) m on `LF` and `LH`, (0, -0.01, 0) m on `RF` and `RH`, identity rotation |
| wheel joint frame rt the `<leg>_SHANK` (KFE) frame | (0.01, -0.102, -0.33797) m on `LF` and `RH`, (0.01, 0.102, -0.33797) m on `RF` and `LH` |
| default posture | HAA 0 on every leg; HFE +0.64 / -0.64 rad and KFE -0.85 / +0.85 rad (front / hind) |
| base height | 0.63588 m: the wheel centers 0.53788 m below the base and 0.098 m above the ground |
| wheel centers rt the base | (+-0.35981, +-0.29811, -0.53788) m, below the HFE joints along x |
| center of mass rt the base | (-0.0809, 0, -0.0506) m |

The HAA angles of the OCS2 `legged_robot` example of ANYmal C (-0.25 / +0.25 rad, left / right) tilt each wheel by
0.25 rad about the base's x axis, so that it would roll on its edge; the default posture keeps the wheels upright with
HAA at 0. Its HFE angle puts each wheel center below its HFE joint for the example's KFE angle of 0.85 rad. Started
from the default posture on the ground with leg joint PD (kp 400, kd 10) and wheel damping (kd 1), the robot stands
on its four wheel cylinders: after 4 s its base has sunk 23 mm and pitched by 0.014 rad, and nothing but the wheels
touches the ground.

## Meshes

The upstream meshes are COLLADA files of 36,268 triangles (in meters, as the URDF uses them). `meshes/` holds one
binary STL per upstream file the URDF uses, 0.9 MB in all. A one-off script, outside the repository, flattened each
COLLADA scene with trimesh 5.2.0 and pycollada 0.9.3 and decimated every mesh of more than 3,000 triangles by quadric
edge collapse (fast-simplification 0.2.0, aggressiveness 4) to 3,000:

| File | Triangles upstream | Triangles shipped |
| --- | --- | --- |
| `base.stl` | 4,276 | 2,999 |
| `top_shell.stl` | 9,543 | 2,998 |
| `bottom_shell.stl` | 10,295 | 3,000 |
| `face.stl` | 6,026 | 2,999 |
| `thigh.stl` | 1,848 | 1,848 |
| `shank_l.stl`, `shank_r.stl` | 1,602 | 1,602 |
| `remote.stl` | 310 | 310 |
| `hip_l.stl`, `hip_r.stl` | 256 | 256 |
| `drive.stl` | 196 | 196 |
| `hatch.stl` | 46 | 46 |
| `battery.stl` | 12 | 12 |

The bounding box of each decimated mesh is within 0.23 mm of the upstream one. The textures (`*.jpg`) and the meshes
the URDF does not use (the cameras, the lidar, the handle, the foot and the lidar cage) are not shipped. The build has
no mesh-processing dependency.
