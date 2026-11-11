# anymal_b_description

The URDF and MuJoCo model of the ANYbotics ANYmal B quadruped, its meshes and its license. Pinocchio loads
`urdf/anymal_b.urdf` for the MPC and the Rerun viewer draws it; the MuJoCo simulator loads `urdf/anymal_b.xml`.

`make launch-anymal-b-sandbox` (`launch/sandbox.textproto`) shows the URDF in Rerun with a slider per joint. The MPC
configuration, its launch targets and the closed-loop results of this model are in
[`../anymal_b_centroidal_mpc`](../anymal_b_centroidal_mpc/README.md) and [`../README.md`](../README.md).

## Sources and license

| Files | Source | Commit |
| --- | --- | --- |
| `urdf/anymal_b.urdf`, `meshes/` | [ANYbotics/anymal_b_simple_description](https://github.com/ANYbotics/anymal_b_simple_description), `urdf/anymal.urdf` and `meshes/*.dae` | `988b5df22b84761bdf08111b1c2ccc883793f456` (2020-11-11) |
| `urdf/anymal_b.xml` | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie), `anybotics_anymal_b/anymal_b.xml` | `370c43965323c0944b433b202aac5058b5f907a9` |

Both are released under the BSD-3-Clause license, copyright 2019 ANYbotics AG. The Menagerie model is derived from the
same URDF and ships the same license text, so the one `LICENSE` file here, ANYbotics' text copied verbatim, covers every
file of the package.

## Changes to the URDF

- The mesh URIs `package://anymal_b_simple_description/meshes/<name>.dae` are `package://anymal_b_description/meshes/<name>.stl`,
  the decimated meshes below.
- The trailing whitespace of its lines is removed (`make format`).

Nothing else is changed: links, joints, inertials, collisions and the Gazebo tags are as upstream.

## The MuJoCo model and its changes

Two MJCF candidates were compared with the URDF, body by body, with the URDF's fixed `ADAPTER` and `FOOT` links lumped
into each `SHANK` as Pinocchio lumps them (the comparison of `robot_models/tests/test_urdf_mjcf_inertial_consistency.py`):
the Menagerie model and the copy in transformer-world-model's `assets/anybotics_anymal_b`, which has the same bodies and
inertials as the Menagerie model and differs only in the type of some collision geoms and in its solver options. Neither
matches the URDF:

| Body | URDF (lumped) | Menagerie |
| --- | --- | --- |
| `base` | 16.7935 kg, center of mass (-0.00196, -0.00141, 0.0502) m | 19.2035 kg, (0.0025, 0, 0.0502) m, inertia 0.46 % off |
| `*_HIP`, `*_THIGH` | 1.42462 kg, 1.63498 kg | the same (inertia within 1.4e-6) |
| `*_SHANK` | 0.347375 kg, center of mass (0.0587, -0.0108, -0.0981) m on `LF` | 0.472163 kg, (0.0696, -0.0132, -0.1509) m, inertia 52 % off |
| total | 30.4214 kg | 33.3306 kg |

The Menagerie model is the base of `anymal_b.xml`: its bodies, joint axes, collision primitives, contact excludes and the
180-degree turn of the hind legs' visuals are kept. The inertials are replaced by the URDF's, so that the simulated robot
is the one the MPC plans for. The changes:

- Every body's inertial is the URDF's, lumped as above, written as `fullinertia` in the body frame (30.4214 kg in total).
- The base is unyawed: Menagerie's `quat="0 0 0 1"`, a 180-degree turn about z, is removed. No child is rotated to
  compensate; at `qpos0` every body frame is its URDF link frame. The base body keeps the URDF root link's name `base`,
  and its free joint is named `floating_base_joint`.
- The position actuators (kp 100, force range +-40 N m) are one `<motor>` per joint of gear 1, named like the joint, with
  `ctrlrange` and `forcerange` +-80 N m, the URDF's effort limit: the simulator writes each joint's torque into its ctrl.
- The joints have the URDF's range, +-9.42 rad, a damping of 2 N m s/rad and no friction loss (Menagerie: no range,
  damping 1, frictionloss 0.1). The MuJoCo integrator treats the damping implicitly
  (`robot_models/tests/test_joint_pd_stability.py`).
- Each foot is one collision sphere of radius 0.031 m in its `SHANK` body, the URDF's foot sphere moved up by 7.75 mm so
  that its lowest point along the shank's z axis is the URDF's `<leg>_FOOT` frame: its center is the `FOOT` frame plus
  (0, 0, 0.031) m. It has `condim="3"`, a point contact with sliding friction only (Menagerie: `condim="6"` and softened
  contact parameters). It is the only sphere of its leg, as the point-contact rule of
  `robot_models/tests/test_default_stance.py` requires.
- A site `<leg>_FOOT` marks the URDF's `FOOT` frame in each `SHANK` body.
- The collision geoms use the contact parameters of the other robots' collision class (friction 1.0, `solimp="0.95
  0.99 0.001"`, `solref="0.004 1.0"`), and the scene has the floor, sky, light and visual settings of
  `robot_models/unitree_r1/unitree_r1_description/urdf/R1.xml`, with the default (Euler) integrator and pyramidal cones.
- An inactive weld `gantry` from the world to `base` is the virtual gantry, which the simulator activates
  (`MujocoSimInterface`, `gantry_hold: "weld_constraint"`).
- The visual meshes are the eight decimated STL files below, one per upstream COLLADA file, each drawn in a single
  color; the Menagerie model's 47 OBJ parts and its two textures are not used.

The `<leg>_FOOT` frames relative to their `<leg>_KFE` joint frames, through the URDF's fixed `SHANK -> ADAPTER -> FOOT`
chain, all with identity rotation:

| Leg | Offset [m] |
| --- | --- |
| `LF` | (0.1, -0.02, -0.32125) |
| `RF` | (0.1, 0.02, -0.32125) |
| `LH` | (-0.1, -0.02, -0.32125) |
| `RH` | (-0.1, 0.02, -0.32125) |

## Meshes

The upstream meshes are COLLADA files of 300,000 triangles and 29 MB (57 MB as Menagerie's OBJ files). `meshes/` holds
one binary STL per upstream file, in the upstream millimeters (both models scale them by 0.001), 3.2 MB in all. A
one-off script, outside the repository, flattened each COLLADA scene with trimesh 5.2.0 and pycollada 0.9.3 and
decimated every sub-mesh by quadric edge collapse (fast-simplification 0.2.0) to its share of the file's target:

| File | Triangles upstream | Triangles shipped |
| --- | --- | --- |
| `anymal_base.stl` | 130,044 | 24,681 |
| `anymal_hip_l.stl`, `anymal_hip_r.stl` | 36,608 | 8,117, 8,125 |
| `anymal_thigh_l.stl`, `anymal_thigh_r.stl` | 38,661, 38,660 | 8,351, 8,310 |
| `anymal_shank_l.stl`, `anymal_shank_r.stl` | 7,959 | 2,999 |
| `anymal_foot.stl` | 4,604 | 2,006 |

The bounding box of each decimated mesh is within 0.7 mm of the upstream one. The build has no mesh-processing
dependency.
