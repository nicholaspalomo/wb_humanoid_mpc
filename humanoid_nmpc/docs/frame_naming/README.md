# Frame naming

Every physical quantity in this repository that depends on frames is named by those frames:

```
<quantity>_<frame1>_rt_<frame2>_ewrt_<frame3>
```

`rt` reads "(with) respect to" and `ewrt` "expressed with respect to". `pos_lfoot_rt_world_ewrt_base` is the position of
the left foot's contact frame relative to the world frame, with its components in the base frame's axes.

This document is the specification the rename follows, and the reference afterwards. Its vocabulary (the quantities,
the frames, the qualifiers and the suffixes) lives in one place, `tools/hooks/frame_naming.py`. The tables here list
the same words, and `tools/hooks/test_frame_naming.py` fails when they drift apart. The `frame-naming` check
measures the rename and then enforces it (section 9).

| Section | What |
|---|---|
| 1 | scope: which code and which names |
| 2 | the grammar, and how it combines with the repository's other naming rules |
| 3 | the quantities: definitions, units, component orders |
| 4 | the frames: definitions, robot links, frames chosen at runtime |
| 5 | exemptions: the names that keep their names, and why |
| 6 | ambiguous cases: read the math, never guess |
| 7 | the rename table: frequent identifiers and their new names |
| 8 | the units of the rename, their order, and what each rename agent does |
| 9 | the `frame-naming` check, and switching it on |

## 1. Scope

The convention covers every identifier that names a frame-dependent value, in all of the repository's own code:

- local variables, parameters, struct and class data members, constants, functions and methods;
- Protocol Buffers field names, and therefore the textproto configurations, the Python GUI, the config registries and
  the recorded text logs that name those fields;
- first-party Python;
- the vendored OCS2 in `lib/ocs2`, but not `lib/ocs2/thirdparty` (CppAD, iit).

The code the check reads is listed in section 8 (`DIRS` of `tools/hooks/frame_naming.py`). The lint machinery and the
proto generator (`tools/hooks`, `tools/nproto`) are not read: a `position` there is an offset into a text. Nor is the
code nproto generates (`*.nproto.h`, `*.nproto.pb.h`, `*.nproto.pb.cc`): its names are copied from the `.proto` fields,
which the check reads, and nproto leaves a `.proto` file's NOLINT markers out of it.

The rename is a pure rename. Behavior stays bit-identical: no golden file, expected output or shipped plan is
regenerated, and a proto field keeps its field number (section 8.3).

## 2. The grammar

### 2.1 Words

A name is a sequence of words. Every word of the vocabulary is one lowercase run of letters and digits with no
underscore (`lfoot`, `linvel`, `eulerzyx`), so a name splits back into its words unambiguously in every spelling of
section 2.4. A word is one of:

- a quantity (section 3): `pos`, `linvel`, `quat`, `wrench`, ...;
- a frame (section 4): `world`, `base`, `lfoot`, `contact`, ...;
- the markers `rt` and `ewrt`;
- a qualifier (section 2.5) or a free context word (`swing`, `liftoff`, `stance`, `planned`);
- a suffix (section 2.6).

### 2.2 The four forms

Each quantity has one form, fixed by its row in section 3. The form says which frames a name carries.

| Form | Pattern | Quantities | Why |
|---|---|---|---|
| relative, expressed | `<q>_<f1>_rt_<f2>_ewrt_<f3>` | `pos`, `linvel`, `angvel`, `linacc`, `angacc`, `twist`, `spatialacc`, `classicalacc`, `specforce`, `linmom`, `angmom`, `cmom`, `jac`, `linjac`, `angjac`, `tilt`, `azimuth`, `elevation` | a vector of one frame's motion relative to another, written in the axes of a third |
| relative | `<q>_<f1>_rt_<f2>` | `quat`, `rotmat`, `rotvec`, `eulerzyx`, `rpy`, `pose`, `poseeulerzyx`, `yaw`, `tiltangle`, `dist`, `speed` | an orientation or a transform relates two frames and is not a vector in a third; a norm is frame-invariant |
| applied | `<q>_<f1>_ewrt_<f3>` | `force`, `torque`, `wrench`, `impulse`, `inertia` | a force has no relative frame (Galilean invariance); frame1 is the point it acts at (the moment's reference point) |
| free | `<q>[_<subject>]_ewrt_<f3>` | `normal`, `axis`, `dir`, `gravity` | a free vector has no origin; it belongs to a subject and is written in a frame |

### 2.3 The grammar, exactly

```
name        ::= [ context "_" ] body [ "_" component ] [ "_" container ]
context     ::= word { "_" word }                 (qualifiers in the order of section 2.5, and free context words)
body        ::= quantity "_" frame1 "_rt_" frame2 "_ewrt_" frame3      (relative, expressed)
              | quantity "_" frame1 "_rt_" frame2                       (relative)
              | quantity "_" frame1 "_ewrt_" frame3                     (applied)
              | quantity [ "_" subject ] "_ewrt_" frame3                (free)
frame1      ::= frame | plural                    (a plural names a container indexed by that role, section 2.7)
frame2      ::= frame
frame3      ::= frame
subject     ::= frame | plural | "terrain" | "joint"
component   ::= "x" | "y" | "z" | "xy" | "xz" | "yz"   (vector forms: the components in frame3's axes)
              | "xyzw" | "wxyz"                         (quat only: the order of stored coefficients)
              | "yaw" | "pitch" | "roll"                (eulerzyx and poseeulerzyx only: one angle)
              | "xx" | "yy" | "zz"                      (inertia only: a diagonal entry, in frame3's axes)
container   ::= "traj" | "samples" | "size"        (a time-indexed sequence; samples indexed otherwise; protobuf's `_size`)
```

- frame1 and frame2 differ, except for `angmom` and `cmom`, whose frame1 is the point the moment is taken about:
  `angmom_world_rt_world_ewrt_world` is the angular momentum about the world origin. frame3 may equal either.
- One name holds one quantity: a second `rt` or `ewrt` is an error. A difference of two quantities is the `err`
  qualifier on one name (section 2.5), not two names joined.
- The context never repeats a quantity word or a legacy word (`velocity`, `position`, ...).

Examples: `pos_lfoot_rt_world_ewrt_world`, `quat_base_rt_world`, `wrench_rfoot_ewrt_world`, `normal_terrain_ewrt_world`,
`gravity_ewrt_world`, `des_pos_feet_rt_world_ewrt_world_traj`, `linvel_base_rt_world_ewrt_heading_xy`,
`swing_ref_linvel_foot_rt_world_ewrt_world_z`.

### 2.4 Spelling in each kind of identifier

A new name keeps the case style of the identifier it replaces: the style that kind of identifier already uses in its
language and file (AGENTS.md "Naming"). The grammar's words are the same in every style; only their spelling changes.
Since every word of the vocabulary is one lowercase run, each spelling splits back into the same words
(`nomPosBaseRtComEwrtBase` is nom | pos | base | rt | com | ewrt | base), and the check reads all of them.

| Kind | Style | Example |
|---|---|---|
| C++ local variable, parameter, struct member | camelCase: the first word lowercase, every later word capitalized | `posBaseRtWorldEwrtWorld`, `nomPosBaseRtComEwrtBase` |
| C++ private or protected data member | camelCase, then the trailing `_` | `quatBaseRtWorld_`, `posBaseRtWorldEwrtWorld_` |
| C++ function or method | camelCase: the verb phrase, its context words, then the name's words | `getPosBaseRtWorldEwrtWorld()`, `setLinvelBaseRtWorldEwrtBase()`, `getMnormRateCmomComRtWorldEwrtWorld()` |
| C++ constant (`kName`: `constexpr`, `static constexpr`, `inline constexpr`) | `k`, then every word capitalized | `kNomPosBaseRtComEwrtBase`, `kNomPosBaseRtSurfaceEwrtWorldZ` |
| C++ identifier that is already snake_case (a struct field that mirrors a proto field, a snake_case field of a vendored struct) | snake_case | `sample.pos_base_rt_world_ewrt_world` |
| proto field | snake_case; the generated accessors follow (`mutable_...`, `set_...`, `..._size()`) | `Vector3 pos_base_rt_world_ewrt_world = 3;` |
| Python variable, attribute, parameter, function | snake_case | `get_pos_base_rt_world_ewrt_world()`, `self._quat_base_rt_world` |
| Python module-level constant | UPPER_SNAKE | `NOM_POS_BASE_RT_SURFACE_EWRT_WORLD_Z` |

- The grammar (sections 2.2 and 2.3), the vocabulary tables (sections 3 and 4) and the library mapping of section 3.1
  write names in snake_case, the canonical spelling of the words; in a C++ local, member or function the same words
  are camelCase. The examples in sections 2.5 to 2.9 and 7 are spelled for the kind of identifier they name.
- Never mix the styles in one name: `getPos_base_rt_world_ewrt_world`, `getPosBase_rt_world_ewrt_world` and
  `nomPos_base_rt_com_ewrt_base` are errors, and so is a camelCase or kName frame name in Python or a proto. A C++
  name keeps the style of the name it replaces: `comToBasePositionNominal` becomes `nomPosBaseRtComEwrtBase`, the
  snake_case local `quat_l_to_w` becomes `quat_base_rt_world`. The check cannot see the replaced name, so in C++ it
  accepts camelCase, kName and snake_case and rejects only the mixed spelling; choosing the right one of them is the
  renamer's job.
- A function's context words sit between the verb and the quantity: `getSwingRefPosFootRtWorldEwrtWorld`,
  `getLinearApproximationPosEesRtWorldEwrtWorld`. The quantity always comes before the frames, and nothing follows
  the frames but the suffixes of section 2.6.
- A function that returns or sets a quantity is named for it with the verb the code already uses (`get`, `set`,
  `compute`, `update`, `fill`). A function that reads a quantity out of a container names the quantity it returns,
  not the container (`getPosBaseRtWorldEwrtWorld(state)`).
- An overload set keeps one name when every overload returns the same quantity (AGENTS.md "Overload only when every
  overload has the same meaning").
- Types and classes are not renamed (section 5): `BasePlanarMotion`, `ContactWrenchConeConstraint`.

### 2.5 Qualifiers

A qualifier is a context word with a fixed meaning. Write it exactly as below, in this order when a name has several
(a C++ `initRefPos...`, `cmdMaxLinvel...`, `refErrPos...`, `getMnormRateCmom...`; a proto field or Python
`init_ref_pos_...`, `cmd_max_linvel_...`: the bound of a command is `cmd_max_`, never `max_cmd_`). The check rejects the spelled-out synonyms and an out-of-order pair. Free context words (`swing`, `stance`, `liftoff`, `touchdown`, `planned`,
`terminal`, `offset`, `norm`, `proxy`) may sit anywhere in the context. `proxy` marks a value that stands in for the
quantity it names on some models: getProxyLinvelComRtWorldEwrtWorld() is the CoM velocity a centroidal state stores, or
the base velocity a whole-body state stores instead (section 6).

<!-- LINT.IfChange(qualifiers) -->
| Qualifier | Meaning | Replaces |
|---|---|---|
| `ad` | the CppAD copy of a value whose `scalar_t` copy has the same name in the same scope (section 2.8) | `Ad`, `AD` suffixes |
| `init` | the value at the start of a horizon, trajectory or episode | `initial` |
| `prev` | the value one sample, iteration or knot earlier | `previous`, `last` |
| `final` | the value at the end of a horizon or trajectory | - |
| `nom` | a nominal value: a configured default or a neutral pose | `nominal`, `default` |
| `ref` | a value of the reference the MPC tracks (OCS2 `TargetTrajectories`, the reference manager) | `reference`, `target` (MPC) |
| `des` | a value a planner, heuristic or command layer hands down to be tracked (footholds, swing targets) | `desired`, `target` (planner) |
| `cmd` | a value of an operator or joystick command | `commanded` |
| `meas` | a value a sensor or the simulator's sensors report | `measured`, `measurement` |
| `est` | a value a state estimator computes | `estimated`, `estimate` |
| `true` | the simulator's ground truth, in a test or a cheater estimator | `actual`, `truth` |
| `filt` | a low-pass or otherwise filtered value | `filtered` |
| `min` | a lower bound | `minimum`, `lower` |
| `max` | an upper bound | `maximum`, `upper` |
| `err` | a difference of two values of the same quantity; the comment says which minus which | `error`, `delta` |
| `mnorm` | divided by the robot's total mass (the centroidal model's normalized momentum) | `normalized` |
| `wnorm` | divided by the robot's weight m g (the multi-contact statics' normalized contact wrench: a force per weight, a moment per weight in meters) | `normalized` |
| `lnorm` | divided by its configured limit, a dimensionless fraction (the operator's walking command, a stick deflection in [-1, 1] that the MPC scales by `cmd_max_...`) | `normalized` |
| `rate` | the time derivative of a quantity that has no derivative quantity of its own (C++ `rateYawBaseRtWorld`, `rateQuatBaseRtWorldXyzw`); the derivative of `pos` is `linvel`, never `ratePos` | `dot`, `derivative`, `Rate` suffixes |
<!-- LINT.ThenChange(//tools/hooks/frame_naming.py:qualifiers) -->

### 2.6 Suffixes

<!-- LINT.IfChange(suffixes) -->
After the frames, a name may end in one component and then one container suffix:

- `x`, `y`, `z`, `xy`, `xz`, `yz`: the components of a vector quantity in frame3's axes (relative-expressed, applied and
  free forms): `posBaseRtWorldEwrtWorldZ` (proto `pos_base_rt_world_ewrt_world_z`) is the base height,
  `linvelBaseRtWorldEwrtHeadingXy` the planar velocity in the heading frame;
- `xyzw`, `wxyz`: for `quat` only, the order of the four stored coefficients when the type does not fix it (a
  `vector4_t` slice of the state, a proto `repeated double`, a MuJoCo `qpos` segment); an `Eigen::Quaternion` needs none;
- `yaw`, `pitch`, `roll`: for `eulerzyx` and `poseeulerzyx` only, one of the three angles;
- `xx`, `yy`, `zz`: for `inertia` only, one diagonal entry of the matrix in frame3's axes: `inertiaComEwrtWorldZz` is
  the composite inertia about the vertical through the center of mass;
- `traj`: a time-indexed sequence (an OCS2 `vector_array_t` over the horizon, a list of knots): `...Traj` in camelCase;
- `samples`: a sequence of samples of one quantity indexed by something other than time (a search direction, a
  grid point), whose frame1 has no plural: `supportPosComRtWorldEwrtWorldXySamples`;
- `size`: only in the `_size()` accessor protobuf generates for a repeated field.
<!-- LINT.ThenChange(//tools/hooks/frame_naming.py:suffixes) -->

### 2.7 Plurals and containers

- A container indexed by a generic role (contact index, foot index, end-effector index) names that role in the plural
  as frame1: `std::vector<vector3_t> posContactsRtWorldEwrtWorld`, `feet`, `hands`, `knees`, `hips`, `ankles`,
  `pads`, `ees`, `links`, `surfaces`, `spheres`. Only frame1 is ever plural.
- frame3 (or frame2) that is "each element's own frame" is the singular role: `forceContactsEwrtContact` holds each
  contact's force in that contact's frame.
- A time-indexed container takes the `traj` suffix: `refPosBaseRtWorldEwrtWorldTraj`. A container that is both is
  plural and `traj`: `desPosFeetRtWorldEwrtWorldTraj`.
- One element of a container drops the plural: `posFeetRtWorldEwrtWorld[i]` is read into `posFootRtWorldEwrtWorld`.

### 2.8 CppAD and templated code

A templated or taped value has the same name as its `scalar_t` counterpart: `SCALAR_T` does not change what a
quantity is. Only when both live in one scope does the taped one take the `ad` qualifier:
`adPosBaseRtWorldEwrtWorld`. Do not encode the scalar type in a suffix (`...Ad`, `...CppAd`). Class names such
as `ZeroVelocityConstraintCppAd` are types and keep their names (section 5).

The CppAD model libraries are keyed by model names and the taped math, not by C++ identifiers; a rename does not
change them. Never delete or regenerate them to make a test pass.

### 2.9 Segments of the state and input vectors

- An accessor of a state or input segment whose frames are fixed is named for that quantity:
  `getPosBaseRtWorldEwrtWorld(state)` for `p_W`, `getQuatBaseRtWorldXyzw(state)` for the stored coefficients of
  `xi`, `getAngvelBaseRtWorldEwrtBase(state)` for `w_B`.
- An accessor whose frame depends on the input parameterization takes the frame `param` (section 4): the
  input-only contact accessors of `MpcRobotModelBase` return the world frame for the wrench-space models and the
  local contact frame for `BasisInputsModelDecorator`, so `getContactWrench` is `getWrenchContactEwrtParam`.
- A segment's start index and dimension keep their names (section 5): `getBaseOrientationStartIndex()`,
  `getContactWrenchStartIndices()`, `kPositionDim`, `kQuaternionCoefficients`, `kCentroidalMomentumDim`.

## 3. The quantities

Frame1 is the moving frame (or the point a force acts at), frame2 the frame the motion is relative to (the observer),
and frame3 the frame whose axes the components are written in. "The origin of f" is the origin of frame f.

<!-- LINT.IfChange(quantities) -->
| Quantity | Form | Definition | Unit | Components |
|---|---|---|---|---|
| `pos` | relative, expressed | position of the origin of f1 relative to the origin of f2 | m | 3, in f3 |
| `linvel` | relative, expressed | time derivative of `pos_f1_rt_f2`, as an observer fixed in f2 sees it: the velocity of the origin of f1 | m/s | 3, in f3 |
| `angvel` | relative, expressed | angular velocity of f1 relative to f2 | rad/s | 3, in f3 |
| `linacc` | relative, expressed | time derivative in f2 of `linvel_f1_rt_f2`: the classical acceleration of the origin of f1 (Pinocchio's `getFrameClassicalAcceleration`) | m/s^2 | 3, in f3 |
| `angacc` | relative, expressed | time derivative of `angvel_f1_rt_f2` | rad/s^2 | 3, in f3 |
| `twist` | relative, expressed | `[linvel; angvel]` of f1 relative to f2, linear first: the velocity of the origin of f1 and the angular velocity (Pinocchio's `LOCAL` for f3 = f1, `LOCAL_WORLD_ALIGNED` for f3 = world) | m/s, rad/s | 6, in f3 |
| `spatialacc` | relative, expressed | Pinocchio's spatial acceleration `[a; alpha]`, the derivative of the spatial velocity, linear first; its linear part differs from `linacc` by `angvel x linvel` | m/s^2, rad/s^2 | 6, in f3 |
| `classicalacc` | relative, expressed | Pinocchio's classical acceleration `[linacc; angacc]` of f1 relative to f2, linear first (`getFrameClassicalAcceleration`), read in parts | m/s^2, rad/s^2 | 6, in f3 |
| `specforce` | relative, expressed | specific force at the origin of f1 relative to the inertial f2: `linacc - gravity`, what an accelerometer there reads | m/s^2 | 3, in f3 |
| `linmom` | relative, expressed | linear momentum of the robot (of the body f1 names), velocities relative to f2 | kg m/s | 3, in f3 |
| `angmom` | relative, expressed | angular momentum of the robot about the origin of f1, velocities relative to f2 (`angmom_com_rt_world_ewrt_world` is the centroidal angular momentum, `angmom_world_rt_world_ewrt_world` the one about the world origin) | kg m^2/s | 3, in f3 |
| `cmom` | relative, expressed | the centroidal momentum `[linmom; angmom]`, linear first (Pinocchio's `data.hg`, the centroidal state's `h` with `mnorm`) | kg m/s, kg m^2/s | 6, in f3 |
| `jac` | relative, expressed | the geometric Jacobian whose rows are `twist_f1_rt_f2_ewrt_f3` (linear rows first) and whose columns are the Pinocchio velocity `v` | - | 6 x nv |
| `linjac` | relative, expressed | the three linear rows of `jac` | - | 3 x nv |
| `angjac` | relative, expressed | the three angular rows of `jac` | - | 3 x nv |
| `tilt` | relative, expressed | the tilt of f1 relative to f2: the vector part, axis * sin(angle / 2), of the shortest-arc rotation that turns the z axis of f2 (a surface's normal) onto the z axis of f1; perpendicular to both, zero when they agree (`rotationMatrixDistanceToPlane()`, the stance foot's orientation rows) | - | 3, in f3 |
| `quat` | relative | the unit quaternion of the rotation that maps components in f1 to components in f2, `v_f2 = q v_f1 q*`: the orientation of f1 in f2 (Eigen's `Quaterniond`, the state's `xi`) | - | 4; stored order by suffix |
| `rotmat` | relative | the rotation matrix `R` with `v_f2 = R v_f1`; its columns are the axes of f1 written in f2 (`world_R_pad`) | - | 3 x 3 |
| `rotvec` | relative | the rotation vector (axis times angle, `log(R)`) of the same rotation; the same written in f1 or f2 | rad | 3 |
| `eulerzyx` | relative | ZYX Euler angles `(yaw, pitch, roll)` with `R = Rz(yaw) Ry(pitch) Rx(roll)`; at the Euler boundary only (`euler-boundary`) | rad | 3 |
| `rpy` | relative | the same three angles in the order `(roll, pitch, yaw)`, `R = Rz(yaw) Ry(pitch) Rx(roll)`: the reverse of `eulerzyx`, as a test or a URDF `rpy` attribute writes them; outside the formulation only | rad | 3 |
| `pose` | relative | the rigid transform of f1 in f2, `p_f2 = R p_f1 + t` (`pinocchio::SE3`, `Eigen::Isometry3d`, proto `Pose`): rotation `rotmat_f1_rt_f2`, translation `pos_f1_rt_f2_ewrt_f2` | m, - | SE(3) |
| `poseeulerzyx` | relative | `[pos_f1_rt_f2_ewrt_f2; eulerzyx_f1_rt_f2]` = `[x, y, z, yaw, pitch, roll]`; at the Euler boundary only | m, rad | 6 |
| `yaw` | relative | the twist heading of f1 relative to f2 about f2's z axis: the angle of the twist-swing split (`MpcRobotModelBase::getYawBaseRtWorld()`, design decision D9), in (-pi, pi]; a ZYX yaw is `eulerzyx_..._yaw` | rad | 1 |
| `tiltangle` | relative | the angle between the z axes of f1 and f2, in [0, pi]: twice the arcsine of the norm of `tilt_f1_rt_f2` (the pad's tilt bound of the multi-contact planner) | rad | 1 |
| `dist` | relative | the distance between the origins of f1 and f2, the norm of `pos_f1_rt_f2` | m | 1 |
| `azimuth` | relative, expressed | the angle of `pos_f1_rt_f2_ewrt_f3` about f3's z axis, from f3's x axis | rad (deg where the field's comment says so) | 1 |
| `elevation` | relative, expressed | the angle of `pos_f1_rt_f2_ewrt_f3` out of f3's xy plane | rad (deg where the field's comment says so) | 1 |
| `speed` | relative | the norm of `linvel_f1_rt_f2` | m/s | 1 |
| `force` | applied | the force the environment exerts on the robot at the origin of f1 (a contact's ground reaction, +z up while a foot stands) | N | 3, in f3 |
| `torque` | applied | the moment about the origin of f1 of the forces the environment exerts on the robot there; never a joint torque | N m | 3, in f3 |
| `wrench` | applied | `[force; torque]`, force first, the moment about the origin of f1 (Pinocchio's `Force` order, the input's contact wrench `[f, tau]`) | N, N m | 6, in f3 |
| `impulse` | applied | the linear impulse, the time integral of a `force`, that the environment applies to the robot at the origin of f1 (a dodgeball push) | N s | 3, in f3 |
| `inertia` | applied | the rotational inertia matrix of the robot (of the body the context names) about the origin of f1 (the centroidal model's `nomInertiaComEwrtBase`, Pinocchio's `data.Ig` at the nominal pose) | kg m^2 | 3 x 3, in f3 |
| `normal` | free | the unit normal of the surface the subject names, pointing out of the solid into free space | - | 3, in f3 |
| `axis` | free | a unit axis of the subject: a joint axis or a rotation axis (`axis_joint_ewrt_base`); an axis of a frame is a column of its `rotmat`, and the context word before `axis` says which column (`x_axis_pad_ewrt_link`, `z_axis_base_ewrt_world`); a `ref` axis is a cost's target for that column and may be off unit | - | 3, in f3 |
| `dir` | free | any other unit direction the subject owns (a push direction, a heading direction) | - | 3, in f3 |
| `gravity` | free | the gravitational acceleration vector, `(0, 0, -9.81)` in the world frame; no subject | m/s^2 | 3, in f3 |
<!-- LINT.ThenChange(//tools/hooks/frame_naming.py:quantities) -->

### 3.1 Third-party conventions, mapped

| Library value | Name |
|---|---|
| Pinocchio `getFrameVelocity(..., LOCAL)` | `twist_<f>_rt_world_ewrt_<f>` |
| Pinocchio `getFrameVelocity(..., LOCAL_WORLD_ALIGNED)` | `twist_<f>_rt_world_ewrt_world` |
| Pinocchio `getFrameVelocity(..., WORLD)` | not a `twist`: its linear part is the velocity of the point at the world origin. Convert, or keep the Pinocchio value local and name it `spatial_twist_<f>_rt_world_ewrt_world` with a comment |
| Pinocchio `getFrameAcceleration` / `getFrameClassicalAcceleration` | `spatialacc_...` / `linacc_...` and `angacc_...` |
| Pinocchio `data.oMf[id]` | `pose_<f>_rt_world` |
| Pinocchio `data.com[0]`, `data.vcom[0]` | `pos_com_rt_world_ewrt_world`, `linvel_com_rt_world_ewrt_world` |
| Pinocchio `data.hg`, `data.Ag` | `cmom_com_rt_world_ewrt_world`, the centroidal momentum matrix (a map, keeps `getCentroidalMomentumMatrix`) |
| MuJoCo free joint `qpos[0:3]`, `qpos[3:7]` | `pos_base_rt_world_ewrt_world`, `quat_base_rt_world_wxyz` |
| MuJoCo free joint `qvel[0:3]`, `qvel[3:6]` | `linvel_base_rt_world_ewrt_world`, `angvel_base_rt_world_ewrt_base` |
| MuJoCo `xpos`, `xquat` of a body | `pos_<f>_rt_world_ewrt_world`, `quat_<f>_rt_world_wxyz` |
| The centroidal state `[h, p_W, xi, q_j]` | `mnorm_cmom_com_rt_world_ewrt_world`, `pos_base_rt_world_ewrt_world`, `quat_base_rt_world_xyzw`, joint angles |
| The whole-body state's base velocity `[dp_W, w_B]` | `linvel_base_rt_world_ewrt_world`, `angvel_base_rt_world_ewrt_base` |

## 4. The frames

<!-- LINT.IfChange(frames) -->
| Frame | Definition |
|---|---|
| `world` | the inertial frame, z up against gravity: MuJoCo's world frame, the MPC's and the state estimator's world (odometry) frame |
| `base` | the floating-base root link's frame: the Pinocchio root joint, MuJoCo's free-joint body, `RobotState`'s "root" (table 4.1) |
| `com` | the centroidal frame: origin at the whole-body center of mass, axes aligned with `world` |
| `acom` | the angular center of mass frame: origin at the whole-body center of mass, axes rotated from the base's by the joint-induced rotation, the attitude `xi_a = xi (x) Exp(Delta_theta(q_j))` of `acom/AngularCenterOfMass.h` |
| `crb` | the composite rigid body: the whole robot locked in its current configuration, origin at the center of mass, rotating as one rigid body; its angular velocity `angvel_crb_rt_world_ewrt_world` is I_G^-1 L_G, the locked (average) angular velocity, which no link and not `com` (axes aligned with `world`) has |
| `heading` | origin at the base origin, z along world z, x along world x turned about z by `yaw_heading_rt_world`: the base's twist heading `yaw_base_rt_world` (`BasePlanarMotion`, the velocity command's axes), except in the contact planners with the heading model on, where it is the heading model's yaw (`ContactPlannerInput::yawHeadingRtWorld`, which starts at `initYawHeadingRtWorld`) |
| `torso` | the upper-body link (table 4.1) |
| `imu` | the IMU's sensor frame; on the shipped robots it is the base frame (`RobotState`: the root's orientation and angular velocity are the IMU's), so code reading `RobotState` writes `base` |
| `contact` | a contact frame chosen at runtime by contact index: one of the task file's `contacts_3dof` / `contact_names_6dof` |
| `mjcontact` | a MuJoCo physics contact between two geoms (`mjData::contact[i]`), not one of the controller's contact frames: origin at `mjContact::pos`, x along the contact normal from `geom[0]` to `geom[1]` (the first row of `mjContact::frame`), y and z the two tangents; the frame of `mj_contactForce` |
| `foot` | a foot contact frame chosen at runtime by foot index |
| `lfoot` | the left foot's contact frame, `foot_l_contact`, at the sole |
| `rfoot` | the right foot's contact frame, `foot_r_contact` |
| `hand` | a hand contact frame chosen at runtime |
| `lhand` | the left hand's contact frame, `hand_l_contact` |
| `rhand` | the right hand's contact frame, `hand_r_contact` |
| `knee` | a knee contact frame chosen at runtime |
| `lknee` | the left knee's contact frame, `knee_l_contact` |
| `rknee` | the right knee's contact frame, `knee_r_contact` |
| `hip` | a hip joint frame chosen at runtime (the locomotion heuristics' hip centers) |
| `lhip` | the left hip joint frame |
| `rhip` | the right hip joint frame |
| `ankle` | an ankle joint frame chosen at runtime, the parent joint of a foot contact |
| `lankle` | the left ankle joint frame (`left_ankle_frame`) |
| `rankle` | the right ankle joint frame |
| `pad` | a multi-contact pad's frame, chosen by index: z = -normal_pad_ewrt_link, x = x_axis_pad_ewrt_link (`contact_set.textproto`; `SurfacePad`'s fields of the same names in the task file); also the forearm, thigh, seat and torso pads |
| `ee` | an end-effector frame of OCS2's `EndEffectorKinematics`, chosen by index |
| `link` | any Pinocchio frame chosen at runtime by name or id (`telemetry_frames`, collision bodies) |
| `linkcom` | the center of mass of one body, with the axes of its principal inertia: MuJoCo's body inertial frame (`xipos`, `ximat`), the point `xfrc_applied` acts at and `mj_objectVelocity(mjOBJ_BODY)` measures |
| `subtreecom` | the center of mass of the bodies a joint carries, the joint's subtree (Pinocchio's `data.com[i]` for a joint `i`, MuJoCo's `subtree_com`), axes aligned with `world`; `com` is the whole robot's |
| `site` | a MuJoCo site (`site_xpos`, `site_xmat`), such as a foot force sensor's |
| `surface` | a contact surface or terrain patch: origin on the surface, z along its outward normal (contact planning's candidate rectangles, multi-contact terrain) |
| `box` | a box of a multi-contact scene: origin at its center, axes along its edges; a terrain primitive (`TerrainScene.Box`, `BoxGeometry`), a box of a link's collision model (`CollisionModel.Box`, in its joint's frame) or a scenario's goal region of base positions (`Scenario.BaseRegion`, axis-aligned) |
| `stance` | the reference point of a multi-contact stance's contact wrench generators (`WrenchGenerators::posStanceRtWorldEwrtWorld`): origin at the mean of the active pads' centers, axes aligned with `world` |
| `icp` | the instantaneous capture point, axes aligned with `world` |
| `dcm` | the divergent component of motion, axes aligned with `world` |
| `zmp` | the zero-moment point, axes aligned with `world` |
| `cop` | a contact's center of pressure, axes aligned with `world` |
| `cam` | the viewer's camera frame (web viewer, MuJoCo free camera) |
| `ball` | the dodgeball projectile's body frame |
| `sphere` | the center of a sphere: a collision sphere at a fixed offset from its contact frame (`TerrainNonPenetrationConstraint`) or in its link's frame (`CollisionModel.Sphere`), or a viewer sphere marker (`humanoid_mpc_msgs.Spheres`: collision spheres, centers of pressure, footholds), axes aligned with the frame it is given in |
| `param` | the frame the input vector parameterizes a contact wrench in: `world` for the wrench-space models, the contact frame for `BasisInputsModelDecorator` (section 2.9) |
| `feetmean` | the mean of the foot contact frames' origins (the soles), axes aligned with `world`: what the LIP consumers measure the pendulum length above (`computePosComRtFeetmeanEwrtWorldZ()`) |
| `supportcenter` | the mean of the foot contact frames' origins weighted by the support blend, (w_l p_l + w_r p_r) / (w_l + w_r), axes aligned with `world`: the support center the DCM terminal cost references (`DcmTerminalCost`); equal to `feetmean` only at equal weights |
| `reachcenter` | the point a contact candidate's reach is measured from in the multi-contact planner (`StanceGraphPlanner`, `ContactSampler`): the origin of the limb root's link, or the pad's center for a trunk contact without a limb root, axes aligned with `world` |
| `lowcontact` | the contact frame whose origin is lowest in world z, axes aligned with `world`: the pendulum anchor of a contact set that is not a biped's (`computePosComRtLowcontactEwrtWorldZ()`) |
| `swingend` | the higher in world z of a swing's lift-off and touch-down foot positions, axes aligned with `world`: what the swing apex is set above (`SwingTrajectoryPlanner::Config::apexPosFootRtSwingendEwrtWorldZ`) |
| `corner` | a corner of a contact's support polygon (`ContactPolygon`, `ContactRectangle`), axes aligned with the contact frame |
| `footprint` | the center of a contact's support rectangle (`ContactRectangle`), axes aligned with the contact frame |
| `patchpoint` | the point a contact wrench cone's generators act at (`ContactWrenchConeConstraint::Config`): the configured patch offset, else the footprint center, axes aligned with the contact frame |
| `geom` | a MuJoCo geom's frame: origin at `geom_xpos`, axes `geom_xmat`; or a URDF `<visual>`'s geometry frame, placed in its link by the visual's `<origin>` (`Visual.pose_geom_rt_link`) |
| `generator` | one wrench generator of a multi-contact stance (`WrenchGenerators`): origin at its point of application, a point pad's center or a corner of a surface pad's rectangle, axes aligned with `world` |
| `capsulestart` | a capsule's frame at the start of its axis segment (`TerrainScene.Capsule`, `CollisionModel.Capsule`): origin at the start point, z along the axis toward its end when the two differ |
| `capsuleend` | the end point of a capsule's axis segment, axes aligned with `capsulestart` |
| `query` | a point a function is evaluated at: a geometric query's point (a terrain primitive's or a pad region's distance query), or the point a point Jacobian is taken at (`RobotStanceModel::computeLinjacQueryRtWorldEwrtWorld`); axes aligned with the frame the point is given in |
| `arrow` | a viewer arrow marker's tail (`humanoid_mpc_msgs.Arrows`), axes aligned with `world` |
| `arrowhead` | a viewer arrow marker's head, axes aligned with `world` |
| `segmentpoint` | a point of a capsule's axis segment: the one nearest a query, or one sampled along it; axes aligned with the frame it is given in |
| `controlpoint` | an inner control point of a contact swing's cubic Bezier arc (`ContactSwingTrajectory`): the lift-off or touch-down point moved along its surface's normal, axes aligned with `world` |
| `chordmid` | the midpoint of the straight chord of a swing traverse, from its lift-off to its touch-down, axes aligned with `world` |
| `strippoint` | a point of a viewer polyline (`humanoid_mpc_msgs.LineStrip`), axes aligned with `world` |
| `camfocus` | the viewer's orbit camera's focal point, the point the camera looks at and orbits, axes aligned with `world` |
| `mjbody` | the MuJoCo body a URDF link becomes: that link with every fixed-joint descendant the MJCF does not model as a body of its own lumped into it; origin and axes are the link's |
| `local` | in a generic helper whose frames are its arguments, the body frame the caller passes (`rotateVectorLocalToWorld`): "some frame f" |

Plurals, for a container indexed by the role (frame1 only):

| Plural | Of |
|---|---|
| `contacts` | `contact` |
| `feet` | `foot` |
| `hands` | `hand` |
| `knees` | `knee` |
| `hips` | `hip` |
| `ankles` | `ankle` |
| `pads` | `pad` |
| `ees` | `ee` |
| `links` | `link` |
| `surfaces` | `surface` |
| `spheres` | `sphere` |
| `corners` | `corner` |
| `mjcontacts` | `mjcontact` |
| `generators` | `generator` |
| `arrows` | `arrow` |
| `arrowheads` | `arrowhead` |
| `strippoints` | `strippoint` |

Subjects of a free vector besides the frames:

| Subject | What |
|---|---|
| `terrain` | the ground or terrain under the robot (`normal_terrain_ewrt_world`) |
| `joint` | a joint (`axis_joint_ewrt_base`) |
<!-- LINT.ThenChange(//tools/hooks/frame_naming.py:frames) -->

### 4.1 Robot links and their roles

The names of links are robot-specific and never appear in an identifier; the role does.

| Role | drc_atlas | unitree_g1 | unitree_r1 | engineai_sa01 |
|---|---|---|---|---|
| `base` | `pelvis` | `pelvis` | `pelvis_link` | `base_link` |
| `torso` | `utorso` | `torso_link` | `waist_yaw_link` | - |
| `lfoot`, `rfoot` | `foot_l_contact`, `foot_r_contact` | `foot_l_contact`, `foot_r_contact` | `foot_l_contact`, `foot_r_contact` | `foot_l_contact`, `foot_r_contact` |
| `lhand`, `rhand`, `lknee`, `rknee` | `hand_l_contact`, `hand_r_contact`, `knee_l_contact`, `knee_r_contact` (multi-contact) | - | - | - |

So the C++ `pelvisAngularVelLocal` is `angvelBaseRtWorldEwrtBase`, and the proto field `desired_pelvis_height` is
`cmd_pos_base_rt_surface_ewrt_world_z`.

### 4.2 Frames chosen at runtime

When the frame is a runtime parameter (a contact index, a frame id, a loop over end effectors), name the generic role,
never a guess at the concrete frame: `posContactRtWorldEwrtWorld` inside a loop over contacts, `forceFoot...` in
code that only handles feet. Use a per-side name (`lfoot`) only where the code fixes the side (a `kContactLeftIndex`
literal, a left-only constant).

## 5. Exemptions

<!-- LINT.IfChange(exemptions) -->
These keep their names. The check knows the word lists below (`EXEMPT_WORDS`, `OPERATION_WORDS`,
`THIRD_PARTY_NAMES` of `tools/hooks/frame_naming.py`); anything else that is genuinely not a frame quantity carries
`// NOLINT(frame-naming): <reason>` (`#` in Python).

- Joint-space quantities (joint positions, velocities, accelerations and torques, and the generalized coordinates
  and velocities `q` and `v`) are not frame quantities: `jointPositions`, `joint_velocities`, `computeJointTorques`,
  `getGeneralizedVelocities`, MuJoCo's `qpos` / `qvel`. A joint-space name says `joint` (or `generalized`,
  `configuration`): a bare `torques` vector of joint torques becomes `jointTorques`.
- Dimensions, indices and layout (`kPositionDim`, `kRotationTangentDim`, `kQuaternionCoefficients`,
  `getBaseOrientationStartIndex()`, `getContactWrenchStartIndices()`, `kWrenchForceZIndex`, tuning-layout slots)
  name a place in a vector, not a value (words `dim`, `index`, `indices`, `idx`, `start`, `count`, `num`, `layout`,
  `slot`, `coefficients`).
- Types, classes, enumerators and macros name a concept, not one value (a Python `enum.Enum` member and a
  `Type.NAME` access too): `ContactWrenchConeConstraint`,
  `WalkingVelocityCommand`, `quaternion_t`, `BasePlanarMotion`; an enumerator names a mode or a registry entry
  (`ContactInputParameterization::kWrench`, `ZERO_TORQUE`); a macro is not a value. Their members and the variables of
  those types follow the convention.
- Objects named after their concept hold a model, a configuration, a cost or constraint term, an
  interface, a callback, a registry, a GUI widget, an execution rule, a locomotion heuristic or a message
  (`quaternionModel`, `wrenchConeConfig`, `velocityCommand`, `positionCppAdInterfacePtr_`, `makeWrenchHeuristic`):
  words `model`, `config`, `settings`, `options`, `parameters`,
  `info`, `cost`, `constraint`, `cone`, `generator`, `interface`, `callback`, `registry`, `gui`, `viewer`, `marker`,
  `message`, `msg`, `command`, `topic`, `name`, `color`, `folder`, `gate`, `rule`, `heuristic`, `mode`, `tracking`,
  `enable`, `disable`, `sensor` (a sensor's id or type is not the value it measures: `isFootForceSensor`).
- Weights, gains and noise are tuning knobs, not frame quantities: `positionWeights`, `velocity_gain`,
  `kinematic_velocity_noise`, `baseVelocityRandomWalk` (words `weight`, `gain`, `penalty`, `noise`, and `walk` for a
  random-walk intensity, the process noise of an integrated state). A bound, a nominal value or a limit IS a
  quantity and is renamed (the C++ `cmdMaxLinvelBaseRtWorldEwrtHeadingX`, the config field
  `cmd_max_linvel_base_rt_world_ewrt_heading_x`).
- Generic operations whose frames are their arguments name the operation: the quaternion algebra of
  `ocs2_core/manifold/UnitQuaternionMath.h` (`quaternionProduct`, `quaternionExp`, `quaternionRotationMatrix`), the
  conversions of `orientation/EulerBoundary.h` and `RotationTransforms.h` (`quaternionFromEulerZyx`,
  `eulerZyxFromQuaternion`, `matrixToQuaternion`), `safelyNormalizedQuaternion`, `randomUnitQuaternion` (words `exp`,
  `log`, `product`, `conjugate`, `slerp`, `tangent`, `safe`, `between`, `distance`, `random`, `unit`, and `<a>From<b>` /
  `<a>To<b>` with a representation `<b>`). Their parameters and locals follow the convention with the frame `local`
  where they are body-generic.
- Momentum matrices (the centroidal momentum matrix and its blocks) map velocities to momentum and hold no frame
  pair of their own (`getCentroidalMomentumMatrix`, `getSingleRigidBodyBaseMomentumMatrix`,
  `computeFloatingBaseCentroidalMomentumMatrixInverse`): a name whose only legacy word is `momentum` and that has the
  word `matrix` keeps its name.
- OCS2's solver and problem interface (`getValue`, `getLinearApproximation`,
  `computeFlowMap`, ...) holds no frame words. OCS2 virtual functions that do (`EndEffectorKinematics::getPosition`,
  `getVelocity`, `getOrientationError`) are renamed in the lib/ocs2 unit together with their base and every override,
  in one change.
- Third-party APIs that Pinocchio, Eigen, MuJoCo, CppAD, protobuf's runtime, numpy, scipy, rerun or tkinter
  define (`toRotationMatrix`, `rotation()`, `getFrameVelocity`, `body_pos`, `Rotation.from_quat`) are not ours; `rotation` is
  exempt only as a member (`transform.rotation()`), so a local or parameter named `rotation` is a legacy name; a name
  qualified by `pinocchio::`, `Eigen::`, `std::`, `np.`, `mujoco.`, `rr.` is not read. A member that copies a MuJoCo array keeps that
  array's name, camelCased (`MjRuntimeModelFields::bodyInertia_` is a copy of `mjModel::body_inertia`). The protobuf accessors
  generated from our own fields are ours and follow the field.
- Modules, files and namespaces (`walking_velocity_command_pb2`, `unit_quaternion_internal`, the
  `quaternion_base_orientation` document): Python imports are not read, and a file or namespace name names a concept.
- Test names (gtest `TEST(Suite, Name)` names and Python `test_*` methods) describe behavior.
- Comments and string literals are not read. Registry names in strings (`"zero_velocity"`, `"contact_wrench_cone"`)
  are user-facing names of terms and stay.
- Generic value messages (the fields of a value type whose frames its user names, such as proto `Pose { position,
  orientation }`, `Wrench { force, torque }`, `Vector3`, `Quaternion`) keep their names with
  `// NOLINT(frame-naming): a component of a frame-generic value; the enclosing field names the frames`. The field
  that holds one is named: `Pose pose_base_rt_world = 1;`.
<!-- LINT.ThenChange(//tools/hooks/frame_naming.py:legacy_words) -->

## 6. Ambiguous cases

Where the current code does not make the frames clear, read the math and decide:

1. Find where the value is computed: the Pinocchio call and its reference frame, the rotation it is multiplied by,
   the state slice it comes from. `R * v` with `R = rotmat_base_rt_world` turns `ewrt_base` into `ewrt_world`.
2. Follow it to its consumers, and check the comments and the design documents
   (`humanoid_nmpc/docs/quaternion_base_orientation/README.md` names the frames of every state slice).
3. A test that compares the value with an independently computed one fixes its frames.

Never guess. A wrong frame in a name is worse than a legacy name. Never leave a `TODO(frame-naming)`: resolve the case
in the same change.

- When the value is a different quantity per model, split it into one function per quantity, each named for its own
  frames, and update every caller; each model implements the one it has, or both. The old `getBaseComVelocity`, the
  normalized centroidal momentum in the centroidal model and the base rates `[dp_W, P_3 w_B]` in the whole-body model,
  became the centroidal model's `getMnormCmomComRtWorldEwrtWorld` and the whole-body model's
  `getTuningLayoutBaseRates`.
- When the callers read the per-model value as one quantity, the models still name each quantity they store, and the
  caller says which it takes. The old `getBaseComLinearVelocity`, the CoM's velocity in the centroidal model and the
  base's in the whole-body model, is `getLinvelComRtWorldEwrtWorld` and `getLinvelBaseRtWorldEwrtWorld`, each
  implemented by the model whose state stores it (`getStoredLinvelPoint()`; the other model aborts). The reference
  managers and contact planners, which run a reduced model of the center of mass on either model, read
  `getProxyLinvelComRtWorldEwrtWorld()`: the CoM's velocity, for which the whole-body state's base velocity stands in
  (the `proxy` context word). The contact planner's heading model starts from the ACoM's yaw when the reference manager
  has the ACoM evaluator and from the base's yaw otherwise: the measurements keep their own names
  (`computeYawAcomRtWorld()`, `getYawBaseRtWorld()`), and the planner input names the frame it plans in, the
  `heading` frame of section 4, by its yaw (`yawHeadingRtWorld`, and the heading model's initial state
  `initYawHeadingRtWorld` / `initRateYawHeadingRtWorld`).
- `param` is the one frame word whose frame a configuration chooses: the frame the input parameterizes a contact wrench
  in (section 2.9). No other value takes a model-dependent frame word.
- When the quantity has no word yet, add the word to the vocabulary, with a definition, in
  `tools/hooks/frame_naming.py` and section 3.
- When the math is ambiguous or inconsistent, work it out from the computation and fix it. That includes the code, if
  the frames are actually wrong. A fix that changes behavior ships with a regression test.

`grep -rn "TODO(frame-naming)"` finds nothing but this document, AGENTS.md, and the check that rejects the marker and
its tests (`tools/hooks/frame_naming.py`, `tools/hooks/test_frame_naming.py`, section 9).

## 7. The rename table

Frequent identifiers and their new names, from the code as it was before the rename. Each new name keeps the case
style of the old one (section 2.4): a camelCase C++ name stays camelCase, a snake_case one, a proto field or a Python
name stays snake_case. Grep the old name repo-wide; the
new name holds wherever the old one had the same meaning. Where the same old name means different frames in
different places (a local `position`), decide each by section 6.

| Where | Old | New |
|---|---|---|
| `MpcRobotModelBase` | `getBasePosition(state)` | `getPosBaseRtWorldEwrtWorld(state)` |
| `MpcRobotModelBase` | `setBasePosition(state, position)` | `setPosBaseRtWorldEwrtWorld(state, posBaseRtWorldEwrtWorld)` |
| `MpcRobotModelBase` | `getBaseOrientation(state)` | `getQuatBaseRtWorldXyzw(state)` |
| `MpcRobotModelBase` | `setBaseOrientation(state, orientation)` | `setQuatBaseRtWorldXyzw(state, quatBaseRtWorldXyzw)` |
| `MpcRobotModelBase` | `getBaseHeading(state)` | `getYawBaseRtWorld(state)` |
| `MpcRobotModelBase` | `getBasePoseEulerZyx` / `setBasePoseEulerZyx` | `getPoseeulerzyxBaseRtWorld` / `setPoseeulerzyxBaseRtWorld` |
| `MpcRobotModelBase` | `getBaseOrientationEulerZyx` / `setBaseOrientationEulerZyx` | `getEulerzyxBaseRtWorld` / `setEulerzyxBaseRtWorld` |
| `MpcRobotModelBase` | `getBaseAngularVelocityLocal` / `setBaseAngularVelocityLocal` | `getAngvelBaseRtWorldEwrtBase` / `setAngvelBaseRtWorldEwrtBase` |
| `MpcRobotModelBase` | `setBaseAngularVelocityFromWorldYawRate(state, yawRate)` | `setTurnAngvelBaseRtWorldEwrtBase(state, angvelBaseRtWorldEwrtWorldZ)` |
| `MpcRobotModelBase` | `getBaseAngularVelocityLocalJacobian` | `getJacobianAngvelBaseRtWorldEwrtBase` (d w_B / d state, a map: `jacobian` is a context word here, since `jac` is the geometric Jacobian of section 3) |
| `MpcRobotModelBase` | `getBaseTwistInWorldFrame(state, jointVelocities)` | `getTwistBaseRtWorldEwrtWorld(state, jointVelocities)` |
| `MpcRobotModelBase` | `getBaseComLinearVelocity` / `setBaseComLinearVelocity` | split (section 6): `getLinvelComRtWorldEwrtWorld` / `setLinvelComRtWorldEwrtWorld` (the centroidal state's) and `getLinvelBaseRtWorldEwrtWorld` / `setLinvelBaseRtWorldEwrtWorld` (the whole-body state's), `getStoredLinvelPoint()`, and the consumers' `getProxyLinvelComRtWorldEwrtWorld` / `setProxyLinvelComRtWorldEwrtWorld`; the values read from it: `measProxyLinvelComRtWorldEwrtWorldXy_`, `HorizontalComState::proxyLinvelComRtWorldEwrtWorldXy`, `ContactPlannerInput::proxyLinvelComRtWorldEwrtWorldXy` (the `proxy` word: on the whole body the base velocity) |
| `MpcRobotModelBase` | `getContactWrench` / `getContactForce` / `getContactMoment` | `getWrenchContactEwrtParam` / `getForceContactEwrtParam` / `getTorqueContactEwrtParam` |
| `MpcRobotModelBase` | `setContactWrench` / `setContactForce` / `setContactMoment` | `setWrenchContactEwrtParam` / `setForceContactEwrtParam` / `setTorqueContactEwrtParam` |
| `MpcRobotModelBase` | `getContactWrenchInWorldFrame` / `getContactForceInWorldFrame` / `getContactMomentInWorldFrame` | `getWrenchContactEwrtWorld` / `getForceContactEwrtWorld` / `getTorqueContactEwrtWorld` |
| `MpcRobotModelBase` | `getContactWrenchInWorldFrameGivenRotation(input, contactFrameRotation, i)` | `getWrenchContactEwrtWorld(input, rotmatContactRtWorld, i)` (an overload: the same quantity) |
| `MpcRobotModelBase` | `setContactWrenchInWorldFrame` / `setContactForceInWorldFrame`, `wrenchInWorld`, `forceInWorld` | `setWrenchContactEwrtWorld` / `setForceContactEwrtWorld`, `wrenchContactEwrtWorld`, `forceContactEwrtWorld` |
| `MpcRobotModelBase` | `fillGeneralizedVelocitiesFromMeasurement(..., linearVelocityLocal, angularVelocityLocal, ...)` | `..., linvelBaseRtWorldEwrtBase, angvelBaseRtWorldEwrtBase, ...` |
| ocs2 centroidal `AccessHelperFunctions.h` | `getBasePosition(state, info)` | `getPosBaseRtWorldEwrtWorld(state, info)` |
| ocs2 centroidal | `getBaseQuaternion(state, info)` | `getQuatBaseRtWorldXyzw(state, info)` |
| ocs2 centroidal | `getNormalizedMomentum(state, info)` | `getMnormCmomComRtWorldEwrtWorld(state, info)` |
| ocs2 centroidal | `getContactForces(input, i, info)` / `getContactTorques(input, i, info)` | `getForceContactEwrtWorld(input, i, info)` / `getTorqueContactEwrtWorld(input, i, info)` |
| ocs2 centroidal `ModelHelperFunctions.h` | `getPositionComToContactPointInWorldFrame` | `getPosContactRtComEwrtWorld` |
| ocs2 centroidal | `getNormalizedCentroidalMomentumRate` | `getMnormRateCmomComRtWorldEwrtWorld` |
| ocs2 `EndEffectorKinematics` | `getPosition(state)` / `getVelocity(state, input)` / `getAngularVelocity` / `getTwist` | `getPosEesRtWorldEwrtWorld` / `getLinvelEesRtWorldEwrtWorld` / `getAngvelEesRtWorldEwrtWorld` / `getTwistEesRtWorldEwrtWorld` |
| ocs2 `EndEffectorKinematics` | `getOrientation(state)` | `getQuatEesRtWorld(state)` |
| ocs2 `EndEffectorKinematics` | `getPositionLinearApproximation` | `getLinearApproximationPosEesRtWorldEwrtWorld` (context words before the quantity, never after the frames) |
| ocs2 `EndEffectorKinematics` | `getVelocityLinearApproximation` / `getAngularVelocityLinearApproximation` / `getTwistLinearApproximation` | `getLinearApproximationLinvelEesRtWorldEwrtWorld` / `getLinearApproximationAngvelEesRtWorldEwrtWorld` / `getLinearApproximationTwistEesRtWorldEwrtWorld` |
| ocs2 `EndEffectorKinematics` | `getOrientationErrorWrtPlane` / `getOrientationErrorWrtPlaneLinearApproximation` | `getTiltEesRtSurfaceEwrtWorld` / `getLinearApproximationTiltEesRtSurfaceEwrtWorld` (the quantity `tilt`), `tiltEesCppAdInterfacePtr_`; the CppAD library names (`_orientation_wrt_plane`) stay |
| ocs2 `PinocchioEndEffectorKinematicsCppAd.cpp` | file-local `getPositionCppAd` / `getVelocityCppAd` / `getOrientationCppAd` / `getAngularVelocityCppAd` / `getTwistCppAd`, the lambdas `positionFunc`, ... | `computePosEesRtWorldEwrtWorld` / `computeLinvel...` / `computeQuatEesRtWorldXyzw` / `computeAngvel...` / `computeTwist...` (not `get...`: a member of that name would hide them in the constructor), `tapePosEesRtWorldEwrtWorld`, ...; a stacked value of every end effector is `stackedPosEesRtWorldEwrtWorld`, its state Jacobian `stackedJacobianPosEesRtWorldEwrtWorld` |
| ocs2 centroidal `CentroidalModelInfo` | `comToBasePositionNominal` | `nomPosBaseRtComEwrtBase` |
| ocs2 centroidal | `rotationBaseToWorld`, `comToBasePositionInWorld`, `centroidalMomentumRate` (before the division by the mass) | `rotmatBaseRtWorld`, `posBaseRtComEwrtWorld`, `rateCmomComRtWorldEwrtWorld` |
| ocs2 centroidal | `CentroidalModelInfo::centroidalInertiaNominal` | `nomInertiaComEwrtBase` (the inertia about the CoM at the nominal pose, in base axes; the new quantity `inertia`) |
| ocs2 centroidal | `getCentroidalMomentumMatrix`, `getSingleRigidBodyBaseMomentumMatrix`, `computeFloatingBaseCentroidalMomentumMatrixInverse` | kept: momentum matrices are maps (section 5) |
| ocs2 `RotationDerivativesTransforms.h` | `getEulerAnglesZyxDerivativesFromGlobalAngularVelocity` / `...FromLocalAngularVelocity` | `getEulerAnglesZyxDerivativesFromWorldRates` / `getEulerAnglesZyxDerivativesFromBodyRates` (generic Euler conversions, as `eulerZyxRatesFromBodyRates`); their parameter is `angvelLocalRtWorldEwrtWorld` / `angvelLocalRtWorldEwrtLocal` |
| ocs2 `RotationDerivativesTransforms.h` | `getMappingFromEulerAnglesZyxDerivativeToLocalAngularVelocity`, `getLocalAngularVelocityFromEulerAnglesZyxDerivatives` | `getMappingFromEulerAnglesZyxDerivativesToBodyRates`, `getBodyRatesFromEulerAnglesZyxDerivatives` |
| ocs2 `StateManifold` | `getMaximumRotationAngle`, `hasRepresentableRotations` | `getMaximumTangentAngle`, `hasRepresentableAttitudes` (a property of the manifold's blocks, not of a frame pair) |
| ocs2 `ProductStateManifold` | `getQuaternionAmbientOffsets` / `getQuaternionTangentOffsets`, `mapQuaternionColumns` / `mapQuaternionRows` | `getUnitQuaternionAmbientOffsets` / `getUnitQuaternionTangentOffsets`, `mapUnitQuaternionColumns` / `mapUnitQuaternionRows` (the unit-quaternion blocks: layout) |
| ocs2 `UnitQuaternionMath.h` | `kQuaternionNormalizationEpsilon`, `kQuaternionHalfAngleFactor`, `kQuaternionDoubleAngleFactor` | `kUnitQuaternionNormalizationEpsilon`, `kUnitQuaternionHalfAngleFactor`, `kUnitQuaternionDoubleAngleFactor` (constants of the generic algebra) |
| ocs2 generic helpers | `rotationMatrix` (of Euler angles), `quaternion_correction` (`RotationTransforms.h`), `forceInitialization` (HPIPM, "force" the verb) | `matrix`, `shortestArc`, `alwaysInitialize` |
| ocs2 SQP | `everyStateHasARotation`, `initialRotation` (the warm start's largest tangent angle) | `everyStateHasAnAttitude`, `initialGapAngle` |
| ocs2 tests | `RigidBodyAttitudeDynamics(withPosition)`, `getQuaternionOffset()`, `jumpRotation()`; `testSqpOnSO3`'s `referenceAttitude(t)`, `referenceAngularVelocity(t)`, `twistHeading(x)`, `kHeadingRate` | `withTranslation`, `getQuaternionStartIndex()`, `jumpUnitQuaternion()`; `getRefQuatBaseRtWorldXyzw(t)`, `getRefAngvelBaseRtWorldEwrtBase(t)`, `yawOf(x)`, `kRefRateYawBaseRtWorld` (a toy rigid body is the `base`) |
| `RobotState` | `getRootPositionInWorldFrame` / `setRootPositionInWorldFrame` | `getPosBaseRtWorldEwrtWorld` / `setPosBaseRtWorldEwrtWorld` |
| `RobotState` | `getRootRotationLocalToWorldFrame` / `setRootRotationLocalToWorldFrame` | `getQuatBaseRtWorld` / `setQuatBaseRtWorld` |
| `RobotState` | `getRootLinearVelocityInLocalFrame` / `setRootLinearVelocityInLocalFrame` | `getLinvelBaseRtWorldEwrtBase` / `setLinvelBaseRtWorldEwrtBase` |
| `RobotState` | `getRootAngularVelocityInLocalFrame` / `setRootAngularVelocityInLocalFrame` | `getAngvelBaseRtWorldEwrtBase` / `setAngvelBaseRtWorldEwrtBase` |
| `RobotState` | `getImuLinearAccelerationInLocalFrame`, `imuLinearAcceleration_` | `getSpecforceBaseRtWorldEwrtBase`, `specforceBaseRtWorldEwrtBase_` (the accelerometer at the root's origin) |
| `RobotState` | `rootPosition_`, `rootOrientation_`, `rootLinearVelocity_`, `rootAngularVelocity_` | `posBaseRtWorldEwrtWorld_`, `quatBaseRtWorld_`, `linvelBaseRtWorldEwrtBase_`, `angvelBaseRtWorldEwrtBase_` |
| `JointState`, `RobotState` | `position`, `velocity`, `positionOf`, `velocityOf`, the output parameters `positions` / `velocities` of `getJointPositions` / `getJointVelocities` | `jointPosition`, `jointVelocity`, `jointPositionOf`, `jointVelocityOf`, `jointPositions` / `jointVelocities` (joint space, section 5) |
| `JointDescription`, `JointAction` | `max_velocity`, `getTotalFeedbackTorque` | `max_joint_velocity`, `getTotalFeedbackJointTorque` (joint space; urdfdom's `JointLimits::velocity` keeps its name with a marker) |
| `MujocoSimInterface` | `quat_l_to_w`, `root_vel_lin_world_frame`, `root_vel_ang_local_frame`, `pelvisAngularVelLocal` | `quat_base_rt_world`, `linvel_base_rt_world_ewrt_world`, `angvel_base_rt_world_ewrt_base` (snake_case locals stay snake_case), `angvelBaseRtWorldEwrtBase` |
| `MujocoSimInterface` | `takeMeasuredFootForces(left, right)`, `footSensorForce(sensorId)`, `footForceBuffer_` | `takeMeasForceFeetEwrtWorld(measForceLfootEwrtWorld, measForceRfootEwrtWorld)`, `getMeasForceFootEwrtWorld(sensorId)`, `bufferedMeasForceFeetEwrtWorld_` |
| `MujocoSimInterface` | `setTargetVelocities(vx, vy, yawRate)`, `getTargetVelocityX` / `Y`, `getTargetYawRate`; `BaseVelocityCommand::velocityX` / `velocityY` / `yawRate` (the viewer's command arrows) | `setBaseMotionCommand(cmdLinvelBaseRtWorldEwrtHeadingX, ...Y, cmdAngvelBaseRtWorldEwrtWorldZ)`, `getCmdLinvelBaseRtWorldEwrtHeadingX` / `Y`, `getCmdAngvelBaseRtWorldEwrtWorldZ`; `cmdLinvelBaseRtWorldEwrtHeadingX` / `Y`, `cmdAngvelBaseRtWorldEwrtWorldZ` |
| `MujocoSimInterface::DodgeballThrow` | `spawnOffset`, `launchVelocity` (in the base's yaw frame) | `spawnPosBallRtBaseEwrtHeading`, `launchLinvelBallRtWorldEwrtHeading`; turned into the world: `spawnPosBallRtBaseEwrtWorld`, `launchLinvelBallRtWorldEwrtWorld` |
| `MujocoSimConfig` | `contactForceThreshold`, `kDefaultContactForceThreshold`; `groundTruthContactMask(..., forceThreshold)` | `minForceMjcontactEwrtMjcontactX`, `kNomMinForceMjcontactEwrtMjcontactX` (the normal component of a MuJoCo contact's force, the x axis of `mjcontact`) |
| `MujocoSimInterface` | `setBaseStateForTesting(basePose, baseVelocity)`, `applyTorqueSwitch`, `kWeldRelposeQuaternionWOffset` | `setBaseStateForTesting(baseQpos, baseQvel)` (the free joint's qpos / qvel: joint space), `applyJointTorqueSwitch`, `kWeldRelposeQuaternionWIndex` |
| `MujocoSimInterface` | `rightFootForceSensor_`, `kNoFootForceSensor`, `isFootForceSensor` | kept: a sensor's id or type (`sensor` is an exempt word); `MujocoOptionFlagVisualization::contactForces()` is `contactForceMarkers()` |
| `MujocoUtils.h` | `footForceInWorldFrame`, `verticalTwistOf(w, z)`, `twistHeadingOf(w, z)`, `kMjQuaternionSize` | `computeMeasForceFootEwrtWorld`, `verticalTwistFromQuaternion(w, z)` (its result is `quatHeadingRtWorld`), `yawOf(w, z)`, `kMjQuaternionCoefficients` |
| `MujocoUtils.h` | `RobotCentroidalState::com` / `comVelocity`, `GroundReaction::force` / `moment` / `zmp`, `groundReaction(..., minNormalForce)`, `divergentComponentOfMotion(com, comVelocity, ..., dcm)` | `posComRtWorldEwrtWorld` / `linvelComRtWorldEwrtWorld`, `forceWorldEwrtWorld` / `torqueWorldEwrtWorld` (about the world origin) / `posZmpRtWorldEwrtWorldXy`, `minForceWorldEwrtWorldZ`, `(posComRtWorldEwrtWorld, linvelComRtWorldEwrtWorld, ..., posDcmRtWorldEwrtWorldXy)` |
| MuJoCo contacts | `mj_contactForce`'s output, its normal component, `mjContact::pos`; `mj_objectVelocity`'s output | `wrenchMjcontactEwrtMjcontact`, `forceMjcontactEwrtMjcontactX`, `posMjcontactRtWorldEwrtWorld` (a local alias with a NOLINT, so that the MuJoCo field is read once); `mjTwistLinkcomRtWorldEwrtWorld` (MuJoCo's order, angular first: the `mj` context word says so) |
| MuJoCo structs | `mjvGeom::pos`, `mjContact::pos`, `mjsBody::pos` | MuJoCo's fields: a `NOLINT(frame-naming)` where they are read (a NOLINTBEGIN / NOLINTEND pair in tests) |
| `Projectile.h` | `ProjectileLaunch::position` / `velocity`, `projectileArrivalVelocity(launchVelocity, ...)`, `kProjectileParkPosition` | `posBallRtWorldEwrtWorld` / `linvelBallRtWorldEwrtWorld`, `computeArrivalLinvelBallRtWorldEwrtWorld(launchLinvelBallRtWorldEwrtWorld, ...)`, `kParkPosBallRtWorldEwrtWorld` |
| mujoco_sim_interface tests | `robotMomentum` (the linear momentum and the angular momentum about the world origin) | `computeCmomWorldRtWorldEwrtWorld` (frame1 of `cmom` is the point the moment is taken about, section 2.3) |
| `BasePlanarMotion` | `velocityInHeadingFrame`, `headingRate` | `linvelBaseRtWorldEwrtHeadingXy`, `rateYawBaseRtWorld` |
| state estimator | `contactPositionInBase`, `contactVelocityInBase` | `posContactRtBaseEwrtBase`, `linvelContactRtBaseEwrtBase` |
| state estimator `ContactKinematics` | `ContactPointSpec::offsetInFrame`, `gravity()`; the `LOCAL_WORLD_ALIGNED` frame velocity of the fixed-base model | `posContactRtLinkEwrtLink`, `gravityEwrtWorld()`; `twistLinkRtBaseEwrtBase` (the fixed-base model's universe is the base) |
| state estimator `KalmanBaseEstimator` | `basePosition()`, `baseLinearVelocity()`, `contactPosition(i)`; `reset(basePosition, baseLinearVelocity, orientation, ...)`, `update(dt, orientation, angularVelocityInBase, imuLinearAccelerationInBase, ...)` | `posBaseRtWorldEwrtWorld()`, `linvelBaseRtWorldEwrtWorld()`, `posContactRtWorldEwrtWorld(i)`; `(posBaseRtWorldEwrtWorld, linvelBaseRtWorldEwrtWorld, quatBaseRtWorld, ...)`, `(dt, quatBaseRtWorld, angvelBaseRtWorldEwrtBase, specforceBaseRtWorldEwrtBase, ...)` |
| state estimator `KalmanBaseEstimator.cpp` | the channel-name constants `kBasePosition` / `kBaseVelocity` / `kImuAcceleration`, the channel indices `kBasePositionChannel` / `kBaseVelocityChannel`, the measurement slots `kRelativePositionMeasurement` / `kVelocityMeasurement` / `kHeightMeasurement`, `kInitialPositionStd` | `kStatePosBaseRtWorldEwrtWorld` / `kStateLinvelBaseRtWorldEwrtWorld` / `kInputLinaccBaseRtWorldEwrtWorld` (the input is a = R f + g; the channel strings stay), `kChannelPosBaseRtWorldEwrtWorld` / `kChannelLinvelBaseRtWorldEwrtWorld`, `kSlotMeasPosContactRtBaseEwrtWorld` / `kSlotMeasLinvelBaseRtWorldEwrtWorld` / `kSlotMeasPosContactRtWorldEwrtWorldZ`, `kInitialPositionNoiseStd` (a tuning knob: `noise`) |
| state estimator `KalmanBaseEstimator` | the process-model block pointers `positionVelocityBlock_` / `positionInputBlock_` / `velocityInputBlock_`, the read-back `positionEstimate_` / `velocityEstimate_` | `transitionDtBlock_` / `inputHalfDtSquaredBlock_` / `inputDtBlock_` (matrix blocks, named for their values), `filterStatePosBaseRtWorldEwrtWorld_` / `filterStateLinvelBaseRtWorldEwrtWorld_` |
| state estimator tests | `WalkingSample::basePosition` / `baseVelocity` / `baseAcceleration` / `orientation` / `angularVelocity` / `imuLinearAcceleration` / `footPositions` / `footVelocities`, `kGravity` | `posBaseRtWorldEwrtWorld` / `linvelBaseRtWorldEwrtWorld` / `linaccBaseRtWorldEwrtWorld` / `quatBaseRtWorld` / `angvelBaseRtWorldEwrtBase` / `specforceBaseRtWorldEwrtBase` / `posFeetRtWorldEwrtWorld` / `linvelFeetRtWorldEwrtWorld`, `kGravityEwrtWorld` |
| state estimator tests | the norm of an error, `maxPositionError`, `rmsVelocity`; a generic Kalman filter test's toy input `acceleration` | `normErrPosBaseRtWorldEwrtWorld`, `maxErrPosBaseRtWorldEwrtWorld`, `rmsErrLinvelBaseRtWorldEwrtWorld` (`norm`, `rms`: free context words); `controlInput` (no frame: a toy channel) |
| state estimator tests | `true_base_position`, `base_velocity_estimate` | `true_pos_base_rt_world_ewrt_world`, `est_linvel_base_rt_world_ewrt_world_` (a member; check the frame, section 6) |
| locomotion heuristics | `hipPositionInBaseFrame[foot]` | `posHipsRtBaseEwrtHeadingXy[foot]` (rotated by the nominal base yaw only: heading axes) |
| command | `commandedVelocityInBaseFrame`, `commandedVelocityWorld` | `cmdLinvelBaseRtWorldEwrtHeadingXy` (rotated by the reference yaw only), `cmdLinvelBaseRtWorldEwrtWorld` |
| common | `comVelocity` (a `vector3_t` of the CoM's velocity) | `linvelComRtWorldEwrtWorld` |
| common | `comPosition` | `posComRtWorldEwrtWorld` |
| common | `basePose` (a `vector6_t` `[x, y, z, yaw, pitch, roll]`) | `poseeulerzyxBaseRtWorld` |
| common | `footPositions` (one per foot, world) | `posFeetRtWorldEwrtWorld` |
| common | `footWrenches` (one per foot, world) | `wrenchFeetEwrtWorld` |
| common | `minNormalForce` | `minForceContactEwrtContactZ` |
| `orientation/BaseOrientation.h` | `twistHeading(q)` (the twist-swing heading, not a spatial twist) | `yawOf(q)`, a generic operation returning `yaw_<f1>_rt_<f2>` of `q` |
| constants | `kGravitationalAcceleration` (the scalar 9.81) | `kGravityMagnitude` |
| multi-contact `RobotStanceModel` | `padPosition(i)` / `padRotation(i)` / `centerOfMass()` | `getPosPadRtWorldEwrtWorld(i)` / `getRotmatPadRtWorld(i)` / `getPosComRtWorldEwrtWorld()` |
| multi-contact `RobotStanceModel` | `RobotConfiguration::basePosition` / `baseOrientation`, `centroidalMomentum(q, v)`, `relaxPositionLimits`, `JointLimits`' `relaxesPositionLimits` | `posBaseRtWorldEwrtWorld` / `quatBaseRtWorld`, `computeCmomComRtWorldEwrtWorld(q, v)`, `relaxJointPositionLimits`, `relaxesJointPositionLimits`; the value types `CentroidalMomentum` / `CentroidalMomentumRate` keep their `linear` / `angular` (the field that holds one names the frames: `CertifierKnot::rateCmomComRtWorldEwrtWorld`) |
| multi-contact `ContactSet`, `StanceContact`, the planner's `ContactSample` / `ContactTarget` / `PadTarget` / `ContactPlacement` | `ContactCandidate::padRotation` (parent_R_pad), their `position` (the pad's center in the world) | `rotmatPadRtLink`, `posPadRtWorldEwrtWorld` (the IK's current pad position is `currentPosPadRtWorldEwrtWorld`) |
| multi-contact statics (`StaticEquilibrium`, `ContactWrenchConeCheck`) | `WrenchGenerators::reference` / `wrenches`, `staticWrench(comXy)`, `wrenchResidual(wrench)`, `requiredContactWrench(mass, com, rate, reference)`, `kMomentXRow` | `posStanceRtWorldEwrtWorld` (the new frame `stance`) / `generatorWrenches`, `computeStaticWnormWrenchStanceEwrtWorld(posComRtWorldEwrtWorldXy)`, `computeResidualWnormWrenchStanceEwrtWorld(wnormWrenchStanceEwrtWorld)`, `computeRequiredWnormWrenchStanceEwrtWorld(mass, posComRtWorldEwrtWorld, rateCmomComRtWorldEwrtWorld, posStanceRtWorldEwrtWorld)` (the new qualifier `wnorm`: per body weight), `kMomentXRowIndex` |
| multi-contact torque feasibility and certifier | `minimumTorqueUtilization`, `isTorqueFeasible(result, torqueMargin)`, `CertifierOptions::torqueMargin`, `reportTorques`, `TorqueFeasibility.cpp`'s generalized `forces` | `minimumJointTorqueUtilization`, `isJointTorqueFeasible(result, jointTorqueMargin)`, `jointTorqueMargin`, `reportJointTorques`, `generatorGeneralizedForces` (joint space) |
| proto `MultiContactPlan` | `Knot.base_position` / `base_orientation_xyzw` / `linear_momentum` / `angular_momentum` / `contact_wrenches`, `ContactPhase.target_pad_pose` | `pos_base_rt_world_ewrt_world` / `quat_base_rt_world_xyzw` / `linmom_com_rt_world_ewrt_world` / `angmom_com_rt_world_ewrt_world` / `wrench_contacts_ewrt_world`, `des_pose_pad_rt_world` (the pad's center, then its z axis: a pose up to the yaw about the normal); `ContactWrench.force` / `torque` keep their names with the generic-value marker, and C++ that reads them a `NOLINT(frame-naming)` |
| proto `Scenario` | `Pose.base_position` / `base_orientation_xyzw`, `JointPosition.position`, `Goal.pose`, `initial_pose`, `torque_margin` | `pos_base_rt_world_ewrt_world` / `quat_base_rt_world_xyzw`, `joint_position` (joint space), `posture`, `initial_posture` (a whole-body posture with its contacts, not a frame pose; the message `Pose` keeps its name), `joint_torque_margin`; C++ locals of `config::Scenario::Pose` are `posture` |
| proto `CertifierReport` | `Knot.com`, `Knot.torque_utilization`, `momentum_rate_feasible` / `momentum_rate_residual`, `torque_margin` | `pos_com_rt_world_ewrt_world`, `joint_torque_utilization` (`max_torque_joint` keeps its name), `wrench_cone_feasible` / `wrench_cone_residual` (the contact wrench cone check; `cone` is exempt), `joint_torque_margin`; the playback's Rerun entity paths keep their names |
| proto `TerrainScene` | `Box.orientation_xyzw` | `quat_box_rt_world_xyzw` (the new frame `box`); `BoxGeometry::rotation` and `PatchFrame::rotation` are `rotmatBoxRtWorld` and `rotmatSurfaceRtWorld` |
| multi-contact planner A | `ActivationCandidate::forceOffset`, `ForceGateWeights::forceScale` / `momentArm`, `scaledWrench` / `wrenchScales`, `kMomentumRateFeatures`, `momentumRateRow`, `kMomentumRateTerm`, `QuasiStaticTorqueConstraint::torques` | `wrenchStartIndex` (layout), `scaleForceContactEwrtWorld` / `leverArm`, `computeScaledWrenchContactEwrtWorld` / `computeScaleWrenchContactEwrtWorld`, `kMomentumRateFeatureCount`, `momentumRateRowIndex`, `kMomentumRateTermName`, `jointTorques` |
| multi-contact stance graph and timing | `poseSample` / `poseSeeds` / `waypointPose` (from a waypoint's posture), `contactWrenches(...)`, `SweepResolution::position`, `TimingOptions::maxComAcceleration`, `kPeakSpeed` / `kPeakAcceleration` (of the unit minimum-jerk profile) | `postureSample` / `postureSeeds` / `waypointPosture`, `computeStaticWrenchContactsEwrtWorld(...)`, `baseTranslation`, `maxNormLinaccComRtWorldEwrtWorld`, `kPeakFirstDerivative` / `kPeakSecondDerivative` |
| multi-contact playback (Python) | `pad_positions`, `_world_link_poses`, `FORCE_ARROW_SCALE`, `base_rotation` | `compute_pos_pads_rt_world_ewrt_world`, `_pose_links_rt_world`, `ARROW_LENGTH_PER_NEWTON`, `rotmat_base_rt_world` |
| proto `RobotStateSample` | `base_position_world`, `base_orientation_world` | `pos_base_rt_world_ewrt_world`, `quat_base_rt_world` |
| proto `RobotStateSample` | `base_linear_velocity_local`, `base_angular_velocity_local` | `linvel_base_rt_world_ewrt_base`, `angvel_base_rt_world_ewrt_base` |
| proto `RobotStateSample` | `measured_contact_wrenches` | `meas_wrench_contacts_ewrt_world` |
| proto `WalkingVelocityCommand` | `linear_velocity_x`, `linear_velocity_y` (normalized to [-1, 1]) | `cmd_lnorm_linvel_base_rt_world_ewrt_heading_x`, `cmd_lnorm_linvel_base_rt_world_ewrt_heading_y` (fractions of the reference file's `cmd_max_linvel_...`, the `lnorm` qualifier) |
| proto `WalkingVelocityCommand` | `angular_velocity_z`, `desired_pelvis_height` | `cmd_lnorm_angvel_base_rt_world_ewrt_world_z`, `cmd_pos_base_rt_surface_ewrt_world_z` |
| proto (centroidal state) | `normalized_linear_momentum`, `normalized_angular_momentum` | `mnorm_linmom_com_rt_world_ewrt_world`, `mnorm_angmom_com_rt_world_ewrt_world` |
| proto / Python | `base_orientation_xyzw`, `contact_wrenches` | `quat_base_rt_world_xyzw`, `wrench_contacts_ewrt_world` |
| proto `Pose`, `Wrench` | `position`, `orientation`, `force`, `torque` | kept, with the generic-value marker of section 5 (and a `NOLINT(frame-naming)` where C++ reads `.force` / `.torque` of one) |
| proto `ViewerAnnotations` | `scaled_velocity_x`, `scaled_velocity_y`, `scaled_yaw_rate` (the walking command after the limits) | `scaled_cmd_linvel_base_rt_world_ewrt_heading_x`, `scaled_cmd_linvel_base_rt_world_ewrt_heading_y`, `scaled_cmd_angvel_base_rt_world_ewrt_world_z` (`scaled` a free context word; the yaw rate is the command's `cmd_angvel_..._z` it copies) |
| proto `TargetContactPatch` | `pose` (where the planner puts the foot's contact frame) | `des_pose_foot_rt_world`; the fields `x`, `y`, `z` of its nested `Pose` are components of it and keep their names, and its `yaw` is `eulerzyx_foot_rt_world_yaw` (it holds `TargetContactPose::eulerzyxFootRtWorldYaw`, a ZYX angle, not the twist heading `yaw`) |
| proto `RobotModelInstance` | `link_poses` (the world pose of every link) | `pose_links_rt_world` |
| proto `JointTargets` | `positions` (joint angles by name) | `joint_positions` (joint space, section 5) |
| proto `DodgeballThrow` | `distance`, `speed`, `spawn_offset`, `launch_velocity`, `launch_speed` | `cmd_dist_ball_rt_base`, `cmd_speed_ball_rt_world` (the operator's settings), `spawn_pos_ball_rt_base_ewrt_heading`, `launch_linvel_ball_rt_world_ewrt_heading`, `launch_speed_ball_rt_world`; `impact_momentum`, the norm of the arriving linear momentum, is `impact_norm_linmom_ball_rt_world_ewrt_world` (`norm`: a free context word) |
| humanoid_mpc_ipc tests | `ManifoldScriptedSolver`'s `velocity` (a rate in the toy manifold's tangent space) | `tangentVelocity` (`tangent`: no frame) |
| proto `StateValues`, `StateWeights` (and the tuning layout's segment names) | `normalized_linear_momentum`, `normalized_angular_momentum`, `base_position`, `base_orientation`, `base_linear_velocity`, `base_angular_velocity` | `mnorm_linmom_com_rt_world_ewrt_world`, `mnorm_angmom_com_rt_world_ewrt_world`, `pos_base_rt_world_ewrt_world`, `eulerzyx_base_rt_world` (the `YawPitchRoll` block: the Euler angles in the values, the heading-tilt rows in the weights), `linvel_base_rt_world_ewrt_world`, `angvel_base_rt_world_ewrt_base` (the body rates in (z, y, x)); a weights block is named for the state quantity it weighs, so that both messages and `StateInputLayout`'s segments share one name |
| proto `InputWeights`, `TaskSpaceWeights` | `contact_wrenches { force moment }`; `TaskSpaceWeights`' `pos_x`, `orientation_x`, `lin_velocity_x`, `ang_velocity_x`, `lin_acceleration_x`, `ang_acceleration_x`, ... | `TaskSpaceWeights`' components take the quantity words (`pos_x`, `rotvec_x`, `linvel_x`, `angvel_x`, `linacc_x`, `angacc_x`) and both are kept with a `NOLINT(frame-naming)`: the components of a generic value message (a weight per component), whose frames the enclosing field names: the input parameterization or the task-space cost's link |
| proto `ReferenceFile` | `target_displacement_velocity`, `target_rotation_velocity` | `ref_speed_base_rt_world`, `ref_rate_yaw_base_rt_world` (the speed and yaw rate a base pose command's reference travels at) |
| proto `ReferenceFile` | `max_displacement_velocity_x` / `_y`, `max_rotation_velocity`, `max_delta_pelvis_height` | `cmd_max_linvel_base_rt_world_ewrt_heading_x` / `_y`, `cmd_max_angvel_base_rt_world_ewrt_world_z`, `cmd_max_err_pos_base_rt_surface_ewrt_world_z` (the command's change from the nominal height) |
| proto `ReferenceFile` | `max_linear_acceleration`, `max_angular_acceleration`, `default_base_height` | `cmd_max_linacc_base_rt_world_ewrt_heading_xy` (the norm of the planar rate limit), `cmd_max_angacc_base_rt_world_ewrt_world_z`, `nom_pos_base_rt_surface_ewrt_world_z` |
| proto `ContactPlanningFile` | `hlip.blend.max_commanded_velocity_x` / `_y`, `max_commanded_yaw_rate`, `max_com_velocity_x` / `_y`; `phase_resetting.late_touchdown_search_velocity` | `cmd_max_linvel_base_rt_world_ewrt_heading_x` / `_y`, `cmd_max_angvel_base_rt_world_ewrt_world_z`, `max_proxy_linvel_com_rt_world_ewrt_heading_x` / `_y`; `late_touchdown_search_speed_foot_rt_world` (a descent speed, positive) |
| proto `ContactPlanningFile` | `hlip.step_width`, `step_width.nominal_step_width`; `hlip.` / `foot_separation.max_step_length`, `max_step_width`, `min_step_width`; `reachability.reach_x`, `reach_y_inner`, `reach_y_outer`; `dcm_step_adjustment.max_offset` | `nom_pos_lfoot_rt_rfoot_ewrt_heading_y`; `max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x` (a bound on the absolute value), `max_pos_lfoot_rt_rfoot_ewrt_heading_y`, `min_pos_lfoot_rt_rfoot_ewrt_heading_y`; `max_norm_pos_foot_rt_com_ewrt_heading_x`, `min_pos_foot_rt_com_ewrt_heading_y`, `max_pos_foot_rt_com_ewrt_heading_y` (the left foot's, mirrored for the right); `max_norm_err_pos_foot_rt_world_ewrt_world_xy`, as the C++ `ContactPlanningConfig` names them |
| proto `NominalFootholdConfig` (and `ModelSettings::NominalFootholdConfig::stepWidth`) | `step_width` | `nom_pos_lfoot_rt_rfoot_ewrt_heading_y` (C++ `nomPosLfootRtRfootEwrtHeadingY`) |
| proto `LocomotionHeuristicsConfig` | `orientation_compensation_parameters.maximum_tilt`, `height_compensation.maximum_height_offset` | `max_offset_eulerzyx_base_rt_world` (each of the ZYX roll and pitch offsets), `max_offset_pos_base_rt_world_ewrt_world_z`, as the C++ heuristic parameters name them |
| proto `TaskFile` | `sim_gantry_catch_lift` | `sim_gantry_catch_err_pos_base_rt_world_ewrt_world_z` (C++ `SimFallRecovery::Config::catchErrPosBaseRtWorldEwrtWorldZ`: the lifted minus the standing gantry height) |
| proto `ContactsConfig` | `contact_frame_translation`, `WrenchCone.min_normal_force`, `WrenchCone.gripper_force` | `pos_contact_rt_link_ewrt_link` (the contact frame in its parent joint's frame), `min_force_contact_ewrt_contact_z`, `grip_force_contact_ewrt_contact_z` (the adhesion the cone admits); `SurfacePad.translation`, `pad_normal`, `pad_x_axis` are `pos_pad_rt_link_ewrt_link`, `normal_pad_ewrt_link`, `x_axis_pad_ewrt_link` (same field numbers), the names of the `ContactSet.Candidate` fields that define the same pad frame |
| proto `humanoid_multi_contact.ContactSet.Candidate`, C++ `ContactCandidate` | `pad_normal`, `pad_x_axis`; `padNormal` | `normal_pad_ewrt_link`, `x_axis_pad_ewrt_link` (same field numbers); `normalPadEwrtLink` |
| proto `ContactImplicitConfig` | `height_reference`, `velocity_reference`, `angular_velocity_reference` | `scale_pos_contact_rt_surface_ewrt_world_z`, `scale_linvel_contact_rt_world_ewrt_world_xy`, `scale_angvel_contact_rt_world_ewrt_world_z` (`scale`: the normalizers of the complementarity gap and of the slip residual) |
| proto `SwingTrajectoryConfig` | `lift_off_velocity`, `touch_down_velocity`, `impact_proximity_factor_lift_off_velocity` / `_touch_down_velocity` | `liftoff_linvel_foot_rt_world_ewrt_world_z`, `touchdown_linvel_foot_rt_world_ewrt_world_z`, `impact_proximity_factor_liftoff_slope` / `_touchdown_slope` (the slopes of a dimensionless factor, not velocities) |
| proto `TaskFile` | `sim_contact_force_threshold`, `sim_auto_reset_base_height` | `sim_min_force_mjcontact_ewrt_mjcontact_x`, `sim_auto_reset_min_pos_base_rt_world_ewrt_world_z` |
| proto `JointPdGainsFile` | `torque_limit` | `max_joint_torque` (joint space, section 5) |
| proto `LocomotionHeuristicsConfig` | `centripetal_acceleration.maximum_force`; the lists `base_pose` / `foothold` / `wrench`; the blocks `orientation_compensation` / `periodic_orientation` / `centripetal_acceleration`; Bledt's `roll_per_lateral_velocity` / `pitch_per_forward_velocity` / `forward_per_forward_velocity` / `lateral_per_lateral_velocity`; `impulse_scaling.maximum_force_ratio` | `max_force_com_ewrt_world_xy` (the bound on the resultant horizontal force, the CoM's `m a`); `base_pose_heuristics` / `foothold_heuristics` / `wrench_heuristics` (heuristicListName()); `orientation_compensation_parameters` / `periodic_orientation_parameters` / `centripetal_acceleration_parameters` (parameterBlockName(); the registry names stay); `..._gain`; `maximum_weight_compensation_ratio`, as the C++ |
| proto `MultiContactPlanConfig`, `ContactPlanningFile`, `DcmTerminalCostConfig` | `force_reference`, `yaw_torque_regularization`, `velocity_offset_factor` | `force_reference_mode` (the C++ `forceReferenceMode`; the registry `multi_contact_force_reference` stays), `yaw_torque_regularization_parameters` (the cost term keeps its registry name), `velocity_command_offset_factor` (the C++ `velocityCommandOffsetFactor`) |
| proto `LocomotionHeuristicsConfig` (the parameter blocks) and the C++ `LocomotionHeuristicConfig` | `capture_point.maximum_offset`; `centripetal_acceleration_parameters.maximum_force_ratio_of_weight` / `maximumForceRatioOfWeight`; `height_compensation.height_offset` / `heightOffset`, `height_per_speed_squared` / `height_per_speed`; `orientation_compensation_parameters.roll_offset` / `pitch_offset`; `periodic_orientation_parameters.roll_amplitude` / `pitch_amplitude`; `translational_stepping` / `in_place_turning` / `high_speed_turning` `.forward_offset` / `.lateral_offset` (and the C++ camelCase of each) | `max_norm_offset_pos_foot_rt_world_ewrt_world_xy` (its C++ field's name); `max_wnorm_force_com_ewrt_world_xy` / `maxWnormForceComEwrtWorldXy` (the clamp over the weight, `wnorm`); `offset_pos_base_rt_world_ewrt_world_z`, `height_per_speed_squared_gain` / `height_per_speed_gain` (gains, as their `orientation_compensation_parameters` siblings); `offset_eulerzyx_base_rt_world_roll` / `_pitch`; `amplitude_eulerzyx_base_rt_world_roll` / `_pitch`; `offset_pos_foot_rt_world_ewrt_heading_x` / `_y` (the lateral one applied per side). Same field numbers |
| proto `ContactPlanningFile.ZmpSupportRegionConfig`, C++ `ZmpSupportRegionParameters` | `half_width_x` / `half_width_y`, `halfWidthX` / `halfWidthY`, `kDefaultHalfWidthX` / `Y` | `max_norm_pos_zmp_rt_foot_ewrt_foot_x` / `_y`, `maxNormPosZmpRtFootEwrtFootX` / `Y`, `kNomMaxNormPosZmpRtFootEwrtFootX` / `Y`: the single-support box in the stance foot's footprint frame, which the double-support rows reuse in heading axes. Same field numbers |
| proto `MultiContactPlanConfig.ClockHold` | `contact_tolerance`, `com_tolerance` | `max_norm_pos_contact_rt_surface_ewrt_surface_z`, `max_norm_err_pos_com_rt_world_ewrt_world_xy` (the names of the C++ settings they feed). Same field numbers |
| `ContactPlanningConfig` / `ContactPlanningModelParameters` (heading model bounds) | `HipYawRangeParameters::lower` / `upper`, `FootYawOffsetBounds::lower` / `upper`, `HipYawRange::lower` / `upper`, `footYawOffsetLower` / `Upper`, `kDefaultFootYawOffset`, `setSymmetricFootYawOffset`, `YawTorqueBudgetParameters::doubleSupportYawCouple`, `doubleSupportYawCouplePerStepWidth` | `minYawFeetRtBase` / `maxYawFeetRtBase` (feet arrays), `minYawFootRtBase` / `maxYawFootRtBase` (one foot), `kNomMaxYawFootRtBase`, `setSymmetricMaxYawFootRtBase`, `doubleSupportMaxTorqueFeetmeanEwrtWorldZ`, `perStepWidthDoubleSupportMaxTorqueFeetmeanEwrtWorldZ` [N] (the couple of the two stance feet about the vertical through `feetmean`: one scalar, so no plural) |
| multi-contact planner tilt bounds | `StanceIkOptions::maxPadTilt` / `surfaceTiltTolerance`, `kDefaultMaxPadTilt` / `kDefaultSurfaceTiltTolerance`, `StanceIkResult::maxPointTilt` / `maxSurfaceTilt`, `ContactInvariantPlannerOptions::maxPointTilt`, `kDefaultMaxPointTilt`; `PlanTiming`'s `kPeakSpeed` | `maxTiltanglePadRtSurface` / `surfaceMaxTiltanglePadRtSurface`, `kNomMaxTiltanglePadRtSurface` / `kNomSurfaceMaxTiltanglePadRtSurface`, `pointMaxTiltanglePadRtSurface` / `surfaceMaxTiltanglePadRtSurface`, `pointMaxTiltanglePadRtSurface`, `kNomPointMaxTiltanglePadRtSurface` (`tiltangle`, section 3); `kPeakFirstDerivative` |
| multi-contact terrain | `TerrainModel::nearestPoint` | `computeNearestPosSurfaceRtWorldEwrtWorld` |
| `config_textproto` (Python), `testConfigNproto` | `position` (an offset into a text), `TextprotoDocument.position(path)`, `RetiredKeyCase::position` | `offset`, `line_and_column(path)`, `location` (not frame quantities) |
| remote_control walking commands (keyboard, Xbox, teleop, GUI) | `KeyboardCommand`'s `max_vel_x` / `max_vel_y` / `max_vel_yaw`, `x_vel` / `y_vel` / `yaw_vel`, `pelvis_height` / `min_pelvis_height` / `max_pelvis_height` / `pelvis_height_step`, `height_change`; `XBoxControllerInterface`'s `current_pelvis_height_target` / `initial_pelvis_height` / `pelvis_height_rate`, `GamepadPoller.pelvis_height_target` / `set_pelvis_height_target`; `teleop.initial_pelvis_height()`, `stop_pelvis_height`; `config_files.DEFAULT_PELVIS_HEIGHT` / `read_default_pelvis_height`; the GUI's `default_pelvis_height` / `set_default_pelvis_height` / `move_default_pelvis_height` / `moved_pelvis_height` | `cmd_max_lnorm_linvel_base_rt_world_ewrt_heading_x` / `_y`, `cmd_max_lnorm_angvel_base_rt_world_ewrt_world_z`, `cmd_lnorm_linvel_base_rt_world_ewrt_heading_x` / `_y`, `cmd_lnorm_angvel_base_rt_world_ewrt_world_z` (stick deflections divided by the reference file's limits, `lnorm`, not the m/s and rad/s of `cmd_max_linvel_...` there), `cmd_pos_base_rt_surface_ewrt_world_z` / `cmd_min_...` / `cmd_max_...` / `cmd_step_pos_base_rt_surface_ewrt_world_z`, `cmd_err_pos_base_rt_surface_ewrt_world_z`; `cmd_pos_base_rt_surface_ewrt_world_z` / `init_cmd_pos_base_rt_surface_ewrt_world_z` / `cmd_linvel_base_rt_surface_ewrt_world_z` (the trigger rate of the height target above the ground), `cmd_pos_base_rt_surface_ewrt_world_z` / `set_cmd_pos_base_rt_surface_ewrt_world_z`; `read_init_cmd_pos_base_rt_surface_ewrt_world_z()`, `stop_cmd_pos_base_rt_surface_ewrt_world_z`; `NOM_POS_BASE_RT_SURFACE_EWRT_WORLD_Z` / `read_nom_pos_base_rt_surface_ewrt_world_z`; `nom_pos_base_rt_surface_ewrt_world_z` / `set_nom_...` / `move_nom_...` / `moved_cmd_pos_base_rt_surface_ewrt_world_z` (a link name such as `pelvis` never appears in an identifier, section 4.1; the `--default_pelvis_height` flag keeps its spelling, with `dest=`). The GUI slider's `min_height` / `max_height` / `height_scale` are `cmd_min_pos_base_rt_surface_ewrt_world_z` / `cmd_max_pos_base_rt_surface_ewrt_world_z` / `cmd_step_pos_base_rt_surface_ewrt_world_z` (per slider step); bare `height` and `slider_height()` keep their names |
| remote_control dodgeball (`tk_app/dodgeball.py`) | `spawn_offset(throw)`, `minimum_launch_speed`, `launch_speed`, `launch_velocity`, `arrival_velocity`, their locals `offset_x` / `velocity_x`, ... | `compute_spawn_pos_ball_rt_base_ewrt_heading`, `compute_min_launch_speed_ball_rt_world`, `compute_launch_speed_ball_rt_world`, `compute_launch_linvel_ball_rt_world_ewrt_heading`, `compute_arrival_linvel_ball_rt_world_ewrt_heading`, `spawn_pos_ball_rt_base_ewrt_heading_x` / `launch_linvel_ball_rt_world_ewrt_heading_x`, ...; `impact_momentum()` is `compute_impact_norm_linmom_ball_rt_world_ewrt_world()`; `DodgeballThrow`'s `azimuth_deg` / `elevation_deg` / `distance_m` / `speed_mps` are `cmd_azimuth_ball_rt_base_ewrt_heading` / `cmd_elevation_ball_rt_base_ewrt_heading` (in degrees, as their comment says) / `cmd_dist_ball_rt_base` / `cmd_speed_ball_rt_world`, as the `DodgeballThrow` message's fields; `GRAVITY` and the `gravity` magnitudes keep theirs (no legacy word) |
| remote_control widgets and joint space | `Joystick.set_position(x_norm, y_norm)`, `App.set_knob_positions` / `center_velocity_sticks`, the window's `pos_x` / `pos_y`, `SliderRow._position` (a slider coordinate), `JointTargetsTab.default_positions`, `operator_bus.joint_targets(positions)`; `ControlMode.ZERO_TORQUE` | `set_deflection`, `set_knob_deflections` / `center_walking_sticks`, `window_x` / `window_y`, `_slider_coordinate`, `default_joint_positions`, `joint_targets(joint_positions)` (not frame quantities); the enumerator keeps its name (an `enum.Enum` member is not read) |
| `MpcRobotModelBase` | `adaptBasePoseHeight(state, heightChange)`, `interpolateWithSlerpedOrientation` (its `leftOrientation` / `rightOrientation`) | `adaptPosBaseRtWorldEwrtWorldZ(state, errPosBaseRtWorldEwrtWorldZ)`, `interpolateWithSlerpedQuatBaseRtWorld` (`leftQuatBaseRtWorldXyzw` / `rightQuatBaseRtWorldXyzw`) |
| `MpcRobotModelBase` | `getBaseComVelocity` | removed from the base, which no production caller read: `CentroidalMpcRobotModel::getMnormCmomComRtWorldEwrtWorld`, `WBAccelMpcRobotModel::getTuningLayoutBaseRates` (section 6) |
| `MpcRobotModelBase` | the parameters `wrench` / `force` / `moment` of the input-only setters | `wrenchContactEwrtParam` / `forceContactEwrtParam` / `torqueContactEwrtParam` (an override keeps its base's names) |
| `BasisInputsModelDecorator` | `getContactFrameRotationLocalToWorld(state, i)`, `rotateWrenchLocalToWorld(state, wrenchLocal, i)` / `(w_R_l, wrenchLocal)`, `rotateWrenchWorldToLocal(state, wrenchWorld, i)` | `getRotmatContactRtWorld(state, i)`, `computeWrenchContactEwrtWorld(state, wrenchContactEwrtContact, i)` / `(rotmatContactRtWorld, wrenchContactEwrtContact)`, `computeWrenchContactEwrtContact(state, wrenchContactEwrtWorld, i)` |
| `BasisInputsModelDecorator`, `BasisInputsCostTransform` | `getLocalBasisToWrenchMap`, `basisToWrenchMap`; the wrapped model's `wrenchInput`, `wrenchJacobians` | `getLocalBasisGeneratorMap`, `basisGeneratorMap` (a map whose blocks are the generators: no frame pair); `wrappedInput`, `wrappedJacobians` |
| `RealtimeRigidBodyDynamics` | `gravityTorques()`, `getBaseAcceleration()`, `centroidalMomentum(q, v)`, `contactFrameRotations(q)`, `footWrenches`, `baseRotation()`, `velocity_` / `acceleration_` | `gravityJointTorques()`, `getBaseGeneralizedAcceleration()` (`[ddp_W, dw_B]`, the base rows of the generalized acceleration), `computeCmomComRtWorldEwrtWorld(q, v)`, `computeRotmatContactsRtWorld(q)`, `wrenchFeetEwrtWorld`, `rotmatBaseRtWorld()`, `generalizedVelocity_` / `generalizedAcceleration_` |
| pinocchio_model `DynamicsHelperFunctions.h` | `computeContactPositions` / `getContactPositions`, `computeFramePositions` / `getFramePositions`, `computeBaseAcceleration`, `getRotationMatrixLocalToWorld(data, frame)` | `computePosContactsRtWorldEwrtWorld` / `getPosContactsRtWorldEwrtWorld`, `computePosLinksRtWorldEwrtWorld` / `getPosLinksRtWorldEwrtWorld`, `computeBaseGeneralizedAcceleration`, `getRotmatLinkRtWorld(data, frame)` (the whole-body MPC's own `computeBaseAcceleration(state, input, ...)` becomes an overload of the same name, `computeBaseGeneralizedAcceleration(state, input, ...)`) |
| `orientation/BaseOrientation.h` | `headingQuaternion(xi)`, `swingQuaternion(xi)`, `kMinimumTwistNorm`, `kQuaternionNormGuard`; file-local `splitTwistSwing`, `twistNorm`, `twistHemisphereSign`, `headingFromTwist` | `verticalTwistFromQuaternion(xi)`, `swingFromQuaternion(xi)`, `kMinimumHeadingCoefficientsNorm` (rho = \|(xi_z, xi_w)\|), `kUnitQuaternionNormGuard`; `twistSwingSplitFromQuaternion`, `headingCoefficientsNorm`, `headingHemisphereSign`, `headingFromCoefficients` (generic algebra of the twist-swing split) |
| `orientation/EulerBoundary.h` | `eulerZyxRateToLocalAngularVelocityMatrix`, `eulerZyxRatesFromLocalAngularVelocity(eulerZyx, localAngularVelocity)`, `localAngularVelocityFromEulerZyxRates`, `localAngularVelocityEulerZyxJacobian` | `eulerZyxRateToBodyRateMatrix`, `eulerZyxRatesFromBodyRates(eulerZyx, angvelLocalRtWorldEwrtLocal)`, `bodyRatesFromEulerZyxRates`, `bodyRateEulerZyxJacobian` (generic Euler conversions: "body rate" is the body angular velocity of the frame the caller passes) |
| `orientation/TuningLayoutDeviation` | the tuning-layout blocks `orientation_` / `angularVelocity_`, `angularVelocityError(...)` | `orientationSlots_` / `angularVelocitySlots_` (layout), `computeErrAngvelBaseRtWorldEwrtBase(quatBaseRtWorldXyzw, refQuatBaseRtWorldXyzw, angvelBaseRtWorldEwrtBase, refAngvelBaseRtWorldEwrtBase)` |
| acom `AcomAttitudeTangentJacobian` | `baseOrientation` (the block of the base's body perturbation) | `baseTangent` |
| `ModelSettings` | `FootConstraintConfig::constrainOrientation`, `ContactImplicitConfig::velocityReference` / `angularVelocityReference` / `kDefaultVelocityReference` | `constrainAttitude`, `scaleLinvelContactRtWorldEwrtWorldXy` / `scaleAngvelContactRtWorldEwrtWorldZ` / `kNomScaleLinvelContactRtWorldEwrtWorldXy` (the config dump keeps its keys) |
| term names | `contact_term::kContactMomentXY` / `kForceWeightedSlip` / `kNormalVelocity` / `kNormalVelocitySoft`, `kZeroVelocityTermSuffix`, `kWrenchContactInputParameterization`, `kWrenchInputsLibraryKey` | `kContactMomentXYName` / `kForceWeightedSlipName` / `kNormalVelocityName` / `kNormalVelocitySoftName`, `kZeroVelocityTermNameSuffix`, `kWrenchContactInputParameterizationName`, `kWrenchInputsLibraryName` (registry names; the enumerators `MpcSoftConstraintType::kNormalVelocity`, ... keep theirs) |
| contact | `contactForceInputJacobian(model, i)`, `normalContactForceRow(model, i)`; `ContactCenterPoint::rotationFromParent` / `translationFromParent`; `ContactPatchFrame::worldToPatchRotation` (p_R_w) | `inputJacobianForceContactEwrtParam(model, i)`, `inputRowForceContactEwrtParamZ(model, i)`; `rotmatContactRtLink` / `posContactRtLinkEwrtLink`; `rotmatWorldRtSurface` |
| contact | the basis-generator helper `wrenchAt(point, force, torsion)`, `ContactWrenchMapper`'s `contactWrench` / `contactPointForceVec`, `ContactWrenchGate::apply(plannedWrenches)` | `generatorWrenchAt(point, generatorForce, torsion)`, `wrenchContactEwrtContact` / `forceCornersEwrtContact`, `apply(plannedWrenchFeetEwrtWorld)` |
| contact_swing | `ContactSwingEndpoint::position`, `ContactSwingReference::position` / `linearVelocity` | `posContactRtWorldEwrtWorld`, `posContactRtWorldEwrtWorld` / `linvelContactRtWorldEwrtWorld` |
| swing_foot_planner | `CubicSpline` / `SplineCpg`'s `position(t)` / `velocity(t)` / `acceleration(t)` and `Node::position` / `velocity` (a generic scalar spline) | `value(t)` / `derivative(t)` / `secondDerivative(t)`, `Node::value` / `derivative` |
| `SwingTrajectoryPlanner` | `getZpositionConstraint` / `getZvelocityConstraint` / `getZaccelerationConstraint`; `Config::liftOffVelocity` / `touchDownVelocity` / `impactProximityFactorLiftOffVelocity` / `impactProximityFactorTouchDownVelocity`; `descentVelocity` | `getDesPosFootRtWorldEwrtWorldZ` / `getDesLinvelFootRtWorldEwrtWorldZ` / `getDesLinaccFootRtWorldEwrtWorldZ`; `liftoffLinvelFootRtWorldEwrtWorldZ` / `touchdownLinvelFootRtWorldEwrtWorldZ` / `impactProximityFactorLiftoffSlope` / `impactProximityFactorTouchdownSlope`; `descentSpeedFootRtWorld` |
| `ContactWrenchConeConstraint::Config` | `minNormalForce` / `gripperForce` / `kDefaultMinNormalForce`, the constructor's `minNormalForceParam` / `gripperForceParam` | `minForceContactEwrtContactZ` / `gripForceContactEwrtContactZ` / `kNomMinForceContactEwrtContactZ`; the constructor parameters take the members' names (the config dump keeps its keys) |
| `FrictionForceConeConstraint::Config` | `gripperForce` | `gripForceContactEwrtSurfaceZ` (this cone is written in the frame of the patch the contact touches) |
| contact cones | `torsionLimitsOfNormalForceAt(rows, point)`, `pointForceWrench(point, force)`, `checkWrenchSpaceInput`, `getContactForceInputJacobian`, `contactForceInputJacobian_` / `forceInputJacobian_`, `contactBlockOfForceJacobian` | `computeLimitsPerNewtonTorqueContactEwrtContactZ(rows, point)`, `generatorWrenchAt(point, generatorForce)`, `checkWrenchInputLayout`, `getInputJacobianForceContactEwrtParam`, `inputJacobianForceContactEwrtParam_`, `computeBlockInputJacobianForceContactEwrtParam` |
| contact cones | `forceInWorld` / `momentInWorld`, `forceInLocal` / `momentInLocal` / `localForce` / `localMoments`, `forceInPatch` / `localForces` (in the patch frame), `worldToPatch` | `forceContactEwrtWorld` / `torqueContactEwrtWorld`, `forceContactEwrtContact` / `torqueContactEwrtContact`, `forceContactEwrtSurface`, `rotmatWorldRtSurface` |
| contact-implicit terms | `forceReference`, `getForceReference`, `inverseForceReference_`, `normalForceRow_`, the normalized `normalForce` | `scaleForceContactEwrtParamZ`, `getScaleForceContactEwrtParamZ`, `inverseScaleForceContactEwrtParamZ_`, `inputRowForceContactEwrtParamZ_`, `scaledForceContactEwrtParamZ` |
| `ForceWeightedSlipConstraint` | `inverseTwistReference_` / `getInverseTwistReference` / `setTwistReferences`, its `twist`, `velocity`, `angularVelocity` | `inverseScaleSlip_` / `getInverseScaleSlip` / `setSlipScales`, `scaledSlip` (the rows `[v_x, v_y, omega_z]` are not a whole twist), `linvelContactRtWorldEwrtWorld`, `angvelContactRtWorldEwrtWorld` |
| `EndEffectorKinematicsTwistConstraint` | `setWorldToPatchRotation` / `clearWorldToPatchRotation` / `getWorldToPatchRotation`, `worldToPatchRotation_`, `getAngularVelocityToOrientationErrorRateMap`, `kOrientationErrorWrtPlaneAntiParallelThreshold` | `setRotmatWorldRtSurface` / `clearRotmatWorldRtSurface` / `getRotmatWorldRtSurface`, `rotmatWorldRtSurface_`, `getTiltRateMap` (de/dt = M w: a map, and `tilt` is a quantity word), `kPlaneNormalAntiParallelThreshold` |
| `EndEffectorKinematicsTwistConstraint` | the `footPose` and `twist` that Ax and Av multiply, `positionApprox` / `orientationApprox` / `twistApprox` | `xee` / `vee`, the class comment's symbols (their angular parts are the tilt error and its rate, not an orientation and an angular velocity); `approxPosEeRtWorldEwrtWorld` / `approxTiltEeRtSurfaceEwrtWorld` / `approxTwistEeRtWorldEwrtWorld` |
| `EndEffectorKinematicsCostElement`, `PlanarEndEffectorKinematicsPlanarReference` | `getPosition` / `getOrientation` / `getLinearVelocity` / `getAngularVelocity` and their setters | `getPosEeRtWorldEwrtWorld` / `getQuatEeRtWorld` / `getLinvelEeRtWorldEwrtWorld` / `getAngvelEeRtWorldEwrtWorld` and `set...` |
| `EndEffectorKinematicCostHelpers` | `tiltAngleToPlane(rotation, planeNormal)`; `computeTaskSpaceErrors`'s `errorQuaternion` / `orientationError` | `tiltAngleToPlane(rotmatEeRtWorld, normalSurfaceEwrtWorld)`; `errQuatEeRtWorld` / `errRotvecEeRtWorld` (the reference times the inverse of the current) |
| foot collision constraints | the collision points' `positions` / `velocities`, `relativePosition` / `relativeVelocity` / `difference` | `posLinksRtWorldEwrtWorld` / `linvelLinksRtWorldEwrtWorld`, `errPosLinkRtWorldEwrtWorld` / `errLinvelLinkRtWorldEwrtWorld` (the first point minus the second) |
| `JointLimitsSoftConstraint` | `positionLimits_`, `upperBoundPositionOffset` / `lowerBoundPositionOffset` | `jointPositionLimits_`, `upperBoundJointPositionOffset` / `lowerBoundJointPositionOffset` (joint space) |
| `ComAndAcomTrackingCost` | the snake_case `p_com`, `p_com_ref`, `p_err` | `pos_com_rt_world_ewrt_world`, `ref_pos_com_rt_world_ewrt_world`, `err_pos_com_rt_world_ewrt_world` (the current minus the reference; snake_case locals stay snake_case) |
| layout | `kBasePoseTuningRow` (`ComAndAcomTrackingCost`), `kNormalForceRow` (`FrictionForceConeConstraint`) | `kBasePoseTuningStartIndex`, `kNormalForceRowIndex` |
| `ReferenceSettings` (and the members of `TargetTrajectoriesCalculatorBase` and `ProceduralMpcMotionManager` that hold them, with `_`) | `targetDisplacementVelocity`, `targetRotationVelocity`, `maxDisplacementVelocityX` / `Y`, `maxDeltaPelvisHeight`, `maxRotationVelocity`, `maxLinearAcceleration`, `maxAngularAcceleration`, `defaultBaseHeight` | the camelCase of the reference file's fields: `refSpeedBaseRtWorld`, `refRateYawBaseRtWorld`, `cmdMaxLinvelBaseRtWorldEwrtHeadingX` / `Y`, `cmdMaxErrPosBaseRtSurfaceEwrtWorldZ`, `cmdMaxAngvelBaseRtWorldEwrtWorldZ`, `cmdMaxLinaccBaseRtWorldEwrtHeadingXy`, `cmdMaxAngaccBaseRtWorldEwrtWorldZ`, `nomPosBaseRtSurfaceEwrtWorldZ` (the config dump keeps its keys) |
| `ReferenceSettings.h` | `kInitialMaxDisplacementVelocityX` / `Y`, `kInitialMaxDeltaPelvisHeight`, `kInitialMaxRotationVelocity` | `kInitCmdMaxLinvelBaseRtWorldEwrtHeadingX` / `Y`, `kInitCmdMaxErrPosBaseRtSurfaceEwrtWorldZ`, `kInitCmdMaxAngvelBaseRtWorldEwrtWorldZ` |
| `TargetTrajectoriesCalculatorBase` | `setTargetDisplacementVelocity` / `setTargetRotationVelocity`, `commandedPositionToTargetTrajectories(commandedVelocities, ...)` / `commandedVelocityToTargetTrajectories(commandedVelocities, ...)` | `setRefSpeedBaseRtWorld` / `setRefRateYawBaseRtWorld`, `poseCommandToTargetTrajectories(poseCommand, ...)` / `velocityCommandToTargetTrajectories(velocityCommand, ...)` (`command`: the operator's command vector, whose entries mix quantities) |
| `TargetTrajectoriesCalculatorBase` | `getCurrentBasePoseTarget(state)`, `integrateTargetBasePose(currentPose, startVelocityWorld, yawRate, baseHeight, deltaT)`, `subdivideYawKnots(..., startPose, ..., targetPose)`, `getDeltaBaseTarget(commadLinePoseTarget, currentPoseTarget)` | `getPoseeulerzyxHeadingRtWorld(state)`, `integrateRefPoseeulerzyxBaseRtWorld(initPoseeulerzyxBaseRtWorld, initLinvelBaseRtWorldEwrtWorldXy, rateYawBaseRtWorld, refPosBaseRtWorldEwrtWorldZ, deltaT)`, `(..., initPoseeulerzyxBaseRtWorld, ..., finalPoseeulerzyxBaseRtWorld)`, `getFinalPoseeulerzyxBaseRtWorld(poseCommand, poseeulerzyxHeadingRtWorld)` (it returns the absolute target pose, not a delta) |
| `TargetTrajectoriesCalculatorBase` | `PoseCommandKnot::basePose`, `VelocityCommandKnot::basePose` / `commandedVelocityWorld` / `commandedYawRate`, `kMaxKnotPositionDeviation` / `kMaxKnotVelocityDeviation`, file-local `kAveragedTwists` | `poseeulerzyxBaseRtWorld`, `poseeulerzyxBaseRtWorld` / `cmdLinvelBaseRtWorldEwrtWorldXy` / `cmdRateYawBaseRtWorld`, `kKnotMaxErrPosBaseRtWorldEwrtWorld` / `kKnotMaxErrLinvelBaseRtWorldEwrtWorld`, `kAveragedMotionCount` |
| `BasePlanarMotion` (the struct) | `velocityInHeadingFrame`, `headingRate` | `linvelBaseRtWorldEwrtHeadingXy`, `rateYawBaseRtWorld` (as `BasePlanarMotion` above) |
| `WalkingVelocityCommand` (the C++ struct, snake_case as the proto) | `linear_velocity_x` / `_y`, `angular_velocity_z`, `desired_pelvis_height`, `kDefaultDesiredPelvisHeight` | split by units: the command the bus carries is the new `NormalizedWalkingVelocityCommand` with `cmd_lnorm_linvel_base_rt_world_ewrt_heading_x` / `_y`, `cmd_lnorm_angvel_base_rt_world_ewrt_world_z` (fractions of the command limits, as the proto) and `cmd_pos_base_rt_surface_ewrt_world_z`, which `node::walkingVelocityCommandFromProto()`, `node::clampWalkingVelocityCommand()` and `ProceduralMpcMotionManager::setAndScaleVelocityCommand()` take; `WalkingVelocityCommand`, the scaled command in m/s and rad/s that `scaleWalkingVelocityCommand()` returns, has `cmd_linvel_base_rt_world_ewrt_heading_x` / `_y`, `cmd_angvel_base_rt_world_ewrt_world_z`, `cmd_pos_base_rt_surface_ewrt_world_z`, `kNomCmdPosBaseRtSurfaceEwrtWorldZ` |
| `ProceduralMpcMotionManager` | `GaitModeStateConfig::minLinVelCmd` / `maxLinVelCmd` / `minAngVelCmd` / `maxAngVelCmd` / `linVelErrorThresh` / `angVelErrorThresh`, `getMaxLinearAcceleration` / `getMaxAngularAcceleration`, `velocityTargetToTargetTrajectories` (the parameter; the type alias keeps its name), `velocityTarget` | `cmdMinLinvelBaseRtWorldEwrtHeadingXy` / `cmdMaxLinvelBaseRtWorldEwrtHeadingXy` / `cmdMinAngvelBaseRtWorldEwrtWorldZ` / `cmdMaxAngvelBaseRtWorldEwrtWorldZ` / `maxErrLinvelBaseRtWorldEwrtHeadingXy` / `maxErrAngvelBaseRtWorldEwrtWorldZ`, `getCmdMaxLinaccBaseRtWorldEwrtHeadingXy` / `getCmdMaxAngaccBaseRtWorldEwrtWorldZ`, `velocityCommandToTargetTrajectories`, `velocityCommand` |
| `StrideMotionHistory.h` | `instantaneousPlanarMotion(orientation, twistWorld)`, `addSample(time, positionXy, heading, seed)`, `lastPosition_` | `(quatBaseRtWorldXyzw, twistBaseRtWorldEwrtWorld)`, `(time, posBaseRtWorldEwrtWorldXy, yawBaseRtWorld, seed)`, `prevPosBaseRtWorldEwrtWorldXy_`; a `Sample`'s unwrapped `heading` is `unwrappedYawBaseRtWorld` (not wrapped into (-pi, pi]) |
| `SwitchedModelReferenceManager` | `SwingFootReference::position` / `linearVelocity`, `getSwingFootVelocityReference`, `getCommandedVelocity` / `getCommandedYawRate`, `PlanarCommandAt::velocityWorld` / `yawRate`, `PlanarBasePose::position` / `yaw`, `commandedPoseChange` | `desPosFootRtWorldEwrtWorld` / `desLinvelFootRtWorldEwrtWorld`, and its `yaw` is `desYawFootRtWorld` (the planned foot is flat, so its twist heading and its ZYX yaw `atan2(R10, R00)` agree), `getSwingRefLinvelFootRtWorldEwrtWorldXy`, `getCmdLinvelBaseRtWorldEwrtWorldXy` / `getCmdRateYawBaseRtWorld` (the operator's command, whatever channel the model stores it in), `cmdLinvelBaseRtWorldEwrtWorldXy` / `cmdRateYawBaseRtWorld`, `posBaseRtWorldEwrtWorldXy` / `yawBaseRtWorld`, `integratePlanarCommand` |
| `SwitchedModelReferenceManager` | `shapeBasePose`, `shapedBodyAngularVelocity(..., nominalBodyAngularVelocity, ...)`, `tiltAngularVelocityInWorld`, `measuredBasePosition_` / `measuredBaseYaw_` / `liftOffPositions_` / `measuredComVelocity_`; file-local `storesBaseAngularVelocity`, `kNormalizedAngularMomentumSegment`, `normalizedAngularMomentumRow` | `shapePoseBaseRtWorld`, `computeShapedAngvelBaseRtWorldEwrtBase(..., refAngvelBaseRtWorldEwrtBase, ...)`, `computeOffsetAngvelBaseRtWorldEwrtWorld`, `measPosBaseRtWorldEwrtWorldXy_` / `measYawBaseRtWorld_` / `liftoffPosFeetRtWorldEwrtWorldXy_` / `measProxyLinvelComRtWorldEwrtWorldXy_`; `storesAngvelBaseRtWorldEwrtBase`, `kMnormAngmomSegmentName`, `mnormAngmomStartIndex` |
| `ArmSwing.cpp` | `ArmJoint::position` / `axis` | `posLinkRtWorldEwrtWorld` / `axisJointEwrtWorld` |
| mrt | `control_mode::kZeroTorque`, `JointPdGainsDefaults::torqueLimit` | `kZeroJointTorque` (its mode name `"ZERO_TORQUE"` and the enumerator `FsmCommandKind::kZeroTorque` stay), `maxJointTorque` (joint space; the config dump keeps its key) |
| config | `kCommandedVelocityTerminalDcmTarget` / `kRestTerminalDcmTarget`, `trackingStateWeightsFromConfig(..., basePose)` | `kCommandedVelocityTerminalDcmTargetName` / `kRestTerminalDcmTargetName` (registry names), `(..., basePoseWeights)`; the C++ reads of `TaskSpaceWeights` and `InputWeights` fields carry the per-component-weight `NOLINT(frame-naming)` |
| contact planning `ContactPlannerInput`, `ContactPlan` | `comPosition` / `comVelocity` / `footPositions` (the input's planar values); the plan's per-node `comPosition` / `comVelocity`, `comPositionAtTime(t)` / `comVelocityAtTime(t)` | `posComRtWorldEwrtWorldXy` / `proxyLinvelComRtWorldEwrtWorldXy` (the CoM velocity the LIP starts from, filled with getProxyLinvelComRtWorldEwrtWorld(), section 6) / `posFeetRtWorldEwrtWorldXy`; `posComRtWorldEwrtWorldXyTraj` / `linvelComRtWorldEwrtWorldXyTraj`, `evaluatePosComRtWorldEwrtWorldXy(t)` / `evaluateLinvelComRtWorldEwrtWorldXy(t)` |
| contact planning `LipState`, `ExecutionContext`, the planners' CoM states | `comVelocity`, `basePosition`, `predictedComVelocity`, `HorizontalComState::position` / `velocity`, `lipOrbitalEnergy(position, velocity, ...)` | `linvelComRtWorldEwrtWorldXy` (the `ExecutionContext`'s measurement: `proxyLinvelComRtWorldEwrtWorldXy`), `posBaseRtWorldEwrtWorldXy`, `predictedProxyLinvelComRtWorldEwrtWorldXy`, `posComRtWorldEwrtWorldXy` / `proxyLinvelComRtWorldEwrtWorldXy`, `(posComRtZmpEwrtLocal, linvelComRtWorldEwrtLocal, ...)` (one axis of the caller's frame) |
| contact planning layout | `LipIndices::vel`, `yawTorque`, `Layout::yawTorque(foot)`, `var::yawTorque(foot)`, `var::kVelX` / `kVelY` | `linvelComRtWorldEwrtWorld`, `torqueFeetEwrtWorldZ`, `torqueFootEwrtWorldZ(foot)` (both), `var::kLinvelComRtWorldEwrtWorldX` / `Y` (the heading model's yaw torques act about world z) |
| `ContactPlanningConfig` | `HlipBlendParameters::maxCommandedVelocityX` / `Y`, `maxCommandedYawRate`, `maxComVelocityX` / `Y` and their `kDefault...`; `TerminalDcmParameters::trackCommandedVelocity`; `PhaseResettingParameters::lateTouchdownSearchVelocity`; `YawTorqueBudgetParameters::torsionalFrictionTorque`; the members `yawTorqueRegularization` / `yawTorqueBudget`; `term::kYawTorqueRegularization` / `kYawTorqueBudget` | `cmdMaxLinvelBaseRtWorldEwrtHeadingX` / `Y`, `cmdMaxAngvelBaseRtWorldEwrtWorldZ`, `maxProxyLinvelComRtWorldEwrtHeadingX` / `Y` and `kNom...`; `trackVelocityCommand`; `lateTouchdownSearchSpeedFootRtWorld`; `maxTorqueFootEwrtFootZ`; `yawTorqueRegularizationParameters` / `yawTorqueBudgetParameters`; `kYawTorqueRegularizationName` / `kYawTorqueBudgetName` (registry names) |
| `ContactPlanningReferenceManager` | `commandedVelocity()`, `commandedVelocity_`, `footPositions_`, `liftOffPositions_`, `computeFootPositions`, `getTargetContactPoses` / `updateTargetContactPoses` / `targetContactPoses_` / `targetPoseMutex_`, `kMaxResampleRotationSpacing`, `boundedRotationResampleTimes` | `cmdLinvelBaseRtWorldEwrtWorldXy()` (the base class's command), `cmdLinvelBaseRtWorldEwrtWorldXy_`, `posFeetRtWorldEwrtWorld_`, `liftoffPosFeetRtWorldEwrtWorld_`, `computePosFeetRtWorldEwrtWorld`, `getDesPoseFeetRtWorld` / `updateDesPoseFeetRtWorld` / `desPoseFeetRtWorld_` / `targetContactMutex_`, `kMaxResampleAngleSpacing`, `boundedAngleResampleTimes` (an angle between two attitudes) |
| `TargetContactPose` | `position`, `computeTargetContactPoses`, its variables `pose` / `poses`, `TargetContactPoseInputs::footPositions` | `posFootRtWorldEwrtWorldXy`, with `height` -> `posFootRtWorldEwrtWorldZ` and `yaw` -> `eulerzyxFootRtWorldYaw`, `computeDesPoseFeetRtWorld`, `desPoseFootRtWorld` / `desPoseFeetRtWorld`, `posFeetRtWorldEwrtWorld` |
| `HlipContactPlanner` | `blendVelocity(input)`, `deadbeatStep(..., commandedVelocity, ...)`, `PhaseRollOut::comPosition` / `comVelocity`, `ComState::position` / `velocity` | `computeBlendLinvelComRtWorldEwrtHeadingXy(input)`, `cmdLinvelBaseRtWorldEwrtHeadingXy`, `posComRtWorldEwrtWorldXy` / `linvelComRtWorldEwrtWorldXy` (both structs); `HlipStandingBlend`'s `comVelocity` is `linvelComRtWorldEwrtHeadingXy` |
| locomotion heuristics contexts | `FootholdHeuristicContext::basePosition` / `measuredVelocity` / `commandedVelocityAtMeasurement` / `commandedVelocity`, `WrenchHeuristicContext::commandedVelocity`, `BasePoseHeuristicContext::commandedVelocityInBaseFrame` | `posBaseRtWorldEwrtWorldXy` / `measProxyLinvelComRtWorldEwrtWorldXy` / `prevCmdLinvelBaseRtWorldEwrtWorldXy` (the command at the last solve) / `cmdLinvelBaseRtWorldEwrtWorldXy`, `cmdLinvelBaseRtWorldEwrtWorldXy`, `cmdLinvelBaseRtWorldEwrtHeadingXy` |
| locomotion heuristics | `WrenchHeuristic::forceOffset`, `producesHorizontalForce`, `LocomotionHeuristicLayer::wrenchOffset` / `basePoseOffset` / `basePoseEmpty` / `wrenchEmpty` / `wrenchNeedsWorldFrame`, `basePose_` / `wrench_`, `LocomotionHeuristicFormulation::basePose` / `wrench`, `heuristic::kOrientationCompensation` / `kPeriodicOrientation` / `kCentripetalAcceleration`, `LocomotionHeuristicEnvironment::footPositionIsUntracked`, `CentripetalAccelerationHeuristic::totalForceLimit` | `computeOffsetForceFootEwrtWorld` (both), `producesForceFootEwrtWorldXy`, `basePoseHeuristicOffset` / `basePoseHeuristicsEmpty` / `wrenchHeuristicsEmpty` / `wrenchHeuristicsNeedWorldFrame`, `basePoseHeuristics_` / `wrenchHeuristics_`, `basePoseHeuristics` / `wrenchHeuristics`, `...Name` (registry names), `footholdIsUntracked`, `getMaxForceComEwrtWorldXy` (`heuristic` is an exempt word, section 5) |
| `LocomotionHeuristicConfig` | the members `orientationCompensation` / `periodicOrientation` / `centripetalAcceleration`, Bledt's `rollPerLateralVelocity` / `pitchPerForwardVelocity` / `forwardPerForwardVelocity` / `lateralPerLateralVelocity`, `ImpulseScalingParameters::maximumForceRatio`, `CentripetalAccelerationParameters::maximumForce`, `LocomotionHeuristicModelParameters::hipPositionInBaseFrame` | `orientationCompensationParameters` / `periodicOrientationParameters` / `centripetalAccelerationParameters`, `rollPerLateralVelocityGain` / ... (gains, section 5), `maximumWeightCompensationRatio`, `maxForceComEwrtWorldXy`, `posHipsRtBaseEwrtHeadingXy`; the proto fields follow the C++ names (`LocomotionHeuristicsConfig` above) |
| `humanoid_common_mpc/test/equivalence` | the quaternion-root model's `quaternionData`, `quaternionState(Of)`, `quaternionMotion(Of)`, `quaternionManifold_`, ...; `velocityMap(s)`, `velocityMapRate`, `chartVelocityMapsAt`; the generalized `torques`, `acceleration`; `worldRotation` (the quaternion g of a turn of the world) and `ambientWorldRotation`, ...; `worldWrenches`, `torqueFreeInput` | `quaternionModelData`, `quaternionModelState(Of)`, `quaternionModelMotion(Of)`, `quaternionModelManifold_`, ... (a representation, not a frame pair: `model` is exempt); `generalizedVelocityMap(s)`, `generalizedVelocityMapRate`, `chartGeneralizedVelocityMapsAt`; `generalizedTorques`, `generalizedAcceleration`; `worldTurn`, `ambientWorldTurn`, ...; `computeWrenchContactsEwrtWorld`, `nonTurningInput` |
| gtest | `TEST(Suite, theVelocityIsTheDerivativeOfThePosition)` | kept: the check skips the test-name argument of the gtest macros (section 5) |
| `CentroidalMpcRobotModel` | `getCentroidalMomentum(state)`, private `baseOrientationOf(state)`, `abortWithoutBaseAngularVelocity`; in `getTwistBaseRtWorldEwrtWorld` the `baseMomentum` (`m h - A_j dq_j`) and `baseVelocity` (`[dp_W, w_B]`) | `getMnormCmomComRtWorldEwrtWorld(state)`, `getDoubleQuatBaseRtWorldXyzw(state)`, `abortWithoutAngvelBaseRtWorldEwrtBase`; `baseMotionCmomComRtWorldEwrtWorld`, `baseGeneralizedVelocity`; `getLinvelComRtWorldEwrtWorld` / `setLinvelComRtWorldEwrtWorld` (the centroidal state's, through `getStoredLinvelPoint()`), with `getProxyLinvelComRtWorldEwrtWorld` for the consumers (section 6); `getBaseComVelocity` is `getMnormCmomComRtWorldEwrtWorld` |
| `CentroidalMpcInterface` | `getBasisToWrenchMap()`, `basisToWrenchMap_`; file-local `footPositionIsTracked`; the contact-implicit `forceReference` | `getBasisGeneratorMap()`, `basisGeneratorMap_` (as `BasisInputsModelDecorator`); `footholdIsTracked`; `scaleForceContactEwrtParamZ` |
| `CentroidalWeightCompInitializer` | `extendNormalizedMomentum` (a flag), `extendNormalizedMomentum_` | `extendMnormCmomComRtWorldEwrtWorld`, `extendMnormCmomComRtWorldEwrtWorld_` |
| `CentroidalDynamicsBasisInputsAD` | `toWorldFrameWrenchInput(...)`, its `wrenchInput`, `wrenchLocal`, `w_R_l` | `toWrappedInput(...)` (the wrapped wrench-space model's input), `wrappedInput`, `wrenchContactEwrtContact`, `rotmatContactRtWorld` |
| `DynamicsHelperFunctions.h` (centroidal) | `totalGravitationalForce` (the scalar m g), `forceInInertialFrame` | `robotWeight`, `forceContactEwrtWorld` |
| `DcmTerminalCost::Config` | `velocityOffsetFactor` (a dimensionless factor) | `velocityCommandOffsetFactor` (the proto field is `velocity_command_offset_factor`); in `dcmResidual` `com`, `comVelocity`, `dcm`, `plannedDcm`, `dcmReference` are `posComRtWorldEwrtWorldXy`, `linvelComRtWorldEwrtWorldXy`, `posDcmRtWorldEwrtWorldXy`, `desPosDcmRtWorldEwrtWorldXy`, `refPosDcmRtWorldEwrtWorldXy` |
| `CentroidalMpcEndEffectorFootCost` | `footYawError(orientation, ...)`; the residual's `position` / `linearVelocity` / `orientation` / `angularVelocity` / `orientationError` / `yawError` | `footYawError(rotmatFootRtWorld, ...)`; `posFootRtWorldEwrtWorld` / `linvelFootRtWorldEwrtWorld` / `rotmatFootRtWorld` / `angvelFootRtWorldEwrtWorld` / `errRotvecFootRtWorld` / `errYawFootRtWorld` |
| `CentroidalMpcTargetTrajectoriesCalculator` | `commandLinePoseTarget`, `commandedVelocities`, `currentPoseTarget`, `targetPose`, `angularMomentum` (`I_G e_z psidot / m`) | `poseCommand`, `velocityCommand` (the base's names), `poseeulerzyxHeadingRtWorld`, `finalPoseeulerzyxBaseRtWorld`, `refMnormAngmomComRtWorldEwrtWorld` |
| `CentroidalMpcMrtJointController` | `fillZeroTorqueAction`, `clampedMpcJointFeedforward(joint, torque)`, `setNominalJointPositions(positions)`; `wrenchFrameState`, `contactFrameRotations`, `footWrenches`, `gravTorques`, `feedforwardTorques` | `fillZeroJointTorqueAction`, `(joint, jointTorque)`, `(jointPositions)`; `contactFrameState`, `rotmatContactsRtWorld`, `wrenchFeetEwrtWorld`, `gravityJointTorques`, `feedforwardJointTorques` |
| `ZeroVelocityGainsApplier.cpp` | `twist` (the constraint), `pose` (a flag) | `twistConstraint`, `constrainsAttitude` |
| `PlannedMotion` | `PlannedContactPhase::targetPosition`, `plannedWrenchesAt(t)`, `plannedWrenches_`, file-local `knotWrenches`, `kTargetPadPoseSize` | `desPosContactRtWorldEwrtWorld`, `plannedInputAt(t)`, `plannedInputs_` (vectors of the MPC's input layout), `plannedInputOfKnot`, `kTargetPadPoseDim`; a loop's proto `ContactWrench` is `wrenchMsg` |
| `PlannedMotionReferenceManager`, `MultiContactPlanSettings` | `measuredContactPositions()` / `_`, `liftOffPositions_`, `plannedContactPosition(i, t)`, `forceReference` (the `ForceReference` choice) | `measPosContactsRtWorldEwrtWorld()` / `_`, `liftoffPosContactsRtWorldEwrtWorld_`, `computePlannedPosContactRtWorldEwrtWorld(i, t)`, `forceReferenceMode` (the proto field `force_reference_mode`) |
| centroidal test support | `CentroidalMpcLoop`'s `initialGapRotation`, `quaternionNormError`, `push(momentumChange)`; `DrcAtlasContactTestModel::makeInput(..., normalForce)` | `initialGapAngle`, `unitQuaternionNormError`, `push(errMnormCmomComRtWorldEwrtWorld)`; `makeInput(..., forceFootEwrtParamZ)` |
| centroidal tests | a state or a heuristics list named for a quantity: `basePose` / `wrench` lists, `basePoseOnly_`, `withBasePoseOffsets`, `basePoseBlock`, `wrenchR`, `wrenchHolder_`, `zeroVelocity` (a constraint), `twists` (constraints) | `basePoseHeuristics` / `wrenchHeuristics`, `basePoseHeuristicsOnly_`, `withBasePoseHeuristicOffsets`, `basePoseWeightBlock`, `wrenchModeInputWeights`, `wrenchModeHolder_`, `zeroVelocityConstraint`, `twistConstraints` (`heuristic`, `weight`, `mode`, `constraint` are exempt words) |
| centroidal tests | `twistHeadingOfEulerZyx`, `yawQuaternion(angle)`, `yawRotation(angle)`, `rotationOf(xi)`, `bodyRotation(R0, R1)`, `kRoundTripRotationTolerance` | `yawOfEulerZyx`, `unitQuaternionOfYaw(angle)`, `rotmatOfYaw(angle)`, `rotmatOf(xi)`, `bodyRotvecBetween(rotmat0, rotmat1)`, `kRoundTripAngleTolerance` (generic operations) |
| whole-body `EndEffectorDynamics` | `getPosition` / `getVelocity` / `getLinearAcceleration` and their `...LinearApproximation` | `getPosEesRtWorldEwrtWorld` / `getLinvelEesRtWorldEwrtWorld` / `getLinaccEesRtWorldEwrtWorld`, `getLinearApproximationPosEesRtWorldEwrtWorld` / `getLinearApproximationLinvelEesRtWorldEwrtWorld` / `getLinearApproximationLinaccEesRtWorldEwrtWorld` (as `EndEffectorKinematics`) |
| whole-body `EndEffectorDynamics` | `getOrientationErrorWrtPlane`, `getPoseWrtPlaneRate` / `getPoseWrtPlaneAcceleration` and their linear approximations | `getTiltEesRtSurfaceEwrtWorld`, `getXeeRate` / `getXeeSecondRate` (`xee` = `[p; e]`, the class comment's symbol), `getLinearApproximationTiltEesRtSurfaceEwrtWorld` / `getLinearApproximationXeeRate` / `getLinearApproximationXeeSecondRate`, `xeeRateCppAdInterfacePtr_` / `xeeSecondRateCppAdInterfacePtr_`; the CppAD library names (`_position`, `_pose_wrt_plane_rate`, ...) stay |
| `OrientationErrorWrtPlaneRates.h` | `orientationErrorWrtPlaneRate(rotation, angularVelocity, planeNormal)`, `orientationErrorWrtPlaneAcceleration(..., angularAcceleration, ...)`, its `normalAcceleration` / `axisAcceleration` / `tiltAcceleration` / ... | `planeTiltRate(rotmatEeRtWorld, angvelEeRtWorldEwrtWorld, normalSurfaceEwrtWorld)`, `planeTiltSecondRate(..., angaccEeRtWorldEwrtWorld, ...)`, `secondRateZAxisEeEwrtWorld` / `secondRateVCrossN` / `secondRateTiltEeRtSurfaceEwrtWorld` / ... (the struct member `normal`, the end effector's z axis, is `zAxisEeEwrtWorld`, and its rates `rateZAxisEeEwrtWorld` / `secondRateZAxisEeEwrtWorld`; the cross product `a = v x n`, which is not unit, is named by its symbol, `vCrossN`, with `rateVCrossN` and `secondRateVCrossN`; the derivatives of the tilt are `rateTiltEeRtSurfaceEwrtWorld` / `secondRateTiltEeRtSurfaceEwrtWorld`) |
| `PinocchioEndEffectorDynamicsCppAd.cpp` | file-local `positionsCppAd` / `velocitiesCppAd` / `linearAccelerationsCppAd` / `orientationErrorsWrtPlaneCppAd` / `poseWrtPlaneRatesCppAd` / `poseWrtPlaneAccelerationsCppAd`, `poseWrtPlaneBlocks`, the lambdas `positionFunc`, ...; Pinocchio's classical `acceleration` | `computePosEesRtWorldEwrtWorld` / `computeLinvelEesRtWorldEwrtWorld` / `computeLinaccEesRtWorldEwrtWorld` / `computeTiltEesRtSurfaceEwrtWorld` / `computeXeeRates` / `computeXeeSecondRates`, `xeeBlocks` (`xee` = `[p; e]`, the class comment's symbol), `tapePosEesRtWorldEwrtWorld`, ...; `classicalaccEeRtWorldEwrtWorld` (the quantity `classicalacc`) |
| `EndEffectorDynamicsAccelerationsConstraint`, `EndEffectorDynamicsLinearAccConstraint` | `footPose` / `footPoseJacobian`, `positionApprox` / `orientationApprox` / `velocityApprox` / `accelApprox` / `accelerationApprox` | `xee` / `xeeJacobian`, `approxPosEeRtWorldEwrtWorld` / `approxTiltEeRtSurfaceEwrtWorld` / `approxLinvelEeRtWorldEwrtWorld` / `approxLinaccEeRtWorldEwrtWorld` / `secondRateApprox` |
| `EndEffectorDynamicsCostHelpers.h` | `PlanarEndEffectorDynamicsReference::getPosition` / `getLinearVelocity` / `getAngularVelocity` / `getLinearAcceleration` / `getAngularAcceleration` and the setters, the plane normal getter, `kFootReferenceLinearVelocityOffset`, ... | `getRefPosFootRtWorldEwrtWorld` / `getRefLinvelFootRtWorldEwrtWorld` / `getRefAngvelFootRtWorldEwrtWorld` / `getRefLinaccFootRtWorldEwrtWorld` / `getRefAngaccFootRtWorldEwrtWorld` and `setRef...`, `getSwingRefNormalSurfaceEwrtWorld` (the swing reference's normal, tilted toe-up during a swing), `kFootReferenceLinearVelocityStartIndex`, ... (layout) |
| `WBAccelMpcRobotModel` | `setBaseLinearVelocity`, private `baseOrientationOf(state)` / `baseTwistOf(state)`, `kBaseAngularVelocityOffset`; the overrides' parameters (`orientation`, `positionWorld`, `angularVelocityLocal`, `yawRate`, `heightChange`, `wrench`, ...) | `setLinvelBaseRtWorldEwrtWorld`, `getDoubleQuatBaseRtWorldXyzw(state)` / `computeTwistBaseRtWorldEwrtWorld(state)`, `kBaseAngularVelocityStartWithoutJoints`; the base's parameter names; `getLinvelBaseRtWorldEwrtWorld` / `setLinvelBaseRtWorldEwrtWorld` (the whole-body state's, through `getStoredLinvelPoint()`), with `getProxyLinvelComRtWorldEwrtWorld` for the consumers (section 6); `getBaseComVelocity` is `getTuningLayoutBaseRates` |
| whole-body `DynamicsHelperFunctions` | `computeBaseAcceleration(state, input, ...)`, file-local `contactWrenchesInWorldFrame`, `baseExternalForces` | `computeBaseGeneralizedAcceleration(state, input, ...)`, `computeWrenchContactsEwrtWorld`, `baseGeneralizedExternalForce` (`J_b^T W`, rows of the generalized force) |
| `WBMpcMrtJointController` | `fillZeroTorqueAction`, `clampedMpcJointFeedforward(joint, torque)`, `setNominalJointPositions(positions)`, `rootOrientation`, `gravityTorques`, `footWrenches` | `fillZeroJointTorqueAction`, `(joint, jointTorque)`, `(jointPositions)`, `quatBaseRtWorld`, `gravityJointTorques`, `wrenchFeetEwrtWorld` (as the centroidal controller) |
| whole-body test support | `AffineEndEffectorDynamics::position` / `velocity` / `linearAcceleration` / `orientationErrorWrtPlane` / `poseWrtPlaneRate` / `poseWrtPlaneAcceleration`, `positionDx_`, ...; `OrientationErrorRateOracle`'s `AttitudeMotion::initialRotation` / `angularVelocity` / `angularAcceleration`, `rotationAlong` / `angularVelocityAlong`, `orientationErrorRateScale`, `differencedOrientationErrorRates`, `DifferencedOrientationErrorRates::acceleration` | `posEeRtWorldEwrtWorld` / `linvelEeRtWorldEwrtWorld` / `linaccEeRtWorldEwrtWorld` / `tiltEeRtSurfaceEwrtWorld` / `xeeRate` / `xeeSecondRate`, `dxPosEeRtWorldEwrtWorld_`, `dxTiltEeRtSurfaceEwrtWorld_`, ...; `initRotmatEeRtWorld` / `angvelEeRtWorldEwrtWorld` / `angaccEeRtWorldEwrtWorld`, `computeRotmatEeRtWorld` / `computeAngvelEeRtWorldEwrtWorld`, `tiltRateScale`, `differencedTiltRates`, `secondRate` (the type keeps its name) |
| whole-body tests | `withoutQuaternionRate`, `quaternionRate(model, derivative)`, `headingRotation(angle)`, `streamOrientation`, `expectSameStateUpToTheQuaternionSign`, `largestQuaternionNormDeviation`, `planarPoseOf`, a `PlanarPose` variable `pose` | `withoutRateQuatBaseRtWorldXyzw`, `getRateQuatBaseRtWorldXyzw(model, derivative)`, `unitQuaternionOfYaw(angle)`, `streamQuatBaseRtWorldXyzw`, `expectSameStateUpToTheUnitQuaternionSign`, `largestUnitQuaternionNormDeviation`, `planarBaseOf`, `planarBase` |
| `EndEffectorKinematicsTwistConstraint` | `setGroundPlaneNormal` / `getGroundPlaneNormal`, `ground_plane_normal_` | `setNormalTerrainEwrtWorld` / `getNormalTerrainEwrtWorld`, `normalTerrainEwrtWorld_` |
| `EndEffectorDynamicsAccelerationsConstraint` | `ground_plane_normal_` | `normalTerrainEwrtWorld_` |
| `RobotBackend` (common_mpc_app) | `readMeasuredContactForces(forces)`, `RobotProcess::measuredContactForces_`, `TelemetrySampler::sample(..., measuredContactForces)` | `readMeasForceFeetEwrtWorld(measForceFeetEwrtWorld)`, `measForceFeetEwrtWorld_`, `(..., measForceFeetEwrtWorld)` (one force per foot, in the world frame) |
| `DecodedRobotState` (common_mpc_app visualization) | `basePosition` / `baseRotation` / `baseOrientation` / `baseLinearVelocity` / `baseAngularVelocity` / `baseAngularVelocityLocal` / `baseEulerAnglesZyx` / `baseEulerZyxRates` / `measuredContactWrenches` | `posBaseRtWorldEwrtWorld` / `rotmatBaseRtWorld` / `quatBaseRtWorld` / `linvelBaseRtWorldEwrtWorld` / `angvelBaseRtWorldEwrtWorld` / `angvelBaseRtWorldEwrtBase` / `eulerzyxBaseRtWorld` / `rateEulerzyxBaseRtWorld` / `measWrenchContactsEwrtWorld` |
| `BalanceQuantities.h`, `SceneContract.h`, `SceneBuilder` | `zeroMomentPoint(positions, wrenches, minNormalForce)`, `divergentComponentOfMotion(com, comVelocity, omega)`; the marker paths `kContactForces` / `kCornerForces`, `kForceScale`, `kMinNormalForceForCop`, `cornerForceMappers_`, `addForceArrow`, `writeLinkPoses` | `computePosZmpRtWorldEwrtWorld(posFeetRtWorldEwrtWorld, wrenchFeetEwrtWorld, minForceWorldEwrtWorldZ)`, `computePosDcmRtWorldEwrtWorldXy(posComRtWorldEwrtWorld, linvelComRtWorldEwrtWorld, omega)`; `kContactForceMarkers` / `kCornerForceMarkers`, `kNewtonsPerArrowMeter`, `kCopMinForceContactEwrtContactZ`, `cornerMappers_`, `addContactArrow`, `writePoseLinksRtWorld` |
| `TelemetryBuilder` | the group-index members `basePosition_`, `normalForce_`, `footVelocity_`, `dofPositions_`, ...; the frame-group kinds `kPose` / `kTwist` / `kAcceleration` / `kWrench`, `kAccelerationHalfInterval`, `kMinNormalForceForZmp`; `Source::contactWrenches` / `com` / `dcm` / `zmp` | `group<quantity>_` (`groupPosBaseRtWorldEwrtWorld_`, `groupForceFeetEwrtWorldZ_`, `groupLinvelFeetRtWorldEwrtWorldZ_`, ...), `generalizedCoordinateGroups_`, ...; `kPoseSlot`, ... (layout), `kCentralDifferenceHalfInterval`, `kZmpMinForceWorldEwrtWorldZ`; `wrenchFeetEwrtWorld` / `posComRtWorldEwrtWorld` / `posDcmRtWorldEwrtWorldXy` / `posZmpRtWorldEwrtWorldXy`. `referenceBaseLinearVelocity_` holds `getProxyLinvelComRtWorldEwrtWorld()` of the reference: `refProxyLinvelComRtWorldEwrtWorld_` |
| visualization tests | `quaternionFromRollPitchYaw`, `TestRobot::makePolicy(..., normalForce, copOffset, ...)`, `withFootForces(proto, normalForce)`, `momentAbout(point, positions, wrenches)` | `unitQuaternionOfRollPitchYaw`, `(..., forceFootEwrtWorldZ, posCopRtContactEwrtContactXy, ...)`, `withMeasForceFeetEwrtWorldZ(proto, measForceFootEwrtWorldZ)`, `computeTorqueLocalEwrtWorld(posLocalRtWorldEwrtWorld, posFeetRtWorldEwrtWorld, wrenchFeetEwrtWorld)` |
| `SimulatorSettings` (common_mpc_app) | `contactForceThreshold` | `minForceMjcontactEwrtMjcontactX`, as `MujocoSimConfig` (the config dump keeps its key) |
| `OperatorCommandMailbox`, `FakeRobotBackend` | `initialNominalPositions`, `ioNominalPositions_`, `takeNominalPosture(nominalPositions)`; `torqueEnables()` / `torqueEnables_` / `torquesEnabled_` | `initialNominalJointPositions`, `ioNominalJointPositions_`, `(nominalJointPositions)`; `jointTorqueEnables()` / `jointTorqueEnables_` / `jointTorquesEnabled_` (joint space) |
| `SimFallRecovery` | `baseTiltAngle(baseRotationLocalToWorld)`, its `baseRotation`, `maxBaseTiltAngle` / `kNomSettleMaxBaseTiltAngle` / `settleMaxBaseTiltAngle`, `sim_max_base_tilt_angle` | `computeTiltangleBaseRtWorld(quatBaseRtWorld)`, `rotmatBaseRtWorld`, `maxTiltangleBaseRtWorld` / `kNomSettleMaxTiltangleBaseRtWorld` / `settleMaxTiltangleBaseRtWorld`, `sim_max_tiltangle_base_rt_world` (same field number) |
| `WalkingVelocityCommandConversions.h` | `kMaxNormalizedVelocity` (the bound of both the planar velocity and the yaw rate), `kMinPelvisHeight` / `kMaxPelvisHeight` | split into `kCmdMaxLnormLinvelBaseRtWorldEwrtHeadingXy` (each planar component) and `kCmdMaxLnormAngvelBaseRtWorldEwrtWorldZ`; `kCmdMinPosBaseRtSurfaceEwrtWorldZ` / `kCmdMaxPosBaseRtSurfaceEwrtWorldZ` |
| `ViewerAnnotations.h` | `targetContactPatchToProto(pose, ...)`, `fillViewerAnnotations(targetContactPoses, ...)` | `(desPoseFootRtWorld, ...)`, `(desPoseFeetRtWorld, ...)`, as `TargetContactPose` |
| `KeyboardCommandLimits` | `kDefaultMaxVelocityX` / `Y`, `kDefaultMaxDeltaPelvisHeight`, `kDefaultMaxYawRate`, `kDefaultBaseHeight`, `defaultBaseHeight` | `kNomCmdMaxLinvelBaseRtWorldEwrtHeadingX` / `Y`, `kNomCmdMaxErrPosBaseRtSurfaceEwrtWorldZ`, `kNomCmdMaxAngvelBaseRtWorldEwrtWorldZ`, `kNomPosBaseRtSurfaceEwrtWorldZ`, `nomPosBaseRtSurfaceEwrtWorldZ` (as `ReferenceSettings`; the config dump keeps its key); the entry indices `kVelocityXIndex`, ... keep their names (layout) |
| `CentroidalMpcNode`, `WBMpcNode` | the target-trajectories lambda's `velocityTarget` | `velocityCommand`, as `ProceduralMpcMotionManager` |
| common_mpc_app and node tests | `ScriptedRobotController::rootPosition()` / `rootLinearVelocity()`, the test helpers `basePose(x, y, z, roll)` (a free joint's qpos), `pose(kind, x)` (a `TargetContactPose`), `referenceWithMaxVelocityX`, `basePosition(remoteControl)` | `getPosBaseRtWorldEwrtWorld()` / `getLinvelBaseRtWorldEwrtBase()`, `baseQpos(...)`, `makeDesPoseFootRtWorld(kind, x)`, `makeReferenceTextproto`, `samplePosBaseRtWorldEwrtWorld(remoteControl)` |
| validation `ControlCycleSample`, `RobotStateRecord` | `basePosition`, `baseQuaternion` (a `Vector4d` `(x, y, z, w)`), `baseLinearVelocityWorld` / `baseLinearVelocityLocal`, `baseAngularVelocityLocal`, `contactPositions`, `referenceVelocity` (forward, lateral, yaw rate after the ramps) | `posBaseRtWorldEwrtWorld`, `quatBaseRtWorldXyzw`, `linvelBaseRtWorldEwrtWorld` / `linvelBaseRtWorldEwrtBase`, `angvelBaseRtWorldEwrtBase`, `posContactsRtWorldEwrtWorld`, `refVelocityCommand` (entries of mixed quantities: `command`); the metrics' JSON keys and the recordings' matrix labels keep their strings |
| validation `SolveSample`, `SqpStepSample`, `InitialStateGap`, `ClosedLoopDriver` | `initialStateRotationGap`, `InitialStateGap::rotation`, `quaternionNormDeviation`, `maxQuaternionNormDeviationOf()`, `swingNormalVelocityResidual`, `normalVelocityTerms_`, `contactPositions(observation)`, `referenceVelocity()` | `initialStateGapAngle`, `angle` (as `initialGapAngle`), `unitQuaternionNormDeviation`, `maxUnitQuaternionNormDeviationOf()`, `swingNormalVelocityConstraintResidual` (a constraint row's residual, [m/s] or [m/s^2]), `normalVelocityConstraintTerms_`, `computePosContactsRtWorldEwrtWorld(observation)`, `refVelocityCommand()` |
| validation `ClosedLoopMetrics`, `TimeSeriesLabels.h`, `ClosedLoopScenario` | `velocityErrorSquaredSum_`, `lastContactPosition_`, `StancePhase::touchDownPosition`, `maxRotationGap_`, `torqueSquaredSum_`; `time_series::kBasePosition` / `kBaseQuaternion` / `kBaseVelocity` (`(v_x, v_y, yaw rate)`) / `kReferenceVelocity` / `kSolveRotationGap`; `CommandSegment::forwardVelocity` / `lateralVelocity`, `kWalkVelocity`, ... | `squaredSumErrLinvelBaseRtWorldEwrtHeadingXy_`, `prevPosContactsRtWorldEwrtWorldXy_`, `touchdownPosContactRtWorldEwrtWorldXy`, `maxGapAngle_`, `jointTorqueSquaredSum_`; `kPosBaseRtWorldEwrtWorld` / `kQuatBaseRtWorldXyzw` / `kBasePlanarMotion` / `kRefVelocityCommand` / `kSolveInitialGapAngle` (the label strings stay); `cmdLinvelBaseRtWorldEwrtHeadingX` / `Y`, `kWalkCmdLinvelBaseRtWorldEwrtHeadingX`, ...; `yawRate` keeps its name (a bare scalar) |
| validation `EulerChart.h`, `RecordedRobotStates.cpp`, `JsonValue.cpp`, `Sha256.cpp` | `localAngularVelocityEulerAngleDerivative`, `kPositionColumns` / `kVelocityColumns`, the parser's `position_`, `kBigSigma0FirstRotation`, ... | `bodyRateEulerZyxAngleDerivative` (as `bodyRateEulerZyxJacobian`), `kPositionColumnCount` / `kVelocityColumnCount` (layout), `offset_` (into a text), `kBigSigma0FirstRotateCount`, ... (bit rotations) |
| rerun viewer `urdf_kinematics.py`, `model_sandbox.py` | `link_poses(positions)`, `nominal_positions()`, `Joint.origin_rotation`, `rotation_about_axis`, `Joint.clamp(position)`; `SandboxScene.positions()` / `set_position(joint, position)`, `_UNLIMITED_ROTATION` | `compute_pose_links_rt_base(joint_positions)` (relative to the root link), `nominal_joint_positions()`, `origin_rotmat` (URDF's `<origin>`, beside `origin_translation`), `matrix_from_axis_angle`, `clamp(joint_position)`; `joint_positions()` / `set_joint_position(joint, joint_position)`, `_UNLIMITED_JOINT_ANGLE_RANGE` |
| rerun viewer `bridge.py`, `telemetry_contract.py`, `blueprint.py` | `_last_link_poses`, `link_poses_logged` / `link_poses_unchanged`, `_MIN_QUATERNION_NORM`; the description constants `_BASE_POSITION`, `_BASE_BODY_ANGULAR_VELOCITY`, `_NORMAL_FORCE`, `_FOOT_VELOCITY`, ..., `CONTACT_WRENCH_GROUPS`; `initial_eye_position()` | `_prev_pose_links_rt_world`, `logged_link_pose_count` / `unchanged_link_pose_count`, `_MIN_UNIT_QUATERNION_NORM`; `_POS_BASE_RT_WORLD_EWRT_WORLD`, `_ANGVEL_BASE_RT_WORLD_EWRT_BASE`, `_FORCE_FEET_EWRT_WORLD_Z`, `_LINVEL_FEET_RT_WORLD_EWRT_WORLD_Z`, ... (as `TelemetryBuilder`'s groups), `SERIES_GROUPS_WRENCH_FEET_EWRT_WORLD`; `init_pos_cam_rt_world_ewrt_world()`. Python reads of a proto `Pose`'s `position` / `orientation`, the URDF `Origin.quaternion_xyzw` and Rerun's keyword arguments (`positions=`, `quaternion=`) carry a `NOLINT(frame-naming)` |
| `humanoid_learning` `envs/base_env.py` | `rotation_matrix(quaternion_wxyz)`, `heading_frame_linear_velocity`, `yaw_quaternion`, `_BASE_POSITION_END` / `_BASE_QUATERNION_END` / `_BASE_LINEAR_VELOCITY_END` / `_BASE_ANGULAR_VELOCITY_END`, `base_quaternion`, `rng_velocity`, `heading` (a ZYX yaw), `world_yaw_rate`, `projected_gravity`, `target_vx` / `target_vy` / `target_wz`, `_FALLEN_BASE_HEIGHT`, `torso_z`; its test's `world_velocity` (the output of `mj_objectVelocity`), `_yaw_rotation`, `_random_quaternion` | `compute_rotmat_base_rt_world(quat_base_rt_world_wxyz)`, `compute_linvel_base_rt_world_ewrt_heading`, `unit_quaternion_of_yaw`, `_BASE_POSITION_END_INDEX`, ... (layout), `quat_base_rt_world_wxyz`, `rng_qvel`, `compute_yaw_base_rt_world` (the twist heading, 2 atan2(q_z, q_w)), `compute_angvel_base_rt_world_ewrt_world_z`, `compute_unit_gravity_ewrt_base`, `cmd_linvel_base_rt_world_ewrt_heading_x` / `_y` / `cmd_angvel_base_rt_world_ewrt_world_z`, `_MIN_POS_BASE_RT_WORLD_EWRT_WORLD_Z`, `pos_base_rt_world_ewrt_world_z`; `angvel_linkcom_rt_world_ewrt_world` / `linvel_linkcom_rt_world_ewrt_world` (at the body's center of mass, `linkcom`; split from the (angular, linear) output, which is not a `twist`), `_rotmat_of_yaw`, `_random_unit_quaternion` |
| `humanoid_learning/acom/train_main.py` | `force_architecture` ("force" the verb) | `replace_architecture`; the `--force_architecture` flag keeps its spelling, with `dest=` |
| robot model playgrounds (`testPinocchioModel.cpp`) | `printFrameRotation`, `rotationLocalToWorld` / `quaternionLocalToWorld`, `testOrientationErrorWrtPlane`, `quaternionCorrection`, `footWrenches`, `baseForceAtZeroBaseAcceleration` (RNEA's base rows) | `printPoseLinkRtWorld`, `rotmatLinkRtWorld` / `quatLinkRtWorld`, `printPlaneTiltError`, `shortestArc`, `wrenchFeetEwrtWorld`, `baseGeneralizedForceAtZeroBaseGeneralizedAcceleration` |
| `robot_models/tests` | `_rotation_about(axis, angle)`, `positions` (joint angles by name), `moments` (a URDF inertia's entries), `child_rotation` / `child_quat` (a fixed joint's origin) | `_matrix_from_axis_angle`, `joint_positions`, `inertia_entries`, `child_origin_rotmat` / `child_origin_wxyz` |
| `tools/locomotion_heuristics` | `largest_force_ratio`, the initial state's `position` / `orientation` / `quaternion`, `foot_positions`, `base_position` | `largest_weight_compensation_ratio` (as `maximumWeightCompensationRatio`), `pos_base_rt_world_ewrt_world` / `eulerzyx_base_rt_world` / `quat_base_rt_world`, `pos_feet_rt_world_ewrt_world`, `pos_base_rt_world_ewrt_world_xy` |
| `tools/config_dump`, `tools/ipc` | `targetBasePosition`, `kSaturatingLinearVelocity`; `message_decoding.py`'s `position` (an index into a field path) | `refPosBaseRtWorldEwrtWorld`, `kSaturatingCmdLinvelBaseRtWorldEwrtHeadingXy`; `depth` |
| `ContactPlannerInput`, `ContactPlan` (the heading model) | the input's `yaw` / `heading` / `headingRate` / `footYaws`; the plan's `yaw` / `heading` / `headingRate` / `footYaws`, `headingAtTime` / `headingRateAtTime` / `footYawAtTime` / `footYawBeforeTime`; `HeadingNominal::heading` / `footYaws`; `TargetContactPoseInputs::footYaws`; the reference manager's `computeHeading`, `readFootYaws`, `footYaws_` / `liftOffYaws_` | `yawHeadingRtWorld` (the frame the planner plans in), `initYawHeadingRtWorld` / `initRateYawHeadingRtWorld` (the heading model's initial state, section 6), `eulerzyxFeetRtWorldYaw` (the ZYX yaw `atan2(R10, R00)` the manager reads); `yawHeadingRtWorld`, `headingModelYawTraj` / `headingModelYawRateTraj` / `desEulerzyxFeetRtWorldYawTraj`, `headingModelYawAtTime` / `headingModelYawRateAtTime` / `getDesEulerzyxFootRtWorldYaw` / `getPrevDesEulerzyxFootRtWorldYaw`; `nomHeadingModelYawTraj` / `nomEulerzyxFeetRtWorldYawTraj`; `eulerzyxFeetRtWorldYaw`; `computeYawAcomRtWorld` (empty without the ACoM evaluator), `readEulerzyxFeetRtWorldYaw`, `eulerzyxFeetRtWorldYaw_` / `liftoffEulerzyxFeetRtWorldYaw_` |
| contact planner `Layout`, `LipIndices`, `var::` (problem/Layout.h) | `var::kComX` / `kComY` / `kZmpX` / `kZmpY`, `var::footX` / `footY` / `footDeltaX` / `footDeltaY` / `footYaw` / `footYawDelta`, `var::kHeading` / `kHeadingRate`; `Layout::heading` / `headingRate` / `footYaw0` / `footYawDelta0` / `footYaw()` / `footYawDelta()`; `LipIndices::heading` / `headingRate` / `footYaw` / `footYawDelta` | `var::kPosComRtWorldEwrtWorldX` / `Y`, `var::kPosZmpRtWorldEwrtWorldX` / `Y`, `var::posFootRtWorldEwrtWorldX` / `Y`, `var::errPosFootRtWorldEwrtWorldX` / `Y`, `var::eulerzyxFootRtWorldYaw`, `var::errEulerzyxFootRtWorldYaw`, `var::kHeadingModelYaw` / `kHeadingModelYawRate`; `Layout::headingModelYaw` / `headingModelYawRate` / `startEulerzyxFeetRtWorldYaw` / `startErrEulerzyxFeetRtWorldYaw` / `eulerzyxFootRtWorldYaw()` / `errEulerzyxFootRtWorldYaw()`; `LipIndices::headingModelYaw` / `headingModelYawRate` / `eulerzyxFeetRtWorldYaw` / `errEulerzyxFeetRtWorldYaw`. The variable strings (`"c_x"`, `"psi_L"`, ...) stay |
| CoM height above the soles | `computeComHeightAboveFeet`, `CentroidalMpcInterface::nominalComHeight_` / `getNominalComHeight`, `DcmTerminalCost::Config::comHeight`, `modelComHeight_`, `measuredComHeight_`, `FootholdHeuristicContext::comHeight`, `LocomotionHeuristicModelParameters::nominalComHeight` | `computeProxyPosComRtFeetmeanEwrtWorldZ` (the new frame `feetmean`; split into `computePosComRtFeetmeanEwrtWorldZ` for the biped contact set and `computePosComRtLowcontactEwrtWorldZ` for any other, which the proxy stands in with), `nomPosComRtFeetmeanEwrtWorldZ_` / `getNomPosComRtFeetmeanEwrtWorldZ`, `nomPosComRtFeetmeanEwrtWorldZ`, `modelNomPosComRtFeetmeanEwrtWorldZ_`, `measPosComRtFeetmeanEwrtWorldZ_`, `measPosComRtFeetmeanEwrtWorldZ`, `nomPosComRtFeetmeanEwrtWorldZ`; the proto fields `dcm_terminal_cost.com_height` and `shared.com_height` are `nom_pos_com_rt_feetmean_ewrt_world_z`, and the contact planner's `shared.comHeight` is `shared.nomPosComRtFeetmeanEwrtWorldZ` |
| `ContactPolygon`, `createPinocchioModel.cpp`, contact cones | `getContactPointTranslation(i)`, `getContactPointTranslationCrossProductMatrix(i)`, `contactPointTranslation`; `footprintCorners`, the cones' `corners` / `corner` / `center` | `getPosCornerRtContactEwrtContact(i)` (the new frame `corner`), `getCrossProductMatrixPosCornerRtContactEwrtContact(i)`, `posCornerRtLinkEwrtLink`; `computePosCornersRtContactEwrtContactXy`, `posCornersRtContactEwrtContactXy` / `posCornerRtContactEwrtContactXy` / `posFootprintRtContactEwrtContact` (the new frame `footprint`) |
| multi-contact collision and terrain | `CollisionSphere::center`, `ClearanceSphere::center`, `PadRegion::center`, `BoxGeometry::center`; proto `CollisionModel.Box.center` / `Sphere.center`, `TerrainScene.Box.center`, `Scenario.BaseRegion.center`, `Spheres.centers` | `posSphereRtLinkEwrtLink` (in the joint's frame), `posSphereRtLinkEwrtLink`, `posPadRtWorldEwrtWorld`, `posBoxRtWorldEwrtWorld`; `pos_box_rt_link_ewrt_link` / `pos_sphere_rt_link_ewrt_link`, `pos_box_rt_world_ewrt_world`, `pos_box_rt_world_ewrt_world`, `pos_spheres_rt_world_ewrt_world` |
| `ForceWeightedSlipConstraint` | `inverseScaleSlip_`, `getInverseScaleSlip`, `setSlipScales`, `scaledSlip` | kept: the slip residual `[v_x, v_y, omega_z]` of a contact mixes a linear velocity with an angular one, so it is no one quantity; its rows are formed from `linvelContactRtWorldEwrtWorld` and `angvelContactRtWorldEwrtWorld`, which carry the frames |
| `DynamicsHelperFunctions.h` | `computeContactCoP` (both overloads; the input-only one treated the `param` frame as the world), `copX` / `copY` | `computePosCopRtWorldEwrtWorld` (the state-aware overload only), `posCopRtContactEwrtContactX` / `posCopRtContactEwrtContactY` |
| `MpcRobotModelBase` | `getTargetPlanarCommand`, `kPlanarCommandYawRateIndex` | `getRefPlanarMotion`, `kRefPlanarMotionYawRateIndex` |
| `TargetTrajectoriesCalculatorBase` | `planarArcDisplacement` | `computeErrPosBaseRtWorldEwrtWorldXy` |
| `SwitchedModelReferenceManager` | `getSwingFootPlaneNormal` | `getSwingRefNormalSurfaceEwrtWorld` |
| the ground's height (`TaskFile`, `ModelSettings`, `SwitchedModelReferenceManager`, `TargetTrajectoriesCalculatorBase`, `ContactComplementarityConstraint`, `GroundPenetrationConstraint`, `ContactImplicitApplier`, `SwingTrajectoryPlanner`) | proto `terrain_height`; `terrainHeight` / `terrainHeight_`, `setTerrainHeight` / `getTerrainHeight` / `getAppliedTerrainHeight`, `appliedTerrainHeight_`, `targetTerrainHeight_`, `setTerrainHeightSource` / `terrainHeightSource_`, `termsTerrainHeight_` / `setContactImplicitTermsTerrainHeight`, `terrainHeightSequence` / `touchDownTerrainHeightSequence` | `pos_surface_rt_world_ewrt_world_z`; `posSurfaceRtWorldEwrtWorldZ` / `posSurfaceRtWorldEwrtWorldZ_`, `setPosSurfaceRtWorldEwrtWorldZ` / `getPosSurfaceRtWorldEwrtWorldZ` / `getAppliedPosSurfaceRtWorldEwrtWorldZ`, `appliedPosSurfaceRtWorldEwrtWorldZ_`, `refPosSurfaceRtWorldEwrtWorldZ_` (the ground the target in use was built on), `setSourcePosSurfaceRtWorldEwrtWorldZ` / `sourcePosSurfaceRtWorldEwrtWorldZ_`, `termsPosSurfaceRtWorldEwrtWorldZ_` / `setContactImplicitTermsPosSurfaceRtWorldEwrtWorldZ`, `posSurfaceRtWorldEwrtWorldZTraj` / `touchdownPosFootRtWorldEwrtWorldZTraj`; the retired `contact_implicit.terrain_height` and the config dump keep their names |
| `SwingTrajectoryPlanner::Config` | proto `swing_height`; `swingHeight`, `kDefaultSwingHeight` (the apex above the higher end of the swing) | `apex_pos_foot_rt_swingend_ewrt_world_z`; `apexPosFootRtSwingendEwrtWorldZ`, `kNomApexPosFootRtSwingendEwrtWorldZ` (the new frame `swingend`); the config dump keeps its key |
| contact planning | `computeDcm`, `dcmStepAdjustment()`, `ContactPlanningReferenceManager::dcmStepAdjustment_` / `getDcmStepAdjustment`, `TargetContactPoseInputs::dcmStepAdjustment`, `StepLengthCost::nominalDisplacementPerNode`, `ContactPlanningModelParameters::comHeight`, `CapturePointParameters::comHeightOverride` | `computePosDcmRtWorldEwrtWorldXy`, `computeErrPosFootRtWorldEwrtWorldXy`, `errPosFeetRtWorldEwrtWorldXy_` / `getErrPosFeetRtWorldEwrtWorldXy`, `errPosFeetRtWorldEwrtWorldXy`, `computeNomErrPosFootRtWorldEwrtWorldXy`, `nomPosComRtFeetmeanEwrtWorldZ`, `overridePosComRtFeetmeanEwrtWorldZ`; the proto field `capture_point.com_height_override` is `override_pos_com_rt_feetmean_ewrt_world_z` |
| `ContactWrenchConeConstraint` | `Config::patchOffset`, `contactPatchReferencePoint` | `posPatchpointRtContactEwrtContact` (the new frame `patchpoint`), `computePosPatchpointRtContactEwrtContact` |
| `DcmTerminalCost` | `computeDcmError`, `computeDcmErrorJacobian` | `computeErrPosDcmRtWorldEwrtWorldXy`, `computeJacobianErrPosDcmRtWorldEwrtWorldXy` |
| `mujoco_sim_interface` | `contactPatchWorldCorners(patch, corners, heightOffset)`, `kMjFreeJointAngularVelocityOffset`, `kMjMotionLinearOffset`, `kMinDrawnYawRate` | `computePosCornersRtWorldEwrtWorld(patch, posCornersRtFootEwrtFootXy, offsetPosCornersRtFootEwrtWorldZ)`, `kMjFreeJointAngularVelocityStartIndex`, `kMjMotionLinearStartIndex`, `kMinDrawnAngvelBaseRtWorldEwrtWorldZ` |
| proto `ContactsConfig.Rectangle`, `ContactSet.Rectangle` | `x_min` / `x_max` / `y_min` / `y_max` | `min_pos_corners_rt_contact_ewrt_contact_x` / `max_pos_corners_rt_contact_ewrt_contact_x` / `min_pos_corners_rt_contact_ewrt_contact_y` / `max_pos_corners_rt_contact_ewrt_contact_y`, as the C++ `PolygonBounds`; in the contact set's pad frame `..._corners_rt_pad_ewrt_pad_...` |
| proto `MultiContactPlanConfig.Swing`, `ContactSwingSettings` | `lift_height`; `liftHeight`, `kDefaultLiftHeight` (the least distance of the Bezier arc's inner control points from its ends) | `min_dist_controlpoint_rt_contact`; `minDistControlpointRtContact`, `kNomMinDistControlpointRtContact` (the new frame `controlpoint`) |
| proto `SwingTrajectoryConfig`, `SwingTrajectoryPlanner` | `swing_pitch_angle`; `Config::swingPitchAngle`, `getSwingPitchAngle(leg, time)` | `swing_max_tiltangle_foot_rt_surface`; `swingMaxTiltangleFootRtSurface`, `getSwingRefTiltangleFootRtSurface(leg, time)` (the toe-up tilt from the ground) |
| proto `ContactPlanningFile`, `ContactPlanningConfig` | `hlip.blend.max_linvel_com_rt_world_ewrt_heading_x` / `_y`; `maxLinvelComRtWorldEwrtHeadingX` / `Y` | `max_proxy_linvel_com_rt_world_ewrt_heading_x` / `_y`; `maxProxyLinvelComRtWorldEwrtHeadingX` / `Y` (the blend reads `getProxyLinvelComRtWorldEwrtWorld()`) |
| `tools/locomotion_heuristics/derive_parameters.py` | `PELVIC_OBLIQUITY`, `inward_room` / `outward_room`, `capture_clamp`, `needed`, `max_pos_lhip_rt_rhip_ewrt_heading_y`, `RobotGeometry.max_pos_hips_rt_base_ewrt_heading_y` | `AMPLITUDE_EULERZYX_BASE_RT_WORLD_ROLL`, `inward_err_pos_lfoot_rt_rfoot_ewrt_heading_y` / `outward_err_pos_lfoot_rt_rfoot_ewrt_heading_y`, `max_norm_offset_pos_foot_rt_world_ewrt_world_xy`, `cmd_max_wnorm_force_com_ewrt_world_xy`, `nom_norm_pos_lhip_rt_rhip_ewrt_heading_y` (the hip width, not a bound), `max_norm_pos_hips_rt_base_ewrt_heading_y` |
| robot model and learning tests, `train_acom.py`, `plan_playback.py` | `tilt_angle_deg`, `bottom` (a sole sphere's), `com_error`, `headings`, `offset_norms_deg` / `mean_offset_deg` / `max_offset_deg`, `polygon` | `tiltangle_link_rt_world`, `min_pos_sphere_rt_contact_ewrt_link_z`, `max_norm_err_pos_linkcom_rt_link_ewrt_link`, `yaw_base_rt_world_samples`, `norm_rotvec_acom_rt_base_samples` / `mean_norm_rotvec_acom_rt_base` / `max_norm_rotvec_acom_rt_base` (in degrees, as their comment says), `support_pos_com_rt_world_ewrt_world_samples` |

## 8. The units of the rename

<!-- LINT.IfChange(units) -->
The check reads these directories (`DIRS` of `tools/hooks/frame_naming.py`): `lib/ocs2` (not its `thirdparty`),
`humanoid_nmpc`, `robot_runtime`, `humanoid_state_estimation`, `robot_models`, `humanoid_learning`,
`tools/config_dump`, `tools/config_registries`, `tools/locomotion_heuristics` and `tools/ipc`.
<!-- LINT.ThenChange(//tools/hooks/frame_naming.py:dirs) -->

### 8.1 Order

The rename runs bottom-up, so that each unit renames what it declares while its callers are still the old code. A
unit owns the identifiers *declared* in it (functions, members, constants, fields) and the locals of its own files.

| # | Unit | Paths | Why here |
|---|---|---|---|
| 1 | OCS2 | `lib/ocs2/core`, `lib/ocs2/robotic_tools`, `lib/ocs2/pinocchio`, `lib/ocs2/oc`, `lib/ocs2/mpc`, `lib/ocs2/sqp` | everything depends on it |
| 2 | Robot model | `robot_runtime/robot_model`, `robot_runtime/robot_core`, `robot_runtime/robot_realtime`, `robot_runtime/robot_ipc` (C++) | `RobotState` is read by the simulator, the estimator and the apps |
| 3 | Message protos | `humanoid_nmpc/humanoid_mpc_msgs`, `humanoid_nmpc/humanoid_rerun_viewer/proto`, `robot_runtime/robot_ipc/proto` | the bus; every C++ and Python user of the fields is updated with it |
| 4 | Configuration protos | `humanoid_nmpc/humanoid_mpc_config` and the protos of `humanoid_nmpc/humanoid_multi_contact` | every textproto, the GUI, `tools/config_dump`, `tools/config_registries` are updated with it |
| 5 | Common MPC | `humanoid_nmpc/humanoid_common_mpc` (split by subdirectory if large: `common`, `orientation`, `pinocchio_model`, `dynamics` / `cost` / `constraint` / `end_effector`, `contact_planning`, `reference_manager` / `command`, `locomotion_heuristics`, the rest) | the formulations derive from `MpcRobotModelBase` |
| 6 | Centroidal MPC | `humanoid_nmpc/humanoid_centroidal_mpc`, `humanoid_centroidal_mpc_test`, `humanoid_centroidal_mpc_app` | |
| 7 | Whole-body MPC | `humanoid_nmpc/humanoid_wb_mpc`, `humanoid_wb_mpc_app` | |
| 8 | Multi-contact | `humanoid_nmpc/humanoid_multi_contact` (C++ and Python) | |
| 9 | Apps and IPC | `humanoid_nmpc/humanoid_common_mpc_app`, `humanoid_nmpc/humanoid_mpc_ipc` | |
| 10 | Validation | `humanoid_nmpc/humanoid_mpc_validation` | |
| 11 | State estimation | `humanoid_state_estimation` | |
| 12 | Simulator | `robot_runtime/mujoco_sim_interface`, `robot_runtime/web_viewer` | |
| 13 | Python | `humanoid_nmpc/remote_control`, `humanoid_nmpc/humanoid_rerun_viewer`, `humanoid_nmpc/humanoid_common_mpc_pyutils`, `humanoid_learning`, `tools/ipc`, `tools/locomotion_heuristics` | |
| 14 | Robot models and tools | `robot_models`, `tools/config_dump`, `tools/config_registries` | |
| 15 | Verify | the whole tree | switches the check on (section 9) |

### 8.2 What each rename agent does

1. List the unit's findings:
   `CI=true python3 -m tools.hooks.lint_code --only frame-naming --paths <unit paths>`.
2. For each identifier the unit declares, pick the new name from sections 2 to 7, deciding unclear frames by section
   6. Rename the declaration and **every use in the whole repository** in the same change, across units: C++, Python,
   proto, textproto, Markdown that names the identifier in backticks, and string literals that spell a field name
   (the GUI registries, `config_registries.textproto`, `getattr` / `HasField` strings, JSON keys). A later unit then
   finds only its own declarations left.
3. Keep the IFTTT pairs: a renamed identifier inside a `LINT.IfChange` block is renamed on the `ThenChange` side too.
4. Keep the comments true: a comment that says "in the world frame" next to a name that now says so may shorten, but
   never contradicts the name. Update the docs under `humanoid_nmpc/docs` that cite a renamed identifier.
5. Keep the other rules: no `auto`, no CTAD, nullability annotations, American spelling, argument comments (an
   argument comment names the callee's new parameter name), `euler-boundary` (`eulerzyx` is an Euler word: only at
   the boundary).
6. Format and lint in the dev container:
   `tools/deploy/in_dev_container.sh 'cd /wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc && make format'`, then
   `CI=true python3 -m tools.hooks.lint_code --paths <changed dirs>` (every check, IFTTT included) and the
   `frame-naming` run of step 1.
7. Build and test, package by package, one bazel command at a time, with `--jobs=4 --local_test_jobs=2`: the unit's
   packages and every package whose files the change touched, e.g.
   `tools/deploy/in_dev_container.sh 'cd /wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc && bazel test --jobs=4 --local_test_jobs=2 //humanoid_nmpc/humanoid_common_mpc/...'`.
   Never `//...` in one command. If a command is killed for low memory, stop and report it.
8. Leave the changes uncommitted.

### 8.3 Protos, configurations and recorded data

- A renamed field keeps its field number; never `reserved` the old name, and do not list it in
  `(nproto.retired_field)` (that retires a field; this one lives on under a new name). Binary data stays valid: the
  shipped multi-contact plans `humanoid_nmpc/humanoid_multi_contact/plans/*.binpb` are not regenerated.
- Text data names fields, so it changes with them: every `.textproto` under `robot_models/` and `humanoid_nmpc/`,
  the plans' `.summary.textproto` files, the test data and the recorded text logs. Grep the old field name in every
  text file, `grep -rn "\b<old_name>\b" --include=*.textproto --include=*.py --include=*.cpp --include=*.h --include=*.md .`,
  and its camelCase JSON spelling where JSON is read.
- `tools/config_dump/testEveryFieldIsConsumed.cpp` lists config fields by name; the config registries and the GUI's
  parameter registry name them too. Update them in the same change.
- Goldens and expected outputs are never regenerated to make a test pass, `testCentroidalContactSetParity` included.
  A golden that stores field names in text is renamed by hand, value by value unchanged, and the change says so.
- Retired-field lists (`(nproto.retired_field)`) keep the historical names they list.

## 9. The check

`frame-naming` (`tools/hooks/frame_naming.py`) reads C++, Python and `.proto` in the directories of section 8. It
reports:

- an identifier with an `rt` or `ewrt` word that departs from the grammar: an unknown quantity or frame, the wrong
  form, a misspelled or misordered qualifier, an unknown suffix, two quantities in one name;
- a frame name spelled in the wrong case style for its kind (section 2.4): a mixed spelling anywhere
  (`getPos_base_rt_world_ewrt_world`), a camelCase or kName name in Python or a proto; the message respells it;
- a legacy name: an identifier with a legacy word (`position`, `pos`, `velocity`, `vel`, `orientation`, `quaternion`,
  `quat`, `rotation`, `rot`, `acceleration`, `accel`, `force`, `torque`, `wrench`, `twist`, `momentum`, `moment`,
  `pose` and their plurals) and no exemption of section 5;
- a C++ or `.proto` declaration of a fixed-size vector (`vector3_t`, `Eigen::Vector3d`, `Vector3`, `AdVector3`,
  `VECTOR3_T<...>` and their 2-D kin) or of a container of them, whose name, its qualifiers aside, is only frame words
  and the point words `center` and `centers` (`center`, `com`, `stanceFoot`, `feet`): a vector named for the point it
  belongs to, without its quantity or frames. A name with another word (`normal`) is left to the reader;
- a `TODO(frame-naming)`, in comments and strings too (section 6).

Comments and string literals are not read otherwise. One finding is reported per name and line. A genuine exception is
`// NOLINT(frame-naming): <reason>`.

```bash
# The findings left in a unit:
CI=true python3 -m tools.hooks.lint_code --only frame-naming --paths humanoid_nmpc/humanoid_common_mpc
# The count per Bazel package:
CI=true python3 -m tools.hooks.lint_code --only frame-naming --summary
```

The check ran opt-in while the rename was under way. The Verify phase (unit 15) switched it on once it reported
nothing for the whole tree, and deleted the `opt_in` field of `Check` with it: `make lint`, the pre-commit hook and CI
run `frame-naming` like every other check.
