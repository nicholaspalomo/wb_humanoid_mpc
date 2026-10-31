# mujoco_sim_interface

MuJoCo simulation backend (`robot::mujoco_sim_interface`), viewer backends (`glfw`, `web`, `none`), and scene
visualizations for the robot runtime (`robot_runtime/robot_control`). MuJoCo is fetched and built from source by Bazel
(`@mujoco` in `MODULE.bazel`).

## Creating a simulator

`MujocoSimInterface::Create(config, urdfPath)` loads the scene of `MujocoSimConfig::scenePath` (compiled with the ball
of `sim_projectile`, if the task file names one) and returns the simulator, or an `absl::Status` saying why it could not:
a scene that does not load, parse or compile, an unknown `sim_projectile` or `gantry_hold` name, viewer settings
`validateViewerSettings()` refuses, simulation data MuJoCo could not create, or a URDF `RobotDescription::Create()`
refuses.

`initSim()` starts the viewer that `MujocoSimConfig::viewer` names (see [Viewers](#viewers)) on threads of its own, and
`viewerStatus()` says whether it did.

## Viewers

A viewer only displays the simulation: nothing it receives (keys, mouse, or browser requests) reaches the physics,
the controller, the gantry or its weld. It is selected by registry name (`include/mujoco_sim_interface/viewer/`):

<!-- LINT.IfChange(viewer_names) -->
| Name | Viewer | What it needs |
|---|---|---|
| `glfw` | `GlfwMujocoViewer`: a GLFW window on `DISPLAY`, the X11/VNC desktop of the containers (`MujocoSimConfig`'s default) | an X display |
| `web` | `WebMujocoViewer`: the simulation in a browser tab, at `http://localhost:6090` by default (the robot app's default, and `make launch-<robot>-sim`'s) | EGL (`libegl1`, `libegl-mesa0`) |
| `none` | `NoMujocoViewer`: no viewer, and no thread | nothing |
<!-- LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/viewer/ViewerNames.h:viewer_names) -->

`MujocoSimConfig::viewer.names` (the robot app's `--viewer`, or without it the `mujoco_viewer` of its `--plot_config`,
the plot configuration the Rerun bridge reads too: `humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto`
names `web`, and `make launch-<robot>-sim VIEWER=glfw` brings back the window) is a preference list: `initSim()`
creates and starts each viewer in turn, destroys one that fails before it tries the next (so a port it bound is free
again), and keeps the first that starts. `web` alone refuses to run without the web viewer (`viewerStatus()` is a FailedPrecondition naming
why each viewer failed, and the robot app stops at start-up), while `web,none` falls back to no viewer with a WARNING.
`parseViewerNames()` refuses unknown, empty and repeated names, anything after `none` (which always starts) and anything
after `glfw`, which cannot fail at start-up: a window it cannot open is logged, `ok()` turns false and the simulation
runs on without it, as it always has.

The two viewers that draw share `MujocoFrameRenderer`: the camera, the visualizations below, the keys (`0`-`5` geometry
groups, `k` camera tracking, `p` the cheatsheet, every visualization's hotkey) and the mouse (left drag rotates, right
drag moves, the wheel and the middle drag zoom, shift for the horizontal variants). Both draw with a private copy of
the simulator's model, made on the simulator's thread before the physics starts, and bring it up to date with each
snapshot's run-time model arrays, so neither render thread reads the model the physics thread writes.

### The web viewer

Run the robot with `--viewer=web` (or with neither `--viewer` nor a plot configuration that names another, as `make
launch-<robot>-sim` does) and open `http://localhost:6090` (the dev and robot containers use the host's
network, so nothing needs to be published). From another machine, forward the port: `ssh -L 6090:127.0.0.1:6090 <host>`,
or VS Code's port forwarding, and open the forwarded address in a browser tab ("Open in Browser"): an editor preview
that frames the page shows a blank pane, since the viewer refuses to be framed. The page pulls one frame at a time, so a slow tunnel delays the picture by at most a round
trip and a frame instead of queueing frames. Click the picture to give it the keys; the same keys and mouse bindings as
the GLFW window apply. The server binds 127.0.0.1 unless `WebViewerOptions::address` says otherwise; its routes, its
trust model and its limits are [robot_runtime/web_viewer/README.md](../web_viewer/README.md).

How it renders:

- A render thread of its own creates a desktop OpenGL context through EGL with no X server: on an EGL device that is a
  GPU (`MUJOCO_EGL_DEVICE_ID` picks one), else on Mesa's surfaceless platform (llvmpipe). `libEGL.so.1` is loaded with
  `dlopen()` (`EglLibrary`), so the robot binary links no EGL; without it, `web` fails with a Status that names the
  packages to install. A GL preflight (`checkGlCapabilities()`) refuses, before MuJoCo would end the process, a context
  without framebuffer objects, an offscreen buffer past `GL_MAX_RENDERBUFFER_SIZE`, more samples than
  `GL_MAX_SAMPLES` or a shadow map past `GL_MAX_TEXTURE_SIZE`.
- It draws with a private copy of the model, made by `WebMujocoViewer::Create()` before the physics starts, and reads
  the simulator's state snapshots only, so it never reads the model the physics thread writes. Each snapshot carries
  the model arrays the simulator changes while it runs (`MjRuntimeModelFields`: the ragdoll damping, the gantry weld's
  relpose, the ball's collision flags, mass and gravity compensation), and each frame writes them into the copy before
  it computes the contacts and forces it draws, so they are the physics', as in the GLFW window, which does the same. Whether the gantry weld
  holds is the snapshot's own `mjData::eq_active`, so the viewer computes the weld only while the robot hangs from the
  gantry, never while it walks.
- Nothing is drawn while no browser watches. With one, it draws, reads the pixels back (`mjr_readPixels`) and encodes
  them as JPEG on the render thread, at up to `fps` (30) frames per second at 1280x720 by default, without multisampling
  and with 2048-pixel shadow maps (`WebViewerOptions`).
- The render thread, the server's threads and llvmpipe's rasterizer threads run on the viewer's cores
  (`WebViewerOptions::cores`, the robot app's `--viewer_cores`) under `SCHED_IDLE` (`WebViewerOptions::threadPriority`):
  they get only the time no other thread wants, and the physics or an MPC thread that wakes on their core takes it at
  once. A higher nice value does not do that: under EEVDF a thread that wakes may wait out the running thread's time
  slice, 3 ms on a machine of 8 or more CPUs, whatever its nice value, and at nice +10 the viewer on the physics' cores
  made 4.6 to 9.5 % of the physics' steps more than 1 ms late (below). One Mesa thread places itself: the shader disk
  cache's queue (`disk$0`) runs on every core, the physics' and the MPC's included, at nice 19. In a process that may
  not raise its scheduling (no `CAP_SYS_NICE`, as in the dev container) it keeps the viewer's `SCHED_IDLE`; in one that
  may, Mesa can move it to a normal policy at nice 19. It has work only when a newly compiled shader is written to the
  cache: in the first frames, and when a marker first appears.
- llvmpipe starts a rasterizer thread per CPU unless `LP_NUM_THREADS` sets their number. When `--viewer` lists `web`
  and the environment does not set it, the robot binaries set it to the number of viewer cores (at least 2) before they
  start a thread (`exportSoftwareRasterizerThreads()`); the robot-sim container sets 2. Another program that starts the
  viewer sets it itself: the viewer warns when it is unset.

Expect about 16 frames per second for Atlas at 1280x720 on llvmpipe with 2 threads, and 25 with 4; 30 needs a GPU. The
cost of one Atlas frame on llvmpipe at 1280x720 (median of 18 frames, idle machine):

| `LP_NUM_THREADS` | multisampling | shadow map | draw (`mjv_updateScene` + `mjr_render`) | `mjr_readPixels` |
|---|---|---|---|---|
| 2 | 4 (the model's) | 4096 (MuJoCo's default) | 86 ms | 43 ms |
| 2 | 4 | 2048 | 69 ms | 42 ms |
| 2 | 0 | 4096 | 61 ms | 15 ms |
| 2 | 0 | 2048 (the viewer's default) | 45 ms | 16 ms |
| 2 | 0 | 1024 | 40 ms | 16 ms |
| 4 | 0 | 1024 | 31 ms | 11 ms |

Encoding a frame as JPEG takes about 1.2 ms.

At a realtime priority the robot process locks all of its memory (`mlockall`), the viewer's threads and Mesa's included.
Measured in the robot-sim image on the DRC Atlas with one client pulling frames (`tools/deploy/README.md`,
"Simulation"), well inside the robot container's 4 GB cap:

| Viewer | priority | resident (`VmRSS`) | locked (`VmLck`) | threads |
|---|---|---|---|---|
| `web` | 80 | 915 MB | 2.49 GB | 28 |
| `none` | 80 | 315 MB | 577 MB | 7 |
| `web` | 0 | 402 MB | 0 | 28 |

### The viewer never delays the physics

The physics thread hands the viewer state snapshots through a lock-free triple buffer, and shares exactly one lock with
it: the contact timeline's. Its side takes that lock with `try_lock`: a sample that finds the viewer copying the timeline
waits for the next one, and of two that find it so the older is dropped (`droppedContactTimelineSamples()`); the
viewer's copy goes into a vector reserved to the timeline's capacity, so it never allocates under the lock. The physics
loop records how late it finishes every step (`StepLatenessHistogram`, `stepLateness()`).

The `test_web_viewer_sim_rate` targets run Atlas in real time, once without a viewer and once with the web viewer (under
`SCHED_IDLE`, as shipped) and a browser pulling frames, with the contact timeline and the target contact patches drawn.
Each places the physics and the viewer as `ThreadAffinity.h`'s default does on a machine of some size, in a process of
its own (llvmpipe's rasterizer threads, started once per process, keep the placement of its first render thread):

<!-- LINT.IfChange(sim_rate_targets) -->
| Target | physics | viewer | browser | `LP_NUM_THREADS` | the default with |
|---|---|---|---|---|---|
| `test_web_viewer_sim_rate` | 2 cores | 2 other cores | on the viewer's | 2 | more than 16 cores |
| `test_web_viewer_sim_rate_oversubscribed` | 2 cores | 2 other cores | on the viewer's | 8 | |
| `test_web_viewer_sim_rate_on_the_physics_cores` | 2 cores | the same 2 | 2 other cores | 2 | 8 to 16 cores |
| `test_web_viewer_sim_rate_on_the_physics_core` | 1 core | the same | 2 other cores | 2 | 4 to 7 cores |
<!-- LINT.ThenChange(//robot_runtime/mujoco_sim_interface/BUILD.bazel:sim_rate_targets) -->

All are skipped with fewer than 4 allowed cores. With the viewer, the 99th percentile of the step lateness may exceed
the run without it by at most the first margin below, the share of steps more than 1 ms late by the second, the maximum
by the third, and the real-time factor must keep the ratio of the fourth:

<!-- LINT.IfChange(sim_rate_bounds) -->
| p99 lateness margin | late-step (> 1 ms) share margin | max lateness margin | real-time factor ratio |
|---|---|---|---|
| 0.25 ms | 0.5 % of the steps | 10 ms | 0.98 |
<!-- LINT.ThenChange(//robot_runtime/mujoco_sim_interface/test/testWebViewerSimRate.cpp:sim_rate_bounds) -->

Calibration, 2026-10-07, in the dev container on a 20-core workstation (cores 0-1 and 18-19 are the targets' pairs),
6000 steps (3 s) per window, five or more runs of each target:

| | windows | p99 lateness | max lateness | steps > 1 ms late | real-time factor | frames pulled |
|---|---|---|---|---|---|---|
| no viewer | 41 | 0 to 5 us | 0 to 3.87 ms | 0 in 33, 3 to 10 in 8 | 0.99994 to 1.00015 | - |
| web on cores of its own, `LP_NUM_THREADS=2` | 5 | 0 | 0 | 0 | 0.99995 to 0.99998 | 36 to 38 (12 fps) |
| web on cores of its own, `LP_NUM_THREADS=8` | 5 | 0 to 5 us | 0 in 4, 2.70 ms | 0 in 4, 11 | 0.99996 to 1.00011 | 34 to 36 (12 fps) |
| web on the physics' two cores | 16 | 5 us | 0.18 to 0.70 ms in 14, 2.93 ms, 26.9 ms | 0 in 14, 5, 1 | 0.99995 to 0.99998 (one 0.991) | 47 to 50 (16 fps) |
| web on the physics' core | 15 | 0 to 5 us | 0 in 12, 1.39 to 4.52 ms in 3 | 0 in 12, 2, 8, 20 | 0.99994 to 0.99998 | 31 to 33 (11 fps) |

No timeline sample was dropped in any run. The machine itself preempts the physics thread now and then, in a window in
five without a viewer here, which is why the bound is on the share of late steps and a generous maximum rather than on
a 2 ms maximum; the robot app's realtime priority and `mlockall` (which keep page faults off the physics thread) are not
part of this test. One window on the physics' two cores stopped the physics once for 27 ms (one late step, after which
the loop rebases its schedule; real-time factor 0.991), past the 10 ms bound: no other window showed a stop of that
length, with or without a viewer, and its cause is not known, so that target can fail now and then.

Two placements are not defaults, and fail the bounds:

- The viewer at nice +10 instead of `SCHED_IDLE`, on the physics' two cores (five windows each): p99 2.25 to 2.56 ms,
  a 3.0 to 3.3 ms maximum and 274 to 371 steps more than 1 ms late with `LP_NUM_THREADS=2`; p99 2.24 to 2.88 ms, up
  to 5.9 ms and 470 to 568 late steps with 8. The physics waits out the viewer threads' time slices.
- Under `SCHED_IDLE` but with more rasterizer threads than cores on the physics' cores (`LP_NUM_THREADS=8` on the two,
  eleven windows): seven within the bounds, four with 33 to 102 late steps and maxima of 5.2 to 8.1 ms, whether or not
  Mesa's shader disk cache is on. Hence the robot binaries' `LP_NUM_THREADS`, one rasterizer thread per viewer core:
  set it no higher where the viewer shares the physics' cores.

## The virtual gantry

The task file's `gantry_hold` names how the gantry holds the floating base while it is locked (`GantryHold` in
`MujocoSimInterface.h`). `weld_constraint`, the default, needs the scene to declare a weld named `gantry` from the world
(`body1`) to the body of the robot's free joint (`body2`):

```xml
<equality>
  <weld name="gantry" body1="world" body2="pelvis" relpose="0 0 1 1 0 0 0" active="false"/>
</equality>
```

The simulator anchors it where the robot is caught and writes the gantry height into its relpose, so the values in the
file are only a starting point. `checkSceneSupportsGantryHold()` checks a compiled scene against a hold; a simulator
whose scene cannot hold the robot by the weld logs an ERROR saying why and falls back to `kinematic_teleport`
(`MujocoSimInterface::gantryHold()` is the hold it runs on). Every shipped scene declares the weld (Atlas, SA01, G1 and
R1), and every task file names `weld_constraint`. `//humanoid_nmpc/humanoid_mpc_validation:test_robot_scene_gantry_hold`
checks every robot configuration's scene against its task file, and that the robot process's simulator runs on that
hold. `test_mujoco_sim_interface_gantry` holds each scene's robot by its weld, and checks that gravity-compensating
torques keep its limbs where they were caught, as GRAVITY_COMP does on a physical gantry, and that on
`kinematic_teleport` they do not.

## Viewer visualizations

Everything the viewer draws on top of the model is a class derived from `MujocoVisualization`
(`include/mujoco_sim_interface/visualization/`), one file pair per marker: the metrics text, the external and contact
force arrows, the base velocity arrows, the contact timeline barcode, the target contact patches, and MuJoCo's own
option-flag markers. A visualization implements up to three per-frame hooks (`beforeSceneUpdate`, `addSceneGeoms`,
`renderOverlay`), reports a `name()` and optionally a `hotkey()`.

Which visualizations run is decided at start-up by the `sim_visualizations` list of the robot's `task.textproto`
(`MujocoSimConfig::visualizations`): every listed name starts enabled, its hotkey toggles it, and `p` prints the
cheatsheet with the current state. Unknown names are reported with the available ones. A task file that lists none
starts none; a `MujocoSimConfig` built in code starts from `defaultVisualizationNames()`.

To add a marker: derive from `MujocoVisualization`, add the source to `BUILD.bazel`, and register it in the table of
`src/visualization/VisualizationRegistry.cpp` (drawing order is the table order); the IFTTT directive there points at
the task file comments to update with the new name.
