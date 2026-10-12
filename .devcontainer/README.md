# Visualization on macOS / Remote Hosts

This guide explains how to see the GUI apps running inside the dev container on your local computer or MacBook, when developing locally or via Remote SSH. There are three ways in, and none needs extra software on your local machine:

- **The VNC desktop (display `:99`)** shows the Tk operator GUI ("Robot Base Controller & Tuning"), and the MuJoCo window with `VIEWER=glfw`, in a browser tab, through noVNC (`http://localhost:6080/vnc.html`).
- **The MuJoCo web viewer**, the default of `make launch-<robot>-sim`, draws the simulation in a browser tab of its own at `http://localhost:6090`, without a display (see [MuJoCo Web Viewer](#mujoco-web-viewer) below).
- **Rerun** shows the robot's state, the MPC's plan and the plots of the telemetry. In the simulation environment it streams by default to a web viewer in your host browser at `http://localhost:9090` (see [Rerun](#rerun) below and `humanoid_nmpc/docs/distributed_runtime/README.md`).

## Checkout Location

The container expects the checkout at `<home>/workspace/wb_humanoid_mpc` on the host (`README.md`, "Repository
Setup"): `docker-compose.yaml` and `devcontainer.json` mount the checkout's grandparent at `/wb_humanoid_mpc_ws` and
open `/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc`, which `post_create.sh` marks as git's safe directory and the
shells' Bazel environment comes from. The host's launch targets (`tools/deploy/in_dev_container.sh`) and the editor's
clangd (`tools/ide/clangd_in_dev_container.sh`) find the container's checkout through the same mount.

## Quick Start

```bash
# Inside the container terminal:
make launch-drc-atlas-dummy-sim-vnc
# On the host (Linux), for the MuJoCo simulation in the hardware topology:
make launch-drc-atlas-sim-vnc
```

Then open the URLs printed in the terminal banner, and for the `-sim` target the MuJoCo web viewer at `http://localhost:6090`:
<!-- LINT.IfChange(vnc_ports) -->
- **Operator GUI, and the MuJoCo window with `VIEWER=glfw` (`:99`)**: `http://localhost:6080/vnc.html` (or `http://<host-lan-ip>:6080/vnc.html`). A VNC client can also connect directly to port `5901`.
<!-- LINT.ThenChange(//.devcontainer/start_vnc.sh:vnc_ports, //docker-compose.bridge.yaml:vnc_ports, //.devcontainer/devcontainer.json:vnc_ports, //Makefile:vnc_ports) -->
- **Rerun web viewer**: `http://localhost:9090/?url=rerun+http://localhost:9876/proxy` (or `http://<host-lan-ip>:9090/?url=rerun+http://<host-lan-ip>:9876/proxy`). If prompted for the server address in the viewer, enter `rerun+http://localhost:9876/proxy`.

Every GUI application in the container renders on `:99`; there is no second display.

## Ports

<!-- LINT.IfChange(ports) -->
On Linux the container runs on the host's network (`network_mode: host` in `docker-compose.yaml`): every process in it
listens on the host's own interfaces, so the ports below answer at the host's address with nothing to publish, and the
ZeroMQ bus reaches the processes of other containers and of other machines at the addresses of the network file. Two
containers with host networking share the host's ports, so run one VNC desktop at a time.

Docker Desktop (macOS, Windows) runs containers in a VM whose host network is not the computer's. There, add the
override `docker-compose.bridge.yaml`, which gives the container a bridge network of its own and publishes the ports
below, each at its own number:

```bash
docker compose -f docker-compose.yaml -f docker-compose.bridge.yaml up -d
```

For the IDE, add `"../docker-compose.bridge.yaml"` to `dockerComposeFile` in `.devcontainer/devcontainer.json`.
<!-- LINT.ThenChange(//docker-compose.yaml:network_mode, //docker-compose.bridge.yaml:published_ports) -->

The ports of the container's processes:

| Port | What |
|------|------|
| `6080` | noVNC, the browser view of display `:99` |
| `5901` | VNC, for a native VNC client |
| `6090` | The MuJoCo web viewer of the robot process (the `-sim` targets' default; [MuJoCo Web Viewer](#mujoco-web-viewer)) |
| `9090`, `9876` | Rerun: the web viewer and the gRPC server it, or a native viewer, reads from |
| `5600`-`5629` | The ZeroMQ bus: each node's PUB socket, at its port in the network file `config/ipc/network.textproto` |
| `8888` | Jupyter (`make train-acom-jupyter`) |

<!-- LINT.IfChange(bus_ports) -->
The bus range `5600`-`5629` covers every node of `config/ipc/network.textproto` (`5600` robot, `5610` MPC, `5620` operator GUI, `5621` teleoperation, `5622` push_robot_config), so a process on the host, or on another machine, can join the bus of the processes in the container. With the bridge network of `docker-compose.bridge.yaml`, a node that is to be reached from outside the container has to bind every interface (`bind_host: "0.0.0.0"` in the network file), not just the container's own loopback.
<!-- LINT.ThenChange(//docker-compose.bridge.yaml:bus_ports) -->

## Rerun

<!-- LINT.IfChange(rerun_ports) -->
The Rerun bridge (`humanoid_nmpc/humanoid_rerun_viewer`) maps the bus's messages onto Rerun. In the simulation environment, it runs with `--rerun_sink serve_web` by default. It serves the web viewer on port `9090` and the recording on port `9876` (gRPC), both published, so you only need to open `http://localhost:9090/?url=rerun+http://localhost:9876/proxy` in your host browser (or enter `rerun+http://localhost:9876/proxy` if prompted for the server address). A native viewer on the host reads the same recording from `rerun+http://localhost:9876/proxy`. Started with `RERUN_SINK=spawn`, the bridge instead opens the native viewer on `:99`, where it draws through Mesa's software GL and occupies the top of the left pane.
<!-- LINT.ThenChange(//docker-compose.bridge.yaml:rerun_ports, //.devcontainer/devcontainer.json:rerun_ports) -->

## MuJoCo Web Viewer

<!-- LINT.IfChange(web_viewer_port) -->
The robot process draws the MuJoCo simulation in a browser tab by default, instead of a window on `:99`: a `-sim`
target opens it at `http://localhost:6090` (the target prints the URL). The plot configuration chooses the viewer by
name (`mujoco_viewer` in `humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto`, or the file `PLOT_CONFIG=`
names), and `VIEWER=` overrides it:

```bash
make launch-drc-atlas-sim                  # then open http://localhost:6090 (the plot configuration's "web")
make launch-drc-atlas-sim VIEWER=web,none  # the same, but without a viewer where the web viewer cannot start
make launch-drc-atlas-sim-vnc VIEWER=glfw  # the window on :99, as before the web viewer
make launch-drc-atlas-sim VIEWER=none      # no viewer
```

It renders offscreen through EGL, with no X server and no VNC, on threads of its own: apart from the physics and the
realtime loop, and at a lower priority. The page fetches one JPEG frame at a time, so a slow link or a tunnel skips
frames instead of queuing them, and nothing is rendered while no page is open. The robot-sim container has no GPU and
renders with Mesa's llvmpipe: expect about 16 fps for the DRC Atlas at 1280x720 (the page's HUD shows the measured
rate, and its settings change the size, the frame rate and the JPEG quality). `--viewer=web` refuses to start when EGL
or the port is missing, and the message names the fix; `web,none` falls back to no viewer instead.

The page moves the viewer's camera and toggles its markers, and nothing else: none of its input reaches the physics,
the controller, the gantry and its weld, or the bus (the operator GUI drives those). It has no authentication and no
TLS, so it listens on `127.0.0.1:6090` only; from another machine, forward the port (`devcontainer.json` forwards
`6090`, or `ssh -L 6090:127.0.0.1:6090 user@remote-host`). It answers only requests whose `Host` is `localhost`, a
loopback address, the address it is bound to or a name of `--web_viewer_allowed_hosts`, and refuses requests that other
pages make, a link from another port of `localhost` (noVNC's, Rerun's) included: type or bookmark the address.

With the bridge network of `docker-compose.bridge.yaml`, the published port reaches the container's own interface and
not its loopback: a robot process run in this container then needs `--web_viewer_address=0.0.0.0` (the override
publishes the port on the computer's loopback only, `127.0.0.1:6090`), and its Host checks stay on.
<!-- LINT.ThenChange(//docker-compose.bridge.yaml:web_viewer_port, //.devcontainer/devcontainer.json:web_viewer_port, //humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotAppFlags.cpp:web_viewer_port) -->

## Remote SSH Development

If your dev container is running on a **remote Linux machine** (e.g. over Remote SSH in Antigravity, Cursor, or VS Code):

1. **Option A (Direct LAN IP - Recommended)**:
   - Connect directly from your browser to `http://<host-ip>:6080/vnc.html` (VNC desktop for the operator GUI, and MuJoCo with `VIEWER=glfw`) and `http://<host-ip>:9090/?url=rerun+http://<host-ip>:9876/proxy` (Rerun web viewer) without needing SSH tunnel proxies.
   - The MuJoCo web viewer listens on the host's loopback only (by default): forward its port `6090` as in Option B.
2. **Option B (Forward Ports 6080, 6090, 9090 & 9876)**:
   - In your IDE: Check the **Ports** panel tab and ensure ports `6080`, `6090`, `9090` and `9876` are forwarded (`devcontainer.json` forwards them).
   - Or from your local terminal: `ssh -L 6080:localhost:6080 -L 6090:127.0.0.1:6090 -L 9090:localhost:9090 -L 9876:localhost:9876 user@remote-host`
   - Navigate to `http://localhost:6080/vnc.html` (operator GUI; MuJoCo with `VIEWER=glfw`), `http://localhost:6090` (MuJoCo web viewer, the default) and `http://localhost:9090/?url=rerun+http://localhost:9876/proxy` (Rerun web viewer) on your local machine.

## Go to Definition

C++ navigation is clangd's, run inside the dev container on the `compile_commands.json` that the container's start and
every Makefile build keep current; Python's is a Pyright-based server on `pyrightconfig.json`. From an editor connected
to the host over SSH, set `"clangd.path": "${workspaceFolder}/tools/ide/clangd_in_dev_container.sh"` in the workspace's
`.vscode/settings.json` (`devcontainer.json` sets it for an editor attached to the container), and install
basedpyright where Pylance is not available. See `tools/ide/README.md`.

## Launch Targets

Use the `-vnc` suffixed Make targets to build, start VNC, and launch with the VNC display and Mesa software rendering:

<!-- LINT.IfChange(launch_targets) -->
| Target | Description |
|--------|-------------|
| `make launch-g1-sim-vnc` | G1 centroidal MPC — MuJoCo sim (robot-sim container + MPC in the dev container) |
| `make launch-g1-dummy-sim-vnc` | G1 centroidal MPC — dummy sim |
| `make launch-g1-sandbox-vnc` | G1 URDF in Rerun, a slider per joint |
| `make launch-wb-g1-sim-vnc` | G1 whole-body MPC — MuJoCo sim |
| `make launch-wb-g1-dummy-sim-vnc` | G1 whole-body MPC — dummy sim |
| `make launch-drc-atlas-sim-vnc` | DRC Atlas centroidal MPC — MuJoCo sim |
| `make launch-drc-atlas-dummy-sim-vnc` | DRC Atlas centroidal MPC — dummy sim |
| `make launch-drc-atlas-sandbox-vnc` | DRC Atlas URDF in Rerun, a slider per joint |
| `make launch-r1-sim-vnc` | Unitree R1 centroidal MPC — MuJoCo sim |
| `make launch-r1-dummy-sim-vnc` | Unitree R1 centroidal MPC — dummy sim |
| `make launch-r1-sandbox-vnc` | Unitree R1 URDF in Rerun, a slider per joint |
| `make launch-sa01-sim-vnc` | EngineAI SA01 centroidal MPC — MuJoCo sim |
| `make launch-sa01-dummy-sim-vnc` | EngineAI SA01 centroidal MPC — dummy sim |
| `make launch-sa01-sandbox-vnc` | EngineAI SA01 URDF in Rerun, a slider per joint |
<!-- LINT.ThenChange(//Makefile:launch_targets, //Makefile:robot_configurations) -->

Each `-vnc` target calls `start-vnc` automatically, so you do **not** need to run `make start-vnc` first.

The non-`-vnc` variants (e.g. `make launch-g1-dummy-sim`) draw on the display of the shell they run in: `:99` in the
container, the host's own display when you run them on the host. They don't start the VNC server. `make help` lists
every launch target and its variables (`RERUN_SINK=serve_web` for Rerun's web viewer, `NETEM=`, and
`VIEWER=glfw|web|none` for the MuJoCo viewer: [MuJoCo Web Viewer](#mujoco-web-viewer)).

**The `-sim` targets mirror the hardware.** The robot process runs in a container of its own, from the robot's image
(`robot-sim`: the robot-runtime image plus the MuJoCo viewer's GL) with the robot's compose file and realtime settings,
and the MPC node, the GUI and the Rerun bridge run in this container, all on the host's network over the ZeroMQ bus
(`tools/deploy/README.md`, "Simulation"). Starting that container needs the host's Docker, so run the `-sim` targets
**on the host** (Linux): they then run the Bazel builds and the laptop side in this container through `docker exec`
(`DEV_CONTAINER=` names it; default `devcontainer-app-1`). The robot container's MuJoCo viewer is by default the web
viewer (the plot configuration's `mujoco_viewer`): it needs no display and serves `http://localhost:6090`. With
`VIEWER=glfw` it draws on `:99` of this container's VNC desktop instead, through the X server's abstract socket, which
containers on the host's network share. The dummy-sim and sandbox targets run
entirely in this container and work from either place. On Docker Desktop (macOS) the containers do not share a host
network: the `-sim` targets do not run there, and the dummy sim (`-dummy-sim-vnc`) is the simulation to use.

`DEV_CONTAINER` must be a container of this checkout's image (Pinocchio in `/opt/openrobots`), on the host's network as
`docker-compose.yaml` creates it. `in_dev_container.sh` checks for Pinocchio and `session.sh` for the network, and each
says so.

## Manual Workflow

If you prefer to run commands yourself:

```bash
# 1. Start VNC (once per session)
make start-vnc

# 2. Open http://localhost:6080/vnc.html in your browser

# 3. Source environment and set Mesa software rendering
source setup_env.sh
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# 4. Launch whatever you want, e.g. the processes of a launch file
#    (humanoid_nmpc/docs/distributed_runtime/README.md, "Launching")
```

> **Note:** The `-vnc` make targets automatically source `setup_env.sh` and set all GL environment variables. The manual exports above are only needed for programs you start yourself.

## Resolution

The VNC display resolution **automatically matches your host monitor**. The
container detects the connected display's native resolution via the kernel DRM
interface (`/sys/class/drm/*/modes`), which is available because the container
runs in privileged mode. If no monitor is detected, the fallback is `1920x1080`.

To **override** the auto-detected resolution (optional):

```bash
# One-off override via make
make start-vnc RESOLUTION=2560x1440

# Or set the env var on the host before starting the container
export VNC_RESOLUTION=2560x1440
```

## Stopping VNC

```bash
make stop-vnc
```

## Plots

The plots are the tabs of the Rerun viewer's blueprint (see [Rerun](#rerun) above): they need no display of their own.

## MuJoCo 3D Viewer Controls & Hotkeys

When focused in the MuJoCo simulation viewport, the following keys and mouse inputs are available: in the web viewer, the
default (`http://localhost:6090`), once its view has the focus (click it; keys with Ctrl, Alt or Meta stay the
browser's), and in the window on `:99` (`VIEWER=glfw`). There, `p` opens the Markers panel, which lists the visualizations with their hotkeys and
toggles them (the window prints its cheatsheet to the robot's log instead), and the toolbar's buttons reset the camera,
switch between tracking and the free camera, save a snapshot and change the size, frame rate and quality. The camera is
shared: every open page shows the same view.

| Key / Input | Action | Effect |
|:---:|---|---|
| **`k`** | Toggle **Camera Tracking** | Switches between robot tracking mode (`mjCAMERA_TRACKING` locked on pelvis) and free manual camera (`mjCAMERA_FREE`) |
| **`0`** | Toggle **Floor / Ground** | Shows/hides ground plane (geom group 0) |
| **`1`** | Toggle **Visual Meshes** | Shows/hides high-res surface meshes (group 1) to inspect underlying collision shapes |
| **`2`** | Toggle **Collision Primitives** | Shows/hides collision capsules, boxes, and cylinders (group 2) |
| **`3` - `5`** | Toggle **Auxiliary Groups** | Shows/hides user/sensor geom groups (groups 3–5) |
| **`t`** | Toggle **Model Transparency** | Alternates between 30% alpha (x-ray mode for inspecting joints/actuators) and 100% opaque |
| **`c`** | Toggle **Contact Points** | Renders small colored spheres at active physical collision contact points |
| **`f`** | Toggle **Contact Forces** | Renders 3D vector arrows depicting normal and friction forces at contacts |
| **`m`** | Toggle **Center of Mass (CoM)** | Displays CoM indicator spheres for kinematic bodies / links |
| **`i`** | Toggle **Inertia Ellipsoids** | Renders equivalent inertia ellipsoids depicting principal moments of inertia |
| **`h`** | Toggle **Convex Hulls** | Displays computed convex hulls enclosing the link meshes |
| **`b`** | Toggle **Contact Timeline** | Barcode along the bottom edge: per contact point, the contact state the MPC policy plans (top strip, blue = contact) against the simulator's ground truth (bottom strip, green = touching) over a sliding window; red = touching while the plan says swing (early touch-down, scuff), orange = in the air while the plan says contact (late touch-down, slip). Window and force threshold: `sim_contact_timeline_window`, `sim_contact_force_threshold` in `task.textproto` |
| **`p`** | **Print Cheatsheet** | Prints the hotkey and mouse control guide to the terminal |
| **Left Click + Drag** | **Orbit Camera** | Rotates camera viewpoint around the robot or focal point |
| **Right Click + Drag** | **Pan Camera** | Translates camera position horizontally and vertically |
| **Scroll / Mid Drag** | **Zoom Camera** | Zooms camera toward or away from the target |
| **Shift + Click + Drag** | **Constrained Pan/Orbit** | Constrains mouse orbit/pan motion to the horizontal plane |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Browser shows "connection refused" | Run `make start-vnc` inside the container first |
| "Failed to connect to server" in a noVNC tab that worked before | The VNC services died with the terminal they were started from (a Ctrl-C on a `make launch-*-vnc` target used to kill them along with the simulation; they now run detached). Run `make start-vnc` and press Connect again; `pgrep -a "Xvfb|websockify"` inside the container shows whether they are alive |
| Port 6080 not reachable | Check `forwardPorts` in `devcontainer.json`; or visit `http://127.0.0.1:6080/vnc.html` |
| Black/blank screen in browser | The WM may not have started. Run `make stop-vnc && make start-vnc` |
| MuJoCo viewer: `GLX` / `Unable to create GL context` | Ensure `LIBGL_ALWAYS_INDIRECT=0` is set (the `-vnc` targets do this). See Manual Workflow above |
| The MuJoCo viewer renders but is slow | Expected with software rendering — use lower resolution: `make start-vnc RESOLUTION=1280x720` |
| `http://localhost:6090` does not answer | `VIEWER=` or the plot configuration's `mujoco_viewer` names another viewer (`glfw`, `none`), or the robot process refused to start: its log (`[robot]`) names the reason, such as no EGL or the port taken by another robot process (`make kill-sims`). In a bridge-networked container the process needs `--web_viewer_address=0.0.0.0` ([MuJoCo Web Viewer](#mujoco-web-viewer)) |
| `403` from the MuJoCo web viewer | The browser reached it under a name that is not `localhost` or a loopback address, or through a link on another page (noVNC, Rerun): type `http://localhost:6090` (through a forwarded port), or add the name to `--web_viewer_allowed_hosts` |
| The MuJoCo web viewer is slow | Expected with llvmpipe (about 16 fps): lower the size or the frame rate in the page's settings, or give it a GPU (NVIDIA's EGL) |
| Rerun web viewer at `:9090` shows no data | The bridge must run with `--rerun_sink serve_web` and be reachable on `9876` too: the page fetches the recording from there |
| "Failed to connect" but the page loads | You are on a port whose container has no VNC server. Containers on the host's network share one 6080, so only one of them can serve it: `ss -ltnp \| grep 6080` on the host shows which process holds it, and `docker ps` the containers |
| Growing zombie process count | Each `start-vnc` restarts Xvfb/x11vnc/websockify and orphans the old ones. `init: true` in `docker-compose.yaml` makes tini PID 1 so they are reaped, but only for containers created after that setting was added: if `cat /proc/1/comm` prints `sleep`, the container predates it and must be recreated (`docker compose up -d --force-recreate`). Run `make check-zombies` to see the holder. Zombies cannot be killed, so rebuild the container to clear an existing backlog |

## Notes

- Uses Mesa software rendering (`llvmpipe`) on `DISPLAY=:99` — reliable and does not require host GPU passthrough.
- VNC session data stays inside the container and is not persisted across rebuilds.
