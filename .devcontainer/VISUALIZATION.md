# Visualization on macOS / Remote Hosts

This guide explains how to see RViz and other GUI apps running inside the dev container on your local computer or MacBook when developing locally or via Remote SSH. The container uses a built-in VNC server with a browser-based viewer (noVNC) — no extra software needed on your local machine.

## Quick Start

```bash
# Inside the container terminal:
make launch-drc-atlas-dummy-sim-vnc
```

Then open the URLs printed in the terminal banner:
<!-- LINT.IfChange(vnc_ports) -->
- **Simulation & RViz (`:99`)**: `http://localhost:6080/vnc.html` (or `http://<host-lan-ip>:6080/vnc.html`)
- **PlotJuggler (`:100`)**: `http://localhost:6082/vnc.html` (or `http://<host-lan-ip>:6082/vnc.html`)
<!-- LINT.ThenChange(//.devcontainer/start_vnc.sh:vnc_ports, //docker-compose.yaml:vnc_ports, //.devcontainer/devcontainer.json:vnc_ports, //Makefile:vnc_ports) -->

All GUI applications (RViz, MuJoCo sim, controllers) render on `:99`, while PlotJuggler opens in its own dedicated, fullscreen window on `:100`.

## Remote SSH Development

If your dev container is running on a **remote Linux machine** (e.g. over Remote SSH in Antigravity, Cursor, or VS Code):

1. **Option A (Direct LAN IP - Recommended)**:
   - Connect directly from your browser to `http://<host-ip>:6080/vnc.html` (RViz) and `http://<host-ip>:6082/vnc.html` (PlotJuggler) without needing SSH tunnel proxies.
2. **Option B (Forward Ports 6080 & 6082)**:
   - In your IDE: Check the **Ports** panel tab and ensure ports `6080` and `6082` are forwarded.
   - Or from your local terminal: `ssh -L 6080:localhost:6080 -L 6082:localhost:6082 user@remote-host`
   - Navigate to `http://localhost:6080/vnc.html` and `http://localhost:6082/vnc.html` on your local machine.

## Launch Targets

Use the `-vnc` suffixed Make targets to automatically build Bazel targets, start VNC, and launch with correctly configured Mesa GL rendering:

<!-- LINT.IfChange(launch_targets) -->
| Target | Description |
|--------|-------------|
| `make launch-g1-dummy-sim-vnc` | G1 centroidal MPC — dummy sim |
| `make launch-g1-sim-vnc` | G1 centroidal MPC — MuJoCo sim |
| `make launch-wb-g1-dummy-sim-vnc` | G1 whole-body MPC — dummy sim |
| `make launch-wb-g1-sim-vnc` | G1 whole-body MPC — MuJoCo sim |
| `make launch-drc-atlas-dummy-sim-vnc` | DRC Atlas centroidal MPC — dummy sim |
| `make launch-drc-atlas-sim-vnc` | DRC Atlas centroidal MPC — MuJoCo sim |
| `make launch-drc-atlas-sandbox-vnc` | DRC Atlas URDF viewer |
| `make launch-r1-dummy-sim-vnc` | Unitree R1 centroidal MPC — dummy sim |
| `make launch-r1-sim-vnc` | Unitree R1 centroidal MPC — MuJoCo sim |
| `make launch-r1-sandbox-vnc` | Unitree R1 URDF viewer |
<!-- LINT.ThenChange(//Makefile:launch_targets, //setup_env.sh:registered_packages) -->

Each `-vnc` target calls `start-vnc` automatically, so you do **not** need to run `make start-vnc` first.

The non-`-vnc` variants (e.g. `make launch-g1-dummy-sim`) also use the VNC display by default (`DISPLAY=:99`) but don't auto-start the VNC server — run `make start-vnc` once before using them.

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

# 4. Launch whatever you want
ros2 launch g1_centroidal_mpc dummy_sim.launch.py
```

> **Note:** The `-vnc` make targets automatically source `setup_env.sh` and set all GL environment variables. The manual exports above are only needed if you run `ros2 launch` or `rviz2` directly.

## Custom Resolution

```bash
make start-vnc RESOLUTION=2560x1440
```

## Stopping VNC

```bash
make stop-vnc
```

## Dedicated PlotJuggler Window

PlotJuggler is routed to dedicated display `:100` (`http://localhost:6082/vnc.html`) so it doesn't crowd RViz and MuJoCo on `:99`. You can also launch it independently anytime:

<!-- LINT.IfChange(plotjuggler_vnc) -->
```bash
make plotjuggler-vnc
```
<!-- LINT.ThenChange(//Makefile:plotjuggler_vnc) -->

## MuJoCo 3D Viewer Controls & Hotkeys

When focused in the MuJoCo simulation viewport on `:99`, the following keys and mouse inputs are available:

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
| **`p`** | **Print Cheatsheet** | Prints the hotkey and mouse control guide to the terminal |
| **Left Click + Drag** | **Orbit Camera** | Rotates camera viewpoint around the robot or focal point |
| **Right Click + Drag** | **Pan Camera** | Translates camera position horizontally and vertically |
| **Scroll / Mid Drag** | **Zoom Camera** | Zooms camera toward or away from the target |
| **Shift + Click + Drag** | **Constrained Pan/Orbit** | Constrains mouse orbit/pan motion to the horizontal plane |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Browser shows "connection refused" | Run `make start-vnc` inside the container first |
| Port 6080 not reachable | Check `forwardPorts` in `devcontainer.json`; or visit `http://127.0.0.1:6080/vnc.html` |
| Black/blank screen in browser | The WM may not have started. Run `make stop-vnc && make start-vnc` |
| RViz: `Unable to create glx context` | Ensure `LIBGL_ALWAYS_INDIRECT=0` is set (the `-vnc` targets do this). See Manual Workflow above |
| RViz renders but is slow | Expected with software rendering — use lower resolution: `make start-vnc RESOLUTION=1280x720` |

## Notes

- Uses Mesa software rendering (`llvmpipe`) on `DISPLAY=:99` — reliable and does not require host GPU passthrough.
- VNC session data stays inside the container and is not persisted across rebuilds.
