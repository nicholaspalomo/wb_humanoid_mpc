# Visualization on macOS / Remote Hosts

This guide explains how to see RViz and other GUI apps running inside the dev container on your local computer or MacBook when developing locally or via Remote SSH. The container uses a built-in VNC server with a browser-based viewer (noVNC) — no extra software needed on your local machine.

## Quick Start

```bash
# Inside the container terminal:
make launch-drc-atlas-dummy-sim-vnc
```

Then open **http://localhost:6080/vnc.html** in your browser.

All GUI applications will render in the browser window automatically.

## Remote SSH Development

If your dev container is running on a **remote Linux machine** (e.g. over Remote SSH in Antigravity, Cursor, or VS Code):

1. **Forward Port 6080**:
   - In your IDE: Check the **Ports** panel tab and ensure port `6080` is forwarded.
   - Or from your local terminal, set up SSH port forwarding:
     ```bash
     ssh -L 6080:localhost:6080 user@remote-host
     ```

2. **Launch Target**:
   ```bash
   make launch-drc-atlas-dummy-sim-vnc
   ```

3. **View in Local Browser**:
   Navigate to **http://localhost:6080/vnc.html** on your local machine and click **Connect**.

## Launch Targets

Use the `-vnc` suffixed Make targets to automatically build Bazel targets, start VNC, and launch with correctly configured Mesa GL rendering:

| Target | Description |
|--------|-------------|
| `make launch-g1-dummy-sim-vnc` | G1 centroidal MPC — dummy sim |
| `make launch-g1-sim-vnc` | G1 centroidal MPC — MuJoCo sim |
| `make launch-wb-g1-dummy-sim-vnc` | G1 whole-body MPC — dummy sim |
| `make launch-wb-g1-sim-vnc` | G1 whole-body MPC — MuJoCo sim |
| `make launch-drc-atlas-dummy-sim-vnc` | DRC Atlas centroidal MPC — dummy sim |
| `make launch-drc-atlas-sandbox-vnc` | DRC Atlas URDF viewer |

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
