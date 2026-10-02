#!/bin/bash
# ------------------------------------------------------------------
# start_vnc.sh – Start the VNC + noVNC session inside the container
# so GUI apps can be viewed in a browser window.
#
# Display :99 (noVNC on port 6080): the MuJoCo viewer, the Tk operator GUI and, when it runs inside the container, the
# native Rerun viewer. The plots and the 3D scene are in Rerun, which normally runs on the host or as a web viewer
# (.devcontainer/README.md).
#
# Usage:
#   ./start_vnc.sh              # start the display (auto-detect resolution)
#   ./start_vnc.sh 2560x1440    # custom resolution override
#   ./start_vnc.sh stop         # tear down all VNC services
# ------------------------------------------------------------------
set -euo pipefail

MODE="start"
# Resolution priority: CLI arg > VNC_RESOLUTION env var > host monitor auto-detect > 1920x1080
# LINT.IfChange(vnc_resolution)
if [ -n "${VNC_RESOLUTION:-}" ]; then
    DEFAULT_RESOLUTION="${VNC_RESOLUTION}"
else
    # Auto-detect from host monitor via DRM (available in privileged containers)
    _detected=$(cat /sys/class/drm/*/modes 2>/dev/null | head -1)
    DEFAULT_RESOLUTION="${_detected:-1920x1080}"
fi
# LINT.ThenChange(//docker-compose.yaml:vnc_resolution, //Makefile:vnc_resolution)
RESOLUTION="${DEFAULT_RESOLUTION}"

if [ "${1:-}" = "stop" ]; then
    MODE="stop"
elif [ -n "${1:-}" ]; then
    # Anything else must be a resolution: there is a single display, and a stale call naming a display mode is refused
    # instead of starting an X server of that geometry.
    if [[ ! "$1" =~ ^[0-9]+x[0-9]+$ ]]; then
        echo "usage: $0 [WIDTHxHEIGHT | stop]  (got '$1'; there is a single display, ${VNC_DISPLAY:-:99})" >&2
        exit 2
    fi
    RESOLUTION="$1"
fi

# LINT.IfChange(vnc_ports)
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_DEPTH="${VNC_DEPTH:-24}"
VNC_DISPLAY="${VNC_DISPLAY:-:99}"
# LINT.ThenChange(//docker-compose.bridge.yaml:vnc_ports, //.devcontainer/devcontainer.json:vnc_ports, //.devcontainer/README.md:vnc_ports, //Makefile:vnc_ports)

# Derive noVNC web root – works on Ubuntu 22.04+ (may be /usr/share/novnc)
NOVNC_DIR="/usr/share/novnc"
if [ ! -d "$NOVNC_DIR" ]; then
    NOVNC_DIR="/usr/share/noVNC"
fi

# Ensure index.html points to vnc.html in noVNC directory
if [ -d "$NOVNC_DIR" ]; then
    sudo -n ln -sf "${NOVNC_DIR}/vnc.html" "${NOVNC_DIR}/index.html" 2>/dev/null || ln -sf "${NOVNC_DIR}/vnc.html" "${NOVNC_DIR}/index.html" 2>/dev/null || true
fi

cleanup() {
    echo "Stopping VNC services..."
    pkill -9 -f "Xvfb.*${VNC_DISPLAY}" 2>/dev/null || true
    pkill -9 -f "x11vnc.*${VNC_PORT}" 2>/dev/null || true
    pkill -9 -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
    pkill -9 -x openbox 2>/dev/null || true
    fuser -k -9 ${NOVNC_PORT}/tcp 2>/dev/null || true
    fuser -k -9 ${VNC_PORT}/tcp 2>/dev/null || true
    sudo -n rm -f /tmp/.X*-lock /tmp/.X11-unix/X* 2>/dev/null || true
    rm -f /tmp/.X*-lock /tmp/.X11-unix/X* 2>/dev/null || true
    echo "VNC services stopped."
}

if [ "$MODE" = "stop" ]; then
    cleanup
    exit 0
fi

# True when a LIVE openbox is managing the given display.
#
# Without a window manager the desktop still renders and noVNC still connects, but windows get no decorations and
# clicking one does not raise it: in X only the client itself or a window manager can restack. That failure used to be
# invisible here, because the health checks below only looked at Xvfb, x11vnc and websockify. Openbox dying while those
# three survived left the script reporting "already running" and skipping the one block that ever starts it, so no
# amount of re-launching brought the window manager back.
#
# pgrep alone is not enough: a zombie keeps its name in the process table (this container accumulates them, because
# `detached` reparents every service to an init that does not reap), so `pgrep -x openbox` happily reports dozens of
# corpses. A zombie has no readable environ, which is what separates it from a live process here, and the environ is
# also where the display it manages is recorded.
wm_running() {
    local pid environ
    for pid in $(pgrep -x openbox 2>/dev/null || true); do
        [ -r "/proc/$pid/environ" ] || continue
        environ=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null) || continue
        case $'\n'"$environ"$'\n' in
            *$'\n'"DISPLAY=$1"$'\n'*) return 0 ;;
        esac
    done
    return 1
}

# The geometry of an already-running X server, as WIDTHxHEIGHT.
#
# The layout rules below are a single shared file, so sizing them from whatever resolution this particular invocation
# was asked for silently mis-places every window whenever the two disagree -- a one-off `start_vnc.sh 1280x720` would
# rewrite the rules of a 1920x1080 desktop that is already up. The running display's real size wins.
running_resolution() {
    local line
    line=$(pgrep -af "Xvfb $1 " 2>/dev/null) || return 1
    [ -n "$line" ] || return 1
    if [[ "$line" =~ ([0-9]+x[0-9]+)x[0-9]+ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

# Tell the live openbox to re-read the rules that were just written.
reconfigure_wm() {
    if wm_running "${VNC_DISPLAY}"; then
        DISPLAY=${VNC_DISPLAY} openbox --reconfigure 2>/dev/null || true
    fi
}

is_running() {
    pgrep -f "Xvfb.*${VNC_DISPLAY}" >/dev/null 2>&1 && \
    pgrep -f "x11vnc.*${VNC_PORT}" >/dev/null 2>&1 && \
    pgrep -f "websockify.*${NOVNC_PORT}" >/dev/null 2>&1 && \
    wm_running "${VNC_DISPLAY}"
}

# Brings up the display, starting only the services that are actually missing. Starting them as a group meant that one
# dead service restarted all of them, which is where the duplicate websockify processes bound to the same port came
# from; it also meant the window manager could only ever be started at the same time as its X server.
start_display() {
    local display="$1" vnc_port="$2" novnc_port="$3" log="$4" root_color="$5"

    if ! pgrep -f "Xvfb.*${display}" >/dev/null 2>&1; then
        echo "Starting Xvfb on display ${display}..."
        detached Xvfb ${display} -screen 0 "${RESOLUTION}x${VNC_DEPTH}" +iglx
        sleep 1
    fi

    if ! pgrep -f "x11vnc.*${vnc_port}" >/dev/null 2>&1; then
        echo "Starting x11vnc on port ${vnc_port}..."
        detached x11vnc -display ${display} \
            -rfbport "${vnc_port}" \
            -nopw \
            -shared \
            -forever \
            -noxdamage \
            -o "${log}"
        sleep 0.5
    fi

    if ! wm_running "${display}"; then
        if command -v openbox >/dev/null 2>&1; then
            echo "Starting openbox on display ${display}..."
            DISPLAY=${display} detached openbox
            sleep 0.5
        else
            echo "WARNING: openbox is not installed. Windows on ${display} will have no title bars and clicking one"
            echo "         will not bring it to the front. Install it with: sudo apt-get install -y openbox"
        fi
    fi

    if command -v xsetroot >/dev/null 2>&1; then
        DISPLAY=${display} xsetroot -solid "${root_color}" 2>/dev/null || true
    fi

    if ! pgrep -f "websockify.*${novnc_port}" >/dev/null 2>&1; then
        echo "Starting noVNC websockify on port ${novnc_port}..."
        detached websockify --web="${NOVNC_DIR}" ${novnc_port} localhost:${vnc_port}
        sleep 0.5
    fi
}

LAYOUT_RESOLUTION="${RESOLUTION}"
if _live_resolution=$(running_resolution "${VNC_DISPLAY}"); then
    LAYOUT_RESOLUTION="${_live_resolution}"
fi

# Setup Openbox window layout rules (sized from the display that is actually running, else from $RESOLUTION).
# Written before the early exit below, so that a healthy session still picks up changed rules instead of keeping
# whatever was generated the first time it came up.
#
# Three windows share this display: the Rerun viewer, the MuJoCo viewer, and the Tkinter "Robot Base Controller &
# Tuning" GUI. openbox_layout.py tiles them so that no window ever covers another (it says why that matters).
mkdir -p "${HOME}/.config/openbox"
if [ -f "/etc/xdg/openbox/rc.xml" ]; then
    cp /etc/xdg/openbox/rc.xml "${HOME}/.config/openbox/rc.xml"
    RESOLUTION="${LAYOUT_RESOLUTION}" python3 "$(dirname "${BASH_SOURCE[0]}")/openbox_layout.py" \
        || echo "WARNING: could not write Openbox layout rules; windows will be placed wherever they ask."
fi
reconfigure_wm

# If the services are already running healthy, keep them active
if is_running; then
    echo "✅ VNC server is already running on ${VNC_DISPLAY}: http://localhost:${NOVNC_PORT}/vnc.html"
    exit 0
fi

# The services are started in their own sessions (setsid), detached from this terminal. Started as plain background
# jobs they belonged to the terminal's foreground process group, so the Ctrl-C that stops a `make launch-*-vnc` target,
# or closing that terminal, killed Xvfb and websockify along with the simulation (x11vnc then exited with its X server).
# The noVNC tab that was still open then reported "Failed to connect to server" until the next launch restarted them.
detached() {
    setsid -f "$@" >/dev/null 2>&1 </dev/null
}

# Clean up stale sockets/locks, but only when the display's X server is actually gone. These commands kill whatever
# holds the VNC and noVNC ports and delete the X socket, so running them unconditionally would tear down a healthy
# session that is only missing its window manager -- and leave an orphaned Xvfb that nothing can connect to, because
# its socket was removed while it kept running.
clear_stale_locks() {
    local display="$1" vnc_port="$2" novnc_port="$3"
    local num="${display#:}"
    if pgrep -f "Xvfb.*${display}" >/dev/null 2>&1; then
        return 0
    fi
    fuser -k -9 "${novnc_port}"/tcp 2>/dev/null || true
    fuser -k -9 "${vnc_port}"/tcp 2>/dev/null || true
    sudo -n rm -f "/tmp/.X${num}-lock" "/tmp/.X11-unix/X${num}" 2>/dev/null || true
    rm -f "/tmp/.X${num}-lock" "/tmp/.X11-unix/X${num}" 2>/dev/null || true
}

clear_stale_locks "${VNC_DISPLAY}" "${VNC_PORT}" "${NOVNC_PORT}"
sleep 0.5

echo "============================================="
echo "  Starting VNC + noVNC visualization server"
echo "  Resolution : ${RESOLUTION}x${VNC_DEPTH}"
echo "  VNC        : port ${VNC_PORT} -> noVNC ${NOVNC_PORT} (display ${VNC_DISPLAY})"
echo "============================================="

# --- Configure Mesa for software rendering (the MuJoCo viewer's GLFW + GLEW context in Xvfb) ---
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

start_display "${VNC_DISPLAY}" "${VNC_PORT}" "${NOVNC_PORT}" /tmp/x11vnc.log "#1e222d"

# Discover host/LAN IPs for easy browser access from remote laptops
LAN_IPS=$(ip -4 addr show 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -vE '^(127\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' || true)

echo ""
echo "============================================="
echo "  VNC visualization server is ready!"
echo ""
echo "  Open in your browser:"
echo "    🎮 MuJoCo viewer & operator GUI : http://localhost:${NOVNC_PORT}/vnc.html"

if [ -n "${LAN_IPS}" ]; then
    echo ""
    echo "  LAN access:"
    for ip in ${LAN_IPS}; do
        echo "    🎮 MuJoCo viewer & operator GUI : http://${ip}:${NOVNC_PORT}/vnc.html"
    done
fi

echo ""
echo "  To stop:  $0 stop"
echo "============================================="

# Export DISPLAY so subsequent commands in this shell use VNC
export DISPLAY=${VNC_DISPLAY}
