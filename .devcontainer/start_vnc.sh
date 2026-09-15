#!/bin/bash
# ------------------------------------------------------------------
# start_vnc.sh – Start VNC + noVNC sessions inside the container
# so GUI apps (RViz, PlotJuggler, etc.) can be viewed in browser windows.
#
# Primary Display (:99, port 6080): RViz2, MuJoCo, Joystick GUI
# Secondary Display (:100, port 6082): PlotJuggler (dedicated window)
#
# Usage:
#   ./start_vnc.sh              # start both displays (auto-detect resolution)
#   ./start_vnc.sh 2560x1440   # custom resolution override
#   ./start_vnc.sh plotjuggler  # start only PlotJuggler display (:100)
#   ./start_vnc.sh stop         # tear down all VNC services
# ------------------------------------------------------------------
set -euo pipefail

MODE="all"
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
elif [ "${1:-}" = "plotjuggler" ]; then
    MODE="plotjuggler"
    RESOLUTION="${2:-${DEFAULT_RESOLUTION}}"
elif [ "${1:-}" = "main" ]; then
    MODE="main"
    RESOLUTION="${2:-${DEFAULT_RESOLUTION}}"
elif [ -n "${1:-}" ]; then
    RESOLUTION="$1"
fi

# LINT.IfChange(vnc_ports)
# Primary display config (RViz + MuJoCo)
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_DEPTH="${VNC_DEPTH:-24}"
VNC_DISPLAY="${VNC_DISPLAY:-:99}"

# Secondary display config (PlotJuggler)
PJ_VNC_PORT="${PJ_VNC_PORT:-5903}"
PJ_NOVNC_PORT="${PJ_NOVNC_PORT:-6082}"
PJ_DISPLAY="${PJ_DISPLAY:-:100}"
# LINT.ThenChange(//docker-compose.yaml:vnc_ports, //.devcontainer/devcontainer.json:vnc_ports, //.devcontainer/README.md:vnc_ports, //Makefile:vnc_ports)

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
    pkill -9 -f "Xvfb.*:99" 2>/dev/null || true
    pkill -9 -f "Xvfb.*:100" 2>/dev/null || true
    pkill -9 -f "x11vnc.*${VNC_PORT}" 2>/dev/null || true
    pkill -9 -f "x11vnc.*${PJ_VNC_PORT}" 2>/dev/null || true
    pkill -9 -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
    pkill -9 -f "websockify.*${PJ_NOVNC_PORT}" 2>/dev/null || true
    pkill -9 -x openbox 2>/dev/null || true
    fuser -k -9 ${NOVNC_PORT}/tcp 2>/dev/null || true
    fuser -k -9 ${PJ_NOVNC_PORT}/tcp 2>/dev/null || true
    fuser -k -9 ${VNC_PORT}/tcp 2>/dev/null || true
    fuser -k -9 ${PJ_VNC_PORT}/tcp 2>/dev/null || true
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
# was asked for silently mis-places every window whenever the two disagree -- a one-off `start_vnc.sh main 1280x720`
# would rewrite the rules of a 1920x1080 desktop that is already up. The running display's real size wins.
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

# Tell every live openbox to re-read the rules that were just written.
reconfigure_wm() {
    local display
    for display in "${VNC_DISPLAY}" "${PJ_DISPLAY}"; do
        if wm_running "${display}"; then
            DISPLAY=${display} openbox --reconfigure 2>/dev/null || true
        fi
    done
}

is_main_running() {
    pgrep -f "Xvfb.*${VNC_DISPLAY}" >/dev/null 2>&1 && \
    pgrep -f "x11vnc.*${VNC_PORT}" >/dev/null 2>&1 && \
    pgrep -f "websockify.*${NOVNC_PORT}" >/dev/null 2>&1 && \
    wm_running "${VNC_DISPLAY}"
}

is_pj_running() {
    pgrep -f "Xvfb.*${PJ_DISPLAY}" >/dev/null 2>&1 && \
    pgrep -f "x11vnc.*${PJ_VNC_PORT}" >/dev/null 2>&1 && \
    pgrep -f "websockify.*${PJ_NOVNC_PORT}" >/dev/null 2>&1 && \
    wm_running "${PJ_DISPLAY}"
}

# Brings up one display, starting only the services that are actually missing. Starting them as a group meant that one
# dead service restarted all of them, which is where the duplicate websockify processes bound to the same port came
# from; it also meant the window manager could only ever be started at the same time as its X server.
start_display() {
    local display="$1" vnc_port="$2" novnc_port="$3" log="$4" root_color="$5" label="$6"

    if ! pgrep -f "Xvfb.*${display}" >/dev/null 2>&1; then
        echo "Starting Xvfb on display ${display} ${label}..."
        detached Xvfb ${display} -screen 0 "${RESOLUTION}x${VNC_DEPTH}" +iglx
        sleep 1
    fi

    if ! pgrep -f "x11vnc.*${vnc_port}" >/dev/null 2>&1; then
        echo "Starting x11vnc on port ${vnc_port} ${label}..."
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
            echo "Starting openbox on display ${display} ${label}..."
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
        echo "Starting noVNC websockify on port ${novnc_port} ${label}..."
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
# Three windows share this display: RViz, the MuJoCo viewer, and the Tkinter "Robot Base Controller & Tuning" GUI.
# The GUI used to have no rule at all, so it landed where it asks (960x700 at +30+50) while RViz was placed across the
# whole left pane, covering all but a ~10 px strip of it. The GUI's own `-topmost` holds for only 1.5 s, which RViz
# beats every time under software GL, so it ended up buried with nothing left to click. The left pane is now split
# between RViz and the GUI so that no window ever covers another.
mkdir -p "${HOME}/.config/openbox"
if [ -f "/etc/xdg/openbox/rc.xml" ]; then
    cp /etc/xdg/openbox/rc.xml "${HOME}/.config/openbox/rc.xml"
    RESOLUTION="${LAYOUT_RESOLUTION}" python3 "$(dirname "${BASH_SOURCE[0]}")/openbox_layout.py" \
        || echo "WARNING: could not write Openbox layout rules; windows will be placed wherever they ask."
fi
reconfigure_wm

# If requested services are already running healthy, keep them active
if [ "$MODE" = "all" ] && is_main_running && is_pj_running; then
    echo "✅ All VNC servers are already running (${VNC_DISPLAY} on ${NOVNC_PORT}, ${PJ_DISPLAY} on ${PJ_NOVNC_PORT})"
    echo "   🎮 Simulation & RViz : http://localhost:${NOVNC_PORT}/vnc.html"
    echo "   📊 PlotJuggler       : http://localhost:${PJ_NOVNC_PORT}/vnc.html"
    exit 0
elif [ "$MODE" = "main" ] && is_main_running; then
    echo "✅ Main VNC server is already running on ${VNC_DISPLAY}: http://localhost:${NOVNC_PORT}/vnc.html"
    exit 0
elif [ "$MODE" = "plotjuggler" ] && is_pj_running; then
    echo "✅ PlotJuggler VNC server is already running on ${PJ_DISPLAY}: http://localhost:${PJ_NOVNC_PORT}/vnc.html"
    exit 0
fi

# The services are started in their own sessions (setsid), detached from this terminal. Started as plain background
# jobs they belonged to the terminal's foreground process group, so the Ctrl-C that stops a `make launch-*-vnc` target,
# or closing that terminal, killed Xvfb and websockify along with the simulation (x11vnc then exited with its X server).
# The noVNC tab that was still open then reported "Failed to connect to server" until the next launch restarted them.
detached() {
    setsid -f "$@" >/dev/null 2>&1 </dev/null
}

# Clean up stale sockets/locks, but only for a display whose X server is actually gone. These commands kill whatever
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

if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    clear_stale_locks "${VNC_DISPLAY}" "${VNC_PORT}" "${NOVNC_PORT}"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "plotjuggler" ]; then
    clear_stale_locks "${PJ_DISPLAY}" "${PJ_VNC_PORT}" "${PJ_NOVNC_PORT}"
fi
sleep 0.5

echo "============================================="
echo "  Starting VNC + noVNC visualization server(s)"
echo "  Resolution : ${RESOLUTION}x${VNC_DEPTH}"
if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    echo "  Main VNC   : port ${VNC_PORT} -> noVNC ${NOVNC_PORT} (display ${VNC_DISPLAY})"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "plotjuggler" ]; then
    echo "  PlotJuggler: port ${PJ_VNC_PORT} -> noVNC ${PJ_NOVNC_PORT} (display ${PJ_DISPLAY})"
fi
echo "============================================="

# --- Configure Mesa for software rendering (required for RViz2/OGRE in VNC) ---
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe


# --- Start Main Display (:99) ---
if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    start_display "${VNC_DISPLAY}" "${VNC_PORT}" "${NOVNC_PORT}" /tmp/x11vnc.log "#1e222d" ""
fi

# --- Start PlotJuggler Dedicated Display (:100) ---
if [ "$MODE" = "all" ] || [ "$MODE" = "plotjuggler" ]; then
    start_display "${PJ_DISPLAY}" "${PJ_VNC_PORT}" "${PJ_NOVNC_PORT}" /tmp/x11vnc_pj.log "#1a1e29" "(PlotJuggler) "
fi

# Discover host/LAN IPs for easy browser access from remote laptops
LAN_IPS=$(ip -4 addr show 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -vE '^(127\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' || true)

echo ""
echo "============================================="
echo "  VNC visualization servers are ready!"
echo ""
echo "  Open in your browser:"
if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    echo "    🎮 Simulation & RViz : http://localhost:${NOVNC_PORT}/vnc.html"
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "plotjuggler" ]; then
    echo "    📊 PlotJuggler       : http://localhost:${PJ_NOVNC_PORT}/vnc.html"
fi

if [ -n "${LAN_IPS}" ]; then
    echo ""
    echo "  LAN access:"
    for ip in ${LAN_IPS}; do
        if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
            echo "    🎮 Simulation & RViz : http://${ip}:${NOVNC_PORT}/vnc.html"
        fi
        if [ "$MODE" = "all" ] || [ "$MODE" = "plotjuggler" ]; then
            echo "    📊 PlotJuggler       : http://${ip}:${PJ_NOVNC_PORT}/vnc.html"
        fi
    done
fi

echo ""
echo "  To stop:  $0 stop"
echo "============================================="

# Export DISPLAY so subsequent commands in this shell use VNC
export DISPLAY=${VNC_DISPLAY}
export PLOTJUGGLER_DISPLAY=${PJ_DISPLAY}

