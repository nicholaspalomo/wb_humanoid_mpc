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
VNC_DISPLAY=":99"

# Secondary display config (PlotJuggler)
PJ_VNC_PORT="${PJ_VNC_PORT:-5903}"
PJ_NOVNC_PORT="${PJ_NOVNC_PORT:-6082}"
PJ_DISPLAY="${PJ_DISPLAY:-:100}"
# LINT.ThenChange(//docker-compose.yaml:vnc_ports, //.devcontainer/devcontainer.json:vnc_ports, //.devcontainer/VISUALIZATION.md:vnc_ports, //Makefile:vnc_ports)

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
    pkill -9 -f "openbox" 2>/dev/null || true
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

is_main_running() {
    pgrep -f "Xvfb.*${VNC_DISPLAY}" >/dev/null 2>&1 && \
    pgrep -f "x11vnc.*${VNC_PORT}" >/dev/null 2>&1 && \
    pgrep -f "websockify.*${NOVNC_PORT}" >/dev/null 2>&1
}

is_pj_running() {
    pgrep -f "Xvfb.*${PJ_DISPLAY}" >/dev/null 2>&1 && \
    pgrep -f "x11vnc.*${PJ_VNC_PORT}" >/dev/null 2>&1 && \
    pgrep -f "websockify.*${PJ_NOVNC_PORT}" >/dev/null 2>&1
}

# If requested services are already running healthy, keep them active
if [ "$MODE" = "all" ] && is_main_running && is_pj_running; then
    echo "✅ All VNC servers are already running (:99 on ${NOVNC_PORT}, :100 on ${PJ_NOVNC_PORT})"
    exit 0
elif [ "$MODE" = "main" ] && is_main_running; then
    echo "✅ Main VNC server is already running on ${VNC_DISPLAY} (noVNC port ${NOVNC_PORT})"
    exit 0
elif [ "$MODE" = "plotjuggler" ] && is_pj_running; then
    echo "✅ PlotJuggler VNC server is already running on ${PJ_DISPLAY} (noVNC port ${PJ_NOVNC_PORT})"
    exit 0
fi

# Clean up any stale sockets/locks for displays we intend to start
if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    fuser -k -9 ${NOVNC_PORT}/tcp 2>/dev/null || true
    fuser -k -9 ${VNC_PORT}/tcp 2>/dev/null || true
    sudo -n rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true
    rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true
fi
if [ "$MODE" = "all" ] || [ "$MODE" = "plotjuggler" ]; then
    fuser -k -9 ${PJ_NOVNC_PORT}/tcp 2>/dev/null || true
    fuser -k -9 ${PJ_VNC_PORT}/tcp 2>/dev/null || true
    sudo -n rm -f /tmp/.X100-lock /tmp/.X11-unix/X100 2>/dev/null || true
    rm -f /tmp/.X100-lock /tmp/.X11-unix/X100 2>/dev/null || true
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

# Setup Openbox window layout rules (dynamically sized from $RESOLUTION)
mkdir -p "${HOME}/.config/openbox"
if [ -f "/etc/xdg/openbox/rc.xml" ]; then
    cp /etc/xdg/openbox/rc.xml "${HOME}/.config/openbox/rc.xml"
    python3 -c "
import os, sys
resolution = '${RESOLUTION}'
try:
    w, h = [int(x) for x in resolution.split('x')]
except ValueError:
    w, h = 1920, 1080

# Split the main display into two side-by-side panes (RViz left, MuJoCo right)
half_w = w // 2
pane_h = h - 40  # leave room for window bar

rc_path = os.path.expanduser('~/.config/openbox/rc.xml')
with open(rc_path, 'r') as f:
    content = f.read()
rules = f'''
  <application class=\"*rviz*\" title=\"*rviz*\" name=\"*rviz*\">
    <position force=\"yes\"><x>0</x><y>0</y></position>
    <size><width>{half_w - 10}</width><height>{pane_h}</height></size>
    <maximized>no</maximized>
  </application>
  <application title=\"*Mujoco*\" name=\"*\" class=\"*\">
    <position force=\"yes\"><x>{half_w}</x><y>0</y></position>
    <size><width>{half_w - 10}</width><height>{pane_h}</height></size>
    <maximized>no</maximized>
    <focus>yes</focus>
  </application>
  <application class=\"*plotjuggler*\" title=\"*plotjuggler*\" name=\"*plotjuggler*\">
    <maximized>yes</maximized>
  </application>
  <application class=\"*PlotJuggler*\" title=\"*PlotJuggler*\" name=\"*PlotJuggler*\">
    <maximized>yes</maximized>
  </application>
'''
if '</applications>' in content:
    content = content.replace('</applications>', rules + '\n</applications>')
    with open(rc_path, 'w') as f:
        f.write(content)
" 2>/dev/null || true
fi

# --- Start Main Display (:99) ---
if [ "$MODE" = "all" ] || [ "$MODE" = "main" ]; then
    if ! is_main_running; then
        echo "Starting Xvfb on display ${VNC_DISPLAY} ..."
        Xvfb ${VNC_DISPLAY} -screen 0 "${RESOLUTION}x${VNC_DEPTH}" +iglx >/dev/null 2>&1 &
        sleep 1

        echo "Starting x11vnc on port ${VNC_PORT} ..."
        x11vnc -display ${VNC_DISPLAY} \
            -rfbport "${VNC_PORT}" \
            -nopw \
            -shared \
            -forever \
            -noxdamage \
            -bg \
            -o /tmp/x11vnc.log \
            >/dev/null 2>&1 || true
        sleep 0.5

        if command -v openbox >/dev/null 2>&1; then
            DISPLAY=${VNC_DISPLAY} openbox &
            sleep 0.5
        fi
        if command -v xsetroot >/dev/null 2>&1; then
            DISPLAY=${VNC_DISPLAY} xsetroot -solid "#1e222d" 2>/dev/null || true
        fi

        echo "Starting noVNC websockify on port ${NOVNC_PORT} ..."
        websockify --web="${NOVNC_DIR}" ${NOVNC_PORT} localhost:${VNC_PORT} >/dev/null 2>&1 &
        sleep 0.5
    fi
fi

# --- Start PlotJuggler Dedicated Display (:100) ---
if [ "$MODE" = "all" ] || [ "$MODE" = "plotjuggler" ]; then
    if ! is_pj_running; then
        echo "Starting Xvfb on display ${PJ_DISPLAY} (PlotJuggler) ..."
        Xvfb ${PJ_DISPLAY} -screen 0 "${RESOLUTION}x${VNC_DEPTH}" +iglx >/dev/null 2>&1 &
        sleep 1

        echo "Starting x11vnc on port ${PJ_VNC_PORT} (PlotJuggler) ..."
        x11vnc -display ${PJ_DISPLAY} \
            -rfbport "${PJ_VNC_PORT}" \
            -nopw \
            -shared \
            -forever \
            -noxdamage \
            -bg \
            -o /tmp/x11vnc_pj.log \
            >/dev/null 2>&1 || true
        sleep 0.5

        if command -v openbox >/dev/null 2>&1; then
            DISPLAY=${PJ_DISPLAY} openbox &
            sleep 0.5
        fi
        if command -v xsetroot >/dev/null 2>&1; then
            DISPLAY=${PJ_DISPLAY} xsetroot -solid "#1a1e29" 2>/dev/null || true
        fi

        echo "Starting noVNC websockify on port ${PJ_NOVNC_PORT} (PlotJuggler) ..."
        websockify --web="${NOVNC_DIR}" ${PJ_NOVNC_PORT} localhost:${PJ_VNC_PORT} >/dev/null 2>&1 &
        sleep 0.5
    fi
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

