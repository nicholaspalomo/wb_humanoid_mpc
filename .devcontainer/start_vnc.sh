#!/bin/bash
# ------------------------------------------------------------------
# start_vnc.sh – Start a VNC + noVNC session inside the container
# so GUI apps (RViz, etc.) can be viewed in a browser on the host.
#
# Uses Xvfb as the X server (provides proper Mesa GLX for RViz2/OGRE)
# and x11vnc to share the framebuffer over VNC.
#
# Usage:
#   ./start_vnc.sh              # default resolution 1920x1080, port 6080
#   ./start_vnc.sh 2560x1440   # custom resolution
#   ./start_vnc.sh stop         # tear everything down
#
# Once running, open http://localhost:6080/vnc.html in your browser.
# ------------------------------------------------------------------
set -euo pipefail

RESOLUTION="${1:-1920x1080}"
VNC_PORT="${VNC_PORT:-5901}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VNC_DEPTH="${VNC_DEPTH:-24}"
VNC_DISPLAY=":1"

# Derive noVNC web root – works on Ubuntu 22.04+ (may be /usr/share/novnc)
NOVNC_DIR="/usr/share/novnc"
if [ ! -d "$NOVNC_DIR" ]; then
    NOVNC_DIR="/usr/share/noVNC"
fi

cleanup() {
    echo "Stopping VNC services..."
    pkill -f "Xvfb ${VNC_DISPLAY}" 2>/dev/null || true
    pkill -f "x11vnc.*display ${VNC_DISPLAY}" 2>/dev/null || true
    pkill -f "x11vnc.*rfbport ${VNC_PORT}" 2>/dev/null || true
    pkill -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
    pkill -f "openbox" 2>/dev/null || true
    rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 2>/dev/null || true
    echo "VNC services stopped."
}

if [ "${1:-}" = "stop" ]; then
    cleanup
    exit 0
fi

# Clean up any stale state
cleanup 2>/dev/null || true
sleep 0.5

echo "============================================="
echo "  Starting VNC + noVNC visualization server"
echo "  Resolution : ${RESOLUTION}x${VNC_DEPTH}"
echo "  VNC port   : ${VNC_PORT}"
echo "  noVNC port : ${NOVNC_PORT}"
echo "============================================="

# --- Configure Mesa for software rendering (required for RViz2/OGRE in VNC) ---
# CRITICAL: Do NOT set LIBGL_ALWAYS_INDIRECT=1 — it breaks GLX context creation
#           in Xvfb.  Direct software rendering via llvmpipe is what we need.
export LIBGL_ALWAYS_SOFTWARE=1
export LIBGL_ALWAYS_INDIRECT=0
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

# --- Start Xvfb (virtual framebuffer with proper Mesa GLX support) ---
echo "Starting Xvfb on display ${VNC_DISPLAY} ..."
Xvfb ${VNC_DISPLAY} -screen 0 "${RESOLUTION}x${VNC_DEPTH}" +iglx &
sleep 1

export DISPLAY=${VNC_DISPLAY}

# --- Start x11vnc to share the Xvfb display over VNC ---
echo "Starting x11vnc on port ${VNC_PORT} ..."
x11vnc -display ${VNC_DISPLAY} \
    -rfbport "${VNC_PORT}" \
    -nopw \
    -shared \
    -forever \
    -noxdamage \
    -bg \
    -o /tmp/x11vnc.log \
    2>/dev/null
sleep 0.5

# Start a lightweight window manager
echo "Starting openbox window manager ..."
openbox &
sleep 0.5

# Launch noVNC (websockify proxy)
echo "Starting noVNC websockify on port ${NOVNC_PORT} ..."
websockify --web="${NOVNC_DIR}" ${NOVNC_PORT} localhost:${VNC_PORT} &
sleep 1

echo ""
echo "============================================="
echo "  VNC is ready!"
echo ""
echo "  Open in your browser:"
echo "    http://localhost:${NOVNC_PORT}/vnc.html"
echo ""
echo "  Any GUI app launched with DISPLAY=${VNC_DISPLAY}"
echo "  will appear in the browser window."
echo ""
echo "  To stop:  $0 stop"
echo "============================================="

# Export DISPLAY so subsequent commands in this shell use VNC
echo ""
echo "Run this in your terminal to use the VNC display:"
echo "  export DISPLAY=${VNC_DISPLAY}"
