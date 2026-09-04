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
VNC_DISPLAY=":99"

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
    pkill -9 -f "Xvfb" 2>/dev/null || true
    pkill -9 -f "x11vnc" 2>/dev/null || true
    pkill -9 -f "websockify" 2>/dev/null || true
    pkill -9 -f "openbox" 2>/dev/null || true
    fuser -k -9 ${NOVNC_PORT}/tcp 2>/dev/null || true
    fuser -k -9 ${VNC_PORT}/tcp 2>/dev/null || true
    sudo -n rm -f /tmp/.X*-lock /tmp/.X11-unix/X* 2>/dev/null || true
    rm -f /tmp/.X*-lock /tmp/.X11-unix/X* 2>/dev/null || true
    echo "VNC services stopped."
}

if [ "${1:-}" = "stop" ]; then
    cleanup
    exit 0
fi

# If VNC is already running healthy, keep it active
if pgrep -f "Xvfb.*:99" >/dev/null 2>&1 && pgrep -f "x11vnc.*${VNC_PORT}" >/dev/null 2>&1 && pgrep -f "websockify.*${NOVNC_PORT}" >/dev/null 2>&1; then
    echo "✅ VNC server is already running on ${VNC_DISPLAY} (noVNC port ${NOVNC_PORT})"
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
Xvfb ${VNC_DISPLAY} -screen 0 "${RESOLUTION}x${VNC_DEPTH}" +iglx >/dev/null 2>&1 &
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
    >/dev/null 2>&1 || true
sleep 0.5

# Setup Openbox window layout rules for side-by-side simulation + RViz viewing
mkdir -p "${HOME}/.config/openbox"
if [ -f "/etc/xdg/openbox/rc.xml" ]; then
    cp /etc/xdg/openbox/rc.xml "${HOME}/.config/openbox/rc.xml"
    python3 -c '
import os
rc_path = os.path.expanduser("~/.config/openbox/rc.xml")
with open(rc_path, "r") as f:
    content = f.read()
rules = """
  <application class="*rviz*" title="*rviz*" name="*rviz*">
    <position force="yes"><x>0</x><y>0</y></position>
    <size><width>950</width><height>1040</height></size>
    <maximized>no</maximized>
  </application>
  <application title="*Mujoco*" name="*" class="*">
    <position force="yes"><x>960</x><y>0</y></position>
    <size><width>950</width><height>1040</height></size>
    <maximized>no</maximized>
    <focus>yes</focus>
  </application>
"""
if "</applications>" in content:
    content = content.replace("</applications>", rules + "\n</applications>")
    with open(rc_path, "w") as f:
        f.write(content)
' 2>/dev/null || true
fi

# Start a lightweight window manager
echo "Starting openbox window manager ..."
openbox &
sleep 0.5
xsetroot -solid "#1e222d" 2>/dev/null || true

# Launch noVNC (websockify proxy)
echo "Starting noVNC websockify on port ${NOVNC_PORT} ..."
websockify --web="${NOVNC_DIR}" ${NOVNC_PORT} localhost:${VNC_PORT} >/dev/null 2>&1 &
sleep 1

# Discover host/LAN IPs for easy browser access from remote laptops
LAN_IPS=$(ip -4 addr show 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -vE '^(127\.|172\.(1[6-9]|2[0-9]|3[0-1])\.)' || true)

echo ""
echo "============================================="
echo "  VNC is ready!"
echo ""
echo "  Open in your browser:"
if [ -n "${LAN_IPS}" ]; then
    for ip in ${LAN_IPS}; do
        echo "    🔗 http://${ip}:${NOVNC_PORT}/vnc.html"
    done
fi
echo "    🔗 http://localhost:${NOVNC_PORT}/vnc.html"
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
