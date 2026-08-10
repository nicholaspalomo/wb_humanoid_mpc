#!/usr/bin/env bash
#
# Usage:
#
# $ cd ~/your_colcon_ws/src/wb_humanoid_mpc/docker
# $ ./launch_wb_mpc.bash    # Launch the WB Humanoid MPC Docker container
#
# (Cross reference this file with the "run" section of ../.devcontainer/devcontainer.json)
#
set -euo pipefail

# Allow GUI applications if DISPLAY is set
if [ -n "${DISPLAY:-}" ]; then
  xhost +SI:localuser:root 2>/dev/null || true

  # Generate Xauthority file for X11 forwarding
  XAUTH=/tmp/.docker.xauth
  if [ ! -f "${XAUTH}" ]; then
    touch "${XAUTH}"
    xauth nlist "${DISPLAY}" 2>/dev/null \
      | sed -e 's/^..../ffff/' \
      | xauth -f "${XAUTH}" nmerge - 2>/dev/null || true
    chmod a+r "${XAUTH}" 2>/dev/null || true
  fi
fi

# Detect repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Check if the container is already running or exists
CONTAINER_NAME="wb-mpc-dev"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Attaching to existing running container '${CONTAINER_NAME}'..."
  exec docker exec -it "${CONTAINER_NAME}" bash
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Starting and attaching to existing stopped container '${CONTAINER_NAME}'..."
  docker start "${CONTAINER_NAME}" >/dev/null
  exec docker exec -it "${CONTAINER_NAME}" bash
else
  # Run a new container instance
  docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --net host \
    --privileged \
    -u ubuntu \
    -e DISPLAY="${DISPLAY:-:99}" \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY="${XAUTH:-/tmp/.docker.xauth}" \
    -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${XAUTH:-/tmp/.docker.xauth}:${XAUTH:-/tmp/.docker.xauth}:rw" \
    -v "${HOME}/.ssh:/tmp/host_ssh:cached" \
    -v "${HOME}/.gitconfig:/tmp/host.gitconfig:cached,ro" \
    -v "${REPO_ROOT}:/wb_humanoid_mpc_ws/src/wb_humanoid_mpc:cached" \
    --workdir /wb_humanoid_mpc_ws/src/wb_humanoid_mpc \
    wb-humanoid-mpc:dev \
    bash -c "if [ -f .devcontainer/post_create.sh ]; then chmod +x .devcontainer/post_create.sh && .devcontainer/post_create.sh; fi; exec bash"
fi

echo "Done."

