#!/usr/bin/env bash
# Runs a shell command in the checkout inside the dev container: here, when this shell is in a container already (the
# dev container's own terminal), or through `docker exec` into the dev container when it runs on the host.
#
#   tools/deploy/in_dev_container.sh 'bazel build //tools/deploy:robot_bundle'
#
# The Makefile's launch and deployment targets build with Bazel and run the laptop side in the dev container this way,
# while the robot side runs in a container of its own (docker-compose.robot.yaml), which only the host's Docker can
# start. So the same `make launch-<robot>-sim` works from the host and, given access to Docker, from the dev container.
#
# Environment:
#   WB_DEV_CONTAINER       the dev container (default devcontainer-app-1, the name docker-compose.yaml gives it)
#   WB_DEV_CONTAINER_REPO  this checkout's path inside it (default /wb_humanoid_mpc_ws/<parent>/<checkout>, where
#                          docker-compose.yaml mounts the checkout's grandparent)
#   DISPLAY                passed on to the command
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: in_dev_container.sh '<shell command>'" >&2
    exit 2
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# /.dockerenv exists in every Docker container.
if [ -f /.dockerenv ]; then
    cd "${repo}"
    exec bash -c "$1"
fi

container="${WB_DEV_CONTAINER:-devcontainer-app-1}"
# LINT.IfChange(dev_container_mount)
container_repo="${WB_DEV_CONTAINER_REPO:-/wb_humanoid_mpc_ws/$(basename "$(dirname "${repo}")")/$(basename "${repo}")}"
# LINT.ThenChange(//docker-compose.yaml:dev_container_mount)
if ! command -v docker > /dev/null 2>&1; then
    echo "in_dev_container.sh: no docker here to reach the dev container '${container}'" >&2
    exit 1
fi
if [ "$(docker inspect --format '{{.State.Running}}' "${container}" 2> /dev/null)" != "true" ]; then
    echo "in_dev_container.sh: the dev container '${container}' is not running. Start it (Dev Containers: Reopen in" \
        "Container, or docker compose up -d), or name another with DEV_CONTAINER=<name>." >&2
    exit 1
fi
# A dev container of this checkout's image has Pinocchio in /opt/openrobots and no ROS. One created from the image
# before the switch from ROS has /opt/ros instead, and cannot build this checkout (bazel/system_libs.bzl).
if ! docker exec "${container}" test -d /opt/openrobots; then
    echo "in_dev_container.sh: the dev container '${container}' has no /opt/openrobots: it was created from the ROS image" \
        "and cannot build this checkout. Recreate it from docker-compose.yaml (Dev Containers: Rebuild Container, or" \
        "docker compose up -d --build --force-recreate), or name another with DEV_CONTAINER=<name>." >&2
    exit 1
fi
flags=(--interactive --workdir "${container_repo}")
if [ -t 0 ] && [ -t 1 ]; then
    flags+=(--tty)
fi
if [ -n "${DISPLAY:-}" ]; then
    flags+=(--env "DISPLAY=${DISPLAY}")
fi
exec docker exec "${flags[@]}" "${container}" bash -c "$1"
