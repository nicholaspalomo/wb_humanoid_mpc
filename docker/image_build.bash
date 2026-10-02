#!/usr/bin/env bash
#
# Usage:
#
# $ cd wb_humanoid_mpc/docker
# $ ./image_build.bash     # Build the WB Humanoid MPC Docker image (Ubuntu 24.04, no ROS)
# $ IMAGE_TAG=wb-humanoid-mpc:test ./image_build.bash    # ... under another tag
#
# (Cross reference this file with the "build" section of ../docker-compose.yaml, which ../.devcontainer/devcontainer.json
# builds the dev container from)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
CONTEXT="${SCRIPT_DIR}/.."
TARGET="base"


: "# Workspace directory inside container"
WB_HUMANOID_MPC_DIR="/wb_humanoid_mpc_ws"
PYTHON_VERSION="3.12"
USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
GIT_USER_NAME="$(git config --global user.name || echo '')"
GIT_USER_EMAIL="$(git config --global user.email || echo '')"

IMAGE_TAG="${IMAGE_TAG:-wb-humanoid-mpc:dev}"

docker build \
  --file "${DOCKERFILE}" \
  --target "${TARGET}" \
  --build-arg WB_HUMANOID_MPC_DIR="${WB_HUMANOID_MPC_DIR}" \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg USER_ID="${USER_ID}" \
  --build-arg GROUP_ID="${GROUP_ID}" \
  --build-arg GIT_USER_NAME="${GIT_USER_NAME}" \
  --build-arg GIT_USER_EMAIL="${GIT_USER_EMAIL}" \
  --tag "${IMAGE_TAG}" \
  "${CONTEXT}"

echo "Built image: ${IMAGE_TAG}"
