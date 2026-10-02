#!/bin/sh
# Builds Pinocchio and coal from source into /opt/openrobots, where docker/install_robotpkg.sh puts robotpkg's packages,
# for an architecture robotpkg publishes no packages for (arm64: an ARM robot computer, docker buildx --platform
# linux/arm64). The robot image's arm64 stage runs it (docker/Dockerfile, robot-pinocchio-arm64); the dev image runs it
# on arm64 in place of install_robotpkg.sh. bazel/system_libs.bzl finds the result where it finds robotpkg's.
#
# It builds the releases install_robotpkg.sh installs, with the same features (URDF parsing, collision through coal),
# so the binaries link the same libraries: libpinocchio_default, libpinocchio_parsers and libcoal. The Python bindings
# are left out unless PINOCCHIO_PYTHON=ON, which also builds eigenpy and needs the dev image's Python.
#
# Usage, as root: sh docker/build_pinocchio_from_source.sh       (PINOCCHIO_PYTHON=ON|OFF, JOBS=<n>)
# POSIX sh, so that it runs in any container before anything else is set up.
set -eu

# LINT.IfChange(pinocchio_source_versions)
# PINOCCHIO_VERSION is install_robotpkg.sh's; COAL_VERSION the coal release robotpkg builds that Pinocchio against.
PINOCCHIO_VERSION="4.1.0"
COAL_VERSION="3.0.4"
EIGENPY_VERSION="3.13.0"
# LINT.ThenChange(//docker/install_robotpkg.sh:robotpkg_packages)
PREFIX="/opt/openrobots"
PINOCCHIO_PYTHON="${PINOCCHIO_PYTHON:-OFF}"
JOBS="${JOBS:-$(nproc)}"

apt-get update
# The libraries Pinocchio and coal link (those robotpkg's packages depend on), and the tools to build them.
apt-get install -y --no-install-recommends \
    ca-certificates git build-essential cmake pkg-config \
    libeigen3-dev libboost-filesystem-dev libboost-serialization-dev libboost-system-dev \
    liburdfdom-dev libconsole-bridge-dev liboctomap-dev libassimp-dev libqhull-dev
if [ "${PINOCCHIO_PYTHON}" = "ON" ]; then
    apt-get install -y --no-install-recommends python3-dev python3-numpy python3-scipy libboost-python-dev
fi

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

build() {
    name="$1"
    url="$2"
    tag="$3"
    shift 3
    git clone --quiet --depth 1 --branch "${tag}" --recurse-submodules --shallow-submodules "${url}" "${work}/${name}"
    cmake -S "${work}/${name}" -B "${work}/${name}/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_INSTALL_RPATH="${PREFIX}/lib" \
        -DBUILD_TESTING=OFF \
        "$@"
    cmake --build "${work}/${name}/build" --parallel "${JOBS}"
    cmake --install "${work}/${name}/build"
}

if [ "${PINOCCHIO_PYTHON}" = "ON" ]; then
    build eigenpy https://github.com/stack-of-tasks/eigenpy "v${EIGENPY_VERSION}"
fi
build coal https://github.com/coal-library/coal "v${COAL_VERSION}" \
    -DBUILD_PYTHON_INTERFACE="${PINOCCHIO_PYTHON}" \
    -DCOAL_HAS_QHULL=ON \
    -DCOAL_BACKWARD_COMPATIBILITY_WITH_HPP_FCL=ON
build pinocchio https://github.com/stack-of-tasks/pinocchio "v${PINOCCHIO_VERSION}" \
    -DCMAKE_PREFIX_PATH="${PREFIX}" \
    -DBUILD_PYTHON_INTERFACE="${PINOCCHIO_PYTHON}" \
    -DBUILD_WITH_URDF_SUPPORT=ON \
    -DBUILD_WITH_COLLISION_SUPPORT=ON \
    -DBUILD_WITH_CASADI_SUPPORT=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DBUILD_UTILS=OFF
echo "build_pinocchio_from_source: Pinocchio ${PINOCCHIO_VERSION} and coal ${COAL_VERSION} in ${PREFIX}"
