#!/bin/sh
# Installs Pinocchio and its Python bindings from robotpkg, the LAAS-CNRS apt repository, into /opt/openrobots.
#
# The dev image (docker/Dockerfile), CI (.github/workflows/build_test.yml) and the local CI emulator (tools/ci_local.sh)
# all run this one script, so all three get the same Pinocchio. ROS 2 Jazzy used to provide it as a ROS package;
# robotpkg ships the same release, built against Ubuntu's Eigen, urdfdom, console_bridge and octomap, together with
# coal, eigenpy and casadi. The image puts /opt/openrobots on PATH, PYTHONPATH, CMAKE_PREFIX_PATH and PKG_CONFIG_PATH,
# but not on LD_LIBRARY_PATH (docker/Dockerfile says why).
#
# Boost: Pinocchio and coal still link Boost (filesystem, serialization, python), so these packages pull Ubuntu's Boost
# 1.83 libraries in as their own dependencies. That is the only way Boost reaches the system: nothing installs it
# explicitly.
#
# Usage, as root: sh docker/install_robotpkg.sh
# It leaves the apt lists in place, because CI installs dependencies.txt right after it without another update; the
# image deletes them in the same layer. POSIX sh, so that it runs in any container before anything else is set up.
set -eu

# LINT.IfChange(robotpkg_packages)
# The release ROS 2 Jazzy shipped. The Python bindings are built for Ubuntu 24.04's Python 3.12, hence py312; the
# image's PYTHONPATH names the same version.
PINOCCHIO_VERSION="4.1.0"
PINOCCHIO_PACKAGES="robotpkg-pinocchio=${PINOCCHIO_VERSION} robotpkg-py312-pinocchio=${PINOCCHIO_VERSION}"
# LINT.ThenChange(//docker/Dockerfile:robotpkg_environment, //docker/build_pinocchio_from_source.sh:pinocchio_source_versions)

# robotpkg serves its key and its packages over plain HTTP only: nothing listens on port 443. The key is therefore
# checked against its fingerprint before apt is told to trust it, and apt then checks every package against the key.
ROBOTPKG_URL="http://robotpkg.openrobots.org/packages/debian"
ROBOTPKG_KEY_FINGERPRINT="F6F93D4D425860C0B0FBE848ADD535E05E56C3FD"
KEYRING="/etc/apt/keyrings/robotpkg.asc"

architecture="$(dpkg --print-architecture)"
if [ "${architecture}" != "amd64" ]; then
  echo "install_robotpkg.sh: robotpkg publishes amd64 packages only, and this system is ${architecture}:" \
    "build Pinocchio from source with docker/build_pinocchio_from_source.sh instead." >&2
  exit 1
fi

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

install -d -m 0755 /etc/apt/keyrings
download="$(mktemp)"
gnupg_home="$(mktemp -d)"
trap 'rm -rf "${download}" "${gnupg_home}"' EXIT
curl -fsSL "${ROBOTPKG_URL}/robotpkg.asc" -o "${download}"
fingerprint="$(GNUPGHOME="${gnupg_home}" gpg --batch --show-keys --with-colons "${download}" \
  | awk -F: '$1 == "fpr" { print $10; exit }')"
if [ "${fingerprint}" != "${ROBOTPKG_KEY_FINGERPRINT}" ]; then
  echo "install_robotpkg.sh: the robotpkg key has fingerprint '${fingerprint}', expected ${ROBOTPKG_KEY_FINGERPRINT}." >&2
  exit 1
fi
install -m 0644 "${download}" "${KEYRING}"

codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
echo "deb [arch=amd64 signed-by=${KEYRING}] ${ROBOTPKG_URL}/pub ${codename} robotpkg" \
  > /etc/apt/sources.list.d/robotpkg.list

apt-get update
# Word splitting of PINOCCHIO_PACKAGES is intended: one argument per package.
# shellcheck disable=SC2086
apt-get install -y --no-install-recommends ${PINOCCHIO_PACKAGES}
