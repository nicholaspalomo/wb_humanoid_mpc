#!/bin/bash
# ==============================================================================
# setup_env.sh - the shell environment of the dev container, CI and the local CI emulator.
#
# Usage:
#   source setup_env.sh
#
# Every shell in the dev container sources it (.devcontainer/shell_init.sh, through BASH_ENV), so it only exports
# variables: it writes no file and starts no process. It sets up
#   - PATH: the Bazel-built binaries (.bazel/bin) and robotpkg's tools (/opt/openrobots/bin);
#   - PYTHONPATH: robotpkg's Pinocchio bindings, for the tools that run under the system Python rather than under
#     Bazel's hermetic one (tools/locomotion_heuristics, the notebooks), and the checkout's Python packages that those
#     tools import (humanoid_learning, humanoid_common_mpc_pyutils);
#   - the display defaults of the VNC desktop and of software GL (.devcontainer/start_vnc.sh).
#
# LD_LIBRARY_PATH is left alone. Robotpkg's libraries and Python modules find each other through their own RUNPATH,
# and every Bazel-built binary through the RUNPATH Bazel links with; robotpkg's /opt/openrobots/lib also holds a
# libblasfeo.so of another version than the solver's (docker/Dockerfile, bazel/system_libs.bzl).
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# ==============================================================================
# The environment must not depend on the build
# ==============================================================================
# .bazelrc passes PATH into every build action (--action_env), so its value is part of every cache key. It used to gain
# .bazel/bin only once a build had created it, so a fresh checkout was set up one way for `bazel build` and another
# way for the `bazel test` after it, and CI's test step rebuilt the whole workspace a second time. So every directory
# below is added whether it exists yet or not; every lookup skips a missing one.

# Where Bazel puts its convenience symlinks. Fixed, whether or not a build has made them yet.
# LINT.IfChange(symlink_prefix)
_BAZEL_SYMLINKS="${SCRIPT_DIR}/.bazel"
# LINT.ThenChange(//.bazelrc:symlink_prefix)

# Robotpkg's prefix (docker/install_robotpkg.sh). The dev image already exports these; CI and tools/ci_local.sh export
# the same values, and a shell that did not inherit them still gets them here.
# LINT.IfChange(robotpkg_environment)
_ROBOTPKG_PREFIX="/opt/openrobots"
_ROBOTPKG_PYTHONPATH=""
for _robotpkg_site_packages in "${_ROBOTPKG_PREFIX}"/lib/python3.*/site-packages; do
    if [ -d "${_robotpkg_site_packages}" ]; then
        _ROBOTPKG_PYTHONPATH="${_ROBOTPKG_PYTHONPATH:+${_ROBOTPKG_PYTHONPATH}:}${_robotpkg_site_packages}"
    fi
done
unset _robotpkg_site_packages
# LINT.ThenChange(//docker/Dockerfile:robotpkg_environment)

export PATH="${_BAZEL_SYMLINKS}/bin:${_ROBOTPKG_PREFIX}/bin:${PATH:-/usr/bin:/bin}"
export PYTHONPATH="${SCRIPT_DIR}/humanoid_learning:${SCRIPT_DIR}/humanoid_nmpc/humanoid_common_mpc_pyutils${_ROBOTPKG_PYTHONPATH:+:${_ROBOTPKG_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}"

# ==============================================================================
# Display
# ==============================================================================
# The VNC desktop's display (.devcontainer/start_vnc.sh) unless the shell has one of its own, and Mesa's software
# renderer for the MuJoCo viewer and the Rerun viewer in a container without a GPU.
if [ "${DISPLAY:-}" = ":1" ] || [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=":99"
fi
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"

# Sourcing this file twice - which a login shell does, and which several shells per build do - used to prepend the
# same directories again each time, so two shells that had set up identically ended up with different values of PATH.
# That is not merely untidy: .bazelrc passes PATH into every build action, so a shell whose PATH carries one extra copy
# of a directory invalidates the entire build. Deduplicating makes sourcing idempotent, and the cache then does its job.
_dedupe_path_var() {
    local name="$1"
    local value="${!name:-}"
    [ -n "$value" ] || return 0
    local out="" entry
    local IFS=':'
    for entry in $value; do
        [ -n "$entry" ] || continue
        case ":${out}:" in
            *":${entry}:"*) continue ;;
        esac
        out="${out:+${out}:}${entry}"
    done
    export "${name}=${out}"
}

_dedupe_path_var PATH
_dedupe_path_var PYTHONPATH

unset _BAZEL_SYMLINKS _ROBOTPKG_PREFIX _ROBOTPKG_PYTHONPATH
unset -f _dedupe_path_var

# Only a person at a prompt is told: every non-interactive bash in the container sources this file too (BASH_ENV), and
# its output would end up in theirs.
case "$-" in
    *i*) echo "Bazel environment ready (setup_env.sh)." ;;
esac
