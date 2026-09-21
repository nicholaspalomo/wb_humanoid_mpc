#!/bin/bash
# ==============================================================================
# setup_env.sh — Source this to set up the ROS2 environment for Bazel-built
# binaries, without needing colcon.
#
# Usage:
#   source setup_env.sh
#   ros2 launch g1_centroidal_mpc dummy_sim.launch.py
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# LINT.IfChange(ros_distro)
# Source base ROS2 installation (temporarily disable strict mode during ROS2 vendor script sourcing)
_SAVED_OPTS="$-"
set +u
set +e
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
elif [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
elif [ -f /bin/ros_setup.sh ]; then
    source /bin/ros_setup.sh
fi
# Restore previous shell options
[[ "$_SAVED_OPTS" =~ e ]] && set -e
[[ "$_SAVED_OPTS" =~ u ]] && set -u
unset _SAVED_OPTS
# LINT.ThenChange(//docker/Dockerfile:ros_distro, //tools/ci_local.sh:ros_distro, //bazel/ros2.bzl:ros_distro, //bazel/system_libs.bzl:ros_distro)

# ==============================================================================
# Create ament_index-compatible directory structure pointing to source tree
# ==============================================================================
BAZEL_INSTALL="${TMPDIR:-/tmp}/.bazel_ros_install"
if [ -e "${SCRIPT_DIR}/.bazel/bin" ]; then
    BAZEL_BIN="${SCRIPT_DIR}/.bazel/bin"
elif [ -e "${SCRIPT_DIR}/bazel-bin" ]; then
    BAZEL_BIN="${SCRIPT_DIR}/bazel-bin"
else
    BAZEL_BIN="${SCRIPT_DIR}/.bazel/bin"
fi

# True for the directory entries _setup_package copies: everything but the Bazel files and the source tree.
_is_share_asset() {
    case "$1" in
        BUILD.bazel|BUILD|src|test|include) return 1 ;;
        *) return 0 ;;
    esac
}

# A fingerprint of the assets _setup_package would copy: their paths, sizes and modification times. Cheap enough to
# run on every shell, and it changes whenever a config, URDF or launch file changes - which is exactly when the
# installed copy has to be rebuilt.
_package_stamp() {
    local source_dir="$1"
    local item base
    for item in "${source_dir}"/*; do
        [ -e "$item" ] || continue
        base="$(basename "$item")"
        _is_share_asset "$base" || continue
        find -L "$item" -printf '%p %s %T@\n' 2>/dev/null
    done | sort | md5sum
}

_setup_package() {
    local pkg_name="$1"
    local source_dir="$2"
    local prefix="${BAZEL_INSTALL}/${pkg_name}"

    mkdir -p "${prefix}/share/ament_index/resource_index/packages"
    touch "${prefix}/share/ament_index/resource_index/packages/${pkg_name}"
    # Create lib directory for executables (required by ros2 launch)
    mkdir -p "${prefix}/lib/${pkg_name}"

    # Every shell sources this file, so the copy below used to run on every terminal, every build and every sim
    # launch - deleting and recreating a directory that other processes were reading out of. A test or a simulator
    # that happened to open a task.yaml or a URDF during that window saw a missing or half-written file and failed
    # with an error that looked like a code defect. Two guards make that impossible:
    #
    #   1. a stamp, so a shell with nothing new to install writes nothing at all, which is the overwhelmingly common
    #      case and the one that was doing the damage;
    #   2. a lock, so that when a copy is genuinely needed, only one shell performs it.
    local stamp stamp_file="${prefix}/.source_stamp"
    stamp="$(_package_stamp "${source_dir}")"
    [ "$(cat "${stamp_file}" 2>/dev/null)" = "${stamp}" ] && return 0

    local lock_fd=""
    if command -v flock >/dev/null 2>&1; then
        exec {lock_fd}>"${BAZEL_INSTALL}/.${pkg_name}.lock" 2>/dev/null || lock_fd=""
        if [ -n "${lock_fd}" ]; then
            flock -x "${lock_fd}"
            # Another shell may have installed the same assets while this one waited for the lock.
            if [ "$(cat "${stamp_file}" 2>/dev/null)" = "${stamp}" ]; then
                exec {lock_fd}>&-
                return 0
            fi
        fi
    fi

    # The stamp is removed first and written last, so an interrupted copy leaves no stamp and the next shell redoes it.
    rm -f "${stamp_file}" 2>/dev/null
    # Copy share assets (config, urdf, launch, package.xml, etc.) excluding Bazel BUILD files and source code
    rm -rf "${prefix}/share/${pkg_name}" 2>/dev/null
    mkdir -p "${prefix}/share/${pkg_name}"
    local item base
    for item in "${source_dir}"/*; do
        if [ -e "$item" ]; then
            base="$(basename "$item")"
            if _is_share_asset "$base"; then
                cp -rL "$item" "${prefix}/share/${pkg_name}/${base}"
            fi
        fi
    done
    echo "${stamp}" > "${stamp_file}"

    [ -n "${lock_fd}" ] && exec {lock_fd}>&-
    return 0
}

# Links a Bazel-built binary into the ament lib directory
_link_node() {
    local pkg_name="$1"       # e.g. humanoid_centroidal_mpc_ros2
    local bazel_pkg="$2"      # e.g. humanoid_nmpc/humanoid_centroidal_mpc_ros2
    local binary_name="$3"    # e.g. humanoid_centroidal_mpc_sqp_node
    local prefix="${BAZEL_INSTALL}/${pkg_name}"

    rm -f "${prefix}/lib/${pkg_name}/${binary_name}" 2>/dev/null
    ln -sf "${BAZEL_BIN}/${bazel_pkg}/${binary_name}" \
           "${prefix}/lib/${pkg_name}/${binary_name}"
}

# LINT.IfChange(registered_packages)
# --- Robot model packages ---
# _setup_package copies a package's share assets (config, urdf, meshes, launch, rviz) into the ament index.
# A package that also builds an executable needs a _link_node next to it, or `ros2 run <pkg> <exe>` answers
# "No executable found": _setup_package creates lib/<pkg> but never puts anything in it, and `test/` is not a
# share asset. That is what the Makefile's test-pinocchio-model-* targets run.
_setup_package "g1_description" \
    "${SCRIPT_DIR}/robot_models/unitree_g1/g1_description"

_setup_package "g1_centroidal_mpc" \
    "${SCRIPT_DIR}/robot_models/unitree_g1/g1_centroidal_mpc"

_setup_package "g1_wb_mpc" \
    "${SCRIPT_DIR}/robot_models/unitree_g1/g1_wb_mpc"

# --- DRC Atlas robot model packages ---
_setup_package "drc_atlas_description" \
    "${SCRIPT_DIR}/robot_models/drc_atlas/drc_atlas_description"

_setup_package "drc_atlas_centroidal_mpc" \
    "${SCRIPT_DIR}/robot_models/drc_atlas/drc_atlas_centroidal_mpc"
_link_node "drc_atlas_centroidal_mpc" "robot_models/drc_atlas/drc_atlas_centroidal_mpc" \
    "test_pinocchio_model"

# --- Unitree R1 robot model packages ---
_setup_package "unitree_r1_description" \
    "${SCRIPT_DIR}/robot_models/unitree_r1/unitree_r1_description"

_setup_package "unitree_r1_centroidal_mpc" \
    "${SCRIPT_DIR}/robot_models/unitree_r1/unitree_r1_centroidal_mpc"
_link_node "unitree_r1_centroidal_mpc" "robot_models/unitree_r1/unitree_r1_centroidal_mpc" \
    "test_pinocchio_model"

# --- EngineAI SA01 robot model packages ---
_setup_package "engineai_sa01_description" \
    "${SCRIPT_DIR}/robot_models/engineai_sa01/engineai_sa01_description"

_setup_package "engineai_sa01_centroidal_mpc" \
    "${SCRIPT_DIR}/robot_models/engineai_sa01/engineai_sa01_centroidal_mpc"
_link_node "engineai_sa01_centroidal_mpc" "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc" \
    "test_pinocchio_model"

# --- Humanoid MPC packages ---
_setup_package "humanoid_common_mpc" \
    "${SCRIPT_DIR}/humanoid_nmpc/humanoid_common_mpc"

_setup_package "humanoid_centroidal_mpc" \
    "${SCRIPT_DIR}/humanoid_nmpc/humanoid_centroidal_mpc"

_setup_package "humanoid_wb_mpc" \
    "${SCRIPT_DIR}/humanoid_nmpc/humanoid_wb_mpc"

# --- Humanoid MPC ROS2 packages (with node executables) ---
_setup_package "humanoid_common_mpc_ros2" \
    "${SCRIPT_DIR}/humanoid_nmpc/humanoid_common_mpc_ros2"
_link_node "humanoid_common_mpc_ros2" "humanoid_nmpc/humanoid_common_mpc_ros2" \
    "gait_keyboard_command_node"
_link_node "humanoid_common_mpc_ros2" "humanoid_nmpc/humanoid_common_mpc_ros2" \
    "velocity_keyboard_command_node"

_setup_package "humanoid_centroidal_mpc_ros2" \
    "${SCRIPT_DIR}/humanoid_nmpc/humanoid_centroidal_mpc_ros2"
_link_node "humanoid_centroidal_mpc_ros2" "humanoid_nmpc/humanoid_centroidal_mpc_ros2" \
    "humanoid_centroidal_mpc_sqp_node"
_link_node "humanoid_centroidal_mpc_ros2" "humanoid_nmpc/humanoid_centroidal_mpc_ros2" \
    "humanoid_centroidal_mpc_dummy_sim_node"
_link_node "humanoid_centroidal_mpc_ros2" "humanoid_nmpc/humanoid_centroidal_mpc_ros2" \
    "humanoid_centroidal_mpc_sim"
_link_node "humanoid_centroidal_mpc_ros2" "humanoid_nmpc/humanoid_centroidal_mpc_ros2" \
    "humanoid_centroidal_mpc_pose_command_node"
_link_node "humanoid_centroidal_mpc_ros2" "humanoid_nmpc/humanoid_centroidal_mpc_ros2" \
    "test_visualizer"

_setup_package "humanoid_wb_mpc_ros2" \
    "${SCRIPT_DIR}/humanoid_nmpc/humanoid_wb_mpc_ros2"
_link_node "humanoid_wb_mpc_ros2" "humanoid_nmpc/humanoid_wb_mpc_ros2" \
    "humanoid_wb_mpc_sqp_node"
_link_node "humanoid_wb_mpc_ros2" "humanoid_nmpc/humanoid_wb_mpc_ros2" \
    "humanoid_wb_mpc_dummy_sim_node"
_link_node "humanoid_wb_mpc_ros2" "humanoid_nmpc/humanoid_wb_mpc_ros2" \
    "humanoid_wb_mpc_sim"
_link_node "humanoid_wb_mpc_ros2" "humanoid_nmpc/humanoid_wb_mpc_ros2" \
    "humanoid_wb_mpc_pose_command_node"

_setup_package "humanoid_centroidal_mpc_test" \
    "${SCRIPT_DIR}/humanoid_nmpc/humanoid_centroidal_mpc_test"

# --- Robot runtime ---
_setup_package "robot_model" \
    "${SCRIPT_DIR}/robot_runtime/robot_model"

_setup_package "robot_core" \
    "${SCRIPT_DIR}/robot_runtime/robot_core"

_setup_package "mujoco_sim_interface" \
    "${SCRIPT_DIR}/robot_runtime/mujoco_sim_interface"

# --- Python-only packages (remote_control) ---
_setup_package "remote_control" \
    "${SCRIPT_DIR}/humanoid_nmpc/remote_control"
# LINT.ThenChange(//Makefile:launch_targets, //.devcontainer/README.md:launch_targets)
# Link the Python entry point for base_velocity_controller_gui
mkdir -p "${BAZEL_INSTALL}/remote_control/lib/remote_control"
cat > "${BAZEL_INSTALL}/remote_control/lib/remote_control/base_velocity_controller_gui" << 'PYEOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, "${SCRIPT_DIR}/humanoid_nmpc/remote_control")
from remote_control.base_velocity_controller_gui import main
main()
PYEOF
chmod +x "${BAZEL_INSTALL}/remote_control/lib/remote_control/base_velocity_controller_gui"
# Fix the path (SCRIPT_DIR wasn't expanded inside heredoc)
sed -i "s|\${SCRIPT_DIR}|${SCRIPT_DIR}|g" \
    "${BAZEL_INSTALL}/remote_control/lib/remote_control/base_velocity_controller_gui"

# ==============================================================================
# Set environment variables
# ==============================================================================

# Build AMENT_PREFIX_PATH from all registered packages
_BAZEL_PREFIXES=""
for d in "${BAZEL_INSTALL}"/*/; do
    _BAZEL_PREFIXES="${_BAZEL_PREFIXES:+${_BAZEL_PREFIXES}:}${d%/}"
done

export AMENT_PREFIX_PATH="${_BAZEL_PREFIXES}:${AMENT_PREFIX_PATH}"

# Add Bazel-generated Python message bindings and shared libraries (humanoid_mpc_msgs, ocs2_ros2_msgs)
_OUTPUT_BASE=""
if [ -e "${SCRIPT_DIR}/.bazel/output_base" ]; then
    _OUTPUT_BASE=$(readlink -f "${SCRIPT_DIR}/.bazel/output_base" 2>/dev/null)
elif [ -e "${SCRIPT_DIR}/bazel-out" ]; then
    _OUTPUT_BASE=$(readlink -f "${SCRIPT_DIR}/bazel-out/../../" 2>/dev/null)
fi
if [ -z "$_OUTPUT_BASE" ] || [ ! -d "$_OUTPUT_BASE" ]; then
    if [ -d "${HOME}/.cache/bazel" ]; then
        _OUTPUT_BASE=$(find "${HOME}/.cache/bazel" -maxdepth 3 -type d -name "external" 2>/dev/null | head -n 1 | sed 's|/external$||')
    fi
fi

if [ -n "$_OUTPUT_BASE" ] && [ -d "$_OUTPUT_BASE" ]; then
    for pypath in "${_OUTPUT_BASE}"/external/*msgs_repo/install/*/lib/python3.*/site-packages; do
        [ -d "$pypath" ] && export PYTHONPATH="${pypath}:${PYTHONPATH}"
    done
    for libpath in "${_OUTPUT_BASE}"/external/*msgs_repo/install/*/lib; do
        [ -d "$libpath" ] && export LD_LIBRARY_PATH="${libpath}:${LD_LIBRARY_PATH}"
    done
    for amentpath in "${_OUTPUT_BASE}"/external/*msgs_repo/install/*; do
        [ -d "$amentpath/share" ] && export AMENT_PREFIX_PATH="${amentpath}:${AMENT_PREFIX_PATH}"
    done
fi
unset _OUTPUT_BASE

# Add Python packages to PYTHONPATH (for launch file imports and RL modules)
# Ensure active workspace source directories take precedence at the front of PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}/humanoid_learning:${SCRIPT_DIR}/humanoid_nmpc/humanoid_common_mpc_ros2:${SCRIPT_DIR}/humanoid_nmpc/humanoid_common_mpc_pyutils:${SCRIPT_DIR}/humanoid_nmpc/remote_control:${PYTHONPATH}"

# Add Bazel-built binaries to PATH
if [ -d "${BAZEL_BIN}" ]; then
    export PATH="${BAZEL_BIN}:${PATH}"
fi

# LD_LIBRARY_PATH for ROS2 system libs
if [ -n "${ROS_DISTRO:-}" ]; then
    export LD_LIBRARY_PATH="/opt/ros/${ROS_DISTRO}/lib:/opt/ros/${ROS_DISTRO}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
fi

# Add workspace-built message libs if they exist
for msg_lib_dir in /wb_humanoid_mpc_ws/install/*/lib; do
    [ -d "$msg_lib_dir" ] && export LD_LIBRARY_PATH="${msg_lib_dir}:${LD_LIBRARY_PATH}"
done

# Host IP for browser visualizer connections
export HOST_IP="${HOST_IP:-192.168.0.3}"
if [ "${DISPLAY:-}" = ":1" ] || [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=":99"
fi
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}"

# Sourcing this file twice - which a login shell does, and which several shells per build do - used to prepend the
# same directories again each time, so two shells that had set up identically ended up with different values of
# PATH and LD_LIBRARY_PATH. That is not merely untidy. .bazelrc passes PATH into every build action with
# --action_env and LD_LIBRARY_PATH into every test action with --test_env, so the value is part of the cache key:
# a shell whose PATH carries one extra copy of a directory invalidates the entire build and re-runs every test.
# Deduplicating makes sourcing idempotent, and the cache then does its job.
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
_dedupe_path_var LD_LIBRARY_PATH
_dedupe_path_var PYTHONPATH
_dedupe_path_var AMENT_PREFIX_PATH
_dedupe_path_var CMAKE_PREFIX_PATH

unset _BAZEL_PREFIXES
unset -f _dedupe_path_var
unset -f _is_share_asset
unset -f _package_stamp
unset -f _setup_package
unset -f _link_node

echo "ROS2 + Bazel environment ready. AMENT_PREFIX_PATH set."
