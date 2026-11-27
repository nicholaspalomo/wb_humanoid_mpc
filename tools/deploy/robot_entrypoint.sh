#!/bin/sh
# The entry point of the robot images (docker/Dockerfile, targets robot-runtime and robot-sim): runs the robot process
# of the robot configuration ROBOT names, through the script exported from its launch/robot.textproto
# (tools/deploy/README.md). The process replaces this script, so `docker stop` (SIGTERM) reaches it and it ends cleanly.
#
# Environment:
#   ROBOT                      the robot configuration, the name of a script in launch/: drc_atlas, unitree_g1_wb, ...
#   WB_ROBOT_<NAME>            the variables of the launch file (the network file, the backend, the realtime priority,
#                              the cores; see the header of launch/<robot>.sh). Unset or empty: the launch file's value.
#   WB_ROBOT_VIEWER            the MuJoCo viewer by name, or names in order of preference: glfw (a window on DISPLAY),
#                              web (a page at http://localhost:6090), none; web,none falls back to none. Unset or
#                              empty: the mujoco_viewer of the plot configuration when WB_ROBOT_PLOT_CONFIG names one;
#                              else WB_ROBOT_DEFAULT_VIEWER, which robot-runtime sets to none (it has no GL); else the
#                              mujoco_viewer of the launch file's plot configuration (robot-sim sets no default: web,
#                              unless the file names another).
#   WB_ROBOT_PLOT_CONFIG       the plot configuration whose mujoco_viewer names the viewer when WB_ROBOT_VIEWER does
#                              not (the simulation's is the checkout's, mounted); unset or empty: the launch file's.
#   NETEM, NETEM_INTERFACE     tc-netem parameters ("delay 3ms 1ms loss 0.5%") for the bus's packets on NETEM_INTERFACE
#                              (default lo) while the robot runs (netem.sh); empty: none. Needs CAP_NET_ADMIN.
#
# POSIX sh: the image has no bash.
set -eu

bundle="$(cd "$(dirname "$0")/.." && pwd)"

robots() {
    for script in "${bundle}"/launch/*.sh; do
        [ -f "${script}" ] && printf '%s ' "$(basename "${script}" .sh)"
    done
}

if [ -z "${ROBOT:-}" ]; then
    echo "robot_entrypoint: set ROBOT to the robot configuration, one of: $(robots)" >&2
    exit 2
fi
case "${ROBOT}" in
    *[!A-Za-z0-9_]*)
        echo "robot_entrypoint: ROBOT=${ROBOT} is not a robot configuration; they are: $(robots)" >&2
        exit 2
        ;;
esac
script="${bundle}/launch/${ROBOT}.sh"
if [ ! -f "${script}" ]; then
    echo "robot_entrypoint: unknown ROBOT=${ROBOT}; the robot configurations are: $(robots)" >&2
    exit 2
fi

# The image's viewer, unless the environment names one or a plot configuration to take it from.
if [ -z "${WB_ROBOT_VIEWER:-}" ] && [ -z "${WB_ROBOT_PLOT_CONFIG:-}" ] && [ -n "${WB_ROBOT_DEFAULT_VIEWER:-}" ]; then
    WB_ROBOT_VIEWER="${WB_ROBOT_DEFAULT_VIEWER}"
    export WB_ROBOT_VIEWER
fi

# What the realtime loop may take here: the robot process reports any setting it is refused, this says why.
limit() {
    awk -v name="$1" 'index($0, name) == 1 { print $(NF - 1); exit }' /proc/self/limits
}
cpus="$(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || echo unknown)"
echo "robot_entrypoint: ROBOT=${ROBOT}; realtime priority limit $(limit 'Max realtime priority')," \
    "locked memory limit $(limit 'Max locked memory'), cpus ${cpus}"

if [ -z "${NETEM:-}" ]; then
    exec "${script}"
fi

interface="${NETEM_INTERFACE:-lo}"
# Word splitting of NETEM is intended: one argument per word of the netem parameters.
# shellcheck disable=SC2086
"${bundle}/bin/netem.sh" apply "${interface}" ${NETEM}
"${script}" &
child=$!
trap 'kill -TERM "${child}" 2>/dev/null || true' TERM INT HUP
status=0
# wait returns early when a trapped signal arrives; wait again until the robot process has ended.
while kill -0 "${child}" 2>/dev/null; do
    if wait "${child}"; then status=0; else status=$?; fi
done
"${bundle}/bin/netem.sh" clear "${interface}" || true
exit "${status}"
