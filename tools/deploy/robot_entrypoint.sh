#!/bin/sh
# The entry point of the robot images (docker/Dockerfile, targets robot-runtime and robot-sim): runs the robot process
# of the robot configuration ROBOT names, through the script exported from its launch/robot.textproto
# (tools/deploy/README.md). The process replaces this script, so `docker stop` (SIGTERM) reaches it and it ends cleanly.
#
# Environment:
#   ROBOT                      the robot configuration, the name of a script in launch/: drc_atlas, unitree_g1_wb, ...
#   WB_ROBOT_<NAME>            the variables of the launch file (the network file, the backend, the realtime priority,
#                              the cores; see the header of launch/<robot>.sh). Unset or empty: the launch file's value.
#   WB_ROBOT_HEADLESS          unset or empty: WB_ROBOT_DEFAULT_HEADLESS, which the image sets (robot-runtime has no GL).
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

if [ -z "${WB_ROBOT_HEADLESS:-}" ] && [ -n "${WB_ROBOT_DEFAULT_HEADLESS:-}" ]; then
    WB_ROBOT_HEADLESS="${WB_ROBOT_DEFAULT_HEADLESS}"
    export WB_ROBOT_HEADLESS
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
