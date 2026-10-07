#!/usr/bin/env bash
# The launch sessions of the Makefile's launch targets (humanoid_nmpc/docs/distributed_runtime/README.md, "Deployment"):
#
#   session.sh sim --robot drc_atlas --launch_file <robot>/launch/mpc.textproto --build "<bazel targets>" \
#       [--netem "delay 3ms 1ms loss 0.5%"] [--headless] [--set name=value ...] [--skip_build]
#   session.sh laptop --launch_file <launch file> --build "<bazel targets>" [--set name=value ...] [--skip_build]
#   session.sh stop          # what a session left running: the robot container, the laptop side, a NETEM qdisc
#   session.sh project       # the compose project of this checkout's simulation (its robot container)
#
# `sim` is simulation in the hardware topology on this machine: it builds the bundle and the robot-sim image
# (deploy_robot.sh image), starts the robot side in the robot-sim container - the robot's own image and compose file
# (docker-compose.robot.yaml) with the MuJoCo viewer added, its realtime settings, the shipped localhost network - and
# follows its log as [robot], then runs the laptop side (the MPC node, the GUI, the Rerun bridge) in the dev container
# over the remote MPC link. When the laptop side ends (Ctrl-C, or its MPC exits) the robot container stops too. The
# robot container's bundled files are this checkout's robot_models (mounted read-only over the image's copy), which seed
# its stored configuration at every start: it starts on the files the MPC node reads, and the GUI's Save reaches it over
# the bus while it runs, as on the robot.
# `laptop` runs only the laptop side of a launch file (the dummy simulator, the MPC against a robot, the sandbox).
#
# The bus's ports (config/ipc/network.textproto) are the host's: a session refuses to start while another holds them -
# a robot deployed on this machine (deploy_robot.sh, compose project wb-humanoid-robot) or the simulation of another
# checkout - and says how to stop it. Each checkout has a compose project of its own, so that `stop` (make kill-sims)
# never touches another checkout's simulation.
#
# The laptop side runs in the dev container (in_dev_container.sh), the robot container with the host's Docker. The PID
# of the laptop side's launcher goes to .deploy/laptop_side.pid of this checkout, so that `stop` ends exactly it.
#
# Environment: DISPLAY (the viewer's and the GUI's), WB_DEV_CONTAINER (in_dev_container.sh).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo}"

# The compose project of this checkout's simulation: apart from a robot deployed on this machine (deploy_robot.sh's
# wb-humanoid-robot) and from the simulation of another checkout.
# LINT.IfChange(sim_project)
PROJECT_PREFIX="wb-humanoid-robot-sim"
DEPLOYMENT_PROJECT="wb-humanoid-robot"
PROJECT="${PROJECT_PREFIX}-$(basename "${repo}" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9_\n-' '-')-$(printf '%s' "${repo}" | cksum | cut -d ' ' -f 1)"
# LINT.ThenChange(//tools/deploy/README.md:sim_project, //tools/deploy/test_deploy_files.py:sim_project)
PID_FILE=".deploy/laptop_side.pid"
LAUNCHER=".bazel/bin/tools/launch/launch"

usage() {
    sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

compose_files=(-f docker-compose.robot.yaml -f docker-compose.robot.sim.yaml)

compose() {
    docker compose -p "${PROJECT}" "${compose_files[@]}" "$@"
}

# The containers of other robot sides that hold the bus's ports on this host: a deployment, another checkout's simulation.
other_robot_containers() {
    docker ps --format '{{.Names}} {{.Label "com.docker.compose.project"}}' 2> /dev/null |
        awk -v own="${PROJECT}" -v deployment="${DEPLOYMENT_PROJECT}" -v prefix="${PROJECT_PREFIX}" \
            '$2 != own && ($2 == deployment || index($2, prefix) == 1) { print $1 " (compose project " $2 ")" }'
}

# Refuses to start a robot side or a dummy simulator while another robot side holds the bus's ports.
refuse_if_ports_taken() {
    command -v docker > /dev/null 2>&1 || return 0
    local others
    others="$(other_robot_containers)"
    [ -z "${others}" ] && return 0
    echo "session.sh: another robot side runs on this host and holds the bus's ports (config/ipc/network.textproto):" >&2
    echo "${others}" | sed 's/^/  /' >&2
    echo "  Stop it first: a deployment with 'make launch-<robot>-robot' ended, 'sudo systemctl stop wb-humanoid-robot@<robot>'" \
        "or 'docker compose --project-directory ~/wb-humanoid-robot down'; another checkout's simulation with its" \
        "'make kill-sims'." >&2
    exit 1
}

# This checkout as the host's Docker sees it, for the robot container's mount: the checkout itself on the host; in a
# container (a dev container with the host's Docker), the source of the mount that holds it.
host_checkout() {
    if [ ! -f /.dockerenv ]; then
        echo "${repo}"
        return 0
    fi
    local source destination
    while read -r source destination; do
        case "${repo}/" in
            "${destination%/}/"*)
                echo "${source%/}${repo#"${destination%/}"}"
                return 0
                ;;
        esac
    done < <(docker inspect --format '{{range .Mounts}}{{.Source}} {{.Destination}}{{"\n"}}{{end}}' "$(hostname)" 2> /dev/null)
    echo "session.sh: cannot tell where the host keeps this checkout (${repo}); set WB_HOST_CHECKOUT to its host path" >&2
    exit 1
}

# Ends whatever a start left: the laptop side in the dev container, the robot container, the NETEM qdisc on lo.
stop() {
    # Only a launcher: the PID of one that ended may since belong to another process.
    tools/deploy/in_dev_container.sh \
        "if [ -f ${PID_FILE} ]; then pid=\$(cat ${PID_FILE}); \
            if grep -q tools/launch/ /proc/\${pid}/cmdline 2> /dev/null; then kill -INT \${pid}; fi; \
            rm -f ${PID_FILE}; fi" || true
    if command -v docker > /dev/null 2>&1; then
        ROBOT="${robot:-none}" compose down --timeout 20 > /dev/null 2>&1 || true
        # The entry point removes its NETEM qdisc when the robot process ends; a container killed before it could
        # leaves it on the host's lo. Removing it is a no-op when there is none, and left to the robot side that runs
        # with it when there is one.
        if [ -z "$(other_robot_containers)" ] && tc qdisc show dev lo 2> /dev/null | grep -q '^qdisc prio 5600: root'; then
            docker run --rm --network host --cap-add NET_ADMIN --entrypoint /opt/wb-humanoid-robot/bin/netem.sh \
                wb-humanoid-robot:latest clear lo > /dev/null 2>&1 || true
        fi
    fi
}

command="${1:-}"
[ -n "${command}" ] || usage
shift
robot=""
launch_file=""
build_targets=""
netem=""
headless=""
skip_build="no"
overrides=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --robot) robot="$2"; shift 2 ;;
        --launch_file) launch_file="$2"; shift 2 ;;
        --build) build_targets="$2"; shift 2 ;;
        --netem) netem="$2"; shift 2 ;;
        --headless) headless="true"; shift ;;
        --set) overrides+=(--set "$2"); shift 2 ;;
        --skip_build) skip_build="yes"; shift ;;
        -h | --help) usage ;;
        *) echo "session.sh: unknown option $1" >&2; usage ;;
    esac
done

# The laptop side, in the foreground. Its PID goes to a file in the checkout so that stop() reaches it inside the dev
# container, where a signal to this script's `docker exec` would not.
run_laptop_side() {
    local arguments=""
    local argument
    for argument in "${overrides[@]+"${overrides[@]}"}"; do
        arguments="${arguments} $(printf '%q' "${argument}")"
    done
    local status=0
    # In the background and waited for: bash runs a trap only once the foreground command has returned, so a SIGTERM
    # or Ctrl-C would otherwise wait for the laptop side to end by itself. The trap's stop() ends it instead.
    tools/deploy/in_dev_container.sh \
        "mkdir -p .deploy && echo \$\$ > ${PID_FILE} && exec ${LAUNCHER} $(printf '%q' "${launch_file}") --machine laptop${arguments}" &
    wait "$!" || status=$?
    tools/deploy/in_dev_container.sh "rm -f ${PID_FILE}" || true
    return "${status}"
}

case "${command}" in
    stop)
        stop
        exit 0
        ;;
    project)
        echo "${PROJECT}"
        exit 0
        ;;
    sim | laptop) ;;
    *) echo "session.sh: unknown command ${command}" >&2; usage ;;
esac
[ -n "${launch_file}" ] || usage

if [ "${command}" = "laptop" ]; then
    # The dummy simulator is the robot node of the shipped localhost network: it binds the robot's port.
    case "${launch_file}" in
        *dummy_sim.textproto) refuse_if_ports_taken ;;
    esac
    if [ "${skip_build}" != "yes" ]; then
        tools/deploy/in_dev_container.sh "bazel build //tools/launch ${build_targets}"
    fi
    trap 'stop' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    run_laptop_side
    exit $?
fi

[ -n "${robot}" ] || usage
# The robot container and the laptop side meet on the host's loopback (config/ipc/network.textproto): the dev
# container must be on the host's network, as docker-compose.yaml puts it (docker-compose.bridge.yaml, for Docker
# Desktop, does not, and the -sim targets do not run there).
if [ ! -f /.dockerenv ] && command -v docker > /dev/null 2>&1; then
    dev_network="$(docker inspect --format '{{.HostConfig.NetworkMode}}' "${WB_DEV_CONTAINER:-devcontainer-app-1}" 2> /dev/null || true)"
    if [ -n "${dev_network}" ] && [ "${dev_network}" != "host" ]; then
        echo "session.sh: the dev container '${WB_DEV_CONTAINER:-devcontainer-app-1}' is on the network '${dev_network}', not" \
            "the host's: the robot container could not reach its MPC node. Recreate it from docker-compose.yaml (host" \
            "network), or use the dummy simulator (make launch-<robot>-dummy-sim)." >&2
        exit 1
    fi
fi
if ! command -v docker > /dev/null 2>&1; then
    echo "session.sh: the robot side runs in a container of its own, which this shell cannot start (no docker here):" \
        "run 'make launch-<robot>-sim' on the host, or give the dev container access to Docker" >&2
    exit 1
fi
if [ -n "${netem}" ]; then
    compose_files+=(-f docker-compose.robot.netem.yaml)
fi

if [ "${skip_build}" != "yes" ]; then
    tools/deploy/deploy_robot.sh image --target robot-sim
    tools/deploy/in_dev_container.sh "bazel build //tools/launch ${build_targets}"
fi

stop
refuse_if_ports_taken
trap 'stop' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export ROBOT="${robot}" NETEM="${netem}" DISPLAY="${DISPLAY:-:99}"
export WB_ROBOT_HEADLESS="${headless}"
# The checkout's robot_models, the seeds of the simulated robot's stored configuration (docker-compose.robot.sim.yaml).
# LINT.IfChange(sim_models)
WB_ROBOT_SIM_MODELS="${WB_HOST_CHECKOUT:-$(host_checkout)}/robot_models"
# LINT.ThenChange(//docker-compose.robot.sim.yaml:sim_models_mount)
export WB_ROBOT_SIM_MODELS
compose up --detach --force-recreate
compose logs --follow --no-log-prefix robot 2>&1 | sed -u 's/^/[robot] /' &
run_laptop_side
