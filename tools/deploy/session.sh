#!/usr/bin/env bash
# The launch sessions of the Makefile's launch targets (humanoid_nmpc/docs/distributed_runtime/README.md, "Deployment"):
#
#   session.sh sim --robot drc_atlas --launch_file <robot>/launch/mpc.textproto --build "<bazel targets>" \
#       [--netem "delay 3ms 1ms loss 0.5%"] [--viewer glfw|web|none] [--plot_config <file>] [--set name=value ...] \
#       [--skip_build]
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
# the bus while it runs, as on the robot. The MuJoCo viewer is the mujoco_viewer of the plot configuration --plot_config
# names (a file of this checkout, mounted read-only, which the laptop side's Rerun bridge reads too; default the shipped
# one, whose viewer is web: a page at http://localhost:6090, which the session prints), unless --viewer names one
# (WB_ROBOT_VIEWER): glfw is the window on DISPLAY, none runs without one, and a list such as web,none falls back in
# order. Without --viewer the file must be a PlotConfig textproto that says so in its header.
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
    sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

# The shipped plot configuration, the Rerun bridge's default too.
# LINT.IfChange(default_plot_config)
DEFAULT_PLOT_CONFIG="humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
# LINT.ThenChange(//Makefile:plot_config_variable, //humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/cli.py:default_plot_config, //docker-compose.robot.sim.yaml:sim_plot_config_mount)
# The launch targets' build, which also refreshes the editor's compilation database (tools/ide/README.md).
BUILD_AND_INDEX="python3 -m tools.ide.compile_commands --after bazel build"

# Whether the file $1 names the message the robot reads it into: the first `# proto-message:` line of its leading
# comment block (the comment lines before the first that is neither blank nor a comment), read as the textproto
# parsers read it.
# LINT.IfChange(plot_config_header)
declares_plot_config() {
    awk '
        /^[[:space:]]*$/ { next }
        !/^[[:space:]]*#/ { exit }
        {
            line = $0
            sub(/^[[:space:]]*#[[:space:]]*/, "", line)
            if (line ~ /^proto-message:/) {
                sub(/^proto-message:[[:space:]]*/, "", line)
                sub(/[[:space:]]*$/, "", line)
                found = line == "humanoid_rerun_viewer_proto.PlotConfig"
                exit
            }
        }
        END { exit !found }
    ' "$1"
}
# LINT.ThenChange(//tools/hooks/textproto_headers.py:header_lines)

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
viewer=""
plot_config="${DEFAULT_PLOT_CONFIG}"
skip_build="no"
overrides=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --robot) robot="$2"; shift 2 ;;
        --launch_file) launch_file="$2"; shift 2 ;;
        --build) build_targets="$2"; shift 2 ;;
        --netem) netem="$2"; shift 2 ;;
        --viewer) viewer="$2"; shift 2 ;;
        --plot_config) plot_config="$2"; shift 2 ;;
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
        tools/deploy/in_dev_container.sh "${BUILD_AND_INDEX} //tools/launch ${build_targets}"
    fi
    trap 'stop' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    run_laptop_side
    exit $?
fi

[ -n "${robot}" ] || usage
# The plot configuration of both sides: its mujoco_viewer names the robot's viewer without --viewer, and the Rerun
# bridge plots what it lists. It is checked before anything is built or stopped: a file of the checkout, by a path
# relative to it, which the robot container mounts and the laptop side reads in the dev container's checkout. The
# bridge's "none" (no plots) leaves the robot the shipped one's viewer.
case "$(printf '%s' "${plot_config}" | tr 'A-Z' 'a-z')" in
    "" | none | off | false | 0) robot_plot_config="${DEFAULT_PLOT_CONFIG}" ;;
    *)
        case "${plot_config}" in
            "${repo}"/*) plot_config="${plot_config#"${repo}"/}" ;;
            /*)
                echo "session.sh: --plot_config ${plot_config} is outside the checkout ${repo}: the robot and the dev" \
                    "container see only the checkout, so copy the file into it" >&2
                exit 2
                ;;
        esac
        robot_plot_config="${plot_config}"
        ;;
esac
if [ ! -f "${robot_plot_config}" ]; then
    echo "session.sh: --plot_config ${robot_plot_config}: no such file in the checkout ${repo}" >&2
    exit 2
fi
# The robot reads the file only without --viewer, and strictly, into a PlotConfig: a plain list of signals, which the
# bridge reads too, would end its start, after the build and the stop of the running simulation.
if [ -z "${viewer}" ] && ! declares_plot_config "${robot_plot_config}"; then
    echo "session.sh: --plot_config ${robot_plot_config} does not declare '# proto-message:" \
        "humanoid_rerun_viewer_proto.PlotConfig' in its leading comment block: without --viewer the robot reads its" \
        "MuJoCo viewer (mujoco_viewer) from it, which only a PlotConfig textproto names. Give it the header of" \
        "${DEFAULT_PLOT_CONFIG}, or name the viewer: make ... VIEWER=web|glfw|none" >&2
    exit 2
fi
# The laptop side reads the same file, by the same checkout-relative path: a later --set wins (tools/launch), over the
# Makefile's of PLOT_CONFIG, whose absolute path of the host the dev container would not find.
overrides+=(--set "plot_config=${plot_config}")
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
    tools/deploy/in_dev_container.sh "${BUILD_AND_INDEX} //tools/launch ${build_targets}"
fi

stop
refuse_if_ports_taken
trap 'stop' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export ROBOT="${robot}" NETEM="${netem}" DISPLAY="${DISPLAY:-:99}"
export WB_ROBOT_VIEWER="${viewer}"
# The checkout's robot_models, the seeds of the simulated robot's stored configuration (docker-compose.robot.sim.yaml).
# LINT.IfChange(sim_models)
WB_ROBOT_SIM_MODELS="${WB_HOST_CHECKOUT:-$(host_checkout)}/robot_models"
# LINT.ThenChange(//docker-compose.robot.sim.yaml:sim_models_mount)
export WB_ROBOT_SIM_MODELS
# The checkout's plot configuration, mounted read-only where the robot container reads it (docker-compose.robot.sim.yaml).
# LINT.IfChange(sim_plot_config)
WB_ROBOT_SIM_PLOT_CONFIG="${WB_HOST_CHECKOUT:-$(host_checkout)}/${robot_plot_config}"
sim_plot_config_target="/etc/wb-humanoid-robot/plot_config.textproto"
# LINT.ThenChange(//docker-compose.robot.sim.yaml:sim_plot_config_mount)
export WB_ROBOT_SIM_PLOT_CONFIG
# The robot's errors about the file name it by the path the container sees.
[ -n "${viewer}" ] ||
    echo "session.sh: the robot reads its MuJoCo viewer from ${robot_plot_config}, which it sees as ${sim_plot_config_target}"
compose up --detach --force-recreate
# Whether the robot starts the web viewer, for the address below: --viewer lists it, or no --viewer is given and the plot
# configuration's mujoco_viewer lists it or nothing (the default, web). A hint only: the robot process reads the file
# strictly and logs the viewer it chose.
# LINT.IfChange(default_viewer_hint)
starts_web_viewer() {
    case ",${viewer}," in
        ,,) ;;
        *,web,*) return 0 ;;
        *) return 1 ;;
    esac
    grep -Eq '^[[:space:]]*mujoco_viewer:.*"web"' "${robot_plot_config}" ||
        ! grep -Eq '^[[:space:]]*mujoco_viewer:' "${robot_plot_config}"
}
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotAppOptions.h:default_viewer)
if starts_web_viewer; then
    # The robot container is on the host's network, and the viewer binds its loopback (--web_viewer_address).
    # LINT.IfChange(web_viewer_port)
    web_viewer_port="${WB_ROBOT_WEB_VIEWER_PORT:-6090}"
    # LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotAppFlags.cpp:web_viewer_port)
    chosen_by=""
    [ -n "${viewer}" ] || chosen_by=" (the mujoco_viewer of ${robot_plot_config}; VIEWER=glfw is the window on DISPLAY)"
    echo "session.sh: the MuJoCo web viewer${chosen_by}: http://localhost:${web_viewer_port} (from another machine:" \
        "ssh -L ${web_viewer_port}:127.0.0.1:${web_viewer_port} <this host>)"
fi
compose logs --follow --no-log-prefix robot 2>&1 | sed -u 's/^/[robot] /' &
run_laptop_side
