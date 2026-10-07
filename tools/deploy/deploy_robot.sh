#!/usr/bin/env bash
# Builds the robot images and deploys the robot side to the robot's computer (tools/deploy/README.md).
#
#   deploy_robot.sh bundle                                 # bazel build //tools/deploy:robot_bundle (dev container)
#   deploy_robot.sh image [--target robot-runtime|robot-sim] [--platform linux/arm64]
#   deploy_robot.sh deploy --robot drc_atlas --host <ssh host>|localhost [--network <file>] [--service enable] ...
#   deploy_robot.sh up --robot drc_atlas --host <host>   # the deployed robot in the foreground; Ctrl-C stops it
#   deploy_robot.sh down --host <host>
#   deploy_robot.sh environment --robot drc_atlas ...     # print the .env a deploy writes
#
# `deploy` builds the bundle in the dev container (in_dev_container.sh) and the robot-runtime image with the host's
# Docker, for the host's architecture, ships the image (`docker save | ssh <host> docker load`, or through --registry),
# and installs the deployment directory on the host: docker-compose.robot.yaml (and the NETEM override), the network
# file, a .env with the settings below, and the systemd unit, which --service installs (`install`) and enables at boot
# (`enable`, which also disables the unit of any other robot on the host: one robot per machine). A deployment that is
# running already is restarted on the new image and settings. --host localhost deploys to this machine without ssh.
# --dry_run prints every command instead of running it.
#
# The Makefile's deploy-robot, launch-<robot>-robot and launch-<robot>-sim targets run it.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo}"

# LINT.IfChange(image_names)
RUNTIME_IMAGE="wb-humanoid-robot"
SIM_IMAGE="wb-humanoid-robot-sim"
# LINT.ThenChange(//docker-compose.robot.yaml, //docker-compose.robot.sim.yaml, //tools/deploy/README.md)
# The robot configurations, the names of the bundle's launch/<robot>.sh scripts.
# LINT.IfChange(robot_names)
ROBOTS="drc_atlas engineai_sa01 unitree_g1 unitree_g1_wb unitree_r1"
# LINT.ThenChange(//tools/deploy/BUILD.bazel:robot_configurations, //tools/deploy/test_deploy_files.py:robot_names)
BUNDLE_TARGET="//tools/deploy:robot_bundle"
BUNDLE_OUTPUT=".bazel/bin/tools/deploy/robot_bundle.tar"
CONTEXT_DIR=".deploy/context"
UNIT="wb-humanoid-robot@.service"
SERVICES="none install enable"
TARGETS="robot-runtime robot-sim"
# LINT.IfChange(config_seed_policies)
CONFIG_SEEDS="when_bundle_changes every_start never"
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/include/humanoid_common_mpc_app/robot/RobotConfigDirectory.h:config_seed_policies)
# The robot's persistent configuration in the deployment directory, which compose binds into the container
# (WB_ROBOT_CONFIG_SOURCE, docker-compose.robot.yaml); a deploy creates it and never empties it.
# LINT.IfChange(robot_config_dir)
ROBOT_CONFIG_DIR="robot_config"
# LINT.ThenChange(//docker-compose.robot.yaml:robot_config_store, //.gitignore)

usage() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    cat >&2 << 'EOF'

Options:
  --robot NAME              the robot configuration (deploy): drc_atlas, engineai_sa01, unitree_g1, unitree_g1_wb,
                            unitree_r1
  --host HOST               the robot's computer as ssh reaches it, or localhost (default)
  --network FILE            the network file to deploy (default: config/ipc/network.textproto, on localhost only)
  --target TARGET           image: robot-runtime (default) or robot-sim, which builds both
  --platform PLATFORM       the image's platform, e.g. linux/arm64 (default: the host's Docker's, asked over ssh, or
                            this machine's for localhost); the bundle must match, and so must the host
  --tag TAG                 the images' tag (default latest)
  --registry REGISTRY       push the image there and pull it on the host, instead of docker save | ssh docker load
  --deploy_dir DIR          the deployment directory on the host, relative to its home (default wb-humanoid-robot)
  --service SERVICE         the systemd unit: none (default), install, enable (install, enable at boot, start)
  --cpuset CPUS             the cores the container may use, e.g. 2-5 (default: all)
  --memory_limit SIZE       the container's memory cap (default 4g)
  --realtime_priority N     SCHED_FIFO priority of the realtime thread (default: the launch file's, 80)
  --realtime_cores LIST     cores of the realtime thread (default: the launch file's)
  --backend_cores LIST      cores of the backend's threads (default: the launch file's)
  --backend NAME            the robot backend (default: the launch file's, mujoco)
  --config_seed POLICY      when the robot's stored configuration is replaced by the deployed files: when_bundle_changes
                            (default: a deploy that changed a file), every_start or never
  --netem PARAMS            tc-netem parameters for the bus's packets, e.g. "delay 3ms 1ms loss 0.5%" (default: none)
  --netem_interface NAME    the interface NETEM shapes (default lo)
  --dev_container NAME      the dev container Bazel runs in from the host (default devcontainer-app-1)
  --skip_bundle             image/deploy: use the bundle already in .deploy/context
  --dry_run                 print the commands instead of running them
EOF
    exit 2
}

fail() {
    echo "deploy_robot.sh: $*" >&2
    exit 1
}

# Runs a command, or prints it with --dry_run.
run() {
    if [ "${dry_run}" = "yes" ]; then
        local word line="+"
        for word in "$@"; do
            if [[ "${word}" =~ ^[A-Za-z0-9_./:@=+,%-]+$ ]]; then
                line="${line} ${word}"
            else
                line="${line} ${word@Q}"
            fi
        done
        echo "${line}"
    else
        "$@"
    fi
}

# Runs a shell command on the host: here for localhost, else over ssh.
on_host() {
    if [ "${host}" = "localhost" ]; then
        run bash -c "$1"
    else
        run ssh "${ssh_flags[@]}" "${host}" "$1"
    fi
}

copy_to_host() {
    local source="$1" destination="$2"
    if [ "${host}" = "localhost" ]; then
        run cp "${source}" "${destination}"
    else
        run scp -q "${source}" "${host}:${destination}"
    fi
}

one_of() {
    local value="$1" choices="$2" what="$3"
    case " ${choices} " in
        *" ${value} "*) ;;
        *) fail "unknown ${what} '${value}'; the ${what}s are: ${choices}" ;;
    esac
}

command="${1:-}"
[ -n "${command}" ] || usage
shift
robot=""
host="localhost"
network=""
target="robot-runtime"
platform=""
tag="latest"
registry=""
deploy_dir="wb-humanoid-robot"
service="none"
cpuset=""
memory_limit=""
realtime_priority=""
realtime_cores=""
backend_cores=""
backend=""
config_seed=""
netem=""
netem_interface="lo"
skip_bundle="no"
dry_run="no"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --robot) robot="$2"; shift 2 ;;
        --host) host="$2"; shift 2 ;;
        --network) network="$2"; shift 2 ;;
        --target) target="$2"; shift 2 ;;
        --platform) platform="$2"; shift 2 ;;
        --tag) tag="$2"; shift 2 ;;
        --registry) registry="$2"; shift 2 ;;
        --deploy_dir) deploy_dir="$2"; shift 2 ;;
        --service) service="$2"; shift 2 ;;
        --cpuset) cpuset="$2"; shift 2 ;;
        --memory_limit) memory_limit="$2"; shift 2 ;;
        --realtime_priority) realtime_priority="$2"; shift 2 ;;
        --realtime_cores) realtime_cores="$2"; shift 2 ;;
        --backend_cores) backend_cores="$2"; shift 2 ;;
        --backend) backend="$2"; shift 2 ;;
        --config_seed) config_seed="$2"; shift 2 ;;
        --netem) netem="$2"; shift 2 ;;
        --netem_interface) netem_interface="$2"; shift 2 ;;
        --dev_container) export WB_DEV_CONTAINER="$2"; shift 2 ;;
        --skip_bundle) skip_bundle="yes"; shift ;;
        --dry_run) dry_run="yes"; shift ;;
        -h | --help) usage ;;
        *) echo "deploy_robot.sh: unknown option $1" >&2; usage ;;
    esac
done
[ -n "${host}" ] || host="localhost"
one_of "${target}" "${TARGETS}" "target"
one_of "${service}" "${SERVICES}" "service"
[ -z "${config_seed}" ] || one_of "${config_seed}" "${CONFIG_SEEDS}" "seed policy"
# A tty for sudo's password prompt and for Ctrl-C of `up`.
ssh_flags=()
if [ -t 0 ]; then
    ssh_flags=(-t)
fi
image_name() {
    local name="$1"
    if [ -n "${registry}" ]; then
        echo "${registry%/}/${name}:${tag}"
    else
        echo "${name}:${tag}"
    fi
}

# The architecture of the bundle's robot binary, from its ELF header (e_machine): amd64 or arm64.
bundle_architecture() {
    local binary="bin/humanoid_nmpc/humanoid_centroidal_mpc_app/humanoid_centroidal_mpc_robot"
    local machine
    machine="$(tar -xOf "${CONTEXT_DIR}/robot_bundle.tar" "${binary}" | head -c 20 | od -An -tx1 -j18 -N2 | tr -d ' \n')"
    case "${machine}" in
        3e00) echo amd64 ;;
        b700) echo arm64 ;;
        *) echo "unknown (e_machine ${machine})" ;;
    esac
}

build_bundle() {
    run tools/deploy/in_dev_container.sh \
        "bazel build ${BUNDLE_TARGET} && mkdir -p ${CONTEXT_DIR} && cp -fL ${BUNDLE_OUTPUT} ${CONTEXT_DIR}/robot_bundle.tar"
}

build_image() {
    command -v docker > /dev/null 2>&1 || [ "${dry_run}" = "yes" ] ||
        fail "the robot images are built with Docker, which this shell has no access to: run it on the host"
    [ "${skip_bundle}" = "yes" ] || build_bundle
    if [ "${dry_run}" != "yes" ]; then
        [ -f "${CONTEXT_DIR}/robot_bundle.tar" ] || fail "no bundle in ${CONTEXT_DIR}: run without --skip_bundle"
        local wanted="${platform#linux/}"
        [ -n "${wanted}" ] || wanted="$(docker version --format '{{.Server.Arch}}')"
        local built
        built="$(bundle_architecture)"
        [ "${built}" = "${wanted}" ] ||
            fail "the bundle is built for ${built}, the image for ${wanted}: build the bundle in a dev container of" \
                "that architecture (tools/deploy/README.md, \"aarch64\")"
    fi
    local revision
    revision="$(git rev-parse HEAD 2> /dev/null || echo unknown)"
    local flags=(-f docker/Dockerfile --build-context "bundle=${CONTEXT_DIR}"
        --label "org.opencontainers.image.revision=${revision}"
        --label "org.opencontainers.image.source=wb_humanoid_mpc")
    if [ -n "${platform}" ]; then
        flags+=(--platform "${platform}")
    fi
    run docker build "${flags[@]}" --target robot-runtime -t "$(image_name "${RUNTIME_IMAGE}")" .
    if [ "${target}" = "robot-sim" ]; then
        run docker build "${flags[@]}" --target robot-sim -t "$(image_name "${SIM_IMAGE}")" .
    fi
}

ship_image() {
    local image
    image="$(image_name "${RUNTIME_IMAGE}")"
    if [ -n "${registry}" ]; then
        run docker push "${image}"
        on_host "docker pull '${image}'"
    elif [ "${host}" != "localhost" ]; then
        if [ "${dry_run}" = "yes" ]; then
            echo "+ docker save ${image} | gzip -1 | ssh ${host} 'gunzip | docker load'"
        else
            docker save "${image}" | gzip -1 | ssh "${host}" 'gunzip | docker load'
        fi
    fi
}

# The settings of the deployment, which docker compose reads from the .env next to docker-compose.robot.yaml.
print_environment() {
    local compose_files="docker-compose.robot.yaml"
    if [ -n "${netem}" ]; then
        compose_files="${compose_files}:docker-compose.robot.netem.yaml"
    fi
    {
        echo "# The robot deployment of ROBOT=${robot}, written by tools/deploy/deploy_robot.sh; docker compose reads it."
        echo "COMPOSE_PROJECT_NAME=wb-humanoid-robot"
        echo "COMPOSE_FILE=${compose_files}"
        echo "ROBOT=${robot}"
        echo "WB_ROBOT_IMAGE=$(image_name "${RUNTIME_IMAGE}")"
        echo "WB_ROBOT_NETWORK_SOURCE=./network.textproto"
        echo "WB_ROBOT_CPUSET=${cpuset}"
        [ -z "${memory_limit}" ] || echo "WB_ROBOT_MEMORY_LIMIT=${memory_limit}"
        # LINT.IfChange(robot_environment)
        [ -z "${backend}" ] || echo "WB_ROBOT_BACKEND=${backend}"
        [ -z "${realtime_priority}" ] || echo "WB_ROBOT_REALTIME_PRIORITY=${realtime_priority}"
        [ -z "${realtime_cores}" ] || echo "WB_ROBOT_REALTIME_CORES=${realtime_cores}"
        [ -z "${backend_cores}" ] || echo "WB_ROBOT_BACKEND_CORES=${backend_cores}"
        [ -z "${config_seed}" ] || echo "WB_ROBOT_CONFIG_SEED=${config_seed}"
        # LINT.ThenChange(//docker-compose.robot.yaml:robot_environment)
        echo "NETEM=\"${netem}\""
        echo "NETEM_INTERFACE=${netem_interface}"
    }
}

check_robot() {
    [ -n "${robot}" ] || fail "${command} needs --robot"
    one_of "${robot}" "${ROBOTS}" "robot configuration"
}

# The architecture of the host's Docker (amd64, arm64), which the image is built for: an image of another one would only
# fail when its container starts, with an exec format error, and restart forever.
check_host_architecture() {
    [ "${host}" != "localhost" ] || return 0
    local remote
    if [ "${dry_run}" = "yes" ]; then
        echo "+ ssh ${host} docker version --format '{{.Server.Arch}}'"
        return 0
    fi
    remote="$(ssh "${host}" "docker version --format '{{.Server.Arch}}'")" ||
        fail "could not ask the Docker of ${host} for its architecture: is Docker installed there, and may this user run it?"
    if [ -z "${platform}" ]; then
        platform="linux/${remote}"
    elif [ "${platform#linux/}" != "${remote}" ]; then
        fail "--platform ${platform}, but the Docker of ${host} runs ${remote}: the robot could not start the image"
    fi
}

deploy() {
    check_robot
    if [ -z "${network}" ]; then
        [ "${host}" = "localhost" ] ||
            fail "a robot on another machine needs --network <file> with both machines' addresses" \
                "(config/ipc/two_machine.example.textproto)"
        network="config/ipc/network.textproto"
    fi
    [ -f "${network}" ] || fail "--network ${network}: no such file"
    target="robot-runtime"
    check_host_architecture
    build_image
    ship_image

    staging="$(mktemp -d)"
    trap 'rm -rf "${staging}"' EXIT
    print_environment > "${staging}/.env"
    local remote_dir="${deploy_dir}"
    if [ "${host}" = "localhost" ] && [ "${deploy_dir#/}" = "${deploy_dir}" ]; then
        remote_dir="${HOME}/${deploy_dir}"
    fi
    # The robot's stored configuration lives in the deployment directory, created here by the deploy user (compose
    # would create a missing one as root); the robot seeds it from the image's bundle when it starts.
    on_host "mkdir -p '${remote_dir}' '${remote_dir}/${ROBOT_CONFIG_DIR}'"
    copy_to_host docker-compose.robot.yaml "${remote_dir}/docker-compose.robot.yaml"
    copy_to_host docker-compose.robot.netem.yaml "${remote_dir}/docker-compose.robot.netem.yaml"
    copy_to_host "${network}" "${remote_dir}/network.textproto"
    copy_to_host "${staging}/.env" "${remote_dir}/.env"
    copy_to_host "tools/deploy/${UNIT}" "${remote_dir}/${UNIT}"
    # The unit names the directory absolutely; the host's home decides what that is.
    on_host "cd '${remote_dir}' && sed -i \"s|@DEPLOY_DIR@|\$(pwd)|\" '${UNIT}'"
    if [ "${service}" != "none" ]; then
        on_host "sudo install -m 0644 '${remote_dir}/${UNIT}' '/etc/systemd/system/${UNIT}' && sudo systemctl daemon-reload"
    fi
    local unit="wb-humanoid-robot@${robot}.service"
    if [ "${service}" = "enable" ]; then
        # One robot per machine: the deployment directory and the compose project are shared, so the unit of another
        # robot deployed here before would recreate the container as its own at the next boot. Its unit goes off.
        on_host "for other in \$(ls /etc/systemd/system/multi-user.target.wants/ 2> /dev/null | grep '^wb-humanoid-robot@.*\.service\$'; \
            systemctl list-units --all --plain --no-legend 'wb-humanoid-robot@*.service' | awk '{ print \$1 }'); do \
            [ \"\${other}\" = '${unit}' ] || sudo systemctl disable --now \"\${other}\"; done"
        # `enable --now` would start nothing when the unit is active (it is oneshot, RemainAfterExit): a restart takes
        # the container down and up again, on the image and the .env just installed.
        on_host "sudo systemctl enable '${unit}' && sudo systemctl restart '${unit}'"
        echo "deploy_robot.sh: ROBOT=${robot} is deployed to ${host}:${remote_dir} ($(image_name "${RUNTIME_IMAGE}")), and runs" \
            "now and at every boot: sudo systemctl status ${unit}"
        return 0
    fi
    # A deployment that runs already takes the new image and settings now: compose recreates its container when either
    # changed. One that does not run is left so.
    on_host "cd '${remote_dir}' && if [ -n \"\$(docker compose ps --quiet robot 2> /dev/null)\" ]; then \
        docker compose up --detach --no-build --remove-orphans && echo 'deploy_robot.sh: the running robot was restarted on the new deployment.'; \
        else echo 'deploy_robot.sh: the robot is not running.'; fi"
    echo "deploy_robot.sh: ROBOT=${robot} is deployed to ${host}:${remote_dir} ($(image_name "${RUNTIME_IMAGE}"))."
    echo "  Start it: make launch-<robot>-robot HOST=${host}, or on the host: cd ${remote_dir} && docker compose up"
}

compose_on_host() {
    local remote_dir="${deploy_dir}"
    if [ "${host}" = "localhost" ] && [ "${deploy_dir#/}" = "${deploy_dir}" ]; then
        remote_dir="${HOME}/${deploy_dir}"
    fi
    on_host "cd '${remote_dir}' && ROBOT='${robot}' docker compose $1"
}

case "${command}" in
    bundle) build_bundle ;;
    image) build_image ;;
    deploy) deploy ;;
    environment) check_robot; print_environment ;;
    up) check_robot; compose_on_host "up" ;;
    down) robot="${robot:-none}"; compose_on_host "down" ;;
    *) echo "deploy_robot.sh: unknown command ${command}" >&2; usage ;;
esac
