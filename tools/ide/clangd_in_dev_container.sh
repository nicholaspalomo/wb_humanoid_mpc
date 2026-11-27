#!/bin/sh
# Runs clangd in the dev container for the editor: set the editor's `clangd.path` to this script (tools/ide/README.md).
#
# The build, Bazel's output base and the compile_commands.json that describes it are inside the dev container, so clangd
# runs there. From the host (an editor on the host, or connected to it over SSH) it runs through `docker exec` and speaks
# the language server protocol over its standard input and output; --path-mappings translates the checkout's path on
# the host into its path in the container and back (docker-compose.yaml mounts the checkout's grandparent at
# /wb_humanoid_mpc_ws). clangd exits when the editor closes the pipe. In the container (an editor attached to it)
# clangd runs directly.
#
# Environment:
#   WB_DEV_CONTAINER       the dev container (default devcontainer-app-1, the name docker-compose.yaml gives it)
#   WB_DEV_CONTAINER_REPO  this checkout's path inside it (default /wb_humanoid_mpc_ws/<parent>/<checkout>)
#   WB_DOCKERENV           the file that marks a container (default /.dockerenv; the tests name another)
#
# clangd is the image's, of the build of its clang-tidy (docker/install_llvm_tools.sh), under its versioned name
# (pick_and_run below). In a container of an image without it, the clangd that the editor's clangd extension downloaded
# on the host runs instead, through the same mount.
#
# Memory: clangd parses translation units outside Bazel's RAM budget (AGENTS.md, "Builds share one machine's memory"),
# so its background index runs two workers at a low priority and keeps preambles on disk.
set -eu

repo="$(cd "$(dirname "$0")/../.." && pwd)"
mount_source="$(dirname "$(dirname "${repo}")")"
# LINT.IfChange(dev_container_mount)
mount_target="/wb_humanoid_mpc_ws"
container="${WB_DEV_CONTAINER:-devcontainer-app-1}"
container_repo="${WB_DEV_CONTAINER_REPO:-${mount_target}/$(basename "$(dirname "${repo}")")/$(basename "${repo}")}"
# LINT.ThenChange(//docker-compose.yaml:dev_container_mount, //tools/deploy/in_dev_container.sh:dev_container_mount)

in_container="no"
if [ -f "${WB_DOCKERENV:-/.dockerenv}" ]; then
    in_container="yes"
    container_repo="${repo}"
fi

# LINT.IfChange(clangd_flags)
set -- \
    --compile-commands-dir="${container_repo}" \
    --query-driver=/usr/bin/gcc,/usr/bin/g++,/usr/bin/c++ \
    --background-index \
    --background-index-priority=low \
    -j=2 \
    --pch-storage=disk \
    --header-insertion=never \
    "$@"
# LINT.ThenChange(//tools/ide/README.md:clangd_flags, //AGENTS.md:clangd_memory)

# Run by the container's shell: the image's clangd, else $1 when it is executable, with the remaining arguments.
# LINT.IfChange(clangd_version)
# shellcheck disable=SC2016  # Expanded by that shell.
pick_and_run='
    clangd="$(command -v clangd-21 || command -v clangd || true)"
    if [ -z "${clangd}" ] && [ -n "$1" ] && [ -x "$1" ]; then clangd="$1"; fi
    if [ -z "${clangd}" ]; then
        echo "clangd_in_dev_container.sh: the dev container has no clangd-21: rebuild it from docker/Dockerfile" >&2
        exit 1
    fi
    shift
    exec "${clangd}" "$@"'
# LINT.ThenChange(//docker/Dockerfile:clang_tidy_version, //docker/install_llvm_tools.sh:llvm_versions, //tools/ide/README.md:clangd_version)

if [ "${in_container}" = "yes" ]; then
    exec sh -c "${pick_and_run}" clangd "" "$@"
fi

if ! command -v docker > /dev/null 2>&1; then
    echo "clangd_in_dev_container.sh: no docker here to reach the dev container '${container}'" >&2
    exit 1
fi
if [ "$(docker inspect --format '{{.State.Running}}' "${container}" 2> /dev/null)" != "true" ]; then
    echo "clangd_in_dev_container.sh: the dev container '${container}' is not running. Start it (docker compose up -d)," \
        "or name another with WB_DEV_CONTAINER=<name>." >&2
    exit 1
fi

# The newest clangd an editor's clangd extension installed on the host, as the container sees it through the mount.
fallback=""
for candidate in \
    "${HOME}"/.*-server/data/User/globalStorage/llvm-vs-code-extensions.vscode-clangd/install/*/clangd_*/bin/clangd \
    "${HOME}"/.config/*/User/globalStorage/llvm-vs-code-extensions.vscode-clangd/install/*/clangd_*/bin/clangd; do
    case "${candidate}" in
        "${mount_source}"/*) [ -x "${candidate}" ] && fallback="${mount_target}${candidate#"${mount_source}"}" ;;
    esac
done

exec docker exec --interactive --workdir "${container_repo}" "${container}" sh -c "${pick_and_run}" clangd "${fallback}" \
    --path-mappings="${repo}=${container_repo}" "$@"
