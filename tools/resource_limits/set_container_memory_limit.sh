#!/usr/bin/env bash
# Caps the dev container's memory, and forbids it swap beyond the cap, by writing WB_CONTAINER_MEMORY_LIMIT into the
# repository's .env, which `docker compose` reads when it interpolates docker-compose.yaml. Run on the HOST before the
# container starts: devcontainer.json runs it as its initializeCommand, and the README's `docker compose up` path runs
# it first.
#
# Why: an unbounded build (twelve CppAD / Pinocchio translation units at 2-3 GB each, next to the simulator, the 3D
# visualizer and the tests' own CppAD code generation) once pushed a 32 GB workstation into swap until it had to be
# power-cycled. With a cap and no swap beyond it, the kernel's OOM killer ends the largest process INSIDE the container - a compiler, a
# test - within seconds, and the host stays usable. .bazelrc bounds Bazel's parallelism by RAM so the cap is not hit in
# the first place; this is the backstop for everything Bazel does not schedule.
#
#   WB_CONTAINER_MEMORY_LIMIT    use this value (docker syntax, e.g. 24g) instead of computing one
#   WB_CONTAINER_MEMORY_PERCENT  share of the host's RAM to allow when computing (default 85)
#   WB_MEMINFO, WB_ENV_FILE      override /proc/meminfo and the .env path (tests)
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly env_file="${WB_ENV_FILE:-${repo_root}/.env}"
readonly meminfo="${WB_MEMINFO:-/proc/meminfo}"
readonly percent="${WB_CONTAINER_MEMORY_PERCENT:-85}"

if [[ -n "${WB_CONTAINER_MEMORY_LIMIT:-}" ]]; then
  limit="${WB_CONTAINER_MEMORY_LIMIT}"
elif [[ -r "${meminfo}" ]]; then
  total_kb="$(awk '/^MemTotal:/ {print $2}' "${meminfo}")"
  if [[ -z "${total_kb}" ]]; then
    echo "set_container_memory_limit: no MemTotal in ${meminfo}; leaving the container's memory unlimited." >&2
    exit 0
  fi
  limit="$((total_kb * percent / 100 / 1024))m"
else
  # macOS and Windows hosts: Docker Desktop's VM already bounds the container's memory.
  echo "set_container_memory_limit: ${meminfo} is not readable; leaving the container's memory unlimited." >&2
  exit 0
fi

# Replace the variable's line, keep every other line of the file.
touch "${env_file}"
tmp="$(mktemp)"
grep -v '^WB_CONTAINER_MEMORY_LIMIT=' "${env_file}" > "${tmp}" || true
echo "WB_CONTAINER_MEMORY_LIMIT=${limit}" >> "${tmp}"
mv "${tmp}" "${env_file}"
echo "set_container_memory_limit: WB_CONTAINER_MEMORY_LIMIT=${limit} written to ${env_file}"
