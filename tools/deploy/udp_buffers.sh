#!/usr/bin/env bash
# Raises a machine's kernel limits on UDP socket buffers to what a network file asks the DDS bus for, and keeps them
# across reboots:
#
#   udp_buffers.sh --network config/ipc/two_machine.example.textproto [--host <ssh host>|localhost]
#
# A network file's `transport { socket_receive_buffer_bytes: N }` (and socket_send_buffer_bytes) is a request that the
# kernel caps at net.core.rmem_max (net.core.wmem_max), which is not namespaced: it is the host's limit, and no
# container can raise it. Cyclone DDS takes the smaller buffer when the kernel caps it and the bus still works, with
# more retransmissions of large policies, so this script never fails a launch: where it cannot raise a limit (inside a
# container, or sudo refused) it says how to and exits 0.
#
# A limit is only ever raised, never lowered, and only when it is below the request. Raising it runs
# `sudo sysctl -w` (sudo asks for the password once; the Makefile's launch and deploy targets run this script first)
# and writes /etc/sysctl.d/60-wb-humanoid-<key>.conf, so that the next boot keeps it. --host <ssh host> does the same
# on that machine over ssh (`ssh -t`, for sudo's prompt).
#
# Environment (for tests): WB_SYSCTL (default sysctl), WB_SUDO (default sudo), WB_SYSCTL_DIR (default /etc/sysctl.d),
# WB_DOCKERENV (default /.dockerenv, the file every Docker container has).
set -euo pipefail

sysctl_command="${WB_SYSCTL:-sysctl}"
sudo_command="${WB_SUDO:-sudo}"
sysctl_dir="${WB_SYSCTL_DIR:-/etc/sysctl.d}"
dockerenv="${WB_DOCKERENV:-/.dockerenv}"

usage() {
    echo "usage: udp_buffers.sh --network <network file> [--host <ssh host>|localhost]" >&2
    exit 2
}

network=""
host="localhost"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --network) [ "$#" -ge 2 ] || usage; network="$2"; shift 2 ;;
        --host) [ "$#" -ge 2 ] || usage; host="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[ -n "${network}" ] || usage
if [ ! -f "${network}" ]; then
    echo "udp_buffers.sh: --network ${network}: no such file" >&2
    exit 2
fi

# The value of a numeric field of the network file, 0 when it is absent: comments are dropped first, and the last
# occurrence counts, as for any field set twice in a textproto.
field_value() {
    local value
    value="$(sed 's/#.*//' "${network}" | grep -oE "\\b$1[[:space:]]*:[[:space:]]*[0-9]+" | grep -oE '[0-9]+$' | tail -n 1 || true)"
    echo "${value:-0}"
}

# Runs a shell command on the host: here for localhost, else over ssh.
on_host() {
    if [ "${host}" = "localhost" ]; then
        bash -c "$1"
    else
        ssh -t "${host}" "$1"
    fi
}

# Raises the kernel limit <key> to <bytes> on the host when it is below, and keeps it in sysctl.d.
ensure_limit() {
    local key="$1" bytes="$2" field="$3" current
    [ "${bytes}" -gt 0 ] || return 0
    if ! current="$(on_host "${sysctl_command} -n ${key}")"; then
        echo "⚠️  udp_buffers.sh: cannot read ${key} on ${host}; the bus works with the kernel's limit." >&2
        return 0
    fi
    current="$(echo "${current}" | tr -d '[:space:]')"
    if [ "${current}" -ge "${bytes}" ]; then
        echo "✅ ${key} on ${host} is ${current} B, at least the ${bytes} B of ${field} in ${network}."
        return 0
    fi
    local hint="sudo sysctl -w ${key}=${bytes}, and keep it in ${sysctl_dir}/60-wb-humanoid-${key}.conf"
    if [ "${host}" = "localhost" ] && [ -f "${dockerenv}" ]; then
        echo "⚠️  ${key} is ${current} B here, below the ${bytes} B of ${field} in ${network}, and a container cannot" \
            "raise the host's limit. On the host run: ${hint} (or make udp-buffers there)." >&2
        return 0
    fi
    echo "🔧 ${key} on ${host} is ${current} B, below the ${bytes} B of ${field} in ${network}: raising it (sudo)."
    local conf="${sysctl_dir}/60-wb-humanoid-${key}.conf"
    if on_host "${sudo_command} ${sysctl_command} -w ${key}=${bytes} > /dev/null &&
            printf '# Set by tools/deploy/udp_buffers.sh for the DDS bus (${field} of the network file).\n%s = %s\n' \
                '${key}' '${bytes}' | ${sudo_command} tee '${conf}' > /dev/null"; then
        echo "✅ ${key} on ${host} is now ${bytes} B, kept in ${conf}."
    else
        echo "⚠️  udp_buffers.sh: could not raise ${key} on ${host}; the bus works with ${current} B. Run: ${hint}." >&2
    fi
}

# LINT.IfChange(socket_buffers)
ensure_limit net.core.rmem_max "$(field_value socket_receive_buffer_bytes)" socket_receive_buffer_bytes
ensure_limit net.core.wmem_max "$(field_value socket_send_buffer_bytes)" socket_send_buffer_bytes
# LINT.ThenChange(//robot_runtime/robot_ipc/proto/network_config.proto:socket_buffers)
