#!/bin/sh
# Delays and drops the ZeroMQ bus's packets with tc-netem, to test the remote MPC link over a link like the robot's:
#
#   netem.sh apply <interface> <netem parameters...>   # e.g. netem.sh apply lo delay 3ms 1ms loss 0.5%
#   netem.sh clear <interface>                         # remove it again (only what `apply` installed)
#   netem.sh show <interface>
#
# Only TCP packets (IPv4 or IPv6) from or to the bus's ports (5600-5631, which holds every node of
# config/ipc/network.textproto) are affected: `apply` replaces the interface's root qdisc with a prio qdisc whose fourth
# band is netem, and u32 filters on the TCP protocol and the ports send the bus's packets to that band; everything else keeps the prio qdisc's default bands. `clear` deletes that root
# qdisc, and the kernel puts the interface's default back. On `lo` both directions of every connection pass the netem
# band; on a network interface only the packets this machine sends do (apply it on both machines, or use an ifb for
# the incoming side). Needs CAP_NET_ADMIN; the robot images run it from robot_entrypoint.sh when NETEM is set.
#
# POSIX sh: the robot image has no bash.
set -eu

# LINT.IfChange(bus_ports)
# 5600-5631: robot 5600, mpc 5610, operator 5620, teleop 5621, config_push 5622.
BUS_PORT=5600
BUS_PORT_MASK=0xffe0
# LINT.ThenChange(//config/ipc/network.textproto:localhost_nodes, //config/ipc/two_machine.example.textproto:two_machine_nodes)
# Distinctive handles (hexadecimal), so that `clear` removes this script's qdisc and nobody else's.
ROOT_HANDLE=5600:
NETEM_HANDLE=5601:

usage() {
    echo "usage: netem.sh apply <interface> <netem parameters...> | clear <interface> | show <interface>" >&2
    exit 2
}

installed() {
    tc qdisc show dev "$1" | grep -q "^qdisc prio ${ROOT_HANDLE} root"
}

clear_netem() {
    if installed "$1"; then
        tc qdisc del dev "$1" root
        echo "netem.sh: removed the bus's netem on $1"
    fi
}

apply_netem() {
    interface="$1"
    shift
    [ "$#" -gt 0 ] || usage
    # A qdisc a previous run left (a container killed before it could clear it) is replaced, not stacked.
    clear_netem "${interface}"
    tc qdisc replace dev "${interface}" root handle "${ROOT_HANDLE}" prio bands 4 \
        priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
    tc qdisc add dev "${interface}" parent "${ROOT_HANDLE}4" handle "${NETEM_HANDLE}" netem "$@"
    # TCP only (IP protocol 6): the u32 port match reads the bytes at the transport header's port offsets of any
    # protocol. IPv4 and IPv6 (whose `ip6 protocol` is the first next header, which TCP is without extension headers).
    for direction in sport dport; do
        tc filter add dev "${interface}" parent "${ROOT_HANDLE}" protocol ip prio 1 u32 \
            match ip protocol 6 0xff match ip "${direction}" "${BUS_PORT}" "${BUS_PORT_MASK}" flowid "${ROOT_HANDLE}4"
        tc filter add dev "${interface}" parent "${ROOT_HANDLE}" protocol ipv6 prio 2 u32 \
            match ip6 protocol 6 0xff match ip6 "${direction}" "${BUS_PORT}" "${BUS_PORT_MASK}" flowid "${ROOT_HANDLE}4"
    done
    echo "netem.sh: the bus's TCP packets (ports ${BUS_PORT}/${BUS_PORT_MASK}, IPv4 and IPv6) on ${interface}: netem $*"
}

[ "$#" -ge 2 ] || usage
command="$1"
interface="$2"
shift 2
case "${command}" in
    apply) apply_netem "${interface}" "$@" ;;
    clear) [ "$#" -eq 0 ] || usage; clear_netem "${interface}" ;;
    show) [ "$#" -eq 0 ] || usage; tc qdisc show dev "${interface}"; tc filter show dev "${interface}" ;;
    *) usage ;;
esac
