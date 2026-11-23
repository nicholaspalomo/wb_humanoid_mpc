# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""The Zenoh configuration of a process of the network, as robot_ipc/SessionConfig.h writes it in C++.

The bus passes the text to zenoh.Config.from_json5() and zenoh.open(), so no JSON5 file and no ZENOH_CONFIG is
involved. The same network, node and port give the same text in C++ and in Python; the cross-language test compares
them.
"""

import dataclasses
import json
import socket

from robot_ipc import network_config

# The TCP ports of a domain: process index i of a machine listens on ZENOH_PORT_BASE + ZENOH_DOMAIN_GAIN * domain_id
# + i, for i below max_processes_per_host. 7447 is Zenoh's own port; domain 232 ends below 32768, where Linux's
# ephemeral ports start.
# LINT.IfChange(zenoh_ports)
ZENOH_PORT_BASE = 7447
ZENOH_DOMAIN_GAIN = 100
# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/SessionConfig.h:zenoh_ports)

# The address a listener on every interface binds.
_EVERY_INTERFACE = "0.0.0.0"


class InterfaceError(OSError):
    """A node whose host is an address of another machine: its process cannot run on this one."""


@dataclasses.dataclass(frozen=True)
class PortRange:
    """A closed range of TCP ports.

    Attributes:
        first: the first port.
        last: the last port.
    """

    first: int
    last: int


def tcp_port_range(network: network_config.NetworkConfig) -> PortRange:
    """The TCP ports the processes of one machine listen on in `network`'s domain (SessionConfig.h tcpPortRange()).

    The process of index i (the lowest port free when it opens its session) listens on 7447 + 100 * domain_id + i.
    Domain 0 with 20 processes: 7447-7466.

    Args:
        network: The network.

    Returns:
        The range of the ports.
    """
    first = ZENOH_PORT_BASE + ZENOH_DOMAIN_GAIN * network.domain_id
    return PortRange(
        first=first,
        last=first + network_config.max_processes_per_host(network) - 1,
    )


def _is_local_address(address: str) -> bool:
    """Whether `address` (IPv4) belongs to an interface of this machine: a socket can bind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind((address, 0))
        except OSError:
            return False
    return True


def interface_address_for(
    network: network_config.NetworkConfig, node_name: str
) -> str:
    """The address the process of `node_name` listens on (SessionConfig.h interfaceAddressFor()).

    Args:
        network: The network.
        node_name: The node the process publishes as; empty for a bus that only subscribes.

    Returns:
        The interface address; empty for every interface.

    Raises:
        ValueError: the node is not in the network.
        InterfaceError: the node's host is an IPv4 address of no interface of this machine.
    """
    if not node_name:
        if all(
            network_config.is_loopback_host(node.host) for node in network.nodes
        ):
            return network_config.LOOPBACK_HOST
        for candidate in network.nodes:
            if (
                network_config.is_ipv4_address(candidate.host)
                and not network_config.is_loopback_host(candidate.host)
                and _is_local_address(candidate.host)
            ):
                return candidate.host
        return ""
    node = network.find(node_name)
    if node is None:
        raise ValueError(f"node '{node_name}' is not in the network")
    if network_config.is_loopback_host(node.host):
        return network_config.LOOPBACK_HOST
    if not network_config.is_ipv4_address(node.host):
        return ""
    if not _is_local_address(node.host):
        raise InterfaceError(
            f"node '{node.name}' is on {node.host}, which is no address of this"
            " machine: run it on that machine, or fix its host"
        )
    return node.host


def is_tcp_port_free(address: str, port: int) -> bool:
    """Whether a TCP listener can bind `port` on `address` (empty: every interface) now (SessionConfig.h).

    Args:
        address: An IPv4 address of this machine, or empty for every interface.
        port: The port.

    Returns:
        True when a socket could bind it, as a listener does (a port in TIME_WAIT is free).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((address or _EVERY_INTERFACE, port))
        except OSError:
            return False
    return True


def _tcp_endpoint(host: str, port: int) -> str:
    return f"tcp/{host}:{port}"


def connect_endpoints(network: network_config.NetworkConfig) -> list[str]:
    """The endpoints every process of `network` connects to (SessionConfig.h connectEndpoints()).

    Args:
        network: The network.

    Returns:
        "tcp/<host>:<port>" for the host of every node and every port of tcp_port_range(), host by host.
    """
    ports = tcp_port_range(network)
    endpoints = []
    for host in network.peer_hosts():
        endpoints += [
            _tcp_endpoint(host, port)
            for port in range(ports.first, ports.last + 1)
        ]
    return endpoints


# LINT.IfChange(zenoh_session_json)
def _json_strings(texts: list[str]) -> str:
    """A JSON array of strings, without spaces, as the C++ writes it."""
    return json.dumps(texts, separators=(",", ":"), ensure_ascii=False)


def _link_section(transport: network_config.TransportOptions) -> str:
    """The transport.link object: the lease and the TCP buffers."""
    lease = (
        transport.lease_duration_ms or network_config.DEFAULT_LEASE_DURATION_MS
    )
    link = f'"link":{{"tx":{{"lease":{lease}}}'
    tcp = []
    if transport.socket_receive_buffer_bytes > 0:
        tcp.append(f'"so_rcvbuf":{transport.socket_receive_buffer_bytes}')
    if transport.socket_send_buffer_bytes > 0:
        tcp.append(f'"so_sndbuf":{transport.socket_send_buffer_bytes}')
    if tcp:
        link += ',"tcp":{' + ",".join(tcp) + "}"
    return link + "}"


# LINT.ThenChange(//robot_runtime/robot_ipc/src/SessionConfig.cpp:zenoh_session_json)


def zenoh_session_config(
    network: network_config.NetworkConfig, node_name: str, listen_port: int
) -> str:
    """The Zenoh configuration (JSON) of a process of `network` publishing as `node_name` (empty: subscribing).

    Args:
        network: The network.
        node_name: The node the process publishes as; empty for a bus that only subscribes.
        listen_port: The TCP port the process listens on.

    Returns:
        The JSON that the bus passes to zenoh.Config.from_json5().

    Raises:
        ValueError: the node is not in the network.
        InterfaceError: the node's host is an address of another machine.
    """
    listen_host = interface_address_for(network, node_name) or _EVERY_INTERFACE
    own_endpoint = _tcp_endpoint(listen_host, listen_port)
    # Every port but the session's own listener, which Zenoh would refuse (a connection to self) on every retry.
    peers = [
        endpoint
        for endpoint in connect_endpoints(network)
        if endpoint != own_endpoint
    ]
    # LINT.IfChange(zenoh_session_layout)
    return (
        '{"mode":"peer","listen":{"endpoints":'
        + _json_strings([own_endpoint])
        + ',"timeout_ms":0,"exit_on_failure":true},"connect":{"endpoints":'
        + _json_strings(peers)
        + ',"timeout_ms":0,"exit_on_failure":false},'
        + '"scouting":{"multicast":{"enabled":false},"gossip":{"enabled":true}},'
        + '"transport":{'
        + _link_section(network.transport)
        + "}}"
    )
    # LINT.ThenChange(//robot_runtime/robot_ipc/src/SessionConfig.cpp:zenoh_session_layout)


__all__ = [
    "InterfaceError",
    "PortRange",
    "ZENOH_DOMAIN_GAIN",
    "ZENOH_PORT_BASE",
    "connect_endpoints",
    "interface_address_for",
    "is_tcp_port_free",
    "tcp_port_range",
    "zenoh_session_config",
]
