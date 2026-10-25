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


"""The Cyclone DDS configuration of a process of the network, as robot_ipc/DomainConfig.h writes it in C++.

The bus passes the text to cyclonedds.domain.Domain (dds_create_domain()), so no XML file and no CYCLONEDDS_URI is
involved. The same network and node give the same text in C++ and in Python; the cross-language test compares them.
"""

import socket
from xml.sax import saxutils

from robot_ipc import network_config

# The entities that an attribute value or an element's text escapes.
_XML_ENTITIES = {'"': "&quot;", "'": "&apos;"}


class InterfaceError(OSError):
    """A node whose host is an address of another machine: its process cannot run on this one."""


def _is_local_address(address: str) -> bool:
    """Whether `address` (IPv4) belongs to an interface of this machine: a UDP socket can bind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind((address, 0))
        except OSError:
            return False
    return True


def interface_address_for(
    network: network_config.NetworkConfig, node_name: str
) -> str:
    """The address of the network interface the process of `node_name` uses (DomainConfig.h interfaceAddressFor()).

    Args:
        network: The network.
        node_name: The node the process publishes as; empty for a bus that only subscribes.

    Returns:
        The interface address; empty when Cyclone DDS picks the interface.

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


# LINT.IfChange(domain_config_xml)
def _escaped(text: str) -> str:
    return saxutils.escape(text, _XML_ENTITIES)


def _bytes(value: int) -> str:
    return f"{value}B"


def _general_section(
    interface_address: str, transport: network_config.TransportOptions
) -> str:
    """The General element: the interface, no multicast, the message size."""
    if interface_address:
        interfaces = (
            "<Interfaces><NetworkInterface"
            f' address="{_escaped(interface_address)}"/></Interfaces>'
        )
    else:
        interfaces = (
            '<Interfaces><NetworkInterface autodetermine="true"/></Interfaces>'
        )
    general = f"<General>{interfaces}<AllowMulticast>false</AllowMulticast>"
    if transport.max_message_size_bytes > 0:
        general += f"<MaxMessageSize>{_bytes(transport.max_message_size_bytes)}</MaxMessageSize>"
    return general + "</General>"


def _discovery_section(network: network_config.NetworkConfig) -> str:
    """The Discovery element: the tag, the process indices, the unicast peers, the lease."""
    lease = (
        network.transport.lease_duration_ms
        or network_config.DEFAULT_LEASE_DURATION_MS
    )
    discovery = "<Discovery>"
    if network.domain_tag:
        discovery += f"<Tag>{_escaped(network.domain_tag)}</Tag>"
    max_index = network_config.max_processes_per_host(network) - 1
    discovery += (
        "<ParticipantIndex>auto</ParticipantIndex>"
        f"<MaxAutoParticipantIndex>{max_index}</MaxAutoParticipantIndex>"
        '<Peers AddLocalhost="false">'
    )
    for host in network.peer_hosts():
        discovery += f'<Peer Address="{_escaped(host)}"/>'
    return (
        discovery
        + f"</Peers><LeaseDuration>{lease}ms</LeaseDuration></Discovery>"
    )


def _internal_section(transport: network_config.TransportOptions) -> str:
    """The Internal element: the socket buffers, the unacknowledged-bytes limit, the simulated loss."""
    unacked = (
        transport.writer_unacked_bytes_limit
        or network_config.DEFAULT_WRITER_UNACKED_BYTES_LIMIT
    )
    internal = "<Internal>"
    if transport.socket_receive_buffer_bytes > 0:
        internal += (
            '<SocketReceiveBufferSize min="default" '
            f'max="{_bytes(transport.socket_receive_buffer_bytes)}"/>'
        )
    if transport.socket_send_buffer_bytes > 0:
        internal += (
            '<SocketSendBufferSize min="default" '
            f'max="{_bytes(transport.socket_send_buffer_bytes)}"/>'
        )
    internal += f"<Watermarks><WhcHigh>{_bytes(unacked)}</WhcHigh><WhcAdaptive>false</WhcAdaptive></Watermarks>"
    if transport.simulated_send_loss_per_mille > 0:
        internal += f"<Test><XmitLossiness>{transport.simulated_send_loss_per_mille}</XmitLossiness></Test>"
    return internal + "</Internal>"


# LINT.ThenChange(//robot_runtime/robot_ipc/src/DomainConfig.cpp:domain_config_xml)


def cyclone_domain_config(
    network: network_config.NetworkConfig, node_name: str
) -> str:
    """The Cyclone DDS configuration (XML) of a process of `network` publishing as `node_name` (empty: subscribing).

    Args:
        network: The network.
        node_name: The node the process publishes as; empty for a bus that only subscribes.

    Returns:
        The XML that the bus passes to cyclonedds.domain.Domain.

    Raises:
        ValueError: the node is not in the network.
        InterfaceError: the node's host is an address of another machine.
    """
    interface_address = interface_address_for(network, node_name)
    return (
        '<CycloneDDS><Domain Id="any">'
        + _general_section(interface_address, network.transport)
        + _discovery_section(network)
        + _internal_section(network.transport)
        + "</Domain></CycloneDDS>"
    )


__all__ = [
    "InterfaceError",
    "cyclone_domain_config",
    "interface_address_for",
]
