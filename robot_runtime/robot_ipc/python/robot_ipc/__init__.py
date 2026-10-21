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


"""robot_ipc: the Cyclone DDS bus of the distributed runtime, in Python (the twin of robot_runtime/robot_ipc in C++).

See robot_runtime/robot_ipc/README.md and humanoid_nmpc/docs/distributed_runtime/README.md.
"""

from robot_ipc import bus
from robot_ipc import codec
from robot_ipc import domain_config
from robot_ipc import envelope
from robot_ipc import isolated_network
from robot_ipc import network_config
from robot_ipc import qos_profile

# The package's API, so that callers import the package and write `robot_ipc.Bus` (Python style 2.2.4: they import a
# module, not its names).
Bus = bus.Bus
BusError = bus.BusError
Delivery = bus.Delivery
parse_delivery = bus.parse_delivery
TopicStatistics = bus.TopicStatistics
WIRE_PROTOCOL_VERSION = envelope.WIRE_PROTOCOL_VERSION
ENVELOPE_TYPE_NAME = envelope.ENVELOPE_TYPE_NAME
Envelope = envelope.Envelope
decode = codec.decode
message_class = codec.message_class
UnknownMessageTypeError = codec.UnknownMessageTypeError
cyclone_domain_config = domain_config.cyclone_domain_config
InterfaceError = domain_config.InterfaceError
unicast_port_range = network_config.unicast_port_range
isolated_test_network = isolated_network.isolated_test_network
format_network_config = network_config.format_network_config
load_network_config = network_config.load_network_config
LOOPBACK_HOST = network_config.LOOPBACK_HOST
MAX_DOMAIN_ID = network_config.MAX_DOMAIN_ID
NetworkConfig = network_config.NetworkConfig
NetworkConfigError = network_config.NetworkConfigError
NodeEndpoint = network_config.NodeEndpoint
parse_network_config = network_config.parse_network_config
TopicQos = network_config.TopicQos
TransportOptions = network_config.TransportOptions
validate_network_config = network_config.validate_network_config
QosProfile = qos_profile.QosProfile
parse_qos_profile = qos_profile.parse_qos_profile

__all__ = [
    "Bus",
    "BusError",
    "Delivery",
    "ENVELOPE_TYPE_NAME",
    "Envelope",
    "InterfaceError",
    "LOOPBACK_HOST",
    "MAX_DOMAIN_ID",
    "NetworkConfig",
    "NetworkConfigError",
    "NodeEndpoint",
    "QosProfile",
    "TopicQos",
    "TopicStatistics",
    "TransportOptions",
    "UnknownMessageTypeError",
    "WIRE_PROTOCOL_VERSION",
    "cyclone_domain_config",
    "decode",
    "format_network_config",
    "isolated_test_network",
    "load_network_config",
    "message_class",
    "parse_delivery",
    "parse_network_config",
    "parse_qos_profile",
    "unicast_port_range",
    "validate_network_config",
]
