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

"""robot_ipc: the ZeroMQ bus of the distributed runtime, in Python (the twin of robot_runtime/robot_ipc in C++).

See robot_runtime/robot_ipc/README.md and humanoid_nmpc/docs/distributed_runtime/README.md.
"""

from robot_ipc import bus
from robot_ipc import codec
from robot_ipc import network_config

# The package's API, so that callers import the package and write `robot_ipc.Bus` (Python style 2.2.4: they import a
# module, not its names).
Bus = bus.Bus
BusError = bus.BusError
Delivery = bus.Delivery
parse_delivery = bus.parse_delivery
TopicStatistics = bus.TopicStatistics
decode = codec.decode
message_class = codec.message_class
UnknownMessageTypeError = codec.UnknownMessageTypeError
EPHEMERAL_PORT = network_config.EPHEMERAL_PORT
format_network_config = network_config.format_network_config
is_wildcard_host = network_config.is_wildcard_host
load_network_config = network_config.load_network_config
localhost_network_config = network_config.localhost_network_config
MAX_PORT = network_config.MAX_PORT
NetworkConfig = network_config.NetworkConfig
NetworkConfigError = network_config.NetworkConfigError
NodeEndpoint = network_config.NodeEndpoint
parse_network_config = network_config.parse_network_config
validate_network_config = network_config.validate_network_config

__all__ = [
    "Bus",
    "BusError",
    "Delivery",
    "EPHEMERAL_PORT",
    "MAX_PORT",
    "NetworkConfig",
    "NetworkConfigError",
    "NodeEndpoint",
    "TopicStatistics",
    "UnknownMessageTypeError",
    "decode",
    "format_network_config",
    "is_wildcard_host",
    "load_network_config",
    "localhost_network_config",
    "message_class",
    "parse_delivery",
    "parse_network_config",
    "validate_network_config",
]
