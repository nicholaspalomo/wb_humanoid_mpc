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


"""The network file of the bus, as robot_ipc/NetworkConfig.h reads it in C++.

The file is a textproto of robot_ipc_proto.NetworkConfig (robot_runtime/robot_ipc/proto/network_config.proto):

    domain_id: 0
    nodes { name: "robot" host: "192.168.1.10" }
    nodes { name: "mpc" host: "192.168.1.20" }
    topics { name: "mpc/policy" qos_profile: "reliable_latest" publishers: "mpc" }

It is parsed strictly with google.protobuf.text_format: a syntax error, an unknown field or a value of the wrong type
is a NetworkConfigError that names the file, the line and the column, e.g.
"config/ipc/network.textproto:3:42: Message type "robot_ipc_proto.NetworkConfig.Node" has no field named "hots".".
The content is then validated, and an error names the node, the topic or the field, e.g.
"config/ipc/network.textproto: domain_id: 300 is not a domain id in [0, 232]".
"""

import dataclasses
import re

from google.protobuf import text_format
from robot_ipc_proto import network_config_pb2

from robot_ipc import qos_profile

# The largest domain id whose ports stay below 65536 (domain_config.py).
MAX_DOMAIN_ID = 232
# The host of a single-machine network.
LOOPBACK_HOST = "127.0.0.1"

# The rules of the network file and the words of their error messages; the C++ loader applies the same ones.
# LINT.IfChange(network_rules)
NODES_FIELD = "nodes"
TOPICS_FIELD = "topics"
TRANSPORT_FIELD = "transport"
# A node name is a non-empty run of letters, digits, '_' and '-' (fullmatch: '$' would let a final newline through).
_NODE_NAME = re.compile(r"[A-Za-z0-9_-]+")
# A domain tag is a run of letters, digits, '_', '-' and '.'.
_DOMAIN_TAG = re.compile(r"[A-Za-z0-9_.-]*")
# A DDS topic name is letters, digits, '_' and '/', not starting with a digit (Cyclone DDS's own rule).
_TOPIC_NAME = re.compile(r"[A-Za-z_/][A-Za-z0-9_/]*")
# The hosts that name every interface rather than a machine.
_WILDCARD_HOSTS = ("0.0.0.0", "*")
_LOCALHOST_NAME = "localhost"
_LOOPBACK_PREFIX = "127."
# A host is printable ASCII without a space (host names and IPv4 addresses are ASCII; an IDN name is written in
# punycode), and without these, which belong to an endpoint or a port.
_FIRST_HOST_CHARACTER = "!"
_LAST_HOST_CHARACTER = "~"
_HOST_REFUSED_PUNCTUATION = "/:"
# The UDP payload sizes Cyclone DDS can send.
MIN_MESSAGE_SIZE_BYTES = 1024
MAX_MESSAGE_SIZE_BYTES = 65500
# The largest process count per host whose unicast ports a domain's 250 ports leave room for.
MAX_PROCESSES_PER_HOST = 115
# Packets per thousand: 1000 drops every one.
_PER_MILLE = 1000
# LINT.ThenChange(//robot_runtime/robot_ipc/src/NetworkConfig.cpp:network_rules)
# The largest UDP port.
_MAX_PORT = 65535
_IPV4_ADDRESS = re.compile(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}")
_MAX_OCTET = 255


class NetworkConfigError(ValueError):
    """A network file or a network built in code that the bus cannot use."""


def is_ipv4_address(host: str) -> bool:
    """True for a dotted-quad IPv4 address, such as "192.168.1.10" (a host name is not)."""
    if not _IPV4_ADDRESS.fullmatch(host):
        return False
    return all(int(octet) <= _MAX_OCTET for octet in host.split("."))


def is_loopback_host(host: str) -> bool:
    """True for an address of the loopback interface ("127.0.0.1", any 127.x.y.z, and "localhost")."""
    return host == _LOCALHOST_NAME or (
        is_ipv4_address(host) and host.startswith(_LOOPBACK_PREFIX)
    )


@dataclasses.dataclass(frozen=True)
class NodeEndpoint:
    """One node: a process that publishes under this name, and the machine it runs on.

    Attributes:
        name: the node name a process publishes as, e.g. "robot".
        host: the machine of the node, an IPv4 address or a host name. Every bus sends its discovery packets to the host
            of every node, and the node's own process uses the network interface of that address.
    """

    name: str
    host: str = LOOPBACK_HOST


@dataclasses.dataclass(frozen=True)
class TopicQos:
    """One topic of the network file and its quality of service.

    Attributes:
        topic: the bus topic, which is also the DDS topic name.
        profile: its QoS profile.
        publishers: the nodes that publish it; their buses create its DDS writer at construction.
    """

    topic: str
    profile: qos_profile.QosProfile = qos_profile.QosProfile.RELIABLE_EVENTS
    publishers: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class TransportOptions:
    """The DDS transport; every field is optional, 0 keeps its default (network_config.proto names them).

    Attributes:
        max_message_size_bytes: General/MaxMessageSize; 0: Cyclone's 14720 B.
        socket_receive_buffer_bytes: Internal/SocketReceiveBufferSize max; 0: Cyclone's 1 MiB request.
        socket_send_buffer_bytes: Internal/SocketSendBufferSize max; 0: the OS default.
        writer_unacked_bytes_limit: Internal/Watermarks/WhcHigh; 0: DEFAULT_WRITER_UNACKED_BYTES_LIMIT.
        max_processes_per_host: Discovery/MaxAutoParticipantIndex + 1; 0: DEFAULT_MAX_PROCESSES_PER_HOST.
        lease_duration_ms: Discovery/LeaseDuration; 0: DEFAULT_LEASE_DURATION_MS.
        simulated_send_loss_per_mille: Internal/Test/XmitLossiness, UDP packets dropped per 1000 sent, for tests and
            rehearsals; 0 on a real network.
    """

    max_message_size_bytes: int = 0
    socket_receive_buffer_bytes: int = 0
    socket_send_buffer_bytes: int = 0
    writer_unacked_bytes_limit: int = 0
    max_processes_per_host: int = 0
    lease_duration_ms: int = 0
    simulated_send_loss_per_mille: int = 0


# LINT.IfChange(transport_defaults)
DEFAULT_WRITER_UNACKED_BYTES_LIMIT = 8 << 20
DEFAULT_MAX_PROCESSES_PER_HOST = 20
DEFAULT_LEASE_DURATION_MS = 3000
# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/NetworkConfig.h:transport_defaults)


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    """The DDS network of the bus: its nodes, domain, topic QoS and transport.

    Attributes:
        nodes: the nodes, in the order of the file.
        domain_id: the DDS domain, in [0, MAX_DOMAIN_ID].
        domain_tag: Discovery/Tag: only processes with the same tag discover each other. Empty: none.
        topics: the topics the file lists, each at most once.
        default_qos_profile: the profile of a topic that `topics` does not list.
        transport: the DDS transport.
    """

    nodes: tuple[NodeEndpoint, ...]
    domain_id: int = 0
    domain_tag: str = ""
    topics: tuple[TopicQos, ...] = ()
    default_qos_profile: qos_profile.QosProfile = (
        qos_profile.QosProfile.RELIABLE_EVENTS
    )
    transport: TransportOptions = TransportOptions()

    def find(self, name: str) -> NodeEndpoint | None:
        """The node of that name, or None."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def node_names(self) -> list[str]:
        """The node names, in order."""
        return [node.name for node in self.nodes]

    def find_topic(self, topic: str) -> TopicQos | None:
        """The entry of `topic` in `topics`, or None."""
        for entry in self.topics:
            if entry.topic == topic:
                return entry
        return None

    def qos_profile_of(self, topic: str) -> qos_profile.QosProfile:
        """The profile of `topic`: its entry's, or default_qos_profile."""
        entry = self.find_topic(topic)
        return self.default_qos_profile if entry is None else entry.profile

    def topics_published_by(self, node_name: str) -> list[str]:
        """The topics that `node_name` publishes, in the order of the file."""
        return [
            entry.topic
            for entry in self.topics
            if node_name in entry.publishers
        ]

    def peer_hosts(self) -> list[str]:
        """The distinct hosts of the nodes, in order: the unicast discovery peers."""
        return list(dict.fromkeys(node.host for node in self.nodes))


# The DDSI port mapping (DDSI-RTPS 2.5, 9.6.1.1), which Cyclone DDS applies with its defaults.
# LINT.IfChange(dds_ports)
DDS_PORT_BASE = 7400
DDS_DOMAIN_GAIN = 250
DDS_PARTICIPANT_GAIN = 2
DDS_UNICAST_DISCOVERY_OFFSET = 10
DDS_UNICAST_DATA_OFFSET = 11
# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/DomainConfig.h:dds_ports)


@dataclasses.dataclass(frozen=True)
class PortRange:
    """A closed range of UDP ports.

    Attributes:
        first: the first port.
        last: the last port.
    """

    first: int
    last: int


def max_processes_per_host(network: NetworkConfig) -> int:
    """TransportOptions.max_processes_per_host, or its default."""
    return (
        network.transport.max_processes_per_host
        or DEFAULT_MAX_PROCESSES_PER_HOST
    )


def unicast_port_range(network: NetworkConfig) -> PortRange:
    """The unicast UDP ports the processes of one machine use on `network` (DomainConfig.h unicastPortRange()).

    Process index i (Cyclone DDS picks the lowest free one) listens on 7400 + 250 * domain_id + 10 + 2 * i (discovery)
    and on the port after it (data), for i below max_processes_per_host. Domain 0 with 20 processes: 7410-7449.

    Args:
        network: The network.

    Returns:
        The range of the ports.
    """
    domain_base = DDS_PORT_BASE + DDS_DOMAIN_GAIN * network.domain_id
    return PortRange(
        first=domain_base + DDS_UNICAST_DISCOVERY_OFFSET,
        last=domain_base
        + DDS_UNICAST_DATA_OFFSET
        + DDS_PARTICIPANT_GAIN * (max_processes_per_host(network) - 1),
    )


def _entry_key(field: str, index: int, name: str) -> str:
    """'nodes[1] (mpc)': the position of an entry in the file, and its name when it has one."""
    position = f"{field}[{index}]"
    return f"{position} ({name})" if name else position


def _field_key(field: str, index: int, name: str, member: str) -> str:
    return f"{_entry_key(field, index, name)}.{member}"


def _validate_host(key: str, host: str) -> None:
    """Raises NetworkConfigError when `host` is not a machine's address or name."""
    if not host:
        raise NetworkConfigError(f"{key}: is missing")
    if host in _WILDCARD_HOSTS:
        raise NetworkConfigError(
            f"{key}: '{host}' names every interface, not a machine: write the"
            " address of the node's machine"
        )
    if any(
        not _FIRST_HOST_CHARACTER <= character <= _LAST_HOST_CHARACTER
        or character in _HOST_REFUSED_PUNCTUATION
        for character in host
    ):
        raise NetworkConfigError(
            f"{key}: '{host}' is not a host: write an IPv4 address or a host"
            " name, without a scheme or a port"
        )


def topic_name_problem(topic: str) -> str | None:
    """Why `topic` is not a DDS topic name, or None when it is one."""
    if not topic:
        return "the topic is empty"
    if not _TOPIC_NAME.fullmatch(topic):
        return (
            f"'{topic}' is not a DDS topic name: letters, digits, '_' and '/',"
            " not starting with a digit"
        )
    return None


def _validate_nodes(config: NetworkConfig) -> None:
    """Raises NetworkConfigError naming the first invalid or repeated node."""
    if not config.nodes:
        raise NetworkConfigError(f"{NODES_FIELD}: names no node")
    names: dict[str, int] = {}
    for index, node in enumerate(config.nodes):
        name_key = _field_key(NODES_FIELD, index, node.name, "name")
        if not node.name:
            raise NetworkConfigError(f"{name_key}: is missing")
        if not isinstance(node.name, str) or not _NODE_NAME.fullmatch(
            node.name
        ):
            raise NetworkConfigError(
                f"{name_key}: a node name is a non-empty run of letters,"
                " digits, '_' and '-'"
            )
        if node.name in names:
            raise NetworkConfigError(
                f"{name_key}: is also the name of"
                f" {_entry_key(NODES_FIELD, names[node.name], node.name)}"
            )
        names[node.name] = index
        _validate_host(
            _field_key(NODES_FIELD, index, node.name, "host"), node.host
        )


def _validate_topics(config: NetworkConfig) -> None:
    """Raises NetworkConfigError naming the first invalid or repeated topic, or a publisher that is no node."""
    topics: dict[str, int] = {}
    for index, entry in enumerate(config.topics):
        name_key = _field_key(TOPICS_FIELD, index, entry.topic, "name")
        problem = topic_name_problem(entry.topic)
        if problem is not None:
            raise NetworkConfigError(f"{name_key}: {problem}")
        if entry.topic in topics:
            raise NetworkConfigError(
                f"{name_key}: is also the name of"
                f" {_entry_key(TOPICS_FIELD, topics[entry.topic], entry.topic)}"
            )
        topics[entry.topic] = index
        for publisher in entry.publishers:
            if config.find(publisher) is None:
                raise NetworkConfigError(
                    f"{_field_key(TOPICS_FIELD, index, entry.topic, 'publishers')}:"
                    f" '{publisher}' is not a node of the network (nodes:"
                    f" {', '.join(config.node_names())})"
                )


def _validate_transport(transport: TransportOptions) -> None:
    """Raises NetworkConfigError naming the first transport value out of range."""
    size = transport.max_message_size_bytes
    if (
        size != 0
        and not MIN_MESSAGE_SIZE_BYTES <= size <= MAX_MESSAGE_SIZE_BYTES
    ):
        raise NetworkConfigError(
            f"{TRANSPORT_FIELD}.max_message_size_bytes: {size} is not 0 (the"
            f" default) or in [{MIN_MESSAGE_SIZE_BYTES},"
            f" {MAX_MESSAGE_SIZE_BYTES}]"
        )
    if (
        min(
            transport.socket_receive_buffer_bytes,
            transport.socket_send_buffer_bytes,
            transport.writer_unacked_bytes_limit,
        )
        < 0
    ):
        raise NetworkConfigError(
            f"{TRANSPORT_FIELD}: the buffer sizes must be >= 0 (0: the default)"
        )
    if not 0 <= transport.max_processes_per_host <= MAX_PROCESSES_PER_HOST:
        raise NetworkConfigError(
            f"{TRANSPORT_FIELD}.max_processes_per_host:"
            f" {transport.max_processes_per_host} is not 0 (the default) or in"
            f" [1, {MAX_PROCESSES_PER_HOST}]"
        )
    if transport.lease_duration_ms < 0:
        raise NetworkConfigError(
            f"{TRANSPORT_FIELD}.lease_duration_ms: must be >= 0 (0: the"
            " default)"
        )
    loss = transport.simulated_send_loss_per_mille
    if not 0 <= loss < _PER_MILLE:
        raise NetworkConfigError(
            f"{TRANSPORT_FIELD}.simulated_send_loss_per_mille: {loss} is not in"
            f" [0, {_PER_MILLE})"
        )


def validate_network_config(config: NetworkConfig) -> None:
    """The checks that also apply to a network built in code (Bus runs them).

    At least one node, valid and unique node names, valid hosts, a domain id in [0, MAX_DOMAIN_ID], a valid tag, valid
    and unique topic names whose publishers are nodes of the network, and transport values in range.

    Args:
        config: The network to check.

    Raises:
        NetworkConfigError: naming the offending node, topic or field.
    """
    _validate_nodes(config)
    if not 0 <= config.domain_id <= MAX_DOMAIN_ID:
        raise NetworkConfigError(
            f"domain_id: {config.domain_id} is not a domain id in [0,"
            f" {MAX_DOMAIN_ID}]"
        )
    if not _DOMAIN_TAG.fullmatch(config.domain_tag):
        raise NetworkConfigError(
            f"domain_tag: '{config.domain_tag}': a domain tag is a run of"
            " letters, digits, '_', '-' and '.'"
        )
    _validate_topics(config)
    _validate_transport(config.transport)
    ports = unicast_port_range(config)
    if ports.last > _MAX_PORT:
        raise NetworkConfigError(
            f"domain_id: domain {config.domain_id} with"
            f" {ports.last - ports.first + 1} ports per host reaches UDP port"
            f" {ports.last}, beyond {_MAX_PORT}: a smaller domain_id or"
            " transport.max_processes_per_host"
        )


def _profile(key: str, name: str) -> qos_profile.QosProfile:
    try:
        return qos_profile.parse_qos_profile(name)
    except ValueError as error:
        raise NetworkConfigError(f"{key}: {error}") from None


def _from_proto(message: network_config_pb2.NetworkConfig) -> NetworkConfig:
    """The network of a parsed file, validated."""
    if message.domain_id > MAX_DOMAIN_ID:
        raise NetworkConfigError(
            f"domain_id: {message.domain_id} is not a domain id in [0,"
            f" {MAX_DOMAIN_ID}]"
        )
    topics = []
    for index, topic in enumerate(message.topics):
        key = _field_key(TOPICS_FIELD, index, topic.name, "qos_profile")
        if not topic.qos_profile:
            raise NetworkConfigError(f"{key}: is missing")
        topics.append(
            TopicQos(
                topic=topic.name,
                profile=_profile(key, topic.qos_profile),
                publishers=tuple(topic.publishers),
            )
        )
    default_profile = qos_profile.QosProfile.RELIABLE_EVENTS
    if message.default_qos_profile:
        default_profile = _profile(
            "default_qos_profile", message.default_qos_profile
        )
    transport = message.transport
    config = NetworkConfig(
        nodes=tuple(
            NodeEndpoint(name=node.name, host=node.host)
            for node in message.nodes
        ),
        domain_id=message.domain_id,
        domain_tag=message.domain_tag,
        topics=tuple(topics),
        default_qos_profile=default_profile,
        transport=TransportOptions(
            max_message_size_bytes=transport.max_message_size_bytes,
            socket_receive_buffer_bytes=transport.socket_receive_buffer_bytes,
            socket_send_buffer_bytes=transport.socket_send_buffer_bytes,
            writer_unacked_bytes_limit=transport.writer_unacked_bytes_limit,
            max_processes_per_host=transport.max_processes_per_host,
            lease_duration_ms=transport.lease_duration_ms,
            simulated_send_loss_per_mille=transport.simulated_send_loss_per_mille,
        ),
    )
    validate_network_config(config)
    return config


def _parse_error_text(error: text_format.ParseError, source: str) -> str:
    """'<source>:<line>:<column>: <problem>', the form the C++ loader reports."""
    line, column = error.GetLine(), error.GetColumn()
    text = str(error)
    if line is None:
        return f"{source}: {text}"
    location = f"{line}:{column}" if column is not None else f"{line}"
    # ParseError prefixes its message with "<line>:<column> : ".
    prefix = f"{location} : "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return f"{source}:{location}: {text}"


def parse_network_config(text: str, source: str = "<string>") -> NetworkConfig:
    """Parses and validates the text of a network file; `source` names it in the error messages.

    Args:
        text: The textproto of a robot_ipc_proto.NetworkConfig.
        source: The file the text came from, which every error message starts with.

    Returns:
        The validated network.

    Raises:
        NetworkConfigError: naming the line and column of a malformed file, or the offending node, topic or field.
    """
    message = network_config_pb2.NetworkConfig()
    try:
        # Strict: Parse() (unlike Merge()) rejects a non-repeated field given twice, and an unknown field is an error.
        text_format.Parse(text, message, allow_unknown_field=False)
    except text_format.ParseError as error:
        raise NetworkConfigError(_parse_error_text(error, source)) from None
    try:
        return _from_proto(message)
    except NetworkConfigError as error:
        raise NetworkConfigError(f"{source}: {error}") from None


def load_network_config(path: str) -> NetworkConfig:
    """Reads and validates a network file.

    Args:
        path: The network file, such as config/ipc/network.textproto.

    Returns:
        The validated network.

    Raises:
        FileNotFoundError: the file does not exist (OSError for other failures to read it).
        NetworkConfigError: naming the line and column of a malformed file, or the offending node, topic or field.
    """
    with open(path, encoding="utf-8") as file:
        return parse_network_config(file.read(), path)


def _to_proto(config: NetworkConfig) -> network_config_pb2.NetworkConfig:
    """The robot_ipc_proto.NetworkConfig of `config`."""
    message = network_config_pb2.NetworkConfig(
        domain_id=config.domain_id, domain_tag=config.domain_tag
    )
    for node in config.nodes:
        message.nodes.add(name=node.name, host=node.host)
    for entry in config.topics:
        message.topics.add(
            name=entry.topic,
            qos_profile=entry.profile.value,
            publishers=list(entry.publishers),
        )
    if config.default_qos_profile is not qos_profile.QosProfile.RELIABLE_EVENTS:
        message.default_qos_profile = config.default_qos_profile.value
    if config.transport != TransportOptions():
        message.transport.CopyFrom(
            network_config_pb2.NetworkConfig.Transport(
                **dataclasses.asdict(config.transport)
            )
        )
    return message


def _one_line(message: network_config_pb2.NetworkConfig) -> str:
    return str(text_format.MessageToString(message, as_one_line=True)).strip()


def format_network_config(config: NetworkConfig) -> str:
    """The network file of `config`, one line per node and per topic.

    parse_network_config() reads it back as `config` when it is valid. The C++ formatNetworkConfig() writes the same.

    Args:
        config: The network to write.

    Returns:
        The textproto.
    """
    message = _to_proto(config)
    scalars = network_config_pb2.NetworkConfig()
    scalars.CopyFrom(message)
    scalars.ClearField("nodes")
    scalars.ClearField("topics")
    scalars.ClearField("transport")
    lines = [_one_line(scalars)] if scalars.ListFields() else []
    for node in message.nodes:
        entry = network_config_pb2.NetworkConfig()
        entry.nodes.add().CopyFrom(node)
        lines.append(_one_line(entry))
    for topic in message.topics:
        entry = network_config_pb2.NetworkConfig()
        entry.topics.add().CopyFrom(topic)
        lines.append(_one_line(entry))
    if message.HasField("transport"):
        entry = network_config_pb2.NetworkConfig()
        entry.transport.CopyFrom(message.transport)
        lines.append(_one_line(entry))
    return "".join(f"{line}\n" for line in lines)


__all__ = [
    "DEFAULT_LEASE_DURATION_MS",
    "DEFAULT_MAX_PROCESSES_PER_HOST",
    "DEFAULT_WRITER_UNACKED_BYTES_LIMIT",
    "LOOPBACK_HOST",
    "MAX_DOMAIN_ID",
    "NetworkConfig",
    "NetworkConfigError",
    "NodeEndpoint",
    "PortRange",
    "TopicQos",
    "TransportOptions",
    "format_network_config",
    "is_ipv4_address",
    "is_loopback_host",
    "load_network_config",
    "parse_network_config",
    "topic_name_problem",
    "unicast_port_range",
    "validate_network_config",
]
