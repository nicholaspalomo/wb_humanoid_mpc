"""The network file of the bus, as robot_ipc/NetworkConfig.h reads it in C++.

The file is a textproto of robot_ipc_proto.NetworkConfig (robot_runtime/robot_ipc/proto/network_config.proto):

    nodes { name: "robot" host: "127.0.0.1" port: 5600 }
    nodes { name: "mpc" host: "192.168.1.20" port: 5610 bind_host: "0.0.0.0" }

It is parsed strictly with google.protobuf.text_format: a syntax error, an unknown field or a value of the wrong type
is a NetworkConfigError that names the file, the line and the column, e.g.
"config/ipc/network.textproto:3:42: Message type "robot_ipc_proto.NetworkConfig.Node" has no field named "prot".".
The nodes are then validated, and an error names the node and its field, e.g.
"config/ipc/network.textproto: nodes[0] (robot).port: 70000 is not a port in [1, 65535]".
"""

import dataclasses
import re
from typing import Dict, List, Optional, Tuple

from google.protobuf import text_format

from robot_ipc_proto import network_config_pb2

# A port that only a network built in code may name (never a network file, where a port of 0 is a missing port): the
# node binds an ephemeral port that the kernel picks, which Bus.bound_endpoint reports. Tests use it so that they never
# collide on a port.
EPHEMERAL_PORT = 0
MAX_PORT = 65535

# The rules of a node and the words of their error messages; the C++ loader applies the same ones.
# LINT.IfChange(node_rules)
MIN_FILE_PORT = 1
NODES_FIELD = "nodes"
# A wildcard host binds every interface, and the other processes reach it over loopback.
_WILDCARD_HOSTS = ("*", "0.0.0.0")
LOOPBACK_HOST = "127.0.0.1"
# A node name is a non-empty run of letters, digits, '_' and '-'.
_NODE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# LINT.ThenChange(//robot_runtime/robot_ipc/src/NetworkConfig.cpp:node_rules)
_NODE = network_config_pb2.NetworkConfig.Node
# The field names of the schema, which the error messages use.
_NAME_FIELD = _NODE.DESCRIPTOR.fields_by_number[_NODE.NAME_FIELD_NUMBER].name
_HOST_FIELD = _NODE.DESCRIPTOR.fields_by_number[_NODE.HOST_FIELD_NUMBER].name
_PORT_FIELD = _NODE.DESCRIPTOR.fields_by_number[_NODE.PORT_FIELD_NUMBER].name
_BIND_HOST_FIELD = _NODE.DESCRIPTOR.fields_by_number[_NODE.BIND_HOST_FIELD_NUMBER].name


class NetworkConfigError(ValueError):
    """A network file or a network built in code that the bus cannot use."""


def is_wildcard_host(host: str) -> bool:
    """True for "*" and "0.0.0.0", the hosts that bind every interface."""
    return host in _WILDCARD_HOSTS


@dataclasses.dataclass(frozen=True)
class NodeEndpoint:
    """One node: where its PUB socket binds, and where every SUB socket of the network connects.

    Attributes:
        name: the node name a process publishes as, e.g. "robot".
        host: the address the other processes connect to, and the one this node binds unless bind_host is set. A
            wildcard host binds every interface, and the others then connect over loopback (a single machine only).
        port: the TCP port, in [1, MAX_PORT], or EPHEMERAL_PORT.
        bind_host: the interface to bind instead of host; empty: host.
    """

    name: str
    host: str
    port: int = EPHEMERAL_PORT
    bind_host: str = ""

    def bind_endpoint(self) -> str:
        """tcp://<bind_host or host>:<port>, with "*" for a wildcard host and for EPHEMERAL_PORT."""
        address = self.bind_host or self.host
        if is_wildcard_host(address):
            address = "*"
        port = "*" if self.port == EPHEMERAL_PORT else str(self.port)
        return f"tcp://{address}:{port}"

    def connect_endpoint(self) -> str:
        """tcp://<host>:<port>, with 127.0.0.1 for a wildcard host."""
        address = LOOPBACK_HOST if is_wildcard_host(self.host) else self.host
        return f"tcp://{address}:{self.port}"


@dataclasses.dataclass(frozen=True)
class NetworkConfig:
    """The nodes of the bus, in the order of the file."""

    nodes: Tuple[NodeEndpoint, ...]

    def find(self, name: str) -> Optional[NodeEndpoint]:
        """The node of that name, or None."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def node_names(self) -> List[str]:
        return [node.name for node in self.nodes]


def _node_key(index: int, name: str) -> str:
    """'nodes[1] (mpc)': the position of the node in the file, and its name when it has one."""
    position = f"{NODES_FIELD}[{index}]"
    return f"{position} ({name})" if name else position


def _field_key(index: int, name: str, field: str) -> str:
    return f"{_node_key(index, name)}.{field}"


def _validate_host(key: str, host: str) -> None:
    if not host:
        raise NetworkConfigError(f"{key}: is missing")
    if is_wildcard_host(host):
        return
    if any(character.isspace() or character in "/:" for character in host):
        raise NetworkConfigError(
            f"{key}: '{host}' is not a host: write an IPv4 address or a host name, without a scheme or a port"
        )


def _port_problem(port: int) -> str:
    return f"{port} is not a port in [{MIN_FILE_PORT}, {MAX_PORT}]"


def validate_network_config(config: NetworkConfig) -> None:
    """The checks that also apply to a network built in code (Bus runs them).

    At least one node, valid and unique names, valid hosts, ports in [1, MAX_PORT] or EPHEMERAL_PORT, and no two nodes
    on one endpoint.

    Raises:
        NetworkConfigError: naming the offending node and field.
    """
    if not config.nodes:
        raise NetworkConfigError(f"{NODES_FIELD}: names no node")
    names: Dict[str, int] = {}
    endpoints: Dict[str, str] = {}
    for index, node in enumerate(config.nodes):
        name_key = _field_key(index, node.name, _NAME_FIELD)
        if not isinstance(node.name, str) or not _NODE_NAME.match(node.name):
            if not node.name:
                raise NetworkConfigError(f"{name_key}: is missing")
            raise NetworkConfigError(
                f"{name_key}: a node name is a non-empty run of letters, digits, '_' and '-'"
            )
        if node.name in names:
            raise NetworkConfigError(
                f"{name_key}: is also the name of {_node_key(names[node.name], node.name)}"
            )
        names[node.name] = index
        _validate_host(_field_key(index, node.name, _HOST_FIELD), node.host)
        if node.bind_host:
            _validate_host(
                _field_key(index, node.name, _BIND_HOST_FIELD), node.bind_host
            )
        if node.port == EPHEMERAL_PORT:
            continue
        if not MIN_FILE_PORT <= node.port <= MAX_PORT:
            raise NetworkConfigError(
                f"{_field_key(index, node.name, _PORT_FIELD)}: {_port_problem(node.port)}"
            )
        endpoint = f"{node.host}:{node.port}"
        if endpoint in endpoints:
            raise NetworkConfigError(
                f"{_node_key(index, node.name)}: {endpoint} is also the endpoint of {endpoints[endpoint]}"
            )
        endpoints[endpoint] = _node_key(index, node.name)


def _from_proto(message: network_config_pb2.NetworkConfig) -> NetworkConfig:
    """The checks only a file needs: a network built in code may leave out what a file must name (EPHEMERAL_PORT)."""
    nodes = []
    for index, node in enumerate(message.nodes):
        if not node.name:
            raise NetworkConfigError(
                f"{_field_key(index, node.name, _NAME_FIELD)}: is missing"
            )
        if node.port == 0:
            raise NetworkConfigError(
                f"{_field_key(index, node.name, _PORT_FIELD)}: is missing (a network file names a port in "
                f"[{MIN_FILE_PORT}, {MAX_PORT}])"
            )
        if node.port > MAX_PORT:
            raise NetworkConfigError(
                f"{_field_key(index, node.name, _PORT_FIELD)}: {_port_problem(node.port)}"
            )
        nodes.append(
            NodeEndpoint(
                name=node.name, host=node.host, port=node.port, bind_host=node.bind_host
            )
        )
    config = NetworkConfig(nodes=tuple(nodes))
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

    Raises:
        NetworkConfigError: naming the line and column of a malformed file, or the offending node.
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

    Raises:
        FileNotFoundError: the file does not exist (OSError for other failures to read it).
        NetworkConfigError: naming the line and column of a malformed file, or the offending node.
    """
    with open(path, encoding="utf-8") as file:
        return parse_network_config(file.read(), path)


def format_network_config(config: NetworkConfig) -> str:
    """The network file of `config`, one line per node.

    parse_network_config() reads it back as `config` when it is valid and names no EPHEMERAL_PORT.
    """
    lines = []
    for node in config.nodes:
        message = network_config_pb2.NetworkConfig()
        message.nodes.add(
            name=node.name,
            host=node.host,
            port=max(node.port, 0),
            bind_host=node.bind_host,
        )
        lines.append(text_format.MessageToString(message, as_one_line=True).strip())
    return "".join(f"{line}\n" for line in lines)


def localhost_network_config() -> NetworkConfig:
    """The shipped network file, config/ipc/network.textproto: robot, mpc, operator and teleop on 127.0.0.1."""
    # LINT.IfChange(localhost_nodes)
    return NetworkConfig(
        nodes=(
            NodeEndpoint(name="robot", host="127.0.0.1", port=5600),
            NodeEndpoint(name="mpc", host="127.0.0.1", port=5610),
            NodeEndpoint(name="operator", host="127.0.0.1", port=5620),
            NodeEndpoint(name="teleop", host="127.0.0.1", port=5621),
        )
    )
    # LINT.ThenChange(//config/ipc/network.textproto:localhost_nodes, //robot_runtime/robot_ipc/src/NetworkConfig.cpp:localhost_nodes)
