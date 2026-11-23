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

"""The key expressions of the bus, as robot_ipc/KeyExprs.h builds them in C++.

A topic's samples are put on robot_ipc/<domain_id>/tag_<domain_tag>/<qos profile>/<topic>, and every bus declares a
liveliness token robot_ipc_endpoints/<domain_id>/tag_<domain_tag>/<role>/<qos profile>/<bus id>/<topic> for each of its
publishers and subscribers, which the other buses count.
"""

import dataclasses
import enum

from robot_ipc import network_config
from robot_ipc import qos_profile

# LINT.IfChange(key_exprs)
DATA_KEY_EXPR_ROOT = "robot_ipc"
ENDPOINT_KEY_EXPR_ROOT = "robot_ipc_endpoints"
DOMAIN_TAG_CHUNK_PREFIX = "tag_"
# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/KeyExprs.h:key_exprs)

# The chunks between the domain prefix and the topic: a data key's profile, and a token key's role, profile and bus
# id.
_DATA_KEY_CHUNKS = 1
_TOKEN_KEY_CHUNKS = 3


class EndpointRole(str, enum.Enum):
    """Whether a liveliness token stands for a publisher or a subscriber."""

    PUBLISHER = "publisher"
    SUBSCRIBER = "subscriber"


@dataclasses.dataclass(frozen=True)
class DataKey:
    """The profile and topic of a data key expression.

    Attributes:
        profile: the topic's QoS profile.
        topic: the bus topic.
    """

    profile: qos_profile.QosProfile
    topic: str


@dataclasses.dataclass(frozen=True)
class EndpointToken:
    """One publisher or subscriber of a bus, as its liveliness token names it.

    Attributes:
        role: publisher or subscriber.
        profile: the QoS profile its bus gives the topic.
        bus_id: the bus that declared it, a random id in hexadecimal.
        topic: the bus topic.
    """

    role: EndpointRole
    profile: qos_profile.QosProfile
    bus_id: str
    topic: str


def _domain_prefix(root: str, network: network_config.NetworkConfig) -> str:
    return f"{root}/{network.domain_id}/{DOMAIN_TAG_CHUNK_PREFIX}{network.domain_tag}"


def _split_after(key_expr: str, prefix: str, chunks: int) -> list[str] | None:
    """The chunks of `key_expr` after `prefix`: `chunks` of them and the rest; None when it has fewer."""
    if not key_expr.startswith(prefix + "/"):
        return None
    parts = key_expr[len(prefix) + 1 :].split("/", chunks)
    if len(parts) != chunks + 1 or not parts[-1]:
        return None
    return parts


def data_key_expr(network: network_config.NetworkConfig, topic: str) -> str:
    """The key expression `topic`'s samples are put on in `network`."""
    profile = network.qos_profile_of(topic)
    return (
        f"{_domain_prefix(DATA_KEY_EXPR_ROOT, network)}/{profile.value}/{topic}"
    )


def all_topics_key_expr(network: network_config.NetworkConfig) -> str:
    """The key expression of every topic of `network`, of every profile."""
    return f"{_domain_prefix(DATA_KEY_EXPR_ROOT, network)}/*/**"


def parse_data_key_expr(
    network: network_config.NetworkConfig, key_expr: str
) -> DataKey | None:
    """The profile and topic of a data key expression of `network`; None for another domain or tag, or no profile."""
    parts = _split_after(
        key_expr, _domain_prefix(DATA_KEY_EXPR_ROOT, network), _DATA_KEY_CHUNKS
    )
    if parts is None:
        return None
    try:
        profile = qos_profile.QosProfile(parts[0])
    except ValueError:
        return None
    return DataKey(profile=profile, topic=parts[1])


def endpoint_token_key_expr(
    network: network_config.NetworkConfig, token: EndpointToken
) -> str:
    """The key expression of the liveliness token `token` in `network`."""
    return (
        f"{_domain_prefix(ENDPOINT_KEY_EXPR_ROOT, network)}/{token.role.value}/"
        f"{token.profile.value}/{token.bus_id}/{token.topic}"
    )


def all_endpoint_tokens_key_expr(network: network_config.NetworkConfig) -> str:
    """The key expression of every liveliness token of `network`."""
    return f"{_domain_prefix(ENDPOINT_KEY_EXPR_ROOT, network)}/**"


def parse_endpoint_token_key_expr(
    network: network_config.NetworkConfig, key_expr: str
) -> EndpointToken | None:
    """The token of a key expression from endpoint_token_key_expr(); None for another domain or tag, or no role."""
    parts = _split_after(
        key_expr,
        _domain_prefix(ENDPOINT_KEY_EXPR_ROOT, network),
        _TOKEN_KEY_CHUNKS,
    )
    if parts is None or not parts[2]:
        return None
    try:
        role = EndpointRole(parts[0])
        profile = qos_profile.QosProfile(parts[1])
    except ValueError:
        return None
    return EndpointToken(
        role=role, profile=profile, bus_id=parts[2], topic=parts[3]
    )


__all__ = [
    "DATA_KEY_EXPR_ROOT",
    "DOMAIN_TAG_CHUNK_PREFIX",
    "ENDPOINT_KEY_EXPR_ROOT",
    "DataKey",
    "EndpointRole",
    "EndpointToken",
    "all_endpoint_tokens_key_expr",
    "all_topics_key_expr",
    "data_key_expr",
    "endpoint_token_key_expr",
    "parse_data_key_expr",
    "parse_endpoint_token_key_expr",
]
