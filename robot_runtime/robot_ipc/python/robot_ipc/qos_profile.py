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


"""The quality of service of a topic, chosen by name in the network file: the twin of robot_ipc/QosProfile.h.

Zenoh sends every sample over TCP; the profiles differ in what a publisher does when its link cannot take a sample at
once, and in its priority. No profile keeps a history: a subscriber gets what is put after it was declared.

    latest           drop, data priority, express          observations, robot state, velocity commands, telemetry
    reliable_latest  drop, data priority, batched          policies, MPC status, FSM state, parameter documents
    reliable_events  block, interactive priority, express  FSM commands, throws, configuration saves (the default)
"""

import dataclasses
import enum


# LINT.IfChange(qos_profile_names)
class QosProfile(str, enum.Enum):
    """A QoS profile, by its name in the network file."""

    LATEST = "latest"
    RELIABLE_LATEST = "reliable_latest"
    RELIABLE_EVENTS = "reliable_events"


# LINT.ThenChange(//robot_runtime/robot_ipc/src/QosProfile.cpp:qos_profile_names)


class CongestionControl(enum.Enum):
    """What a Zenoh publisher does with a sample its link cannot take at once."""

    DROP = "drop"
    BLOCK = "block"


class Priority(enum.Enum):
    """The Zenoh priority of a profile's samples."""

    INTERACTIVE_HIGH = "interactive_high"
    DATA = "data"


@dataclasses.dataclass(frozen=True)
class QosPolicy:
    """The Zenoh publisher options of a profile.

    Attributes:
        congestion_control: DROP or BLOCK.
        priority: the samples' priority.
        express: sent at once, not batched.
    """

    congestion_control: CongestionControl
    priority: Priority
    express: bool


# LINT.IfChange(qos_policies)
_POLICIES = {
    QosProfile.LATEST: QosPolicy(
        CongestionControl.DROP, Priority.DATA, express=True
    ),
    QosProfile.RELIABLE_LATEST: QosPolicy(
        CongestionControl.DROP, Priority.DATA, express=False
    ),
    QosProfile.RELIABLE_EVENTS: QosPolicy(
        CongestionControl.BLOCK, Priority.INTERACTIVE_HIGH, express=True
    ),
}
# LINT.ThenChange(//robot_runtime/robot_ipc/src/QosProfile.cpp:qos_policies)


def parse_qos_profile(value: "str | QosProfile") -> QosProfile:
    """The profile of that name.

    Args:
        value: "latest", "reliable_latest" or "reliable_events", or a QosProfile, which is returned as it is.

    Returns:
        The QosProfile that `value` names.

    Raises:
        ValueError: an unknown name; the message lists the valid ones.
    """
    if isinstance(value, QosProfile):
        return value
    try:
        return QosProfile(value)
    except ValueError:
        valid = ", ".join(profile.value for profile in QosProfile)
        raise ValueError(
            f"unknown QoS profile '{value}' (valid: {valid})"
        ) from None


def qos_policy(profile: QosProfile) -> QosPolicy:
    """The Zenoh publisher options of `profile`."""
    return _POLICIES[profile]


__all__ = [
    "CongestionControl",
    "Priority",
    "QosPolicy",
    "QosProfile",
    "parse_qos_profile",
    "qos_policy",
]
