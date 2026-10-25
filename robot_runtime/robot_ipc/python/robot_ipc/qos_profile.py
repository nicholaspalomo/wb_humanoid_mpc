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


"""The DDS quality of service of a topic, chosen by name in the network file: the twin of robot_ipc/QosProfile.h.

Every profile is VOLATILE: a reader gets what is written after it matched the writer, never a history.

    latest           BEST_EFFORT, KEEP_LAST 1      observations, robot state, velocity commands, telemetry
    reliable_latest  RELIABLE, KEEP_LAST 1         policies, MPC status, FSM state, parameter documents
    reliable_events  RELIABLE, KEEP_LAST 64        FSM commands, throws, configuration saves (the default)
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


class Reliability(enum.Enum):
    """Whether DDS sends a lost sample (or fragment) again."""

    BEST_EFFORT = "best_effort"
    RELIABLE = "reliable"


@dataclasses.dataclass(frozen=True)
class QosPolicy:
    """The DDS policies of a profile.

    Attributes:
        reliability: BEST_EFFORT or RELIABLE.
        writer_history_depth: the writer's KEEP_LAST depth.
    """

    reliability: Reliability
    writer_history_depth: int


# LINT.IfChange(qos_profiles)
RELIABLE_EVENTS_HISTORY_DEPTH = 64
_POLICIES = {
    QosProfile.LATEST: QosPolicy(Reliability.BEST_EFFORT, 1),
    QosProfile.RELIABLE_LATEST: QosPolicy(Reliability.RELIABLE, 1),
    QosProfile.RELIABLE_EVENTS: QosPolicy(
        Reliability.RELIABLE, RELIABLE_EVENTS_HISTORY_DEPTH
    ),
}
# LINT.ThenChange(//robot_runtime/robot_ipc/include/robot_ipc/QosProfile.h:qos_profiles)


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
    """The DDS policies of `profile`."""
    return _POLICIES[profile]


__all__ = [
    "RELIABLE_EVENTS_HISTORY_DEPTH",
    "QosPolicy",
    "QosProfile",
    "Reliability",
    "parse_qos_profile",
    "qos_policy",
]
