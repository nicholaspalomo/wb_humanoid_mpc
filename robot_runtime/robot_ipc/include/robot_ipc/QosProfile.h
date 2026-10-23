/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#pragma once

#include <array>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace robot::ipc {

/**
 * The DDS quality of service of a topic, chosen by name in the network file (NetworkConfig::topics, e.g.
 * `topics { name: "mpc/policy" qos_profile: "reliable_latest" }`). Every profile is VOLATILE: a reader gets what is
 * written after it matched the writer, never a history, so a process that restarts is not handed old commands. The
 * publisher and the subscribers of a topic read the profile from the same network file; a reader that asks for
 * RELIABLE does not match a BEST_EFFORT writer, which both sides count (TopicStatistics::incompatibleQos).
 */
enum class QosProfile {
  /**
   * "latest": BEST_EFFORT, KEEP_LAST 1. Streams whose newest sample is all that matters and that come again within a
   * few milliseconds: observations, robot state, velocity commands, telemetry. A lost sample is replaced by the next
   * one instead of being sent again, so a lossy link adds no latency.
   */
  kLatest,
  /**
   * "reliable_latest": RELIABLE, KEEP_LAST 1. Large or infrequent latest values: MPC policies (0.5-0.75 MB, hundreds
   * of UDP fragments, so that one lost fragment would lose the whole policy under BEST_EFFORT), the MPC status, the FSM
   * state and the operator's parameter documents. Lost fragments are sent again, but a newer sample replaces one that
   * is still being repaired, so a slow link never builds a backlog.
   */
  kReliableLatest,
  /**
   * "reliable_events": RELIABLE, KEEP_LAST kReliableEventsHistoryDepth. Discrete events that must each arrive: FSM
   * commands, dodgeball throws, configuration saves and their status. The default of a topic the network file does
   * not list.
   */
  kReliableEvents,
};

/** Whether DDS sends a lost sample (or fragment) again. */
enum class Reliability {
  kBestEffort,
  kReliable,
};

/** The DDS policies of a QosProfile. */
struct QosPolicy {
  Reliability reliability = Reliability::kReliable;
  /** The writer's KEEP_LAST depth: the samples it keeps for readers that have not acknowledged them yet. */
  int writerHistoryDepth = 1;
};

// The Python bus has the same profiles (robot_ipc/qos_profile.py).
// LINT.IfChange(qos_profiles)
/** The writer history of kReliableEvents: events written faster than a reader acknowledges them beyond it are lost. */
inline constexpr int kReliableEventsHistoryDepth = 64;
/** Every profile, in the order of the enum. */
inline constexpr std::array<QosProfile, 3> kAllQosProfiles = {QosProfile::kLatest, QosProfile::kReliableLatest,
                                                              QosProfile::kReliableEvents};
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/qos_profile.py:qos_profiles)

/** The name of a profile in the network file: "latest", "reliable_latest" or "reliable_events". */
absl::string_view qosProfileName(QosProfile profile);

/** The profile of that name; an unknown name is an InvalidArgument error that lists the valid ones. */
absl::StatusOr<QosProfile> parseQosProfile(absl::string_view name);

/** The DDS policies of `profile`. */
QosPolicy qosPolicy(QosProfile profile);

}  // namespace robot::ipc
