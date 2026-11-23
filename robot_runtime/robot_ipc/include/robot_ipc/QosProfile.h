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
 * The quality of service of a topic, chosen by name in the network file
 * (NetworkConfig::topics, e.g. `topics { name: "mpc/policy" qos_profile:
 * "reliable_latest" }`). Zenoh sends every sample over the TCP links between
 * the processes, so a sample that leaves a publisher arrives unless its link
 * breaks; the profiles differ in what a publisher does when its link cannot
 * take a sample at once (CongestionControl), and in its priority. No profile
 * keeps a history: a subscriber gets what is put after it was declared, so a
 * process that restarts is not handed old commands. The publisher and the
 * subscribers of a topic read the profile from the same network file, and the
 * profile is a chunk of the topic's key expression (KeyExprs.h): two processes
 * that disagree about it never exchange samples, which both count
 * (TopicStatistics::incompatibleQos).
 */
enum class QosProfile {
  /**
   * "latest": drop under congestion, data priority, sent at once (express).
   * Streams whose newest sample is all that matters and that come again within
   * a few milliseconds: observations, robot state, velocity commands,
   * telemetry. A sample the link cannot take is dropped instead of queued, so a
   * slow link adds no latency.
   */
  kLatest,
  /**
   * "reliable_latest": drop under congestion, data priority, batched. Large or
   * infrequent latest values: MPC policies (0.5-0.75 MB), the MPC status, the
   * FSM state and the operator's parameter documents. Zenoh waits briefly for
   * room before it drops a sample, so a policy crosses a busy link whole, while
   * a link too slow for the rate drops samples rather than building a backlog.
   */
  kReliableLatest,
  /**
   * "reliable_events": block under congestion, interactive priority, sent at
   * once. Discrete events that must each arrive: FSM commands, dodgeball
   * throws, configuration saves and their status. The default of a topic the
   * network file does not list. A put waits for room in the link's queue, on
   * the IO thread; a broken link is closed when its lease runs out.
   */
  kReliableEvents,
};

/** What a Zenoh publisher does with a sample its link cannot take at once. */
enum class CongestionControl {
  kDrop,
  kBlock,
};

/** The Zenoh priority of a profile's samples, highest first. */
enum class Priority {
  kInteractiveHigh,
  kData,
};

/** The Zenoh publisher options of a QosProfile. */
struct QosPolicy {
  CongestionControl congestionControl = CongestionControl::kBlock;
  Priority priority = Priority::kData;
  /** Sent at once, not batched with other samples. */
  bool express = false;
};

// The Python bus has the same profiles (robot_ipc/qos_profile.py).
// LINT.IfChange(qos_profiles)
/** Every profile, in the order of the enum. */
inline constexpr std::array<QosProfile, 3> kAllQosProfiles = {
    QosProfile::kLatest, QosProfile::kReliableLatest,
    QosProfile::kReliableEvents};
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/qos_profile.py:qos_profile_names)

/** The name of a profile in the network file: "latest", "reliable_latest" or
 * "reliable_events". */
absl::string_view qosProfileName(QosProfile profile);

/** The profile of that name; an unknown name is an InvalidArgument error that
 * lists the valid ones. */
absl::StatusOr<QosProfile> parseQosProfile(absl::string_view name);

/** The Zenoh publisher options of `profile`. */
QosPolicy qosPolicy(QosProfile profile);

}  // namespace robot::ipc
