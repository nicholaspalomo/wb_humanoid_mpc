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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "robot_realtime/SpscQueue.h"

namespace ocs2::humanoid {

/**
 * The telemetry of the realtime loop: every `decimation`-th cycle the realtime thread copies the robot state, the joint
 * action it applied, the measured contact flags and the measured foot forces into a preallocated slot of a
 * robot::realtime::SpscQueue (an msgs::RobotStateSample, the nproto struct of robot/state), and the communication
 * thread drains the queue, converts each sample with nproto's ToProto() and hands it to the telemetry sinks (the bus
 * publishes robot/state).
 *
 * Every slot is built at construction with the sample's shape - one entry per joint, the joint names filled in, one
 * entry per contact - so sample() writes numbers into storage that already exists: it allocates nothing, takes no lock
 * and never waits. The joint names are written once, at construction; the realtime thread never touches them. A sample
 * taken while the queue is full is dropped and counted (LoopTiming.telemetry_samples_dropped).
 */
class TelemetrySampler {
 public:
  struct Config {
    /** The robot's joints, by joint index (jointNamesByIndex()). */
    std::vector<std::string> jointNames;
    /** One sample every this many cycles: round(mpc.mrt_desired_frequency / telemetry_frequency), at least 1. */
    size_t decimation = 1;
    /** Slots of the queue: a second of telemetry at 100 Hz rides out a stalled communication thread. */
    size_t capacity = 128;
  };

  explicit TelemetrySampler(const Config& config);

  TelemetrySampler(const TelemetrySampler&) = delete;
  TelemetrySampler& operator=(const TelemetrySampler&) = delete;
  ~TelemetrySampler() = default;

  /**
   * Realtime thread, once per cycle: counts the cycle and, every decimation-th one, copies it into a slot. True when a
   * sample was taken (or dropped because the queue was full).
   */
  bool sample(const robot::model::RobotState& robotState,
              const robot::model::RobotJointAction& jointAction,
              absl::string_view controlMode,
              const contact_flag_t& measuredContactFlags,
              const std::array<vector3_t, kNumContacts>& measuredContactForces);

  /** Communication thread: hands every sample taken so far to `consumer`, oldest first; returns how many. */
  size_t drain(const std::function<void(const msgs::RobotStateSample& sample)>& consumer);

  /** Samples dropped because the queue was full. Any thread. */
  uint64_t dropped() const { return queue_.droppedCount(); }
  /** Samples taken so far, the dropped ones included. Any thread. */
  uint64_t samplesTaken() const { return samplesTaken_.load(std::memory_order_relaxed); }

  size_t decimation() const { return decimation_; }

  /** A sample with the shape every slot has: `jointNames`, and zeros. */
  static msgs::RobotStateSample prototype(const std::vector<std::string>& jointNames);

 private:
  const size_t decimation_;
  const size_t numJoints_;
  robot::realtime::SpscQueue<msgs::RobotStateSample> queue_;
  size_t cycle_ = 0;  // realtime thread
  std::atomic<uint64_t> samplesTaken_{0};
};

}  // namespace ocs2::humanoid
