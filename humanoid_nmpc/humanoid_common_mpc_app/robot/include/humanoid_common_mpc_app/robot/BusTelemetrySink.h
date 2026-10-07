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

#include <cstdint>

#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/robot/TelemetrySink.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid {

/**
 * The `bus` telemetry sink: publishes every sample on robot/state, where the MPC node's visualization publisher, the
 * Rerun bridge and `ipc_tool echo robot/state` read it. Publishes from the bus's IO thread, so it must be written to from
 * a handler or a periodic callback of `bus`.
 */
class BusTelemetrySink final : public TelemetrySink {
 public:
  explicit BusTelemetrySink(robot::ipc::Bus& bus) : bus_(bus) {}

  absl::string_view name() const override;
  void write(const humanoid_mpc_msgs::RobotStateSample& sample) override;

  /** Samples the bus refused (its send queue full, or not called on its IO thread). */
  uint64_t publishFailures() const { return publishFailures_; }

 private:
  robot::ipc::Bus& bus_;
  uint64_t publishFailures_ = 0;
};

}  // namespace ocs2::humanoid
