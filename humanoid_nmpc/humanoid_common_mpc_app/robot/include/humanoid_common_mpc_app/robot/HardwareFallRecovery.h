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

#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/FallRecovery.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/RobotBackend.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid {

/**
 * The fall recovery of a robot without a simulator (the `hardware` supervisor
 * of RobotSupervisorRegistry). Nothing catches a falling robot and nothing
 * teleports it, so update() reports no discontinuity and never relocates the
 * base; what it does is the last line of defense against a fall under torque:
 * once the base tilts past `maxBaseTiltAngle` (task file
 * `sim_max_base_tilt_angle`; <= 0 disables it) while the torques are on, it
 * puts the backend in its safe state (enterSafeState(): its damping) and the
 * mode in ZERO_TORQUE, and reports it once. An FSM command switches the torques
 * on again. Realtime thread: no lock, no log line, no allocation.
 */
class HardwareFallRecovery final : public FallRecovery {
 public:
  /** `backend` and `eventLog` must outlive the recovery; `eventLog` may be
   * nullptr. */
  HardwareFallRecovery(scalar_t maxBaseTiltAngle,
                       RobotBackend* absl_nonnull backend,
                       RealtimeEventLog* absl_nullable eventLog = nullptr);

  HardwareFallRecovery(const HardwareFallRecovery&) = delete;
  HardwareFallRecovery& operator=(const HardwareFallRecovery&) = delete;
  ~HardwareFallRecovery() override = default;

  Cycle update(const robot::model::RobotState& robotState,
               const std::vector<scalar_t>& nominalJointPositions,
               std::string& currentModeName) override;

 private:
  const scalar_t maxBaseTiltAngle_;
  RobotBackend* absl_nonnull backend_;
  RealtimeEventLog* absl_nullable const eventLog_;
};

}  // namespace ocs2::humanoid
