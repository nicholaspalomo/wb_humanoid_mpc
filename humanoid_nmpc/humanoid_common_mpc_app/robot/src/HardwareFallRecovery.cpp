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

#include "humanoid_common_mpc_app/robot/HardwareFallRecovery.h"

#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/mrt/ControlMode.h"
#include "humanoid_common_mpc_app/robot/SimFallRecovery.h"

namespace ocs2::humanoid {

HardwareFallRecovery::HardwareFallRecovery(
    scalar_t maxBaseTiltAngle,
    RobotBackend* absl_nonnull backend,
    RealtimeEventLog* absl_nullable eventLog)
    : maxBaseTiltAngle_(maxBaseTiltAngle),
      backend_(backend),
      eventLog_(eventLog) {}

FallRecovery::Cycle HardwareFallRecovery::update(
    const robot::model::RobotState& robotState,
    const std::vector<scalar_t>& /*nominalJointPositions*/,
    std::string& currentModeName) {
  Cycle cycle;
  if (!(maxBaseTiltAngle_ > 0.0) || !backend_->acceptsJointAction()) {
    return cycle;
  }
  const scalar_t tilt = SimFallRecovery::baseTiltAngle(
      robotState.getRootRotationLocalToWorldFrame());
  if (!(tilt > maxBaseTiltAngle_)) {
    return cycle;
  }
  backend_->enterSafeState();
  currentModeName.assign(control_mode::kZeroTorque.data(),
                         control_mode::kZeroTorque.size());
  cycle.modeChanged = true;
  cycle.tilt = tilt;
  if (eventLog_ != nullptr) {
    eventLog_->post(RealtimeEventCode::kFallTorquesOff, /*detail=*/0,
                    /*text=*/{}, tilt, maxBaseTiltAngle_);
  }
  return cycle;
}

}  // namespace ocs2::humanoid
