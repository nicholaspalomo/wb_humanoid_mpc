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

#include "humanoid_common_mpc_app/teleop/VelocityCommandRepeater.h"

#include <functional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"

#include "humanoid_mpc_ipc/Topics.h"

namespace ocs2::humanoid::teleop {

absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>> VelocityCommandRepeater::Create(robot::ipc::Bus& bus, absl::Duration period) {
  const std::shared_ptr<State> state = std::make_shared<State>();
  robot::ipc::Bus* busPointer = &bus;
  const absl::Status registered = bus.addPeriodicCallback(period, [state, busPointer]() {
    std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> command;
    {
      absl::MutexLock lock(state->mutex);
      command = state->command;
    }
    if (!command.has_value()) return;
    const absl::Status sent = busPointer->publishFromIoThread(ipc::topics::kOperatorWalkingVelocityCommand, *command);
    if (sent.ok()) {
      state->published.fetch_add(1);
    } else {
      LOG_EVERY_N_SEC(WARNING, 5.0) << "[teleop] Publishing the walking command failed: " << sent.message();
    }
  });
  if (!registered.ok()) return registered;
  return std::unique_ptr<VelocityCommandRepeater>(new VelocityCommandRepeater(state));
}

void VelocityCommandRepeater::setCommand(const humanoid_mpc_msgs::WalkingVelocityCommand& command) {
  absl::MutexLock lock(state_->mutex);
  state_->command = command;
}

}  // namespace ocs2::humanoid::teleop
