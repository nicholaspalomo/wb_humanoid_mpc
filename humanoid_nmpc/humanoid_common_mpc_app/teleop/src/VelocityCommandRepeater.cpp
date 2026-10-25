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

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"

#include "humanoid_mpc_ipc/Topics.h"

namespace ocs2::humanoid::teleop {
namespace {

/** [s] How often a failing publication of the walking command, or a reference
 * file that no longer reads, is logged. */
constexpr double kPublishFailureLogPeriodSeconds = 5.0;
/** How often publishStopCommand() looks whether the bus has sent the stop
 * command. */
constexpr absl::Duration kStopCommandPollPeriod = absl::Milliseconds(5);

}  // namespace

absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>>
VelocityCommandRepeater::Create(robot::ipc::Bus& bus,
                                std::string referenceFile,
                                const KeyboardCommandLimits& limits,
                                absl::Duration period,
                                absl::Duration limitsCheckPeriod) {
  if (period <= absl::ZeroDuration() ||
      limitsCheckPeriod <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError(
        "The repeater's period and the command limits' check period must be "
        "positive");
  }
  const std::shared_ptr<State> state = std::make_shared<State>();
  {
    absl::MutexLock lock(state->mutex);
    state->limits = limits;
  }
  state->referenceFile = std::move(referenceFile);
  absl::Duration remainder;
  const int64_t callbacksPerLimitsCheck =
      absl::IDivDuration(limitsCheckPeriod, period, &remainder);
  state->callbacksPerLimitsCheck =
      callbacksPerLimitsCheck > 0 ? callbacksPerLimitsCheck : 1;
  state->callbacksUntilLimitsCheck = state->callbacksPerLimitsCheck;
  robot::ipc::Bus* absl_nonnull busPointer = &bus;
  absl::Status registered = bus.addPeriodicCallback(
      period,
      [state, busPointer]() { publishFromIoThread(*state, *busPointer); });
  if (!registered.ok()) return registered;
  return absl::WrapUnique(new VelocityCommandRepeater(state));
}

void VelocityCommandRepeater::reloadLimitsIfChanged(State& state) {
  std::error_code error;
  const std::filesystem::file_time_type writeTime =
      std::filesystem::last_write_time(state.referenceFile, error);
  if (error) {
    LOG_EVERY_N_SEC(WARNING, kPublishFailureLogPeriodSeconds)
        << "[teleop] Keeping the previous command limits: cannot read the time "
           "of "
        << state.referenceFile << ": " << error.message();
    return;
  }
  if (state.referenceFileTime.has_value() &&
      *state.referenceFileTime == writeTime)
    return;
  absl::StatusOr<KeyboardCommandLimits> reloaded =
      loadKeyboardCommandLimits(state.referenceFile);
  if (!reloaded.ok()) {
    // The time is not recorded, so that the next look tries again: an editor
    // may have saved half a file.
    LOG_EVERY_N_SEC(WARNING, kPublishFailureLogPeriodSeconds)
        << "[teleop] Keeping the previous command limits: "
        << reloaded.status();
    return;
  }
  state.referenceFileTime = writeTime;
  absl::MutexLock lock(state.mutex);
  state.limits = *std::move(reloaded);
  state.limitsReloads.fetch_add(1);
}

void VelocityCommandRepeater::publishFromIoThread(State& state,
                                                  robot::ipc::Bus& bus) {
  if (--state.callbacksUntilLimitsCheck <= 0) {
    state.callbacksUntilLimitsCheck = state.callbacksPerLimitsCheck;
    reloadLimitsIfChanged(state);
  }
  humanoid_mpc_msgs::WalkingVelocityCommand message;
  uint64_t generation = 0;
  {
    absl::MutexLock lock(state.mutex);
    if (!state.command.has_value()) return;
    message = keyboardCommandToMessage(*state.command, state.limits);
    generation = state.commandGeneration;
  }
  const absl::Status sent = bus.publishFromIoThread(
      ipc::topics::kOperatorWalkingVelocityCommand, message);
  if (!sent.ok()) {
    LOG_EVERY_N_SEC(WARNING, kPublishFailureLogPeriodSeconds)
        << "[teleop] Publishing the walking command failed: " << sent.message();
    return;
  }
  state.published.fetch_add(1);
  absl::MutexLock lock(state.mutex);
  state.publishedGeneration = std::max(state.publishedGeneration, generation);
}

void VelocityCommandRepeater::setCommand(const vector4_t& command,
                                         const KeyboardCommandLimits& limits) {
  absl::MutexLock lock(state_->mutex);
  state_->command = command;
  state_->limits = limits;
  ++state_->commandGeneration;
}

bool VelocityCommandRepeater::publishStopCommand(absl::Duration timeout,
                                                 absl::Duration gracePeriod) {
  uint64_t stopGeneration = 0;
  {
    absl::MutexLock lock(state_->mutex);
    if (!state_->command.has_value()) return false;
    vector4_t stop = vector4_t::Zero();
    stop(KeyboardCommandLimits::kPelvisHeightChangeIndex) =
        (*state_->command)(KeyboardCommandLimits::kPelvisHeightChangeIndex);
    state_->command = stop;
    stopGeneration = ++state_->commandGeneration;
  }
  const absl::Time deadline = absl::Now() + timeout;
  while (true) {
    {
      absl::MutexLock lock(state_->mutex);
      if (state_->publishedGeneration >= stopGeneration) break;
    }
    if (absl::Now() >= deadline) {
      LOG(WARNING) << "[teleop] The stop command was not sent within "
                   << timeout;
      return false;
    }
    absl::SleepFor(kStopCommandPollPeriod);
  }
  absl::SleepFor(gracePeriod);
  return true;
}

}  // namespace ocs2::humanoid::teleop
