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

#include "humanoid_common_mpc/contact_planning/execution/PlannedHeightOverride.h"

#include <cmath>
#include <sstream>

namespace ocs2::humanoid {

std::string PlannedHeightOverride::describe() const {
  return "the base height reference of the MPC target follows the plan's rise and fall, z_plan(t) - z_plan(now), and only "
         "for a plan that leaves the ground";
}

void PlannedHeightOverride::overrideTarget(const ExecutionContext& ctx, TargetTrajectories& targetTrajectories) const {
  const size_t n = targetTrajectories.timeTrajectory.size();
  if (n == 0 || targetTrajectories.stateTrajectory.size() != n) return;
  // Only a plan that leaves the ground says anything about the height; a walk must not touch the reference at all.
  const ContactPlan* plan =
      (ctx.hasPlan() && ctx.activePlan->hasHeight() && ctx.activePlan->numFlightIntervals() > 0) ? ctx.activePlan : nullptr;
  const std::optional<scalar_t> heightNow = plan != nullptr ? plan->heightAtTime(ctx.time) : std::nullopt;
  const auto readable = [this](const vector_t& state) { return state.size() >= static_cast<Eigen::Index>(mpcRobotModel_->getStateDim()); };

  // Is this the trajectory the last call wrote into, unchanged since? Then its heights still carry the offsets recorded
  // then, and those come out before the current ones go in.
  constexpr scalar_t kSame = 1e-9;
  bool carriesTheLastOffsets = appliedTimes_.size() == n && appliedOffsets_.size() == n && appliedHeights_.size() == n;
  for (size_t i = 0; carriesTheLastOffsets && i < n; ++i) {
    carriesTheLastOffsets =
        readable(targetTrajectories.stateTrajectory[i]) && std::abs(targetTrajectories.timeTrajectory[i] - appliedTimes_[i]) < kSame &&
        std::abs(mpcRobotModel_->getBasePosition(targetTrajectories.stateTrajectory[i])(2) - appliedHeights_[i]) < kSame;
  }

  scalar_array_t offsets(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    if (!readable(targetTrajectories.stateTrajectory[i])) continue;
    if (plan != nullptr) {
      const std::optional<scalar_t> height = plan->heightAtTime(targetTrajectories.timeTrajectory[i]);
      if (height.has_value() && heightNow.has_value()) offsets[i] = *height - *heightNow;
    }
    const scalar_t change = offsets[i] - (carriesTheLastOffsets ? appliedOffsets_[i] : 0.0);
    if (change != 0.0) mpcRobotModel_->adaptBasePoseHeight(targetTrajectories.stateTrajectory[i], change);
  }

  appliedTimes_ = targetTrajectories.timeTrajectory;
  appliedOffsets_ = std::move(offsets);
  appliedHeights_.assign(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    if (readable(targetTrajectories.stateTrajectory[i])) {
      appliedHeights_[i] = mpcRobotModel_->getBasePosition(targetTrajectories.stateTrajectory[i])(2);
    }
  }
}

}  // namespace ocs2::humanoid
