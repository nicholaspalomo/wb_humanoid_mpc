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

#include "humanoid_common_mpc/parameter_update/ReferenceManagerApplier.h"

#include <array>
#include <cmath>
#include <memory>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/parameter_update/ContactImplicitApplier.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(reference_manager_fields)
constexpr std::array<absl::string_view, 2> kFields = {
    "terrain_height",
    "swing_trajectory_config",
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto)

}  // namespace

absl::Span<const absl::string_view> ReferenceManagerApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void ReferenceManagerApplier::apply(HotUpdateTarget& target) {
  SwitchedModelReferenceManager* absl_nullable referenceManager = target.referenceManager();
  // Where the ground is: one field, shared with the swing trajectories. The reference manager owns it: the swing
  // trajectories and the landing targets are rebuilt on it at the next solve, and the contact-implicit terms follow it in
  // that same solve - not in this one, whose references were built before this reload. Without a reference manager there
  // are no references to agree with, and the terms take the height at once.
  const scalar_t terrainHeight = target.task().terrain_height;
  if (!std::isfinite(terrainHeight)) {
    target.reportNotApplied("terrain_height",
                            absl::InvalidArgumentError(absl::StrCat("terrain_height is ", terrainHeight, ", which is not finite.")));
  } else if (referenceManager != nullptr) {
    referenceManager->setTerrainHeight(terrainHeight);
  } else {
    setContactImplicitTermsTerrainHeight(target.problems(), target.contactNames(), terrainHeight);
  }

  if (referenceManager == nullptr) return;
  if (const std::optional<SwingTrajectoryPlanner::Config> swingTrajectory = convertedOrReported(
          swingTrajectorySettingsFromConfig(target.task().swing_trajectory_config), target.source(), "swing_trajectory_config")) {
    if (const std::shared_ptr<SwingTrajectoryPlanner>& swingPlanner = referenceManager->getSwingTrajectoryPlanner()) {
      swingPlanner->setConfig(*swingTrajectory);
    }
  }
}

}  // namespace ocs2::humanoid
