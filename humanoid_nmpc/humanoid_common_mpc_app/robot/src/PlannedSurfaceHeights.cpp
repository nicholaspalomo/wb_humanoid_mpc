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

#include "humanoid_common_mpc_app/robot/PlannedSurfaceHeights.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"

namespace ocs2::humanoid {
namespace {

constexpr Eigen::Index kPositionDim = 3;
/** des_pose_pad_rt_world: the pad's center, then its normal. */
constexpr Eigen::Index kPadPoseDim = 2 * kPositionDim;
/** How far the planned normal's norm may be from 1, as the MPC accepts it
 * (PlannedMotion). */
constexpr scalar_t kUnitNormalTolerance = 1.0e-3;

/** A planned pad: its center and its unit normal, in the world. */
struct Pad {
  vector3_t posSurfaceRtWorldEwrtWorld = vector3_t::Zero();
  vector3_t unitNormalSurfaceEwrtWorld = vector3_t::UnitZ();
};

/**
 * Returns the pad of `phase`, plan phase `index`: the center and the
 * normalized normal of its des_pose_pad_rt_world, or InvalidArgument naming
 * the phase and its contact when that is not six finite values with a unit
 * normal.
 */
absl::StatusOr<Pad> padOfPhase(
    const multi_contact::config::MultiContactPlan::ContactPhase& phase,
    size_t index) {
  if (phase.des_pose_pad_rt_world.size() != kPadPoseDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[PlannedSurfaceHeights] phase ", index, " of '", phase.contact,
        "' has a des_pose_pad_rt_world of ", phase.des_pose_pad_rt_world.size(),
        " values, not ", kPadPoseDim, " (its center, then its normal)."));
  }
  const vector3_t posSurfaceRtWorldEwrtWorld =
      phase.des_pose_pad_rt_world.head<kPositionDim>();
  const vector3_t normalSurfaceEwrtWorld =
      phase.des_pose_pad_rt_world.tail<kPositionDim>();
  if (!posSurfaceRtWorldEwrtWorld.allFinite() ||
      !normalSurfaceEwrtWorld.allFinite() ||
      std::abs(normalSurfaceEwrtWorld.norm() - 1.0) > kUnitNormalTolerance) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[PlannedSurfaceHeights] phase ", index, " of '", phase.contact,
        "' has a pad center that is not finite or a normal that is not a "
        "unit vector."));
  }
  return Pad{.posSurfaceRtWorldEwrtWorld = posSurfaceRtWorldEwrtWorld,
             .unitNormalSurfaceEwrtWorld = normalSurfaceEwrtWorld.normalized()};
}

}  // namespace

absl::StatusOr<std::unique_ptr<PlannedSurfaceHeights>>
PlannedSurfaceHeights::Create(
    const multi_contact::config::MultiContactPlan& plan,
    const ModelSettings& modelSettings) {
  const scalar_t minNormalSurfaceEwrtWorldZ = std::cos(kMaxSurfaceTilt);
  std::vector<std::vector<Patch>> patches(modelSettings.numContacts());
  for (size_t index = 0; index < plan.phases.size(); ++index) {
    const multi_contact::config::MultiContactPlan::ContactPhase& phase =
        plan.phases[index];
    const std::vector<std::string>::const_iterator name =
        std::find(modelSettings.contactNames.begin(),
                  modelSettings.contactNames.end(), phase.contact);
    if (name == modelSettings.contactNames.end()) continue;
    const size_t contact = static_cast<size_t>(
        std::distance(modelSettings.contactNames.begin(), name));
    ASSIGN_OR_RETURN(const Pad pad, padOfPhase(phase, index));
    std::vector<Patch>& contactPatches = patches[contact];
    if (hasPatch(contactPatches, pad.posSurfaceRtWorldEwrtWorld,
                 pad.unitNormalSurfaceEwrtWorld)) {
      continue;
    }
    contactPatches.push_back(Patch{
        .posSurfaceRtWorldEwrtWorld = pad.posSurfaceRtWorldEwrtWorld,
        .normalSurfaceEwrtWorld = pad.unitNormalSurfaceEwrtWorld,
        .givesHeight =
            pad.unitNormalSurfaceEwrtWorld.z() >= minNormalSurfaceEwrtWorldZ});
  }
  return absl::WrapUnique(new PlannedSurfaceHeights(std::move(patches)));
}

bool PlannedSurfaceHeights::hasPatch(
    absl::Span<const Patch> patches,
    const vector3_t& posSurfaceRtWorldEwrtWorld,
    const vector3_t& normalSurfaceEwrtWorld) {
  return std::any_of(patches.begin(), patches.end(), [&](const Patch& patch) {
    return patch.posSurfaceRtWorldEwrtWorld == posSurfaceRtWorldEwrtWorld &&
           patch.normalSurfaceEwrtWorld == normalSurfaceEwrtWorld;
  });
}

void PlannedSurfaceHeights::fill(
    absl::Span<const vector3_t> estPosContactsRtWorldEwrtWorld,
    estimation::ContactEstimationInputs& inputs) const {
  const size_t numContacts = inputs.numContacts();
  const bool haveEstimates =
      estPosContactsRtWorldEwrtWorld.size() == numContacts;
  for (size_t contact = 0; contact < numContacts; ++contact) {
    inputs.clearNomPosContactRtWorldEwrtWorldZ(contact);
    if (!haveEstimates || contact >= patchesByContact_.size()) continue;
    const vector3_t& estPosContactRtWorldEwrtWorld =
        estPosContactsRtWorldEwrtWorld[contact];
    const Patch* absl_nullable nearest = nullptr;
    scalar_t nearestSquaredDistance = 0.0;
    for (const Patch& patch : patchesByContact_[contact]) {
      const scalar_t squaredDistance =
          (estPosContactRtWorldEwrtWorld - patch.posSurfaceRtWorldEwrtWorld)
              .squaredNorm();
      if (nearest == nullptr || squaredDistance < nearestSquaredDistance) {
        nearest = &patch;
        nearestSquaredDistance = squaredDistance;
      }
    }
    if (nearest == nullptr || !nearest->givesHeight ||
        !(nearestSquaredDistance <
          kMaxSurfaceMatchDistance * kMaxSurfaceMatchDistance)) {
      continue;
    }
    const vector3_t& posSurfaceRtWorldEwrtWorld =
        nearest->posSurfaceRtWorldEwrtWorld;
    const vector3_t& normalSurfaceEwrtWorld = nearest->normalSurfaceEwrtWorld;
    // The patch's plane n . (p - c) = 0 at the contact's x and y.
    inputs.setNomPosContactRtWorldEwrtWorldZ(
        contact,
        posSurfaceRtWorldEwrtWorld.z() -
            (normalSurfaceEwrtWorld.x() * (estPosContactRtWorldEwrtWorld.x() -
                                           posSurfaceRtWorldEwrtWorld.x()) +
             normalSurfaceEwrtWorld.y() * (estPosContactRtWorldEwrtWorld.y() -
                                           posSurfaceRtWorldEwrtWorld.y())) /
                normalSurfaceEwrtWorld.z());
  }
}

}  // namespace ocs2::humanoid
