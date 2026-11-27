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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "legged_centroidal_mpc/constraint/SwingLandingAlignmentConstraint.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_core/constraint/ConstraintOrder.h"
#include "ocs2_core/constraint/StateConstraintCppAd.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"   // IWYU pragma: keep
#include "pinocchio/multibody/model.hpp"  // IWYU pragma: keep
#include "pinocchio/spatial.hpp"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact/ContactPatchFrame.h"
#include "legged_common_mpc/gait/MotionPhaseDefinition.h"
#include "legged_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::legged {

SwingLandingAlignmentConstraint::SwingLandingAlignmentConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
    size_t contactIndex,
    scalar_t window,
    absl::string_view libraryName,
    const ModelSettings& modelSettings)
    : StateConstraintCppAd(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfaceCppAd_(pinocchioInterface.toCppAd()),
      mpcRobotModelPtr_(&mpcRobotModel),
      contactIndex_(contactIndex),
      numContacts_(modelSettings.contactNames.size()),
      frameName_(modelSettings.contactNames[contactIndex]),
      window_(window) {
  initialize(mpcRobotModelPtr_->getStateDim(), kPositionDim,
             std::string(libraryName), modelSettings.modelFolderCppAd,
             modelSettings.recompileLibrariesCppAd, modelSettings.verboseCppAd);
}

SwingLandingAlignmentConstraint::SwingLandingAlignmentConstraint(
    const SwingLandingAlignmentConstraint& other) = default;

std::optional<scalar_t> SwingLandingAlignmentConstraint::timeToTouchDown(
    scalar_t time) const {
  const ModeSchedule& schedule = referenceManagerPtr_->getModeSchedule();
  // modeSequence[k] holds before eventTimes[k] and from eventTimes[k - 1].
  const std::vector<scalar_t>& events = schedule.eventTimes;
  const size_t current = static_cast<size_t>(
      std::upper_bound(events.begin(), events.end(), time) - events.begin());
  if (contactIsActiveInMode(schedule.modeSequence[current], contactIndex_,
                            numContacts_)) {
    return std::nullopt;
  }
  for (size_t k = current + 1; k < schedule.modeSequence.size(); ++k) {
    if (contactIsActiveInMode(schedule.modeSequence[k], contactIndex_,
                              numContacts_)) {
      return events[k - 1] - time;
    }
  }
  return std::nullopt;
}

bool SwingLandingAlignmentConstraint::isActive(scalar_t time) const {
  const std::optional<scalar_t> remaining = timeToTouchDown(time);
  return remaining.has_value() && *remaining <= window_;
}

vector_t SwingLandingAlignmentConstraint::getParameters(
    scalar_t time, const PreComputation& /*preComputation*/) const {
  return getNormalSurfaceEwrtWorld(
      referenceManagerPtr_->getContactPatchFrame(contactIndex_, time));
}

ad_vector_t SwingLandingAlignmentConstraint::constraintFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& parameters) const {
  const PinocchioInterfaceCppAd::Model& model =
      pinocchioInterfaceCppAd_.getModel();
  PinocchioInterfaceCppAd::Data data = pinocchioInterfaceCppAd_.getData();
  const ad_vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(model, data, q);
  const pinocchio::SE3Tpl<ad_scalar_t>& poseContactRtWorld =
      pinocchio::updateFramePlacement(model, data,
                                      model.getFrameId(frameName_));
  const ad_vector3_t zAxisContactEwrtWorld =
      poseContactRtWorld.rotation().col(kZAxisIndex);
  const ad_vector3_t desNormalSurfaceEwrtWorld =
      parameters.head<kPositionDim>();
  return zAxisContactEwrtWorld.cross(desNormalSurfaceEwrtWorld);
}

}  // namespace ocs2::legged
