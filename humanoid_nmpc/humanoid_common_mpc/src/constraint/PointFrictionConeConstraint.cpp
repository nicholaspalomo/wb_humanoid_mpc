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

#include "humanoid_common_mpc/constraint/PointFrictionConeConstraint.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact/ContactInputJacobian.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"

namespace ocs2::humanoid {

absl::Status PointFrictionConeConstraint::checkPointContactInput(
    const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactIndex) {
  const size_t inputDim = mpcRobotModel.getContactInputDim(contactIndex);
  const size_t forceStart =
      mpcRobotModel.getContactForceStartIndices(contactIndex);
  const size_t blockStart =
      mpcRobotModel.getContactWrenchStartIndices(contactIndex);
  if (inputDim != kForceDim || forceStart != blockStart) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[PointFrictionConeConstraint] contact ", contactIndex,
        " is parameterized by ", inputDim, " inputs starting at ", blockStart,
        ", not by its three-entry force: the point friction cone is the cone "
        "of a 3-DoF contact (model_settings.contacts_3dof) on wrench inputs; "
        "a 6-DoF contact takes the contact wrench cone."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PointFrictionConeConstraint>>
PointFrictionConeConstraint::Create(
    const SwitchedModelReferenceManager& referenceManager,
    size_t contactIndex,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const ContactWrenchConeConstraint::Config& config,
    bool scheduleGated) {
  RETURN_IF_ERROR(checkPointContactInput(mpcRobotModel, contactIndex));
  RETURN_IF_ERROR(ContactWrenchConeConstraint::validateConfig(config));
  // The constructor is private: Create() is the only way to one that passed
  // the checks above.
  return absl::WrapUnique(new PointFrictionConeConstraint(
      referenceManager, contactIndex, mpcRobotModel, config, scheduleGated));
}

PointFrictionConeConstraint::PointFrictionConeConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    size_t contactIndex,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const ContactWrenchConeConstraint::Config& config,
    bool scheduleGated)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      mpcRobotModelPtr_(&mpcRobotModel),
      contactIndex_(contactIndex),
      scheduleGated_(scheduleGated),
      forceInputJacobian_(
          contactForceInputJacobian(mpcRobotModel, contactIndex)) {
  const ContactWrenchConeRows rows = buildFrictionPyramidRows(
      scheduleGated
          ? config
          : ContactWrenchConeConstraint::withoutLoadedFootOffsets(config));
  A_f_ = rows.A_f;
  b_ = rows.b;
}

bool PointFrictionConeConstraint::isActive(scalar_t time) const {
  if (!isActive_) return false;
  if (!scheduleGated_) return true;
  return referenceManagerPtr_->isInContact(time, contactIndex_);
}

vector_t PointFrictionConeConstraint::getValue(
    scalar_t time,
    const vector_t& /*state*/,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const matrix3_t worldToPatch =
      referenceManagerPtr_->getContactPatchFrame(contactIndex_, time)
          .worldToPatchRotation;
  const vector3_t forceInPatch =
      worldToPatch * mpcRobotModelPtr_->getContactForce(input, contactIndex_);
  return A_f_ * forceInPatch + b_;
}

VectorFunctionLinearApproximation
PointFrictionConeConstraint::getLinearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const matrix3_t worldToPatch =
      referenceManagerPtr_->getContactPatchFrame(contactIndex_, time)
          .worldToPatchRotation;
  const vector3_t forceInPatch =
      worldToPatch * mpcRobotModelPtr_->getContactForce(input, contactIndex_);
  VectorFunctionLinearApproximation linearApproximation;
  linearApproximation.f = A_f_ * forceInPatch + b_;
  linearApproximation.dfdx = matrix_t::Zero(b_.size(), state.size());
  linearApproximation.dfdu = A_f_ * worldToPatch * forceInputJacobian_;
  return linearApproximation;
}

}  // namespace ocs2::humanoid
