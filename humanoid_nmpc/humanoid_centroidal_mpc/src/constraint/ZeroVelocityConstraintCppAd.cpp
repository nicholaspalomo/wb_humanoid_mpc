/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"

#include <memory>
#include <utility>

#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ZeroVelocityConstraintCppAd::ZeroVelocityConstraintCppAd(
    const SwitchedModelReferenceManager& referenceManager,
    const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
    size_t contactPointIndex,
    size_t numConstraints,
    EndEffectorKinematicsTwistConstraint::Config config,
    ContactKind contactKind)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      eeTwistConstraintPtr_(
          std::make_unique<EndEffectorKinematicsTwistConstraint>(
              endEffectorKinematics,
              contactKind == ContactKind::kPoint ? kPositionDim
                                                 : numConstraints,
              std::move(config))),
      contactPointIndex_(contactPointIndex),
      contactKind_(contactKind) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ZeroVelocityConstraintCppAd::ZeroVelocityConstraintCppAd(
    const ZeroVelocityConstraintCppAd& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      eeTwistConstraintPtr_(rhs.eeTwistConstraintPtr_->clone()),
      contactPointIndex_(rhs.contactPointIndex_),
      contactKind_(rhs.contactKind_),
      // isActive_ is copied deliberately: the SQP solver clones the whole
      // problem once per worker thread, and a copy constructor that dropped
      // this flag silently reverted a deactivated term to active.
      isActive_(rhs.isActive_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool ZeroVelocityConstraintCppAd::isActive(scalar_t time) const {
  if (!isActive_) return false;
  return referenceManagerPtr_->isInContact(time, contactPointIndex_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void ZeroVelocityConstraintCppAd::configureForEvaluation(
    scalar_t time, const PreComputation& preComp) const {
  const HumanoidPreComputation& humanoidPreComp =
      cast<HumanoidPreComputation>(preComp);
  const ContactPatchFrame patch =
      referenceManagerPtr_->getContactPatchFrame(contactPointIndex_, time);
  EndEffectorKinematicsTwistConstraint::Config& config =
      eeTwistConstraintPtr_->getConfig();
  if (isWorldAligned(patch) &&
      humanoidPreComp.hasFootReferenceHeight(contactPointIndex_)) {
    // Modify to actual ground height
    eeTwistConstraintPtr_->clearRotmatWorldRtSurface();
    config.b[kZAxisIndex] =
        -config.Ax(kZAxisIndex, kZAxisIndex) *
        humanoidPreComp.getFootReferenceHeight(contactPointIndex_);
    return;
  }
  if (isWorldAligned(patch)) {
    eeTwistConstraintPtr_->clearRotmatWorldRtSurface();
  } else {
    eeTwistConstraintPtr_->setRotmatWorldRtSurface(patch.rotmatWorldRtSurface);
  }
  config.b[kZAxisIndex] =
      -config.Ax(kZAxisIndex, kZAxisIndex) * patch.normalOffset;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t ZeroVelocityConstraintCppAd::getValue(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& preComp) const {
  configureForEvaluation(time, preComp);
  return eeTwistConstraintPtr_->getValue(time, state, input, preComp);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation
ZeroVelocityConstraintCppAd::getLinearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& preComp) const {
  configureForEvaluation(time, preComp);
  return eeTwistConstraintPtr_->getLinearApproximation(time, state, input,
                                                       preComp);
}

}  // namespace ocs2::humanoid
