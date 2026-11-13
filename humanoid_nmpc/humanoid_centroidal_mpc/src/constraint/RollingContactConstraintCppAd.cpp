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

#include "humanoid_centroidal_mpc/constraint/RollingContactConstraintCppAd.h"

#include <cmath>
#include <cstddef>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/ConstraintOrder.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_robotic_tools/common/SkewSymmetricMatrix.h"
#include "ocs2_robotic_tools/end_effector/EndEffectorKinematics.h"

#include "humanoid_common_mpc/HumanoidPreComputation.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/EndEffectorKinematicsTwistConstraint.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"
#include "humanoid_common_mpc/contact/RollingContact.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

namespace {

/** Returns whether `gains` is empty or has at least the translation block. */
bool hasTranslationBlockOrIsEmpty(const matrix_t& gains) {
  return gains.size() == 0 ||
         (gains.rows() >= kPositionDim && gains.cols() >= kPositionDim);
}

/** Returns the translation block of `gains`, zero when it is empty. */
matrix3_t translationBlock(const matrix_t& gains) {
  if (gains.size() == 0) return matrix3_t::Zero();
  return gains.topLeftCorner<kPositionDim, kPositionDim>();
}

}  // namespace

absl::StatusOr<std::unique_ptr<RollingContactConstraintCppAd>>
RollingContactConstraintCppAd::Create(
    const SwitchedModelReferenceManager& referenceManager,
    const EndEffectorKinematics<scalar_t>& wheelKinematics,
    size_t contactIndex,
    scalar_t wheelRadius,
    const EndEffectorKinematicsTwistConstraint::Config& gains) {
  if (!std::isfinite(wheelRadius) || !(wheelRadius > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[RollingContactConstraintCppAd] contact ", contactIndex,
                     ": the wheel radius must be finite and positive, not ",
                     wheelRadius, " m."));
  }
  if (wheelKinematics.getIds().size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[RollingContactConstraintCppAd] contact ", contactIndex,
        ": the kinematics must have exactly one end-effector, the wheel "
        "center, not ",
        wheelKinematics.getIds().size(), "."));
  }
  if (!hasTranslationBlockOrIsEmpty(gains.Ax) ||
      !hasTranslationBlockOrIsEmpty(gains.Av)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[RollingContactConstraintCppAd] contact ", contactIndex,
        ": the gains Ax (", gains.Ax.rows(), " x ", gains.Ax.cols(),
        ") and Av (", gains.Av.rows(), " x ", gains.Av.cols(),
        ") must each be empty or at least 3 x 3."));
  }
  // The constructor is private: Create() checks what it relies on.
  return absl::WrapUnique(new RollingContactConstraintCppAd(
      referenceManager, wheelKinematics, contactIndex, wheelRadius, gains));
}

RollingContactConstraintCppAd::RollingContactConstraintCppAd(
    const SwitchedModelReferenceManager& referenceManager,
    const EndEffectorKinematics<scalar_t>& wheelKinematics,
    size_t contactIndex,
    scalar_t wheelRadius,
    const EndEffectorKinematicsTwistConstraint::Config& gains)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      wheelKinematicsPtr_(wheelKinematics.clone()),
      contactIndex_(contactIndex),
      wheelRadius_(wheelRadius) {
  setGains(gains);
}

RollingContactConstraintCppAd::RollingContactConstraintCppAd(
    const RollingContactConstraintCppAd& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      wheelKinematicsPtr_(rhs.wheelKinematicsPtr_->clone()),
      contactIndex_(rhs.contactIndex_),
      wheelRadius_(rhs.wheelRadius_),
      posErrorGains_(rhs.posErrorGains_),
      linvelErrorGains_(rhs.linvelErrorGains_),
      // Copied, as every stance term copies it: the SQP solver clones the
      // problem once per worker thread.
      isActive_(rhs.isActive_) {}

void RollingContactConstraintCppAd::setGains(
    const EndEffectorKinematicsTwistConstraint::Config& gains) {
  posErrorGains_ = translationBlock(gains.Ax);
  linvelErrorGains_ = translationBlock(gains.Av);
}

bool RollingContactConstraintCppAd::isActive(scalar_t time) const {
  if (!isActive_) return false;
  return referenceManagerPtr_->isInContact(time, contactIndex_);
}

scalar_t RollingContactConstraintCppAd::getDesPosContactRtWorldEwrtWorldZ(
    scalar_t time, const PreComputation& preComp) const {
  const HumanoidPreComputation& humanoidPreComp =
      cast<HumanoidPreComputation>(preComp);
  if (humanoidPreComp.hasDesPosFootRtWorldEwrtWorldZ(contactIndex_)) {
    return humanoidPreComp.getDesPosFootRtWorldEwrtWorldZ(contactIndex_);
  }
  // A contact the swing planner does not plan stands on its patch, which is
  // world aligned for a rolling contact (checkCentroidalContactSet() refuses
  // the sources of tilted patches on a set with one).
  const ContactPatchFrame patch =
      referenceManagerPtr_->getContactPatchFrame(contactIndex_, time);
  return patch.posSurfaceRtWorldEwrtSurfaceZ;
}

Eigen::Matrix<scalar_t, kPositionDim, kTwistDim>
RollingContactConstraintCppAd::getNoSlipMap() const {
  // v_contact = v_wheel + w_wheel x (-r n) = v_wheel + r [n]x w_wheel.
  Eigen::Matrix<scalar_t, kPositionDim, kTwistDim> map;
  map.leftCols<kPositionDim>().setIdentity();
  map.rightCols<kRotationTangentDim>() =
      wheelRadius_ *
      skewSymmetricMatrix<scalar_t>(normalTerrainEwrtWorld<scalar_t>());
  return map;
}

vector_t RollingContactConstraintCppAd::getValue(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& preComp) const {
  const vector3_t normalTerrain = normalTerrainEwrtWorld<scalar_t>();
  vector_t value = vector_t::Zero(kPositionDim);
  if (!posErrorGains_.isZero(0.0)) {
    const vector3_t errPosContactRtWorldEwrtWorld =
        posContactRtWorldEwrtWorld<scalar_t>(
            wheelKinematicsPtr_->getPosEesRtWorldEwrtWorld(state).front(),
            wheelRadius_, normalTerrain) -
        getDesPosContactRtWorldEwrtWorldZ(time, preComp) * normalTerrain;
    value.noalias() += posErrorGains_ * errPosContactRtWorldEwrtWorld;
  }
  const vector6_t twistWheelRtWorldEwrtWorld =
      wheelKinematicsPtr_->getTwistEesRtWorldEwrtWorld(state, input).front();
  value.noalias() += linvelErrorGains_ *
                     linvelContactRtWorldEwrtWorld<scalar_t>(
                         twistWheelRtWorldEwrtWorld.head<kPositionDim>(),
                         twistWheelRtWorldEwrtWorld.tail<kRotationTangentDim>(),
                         wheelRadius_, normalTerrain);
  return value;
}

VectorFunctionLinearApproximation
RollingContactConstraintCppAd::getLinearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& preComp) const {
  VectorFunctionLinearApproximation linearApproximation =
      VectorFunctionLinearApproximation::Zero(kPositionDim, state.size(),
                                              input.size());
  const vector3_t normalTerrain = normalTerrainEwrtWorld<scalar_t>();
  if (!posErrorGains_.isZero(0.0)) {
    // p_contact - p_wheel = -r n is constant: d p_contact / dx = J_p.
    const VectorFunctionLinearApproximation approxPosWheelRtWorldEwrtWorld =
        wheelKinematicsPtr_->getLinearApproximationPosEesRtWorldEwrtWorld(state)
            .front();
    const vector3_t errPosContactRtWorldEwrtWorld =
        posContactRtWorldEwrtWorld<scalar_t>(approxPosWheelRtWorldEwrtWorld.f,
                                             wheelRadius_, normalTerrain) -
        getDesPosContactRtWorldEwrtWorldZ(time, preComp) * normalTerrain;
    linearApproximation.f.noalias() +=
        posErrorGains_ * errPosContactRtWorldEwrtWorld;
    linearApproximation.dfdx.noalias() +=
        posErrorGains_ * approxPosWheelRtWorldEwrtWorld.dfdx;
  }
  const VectorFunctionLinearApproximation approxTwistWheelRtWorldEwrtWorld =
      wheelKinematicsPtr_
          ->getLinearApproximationTwistEesRtWorldEwrtWorld(state, input)
          .front();
  // Av [I, r [n]x]: the gains of the wheel-center twist.
  const Eigen::Matrix<scalar_t, kPositionDim, kTwistDim> twistGains =
      linvelErrorGains_ * getNoSlipMap();
  linearApproximation.f.noalias() +=
      twistGains * approxTwistWheelRtWorldEwrtWorld.f;
  linearApproximation.dfdx.noalias() +=
      twistGains * approxTwistWheelRtWorldEwrtWorld.dfdx;
  linearApproximation.dfdu.noalias() +=
      twistGains * approxTwistWheelRtWorldEwrtWorld.dfdu;
  return linearApproximation;
}

}  // namespace ocs2::humanoid
