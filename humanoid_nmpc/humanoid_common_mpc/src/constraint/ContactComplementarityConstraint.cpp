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

#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"

#include <cmath>
#include <cstddef>
#include <memory>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/ConstraintOrder.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact/ContactInputJacobian.h"
#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<ContactComplementarityConstraint>>
ContactComplementarityConstraint::Create(
    const FootprintCornerHeights& cornerHeights,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    size_t contactPointIndex,
    scalar_t posSurfaceRtWorldEwrtWorldZ,
    scalar_t scaleForceContactEwrtParamZ,
    scalar_t scalePosContactRtSurfaceEwrtWorldZ,
    scalar_t gapSmoothing) {
  // Negated comparisons, so that NaN is refused too.
  if (!(scaleForceContactEwrtParamZ > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactComplementarityConstraint] the force reference "
                     "must be positive, got ",
                     scaleForceContactEwrtParamZ,
                     "; it is the robot's weight, robotMass * g, taken from "
                     "the URDF rather than configured"));
  }
  if (!(scalePosContactRtSurfaceEwrtWorldZ > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactComplementarityConstraint] "
                     "contact_implicit.scale_pos_contact_rt_surface_ewrt_world_"
                     "z must be positive, got ",
                     scalePosContactRtSurfaceEwrtWorldZ));
  }
  if (!(gapSmoothing > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactComplementarityConstraint] "
                     "contact_implicit.gap_smoothing must be positive, got ",
                     gapSmoothing));
  }
  // The constructor is private, so std::make_unique cannot call it (ToTW #134).
  return absl::WrapUnique(new ContactComplementarityConstraint(
      cornerHeights, mpcRobotModel, contactPointIndex,
      posSurfaceRtWorldEwrtWorldZ, scaleForceContactEwrtParamZ,
      scalePosContactRtSurfaceEwrtWorldZ, gapSmoothing));
}

ContactComplementarityConstraint::ContactComplementarityConstraint(
    const FootprintCornerHeights& cornerHeights,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    size_t contactPointIndex,
    scalar_t posSurfaceRtWorldEwrtWorldZ,
    scalar_t scaleForceContactEwrtParamZ,
    scalar_t scalePosContactRtSurfaceEwrtWorldZ,
    scalar_t gapSmoothing)
    : StateInputConstraint(ConstraintOrder::Linear),
      cornerHeightsPtr_(cornerHeights.clone()),
      contactPointIndex_(contactPointIndex),
      posSurfaceRtWorldEwrtWorldZ_(posSurfaceRtWorldEwrtWorldZ),
      gapSmoothing_(gapSmoothing),
      inverseScaleForceContactEwrtParamZ_(1.0 / scaleForceContactEwrtParamZ),
      inverseScalePosContactRtSurfaceEwrtWorldZ_(
          1.0 / scalePosContactRtSurfaceEwrtWorldZ),
      inputRowForceContactEwrtParamZ_(
          inputRowForceContactEwrtParamZ(mpcRobotModel, contactPointIndex)) {}

ContactComplementarityConstraint::ContactComplementarityConstraint(
    const ContactComplementarityConstraint& rhs)
    : StateInputConstraint(rhs),
      cornerHeightsPtr_(rhs.cornerHeightsPtr_->clone()),
      contactPointIndex_(rhs.contactPointIndex_),
      posSurfaceRtWorldEwrtWorldZ_(rhs.posSurfaceRtWorldEwrtWorldZ_),
      gapSmoothing_(rhs.gapSmoothing_),
      inverseScaleForceContactEwrtParamZ_(
          rhs.inverseScaleForceContactEwrtParamZ_),
      inverseScalePosContactRtSurfaceEwrtWorldZ_(
          rhs.inverseScalePosContactRtSurfaceEwrtWorldZ_),
      inputRowForceContactEwrtParamZ_(rhs.inputRowForceContactEwrtParamZ_) {}

void ContactComplementarityConstraint::setGapSmoothing(scalar_t gapSmoothing) {
  CHECK_GT(gapSmoothing, 0.0)
      << "[ContactComplementarityConstraint] contact_implicit.gap_smoothing "
         "must be positive";
  gapSmoothing_ = gapSmoothing;
}

scalar_t ContactComplementarityConstraint::getGap(const vector_t& state) const {
  return smoothMinimumHeight(
             cornerHeightsPtr_->getPosLinksRtWorldEwrtWorldZ(state),
             gapSmoothing_)
             .value -
         posSurfaceRtWorldEwrtWorldZ_;
}

vector_t ContactComplementarityConstraint::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const scalar_t scaledPosContactRtSurfaceEwrtWorldZ =
      getGap(state) * inverseScalePosContactRtSurfaceEwrtWorldZ_;
  const scalar_t scaledForceContactEwrtParamZ =
      inputRowForceContactEwrtParamZ_.dot(input) *
      inverseScaleForceContactEwrtParamZ_;
  return (vector_t(1) << scaledForceContactEwrtParamZ *
                             scaledPosContactRtSurfaceEwrtWorldZ)
      .finished();
}

VectorFunctionLinearApproximation
ContactComplementarityConstraint::getLinearApproximation(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const SmoothMinimumHeight gap = smoothMinimumHeight(
      cornerHeightsPtr_->getPosLinksRtWorldEwrtWorldZ(state), gapSmoothing_);
  // Both factors are carried in their normalized form, so each derivative block
  // picks up the scale of the factor that survives the product rule: d(f_hat
  // h_hat)/dx = f_hat dh_hat/dx, and dh_hat/dx is dh/dx over the reference
  // height. The gap's own derivative is the softmin weights contracted with the
  // corner Jacobian - a convex combination of the corners' rows, which
  // collapses to the touching corner's row as the foot rocks onto it.
  const scalar_t scaledPosContactRtSurfaceEwrtWorldZ =
      (gap.value - posSurfaceRtWorldEwrtWorldZ_) *
      inverseScalePosContactRtSurfaceEwrtWorldZ_;
  const scalar_t scaledForceContactEwrtParamZ =
      inputRowForceContactEwrtParamZ_.dot(input) *
      inverseScaleForceContactEwrtParamZ_;

  VectorFunctionLinearApproximation approximation;
  approximation.f = (vector_t(1) << scaledForceContactEwrtParamZ *
                                        scaledPosContactRtSurfaceEwrtWorldZ)
                        .finished();
  approximation.dfdx =
      (scaledForceContactEwrtParamZ *
       inverseScalePosContactRtSurfaceEwrtWorldZ_) *
      (gap.weights.transpose() *
       cornerHeightsPtr_->getJacobianPosLinksRtWorldEwrtWorldZ(state));
  approximation.dfdu = (scaledPosContactRtSurfaceEwrtWorldZ *
                        inverseScaleForceContactEwrtParamZ_) *
                       inputRowForceContactEwrtParamZ_.transpose();
  return approximation;
}

}  // namespace ocs2::humanoid
