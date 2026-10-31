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

#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/contact/ContactInputJacobian.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<ForceWeightedSlipConstraint>>
ForceWeightedSlipConstraint::Create(
    const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    size_t contactPointIndex,
    scalar_t scaleForceContactEwrtParamZ,
    scalar_t scaleLinvelContactRtWorldEwrtWorldXy,
    scalar_t scaleAngvelContactRtWorldEwrtWorldZ) {
  if (endEffectorKinematics.getIds().size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ForceWeightedSlipConstraint] expects exactly one "
                     "end-effector, the contact frame of "
                     "this foot, got ",
                     endEffectorKinematics.getIds().size()));
  }
  // Negated comparisons, so that NaN is refused too.
  if (!(scaleForceContactEwrtParamZ > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ForceWeightedSlipConstraint] the force reference must "
                     "be positive, got ",
                     scaleForceContactEwrtParamZ,
                     "; it is the robot's weight, robotMass * g, taken from "
                     "the URDF rather than configured"));
  }
  if (!(scaleLinvelContactRtWorldEwrtWorldXy > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ForceWeightedSlipConstraint] "
        "contact_implicit.scale_linvel_contact_rt_world_ewrt_world_xy "
        "must be positive, got ",
        scaleLinvelContactRtWorldEwrtWorldXy));
  }
  if (!(scaleAngvelContactRtWorldEwrtWorldZ > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ForceWeightedSlipConstraint] "
                     "contact_implicit.scale_angvel_contact_rt_world_ewrt_"
                     "world_z must be positive, got ",
                     scaleAngvelContactRtWorldEwrtWorldZ));
  }
  // The constructor is private, so std::make_unique cannot call it (ToTW #134).
  return absl::WrapUnique(new ForceWeightedSlipConstraint(
      endEffectorKinematics, mpcRobotModel, contactPointIndex,
      scaleForceContactEwrtParamZ, scaleLinvelContactRtWorldEwrtWorldXy,
      scaleAngvelContactRtWorldEwrtWorldZ));
}

ForceWeightedSlipConstraint::ForceWeightedSlipConstraint(
    const EndEffectorKinematics<scalar_t>& endEffectorKinematics,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    size_t contactPointIndex,
    scalar_t scaleForceContactEwrtParamZ,
    scalar_t scaleLinvelContactRtWorldEwrtWorldXy,
    scalar_t scaleAngvelContactRtWorldEwrtWorldZ)
    : StateInputConstraint(ConstraintOrder::Linear),
      endEffectorKinematicsPtr_(endEffectorKinematics.clone()),
      contactPointIndex_(contactPointIndex),
      inverseScaleForceContactEwrtParamZ_(1.0 / scaleForceContactEwrtParamZ),
      inverseScaleSlip_(1.0 / scaleLinvelContactRtWorldEwrtWorldXy,
                        1.0 / scaleLinvelContactRtWorldEwrtWorldXy,
                        1.0 / scaleAngvelContactRtWorldEwrtWorldZ),
      inputRowForceContactEwrtParamZ_(
          inputRowForceContactEwrtParamZ(mpcRobotModel, contactPointIndex)) {}

ForceWeightedSlipConstraint::ForceWeightedSlipConstraint(
    const ForceWeightedSlipConstraint& rhs)
    : StateInputConstraint(rhs),
      endEffectorKinematicsPtr_(rhs.endEffectorKinematicsPtr_->clone()),
      contactPointIndex_(rhs.contactPointIndex_),
      inverseScaleForceContactEwrtParamZ_(
          rhs.inverseScaleForceContactEwrtParamZ_),
      inverseScaleSlip_(rhs.inverseScaleSlip_),
      inputRowForceContactEwrtParamZ_(rhs.inputRowForceContactEwrtParamZ_) {}

vector_t ForceWeightedSlipConstraint::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const vector3_t linvelContactRtWorldEwrtWorld =
      endEffectorKinematicsPtr_->getLinvelEesRtWorldEwrtWorld(state, input)
          .front();
  const vector3_t angvelContactRtWorldEwrtWorld =
      endEffectorKinematicsPtr_->getAngvelEesRtWorldEwrtWorld(state, input)
          .front();
  const scalar_t scaledForceContactEwrtParamZ =
      inputRowForceContactEwrtParamZ_.dot(input) *
      inverseScaleForceContactEwrtParamZ_;
  vector3_t scaledSlip;
  scaledSlip << linvelContactRtWorldEwrtWorld.head<kHorizontalPositionDim>(),
      angvelContactRtWorldEwrtWorld(kZAxisIndex);
  return vector_t(scaledForceContactEwrtParamZ *
                  scaledSlip.cwiseProduct(inverseScaleSlip_));
}

VectorFunctionLinearApproximation
ForceWeightedSlipConstraint::getLinearApproximation(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const VectorFunctionLinearApproximation linvelContactRtWorldEwrtWorld =
      endEffectorKinematicsPtr_
          ->getLinearApproximationLinvelEesRtWorldEwrtWorld(state, input)
          .front();
  const VectorFunctionLinearApproximation angvelContactRtWorldEwrtWorld =
      endEffectorKinematicsPtr_
          ->getLinearApproximationAngvelEesRtWorldEwrtWorld(state, input)
          .front();
  const scalar_t scaledForceContactEwrtParamZ =
      inputRowForceContactEwrtParamZ_.dot(input) *
      inverseScaleForceContactEwrtParamZ_;

  // The constrained components of the twist: the two tangential linear
  // velocities and the spin about the normal, each divided by a reference in
  // its own units so that the three rows are comparable dimensionless numbers.
  const vector3_t scaledSlip =
      (vector3_t()
           << linvelContactRtWorldEwrtWorld.f.head<kHorizontalPositionDim>(),
       angvelContactRtWorldEwrtWorld.f(kZAxisIndex))
          .finished()
          .cwiseProduct(inverseScaleSlip_);
  matrix_t dScaledSlipdx(kNumConstraints,
                         linvelContactRtWorldEwrtWorld.dfdx.cols());
  dScaledSlipdx
      << linvelContactRtWorldEwrtWorld.dfdx.topRows<kHorizontalPositionDim>(),
      angvelContactRtWorldEwrtWorld.dfdx.row(kZAxisIndex);
  dScaledSlipdx = inverseScaleSlip_.asDiagonal() * dScaledSlipdx;
  matrix_t dScaledSlipdu(kNumConstraints,
                         linvelContactRtWorldEwrtWorld.dfdu.cols());
  dScaledSlipdu
      << linvelContactRtWorldEwrtWorld.dfdu.topRows<kHorizontalPositionDim>(),
      angvelContactRtWorldEwrtWorld.dfdu.row(kZAxisIndex);
  dScaledSlipdu = inverseScaleSlip_.asDiagonal() * dScaledSlipdu;

  VectorFunctionLinearApproximation approximation;
  approximation.f = scaledForceContactEwrtParamZ * scaledSlip;
  approximation.dfdx = scaledForceContactEwrtParamZ * dScaledSlipdx;
  // d(f_hat v_hat) / du = f_hat dv_hat/du + v_hat (df_hat/du), the second term
  // being the constant normal-force row carrying the force scale.
  approximation.dfdu = scaledForceContactEwrtParamZ * dScaledSlipdu +
                       (scaledSlip * inverseScaleForceContactEwrtParamZ_) *
                           inputRowForceContactEwrtParamZ_.transpose();
  return approximation;
}

}  // namespace ocs2::humanoid
