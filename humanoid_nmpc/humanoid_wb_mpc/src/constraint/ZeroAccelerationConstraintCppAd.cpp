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

#include "humanoid_wb_mpc/constraint/ZeroAccelerationConstraintCppAd.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "ocs2_core/misc/Numerics.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

// The row of the stance constraint that holds the foot's height (stanceFootAccelerationConstraintConfig()).
constexpr Eigen::Index kHeightRow = 2;

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::StatusOr<std::unique_ptr<ZeroAccelerationConstraintCppAd>> ZeroAccelerationConstraintCppAd::Create(
    const SwitchedModelReferenceManager& referenceManager,
    const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
    size_t contactPointIndex,
    EndEffectorDynamicsAccelerationsConstraint::Config config) {
  ASSIGN_OR_RETURN(std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> eeAccelConstraint,
                   EndEffectorDynamicsAccelerationsConstraint::Create(endEffectorDynamics, /*numConstraints=*/6, std::move(config)));
  return absl::WrapUnique(new ZeroAccelerationConstraintCppAd(referenceManager, std::move(eeAccelConstraint), contactPointIndex));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ZeroAccelerationConstraintCppAd::ZeroAccelerationConstraintCppAd(
    const SwitchedModelReferenceManager& referenceManager,
    std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint> eeAccelConstraint,
    size_t contactPointIndex)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      eeAccelConstraintPtr_(std::move(eeAccelConstraint)),
      contactPointIndex_(contactPointIndex) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ZeroAccelerationConstraintCppAd::ZeroAccelerationConstraintCppAd(const ZeroAccelerationConstraintCppAd& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      eeAccelConstraintPtr_(rhs.eeAccelConstraintPtr_->clone()),
      contactPointIndex_(rhs.contactPointIndex_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool ZeroAccelerationConstraintCppAd::isActive(scalar_t time) const {
  return referenceManagerPtr_->getContactFlags(time)[contactPointIndex_];
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t ZeroAccelerationConstraintCppAd::groundTerm(scalar_t time) const {
  const matrix_t& positionGains = eeAccelConstraintPtr_->getConfig().Ax;
  if (positionGains.rows() <= kHeightRow || positionGains.cols() <= kHeightRow) return 0.0;
  return positionGains(kHeightRow, kHeightRow) *
         referenceManagerPtr_->getSwingTrajectoryPlanner()->getZpositionConstraint(contactPointIndex_, time);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t ZeroAccelerationConstraintCppAd::getValue(scalar_t time,
                                                   const vector_t& state,
                                                   const vector_t& input,
                                                   const PreComputation& preComp) const {
  vector_t value = eeAccelConstraintPtr_->getValue(time, state, input, preComp);
  value(kHeightRow) -= groundTerm(time);
  return value;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation ZeroAccelerationConstraintCppAd::getLinearApproximation(scalar_t time,
                                                                                          const vector_t& state,
                                                                                          const vector_t& input,
                                                                                          const PreComputation& preComp) const {
  VectorFunctionLinearApproximation linearization = eeAccelConstraintPtr_->getLinearApproximation(time, state, input, preComp);
  linearization.f(kHeightRow) -= groundTerm(time);
  return linearization;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
EndEffectorDynamicsAccelerationsConstraint::Config stanceFootAccelerationConstraintConfig(
    const ModelSettings::FootConstraintConfig& gains) {
  EndEffectorDynamicsAccelerationsConstraint::Config config;
  config.b.setZero(6);
  config.Ax.setZero(6, 6);
  config.Av.setIdentity(6, 6);
  config.Aa.setIdentity(6, 6);
  if (!numerics::almost_eq(gains.positionErrorGain_z, /*y=*/0.0)) {
    config.Ax(2, 2) = gains.positionErrorGain_z;
  }
  if (!numerics::almost_eq(gains.orientationErrorGain, /*y=*/0.0)) {
    config.Ax.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * gains.orientationErrorGain;
  }
  config.Av.block(0, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2) * gains.linearVelocityErrorGain_xy;
  config.Av(2, 2) = gains.linearVelocityErrorGain_z;
  config.Av.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * gains.angularVelocityErrorGain;
  config.Aa.block(0, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2) * gains.linearAccelerationErrorGain_xy;
  config.Aa(2, 2) = gains.linearAccelerationErrorGain_z;
  config.Aa.block(3, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * gains.angularAccelerationErrorGain;
  return config;
}

}  // namespace ocs2::humanoid
