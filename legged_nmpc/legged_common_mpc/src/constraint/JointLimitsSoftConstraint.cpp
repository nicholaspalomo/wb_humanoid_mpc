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

#include "legged_common_mpc/constraint/JointLimitsSoftConstraint.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/Types.h"

namespace ocs2::legged {
namespace {

/** Which derivative of the barrier FiniteLimitBarrier evaluates. */
enum class BarrierOrder {
  kValue,
  kFirstDerivative,
  kSecondDerivative,
};

/**
 * The barrier of an offset `hi` from a joint limit, or 0 for an infinite limit,
 * which bounds nothing: a joint without that bound has no row. Applied entry by
 * entry (Eigen's binaryExpr), so that the sums over the joints add the same
 * terms in the same order as the barrier of the offsets alone did. Keeps a
 * pointer to the penalty, which outlives it. Thread-safe as the penalty's const
 * methods are.
 */
class FiniteLimitBarrier {
 public:
  explicit FiniteLimitBarrier(const PieceWisePolynomialBarrierPenalty& penalty,
                              BarrierOrder order)
      : penalty_(&penalty), order_(order) {}
  FiniteLimitBarrier(const FiniteLimitBarrier&) = default;
  FiniteLimitBarrier& operator=(const FiniteLimitBarrier&) = default;
  FiniteLimitBarrier(FiniteLimitBarrier&&) noexcept = default;
  FiniteLimitBarrier& operator=(FiniteLimitBarrier&&) noexcept = default;
  ~FiniteLimitBarrier() = default;

  scalar_t operator()(scalar_t hi, scalar_t limit) const {
    if (!std::isfinite(limit)) return 0.0;
    switch (order_) {
      case BarrierOrder::kValue:
        return penalty_->getValue(/*t=*/0.0, hi);
      case BarrierOrder::kFirstDerivative:
        return penalty_->getDerivative(/*t=*/0.0, hi);
      case BarrierOrder::kSecondDerivative:
        return penalty_->getSecondDerivative(/*t=*/0.0, hi);
    }
    return 0.0;
  }

 private:
  const PieceWisePolynomialBarrierPenalty* absl_nonnull penalty_;
  BarrierOrder order_;
};

/** Returns the sum over the joints of the barrier of `offsets` from the
 * position bounds `limits`; an infinite bound adds nothing. */
scalar_t sumOfFiniteLimitBarriers(
    const vector_t& offsets,
    const vector_t& limits,
    const PieceWisePolynomialBarrierPenalty& penalty) {
  return offsets
      .binaryExpr(limits, FiniteLimitBarrier(penalty, BarrierOrder::kValue))
      .sum();
}

/** Returns, joint by joint, the derivative `order` of the barrier of `offsets`
 * from the position bounds `limits`; 0 at an infinite bound. */
vector_t finiteLimitBarrierDerivatives(
    const vector_t& offsets,
    const vector_t& limits,
    const PieceWisePolynomialBarrierPenalty& penalty,
    BarrierOrder order) {
  return offsets.binaryExpr(limits, FiniteLimitBarrier(penalty, order));
}

}  // namespace

JointLimitsSoftConstraint::JointLimitsSoftConstraint(
    JointPositionLimits jointPositionLimits,
    ocs2::PieceWisePolynomialBarrierPenalty::Config barrierSettings,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : jointPositionPenaltyPtr_(
          std::make_unique<ocs2::PieceWisePolynomialBarrierPenalty>(
              barrierSettings)),
      mpcRobotModelPtr_(&mpcRobotModel),
      jointPositionLimits_(std::move(jointPositionLimits)) {}

JointLimitsSoftConstraint::JointLimitsSoftConstraint(
    const JointLimitsSoftConstraint& rhs)
    : jointPositionPenaltyPtr_(rhs.jointPositionPenaltyPtr_->clone()),
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_),
      jointPositionLimits_(rhs.jointPositionLimits_),
      // isActive_ is copied deliberately: the SQP solver clones the whole
      // problem once per worker thread, and a copy constructor that dropped
      // this flag silently reverted a deactivated term to active.
      isActive_(rhs.isActive_) {}

scalar_t JointLimitsSoftConstraint::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const ocs2::TargetTrajectories& /*targetTrajectories*/,
    const ocs2::PreComputation& /*preComp*/) const {
  return getValue(mpcRobotModelPtr_->getJointAngles(state));
}

ScalarFunctionQuadraticApproximation
JointLimitsSoftConstraint::getQuadraticApproximation(
    scalar_t /*time*/,
    const vector_t& state,
    const ocs2::TargetTrajectories& /*targetTrajectories*/,
    const ocs2::PreComputation& /*preComp*/) const {
  return getQuadraticApproximation(mpcRobotModelPtr_->getJointAngles(state));
}

scalar_t JointLimitsSoftConstraint::getValue(
    const vector_t& jointPositions) const {
  const vector_t upperBoundJointPositionOffset =
      jointPositionLimits_.upper - jointPositions;
  const vector_t lowerBoundJointPositionOffset =
      jointPositions - jointPositionLimits_.lower;
  return sumOfFiniteLimitBarriers(upperBoundJointPositionOffset,
                                  jointPositionLimits_.upper,
                                  *jointPositionPenaltyPtr_) +
         sumOfFiniteLimitBarriers(lowerBoundJointPositionOffset,
                                  jointPositionLimits_.lower,
                                  *jointPositionPenaltyPtr_);
}

ScalarFunctionQuadraticApproximation
JointLimitsSoftConstraint::getQuadraticApproximation(
    const vector_t& jointPositions) const {
  const vector_t upperBoundJointPositionOffset =
      jointPositionLimits_.upper - jointPositions;
  const vector_t lowerBoundJointPositionOffset =
      jointPositions - jointPositionLimits_.lower;
  const PieceWisePolynomialBarrierPenalty& penalty = *jointPositionPenaltyPtr_;

  const size_t stateDim = mpcRobotModelPtr_->getStateDim();
  const size_t jointDim = mpcRobotModelPtr_->getJointDim();
  const size_t jointStartIndex = mpcRobotModelPtr_->getJointStartindex();

  ScalarFunctionQuadraticApproximation cost;
  cost.f = getValue(jointPositions);

  cost.dfdx = vector_t::Zero(stateDim);
  cost.dfdx.segment(jointStartIndex, jointDim) =
      finiteLimitBarrierDerivatives(lowerBoundJointPositionOffset,
                                    jointPositionLimits_.lower, penalty,
                                    BarrierOrder::kFirstDerivative) -
      finiteLimitBarrierDerivatives(upperBoundJointPositionOffset,
                                    jointPositionLimits_.upper, penalty,
                                    BarrierOrder::kFirstDerivative);

  cost.dfdxx = matrix_t::Zero(stateDim, stateDim);
  cost.dfdxx.block(jointStartIndex, jointStartIndex, jointDim, jointDim)
      .diagonal() =
      finiteLimitBarrierDerivatives(lowerBoundJointPositionOffset,
                                    jointPositionLimits_.lower, penalty,
                                    BarrierOrder::kSecondDerivative) +
      finiteLimitBarrierDerivatives(upperBoundJointPositionOffset,
                                    jointPositionLimits_.upper, penalty,
                                    BarrierOrder::kSecondDerivative);

  return cost;
}

void JointLimitsSoftConstraint::setGains(scalar_t mu, scalar_t delta) {
  jointPositionPenaltyPtr_->setConfig(
      ocs2::PieceWisePolynomialBarrierPenalty::Config(mu, delta));
}

PieceWisePolynomialBarrierPenalty::Config JointLimitsSoftConstraint::getGains()
    const {
  ocs2::PieceWisePolynomialBarrierPenalty::Config config;
  jointPositionPenaltyPtr_->getConfig(config);
  return config;
}

}  // namespace ocs2::legged
