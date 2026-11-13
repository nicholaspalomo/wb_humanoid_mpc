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

#include "legged_wb_mpc/constraint/EndEffectorDynamicsAccelerationsConstraint.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/ConstraintOrder.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "legged_common_mpc/common/Types.h"
#include "legged_wb_mpc/end_effector/EndEffectorDynamics.h"

namespace ocs2::legged {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

absl::StatusOr<std::unique_ptr<EndEffectorDynamicsAccelerationsConstraint>>
EndEffectorDynamicsAccelerationsConstraint::Create(
    const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
    size_t numConstraints,
    Config config) {
  if (endEffectorDynamics.getIds().size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("[EndEffectorDynamicsAccelerationsConstraint] this class "
                     "only accepts a single end-effector; "
                     "the end-effector dynamics has ",
                     endEffectorDynamics.getIds().size(), ": [",
                     absl::StrJoin(endEffectorDynamics.getIds(), ", "), "]"));
  }
  if (numConstraints < 1 ||
      numConstraints > static_cast<size_t>(kPoseTangentDim)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[EndEffectorDynamicsAccelerationsConstraint] ", numConstraints,
        " constraints: between 1 and ", kPoseTangentDim));
  }
  const Eigen::Index rows = static_cast<Eigen::Index>(numConstraints);
  // An empty config is set later, by configure(); a given one has the
  // dimensions of the constraint.
  const bool unconfigured = config.b.size() == 0 && config.Ax.size() == 0 &&
                            config.Av.size() == 0 && config.Aa.size() == 0;
  if (!unconfigured && config.b.size() != rows) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[EndEffectorDynamicsAccelerationsConstraint] b has ", config.b.size(),
        " rows for a constraint of ", numConstraints));
  }
  for (const std::pair<const char* absl_nonnull, const matrix_t* absl_nonnull>&
           gain :
       {std::pair<const char* absl_nonnull, const matrix_t* absl_nonnull>(
            "Ax", &config.Ax),
        std::pair<const char* absl_nonnull, const matrix_t* absl_nonnull>(
            "Av", &config.Av),
        std::pair<const char* absl_nonnull, const matrix_t* absl_nonnull>(
            "Aa", &config.Aa)}) {
    const matrix_t& matrix = *gain.second;
    if (matrix.size() > 0 &&
        (matrix.rows() != rows || matrix.cols() != kPoseTangentDim)) {
      return absl::InvalidArgumentError(
          absl::StrCat("[EndEffectorDynamicsAccelerationsConstraint] ",
                       gain.first, " is ", matrix.rows(), "x", matrix.cols(),
                       ": empty, or ", numConstraints, "x", kPoseTangentDim));
    }
  }
  return absl::WrapUnique(new EndEffectorDynamicsAccelerationsConstraint(
      endEffectorDynamics, numConstraints, std::move(config)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

EndEffectorDynamicsAccelerationsConstraint::
    EndEffectorDynamicsAccelerationsConstraint(
        const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
        size_t numConstraints,
        Config config)
    : StateInputConstraint(ConstraintOrder::Linear),
      normalTerrainEwrtWorld_(0.0, 0.0, 1.0),
      endEffectorDynamicsPtr_(endEffectorDynamics.clone()),
      numConstraints_(numConstraints),
      config_(std::move(config)) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

EndEffectorDynamicsAccelerationsConstraint::
    EndEffectorDynamicsAccelerationsConstraint(
        const EndEffectorDynamicsAccelerationsConstraint& rhs)
    : StateInputConstraint(rhs),
      normalTerrainEwrtWorld_(rhs.normalTerrainEwrtWorld_),
      endEffectorDynamicsPtr_(rhs.endEffectorDynamicsPtr_->clone()),
      numConstraints_(rhs.numConstraints_),
      config_(rhs.config_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void EndEffectorDynamicsAccelerationsConstraint::configure(Config&& config) {
  config_ = std::move(config);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t EndEffectorDynamicsAccelerationsConstraint::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  vector_t f = config_.b;
  if (config_.Ax.size() > 0) {
    // foot pose is a 6D vector containing the foot position and orientation
    // error wrt. to the ground normal
    vector6_t xee;
    xee << endEffectorDynamicsPtr_->getPosEesRtWorldEwrtWorld(state).front(),
        endEffectorDynamicsPtr_
            ->getTiltEesRtSurfaceEwrtWorld(state, {normalTerrainEwrtWorld_})
            .front();
    f.noalias() += config_.Ax * xee;
  }
  // The velocity and acceleration gains act on the time derivatives of that
  // pose: the linear velocity and acceleration, and the rate and acceleration
  // of the orientation error (completed along the ground normal by half the
  // rotation rate about the foot normal), so that the orientation rows are the
  // PD law on the error, Kp e + Kd de/dt + Ka d2e/dt2.
  if (config_.Av.size() > 0) {
    f.noalias() +=
        config_.Av * endEffectorDynamicsPtr_
                         ->getXeeRate(state, input, {normalTerrainEwrtWorld_})
                         .front();
  }
  if (config_.Aa.size() > 0) {
    f.noalias() += config_.Aa * endEffectorDynamicsPtr_
                                    ->getXeeSecondRate(
                                        state, input, {normalTerrainEwrtWorld_})
                                    .front();
  }
  return f;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

VectorFunctionLinearApproximation
EndEffectorDynamicsAccelerationsConstraint::getLinearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  VectorFunctionLinearApproximation linearApproximation =
      VectorFunctionLinearApproximation::Zero(getNumConstraints(time),
                                              state.size(), input.size());

  linearApproximation.f = config_.b;

  // The pose [p; e] and its Jacobian, stacked as getValue() stacks the pose,
  // through all of Ax: the linearization of getValue() for any Ax and any
  // number of rows. The pose depends on the state alone.
  if (config_.Ax.size() > 0) {
    const VectorFunctionLinearApproximation approxPosEeRtWorldEwrtWorld =
        endEffectorDynamicsPtr_
            ->getLinearApproximationPosEesRtWorldEwrtWorld(state)
            .front();
    const VectorFunctionLinearApproximation approxTiltEeRtSurfaceEwrtWorld =
        endEffectorDynamicsPtr_
            ->getLinearApproximationTiltEesRtSurfaceEwrtWorld(
                state, {normalTerrainEwrtWorld_})
            .front();
    vector6_t xee;
    xee << approxPosEeRtWorldEwrtWorld.f, approxTiltEeRtSurfaceEwrtWorld.f;
    matrix_t xeeJacobian(kPoseTangentDim, state.size());
    xeeJacobian << approxPosEeRtWorldEwrtWorld.dfdx,
        approxTiltEeRtSurfaceEwrtWorld.dfdx;
    linearApproximation.f.noalias() += config_.Ax * xee;
    linearApproximation.dfdx.noalias() += config_.Ax * xeeJacobian;
  }

  // The rates of getValue(), from the libraries that give their values: the
  // Jacobians are their exact derivatives, the state-dependent map from the
  // angular velocity and acceleration to the error's rates included.
  if (config_.Av.size() > 0) {
    const VectorFunctionLinearApproximation rateApprox =
        endEffectorDynamicsPtr_
            ->getLinearApproximationXeeRate(state, input,
                                            {normalTerrainEwrtWorld_})
            .front();
    linearApproximation.f.noalias() += config_.Av * rateApprox.f;
    linearApproximation.dfdx.noalias() += config_.Av * rateApprox.dfdx;
    linearApproximation.dfdu.noalias() += config_.Av * rateApprox.dfdu;
  }

  if (config_.Aa.size() > 0) {
    const VectorFunctionLinearApproximation secondRateApprox =
        endEffectorDynamicsPtr_
            ->getLinearApproximationXeeSecondRate(state, input,
                                                  {normalTerrainEwrtWorld_})
            .front();
    linearApproximation.f.noalias() += config_.Aa * secondRateApprox.f;
    linearApproximation.dfdx.noalias() += config_.Aa * secondRateApprox.dfdx;
    linearApproximation.dfdu.noalias() += config_.Aa * secondRateApprox.dfdu;
  }

  return linearApproximation;
}

}  // namespace ocs2::legged
