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

#include "legged_wb_mpc/constraint/EndEffectorDynamicsLinearAccConstraint.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/constraint/ConstraintOrder.h"
#include "ocs2_core/constraint/StateInputConstraint.h"

#include "legged_wb_mpc/end_effector/EndEffectorDynamics.h"

namespace ocs2::legged {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
absl::StatusOr<std::unique_ptr<EndEffectorDynamicsLinearAccConstraint>>
EndEffectorDynamicsLinearAccConstraint::Create(
    const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
    size_t numConstraints,
    Config config) {
  if (endEffectorDynamics.getIds().size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("[EndEffectorDynamicsLinearAccConstraint] this class only "
                     "accepts a single end-effector; "
                     "the end-effector dynamics has ",
                     endEffectorDynamics.getIds().size(), ": [",
                     absl::StrJoin(endEffectorDynamics.getIds(), ", "), "]"));
  }
  return absl::WrapUnique(new EndEffectorDynamicsLinearAccConstraint(
      endEffectorDynamics, numConstraints, std::move(config)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
EndEffectorDynamicsLinearAccConstraint::EndEffectorDynamicsLinearAccConstraint(
    const EndEffectorDynamics<scalar_t>& endEffectorDynamics,
    size_t numConstraints,
    Config config)
    : StateInputConstraint(ConstraintOrder::Linear),
      endEffectorDynamicsPtr_(endEffectorDynamics.clone()),
      numConstraints_(numConstraints),
      config_(std::move(config)) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
EndEffectorDynamicsLinearAccConstraint::EndEffectorDynamicsLinearAccConstraint(
    const EndEffectorDynamicsLinearAccConstraint& rhs)
    : StateInputConstraint(rhs),
      endEffectorDynamicsPtr_(rhs.endEffectorDynamicsPtr_->clone()),
      numConstraints_(rhs.numConstraints_),
      config_(rhs.config_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void EndEffectorDynamicsLinearAccConstraint::configure(Config&& config) {
  config_ = std::move(config);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t EndEffectorDynamicsLinearAccConstraint::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  vector_t f = config_.b;
  if (config_.Ax.size() > 0) {
    f.noalias() +=
        config_.Ax *
        endEffectorDynamicsPtr_->getPosEesRtWorldEwrtWorld(state).front();
  }
  if (config_.Av.size() > 0) {
    f.noalias() += config_.Av * endEffectorDynamicsPtr_
                                    ->getLinvelEesRtWorldEwrtWorld(state, input)
                                    .front();
  }
  if (config_.Aa.size() > 0) {
    f.noalias() += config_.Aa * endEffectorDynamicsPtr_
                                    ->getLinaccEesRtWorldEwrtWorld(state, input)
                                    .front();
  }
  return f;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation
EndEffectorDynamicsLinearAccConstraint::getLinearApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  VectorFunctionLinearApproximation linearApproximation =
      VectorFunctionLinearApproximation::Zero(getNumConstraints(time),
                                              state.size(), input.size());

  linearApproximation.f = config_.b;

  if (config_.Ax.size() > 0) {
    const VectorFunctionLinearApproximation approxPosEeRtWorldEwrtWorld =
        endEffectorDynamicsPtr_
            ->getLinearApproximationPosEesRtWorldEwrtWorld(state)
            .front();
    linearApproximation.f.noalias() +=
        config_.Ax * approxPosEeRtWorldEwrtWorld.f;
    linearApproximation.dfdx.noalias() +=
        config_.Ax * approxPosEeRtWorldEwrtWorld.dfdx;
  }

  if (config_.Av.size() > 0) {
    const VectorFunctionLinearApproximation approxLinvelEeRtWorldEwrtWorld =
        endEffectorDynamicsPtr_
            ->getLinearApproximationLinvelEesRtWorldEwrtWorld(state, input)
            .front();
    linearApproximation.f.noalias() +=
        config_.Av * approxLinvelEeRtWorldEwrtWorld.f;
    linearApproximation.dfdx.noalias() +=
        config_.Av * approxLinvelEeRtWorldEwrtWorld.dfdx;
    linearApproximation.dfdu.noalias() +=
        config_.Av * approxLinvelEeRtWorldEwrtWorld.dfdu;
  }

  if (config_.Aa.size() > 0) {
    const VectorFunctionLinearApproximation approxLinaccEeRtWorldEwrtWorld =
        endEffectorDynamicsPtr_
            ->getLinearApproximationLinaccEesRtWorldEwrtWorld(state, input)
            .front();
    linearApproximation.f.noalias() +=
        config_.Aa * approxLinaccEeRtWorldEwrtWorld.f;
    linearApproximation.dfdx.noalias() +=
        config_.Aa * approxLinaccEeRtWorldEwrtWorld.dfdx;
    linearApproximation.dfdu.noalias() +=
        config_.Aa * approxLinaccEeRtWorldEwrtWorld.dfdu;
  }

  return linearApproximation;
}

}  // namespace ocs2::legged
