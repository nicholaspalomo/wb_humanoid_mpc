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

#include "humanoid_common_mpc/cost/BasePoseShapedQuadraticStateCost.h"

#include <memory>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<BasePoseShapedQuadraticStateCost>> BasePoseShapedQuadraticStateCost::Create(
    matrix_t stateWeights,
    const SwitchedModelReferenceManager* absl_nonnull referenceManager,
    const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel) {
  // The constructor is private, so std::make_unique cannot reach it.
  std::unique_ptr<BasePoseShapedQuadraticStateCost> cost =
      absl::WrapUnique(new BasePoseShapedQuadraticStateCost(referenceManager, mpcRobotModel));
  RETURN_IF_ERROR(cost->setStateWeights(std::move(stateWeights)));
  return cost;
}

BasePoseShapedQuadraticStateCost::BasePoseShapedQuadraticStateCost(const SwitchedModelReferenceManager* absl_nonnull referenceManager,
                                                                   const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel)
    : TuningQuadraticStateCost(mpcRobotModel), referenceManagerPtr_(ABSL_DIE_IF_NULL(referenceManager)) {}

vector_t BasePoseShapedQuadraticStateCost::getReference(scalar_t time,
                                                        const vector_t& /*state*/,
                                                        const TargetTrajectories& targetTrajectories,
                                                        const PreComputation& /*preComputation*/) const {
  return referenceManagerPtr_->shapeBasePose(time, mpcRobotModel().getDesiredState(targetTrajectories, time));
}

}  // namespace ocs2::humanoid
