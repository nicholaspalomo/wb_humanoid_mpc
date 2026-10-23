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

#pragma once

#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateCost.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The terminal cost (terminal_cost), 1/2 r' Q_final r of the tuning-layout residual against the reference state with
 * its base pose shaped by the locomotion heuristics exactly as the running state costs shape it.
 *
 * Measured against the target trajectories' desired state directly, the reference would be the UNSHAPED target: level
 * roll and pitch and the operator's height. On a robot whose base-pose heuristics are live, that left one term -
 * weighted by terminalCostScaling on the last node, easily more than the whole running horizon's base-pose weight -
 * pulling the end of every horizon back to the pose the heuristics had just moved every other node away from. This
 * routes the reference through SwitchedModelReferenceManager::shapeBasePose(), the one definition of the shaped pose,
 * and nothing else: in particular not through the reference manager's getDesiredState(), which would also pull the
 * procedural arm swing into Q_final. The target is interpolated by the robot model (getDesiredState()).
 *
 * With no base-pose heuristic listed it is TuningQuadraticStateCost of the unshaped target. The reference manager and
 * the robot model must outlive it. Like every OCS2 term it is cloned for each solver thread; its const functions are safe to call
 * concurrently, setStateWeights() is not.
 */
class BasePoseShapedQuadraticStateCost final : public TuningQuadraticStateCost {
 public:
  /**
   * The cost of `stateWeights` (Q_final, the tuning layout's: getStateTangentDim() square) on `mpcRobotModel`, retained
   * with `referenceManager`. InvalidArgument when Q_final is refused (setStateWeights()).
   */
  static absl::StatusOr<std::unique_ptr<BasePoseShapedQuadraticStateCost>> Create(
      matrix_t stateWeights,
      const SwitchedModelReferenceManager* absl_nonnull referenceManager,
      const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel);

  ~BasePoseShapedQuadraticStateCost() override = default;
  BasePoseShapedQuadraticStateCost& operator=(const BasePoseShapedQuadraticStateCost&) = delete;
  BasePoseShapedQuadraticStateCost(BasePoseShapedQuadraticStateCost&&) = delete;
  BasePoseShapedQuadraticStateCost& operator=(BasePoseShapedQuadraticStateCost&&) = delete;
  BasePoseShapedQuadraticStateCost* absl_nonnull clone() const override { return new BasePoseShapedQuadraticStateCost(*this); }

 private:
  BasePoseShapedQuadraticStateCost(const SwitchedModelReferenceManager* absl_nonnull referenceManager,
                                   const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel);
  BasePoseShapedQuadraticStateCost(const BasePoseShapedQuadraticStateCost& other) = default;

  vector_t getReference(scalar_t time,
                        const vector_t& state,
                        const TargetTrajectories& targetTrajectories,
                        const PreComputation& preComputation) const override;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
};

}  // namespace ocs2::humanoid
