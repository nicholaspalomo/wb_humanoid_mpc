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

#include <ocs2_core/cost/QuadraticStateCost.h>

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The terminal cost, 0.5 (x - x_ref)' Q_final (x - x_ref), with x_ref's base pose shaped by the locomotion heuristics
 * exactly as the running state costs shape it.
 *
 * OCS2's QuadraticStateCost measures the deviation from `targetTrajectories.getDesiredState(time)` directly, which is
 * the UNSHAPED target: level roll and pitch and the operator's height. On a robot whose base-pose heuristics are live,
 * that left one term - weighted by terminalCostScaling on the last node, easily more than the whole running horizon's
 * base-pose weight - pulling the end of every horizon back to the pose the heuristics had just moved every other node
 * away from. This routes the reference through SwitchedModelReferenceManager::shapeBasePose(), the one definition of
 * the shaped pose, and nothing else: in particular not through getDesiredState(), which would also pull the procedural
 * arm swing into Q_final.
 *
 * Bit for bit the plain QuadraticStateCost when no base-pose heuristic is listed. A subclass rather than a new term,
 * so that the gains updater's `get<QuadraticStateCost>("terminalCost").setGains()` keeps working.
 */
class BasePoseShapedQuadraticStateCost final : public QuadraticStateCost {
 public:
  BasePoseShapedQuadraticStateCost(matrix_t Q, const SwitchedModelReferenceManager& referenceManager);
  ~BasePoseShapedQuadraticStateCost() override = default;
  BasePoseShapedQuadraticStateCost* clone() const override { return new BasePoseShapedQuadraticStateCost(*this); }

 protected:
  BasePoseShapedQuadraticStateCost(const BasePoseShapedQuadraticStateCost& other) = default;
  vector_t getStateDeviation(scalar_t time, const vector_t& state, const TargetTrajectories& targetTrajectories) const override;

 private:
  const SwitchedModelReferenceManager* referenceManagerPtr_;
};

}  // namespace ocs2::humanoid
