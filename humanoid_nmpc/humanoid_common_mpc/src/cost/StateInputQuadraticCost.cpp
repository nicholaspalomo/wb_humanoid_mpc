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

#include "humanoid_common_mpc/cost/StateInputQuadraticCost.h"

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

#include <cmath>
#include <numbers>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
StateInputQuadraticCost::StateInputQuadraticCost(matrix_t Q,
                                                 matrix_t R,
                                                 const SwitchedModelReferenceManager& referenceManager,
                                                 const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : QuadraticStateInputCost(std::move(Q), std::move(R)), referenceManagerPtr_(&referenceManager), mpcRobotModelPtr_(&mpcRobotModel) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

StateInputQuadraticCost::StateInputQuadraticCost(const StateInputQuadraticCost& rhs)
    : QuadraticStateInputCost(rhs), referenceManagerPtr_(rhs.referenceManagerPtr_), mpcRobotModelPtr_(rhs.mpcRobotModelPtr_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::pair<vector_t, vector_t> StateInputQuadraticCost::getStateInputDeviation(scalar_t time,
                                                                              const vector_t& state,
                                                                              const vector_t& input,
                                                                              const TargetTrajectories& targetTrajectories) const {
  // The nominal input hands the robot's weight to the feet the mode schedule calls stance feet, and none to a swinging
  // one. This stays schedule-derived under the contact-implicit formulation, deliberately.
  //
  // It is tempting to see it as a soft `zero_wrench` and remove it along with the hard one. That is a mistake, and an
  // expensive one: it is the difference between a schedule-derived CONSTRAINT, which makes departing from the plan
  // impossible, and a schedule-derived REFERENCE, which is how the reduced-order plan is supposed to guide the
  // whole-body MPC in the first place. The solver overrules this reference whenever anything else pays more.
  //
  // Removing it leaves the contact problem symmetric, so the solver has no reason to prefer lifting either foot. Worse,
  // spreading the weight over ALL feet makes the nominal force on the foot that should be in the air half the body
  // weight, and the complementarity penalty's curvature on that foot's height is
  // complementarityWeight * (f_n/f_ref)^2 / heightReference^2 - about 1950 at half body weight, against the swing
  // height reference's 150. The foot then rises a few millimetres and stops, which is what was observed on hardware-like
  // simulation when this was tried. The contact flags it rests on are now read inside getDesiredInput() rather than
  // here, which is the only change: the reference is still derived from the mode schedule, node by node.
  vector_t xNominal = referenceManagerPtr_->getDesiredState(targetTrajectories, state, time);

  // The contact-force reference, from the reference manager rather than computed here - the same move, and for the
  // same reasons, as in InputQuadraticCost. With no wrench heuristic listed getDesiredInput() makes exactly the call
  // this line used to make, so nothing changes on any robot shipped here; with one listed it is the seam Bledt's H_f
  // heuristics shape (humanoid_nmpc/docs/locomotion_heuristics/README.md).
  const vector_t uNominal = referenceManagerPtr_->getDesiredInput(targetTrajectories, state, time);

  return {state - xNominal, input - uNominal};
}

}  // namespace ocs2::humanoid
