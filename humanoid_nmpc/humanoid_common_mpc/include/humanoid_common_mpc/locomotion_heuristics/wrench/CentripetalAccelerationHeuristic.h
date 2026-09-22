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

#include <string>

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/WrenchHeuristic.h"

namespace ocs2::humanoid {

/**
 * `centripetal_acceleration`: H_f(Theta x pdot) = m omega x pdot (Bledt, Appendix C, Table C.1).
 *
 * The horizontal force that holds the robot on a circular path, shared over the stance feet.
 *
 * Analytic, and the companion of the `high_speed_turning` foot placement: one says where to put the foot for a turn
 * at speed, the other says which way to push once it is there. With omega = psidot e_z and a horizontal velocity the
 * cross product is (-psidot v_y, psidot v_x, 0), a force pointing towards the centre of the turn.
 *
 * TWO THINGS TO KNOW BEFORE LISTING IT.
 *
 * First, it is the only heuristic of the ten whose force is horizontal, so it is the only one whose expression
 * depends on the foot's orientation. Under `useContactBasisVectorInputs: true` the input is parameterized in the
 * local contact frame, so the layer has to route the whole reference through the state-aware
 * setContactForceInWorldFrame() and pay a forward-kinematics pass per shooting node. The layer switches paths by
 * itself when this is listed; the cost is real and is the reason the other nine do not pay it.
 *
 * Second, and more important: this may well be the wrong thing to want on THIS controller. Bledt's control model is a
 * single rigid body over a short horizon, and its force reference is the only thing telling the QP how hard to push.
 * The dynamics here are full centroidal and the state cost already tracks the commanded linear and angular momentum,
 * so m omega x v is the derivative of a momentum the problem is already regulating - an output the optimizer has to
 * produce, not information the reference lacks. Supplying it again at lower fidelity can only help where the
 * optimizer would have been slow to find it, and can fight the momentum cost wherever the two disagree, which they do
 * in double support and wherever the friction cone binds. It is implemented because it is one of Bledt's ten and the
 * seam for it is clean; whether it earns its place here is a question for simulation, and `scale` is the knob.
 */
class CentripetalAccelerationHeuristic final : public WrenchHeuristic {
 public:
  CentripetalAccelerationHeuristic() = default;
  ~CentripetalAccelerationHeuristic() override = default;

  absl::string_view name() const override { return heuristic::kCentripetalAcceleration; }
  absl::Status configure(const LocomotionHeuristicConfig& config, const LocomotionHeuristicModelParameters& model) override;
  std::string describe() const override;

  vector3_t forceOffset(const WrenchHeuristicContext& context, size_t contactIndex) const override;
  bool producesHorizontalForce() const override { return true; }

 private:
  CentripetalAccelerationParameters parameters_;
  scalar_t totalMass_ = 0.0;
  scalar_t totalWeight_ = 0.0;
};

}  // namespace ocs2::humanoid
