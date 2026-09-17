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

#include "humanoid_common_mpc/contact_planning/cost/StepLengthCost.h"

#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string StepLengthCost::describe() const {
  std::ostringstream out;
  out << weightLine("w sum_i ||dp_{i,k} - d_nom (1 - c_{i,k})||^2 on the running nodes, d_nom = v_cmd dt T_stride / T_swing")
      << ", T_stride / T_swing = " << strideToSwingRatio_;
  return out.str();
}

void StepLengthCost::configure(const ContactPlanningConfig& config) {
  checkWeight("step_length", config.stepLength.weight);
  weight_ = config.stepLength.weight;
  const GaitLimits& limits = config.shared.gaitLimits;
  if (limits.minSwingDuration <= 0.0) throw std::invalid_argument("step_length: shared.gait_limits.minSwingDuration must be positive");
  const scalar_t doubleSupport = std::max(0.0, limits.minDoubleSupportDuration);
  // T_stride / T_swing of the nominal cadence. Walking: a stride is the two swings plus the two double supports between
  // them, T_stride = 2 (T_swing + T_ds). Running: the other foot's stance is shortened at both ends by a flight, and a
  // foot's own air time already contains those two flights, so T_stance = T_swing - 2 T_f and
  // T_stride = 2 (T_stance + T_f) = 2 (T_swing - T_f). One expression covers both, with the flight counted only where
  // the gait limits allow one; otherwise a running gait would be given the walking ratio and its steps planned too long.
  const scalar_t flight = limits.maxFlightDuration > 0.0 ? std::max(0.0, limits.minFlightDuration) : 0.0;
  strideToSwingRatio_ = std::max(0.0, 2.0 * (limits.minSwingDuration + doubleSupport - flight) / limits.minSwingDuration);
}

vector2_t StepLengthCost::nominalDisplacementPerNode(const vector2_t& velocityCommand, scalar_t dt) const {
  return velocityCommand * (dt * strideToSwingRatio_);
}

void StepLengthCost::addToStage(const ContactPlanningContext& ctx, int /*node*/, StageAccumulator& stage) const {
  const vector2_t dNominal = nominalDisplacementPerNode(ctx.input->velocityCommand, ctx.dt);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int axis = 0; axis < 2; ++axis) {
      // dp - d_nom (1 - c) = dp + d_nom c - d_nom
      stage.addQuadraticResidual({}, {{idx_.footDelta[foot][axis], 1.0}, {idx_.contact[foot], dNominal(axis)}}, -dNominal(axis), weight_);
    }
  }
}

}  // namespace ocs2::humanoid
