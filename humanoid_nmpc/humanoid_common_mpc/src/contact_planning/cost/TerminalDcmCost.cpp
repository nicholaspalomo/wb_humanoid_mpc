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

#include "humanoid_common_mpc/contact_planning/cost/TerminalDcmCost.h"

#include <cmath>
#include <string>

namespace ocs2::humanoid {

std::string TerminalDcmCost::describe() const {
  return trackCommandedVelocity_ ? weightLine("w ||xi_N - zmp_{N-1} - v_cmd / omega||^2 on the last running node (keeps walking)")
                                 : weightLine("w ||xi_N - zmp_{N-1}||^2 on the last running node (terminal capturability)");
}

void TerminalDcmCost::configure(const ContactPlanningConfig& config) {
  weight_ = config.terminalDcm.weight;
  trackCommandedVelocity_ = config.terminalDcm.trackCommandedVelocity;
}

void TerminalDcmCost::addToStage(const ContactPlanningContext& ctx, int /*node*/, StageAccumulator& stage) const {
  // The terminal DCM is written on the last running node, where the ZMP of the last interval is a variable:
  // xi_N - zmp_{N-1} = e^{omega dt} (xi_{N-1} - zmp_{N-1}), so w ||xi_N - zmp_{N-1} - r||^2 is
  // w e^{2 omega dt} ||xi_{N-1} - zmp_{N-1} - e^{-omega dt} r||^2 exactly.
  const scalar_t omega = ctx.omega;
  const scalar_t growth = std::exp(omega * ctx.dt);
  for (int axis = 0; axis < 2; ++axis) {
    // With trackCommandedVelocity the target of the terminal DCM is zmp_{N-1} + v_cmd / omega, the offset of a CoM
    // over the foot that keeps moving at the commanded velocity, instead of the rest condition xi_N = zmp_{N-1}. The
    // residual used to apply that offset to xi_{N-1} instead, which put the target of xi_N at e^{omega dt} v_cmd /
    // omega - 35 % further ahead than documented at dt 0.1 on the Atlas's 1.0805 m pendulum.
    const scalar_t target = trackCommandedVelocity_ ? ctx.input->velocityCommand(axis) / omega : 0.0;
    stage.addQuadraticResidual({{idx_.com[axis], 1.0}, {idx_.vel[axis], 1.0 / omega}}, {{idx_.zmp[axis], -1.0}}, -target / growth,
                               weight_ * growth * growth);
  }
}

}  // namespace ocs2::humanoid
