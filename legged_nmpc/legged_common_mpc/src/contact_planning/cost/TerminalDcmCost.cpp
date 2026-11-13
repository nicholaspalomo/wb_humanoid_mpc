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

#include "legged_common_mpc/contact_planning/cost/TerminalDcmCost.h"

#include <cmath>
#include <string>

#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "legged_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "legged_common_mpc/contact_planning/problem/StageAccumulator.h"

namespace ocs2::legged {

std::string TerminalDcmCost::describe() const {
  return trackVelocityCommand_
             ? weightLine(
                   "w ||xi_N - zmp_{N-1} - R_z(omega_cmd N dt) v_cmd / "
                   "omega||^2 "
                   "on the last running node (keeps walking)")
             : weightLine(
                   "w ||xi_N - zmp_{N-1}||^2 on the last running node "
                   "(terminal capturability)");
}

void TerminalDcmCost::configure(const ContactPlanningConfig& config) {
  weight_ = config.terminalDcm.weight;
  trackVelocityCommand_ = config.terminalDcm.trackVelocityCommand;
}

void TerminalDcmCost::addToStage(const ContactPlanningContext& ctx,
                                 int node,
                                 StageAccumulator& stage) const {
  // The terminal DCM is written on the last running node, where the ZMP of the
  // last interval is a variable: xi_N - zmp_{N-1} = e^{omega dt} (xi_{N-1} -
  // zmp_{N-1}), so w ||xi_N - zmp_{N-1} - r||^2 is w e^{2 omega dt} ||xi_{N-1}
  // - zmp_{N-1} - e^{-omega dt} r||^2 exactly.
  const scalar_t omega = ctx.omega;
  const scalar_t growth = std::exp(omega * ctx.dt);
  // xi_N, the DCM the target is for, is at node N = node + 1, so its velocity
  // is the command of that node: the captured one turned by omega_cmd N dt.
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
      ctx.getCmdLinvelBaseRtWorldEwrtWorldXy(node + 1);
  for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
    // With trackVelocityCommand the target of the terminal DCM is zmp_{N-1} +
    // v_cmd / omega, the offset of a CoM over the foot that keeps moving at the
    // commanded velocity, instead of the rest condition xi_N = zmp_{N-1}. The
    // residual used to apply that offset to xi_{N-1} instead, which put the
    // target of xi_N at e^{omega dt} v_cmd / omega - 35 % further ahead than
    // documented at dt 0.1 on the Atlas's 1.0805 m pendulum.
    const scalar_t refPosDcmRtZmpEwrtWorld =
        trackVelocityCommand_ ? cmdLinvelBaseRtWorldEwrtWorldXy(axis) / omega
                              : 0.0;
    stage.addQuadraticResidual(
        {{idx_.posComRtWorldEwrtWorld[axis], 1.0},
         {idx_.linvelComRtWorldEwrtWorld[axis], 1.0 / omega}},
        {{idx_.posZmpRtWorldEwrtWorld[axis], -1.0}},
        -refPosDcmRtZmpEwrtWorld / growth, weight_ * growth * growth);
  }
}

}  // namespace ocs2::legged
