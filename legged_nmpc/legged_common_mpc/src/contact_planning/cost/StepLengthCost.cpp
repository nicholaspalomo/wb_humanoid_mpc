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

#include "legged_common_mpc/contact_planning/cost/StepLengthCost.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "legged_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "legged_common_mpc/contact_planning/problem/StageAccumulator.h"

namespace ocs2::legged {

std::string StepLengthCost::describe() const {
  return absl::StrCat(
      weightLine("w sum_i ||dp_{i,k} - d_nom,k (1 - c_{i,k})||^2 on the "
                 "running nodes, d_nom = v_cmd dt T_stride / T_swing, d_nom,k "
                 "= R_z(omega_cmd k dt) d_nom"),
      ", T_stride / T_swing = ", strideToSwingRatio_);
}

void StepLengthCost::configure(const ContactPlanningConfig& config) {
  weight_ = config.stepLength.weight;
  const GaitLimits& limits = config.shared.gaitLimits;
  const scalar_t doubleSupport = std::max(0.0, limits.minDoubleSupportDuration);
  strideToSwingRatio_ =
      2.0 * (limits.minSwingDuration + doubleSupport) / limits.minSwingDuration;
}

vector2_t StepLengthCost::computeNomErrPosFootRtWorldEwrtWorldXy(
    const vector2_t& cmdLinvelBaseRtWorldEwrtWorldXy, scalar_t dt) const {
  return cmdLinvelBaseRtWorldEwrtWorldXy * (dt * strideToSwingRatio_);
}

void StepLengthCost::addToStage(const ContactPlanningContext& ctx,
                                int node,
                                StageAccumulator& stage) const {
  // The input's command is the heading-frame command rotated by the commanded
  // heading at the planning instant; at node k the commanded heading has
  // turned by omega_cmd k dt (the ramp heading_tracking follows).
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
      ctx.getCmdLinvelBaseRtWorldEwrtWorldXy(node);
  const vector2_t nomErrPosFootRtWorldEwrtWorldXy =
      computeNomErrPosFootRtWorldEwrtWorldXy(cmdLinvelBaseRtWorldEwrtWorldXy,
                                             ctx.dt);
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
      // dp - d_nom (1 - c) = dp + d_nom c - d_nom
      stage.addQuadraticResidual(
          /*xCoefficients=*/{},
          {{idx_.errPosFeetRtWorldEwrtWorld[foot][axis], 1.0},
           {idx_.contact[foot], nomErrPosFootRtWorldEwrtWorldXy(axis)}},
          -nomErrPosFootRtWorldEwrtWorldXy(axis), weight_);
    }
  }
}

}  // namespace ocs2::legged
