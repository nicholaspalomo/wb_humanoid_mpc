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

#include "legged_common_mpc/contact_planning/constraint/ReachabilityConstraint.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "legged_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "legged_common_mpc/contact_planning/problem/LipCoefficients.h"
#include "legged_common_mpc/contact_planning/problem/RowBuilder.h"

namespace ocs2::legged {

std::string ReachabilityConstraint::describe() const {
  return absl::StrCat(
      "|e_x . (p_i - c)| <= ", params_.maxNormPosFootRtComEwrtHeadingX, " m, ",
      params_.minPosFootRtComEwrtHeadingY,
      " <= side_i e_y . (p_i - c) <= ", params_.maxPosFootRtComEwrtHeadingY,
      " m at every node (first-order in the heading), ", penaltyText());
}

void ReachabilityConstraint::configure(const ContactPlanningConfig& config) {
  params_ = config.reachability;
  configurePenalty(config, config.reachability.slack);
}

void ReachabilityConstraint::addRows(const ContactPlanningContext& ctx,
                                     int node,
                                     RowBuilder& rows) const {
  const std::array<vector2_t, kHorizontalPositionDim>& rotmatHeadingRtWorld =
      ctx.getRotmatHeadingRtWorld(node);
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
      Coefficients xc;
      for (int w = 0; w < kHorizontalPositionDim; ++w) {
        xc.emplace_back(idx_.posFeetRtWorldEwrtWorld[foot][w],
                        rotmatHeadingRtWorld[static_cast<size_t>(axis)](w));
        xc.emplace_back(idx_.posComRtWorldEwrtWorld[w],
                        -rotmatHeadingRtWorld[static_cast<size_t>(axis)](w));
      }
      scalar_t lower = 0.0;
      scalar_t upper = 0.0;
      if (axis == 0) {
        lower = -params_.maxNormPosFootRtComEwrtHeadingX;
        upper = params_.maxNormPosFootRtComEwrtHeadingX;
      } else if (foot == 0) {
        lower = params_.minPosFootRtComEwrtHeadingY;
        upper = params_.maxPosFootRtComEwrtHeadingY;
      } else {
        lower = -params_.maxPosFootRtComEwrtHeadingY;
        upper = -params_.minPosFootRtComEwrtHeadingY;
      }
      const vector2_t nomPosFootRtComEwrtWorldXy =
          ctx.hasHeading()
              ? vector2_t(
                    ctx.nominal
                        ->nomPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(
                            node)][foot] -
                    ctx.nominal
                        ->nomPosComRtWorldEwrtWorldXyTraj[static_cast<size_t>(
                            node)])
              : vector2_t(vector2_t::Zero());
      const ContactPlanningContext::HeadingFrameTerm frameTerm =
          ctx.frameTerm(node, axis, nomPosFootRtComEwrtWorldXy);
      const scalar_t g = frameTerm.gain;
      const scalar_t offset = frameTerm.offset;
      if (ctx.hasHeading()) xc.emplace_back(idx_.headingModelYaw, g);
      rows.addSoft(xc, /*uCoefficients=*/{}, lower - offset, upper - offset,
                   penalty_);
    }
  }
}

}  // namespace ocs2::legged
