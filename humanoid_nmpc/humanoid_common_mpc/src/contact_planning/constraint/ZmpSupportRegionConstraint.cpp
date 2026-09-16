/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/contact_planning/constraint/ZmpSupportRegionConstraint.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the support region rows are written for a biped");

std::string ZmpSupportRegionConstraint::describe() const {
  std::ostringstream out;
  out << "zmp in the support region, box half-widths (" << halfWidthX_ << ", " << halfWidthY_
      << ") m around the stance foot / between both feet, big-M disjunction, " << penaltyText();
  return out.str();
}

void ZmpSupportRegionConstraint::configure(const ContactPlanningConfig& config) {
  if (config.zmpSupportRegion.halfWidthX <= 0.0 || config.zmpSupportRegion.halfWidthY <= 0.0) {
    throw std::invalid_argument("[zmp_support_region] ZMP half widths must be positive");
  }
  halfWidthX_ = config.zmpSupportRegion.halfWidthX;
  halfWidthY_ = config.zmpSupportRegion.halfWidthY;
  configurePenalty(config, config.zmpSupportRegion.slack, "zmp_support_region");
}

void ZmpSupportRegionConstraint::addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const {
  const scalar_t M = ctx.bigM;
  const std::array<vector2_t, 2>& axes = ctx.axesAt(node);
  const std::array<scalar_t, 2> halfWidth{halfWidthX_, halfWidthY_};
  // +-e_j'(zmp - p_i), with the sign folded in.
  const auto zmpMinusFoot = [&](size_t foot, int axis, scalar_t sign, Coefficients& xc, Coefficients& uc) {
    for (int w = 0; w < 2; ++w) {
      xc.push_back({idx_.foot[foot][w], -sign * axes[static_cast<size_t>(axis)](w)});
      uc.push_back({idx_.zmp[w], sign * axes[static_cast<size_t>(axis)](w)});
    }
  };
  // Single-support boxes: +-e_j'(zmp - p_i) <= r_j + M (1 - c_i) + M c_other.
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    const size_t other = 1 - foot;
    for (int axis = 0; axis < 2; ++axis) {
      for (const scalar_t sign : {1.0, -1.0}) {
        Coefficients xc, uc;
        zmpMinusFoot(foot, axis, sign, xc, uc);
        uc.push_back({idx_.contact[foot], M});
        uc.push_back({idx_.contact[other], -M});
        rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[static_cast<size_t>(axis)] + M, penalty_);
      }
    }
  }
  // Double support, heading axis: +-e_x'(zmp - (p_L + p_R) / 2) <= r_x + M (1 - c_L) + M (1 - c_R).
  for (const scalar_t sign : {1.0, -1.0}) {
    Coefficients xc, uc;
    for (int w = 0; w < 2; ++w) {
      xc.push_back({idx_.foot[0][w], -0.5 * sign * axes[0](w)});
      xc.push_back({idx_.foot[1][w], -0.5 * sign * axes[0](w)});
      uc.push_back({idx_.zmp[w], sign * axes[0](w)});
    }
    uc.push_back({idx_.contact[0], M});
    uc.push_back({idx_.contact[1], M});
    rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[0] + 2.0 * M, penalty_);
  }
  // Double support, lateral axis: upper bound from the left foot, lower bound from the right foot.
  {
    Coefficients xc, uc;  // e_y'(zmp - p_L) <= r + M (1 - c_L)
    zmpMinusFoot(0, 1, 1.0, xc, uc);
    uc.push_back({idx_.contact[0], M});
    rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[1] + M, penalty_);
  }
  {
    Coefficients xc, uc;  // -e_y'(zmp - p_R) <= r + M (1 - c_R)
    zmpMinusFoot(1, 1, -1.0, xc, uc);
    uc.push_back({idx_.contact[1], M});
    rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[1] + M, penalty_);
  }
}

}  // namespace ocs2::humanoid
