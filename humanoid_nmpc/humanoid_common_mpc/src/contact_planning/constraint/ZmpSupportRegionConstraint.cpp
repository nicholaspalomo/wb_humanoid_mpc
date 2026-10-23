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

#include "humanoid_common_mpc/contact_planning/constraint/ZmpSupportRegionConstraint.h"

#include <array>
#include <cmath>
#include <functional>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::humanoid {

static_assert(kNumContacts == kBipedFootCount, "the support region rows are written for a biped");

std::string ZmpSupportRegionConstraint::describe() const {
  return absl::StrCat("zmp in the support region, box half-widths (", halfWidthX_, ", ", halfWidthY_,
                      ") m around the stance foot / between both feet, big-M disjunction, ", penaltyText());
}

void ZmpSupportRegionConstraint::configure(const ContactPlanningConfig& config) {
  halfWidthX_ = config.zmpSupportRegion.halfWidthX;
  halfWidthY_ = config.zmpSupportRegion.halfWidthY;
  configurePenalty(config, config.zmpSupportRegion.slack);
}

void ZmpSupportRegionConstraint::addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const {
  const scalar_t M = ctx.bigM;
  const std::array<vector2_t, kHorizontalPositionDim>& axes = ctx.axesAt(node);
  const std::array<scalar_t, kHorizontalPositionDim> halfWidth{halfWidthX_, halfWidthY_};
  // +-e_j'(zmp - p_i) in the frame `frame`, with the sign folded in.
  const std::function<void(const std::array<vector2_t, kHorizontalPositionDim>&, size_t, int, scalar_t, Coefficients&, Coefficients&)>
      zmpMinusFoot = [&](const std::array<vector2_t, kHorizontalPositionDim>& frame, size_t foot, int axis, scalar_t sign, Coefficients& xc,
                         Coefficients& uc) {
        for (int w = 0; w < kHorizontalPositionDim; ++w) {
          xc.emplace_back(idx_.foot[foot][w], -sign * frame[static_cast<size_t>(axis)](w));
          uc.emplace_back(idx_.zmp[w], sign * frame[static_cast<size_t>(axis)](w));
        }
      };
  // Single-support boxes, in the stance foot's own footprint frame: +-e_j(psi_i)'(zmp - p_i) <= r_j + M (1 - c_i) +
  // M c_other. The box is the footprint, so it turns with the foot and not with the heading: the heading model lets a
  // stance foot sit up to the hip yaw range away from the heading, and a box aligned with the heading then admitted
  // ZMPs past the footprint's corners and refused ones inside it. psi_i is a decision variable, so the frame is the
  // foot's NOMINAL yaw psi_n (ContactPlanningContext::footAxes), the linearization of e_j(psi)'(zmp - p_i) about
  // (psi_n, zmp - p_i = 0): its first-order heading term (e_j'(psi_n) . 0) (psi - psi_n) vanishes, and what is left out
  // is the product |zmp - p_i| |psi_i - psi_n| <= |r| |psi_i - psi_n|, which the heading re-linearization drives to zero.
  // A stance foot's yaw is pinned while it is down (foot_yaw_pinned_in_contact), so for the foot standing at planning
  // time the nominal is its measured yaw and the box is exact.
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const size_t other = 1 - foot;
    const std::array<vector2_t, kHorizontalPositionDim>& footprint = ctx.footAxesAt(node, foot);
    for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
      for (const scalar_t sign : {1.0, -1.0}) {
        Coefficients xc;
        Coefficients uc;
        zmpMinusFoot(footprint, foot, axis, sign, xc, uc);
        uc.emplace_back(idx_.contact[foot], M);
        uc.emplace_back(idx_.contact[other], -M);
        rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[static_cast<size_t>(axis)] + M, penalty_);
      }
    }
  }
  // Double support, in the heading frame (the region is the bounding box of the two feet, see the class comment, and no
  // single foot's frame is the right one for it): +-e_x'(zmp - (p_L + p_R) / 2) <= r_x + M (1 - c_L) + M (1 - c_R).
  for (const scalar_t sign : {1.0, -1.0}) {
    Coefficients xc;
    Coefficients uc;
    for (int w = 0; w < kHorizontalPositionDim; ++w) {
      xc.emplace_back(idx_.foot[0][w], -0.5 * sign * axes[0](w));
      xc.emplace_back(idx_.foot[1][w], -0.5 * sign * axes[0](w));
      uc.emplace_back(idx_.zmp[w], sign * axes[0](w));
    }
    uc.emplace_back(idx_.contact[0], M);
    uc.emplace_back(idx_.contact[1], M);
    rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[0] + 2.0 * M, penalty_);
  }
  // Double support, lateral axis: upper bound from the left foot, lower bound from the right foot.
  //
  // Both are relaxed by BOTH binaries, exactly like the heading row above, so that they bind only in double support.
  // They used to carry one big-M each, which left one of the two active throughout single support: standing on the
  // left foot, `e_y'(zmp - p_L) <= r_y` is not relaxed by c_L = 1 and states precisely what that foot's own
  // single-support box already states. The bound was therefore correct but duplicated, and since these are SOFT rows
  // the duplicate is not free - a lateral excursion past the support foot paid its slack penalty twice on the
  // outward side and once everywhere else, quietly making the lateral ZMP cost asymmetric in single support, which is
  // most of a stride. Relaxing by the other foot as well removes the duplicate without loosening anything: in single
  // support the surviving constraint is the single-support box, which is the same inequality.
  {
    Coefficients xc;
    Coefficients uc;  // e_y'(zmp - p_L) <= r + M (1 - c_L) + M (1 - c_R)
    zmpMinusFoot(axes, /*foot=*/0, /*axis=*/1, /*sign=*/1.0, xc, uc);
    uc.emplace_back(idx_.contact[0], M);
    uc.emplace_back(idx_.contact[1], M);
    rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[1] + 2.0 * M, penalty_);
  }
  {
    Coefficients xc;
    Coefficients uc;  // -e_y'(zmp - p_R) <= r + M (1 - c_L) + M (1 - c_R)
    zmpMinusFoot(axes, /*foot=*/1, /*axis=*/1, /*sign=*/-1.0, xc, uc);
    uc.emplace_back(idx_.contact[0], M);
    uc.emplace_back(idx_.contact[1], M);
    rows.addSoft(xc, uc, -kLipLooseBound, halfWidth[1] + 2.0 * M, penalty_);
  }
}

}  // namespace ocs2::humanoid
