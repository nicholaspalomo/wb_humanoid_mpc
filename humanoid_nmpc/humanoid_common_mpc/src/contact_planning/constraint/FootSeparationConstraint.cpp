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

#include "humanoid_common_mpc/contact_planning/constraint/FootSeparationConstraint.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the separation rows are written for a biped");

std::string FootSeparationConstraint::describe() const {
  std::ostringstream out;
  out << "|e_x . (p_L - p_R)| <= " << params_.maxStepLength << " m, " << params_.minStepWidth
      << " <= e_y . (p_L - p_R) <= " << params_.maxStepWidth << " m at every node (first-order in the heading), " << penaltyText();
  return out.str();
}

void FootSeparationConstraint::configure(const ContactPlanningConfig& config) {
  if (config.footSeparation.minStepWidth <= 0.0 || config.footSeparation.maxStepWidth < config.footSeparation.minStepWidth) {
    throw std::invalid_argument("[foot_separation] need 0 < minStepWidth <= maxStepWidth");
  }
  if (config.footSeparation.maxStepLength <= 0.0) throw std::invalid_argument("[foot_separation] maxStepLength must be positive");
  params_ = config.footSeparation;
  configurePenalty(config, config.footSeparation.slack, "foot_separation");
}

void FootSeparationConstraint::addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const {
  const std::array<vector2_t, 2>& axes = ctx.axesAt(node);
  for (int axis = 0; axis < 2; ++axis) {
    Coefficients xc;
    for (int w = 0; w < 2; ++w) {
      xc.push_back({idx_.foot[0][w], axes[static_cast<size_t>(axis)](w)});
      xc.push_back({idx_.foot[1][w], -axes[static_cast<size_t>(axis)](w)});
    }
    const vector2_t dNominal =
        ctx.hasHeading() ? vector2_t(ctx.nominal->feet[static_cast<size_t>(node)][0] - ctx.nominal->feet[static_cast<size_t>(node)][1])
                         : vector2_t(vector2_t::Zero());
    const auto [g, offset] = ctx.frameTerm(node, axis, dNominal);
    if (ctx.hasHeading()) xc.push_back({idx_.heading, g});
    if (axis == 0) {
      rows.addSoft(xc, {}, -params_.maxStepLength - offset, params_.maxStepLength - offset, penalty_);
    } else {
      rows.addSoft(xc, {}, params_.minStepWidth - offset, params_.maxStepWidth - offset, penalty_);
    }
  }
}

}  // namespace ocs2::humanoid
