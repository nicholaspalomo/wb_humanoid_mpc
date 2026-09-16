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

#include "humanoid_common_mpc/contact_planning/constraint/ReachabilityConstraint.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string ReachabilityConstraint::describe() const {
  std::ostringstream out;
  out << "|e_x . (p_i - c)| <= " << params_.reachX << " m, " << params_.reachYInner
      << " <= side_i e_y . (p_i - c) <= " << params_.reachYOuter << " m at every node (first-order in the heading), " << penaltyText();
  return out.str();
}

void ReachabilityConstraint::configure(const ContactPlanningConfig& config) {
  if (config.reachability.reachX <= 0.0) throw std::invalid_argument("[reachability] reachX must be positive");
  if (config.reachability.reachYOuter <= config.reachability.reachYInner) {
    throw std::invalid_argument("[reachability] reachYOuter must exceed reachYInner");
  }
  params_ = config.reachability;
  configurePenalty(config, config.reachability.slack, "reachability");
}

void ReachabilityConstraint::addRows(const ContactPlanningContext& ctx, int node, RowBuilder& rows) const {
  const std::array<vector2_t, 2>& axes = ctx.axesAt(node);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int axis = 0; axis < 2; ++axis) {
      Coefficients xc;
      for (int w = 0; w < 2; ++w) {
        xc.push_back({idx_.foot[foot][w], axes[static_cast<size_t>(axis)](w)});
        xc.push_back({idx_.com[w], -axes[static_cast<size_t>(axis)](w)});
      }
      scalar_t lower, upper;
      if (axis == 0) {
        lower = -params_.reachX;
        upper = params_.reachX;
      } else if (foot == 0) {
        lower = params_.reachYInner;
        upper = params_.reachYOuter;
      } else {
        lower = -params_.reachYOuter;
        upper = -params_.reachYInner;
      }
      const vector2_t dNominal =
          ctx.hasHeading() ? vector2_t(ctx.nominal->feet[static_cast<size_t>(node)][foot] - ctx.nominal->com[static_cast<size_t>(node)])
                           : vector2_t(vector2_t::Zero());
      const auto [g, offset] = ctx.frameTerm(node, axis, dNominal);
      if (ctx.hasHeading()) xc.push_back({idx_.heading, g});
      rows.addSoft(xc, {}, lower - offset, upper - offset, penalty_);
    }
  }
}

}  // namespace ocs2::humanoid
