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

#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"

#include <cmath>

namespace ocs2::humanoid {

std::pair<scalar_t, scalar_t> ContactPlanningContext::frameTerm(int node, int axis, const vector2_t& dNominal) const {
  if (!hasHeading()) return {0.0, 0.0};
  const std::array<vector2_t, 2>& a = axesAt(node);
  const vector2_t de = (axis == 0) ? a[1] : vector2_t(-a[0]);
  const scalar_t g = de.dot(dNominal);
  return {g, -g * nominal->heading[static_cast<size_t>(node)]};
}

void ContactPlanningContext::computeAxes() {
  axes.resize(static_cast<size_t>(numNodes) + 1);
  for (int k = 0; k <= numNodes; ++k) {
    const scalar_t theta = hasHeading() ? nominal->heading[static_cast<size_t>(k)] : input->yaw;
    axes[static_cast<size_t>(k)] = {vector2_t(std::cos(theta), std::sin(theta)), vector2_t(-std::sin(theta), std::cos(theta))};
  }
}

HeadingNominal nominalFromSolution(const Layout& layout, const std::vector<vector_t>& x) {
  HeadingNominal nominal;
  const size_t n = x.size();
  nominal.heading.resize(n);
  nominal.com.resize(n);
  nominal.feet.resize(n);
  const int cx = layout.state(var::kComX);
  for (size_t k = 0; k < n; ++k) {
    nominal.heading[k] = layout.hasHeading ? x[k](layout.heading) : 0.0;
    nominal.com[k] = x[k].segment<2>(cx);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) nominal.feet[k][foot] = x[k].segment<2>(layout.state(var::footX(foot)));
  }
  return nominal;
}

}  // namespace ocs2::humanoid
