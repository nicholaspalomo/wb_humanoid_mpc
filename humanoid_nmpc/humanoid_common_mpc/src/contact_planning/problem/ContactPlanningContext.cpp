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

#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "ocs2_core/Dimensions.h"

namespace ocs2::humanoid {

ContactPlanningContext::HeadingFrameTerm ContactPlanningContext::frameTerm(int node, int axis, const vector2_t& dNominal) const {
  if (!hasHeading()) return {};
  const std::array<vector2_t, kHorizontalPositionDim>& a = axesAt(node);
  const vector2_t de = (axis == 0) ? a[1] : vector2_t(-a[0]);
  const scalar_t g = de.dot(dNominal);
  return {.gain = g, .offset = -g * nominal->heading[static_cast<size_t>(node)]};
}

namespace {
std::array<vector2_t, kHorizontalPositionDim> frameAxes(scalar_t yaw) {
  return {vector2_t(std::cos(yaw), std::sin(yaw)), vector2_t(-std::sin(yaw), std::cos(yaw))};
}
}  // namespace

void ContactPlanningContext::computeAxes() {
  axes.resize(static_cast<size_t>(numNodes) + 1);
  footAxes.resize(static_cast<size_t>(numNodes) + 1);
  const bool withFootYaws = hasHeading() && nominal->footYaws.size() > static_cast<size_t>(numNodes);
  for (int k = 0; k <= numNodes; ++k) {
    const size_t node = static_cast<size_t>(k);
    axes[node] = frameAxes(hasHeading() ? nominal->heading[node] : input->yaw);
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      footAxes[node][foot] = withFootYaws ? frameAxes(nominal->footYaws[node][foot]) : axes[node];
    }
  }
}

HeadingNominal nominalFromSolution(const Layout& layout, const std::vector<vector_t>& x) {
  HeadingNominal nominal;
  const size_t n = x.size();
  nominal.heading.resize(n);
  nominal.com.resize(n);
  nominal.feet.resize(n);
  if (layout.hasHeading) nominal.footYaws.resize(n);
  const int cx = layout.state(var::kComX);
  for (size_t k = 0; k < n; ++k) {
    nominal.heading[k] = layout.hasHeading ? x[k](layout.heading) : 0.0;
    nominal.com[k] = x[k].segment<kHorizontalPositionDim>(cx);
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      nominal.feet[k][foot] = x[k].segment<kHorizontalPositionDim>(layout.state(var::footX(foot)));
      if (layout.hasHeading) nominal.footYaws[k][foot] = x[k](layout.footYaw(foot));
    }
  }
  return nominal;
}

}  // namespace ocs2::humanoid
