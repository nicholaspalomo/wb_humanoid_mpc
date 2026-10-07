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

#pragma once

#include <cassert>

#include "absl/base/nullability.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/manifold/StateManifold.h>
#include <ocs2_core/misc/LinearInterpolation.h>

namespace ocs2 {

/**
 * Interpolates a state trajectory at `time`: linearly (LinearInterpolation::interpolate, bit for bit) when
 * `stateManifold` is null, and along the manifold's shortest-path geodesic between the two neighboring nodes otherwise
 * (a slerp on every quaternion block). The time lookup, the treatment of repeated (event) times and the clamping outside
 * the time range are LinearInterpolation's.
 */
inline vector_t interpolateStateTrajectory(const StateManifold* absl_nullable stateManifold,
                                           scalar_t time,
                                           const scalar_array_t& timeTrajectory,
                                           const vector_array_t& stateTrajectory) {
  if (stateManifold == nullptr) {
    return LinearInterpolation::interpolate(time, timeTrajectory, stateTrajectory);
  }
  assert(!stateTrajectory.empty());
  if (stateTrajectory.size() == 1) {
    return stateTrajectory.front();
  }
  const LinearInterpolation::index_alpha_t indexAlpha = LinearInterpolation::timeSegment(time, timeTrajectory);
  const vector_t& lhs = stateTrajectory[indexAlpha.first];
  const vector_t& rhs = stateTrajectory[indexAlpha.first + 1];
  // LinearInterpolation's alpha is the weight of the left node: 1 at its time, 0 at the right node's. Nodes of different
  // sizes snap to the closer one, as in LinearInterpolation.
  if (lhs.size() != rhs.size()) {
    return (indexAlpha.second > 0.5) ? lhs : rhs;
  }
  if (indexAlpha.second >= 1.0) {
    return lhs;
  }
  if (indexAlpha.second <= 0.0) {
    return rhs;
  }
  return stateManifold->interpolate(lhs, rhs, 1.0 - indexAlpha.second);
}

}  // namespace ocs2
