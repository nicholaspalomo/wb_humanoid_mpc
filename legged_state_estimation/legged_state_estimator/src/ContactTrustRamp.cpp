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

#include "legged_state_estimator/ContactTrustRamp.h"

#include <algorithm>

#include "ocs2_core/Types.h"

namespace ocs2::legged::estimation {
namespace {

/** The cubic smoothstep w^2 (kSmoothstepCubicOffset - 2 w). */
constexpr scalar_t kSmoothstepCubicOffset = 3.0;

}  // namespace

scalar_t contactTrustScale(scalar_t timeToNearestSwitch,
                           scalar_t rampDuration,
                           scalar_t maxScale) {
  // Also false for a NaN or infinite distance: no known switch.
  if (!(rampDuration > 0.0 && timeToNearestSwitch < rampDuration)) return 1.0;
  const scalar_t weight =
      std::clamp(1.0 - timeToNearestSwitch / rampDuration, 0.0, 1.0);
  return 1.0 + (maxScale - 1.0) * weight * weight *
                   (kSmoothstepCubicOffset - 2.0 * weight);
}

}  // namespace ocs2::legged::estimation
