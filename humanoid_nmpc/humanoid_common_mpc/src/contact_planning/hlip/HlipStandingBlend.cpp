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

#include "humanoid_common_mpc/contact_planning/hlip/HlipStandingBlend.h"

#include <cmath>
#include <functional>

namespace ocs2::humanoid {

scalar_t HlipStandingBlend::activity(const vector2_t& velocityCommand, scalar_t yawRateCommand, const vector2_t& baseVelocity) const {
  const HlipBlendParameters& p = parameters_;
  const std::function<scalar_t(scalar_t, scalar_t)> squaredRatio = [](scalar_t value, scalar_t threshold) {
    const scalar_t ratio = value / threshold;
    return ratio * ratio;
  };
  return squaredRatio(velocityCommand(0), p.maxCommandedVelocityX) + squaredRatio(velocityCommand(1), p.maxCommandedVelocityY) +
         squaredRatio(yawRateCommand, p.maxCommandedYawRate) + squaredRatio(baseVelocity(0), p.maxBaseVelocityX) +
         squaredRatio(baseVelocity(1), p.maxBaseVelocityY);
}

scalar_t HlipStandingBlend::weight(const vector2_t& velocityCommand, scalar_t yawRateCommand, const vector2_t& baseVelocity) const {
  const scalar_t phi = activity(velocityCommand, yawRateCommand, baseVelocity);
  return 0.5 * std::tanh(parameters_.sharpness * (phi - parameters_.threshold)) + 0.5;
}

}  // namespace ocs2::humanoid
