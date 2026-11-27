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

#include "legged_common_mpc/contact_planning/hlip/HlipStandingBlend.h"

#include <cmath>

#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::legged {
namespace {

/** (value / threshold)^2: one component of phi under the threshold metric P. */
scalar_t squaredRatio(scalar_t value, scalar_t threshold) {
  const scalar_t ratio = value / threshold;
  return ratio * ratio;
}

}  // namespace

scalar_t HlipStandingBlend::activity(
    const vector2_t& cmdLinvelBaseRtWorldEwrtHeadingXy,
    scalar_t cmdRateYawBaseRtWorld,
    const vector2_t& linvelComRtWorldEwrtHeadingXy) const {
  const HlipBlendParameters& p = parameters_;
  return squaredRatio(cmdLinvelBaseRtWorldEwrtHeadingXy(0),
                      p.cmdMaxLinvelBaseRtWorldEwrtHeadingX) +
         squaredRatio(cmdLinvelBaseRtWorldEwrtHeadingXy(1),
                      p.cmdMaxLinvelBaseRtWorldEwrtHeadingY) +
         squaredRatio(cmdRateYawBaseRtWorld,
                      p.cmdMaxAngvelBaseRtWorldEwrtWorldZ) +
         squaredRatio(linvelComRtWorldEwrtHeadingXy(0),
                      p.maxProxyLinvelComRtWorldEwrtHeadingX) +
         squaredRatio(linvelComRtWorldEwrtHeadingXy(1),
                      p.maxProxyLinvelComRtWorldEwrtHeadingY);
}

scalar_t HlipStandingBlend::weight(
    const vector2_t& cmdLinvelBaseRtWorldEwrtHeadingXy,
    scalar_t cmdRateYawBaseRtWorld,
    const vector2_t& linvelComRtWorldEwrtHeadingXy) const {
  const scalar_t phi =
      activity(cmdLinvelBaseRtWorldEwrtHeadingXy, cmdRateYawBaseRtWorld,
               linvelComRtWorldEwrtHeadingXy);
  return 0.5 *
             std::tanh(parameters_.sharpness * (phi - parameters_.threshold)) +
         kHalfPointWeight;
}

}  // namespace ocs2::legged
