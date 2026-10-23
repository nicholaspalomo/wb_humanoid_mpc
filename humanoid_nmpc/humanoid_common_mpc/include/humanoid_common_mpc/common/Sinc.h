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

#include <cmath>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/** Below this |x| sinc() is its two-term series, whose next term is below 1e-18 there. */
inline constexpr scalar_t kSincSeriesThreshold = 1.0e-4;
/** 3!: the series sin(x) / x = 1 - x^2 / 3! + ... */
inline constexpr scalar_t kSincSeriesQuadraticDivisor = 6.0;

/**
 * Returns sin(x) / x, 1 at x = 0, and its two-term series below |x| = kSincSeriesThreshold, where the quotient loses
 * its precision: the factor of the chord of a constant planar twist (TargetTrajectoriesCalculatorBase, the velocity
 * command's arc; StrideMotionHistory, the displacement of a constant body twist). Allocates nothing.
 */
inline scalar_t sinc(scalar_t x) {
  return std::abs(x) < kSincSeriesThreshold ? 1.0 - x * x / kSincSeriesQuadraticDivisor : std::sin(x) / x;
}

}  // namespace ocs2::humanoid
