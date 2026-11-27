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

/*
 * The physical constants and the unit conversions shared by the OCS2 fork and the first-party code, so that the models,
 * the costs, the simulator and the logs agree on them (AGENTS.md, C++ "Magic numbers"). Like ocs2_core/Dimensions.h,
 * the header has no dependencies (//lib/ocs2:ocs2_physical_constants), so code that uses neither Eigen nor OCS2, such as
 * the robot runtime, may include it too.
 *
 * Every constant is a `double` (ocs2::scalar_t), the type of the floating-point literal it replaces, so that the
 * arithmetic is what it was with the literal.
 */

namespace ocs2 {

/**
 * [m/s^2] The magnitude of the gravitational acceleration every model, cost, heuristic and simulation of the repository
 * assumes: 9.81, not the standard 9.80665. Gravity points along -z (ocs2_core/Dimensions.h, kZAxisIndex).
 */
inline constexpr double kGravityMagnitude = 9.81;

/** The submultiples of a second in one second: the factors of a time in seconds to one in ms, us and ns. */
inline constexpr double kMillisecondsPerSecond = 1.0e3;
inline constexpr double kMicrosecondsPerSecond = 1.0e6;
inline constexpr double kNanosecondsPerSecond = 1.0e9;

}  // namespace ocs2
