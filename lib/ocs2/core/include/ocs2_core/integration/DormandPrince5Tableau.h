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

#include <ocs2_core/Types.h>

namespace ocs2 {

/**
 * The Butcher tableau of the Dormand-Prince 5(4) method and the step-size control of its adaptive steps, shared by
 * RungeKuttaDormandPrince5 and the odeint port's steppers::DormandPrince5 (steppers.h). kA<i> is the node of stage i,
 * the fraction of dt at which it evaluates the system; kB<i><j> the weight of stage j in the state of stage i; kC<i>
 * the weight of stage i in the fifth-order solution (c2 = 0); kDc<i> the fifth-order minus the fourth-order weight of
 * stage i, which estimates the error of the step (dc2 = 0). Stage 7 is the derivative at the new state.
 */
namespace dormand_prince5 {

inline constexpr scalar_t kA2 = 1.0 / 5;
inline constexpr scalar_t kA3 = 3.0 / 10;
inline constexpr scalar_t kA4 = 4.0 / 5;
inline constexpr scalar_t kA5 = 8.0 / 9;

inline constexpr scalar_t kB21 = 1.0 / 5;

inline constexpr scalar_t kB31 = 3.0 / 40;
inline constexpr scalar_t kB32 = 9.0 / 40;

inline constexpr scalar_t kB41 = 44.0 / 45;
inline constexpr scalar_t kB42 = -56.0 / 15;
inline constexpr scalar_t kB43 = 32.0 / 9;

inline constexpr scalar_t kB51 = 19372.0 / 6561;
inline constexpr scalar_t kB52 = -25360.0 / 2187;
inline constexpr scalar_t kB53 = 64448.0 / 6561;
inline constexpr scalar_t kB54 = -212.0 / 729;

inline constexpr scalar_t kB61 = 9017.0 / 3168;
inline constexpr scalar_t kB62 = -355.0 / 33;
inline constexpr scalar_t kB63 = 46732.0 / 5247;
inline constexpr scalar_t kB64 = 49.0 / 176;
inline constexpr scalar_t kB65 = -5103.0 / 18656;

inline constexpr scalar_t kC1 = 35.0 / 384;
inline constexpr scalar_t kC3 = 500.0 / 1113;
inline constexpr scalar_t kC4 = 125.0 / 192;
inline constexpr scalar_t kC5 = -2187.0 / 6784;
inline constexpr scalar_t kC6 = 11.0 / 84;

inline constexpr scalar_t kDc1 = kC1 - 5179.0 / 57600;
inline constexpr scalar_t kDc3 = kC3 - 7571.0 / 16695;
inline constexpr scalar_t kDc4 = kC4 - 393.0 / 640;
inline constexpr scalar_t kDc5 = kC5 - -92097.0 / 339200;
inline constexpr scalar_t kDc6 = kC6 - 187.0 / 2100;
inline constexpr scalar_t kDc7 = -1.0 / 40;

// The step-size control of odeint's default_step_adjuster.
/** The safety factor of every change of the step size. */
inline constexpr scalar_t kStepSizeSafetyFactor = 0.9;
/** A decreased step is no less than this fraction of the step it replaces. */
inline constexpr scalar_t kMinStepDecreaseFactor = 0.2;
/** A step grows only when its maximal relative error is below this. */
inline constexpr scalar_t kStepIncreaseErrorThreshold = 0.5;
/** The growth of a step caps error^(-1 / stepperOrder) at this. */
inline constexpr scalar_t kStepIncreaseLimit = 5.0;

}  // namespace dormand_prince5
}  // namespace ocs2
