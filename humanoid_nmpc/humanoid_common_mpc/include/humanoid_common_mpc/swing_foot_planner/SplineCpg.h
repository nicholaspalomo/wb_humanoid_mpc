/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/swing_foot_planner/CubicSpline.h"

namespace ocs2::humanoid {

/**
 * The height profile of one swing: two cubic splines from lift-off to the mid-swing apex and from there to touch-down,
 * with zero vertical velocity at the apex. An immutable value.
 */
class SplineCpg {
 public:
  SplineCpg(CubicSpline::Node liftOff, scalar_t midHeight, CubicSpline::Node touchDown);

  scalar_t position(scalar_t time) const;

  scalar_t velocity(scalar_t time) const;

  scalar_t acceleration(scalar_t time) const;

  scalar_t startTimeDerivative(scalar_t time) const;

  scalar_t finalTimeDerivative(scalar_t time) const;

 private:
  scalar_t midTime_;  // NOLINT(modernize-use-default-member-init): set from the constructor's parameters
  CubicSpline leftSpline_;
  CubicSpline rightSpline_;
};

}  // namespace ocs2::humanoid
