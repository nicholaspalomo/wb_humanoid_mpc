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

#include <cstddef>

/*
 * The contacts of the humanoid MPCs, its feet, without the rest of
 * humanoid_common_mpc/common/Types.h (which includes it): code that only counts
 * the feet, such as the validation's metrics, includes this header and none of
 * the Eigen, OCS2 and CppAD headers Types.h brings with it.
 */

namespace ocs2 {

/** The contacts of the MPC, one per foot, and the index of each in a per-foot
 * array (feet_array_t). */
inline constexpr size_t kNumContacts = 2;
inline constexpr size_t kContactLeftIndex = 0;
inline constexpr size_t kContactRightIndex = 1;

/**
 * The feet of a biped, one per leg: what the code written for a left and a
 * right foot (the contact planner's rows, rules and variable layout, one wrench
 * cone basis or knee mimic block per foot) static_asserts that kNumContacts is.
 */
inline constexpr int kBipedFootCount = 2;

}  // namespace ocs2
