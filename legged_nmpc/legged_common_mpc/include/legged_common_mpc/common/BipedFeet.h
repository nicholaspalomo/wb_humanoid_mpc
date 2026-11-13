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

#include <array>
#include <cstddef>

/*
 * The feet of a biped, without the Eigen, OCS2 and CppAD headers of
 * legged_common_mpc/common/Types.h.
 *
 * Code that is meaningful for any contact set holds one value per contact in
 * a ContactArray and the active contacts in a ContactMask
 * (common/ContactArray.h, common/ContactMask.h), sized from
 * ModelSettings::numContacts(). BipedFeetArray is for the code written for a
 * left and a right foot only, which refuses any other contact set
 * (ModelSettings::isBipedContactSet()).
 */

namespace ocs2 {

/** The index of each foot in a BipedFeetArray, and in the contact set of a
 * biped (two 6-DoF contacts, left then right). */
inline constexpr size_t kContactLeftIndex = 0;
inline constexpr size_t kContactRightIndex = 1;

/**
 * The feet of a biped, one per leg. The code written for a left and a right
 * foot (the contact planners, their biped-only rows and rules, the knee mimic
 * blocks) refuses a contact set of any other size when it is built
 * (common/TwoContactCount.h).
 */
inline constexpr int kBipedFootCount = 2;

/** One value per foot of a biped, indexed by kContactLeftIndex and
 * kContactRightIndex. */
template <typename T>
using BipedFeetArray = std::array<T, static_cast<size_t>(kBipedFootCount)>;

/** A per-foot array with both entries set to `value`. */
template <typename T>
BipedFeetArray<T> makeBipedFeetArray(const T& value) {
  BipedFeetArray<T> array;
  array.fill(value);
  return array;
}

}  // namespace ocs2
