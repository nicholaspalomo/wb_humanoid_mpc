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

#include "humanoid_common_mpc/common/BipedFeet.h"

namespace ocs2::humanoid {

/*
 * The mode numbers of a biped's two feet, left then right, for the code
 * written for a left and a right foot (gait files with FLY, RF, LF and STANCE,
 * the biped contact planners, the WB MPC). A mode number is the bitmask of the
 * active contacts with the first contact the most significant bit
 * (ContactMask, contactIsActiveInMode()), so these are its values for the set
 * {left foot, right foot}; generic code reads a mode through ContactMask.
 */
/** Neither foot in contact. */
inline constexpr size_t kBipedModeFly = 0;
/** The right foot alone in contact. */
inline constexpr size_t kBipedModeRf = 1;
/** The left foot alone in contact. */
inline constexpr size_t kBipedModeLf = 2;
/** Both feet in contact. */
inline constexpr size_t kBipedModeStance = 3;
/** The biped's mode numbers, FLY to STANCE. */
inline constexpr size_t kNumBipedModes = kBipedModeStance + 1;

/** Returns the stance flags {left, right} of `modeNumber`; neither foot is in
 * stance for a number that is not a biped mode. Realtime-safe. */
inline BipedFeetArray<bool> bipedFootContactsOfMode(size_t modeNumber) {
  if (modeNumber >= kNumBipedModes) return BipedFeetArray<bool>{false, false};
  return BipedFeetArray<bool>{(modeNumber & kBipedModeLf) != 0,
                              (modeNumber & kBipedModeRf) != 0};
}

/** Returns the mode number of the stance flags {left, right}, the inverse of
 * bipedFootContactsOfMode() on the biped modes. */
inline size_t bipedModeOfFootContacts(
    const BipedFeetArray<bool>& footContacts) {
  return (footContacts[kContactLeftIndex] ? kBipedModeLf : kBipedModeFly) +
         (footContacts[kContactRightIndex] ? kBipedModeRf : kBipedModeFly);
}

}  // namespace ocs2::humanoid
