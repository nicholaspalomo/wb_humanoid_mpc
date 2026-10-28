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

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The contact modes of the two feet. A mode number is the stance flags {LF, RF}
 * read as the bits of a two-bit number (stanceLeg2ModeNumber()), which is how
 * ModeSchedule stores it: as a size_t.
 */
// NOLINTNEXTLINE(totw-unscoped-enum): a mode number IS the size_t ModeSchedule stores; every use spells ModeNumber::k...
enum ModeNumber {
  kFly = 0,
  kRf = 1,
  kLf = 2,
  kStance = 3,
};

/** The mode numbers, FLY to STANCE: the modes a humanoid MPC's mode schedule
 * and policy may carry. */
inline constexpr size_t kNumModeNumbers =
    static_cast<size_t>(ModeNumber::kStance) + 1;

/**
 * The left foot's stance flag in a mode number, bit 1: as the mask
 * modeNumber2StanceLeg() reads it with, and as the place value
 * stanceLeg2ModeNumber() weighs it with. The right foot's is bit 0, the
 * value 1.
 */
inline constexpr unsigned int kLeftStanceModeBitMask = 2U;
inline constexpr int kLeftStanceModePlaceValue = 2;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
/** The stance flags {LF, RF} of `modeNumber`; no foot is in stance for a number
 * that is not a mode. Realtime-safe. */
inline contact_flag_t modeNumber2StanceLeg(size_t modeNumber) {
  if (modeNumber > static_cast<size_t>(ModeNumber::kStance)) {
    return contact_flag_t{false, false};
  }
  // The inverse of stanceLeg2ModeNumber(): bit 1 is the left foot, bit 0 the
  // right one.
  return contact_flag_t{(modeNumber & kLeftStanceModeBitMask) != 0,
                        (modeNumber & 1U) != 0};
}

/**
 * Whether contact `contactIndex` of a set of `numContacts` contacts
 * (ModelSettings::contactNames) is active in `modeNumber`, the bitmask of the
 * active contacts in contact-set order with the first contact the most
 * significant bit: contact i is bit numContacts - 1 - i. For the biped's two
 * feet that is modeNumber2StanceLeg()'s reading, FLY, RF, LF and STANCE
 * unchanged, and in a larger set whose last two contacts are the feet those
 * four numbers still mean the same feet with every other contact inactive. A
 * number with a bit past the contacts is no mode: no contact is active in it.
 * Realtime-safe.
 */
inline bool contactIsActiveInMode(size_t modeNumber,
                                  size_t contactIndex,
                                  size_t numContacts) {
  constexpr size_t kModeNumberBits = std::numeric_limits<size_t>::digits;
  if (contactIndex >= numContacts || numContacts >= kModeNumberBits ||
      (modeNumber >> numContacts) != 0) {
    return false;
  }
  return ((modeNumber >> (numContacts - 1 - contactIndex)) & 1U) != 0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/**
 * The mode number in which contact `contactIndex` alone of a set of
 * `numContacts` contacts is active: bit numContacts - 1 - contactIndex
 * (contactIsActiveInMode()). Requires contactIndex < numContacts < the bits of
 * a size_t.
 */
inline size_t contactModeBit(size_t contactIndex, size_t numContacts) {
  return size_t{1} << (numContacts - 1 - contactIndex);
}

/**
 * The number of mode numbers of a set of `numContacts` contacts, 2^numContacts:
 * the bound a mode of its schedules and policies is checked against.
 * kNumModeNumbers for the biped's two feet. Requires numContacts < the bits of
 * a size_t.
 */
inline size_t numModeNumbersOf(size_t numContacts) {
  return size_t{1} << numContacts;
}

inline size_t stanceLeg2ModeNumber(const contact_flag_t& stanceLegs) {
  return static_cast<size_t>(static_cast<size_t>(stanceLegs[1]) +
                             kLeftStanceModePlaceValue *
                                 static_cast<size_t>(stanceLegs[0]));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/** The name of `modeNumber` in the gait files (FLY, RF, LF or STANCE), or an
 * empty string for a number that is not a mode. */
inline std::string modeNumber2String(size_t modeNumber) {
  // Indexed by the mode number.
  // LINT.IfChange(mode_names)
  constexpr std::array<absl::string_view, kNumModeNumbers> kModeNames = {
      "FLY", "RF", "LF", "STANCE"};
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/mode_sequence_template_config.proto:mode_names)
  return modeNumber < kModeNames.size() ? std::string(kModeNames[modeNumber])
                                        : std::string();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/** The mode number a gait file names `modeString` (FLY, RF, LF or STANCE), or
 * InvalidArgument naming the valid names. */
inline absl::StatusOr<size_t> parseModeNumber(absl::string_view modeString) {
  for (const ModeNumber mode : {ModeNumber::kFly, ModeNumber::kRf,
                                ModeNumber::kLf, ModeNumber::kStance}) {
    if (modeString == modeNumber2String(mode)) {
      return static_cast<size_t>(mode);
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("'", modeString,
                   "' is not a mode; the modes are FLY, RF, LF and STANCE."));
}

}  // namespace ocs2::humanoid
