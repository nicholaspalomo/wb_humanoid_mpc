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
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/gait/BipedModes.h"

namespace ocs2::humanoid {

/*
 * Mode numbers. A mode number is the bitmask of the active contacts of a
 * contact set (ModelSettings::contactNames, 3-DoF contacts first): contact i
 * is bit numContacts - 1 - i, so the first contact is the most significant
 * bit. ModeSchedule and the policies store it as a size_t, and
 * ContactMask::fromModeNumber() (common/ContactMask.h) reads it as the flags
 * of every contact. The biped's four modes and their gait-file names FLY, RF,
 * LF and STANCE are gait/BipedModes.h's kBipedMode* constants; they are the
 * same numbers when the two feet are the last contacts of a larger set.
 */

/**
 * Whether contact `contactIndex` of a set of `numContacts` contacts
 * (ModelSettings::contactNames) is active in `modeNumber`, the bitmask of the
 * active contacts in contact-set order with the first contact the most
 * significant bit: contact i is bit numContacts - 1 - i. For the biped's two
 * feet that is bipedFootContactsOfMode()'s reading, FLY, RF, LF and STANCE
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
 * kNumBipedModes for the biped's two feet. Requires numContacts < the bits of
 * a size_t.
 */
inline size_t numModeNumbersOf(size_t numContacts) {
  return size_t{1} << numContacts;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/** The name of `modeNumber` in the gait files (FLY, RF, LF or STANCE), or an
 * empty string for a number that is not a mode. */
inline std::string modeNumber2String(size_t modeNumber) {
  // Indexed by the mode number.
  // LINT.IfChange(mode_names)
  constexpr std::array<absl::string_view, kNumBipedModes> kModeNames = {
      "FLY", "RF", "LF", "STANCE"};
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/mode_sequence_template_config.proto:mode_names)
  return modeNumber < kModeNames.size() ? std::string(kModeNames[modeNumber])
                                        : std::string();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/**
 * The contact set a gait file's modes are read against (parseModeNumber()): its
 * number of contacts and its stance mode, the mode GaitSchedule inserts between
 * gaits and appends after the last one. GaitSchedule.h's gaitModeLayoutOf()
 * derives it from ModelSettings. The default is the biped's two feet.
 */
struct GaitModeLayout {
  size_t numContacts = static_cast<size_t>(kBipedFootCount);
  // The mode number "STANCE" names: kBipedModeStance (the two feet, which are
  // the last two contacts) on a set whose last two contacts are 6-DoF feet, all
  // contacts active otherwise.
  size_t stanceMode = kBipedModeStance;
};

/**
 * The mode number a gait file names `modeString` on the contact set `layout`:
 *  - "FLY", no contact active, and "STANCE", layout.stanceMode, on any set;
 *  - "LF" and "RF", the left or the right foot alone, on a set whose stance
 *    mode is the biped's (the feet are its last two contacts);
 *  - a string of layout.numContacts digits '0' and '1', the bitmask of the
 *    active contacts in contact order (contact 0 first, the most significant
 *    bit), on any set: "1001" is contacts 0 and 3 of a set of four.
 * InvalidArgument naming the valid forms and the contact count otherwise.
 */
inline absl::StatusOr<size_t> parseModeNumber(absl::string_view modeString,
                                              GaitModeLayout layout) {
  if (modeString == modeNumber2String(kBipedModeFly)) return kBipedModeFly;
  if (modeString == modeNumber2String(kBipedModeStance)) {
    return layout.stanceMode;
  }
  const bool bipedFeetLast = layout.stanceMode == kBipedModeStance;
  if (bipedFeetLast) {
    if (modeString == modeNumber2String(kBipedModeLf)) return kBipedModeLf;
    if (modeString == modeNumber2String(kBipedModeRf)) return kBipedModeRf;
  }
  constexpr size_t kModeNumberBits = std::numeric_limits<size_t>::digits;
  const bool isMask =
      modeString.size() == layout.numContacts &&
      layout.numContacts < kModeNumberBits &&
      modeString.find_first_not_of("01") == absl::string_view::npos;
  if (isMask) {
    size_t mode = 0;
    for (const char digit : modeString) {
      mode = (mode << 1U) | static_cast<size_t>(digit == '1');
    }
    return mode;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "'", modeString, "' is not a mode of a set of ", layout.numContacts,
      " contacts; the modes are FLY, STANCE", bipedFeetLast ? ", LF, RF" : "",
      " and a string of ", layout.numContacts,
      " digits 0 and 1, one per contact in contact order (1 in contact)."));
}

}  // namespace ocs2::humanoid
