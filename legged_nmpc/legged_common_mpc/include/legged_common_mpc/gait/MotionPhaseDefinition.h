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

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/gait/BipedModes.h"

namespace ocs2::legged {

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

// The spellings of the gait files' mode names (parseModeNumber()). FLY and
// STANCE name a mode on every contact set, LF and RF on a set whose stance is
// the biped's; leg names are joined by the leg separator and contact names by
// the contact separator.
// LINT.IfChange(mode_names)
inline constexpr char kModeNameFly[] = "FLY";
inline constexpr char kModeNameStance[] = "STANCE";
inline constexpr char kModeNameLeftFoot[] = "LF";
inline constexpr char kModeNameRightFoot[] = "RF";
inline constexpr char kModeLegSeparator = '_';
inline constexpr char kModeContactSeparator = '+';
// LINT.ThenChange(//legged_nmpc/legged_mpc_config/mode_sequence_template_config.proto:mode_names, //legged_nmpc/legged_common_mpc/src/config/model/ModelSettingsFromConfig.cpp:reserved_leg_names, //legged_nmpc/legged_mpc_config/model_settings_config.proto:legs)

/**
 * The contact set a gait file's modes are read against (parseModeNumber())
 * and printed for (modeNumber2String()): its number of contacts, its stance
 * mode, the mode GaitSchedule inserts between gaits and appends after the last
 * one, and the names of its legs and contacts. GaitSchedule.h's
 * gaitModeLayoutOf() derives it from ModelSettings. The default is the biped's
 * two feet without leg or contact names, which reads and prints FLY, RF, LF
 * and STANCE only.
 */
struct GaitModeLayout {
  /** A leg of the set (ModelSettings::Leg) and the mode number in which its
   * contacts alone are active. parseModeNumber() reads the contact names with
   * the same pair, one contact per entry. */
  struct LegModeBits {
    std::string name;
    // The OR of contactModeBit() over the leg's contacts (the contact's bit
    // for a contact); nonzero.
    size_t modeBits = 0;
  };

  size_t numContacts = static_cast<size_t>(kBipedFootCount);
  // The mode number "STANCE" names: kBipedModeStance (the two feet, which are
  // the last two contacts) on a set whose last two contacts are 6-DoF feet and
  // that is not legged other than as the biped (gaitModeLayoutOf()), all
  // contacts active otherwise.
  size_t stanceMode = kBipedModeStance;
  // The legs in ModelSettings::legs order; empty on a set without legs.
  std::vector<LegModeBits> legs;
  // ModelSettings::contactNames, in contact order; empty when the layout names
  // no contacts, so the '+' form is not read.
  std::vector<std::string> contactNames;
};

/**
 * Returns the mode a new gait aligns its start to when its template has no mode
 * other than the stance (updateGaitSchedule()): kBipedModeLf, the left foot
 * alone, on a set whose stance is the biped's (layout.stanceMode ==
 * kBipedModeStance), and none on any other set, whose stance template is not
 * moved. A function of the stance mode rather than a field of GaitModeLayout,
 * so that a layout built by hand cannot contradict its own stance.
 */
std::optional<size_t> defaultCycleStartMode(const GaitModeLayout& layout);

/**
 * Returns the name of `modeNumber` on the contact set `layout`, the inverse of
 * parseModeNumber():
 *  - "FLY" for no contact and "STANCE" for layout.stanceMode;
 *  - on a set whose stance is the biped's (layout.stanceMode ==
 *    kBipedModeStance), "LF" and "RF" for the left or the right foot alone;
 *  - the names of the legs joined by '_' in leg order when the mode is the
 *    union of whole legs of the layout ("LF_RH");
 *  - otherwise one digit 0 or 1 per contact, contact 0 first ("1001").
 * On the default layout that is FLY, RF, LF and STANCE for modes 0 to 3, as the
 * gait files name them. An empty string for a number that is not a mode of the
 * set (a bit past its contacts).
 */
std::string modeNumber2String(size_t modeNumber,
                              const GaitModeLayout& layout = GaitModeLayout{});

/**
 * Parses `modeString`, a mode name of a gait file, into the mode number on the
 * contact set `layout`,
 * tried in this order:
 *  - "FLY", no contact active, and "STANCE", layout.stanceMode, on any set;
 *  - "LF" and "RF", the left or the right foot alone, on a set whose stance
 *    mode is the biped's (the feet are its last two contacts);
 *  - distinct leg names of layout.legs joined by '_', the contacts of those
 *    legs: "LF_RH" is 0b1001 for legs LF, RF, LH and RH of one contact each;
 *  - distinct contact names of layout.contactNames joined by '+', those
 *    contacts: "sole_l_toe_contact+sole_r_heel_contact";
 *  - a string of layout.numContacts digits '0' and '1', the bitmask of the
 *    active contacts in contact order (contact 0 first, the most significant
 *    bit), on any set: "1001" is contacts 0 and 3 of a set of four.
 * A string of leg or contact names that repeats one is refused naming it.
 * InvalidArgument naming the valid forms, the contact count and the legs
 * otherwise.
 */
absl::StatusOr<size_t> parseModeNumber(absl::string_view modeString,
                                       const GaitModeLayout& layout);

}  // namespace ocs2::legged
