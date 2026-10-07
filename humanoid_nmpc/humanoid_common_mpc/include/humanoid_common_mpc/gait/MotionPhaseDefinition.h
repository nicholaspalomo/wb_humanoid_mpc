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
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The contact modes of the two feet. A mode number is the stance flags {LF, RF} read as the bits of a two-bit number
 * (stanceLeg2ModeNumber()), which is how ModeSchedule stores it: as a size_t.
 */
// NOLINTNEXTLINE(totw-unscoped-enum): a mode number IS the size_t ModeSchedule stores; every use spells ModeNumber::k...
enum ModeNumber {
  kFly = 0,
  kRf = 1,
  kLf = 2,
  kStance = 3,
};

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
/** The stance flags {LF, RF} of `modeNumber`; no foot is in stance for a number that is not a mode. Realtime-safe. */
inline contact_flag_t modeNumber2StanceLeg(size_t modeNumber) {
  if (modeNumber > static_cast<size_t>(ModeNumber::kStance)) {
    return contact_flag_t{false, false};
  }
  // The inverse of stanceLeg2ModeNumber(): bit 1 is the left foot, bit 0 the right one.
  return contact_flag_t{(modeNumber & 2U) != 0, (modeNumber & 1U) != 0};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

inline size_t stanceLeg2ModeNumber(const contact_flag_t& stanceLegs) {
  return static_cast<size_t>(static_cast<size_t>(stanceLegs[1]) + 2 * static_cast<size_t>(stanceLegs[0]));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/** The name of `modeNumber` in the gait files (FLY, RF, LF or STANCE), or an empty string for a number that is not a mode. */
inline std::string modeNumber2String(size_t modeNumber) {
  // Indexed by the mode number.
  // LINT.IfChange(mode_names)
  constexpr std::array<absl::string_view, static_cast<size_t>(ModeNumber::kStance) + 1> kModeNames = {"FLY", "RF", "LF", "STANCE"};
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/mode_sequence_template_config.proto:mode_names)
  return modeNumber < kModeNames.size() ? std::string(kModeNames[modeNumber]) : std::string();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/** The mode number a gait file names `modeString` (FLY, RF, LF or STANCE), or InvalidArgument naming the valid names. */
inline absl::StatusOr<size_t> parseModeNumber(absl::string_view modeString) {
  for (const ModeNumber mode : {ModeNumber::kFly, ModeNumber::kRf, ModeNumber::kLf, ModeNumber::kStance}) {
    if (modeString == modeNumber2String(mode)) {
      return static_cast<size_t>(mode);
    }
  }
  return absl::InvalidArgumentError(absl::StrCat("'", modeString, "' is not a mode; the modes are FLY, RF, LF and STANCE."));
}

}  // namespace ocs2::humanoid
