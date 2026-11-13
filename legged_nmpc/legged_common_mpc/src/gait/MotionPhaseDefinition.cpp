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

#include "legged_common_mpc/gait/MotionPhaseDefinition.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "legged_common_mpc/gait/BipedModes.h"

namespace ocs2::legged {
namespace {

constexpr size_t kModeNumberBits = std::numeric_limits<size_t>::digits;
// The separators as strings, for joining and for the messages.
constexpr char kLegSeparatorString[] = {kModeLegSeparator, '\0'};
constexpr char kContactSeparatorString[] = {kModeContactSeparator, '\0'};

/** Returns whether `modeNumber` is a mode of `layout`: no bit past its
 * contacts. */
bool isModeOf(size_t modeNumber, const GaitModeLayout& layout) {
  return layout.numContacts < kModeNumberBits &&
         (modeNumber >> layout.numContacts) == 0;
}

/** Returns the contacts of `layout` with their mode bits, none when the layout
 * names no contacts or a number of them other than its contact count. */
std::vector<GaitModeLayout::LegModeBits> contactModeBits(
    const GaitModeLayout& layout) {
  std::vector<GaitModeLayout::LegModeBits> contacts;
  if (layout.contactNames.size() != layout.numContacts ||
      layout.numContacts >= kModeNumberBits) {
    return contacts;
  }
  contacts.reserve(layout.numContacts);
  for (size_t i = 0; i < layout.numContacts; ++i) {
    contacts.push_back(GaitModeLayout::LegModeBits{
        .name = layout.contactNames[i],
        .modeBits = contactModeBit(i, layout.numContacts)});
  }
  return contacts;
}

/** Joins the names of `named` with ", ". */
std::string joinedNames(absl::Span<const GaitModeLayout::LegModeBits> named) {
  std::vector<absl::string_view> names;
  names.reserve(named.size());
  for (const GaitModeLayout::LegModeBits& entry : named) {
    names.push_back(entry.name);
  }
  return absl::StrJoin(names, ", ");
}

/**
 * Returns the union of the modes of the names `modeString` lists, separated by
 * `separator`, when every part is a name of `named`; nullopt when one is not,
 * and InvalidArgument naming the `kind` ("leg", "contact") when one is listed
 * twice.
 */
absl::StatusOr<std::optional<size_t>> unionOfNames(
    absl::string_view modeString,
    char separator,
    absl::Span<const GaitModeLayout::LegModeBits> named,
    absl::string_view kind) {
  if (named.empty()) return std::nullopt;
  const std::vector<absl::string_view> parts =
      absl::StrSplit(modeString, separator);
  std::vector<bool> listed(named.size(), false);
  size_t modeNumber = 0;
  std::optional<absl::string_view> repeated;
  for (const absl::string_view part : parts) {
    size_t index = named.size();
    for (size_t i = 0; i < named.size(); ++i) {
      if (named[i].name == part) {
        index = i;
        break;
      }
    }
    if (index == named.size()) return std::nullopt;
    if (listed[index] && !repeated.has_value()) repeated = part;
    listed[index] = true;
    modeNumber |= named[index].modeBits;
  }
  if (repeated.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "'", modeString, "' names the ", kind, " '", *repeated,
        "' more than once; the ", kind, "s are ", joinedNames(named)));
  }
  return modeNumber;
}

/** Returns the digit mask of `modeNumber`: one digit 0 or 1 per contact of a
 * set of `numContacts`, contact 0 first. */
std::string digitMask(size_t modeNumber, size_t numContacts) {
  std::string mask(numContacts, '0');
  for (size_t i = 0; i < numContacts; ++i) {
    if (contactIsActiveInMode(modeNumber, i, numContacts)) mask[i] = '1';
  }
  return mask;
}

/** Returns the legs of `layout` whose union is `modeNumber`, joined by
 * kModeLegSeparator in leg order, or an empty string when the mode is no union
 * of whole legs. */
std::string legNamesOfMode(size_t modeNumber, const GaitModeLayout& layout) {
  std::vector<absl::string_view> names;
  size_t covered = 0;
  for (const GaitModeLayout::LegModeBits& leg : layout.legs) {
    if (leg.modeBits != 0 && (leg.modeBits & modeNumber) == leg.modeBits) {
      names.push_back(leg.name);
      covered |= leg.modeBits;
    }
  }
  if (names.empty() || covered != modeNumber) return std::string();
  return absl::StrJoin(names, kLegSeparatorString);
}

}  // namespace

std::optional<size_t> defaultCycleStartMode(const GaitModeLayout& layout) {
  if (layout.stanceMode == kBipedModeStance) return kBipedModeLf;
  return std::nullopt;
}

std::string modeNumber2String(size_t modeNumber, const GaitModeLayout& layout) {
  if (!isModeOf(modeNumber, layout)) return std::string();
  if (modeNumber == kBipedModeFly) return kModeNameFly;
  if (modeNumber == layout.stanceMode) return kModeNameStance;
  const bool bipedStance = layout.stanceMode == kBipedModeStance;
  if (bipedStance && modeNumber == kBipedModeLf) return kModeNameLeftFoot;
  if (bipedStance && modeNumber == kBipedModeRf) return kModeNameRightFoot;
  std::string legs = legNamesOfMode(modeNumber, layout);
  // On a biped-stance set the foot aliases come first in parseModeNumber(), so
  // a leg named LF or RF is printed as its digit mask to read back the same.
  const bool shadowedByAlias =
      bipedStance && (legs == kModeNameLeftFoot || legs == kModeNameRightFoot);
  if (!legs.empty() && !shadowedByAlias) return legs;
  return digitMask(modeNumber, layout.numContacts);
}

absl::StatusOr<size_t> parseModeNumber(absl::string_view modeString,
                                       const GaitModeLayout& layout) {
  if (modeString == kModeNameFly) return kBipedModeFly;
  if (modeString == kModeNameStance) return layout.stanceMode;
  const bool bipedStance = layout.stanceMode == kBipedModeStance;
  if (bipedStance) {
    if (modeString == kModeNameLeftFoot) return kBipedModeLf;
    if (modeString == kModeNameRightFoot) return kBipedModeRf;
  }
  const absl::StatusOr<std::optional<size_t>> ofLegs =
      unionOfNames(modeString, kModeLegSeparator, layout.legs,
                   /*kind=*/"leg");
  if (!ofLegs.ok()) return ofLegs.status();
  if (ofLegs->has_value()) return **ofLegs;
  const std::vector<GaitModeLayout::LegModeBits> contacts =
      contactModeBits(layout);
  const absl::StatusOr<std::optional<size_t>> ofContacts = unionOfNames(
      modeString, kModeContactSeparator, contacts, /*kind=*/"contact");
  if (!ofContacts.ok()) return ofContacts.status();
  if (ofContacts->has_value()) return **ofContacts;
  const bool isMask =
      modeString.size() == layout.numContacts &&
      layout.numContacts < kModeNumberBits &&
      modeString.find_first_not_of("01") == absl::string_view::npos;
  if (isMask) {
    size_t modeNumber = 0;
    for (const char digit : modeString) {
      modeNumber = (modeNumber << 1U) | static_cast<size_t>(digit == '1');
    }
    return modeNumber;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "'", modeString, "' is not a mode of a set of ", layout.numContacts,
      " contacts; the modes are ", kModeNameFly, ", ", kModeNameStance,
      bipedStance
          ? absl::StrCat(", ", kModeNameLeftFoot, ", ", kModeNameRightFoot)
          : std::string(),
      layout.legs.empty()
          ? std::string()
          : absl::StrCat(", distinct names of the legs ",
                         joinedNames(layout.legs), " joined by '",
                         kLegSeparatorString, "'"),
      contacts.empty() ? std::string()
                       : absl::StrCat(", distinct names of the contacts ",
                                      joinedNames(contacts), " joined by '",
                                      kContactSeparatorString, "'"),
      " and a string of ", layout.numContacts,
      " digits 0 and 1, one per contact in contact order (1 in contact)."));
}

}  // namespace ocs2::legged
