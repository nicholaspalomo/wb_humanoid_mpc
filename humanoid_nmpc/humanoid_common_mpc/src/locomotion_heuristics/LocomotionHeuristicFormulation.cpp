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

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"

#include <algorithm>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ocs2::humanoid {

namespace {

/** The rejection message's list of what WOULD have been valid. */
std::string joinNames(const std::vector<std::string>& names) {
  return names.empty() ? "(none)" : absl::StrJoin(names, ", ");
}

}  // namespace

const std::array<HeuristicKind, kNumHeuristicKinds>& allHeuristicKinds() {
  static const std::array<HeuristicKind, kNumHeuristicKinds> kinds{HeuristicKind::BASE_POSE, HeuristicKind::FOOTHOLD,
                                                                   HeuristicKind::WRENCH};
  return kinds;
}

absl::string_view heuristicKindName(HeuristicKind kind) {
  switch (kind) {
    case HeuristicKind::BASE_POSE:
      return "base_pose";
    case HeuristicKind::FOOTHOLD:
      return "foothold";
    case HeuristicKind::WRENCH:
      return "wrench";
  }
  return "base_pose";
}

std::string normalizeHeuristicName(absl::string_view name) {
  std::string output;
  output.reserve(name.size());
  for (char character : name) {
    if (character != '_' && character != '-' && character != ' ') {
      output.push_back(absl::ascii_tolower(character));
    }
  }
  return output;
}

const std::vector<std::string>& knownHeuristicNames(HeuristicKind kind) {
  // The single source of truth for which heuristics exist. In the order of Bledt's Appendix C: the analytic ones of
  // Table C.1 before the data-extracted ones of Table C.2 within each kind, because that is the order the
  // dissertation introduces them and the order the README derives them.
  // LINT.IfChange(known_heuristic_names)
  static const std::vector<std::string> basePose{heuristic::kOrientationCompensation, heuristic::kPeriodicOrientation,
                                                 heuristic::kHeightCompensation};
  static const std::vector<std::string> foothold{heuristic::kHipCenteredStepping, heuristic::kCapturePoint,
                                                 heuristic::kTranslationalStepping, heuristic::kInPlaceTurning,
                                                 heuristic::kHighSpeedTurning};
  static const std::vector<std::string> wrench{heuristic::kImpulseScaling, heuristic::kCentripetalAcceleration};
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicFactory.cpp:heuristic_factory, //humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicConfig.cpp:locomotion_heuristic_keys)
  // clang-format on
  switch (kind) {
    case HeuristicKind::BASE_POSE:
      return basePose;
    case HeuristicKind::FOOTHOLD:
      return foothold;
    case HeuristicKind::WRENCH:
      return wrench;
  }
  return basePose;
}

std::string canonicalHeuristicName(HeuristicKind kind, absl::string_view name) {
  const std::string normalized = normalizeHeuristicName(name);
  for (const std::string& known : knownHeuristicNames(kind)) {
    if (normalizeHeuristicName(known) == normalized) return known;
  }
  return std::string();
}

std::optional<HeuristicKind> heuristicKindOf(absl::string_view name) {
  for (HeuristicKind kind : allHeuristicKinds()) {
    if (!canonicalHeuristicName(kind, name).empty()) return kind;
  }
  return std::nullopt;
}

std::vector<std::string>& LocomotionHeuristicFormulation::list(HeuristicKind kind) {
  switch (kind) {
    case HeuristicKind::BASE_POSE:
      return basePose;
    case HeuristicKind::FOOTHOLD:
      return foothold;
    case HeuristicKind::WRENCH:
      return wrench;
  }
  return basePose;
}

const std::vector<std::string>& LocomotionHeuristicFormulation::list(HeuristicKind kind) const {
  return const_cast<LocomotionHeuristicFormulation*>(this)->list(kind);
}

bool LocomotionHeuristicFormulation::listed(HeuristicKind kind, absl::string_view name) const {
  const std::string canonical = canonicalHeuristicName(kind, name);
  if (canonical.empty()) return false;
  const std::vector<std::string>& names = list(kind);
  return std::any_of(names.begin(), names.end(),
                     [&canonical, kind](const std::string& entry) { return canonicalHeuristicName(kind, entry) == canonical; });
}

absl::Status LocomotionHeuristicFormulation::validate() const {
  for (HeuristicKind kind : allHeuristicKinds()) {
    absl::flat_hash_set<std::string> seen;
    for (const std::string& entry : list(kind)) {
      const std::string canonical = canonicalHeuristicName(kind, entry);
      if (canonical.empty()) {
        // A name that IS a heuristic but sits in the wrong list gets told where it belongs. The three kinds shape
        // three different reference channels and are evaluated at three different seams, so this is a genuine
        // mistake rather than a spelling one, and "unknown foothold heuristic 'periodic_orientation'" would send the
        // operator looking for a typo that is not there.
        const std::optional<HeuristicKind> actualKind = heuristicKindOf(entry);
        if (actualKind.has_value()) {
          return absl::InvalidArgumentError(
              absl::StrCat("[LocomotionHeuristicFormulation] '", entry, "' is a ", heuristicKindName(*actualKind),
                           " heuristic and is listed under locomotion_heuristics.", heuristicKindName(kind),
                           ". Move it to the locomotion_heuristics.", heuristicKindName(*actualKind), " list. Valid ",
                           heuristicKindName(kind), " heuristics are: ", joinNames(knownHeuristicNames(kind)), "."));
        }
        return absl::InvalidArgumentError(absl::StrCat("[LocomotionHeuristicFormulation] unknown ", heuristicKindName(kind), " heuristic '",
                                                       entry, "' in locomotion_heuristics.", heuristicKindName(kind),
                                                       "; supported: ", joinNames(knownHeuristicNames(kind)), "."));
      }
      if (!seen.insert(canonical).second) {
        // Listing a heuristic twice would double its offset, which looks exactly like a coefficient that is twice
        // what the file says and is correspondingly hard to find by tuning.
        return absl::InvalidArgumentError(absl::StrCat("[LocomotionHeuristicFormulation] '", canonical,
                                                       "' is listed more than once in locomotion_heuristics.", heuristicKindName(kind),
                                                       ". The offsets of a list are summed, so a repeated name would "
                                                       "silently double its contribution."));
      }
    }
  }
  return absl::OkStatus();
}

std::vector<std::string> LocomotionHeuristicFormulation::warnings() const {
  std::vector<std::string> out;
  // hip_centered_stepping is the base term of Bledt's foot placement, the one the velocity-dependent heuristics are
  // summed on top of (section 4.3, figures 4-8 to 4-11). Neither half of that sum on its own is a configuration the
  // dissertation validates, and each fails in its own way, so both halves are reported.
  const bool hasAnchor = listed(HeuristicKind::FOOTHOLD, heuristic::kHipCenteredStepping);
  if (hasAnchor && foothold.size() == 1) {
    out.push_back(
        "locomotion_heuristics.foothold lists only 'hip_centered_stepping'. That places each foot under its own hip "
        "and nowhere else, which is the base case Bledt measures the others against (figure 4-8): it stands and takes "
        "a few slow steps, and falls as soon as it has any speed, because a foot placed under the hip at lift-off is "
        "behind the robot by touch-down. It also re-anchors the landing target on the measured base, which "
        "SwitchedModelReferenceManager documents as giving less foot separation than intended in single support. Add "
        "'translational_stepping' before drawing conclusions from it.");
  }
  if (!hasAnchor && !foothold.empty()) {
    out.push_back(
        "locomotion_heuristics.foothold does not list 'hip_centered_stepping'. The other foothold heuristics are "
        "corrections that Bledt sums onto it, and they are being applied here to this controller's own landing target "
        "instead - a step width to the side of the stance foot. That is a reasonable thing to do and it is why the "
        "anchor is not mandatory, but it is NOT the formulation of the dissertation, and the coefficients fitted "
        "there do not transfer to it.");
  }
  return out;
}

std::string LocomotionHeuristicFormulation::summary() const {
  std::string out;
  for (HeuristicKind kind : allHeuristicKinds()) {
    absl::StrAppend(&out, "  ", heuristicKindName(kind), ": ", joinNames(list(kind)), "\n");
  }
  return out;
}

}  // namespace ocs2::humanoid
