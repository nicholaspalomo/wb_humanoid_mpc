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
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * Canonical names of the locomotion heuristics the reference-shaping layer can be assembled from: the ten of Gerardo
 * Bledt's "Regularized Predictive Control Framework for Robust Dynamic Legged Locomotion", MIT 2020, Appendix C.
 *
 * These are the entries of the three lists of the `locomotion_heuristics` block of the task file, and each is also the
 * key of that heuristic's own parameter block. Lookups normalise a name the way the rest of this repository does
 * (case-insensitive, `_`, `-` and spaces ignored), so `capturePoint` and `capture_point` are the same heuristic.
 */
namespace heuristic {

// ---- Base-pose heuristics: the roll, pitch and height of the base-pose reference (H_Theta and H_z) ----

/** Table C.2. H_Theta(pdot) = a1 pdot + a0: roll from lateral speed, pitch from forward speed. */
inline constexpr const char* kOrientationCompensation = "orientation_compensation";
/** Table C.2. H_Theta(Phi) = b1 sin(c1 Phi + d1): the orientation limit cycle of the gait. */
inline constexpr const char* kPeriodicOrientation = "periodic_orientation";
/** Table C.2. H_z(v) = a2 v^2 + a1 v + a0: base height against forward speed. */
inline constexpr const char* kHeightCompensation = "height_compensation";

// ---- Foothold heuristics: where the swing foot lands (H_r) ----

/** Table C.1. H_r(Theta) = PTP(R(Theta) r_hip): the foot under its own hip, ground-projected. */
inline constexpr const char* kHipCenteredStepping = "hip_centered_stepping";
/** Table C.1. H_r(pdot, Phi) = PTP(sqrt(p_z/g) (pdot - pdot_d)): the capture point of the velocity error. */
inline constexpr const char* kCapturePoint = "capture_point";
/** Table C.2. H_r(pdot) = a1 pdot + a0: step in the direction of travel. */
inline constexpr const char* kTranslationalStepping = "translational_stepping";
/** Table C.2. H_r(psidot) = a1 psidot + a0: step along the arc of an in-place turn. */
inline constexpr const char* kInPlaceTurning = "in_place_turning";
/** Table C.2. H_r(pdot x omega) = a1 (pdot x omega) + a0: step outward along the turn radius at speed. */
inline constexpr const char* kHighSpeedTurning = "high_speed_turning";

// ---- Contact-wrench heuristics: the contact force reference (H_f) ----

/** Table C.1. H_f(Phi) = m g / (F beta): the vertical stance force scaled by the reciprocal of the duty factor. */
inline constexpr const char* kImpulseScaling = "impulse_scaling";
/** Table C.1. H_f(Theta x pdot) = m omega x pdot: the centripetal force of a turn taken at speed. */
inline constexpr const char* kCentripetalAcceleration = "centripetal_acceleration";

}  // namespace heuristic

/**
 * Which reference channel a heuristic shapes, and therefore which of the three lists it belongs to and which base
 * class it derives from.
 *
 * The three are kept apart rather than merged into one list because they are evaluated at different places, at
 * different rates and with different arguments: base-pose offsets per shooting node from the node's reference,
 * foothold offsets once per swing from the last measurement, contact-force offsets per node from the contact flags.
 * A single list would make `orientation_compensation` in the foothold position a run-time surprise instead of a
 * load-time error naming the three foothold heuristics that would have been valid there.
 */
enum class HeuristicKind { BASE_POSE, FOOTHOLD, WRENCH };

/** The three kinds, for the loops that must visit every list. Use it instead of writing the three out again. */
inline constexpr size_t kNumHeuristicKinds = 3;
const std::array<HeuristicKind, kNumHeuristicKinds>& allHeuristicKinds();

/** The name of a kind as the task file spells its list: "base_pose", "foothold", "wrench". */
absl::string_view heuristicKindName(HeuristicKind kind);

/** Normalises a heuristic name for comparison: lower case, `_`, `-` and spaces removed. */
std::string normalizeHeuristicName(absl::string_view name);

/** Canonical names of every heuristic of a kind, in the order of Bledt's Appendix C tables. */
const std::vector<std::string>& knownHeuristicNames(HeuristicKind kind);

/** Canonical spelling of `name` among the heuristics of `kind`, or empty if the name is not one of them. */
std::string canonicalHeuristicName(HeuristicKind kind, absl::string_view name);

/**
 * The kind a name belongs to, whichever list it was found in, or empty when it is not a heuristic name at all.
 *
 * Used only to improve the rejection message: a name that IS a heuristic but is in the wrong list gets told which
 * list it belongs in, rather than "unknown foothold heuristic".
 */
std::optional<HeuristicKind> heuristicKindOf(absl::string_view name);

/**
 * The three name lists of the layer's formulation: which heuristics shape which reference channel.
 *
 * EVERY LIST IS EMPTY BY DEFAULT, and that is the whole safety story of this subsystem. Each of these changes the
 * closed loop, none has been validated in simulation on hardware-like conditions on these robots, and Bledt's own
 * coefficients were fitted to a 9 kg quadruped rather than to a humanoid. So the layer ships inert and every name is
 * turned on one at a time, which is also how the dissertation itself proceeds (section 4.3, figure 4-13: one
 * heuristic added per panel, with the viable operating region measured after each).
 */
struct LocomotionHeuristicFormulation {
  std::vector<std::string> basePose{};
  std::vector<std::string> foothold{};
  std::vector<std::string> wrench{};

  std::vector<std::string>& list(HeuristicKind kind);
  const std::vector<std::string>& list(HeuristicKind kind) const;

  bool empty() const { return basePose.empty() && foothold.empty() && wrench.empty(); }
  bool listed(HeuristicKind kind, absl::string_view name) const;

  /**
   * Rejects a formulation that cannot do what it says: an unknown name, a name in the wrong list, or a duplicate.
   *
   * Returns the valid names of the kind in the message, so that a typo is one line away from being fixed without
   * opening a header.
   */
  absl::Status validate() const;

  /**
   * Configurations that are legal and self-inconsistent, reported rather than rejected.
   *
   * They cost performance or make a heuristic inert; they do not make the controller wrong, and refusing to start over
   * one of them would stop a shipped robot's file from loading over a tuning choice. validate() does not emit them;
   * the layer does, once, at construction.
   */
  std::vector<std::string> warnings() const;

  /** One line per listed heuristic, for the start-up banner. */
  std::string summary() const;
};

}  // namespace ocs2::humanoid
