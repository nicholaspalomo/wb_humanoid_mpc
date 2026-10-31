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

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::contact_term {

/**
 * The names the per-foot contact terms are registered under in the optimal
 * control problem's collections: the foot's contact name followed by one of
 * these suffixes.
 *
 * The MPC interfaces register the terms under these names,
 * MpcParameterUpdaterModule finds them again by the same names to hot-reload
 * their weights, and the validation's SQP diagnostics find the hard
 * normal-velocity rows by theirs. The updater has to treat an absent term as
 * the normal case - most of them are only built for some task lists - so it
 * could never tell a term that is not listed from a term whose name had
 * drifted, and a rename on one side used to turn every hot reload of that term
 * into a silent no-op. With one definition, a rename is a rename on both sides.
 */
// The contact cones, schedule-gated or not
// (contactConstraintsAreScheduleGated()).
inline constexpr absl::string_view kContactWrenchCone = "_contactWrenchCone";
inline constexpr absl::string_view kFrictionForceCone = "_frictionForceCone";
inline constexpr absl::string_view kContactMomentXYName = "_contactMomentXY";
// The friction cone a 3-DoF contact takes where a 6-DoF one takes the contact
// wrench cone (PointFrictionConeConstraint).
inline constexpr absl::string_view kPointFrictionCone = "_pointFrictionCone";
// The contact-implicit formulation
// (humanoid_nmpc/docs/contact_implicit_mpc/README.md).
inline constexpr absl::string_view kContactComplementarity =
    "_contactComplementarity";
inline constexpr absl::string_view kForceWeightedSlipName =
    "_forceWeightedSlip";
inline constexpr absl::string_view kGroundPenetration = "_groundPenetration";
// The swing-foot vertical servo listed as a hard constraint, and as a soft one,
// which the contact-implicit formulation requires.
inline constexpr absl::string_view kNormalVelocityName = "_normalVelocity";
inline constexpr absl::string_view kNormalVelocitySoftName =
    "_normalVelocitySoft";
// The terms of a tracked multi-contact plan: the swing servo of any contact
// (ContactSwingConstraint), the landing alignment of a 6-DoF one
// (SwingLandingAlignmentConstraint) and the collision sphere kept out of the
// terrain (TerrainNonPenetrationConstraint).
inline constexpr absl::string_view kContactSwing = "_contactSwing";
inline constexpr absl::string_view kSwingLandingAlignment =
    "_swingLandingAlignment";
inline constexpr absl::string_view kTerrainNonPenetration =
    "_terrainNonPenetration";

/** The name the term with `suffix` of the foot `footName` is registered under.
 */
inline std::string name(absl::string_view footName, absl::string_view suffix) {
  return absl::StrCat(footName, suffix);
}

}  // namespace ocs2::humanoid::contact_term
