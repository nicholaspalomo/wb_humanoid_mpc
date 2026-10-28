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

#include "ocs2_core/Dimensions.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The rows of the contact features of a contact-invariant problem
 * (ContactFeatureFunction), per candidate in layout order and then for the
 * whole robot:
 *
 *  - kDistanceFeature: d_i, the signed distance of the pad to the terrain [m];
 *  - kSlipFeature (3 rows): v_t, the pad's velocity tangent to the terrain
 *    (along the blended normal n(p_i)) [m/s];
 *  - kTiltFeature: the pad's misalignment with n(p_i): 1 - n_pad . n for a
 *    surface pad, max(0, cos(max tilt) - n_pad . n) for a point pad;
 *  - kConeFeature: mu n . f - sqrt(|f_t|^2 + eps^2) in units of the robot's
 *    weight, not negative in the friction cone about n(p_i) (a point pad; 0
 *    for a surface pad, whose wrench cone is another term);
 *  - kClearanceFeature: the clearance of the pad's collision geometry to the
 *    terrain [m], not negative out of it;
 *
 * then kMomentumRateFeatures rows of the momentum rate the forces produce,
 * normalized by the mass: (sum f + m g) / m and sum (p - c) x f + tau over m.
 */
inline constexpr int kDistanceFeature = 0;
inline constexpr int kSlipFeature = 1;
inline constexpr int kTiltFeature = kSlipFeature + kPositionDim;
inline constexpr int kConeFeature = kTiltFeature + 1;
inline constexpr int kClearanceFeature = kConeFeature + 1;
inline constexpr int kFeaturesPerCandidate = kClearanceFeature + 1;
/** The rows of the contact-invariant residual e_i: distance, slip, tilt. */
inline constexpr int kResidualFeatures = kConeFeature;
inline constexpr int kMomentumRateFeatures = kPoseTangentDim;

/** The row of `feature` of candidate `candidate`. */
inline int featureRow(int candidate, int feature) {
  return candidate * kFeaturesPerCandidate + feature;
}

/** The first row of the momentum rate of `numCandidates` candidates. */
inline int momentumRateRow(int numCandidates) {
  return numCandidates * kFeaturesPerCandidate;
}

/** The number of feature rows of `numCandidates` candidates. */
inline int numFeatureRows(int numCandidates) {
  return momentumRateRow(numCandidates) + kMomentumRateFeatures;
}

}  // namespace ocs2::humanoid::multi_contact
