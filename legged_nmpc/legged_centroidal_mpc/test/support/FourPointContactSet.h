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

#include "legged_centroidal_mpc/CentroidalMpcConfig.h"

/**
 * A contact set of four 3-DoF point contacts and no 6-DoF contact, the contact
 * set of a point-footed quadruped, on the DRC Atlas: the toe and the heel of
 * each sole, listed front-left, front-right, hind-left, hind-right. The tests
 * of the centroidal MPC on a non-biped set build on it.
 */
namespace ocs2::legged::four_point_set {

inline constexpr size_t kNumContacts = 4;

// The four contacts in contact order, a quadruped's leg order.
inline constexpr char kFrontLeft[] = "sole_l_toe_contact";
inline constexpr char kFrontRight[] = "sole_r_toe_contact";
inline constexpr char kHindLeft[] = "sole_l_heel_contact";
inline constexpr char kHindRight[] = "sole_r_heel_contact";

// The modes of the set: all four, and the two diagonal pairs of a trot.
inline constexpr size_t kAllFour = 0b1111;
inline constexpr size_t kFrontLeftHindRight = 0b1001;
inline constexpr size_t kFrontRightHindLeft = 0b0110;

/**
 * Returns the shipped DRC Atlas configuration `atlas` on the four sole corners
 * alone, on wrench inputs and the gait schedule, with only the terms that run
 * on any contact set and the procedural arm swing off. The set has no legs.
 */
CentroidalMpcConfig fourPointConfig(const CentroidalMpcConfig& atlas);

/** Returns fourPointConfig() described by its legs LF, RF, LH and RH, one
 * contact each, in contact order. */
CentroidalMpcConfig leggedFourPointConfig(const CentroidalMpcConfig& atlas);

}  // namespace ocs2::legged::four_point_set
