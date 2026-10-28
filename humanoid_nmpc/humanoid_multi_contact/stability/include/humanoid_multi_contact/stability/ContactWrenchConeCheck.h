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

#include "Eigen/Core"
#include "absl/status/statusor.h"

#include "humanoid_multi_contact/stability/Stance.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"

namespace ocs2::humanoid::multi_contact {

/** The rate of change of the centroidal momentum, in the world frame. Passive
 * data. */
struct CentroidalMomentumRate {
  // [N] d/dt (m c_dot).
  Eigen::Vector3d linear = Eigen::Vector3d::Zero();
  // [N m] d/dt L, the angular momentum about the center of mass.
  Eigen::Vector3d angular = Eigen::Vector3d::Zero();
};

/** What checkContactWrenchCone() returns. Passive data. */
struct ContactWrenchConeResult {
  bool feasible = false;
  // The L1 norm of the smallest mismatch of the normalized contact wrench.
  double residual = 0.0;
};

/**
 * The contact wrench the stance must apply for the momentum rate `rate` of a
 * robot of `mass` [kg] with its CoM at `com`, gravity along -z, normalized by
 * the weight m g and taken about `reference`:
 *
 *   F = (rate.linear + m g e_z) / (m g),
 *   M = rate.angular / (m g) + (com - reference) x F,
 *
 * from sum_i f_i + m g_vec = d/dt (m c_dot) and sum_i (p_i - c) x f_i = d/dt L
 * (design section 2). With rate = 0 it is StaticEquilibrium::staticWrench().
 */
Wrench requiredContactWrench(double mass,
                             const Eigen::Vector3d& com,
                             const CentroidalMomentumRate& rate,
                             const Eigen::Vector3d& reference);

/**
 * The dynamic stability of a motion at one instant (design section 2, Caron,
 * Pham and Nakamura): whether the gravito-inertial wrench of the planned
 * momentum rate lies in the linearized contact wrench cone CWC(S) of the
 * stance of `equilibrium`, by the feasibility LP of
 * StaticEquilibrium::wrenchResidual(). Feasible when the residual is at most
 * the equilibrium's feasibility tolerance.
 */
absl::StatusOr<ContactWrenchConeResult> checkContactWrenchCone(
    const StaticEquilibrium& equilibrium,
    double mass,
    const Eigen::Vector3d& com,
    const CentroidalMomentumRate& rate);

}  // namespace ocs2::humanoid::multi_contact
