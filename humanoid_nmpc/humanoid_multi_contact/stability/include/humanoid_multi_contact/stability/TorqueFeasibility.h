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

#include "Eigen/Core"
#include "absl/status/statusor.h"

#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"

namespace ocs2::humanoid::multi_contact {

/** What minimumTorqueUtilization() returns. Passive data. */
struct TorqueFeasibilityResult {
  // min over the contact forces that hold the stance of max_j |tau_j| /
  // effortLimit_j, over the joints with an effort limit.
  double utilization = 0.0;
  // The joint with the largest |tau_j| / effortLimit_j at those forces.
  std::string maxTorqueJoint;
  // [N m] The joint torques at those forces, one per actuated joint
  // (RobotStanceModel::actuatedJoints()).
  Eigen::VectorXd jointTorques;
  // The generator scalings of those forces, in units of the robot's weight.
  Eigen::VectorXd scalings;
};

/**
 * The torque feasibility of a stance (design section 2): for the configuration
 * q that `model` was last set to, the joint torques that hold the stance are
 * tau = g(q) - sum_g J_g(q)^T f_g, linear in the contact forces f_g, and this
 * finds the forces that minimize the largest |tau_j| / effortLimit_j, with
 * one LP on the static equilibrium rows of `equilibrium` plus two rows per
 * joint (|tau_j| <= u effortLimit_j, minimize u). The stance is torque-feasible
 * within a margin when utilization <= that margin (isTorqueFeasible()).
 *
 * Requires that `equilibrium`'s stance is the model's at q: its contacts'
 * candidateIndex name the model's candidates, and the robot's CoM at q is in
 * static equilibrium on it (StaticEquilibrium::equilibriumResidual() within the
 * tolerance); otherwise the LP has no solution and HPIPM's failure is an
 * Internal error. Reads `model` only.
 */
absl::StatusOr<TorqueFeasibilityResult> minimumTorqueUtilization(
    const RobotStanceModel& model, const StaticEquilibrium& equilibrium);

/** Whether the torques use at most `torqueMargin` of the effort limits. */
inline bool isTorqueFeasible(const TorqueFeasibilityResult& result,
                             double torqueMargin) {
  return result.utilization <= torqueMargin;
}

}  // namespace ocs2::humanoid::multi_contact
