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

#include "humanoid_multi_contact/stability/ContactWrenchConeCheck.h"

#include "Eigen/Geometry"
#include "absl/status/statusor.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::multi_contact {

Wrench requiredContactWrench(double mass,
                             const Eigen::Vector3d& com,
                             const CentroidalMomentumRate& rate,
                             const Eigen::Vector3d& reference) {
  const double weight = mass * kGravitationalAcceleration;
  const Eigen::Vector3d force = rate.linear / weight + Eigen::Vector3d::UnitZ();
  Wrench wrench;
  wrench.head<kForceDim>() = force;
  wrench.tail<kTorqueDim>() =
      rate.angular / weight + (com - reference).cross(force);
  return wrench;
}

absl::StatusOr<ContactWrenchConeResult> checkContactWrenchCone(
    const StaticEquilibrium& equilibrium,
    double mass,
    const Eigen::Vector3d& com,
    const CentroidalMomentumRate& rate) {
  ASSIGN_OR_RETURN(const double residual,
                   equilibrium.wrenchResidual(requiredContactWrench(
                       mass, com, rate, equilibrium.generators().reference)));
  return ContactWrenchConeResult{
      .feasible = residual <= equilibrium.options().feasibilityTolerance,
      .residual = residual};
}

}  // namespace ocs2::humanoid::multi_contact
