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

#include "humanoid_common_mpc_app/visualization/BalanceQuantities.h"

#include <cmath>
#include <optional>

#include "ocs2_core/Dimensions.h"

namespace ocs2::humanoid::visualization {

std::optional<scalar_t> pendulumFrequency(scalar_t gravity, scalar_t pendulumHeight) {
  if (!std::isfinite(gravity) || !std::isfinite(pendulumHeight) || !(gravity > 0.0) || !(pendulumHeight > 0.0)) {
    return std::nullopt;
  }
  return std::sqrt(gravity / pendulumHeight);
}

vector2_t divergentComponentOfMotion(const vector3_t& com, const vector3_t& comVelocity, scalar_t omega) {
  return com.head<kHorizontalPositionDim>() + comVelocity.head<kHorizontalPositionDim>() / omega;
}

std::optional<vector3_t> zeroMomentPoint(const feet_array_t<vector3_t>& positions,
                                         const feet_array_t<vector6_t>& wrenches,
                                         scalar_t minNormalForce) {
  vector3_t force = vector3_t::Zero();
  vector3_t moment = vector3_t::Zero();
  scalar_t weightedHeight = 0.0;
  for (size_t contact = 0; contact < positions.size(); ++contact) {
    const vector3_t contactForce = wrenches[contact].head<kForceDim>();
    force += contactForce;
    moment += positions[contact].cross(contactForce) + wrenches[contact].tail<kTorqueDim>();
    weightedHeight += contactForce.z() * positions[contact].z();
  }
  if (!(force.z() > minNormalForce)) {
    return std::nullopt;
  }
  const scalar_t height = weightedHeight / force.z();
  return vector3_t((height * force.x() - moment.y()) / force.z(), (height * force.y() + moment.x()) / force.z(), height);
}

}  // namespace ocs2::humanoid::visualization
