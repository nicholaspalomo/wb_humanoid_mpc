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
#include <cstddef>
#include <optional>

#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid::visualization {

std::optional<scalar_t> pendulumFrequency(
    scalar_t gravityMagnitude, scalar_t proxyPosComRtFeetmeanEwrtWorldZ) {
  if (!std::isfinite(gravityMagnitude) ||
      !std::isfinite(proxyPosComRtFeetmeanEwrtWorldZ) ||
      !(gravityMagnitude > 0.0) || !(proxyPosComRtFeetmeanEwrtWorldZ > 0.0)) {
    return std::nullopt;
  }
  return std::sqrt(gravityMagnitude / proxyPosComRtFeetmeanEwrtWorldZ);
}

vector2_t computePosDcmRtWorldEwrtWorldXy(
    const vector3_t& posComRtWorldEwrtWorld,
    const vector3_t& linvelComRtWorldEwrtWorld,
    scalar_t omega) {
  return posComRtWorldEwrtWorld.head<kHorizontalPositionDim>() +
         linvelComRtWorldEwrtWorld.head<kHorizontalPositionDim>() / omega;
}

std::optional<vector3_t> computePosZmpRtWorldEwrtWorld(
    const feet_array_t<vector3_t>& posFeetRtWorldEwrtWorld,
    const feet_array_t<vector6_t>& wrenchFeetEwrtWorld,
    scalar_t minForceWorldEwrtWorldZ) {
  // The net force, and the net moment about the world origin.
  vector3_t forceWorldEwrtWorld = vector3_t::Zero();
  vector3_t torqueWorldEwrtWorld = vector3_t::Zero();
  // The sum of the feet's heights, each weighted by its normal force.
  scalar_t weightedSumPosFeetRtWorldEwrtWorldZ = 0.0;
  for (size_t contact = 0; contact < posFeetRtWorldEwrtWorld.size();
       ++contact) {
    const vector3_t forceFootEwrtWorld =
        wrenchFeetEwrtWorld[contact].head<kForceDim>();
    forceWorldEwrtWorld += forceFootEwrtWorld;
    torqueWorldEwrtWorld +=
        posFeetRtWorldEwrtWorld[contact].cross(forceFootEwrtWorld) +
        wrenchFeetEwrtWorld[contact].tail<kTorqueDim>();
    weightedSumPosFeetRtWorldEwrtWorldZ +=
        forceFootEwrtWorld.z() * posFeetRtWorldEwrtWorld[contact].z();
  }
  if (!(forceWorldEwrtWorld.z() > minForceWorldEwrtWorldZ)) {
    return std::nullopt;
  }
  const scalar_t posZmpRtWorldEwrtWorldZ =
      weightedSumPosFeetRtWorldEwrtWorldZ / forceWorldEwrtWorld.z();
  return vector3_t((posZmpRtWorldEwrtWorldZ * forceWorldEwrtWorld.x() -
                    torqueWorldEwrtWorld.y()) /
                       forceWorldEwrtWorld.z(),
                   (posZmpRtWorldEwrtWorldZ * forceWorldEwrtWorld.y() +
                    torqueWorldEwrtWorld.x()) /
                       forceWorldEwrtWorld.z(),
                   posZmpRtWorldEwrtWorldZ);
}

}  // namespace ocs2::humanoid::visualization
