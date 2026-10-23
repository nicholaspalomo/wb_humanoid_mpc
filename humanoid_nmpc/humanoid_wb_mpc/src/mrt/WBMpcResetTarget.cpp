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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"

#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

TargetTrajectories wbMpcResetTargetTrajectories(const SystemObservation& observation,
                                                const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                const PinocchioInterface& pinocchioInterface) {
  vector_t targetState = observation.state;

  // Zero out the generalized velocities v, the tail of the whole-body state [q, v].
  targetState.tail(mpcRobotModel.getVelocityDim()).setZero();

  // Keep only the heading of the base orientation, so that the target base is upright: the heading quaternion of the
  // observed attitude, its twist about the world z axis with the tilt dropped (design decision D9). A measured attitude,
  // so it is written as one (setBaseOrientation).
  mpcRobotModel.setBaseOrientation(targetState, headingQuaternion(mpcRobotModel.getBaseOrientation(targetState)));

  const vector_t targetInput = weightCompensatingInput(pinocchioInterface, {true, true}, mpcRobotModel);
  const scalar_t startTime = observation.time;
  return TargetTrajectories({startTime, startTime + kResetTargetDuration}, {targetState, targetState}, {targetInput, targetInput});
}

}  // namespace ocs2::humanoid
