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

#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The target trajectories the whole-body MPC restarts from after a reset served at `observation`: the observed
 * configuration held still and upright, with the weight on both feet.
 *
 *  - the state is the observation's, with every generalized velocity zeroed and the base pitch and roll zeroed;
 *  - the input carries the robot's weight evenly on both feet (weightCompensatingInput()), as the centroidal MPC's
 *    reset target does: a zero input asked the first solve after a reset to hold the robot up with no contact force;
 *  - two nodes, at the observation time and 2 s later, so that the target holds over the horizon rather than being a
 *    single knot extrapolated.
 *
 * The one definition of the reset target for every path that serves a reset: WBMpcMrtJointController hands it to its
 * MPC link (InProcessMpcLink serves its resets from it), and the MPC node's MpcServer resets from it. Pure: it reads
 * `pinocchioInterface` (the robot's mass) and the model, and may be called on any thread.
 */
TargetTrajectories wbMpcResetTargetTrajectories(const SystemObservation& observation,
                                                const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                const PinocchioInterface& pinocchioInterface);

}  // namespace ocs2::humanoid
