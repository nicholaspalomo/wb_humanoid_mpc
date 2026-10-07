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

#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The target trajectories the centroidal MPC restarts from after a reset served at `observation`: the observed
 * configuration held still and upright, with the weight on both feet.
 *
 *  - the state is the observation's, with the normalized linear and angular momentum zeroed and the base pitch and roll
 *    zeroed, so the target base is upright; the joint positions are the observed ones (the nominal ones would make a
 *    kinematically inconsistent target - the current base height with the nominal joint angles - that the MPC tries to
 *    "correct" by shooting the base upward);
 *  - the input carries the robot's weight evenly on both feet in the effective model's input parameterization at that
 *    state (weightCompensatingInput());
 *  - two nodes, at the observation time and 2 s later, so that the target holds over the horizon.
 *
 * The one definition of the reset target for every path that serves a reset: CentroidalMpcMrtJointController hands it
 * to its MPC link (InProcessMpcLink serves its resets from it), and the MPC node's MpcServer resets from it. Pure: it
 * reads `pinocchioInterface` (the robot's mass) and the models, and may be called on any thread.
 *
 * @param info            The centroidal model of the MPC.
 * @param effectiveModel  The model the OCP's inputs are laid out in (the basis-vector decorator when it is active).
 */
TargetTrajectories centroidalMpcResetTargetTrajectories(const SystemObservation& observation,
                                                        const CentroidalModelInfo& info,
                                                        const MpcRobotModelBase<scalar_t>& effectiveModel,
                                                        const PinocchioInterface& pinocchioInterface);

}  // namespace ocs2::humanoid
