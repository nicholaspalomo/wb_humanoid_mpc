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

#include "support/RollingContactTestSupport.h"

#include <string>

#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_centroidal_mpc/CentroidalMpcConfig.h"
#include "legged_centroidal_mpc/CentroidalMpcInterface.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "legged_nmpc/legged_common_mpc/test/support/StatusTestMacros.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::legged::rolling_contact_test {

namespace {
// [s] The end of the standing target.
constexpr scalar_t kTargetEndTime = 2.0;
// [s] The horizon the reference manager prepares.
constexpr scalar_t kHorizonEndTime = 1.0;
}  // namespace

CentroidalMpcConfig tron1Config() {
  return checkedValue(loadConfigOf(tron1Files()));
}

ModeSchedule rightWheelSwing() {
  return ModeSchedule({kSwingStartTime, kSwingEndTime},
                      {kBothWheels, kLeftWheelOnly, kBothWheels});
}

ModeSchedule bothWheelsStand() {
  return ModeSchedule({kSwingStartTime}, {kBothWheels, kBothWheels});
}

void prepareSolve(CentroidalMpcInterface& mpc, const ModeSchedule& schedule) {
  SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  const vector_t& state = mpc.getInitialState();
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc.getMpcRobotModel().getInputDim()));
  referenceManager.setTargetTrajectories(TargetTrajectories(
      {0.0, kTargetEndTime}, {state, state}, {input, input}));
  referenceManager.preSolverRun(/*initTime=*/0.0,
                                /*finalTime=*/kHorizonEndTime, state,
                                schedule.modeAtTime(0.0));
  mpc.getOptimalControlProblemRef().targetTrajectoriesPtr =
      &referenceManager.getTargetTrajectories();
}

vector3_t posWheelRtWorldEwrtWorld(CentroidalMpcInterface& mpc,
                                   const vector_t& state,
                                   absl::string_view frameName) {
  const pinocchio::Model& model = mpc.getPinocchioInterface().getModel();
  pinocchio::Data data(model);
  pinocchio::forwardKinematics(
      model, data, mpc.getMpcRobotModel().getGeneralizedCoordinates(state));
  pinocchio::updateFramePlacements(model, data);
  return data.oMf[model.getFrameId(std::string(frameName))].translation();
}

}  // namespace ocs2::legged::rolling_contact_test
