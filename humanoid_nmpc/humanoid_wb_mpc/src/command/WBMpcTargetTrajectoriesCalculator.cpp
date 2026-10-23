/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"

namespace ocs2::humanoid {

namespace {

/**
 * A target knot of the whole-body state of `model`: the base at the pose `basePoseEulerZyx` ([x, y, z, yaw, pitch,
 * roll]) and the joints at `jointAngles` (one per joint of the model), every velocity zero.
 */
vector_t restingTargetKnot(const MpcRobotModelBase<scalar_t>& model, const vector6_t& basePoseEulerZyx, const vector_t& jointAngles) {
  vector_t knot = vector_t::Zero(model.getStateDim());
  model.setBasePoseEulerZyx(knot, basePoseEulerZyx);
  model.setJointAngles(knot, jointAngles);
  return knot;
}

/**
 * restingTargetKnot() with the base moving at the world-frame planar velocity (`planarVelocityWorld`) and turning at
 * `yawRate` [rad/s] about the world z axis, the joints still.
 */
vector_t movingTargetKnot(const MpcRobotModelBase<scalar_t>& model,
                          const vector6_t& basePoseEulerZyx,
                          const vector_t& jointAngles,
                          const vector2_t& planarVelocityWorld,
                          scalar_t yawRate) {
  vector_t knot = restingTargetKnot(model, basePoseEulerZyx, jointAngles);
  model.setBaseComLinearVelocity(knot, vector3_t(planarVelocityWorld(0), planarVelocityWorld(1), 0.0));
  model.setBaseAngularVelocityFromWorldYawRate(knot, yawRate);
  return knot;
}

}  // namespace

absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>> WBMpcTargetTrajectoriesCalculator::Create(
    const mpc_config::ReferenceFile& referenceFile, const MpcRobotModelBase<scalar_t>& mpcRobotModel, scalar_t mpcHorizon) {
  ASSIGN_OR_RETURN(const ReferenceSettings referenceSettings, referenceSettingsFromConfig(referenceFile));
  ASSIGN_OR_RETURN(const vector_t defaultJointState,
                   defaultJointStateFromConfig(referenceFile, mpcRobotModel.modelSettings.mpcModelJointNames,
                                               mpcRobotModel.modelSettings.fixedJointNames));
  // The constructor is private, so std::make_unique cannot reach it.
  return absl::WrapUnique(new WBMpcTargetTrajectoriesCalculator(referenceSettings, defaultJointState, mpcRobotModel, mpcHorizon));
}

absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>> WBMpcTargetTrajectoriesCalculator::Create(
    const std::string& referenceFile, const MpcRobotModelBase<scalar_t>& mpcRobotModel, scalar_t mpcHorizon) {
  LOG(INFO) << "[WBMpcTargetTrajectoriesCalculator] reference file: " << referenceFile;
  ASSIGN_OR_RETURN(const mpc_config::ReferenceFile reference, loadReferenceFile(referenceFile));
  absl::StatusOr<std::unique_ptr<WBMpcTargetTrajectoriesCalculator>> calculator = Create(reference, mpcRobotModel, mpcHorizon);
  if (!calculator.ok()) {
    return withConfigFile(calculator.status(), referenceFile);
  }
  return calculator;
}

WBMpcTargetTrajectoriesCalculator::WBMpcTargetTrajectoriesCalculator(const ReferenceSettings& referenceSettings,
                                                                     const vector_t& defaultJointState,
                                                                     const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                     scalar_t mpcHorizon)
    : TargetTrajectoriesCalculatorBase(referenceSettings, defaultJointState, mpcRobotModel, mpcHorizon) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

TargetTrajectories WBMpcTargetTrajectoriesCalculator::commandedPositionToTargetTrajectories(const vector4_t& commadLinePoseTarget,
                                                                                            scalar_t initTime,
                                                                                            const vector_t& initState) {
  vector_t currentPose = getCurrentBasePoseTarget(initState);

  const vector_t targetPose = getDeltaBaseTarget(commadLinePoseTarget, currentPose);

  scalar_t targetReachingTime = initTime + estimateTimeToTarget(targetPose - currentPose);

  // desired time and state trajectories: resting knots at most a quarter turn apart in yaw, so that the slerp of the
  // orientation between them turns the commanded way (subdivideYawKnots())
  const std::vector<PoseCommandKnot> knots = subdivideYawKnots(initTime, currentPose, targetReachingTime, targetPose);
  scalar_array_t timeTrajectory;
  timeTrajectory.reserve(knots.size());
  vector_array_t stateTrajectory;
  stateTrajectory.reserve(knots.size());
  for (const PoseCommandKnot& knot : knots) {
    timeTrajectory.push_back(knot.time);
    stateTrajectory.push_back(restingTargetKnot(*mpcRobotModelPtr_, knot.basePose, targetJointState_));
  }

  // desired input trajectory (just right dimensions, they are not used)
  const vector_array_t inputTrajectory(knots.size(), vector_t::Zero(mpcRobotModelPtr_->getInputDim()));

  TargetTrajectories targetTrajectories{timeTrajectory, stateTrajectory, inputTrajectory};

  return targetTrajectories;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

TargetTrajectories WBMpcTargetTrajectoriesCalculator::commandedVelocityToTargetTrajectories(const vector4_t& commandedVelocities,
                                                                                            scalar_t initTime,
                                                                                            const vector_t& initState,
                                                                                            const BasePlanarMotion& measuredMotion) {
  // The base knots of the command on the arc of its twist, the first segment starting from the measured motion
  // (TargetTrajectoriesCalculatorBase's class comment; design defect D1). Every knot's base
  // moves at the command in its own heading frame and turns at the commanded yaw rate about the world z axis.
  const std::vector<VelocityCommandKnot> knots = velocityCommandKnots(commandedVelocities, initTime, initState, measuredMotion);

  scalar_array_t timeTrajectory;
  vector_array_t stateTrajectory;
  for (const VelocityCommandKnot& knot : knots) {
    timeTrajectory.push_back(knot.time);
    stateTrajectory.push_back(
        movingTargetKnot(*mpcRobotModelPtr_, knot.basePose, targetJointState_, knot.commandedVelocityWorld, knot.commandedYawRate));
  }

  // desired input trajectory (just right dimensions, they are not used)
  const vector_array_t inputTrajectory(stateTrajectory.size(), vector_t::Zero(mpcRobotModelPtr_->getInputDim()));

  TargetTrajectories targetTrajectories{timeTrajectory, stateTrajectory, inputTrajectory};

  return targetTrajectories;
}

}  // namespace ocs2::humanoid
