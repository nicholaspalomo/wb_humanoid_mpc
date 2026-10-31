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

#include "support/CentroidalMpcLoop.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/misc/LinearInterpolation.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcResetTarget.h"
#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/state_weights.nproto.h"
#include "humanoid_mpc_config/yaw_pitch_roll.nproto.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

/** max | |xi| - 1 | of the base quaternion of `state`. */
scalar_t quaternionNormError(const MpcRobotModelBase<scalar_t>& model,
                             const vector_t& state) {
  return std::abs(state
                      .segment<4>(static_cast<Eigen::Index>(
                          model.getBaseOrientationStartIndex()))
                      .norm() -
                  1.0);
}

}  // namespace

absl::StatusOr<std::unique_ptr<CentroidalMpcLoop>> CentroidalMpcLoop::Create(
    ShippedCentroidalRobot robot, const CentroidalMpcLoopOptions& options) {
  std::unique_ptr<CentroidalMpcLoop> loop =
      absl::WrapUnique(new CentroidalMpcLoop());
  const CentroidalRobotFiles files = shippedCentroidalRobotFiles(robot);
  ASSIGN_OR_RETURN(CentroidalMpcConfig config, loadConfigOf(files));
  if (options.baseYawWeight.has_value()) {
    for (mpc_config::StateWeights* absl_nonnull weights :
         {&config.task.state_weights, &config.task.final_state_weights}) {
      mpc_config::YawPitchRoll orientation =
          weights->eulerzyx_base_rt_world.value_or(mpc_config::YawPitchRoll{});
      orientation.yaw = *options.baseYawWeight;
      weights->eulerzyx_base_rt_world = orientation;
    }
  }
  const mpc_config::ReferenceFile reference = config.reference;
  ASSIGN_OR_RETURN(loop->interface_, CentroidalMpcInterface::Create(
                                         std::move(config), files.urdfFile));
  CentroidalMpcInterface& interface = *loop->interface_;
  loop->period_ = 1.0 / interface.mpcSettings().mpcDesiredFrequency_;
  loop->mpc_ = std::make_unique<SqpMpc>(
      interface.mpcSettings(), interface.sqpSettings(),
      interface.getOptimalControlProblem(), interface.getInitializer());
  const MpcRobotModelBase<scalar_t>& effectiveModel =
      interface.getEffectiveMpcRobotModel();
  ASSIGN_OR_RETURN(
      loop->calculator_,
      CentroidalMpcTargetTrajectoriesCalculator::Create(
          reference, effectiveModel, interface.getPinocchioInterface(),
          interface.mpcSettings().timeHorizon_));
  loop->calculator_->setTerrainHeightSource(
      [referenceManager = interface.getSwitchedModelReferenceManagerPtr()]() {
        return referenceManager->getAppliedTerrainHeight();
      });
  CentroidalMpcTargetTrajectoriesCalculator* absl_nonnull calculator =
      loop->calculator_.get();
  ASSIGN_OR_RETURN(
      const std::string gaitFile,
      robot::resolveResourcePath(
          "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto"));
  ASSIGN_OR_RETURN(
      loop->motionManager_,
      ProceduralMpcMotionManager::Create(
          gaitFile, files.referenceFile,
          interface.getSwitchedModelReferenceManagerPtr(), effectiveModel,
          [calculator](const vector4_t& velocity, scalar_t initTime,
                       scalar_t /*finalTime*/, const vector_t& initState,
                       const BasePlanarMotion& measuredMotion) {
            return calculator->velocityCommandToTargetTrajectories(
                velocity, initTime, initState, measuredMotion);
          }));
  loop->motionManager_->setResetHook([calculator]() { calculator->reset(); });
  // The command as given: no filter, no ramp, limits of 1, so that a command in
  // m/s and rad/s is the scaled one.
  ASSIGN_OR_RETURN(ReferenceSettings settings,
                   referenceSettingsFromConfig(reference));
  settings.cmdMaxLinvelBaseRtWorldEwrtHeadingX = 1.0;
  settings.cmdMaxLinvelBaseRtWorldEwrtHeadingY = 1.0;
  settings.cmdMaxAngvelBaseRtWorldEwrtWorldZ = 1.0;
  settings.cmdMaxLinaccBaseRtWorldEwrtHeadingXy = 0.0;
  settings.cmdMaxAngaccBaseRtWorldEwrtWorldZ = 0.0;
  settings.velocityCommandFilterBreakFrequency = 0.0;
  RETURN_IF_ERROR(loop->motionManager_->applyCommandLimits(settings));
  loop->mpc_->getSolverPtr()->setReferenceManager(
      interface.getReferenceManagerPtr());
  loop->mpc_->getSolverPtr()->addSynchronizedModule(loop->motionManager_);
  loop->mrt_ = std::make_unique<MPC_MRT_Interface>(*loop->mpc_);
  // The observation of every solve to the motion manager, before the solve, as
  // the MPC node's server and the lockstep closed loop hand it over
  // (ProceduralMpcMotionManager::observe()): its input carries the joint
  // velocities of the measured base twist. The loop owns both.
  ProceduralMpcMotionManager* absl_nonnull motionManager =
      loop->motionManager_.get();
  loop->mrt_->setSolverObservationObserver(
      [motionManager](const SystemObservation& observation) {
        motionManager->observe(observation);
      });
  loop->mrt_->setStateManifold(interface.getStateManifold());
  loop->mrt_->initRollout(&interface.getRollout());
  loop->observation_.time = 0.0;
  loop->observation_.state = interface.getInitialState();
  loop->observation_.input =
      vector_t::Zero(static_cast<Eigen::Index>(effectiveModel.getInputDim()));
  loop->observation_.mode = ModeNumber::kStance;
  loop->mrt_->setCurrentObservation(loop->observation_);
  loop->mrt_->resetMpcNode(centroidalMpcResetTargetTrajectories(
      loop->observation_, interface.getCentroidalModelInfo(), effectiveModel,
      interface.getPinocchioInterface()));
  return loop;
}

void CentroidalMpcLoop::command(scalar_t forward,
                                scalar_t lateral,
                                scalar_t yawRate) {
  motionManager_->setAndScaleVelocityCommand(WalkingVelocityCommand(
      forward, lateral, /*desired_pelvis_h=*/0.0, yawRate));
}

absl::StatusOr<CentroidalMpcLoopStep> CentroidalMpcLoop::step() {
  const StateManifold& manifold = *interface_->getStateManifold();
  const MpcRobotModelBase<scalar_t>& model = interface_->getMpcRobotModel();
  CentroidalMpcLoopStep record;
  if (mrt_->initialPolicyReceived()) {
    const PrimalSolution& previous = mrt_->getPolicy();
    const vector_t planned = model.interpolateStateTrajectory(
        observation_.time, previous.timeTrajectory_, previous.stateTrajectory_);
    record.initialGapRotation = manifold.getMaximumTangentAngle(
        manifold.difference(planned, observation_.state));
  }
  mrt_->setCurrentObservation(observation_);
  RETURN_IF_ERROR(mrt_->advanceMpc());
  mrt_->updatePolicy();
  record.performance = mpc_->getSolverPtr()->getPerformanceIndeces();
  const PrimalSolution& solution = mrt_->getPolicy();
  for (const vector_t& node : solution.stateTrajectory_) {
    record.quaternionNormError =
        std::max(record.quaternionNormError, quaternionNormError(model, node));
  }
  // The period starts from the observation, pushed (push()); the feedback the
  // policy applies there, against its input at the plan.
  vector_t plantState = observation_.state;
  plantState.head<6>() += pendingPush_;
  pendingPush_.setZero();
  vector_t plannedState;
  vector_t input;
  size_t mode = 0;
  mrt_->evaluatePolicy(observation_.time, plantState, plannedState, input,
                       mode);
  const vector_t feedforward =
      solution.controllerPtr_->computeInput(observation_.time, plannedState);
  record.feedbackCorrection = (input - feedforward).cwiseAbs().maxCoeff();
  record.feedforwardMagnitude = feedforward.cwiseAbs().maxCoeff();
  // The plant: the policy rolled out from the observation over one period,
  // projected onto the manifold.
  vector_t nextState;
  vector_t nextInput;
  mrt_->rolloutPolicy(observation_.time, plantState, period_, nextState,
                      nextInput, mode);
  observation_.time += period_;
  observation_.state = std::move(nextState);
  observation_.input = std::move(nextInput);
  observation_.mode = mode;
  record.quaternionNormError =
      std::max(record.quaternionNormError,
               quaternionNormError(model, observation_.state));
  return record;
}

}  // namespace ocs2::humanoid
