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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_mpc_validation/closed_loop/ClosedLoopDriver.h"

#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <mujoco_sim_interface/CheaterSimContactEstimator.h>
#include <ocs2_core/misc/LoadData.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/robot/InitialSimState.h"
#include "humanoid_common_mpc_app/robot/MujocoRobotBackend.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"

namespace ocs2::humanoid::validation {
namespace {

/** [s] How often the controller-side keys of the task file are checked, as RobotProcess checks them (kTaskFileCheckPeriod). */
constexpr double kTaskFileCheckPeriod = 1.0;

}  // namespace

absl::StatusOr<robot::mujoco_sim_interface::MujocoSimConfig> ClosedLoopDriver::simulatorConfig() const {
  // LINT.IfChange(robot_backend_options)
  RobotBackendOptions options;
  options.robotName = configuration_.name;
  options.urdfFile = configuration_.urdfFile;
  options.mjcfFile = configuration_.sceneFile;
  options.initialState.emplace(*initialRobotState_);
  options.contactFrameNames = modelSettings_->contactNames;
  options.contactParentJointNames = modelSettings_->contactParentJointNames;
  options.simulator = settings_.simulator;
  options.headless = true;
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalMpcRobotMain.cpp:robot_backend_options, //humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcRobotMain.cpp:robot_backend_options)
  // clang-format on
  ASSIGN_OR_RETURN(robot::mujoco_sim_interface::MujocoSimConfig config, MujocoRobotBackend::makeConfig(options));
  config.verbose = false;
  return config;
}

void ClosedLoopDriver::setGuiVelocityCommand(const Eigen::Vector4d& message) {
  // The message the GUI publishes, applied as the MPC node applies a received one.
  humanoid_mpc_msgs::WalkingVelocityCommand proto;
  proto.set_linear_velocity_x(message(0));
  proto.set_linear_velocity_y(message(1));
  proto.set_desired_pelvis_height(message(2));
  proto.set_angular_velocity_z(message(3));
  const absl::Status applied = node::applyWalkingVelocityCommand(proto, *motionManager_);
  if (!applied.ok()) LOG(ERROR) << "[ClosedLoopDriver] The MPC node refuses the GUI's message: " << applied;
}

Eigen::Vector3d ClosedLoopDriver::referenceVelocity() const {
  const vector4_t& ramped = motionManager_->getRampedVelocityCommand();
  return Eigen::Vector3d(ramped(0), ramped(1), ramped(3));
}

double ClosedLoopDriver::referenceBaseHeight(double pelvisHeight) const {
  return calculator_->commandedBaseHeight(pelvisHeight);
}

DriverSolveOutcome ClosedLoopDriver::solve() {
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  InProcessMpcLink::SolverIterationResult iteration = mpcLink_->runSolverIteration();
  const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  DriverSolveOutcome outcome;
  outcome.status = std::move(iteration.status);
  outcome.retryDelay = iteration.retryDelay.count();
  outcome.wallTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
  if (outcome.status.ok()) {
    // The SQP runs one iteration per solve on every robot (sqpIteration: 1), so the last interval is the solve's.
    const SqpSolver::Benchmarks benchmarks = mpc_->getSolverPtr()->getBenchmarks();
    outcome.lqApproximationMs = benchmarks.linearQuadraticApproximationTime;
    outcome.solveQpMs = benchmarks.solveQpTime;
    outcome.linesearchMs = benchmarks.linesearchTime;
    outcome.computeControllerMs = benchmarks.computeControllerTime;
  }
  return outcome;
}

std::optional<InitialStateGap> ClosedLoopDriver::lastSolveInitialStateGap() const {
  // delta_x0 of the SQP's first iteration, as the solver recorded it: after the reset the iteration served first (which
  // discards the previous solution, so the gap is zero), after trajectorySpread, on the warm start the SQP built.
  const vector_t& delta = mpc_->getSolverPtr()->getInitialStateGap();
  if (delta.size() == 0) return std::nullopt;
  // The Euler layout: the stored orientation is (yaw, pitch, roll), and delta_x0 holds their differences.
  // LINT.IfChange(initial_state_gap_layout)
  InitialStateGap gap;
  gap.rotation = measurementModel_->getBaseOrientationEulerZYX(delta).norm();
  gap.norm = delta.norm();
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/StateLayout.h)
  return gap;
}

std::array<Eigen::Vector3d, 2> ClosedLoopDriver::contactPositions(const SystemObservation& observation) {
  const vector_t q = measurementModel_->getGeneralizedCoordinates(observation.state);
  const PinocchioInterface::Model& model = measurementPinocchio_->getModel();
  PinocchioInterface::Data& data = measurementPinocchio_->getData();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  return {data.oMf[contactFrameIds_[0]].translation(), data.oMf[contactFrameIds_[1]].translation()};
}

absl::Status ClosedLoopDriver::connectSimulator(const robot::mujoco_sim_interface::MujocoSimInterface& sim) {
  // The backend's estimators next to the built-in ones (MujocoRobotBackend::registerContactEstimators()).
  if (!contactEstimators_.has(robot::mujoco_sim_interface::kCheaterSimContactEstimatorName)) {
    robot::mujoco_sim_interface::registerCheaterSimContactEstimator(contactEstimators_, sim);
  }
  const std::string canonical = robot::model::ContactEstimatorRegistry::canonicalName(settings_.contactEstimator);
  if (!contactEstimators_.has(canonical)) {
    return absl::InvalidArgumentError(absl::StrCat("contactEstimator of the task file: there is no contact estimator '",
                                                   settings_.contactEstimator, "'. Available: ", contactEstimators_.availableNames(), "."));
  }
  try {
    robotController_->setContactEstimator(contactEstimators_.create(canonical));
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(error.what());
  }
  contactEstimatorName_ = canonical;
  return absl::OkStatus();
}

void ClosedLoopDriver::startMpc(const robot::model::RobotState& robotState) {
  if (!mpcStarted_) {
    robotController_->startMpc(robotState);
    mpcStarted_ = true;
    return;
  }
  robotController_->requestMpcReset();
}

void ClosedLoopDriver::applyControllerSideSettings(double time) {
  if (time >= nextTaskFileCheckTime_) {
    nextTaskFileCheckTime_ = time + kTaskFileCheckPeriod;
    taskFileWatcher_->poll();
  }
  if (!pendingControllerSideSettings_.has_value()) return;
  const ControllerSideSettings settings = *std::move(pendingControllerSideSettings_);
  pendingControllerSideSettings_.reset();
  // LINT.IfChange(controller_side_updates)
  // Touch-down shaping of the contact wrenches.
  if (settings.contactWrenchGate.has_value()) {
    const ContactWrenchGate::Config& current = robotController_->contactWrenchGateConfig();
    if (settings.contactWrenchGate->debounceTime != current.debounceTime || settings.contactWrenchGate->rampTime != current.rampTime) {
      robotController_->setContactWrenchGateConfig(*settings.contactWrenchGate);
    }
  }
  // The contact estimator, by name; an unknown name keeps the estimator in use.
  if (settings.contactEstimator.has_value()) {
    const std::string canonical = robot::model::ContactEstimatorRegistry::canonicalName(*settings.contactEstimator);
    if (!contactEstimators_.has(canonical)) {
      LOG(ERROR) << "[ClosedLoopDriver] Unknown contactEstimator '" << *settings.contactEstimator << "' in " << configuration_.taskFile
                 << "; keeping '" << contactEstimatorName_ << "'. Available: " << contactEstimators_.availableNames() << ".";
    } else if (canonical != contactEstimatorName_) {
      robotController_->setContactEstimator(contactEstimators_.create(canonical));
      contactEstimatorName_ = canonical;
    }
  }
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotProcess.cpp:controller_side_updates)
}

absl::Status ClosedLoopDriver::initializeShared(const RobotConfiguration& configuration,
                                                const ClosedLoopDriverOptions& options,
                                                const ModelSettings& modelSettings,
                                                const mpc::Settings& mpcSettings,
                                                const sqp::Settings& sqpSettings,
                                                const OptimalControlProblem& problem,
                                                const Initializer& initializer,
                                                std::shared_ptr<ReferenceManagerInterface> referenceManager,
                                                const MpcRobotModelBase<scalar_t>& robotModel,
                                                const PinocchioInterface& pinocchioInterface,
                                                const vector_t& initialMpcState) {
  configuration_ = configuration;
  // The robot binaries' keys, retired ones refused with their replacement.
  ASSIGN_OR_RETURN(settings_, loadRobotProcessSettings(configuration.taskFile));
  robotDescription_ = std::make_unique<robot::model::RobotDescription>(configuration.urdfFile);
  modelSettings_ = &modelSettings;
  mpcJointIndices_ = robotDescription_->getJointIndices(modelSettings.mpcModelJointNames);
  mpcFrequency_ = mpcSettings.mpcDesiredFrequency_;
  mrtFrequency_ = mpcSettings.mrtDesiredFrequency_;
  if (mpcFrequency_ <= 0.0 || mrtFrequency_ <= 0.0) {
    return absl::InvalidArgumentError(absl::StrCat("[ClosedLoopDriver] ", configuration.taskFile,
                                                   " must set positive mpcDesiredFrequency and mrtDesiredFrequency for a lockstep run"));
  }
  initialMpcState_ = initialMpcState;
  initialRobotState_.emplace(createInitialSimState(*robotDescription_, modelSettings, robotModel, initialMpcState_));

  // The MPC of the MPC node, with the solver threads of the options.
  sqp::Settings settings = sqpSettings;
  if (options.solverThreads.has_value()) settings.nThreads = *options.solverThreads;
  solverThreads_ = settings.nThreads;
  mpc_ = std::make_unique<SqpMpc>(mpcSettings, settings, problem, initializer);
  mpc_->getSolverPtr()->setReferenceManager(std::move(referenceManager));

  // The configuration the observation describes, for the contact positions.
  measurementModel_.reset(robotModel.clone());
  measurementPinocchio_ = std::make_unique<PinocchioInterface>(pinocchioInterface);
  if (modelSettings.contactNames.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat("[ClosedLoopDriver] ", configuration.taskFile, " names fewer than two contact points"));
  }
  for (size_t contact = 0; contact < 2; ++contact) {
    const std::string& frame = modelSettings.contactNames[contact];
    if (!measurementPinocchio_->getModel().existFrame(frame)) {
      return absl::NotFoundError(absl::StrCat("[ClosedLoopDriver] the model has no contact frame '", frame, "'"));
    }
    contactFrameIds_[contact] = measurementPinocchio_->getModel().getFrameId(frame);
  }

  // The robot process's watcher of the controller-side keys: an unchanged file never reports.
  taskFileWatcher_ = std::make_unique<TaskFileWatcher>(configuration.taskFile, [this](const std::string& file) {
    absl::StatusOr<ControllerSideSettings> loaded = loadControllerSideSettings(file);
    if (!loaded.ok()) {
      LOG(WARNING) << "[ClosedLoopDriver] Not applying the controller-side keys of " << file << ": " << loaded.status().message();
      return;
    }
    pendingControllerSideSettings_ = *std::move(loaded);
  });
  return absl::OkStatus();
}

absl::Status ClosedLoopDriver::initializeCommandPath(std::unique_ptr<TargetTrajectoriesCalculatorBase> calculator,
                                                     std::shared_ptr<SwitchedModelReferenceManager> referenceManager,
                                                     const MpcRobotModelBase<scalar_t>& commandModel) {
  calculator_ = std::move(calculator);
  // The GUI's command limits and pelvis height, from the same file the motion manager scales with.
  loadData::loadCppDataType(configuration_.referenceFile, "maxDisplacementVelocityX", commandLimits_(0));
  loadData::loadCppDataType(configuration_.referenceFile, "maxDisplacementVelocityY", commandLimits_(1));
  loadData::loadCppDataType(configuration_.referenceFile, "maxRotationVelocity", commandLimits_(2));
  loadData::loadCppDataType(configuration_.referenceFile, "defaultBaseHeight", defaultPelvisHeight_);
  if ((commandLimits_.array() <= 0.0).any()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ClosedLoopDriver] the command limits of ", configuration_.referenceFile, " must be positive"));
  }

  // LINT.IfChange(command_path)
  TargetTrajectoriesCalculatorBase* calculatorPtr = calculator_.get();
  ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajectories =
      [calculatorPtr](const vector4_t& velocityTarget, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
        return calculatorPtr->commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      };
  motionManager_ = std::make_shared<ProceduralMpcMotionManager>(configuration_.gaitFile, configuration_.referenceFile,
                                                                std::move(referenceManager), commandModel, targetTrajectories);
  // A reset of the MPC resets the command path with it, as the MPC nodes wire it.
  motionManager_->setResetHook([calculatorPtr]() { calculatorPtr->reset(); });
  mpc_->getSolverPtr()->addSynchronizedModule(motionManager_);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalMpcNode.cpp:mpc_wiring, //humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcNode.cpp:mpc_wiring)
  // clang-format on
  return absl::OkStatus();
}

MpcLinkFactory ClosedLoopDriver::lockstepMpcLinkFactory(absl::string_view solverName) {
  InProcessMpcLink::Config config;
  config.solverThreadName = std::string(solverName);
  config.execution = InProcessMpcLink::Execution::kCaller;
  return InProcessMpcLink::factory(*mpc_, std::move(config), &mpcLink_);
}

void ClosedLoopDriver::releaseControllerAndMpc() {
  // The controller first: it owns the link, which advances the MPC and calls back into the controller's reset target.
  robotController_.reset();
  mpcLink_ = nullptr;
  mpc_.reset();
  motionManager_.reset();
  calculator_.reset();
}

}  // namespace ocs2::humanoid::validation
