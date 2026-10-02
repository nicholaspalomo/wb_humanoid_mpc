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

#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include <humanoid_common_mpc/common/StatusMacros.h>
#include <humanoid_common_mpc/common/ThreadAffinity.h>
#include <humanoid_wb_mpc/WBMpcInterface.h>
#include <humanoid_wb_mpc/mrt/WBMpcMrtJointController.h>
#include <robot_model/ContactEstimatorRegistry.h>
#include <robot_model/RobotDescription.h>

#include "humanoid_common_mpc_app/node/NodeBus.h"
#include "humanoid_common_mpc_app/node/ShutdownSignal.h"
#include "humanoid_common_mpc_app/robot/InitialSimState.h"
#include "humanoid_common_mpc_app/robot/MrtRobotController.h"
#include "humanoid_common_mpc_app/robot/RemoteMpcLinkAdapter.h"
#include "humanoid_common_mpc_app/robot/RobotAppFlags.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotProcess.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"

/*
 * humanoid_wb_mpc_robot: the robot process of the whole-body MPC (humanoid_nmpc/docs/distributed_runtime/README.md,
 * "Processes"). The realtime loop over the backend --backend names (mujoco: the MuJoCo simulator) and the whole-body MRT
 * joint controller, which reaches the MPC node over the bus (RemoteMpcLink), in simulation as on the robot. What the ROS
 * sim WBMpcRobotSim did, without ROS and without the MPC in the process. See
 * humanoid_nmpc/humanoid_common_mpc_app/robot/README.md.
 */

namespace ocs2::humanoid {
namespace {

/** The modes of a humanoid MPC: the contact combinations of its two feet (MotionPhaseDefinition.h). */
constexpr size_t kNumModes = 4;

absl::Status run() {
  const SystemCoreAllocation cores = getDefaultCoreAllocation();
  ASSIGN_OR_RETURN(const RobotAppOptions options, robotAppOptionsFromFlags(cores.mrtCores, cores.simCores));
  const node::MpcFiles& files = options.files;

  // The models the controller needs, nothing more: no reference manager and no optimal control problem, so nothing is
  // taped or loaded with CppAD (the MPC is the MPC node's).
  ASSIGN_OR_RETURN(std::unique_ptr<WBMpcInterface> interface,
                   WBMpcInterface::CreateControllerModels(files.taskFile, files.urdfFile, files.referenceFile));
  ASSIGN_OR_RETURN(const RobotProcessSettings settings, loadRobotProcessSettings(files.taskFile));
  if (settings.wbMpcFeedforward != WbMpcFeedforward::kInverseDynamics) {
    return absl::InvalidArgumentError(absl::StrCat(files.taskFile, ": wbMpcFeedforward: ", wbMpcFeedforwardName(settings.wbMpcFeedforward),
                                                   "; the whole-body controller computes its feedforward with the inverse dynamics only."));
  }
  // The whole-body controller keeps a reference to the model settings: the interface outlives it.
  const ModelSettings& modelSettings = interface->modelSettings();
  const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel = interface->getMpcRobotModel();

  // Init Sim state
  const robot::model::RobotDescription robotDescription(files.urdfFile);
  const robot::model::RobotState initState =
      createInitialSimState(robotDescription, modelSettings, interface->getMpcRobotModel(), interface->getInitialState());
  LOG(INFO) << "initState: " << initState.getRootPositionInWorldFrame().transpose();

  // The bus first: it outlives everything registered on it.
  ASSIGN_OR_RETURN(std::unique_ptr<robot::ipc::Bus> bus, node::createNodeBus(options.networkConfig, options.ipcNode));

  // LINT.IfChange(robot_backend_options)
  RobotBackendOptions backendOptions;
  backendOptions.robotName = options.robotName;
  backendOptions.urdfFile = files.urdfFile;
  backendOptions.mjcfFile = options.mjcfFile;
  backendOptions.initialState.emplace(initState);
  backendOptions.contactFrameNames = modelSettings.contactNames;
  backendOptions.contactParentJointNames = modelSettings.contactParentJointNames;
  backendOptions.simulator = settings.simulator;
  backendOptions.headless = options.headless;
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:robot_backend_options)
  ASSIGN_OR_RETURN(std::unique_ptr<RobotBackend> backend, RobotBackendRegistry().create(options.backend, backendOptions));
  robot::model::ContactEstimatorRegistry contactEstimators;
  backend->registerContactEstimators(contactEstimators);

  // The MPC link.
  const ipc::ModelDimensions dimensions{
      .stateDim = mpcRobotModel.getStateDim(), .inputDim = mpcRobotModel.getInputDim(), .numModes = kNumModes};
  ipc::RemoteMpcLink::Config linkConfig;
  linkConfig.dimensions = dimensions;
  linkConfig.policyTimeout = settings.mpcLinkPolicyTimeout;
  ASSIGN_OR_RETURN(std::unique_ptr<RemoteMpcLinkAdapter> link, RemoteMpcLinkAdapter::Create(*bus, std::move(linkConfig)));
  RemoteMpcLinkAdapter* const remoteLink = link.get();
  const MpcLinkFactory linkFactory = handOverMpcLink(std::move(link));

  const std::filesystem::path configDir = std::filesystem::path(files.taskFile).parent_path().parent_path();
  const std::string pdGainsFile = (configDir / "controller" / "joint_pd_gains.yaml").string();
  // LINT.IfChange(whole_body_robot_controller)
  std::unique_ptr<WBMpcMrtJointController> jointController;
  try {
    jointController = std::make_unique<WBMpcMrtJointController>(robotDescription, modelSettings, linkFactory,
                                                                interface->getPinocchioInterface(), pdGainsFile);
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat("the whole-body MRT joint controller did not start: ", error.what()));
  }
  // The controller settings of the task file.
  if (settings.contactWrenchGate.has_value()) {
    jointController->setContactWrenchGateConfig(*settings.contactWrenchGate);
  }
  // SAFETY: time constant of the exponential decay applied to the joint PD gains once the mode is entered (the
  // controller's default when the task file has none).
  if (settings.safetyDecayTimeConstant.has_value()) {
    jointController->setSafetyDecayTimeConstant(*settings.safetyDecayTimeConstant);
  }
  LOG(INFO) << "MPC MRT joint controller is set up with PD gains from: " << pdGainsFile;
  // The whole-body sim handed the controller its posture before its mode.
  MrtRobotController<WBMpcMrtJointController> controller(std::move(jointController), CycleInputOrder::kPostureThenMode);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/WholeBodyClosedLoopDriver.cpp:whole_body_robot_controller)

  const scalar_t mrtDesiredFrequency = interface->mpcSettings().mrtDesiredFrequency_;
  RobotProcess::Config config;
  config.controlFrequency = mrtDesiredFrequency > 0.0 ? mrtDesiredFrequency : 100.0;
  config.realtimePriority = options.realtimePriority;
  config.realtimeCores = options.realtimeCores;
  config.backendCores = options.backendCores;
  config.settings = settings;
  config.initialState.emplace(initState);
  config.taskFile = files.taskFile;
  // Whether a caught robot is at rest is judged on the joints of the MPC model: the ones JOINT_PD brings to the nominal
  // posture and the MPC starts from.
  config.restJointIndices = robotDescription.getJointIndices(modelSettings.mpcModelJointNames);
  // The joint_pd_gains.yaml file watcher, every 500 control periods (~1 Hz at ~500 Hz).
  config.pdGainsFileCheckInterval = 500;

  RobotProcess::Hooks hooks;
  hooks.takeViewerAnnotations = [remoteLink](msgs::ViewerAnnotations& annotations) {
    return remoteLink->remote().takeAnnotations(annotations);
  };
  hooks.fillLinkStatistics = [remoteLink](humanoid_mpc_msgs::LoopTiming& loopTiming) {
    const ipc::RemoteMpcLink::Statistics statistics = remoteLink->remote().statistics();
    loopTiming.set_stale_policies_dropped(statistics.stalePoliciesDropped);
    loopTiming.set_policy_age_s(statistics.policyAge);
  };

  ASSIGN_OR_RETURN(std::unique_ptr<RobotProcess> process,
                   RobotProcess::Create(*bus, *backend, controller, contactEstimators, std::move(config), std::move(hooks)));
  RETURN_IF_ERROR(bus->start());
  RETURN_IF_ERROR(process->start());
  // Until SIGINT or SIGTERM, or until a cycle of the realtime loop throws (the backend is then in its safe state and the
  // binary exits with a failure, which the container's restart policy answers).
  const absl::Status ended = process->runUntilShutdown(&node::shutdownRequested);
  LOG(INFO) << "The robot process has stopped.";
  return ended;
}

}  // namespace
}  // namespace ocs2::humanoid

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "The robot process of the whole-body MPC: the realtime loop over a robot backend and the MRT joint controller, with "
      "its MPC behind the bus. humanoid_wb_mpc_robot --robot_name=unitree_g1 --task_file=... --reference_file=... "
      "--urdf_file=... --mjcf_file=... [--network_config=...] [--backend=mujoco] [--realtime_priority=80]");
  absl::ParseCommandLine(argc, argv);
  // Route Abseil log records to stderr, INFO included.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  ocs2::humanoid::node::installShutdownSignalHandlers();
  const absl::Status status = ocs2::humanoid::run();
  if (!status.ok()) {
    LOG(ERROR) << "humanoid_wb_mpc_robot: " << status;
    return 1;
  }
  return 0;
}
