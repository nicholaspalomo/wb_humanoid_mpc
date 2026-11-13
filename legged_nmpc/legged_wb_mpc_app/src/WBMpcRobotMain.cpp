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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/base/nullability.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/StateManifoldFactory.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/ThreadAffinity.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/gait/MotionPhaseDefinition.h"
#include "legged_common_mpc/mrt/MpcLink.h"
#include "legged_common_mpc_app/node/NodeBus.h"
#include "legged_common_mpc_app/node/ShutdownSignal.h"
#include "legged_common_mpc_app/robot/InitialSimState.h"
#include "legged_common_mpc_app/robot/MrtRobotController.h"
#include "legged_common_mpc_app/robot/RemoteMpcLinkAdapter.h"
#include "legged_common_mpc_app/robot/RobotAppFlags.h"
#include "legged_common_mpc_app/robot/RobotAppOptions.h"
#include "legged_common_mpc_app/robot/RobotBackendRegistry.h"
#include "legged_common_mpc_app/robot/RobotConfigDirectory.h"
#include "legged_common_mpc_app/robot/RobotConfigurationCheck.h"
#include "legged_common_mpc_app/robot/RobotController.h"
#include "legged_common_mpc_app/robot/RobotProcess.h"
#include "legged_common_mpc_app/robot/RobotProcessSettings.h"
#include "legged_common_mpc_app/robot/RobotSetUpSteps.h"
#include "legged_common_mpc_app/robot/RobotStack.h"
#include "legged_common_mpc_app/robot/RobotStartup.h"
#include "legged_common_mpc_app/robot/config/robot/RobotProcessSettingsFromConfig.h"
#include "legged_mpc_ipc/MpcMessageConversions.h"
#include "legged_mpc_ipc/RemoteMpcLink.h"
#include "legged_wb_mpc/WBMpcInterface.h"
#include "legged_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "legged_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_core/Types.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * legged_wb_mpc_robot: the robot process of the whole-body MPC
 * (legged_nmpc/docs/distributed_runtime/README.md, "Processes"). The realtime
 * loop over the backend --backend names (mujoco: the MuJoCo simulator) and the
 * whole-body MRT joint controller, which reaches the MPC node over the bus
 * (RemoteMpcLink), in simulation as on the robot. The MPC never runs in this
 * process. See legged_nmpc/legged_common_mpc_app/robot/README.md.
 */

namespace ocs2::legged {
namespace {

/** The control periods between two checks of the joint_pd_gains.textproto file
 * watcher: ~1 Hz at ~500 Hz. */
constexpr size_t kPdGainsFileCheckInterval = 500;

/**
 * The models the controller needs, nothing more: no reference manager and no
 * optimal control problem, so nothing is taped or loaded with CppAD (the MPC is
 * the MPC node's). What the set-up builds, and what a save is checked with.
 */
absl::StatusOr<std::unique_ptr<WBMpcInterface>> controllerModelsOf(
    const RobotConfigFiles& files, absl::string_view urdfFile) {
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> interface =
      WBMpcInterface::CreateControllerModels(files.task, urdfFile,
                                             files.reference);
  if (!interface.ok()) {
    return withConfigFile(interface.status(), files.taskSource);
  }
  return interface;
}

/** The whole-body controller computes its feedforward with the inverse dynamics
 * only. */
absl::Status checkFeedforward(const RobotProcessSettings& settings,
                              absl::string_view taskSource) {
  if (settings.wbMpcFeedforward == WbMpcFeedforward::kInverseDynamics) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat(taskSource, ": wb_mpc_feedforward: \"",
                   wbMpcFeedforwardName(settings.wbMpcFeedforward),
                   "\"; the whole-body controller computes its feedforward "
                   "with the inverse dynamics only."));
}

/** The whole-body MPC's check of saved files: its controller models and its
 * feedforward. */
absl::Status checkWholeBodyFiles(const RobotConfigFiles& files,
                                 absl::string_view urdfFile) {
  RETURN_IF_ERROR(controllerModelsOf(files, urdfFile).status());
  ASSIGN_OR_RETURN(const RobotProcessSettings settings,
                   robotProcessSettingsFromConfig(files.task));
  return checkFeedforward(settings, files.taskSource);
}

/** The whole-body MRT joint controller over the MPC link `linkFactory`, with
 * the controller settings of the task file. */
absl::StatusOr<std::unique_ptr<RobotController>> createController(
    const robot::model::RobotDescription& robotDescription,
    WBMpcInterface& models,
    const MpcLinkFactory& linkFactory,
    absl::string_view pdGainsFile,
    const RobotProcessSettings& settings) {
  // The whole-body controller keeps a reference to the model settings: the
  // interface outlives it.
  const ModelSettings& modelSettings = models.modelSettings();
  // LINT.IfChange(whole_body_robot_controller)
  absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> createdController =
      WBMpcMrtJointController::Create(
          robotDescription, modelSettings, linkFactory,
          models.getPinocchioInterface(), pdGainsFile);
  if (!createdController.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("the whole-body MRT joint controller did not start: ",
                     createdController.status().message()));
  }
  std::unique_ptr<WBMpcMrtJointController> jointController =
      *std::move(createdController);
  // The controller settings of the task file.
  if (settings.contactWrenchGate.has_value()) {
    jointController->setContactWrenchGateConfig(*settings.contactWrenchGate);
  }
  // SAFETY: time constant of the exponential decay applied to the joint PD
  // gains once the mode is entered (the controller's default when the task file
  // has none).
  if (settings.safetyDecayTimeConstant.has_value()) {
    jointController->setSafetyDecayTimeConstant(
        *settings.safetyDecayTimeConstant);
  }
  LOG(INFO) << "MPC MRT joint controller is set up with PD gains from: "
            << pdGainsFile;
  // The whole-body sim handed the controller its posture before its mode.
  return std::make_unique<MrtRobotController<WBMpcMrtJointController>>(
      std::move(jointController), CycleInputOrder::kPostureThenMode);
  // LINT.ThenChange(//legged_nmpc/legged_mpc_validation/src/closed_loop/WholeBodyClosedLoopDriver.cpp:whole_body_robot_controller)
}

/**
 * Everything the robot process of the files of `directory` needs, built and
 * checked, up to its start (runRobot()). The checks a save is held to run
 * before the bus binds its port, so that a configuration the robot refuses
 * fails here, where runRobot() can fall back to the bundled files.
 */
absl::StatusOr<RobotStack> setUpRobot(const RobotAppOptions& options,
                                      const RobotConfigDirectory& directory) {
  const RobotConfigDirectory::Files& files = directory.files();
  const std::string& urdfFile = options.files.urdfFile;
  ASSIGN_OR_RETURN(RobotConfiguration configuration,
                   loadRobotConfiguration(directory));
  const RobotProcessSettings& settings = configuration.settings;

  RobotStack stack;
  ASSIGN_OR_RETURN(std::unique_ptr<WBMpcInterface> interface,
                   controllerModelsOf(configuration.files, urdfFile));
  RETURN_IF_ERROR(checkFeedforward(settings, files.taskFile));
  WBMpcInterface& models = *interface;
  stack.interface = std::move(interface);
  const ModelSettings& modelSettings = models.modelSettings();

  // Init Sim state
  ASSIGN_OR_RETURN(robot::model::RobotDescription description,
                   robot::model::RobotDescription::Create(urdfFile));
  stack.robotDescription =
      std::make_unique<robot::model::RobotDescription>(std::move(description));
  const robot::model::RobotDescription& robotDescription =
      *stack.robotDescription;
  // A task file naming an MPC joint the URDF does not have is refused here,
  // before anything indexes the joints by it.
  ASSIGN_OR_RETURN(
      const std::vector<robot::joint_index_t> mpcJointIndices,
      robotDescription.findJointIndices(modelSettings.mpcModelJointNames));
  const robot::model::RobotState initState = createInitialSimState(
      robotDescription, modelSettings, models.getMpcRobotModel(),
      models.getInitialState());
  LOG(INFO) << "initState: "
            << initState.getPosBaseRtWorldEwrtWorld().transpose();

  // [Hz] The robot process's control rate.
  const scalar_t controlFrequency =
      models.mpcSettings().mrtDesiredFrequency_ > 0.0
          ? models.mpcSettings().mrtDesiredFrequency_
          : kMrtFallbackControlFrequency;
  // LINT.IfChange(robot_backend_options)
  RobotBackendOptions backendOptions;
  backendOptions.robotName = options.robotName;
  backendOptions.urdfFile = urdfFile;
  backendOptions.mjcfFile = options.mjcfFile;
  backendOptions.initialState.emplace(initState);
  backendOptions.contactFrameNames = modelSettings.contactNames;
  backendOptions.contactParentJointNames =
      modelSettings.contactParentJointNamesInContactOrder();
  backendOptions.numPointContacts = modelSettings.contactNames3DoF.size();
  backendOptions.simulator = settings.simulator;
  backendOptions.controlPeriod = 1.0 / controlFrequency;
  backendOptions.viewer = options.viewer;
  // clang-format off
  // LINT.ThenChange(//legged_nmpc/legged_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:robot_backend_options, //legged_nmpc/legged_centroidal_mpc_app/src/CentroidalMpcRobotMain.cpp:robot_backend_options)
  // clang-format on
  ASSIGN_OR_RETURN(stack.backend, RobotBackendRegistry().create(
                                      options.backend, backendOptions));
  stack.contactEstimators =
      std::make_unique<robot::model::ContactEstimatorRegistry>();
  stack.backend->registerContactEstimators(*stack.contactEstimators);

  RobotConfigurationCheckContext context = robotConfigurationCheckContext(
      options, configuration.bundledRobotName, modelSettings, backendOptions,
      stack.contactEstimators.get());
  RETURN_IF_ERROR(checkRobotConfiguration(configuration.files, context));
  // A save is checked with the models of the saved files too; the set-up built
  // them above.
  context.checkFormulation = [urdfFile](const RobotConfigFiles& saved) {
    return checkWholeBodyFiles(saved, urdfFile);
  };

  // The bus first: it outlives everything registered on it.
  ASSIGN_OR_RETURN(stack.bus,
                   node::createNodeBus(options.networkConfig, options.ipcNode));

  // The MPC link, on the state's manifold, on which the link decodes the node's
  // manifold-linear policies and interpolates their states.
  // LINT.IfChange(robot_mpc_link)
  const WBAccelMpcRobotModel<scalar_t>& mpcRobotModel =
      models.getMpcRobotModel();
  const ipc::ModelDimensions dimensions{
      .stateDim = mpcRobotModel.getStateDim(),
      .inputDim = mpcRobotModel.getInputDim(),
      .numModes = numModeNumbersOf(modelSettings.numContacts())};
  ASSIGN_OR_RETURN(std::shared_ptr<const StateManifold> stateManifold,
                   stateManifoldOf(mpcRobotModel));
  ipc::RemoteMpcLink::Config linkConfig;
  linkConfig.dimensions = dimensions;
  linkConfig.stateManifold = std::move(stateManifold);
  linkConfig.policyTimeout = settings.mpcLinkPolicyTimeout;
  // A policy of another layout (equal dimensions, other joints) is refused, as
  // one of other dimensions.
  linkConfig.modelFingerprint = modelLayoutFingerprint(modelSettings);
  ASSIGN_OR_RETURN(std::unique_ptr<RemoteMpcLinkAdapter> link,
                   RemoteMpcLinkAdapter::Create(stack.bus.get(), linkConfig));
  RemoteMpcLinkAdapter* absl_nonnull const remoteLink = link.get();
  const MpcLinkFactory linkFactory = handOverMpcLink(std::move(link));
  // LINT.ThenChange(//legged_nmpc/legged_centroidal_mpc_app/src/CentroidalMpcRobotMain.cpp:robot_mpc_link)
  ASSIGN_OR_RETURN(stack.controller,
                   createController(robotDescription, models, linkFactory,
                                    files.pdGainsFile, settings));

  ASSIGN_OR_RETURN(
      RobotProcess::Config config,
      // The whole-body MPC tracks no multi-contact plan (WBMpcInterface
      // refuses contact_schedule_source: "multi_contact_plan"): its contacts
      // stand on the flat ground.
      robotProcessConfig(options, directory, configuration, modelSettings,
                         /*plan=*/nullptr));
  config.controlFrequency = controlFrequency;
  config.initialState.emplace(initState);
  // Whether a caught robot is at rest is judged on the joints of the MPC model:
  // the ones JOINT_PD brings to the nominal posture and the MPC starts from.
  config.restJointIndices = mpcJointIndices;
  config.pdGainsFileCheckInterval = kPdGainsFileCheckInterval;

  ASSIGN_OR_RETURN(
      stack.process,
      RobotProcess::Create(
          stack.bus.get(), stack.backend.get(), stack.controller.get(),
          stack.contactEstimators.get(), std::move(config),
          robotProcessHooks(remoteLink, files, std::move(context))));
  return stack;
}

absl::Status run() {
  ASSIGN_OR_RETURN(const RobotAppOptions options,
                   robotAppOptionsFromFlags(getDefaultCoreAllocation()));
  exportSoftwareRasterizerThreads(
      options.viewer);  // before the process starts a thread
  return runRobot(
      options,
      [&options](const RobotConfigDirectory& directory) {
        return setUpRobot(options, directory);
      },
      &node::shutdownRequested);
}

}  // namespace
}  // namespace ocs2::legged

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::SetProgramUsageMessage(
      "The robot process of the whole-body MPC: the realtime loop over a robot "
      "backend and the MRT joint controller, with "
      "its MPC behind the bus. legged_wb_mpc_robot --robot_name=unitree_g1 "
      "--task_file=... --reference_file=... "
      "--urdf_file=... --mjcf_file=... [--network_config=...] "
      "[--backend=mujoco] [--realtime_priority=80]");
  absl::ParseCommandLine(argc, argv);
  // Route Abseil log records to stderr, INFO included.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  ocs2::legged::node::installShutdownSignalHandlers();
  const absl::Status status = ocs2::legged::run();
  if (!status.ok()) {
    LOG(ERROR) << "legged_wb_mpc_robot: " << status;
    return 1;
  }
  return 0;
}
