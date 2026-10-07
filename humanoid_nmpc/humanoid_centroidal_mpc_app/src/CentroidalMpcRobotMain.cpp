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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/ThreadAffinity.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc_app/node/NodeBus.h"
#include "humanoid_common_mpc_app/node/ShutdownSignal.h"
#include "humanoid_common_mpc_app/robot/InitialSimState.h"
#include "humanoid_common_mpc_app/robot/MrtRobotController.h"
#include "humanoid_common_mpc_app/robot/RemoteMpcLinkAdapter.h"
#include "humanoid_common_mpc_app/robot/RobotAppFlags.h"
#include "humanoid_common_mpc_app/robot/RobotBackendRegistry.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/RobotConfigurationCheck.h"
#include "humanoid_common_mpc_app/robot/RobotController.h"
#include "humanoid_common_mpc_app/robot/RobotProcess.h"
#include "humanoid_common_mpc_app/robot/RobotProcessSettings.h"
#include "humanoid_common_mpc_app/robot/RobotSetUpSteps.h"
#include "humanoid_common_mpc_app/robot/RobotStack.h"
#include "humanoid_common_mpc_app/robot/RobotStartup.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"

/*
 * humanoid_centroidal_mpc_robot: the robot process of the centroidal MPC (humanoid_nmpc/docs/distributed_runtime/
 * README.md, "Processes"). The realtime loop over the backend --backend names (mujoco: the MuJoCo simulator) and the
 * centroidal MRT joint controller, which reaches the MPC node over the bus (RemoteMpcLink), in simulation as on the robot.
 * What the ROS sim CentroidalMpcRobotSim did, without ROS and without the MPC in the process. See
 * humanoid_nmpc/humanoid_common_mpc_app/robot/README.md.
 */

namespace ocs2::humanoid {
namespace {

/** The modes of a humanoid MPC: the contact combinations of its two feet (MotionPhaseDefinition.h). */
constexpr size_t kNumModes = 4;

/**
 * The target contact patches the viewer draws, one per contact: the contact rectangle of every foot (the task file's
 * contacts block of `config`), drawn at the pose the contact planner wants the foot on the ground. Without a contact
 * planner (contact_schedule_source: "gait_schedule") there is no target and nothing is drawn.
 */
std::vector<robot::mujoco_sim_interface::ContactPatchCorners> targetContactPatchCorners(const CentroidalMpcConfig& config,
                                                                                        const ModelSettings& modelSettings) {
  std::vector<robot::mujoco_sim_interface::ContactPatchCorners> patches;
  const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromConfig(config.task);
  if (!source.ok() || *source != ContactScheduleSource::kContactPlanner) {
    return patches;
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    robot::mujoco_sim_interface::ContactPatchCorners corners;
    const absl::StatusOr<ContactRectangle> rectangle =
        contactRectangleFromConfig(config.task.contacts, modelSettings, static_cast<int>(contact));
    if (rectangle.ok()) {
      const PolygonBounds& bounds = rectangle->getBounds();
      if (bounds.x_max > bounds.x_min && bounds.y_max > bounds.y_min) {
        for (size_t corner = 0; corner < rectangle->getNumberOfContactPoints(); ++corner) {
          const vector3_t point = rectangle->getContactPointTranslation(static_cast<int>(corner));
          corners.push_back({point(0), point(1)});
        }
      }
    } else {
      LOG(WARNING) << "No contact rectangle for contact point " << contact
                   << " in the task file's contacts: " << rectangle.status().message() << "; the viewer draws a generic outline.";
    }
    patches.push_back(std::move(corners));
  }
  return patches;
}

/**
 * The warning of the ROS sim: phase resetting needs a measured contact state that is not the plan's own. `config` is the
 * configuration of the task file at `taskFile`, which the warning names.
 */
void warnAboutPhaseResettingWithoutContactSensing(const CentroidalMpcConfig& config,
                                                  const std::string& taskFile,
                                                  const std::string& contactEstimator) {
  if (robot::model::ContactEstimatorRegistry::canonicalName(contactEstimator) != robot::model::ContactEstimatorRegistry::kAlwaysInContact) {
    return;
  }
  const absl::StatusOr<ContactScheduleSource> source = contactScheduleSourceFromConfig(config.task);
  if (!source.ok() || *source != ContactScheduleSource::kContactPlanner) {
    return;
  }
  const absl::StatusOr<ContactPlanningConfig> planning =
      contactPlanningConfigFromOptionalFile(config.contactPlanning.has_value() ? &*config.contactPlanning : nullptr,
                                            ContactPlanningValidation::kDeferUntilModelParametersApplied);
  if (planning.ok() && planning->formulation.hasExecutionRule(term::kPhaseResetting)) {
    LOG(WARNING) << "contact_planning.textproto lists the phase_resetting execution rule but the contact estimator is always_in_contact: "
                    "every contact point reads as touching, so phase resetting would end every swing at its scuffing window. Select "
                    "contact_estimator: \"cheater_sim\" in "
                 << taskFile << ".";
  }
}

/** The centroidal MPC's configuration of the parsed files. */
CentroidalMpcConfig centroidalMpcConfigOf(const RobotConfigFiles& files) {
  CentroidalMpcConfig config;
  config.task = files.task;
  config.reference = files.reference;
  config.contactPlanning = files.contactPlanning;
  return config;
}

/**
 * The models the controller needs, nothing more: no reference manager and no optimal control problem, so nothing is
 * taped or loaded with CppAD (the MPC is the MPC node's). What the set-up builds, and what a save is checked with.
 */
absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> controllerModelsOf(const RobotConfigFiles& files, const std::string& urdfFile) {
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::CreateControllerModels(centroidalMpcConfigOf(files), urdfFile);
  if (!interface.ok()) return withConfigFile(interface.status(), files.taskSource);
  return interface;
}

/** The centroidal MRT joint controller over the MPC link `linkFactory`, with the controller settings of the task file. */
absl::StatusOr<std::unique_ptr<RobotController>> createController(const robot::model::RobotDescription& robotDescription,
                                                                  CentroidalMpcInterface& models,
                                                                  const MpcLinkFactory& linkFactory,
                                                                  const std::string& pdGainsFile,
                                                                  const RobotProcessSettings& settings) {
  const ModelSettings& modelSettings = models.modelSettings();
  // LINT.IfChange(centroidal_robot_controller)
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> createdController =
      CentroidalMpcMrtJointController::Create(robotDescription, modelSettings, models.getMpcRobotModel(), linkFactory,
                                              models.getPinocchioInterface(), pdGainsFile, &models.getEffectiveMpcRobotModel());
  if (!createdController.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("the centroidal MRT joint controller did not start: ", createdController.status().message()));
  }
  std::unique_ptr<CentroidalMpcMrtJointController> jointController = *std::move(createdController);
  // The controller settings of the task file.
  if (settings.wbMpcFeedforward == WbMpcFeedforward::kGravityCompensation) {
    jointController->setUseGravityCompFeedforward(true);
    LOG(INFO) << "Using gravity-comp feedforward in WB_MPC mode (wb_mpc_feedforward: \"gravity_compensation\").";
  }
  if (settings.mpcEntryBlendTime.has_value()) {
    jointController->setMpcEntryBlendTime(*settings.mpcEntryBlendTime);
    LOG(INFO) << "WB_MPC entry blend time: " << jointController->getMpcEntryBlendTime() << " s (mpc_entry_blend_time).";
  }
  if (settings.safetyDecayTimeConstant.has_value()) {
    jointController->setSafetyDecayTimeConstant(*settings.safetyDecayTimeConstant);
  }
  LOG(INFO) << "SAFETY decay time constant: " << jointController->getSafetyDecayTimeConstant() << " s (safety_decay_time_constant).";
  if (settings.contactWrenchGate.has_value()) {
    jointController->setContactWrenchGateConfig(*settings.contactWrenchGate);
  }
  LOG(INFO) << "MPC MRT joint controller is set up with PD gains from: " << pdGainsFile;
  return std::make_unique<MrtRobotController<CentroidalMpcMrtJointController>>(std::move(jointController),
                                                                               CycleInputOrder::kModeThenPosture);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/CentroidalClosedLoopDriver.cpp:centroidal_robot_controller)
}

/**
 * Everything the robot process of the files of `directory` needs, built and checked, up to its start (runRobot()). The
 * checks a save is held to run before the bus binds its port, so that a configuration the robot refuses fails here,
 * where runRobot() can fall back to the bundled files.
 */
absl::StatusOr<RobotStack> setUpRobot(const RobotAppOptions& options, const RobotConfigDirectory& directory) {
  const RobotConfigDirectory::Files& files = directory.files();
  const std::string& urdfFile = options.files.urdfFile;
  ASSIGN_OR_RETURN(RobotConfiguration configuration, loadRobotConfiguration(directory));
  const RobotProcessSettings& settings = configuration.settings;

  RobotStack stack;
  ASSIGN_OR_RETURN(std::unique_ptr<CentroidalMpcInterface> interface, controllerModelsOf(configuration.files, urdfFile));
  CentroidalMpcInterface& models = *interface;
  stack.interface = std::move(interface);
  const ModelSettings& modelSettings = models.modelSettings();

  // Init Sim state
  ASSIGN_OR_RETURN(robot::model::RobotDescription description, robot::model::RobotDescription::Create(urdfFile));
  stack.robotDescription = std::make_unique<robot::model::RobotDescription>(std::move(description));
  const robot::model::RobotDescription& robotDescription = *stack.robotDescription;
  // A task file naming an MPC joint the URDF does not have is refused here, before anything indexes the joints by it.
  ASSIGN_OR_RETURN(const std::vector<robot::joint_index_t> mpcJointIndices,
                   robotDescription.findJointIndices(modelSettings.mpcModelJointNames));
  const robot::model::RobotState initState =
      createInitialSimState(robotDescription, modelSettings, models.getMpcRobotModel(), models.getInitialState());
  LOG(INFO) << "initState: " << initState.getRootPositionInWorldFrame().transpose();

  // LINT.IfChange(robot_backend_options)
  RobotBackendOptions backendOptions;
  backendOptions.robotName = options.robotName;
  backendOptions.urdfFile = urdfFile;
  backendOptions.mjcfFile = options.mjcfFile;
  backendOptions.initialState.emplace(initState);
  backendOptions.contactFrameNames = modelSettings.contactNames;
  backendOptions.contactParentJointNames = modelSettings.contactParentJointNames;
  backendOptions.contactPatchCorners = targetContactPatchCorners(models.config(), modelSettings);
  backendOptions.simulator = settings.simulator;
  backendOptions.headless = options.headless;
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:robot_backend_options, //humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcRobotMain.cpp:robot_backend_options)
  // clang-format on
  ASSIGN_OR_RETURN(stack.backend, RobotBackendRegistry().create(options.backend, backendOptions));
  stack.contactEstimators = std::make_unique<robot::model::ContactEstimatorRegistry>();
  stack.backend->registerContactEstimators(*stack.contactEstimators);
  warnAboutPhaseResettingWithoutContactSensing(models.config(), files.taskFile, settings.contactEstimator);

  RobotConfigurationCheckContext context =
      robotConfigurationCheckContext(options, configuration.bundledRobotName, modelSettings,
                                     CentroidalMpcMrtJointController::pdGainsDefaults(), backendOptions, stack.contactEstimators.get());
  RETURN_IF_ERROR(checkRobotConfiguration(configuration.files, context));
  // A save is checked with the models of the saved files too; the set-up built them above.
  context.checkFormulation = [urdfFile](const RobotConfigFiles& saved) { return controllerModelsOf(saved, urdfFile).status(); };

  // The bus first: it outlives everything registered on it.
  ASSIGN_OR_RETURN(stack.bus, node::createNodeBus(options.networkConfig, options.ipcNode));

  // The MPC link, on the effective model (the basis-vector inputs' when the contact inputs are basis vectors).
  // LINT.IfChange(robot_mpc_link)
  const MpcRobotModelBase<scalar_t>& effectiveModel = models.getEffectiveMpcRobotModel();
  const ipc::ModelDimensions dimensions{
      .stateDim = effectiveModel.getStateDim(), .inputDim = effectiveModel.getInputDim(), .numModes = kNumModes};
  ipc::RemoteMpcLink::Config linkConfig;
  linkConfig.dimensions = dimensions;
  linkConfig.policyTimeout = settings.mpcLinkPolicyTimeout;
  ASSIGN_OR_RETURN(std::unique_ptr<RemoteMpcLinkAdapter> link, RemoteMpcLinkAdapter::Create(*stack.bus, linkConfig));
  RemoteMpcLinkAdapter* absl_nonnull const remoteLink = link.get();
  const MpcLinkFactory linkFactory = handOverMpcLink(std::move(link));
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcRobotMain.cpp:robot_mpc_link)
  ASSIGN_OR_RETURN(stack.controller, createController(robotDescription, models, linkFactory, files.pdGainsFile, settings));

  const scalar_t mrtDesiredFrequency = models.mpcSettings().mrtDesiredFrequency_;
  RobotProcess::Config config = robotProcessConfig(options, directory, configuration, modelSettings);
  config.controlFrequency = mrtDesiredFrequency > 0.0 ? mrtDesiredFrequency : 100.0;
  config.initialState.emplace(initState);
  // Whether a caught robot is at rest is judged on the joints of the MPC model: the ones JOINT_PD brings to the nominal
  // posture and the MPC starts from.
  config.restJointIndices = mpcJointIndices;
  // The joint_pd_gains.textproto file watcher, every 100 control periods (~1 Hz at ~100 Hz).
  config.pdGainsFileCheckInterval = 100;

  ASSIGN_OR_RETURN(stack.process, RobotProcess::Create(*stack.bus, *stack.backend, *stack.controller, *stack.contactEstimators,
                                                       std::move(config), robotProcessHooks(remoteLink, files, std::move(context))));
  return stack;
}

absl::Status run() {
  const SystemCoreAllocation cores = getDefaultCoreAllocation();
  ASSIGN_OR_RETURN(const RobotAppOptions options, robotAppOptionsFromFlags(cores.mrtCores, cores.simCores));
  return runRobot(
      options, [&options](const RobotConfigDirectory& directory) { return setUpRobot(options, directory); }, &node::shutdownRequested);
}

}  // namespace
}  // namespace ocs2::humanoid

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::SetProgramUsageMessage(
      "The robot process of the centroidal MPC: the realtime loop over a robot backend and the MRT joint controller, with "
      "its MPC behind the bus. humanoid_centroidal_mpc_robot --robot_name=drc_atlas --task_file=... --reference_file=... "
      "--urdf_file=... --mjcf_file=... [--network_config=...] [--backend=mujoco] [--realtime_priority=80]");
  absl::ParseCommandLine(argc, argv);
  // Route Abseil log records to stderr, INFO included.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  ocs2::humanoid::node::installShutdownSignalHandlers();
  const absl::Status status = ocs2::humanoid::run();
  if (!status.ok()) {
    LOG(ERROR) << "humanoid_centroidal_mpc_robot: " << status;
    return 1;
  }
  return 0;
}
