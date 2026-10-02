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

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include <humanoid_common_mpc/common/MpcFormulationConfig.h>
#include <humanoid_common_mpc/common/StatusMacros.h>
#include <humanoid_common_mpc/common/ThreadAffinity.h>
#include <humanoid_common_mpc/contact/ContactRectangle.h>
#include <humanoid_common_mpc/contact_planning/ContactPlanningConfig.h>
#include <humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h>
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
 * The target contact patches the viewer draws, one per contact: the contact rectangle of every foot, drawn at the pose
 * the contact planner wants the foot on the ground. Without a contact planner (contactScheduleSource: gait_schedule)
 * there is no target and nothing is drawn.
 */
std::vector<robot::mujoco_sim_interface::ContactPatchCorners> targetContactPatchCorners(const std::string& taskFile,
                                                                                        const ModelSettings& modelSettings) {
  std::vector<robot::mujoco_sim_interface::ContactPatchCorners> patches;
  const absl::StatusOr<ContactScheduleSource> source = loadContactScheduleSource(taskFile);
  if (!source.ok() || *source != ContactScheduleSource::kContactPlanner) {
    return patches;
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    robot::mujoco_sim_interface::ContactPatchCorners corners;
    try {
      const ContactRectangle rectangle =
          ContactRectangle::loadContactRectangle(taskFile, modelSettings, static_cast<int>(contact), /*verbose=*/false);
      const PolygonBounds& bounds = rectangle.getBounds();
      if (bounds.x_max > bounds.x_min && bounds.y_max > bounds.y_min) {
        for (size_t corner = 0; corner < rectangle.getNumberOfContactPoints(); ++corner) {
          const vector3_t point = rectangle.getContactPointTranslation(static_cast<int>(corner));
          corners.push_back({point(0), point(1)});
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "No contact rectangle for contact point " << contact << " in " << taskFile << ": " << e.what()
                   << "; the viewer draws a generic outline.";
    }
    patches.push_back(std::move(corners));
  }
  return patches;
}

/** The warning of the ROS sim: phase resetting needs a measured contact state that is not the plan's own. */
void warnAboutPhaseResettingWithoutContactSensing(const std::string& taskFile, const std::string& contactEstimator) {
  if (robot::model::ContactEstimatorRegistry::canonicalName(contactEstimator) != robot::model::ContactEstimatorRegistry::kAlwaysInContact) {
    return;
  }
  const absl::StatusOr<ContactScheduleSource> source = loadContactScheduleSource(taskFile);
  if (!source.ok() || *source != ContactScheduleSource::kContactPlanner) {
    return;
  }
  const absl::StatusOr<ContactPlanningConfig> config = loadContactPlanningConfigStatus(
      resolveContactPlanningConfigFile(taskFile), "contact_planning.", /*verbose=*/false, /*validate=*/false);
  if (config.ok() && config->formulation.hasExecutionRule(term::kPhaseResetting)) {
    LOG(WARNING) << "contact_planning lists the phase_resetting execution rule but the contact estimator is always_in_contact: every "
                    "contact point reads as touching, so phase resetting would end every swing at its scuffing window. Select "
                    "contactEstimator: cheater_sim in "
                 << taskFile << ".";
  }
}

absl::Status run() {
  const SystemCoreAllocation cores = getDefaultCoreAllocation();
  ASSIGN_OR_RETURN(const RobotAppOptions options, robotAppOptionsFromFlags(cores.mrtCores, cores.simCores));
  const node::MpcFiles& files = options.files;

  // The models the controller needs, nothing more: no reference manager and no optimal control problem, so nothing is
  // taped or loaded with CppAD (the MPC is the MPC node's).
  ASSIGN_OR_RETURN(std::unique_ptr<CentroidalMpcInterface> interface,
                   CentroidalMpcInterface::CreateControllerModels(files.taskFile, files.urdfFile, files.referenceFile));
  ASSIGN_OR_RETURN(const RobotProcessSettings settings, loadRobotProcessSettings(files.taskFile));
  const ModelSettings& modelSettings = interface->modelSettings();
  const MpcRobotModelBase<scalar_t>& effectiveModel = interface->getEffectiveMpcRobotModel();

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
  backendOptions.contactPatchCorners = targetContactPatchCorners(files.taskFile, modelSettings);
  backendOptions.simulator = settings.simulator;
  backendOptions.headless = options.headless;
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/ClosedLoopDriver.cpp:robot_backend_options)
  ASSIGN_OR_RETURN(std::unique_ptr<RobotBackend> backend, RobotBackendRegistry().create(options.backend, backendOptions));
  robot::model::ContactEstimatorRegistry contactEstimators;
  backend->registerContactEstimators(contactEstimators);
  warnAboutPhaseResettingWithoutContactSensing(files.taskFile, settings.contactEstimator);

  // The MPC link.
  const ipc::ModelDimensions dimensions{
      .stateDim = effectiveModel.getStateDim(), .inputDim = effectiveModel.getInputDim(), .numModes = kNumModes};
  ipc::RemoteMpcLink::Config linkConfig;
  linkConfig.dimensions = dimensions;
  linkConfig.policyTimeout = settings.mpcLinkPolicyTimeout;
  ASSIGN_OR_RETURN(std::unique_ptr<RemoteMpcLinkAdapter> link, RemoteMpcLinkAdapter::Create(*bus, std::move(linkConfig)));
  RemoteMpcLinkAdapter* const remoteLink = link.get();
  const MpcLinkFactory linkFactory = handOverMpcLink(std::move(link));

  const std::filesystem::path configDir = std::filesystem::path(files.taskFile).parent_path().parent_path();
  const std::string pdGainsFile = (configDir / "controller" / "joint_pd_gains.yaml").string();
  // LINT.IfChange(centroidal_robot_controller)
  std::unique_ptr<CentroidalMpcMrtJointController> jointController;
  try {
    jointController =
        std::make_unique<CentroidalMpcMrtJointController>(robotDescription, modelSettings, interface->getMpcRobotModel(), linkFactory,
                                                          interface->getPinocchioInterface(), pdGainsFile, &effectiveModel);
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat("the centroidal MRT joint controller did not start: ", error.what()));
  }
  // The controller settings of the task file.
  if (settings.wbMpcFeedforward == WbMpcFeedforward::kGravityCompensation) {
    jointController->setUseGravityCompFeedforward(true);
    LOG(INFO) << "Using gravity-comp feedforward in WB_MPC mode (wbMpcFeedforward: gravity_compensation).";
  }
  if (settings.mpcEntryBlendTime.has_value()) {
    jointController->setMpcEntryBlendTime(*settings.mpcEntryBlendTime);
    LOG(INFO) << "WB_MPC entry blend time: " << jointController->getMpcEntryBlendTime() << " s (mpcEntryBlendTime).";
  }
  if (settings.safetyDecayTimeConstant.has_value()) {
    jointController->setSafetyDecayTimeConstant(*settings.safetyDecayTimeConstant);
  }
  LOG(INFO) << "SAFETY decay time constant: " << jointController->getSafetyDecayTimeConstant() << " s (safetyDecayTimeConstant).";
  if (settings.contactWrenchGate.has_value()) {
    jointController->setContactWrenchGateConfig(*settings.contactWrenchGate);
  }
  LOG(INFO) << "MPC MRT joint controller is set up with PD gains from: " << pdGainsFile;
  MrtRobotController<CentroidalMpcMrtJointController> controller(std::move(jointController), CycleInputOrder::kModeThenPosture);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/src/closed_loop/CentroidalClosedLoopDriver.cpp:centroidal_robot_controller)

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
  // The joint_pd_gains.yaml file watcher, every 100 control periods (~1 Hz at ~100 Hz).
  config.pdGainsFileCheckInterval = 100;

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
