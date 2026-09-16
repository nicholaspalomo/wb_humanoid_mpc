/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
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

#include <ocs2_sqp/SqpMpc.h>
#include <cmath>
#include <fstream>
#include <rclcpp/rclcpp.hpp>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "absl/log/check.h"

#include <absl/log/log.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include <humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h>
#include <humanoid_common_mpc/common/ThreadAffinity.h>
#include <humanoid_common_mpc/contact/ContactRectangle.h>
#include <humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h>
#include "humanoid_common_mpc_ros2/fsm/SimFsmBridge.h"
#include "humanoid_common_mpc_ros2/ros_comm/Ros2ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc_ros2/telemetry/PinocchioTelemetryPublisher.h"
#include "humanoid_common_mpc_ros2/visualization/HumanoidVisualizer.h"

using namespace ocs2;
using namespace ocs2::humanoid;

int main(int argc, char** argv) {
  std::set_terminate([]() {
    std::exception_ptr ex = std::current_exception();
    if (ex) {
      try {
        std::rethrow_exception(ex);
      } catch (const std::exception& e) {
        std::cerr << "\nFATAL: Unhandled exception in CentroidalMpcRobotSim: " << e.what() << std::endl;
      } catch (...) {
        std::cerr << "\nFATAL: Unknown unhandled exception in CentroidalMpcRobotSim." << std::endl;
      }
    } else {
      std::cerr << "\nFATAL: std::terminate called without active exception in CentroidalMpcRobotSim." << std::endl;
    }
    std::abort();
  });

  std::vector<std::string> programArgs;
  programArgs = rclcpp::remove_ros_arguments(argc, argv);
  if (programArgs.size() < 6) {
    throw std::runtime_error("No robot name, config folder, target command file, or description name specified. Aborting.");
  }

  const std::string robotName(argv[1]);
  const std::string taskFile(argv[2]);
  const std::string referenceFile(argv[3]);
  const std::string urdfFile(argv[4]);
  const std::string gaitFile(argv[5]);
  const std::string mjxFile(argv[6]);

  rclcpp::init(argc, argv);

  // Robot interface
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> create_result = CentroidalMpcInterface::Create(taskFile, urdfFile, referenceFile);
  CHECK(create_result.ok()) << "Failed to create CentroidalMpcInterface: " << create_result.status();
  CentroidalMpcInterface& interface = **create_result;

  // MPC
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(), interface.getInitializer());

  // Launch MPC ROS node
  rclcpp::Node::SharedPtr nodeHandle = std::make_shared<rclcpp::Node>(robotName + "_centroidal_mpc");

  auto qos = rclcpp::QoS(1);
  qos.best_effort();

  // Everything that produces or consumes OCP input vectors (reference inputs, policy visualization, telemetry) must use
  // the effective model: in basis-vector mode its input layout is [λ, joint velocities], not the 6D-wrench layout.
  const MpcRobotModelBase<scalar_t>& effectiveMpcRobotModel = interface.getEffectiveMpcRobotModel();

  std::shared_ptr<HumanoidVisualizer> humanoidVisualizer(
      new HumanoidVisualizer(taskFile, interface.getPinocchioInterface(), effectiveMpcRobotModel, nodeHandle));

  // Reference and motion management for Procedural MPC
  CentroidalMpcTargetTrajectoriesCalculator mpcTargetTrajectoriesCalculator(
      referenceFile, effectiveMpcRobotModel, interface.getPinocchioInterface(), interface.getCentroidalModelInfo(),
      interface.mpcSettings().timeHorizon_);
  ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajectoriesFunc =
      [&mpcTargetTrajectoriesCalculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t finalTime,
                                         const vector_t& initState) mutable {
        return mpcTargetTrajectoriesCalculator.commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      };
  auto ros2ProceduralMpcMotionManager = std::make_shared<Ros2ProceduralMpcMotionManager>(
      gaitFile, referenceFile, interface.getSwitchedModelReferenceManagerPtr(), effectiveMpcRobotModel, targetTrajectoriesFunc);

  ros2ProceduralMpcMotionManager->subscribe(nodeHandle, qos);

  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  mpc.getSolverPtr()->addSynchronizedModule(ros2ProceduralMpcMotionManager);
  // Online contact planning (useContactPlanning: true in task.yaml): the planner module feeds mode schedules and footholds
  // to the reference manager and, like the other synchronized modules, has to run before every solve.
  if (auto contactPlannerModule = interface.getContactPlannerModulePtr()) {
    mpc.getSolverPtr()->addSynchronizedModule(contactPlannerModule);
  }

  // Register real-time MPC parameter hot-reloading. The updater is sized to the OCP input and, in basis-vector mode,
  // transforms the wrench-space R of task.yaml exactly as the OCP factory did.
  auto mpcParameterUpdater = std::make_shared<MpcParameterUpdaterModule>(
      &mpc, taskFile, urdfFile, referenceFile, interface.getMpcRobotModel().getStateDim(), effectiveMpcRobotModel.getInputDim(),
      interface.modelSettings().contactNames, dynamic_cast<const SwitchedModelReferenceManager*>(interface.getReferenceManagerPtr().get()),
      interface.getBasisInputsCostTransformConfig());
  mpcParameterUpdater->setContactPlannerModule(interface.getContactPlannerModulePtr());
  mpcParameterUpdater->subscribe(nodeHandle);
  mpc.getSolverPtr()->addSynchronizedModule(mpcParameterUpdater);

  // Init Sim state
  robot::model::RobotDescription robotDescription(urdfFile);
  robot::model::RobotState initState =
      createInitialSimState(robotDescription, interface.modelSettings(), interface.getMpcRobotModel(), interface.getInitialState());

  LOG(INFO) << "initState: " << initState.getRootPositionInWorldFrame().transpose();

  SimFsmBridge fsmBridge(robotDescription, initState, nodeHandle);

  // Ground-truth contact detection in MuJoCo. It drives the viewer's contact timeline ('b' toggles it) and, with
  // simReportsGroundTruthContacts, the contact flags the MPC receives as its measured contact state; without it every
  // contact point is reported as touching.
  bool simReportsGroundTruthContacts = false;
  double simContactForceThreshold = 5.0;
  double simContactTimelineWindow = 5.0;
  // Viewer visualizations by name (VisualizationRegistry.h); absent: the viewer's default set.
  std::vector<std::string> simVisualizations = robot::mujoco_sim_interface::defaultVisualizationNames();
  try {
    YAML::Node taskYaml = YAML::LoadFile(taskFile);
    if (taskYaml["simReportsGroundTruthContacts"]) simReportsGroundTruthContacts = taskYaml["simReportsGroundTruthContacts"].as<bool>();
    if (taskYaml["simContactForceThreshold"]) simContactForceThreshold = taskYaml["simContactForceThreshold"].as<double>();
    if (taskYaml["simContactTimelineWindow"]) simContactTimelineWindow = taskYaml["simContactTimelineWindow"].as<double>();
    if (taskYaml["simVisualizations"]) simVisualizations = taskYaml["simVisualizations"].as<std::vector<std::string>>();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to read the simulator contact settings from " << taskFile << ": " << e.what();
  }

  robot::mujoco_sim_interface::MujocoSimConfig config;

  config.scenePath = mjxFile;
  config.verbose = true;
  config.initStatePtr_ = std::make_shared<robot::model::RobotState>(std::move(initState));
  config.contactFrameNames = interface.modelSettings().contactNames;
  config.contactParentJointNames = interface.modelSettings().contactParentJointNames;
  config.contactForceThreshold = simContactForceThreshold;
  config.contactTimelineWindow = simContactTimelineWindow;
  config.reportGroundTruthContacts = simReportsGroundTruthContacts;
  config.visualizations = simVisualizations;
  // Target contact patches in the viewer ('g' toggles them): the contact rectangle of every foot, drawn at the pose the
  // contact planner wants the foot on the ground. Without a contact planner there is no target and nothing is drawn.
  const auto planningReferenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface.getReferenceManagerPtr());
  if (planningReferenceManager) {
    for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
      robot::mujoco_sim_interface::ContactPatchCorners corners;
      try {
        const ContactRectangle rectangle =
            ContactRectangle::loadContactRectangle(taskFile, interface.modelSettings(), static_cast<int>(contact), false);
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
      config.contactPatchCorners.push_back(std::move(corners));
    }
  }

  robot::mujoco_sim_interface::MujocoSimInterface robotInterface(config, urdfFile);

  if (auto plannerModule = interface.getContactPlannerModulePtr();
      plannerModule && plannerModule->getConfig().formulation.hasExecutionRule(term::kPhaseResetting) && !simReportsGroundTruthContacts) {
    LOG(WARNING) << "contact_planning lists the phase_resetting execution rule but simReportsGroundTruthContacts is off: the simulator "
                    "reports every "
                    "contact point as touching, so phase resetting would end every swing at its scuffing window. Set "
                    "simReportsGroundTruthContacts: true in "
                 << taskFile << ".";
  }

  std::filesystem::path configDir = std::filesystem::path(taskFile).parent_path().parent_path();
  std::string pdGainsFile = (configDir / "controller" / "joint_pd_gains.yaml").string();

  CentroidalMpcMrtJointController mpcJointController(
      robotInterface.getRobotDescription(), interface.modelSettings(), interface.getMpcRobotModel(), mpc, interface.getPinocchioInterface(),
      interface.mpcSettings().mpcDesiredFrequency_, humanoidVisualizer, pdGainsFile, &interface.getEffectiveMpcRobotModel());
  mpcJointController.subscribePdGains(nodeHandle);
  fsmBridge.subscribeJointTargets(nodeHandle);

  // Read gravity-comp feedforward fallback flag from task.yaml
  // Set `useGravityCompFeedforward: true` in task.yaml to use pure gravity comp
  // instead of full inverse dynamics in WB_MPC mode (for debugging ID issues).
  try {
    YAML::Node taskYaml = YAML::LoadFile(taskFile);
    if (taskYaml["useGravityCompFeedforward"] && taskYaml["useGravityCompFeedforward"].as<bool>()) {
      mpcJointController.setUseGravityCompFeedforward(true);
      LOG(INFO) << "Using gravity-comp feedforward in WB_MPC mode (useGravityCompFeedforward=true).";
    }
    // Hand-over into WB_MPC: hold the previous mode until a post-reset policy is active, then ramp over this duration.
    if (taskYaml["mpcEntryBlendTime"]) {
      mpcJointController.setMpcEntryBlendTime(taskYaml["mpcEntryBlendTime"].as<double>());
      LOG(INFO) << "WB_MPC entry blend time: " << mpcJointController.getMpcEntryBlendTime() << " s (mpcEntryBlendTime).";
    }
  } catch (...) {
  }

  bool enableTelemetry = true;
  std::vector<std::string> telemetryFrames;
  const scalar_t mrtDesiredFrequency = interface.mpcSettings().mrtDesiredFrequency_;
  double telemetryFrequency = std::min(100.0, static_cast<double>(mrtDesiredFrequency > 0.0 ? mrtDesiredFrequency : 100.0));
  try {
    YAML::Node taskYaml = YAML::LoadFile(taskFile);
    if (taskYaml["enableTelemetry"]) {
      enableTelemetry = taskYaml["enableTelemetry"].as<bool>();
    } else if (taskYaml["enable_telemetry"]) {
      enableTelemetry = taskYaml["enable_telemetry"].as<bool>();
    }
    if (taskYaml["telemetryFrequency"]) {
      telemetryFrequency = taskYaml["telemetryFrequency"].as<double>();
    } else if (taskYaml["telemetry_frequency"]) {
      telemetryFrequency = taskYaml["telemetry_frequency"].as<double>();
    }
    if (taskYaml["telemetryFrames"]) {
      telemetryFrames = taskYaml["telemetryFrames"].as<std::vector<std::string>>();
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to read telemetry config from " << taskFile << ": " << e.what();
  }

  const size_t mrtDeltaTMicroSeconds_ = 1000000 / static_cast<size_t>(mrtDesiredFrequency > 0.0 ? mrtDesiredFrequency : 100.0);
  const size_t telemetryDecimation = (telemetryFrequency > 0.0 && mrtDesiredFrequency > 0.0)
                                         ? std::max<size_t>(1, static_cast<size_t>(std::round(mrtDesiredFrequency / telemetryFrequency)))
                                         : 1;

  std::unique_ptr<PinocchioTelemetryPublisher> telemetryPublisher;
  if (enableTelemetry) {
    telemetryPublisher =
        std::make_unique<PinocchioTelemetryPublisher>(nodeHandle, interface.getPinocchioInterface(), interface.modelSettings(),
                                                      effectiveMpcRobotModel, robotDescription, telemetryFrames);
    LOG(INFO) << "Pinocchio telemetry publishing enabled (" << (mrtDesiredFrequency / telemetryDecimation)
              << " Hz, decimation=" << telemetryDecimation << ").";
  } else {
    LOG(INFO) << "Telemetry publishing disabled in task.yaml.";
  }

  LOG(INFO) << "MPC MRT joint controller is set up with PD gains from: " << pdGainsFile;
  LOG(INFO) << "MRT joint control loop configured at " << mrtDesiredFrequency << " Hz (" << mrtDeltaTMicroSeconds_ << " us).";

  robotInterface.initSim();
  robotInterface.updateInterfaceStateFromRobot();
  mpcJointController.startMpcThread(robotInterface.getRobotState());

  while (!mpcJointController.ready()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  LOG(INFO) << "Initial MPC policy received.";

  // Wait to allow MPC policy to initialize
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Start sim loop in zero-torque mode: the robot spawns passively held by the gantry.
  // The MPC solver continues to receive state feedback and refine its policy.
  const auto coreAlloc = ocs2::humanoid::getDefaultCoreAllocation();
  robotInterface.startSim();
  ocs2::humanoid::setThreadCpuAffinity(coreAlloc.simCores, robotInterface.getSimulationThread().native_handle(), "MuJoCo Simulation");
  ocs2::humanoid::setThreadCpuAffinity(coreAlloc.mrtCores, pthread_self(), "MRT Joint Control Loop");

  rclcpp::spin_some(nodeHandle);
  LOG(INFO) << "Zero-torque mode: robot spawned. Waiting for FSM command to enable torques...";

  // Unified control loop: processes /humanoid/fsm_command ROS 2 topics for mode transitions.
  std::string currentModeName = "ZERO_TORQUE";
  size_t mrtSlowCount = 0;
  size_t telemetryCounter = 0;
  while (true) {
    auto targetTimeForNextIteration = std::chrono::steady_clock::now() + std::chrono::microseconds(mrtDeltaTMicroSeconds_);

    // Always publish state to MPC so the solver's plan stays current.
    // In zero-torque mode, we still compute the control action but don't apply it,
    // keeping the MPC solver warm for instant transitions back to active mode.
    robotInterface.updateInterfaceStateFromRobot();

    // Propagate FSM mode to the controller so it can handle JOINT_PD with gravity comp internally.
    mpcJointController.setControlMode(currentModeName);
    fsmBridge.applyJointTargetUpdates();
    mpcJointController.setNominalJointPositions(fsmBridge.getNominalJointPositions());
    mpcJointController.computeJointControlAction(0.0, robotInterface.getRobotState(), robotInterface.getRobotJointAction());

    // Contact timeline in the MuJoCo viewer: the contact state the executed policy plans for now, against the physics.
    if (const auto planned = mpcJointController.getPlannedContactFlags(mpcJointController.getCurrentObservation().time)) {
      robotInterface.setTargetContactFlags(std::vector<bool>(planned->begin(), planned->end()));
    } else {
      robotInterface.setTargetContactFlags({});
    }

    // Target contact patches in the MuJoCo viewer: where the contact planner wants each foot next (or where it holds it).
    if (planningReferenceManager) {
      const feet_array_t<TargetContactPose> poses = planningReferenceManager->getTargetContactPoses();
      std::vector<robot::mujoco_sim_interface::TargetContactPatch> patches(N_CONTACTS);
      for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
        const TargetContactPose& pose = poses[contact];
        robot::mujoco_sim_interface::TargetContactPatch& patch = patches[contact];
        patch.valid = pose.valid && pose.position.allFinite() && std::isfinite(pose.height) && std::isfinite(pose.yaw);
        switch (pose.kind) {
          case TargetContactPose::Kind::SWING_IN_FLIGHT:
            patch.kind = robot::mujoco_sim_interface::TargetContactPatch::Kind::SWING_IN_FLIGHT;
            break;
          case TargetContactPose::Kind::NEXT_SWING:
            patch.kind = robot::mujoco_sim_interface::TargetContactPatch::Kind::NEXT_SWING;
            break;
          case TargetContactPose::Kind::STANCE:
          default:
            patch.kind = robot::mujoco_sim_interface::TargetContactPatch::Kind::STANCE;
            break;
        }
        patch.x = pose.position(0);
        patch.y = pose.position(1);
        patch.z = pose.height;
        patch.yaw = pose.yaw;
        patch.yawPlanned = pose.yawPlanned;
      }
      robotInterface.setTargetContactPatches(patches);
    }

    WalkingVelocityCommand targetCmd = ros2ProceduralMpcMotionManager->getScaledWalkingVelocityCommand();
    robotInterface.setTargetVelocities(targetCmd.linear_velocity_x, targetCmd.linear_velocity_y, targetCmd.angular_velocity_z);

    // Apply mode-specific overrides for modes other than JOINT_PD (which is handled by the controller)
    fsmBridge.applyModeAction(currentModeName, robotDescription, robotInterface.getRobotJointAction());

    if (!robotInterface.isZeroTorqueMode()) {
      robotInterface.applyJointAction();
    }

    // Publish telemetry for PlotJuggler visualization at configured rate
    if (telemetryPublisher && (++telemetryCounter % telemetryDecimation == 0)) {
      try {
        telemetryPublisher->publish(robotInterface.getRobotState(), robotInterface.getRobotJointAction(),
                                    mpcJointController.getCurrentObservation(), mpcJointController.getLatestPolicyInput(),
                                    mpcJointController.getCommandData(), robotInterface.getLeftFootMeasuredForce(),
                                    robotInterface.getRightFootMeasuredForce());
      } catch (const std::exception& e) {
        LOG_EVERY_N(ERROR, 100) << "Telemetry publish exception caught in sim loop: " << e.what();
      } catch (...) {
        LOG_EVERY_N(ERROR, 100) << "Telemetry publish unknown exception caught in sim loop";
      }
    }

    rclcpp::spin_some(nodeHandle);
    bool gantryBefore = robotInterface.isGantryLocked();
    fsmBridge.processCommands(currentModeName, robotInterface);
    bool gantryAfter = robotInterface.isGantryLocked();

    // Reset MPC and switch to JOINT_PD when gantry is locked
    if (gantryBefore != gantryAfter) {
      mpcJointController.requestMpcReset();
      if (gantryAfter) {
        // Gantry locked: switch to safe PD mode (MPC is overconstrained on the gantry)
        currentModeName = "JOINT_PD";
        fsmBridge.publishFsmState(currentModeName, gantryAfter);
        LOG(INFO) << "Gantry locked — switching to JOINT_PD mode and resetting MPC.";
      } else {
        LOG(INFO) << "Gantry unlocked — resetting MPC.";
      }
    }

    auto currentTime = std::chrono::steady_clock::now();
    if (currentTime > targetTimeForNextIteration) {
      // Only warn in MPC-active mode and for significant delays (>1ms).
      // Sub-millisecond overruns are normal OS scheduling jitter.
      if (!robotInterface.isZeroTorqueMode()) {
        auto delay = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - targetTimeForNextIteration).count();
        if (delay > 1000 && (++mrtSlowCount % 20 == 0)) {
          LOG(WARNING) << "MRT loop running slow by " << delay << " microseconds.";
        }
      }
    } else {
      // Sleep in case sim loop is faster than specified
      std::this_thread::sleep_until(targetTimeForNextIteration);
    }
  }

  std::cout << "ende..." << std::endl;

  return 0;
}
