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
#include <fstream>
#include <rclcpp/rclcpp.hpp>

#include <humanoid_wb_mpc/WBMpcInterface.h>
#include <mujoco_sim_interface/CheaterSimContactEstimator.h>
#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include "absl/log/check.h"

#include <absl/log/log.h>
#include <humanoid_common_mpc/common/ThreadAffinity.h>
#include <humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h>
#include <humanoid_wb_mpc/mrt/WBMpcMrtJointController.h>
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "humanoid_common_mpc_ros2/fsm/SimFsmBridge.h"
#include "humanoid_common_mpc_ros2/ros_comm/Ros2ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc_ros2/telemetry/PinocchioTelemetryPublisher.h"
#include "humanoid_common_mpc_ros2/visualization/HumanoidVisualizer.h"

using namespace ocs2;
using namespace ocs2::humanoid;

int main(int argc, char** argv) {
  // Route Abseil log records to stderr. Without InitializeLog() Abseil warns once and writes everything to
  // stderr anyway; with it the default stderr threshold is ERROR, so the INFO records have to be asked for.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  std::set_terminate([]() {
    std::exception_ptr ex = std::current_exception();
    if (ex) {
      try {
        std::rethrow_exception(ex);
      } catch (const std::exception& e) {
        LOG(ERROR) << "\nUnhandled exception in WBMpcRobotSim: " << e.what();
      } catch (...) {
        LOG(ERROR) << "\nUnknown unhandled exception in WBMpcRobotSim.";
      }
    } else {
      LOG(ERROR) << "\nstd::terminate called without active exception in WBMpcRobotSim.";
    }
    std::abort();
  });

  std::vector<std::string> programArgs;
  programArgs = rclcpp::remove_ros_arguments(argc, argv);
  // argv[0] .. argv[6] are dereferenced below, so 7 arguments must be present.
  if (programArgs.size() < 7) {
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
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> create_result = WBMpcInterface::Create(taskFile, urdfFile, referenceFile);
  CHECK(create_result.ok()) << "Failed to create WBMpcInterface: " << create_result.status();
  WBMpcInterface& interface = **create_result;

  // MPC
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(), interface.getInitializer());

  // Launch MPC ROS node
  rclcpp::Node::SharedPtr nodeHandle = std::make_shared<rclcpp::Node>(robotName + "_wb_mpc");

  auto qos = rclcpp::QoS(1);
  qos.best_effort();

  std::shared_ptr<HumanoidVisualizer> humanoidVisualizer(
      new HumanoidVisualizer(taskFile, interface.getPinocchioInterface(), interface.getMpcRobotModel(), nodeHandle));

  // Reference and motion management for Procedural MPC
  WBMpcTargetTrajectoriesCalculator mpcTargetTrajectoriesCalculator(referenceFile, interface.getMpcRobotModel(),
                                                                    interface.mpcSettings().timeHorizon_);
  ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajectoriesFunc =
      [&mpcTargetTrajectoriesCalculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t finalTime,
                                         const vector_t& initState) mutable {
        return mpcTargetTrajectoriesCalculator.commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      };
  auto ros2ProceduralMpcMotionManager = std::make_shared<Ros2ProceduralMpcMotionManager>(
      gaitFile, referenceFile, interface.getSwitchedModelReferenceManagerPtr(), interface.getMpcRobotModel(), targetTrajectoriesFunc);

  ros2ProceduralMpcMotionManager->subscribe(nodeHandle, qos);

  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  mpc.getSolverPtr()->addSynchronizedModule(ros2ProceduralMpcMotionManager);

  // Init Sim state
  robot::model::RobotDescription robotDescription(urdfFile);
  robot::model::RobotState initState =
      createInitialSimState(robotDescription, interface.modelSettings(), interface.getMpcRobotModel(), interface.getInitialState());

  LOG(INFO) << "initState: " << initState.getRootPositionInWorldFrame().transpose();

  SimFsmBridge fsmBridge(robotDescription, initState, nodeHandle);

  // Ground-truth contact detection in MuJoCo: the viewer's contact timeline ('b' toggles it) and the cheater_sim contact
  // estimator, the measured contact state of the controller (task file `contactEstimator`, a name of the
  // ContactEstimatorRegistry: MPC observation mode and the contact wrenches the inverse dynamics projects).
  std::string contactEstimatorName = robot::mujoco_sim_interface::kCheaterSimContactEstimatorName;
  double simContactForceThreshold = 5.0;
  // [rad] tilt of the base past which the robot is caught on the gantry; 0 leaves a fallen robot where it lands.
  double simMaxBaseTiltAngle = 0.0;
  double simContactTimelineWindow = 5.0;
  // Viewer visualizations by name (VisualizationRegistry.h); absent: the viewer's default set.
  std::vector<std::string> simVisualizations = robot::mujoco_sim_interface::defaultVisualizationNames();
  // Which implementation holds the base while the virtual gantry is locked (GantryHold in MujocoSimInterface.h).
  std::string simGantryHold = "weld_constraint";
  try {
    YAML::Node taskYaml = YAML::LoadFile(taskFile);
    if (taskYaml["contactEstimator"]) contactEstimatorName = taskYaml["contactEstimator"].as<std::string>();
    if (taskYaml["simContactForceThreshold"]) simContactForceThreshold = taskYaml["simContactForceThreshold"].as<double>();
    if (taskYaml["simMaxBaseTiltAngle"]) simMaxBaseTiltAngle = taskYaml["simMaxBaseTiltAngle"].as<double>();
    if (taskYaml["simContactTimelineWindow"]) simContactTimelineWindow = taskYaml["simContactTimelineWindow"].as<double>();
    if (taskYaml["simVisualizations"]) simVisualizations = taskYaml["simVisualizations"].as<std::vector<std::string>>();
    if (taskYaml["gantryHold"]) simGantryHold = taskYaml["gantryHold"].as<std::string>();
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to read the simulator contact settings from " << taskFile << ": " << e.what();
  }

  fsmBridge.setMaxBaseTiltAngle(simMaxBaseTiltAngle);

  robot::mujoco_sim_interface::MujocoSimConfig config;

  config.scenePath = mjxFile;
  config.verbose = true;
  config.initStatePtr_ = std::make_shared<robot::model::RobotState>(std::move(initState));
  config.contactFrameNames = interface.modelSettings().contactNames;
  config.contactParentJointNames = interface.modelSettings().contactParentJointNames;
  config.contactForceThreshold = simContactForceThreshold;
  config.contactTimelineWindow = simContactTimelineWindow;
  config.visualizations = simVisualizations;
  config.gantryHold = simGantryHold;

  robot::mujoco_sim_interface::MujocoSimInterface robotInterface(config, urdfFile);

  std::filesystem::path configDir = std::filesystem::path(taskFile).parent_path().parent_path();
  std::string pdGainsFile = (configDir / "controller" / "joint_pd_gains.yaml").string();

  WBMpcMrtJointController mpcJointController(robotInterface.getRobotDescription(), interface.modelSettings(), mpc,
                                             interface.getPinocchioInterface(), interface.mpcSettings().mpcDesiredFrequency_,
                                             humanoidVisualizer, pdGainsFile);
  mpcJointController.subscribePdGains(nodeHandle);
  // Measured contact state of the controller, by name (task file `contactEstimator`; an unknown name is fatal and the
  // message lists the available ones).
  {
    robot::model::ContactEstimatorRegistry contactEstimators;
    robot::mujoco_sim_interface::registerCheaterSimContactEstimator(contactEstimators, robotInterface);
    mpcJointController.setContactEstimator(contactEstimators.create(contactEstimatorName));
  }
  try {
    YAML::Node taskYaml = YAML::LoadFile(taskFile);
    // Touch-down shaping of the planned contact wrenches in the inverse dynamics (contact_wrench_gate, default: instant).
    if (taskYaml["contact_wrench_gate"]) {
      ContactWrenchGate::Config gate;
      const YAML::Node block = taskYaml["contact_wrench_gate"];
      if (block["debounceTime"]) gate.debounceTime = block["debounceTime"].as<double>();
      if (block["rampTime"]) gate.rampTime = block["rampTime"].as<double>();
      mpcJointController.setContactWrenchGateConfig(gate);
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to read the controller settings from " << taskFile << ": " << e.what();
  }
  fsmBridge.subscribeJointTargets(nodeHandle);
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
                                                      interface.getMpcRobotModel(), robotDescription, telemetryFrames);
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
    mpcJointController.computeJointControlAction(0.0, robotInterface.getRobotState(), robotInterface.getRobotJointAction());

    // Contact timeline in the MuJoCo viewer: the contact state the executed policy plans for now, against the physics.
    if (const auto planned = mpcJointController.getPlannedContactFlags(mpcJointController.getCurrentObservation().time)) {
      robotInterface.setTargetContactFlags(std::vector<bool>(planned->begin(), planned->end()));
    } else {
      robotInterface.setTargetContactFlags({});
    }

    WalkingVelocityCommand targetCmd = ros2ProceduralMpcMotionManager->getScaledWalkingVelocityCommand();
    robotInterface.setTargetVelocities(targetCmd.linear_velocity_x, targetCmd.linear_velocity_y, targetCmd.angular_velocity_z);

    // Apply any pending joint target position updates from the GUI
    fsmBridge.applyJointTargetUpdates();

    // Apply mode-specific overrides (e.g. pure nominal position tracking in JOINT_PD mode)
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
        LOG_EVERY_N(ERROR, 100) << "Telemetry publish exception caught in WB sim loop: " << e.what();
      } catch (...) {
        LOG_EVERY_N(ERROR, 100) << "Telemetry publish unknown exception caught in WB sim loop";
      }
    }

    rclcpp::spin_some(nodeHandle);
    const bool gantryBefore = robotInterface.isGantryLocked();
    // A robot that has tipped past the configured tilt is caught on the gantry and put in JOINT_PD. Placed before
    // processCommands() so that the lock is part of the same transition the block below reacts to.
    fsmBridge.recoverFromFall(robotInterface.getRobotState(), robotInterface, currentModeName);
    fsmBridge.processCommands(currentModeName, robotInterface);
    const bool gantryAfter = robotInterface.isGantryLocked();

    // The gantry changing state invalidates the solver's warm start either way: locking pins the floating base the
    // trajectory was solved for, unlocking releases a base the solver still believes is pinned.
    if (gantryBefore != gantryAfter) {
      mpcJointController.requestMpcReset();
      if (gantryAfter) {
        // Locked: the whole-body MPC is overconstrained against a pinned base, so hold the posture instead.
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

  LOG(INFO) << "ende...";

  return 0;
}
