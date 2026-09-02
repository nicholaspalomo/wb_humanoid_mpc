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

#include <ocs2_sqp/SqpMpc.h>
#include <fstream>
#include <rclcpp/rclcpp.hpp>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include "humanoid_common_mpc_ros2/ros_comm/Ros2ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc_ros2/visualization/HumanoidVisualizer.h"

using namespace ocs2;
using namespace ocs2::humanoid;

int main(int argc, char** argv) {
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
  CentroidalMpcInterface interface(taskFile, urdfFile, referenceFile);

  // MPC
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(), interface.getInitializer());

  // Launch MPC ROS node
  rclcpp::Node::SharedPtr nodeHandle = std::make_shared<rclcpp::Node>(robotName + "_centroidal_mpc");

  auto qos = rclcpp::QoS(1);
  qos.best_effort();

  std::shared_ptr<HumanoidVisualizer> humanoidVisualizer(
      new HumanoidVisualizer(taskFile, interface.getPinocchioInterface(), interface.getMpcRobotModel(), nodeHandle));

  // Reference and motion management for Procedural MPC
  CentroidalMpcTargetTrajectoriesCalculator mpcTargetTrajectoriesCalculator(
      referenceFile, interface.getMpcRobotModel(), interface.getPinocchioInterface(), interface.getCentroidalModelInfo(),
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
  robot::model::RobotState initState(robotDescription, 2);
  initState.setConfigurationToZero();

  const vector_t& initMpcState = interface.getInitialState();
  const auto& mpcModel = interface.getMpcRobotModel();
  initState.setRootPositionInWorldFrame(mpcModel.getBasePosition(initMpcState));
  vector3_t baseOriEulerZyx = mpcModel.getBaseOrientationEulerZYX(initMpcState);
  initState.setRootRotationLocalToWorldFrame(ocs2::getQuaternionFromEulerAnglesZyx(baseOriEulerZyx));

  vector_t mpcJointAngles = mpcModel.getJointAngles(initMpcState);
  std::vector<robot::joint_index_t> mpcJointIndices = robotDescription.getJointIndices(interface.modelSettings().mpcModelJointNames);
  for (size_t i = 0; i < mpcJointIndices.size(); i++) {
    initState.setJointPosition(mpcJointIndices[i], mpcJointAngles[i]);
  }

  std::cerr << "initState: " << initState.getRootPositionInWorldFrame().transpose() << std::endl;

  robot::mujoco_sim_interface::MujocoSimConfig config;

  config.scenePath = mjxFile;
  config.verbose = true;
  config.initStatePtr_ = std::make_shared<robot::model::RobotState>(std::move(initState));

  robot::mujoco_sim_interface::MujocoSimInterface robotInterface(config, urdfFile);

  std::filesystem::path configDir = std::filesystem::path(taskFile).parent_path().parent_path();
  std::string pdGainsFile = (configDir / "controller" / "joint_pd_gains.yaml").string();

  CentroidalMpcMrtJointController mpcJointController(robotInterface.getRobotDescription(), interface.modelSettings(),
                                                     interface.getMpcRobotModel(), mpc, interface.getPinocchioInterface(),
                                                     interface.mpcSettings().mpcDesiredFrequency_, humanoidVisualizer, pdGainsFile);

  std::cout << "MPC MRT joint controller is set up with PD gains from: " << pdGainsFile << std::endl;

  // size_t mrtDeltaTMicroSeconds_ = 1000000 / (interface.mpcSettings().mrtDesiredFrequency_);
  size_t mrtDeltaTMicroSeconds_ = 1000000 / (500);
  robotInterface.initSim();
  robotInterface.updateInterfaceStateFromRobot();
  mpcJointController.startMpcThread(robotInterface.getRobotState());

  while (!mpcJointController.ready()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cout << "Initial MPC policy received. " << std::endl;

  // Wait to allow MPC policy to initialize
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Start sim loop in zero-torque mode: the robot spawns passively held by the gantry.
  // The MPC solver continues to receive state feedback and refine its policy.
  robotInterface.startSim();

  rclcpp::spin_some(nodeHandle);
  std::cout << "Zero-torque mode: robot spawned. Waiting for FSM command to enable torques..." << std::endl;

  // Write initial FSM state file so the notebook/GUI knows the sim is ready
  {
    std::ofstream f("/tmp/humanoid_fsm_state");
    f << "ZERO_TORQUE,GANTRY_LOCKED" << std::endl;
  }

  // Unified control loop: polls /tmp/humanoid_fsm_command for mode transitions.
  // The Python FSM writes "ENABLE_TORQUES" or "DISABLE_TORQUES" to this file.
  size_t pollCounter = 0;
  std::string currentModeName = "ZERO_TORQUE";
  while (true) {
    auto targetTimeForNextIteration = std::chrono::steady_clock::now() + std::chrono::microseconds(mrtDeltaTMicroSeconds_);

    // Always publish state to MPC so the solver's plan stays current.
    // In zero-torque mode, we still compute the control action but don't apply it,
    // keeping the MPC solver warm for instant transitions back to active mode.
    robotInterface.updateInterfaceStateFromRobot();
    mpcJointController.computeJointControlAction(0.0, robotInterface.getRobotState(), robotInterface.getRobotJointAction());
    if (!robotInterface.isZeroTorqueMode()) {
      robotInterface.applyJointAction();
    }

    rclcpp::spin_some(nodeHandle);

    // Poll for FSM commands every ~100ms (every 50 iterations at 500 Hz)
    if (++pollCounter % 50 == 0) {
      std::ifstream cmdFile("/tmp/humanoid_fsm_command");
      if (cmdFile.is_open()) {
        std::string cmd;
        std::getline(cmdFile, cmd);
        cmdFile.close();

        auto writeState = [&]() {
          std::ofstream f("/tmp/humanoid_fsm_state");
          f << currentModeName << "," << (robotInterface.isGantryLocked() ? "GANTRY_LOCKED" : "GANTRY_UNLOCKED") << std::endl;
        };

        // Handle mode commands: ZERO_TORQUE disables torques, all others enable.
        // The mode name is preserved in the state file for the GUI to read back.
        if (cmd == "ZERO_TORQUE" || cmd == "DISABLE_TORQUES") {
          if (!robotInterface.isZeroTorqueMode()) {
            std::cout << "FSM command received: " << cmd << " — zero-torque mode." << std::endl;
            robotInterface.disableTorques();
          }
          currentModeName = "ZERO_TORQUE";
          writeState();
        } else if (cmd == "JOINT_PD" || cmd == "GRAVITY_COMP" || cmd == "WB_MPC" || cmd == "SAFETY" || cmd == "MPC_ACTIVE" ||
                   cmd == "ENABLE_TORQUES") {
          if (robotInterface.isZeroTorqueMode()) {
            std::cout << "FSM command received: " << cmd << " — enabling torques." << std::endl;
            robotInterface.enableTorques();
          }
          // Preserve the actual mode name (map legacy commands to WB_MPC)
          currentModeName = (cmd == "ENABLE_TORQUES" || cmd == "MPC_ACTIVE") ? "WB_MPC" : cmd;
          writeState();
        } else if (cmd == "LOCK_GANTRY" && !robotInterface.isGantryLocked()) {
          std::cout << "FSM command received: Locking gantry." << std::endl;
          robotInterface.lockGantry();
          writeState();
        } else if (cmd == "UNLOCK_GANTRY" && robotInterface.isGantryLocked()) {
          std::cout << "FSM command received: Unlocking gantry." << std::endl;
          robotInterface.unlockGantry();
          writeState();
        }
      }
    }

    auto currentTime = std::chrono::steady_clock::now();
    if (currentTime > targetTimeForNextIteration) {
      // Only warn in MPC-active mode and for significant delays (>1ms).
      // Sub-millisecond overruns are normal OS scheduling jitter.
      if (!robotInterface.isZeroTorqueMode()) {
        auto delay = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - targetTimeForNextIteration).count();
        if (delay > 1000) {
          std::cerr << "Warning: MRT loop running slow by " << delay << " microseconds." << std::endl;
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
