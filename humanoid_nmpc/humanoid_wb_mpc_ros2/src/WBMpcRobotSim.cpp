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

#include <humanoid_wb_mpc/WBMpcInterface.h>
#include <mujoco_sim_interface/MujocoSimInterface.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h>
#include <humanoid_wb_mpc/mrt/WBMpcMrtJointController.h>
#include <absl/log/log.h>
#include "humanoid_common_mpc_ros2/fsm/SimFsmBridge.h"
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
  WBMpcInterface interface(taskFile, urdfFile, referenceFile);

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

  robot::mujoco_sim_interface::MujocoSimConfig config;

  config.scenePath = mjxFile;
  config.verbose = true;
  config.initStatePtr_ = std::make_shared<robot::model::RobotState>(std::move(initState));

  robot::mujoco_sim_interface::MujocoSimInterface robotInterface(config, urdfFile);

  std::filesystem::path configDir = std::filesystem::path(taskFile).parent_path().parent_path();
  std::string pdGainsFile = (configDir / "controller" / "joint_pd_gains.yaml").string();

  WBMpcMrtJointController mpcJointController(robotInterface.getRobotDescription(), interface.modelSettings(), mpc,
                                             interface.getPinocchioInterface(), interface.mpcSettings().mpcDesiredFrequency_,
                                             humanoidVisualizer, pdGainsFile);

  LOG(INFO) << "MPC MRT joint controller is set up with PD gains from: " << pdGainsFile;

  // size_t mrtDeltaTMicroSeconds_ = 1000000 / (interface.mpcSettings().mrtDesiredFrequency_);
  size_t mrtDeltaTMicroSeconds_ = 1000000 / (500);
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
  robotInterface.startSim();

  rclcpp::spin_some(nodeHandle);
  LOG(INFO) << "Zero-torque mode: robot spawned. Waiting for FSM command to enable torques...";

  // Unified control loop: processes /humanoid/fsm_command ROS 2 topics for mode transitions.
  std::string currentModeName = "ZERO_TORQUE";
  size_t mrtSlowCount = 0;
  while (true) {
    auto targetTimeForNextIteration = std::chrono::steady_clock::now() + std::chrono::microseconds(mrtDeltaTMicroSeconds_);

    // Always publish state to MPC so the solver's plan stays current.
    // In zero-torque mode, we still compute the control action but don't apply it,
    // keeping the MPC solver warm for instant transitions back to active mode.
    robotInterface.updateInterfaceStateFromRobot();
    mpcJointController.computeJointControlAction(0.0, robotInterface.getRobotState(), robotInterface.getRobotJointAction());

    // Apply mode-specific overrides (e.g. pure nominal position tracking in JOINT_PD mode)
    fsmBridge.applyModeAction(currentModeName, robotDescription, robotInterface.getRobotJointAction());

    if (!robotInterface.isZeroTorqueMode()) {
      robotInterface.applyJointAction();
    }

    rclcpp::spin_some(nodeHandle);
    fsmBridge.processCommands(currentModeName, robotInterface);

    auto currentTime = std::chrono::steady_clock::now();
    if (currentTime > targetTimeForNextIteration) {
      // Only warn in MPC-active mode and for significant delays (>1ms).
      // Sub-millisecond overruns are normal OS scheduling jitter.
      if (!robotInterface.isZeroTorqueMode()) {
        auto delay = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - targetTimeForNextIteration).count();
        if (delay > 1000 && (++mrtSlowCount % 10 == 0)) {
          LOG(WARNING) << "MRT loop running slow by " << delay << " microseconds (showing 1 in 10).";
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
