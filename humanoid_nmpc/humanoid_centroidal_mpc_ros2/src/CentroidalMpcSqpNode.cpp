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

#include <rclcpp/rclcpp.hpp>

#include <ocs2_ros2_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_ros2_interfaces/synchronized_module/RosReferenceManager.h>
#include <ocs2_sqp/SqpMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include "absl/log/check.h"

#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include <humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h>
#include <humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h>
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "humanoid_common_mpc_ros2/ros_comm/Ros2ProceduralMpcMotionManager.h"

using namespace ocs2;
using namespace ocs2::humanoid;

int main(int argc, char** argv) {
  // Route Abseil log records to stderr. Without InitializeLog() Abseil warns once and writes everything to
  // stderr anyway; with it the default stderr threshold is ERROR, so the INFO records have to be asked for.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  std::vector<std::string> programArgs;
  programArgs = rclcpp::remove_ros_arguments(argc, argv);
  // argv[0] .. argv[5] are dereferenced below, so 6 arguments must be present.
  if (programArgs.size() < 6) {
    throw std::runtime_error("No robot name, config folder, target command file, or description name specified. Aborting.");
  }

  const std::string robotName(argv[1]);
  const std::string taskFile(argv[2]);
  const std::string referenceFile(argv[3]);
  const std::string urdfFile(argv[4]);
  const std::string gaitFile(argv[5]);

  rclcpp::init(argc, argv);

  // Robot interface
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> create_result = CentroidalMpcInterface::Create(taskFile, urdfFile, referenceFile);
  CHECK(create_result.ok()) << "Failed to create CentroidalMpcInterface: " << create_result.status();
  CentroidalMpcInterface& interface = **create_result;

  // MPC
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(), interface.getInitializer());

  // Launch MPC ROS node
  rclcpp::Node::SharedPtr nodeHandle = std::make_shared<rclcpp::Node>(robotName + "_mpc");

  rclcpp::QoS qos(1);
  qos.best_effort();

  // Reference inputs must be laid out like the OCP input: in basis-vector mode that is [λ, joint velocities], so the
  // effective model (decorator when active) is used rather than the wrench-space model.
  const MpcRobotModelBase<scalar_t>& effectiveMpcRobotModel = interface.getEffectiveMpcRobotModel();

  // Reference and motion management for Procedural MPC
  CentroidalMpcTargetTrajectoriesCalculator mpcTargetTrajectoriesCalculator(
      referenceFile, effectiveMpcRobotModel, interface.getPinocchioInterface(), interface.getCentroidalModelInfo(),
      interface.mpcSettings().timeHorizon_);
  // The commanded base height stands on the ground the reference manager applied in this solve, so it follows a hot
  // reload of terrainHeight (TargetTrajectoriesCalculatorBase::setTerrainHeightSource).
  mpcTargetTrajectoriesCalculator.setTerrainHeightSource(
      [referenceManager = interface.getSwitchedModelReferenceManagerPtr()]() { return referenceManager->getAppliedTerrainHeight(); });
  ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajectoriesFunc =
      [&mpcTargetTrajectoriesCalculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t finalTime,
                                         const vector_t& initState) mutable {
        return mpcTargetTrajectoriesCalculator.commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      };
  auto ros2ProceduralMpcMotionManager = std::make_shared<Ros2ProceduralMpcMotionManager>(
      gaitFile, referenceFile, interface.getSwitchedModelReferenceManagerPtr(), effectiveMpcRobotModel, targetTrajectoriesFunc);

  ros2ProceduralMpcMotionManager->subscribe(nodeHandle, qos);
  // A reset of the MPC (MPC_BASE::reset(): the /mpc_reset service) resets the command path with it: the motion manager
  // itself, and through this hook the target calculator behind targetTrajectoriesFunc, whose filters are its state.
  ros2ProceduralMpcMotionManager->setResetHook([&mpcTargetTrajectoriesCalculator]() { mpcTargetTrajectoriesCalculator.reset(); });

  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  mpc.getSolverPtr()->addSynchronizedModule(ros2ProceduralMpcMotionManager);
  // Online contact planning (contactScheduleSource: contact_planner in task.yaml): the planner module feeds mode schedules and footholds
  // to the reference manager and, like the other synchronized modules, has to run before every solve.
  if (const std::shared_ptr<ContactPlannerModule> contactPlannerModule = interface.getContactPlannerModulePtr()) {
    mpc.getSolverPtr()->addSynchronizedModule(contactPlannerModule);
  }

  // Register real-time MPC parameter hot-reloading, wired by the one function both MPC nodes use: the updater is sized
  // to the OCP input, reaches the reference manager, the contact planner and the locomotion-heuristic layer, and
  // reloads the consumers of reference.yaml built above (both outlive the updater: they live in this scope, as it does)
  // so that the Command Limits tab of the remote control changes them on the running controller.
  absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> updaterResult = makeCentroidalMpcParameterUpdater(
      &mpc, interface, taskFile, urdfFile, referenceFile,
      {[&mpcTargetTrajectoriesCalculator](const std::string& file) { mpcTargetTrajectoriesCalculator.reloadCommandLimits(file); },
       [ros2ProceduralMpcMotionManager](const std::string& file) { ros2ProceduralMpcMotionManager->reloadCommandLimits(file); }});
  CHECK(updaterResult.ok()) << "Failed to create the MPC parameter updater: " << updaterResult.status();
  const std::shared_ptr<MpcParameterUpdaterModule> mpcParameterUpdater = *std::move(updaterResult);
  mpcParameterUpdater->subscribe(nodeHandle);
  mpc.getSolverPtr()->addSynchronizedModule(mpcParameterUpdater);

  MPC_ROS_Interface mpcNode(mpc, robotName);
  mpcNode.launchNodes(nodeHandle, qos);

  return 0;
}
