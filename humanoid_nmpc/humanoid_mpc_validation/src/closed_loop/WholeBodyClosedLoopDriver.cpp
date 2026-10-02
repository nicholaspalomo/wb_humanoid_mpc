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

#include "humanoid_mpc_validation/closed_loop/WholeBodyClosedLoopDriver.h"

#include <exception>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_wb_mpc/command/WBMpcTargetTrajectoriesCalculator.h"

namespace ocs2::humanoid::validation {

absl::StatusOr<std::unique_ptr<WholeBodyClosedLoopDriver>> WholeBodyClosedLoopDriver::create(const RobotConfiguration& configuration,
                                                                                             const ClosedLoopDriverOptions& options) {
  if (configuration.formulation != MpcFormulation::kWholeBody) {
    return absl::InvalidArgumentError(
        absl::StrCat("[WholeBodyClosedLoopDriver] ", configuration.name, " is not a whole-body configuration"));
  }
  std::unique_ptr<WholeBodyClosedLoopDriver> driver(new WholeBodyClosedLoopDriver());
  RETURN_IF_ERROR(driver->initialize(configuration, options));
  return driver;
}

WholeBodyClosedLoopDriver::~WholeBodyClosedLoopDriver() {
  // The controller and the MPC first: they hold references into the command path and the interface.
  releaseControllerAndMpc();
}

absl::Status WholeBodyClosedLoopDriver::initialize(const RobotConfiguration& configuration, const ClosedLoopDriverOptions& options) {
  ASSIGN_OR_RETURN(interface_, WBMpcInterface::Create(configuration.taskFile, configuration.urdfFile, configuration.referenceFile));
  WBMpcInterface& interface = *interface_;
  RETURN_IF_ERROR(initializeShared(configuration, options, interface.modelSettings(), interface.mpcSettings(), interface.sqpSettings(),
                                   interface.getOptimalControlProblem(), interface.getInitializer(), interface.getReferenceManagerPtr(),
                                   interface.getMpcRobotModel(), interface.getPinocchioInterface(), interface.getInitialState()));
  if (settings_.wbMpcFeedforward != WbMpcFeedforward::kInverseDynamics) {
    return absl::InvalidArgumentError(absl::StrCat(configuration.taskFile,
                                                   ": wbMpcFeedforward: ", wbMpcFeedforwardName(settings_.wbMpcFeedforward),
                                                   "; the whole-body controller computes its feedforward with the inverse dynamics only."));
  }

  // The MPC node's side (WBMpcNode::Create()).
  // LINT.IfChange(whole_body_mpc_node_wiring)
  std::unique_ptr<WBMpcTargetTrajectoriesCalculator> calculator = std::make_unique<WBMpcTargetTrajectoriesCalculator>(
      configuration.referenceFile, interface.getMpcRobotModel(), interface.mpcSettings().timeHorizon_);
  calculator->setTerrainHeightSource(
      [referenceManager = interface.getSwitchedModelReferenceManagerPtr()]() { return referenceManager->getAppliedTerrainHeight(); });
  RETURN_IF_ERROR(
      initializeCommandPath(std::move(calculator), interface.getSwitchedModelReferenceManagerPtr(), interface.getMpcRobotModel()));
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcNode.cpp:mpc_wiring)

  // The robot binary's side (WBMpcRobotMain.cpp), with the lockstep link in place of the remote one.
  // LINT.IfChange(whole_body_robot_controller)
  std::unique_ptr<WBMpcMrtJointController> jointController;
  try {
    jointController = std::make_unique<WBMpcMrtJointController>(*robotDescription_, interface.modelSettings(),
                                                                lockstepMpcLinkFactory(/*solverName=*/"WB MPC Solver Thread"),
                                                                interface.getPinocchioInterface(), configuration.pdGainsFile);
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat("the whole-body MRT joint controller did not start: ", error.what()));
  }
  if (settings_.contactWrenchGate.has_value()) jointController->setContactWrenchGateConfig(*settings_.contactWrenchGate);
  if (settings_.safetyDecayTimeConstant.has_value()) jointController->setSafetyDecayTimeConstant(*settings_.safetyDecayTimeConstant);
  // The whole-body sim handed the controller its posture before its mode.
  std::unique_ptr<MrtRobotController<WBMpcMrtJointController>> controller =
      std::make_unique<MrtRobotController<WBMpcMrtJointController>>(std::move(jointController), CycleInputOrder::kPostureThenMode);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcRobotMain.cpp:whole_body_robot_controller)
  // clang-format on
  controller_ = controller.get();
  robotController_ = std::move(controller);
  if (mpcLink_ == nullptr) {
    return absl::InternalError("[WholeBodyClosedLoopDriver] the controller did not make its MPC link");
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::validation
