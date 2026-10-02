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

#include "humanoid_mpc_validation/closed_loop/CentroidalClosedLoopDriver.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"

namespace ocs2::humanoid::validation {

absl::StatusOr<std::unique_ptr<CentroidalClosedLoopDriver>> CentroidalClosedLoopDriver::create(const RobotConfiguration& configuration,
                                                                                               const ClosedLoopDriverOptions& options) {
  if (configuration.formulation != MpcFormulation::kCentroidal) {
    return absl::InvalidArgumentError(
        absl::StrCat("[CentroidalClosedLoopDriver] ", configuration.name, " is not a centroidal configuration"));
  }
  std::unique_ptr<CentroidalClosedLoopDriver> driver(new CentroidalClosedLoopDriver());
  RETURN_IF_ERROR(driver->initialize(configuration, options));
  return driver;
}

CentroidalClosedLoopDriver::~CentroidalClosedLoopDriver() {
  // The controller and the MPC first: they hold references into the command path and the interface.
  releaseControllerAndMpc();
  parameterUpdater_.reset();
}

absl::Status CentroidalClosedLoopDriver::initialize(const RobotConfiguration& configuration, const ClosedLoopDriverOptions& options) {
  ASSIGN_OR_RETURN(interface_, CentroidalMpcInterface::Create(configuration.taskFile, configuration.urdfFile, configuration.referenceFile));
  CentroidalMpcInterface& interface = *interface_;
  RETURN_IF_ERROR(initializeShared(configuration, options, interface.modelSettings(), interface.mpcSettings(), interface.sqpSettings(),
                                   interface.getOptimalControlProblem(), interface.getInitializer(), interface.getReferenceManagerPtr(),
                                   interface.getMpcRobotModel(), interface.getPinocchioInterface(), interface.getInitialState()));

  // The MPC node's side (CentroidalMpcNode::Create()).
  // LINT.IfChange(centroidal_mpc_node_wiring)
  // Everything that produces or consumes OCP inputs uses the effective model (the basis-vector input layout).
  const MpcRobotModelBase<scalar_t>& effectiveModel = interface.getEffectiveMpcRobotModel();
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator = std::make_unique<CentroidalMpcTargetTrajectoriesCalculator>(
      configuration.referenceFile, effectiveModel, interface.getPinocchioInterface(), interface.getCentroidalModelInfo(),
      interface.mpcSettings().timeHorizon_);
  calculator->setTerrainHeightSource(
      [referenceManager = interface.getSwitchedModelReferenceManagerPtr()]() { return referenceManager->getAppliedTerrainHeight(); });
  RETURN_IF_ERROR(initializeCommandPath(std::move(calculator), interface.getSwitchedModelReferenceManagerPtr(), effectiveModel));

  if (const std::shared_ptr<ContactPlannerModule> contactPlannerModule = interface.getContactPlannerModulePtr()) {
    mpc_->getSolverPtr()->addSynchronizedModule(contactPlannerModule);
  }

  TargetTrajectoriesCalculatorBase* calculatorPtr = calculator_.get();
  const std::shared_ptr<ProceduralMpcMotionManager> motionManager = motionManager_;
  ASSIGN_OR_RETURN(
      parameterUpdater_,
      makeCentroidalMpcParameterUpdater(mpc_.get(), interface, configuration.taskFile, configuration.urdfFile, configuration.referenceFile,
                                        {[calculatorPtr](const std::string& file) { calculatorPtr->reloadCommandLimits(file); },
                                         [motionManager](const std::string& file) { motionManager->reloadCommandLimits(file); }}));
  mpc_->getSolverPtr()->addSynchronizedModule(parameterUpdater_);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalMpcNode.cpp:mpc_wiring)
  // clang-format on

  // The robot binary's side (CentroidalMpcRobotMain.cpp), with the lockstep link in place of the remote one.
  // LINT.IfChange(centroidal_robot_controller)
  std::unique_ptr<CentroidalMpcMrtJointController> jointController;
  try {
    jointController =
        std::make_unique<CentroidalMpcMrtJointController>(*robotDescription_, interface.modelSettings(), interface.getMpcRobotModel(),
                                                          lockstepMpcLinkFactory(/*solverName=*/"Centroidal MPC Solver Thread"),
                                                          interface.getPinocchioInterface(), configuration.pdGainsFile, &effectiveModel);
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat("the centroidal MRT joint controller did not start: ", error.what()));
  }
  if (settings_.wbMpcFeedforward == WbMpcFeedforward::kGravityCompensation) jointController->setUseGravityCompFeedforward(/*enable=*/true);
  if (settings_.mpcEntryBlendTime.has_value()) jointController->setMpcEntryBlendTime(*settings_.mpcEntryBlendTime);
  if (settings_.safetyDecayTimeConstant.has_value()) jointController->setSafetyDecayTimeConstant(*settings_.safetyDecayTimeConstant);
  if (settings_.contactWrenchGate.has_value()) jointController->setContactWrenchGateConfig(*settings_.contactWrenchGate);
  std::unique_ptr<MrtRobotController<CentroidalMpcMrtJointController>> controller =
      std::make_unique<MrtRobotController<CentroidalMpcMrtJointController>>(std::move(jointController), CycleInputOrder::kModeThenPosture);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalMpcRobotMain.cpp:centroidal_robot_controller)
  // clang-format on
  controller_ = controller.get();
  robotController_ = std::move(controller);
  if (mpcLink_ == nullptr) {
    return absl::InternalError("[CentroidalClosedLoopDriver] the controller did not make its MPC link");
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::validation
