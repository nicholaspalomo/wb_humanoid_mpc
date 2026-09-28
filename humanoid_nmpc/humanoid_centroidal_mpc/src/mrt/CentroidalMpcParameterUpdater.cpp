/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> makeCentroidalMpcParameterUpdater(
    MPC_BASE* mpc,
    const CentroidalMpcInterface& interface,
    const std::string& taskFile,
    const std::string& urdfFile,
    const std::string& referenceFile,
    std::vector<ReferenceFileReloader> referenceFileReloaders) {
  // The input layout the solver optimizes over is the effective model's: in basis-vector mode [lambda, joint
  // velocities], and the wrench-space R of task.yaml is transformed exactly as the OCP factory did.
  ASSIGN_OR_RETURN(std::unique_ptr<MpcParameterUpdaterModule> created,
                   MpcParameterUpdaterModule::Create(
                       mpc, taskFile, urdfFile, referenceFile, interface.getMpcRobotModel().getStateDim(),
                       interface.getEffectiveMpcRobotModel().getInputDim(), interface.modelSettings().contactNames,
                       interface.getSwitchedModelReferenceManagerPtr().get(), interface.getBasisInputsCostTransformConfig()));
  const std::shared_ptr<MpcParameterUpdaterModule> updater = std::move(created);
  // Null when contact planning is off, which the updater takes as "no contact_planning block to reload".
  updater->setContactPlannerModule(interface.getContactPlannerModulePtr());
  // Without this the locomotion_heuristics coefficients are launch-time only and the tuning GUI's sliders for them
  // write the file without reaching the running controller.
  updater->setLocomotionHeuristicLayer(interface.getLocomotionHeuristicLayerPtr());
  // The command limits and ramps come from reference.yaml, which its consumers read once at construction. Each consumer
  // must outlive the updater, and is responsible for the thread safety of what it writes.
  for (ReferenceFileReloader& reloader : referenceFileReloaders) {
    updater->addReferenceFileReloader(std::move(reloader));
  }
  return updater;
}

}  // namespace ocs2::humanoid
