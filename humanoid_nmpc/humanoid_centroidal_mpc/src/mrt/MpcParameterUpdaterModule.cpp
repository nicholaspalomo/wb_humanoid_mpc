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

#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"

#include <absl/log/log.h>
#include <ocs2_sqp/SqpSolver.h>
#include <iostream>

namespace ocs2::humanoid {

MpcParameterUpdaterModule::MpcParameterUpdaterModule(MPC_BASE* mpcPtr,
                                                     const std::string& taskFile,
                                                     const std::string& urdfFile,
                                                     const std::string& referenceFile)
    : mpcPtr_(mpcPtr), taskFile_(taskFile), urdfFile_(urdfFile), referenceFile_(referenceFile) {
  if (!taskFile_.empty() && std::filesystem::exists(taskFile_)) {
    std::error_code ec;
    taskFileLastWriteTime_ = std::filesystem::last_write_time(taskFile_, ec);
  }
}

void MpcParameterUpdaterModule::preSolverRun(scalar_t initTime,
                                             scalar_t finalTime,
                                             const vector_t& currentState,
                                             const ReferenceManagerInterface& referenceManager) {
  // Check task.yaml modification time at roughly 1Hz (assuming solver runs around 100Hz)
  if (!taskFile_.empty() && checkCounter_++ % 100 == 0) {
    std::error_code ec;
    auto last_write = std::filesystem::last_write_time(taskFile_, ec);
    if (!ec && last_write != taskFileLastWriteTime_) {
      taskFileLastWriteTime_ = last_write;

      LOG(INFO) << "[MpcParameterUpdaterModule] Detected changes in " << taskFile_ << ". Regenerating OCP...";

      // Recreate the entire CentroidalMpcInterface
      // This will re-parse task.yaml, rebuild AD models if necessary (usually cached), and set up the OCP.
      auto newInterfaceStatus = CentroidalMpcInterface::Create(taskFile_, urdfFile_, referenceFile_);

      if (!newInterfaceStatus.ok()) {
        LOG(ERROR) << "[MpcParameterUpdaterModule] Failed to recreate CentroidalMpcInterface: " << newInterfaceStatus.status();
        return;
      }

      // Keep the new interface alive so its internal pointers (e.g. mpcRobotModelPtr, swingTrajectoryPlannerPtr) remain valid
      latestInterface_ = *std::move(newInterfaceStatus);

      // Extract the newly generated OptimalControlProblem
      auto newOcp = latestInterface_->getOptimalControlProblem();

      // Overwrite the solver's thread-local OCP definitions
      auto* solverBasePtr = mpcPtr_->getSolverPtr();
      if (solverBasePtr != nullptr) {
        auto* sqpSolverPtr = dynamic_cast<SqpSolver*>(solverBasePtr);
        if (sqpSolverPtr) {
          for (auto& ocp : sqpSolverPtr->getOcpDefinitions()) {
            ocp = newOcp;
          }
          LOG(INFO) << "[MpcParameterUpdaterModule] Successfully updated OCP definitions in SqpSolver.";
        } else {
          LOG(ERROR) << "[MpcParameterUpdaterModule] Underlying solver is not SqpSolver. Cannot update parameters.";
        }
      }
    }
  }
}

}  // namespace ocs2::humanoid
