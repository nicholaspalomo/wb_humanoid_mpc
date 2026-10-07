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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_mpc/MPC_BASE.h"

#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

namespace ocs2::humanoid {

/**
 * Returns the parameter updater of a node that runs the whole-body MPC `mpc` (may be nullptr: nothing is then applied),
 * wired to what `interface` built that a hot reload has to reach: the task file the problem was built from
 * (WBMpcInterface::taskFile()), which a reload's start-up fields are compared with; the whole-body coordinates and OCP
 * input; the reference manager (the ground, the swing trajectories); the whole-body appliers
 * (wholeBodyHotFieldAppliers()); and `referenceFileReloaders`, the consumers of the reference file the node built
 * (makeCommandLimitsReloaders()), each handed the command limits of a reloaded file. It watches `taskFile` and
 * `referenceFile`; the whole-body MPC has no contact planner's file. `interface` must outlive the updater.
 *
 * Every node and driver that runs the whole-body MPC builds its updater with this one function, as the centroidal ones
 * use makeCentroidalMpcParameterUpdater(). The caller hands it the updates of operator/mpc_parameters
 * (enqueueParameterUpdate()) and registers it with the solver after the motion manager; OCS2's addSynchronizedModule()
 * takes the shared_ptr. Returns the errors of MpcParameterUpdaterModule::Create().
 */
absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> makeWholeBodyMpcParameterUpdater(
    MPC_BASE* absl_nullable mpc,
    const WBMpcInterface& interface,
    const std::string& taskFile,
    const std::string& referenceFile,
    std::vector<MpcParameterUpdaterModule::ReferenceFileReloader> referenceFileReloaders);

}  // namespace ocs2::humanoid
