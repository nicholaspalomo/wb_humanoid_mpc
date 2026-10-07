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
#include <vector>

#include "absl/base/nullability.h"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"

namespace ocs2::humanoid {

/**
 * The reference-file reloaders of an MPC node or driver of either formulation (MpcParameterUpdaterModule::
 * addReferenceFileReloader()): the command limits of a reloaded reference file go to the target trajectories
 * calculator `calculator` and to the procedural motion manager `motionManager` (an OCS2 synchronized module, hence
 * shared), the two consumers that read them at construction, so that the Command Limits tab of the remote control
 * changes them on the running MPC. `calculator` is kept and must outlive the updater the reloaders are registered with;
 * both hold their limits in atomics, so the reloaders may run on the solver thread while targets are built on another.
 */
std::vector<MpcParameterUpdaterModule::ReferenceFileReloader> makeCommandLimitsReloaders(
    TargetTrajectoriesCalculatorBase* absl_nonnull calculator, std::shared_ptr<ProceduralMpcMotionManager> motionManager);

}  // namespace ocs2::humanoid
