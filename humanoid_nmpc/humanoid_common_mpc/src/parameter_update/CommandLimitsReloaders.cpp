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

#include "humanoid_common_mpc/parameter_update/CommandLimitsReloaders.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"

#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"

namespace ocs2::humanoid {

std::vector<MpcParameterUpdaterModule::ReferenceFileReloader> makeCommandLimitsReloaders(
    TargetTrajectoriesCalculatorBase* absl_nonnull calculator, std::shared_ptr<ProceduralMpcMotionManager> motionManager) {
  std::vector<MpcParameterUpdaterModule::ReferenceFileReloader> reloaders;
  reloaders.emplace_back([calculator](const ReferenceSettings& settings) {
    calculator->applyCommandLimits(settings);
    return absl::OkStatus();
  });
  reloaders.emplace_back([motionManager = std::move(motionManager)](const ReferenceSettings& settings) {
    return motionManager->applyCommandLimits(settings);
  });
  return reloaders;
}

}  // namespace ocs2::humanoid
