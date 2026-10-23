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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <memory>
#include <utility>

#include "absl/status/statusor.h"

#include "humanoid_mpc_validation/closed_loop/CentroidalClosedLoopDriver.h"
#include "humanoid_mpc_validation/closed_loop/ClosedLoopDriver.h"
#include "humanoid_mpc_validation/closed_loop/WholeBodyClosedLoopDriver.h"

namespace ocs2::humanoid::validation {

absl::StatusOr<std::unique_ptr<ClosedLoopDriver>> createClosedLoopDriver(const RobotConfiguration& configuration,
                                                                         const ClosedLoopDriverOptions& options) {
  if (configuration.formulation == MpcFormulation::kWholeBody) {
    absl::StatusOr<std::unique_ptr<WholeBodyClosedLoopDriver>> driver = WholeBodyClosedLoopDriver::create(configuration, options);
    if (!driver.ok()) return driver.status();
    return std::unique_ptr<ClosedLoopDriver>(*std::move(driver));
  }
  absl::StatusOr<std::unique_ptr<CentroidalClosedLoopDriver>> driver = CentroidalClosedLoopDriver::create(configuration, options);
  if (!driver.ok()) return driver.status();
  return std::unique_ptr<ClosedLoopDriver>(*std::move(driver));
}

}  // namespace ocs2::humanoid::validation
