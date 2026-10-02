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

#include <string>

#include "absl/status/status.h"

#include "humanoid_mpc_validation/closed_loop/LockstepClosedLoop.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"

namespace ocs2::humanoid::validation {

// LINT.IfChange(output_names)
/** <robot>_<scenario>.json: the metrics document. */
std::string metricsFileName(const std::string& robot, const std::string& scenario);
/** <robot>_<scenario>_timeseries.txt: the base, its reference and each solve's rotation gap over the commands, a golden file. */
std::string timeSeriesFileName(const std::string& robot, const std::string& scenario);
/** <robot>_<scenario>_states.txt: the robot at the recorded solves, the input of the solve benchmark. */
std::string recordedStatesFileName(const std::string& robot, const std::string& scenario);
// LINT.ThenChange(//Makefile:closed_loop_targets, //humanoid_nmpc/humanoid_mpc_validation/README.md:output_names)

/**
 * Writes `result` into `outputDir`: the metrics document, the time series, and the recorded states when the run recorded
 * any, the golden files with `provenance`. The first error of a file that cannot be written.
 */
absl::Status writeLockstepOutputs(const LockstepResult& result,
                                  const std::string& outputDir,
                                  const std::string& robot,
                                  const std::string& scenario,
                                  const GoldenProvenance& provenance);

}  // namespace ocs2::humanoid::validation
