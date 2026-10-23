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

#include "absl/status/statusor.h"

#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"

namespace ocs2::humanoid::config_dump {

/**
 * Every value the start-up of `configuration`'s stack builds from its files, as the MPC node, the robot process and the
 * lockstep driver build them: through the entry points that take the files by path, the calls the binaries make at
 * start-up, and nothing else (tools/config_dump/README.md). The MPC interface's
 * settings and problem (dumpProblem()), the target trajectories of a saturating command, the motion manager's gaits and
 * ramps, the MRT joint controller's resolved PD gains, the robot process's settings, the visualization's and the
 * keyboard teleoperation's limits.
 *
 * @return The dump, or the error of the root that refused the configuration.
 */
absl::StatusOr<std::string> dumpConfiguration(const validation::RobotConfiguration& configuration);

}  // namespace ocs2::humanoid::config_dump
