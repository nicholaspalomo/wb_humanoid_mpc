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

#include <cstddef>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_mpc_validation/closed_loop/ClosedLoopDriver.h"
#include "humanoid_mpc_validation/closed_loop/RecordedRobotStates.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

/** The `schema` member of every solve benchmark document. */
inline constexpr absl::string_view kSolveBenchmarkSchemaName = "humanoid_mpc_validation.solve_benchmark.v1";

/** How the solve benchmark runs. */
struct SolveBenchmarkOptions {
  ClosedLoopDriverOptions driver;  ///< the task file's threads unless overridden: the real-time gate uses the configured ones
  size_t warmupSolves = 25;        ///< solves after each start that are not timed: the plan converges from the reset
  size_t repeats = 3;              ///< passes over the recording, each from a fresh start
};

/**
 * The real-time benchmark of the quaternion design's section 4.6: the recorded robot states (RecordedRobotStates, from a
 * lockstep walking run) are replayed through the formulation's MRT joint controller in WB_MPC, which turns each into its
 * observation exactly as in the closed loop, and the controller's solver iteration runs once per record. Reported, over
 * the timed solves of every pass:
 *  - the wall time of the solver iteration [ms] (mean, p50, p99, max) and its share of the MPC period;
 *  - each SQP phase (SqpSolver::getBenchmarks()) [ms];
 *  - the tape operation count of every CppAD library the problem builds (CppAdInterface::getTapeOperationCount()),
 *    machine independent, and their sum.
 * InvalidArgument when the recording does not fit the configuration's robot.
 */
absl::StatusOr<JsonValue> runSolveBenchmark(const RobotConfiguration& configuration,
                                            const RecordedRobotStates& states,
                                            const SolveBenchmarkOptions& options,
                                            const std::string& label,
                                            JsonValue provenance);

/**
 * A library's name in the benchmark document: its folder without the "cppad_code_gen/" every robot's libraries live in
 * and without CppAdInterface's "/cppad_generated", e.g. "cppad_centroidal_mpc_atlas/FlowMap".
 */
std::string libraryNameOfFolder(absl::string_view libraryFolder);

}  // namespace ocs2::humanoid::validation
