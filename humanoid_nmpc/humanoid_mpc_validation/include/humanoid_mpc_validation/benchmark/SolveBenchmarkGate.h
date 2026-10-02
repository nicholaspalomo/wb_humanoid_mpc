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
#include <vector>

#include "absl/strings/string_view.h"

#include "humanoid_mpc_validation/io/JsonValue.h"

namespace ocs2::humanoid::validation {

// LINT.IfChange(real_time_gate)
/**
 * The real-time gate of the quaternion design's section 4.6, a candidate solve benchmark (B3) against its baseline (B0),
 * both documents of runSolveBenchmark() for the same robot on the same recorded states and machine:
 *  - the solver iteration's mean at most +5 % and its p99 at most +10 %;
 *  - the LQ-approximation phase's mean at most +8 %;
 *  - the candidate's p99 below 80 % of the robot's MPC period;
 *  - every CppAD library's tape operation count at most +5 %, and their total too.
 */
struct SolveBenchmarkGate {
  double meanIncrease = 0.05;
  double p99Increase = 0.10;
  double lqApproximationIncrease = 0.08;
  double maxP99FractionOfPeriod = 0.8;
  double tapeOperationIncrease = 0.05;
};
// LINT.ThenChange(//humanoid_nmpc/docs/quaternion_base_orientation/README.md)

/**
 * A library's key in tape_operation_counts with the state-layout tag of the CppAD folder removed: Step 8 of the design
 * moves every library from cppad_<mpc><robot>/<rest> to cppad_<mpc><robot>/<kStateLayoutTag>/<rest>, and the gate
 * compares a library with itself across that move.
 */
std::string normalizedLibraryKey(absl::string_view key);

/** The candidate against the baseline, one sentence per breach of `gate`; empty when it passes. */
std::vector<std::string> compareSolveBenchmarks(const JsonValue& candidate, const JsonValue& baseline, const SolveBenchmarkGate& gate);

}  // namespace ocs2::humanoid::validation
