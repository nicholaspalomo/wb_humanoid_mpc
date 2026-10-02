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

#include <optional>

namespace ocs2::humanoid::validation {

/**
 * When the lockstep closed loop (LockstepClosedLoop) solves: steps 4 and 5 of InProcessMpcLink's solver thread, which
 * Execution::kCaller leaves to the caller, on the simulation's clock.
 *
 * A solve is due every `period` from the first one (step 5, the pacing; a solve falls in the first control cycle at or
 * after its time). A solve that fails and asks for a pause before the next attempt (MpcResetSupervisor::onSolveResult()'s
 * retry delay) holds the solves back until the delay has passed, or until a reset is requested after that failure
 * (MpcResetSupervisor::resetRequestedSinceLastFailure()), whichever comes first (step 4, waitBeforeRetry()). The solve
 * that ends the pause runs at once, and the period is counted again from it, as the solver thread starts the next
 * iteration straight after the wait and paces from there.
 */
class LockstepSolveSchedule {
 public:
  /**
   * @param firstSolveTime [s] When the first solve is due.
   * @param period         [s] 1 / mpcDesiredFrequency; <= 0: a solve in every control cycle, as the solver thread
   *                       runs back to back without a frequency.
   * @param timeTolerance  [s] Times this close are the same: the clock is a sum of simulation steps.
   */
  LockstepSolveSchedule(double firstSolveTime, double period, double timeTolerance);

  /**
   * Whether a solve is due in the control cycle at `time`. `resetRequestedSinceFailure` is
   * MpcResetSupervisor::resetRequestedSinceLastFailure(); it matters only while a pause holds the solves back.
   */
  bool isDue(double time, bool resetRequestedSinceFailure) const;

  /** Accounts for the solve run at `time`, which asked for a pause of `retryDelay` [s] before the next (zero: none). */
  void onSolve(double time, double retryDelay);

  /** Whether a failed solve's pause holds the solves back. */
  bool isPaused() const { return retryUntil_.has_value(); }

 private:
  const double period_;
  const double timeTolerance_;
  double nextSolveTime_;
  std::optional<double> retryUntil_;
};

}  // namespace ocs2::humanoid::validation
