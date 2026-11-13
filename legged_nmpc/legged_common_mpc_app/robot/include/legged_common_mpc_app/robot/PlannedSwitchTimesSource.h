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
#include <vector>

#include "ocs2_core/Types.h"

#include "legged_common_mpc_app/robot/RobotController.h"
#include "legged_state_estimator/ContactEstimationInputs.h"

namespace ocs2::legged {

/**
 * Gives the state estimator the planned contact switch times of the policy in
 * use: every cycle, before StateEstimator::estimate(), fill() asks the
 * controller (RobotController::plannedContactSwitchTimes()) and writes its
 * answer into the ContactEstimationInputs, or clears them while there is no
 * policy. The kalman estimator's contact-trust ramp reads them; with the ramp
 * off nothing does. RobotProcess::cycle() and the lockstep closed loop of
 * legged_mpc_validation fill the same way.
 *
 * REALTIME. The scratch storage is sized by the constructor; fill() allocates
 * nothing. Not thread-safe.
 */
class PlannedSwitchTimesSource {
 public:
  /** Sizes the scratch storage for a controller of `numContacts` contacts.
   * Allocates. */
  explicit PlannedSwitchTimesSource(size_t numContacts);

  PlannedSwitchTimesSource(const PlannedSwitchTimesSource&) = delete;
  PlannedSwitchTimesSource& operator=(const PlannedSwitchTimesSource&) = delete;
  PlannedSwitchTimesSource(PlannedSwitchTimesSource&&) noexcept = default;
  PlannedSwitchTimesSource& operator=(PlannedSwitchTimesSource&&) noexcept =
      default;
  ~PlannedSwitchTimesSource() = default;

  /** Writes `controller`'s planned switch times into `inputs`, one per contact
   * of both (the first min of their counts), or clears every contact's when
   * the controller has none. */
  void fill(const RobotController& controller,
            estimation::ContactEstimationInputs& inputs);

 private:
  /** [s] The controller's answer, one per contact. */
  std::vector<scalar_t> timesSinceSwitch_;
  std::vector<scalar_t> timesToSwitch_;
};

}  // namespace ocs2::legged
