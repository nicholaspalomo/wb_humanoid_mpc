/******************************************************************************
Copyright (c) 2017, Farbod Farshidian. All rights reserved.

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

#include <algorithm>
#include <cstddef>

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/ModeSchedule.h>

#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/TimeDiscretization.h"
#include "ocs2_oc/trajectory_adjustment/TrajectorySpreading.h"

namespace ocs2 {

/**
 * Stamps each post-event sample of a primal solution that shares its event's time with the sample before it at the start of its
 * interval, getIntervalStart() of a post-event node: the event time plus weakEpsilon.
 *
 * multiple_shooting::toPrimalSolution() stamps an event's pre-event and post-event samples both at the event time. TrajectorySpreading
 * takes a post-event sample to lie after its event, as a rollout's does, and finds it as the first sample later than the event time. For
 * two samples at the event time that is the node after the post-event one, so an unchanged mode schedule would move that node's time to
 * the event and hand the post-event interval that node's input. The post-event input is read at the same time when the next solve is
 * initialized (multiple_shooting::initializeStateInputTrajectories()), so the stamp leaves it where the previous solution had it.
 *
 * @param [in, out] primalSolution: A primal solution whose postEventIndices_ name its post-event samples. Samples that already lie after
 * the sample before them keep their time.
 */
inline void stampPostEventSamplesAtIntervalStart(PrimalSolution& primalSolution) {
  scalar_array_t& timeTrajectory = primalSolution.timeTrajectory_;
  for (const std::size_t postEventIndex : primalSolution.postEventIndices_) {
    if (postEventIndex == 0 || postEventIndex >= timeTrajectory.size()) continue;
    if (timeTrajectory[postEventIndex] != timeTrajectory[postEventIndex - 1]) continue;
    scalar_t intervalStart = getIntervalStart(AnnotatedTime(timeTrajectory[postEventIndex], AnnotatedTime::Event::PostEvent));
    // timeDiscretizationWithEvents() keeps nodes further apart than weakEpsilon; the bound only keeps the times sorted otherwise.
    if (postEventIndex + 1 < timeTrajectory.size()) intervalStart = std::min(intervalStart, timeTrajectory[postEventIndex + 1]);
    timeTrajectory[postEventIndex] = intervalStart;
  }
}

/**
 * Adjusts in-place a primal solution based on the last changes in mode schedule using a TrajectorySpreading strategy.
 * Note: PrimalSolution::controllerPtr_ will not be adjusted.
 *
 * @param [in] oldModeSchedule: The old mode schedule associated to the trajectories which should be adjusted.
 * @param [in] newModeSchedule: The new mode schedule that should be adapted to.
 * @param [in, out] primalSolution: The primal solution that is associated with the old mode schedule. Its post-event samples end up after
 * their events (stampPostEventSamplesAtIntervalStart()).
 */
inline void trajectorySpread(const ModeSchedule& oldModeSchedule, const ModeSchedule& newModeSchedule, PrimalSolution& primalSolution) {
  stampPostEventSamplesAtIntervalStart(primalSolution);

  // trajectory spreading
  TrajectorySpreading trajectorySpreading;
  trajectorySpreading.set(oldModeSchedule, newModeSchedule, primalSolution.timeTrajectory_);

  // adjust modeSchedule, state, input, time, postEventIndices
  primalSolution.modeSchedule_ = newModeSchedule;
  trajectorySpreading.adjustTrajectory(primalSolution.stateTrajectory_);
  trajectorySpreading.adjustTrajectory(primalSolution.inputTrajectory_);
  trajectorySpreading.adjustTimeTrajectory(primalSolution.timeTrajectory_);
  primalSolution.postEventIndices_ = trajectorySpreading.getPostEventIndices();
}

}  // namespace ocs2
