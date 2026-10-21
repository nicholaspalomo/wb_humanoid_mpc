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

#include "ocs2_oc/trajectory_adjustment/TrajectorySpreading.h"

#include <algorithm>
#include <cassert>

#include <ocs2_core/NumericTraits.h>

namespace ocs2 {

namespace {

/** [s] How far outside the old trajectory's time range set() places an event time that it shifts out of that range. */
constexpr scalar_t kOutOfRangeEventTimeMargin = 1e-4;

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void TrajectorySpreading::set(const ModeSchedule& oldModeSchedule, const ModeSchedule& newModeSchedule,
                              const scalar_array_t& oldTimeTrajectory) {
  // step 1: What modes do the original primal solution contain?
  const scalar_t oldInitTime = oldTimeTrajectory.front();
  const scalar_t oldFinalTime = oldTimeTrajectory.back();

  // note: sizeof(eventTimes) + 1 == sizeof(modeSequence)
  const int oldFirstActiveModeIndex = upperBoundIndex(oldModeSchedule.eventTimes, oldInitTime);  // no event at initial time
  const int oldLastActiveModeIndex = upperBoundIndex(oldModeSchedule.eventTimes, oldFinalTime);  // can be an event at final time

  // step 2: What modes do the new mode schedule need?
  const int newFirstActiveModeIndex = upperBoundIndex(newModeSchedule.eventTimes, oldInitTime);  // no event at initial time
  const int newLastActiveModeIndex = upperBoundIndex(newModeSchedule.eventTimes, oldFinalTime);  // can be an event at final time

  // the starting mode of the matched sequence may differ from the leading mode of the old mode schedule.
  auto oldStartIndexOfMatchedSequence = oldFirstActiveModeIndex;
  // the starting mode of the matched sequence must cover the newInitTime
  const auto newStartIndexOfMatchedSequence = newFirstActiveModeIndex;

  // step 3: What is the longest window of matched mode sequence?
  // the size of the window where the active modes are identical.
  // this means: mode[firstMatchingModeIndex + w] != updatedMode[updatedFirstMatchingModeIndex + w]
  // Only the modes of the old trajectory are matched: past oldLastActiveModeIndex the range std::mismatch compares would be
  // reversed, and a mode after the old trajectory has no samples to keep.
  size_t w = 0;
  while (oldStartIndexOfMatchedSequence <= oldLastActiveModeIndex) {
    // +1 to include the last active mode
    const auto mismatchedIndex = std::mismatch(oldModeSchedule.modeSequence.cbegin() + oldStartIndexOfMatchedSequence,
                                               oldModeSchedule.modeSequence.cbegin() + oldLastActiveModeIndex + 1,
                                               newModeSchedule.modeSequence.cbegin() + newStartIndexOfMatchedSequence,
                                               newModeSchedule.modeSequence.cbegin() + newLastActiveModeIndex + 1);

    w = std::distance(oldModeSchedule.modeSequence.begin() + oldStartIndexOfMatchedSequence, mismatchedIndex.first);

    if (w > 0) {
      break;
    } else {
      // move starting index forward to find the first matched mode in the old mode schedule
      ++oldStartIndexOfMatchedSequence;
    }
  }

  // No mode of the old trajectory continues the new schedule's first mode: nothing to keep, erase all. The phases below index
  // the event times after the w matched modes, which do not exist for w == 0.
  if (w == 0) {
    eraseFromIndex_ = 0;
    computeSpreadingStrategy(oldTimeTrajectory, scalar_array_t(), scalar_array_t());
    return;
  }

  // matched event times. Used to store event time that is likely to be spread.
  // note: (beginEventItr + w - 1) <= endEventItr. There are w - 1 event times between w modes.
  scalar_array_t oldMatchedEventTimes;
  scalar_array_t newMatchedEventTimes;
  // Add corresponding matched event time
  // Phase 1: Add all event time that are between modes. Those event time show up in both original and modified trajectories.
  if (w > 0) {
    const auto oldBeginEventItr = oldModeSchedule.eventTimes.begin() + oldStartIndexOfMatchedSequence;
    oldMatchedEventTimes.assign(oldBeginEventItr, oldBeginEventItr + w - 1);

    const auto newBeginEventItr = newModeSchedule.eventTimes.begin() + newStartIndexOfMatchedSequence;
    newMatchedEventTimes.assign(newBeginEventItr, newBeginEventItr + w - 1);
  }

  // Phase 2: Add triggering event time if the first matched mode is NOT the leading mode of the old mode schedule.
  // As the first matched mode must be the leading mode of the new mode schedule, in the case that the first matched mode is NOT the
  // leading mode of the old mode schedule, the triggering event time need to be shifted back to the beginning of trajectories
  // In summary, Triggering event time can be either ignored (leading mode of both old and new schedule are matched) or shifted backward.
  if (w > 0 && oldStartIndexOfMatchedSequence > oldFirstActiveModeIndex) {
    const auto oldLastTriggeredEvent = oldModeSchedule.eventTimes[oldStartIndexOfMatchedSequence - 1];
    const auto newLastTriggeredEvent = oldInitTime - kOutOfRangeEventTimeMargin;  // a bit before the updated initial time

    oldMatchedEventTimes.insert(oldMatchedEventTimes.begin(), oldLastTriggeredEvent);
    newMatchedEventTimes.insert(newMatchedEventTimes.begin(), newLastTriggeredEvent);
  }

  // Phase 3: Add ending event time if that of the old mode schedule is less than that of the new mode schedule.
  // The situation is similar to phase 2 where ending event time is added if it has to be spread till the end of truncated trajectories.
  // If the last active mode of the old mode schedule is matched, the ending event time can be either left untouched or moved backward(no
  // space for spreading forward). But in both cases, we don't have to add the ending event time as everything after the new ending event
  // time will be erased.
  // If the last active mode of the old mode schedule is NOT matched but the last active mode of the new mode schedule is matched, that
  // indicates the old ending event time need to be spread till the end of trajectories.
  // If neither new last active mode and old last active mode is matched, we have to compare the corresponding ending event time.
  const bool isLastActiveModeOfOldModeSequenceMatched = oldStartIndexOfMatchedSequence + w - 1 == static_cast<size_t>(oldLastActiveModeIndex);
  const bool isLastActiveModeOfNewModeSequenceMatched = newStartIndexOfMatchedSequence + w - 1 == static_cast<size_t>(newLastActiveModeIndex);
  if (!isLastActiveModeOfOldModeSequenceMatched &&
      (isLastActiveModeOfNewModeSequenceMatched || oldModeSchedule.eventTimes[oldStartIndexOfMatchedSequence + w - 1] <
                                                       newModeSchedule.eventTimes[newStartIndexOfMatchedSequence + w - 1])) {
    const auto oldLEndEvent = oldModeSchedule.eventTimes[oldStartIndexOfMatchedSequence + w - 1];
    const auto newEndEvent = isLastActiveModeOfNewModeSequenceMatched ? oldFinalTime + kOutOfRangeEventTimeMargin
                                                                      : newModeSchedule.eventTimes[newStartIndexOfMatchedSequence + w - 1];

    oldMatchedEventTimes.push_back(oldLEndEvent);
    newMatchedEventTimes.push_back(newEndEvent);
  }

  assert(oldMatchedEventTimes.size() == newMatchedEventTimes.size());

  // find the start index of erasing(include the start index) (by default preserve all)
  eraseFromIndex_ = oldTimeTrajectory.size();
  // if last mode of the new mode sequence is NOT matched
  if (!isLastActiveModeOfNewModeSequenceMatched) {
    const auto mismatchEventTime = newModeSchedule.eventTimes[newStartIndexOfMatchedSequence + w - 1];
    eraseFromIndex_ = lowerBoundIndex(oldTimeTrajectory, mismatchEventTime);
  }

  // computes the index of the spreading values and intervals
  computeSpreadingStrategy(oldTimeTrajectory, oldMatchedEventTimes, newMatchedEventTimes);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void TrajectorySpreading::computeSpreadingStrategy(const scalar_array_t& oldTimeTrajectory, const scalar_array_t& oldMatchedEventTimes,
                                                   const scalar_array_t& newMatchedEventTimes) {
  // output containers
  beginIndices_.clear();
  endIndices_.clear();
  spreadingValueIndices_.clear();
  updatedPostEventIndices_.clear();
  updatedMatchedEventTimes_.clear();

  // finds the indices of the post event time
  size_array_t oldPostEventIndices = findPostEventIndices(oldMatchedEventTimes, oldTimeTrajectory);

  // finds the indices of the new post event time
  size_array_t newPostEventIndices = findPostEventIndices(newMatchedEventTimes, oldTimeTrajectory);

  // spreading periods and value indices
  for (size_t j = 0; j < oldPostEventIndices.size(); j++) {
    const size_t oldPreEventIndex = oldPostEventIndices[j] - 1;

    /**
     * Backward spreading
     *
     * |- - - - - - - - * - - - - - -|  Old event
     *           _______|
     *          /
     * |- -  - * - |(erase from here)| New event
     *
     *
     * The end index should be smaller or equal to the index of the first erased time stamp. The value index is not: when the new
     * schedule starts another mode before the old event's samples, the old post-event sample is erased, so adjustTrajectory() reads
     * the spreading values before it erases.
     */
    if (newPostEventIndices[j] < oldPostEventIndices[j]) {
      const size_t endIndex = std::min(oldPostEventIndices[j], eraseFromIndex_);
      beginIndices_.push_back(newPostEventIndices[j]);
      endIndices_.push_back(endIndex);
      spreadingValueIndices_.push_back(oldPostEventIndices[j]);

      // Check the assumption that zero length adjustment is excluded. Note that (newPostEventIndices[j] != eraseFromIndex_) since
      // eraseFromIndex_ will be the next postEventIndex after newPostEventIndices[j]. See phase 3 of adding matched event time.
      assert(beginIndices_.back() != endIndices_.back());
    }

    /**
     * Forward spreading
     *
     * |- * - - (*) - - * - - - -|  Old event
     *    |      |__________
     *    |_________        |
     *              |       |
     * |- - - - - - * - - -(*) - - | New event
     *
     *
     * The begin index should be larger than the previous new event index when overlapping occurs in the forward spreadind case. Otherwise
     * the current mode will override the previous mode resulting in wrong mode sequence.
     * The end index should be smaller or equal to the index of the first erased time stamp: the new event can be the one from which
     * the trajectory is erased, and an old sample at its time lies between eraseFromIndex_ (lower bound) and its post-event index
     * (upper bound).
     */
    else if (newPostEventIndices[j] > oldPostEventIndices[j]) {
      const size_t beginIndex = (j == 0 ? oldPostEventIndices[j] : std::max(oldPostEventIndices[j], newPostEventIndices[j - 1]));
      beginIndices_.push_back(beginIndex);
      endIndices_.push_back(std::min(newPostEventIndices[j], eraseFromIndex_));
      spreadingValueIndices_.push_back(oldPreEventIndex);

      // Check the assumption that zero length adjustment is excluded. Note that oldPostEventIndices[j] and newPostEventIndices[j - 1]
      // must be different.
      assert(beginIndices_.back() != endIndices_.back());
    }

    // the first index can be post event index but it is meaningless to store it in the indices array.
    // similarly, the last index can be pre-event index but it is meaningless to store it in the indices array.
    if (newPostEventIndices[j] != 0 && newPostEventIndices[j] < eraseFromIndex_) {
      updatedPostEventIndices_.push_back(newPostEventIndices[j]);
      updatedMatchedEventTimes_.push_back(newMatchedEventTimes[j]);
    }
  }  // end of j loop
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void TrajectorySpreading::adjustTimeTrajectory(scalar_array_t& timeTrajectory) const {
  timeTrajectory.erase(timeTrajectory.begin() + eraseFromIndex_, timeTrajectory.end());

  for (size_t i = 0; i < updatedPostEventIndices_.size(); i++) {
    assert(updatedPostEventIndices_[i] < timeTrajectory.size());
    constexpr auto eps = numeric_traits::weakEpsilon<scalar_t>();
    timeTrajectory[updatedPostEventIndices_[i] - 1] = updatedMatchedEventTimes_[i];
    timeTrajectory[updatedPostEventIndices_[i]] = std::min(updatedMatchedEventTimes_[i] + eps, timeTrajectory.back());
  }  // end of i loop
}

}  // namespace ocs2
