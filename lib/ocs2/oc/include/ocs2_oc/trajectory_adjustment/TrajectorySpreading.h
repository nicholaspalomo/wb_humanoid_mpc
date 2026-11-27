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
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/ModeSchedule.h>

namespace ocs2 {

class TrajectorySpreading final {
 public:
  /**
   * Initialize trajectory spreading strategy. Run it before invoking other member functions
   *
   * @param [in] oldModeSchedule: The old mode schedule associated to the trajectories which should be adjusted.
   * @param [in] newModeSchedule: The new mode schedule that should be adapted to.
   * @param [in] oldTimeTrajectory: The old time trajectories that is associated with the old mode schedule. A post-event sample lies
   * after its event, as a rollout's does; a multiple-shooting solution's is first stamped there (stampPostEventSamplesAtIntervalStart()).
   */
  void set(const ModeSchedule& oldModeSchedule, const ModeSchedule& newModeSchedule, const scalar_array_t& oldTimeTrajectory);

  /**
   * Adjust continuous-time trajectory.
   *
   * @tparam data type.
   * @param [in, out] trajectory: trajectory for rectification.
   */
  template <typename T>
  void adjustTrajectory(std::vector<T>& trajectory) const;

  /**
   * Adjust time stamp of the trajectories and post event indices of the trajectories.
   *
   * @param [in, out] timeTrajectory: The time stamp of the trajectories.
   */
  void adjustTimeTrajectory(scalar_array_t& timeTrajectory) const;

  /**
   * Get adjusted post event index array.
   *
   * @return The post event indices of the trajectories.
   */
  const size_array_t& getPostEventIndices() const { return updatedPostEventIndices_; }

 private:
  /**
   * Computing spreading strategy based on the correspondence between the old event time and the new event time.
   *
   * @param oldTimeTrajectory: The old time trajectories that is associated with the old mode schedule.
   * @param oldMatchedEventTimes: The old event time that need adjustment potentially.
   * @param newMatchedEventTimes: The corresponding new event time.
   */
  void computeSpreadingStrategy(const scalar_array_t& oldTimeTrajectory, const scalar_array_t& oldMatchedEventTimes,
                                const scalar_array_t& newMatchedEventTimes);

  // helper function: get index of the iterator returned by upper_bound function
  int upperBoundIndex(const scalar_array_t& timeTrajectory, scalar_t queryTime) const {
    return std::distance(timeTrajectory.begin(), std::upper_bound(timeTrajectory.begin(), timeTrajectory.end(), queryTime));
  }

  // helper function: get index of the iterator returned by lower_bound function
  int lowerBoundIndex(const scalar_array_t& timeTrajectory, scalar_t queryTime) const {
    return std::distance(timeTrajectory.begin(), std::lower_bound(timeTrajectory.begin(), timeTrajectory.end(), queryTime));
  }

  /**
   * Finds post event indices of an array of event time for the given time trajectory.
   *
   * @param eventTimes: An array of event time.
   * @param timeTrajectory: Time trajectory.
   * @return An array containing post event indices of the given event time.
   */
  size_array_t findPostEventIndices(const scalar_array_t& eventTimes, const scalar_array_t& timeTrajectory) const {
    size_array_t postEventIndices(eventTimes.size());
    for (std::size_t i = 0; i < eventTimes.size(); i++) {
      if (i == eventTimes.size() - 1 && eventTimes[i] == timeTrajectory.back()) {
        postEventIndices[i] = timeTrajectory.size() - 1;
      } else {
        postEventIndices[i] = upperBoundIndex(timeTrajectory, eventTimes[i]);
      }
    }
    return postEventIndices;
  }

  /***********
   * Variables
   ***********/
  size_t eraseFromIndex_; /**< The first index to erase **/

  size_array_t beginIndices_;
  size_array_t endIndices_;
  size_array_t spreadingValueIndices_;

  size_array_t updatedPostEventIndices_;
  scalar_array_t updatedMatchedEventTimes_;
};

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename T>
void TrajectorySpreading::adjustTrajectory(std::vector<T>& trajectory) const {
  // extract spreading values beforehand since they might be overridden, and before the erase: a backward spreading carries
  // the post-event value of an old event, which lies at or after eraseFromIndex_ when the new schedule starts another mode
  // before that event's samples
  std::vector<T> spreadingValues;
  spreadingValues.reserve(spreadingValueIndices_.size());
  for (const size_t valueIndex : spreadingValueIndices_) {
    spreadingValues.push_back(trajectory[valueIndex]);
  }

  // erase segment of trajectory associated to mismatched modes
  trajectory.erase(trajectory.begin() + eraseFromIndex_, trajectory.end());

  // spread over the kept samples only (computeSpreadingStrategy() already ends every spreading at eraseFromIndex_)
  for (size_t i = 0; i < spreadingValueIndices_.size(); i++) {
    const size_t endIndex = std::min(endIndices_[i], trajectory.size());
    for (size_t j = beginIndices_[i]; j < endIndex; j++) {
      trajectory[j] = spreadingValues[i];
    }  // end of j loop
  }    // end of i loop
}

}  // namespace ocs2
