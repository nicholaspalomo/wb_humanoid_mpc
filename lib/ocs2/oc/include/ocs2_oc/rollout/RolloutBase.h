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

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/control/ControllerBase.h>
#include <ocs2_core/manifold/StateManifold.h>
#include "ocs2_core/reference/ModeSchedule.h"

#include "ocs2_oc/rollout/RolloutSettings.h"

namespace ocs2 {

/**
 * This class is an interface class for forward rollout of the system dynamics.
 */
class RolloutBase {
 public:
  /**
   * Default constructor.
   *
   * @param [in] rolloutSettings: The rollout settings.
   */
  explicit RolloutBase(rollout::Settings rolloutSettings) : rolloutSettings_(std::move(rolloutSettings)) {}

  /**
   * Default destructor.
   */
  virtual ~RolloutBase() = default;

  /**
   * Returns the rollout settings.
   *
   * @return The rollout settings.
   */
  const rollout::Settings& settings() const { return rolloutSettings_; }

  /**
   * Returns pointer to the class.
   *
   * @return A raw pointer to the class.
   */
  virtual RolloutBase* absl_nonnull clone() const = 0;

  /**
   * Sets the manifold the state lives on (nullptr, the default, for a flat state). The rollout projects every state it
   * outputs onto the manifold (TimeTriggeredRollout normalizes every quaternion block), so a simulated or fed-back state
   * stays a valid configuration however long it is integrated, and its clones keep the manifold.
   */
  void setStateManifold(std::shared_ptr<const StateManifold> stateManifold) { stateManifoldPtr_ = std::move(stateManifold); }

  /** The manifold the state lives on, or nullptr. */
  const std::shared_ptr<const StateManifold>& getStateManifold() const { return stateManifoldPtr_; }

  /**
   * Forward integrate the system dynamics with given controller. It uses the given control policies and initial state,
   * to integrate the system dynamics in time period [initTime, finalTime].
   *
   * @param [in] initTime: The initial time.
   * @param [in] initState: The initial state.
   * @param [in] finalTime: The final time.
   * @param [in] controller: control policy.
   * @param [in] modeSchedule: Defines the sequence of modes and the associated event times.
   * @param [out] timeTrajectory: The time trajectory stamp.
   * @param [out] postEventIndices: Indices containing past-the-end index of events trigger.
   * @param [out] stateTrajectory: The state trajectory.
   * @param [out] inputTrajectory: The control input trajectory.
   *
   * @return The final state (state jump is considered if it took place)
   */
  virtual vector_t run(scalar_t initTime,
                       const vector_t& initState,
                       scalar_t finalTime,
                       ControllerBase* absl_nullable controller,
                       const ModeSchedule& modeSchedule,
                       scalar_array_t& timeTrajectory,
                       size_array_t& postEventIndices,
                       vector_array_t& stateTrajectory,
                       vector_array_t& inputTrajectory) = 0;

  /**
   * Prints out the rollout.
   *
   * @param [in] timeTrajectory: The time trajectory stamp.
   * @param [in] postEventIndices: An array of the post-event indices.
   * @param [in] stateTrajectory: The state trajectory.
   * @param [in] inputTrajectory: The control input trajectory.
   */
  static void display(const scalar_array_t& timeTrajectory,
                      const size_array_t& postEventIndices,
                      const vector_array_t& stateTrajectory,
                      const vector_array_t* absl_nullable const inputTrajectory);

 protected:
  /** Extracts an array of the rollout's start and final times for each active mode. */
  std::vector<std::pair<scalar_t, scalar_t>> findActiveModesTimeInterval(scalar_t initTime, scalar_t finalTime,
                                                                         const scalar_array_t& eventTimes) const;

  /** Checks for the numerical stability if rollout::Settings::checkNumericalStability is true. */
  void checkNumericalStability(const ControllerBase& controller, const scalar_array_t& timeTrajectory, const size_array_t& postEventIndices,
                               const vector_array_t& stateTrajectory, const vector_array_t& inputTrajectory) const;

  const rollout::Settings rolloutSettings_;

  std::shared_ptr<const StateManifold> stateManifoldPtr_;
};

}  // namespace ocs2
