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

#include <cstdint>

#include "absl/strings/string_view.h"

#include "robot_model/RobotState.h"

namespace ocs2::humanoid::estimation {

/**
 * A stage of the robot process's realtime cycle between the backend and the
 * controller: it takes the state the backend measured and hands on the state
 * the controller consumes, in particular the base's position and linear
 * velocity, which no sensor measures directly on hardware. Selected by name
 * (StateEstimatorRegistry, the task file's state_estimator).
 *
 * THREADS. reset() and estimate() are for the realtime thread: both are
 * realtime-safe (no allocation, no lock, no I/O) once the estimator is built.
 * failedUpdates() is for any thread. The class is otherwise not thread-safe.
 */
class StateEstimator {
 public:
  virtual ~StateEstimator() = default;

  /** The registry name of this estimator. */
  virtual absl::string_view name() const = 0;

  /**
   * Restarts the estimate from `measured`, taken as the truth: at start-up, and
   * after the plant jumped by a means no sensor saw (a simulator putting the
   * robot back). Realtime-safe.
   */
  virtual void reset(const robot::model::RobotState& measured) = 0;

  /**
   * The state the controller consumes this cycle, from the state `measured` the
   * backend published. The reference stays valid until the next call of
   * estimate() or reset() and the lifetime of `measured` (the ground truth
   * hands back `measured` itself). Allocates nothing.
   */
  virtual const robot::model::RobotState& estimate(
      const robot::model::RobotState& measured) = 0;

  /** The updates that failed (a non-finite reading) and kept the previous
   * estimate. Any thread. */
  virtual uint64_t failedUpdates() const = 0;
};

}  // namespace ocs2::humanoid::estimation
