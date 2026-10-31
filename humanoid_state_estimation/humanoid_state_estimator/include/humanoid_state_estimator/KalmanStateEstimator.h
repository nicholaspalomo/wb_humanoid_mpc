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

#include <atomic>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"

#include "humanoid_state_estimator/KalmanBaseEstimator.h"
#include "humanoid_state_estimator/StateEstimator.h"
#include "humanoid_state_estimator/StateEstimatorNames.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid::estimation {

/**
 * The `kalman` state estimator: the backend's state with the base position and
 * linear velocity replaced by KalmanBaseEstimator's, from the IMU (the state's
 * root orientation, angular velocity and
 * getSpecforceBaseRtWorldEwrtBase()), the joint positions and velocities
 * and the state's contact flags. Everything else - the orientation, the angular
 * velocity, the joints, the flags, the time - is the backend's.
 *
 * The time step is the difference of the states' times; a state whose time did
 * not advance (the backend has not published a new one) or is not finite is
 * handed on with the previous estimate, without an update. An update that fails
 * (a non-finite reading, flags of another number than the contact points) is
 * counted and the estimate of the prediction, or the previous one, is handed
 * on.
 *
 * THREADS. As StateEstimator: estimate() and reset() allocate nothing, and the
 * first estimate() resets from its state when reset() was never called. A reset
 * that fails (a non-finite state) counts in failedUpdates().
 */
class KalmanStateEstimator final : public StateEstimator {
 public:
  /** The estimator over the joints of `description` (the RobotState's), which
   * must outlive it; `estimator` not null. */
  static absl::StatusOr<std::unique_ptr<KalmanStateEstimator>> Create(
      const robot::model::RobotDescription& description,
      std::unique_ptr<KalmanBaseEstimator> estimator);

  KalmanStateEstimator(const KalmanStateEstimator&) = delete;
  KalmanStateEstimator& operator=(const KalmanStateEstimator&) = delete;
  ~KalmanStateEstimator() override = default;

  absl::string_view name() const override { return kKalmanStateEstimatorName; }
  void reset(const robot::model::RobotState& measured) override;
  const robot::model::RobotState& estimate(
      const robot::model::RobotState& measured) override;
  uint64_t failedUpdates() const override {
    return failedUpdates_.load(std::memory_order_relaxed);
  }

  const KalmanBaseEstimator& baseEstimator() const { return *estimator_; }

 private:
  KalmanStateEstimator(const robot::model::RobotDescription& description,
                       std::unique_ptr<KalmanBaseEstimator> estimator);

  /** The joint positions and velocities of `state` into jointPositions_ and
   * jointVelocities_, by joint index. */
  void readJoints(const robot::model::RobotState& state);
  /** Writes the base estimate into estimated_. */
  void writeEstimate();

  std::unique_ptr<KalmanBaseEstimator> estimator_;
  robot::model::RobotState estimated_;
  vector_t jointPositions_;
  vector_t jointVelocities_;
  /** The time of the last state reset() or an update took; meaningful once
   * initialized_. */
  scalar_t previousTime_ = 0.0;
  bool initialized_ = false;
  std::atomic<uint64_t> failedUpdates_{0};
};

}  // namespace ocs2::humanoid::estimation
