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

#include "humanoid_state_estimator/StateEstimator.h"
#include "humanoid_state_estimator/StateEstimatorNames.h"
#include "robot_model/RobotState.h"

namespace ocs2::humanoid::estimation {

/**
 * The `ground_truth` state estimator: the backend's state, unchanged. On the
 * simulator that is the truth; on hardware the base position and linear
 * velocity are whatever the backend reports. Stateless and thread-compatible.
 */
class GroundTruthStateEstimator final : public StateEstimator {
 public:
  GroundTruthStateEstimator() = default;
  GroundTruthStateEstimator(const GroundTruthStateEstimator&) = delete;
  GroundTruthStateEstimator& operator=(const GroundTruthStateEstimator&) =
      delete;
  ~GroundTruthStateEstimator() override = default;

  absl::string_view name() const override {
    return kGroundTruthStateEstimatorName;
  }
  void reset(const robot::model::RobotState& /*measured*/) override {}
  /** `measured` itself, valid as long as `measured` is (the contract of
   * StateEstimator::estimate()). */
  // NOLINTNEXTLINE(bugprone-return-const-ref-from-parameter): the base contract bounds the reference by `measured`'s life.
  const robot::model::RobotState& estimate(
      const robot::model::RobotState& measured) override {
    return measured;
  }
  uint64_t failedUpdates() const override { return 0; }
};

}  // namespace ocs2::humanoid::estimation
