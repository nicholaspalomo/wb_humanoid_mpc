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

#include "humanoid_state_estimator/KalmanStateEstimator.h"

#include <cmath>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid::estimation {

absl::StatusOr<std::unique_ptr<KalmanStateEstimator>> KalmanStateEstimator::Create(const robot::model::RobotDescription& description,
                                                                                   std::unique_ptr<KalmanBaseEstimator> estimator) {
  if (estimator == nullptr) {
    return absl::InvalidArgumentError("KalmanStateEstimator: no base estimator");
  }
  if (estimator->kinematics().numJoints() != description.getNumJoints()) {
    return absl::InvalidArgumentError(absl::StrCat("KalmanStateEstimator: the kinematics take ", estimator->kinematics().numJoints(),
                                                   " joints, the robot has ", description.getNumJoints()));
  }
  return absl::WrapUnique(new KalmanStateEstimator(description, std::move(estimator)));
}

KalmanStateEstimator::KalmanStateEstimator(const robot::model::RobotDescription& description,
                                           std::unique_ptr<KalmanBaseEstimator> estimator)
    : estimator_(std::move(estimator)),
      estimated_(description, estimator_->numContacts()),
      jointPositions_(vector_t::Zero(static_cast<Eigen::Index>(description.getNumJoints()))),
      jointVelocities_(vector_t::Zero(static_cast<Eigen::Index>(description.getNumJoints()))) {}

void KalmanStateEstimator::readJoints(const robot::model::RobotState& state) {
  for (Eigen::Index joint = 0; joint < jointPositions_.size(); ++joint) {
    jointPositions_[joint] = state.getJointPosition(static_cast<size_t>(joint));
    jointVelocities_[joint] = state.getJointVelocity(static_cast<size_t>(joint));
  }
}

void KalmanStateEstimator::writeEstimate() {
  estimated_.setRootPositionInWorldFrame(estimator_->basePosition());
  estimated_.setRootLinearVelocityInLocalFrame(estimated_.getRootRotationLocalToWorldFrame().inverse() * estimator_->baseLinearVelocity());
}

void KalmanStateEstimator::reset(const robot::model::RobotState& measured) {
  readJoints(measured);
  const quaternion_t orientation = measured.getRootRotationLocalToWorldFrame();
  const vector3_t worldVelocity = orientation * measured.getRootLinearVelocityInLocalFrame();
  if (!estimator_->reset(measured.getRootPositionInWorldFrame(), worldVelocity, orientation, jointPositions_, jointVelocities_).ok()) {
    failedUpdates_.fetch_add(1, std::memory_order_relaxed);
  }
  previousTime_ = measured.getTime();
  initialized_ = true;
}

const robot::model::RobotState& KalmanStateEstimator::estimate(const robot::model::RobotState& measured) {
  if (!initialized_) reset(measured);
  estimated_ = measured;
  const scalar_t dt = measured.getTime() - previousTime_;
  if (std::isfinite(dt) && dt > 0.0) {
    readJoints(measured);
    const absl::Status updated =
        estimator_->update(dt, measured.getRootRotationLocalToWorldFrame(), measured.getRootAngularVelocityInLocalFrame(),
                           measured.getImuLinearAccelerationInLocalFrame(), jointPositions_, jointVelocities_, measured.getContactFlags());
    if (!updated.ok()) failedUpdates_.fetch_add(1, std::memory_order_relaxed);
    previousTime_ = measured.getTime();
  }
  writeEstimate();
  return estimated_;
}

}  // namespace ocs2::humanoid::estimation
