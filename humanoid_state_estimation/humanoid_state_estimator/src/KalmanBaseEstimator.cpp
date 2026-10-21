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

#include "humanoid_state_estimator/KalmanBaseEstimator.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::estimation {
namespace {

constexpr Eigen::Index kDim = 3;
constexpr char kBasePosition[] = "base_position";
constexpr char kBaseVelocity[] = "base_velocity";
constexpr char kImuAcceleration[] = "imu_acceleration";
/** The state channels in their order: the base position, the base velocity, then one per contact point. */
constexpr size_t kBasePositionChannel = 0;
constexpr size_t kBaseVelocityChannel = 1;
constexpr size_t kFirstContactChannel = 2;
/** The measurements of one stance contact point: relative position (3), velocity (3) and height (1). */
constexpr Eigen::Index kMeasurementRowsPerContact = 7;
constexpr size_t kRelativePositionMeasurement = 0;
constexpr size_t kVelocityMeasurement = 1;
constexpr size_t kHeightMeasurement = 2;
/** The standard deviations reset() starts the estimate with: the base [m], [m/s], and its contact points [m]. */
constexpr scalar_t kInitialPositionStd = 0.01;
constexpr scalar_t kInitialVelocityStd = 0.1;
constexpr scalar_t kInitialContactStd = 0.01;

matrix_t identity() {
  return matrix_t::Identity(kDim, kDim);
}

absl::Status checkStd(scalar_t value, absl::string_view field) {
  if (!(std::isfinite(value) && value > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("kalman_state_estimator.", field, " must be a positive standard deviation, got ", value));
  }
  return absl::OkStatus();
}

std::string contactStateName(size_t contact) {
  return absl::StrCat("contact_", contact, "_position");
}

}  // namespace

absl::Status KalmanBaseEstimator::validateConfig(const Config& config) {
  RETURN_IF_ERROR(checkStd(config.imuAccelerationNoise, "imu_acceleration_noise"));
  RETURN_IF_ERROR(checkStd(config.baseVelocityRandomWalk, "base_velocity_random_walk"));
  RETURN_IF_ERROR(checkStd(config.basePositionRandomWalk, "base_position_random_walk"));
  RETURN_IF_ERROR(checkStd(config.stanceContactRandomWalk, "stance_contact_random_walk"));
  RETURN_IF_ERROR(checkStd(config.swingContactRandomWalk, "swing_contact_random_walk"));
  RETURN_IF_ERROR(checkStd(config.kinematicPositionNoise, "kinematic_position_noise"));
  RETURN_IF_ERROR(checkStd(config.kinematicVelocityNoise, "kinematic_velocity_noise"));
  RETURN_IF_ERROR(checkStd(config.contactHeightNoise, "contact_height_noise"));
  if (!std::isfinite(config.contactHeight)) {
    return absl::InvalidArgumentError(absl::StrCat("kalman_state_estimator.contact_height must be finite, got ", config.contactHeight));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<KalmanBaseEstimator>> KalmanBaseEstimator::Create(const Config& config,
                                                                                 std::unique_ptr<ContactKinematics> kinematics) {
  RETURN_IF_ERROR(validateConfig(config));
  if (kinematics == nullptr) {
    return absl::InvalidArgumentError("KalmanBaseEstimator: no contact kinematics");
  }
  std::unique_ptr<KalmanBaseEstimator> estimator = absl::WrapUnique(new KalmanBaseEstimator(config, std::move(kinematics)));
  RETURN_IF_ERROR(estimator->build());
  return estimator;
}

KalmanBaseEstimator::KalmanBaseEstimator(const Config& config, std::unique_ptr<ContactKinematics> kinematics)
    : config_(config), kinematics_(std::move(kinematics)) {}

absl::Status KalmanBaseEstimator::build() {
  const size_t numContacts = kinematics_->numContacts();
  const matrix_t eye = identity();
  initialStates_.push_back(
      {.name = kBasePosition, .state = vector_t::Zero(kDim), .stateCovariance = kInitialPositionStd * kInitialPositionStd * eye});
  initialStates_.push_back(
      {.name = kBaseVelocity, .state = vector_t::Zero(kDim), .stateCovariance = kInitialVelocityStd * kInitialVelocityStd * eye});

  // The blocks that depend on dt are written by update(); their values here only size them.
  processModel_.push_back({.stateName = kBasePosition,
                           .stateTransition = {{kBaseVelocity, eye}},
                           .controlInputMatrix = {{kImuAcceleration, eye}},
                           .processNoiseCovariance = eye});
  processModel_.push_back({.stateName = kBaseVelocity, .controlInputMatrix = {{kImuAcceleration, eye}}, .processNoiseCovariance = eye});
  inputs_.push_back({.name = kImuAcceleration,
                     .input = vector_t::Zero(kDim),
                     .inputNoiseCovariance = config_.imuAccelerationNoise * config_.imuAccelerationNoise * eye});

  matrix_t heightRow = matrix_t::Zero(1, kDim);
  heightRow.rightCols<1>().setOnes();  // e_z^T
  for (size_t contact = 0; contact < numContacts; ++contact) {
    const std::string state = contactStateName(contact);
    initialStates_.push_back(
        {.name = state, .state = vector_t::Zero(kDim), .stateCovariance = kInitialContactStd * kInitialContactStd * eye});
    processModel_.push_back({.stateName = state, .processNoiseCovariance = eye});
    ContactMeasurements measurements;
    measurements.stateName = state;
    measurements.measurements.push_back(
        {.name = absl::StrCat("contact_", contact, "_relative_position"),
         .measurement = vector_t::Zero(kDim),
         .measurementModel = {{state, eye}, {kBasePosition, -eye}},
         .measurementNoiseCovariance = config_.kinematicPositionNoise * config_.kinematicPositionNoise * eye});
    measurements.measurements.push_back(
        {.name = absl::StrCat("contact_", contact, "_velocity"),
         .measurement = vector_t::Zero(kDim),
         .measurementModel = {{kBaseVelocity, eye}},
         .measurementNoiseCovariance = config_.kinematicVelocityNoise * config_.kinematicVelocityNoise * eye});
    measurements.measurements.push_back(
        {.name = absl::StrCat("contact_", contact, "_height"),
         .measurement = vector_t::Constant(1, config_.contactHeight),
         .measurementModel = {{state, heightRow}},
         .measurementNoiseCovariance = matrix_t::Constant(1, 1, config_.contactHeightNoise * config_.contactHeightNoise)});
    contacts_.push_back(std::move(measurements));
  }

  // The maps are final from here on, so the pointers into them stay valid.
  positionVelocityBlock_ = &processModel_[kBasePositionChannel].stateTransition.find(kBaseVelocity)->second;
  positionInputBlock_ = &processModel_[kBasePositionChannel].controlInputMatrix.find(kImuAcceleration)->second;
  velocityInputBlock_ = &processModel_[kBaseVelocityChannel].controlInputMatrix.find(kImuAcceleration)->second;

  RETURN_IF_ERROR(filter_.reset(initialStates_));
  RETURN_IF_ERROR(filter_.reserve(/*maxInputDim=*/kDim, /*maxMeasurementDim=*/kMeasurementRowsPerContact));
  return readEstimate();
}

absl::Status KalmanBaseEstimator::reset(const vector3_t& basePosition,
                                        const vector3_t& baseLinearVelocity,
                                        const quaternion_t& orientation,
                                        const vector_t& jointPositions,
                                        const vector_t& jointVelocities) {
  kinematics_->update(jointPositions, jointVelocities);
  const matrix3_t rotation = orientation.normalized().toRotationMatrix();
  initialStates_[kBasePositionChannel].state = basePosition;
  initialStates_[kBaseVelocityChannel].state = baseLinearVelocity;
  for (size_t contact = 0; contact < contacts_.size(); ++contact) {
    initialStates_[kFirstContactChannel + contact].state = basePosition + rotation * kinematics_->contactPositionInBase(contact);
  }
  RETURN_IF_ERROR(filter_.resetEstimate(initialStates_));
  return readEstimate();
}

absl::Status KalmanBaseEstimator::update(scalar_t dt,
                                         const quaternion_t& orientation,
                                         const vector3_t& angularVelocityInBase,
                                         const vector3_t& imuLinearAccelerationInBase,
                                         const vector_t& jointPositions,
                                         const vector_t& jointVelocities,
                                         const std::vector<bool>& contactFlags) {
  if (!(std::isfinite(dt) && dt > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("KalmanBaseEstimator::update(): the time step must be positive, got ", dt, " s"));
  }
  if (contactFlags.size() != contacts_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("KalmanBaseEstimator::update(): ", contactFlags.size(), " contact flags for ", contacts_.size(), " contact points"));
  }
  if (static_cast<size_t>(jointPositions.size()) != kinematics_->numJoints() ||
      static_cast<size_t>(jointVelocities.size()) != kinematics_->numJoints()) {
    return absl::InvalidArgumentError(absl::StrCat("KalmanBaseEstimator::update(): joint vectors of ", jointPositions.size(), " and ",
                                                   jointVelocities.size(), " entries for ", kinematics_->numJoints(), " joints"));
  }
  const matrix3_t rotation = orientation.normalized().toRotationMatrix();

  // The prediction: the world acceleration a = R f + g drives p and v.
  inputs_[0].input = rotation * imuLinearAccelerationInBase + kinematics_->gravity();
  positionVelocityBlock_->setIdentity();
  *positionVelocityBlock_ *= dt;
  positionInputBlock_->setIdentity();
  *positionInputBlock_ *= 0.5 * dt * dt;
  velocityInputBlock_->setIdentity();
  *velocityInputBlock_ *= dt;
  processModel_[kBasePositionChannel].processNoiseCovariance.setIdentity();
  processModel_[kBasePositionChannel].processNoiseCovariance *= config_.basePositionRandomWalk * config_.basePositionRandomWalk * dt;
  processModel_[kBaseVelocityChannel].processNoiseCovariance.setIdentity();
  processModel_[kBaseVelocityChannel].processNoiseCovariance *= config_.baseVelocityRandomWalk * config_.baseVelocityRandomWalk * dt;
  for (size_t contact = 0; contact < contacts_.size(); ++contact) {
    const scalar_t randomWalk = contactFlags[contact] ? config_.stanceContactRandomWalk : config_.swingContactRandomWalk;
    matrix_t& noise = processModel_[kFirstContactChannel + contact].processNoiseCovariance;
    noise.setIdentity();
    noise *= randomWalk * randomWalk * dt;
  }
  RETURN_IF_ERROR(filter_.predict(processModel_, inputs_));

  // The corrections of the stance contact points, one at a time.
  kinematics_->update(jointPositions, jointVelocities);
  for (size_t contact = 0; contact < contacts_.size(); ++contact) {
    if (!contactFlags[contact]) continue;
    const vector3_t& positionInBase = kinematics_->contactPositionInBase(contact);
    const vector3_t& velocityInBase = kinematics_->contactVelocityInBase(contact);
    std::vector<KalmanFilterMeasurement>& measurements = contacts_[contact].measurements;
    measurements[kRelativePositionMeasurement].measurement = rotation * positionInBase;
    measurements[kVelocityMeasurement].measurement = -(rotation * (angularVelocityInBase.cross(positionInBase) + velocityInBase));
    measurements[kHeightMeasurement].measurement(0) = config_.contactHeight;
    absl::Status corrected = filter_.correct(measurements);
    if (!corrected.ok()) {
      RETURN_IF_ERROR(readEstimate());
      return corrected;
    }
  }
  return readEstimate();
}

absl::Status KalmanBaseEstimator::readEstimate() {
  RETURN_IF_ERROR(filter_.getState(kBasePosition, positionEstimate_));
  RETURN_IF_ERROR(filter_.getState(kBaseVelocity, velocityEstimate_));
  basePosition_ = positionEstimate_.state;
  baseLinearVelocity_ = velocityEstimate_.state;
  return absl::OkStatus();
}

absl::StatusOr<vector3_t> KalmanBaseEstimator::contactPosition(size_t contact) const {
  if (contact >= contacts_.size()) {
    return absl::OutOfRangeError(absl::StrCat("KalmanBaseEstimator: no contact point ", contact));
  }
  ASSIGN_OR_RETURN(const KalmanFilterState state, filter_.getState(contacts_[contact].stateName));
  return vector3_t(state.state);
}

}  // namespace ocs2::humanoid::estimation
