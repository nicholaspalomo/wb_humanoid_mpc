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
/** The names of the filter's base channels and of its input, the world
 * acceleration a = R f + g. */
constexpr char kStatePosBaseRtWorldEwrtWorld[] = "base_position";
constexpr char kStateLinvelBaseRtWorldEwrtWorld[] = "base_velocity";
constexpr char kInputLinaccBaseRtWorldEwrtWorld[] = "imu_acceleration";
/** The state channels in their order: the base position, the base velocity,
 * then one per contact point. */
constexpr size_t kChannelPosBaseRtWorldEwrtWorld = 0;
constexpr size_t kChannelLinvelBaseRtWorldEwrtWorld = 1;
constexpr size_t kFirstContactChannel = 2;
/** The measurements of one stance contact point: relative position (3),
 * velocity (3) and height (1). */
constexpr Eigen::Index kMeasurementRowsPerContact = 7;
/** Their slots in ContactMeasurements::measurements. */
constexpr size_t kSlotMeasPosContactRtBaseEwrtWorld = 0;
constexpr size_t kSlotMeasLinvelBaseRtWorldEwrtWorld = 1;
constexpr size_t kSlotMeasPosContactRtWorldEwrtWorldZ = 2;
/** The standard deviations reset() starts the estimate with: the base [m],
 * [m/s], and its contact points [m]. */
constexpr scalar_t kInitialPositionNoiseStd = 0.01;
constexpr scalar_t kInitialVelocityNoiseStd = 0.1;
constexpr scalar_t kInitialContactNoiseStd = 0.01;

matrix_t identity() {
  return matrix_t::Identity(kDim, kDim);
}

absl::Status checkStd(scalar_t value, absl::string_view field) {
  if (!(std::isfinite(value) && value > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("kalman_state_estimator.", field,
                     " must be a positive standard deviation, got ", value));
  }
  return absl::OkStatus();
}

std::string contactStateName(size_t contact) {
  return absl::StrCat("contact_", contact, "_position");
}

}  // namespace

absl::Status KalmanBaseEstimator::validateConfig(const Config& config) {
  RETURN_IF_ERROR(
      checkStd(config.imuAccelerationNoise, "imu_acceleration_noise"));
  RETURN_IF_ERROR(
      checkStd(config.baseVelocityRandomWalk, "base_velocity_random_walk"));
  RETURN_IF_ERROR(
      checkStd(config.basePositionRandomWalk, "base_position_random_walk"));
  RETURN_IF_ERROR(
      checkStd(config.stanceContactRandomWalk, "stance_contact_random_walk"));
  RETURN_IF_ERROR(
      checkStd(config.swingContactRandomWalk, "swing_contact_random_walk"));
  RETURN_IF_ERROR(
      checkStd(config.kinematicPositionNoise, "kinematic_position_noise"));
  RETURN_IF_ERROR(
      checkStd(config.kinematicVelocityNoise, "kinematic_velocity_noise"));
  RETURN_IF_ERROR(checkStd(config.contactHeightNoise, "contact_height_noise"));
  if (!std::isfinite(config.contactHeight)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "kalman_state_estimator.contact_height must be finite, got ",
        config.contactHeight));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<KalmanBaseEstimator>>
KalmanBaseEstimator::Create(const Config& config,
                            std::unique_ptr<ContactKinematics> kinematics) {
  RETURN_IF_ERROR(validateConfig(config));
  if (kinematics == nullptr) {
    return absl::InvalidArgumentError(
        "KalmanBaseEstimator: no contact kinematics");
  }
  std::unique_ptr<KalmanBaseEstimator> estimator =
      absl::WrapUnique(new KalmanBaseEstimator(config, std::move(kinematics)));
  RETURN_IF_ERROR(estimator->build());
  return estimator;
}

KalmanBaseEstimator::KalmanBaseEstimator(
    const Config& config, std::unique_ptr<ContactKinematics> kinematics)
    : config_(config), kinematics_(std::move(kinematics)) {}

absl::Status KalmanBaseEstimator::build() {
  const size_t numContacts = kinematics_->numContacts();
  const matrix_t eye = identity();
  initialStates_.push_back({.name = kStatePosBaseRtWorldEwrtWorld,
                            .state = vector_t::Zero(kDim),
                            .stateCovariance = kInitialPositionNoiseStd *
                                               kInitialPositionNoiseStd * eye});
  initialStates_.push_back({.name = kStateLinvelBaseRtWorldEwrtWorld,
                            .state = vector_t::Zero(kDim),
                            .stateCovariance = kInitialVelocityNoiseStd *
                                               kInitialVelocityNoiseStd * eye});

  // The blocks that depend on dt are written by update(); their values here
  // only size them.
  processModel_.push_back(
      {.stateName = kStatePosBaseRtWorldEwrtWorld,
       .stateTransition = {{kStateLinvelBaseRtWorldEwrtWorld, eye}},
       .controlInputMatrix = {{kInputLinaccBaseRtWorldEwrtWorld, eye}},
       .processNoiseCovariance = eye});
  processModel_.push_back(
      {.stateName = kStateLinvelBaseRtWorldEwrtWorld,
       .controlInputMatrix = {{kInputLinaccBaseRtWorldEwrtWorld, eye}},
       .processNoiseCovariance = eye});
  inputs_.push_back(
      {.name = kInputLinaccBaseRtWorldEwrtWorld,
       .input = vector_t::Zero(kDim),
       .inputNoiseCovariance =
           config_.imuAccelerationNoise * config_.imuAccelerationNoise * eye});

  matrix_t heightRow = matrix_t::Zero(1, kDim);
  heightRow.rightCols<1>().setOnes();  // e_z^T
  for (size_t contact = 0; contact < numContacts; ++contact) {
    const std::string state = contactStateName(contact);
    initialStates_.push_back(
        {.name = state,
         .state = vector_t::Zero(kDim),
         .stateCovariance =
             kInitialContactNoiseStd * kInitialContactNoiseStd * eye});
    processModel_.push_back(
        {.stateName = state, .processNoiseCovariance = eye});
    ContactMeasurements measurements;
    measurements.stateName = state;
    measurements.measurements.push_back(
        {.name = absl::StrCat("contact_", contact, "_relative_position"),
         .measurement = vector_t::Zero(kDim),
         .measurementModel = {{state, eye},
                              {kStatePosBaseRtWorldEwrtWorld, -eye}},
         .measurementNoiseCovariance = config_.kinematicPositionNoise *
                                       config_.kinematicPositionNoise * eye});
    measurements.measurements.push_back(
        {.name = absl::StrCat("contact_", contact, "_velocity"),
         .measurement = vector_t::Zero(kDim),
         .measurementModel = {{kStateLinvelBaseRtWorldEwrtWorld, eye}},
         .measurementNoiseCovariance = config_.kinematicVelocityNoise *
                                       config_.kinematicVelocityNoise * eye});
    measurements.measurements.push_back(
        {.name = absl::StrCat("contact_", contact, "_height"),
         .measurement = vector_t::Constant(1, config_.contactHeight),
         .measurementModel = {{state, heightRow}},
         .measurementNoiseCovariance = matrix_t::Constant(
             1, 1, config_.contactHeightNoise * config_.contactHeightNoise)});
    contacts_.push_back(std::move(measurements));
  }

  // The maps are final from here on, so the pointers into them stay valid.
  transitionDtBlock_ =
      &processModel_[kChannelPosBaseRtWorldEwrtWorld]
           .stateTransition.find(kStateLinvelBaseRtWorldEwrtWorld)
           ->second;
  inputHalfDtSquaredBlock_ =
      &processModel_[kChannelPosBaseRtWorldEwrtWorld]
           .controlInputMatrix.find(kInputLinaccBaseRtWorldEwrtWorld)
           ->second;
  inputDtBlock_ =
      &processModel_[kChannelLinvelBaseRtWorldEwrtWorld]
           .controlInputMatrix.find(kInputLinaccBaseRtWorldEwrtWorld)
           ->second;

  RETURN_IF_ERROR(filter_.reset(initialStates_));
  RETURN_IF_ERROR(filter_.reserve(
      /*maxInputDim=*/kDim, /*maxMeasurementDim=*/kMeasurementRowsPerContact));
  return readEstimate();
}

absl::Status KalmanBaseEstimator::reset(
    const vector3_t& posBaseRtWorldEwrtWorld,
    const vector3_t& linvelBaseRtWorldEwrtWorld,
    const quaternion_t& quatBaseRtWorld,
    const vector_t& jointPositions,
    const vector_t& jointVelocities) {
  kinematics_->update(jointPositions, jointVelocities);
  const matrix3_t rotmatBaseRtWorld =
      quatBaseRtWorld.normalized().toRotationMatrix();
  initialStates_[kChannelPosBaseRtWorldEwrtWorld].state =
      posBaseRtWorldEwrtWorld;
  initialStates_[kChannelLinvelBaseRtWorldEwrtWorld].state =
      linvelBaseRtWorldEwrtWorld;
  for (size_t contact = 0; contact < contacts_.size(); ++contact) {
    initialStates_[kFirstContactChannel + contact].state =
        posBaseRtWorldEwrtWorld +
        rotmatBaseRtWorld * kinematics_->posContactRtBaseEwrtBase(contact);
  }
  RETURN_IF_ERROR(filter_.resetEstimate(initialStates_));
  return readEstimate();
}

absl::Status KalmanBaseEstimator::update(
    scalar_t dt,
    const quaternion_t& quatBaseRtWorld,
    const vector3_t& angvelBaseRtWorldEwrtBase,
    const vector3_t& specforceBaseRtWorldEwrtBase,
    const vector_t& jointPositions,
    const vector_t& jointVelocities,
    const std::vector<bool>& contactFlags) {
  if (!(std::isfinite(dt) && dt > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "KalmanBaseEstimator::update(): the time step must be positive, got ",
        dt, " s"));
  }
  if (contactFlags.size() != contacts_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "KalmanBaseEstimator::update(): ", contactFlags.size(),
        " contact flags for ", contacts_.size(), " contact points"));
  }
  if (static_cast<size_t>(jointPositions.size()) != kinematics_->numJoints() ||
      static_cast<size_t>(jointVelocities.size()) != kinematics_->numJoints()) {
    return absl::InvalidArgumentError(
        absl::StrCat("KalmanBaseEstimator::update(): joint vectors of ",
                     jointPositions.size(), " and ", jointVelocities.size(),
                     " entries for ", kinematics_->numJoints(), " joints"));
  }
  const matrix3_t rotmatBaseRtWorld =
      quatBaseRtWorld.normalized().toRotationMatrix();

  // The prediction: the world acceleration a = R f + g drives p and v.
  inputs_[0].input = rotmatBaseRtWorld * specforceBaseRtWorldEwrtBase +
                     kinematics_->gravityEwrtWorld();
  transitionDtBlock_->setIdentity();
  *transitionDtBlock_ *= dt;
  inputHalfDtSquaredBlock_->setIdentity();
  *inputHalfDtSquaredBlock_ *= 0.5 * dt * dt;
  inputDtBlock_->setIdentity();
  *inputDtBlock_ *= dt;
  processModel_[kChannelPosBaseRtWorldEwrtWorld]
      .processNoiseCovariance.setIdentity();
  processModel_[kChannelPosBaseRtWorldEwrtWorld].processNoiseCovariance *=
      config_.basePositionRandomWalk * config_.basePositionRandomWalk * dt;
  processModel_[kChannelLinvelBaseRtWorldEwrtWorld]
      .processNoiseCovariance.setIdentity();
  processModel_[kChannelLinvelBaseRtWorldEwrtWorld].processNoiseCovariance *=
      config_.baseVelocityRandomWalk * config_.baseVelocityRandomWalk * dt;
  for (size_t contact = 0; contact < contacts_.size(); ++contact) {
    const scalar_t randomWalk = contactFlags[contact]
                                    ? config_.stanceContactRandomWalk
                                    : config_.swingContactRandomWalk;
    matrix_t& noise =
        processModel_[kFirstContactChannel + contact].processNoiseCovariance;
    noise.setIdentity();
    noise *= randomWalk * randomWalk * dt;
  }
  RETURN_IF_ERROR(filter_.predict(processModel_, inputs_));

  // The corrections of the stance contact points, one at a time.
  kinematics_->update(jointPositions, jointVelocities);
  for (size_t contact = 0; contact < contacts_.size(); ++contact) {
    if (!contactFlags[contact]) continue;
    const vector3_t& posContactRtBaseEwrtBase =
        kinematics_->posContactRtBaseEwrtBase(contact);
    const vector3_t& linvelContactRtBaseEwrtBase =
        kinematics_->linvelContactRtBaseEwrtBase(contact);
    std::vector<KalmanFilterMeasurement>& measurements =
        contacts_[contact].measurements;
    measurements[kSlotMeasPosContactRtBaseEwrtWorld].measurement =
        rotmatBaseRtWorld * posContactRtBaseEwrtBase;
    measurements[kSlotMeasLinvelBaseRtWorldEwrtWorld].measurement =
        -(rotmatBaseRtWorld *
          (angvelBaseRtWorldEwrtBase.cross(posContactRtBaseEwrtBase) +
           linvelContactRtBaseEwrtBase));
    measurements[kSlotMeasPosContactRtWorldEwrtWorldZ].measurement(0) =
        config_.contactHeight;
    absl::Status corrected = filter_.correct(measurements);
    if (!corrected.ok()) {
      RETURN_IF_ERROR(readEstimate());
      return corrected;
    }
  }
  return readEstimate();
}

absl::Status KalmanBaseEstimator::readEstimate() {
  RETURN_IF_ERROR(filter_.getState(kStatePosBaseRtWorldEwrtWorld,
                                   filterStatePosBaseRtWorldEwrtWorld_));
  RETURN_IF_ERROR(filter_.getState(kStateLinvelBaseRtWorldEwrtWorld,
                                   filterStateLinvelBaseRtWorldEwrtWorld_));
  posBaseRtWorldEwrtWorld_ = filterStatePosBaseRtWorldEwrtWorld_.state;
  linvelBaseRtWorldEwrtWorld_ = filterStateLinvelBaseRtWorldEwrtWorld_.state;
  return absl::OkStatus();
}

absl::StatusOr<vector3_t> KalmanBaseEstimator::posContactRtWorldEwrtWorld(
    size_t contact) const {
  if (contact >= contacts_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("KalmanBaseEstimator: no contact point ", contact));
  }
  ASSIGN_OR_RETURN(const KalmanFilterState state,
                   filter_.getState(contacts_[contact].stateName));
  return vector3_t(state.state);
}

}  // namespace ocs2::humanoid::estimation
