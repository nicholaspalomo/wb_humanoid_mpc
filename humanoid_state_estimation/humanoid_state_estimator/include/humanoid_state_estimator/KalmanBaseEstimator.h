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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"
#include "humanoid_state_estimator/ContactKinematics.h"
#include "humanoid_state_estimator/KalmanFilter.h"

namespace ocs2::humanoid::estimation {

/**
 * The base position and velocity of a legged robot from its IMU and its leg
 * kinematics: a linear Kalman filter (KalmanFilter) whose state is the base
 * position p and velocity v in the world and the world position c_i of every
 * contact point, the formulation of Bloesch et al. (2013) with the orientation
 * taken from the IMU, as the MIT Cheetah's linear estimator does.
 *
 *   predict:  p+ = p + dt v + dt^2/2 a,  v+ = v + dt a,  c_i+ = c_i,
 *             a = R f + g,
 *   correct, per contact i in stance:
 *     c_i - p = R p_rel_i                    (where the leg kinematics put
 *                                             the contact point)
 *     v = -R (omega x p_rel_i + v_rel_i)     (the stance contact point does
 *                                             not move)
 *     e_z^T c_i = h_i                        (the surface height the caller
 *                                             gives contact i,
 *                                             ContactEstimationInputs; no
 *                                             such row without one)
 *
 * with R the IMU's orientation (base to world), omega the base's angular
 * velocity and f the accelerometer's specific force R^T (a - g), both in the
 * base frame, g the gravity, and p_rel_i, v_rel_i the contact point's position
 * and velocity relative to the base (ContactKinematics). The accelerometer
 * noise enters as input noise through B = [dt^2/2 I; dt I], so that it
 * correlates p and v as it physically does; the base position and velocity and
 * every contact point take a random walk, the contact points a small one in
 * stance (slip) and a large one in swing, so that a lifted foot is forgotten
 * and placed anew by the kinematics when it touches down. Each stance contact
 * is a correct() of its own, which equals the joint update because the
 * measurements' noise is independent.
 *
 * Optionally (Config::contactTrustRampDuration > 0) a stance contact is
 * trusted less near a planned liftoff or touchdown: with d the time to the
 * nearer planned switch the caller gives (ContactEstimationInputs), its stance
 * random walk and its velocity and height noise standard deviations are
 * multiplied by contactTrustScale(d, ...) (ContactTrustRamp.h), 1 outside the
 * window. Off, the noise is never rewritten for it.
 *
 * The orientation is not estimated: it is the IMU's (on a simulator, the
 * truth). The base height is observable through the contact heights only: a
 * stance contact without a surface height keeps the height it had at
 * touch-down, and with no height at all the position's z drifts.
 *
 * REALTIME. The models, inputs and measurements are built once by Create() and
 * the filter's workspace is reserved; update() overwrites their values in place
 * and allocates nothing. Not thread-safe.
 */
class KalmanBaseEstimator {
 public:
  /** The noise model (humanoid_mpc_config.KalmanStateEstimatorConfig); every
   * standard deviation positive. */
  struct Config {
    // LINT.IfChange(kalman_state_estimator_defaults)
    static constexpr scalar_t kDefaultImuAccelerationNoise = 0.5;
    static constexpr scalar_t kDefaultBaseVelocityRandomWalk = 0.1;
    static constexpr scalar_t kDefaultBasePositionRandomWalk = 0.001;
    static constexpr scalar_t kDefaultStanceContactRandomWalk = 0.01;
    static constexpr scalar_t kDefaultSwingContactRandomWalk = 10.0;
    static constexpr scalar_t kDefaultKinematicPositionNoise = 0.005;
    static constexpr scalar_t kDefaultKinematicVelocityNoise = 0.05;
    static constexpr scalar_t kDefaultContactHeightNoise = 0.01;
    static constexpr scalar_t kNomPosContactRtWorldEwrtWorldZ = 0.0;
    static constexpr scalar_t kDefaultContactTrustRampDuration = 0.0;
    static constexpr scalar_t kDefaultContactTrustRampMaxScale = 1.0;
    // LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/kalman_state_estimator_config.proto:kalman_state_estimator_defaults)

    /** [m/s^2] The accelerometer's noise on one sample
     * (imu_acceleration_noise). */
    scalar_t imuAccelerationNoise = kDefaultImuAccelerationNoise;
    /** [m/s/sqrt(s)] (base_velocity_random_walk). */
    scalar_t baseVelocityRandomWalk = kDefaultBaseVelocityRandomWalk;
    /** [m/sqrt(s)] (base_position_random_walk). */
    scalar_t basePositionRandomWalk = kDefaultBasePositionRandomWalk;
    /** [m/sqrt(s)] A contact point in stance (stance_contact_random_walk). */
    scalar_t stanceContactRandomWalk = kDefaultStanceContactRandomWalk;
    /** [m/sqrt(s)] A contact point in swing (swing_contact_random_walk). */
    scalar_t swingContactRandomWalk = kDefaultSwingContactRandomWalk;
    /** [m] (kinematic_position_noise). */
    scalar_t kinematicPositionNoise = kDefaultKinematicPositionNoise;
    /** [m/s] (kinematic_velocity_noise). */
    scalar_t kinematicVelocityNoise = kDefaultKinematicVelocityNoise;
    /** [m] (contact_height_noise). */
    scalar_t contactHeightNoise = kDefaultContactHeightNoise;
    /** [m] The flat ground's height (nom_pos_contact_rt_world_ewrt_world_z);
     * any finite value. The estimator does not read it: the robot process
     * gives it to every contact while no multi-contact plan names a surface
     * (FlatGroundSurfaceHeights). The contact points are the MPC's sole and
     * pad centers and point offsets (ContactPointSpec), so 0 is flat ground at
     * z = 0. */
    scalar_t nomPosContactRtWorldEwrtWorldZ = kNomPosContactRtWorldEwrtWorldZ;
    /** [s] The contact-trust ramp's window around a planned contact switch
     * (contact_trust_ramp_duration), >= 0; 0 turns the ramp off, and update()
     * then rewrites no noise for it (ContactTrustRamp.h). */
    scalar_t contactTrustRampDuration = kDefaultContactTrustRampDuration;
    /** The ramp's factor on the standard deviations at the switch
     * (contact_trust_ramp_max_scale), >= 1. */
    scalar_t contactTrustRampMaxScale = kDefaultContactTrustRampMaxScale;
  };

  /**
   * The refusals Create() makes on `config`, without building an estimator:
   * InvalidArgument naming the field by its task file name
   * ("kalman_state_estimator.imu_acceleration_noise must be a positive standard
   * deviation, got -1").
   */
  static absl::Status validateConfig(const Config& config);

  /**
   * An estimator over the contact points of `kinematics`, at the origin with
   * zero velocity until reset(). The errors of validateConfig().
   */
  static absl::StatusOr<std::unique_ptr<KalmanBaseEstimator>> Create(
      const Config& config, std::unique_ptr<ContactKinematics> kinematics);

  KalmanBaseEstimator(const KalmanBaseEstimator&) = delete;
  KalmanBaseEstimator& operator=(const KalmanBaseEstimator&) = delete;
  KalmanBaseEstimator(KalmanBaseEstimator&&) = delete;
  KalmanBaseEstimator& operator=(KalmanBaseEstimator&&) = delete;
  ~KalmanBaseEstimator() = default;

  /**
   * Restarts the estimate at the base position `posBaseRtWorldEwrtWorld` and
   * velocity `linvelBaseRtWorldEwrtWorld` (world), with the contact points
   * where the kinematics of `jointPositions` put them for the base orientation
   * `quatBaseRtWorld` (base to world). The joint vectors are in the robot's
   * joint order. Allocates nothing (KalmanFilter::resetEstimate()), so the
   * realtime thread may call it after a discontinuity.
   */
  absl::Status reset(const vector3_t& posBaseRtWorldEwrtWorld,
                     const vector3_t& linvelBaseRtWorldEwrtWorld,
                     const quaternion_t& quatBaseRtWorld,
                     const vector_t& jointPositions,
                     const vector_t& jointVelocities);

  /**
   * Advances the estimate by `dt` [s] (> 0) with the IMU's orientation (base to
   * world), angular velocity and specific force (base frame), and corrects it
   * with the kinematics of every contact point `contactFlags` (one per contact)
   * has in stance, and with the surface height `contactInputs` gives each of
   * them (none for a contact without one); with the contact-trust ramp on,
   * also with the planned switch times it gives. A non-finite IMU reading, a
   * bad dt, or flags or inputs of another number than the contact points fail
   * before anything changes; a non-finite joint reading fails the correction
   * after the prediction was applied, so the estimate is then the IMU's
   * prediction alone. Allocates nothing.
   */
  absl::Status update(scalar_t dt,
                      const quaternion_t& quatBaseRtWorld,
                      const vector3_t& angvelBaseRtWorldEwrtBase,
                      const vector3_t& specforceBaseRtWorldEwrtBase,
                      const vector_t& jointPositions,
                      const vector_t& jointVelocities,
                      const std::vector<bool>& contactFlags,
                      const ContactEstimationInputs& contactInputs);

  /** [m] The base position in the world. */
  const vector3_t& posBaseRtWorldEwrtWorld() const {
    return posBaseRtWorldEwrtWorld_;
  }
  /** [m/s] The base velocity in the world. */
  const vector3_t& linvelBaseRtWorldEwrtWorld() const {
    return linvelBaseRtWorldEwrtWorld_;
  }
  /** [m] The estimated world positions of the contact points, in contact
   * order, as of the last reset() or update(). Allocates nothing; the span is
   * valid as long as the estimator. */
  absl::Span<const vector3_t> estPosContactsRtWorldEwrtWorld() const {
    return estPosContactsRtWorldEwrtWorld_;
  }
  /** Returns contact `contact`'s estimated position [m] in the world.
   * Allocates: for tests and tools, not the realtime thread. */
  absl::StatusOr<vector3_t> posContactRtWorldEwrtWorld(size_t contact) const;
  /** Returns the 3 x 3 covariance of contact `contact`'s estimated world
   * position. Allocates: for tests and tools, not the realtime thread. */
  absl::StatusOr<matrix_t> contactStateCovariance(size_t contact) const;
  size_t numContacts() const { return kinematics_->numContacts(); }
  const ContactKinematics& kinematics() const { return *kinematics_; }

 private:
  /** The three measurements of one stance contact point and the blocks update()
   * overwrites. */
  struct ContactMeasurements {
    /** Relative position, velocity and height, in that order. */
    std::vector<KalmanFilterMeasurement> measurements;
    /** The contact's state channel, "contact_<i>_position". */
    std::string stateName;
  };

  KalmanBaseEstimator(const Config& config,
                      std::unique_ptr<ContactKinematics> kinematics);

  absl::Status build();
  absl::Status readEstimate();
  /** Rewrites contact `contact`'s velocity and height measurement noise in
   * place as `scale` times the configured standard deviations (the
   * contact-trust ramp). Allocates nothing. */
  void scaleContactMeasurementNoise(size_t contact, scalar_t scale);
  /** Writes the prediction's input, the world acceleration of the specific
   * force `specforceBaseRtWorldEwrtBase` rotated by `rotmatBaseRtWorld`, and
   * the base channels' transition, input and process noise blocks for the
   * step `dt`. Allocates nothing. */
  void setBaseProcessModel(scalar_t dt,
                           const matrix3_t& rotmatBaseRtWorld,
                           const vector3_t& specforceBaseRtWorldEwrtBase);
  /** Returns the contact-trust ramp's scale of contact `contact`:
   * contactTrustScale() of the nearer planned switch of `contactInputs`. */
  scalar_t contactTrustScaleOf(
      size_t contact, const ContactEstimationInputs& contactInputs) const;
  /** Writes contact `contact`'s process noise for the step `dt` from its
   * random walk `randomWalk` [m/sqrt(s)]. Allocates nothing. */
  void setContactProcessNoise(size_t contact, scalar_t randomWalk, scalar_t dt);
  /** Corrects the filter with stance contact `contact`'s relative position and
   * velocity rows, and its height row where `contactInputs` gives the surface
   * height, from the kinematics of the last ContactKinematics::update().
   * Returns the filter's error. Allocates nothing. */
  absl::Status correctContact(size_t contact,
                              const matrix3_t& rotmatBaseRtWorld,
                              const vector3_t& angvelBaseRtWorldEwrtBase,
                              const ContactEstimationInputs& contactInputs);

  const Config config_;
  std::unique_ptr<ContactKinematics> kinematics_;
  KalmanFilter filter_;

  std::vector<KalmanFilterState> initialStates_;
  std::vector<KalmanFilterProcessModel> processModel_;
  std::vector<KalmanFilterInput> inputs_;
  std::vector<ContactMeasurements> contacts_;
  // Pointers into the blocks of processModel_ that change with dt and the
  // contact state: the maps are not changed after build(), so the pointers stay
  // valid. They hold dt I (the base position's block of the base velocity in
  // the transition), dt^2/2 I and dt I (the base position's and the base
  // velocity's blocks of the acceleration input).
  matrix_t* absl_nullable transitionDtBlock_ = nullptr;
  matrix_t* absl_nullable inputHalfDtSquaredBlock_ = nullptr;
  matrix_t* absl_nullable inputDtBlock_ = nullptr;

  KalmanFilterState filterStatePosBaseRtWorldEwrtWorld_;
  KalmanFilterState filterStateLinvelBaseRtWorldEwrtWorld_;
  vector3_t posBaseRtWorldEwrtWorld_ = vector3_t::Zero();
  vector3_t linvelBaseRtWorldEwrtWorld_ = vector3_t::Zero();
  // One per contact point: the filter's channel, read in place by
  // readEstimate() (sized by build(), so that reading allocates nothing), and
  // its position.
  std::vector<KalmanFilterState> filterStateContacts_;
  std::vector<vector3_t> estPosContactsRtWorldEwrtWorld_;
};

}  // namespace ocs2::humanoid::estimation
