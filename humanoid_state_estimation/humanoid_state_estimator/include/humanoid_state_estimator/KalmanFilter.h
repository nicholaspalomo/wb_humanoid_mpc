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
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include <ocs2_core/Types.h>

namespace ocs2::humanoid::estimation {

// Every field of the four argument structs below has a default member initializer, so that a designated initializer
// may leave out the optional ones, {.state_name = "foot_position", .Q_process_noise = Q}, without GCC's
// -Wmissing-field-initializers firing on every call site.

/**
 * One named block of the filter state, e.g. "base_position" or "left_foot_position", with its mean and covariance.
 *
 * reset() takes one per state channel, in the order the channels are stacked; getState() and getStates() return the
 * estimate in the same form, so a result is read by channel name rather than sliced out of a joint state vector.
 */
struct KalmanFilterState {
  std::string name{};
  /// x_hat_i: the mean of the channel.
  vector_t state{};
  /// P_ii: the covariance of the channel, size(state) x size(state), symmetric positive semi-definite.
  matrix_t P_state_estimate{};
};

/**
 * One named input to the process model, e.g. "imu_linear_acceleration", given to predict().
 *
 * An input drives every state channel whose KalmanFilterProcessModel lists it in B_control_input. Its noise reaches the
 * state through those same blocks, as B Q_input_noise B^T, so the process noise it causes is correlated across all the
 * channels it drives, as it physically is: accelerometer noise moves the position and the velocity of the base
 * together.
 */
struct KalmanFilterInput {
  std::string name{};
  /// u_m: the value of the input.
  vector_t input{};
  /// Covariance of the noise on `input`, size(input) x size(input), symmetric positive semi-definite. Empty when the
  /// input is noise-free.
  matrix_t Q_input_noise{};
};

/**
 * The process model of one state channel i over one predict() step:
 *
 *   x_i[k+1] = sum_j A_ij x_j[k] + sum_m B_im u_m[k] + w_i,    w_i ~ N(0, Q_i).
 *
 * A_state_transition is keyed by the name of the state channel j, and B_control_input by the name of the input m. A
 * block that is not listed takes its value from the identity transition x_i[k+1] = x_i[k]: the diagonal block A_ii is
 * the identity and every other block is zero. An integrator therefore names only what it integrates,
 *
 *   {.state_name = "base_position", .A_state_transition = {{"base_velocity", dt * I}}}    // p[k+1] = p[k] + dt v[k],
 *
 * a channel that must forget its past lists A_ii as zero, and a channel that predict() is given no model for keeps its
 * mean and its covariance.
 */
struct KalmanFilterProcessModel {
  std::string state_name{};
  /// A_ij, size(x_i) x size(x_j), keyed by the state channel j.
  absl::flat_hash_map<std::string, matrix_t> A_state_transition{};
  /// B_im, size(x_i) x size(u_m), keyed by the input m.
  absl::flat_hash_map<std::string, matrix_t> B_control_input{};
  /// Q_i: the additive process noise covariance, size(x_i) x size(x_i), symmetric positive semi-definite. Empty for none.
  matrix_t Q_process_noise{};
};

/**
 * One named measurement, e.g. "left_foot_relative_position", given to correct():
 *
 *   z = sum_j H_j x_j + v,    v ~ N(0, R).
 *
 * H_measurement_model is keyed by the name of every state channel the measurement observes; a channel it does not list
 * does not enter the measurement. The noise of different measurements is independent.
 */
struct KalmanFilterMeasurement {
  std::string name{};
  /// z: the measured value.
  vector_t measurement{};
  /// H_j, size(z) x size(x_j), keyed by the state channel j. At least one channel.
  absl::flat_hash_map<std::string, matrix_t> H_measurement_model{};
  /// R: the measurement noise covariance, size(z) x size(z), symmetric positive semi-definite. Required: a noise-free
  /// measurement says so with an explicit zero rather than by omission.
  matrix_t R_measurement_noise{};
};

/**
 * A linear Kalman filter whose state, inputs and measurements are organized in named channels.
 *
 * The caller never assembles or slices a block matrix. States are declared by name in reset(), the process model is
 * written per state channel with blocks keyed by state and input names, measurements carry their observation blocks
 * keyed by state names, and the estimate is read back per channel with getState(), getStates() and getCovariance().
 * Internally the filter keeps the joint mean and the joint covariance, including every cross-covariance between
 * channels. They are what let a position measurement correct the velocity through p[k+1] = p[k] + dt v[k], and what
 * let a leg-kinematics measurement of foot minus base correct both the foot and the base.
 *
 * Example: the base of a legged robot, driven by a world-frame IMU acceleration and observed through leg kinematics.
 *
 *   RETURN_IF_ERROR(filter.reset({{.name = "base_position", .state = p0, .P_state_estimate = P_p0},
 *                                 {.name = "base_velocity", .state = v0, .P_state_estimate = P_v0},
 *                                 {.name = "foot_position", .state = f0, .P_state_estimate = P_f0}}));
 *   RETURN_IF_ERROR(filter.predict(
 *       {{.state_name = "base_position",
 *         .A_state_transition = {{"base_velocity", dt * I}},
 *         .B_control_input = {{"imu_acceleration", 0.5 * dt * dt * I}}},
 *        {.state_name = "base_velocity", .B_control_input = {{"imu_acceleration", dt * I}}},
 *        {.state_name = "foot_position", .Q_process_noise = q_foot * I}},
 *       {{.name = "imu_acceleration", .input = a_world, .Q_input_noise = sigma_a * sigma_a * I}}));
 *   RETURN_IF_ERROR(filter.correct({{.name = "foot_relative_position",
 *                                    .measurement = R_world_base * foot_position_in_base,
 *                                    .H_measurement_model = {{"foot_position", I}, {"base_position", -I}},
 *                                    .R_measurement_noise = r * I}}));
 *   ASSIGN_OR_RETURN(const KalmanFilterState base_velocity, filter.getState("base_velocity"));
 *
 * The models are passed to every step, so they may change from one step to the next: a varying time step, a rotation
 * that maps a body-frame quantity into the world, or a foot whose process noise grows while it swings.
 *
 * Real-time use. predict() and correct() compute in storage the filter keeps between steps. reset() sizes it and a step
 * grows it only when it needs more room than any step before, so a steady run of steps makes no heap allocation; call
 * reserve() after reset() with the largest input and measurement dimensions to make the first steps allocation-free
 * too. The caller's side stays allocation-free when it builds its model, input and measurement vectors once and
 * overwrites their values in place every step, and reads the estimate into a KalmanFilterState it owns:
 *
 *   inputs[0].input = a_world;                                          // same size: no allocation
 *   measurements[0].measurement = R_world_base * foot_position_in_base;
 *   measurements[0].H_measurement_model.at("base_position") = -I;      // existing key: no allocation
 *   RETURN_IF_ERROR(filter.predict(process_model, inputs));
 *   RETURN_IF_ERROR(filter.correct(measurements));
 *   RETURN_IF_ERROR(filter.getState("base_velocity", base_velocity));
 *
 * Every call validates its arguments before it changes anything, and a call that fails leaves the estimate exactly as
 * it was, so a sensor glitch that produces a non-finite reading is rejected rather than absorbed into the estimate.
 * Error messages are formatted only on failure. A covariance passed to a step gets the checks a step can afford without
 * allocating: finite, symmetric, non-negative variances and correlations within [-1, 1]. Those are necessary for
 * positive semi-definiteness but not sufficient; the full eigenvalue check runs on the covariances given to reset().
 * The class is not thread-safe.
 */
class KalmanFilter {
 public:
  /** A filter over the given state channels, see reset(). */
  static absl::StatusOr<std::unique_ptr<KalmanFilter>> Create(absl::Span<const KalmanFilterState> initial_states);

  /** A filter with no state channels. predict() and correct() fail until reset() declares them. */
  KalmanFilter() = default;
  virtual ~KalmanFilter() = default;

  KalmanFilter(const KalmanFilter&) = delete;
  KalmanFilter& operator=(const KalmanFilter&) = delete;

  /**
   * Declares the state channels, stacked in the order given, and sets the estimate to their means and covariances,
   * uncorrelated with each other. Replaces any channels declared before.
   *
   * Fails without changing the filter when the list is empty, or when a channel has an empty or repeated name, an empty
   * or non-finite state, or a covariance of the wrong size or that is not symmetric positive semi-definite.
   */
  virtual absl::Status reset(absl::Span<const KalmanFilterState> initial_states);

  /**
   * Sizes the storage of predict() and correct() for steps whose inputs add up to at most `max_input_dim` entries and
   * whose measurements add up to at most `max_measurement_dim` rows, so that no step up to that size allocates. The
   * storage only grows, and it carries over to the channels of a later reset(). Fails on a negative dimension.
   */
  absl::Status reserve(Eigen::Index max_input_dim, Eigen::Index max_measurement_dim);

  /**
   * Propagates the estimate over one step:
   *
   *   x_hat <- A x_hat + B u,    P <- A P A^T + B Q_u B^T + Q,
   *
   * where A, B and Q are assembled from the per-channel process models, as KalmanFilterProcessModel describes, and Q_u
   * from the noise of the inputs. Each state channel takes at most one process model. An input that no
   * B_control_input block names has no effect.
   *
   * Fails without changing the estimate before reset(), and when a model or a block names a state channel or an input
   * that does not exist, when a block or a covariance has the wrong size, when a value is not finite, or when a
   * covariance is not symmetric or has a negative variance or a correlation outside [-1, 1].
   */
  virtual absl::Status predict(absl::Span<const KalmanFilterProcessModel> process_model, absl::Span<const KalmanFilterInput> inputs);

  /**
   * Corrects the estimate with all the measurements in one joint update, which equals correcting with them one after
   * the other because their noise is independent:
   *
   *   S = H P H^T + R,    K = P H^T S^-1,    x_hat <- x_hat + K (z - H x_hat),
   *   P <- (I - K H) P (I - K H)^T + K R K^T.
   *
   * The covariance is updated in Joseph form, which keeps it symmetric positive semi-definite under rounding. An empty
   * list leaves the estimate unchanged, e.g. when no foot is in contact.
   *
   * Fails without changing the estimate before reset(), and when a measurement has an empty or repeated name, an empty
   * or non-finite value, no observation block, a block that names an unknown state channel or has the wrong size, or a
   * missing or invalid noise covariance (checked as in predict()). It also fails when S is not numerically positive
   * definite, i.e. when a Cholesky pivot of S falls below machine epsilon times its largest variance, which takes a
   * noise-free measurement of a direction the estimate is already certain of.
   */
  virtual absl::Status correct(absl::Span<const KalmanFilterMeasurement> measurements);

  /** The mean and covariance of one state channel. NotFound for a name reset() did not declare. */
  absl::StatusOr<KalmanFilterState> getState(absl::string_view name) const;

  /**
   * The same, written into a KalmanFilterState the caller owns. Its name, state and covariance keep their storage, so
   * reading a channel every step allocates only the first time. NotFound, leaving `state` as it was, for a name reset()
   * did not declare.
   */
  absl::Status getState(absl::string_view name, KalmanFilterState& state) const;

  /** The mean and covariance of every state channel, in the order reset() declared them. Empty before reset(). */
  std::vector<KalmanFilterState> getStates() const;

  /**
   * The covariance block P_ij = E[(x_i - x_hat_i) (x_j - x_hat_j)^T] between two state channels, size(x_i) x size(x_j).
   * getCovariance(name, name) is the covariance of that channel. NotFound for a name reset() did not declare.
   */
  absl::StatusOr<matrix_t> getCovariance(absl::string_view row_state_name, absl::string_view col_state_name) const;

 private:
  /// Where a state channel lives in the stacked state, and its position in state_names_.
  struct ChannelBlock {
    Eigen::Index offset = 0;
    Eigen::Index size = 0;
    size_t index = 0;
  };

  /**
   * The storage predict() and correct() compute in, kept between steps so that a step does not allocate. The blocks
   * sized by the state are set by reset(); those sized by the inputs or the measurements of a step only grow, and a step
   * uses their leading rows and columns.
   */
  struct Workspace {
    Eigen::Index input_capacity = 0;
    Eigen::Index measurement_capacity = 0;
    /// Offset of each input of the current predict() in the stacked input.
    std::vector<Eigen::Index> input_offsets;
    /// Whether the current predict() has seen a process model for each state channel, to reject a second one.
    std::vector<char> state_has_process_model;

    // predict(): x_hat <- A x_hat + B u, P <- A P A^T + B Q_u B^T + Q.
    matrix_t A_state_transition;
    matrix_t B_control_input;
    matrix_t Q_process_noise;
    vector_t u_control_input;
    matrix_t Q_input_noise;
    matrix_t BQ_input_noise;
    matrix_t AP;

    // correct(): S = H P H^T + R, K^T = S^-1 H P, and the Joseph-form covariance update.
    matrix_t H_measurement_model;
    matrix_t R_measurement_noise;
    vector_t innovation;
    matrix_t PHt;
    matrix_t S_innovation_covariance;
    matrix_t K_transpose;
    matrix_t KR;
    matrix_t I_minus_KH;
    matrix_t I_minus_KH_P;

    /// The estimate a step computes, swapped with the current one once the step has succeeded.
    vector_t x_hat_next;
    matrix_t P_next;
  };

  void fillState(absl::string_view name, const ChannelBlock& block, KalmanFilterState& state) const;
  std::optional<ChannelBlock> findStateChannel(absl::string_view name) const;
  absl::Status unknownStateChannelError(absl::StatusCode code, absl::string_view name, absl::string_view context) const;
  absl::Status checkInitialized(absl::string_view method) const;
  /// Sizes every workspace block for the current state channels and the current capacities.
  void resizeWorkspace();
  /// Grows the input and measurement blocks of the workspace to at least the given dimensions.
  void growWorkspace(Eigen::Index input_dim, Eigen::Index measurement_dim);

  /// The state channel names, in the order reset() declared them.
  std::vector<std::string> state_names_;
  absl::flat_hash_map<std::string, ChannelBlock> state_channels_;

  /// The joint mean, the channels stacked in the order of state_names_.
  vector_t x_hat_state_estimate_;
  /// The joint covariance, including the cross-covariances between channels.
  matrix_t P_state_estimate_;

  Workspace workspace_;
};

}  // namespace ocs2::humanoid::estimation
