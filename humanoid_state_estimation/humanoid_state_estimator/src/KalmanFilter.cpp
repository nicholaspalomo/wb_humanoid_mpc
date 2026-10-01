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

#include "humanoid_state_estimator/KalmanFilter.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::estimation {
namespace {

/// Relative tolerance of the symmetry and positive semi-definiteness checks on a covariance, scaled by its largest
/// entry so that it applies alike to variances in square millimeters and in square radians.
constexpr scalar_t kCovarianceTolerance = 1e-9;

matrix_t symmetrized(const matrix_t& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

std::string joinedNamesOrNone(absl::Span<const std::string> names) {
  return names.empty() ? std::string("none") : absl::StrJoin(names, ", ");
}

template <typename Derived>
absl::Status checkFinite(const Eigen::MatrixBase<Derived>& value, absl::string_view description) {
  if (!value.allFinite()) {
    return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is not finite"));
  }
  return absl::OkStatus();
}

absl::Status checkBlock(const matrix_t& block, Eigen::Index rows, Eigen::Index cols, absl::string_view description) {
  if (block.rows() != rows || block.cols() != cols) {
    return absl::InvalidArgumentError(
        absl::StrCat("[KalmanFilter] ", description, " is ", block.rows(), "x", block.cols(), ", expected ", rows, "x", cols));
  }
  return checkFinite(block, description);
}

/// A covariance must be dim x dim (dim > 0), finite, symmetric and positive semi-definite, up to rounding.
absl::Status checkCovariance(const matrix_t& covariance, Eigen::Index dim, absl::string_view description) {
  RETURN_IF_ERROR(checkBlock(covariance, dim, dim, description));
  const scalar_t tolerance = kCovarianceTolerance * covariance.cwiseAbs().maxCoeff();
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > tolerance) {
    return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is not symmetric"));
  }
  const Eigen::SelfAdjointEigenSolver<matrix_t> eigen_solver(covariance, Eigen::EigenvaluesOnly);
  if (eigen_solver.info() != Eigen::Success || eigen_solver.eigenvalues().minCoeff() < -tolerance) {
    return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description,
                                                   " is not positive semi-definite: its smallest eigenvalue is ",
                                                   eigen_solver.eigenvalues().minCoeff()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<KalmanFilter>> KalmanFilter::Create(absl::Span<const KalmanFilterState> initial_states) {
  auto filter = std::make_unique<KalmanFilter>();
  RETURN_IF_ERROR(filter->reset(initial_states));
  return filter;
}

absl::Status KalmanFilter::reset(absl::Span<const KalmanFilterState> initial_states) {
  if (initial_states.empty()) {
    return absl::InvalidArgumentError("[KalmanFilter] reset() needs at least one state channel");
  }

  std::vector<std::string> state_names;
  state_names.reserve(initial_states.size());
  absl::flat_hash_map<std::string, ChannelBlock> state_channels;
  Eigen::Index state_dim = 0;
  for (const KalmanFilterState& channel : initial_states) {
    if (channel.name.empty()) {
      return absl::InvalidArgumentError("[KalmanFilter] a state channel has an empty name");
    }
    const std::string description = absl::StrCat("state channel '", channel.name, "'");
    if (channel.state.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is empty"));
    }
    RETURN_IF_ERROR(checkFinite(channel.state, absl::StrCat("the state of ", description)));
    RETURN_IF_ERROR(checkCovariance(channel.P_state_estimate, channel.state.size(), absl::StrCat("P_state_estimate of ", description)));
    if (!state_channels.try_emplace(channel.name, ChannelBlock{state_dim, channel.state.size()}).second) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is declared twice"));
    }
    state_names.push_back(channel.name);
    state_dim += channel.state.size();
  }

  vector_t x_hat_state_estimate(state_dim);
  matrix_t P_state_estimate = matrix_t::Zero(state_dim, state_dim);
  for (const KalmanFilterState& channel : initial_states) {
    const ChannelBlock& block = state_channels.at(channel.name);
    x_hat_state_estimate.segment(block.offset, block.size) = channel.state;
    P_state_estimate.block(block.offset, block.offset, block.size, block.size) = symmetrized(channel.P_state_estimate);
  }

  state_names_ = std::move(state_names);
  state_channels_ = std::move(state_channels);
  x_hat_state_estimate_ = std::move(x_hat_state_estimate);
  P_state_estimate_ = std::move(P_state_estimate);
  return absl::OkStatus();
}

absl::Status KalmanFilter::predict(absl::Span<const KalmanFilterProcessModel> process_model, absl::Span<const KalmanFilterInput> inputs) {
  RETURN_IF_ERROR(checkInitialized("predict"));

  // The inputs, stacked in the order given, and the block-diagonal covariance of their noise.
  std::vector<std::string> input_names;
  input_names.reserve(inputs.size());
  absl::flat_hash_map<std::string, ChannelBlock> input_channels;
  Eigen::Index input_dim = 0;
  for (const KalmanFilterInput& input : inputs) {
    if (input.name.empty()) {
      return absl::InvalidArgumentError("[KalmanFilter] an input has an empty name");
    }
    const std::string description = absl::StrCat("input '", input.name, "'");
    if (input.input.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is empty"));
    }
    RETURN_IF_ERROR(checkFinite(input.input, absl::StrCat("the value of ", description)));
    if (input.Q_input_noise.size() != 0) {
      RETURN_IF_ERROR(checkCovariance(input.Q_input_noise, input.input.size(), absl::StrCat("Q_input_noise of ", description)));
    }
    if (!input_channels.try_emplace(input.name, ChannelBlock{input_dim, input.input.size()}).second) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is given twice"));
    }
    input_names.push_back(input.name);
    input_dim += input.input.size();
  }
  vector_t u_control_input(input_dim);
  matrix_t Q_input_noise = matrix_t::Zero(input_dim, input_dim);
  for (const KalmanFilterInput& input : inputs) {
    const ChannelBlock& block = input_channels.at(input.name);
    u_control_input.segment(block.offset, block.size) = input.input;
    if (input.Q_input_noise.size() != 0) {
      Q_input_noise.block(block.offset, block.offset, block.size, block.size) = symmetrized(input.Q_input_noise);
    }
  }

  // The joint matrices start from the identity transition and the per-channel models overwrite their blocks.
  const Eigen::Index state_dim = x_hat_state_estimate_.size();
  matrix_t A_state_transition = matrix_t::Identity(state_dim, state_dim);
  matrix_t B_control_input = matrix_t::Zero(state_dim, input_dim);
  matrix_t Q_process_noise = matrix_t::Zero(state_dim, state_dim);
  absl::flat_hash_set<std::string> modeled_state_names;
  for (const KalmanFilterProcessModel& model : process_model) {
    const std::optional<ChannelBlock> row = findStateChannel(model.state_name);
    if (!row.has_value()) {
      return unknownStateChannelError(absl::StatusCode::kInvalidArgument, model.state_name, "a process model");
    }
    if (!modeled_state_names.insert(model.state_name).second) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] state channel '", model.state_name, "' is given two process models"));
    }
    for (const std::pair<const std::string, matrix_t>& entry : model.A_state_transition) {
      const std::optional<ChannelBlock> col = findStateChannel(entry.first);
      if (!col.has_value()) {
        return unknownStateChannelError(absl::StatusCode::kInvalidArgument, entry.first,
                                        absl::StrCat("A_state_transition of state channel '", model.state_name, "'"));
      }
      RETURN_IF_ERROR(checkBlock(entry.second, row->size, col->size,
                                 absl::StrCat("A_state_transition block ('", model.state_name, "', '", entry.first, "')")));
      A_state_transition.block(row->offset, col->offset, row->size, col->size) = entry.second;
    }
    for (const std::pair<const std::string, matrix_t>& entry : model.B_control_input) {
      const absl::flat_hash_map<std::string, ChannelBlock>::const_iterator input = input_channels.find(entry.first);
      if (input == input_channels.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("[KalmanFilter] B_control_input of state channel '", model.state_name, "' names input '", entry.first,
                         "', which predict() was not given; the inputs are: ", joinedNamesOrNone(input_names)));
      }
      RETURN_IF_ERROR(checkBlock(entry.second, row->size, input->second.size,
                                 absl::StrCat("B_control_input block ('", model.state_name, "', '", entry.first, "')")));
      B_control_input.block(row->offset, input->second.offset, row->size, input->second.size) = entry.second;
    }
    if (model.Q_process_noise.size() != 0) {
      RETURN_IF_ERROR(
          checkCovariance(model.Q_process_noise, row->size, absl::StrCat("Q_process_noise of state channel '", model.state_name, "'")));
      Q_process_noise.block(row->offset, row->offset, row->size, row->size) = model.Q_process_noise;
    }
  }

  vector_t x_hat_state_estimate = A_state_transition * x_hat_state_estimate_ + B_control_input * u_control_input;
  matrix_t P_state_estimate = symmetrized(A_state_transition * P_state_estimate_ * A_state_transition.transpose() +
                                          B_control_input * Q_input_noise * B_control_input.transpose() + Q_process_noise);
  RETURN_IF_ERROR(checkFinite(x_hat_state_estimate, "the predicted state"));
  RETURN_IF_ERROR(checkFinite(P_state_estimate, "the predicted covariance"));

  x_hat_state_estimate_ = std::move(x_hat_state_estimate);
  P_state_estimate_ = std::move(P_state_estimate);
  return absl::OkStatus();
}

absl::Status KalmanFilter::correct(absl::Span<const KalmanFilterMeasurement> measurements) {
  RETURN_IF_ERROR(checkInitialized("correct"));
  if (measurements.empty()) {
    return absl::OkStatus();
  }

  std::vector<std::string> measurement_names;
  measurement_names.reserve(measurements.size());
  absl::flat_hash_set<std::string> unique_measurement_names;
  Eigen::Index measurement_dim = 0;
  for (const KalmanFilterMeasurement& measurement : measurements) {
    if (measurement.name.empty()) {
      return absl::InvalidArgumentError("[KalmanFilter] a measurement has an empty name");
    }
    const std::string description = absl::StrCat("measurement '", measurement.name, "'");
    if (!unique_measurement_names.insert(measurement.name).second) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is given twice"));
    }
    if (measurement.measurement.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", description, " is empty"));
    }
    RETURN_IF_ERROR(checkFinite(measurement.measurement, absl::StrCat("the value of ", description)));
    if (measurement.H_measurement_model.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("[KalmanFilter] ", description, " observes no state channel: its H_measurement_model is empty"));
    }
    if (measurement.R_measurement_noise.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] R_measurement_noise of ", description,
                                                     " is missing; a noise-free measurement needs an explicit zero"));
    }
    RETURN_IF_ERROR(checkCovariance(measurement.R_measurement_noise, measurement.measurement.size(),
                                    absl::StrCat("R_measurement_noise of ", description)));
    measurement_names.push_back(measurement.name);
    measurement_dim += measurement.measurement.size();
  }

  // The measurements, stacked in the order given, with block-diagonal noise.
  const Eigen::Index state_dim = x_hat_state_estimate_.size();
  vector_t z_measurement(measurement_dim);
  matrix_t H_measurement_model = matrix_t::Zero(measurement_dim, state_dim);
  matrix_t R_measurement_noise = matrix_t::Zero(measurement_dim, measurement_dim);
  Eigen::Index row = 0;
  for (const KalmanFilterMeasurement& measurement : measurements) {
    const Eigen::Index rows = measurement.measurement.size();
    z_measurement.segment(row, rows) = measurement.measurement;
    for (const std::pair<const std::string, matrix_t>& entry : measurement.H_measurement_model) {
      const std::optional<ChannelBlock> col = findStateChannel(entry.first);
      if (!col.has_value()) {
        return unknownStateChannelError(absl::StatusCode::kInvalidArgument, entry.first,
                                        absl::StrCat("H_measurement_model of measurement '", measurement.name, "'"));
      }
      RETURN_IF_ERROR(checkBlock(entry.second, rows, col->size,
                                 absl::StrCat("H_measurement_model block ('", measurement.name, "', '", entry.first, "')")));
      H_measurement_model.block(row, col->offset, rows, col->size) = entry.second;
    }
    R_measurement_noise.block(row, row, rows, rows) = symmetrized(measurement.R_measurement_noise);
    row += rows;
  }

  const vector_t innovation = z_measurement - H_measurement_model * x_hat_state_estimate_;
  // P H^T, the covariance between the state and the predicted measurement.
  const matrix_t PHt = P_state_estimate_ * H_measurement_model.transpose();
  const matrix_t S_innovation_covariance = symmetrized(H_measurement_model * PHt + R_measurement_noise);
  const Eigen::LLT<matrix_t> S_cholesky(S_innovation_covariance);
  // The negated comparison also rejects a NaN estimate of the reciprocal condition number.
  if (S_cholesky.info() != Eigen::Success || !(S_cholesky.rcond() >= std::numeric_limits<scalar_t>::epsilon())) {
    return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] the innovation covariance H P H^T + R of measurements ",
                                                   joinedNamesOrNone(measurement_names),
                                                   " is not positive definite: a measurement with zero noise observes a direction "
                                                   "whose covariance is already zero"));
  }
  // K = P H^T S^-1, computed as the transpose of S^-1 H P because P and S are symmetric.
  const matrix_t K_kalman_gain = S_cholesky.solve(PHt.transpose()).transpose();
  vector_t x_hat_state_estimate = x_hat_state_estimate_ + K_kalman_gain * innovation;
  const matrix_t I_minus_KH = matrix_t::Identity(state_dim, state_dim) - K_kalman_gain * H_measurement_model;
  matrix_t P_state_estimate = symmetrized(I_minus_KH * P_state_estimate_ * I_minus_KH.transpose() +
                                          K_kalman_gain * R_measurement_noise * K_kalman_gain.transpose());
  RETURN_IF_ERROR(checkFinite(x_hat_state_estimate, "the corrected state"));
  RETURN_IF_ERROR(checkFinite(P_state_estimate, "the corrected covariance"));

  x_hat_state_estimate_ = std::move(x_hat_state_estimate);
  P_state_estimate_ = std::move(P_state_estimate);
  return absl::OkStatus();
}

absl::StatusOr<KalmanFilterState> KalmanFilter::getState(absl::string_view name) const {
  const std::optional<ChannelBlock> block = findStateChannel(name);
  if (!block.has_value()) {
    return unknownStateChannelError(absl::StatusCode::kNotFound, name, "getState()");
  }
  return makeState(name, *block);
}

std::vector<KalmanFilterState> KalmanFilter::getStates() const {
  std::vector<KalmanFilterState> states;
  states.reserve(state_names_.size());
  for (const std::string& name : state_names_) {
    states.push_back(makeState(name, state_channels_.at(name)));
  }
  return states;
}

absl::StatusOr<matrix_t> KalmanFilter::getCovariance(absl::string_view row_state_name, absl::string_view col_state_name) const {
  const std::optional<ChannelBlock> row = findStateChannel(row_state_name);
  if (!row.has_value()) {
    return unknownStateChannelError(absl::StatusCode::kNotFound, row_state_name, "getCovariance()");
  }
  const std::optional<ChannelBlock> col = findStateChannel(col_state_name);
  if (!col.has_value()) {
    return unknownStateChannelError(absl::StatusCode::kNotFound, col_state_name, "getCovariance()");
  }
  return matrix_t(P_state_estimate_.block(row->offset, col->offset, row->size, col->size));
}

KalmanFilterState KalmanFilter::makeState(absl::string_view name, const ChannelBlock& block) const {
  return KalmanFilterState{
      .name = std::string(name),
      .state = x_hat_state_estimate_.segment(block.offset, block.size),
      .P_state_estimate = P_state_estimate_.block(block.offset, block.offset, block.size, block.size),
  };
}

std::optional<KalmanFilter::ChannelBlock> KalmanFilter::findStateChannel(absl::string_view name) const {
  const absl::flat_hash_map<std::string, ChannelBlock>::const_iterator it = state_channels_.find(name);
  if (it == state_channels_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status KalmanFilter::unknownStateChannelError(absl::StatusCode code, absl::string_view name, absl::string_view context) const {
  return absl::Status(code, absl::StrCat("[KalmanFilter] ", context, " names state channel '", name,
                                         "', which reset() did not declare; the state channels are: ", joinedNamesOrNone(state_names_)));
}

absl::Status KalmanFilter::checkInitialized(absl::string_view method) const {
  if (state_names_.empty()) {
    return absl::FailedPreconditionError(absl::StrCat("[KalmanFilter] ", method, "() before reset(): the filter has no state channels"));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::estimation
