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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
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

/// Names what a check is about, e.g. "Q_input_noise of input 'imu_acceleration'". Called only when the check fails,
/// so that a step that succeeds formats no strings and allocates nothing for its error messages.
using Describe = absl::FunctionRef<std::string()>;

/// Relative tolerance of the covariance checks, scaled by the largest entry so that it applies alike to variances in
/// square millimeters and in square radians.
constexpr scalar_t kCovarianceTolerance = 1e-9;

void symmetrizeInPlace(matrix_t& matrix) {
  for (Eigen::Index col = 1; col < matrix.cols(); ++col) {
    for (Eigen::Index row = 0; row < col; ++row) {
      const scalar_t mean = 0.5 * (matrix(row, col) + matrix(col, row));
      matrix(row, col) = mean;
      matrix(col, row) = mean;
    }
  }
}

std::string joinedNamesOrNone(absl::Span<const std::string> names) {
  return names.empty() ? std::string("none") : absl::StrJoin(names, ", ");
}

std::string joinedInputNamesOrNone(absl::Span<const KalmanFilterInput> inputs) {
  if (inputs.empty()) {
    return "none";
  }
  return absl::StrJoin(inputs, ", ", [](std::string* out, const KalmanFilterInput& input) { absl::StrAppend(out, input.name); });
}

std::string joinedMeasurementNames(absl::Span<const KalmanFilterMeasurement> measurements) {
  return absl::StrJoin(measurements, ", ",
                       [](std::string* out, const KalmanFilterMeasurement& measurement) { absl::StrAppend(out, measurement.name); });
}

/// The position of the input called `name`; a linear scan, since a step has few inputs and a hash map would allocate.
std::optional<size_t> findInput(absl::Span<const KalmanFilterInput> inputs, absl::string_view name) {
  for (size_t index = 0; index < inputs.size(); ++index) {
    if (inputs[index].name == name) {
      return index;
    }
  }
  return std::nullopt;
}

template <typename Derived>
absl::Status checkFinite(const Eigen::MatrixBase<Derived>& value, Describe describe) {
  if (!value.allFinite()) {
    return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", describe(), " is not finite"));
  }
  return absl::OkStatus();
}

absl::Status checkBlock(const matrix_t& block, Eigen::Index rows, Eigen::Index cols, Describe describe) {
  if (block.rows() != rows || block.cols() != cols) {
    return absl::InvalidArgumentError(
        absl::StrCat("[KalmanFilter] ", describe(), " is ", block.rows(), "x", block.cols(), ", expected ", rows, "x", cols));
  }
  return checkFinite(block, describe);
}

/// The covariance checks a step can afford, in O(dim^2) and without allocating: dim x dim (dim > 0), finite,
/// symmetric, non-negative variances and correlations within [-1, 1]. Necessary for positive semi-definiteness, not
/// sufficient.
absl::Status checkCovarianceStructure(const matrix_t& covariance, Eigen::Index dim, Describe describe) {
  RETURN_IF_ERROR(checkBlock(covariance, dim, dim, describe));
  const scalar_t tolerance = kCovarianceTolerance * covariance.cwiseAbs().maxCoeff();
  for (Eigen::Index col = 0; col < dim; ++col) {
    if (covariance(col, col) < -tolerance) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", describe(), " is not positive semi-definite: its variance (", col,
                                                     ", ", col, ") is ", covariance(col, col)));
    }
    for (Eigen::Index row = 0; row < col; ++row) {
      if (std::abs(covariance(row, col) - covariance(col, row)) > tolerance) {
        return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", describe(), " is not symmetric"));
      }
      const scalar_t bound = std::sqrt(std::max(covariance(row, row), 0.0) * std::max(covariance(col, col), 0.0));
      if (std::abs(covariance(row, col)) > bound + tolerance) {
        return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", describe(), " is not positive semi-definite: its entry (", row,
                                                       ", ", col, ") = ", covariance(row, col),
                                                       " exceeds the geometric mean of the two variances, ", bound));
      }
    }
  }
  return absl::OkStatus();
}

/// The full check: the structure above and a smallest eigenvalue that is not negative, up to rounding. It allocates,
/// so it is for reset() only.
absl::Status checkCovariance(const matrix_t& covariance, Eigen::Index dim, Describe describe) {
  RETURN_IF_ERROR(checkCovarianceStructure(covariance, dim, describe));
  const scalar_t tolerance = kCovarianceTolerance * covariance.cwiseAbs().maxCoeff();
  const Eigen::SelfAdjointEigenSolver<matrix_t> eigen_solver(covariance, Eigen::EigenvaluesOnly);
  if (eigen_solver.info() != Eigen::Success || eigen_solver.eigenvalues().minCoeff() < -tolerance) {
    return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] ", describe(),
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
    if (channel.state.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] state channel '", channel.name, "' is empty"));
    }
    RETURN_IF_ERROR(checkFinite(channel.state, [&] { return absl::StrCat("the state of state channel '", channel.name, "'"); }));
    RETURN_IF_ERROR(checkCovariance(channel.P_state_estimate, channel.state.size(),
                                    [&] { return absl::StrCat("P_state_estimate of state channel '", channel.name, "'"); }));
    if (!state_channels.try_emplace(channel.name, ChannelBlock{state_dim, channel.state.size(), state_names.size()}).second) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] state channel '", channel.name, "' is declared twice"));
    }
    state_names.push_back(channel.name);
    state_dim += channel.state.size();
  }

  vector_t x_hat_state_estimate(state_dim);
  matrix_t P_state_estimate = matrix_t::Zero(state_dim, state_dim);
  for (const KalmanFilterState& channel : initial_states) {
    const ChannelBlock& block = state_channels.at(channel.name);
    x_hat_state_estimate.segment(block.offset, block.size) = channel.state;
    P_state_estimate.block(block.offset, block.offset, block.size, block.size) =
        0.5 * (channel.P_state_estimate + channel.P_state_estimate.transpose());
  }

  state_names_ = std::move(state_names);
  state_channels_ = std::move(state_channels);
  x_hat_state_estimate_ = std::move(x_hat_state_estimate);
  P_state_estimate_ = std::move(P_state_estimate);
  resizeWorkspace();
  return absl::OkStatus();
}

absl::Status KalmanFilter::reserve(Eigen::Index max_input_dim, Eigen::Index max_measurement_dim) {
  if (max_input_dim < 0 || max_measurement_dim < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("[KalmanFilter] reserve() needs non-negative dimensions, not ", max_input_dim, " and ", max_measurement_dim));
  }
  growWorkspace(max_input_dim, max_measurement_dim);
  return absl::OkStatus();
}

absl::Status KalmanFilter::predict(absl::Span<const KalmanFilterProcessModel> process_model, absl::Span<const KalmanFilterInput> inputs) {
  RETURN_IF_ERROR(checkInitialized("predict"));

  // The inputs, stacked in the order given. A step has few inputs, so the duplicate check and the lookups by name are
  // linear scans: a hash map built per step would allocate.
  workspace_.input_offsets.clear();
  Eigen::Index input_dim = 0;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const KalmanFilterInput& input = inputs[index];
    if (input.name.empty()) {
      return absl::InvalidArgumentError("[KalmanFilter] an input has an empty name");
    }
    if (input.input.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] input '", input.name, "' is empty"));
    }
    RETURN_IF_ERROR(checkFinite(input.input, [&] { return absl::StrCat("the value of input '", input.name, "'"); }));
    if (input.Q_input_noise.size() != 0) {
      RETURN_IF_ERROR(checkCovarianceStructure(input.Q_input_noise, input.input.size(),
                                               [&] { return absl::StrCat("Q_input_noise of input '", input.name, "'"); }));
    }
    for (size_t previous = 0; previous < index; ++previous) {
      if (inputs[previous].name == input.name) {
        return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] input '", input.name, "' is given twice"));
      }
    }
    workspace_.input_offsets.push_back(input_dim);
    input_dim += input.input.size();
  }
  growWorkspace(input_dim, /*measurement_dim=*/0);

  Eigen::Ref<vector_t> u_control_input = workspace_.u_control_input.head(input_dim);
  Eigen::Ref<matrix_t> Q_input_noise = workspace_.Q_input_noise.topLeftCorner(input_dim, input_dim);
  Q_input_noise.setZero();
  for (size_t index = 0; index < inputs.size(); ++index) {
    const KalmanFilterInput& input = inputs[index];
    const Eigen::Index offset = workspace_.input_offsets[index];
    const Eigen::Index size = input.input.size();
    u_control_input.segment(offset, size) = input.input;
    if (input.Q_input_noise.size() != 0) {
      Q_input_noise.block(offset, offset, size, size) = 0.5 * (input.Q_input_noise + input.Q_input_noise.transpose());
    }
  }

  // The joint matrices start from the identity transition, and the per-channel models overwrite their blocks.
  matrix_t& A_state_transition = workspace_.A_state_transition;
  Eigen::Ref<matrix_t> B_control_input = workspace_.B_control_input.leftCols(input_dim);
  matrix_t& Q_process_noise = workspace_.Q_process_noise;
  A_state_transition.setIdentity();
  B_control_input.setZero();
  Q_process_noise.setZero();
  std::fill(workspace_.state_has_process_model.begin(), workspace_.state_has_process_model.end(), 0);
  for (const KalmanFilterProcessModel& model : process_model) {
    const std::optional<ChannelBlock> row = findStateChannel(model.state_name);
    if (!row.has_value()) {
      return unknownStateChannelError(absl::StatusCode::kInvalidArgument, model.state_name, "a process model");
    }
    char& has_process_model = workspace_.state_has_process_model[row->index];
    if (has_process_model != 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] state channel '", model.state_name, "' is given two process models"));
    }
    has_process_model = 1;
    for (const std::pair<const std::string, matrix_t>& entry : model.A_state_transition) {
      const std::optional<ChannelBlock> col = findStateChannel(entry.first);
      if (!col.has_value()) {
        return unknownStateChannelError(absl::StatusCode::kInvalidArgument, entry.first,
                                        absl::StrCat("A_state_transition of state channel '", model.state_name, "'"));
      }
      RETURN_IF_ERROR(checkBlock(entry.second, row->size, col->size,
                                 [&] { return absl::StrCat("A_state_transition block ('", model.state_name, "', '", entry.first, "')"); }));
      A_state_transition.block(row->offset, col->offset, row->size, col->size) = entry.second;
    }
    for (const std::pair<const std::string, matrix_t>& entry : model.B_control_input) {
      const std::optional<size_t> input = findInput(inputs, entry.first);
      if (!input.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("[KalmanFilter] B_control_input of state channel '", model.state_name, "' names input '", entry.first,
                         "', which predict() was not given; the inputs are: ", joinedInputNamesOrNone(inputs)));
      }
      const Eigen::Index input_size = inputs[*input].input.size();
      RETURN_IF_ERROR(checkBlock(entry.second, row->size, input_size,
                                 [&] { return absl::StrCat("B_control_input block ('", model.state_name, "', '", entry.first, "')"); }));
      B_control_input.block(row->offset, workspace_.input_offsets[*input], row->size, input_size) = entry.second;
    }
    if (model.Q_process_noise.size() != 0) {
      RETURN_IF_ERROR(checkCovarianceStructure(model.Q_process_noise, row->size,
                                               [&] { return absl::StrCat("Q_process_noise of state channel '", model.state_name, "'"); }));
      Q_process_noise.block(row->offset, row->offset, row->size, row->size) = model.Q_process_noise;
    }
  }

  // x_hat <- A x_hat + B u and P <- A P A^T + B Q_u B^T + Q, every product into storage the workspace already holds.
  Eigen::Ref<matrix_t> BQ_input_noise = workspace_.BQ_input_noise.leftCols(input_dim);
  BQ_input_noise.noalias() = B_control_input * Q_input_noise;
  Q_process_noise.noalias() += BQ_input_noise * B_control_input.transpose();
  vector_t& x_hat_next = workspace_.x_hat_next;
  x_hat_next.noalias() = A_state_transition * x_hat_state_estimate_;
  x_hat_next.noalias() += B_control_input * u_control_input;
  workspace_.AP.noalias() = A_state_transition * P_state_estimate_;
  matrix_t& P_next = workspace_.P_next;
  P_next.noalias() = workspace_.AP * A_state_transition.transpose();
  P_next += Q_process_noise;
  symmetrizeInPlace(P_next);
  RETURN_IF_ERROR(checkFinite(x_hat_next, [] { return std::string("the predicted state"); }));
  RETURN_IF_ERROR(checkFinite(P_next, [] { return std::string("the predicted covariance"); }));

  x_hat_state_estimate_.swap(x_hat_next);
  P_state_estimate_.swap(P_next);
  return absl::OkStatus();
}

absl::Status KalmanFilter::correct(absl::Span<const KalmanFilterMeasurement> measurements) {
  RETURN_IF_ERROR(checkInitialized("correct"));
  if (measurements.empty()) {
    return absl::OkStatus();
  }

  Eigen::Index measurement_dim = 0;
  for (size_t index = 0; index < measurements.size(); ++index) {
    const KalmanFilterMeasurement& measurement = measurements[index];
    if (measurement.name.empty()) {
      return absl::InvalidArgumentError("[KalmanFilter] a measurement has an empty name");
    }
    for (size_t previous = 0; previous < index; ++previous) {
      if (measurements[previous].name == measurement.name) {
        return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] measurement '", measurement.name, "' is given twice"));
      }
    }
    if (measurement.measurement.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] measurement '", measurement.name, "' is empty"));
    }
    RETURN_IF_ERROR(
        checkFinite(measurement.measurement, [&] { return absl::StrCat("the value of measurement '", measurement.name, "'"); }));
    if (measurement.H_measurement_model.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("[KalmanFilter] measurement '", measurement.name, "' observes no state channel: its H_measurement_model is empty"));
    }
    if (measurement.R_measurement_noise.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] R_measurement_noise of measurement '", measurement.name,
                                                     "' is missing; a noise-free measurement needs an explicit zero"));
    }
    RETURN_IF_ERROR(checkCovarianceStructure(measurement.R_measurement_noise, measurement.measurement.size(),
                                             [&] { return absl::StrCat("R_measurement_noise of measurement '", measurement.name, "'"); }));
    measurement_dim += measurement.measurement.size();
  }
  growWorkspace(/*input_dim=*/0, measurement_dim);

  // The measurements, stacked in the order given, with block-diagonal noise. The innovation starts as z.
  Eigen::Ref<matrix_t> H_measurement_model = workspace_.H_measurement_model.topRows(measurement_dim);
  Eigen::Ref<matrix_t> R_measurement_noise = workspace_.R_measurement_noise.topLeftCorner(measurement_dim, measurement_dim);
  Eigen::Ref<vector_t> innovation = workspace_.innovation.head(measurement_dim);
  H_measurement_model.setZero();
  R_measurement_noise.setZero();
  Eigen::Index row = 0;
  for (const KalmanFilterMeasurement& measurement : measurements) {
    const Eigen::Index rows = measurement.measurement.size();
    innovation.segment(row, rows) = measurement.measurement;
    for (const std::pair<const std::string, matrix_t>& entry : measurement.H_measurement_model) {
      const std::optional<ChannelBlock> col = findStateChannel(entry.first);
      if (!col.has_value()) {
        return unknownStateChannelError(absl::StatusCode::kInvalidArgument, entry.first,
                                        absl::StrCat("H_measurement_model of measurement '", measurement.name, "'"));
      }
      RETURN_IF_ERROR(checkBlock(entry.second, rows, col->size, [&] {
        return absl::StrCat("H_measurement_model block ('", measurement.name, "', '", entry.first, "')");
      }));
      H_measurement_model.block(row, col->offset, rows, col->size) = entry.second;
    }
    R_measurement_noise.block(row, row, rows, rows) = 0.5 * (measurement.R_measurement_noise + measurement.R_measurement_noise.transpose());
    row += rows;
  }

  // innovation = z - H x_hat, P H^T, and S = H P H^T + R, every product into storage the workspace already holds.
  innovation.noalias() -= H_measurement_model * x_hat_state_estimate_;
  Eigen::Ref<matrix_t> PHt = workspace_.PHt.leftCols(measurement_dim);
  PHt.noalias() = P_state_estimate_ * H_measurement_model.transpose();
  Eigen::Ref<matrix_t> S_innovation_covariance = workspace_.S_innovation_covariance.topLeftCorner(measurement_dim, measurement_dim);
  S_innovation_covariance.noalias() = H_measurement_model * PHt;
  S_innovation_covariance += R_measurement_noise;
  const scalar_t largest_variance = S_innovation_covariance.diagonal().maxCoeff();
  // In place: the Cholesky factor L overwrites the lower triangle of S, so the decomposition does not allocate. A pivot
  // L_ii^2 below machine epsilon times the largest variance makes S singular to working precision; the negated
  // comparison also rejects a NaN.
  const Eigen::LLT<Eigen::Ref<matrix_t>> S_cholesky(S_innovation_covariance);
  const scalar_t smallest_pivot_root = S_innovation_covariance.diagonal().minCoeff();
  if (S_cholesky.info() != Eigen::Success ||
      !(smallest_pivot_root * smallest_pivot_root > std::numeric_limits<scalar_t>::epsilon() * largest_variance)) {
    return absl::InvalidArgumentError(absl::StrCat("[KalmanFilter] the innovation covariance H P H^T + R of measurements ",
                                                   joinedMeasurementNames(measurements),
                                                   " is not positive definite: a measurement with zero noise observes a direction "
                                                   "whose covariance is already zero"));
  }

  // K^T = S^-1 H P, the transpose of the gain K = P H^T S^-1 because P and S are symmetric.
  Eigen::Ref<matrix_t> K_transpose = workspace_.K_transpose.topRows(measurement_dim);
  K_transpose = PHt.transpose();
  S_cholesky.solveInPlace(K_transpose);
  vector_t& x_hat_next = workspace_.x_hat_next;
  x_hat_next = x_hat_state_estimate_;
  x_hat_next.noalias() += K_transpose.transpose() * innovation;
  // Joseph form: P <- (I - K H) P (I - K H)^T + K R K^T.
  matrix_t& I_minus_KH = workspace_.I_minus_KH;
  I_minus_KH.setIdentity();
  I_minus_KH.noalias() -= K_transpose.transpose() * H_measurement_model;
  workspace_.I_minus_KH_P.noalias() = I_minus_KH * P_state_estimate_;
  matrix_t& P_next = workspace_.P_next;
  P_next.noalias() = workspace_.I_minus_KH_P * I_minus_KH.transpose();
  Eigen::Ref<matrix_t> KR = workspace_.KR.leftCols(measurement_dim);
  KR.noalias() = K_transpose.transpose() * R_measurement_noise;
  P_next.noalias() += KR * K_transpose;
  symmetrizeInPlace(P_next);
  RETURN_IF_ERROR(checkFinite(x_hat_next, [] { return std::string("the corrected state"); }));
  RETURN_IF_ERROR(checkFinite(P_next, [] { return std::string("the corrected covariance"); }));

  x_hat_state_estimate_.swap(x_hat_next);
  P_state_estimate_.swap(P_next);
  return absl::OkStatus();
}

absl::StatusOr<KalmanFilterState> KalmanFilter::getState(absl::string_view name) const {
  KalmanFilterState state;
  RETURN_IF_ERROR(getState(name, state));
  return state;
}

absl::Status KalmanFilter::getState(absl::string_view name, KalmanFilterState& state) const {
  const std::optional<ChannelBlock> block = findStateChannel(name);
  if (!block.has_value()) {
    return unknownStateChannelError(absl::StatusCode::kNotFound, name, "getState()");
  }
  fillState(name, *block, state);
  return absl::OkStatus();
}

std::vector<KalmanFilterState> KalmanFilter::getStates() const {
  std::vector<KalmanFilterState> states(state_names_.size());
  for (size_t index = 0; index < state_names_.size(); ++index) {
    fillState(state_names_[index], state_channels_.at(state_names_[index]), states[index]);
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

void KalmanFilter::fillState(absl::string_view name, const ChannelBlock& block, KalmanFilterState& state) const {
  state.name.assign(name.data(), name.size());
  state.state = x_hat_state_estimate_.segment(block.offset, block.size);
  state.P_state_estimate = P_state_estimate_.block(block.offset, block.offset, block.size, block.size);
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

void KalmanFilter::resizeWorkspace() {
  const Eigen::Index state_dim = x_hat_state_estimate_.size();
  const Eigen::Index input_dim = workspace_.input_capacity;
  const Eigen::Index measurement_dim = workspace_.measurement_capacity;
  workspace_.state_has_process_model.assign(state_names_.size(), 0);
  workspace_.input_offsets.reserve(static_cast<size_t>(input_dim));

  workspace_.A_state_transition.resize(state_dim, state_dim);
  workspace_.B_control_input.resize(state_dim, input_dim);
  workspace_.Q_process_noise.resize(state_dim, state_dim);
  workspace_.u_control_input.resize(input_dim);
  workspace_.Q_input_noise.resize(input_dim, input_dim);
  workspace_.BQ_input_noise.resize(state_dim, input_dim);
  workspace_.AP.resize(state_dim, state_dim);

  workspace_.H_measurement_model.resize(measurement_dim, state_dim);
  workspace_.R_measurement_noise.resize(measurement_dim, measurement_dim);
  workspace_.innovation.resize(measurement_dim);
  workspace_.PHt.resize(state_dim, measurement_dim);
  workspace_.S_innovation_covariance.resize(measurement_dim, measurement_dim);
  workspace_.K_transpose.resize(measurement_dim, state_dim);
  workspace_.KR.resize(state_dim, measurement_dim);
  workspace_.I_minus_KH.resize(state_dim, state_dim);
  workspace_.I_minus_KH_P.resize(state_dim, state_dim);

  workspace_.x_hat_next.resize(state_dim);
  workspace_.P_next.resize(state_dim, state_dim);
}

void KalmanFilter::growWorkspace(Eigen::Index input_dim, Eigen::Index measurement_dim) {
  if (input_dim <= workspace_.input_capacity && measurement_dim <= workspace_.measurement_capacity) {
    return;
  }
  workspace_.input_capacity = std::max(workspace_.input_capacity, input_dim);
  workspace_.measurement_capacity = std::max(workspace_.measurement_capacity, measurement_dim);
  resizeWorkspace();
}

}  // namespace ocs2::humanoid::estimation
