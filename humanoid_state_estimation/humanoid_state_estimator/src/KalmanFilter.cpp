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

#include "Eigen/Cholesky"
#include "Eigen/Eigenvalues"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid::estimation {
namespace {

/// Names what a check is about, e.g. "inputNoiseCovariance of input
/// 'imu_acceleration'". Called only when the check fails, so that a step that
/// succeeds formats no strings and allocates nothing for its error messages.
using Describe = absl::FunctionRef<std::string()>;

/// Relative tolerance of the covariance checks, scaled by the largest entry so
/// that it applies alike to variances in square millimeters and in square
/// radians.
constexpr scalar_t kCovarianceTolerance = 1.0e-9;

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
  return absl::StrJoin(
      inputs, ", ",
      [](std::string* absl_nonnull out, const KalmanFilterInput& input) {
        absl::StrAppend(out, input.name);
      });
}

std::string joinedMeasurementNames(
    absl::Span<const KalmanFilterMeasurement> measurements) {
  return absl::StrJoin(measurements, ", ",
                       [](std::string* absl_nonnull out,
                          const KalmanFilterMeasurement& measurement) {
                         absl::StrAppend(out, measurement.name);
                       });
}

/// The position of the input called `name`; a linear scan, since a step has few
/// inputs and a hash map would allocate.
std::optional<size_t> findInput(absl::Span<const KalmanFilterInput> inputs,
                                absl::string_view name) {
  for (size_t index = 0; index < inputs.size(); ++index) {
    if (inputs[index].name == name) {
      return index;
    }
  }
  return std::nullopt;
}

template <typename Derived>
absl::Status checkFinite(const Eigen::MatrixBase<Derived>& value,
                         Describe describe) {
  if (!value.allFinite()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[KalmanFilter] ", describe(), " is not finite"));
  }
  return absl::OkStatus();
}

absl::Status checkBlock(const matrix_t& block,
                        Eigen::Index rows,
                        Eigen::Index cols,
                        Describe describe) {
  if (block.rows() != rows || block.cols() != cols) {
    return absl::InvalidArgumentError(
        absl::StrCat("[KalmanFilter] ", describe(), " is ", block.rows(), "x",
                     block.cols(), ", expected ", rows, "x", cols));
  }
  return checkFinite(block, describe);
}

/// The covariance checks a step can afford, in O(dim^2) and without allocating:
/// dim x dim (dim > 0), finite, symmetric, non-negative variances and
/// correlations within [-1, 1]. Necessary for positive semi-definiteness, not
/// sufficient.
absl::Status checkCovarianceStructure(const matrix_t& covariance,
                                      Eigen::Index dim,
                                      Describe describe) {
  RETURN_IF_ERROR(checkBlock(covariance, dim, dim, describe));
  const scalar_t tolerance =
      kCovarianceTolerance * covariance.cwiseAbs().maxCoeff();
  for (Eigen::Index col = 0; col < dim; ++col) {
    if (covariance(col, col) < -tolerance) {
      return absl::InvalidArgumentError(
          absl::StrCat("[KalmanFilter] ", describe(),
                       " is not positive semi-definite: its variance (", col,
                       ", ", col, ") is ", covariance(col, col)));
    }
    for (Eigen::Index row = 0; row < col; ++row) {
      if (std::abs(covariance(row, col) - covariance(col, row)) > tolerance) {
        return absl::InvalidArgumentError(
            absl::StrCat("[KalmanFilter] ", describe(), " is not symmetric"));
      }
      const scalar_t bound = std::sqrt(std::max(covariance(row, row), 0.0) *
                                       std::max(covariance(col, col), 0.0));
      if (std::abs(covariance(row, col)) > bound + tolerance) {
        return absl::InvalidArgumentError(absl::StrCat(
            "[KalmanFilter] ", describe(),
            " is not positive semi-definite: its entry (", row, ", ", col,
            ") = ", covariance(row, col),
            " exceeds the geometric mean of the two variances, ", bound));
      }
    }
  }
  return absl::OkStatus();
}

/// The full check: the structure above and a smallest eigenvalue that is not
/// negative, up to rounding. It allocates, so it is for reset() only.
absl::Status checkCovariance(const matrix_t& covariance,
                             Eigen::Index dim,
                             Describe describe) {
  RETURN_IF_ERROR(checkCovarianceStructure(covariance, dim, describe));
  const scalar_t tolerance =
      kCovarianceTolerance * covariance.cwiseAbs().maxCoeff();
  const Eigen::SelfAdjointEigenSolver<matrix_t> eigenSolver(
      covariance, Eigen::EigenvaluesOnly);
  if (eigenSolver.info() != Eigen::Success ||
      eigenSolver.eigenvalues().minCoeff() < -tolerance) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[KalmanFilter] ", describe(),
        " is not positive semi-definite: its smallest eigenvalue is ",
        eigenSolver.eigenvalues().minCoeff()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<KalmanFilter>> KalmanFilter::Create(
    absl::Span<const KalmanFilterState> initialStates) {
  auto filter = std::make_unique<KalmanFilter>();
  RETURN_IF_ERROR(filter->reset(initialStates));
  return filter;
}

absl::Status KalmanFilter::reset(
    absl::Span<const KalmanFilterState> initialStates) {
  if (initialStates.empty()) {
    return absl::InvalidArgumentError(
        "[KalmanFilter] reset() needs at least one state channel");
  }

  std::vector<std::string> stateNames;
  stateNames.reserve(initialStates.size());
  absl::flat_hash_map<std::string, ChannelBlock> stateChannels;
  Eigen::Index stateDim = 0;
  for (const KalmanFilterState& channel : initialStates) {
    if (channel.name.empty()) {
      return absl::InvalidArgumentError(
          "[KalmanFilter] a state channel has an empty name");
    }
    if (channel.state.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[KalmanFilter] state channel '", channel.name, "' is empty"));
    }
    RETURN_IF_ERROR(checkFinite(channel.state, [&] {
      return absl::StrCat("the state of state channel '", channel.name, "'");
    }));
    RETURN_IF_ERROR(
        checkCovariance(channel.stateCovariance, channel.state.size(), [&] {
          return absl::StrCat("stateCovariance of state channel '",
                              channel.name, "'");
        }));
    if (!stateChannels
             .try_emplace(channel.name,
                          ChannelBlock{.offset = stateDim,
                                       .size = channel.state.size(),
                                       .index = stateNames.size()})
             .second) {
      return absl::InvalidArgumentError(
          absl::StrCat("[KalmanFilter] state channel '", channel.name,
                       "' is declared twice"));
    }
    stateNames.push_back(channel.name);
    stateDim += channel.state.size();
  }

  vector_t stateEstimate(stateDim);
  matrix_t stateCovariance = matrix_t::Zero(stateDim, stateDim);
  // The channels stacked in their order, at the offsets the loop above gave
  // them.
  Eigen::Index offset = 0;
  for (const KalmanFilterState& channel : initialStates) {
    const Eigen::Index size = channel.state.size();
    stateEstimate.segment(offset, size) = channel.state;
    stateCovariance.block(offset, offset, size, size) =
        0.5 * (channel.stateCovariance + channel.stateCovariance.transpose());
    offset += size;
  }

  stateNames_ = std::move(stateNames);
  stateChannels_ = std::move(stateChannels);
  stateEstimate_ = std::move(stateEstimate);
  stateCovariance_ = std::move(stateCovariance);
  resizeWorkspace();
  return absl::OkStatus();
}

absl::Status KalmanFilter::resetEstimate(
    absl::Span<const KalmanFilterState> states) {
  RETURN_IF_ERROR(checkInitialized("resetEstimate"));
  if (states.size() != stateNames_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[KalmanFilter] resetEstimate() takes the ", stateNames_.size(),
        " declared state channels, got ", states.size()));
  }
  for (size_t index = 0; index < states.size(); ++index) {
    const KalmanFilterState& channel = states[index];
    if (channel.name != stateNames_[index]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[KalmanFilter] resetEstimate(): channel ", index, " is '",
          channel.name, "', declared as '", stateNames_[index], "'"));
    }
    const Eigen::Index size = stateChannels_.find(channel.name)->second.size;
    if (channel.state.size() != size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[KalmanFilter] resetEstimate(): state channel '", channel.name,
          "' has ", channel.state.size(), " entries, declared with ", size));
    }
    RETURN_IF_ERROR(checkFinite(channel.state, [&] {
      return absl::StrCat("the state of state channel '", channel.name, "'");
    }));
    RETURN_IF_ERROR(
        checkCovarianceStructure(channel.stateCovariance, size, [&] {
          return absl::StrCat("stateCovariance of state channel '",
                              channel.name, "'");
        }));
  }
  stateCovariance_.setZero();
  Eigen::Index offset = 0;
  for (const KalmanFilterState& channel : states) {
    const Eigen::Index size = channel.state.size();
    stateEstimate_.segment(offset, size) = channel.state;
    stateCovariance_.block(offset, offset, size, size) =
        0.5 * (channel.stateCovariance + channel.stateCovariance.transpose());
    offset += size;
  }
  return absl::OkStatus();
}

absl::Status KalmanFilter::reserve(Eigen::Index maxInputDim,
                                   Eigen::Index maxMeasurementDim) {
  if (maxInputDim < 0 || maxMeasurementDim < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[KalmanFilter] reserve() needs non-negative dimensions, not ",
        maxInputDim, " and ", maxMeasurementDim));
  }
  growWorkspace(maxInputDim, maxMeasurementDim);
  return absl::OkStatus();
}

absl::Status KalmanFilter::predict(
    absl::Span<const KalmanFilterProcessModel> processModel,
    absl::Span<const KalmanFilterInput> inputs) {
  RETURN_IF_ERROR(checkInitialized("predict"));

  // The inputs, stacked in the order given. A step has few inputs, so the
  // duplicate check and the lookups by name are linear scans: a hash map built
  // per step would allocate.
  workspace_.inputOffsets.clear();
  Eigen::Index inputDim = 0;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const KalmanFilterInput& input = inputs[index];
    if (input.name.empty()) {
      return absl::InvalidArgumentError(
          "[KalmanFilter] an input has an empty name");
    }
    if (input.input.size() == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("[KalmanFilter] input '", input.name, "' is empty"));
    }
    RETURN_IF_ERROR(checkFinite(input.input, [&] {
      return absl::StrCat("the value of input '", input.name, "'");
    }));
    if (input.inputNoiseCovariance.size() != 0) {
      RETURN_IF_ERROR(checkCovarianceStructure(
          input.inputNoiseCovariance, input.input.size(), [&] {
            return absl::StrCat("inputNoiseCovariance of input '", input.name,
                                "'");
          }));
    }
    for (size_t previous = 0; previous < index; ++previous) {
      if (inputs[previous].name == input.name) {
        return absl::InvalidArgumentError(absl::StrCat(
            "[KalmanFilter] input '", input.name, "' is given twice"));
      }
    }
    workspace_.inputOffsets.push_back(inputDim);
    inputDim += input.input.size();
  }
  growWorkspace(inputDim, /*measurementDim=*/0);

  Eigen::Ref<vector_t> controlInput = workspace_.controlInput.head(inputDim);
  Eigen::Ref<matrix_t> inputNoiseCovariance =
      workspace_.inputNoiseCovariance.topLeftCorner(inputDim, inputDim);
  inputNoiseCovariance.setZero();
  for (size_t index = 0; index < inputs.size(); ++index) {
    const KalmanFilterInput& input = inputs[index];
    const Eigen::Index offset = workspace_.inputOffsets[index];
    const Eigen::Index size = input.input.size();
    controlInput.segment(offset, size) = input.input;
    if (input.inputNoiseCovariance.size() != 0) {
      inputNoiseCovariance.block(offset, offset, size, size) =
          0.5 *
          (input.inputNoiseCovariance + input.inputNoiseCovariance.transpose());
    }
  }

  // The joint matrices start from the identity transition, and the per-channel
  // models overwrite their blocks.
  matrix_t& stateTransition = workspace_.stateTransition;
  Eigen::Ref<matrix_t> controlInputMatrix =
      workspace_.controlInputMatrix.leftCols(inputDim);
  matrix_t& processNoiseCovariance = workspace_.processNoiseCovariance;
  stateTransition.setIdentity();
  controlInputMatrix.setZero();
  processNoiseCovariance.setZero();
  std::fill(workspace_.stateHasProcessModel.begin(),
            workspace_.stateHasProcessModel.end(), 0);
  for (const KalmanFilterProcessModel& model : processModel) {
    const std::optional<ChannelBlock> row = findStateChannel(model.stateName);
    if (!row.has_value()) {
      return unknownStateChannelError(absl::StatusCode::kInvalidArgument,
                                      model.stateName, "a process model");
    }
    char& hasProcessModel = workspace_.stateHasProcessModel[row->index];
    if (hasProcessModel != 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("[KalmanFilter] state channel '", model.stateName,
                       "' is given two process models"));
    }
    hasProcessModel = 1;
    for (const std::pair<const std::string, matrix_t>& entry :
         model.stateTransition) {
      const std::optional<ChannelBlock> col = findStateChannel(entry.first);
      if (!col.has_value()) {
        return unknownStateChannelError(
            absl::StatusCode::kInvalidArgument, entry.first,
            absl::StrCat("stateTransition of state channel '", model.stateName,
                         "'"));
      }
      RETURN_IF_ERROR(checkBlock(entry.second, row->size, col->size, [&] {
        return absl::StrCat("stateTransition block ('", model.stateName, "', '",
                            entry.first, "')");
      }));
      stateTransition.block(row->offset, col->offset, row->size, col->size) =
          entry.second;
    }
    for (const std::pair<const std::string, matrix_t>& entry :
         model.controlInputMatrix) {
      const std::optional<size_t> input = findInput(inputs, entry.first);
      if (!input.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("[KalmanFilter] controlInputMatrix of state channel '",
                         model.stateName, "' names input '", entry.first,
                         "', which predict() was not given; the inputs are: ",
                         joinedInputNamesOrNone(inputs)));
      }
      const Eigen::Index inputSize = inputs[*input].input.size();
      RETURN_IF_ERROR(checkBlock(entry.second, row->size, inputSize, [&] {
        return absl::StrCat("controlInputMatrix block ('", model.stateName,
                            "', '", entry.first, "')");
      }));
      controlInputMatrix.block(row->offset, workspace_.inputOffsets[*input],
                               row->size, inputSize) = entry.second;
    }
    if (model.processNoiseCovariance.size() != 0) {
      RETURN_IF_ERROR(checkCovarianceStructure(
          model.processNoiseCovariance, row->size, [&] {
            return absl::StrCat("processNoiseCovariance of state channel '",
                                model.stateName, "'");
          }));
      processNoiseCovariance.block(row->offset, row->offset, row->size,
                                   row->size) = model.processNoiseCovariance;
    }
  }

  // x_hat <- A x_hat + B u and P <- A P A^T + B Q_u B^T + Q, every product into
  // storage the workspace already holds.
  Eigen::Ref<matrix_t> controlInputTimesInputNoise =
      workspace_.controlInputTimesInputNoise.leftCols(inputDim);
  controlInputTimesInputNoise.noalias() =
      controlInputMatrix * inputNoiseCovariance;
  processNoiseCovariance.noalias() +=
      controlInputTimesInputNoise * controlInputMatrix.transpose();
  vector_t& stateEstimateNext = workspace_.stateEstimateNext;
  stateEstimateNext.noalias() = stateTransition * stateEstimate_;
  stateEstimateNext.noalias() += controlInputMatrix * controlInput;
  workspace_.AP.noalias() = stateTransition * stateCovariance_;
  matrix_t& covarianceNext = workspace_.covarianceNext;
  covarianceNext.noalias() = workspace_.AP * stateTransition.transpose();
  covarianceNext += processNoiseCovariance;
  symmetrizeInPlace(covarianceNext);
  RETURN_IF_ERROR(checkFinite(
      stateEstimateNext, [] { return std::string("the predicted state"); }));
  RETURN_IF_ERROR(checkFinite(
      covarianceNext, [] { return std::string("the predicted covariance"); }));

  stateEstimate_.swap(stateEstimateNext);
  stateCovariance_.swap(covarianceNext);
  return absl::OkStatus();
}

absl::Status KalmanFilter::correct(
    absl::Span<const KalmanFilterMeasurement> measurements) {
  RETURN_IF_ERROR(checkInitialized("correct"));
  if (measurements.empty()) {
    return absl::OkStatus();
  }

  Eigen::Index measurementDim = 0;
  for (size_t index = 0; index < measurements.size(); ++index) {
    const KalmanFilterMeasurement& measurement = measurements[index];
    if (measurement.name.empty()) {
      return absl::InvalidArgumentError(
          "[KalmanFilter] a measurement has an empty name");
    }
    for (size_t previous = 0; previous < index; ++previous) {
      if (measurements[previous].name == measurement.name) {
        return absl::InvalidArgumentError(
            absl::StrCat("[KalmanFilter] measurement '", measurement.name,
                         "' is given twice"));
      }
    }
    if (measurement.measurement.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[KalmanFilter] measurement '", measurement.name, "' is empty"));
    }
    RETURN_IF_ERROR(checkFinite(measurement.measurement, [&] {
      return absl::StrCat("the value of measurement '", measurement.name, "'");
    }));
    if (measurement.measurementModel.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[KalmanFilter] measurement '", measurement.name,
          "' observes no state channel: its measurementModel is empty"));
    }
    if (measurement.measurementNoiseCovariance.size() == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[KalmanFilter] measurementNoiseCovariance of measurement '",
          measurement.name,
          "' is missing; a noise-free measurement needs an explicit zero"));
    }
    RETURN_IF_ERROR(checkCovarianceStructure(
        measurement.measurementNoiseCovariance, measurement.measurement.size(),
        [&] {
          return absl::StrCat("measurementNoiseCovariance of measurement '",
                              measurement.name, "'");
        }));
    measurementDim += measurement.measurement.size();
  }
  growWorkspace(/*inputDim=*/0, measurementDim);

  // The measurements, stacked in the order given, with block-diagonal noise.
  // The innovation starts as z.
  Eigen::Ref<matrix_t> measurementModel =
      workspace_.measurementModel.topRows(measurementDim);
  Eigen::Ref<matrix_t> measurementNoiseCovariance =
      workspace_.measurementNoiseCovariance.topLeftCorner(measurementDim,
                                                          measurementDim);
  Eigen::Ref<vector_t> innovation = workspace_.innovation.head(measurementDim);
  measurementModel.setZero();
  measurementNoiseCovariance.setZero();
  Eigen::Index row = 0;
  for (const KalmanFilterMeasurement& measurement : measurements) {
    const Eigen::Index rows = measurement.measurement.size();
    innovation.segment(row, rows) = measurement.measurement;
    for (const std::pair<const std::string, matrix_t>& entry :
         measurement.measurementModel) {
      const std::optional<ChannelBlock> col = findStateChannel(entry.first);
      if (!col.has_value()) {
        return unknownStateChannelError(
            absl::StatusCode::kInvalidArgument, entry.first,
            absl::StrCat("measurementModel of measurement '", measurement.name,
                         "'"));
      }
      RETURN_IF_ERROR(checkBlock(entry.second, rows, col->size, [&] {
        return absl::StrCat("measurementModel block ('", measurement.name,
                            "', '", entry.first, "')");
      }));
      measurementModel.block(row, col->offset, rows, col->size) = entry.second;
    }
    measurementNoiseCovariance.block(row, row, rows, rows) =
        0.5 * (measurement.measurementNoiseCovariance +
               measurement.measurementNoiseCovariance.transpose());
    row += rows;
  }

  // innovation = z - H x_hat, P H^T, and S = H P H^T + R, every product into
  // storage the workspace already holds.
  innovation.noalias() -= measurementModel * stateEstimate_;
  Eigen::Ref<matrix_t> PHt = workspace_.PHt.leftCols(measurementDim);
  PHt.noalias() = stateCovariance_ * measurementModel.transpose();
  Eigen::Ref<matrix_t> innovationCovariance =
      workspace_.innovationCovariance.topLeftCorner(measurementDim,
                                                    measurementDim);
  innovationCovariance.noalias() = measurementModel * PHt;
  innovationCovariance += measurementNoiseCovariance;
  const scalar_t largestVariance = innovationCovariance.diagonal().maxCoeff();
  // In place: the Cholesky factor L overwrites the lower triangle of S, so the
  // decomposition does not allocate. A pivot L_ii^2 below machine epsilon times
  // the largest variance makes S singular to working precision; the negated
  // comparison also rejects a NaN.
  const Eigen::LLT<Eigen::Ref<matrix_t>> innovationCholesky(
      innovationCovariance);
  const scalar_t smallestPivotRoot = innovationCovariance.diagonal().minCoeff();
  if (innovationCholesky.info() != Eigen::Success ||
      !(smallestPivotRoot * smallestPivotRoot >
        std::numeric_limits<scalar_t>::epsilon() * largestVariance)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[KalmanFilter] the innovation covariance H P H^T + R of measurements ",
        joinedMeasurementNames(measurements),
        " is not positive definite: a measurement with zero noise observes a "
        "direction "
        "whose covariance is already zero"));
  }

  // K^T = S^-1 H P, the transpose of the gain K = P H^T S^-1 because P and S
  // are symmetric.
  Eigen::Ref<matrix_t> gainTranspose =
      workspace_.gainTranspose.topRows(measurementDim);
  gainTranspose = PHt.transpose();
  innovationCholesky.solveInPlace(gainTranspose);
  vector_t& stateEstimateNext = workspace_.stateEstimateNext;
  stateEstimateNext = stateEstimate_;
  stateEstimateNext.noalias() += gainTranspose.transpose() * innovation;
  // Joseph form: P <- (I - K H) P (I - K H)^T + K R K^T.
  matrix_t& identityMinusKh = workspace_.identityMinusKh;
  identityMinusKh.setIdentity();
  identityMinusKh.noalias() -= gainTranspose.transpose() * measurementModel;
  workspace_.identityMinusKhP.noalias() = identityMinusKh * stateCovariance_;
  matrix_t& covarianceNext = workspace_.covarianceNext;
  covarianceNext.noalias() =
      workspace_.identityMinusKhP * identityMinusKh.transpose();
  Eigen::Ref<matrix_t> KR = workspace_.KR.leftCols(measurementDim);
  KR.noalias() = gainTranspose.transpose() * measurementNoiseCovariance;
  covarianceNext.noalias() += KR * gainTranspose;
  symmetrizeInPlace(covarianceNext);
  RETURN_IF_ERROR(checkFinite(
      stateEstimateNext, [] { return std::string("the corrected state"); }));
  RETURN_IF_ERROR(checkFinite(
      covarianceNext, [] { return std::string("the corrected covariance"); }));

  stateEstimate_.swap(stateEstimateNext);
  stateCovariance_.swap(covarianceNext);
  return absl::OkStatus();
}

absl::StatusOr<KalmanFilterState> KalmanFilter::getState(
    absl::string_view name) const {
  KalmanFilterState state;
  RETURN_IF_ERROR(getState(name, state));
  return state;
}

absl::Status KalmanFilter::getState(absl::string_view name,
                                    KalmanFilterState& state) const {
  const std::optional<ChannelBlock> block = findStateChannel(name);
  if (!block.has_value()) {
    return unknownStateChannelError(absl::StatusCode::kNotFound, name,
                                    "getState()");
  }
  fillState(name, *block, state);
  return absl::OkStatus();
}

std::vector<KalmanFilterState> KalmanFilter::getStates() const {
  std::vector<KalmanFilterState> states(stateNames_.size());
  // Each channel into its place in stateNames_, so that the hash map's order
  // does not matter.
  for (const std::pair<const std::string, ChannelBlock>& channel :
       stateChannels_) {
    fillState(channel.first, channel.second, states[channel.second.index]);
  }
  return states;
}

absl::StatusOr<matrix_t> KalmanFilter::getCovariance(
    absl::string_view rowStateName, absl::string_view colStateName) const {
  const std::optional<ChannelBlock> row = findStateChannel(rowStateName);
  if (!row.has_value()) {
    return unknownStateChannelError(absl::StatusCode::kNotFound, rowStateName,
                                    "getCovariance()");
  }
  const std::optional<ChannelBlock> col = findStateChannel(colStateName);
  if (!col.has_value()) {
    return unknownStateChannelError(absl::StatusCode::kNotFound, colStateName,
                                    "getCovariance()");
  }
  return matrix_t(
      stateCovariance_.block(row->offset, col->offset, row->size, col->size));
}

void KalmanFilter::fillState(absl::string_view name,
                             const ChannelBlock& block,
                             KalmanFilterState& state) const {
  state.name.assign(name.data(), name.size());
  state.state = stateEstimate_.segment(block.offset, block.size);
  state.stateCovariance = stateCovariance_.block(block.offset, block.offset,
                                                 block.size, block.size);
}

std::optional<KalmanFilter::ChannelBlock> KalmanFilter::findStateChannel(
    absl::string_view name) const {
  const absl::flat_hash_map<std::string, ChannelBlock>::const_iterator it =
      stateChannels_.find(name);
  if (it == stateChannels_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status KalmanFilter::unknownStateChannelError(
    absl::StatusCode code,
    absl::string_view name,
    absl::string_view context) const {
  return absl::Status(
      code,
      absl::StrCat("[KalmanFilter] ", context, " names state channel '", name,
                   "', which reset() did not declare; the state channels are: ",
                   joinedNamesOrNone(stateNames_)));
}

absl::Status KalmanFilter::checkInitialized(absl::string_view method) const {
  if (stateNames_.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("[KalmanFilter] ", method,
                     "() before reset(): the filter has no state channels"));
  }
  return absl::OkStatus();
}

void KalmanFilter::resizeWorkspace() {
  const Eigen::Index stateDim = stateEstimate_.size();
  const Eigen::Index inputDim = workspace_.inputCapacity;
  const Eigen::Index measurementDim = workspace_.measurementCapacity;
  workspace_.stateHasProcessModel.assign(stateNames_.size(), 0);
  workspace_.inputOffsets.reserve(static_cast<size_t>(inputDim));

  workspace_.stateTransition.resize(stateDim, stateDim);
  workspace_.controlInputMatrix.resize(stateDim, inputDim);
  workspace_.processNoiseCovariance.resize(stateDim, stateDim);
  workspace_.controlInput.resize(inputDim);
  workspace_.inputNoiseCovariance.resize(inputDim, inputDim);
  workspace_.controlInputTimesInputNoise.resize(stateDim, inputDim);
  workspace_.AP.resize(stateDim, stateDim);

  workspace_.measurementModel.resize(measurementDim, stateDim);
  workspace_.measurementNoiseCovariance.resize(measurementDim, measurementDim);
  workspace_.innovation.resize(measurementDim);
  workspace_.PHt.resize(stateDim, measurementDim);
  workspace_.innovationCovariance.resize(measurementDim, measurementDim);
  workspace_.gainTranspose.resize(measurementDim, stateDim);
  workspace_.KR.resize(stateDim, measurementDim);
  workspace_.identityMinusKh.resize(stateDim, stateDim);
  workspace_.identityMinusKhP.resize(stateDim, stateDim);

  workspace_.stateEstimateNext.resize(stateDim);
  workspace_.covarianceNext.resize(stateDim, stateDim);
}

void KalmanFilter::growWorkspace(Eigen::Index inputDim,
                                 Eigen::Index measurementDim) {
  if (inputDim <= workspace_.inputCapacity &&
      measurementDim <= workspace_.measurementCapacity) {
    return;
  }
  workspace_.inputCapacity = std::max(workspace_.inputCapacity, inputDim);
  workspace_.measurementCapacity =
      std::max(workspace_.measurementCapacity, measurementDim);
  resizeWorkspace();
}

}  // namespace ocs2::humanoid::estimation
