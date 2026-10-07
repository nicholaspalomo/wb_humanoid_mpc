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

#include "ocs2_core/control/ManifoldLinearController.h"

#include <iostream>
#include <stdexcept>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "ocs2_core/NumericTraits.h"
#include "ocs2_core/misc/LinearInterpolation.h"

namespace ocs2 {

ManifoldLinearController::ManifoldLinearController(scalar_array_t timeStamp,
                                                   vector_array_t anchorStates,
                                                   vector_array_t nominalInputs,
                                                   matrix_array_t feedbackGains,
                                                   std::shared_ptr<const StateManifold> stateManifold)
    : timeStamp_(std::move(timeStamp)),
      anchorStateArray_(std::move(anchorStates)),
      nominalInputArray_(std::move(nominalInputs)),
      gainArray_(std::move(feedbackGains)),
      stateManifoldPtr_(std::move(stateManifold)) {
  CHECK(stateManifoldPtr_ != nullptr) << "[ManifoldLinearController] The state manifold must not be null.";
  CHECK_EQ(anchorStateArray_.size(), timeStamp_.size());
  CHECK_EQ(nominalInputArray_.size(), timeStamp_.size());
  CHECK_EQ(gainArray_.size(), timeStamp_.size());
}

vector_t ManifoldLinearController::nodeInput(size_t k, const vector_t& x) const {
  vector_t input = nominalInputArray_[k];
  input.noalias() += gainArray_[k] * stateManifoldPtr_->difference(anchorStateArray_[k], x);
  return input;
}

vector_t ManifoldLinearController::computeInput(scalar_t t, const vector_t& x) {
  return blendedInput(t, x);
}

vector_t ManifoldLinearController::blendedInput(scalar_t t, const vector_t& x) const {
  CHECK(!timeStamp_.empty()) << "[ManifoldLinearController::computeInput] The controller is empty.";
  const LinearInterpolation::index_alpha_t indexAlpha = LinearInterpolation::timeSegment(t, timeStamp_);
  const size_t k = static_cast<size_t>(indexAlpha.first);
  const scalar_t alpha = indexAlpha.second;  // the weight of node k; 1 - alpha is the weight of node k + 1
  if (timeStamp_.size() == 1 || alpha >= 1.0) {
    return nodeInput(k, x);
  }
  if (alpha <= 0.0) {
    return nodeInput(k + 1, x);
  }
  return alpha * nodeInput(k, x) + (1.0 - alpha) * nodeInput(k + 1, x);
}

matrix_t ManifoldLinearController::getFeedbackGain(scalar_t time) const {
  return LinearInterpolation::interpolate(time, timeStamp_, gainArray_);
}

vector_t ManifoldLinearController::getNominalInput(scalar_t time) const {
  return LinearInterpolation::interpolate(time, timeStamp_, nominalInputArray_);
}

vector_t ManifoldLinearController::getAnchorState(scalar_t time) const {
  CHECK(!timeStamp_.empty()) << "[ManifoldLinearController::getAnchorState] The controller is empty.";
  const LinearInterpolation::index_alpha_t indexAlpha = LinearInterpolation::timeSegment(time, timeStamp_);
  const size_t k = static_cast<size_t>(indexAlpha.first);
  if (timeStamp_.size() == 1 || indexAlpha.second >= 1.0) {
    return anchorStateArray_[k];
  }
  if (indexAlpha.second <= 0.0) {
    return anchorStateArray_[k + 1];
  }
  return stateManifoldPtr_->interpolate(anchorStateArray_[k], anchorStateArray_[k + 1], 1.0 - indexAlpha.second);
}

// LINT.IfChange(flat_layout)
void ManifoldLinearController::flattenSingle(scalar_t time, std::vector<double>& flatArray) const {
  // [u*; xbar; vec(K)], K column-major. Between two nodes the written input is the blended law evaluated at the
  // interpolated anchor, not the interpolated nominal input: the two differ by w (1 - w) (K_k - K_{k+1}) (xbar_k -
  // xbar_{k+1}), which a reader would otherwise get wrong even on a flat manifold. At a node, and before the first or
  // after the last, the node's own values are written, bit for bit.
  const vector_t anchorState = getAnchorState(time);
  const matrix_t gain = getFeedbackGain(time);
  const LinearInterpolation::index_alpha_t indexAlpha = LinearInterpolation::timeSegment(time, timeStamp_);
  const bool betweenNodes = timeStamp_.size() > 1 && indexAlpha.second > 0.0 && indexAlpha.second < 1.0;
  const vector_t nominalInput = betweenNodes ? blendedInput(time, anchorState) : getNominalInput(time);
  flatArray.resize(nominalInput.size() + anchorState.size() + gain.size());
  Eigen::Map<vector_t> data(flatArray.data(), flatArray.size());
  data.head(nominalInput.size()) = nominalInput;
  data.segment(nominalInput.size(), anchorState.size()) = anchorState;
  data.tail(gain.size()) = Eigen::Map<const vector_t>(gain.data(), gain.size());
}
// LINT.ThenChange(:unflatten_layout, //lib/ocs2/core/include/ocs2_core/control/ManifoldLinearController.h)

void ManifoldLinearController::flatten(const scalar_array_t& timeArray,
                                       const std::vector<std::vector<double>* absl_nonnull>& flatArray2) const {
  if (timeArray.size() != flatArray2.size()) {
    throw std::runtime_error("[ManifoldLinearController::flatten] timeArray and flatArray2 must have the same size.");
  }
  for (size_t i = 0; i < timeArray.size(); ++i) {
    flattenSingle(timeArray[i], *flatArray2[i]);
  }
}

absl::StatusOr<ManifoldLinearController> ManifoldLinearController::unFlatten(
    const size_array_t& stateDim,
    const size_array_t& inputDim,
    const size_array_t& tangentDim,
    const scalar_array_t& timeArray,
    const std::vector<const std::vector<double>* absl_nullable>& flatArray2,
    std::shared_ptr<const StateManifold> stateManifold) {
  if (stateManifold == nullptr) {
    return absl::InvalidArgumentError("[ManifoldLinearController::unFlatten] The state manifold must not be null.");
  }
  const size_t numNodes = timeArray.size();
  if (stateDim.size() != numNodes || inputDim.size() != numNodes || tangentDim.size() != numNodes || flatArray2.size() != numNodes) {
    return absl::InvalidArgumentError(absl::StrCat("[ManifoldLinearController::unFlatten] ", numNodes, " times, but ", stateDim.size(),
                                                   " state, ", inputDim.size(), " input and ", tangentDim.size(), " tangent sizes and ",
                                                   flatArray2.size(), " arrays."));
  }
  vector_array_t anchorStates;
  vector_array_t nominalInputs;
  matrix_array_t gains;
  anchorStates.reserve(numNodes);
  nominalInputs.reserve(numNodes);
  gains.reserve(numNodes);
  // LINT.IfChange(unflatten_layout)
  for (size_t k = 0; k < numNodes; ++k) {
    if (stateDim[k] != stateManifold->getAmbientDim() || tangentDim[k] != stateManifold->getTangentDim()) {
      return absl::InvalidArgumentError(absl::StrCat("[ManifoldLinearController::unFlatten] Node ", k, " has state size ", stateDim[k],
                                                     " and tangent size ", tangentDim[k], ", the manifold ", stateManifold->getAmbientDim(),
                                                     " and ", stateManifold->getTangentDim(), "."));
    }
    const size_t expectedSize = inputDim[k] + stateDim[k] + inputDim[k] * tangentDim[k];
    if (flatArray2[k] == nullptr || flatArray2[k]->size() != expectedSize) {
      return absl::InvalidArgumentError(absl::StrCat("[ManifoldLinearController::unFlatten] Node ", k, " has ",
                                                     flatArray2[k] == nullptr ? 0 : flatArray2[k]->size(), " entries, expected ",
                                                     expectedSize, " = inputs + state + inputs * tangent."));
    }
    const Eigen::Map<const vector_t> data(flatArray2[k]->data(), expectedSize);
    nominalInputs.push_back(data.head(inputDim[k]));
    anchorStates.push_back(data.segment(inputDim[k], stateDim[k]));
    gains.push_back(Eigen::Map<const matrix_t>(data.data() + inputDim[k] + stateDim[k], inputDim[k], tangentDim[k]));
  }
  // LINT.ThenChange(:flat_layout)
  return ManifoldLinearController(timeArray, std::move(anchorStates), std::move(nominalInputs), std::move(gains), std::move(stateManifold));
}

void ManifoldLinearController::concatenate(const ControllerBase* absl_nonnull nextController, int index, int length) {
  const ManifoldLinearController* absl_nullable next = dynamic_cast<const ManifoldLinearController*>(nextController);
  if (next == nullptr) {
    throw std::runtime_error("[ManifoldLinearController::concatenate] Concatenate only works with controllers of the same type.");
  }
  if (!timeStamp_.empty() && !next->timeStamp_.empty() && timeStamp_.back() > next->timeStamp_.front()) {
    throw std::runtime_error("[ManifoldLinearController::concatenate] The next controller must come later in time.");
  }
  if (stateManifoldPtr_ == nullptr) {
    stateManifoldPtr_ = next->stateManifoldPtr_;
  } else if (next->stateManifoldPtr_ != stateManifoldPtr_) {
    throw std::runtime_error("[ManifoldLinearController::concatenate] The controllers live on different state manifolds.");
  }
  const int last = index + length;
  timeStamp_.insert(timeStamp_.end(), next->timeStamp_.begin() + index, next->timeStamp_.begin() + last);
  anchorStateArray_.insert(anchorStateArray_.end(), next->anchorStateArray_.begin() + index, next->anchorStateArray_.begin() + last);
  nominalInputArray_.insert(nominalInputArray_.end(), next->nominalInputArray_.begin() + index, next->nominalInputArray_.begin() + last);
  gainArray_.insert(gainArray_.end(), next->gainArray_.begin() + index, next->gainArray_.begin() + last);
}

void ManifoldLinearController::clear() {
  timeStamp_.clear();
  anchorStateArray_.clear();
  nominalInputArray_.clear();
  gainArray_.clear();
}

void ManifoldLinearController::display() const {
  std::cerr << *this;
}

scalar_array_t ManifoldLinearController::controllerEventTimes() const {
  // The same detection as LinearController::controllerEventTimes(): an event is a pair of (nearly) equal node times.
  if (timeStamp_.size() < 2) {
    return scalar_array_t(0);
  }
  scalar_array_t eventTimes{0.0};
  scalar_t lastEvent = timeStamp_.front();
  for (size_t i = 0; i + 1 < timeStamp_.size(); ++i) {
    const bool eventDetected = timeStamp_[i + 1] - timeStamp_[i] < 2.0 * numeric_traits::weakEpsilon<scalar_t>();
    const bool sufficientTimeSinceEvent = timeStamp_[i] - lastEvent > 2.0 * numeric_traits::weakEpsilon<scalar_t>();
    if (eventDetected && sufficientTimeSinceEvent) {
      eventTimes.push_back(timeStamp_[i]);
      lastEvent = eventTimes.back();
    } else if (eventDetected) {
      eventTimes.back() = timeStamp_[i];
      lastEvent = eventTimes.back();
    }
  }
  return eventTimes;
}

std::ostream& operator<<(std::ostream& out, const ManifoldLinearController& controller) {
  for (size_t k = 0; k < controller.getTimeStamps().size(); ++k) {
    out << "k: " << k << '\n';
    out << "time: " << controller.getTimeStamps()[k] << '\n';
    out << "anchor state: " << controller.getAnchorStates()[k].transpose() << '\n';
    out << "nominal input: " << controller.getNominalInputs()[k].transpose() << '\n';
    out << "gain: " << controller.getFeedbackGains()[k] << '\n';
  }
  return out;
}

}  // namespace ocs2
