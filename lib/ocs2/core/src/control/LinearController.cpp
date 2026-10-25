/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <iostream>
#include <utility>

#include "absl/base/nullability.h"

#include <ocs2_core/control/LinearController.h>

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
LinearController::LinearController(const LinearController& other)
    : LinearController(other.timeStamp_, other.biasArray_, other.gainArray_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
LinearController::LinearController(LinearController&& other) : ControllerBase(other) {
  swap(other, *this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
LinearController& LinearController::operator=(LinearController rhs) {
  swap(rhs, *this);
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
LinearController* absl_nonnull LinearController::clone() const {
  return new LinearController(*this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector_t LinearController::computeInput(scalar_t t, const vector_t& x) {
  const auto indexAlpha = LinearInterpolation::timeSegment(t, timeStamp_);

  vector_t uff = LinearInterpolation::interpolate(indexAlpha, biasArray_);
  const matrix_t k = LinearInterpolation::interpolate(indexAlpha, gainArray_);

  uff.noalias() += k * x;
  return uff;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void LinearController::flatten(const scalar_array_t& timeArray, const std::vector<std::vector<double>* absl_nonnull>& flatArray2) const {
  const auto timeSize = timeArray.size();
  const auto dataSize = flatArray2.size();

  if (timeSize != dataSize) {
    throw std::runtime_error("timeSize and dataSize must be equal in flatten method.");
  }

  for (size_t i = 0; i < timeSize; i++) {
    flattenSingle(timeArray[i], *(flatArray2[i]));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void LinearController::flattenSingle(scalar_t time, std::vector<double>& flatArray) const {
  /* Serialized linear controller:
   * data = [
   *   // t0
   *   [
   *     uff[0], k[0, :],
   *     uff[1], k[1, :],
   *     ...
   *     uff[m], k[m, :]
   *   ],
   *
   *   ...
   *
   *   // tN
   *   [
   *     uff[0], k[0, :],
   *     uff[1], k[1, :],
   *     ...
   *     uff[m], k[m, :]
   *   ]
   * ]
   */

  const auto indexAlpha = LinearInterpolation::timeSegment(time, timeStamp_);
  const vector_t uff = LinearInterpolation::interpolate(indexAlpha, biasArray_);
  const matrix_t k = LinearInterpolation::interpolate(indexAlpha, gainArray_);

  const size_t stateDim = k.cols();
  const size_t inputDim = k.rows();

  flatArray.clear();
  flatArray.resize(uff.size() + k.size());

  for (size_t i = 0; i < inputDim; i++) {  // i loops through rows of uff and k
    flatArray[i * (stateDim + 1) + 0] = uff(i);
    for (size_t j = 0; j < stateDim; j++) {  // j loops through cols of k
      flatArray[i * (stateDim + 1) + j + 1] = k(i, j);
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
LinearController LinearController::unFlatten(const size_array_t& stateDim,
                                             const size_array_t& inputDim,
                                             const scalar_array_t& timeArray,
                                             const std::vector<const std::vector<double>* absl_nonnull>& flatArray2) {
  vector_array_t bias;
  matrix_array_t gain;

  bias.reserve(flatArray2.size());
  gain.reserve(flatArray2.size());

  for (size_t k = 0; k < timeArray.size(); k++) {  // loop through time
    if (flatArray2[k]->size() != inputDim[k] + inputDim[k] * stateDim[k]) {
      throw std::runtime_error("LinearController::unFlatten received array of wrong length.");
    }

    bias.emplace_back(vector_t(inputDim[k]));
    gain.emplace_back(matrix_t(inputDim[k], stateDim[k]));

    const auto& arr = *flatArray2[k];
    for (size_t i = 0; i < inputDim[k]; i++) {  // loop through input dim
      bias.back()(i) = static_cast<scalar_t>(arr[i * (stateDim[k] + 1) + 0]);
      gain.back().row(i) = Eigen::Map<const Eigen::VectorXd>(&(arr[i * (stateDim[k] + 1) + 1]), stateDim[k]);
    }
  }
  return LinearController(timeArray, std::move(bias), std::move(gain));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
int LinearController::size() const {
  return timeStamp_.size();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ControllerType LinearController::getType() const {
  return ControllerType::LINEAR;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void LinearController::clear() {
  timeStamp_.clear();
  biasArray_.clear();
  gainArray_.clear();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool LinearController::empty() const {
  return timeStamp_.empty();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void LinearController::display() const {
  std::cerr << *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void swap(LinearController& a, LinearController& b) noexcept {
  std::swap(a.timeStamp_, b.timeStamp_);
  std::swap(a.biasArray_, b.biasArray_);
  std::swap(a.gainArray_, b.gainArray_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::ostream& operator<<(std::ostream& out, const LinearController& controller) {
  for (size_t k = 0; k < controller.timeStamp_.size(); k++) {
    out << "k: " << k << '\n';
    out << "time: " << controller.timeStamp_[k] << '\n';
    out << "bias: " << controller.biasArray_[k].transpose() << '\n';
    out << "gain: " << controller.gainArray_[k] << '\n';
  }
  return out;
}

}  // namespace ocs2
