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

#include "humanoid_common_mpc/contact_planning/hlip/HlipModel.h"

#include <cmath>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {

namespace {
// The linear system of the period-two orbit: the pre-impact states of its two steps, stacked.
constexpr int kPeriodTwoOrbitSteps = 2;
constexpr int kPeriodTwoSystemDim = kPeriodTwoOrbitSteps * HlipModel::kStateDim;
}  // namespace

absl::StatusOr<HlipModel> HlipModel::Create(scalar_t sspDuration, scalar_t dspDuration, scalar_t comHeight, scalar_t gravity) {
  // Negated comparisons, so that NaN is refused too.
  if (!(sspDuration > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("[HlipModel] the single support duration must be positive, got ", sspDuration, " s"));
  }
  if (!(dspDuration >= 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[HlipModel] the double support duration must not be negative, got ", dspDuration, " s"));
  }
  if (!(comHeight > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("[HlipModel] the pendulum height must be positive, got ", comHeight, " m"));
  }
  if (!(gravity > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("[HlipModel] gravity must be positive, got ", gravity, " m/s^2"));
  }
  return HlipModel(sspDuration, dspDuration, comHeight, gravity);
}

HlipModel::HlipModel(scalar_t sspDuration, scalar_t dspDuration, scalar_t comHeight, scalar_t gravity)
    : sspDuration_(sspDuration), dspDuration_(dspDuration), comHeight_(comHeight), gravity_(gravity) {
  omega_ = std::sqrt(gravity_ / comHeight_);

  const scalar_t coshTerm = std::cosh(omega_ * sspDuration_);
  const scalar_t sinhTerm = std::sinh(omega_ * sspDuration_);

  Matrix2 singleSupport;
  singleSupport << coshTerm, sinhTerm / omega_, omega_ * sinhTerm, coshTerm;
  Matrix2 doubleSupport;
  doubleSupport << 1.0, dspDuration_, 0.0, 1.0;

  stepToStepA_ = singleSupport * doubleSupport;
  // The impact subtracts the step length from the position, so the input column is minus the first column of A.
  stepToStepB_ = -stepToStepA_.col(0);
  // K = [1, T_dsp + coth(w T_ssp) / w] makes A + B K nilpotent; see the class documentation.
  deadbeatGain_ << 1.0, dspDuration_ + coshTerm / (omega_ * sinhTerm);
}

HlipModel::State HlipModel::flowSingleSupport(const State& state, scalar_t duration) const {
  const scalar_t coshTerm = std::cosh(omega_ * duration);
  const scalar_t sinhTerm = std::sinh(omega_ * duration);
  State flowed;
  flowed(0) = coshTerm * state(0) + sinhTerm / omega_ * state(1);
  flowed(1) = omega_ * sinhTerm * state(0) + coshTerm * state(1);
  return flowed;
}

HlipModel::State HlipModel::flowDoubleSupport(const State& state, scalar_t duration) {
  return State(state(0) + duration * state(1), state(1));
}

HlipModel::State HlipModel::applyStepTransition(const State& state, scalar_t stepLength) {
  return State(state(0) - stepLength, state(1));
}

HlipModel::State HlipModel::applyStepToStep(const State& state, scalar_t stepLength) const {
  return stepToStepA_ * state + stepToStepB_ * stepLength;
}

HlipModel::State HlipModel::periodOneOrbit(scalar_t stepLength) const {
  // x = A x + B u, which is regular because A has no unit eigenvalue for a positive step duration.
  const Matrix2 system = Matrix2::Identity() - stepToStepA_;
  return system.colPivHouseholderQr().solve(State(stepToStepB_ * stepLength));
}

HlipModel::PeriodTwoOrbit HlipModel::periodTwoOrbit(scalar_t firstStepLength, scalar_t secondStepLength) const {
  // The orbit alternates the two steps: x1 = A x2 + B u2 and x2 = A x1 + B u1, one 4 x 4 linear system.
  Eigen::Matrix<scalar_t, kPeriodTwoSystemDim, kPeriodTwoSystemDim> system =
      Eigen::Matrix<scalar_t, kPeriodTwoSystemDim, kPeriodTwoSystemDim>::Identity();
  system.topRightCorner<kStateDim, kStateDim>() = -stepToStepA_;
  system.bottomLeftCorner<kStateDim, kStateDim>() = -stepToStepA_;
  Eigen::Matrix<scalar_t, kPeriodTwoSystemDim, 1> rhs;
  rhs.head<kStateDim>() = stepToStepB_ * secondStepLength;
  rhs.tail<kStateDim>() = stepToStepB_ * firstStepLength;
  const Eigen::Matrix<scalar_t, kPeriodTwoSystemDim, 1> solution = system.colPivHouseholderQr().solve(rhs);
  return {.beforeFirstStep = State(solution.head<kStateDim>()), .beforeSecondStep = State(solution.tail<kStateDim>())};
}

scalar_t HlipModel::deadbeatStepLength(const State& state, const State& nominalState, scalar_t nominalStepLength) const {
  return nominalStepLength + deadbeatGain_.dot(state - nominalState);
}

}  // namespace ocs2::humanoid
