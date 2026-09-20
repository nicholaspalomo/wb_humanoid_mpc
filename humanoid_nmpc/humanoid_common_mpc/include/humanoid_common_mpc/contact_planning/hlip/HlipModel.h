/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <utility>

#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The Hybrid Linear Inverted Pendulum (H-LIP) of Xiong and Ames, as used by the reduced-order layer of
 * "Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion"
 * (Esteban, Kurtz, Ghansah, Ames, arXiv:2502.15630).
 *
 * One planar pendulum of fixed height z0. The state is the horizontal position and velocity of the point mass
 * relative to the stance foot, x = [p, v]^T. A step consists of a single support phase (SSP) of `sspDuration`, during
 * which the pendulum is unactuated, an instantaneous impact that hands the stance over to the foot placed a step
 * length ahead, and a double support phase (DSP) of `dspDuration`, during which the mass is assumed to drift at
 * constant velocity. Composing the three gives the step-to-step (S2S) dynamics on the pre-impact state,
 *
 *   x_{k+1} = A x_k + B u_k,   A = A_ssp A_dsp,   B = -A e_1,
 *
 * with A_ssp = [[cosh(w T_ssp), sinh(w T_ssp) / w], [w sinh(w T_ssp), cosh(w T_ssp)]], A_dsp = [[1, T_dsp], [0, 1]]
 * and w = sqrt(g / z0). The S2S map is the whole planner: the step length is the only input, and the gain that drives
 * the error to zero in two steps is closed form,
 *
 *   K = [1, T_dsp + coth(w T_ssp) / w],   A + B K = [[0, -1 / (w sinh(w T_ssp))], [0, 0]],
 *
 * which is nilpotent, so no gain is tuned anywhere in this class. Three dimensional walking is the orthogonal
 * composition of two of these models, a period-one orbit along the heading and a period-two orbit laterally; both
 * orbits are solved here as linear systems rather than approximated.
 *
 * The model carries no state. It is a cheap value type: construct it from the configuration and copy it freely.
 */
class HlipModel {
 public:
  using State = vector2_t;  // [p, v] of the point mass relative to the stance foot
  using Matrix2 = Eigen::Matrix<scalar_t, 2, 2>;

  /**
   * @param sspDuration  [s] single support duration, must be positive.
   * @param dspDuration  [s] double support duration, must not be negative (the paper uses 0).
   * @param comHeight    [m] pendulum height z0, must be positive.
   * @param gravity      [m/s^2] must be positive.
   *
   * The preconditions are checked; ContactPlanningConfig::validate() rejects a configuration that would violate them
   * before a model is ever built from it.
   */
  HlipModel(scalar_t sspDuration, scalar_t dspDuration, scalar_t comHeight, scalar_t gravity);

  scalar_t sspDuration() const { return sspDuration_; }
  scalar_t dspDuration() const { return dspDuration_; }
  /** [s] duration of a whole step, SSP and DSP together. */
  scalar_t stepDuration() const { return sspDuration_ + dspDuration_; }
  /** [1/s] the pendulum's natural frequency w = sqrt(g / z0). */
  scalar_t naturalFrequency() const { return omega_; }

  const Matrix2& stepToStepA() const { return stepToStepA_; }
  const vector2_t& stepToStepB() const { return stepToStepB_; }
  /** The deadbeat row K, so that u = uNominal + K (x - xNominal). */
  const vector2_t& deadbeatGain() const { return deadbeatGain_; }

  /** Flows the unactuated single support dynamics for `duration` seconds. */
  State flowSingleSupport(const State& state, scalar_t duration) const;
  /** Drifts at constant velocity for `duration` seconds, the model's double support. */
  static State flowDoubleSupport(const State& state, scalar_t duration);
  /** The impact: the foot placed `stepLength` ahead of the stance foot becomes the stance foot. */
  static State applyStepTransition(const State& state, scalar_t stepLength);

  /** The pre-impact state after one whole step of length `stepLength`, i.e. A x + B u. */
  State applyStepToStep(const State& state, scalar_t stepLength) const;

  /** The pre-impact fixed point of the period-one orbit walked with a constant step length. */
  State periodOneOrbit(scalar_t stepLength) const;
  /**
   * The two pre-impact states of the period-two orbit that alternates the two step lengths, the lateral orbit of a
   * biped. The first entry is the state that precedes a step of `firstStepLength`.
   */
  std::pair<State, State> periodTwoOrbit(scalar_t firstStepLength, scalar_t secondStepLength) const;

  /**
   * The deadbeat step length u = nominalStepLength + K (state - nominalState). Applied at the pre-impact state of a
   * step, it returns the model to the nominal orbit within two steps.
   */
  scalar_t deadbeatStepLength(const State& state, const State& nominalState, scalar_t nominalStepLength) const;

 private:
  scalar_t sspDuration_;
  scalar_t dspDuration_;
  scalar_t comHeight_;
  scalar_t gravity_;
  scalar_t omega_;
  Matrix2 stepToStepA_;
  vector2_t stepToStepB_;
  vector2_t deadbeatGain_;
};

}  // namespace ocs2::humanoid
