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

#include <deque>

#include "absl/functional/function_ref.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The planar motion of the base at the attitude `orientation` (quaternion coefficients x, y, z, w; unit) moving at the
 * world-frame twist `twistWorld` = [v_W, w_W] (MpcRobotModelBase::getBaseTwistInWorldFrame()), at one instant: the
 * velocity R_z(psi_h)^T (v_W,x, v_W,y) in the heading frame and the heading rate d psi_h / dt = J_HT(xi).row(0) R(xi)^T
 * w_W, the heading row of the heading-tilt Jacobian (headingTiltJacobian(), d psi_h / d(dphi) for the body increment
 * dphi = w_B dt) times the body angular velocity. Exact; the heading rate differs from the world yaw rate w_W,z by
 * (theta dphi/dt - dtheta/dt phi) / 2 at second order in the tilt.
 */
BasePlanarMotion instantaneousPlanarMotion(const vector4_t& orientation, const vector6_t& twistWorld);

/**
 * The mean planar motion of the base over its last stride, from the measured base poses of the solves: what the gait
 * switches and a velocity command's intermediate knot compare with the command (design defects D1, D1b, D2).
 *
 * Why a stride mean. A walking humanoid's base sways with the gait cycle P: the pelvis yaws and moves sideways at the
 * stride frequency 1 / P and its harmonics, by more than the gait thresholds (on its recorded straight walk the Unitree
 * G1's pelvis yaw rate (R w_B)_z reaches 0.56 rad/s, RMS 0.27, while its mean over a stride stays below 0.04 rad/s,
 * against the 0.15 rad/s that raises slow_walk to walk), and the instantaneous base twist reads that sway as a turn or a
 * sideways walk. The boxcar mean over the window W,
 *
 *   m(t) = (1 / W) int_{t - W}^{t} y(s) ds,   with the frequency response H(f) = sinc(pi f W) exp(-i pi f W),
 *
 * passes the mean motion (f = 0) with unit gain and has zeros at f = k / W for every k >= 1: with W = P it removes every
 * harmonic of a P-periodic sway exactly, whatever its shape. A sway of another period P' leaks by |sin(pi W / P') / (pi
 * W / P')| (0.20 at W / P' = 1.4 / 1.7). The price is a delay of W / 2 when the mean motion changes. A first-order low
 * pass of time constant P instead would leave 1 / sqrt(1 + (2 pi)^2) = 16 % of the fundamental, and every harmonic.
 *
 * What is averaged, and why it is exact. Only the measured base pose is needed, no velocity:
 *   - the heading rate: the twist heading psi_h of the base attitude (MpcRobotModelBase::getBaseHeading()), unwrapped
 *     into Psi from sample to sample, has the mean (Psi(t) - Psi(t - W)) / W, the exact integral of d psi_h / dt. It is a
 *     function of the base's orientation alone: no joint motion enters it (design README section 2.9). It is the rate the
 *     commands and the knots' yaws are given in. The world yaw rate w_W,z differs from it by (theta dphi/dt - dtheta/dt
 *     phi) / 2 at second order in the tilt, whose mean over a stride is the signed area the (roll, pitch) path encloses
 *     over the stride divided by its duration (zero for the figure eight of a symmetric gait), and its integral is not
 *     a function of the poses, so the samples give no exact mean of it;
 *   - the velocity of the base origin in its heading frame, v_h = R_z(psi_h)^T dp / dt (x, y), the frame of the command:
 *     there the gait's sway is periodic also while the robot turns, where a world-frame mean would lag the heading by
 *     omega W / 2 and fold the sway into the turn. Its integral D = int R_z(psi_h)^T dp is accumulated from sample i to
 *     i + 1 as dD = R_z(Psi_mid)^T dp / sinc(dPsi / 2), Psi_mid = (Psi_i + Psi_i+1) / 2: exact when the base moves at a
 *     constant body planar twist (v_h, omega) between the two samples, whose chord is dp = dt sinc(omega dt / 2)
 *     R_z(Psi_i + omega dt / 2) v_h, and second order in the sample spacing otherwise. The mean is (D(t) - D(t - W)) / W.
 * Between samples Psi and D are interpolated linearly in time: exact under the same constant-twist model, and otherwise
 * off at a window's start by at most max |Psi''| h^2 / 8 for samples h apart (divided by W: below 1e-3 rad/s at the
 * shipped solve rates of 50 to 80 Hz, 0.023 rad/s at h = kMaxSampleGap with the largest heading acceleration of the
 * recorded walks, about 25 rad/s^2, the whole-body G1's).
 *
 * Gaps and restarts. A gap of up to kMaxBridgedGap between two samples (a network stall between the robot and the MPC
 * node, a starved solver thread) is bridged: across it Psi takes the branch of |dPsi| <= pi, the true one while the
 * heading changes by less than pi over the gap g, i.e. while its mean rate there stays below (pi - s) / g, s the
 * heading's peak-to-peak sway (at most 0.24 rad on the recorded walks): below 1.9 rad/s for every bridged gap, against
 * the shipped command limit of 1 rad/s; and D takes the chord step above, exact for a constant body twist across the
 * gap and otherwise off by about |v| times the mean deviation of the heading from its chord there (in the mean of a
 * window that spans a stall of up to 1 s, at most 0.007 m/s on the recorded walks, the heading rate exact). A window
 * that starts inside a bridged gap interpolates Psi and D linearly across it, off by the heading's deviation from that
 * line divided by W (up to 0.044 rad/s and 0.050 m/s for a stall of 1 s on the recorded walks, the whole-body G1's). A
 * longer gap outlasts every shipped MPC horizon (1.0 to 1.2 s), past which the robot has no plan left to walk on and
 * MPC_BASE::run() does not run before a reset: the history restarts - its next sample is a first one again - as it
 * does after reset() and when the clock runs backwards.
 *
 * Before the first sample. A history that has just (re)started holds less than a window. The base is taken to have
 * moved before its first sample at the constant `seed` motion given with that sample (the motion manager gives the
 * exact instantaneous planar motion of the observation the solve starts from, instantaneousPlanarMotion(); in
 * operation a history restarts at a full reset of the MPC, which the controllers make while they hold the robot or it
 * stands, docs/mpc_reset/README.md), so a window that reaches back before the first sample blends the seed with the
 * measured motion in proportion to the time each covers. A seed is one
 * instant, sway and all: on a walking robot it reads the sway as motion for up to a window (seeded with the exact
 * pelvis twist, 67 % and 81 % of the restart instants of the G1's and R1's recorded walks read a turn above 0.15
 * rad/s, for up to 1.3 s). isMeasuredOver() therefore tells a mean measured over its whole window - after the first
 * sample, the samples at most kMaxSampleGap apart throughout - from one that is not, and ProceduralMpcMotionManager's
 * gait switches read only the former.
 *
 * A copyable value; not thread-safe (ProceduralMpcMotionManager uses it on the solver thread).
 */
class StrideMotionHistory {
 public:
  /** [s] The largest spacing of the samples a measured window holds (isMeasuredOver()). */
  static constexpr scalar_t kMaxSampleGap = 0.1;
  /** [s] A gap between two samples beyond which the history restarts; a shorter one is bridged. */
  static constexpr scalar_t kMaxBridgedGap = 1.5;

  /** A history that keeps the samples of the last `retention` seconds (positive): mean() averages over at most that. */
  explicit StrideMotionHistory(scalar_t retention);

  ~StrideMotionHistory() = default;
  StrideMotionHistory(const StrideMotionHistory&) = default;
  StrideMotionHistory& operator=(const StrideMotionHistory&) = default;
  StrideMotionHistory(StrideMotionHistory&&) noexcept = default;
  StrideMotionHistory& operator=(StrideMotionHistory&&) noexcept = default;

  /** Forgets every sample; the next addSample() starts the history again. */
  void reset();

  /**
   * Appends the measured base at `time` [s]: the position `positionXy` [m] of its origin in the world's ground plane and
   * its twist heading `heading` [rad], on any branch. When the history (re)starts with this sample (see the class
   * comment), `seed` is called once for the motion before it; otherwise it is not called. A sample at the time of the
   * last one is dropped.
   */
  void addSample(scalar_t time, const vector2_t& positionXy, scalar_t heading, absl::FunctionRef<BasePlanarMotion()> seed);

  /**
   * The mean planar motion of the base over the `window` seconds (positive, unchecked) up to the last sample, the window
   * cut to the retention. Before the first sample (see the class comment) the motion is the seed's. Zero motion while
   * the history is empty.
   */
  BasePlanarMotion mean(scalar_t window) const;

  /**
   * Whether mean(`window`) is measured over its whole window: the history holds samples over the last `window` seconds
   * (cut to the retention) at most kMaxSampleGap apart, so no seed and no bridged gap enters the mean. False while the
   * history is empty.
   */
  bool isMeasuredOver(scalar_t window) const;

  bool empty() const { return samples_.empty(); }

 private:
  /** A sample: the cumulative heading Psi and heading-frame displacement D at a time. */
  struct Sample {
    scalar_t time = 0.0;
    /** [rad] Psi, the unwrapped twist heading. */
    scalar_t heading = 0.0;
    /** [m] D, the displacement integrated in the heading frame since the first sample. */
    vector2_t displacement = vector2_t::Zero();
  };

  /** Psi and D at `time`, at most the last sample's: interpolated between samples, extrapolated with the seed before them. */
  Sample at(scalar_t time) const;

  // [s] Positive.
  scalar_t retention_ = 0.0;
  // In increasing time; samples_.front() is the first sample of the history until it is pruned.
  std::deque<Sample> samples_;
  // The base position of samples_.back(), to which the next displacement is taken.
  vector2_t lastPosition_ = vector2_t::Zero();
  // The motion before the first sample; read only while samples_.front() is that sample (firstSampleKept_).
  BasePlanarMotion seed_;
  bool firstSampleKept_ = false;
  // [s] The time of the first sample from which on the samples are at most kMaxSampleGap apart: the first sample of the
  // history, or the first after its last bridged gap. Read only while the history is not empty.
  scalar_t measuredSince_ = 0.0;
};

}  // namespace ocs2::humanoid
