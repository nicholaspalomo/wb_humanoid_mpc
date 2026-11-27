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
#include <vector>

#include "robot_core/Types.h"

namespace robot::mujoco_sim_interface {

/**
 * The simulated accelerometer's acceleration of the base: its mean over the
 * last few physics steps, from the base's world-frame velocity at the end of
 * each step.
 *
 * The simulator records the base velocity after every step (record()) and
 * reads linaccBaseRtWorldEwrtWorld() = (v_now - v_then) / (t_now - t_then),
 * with `then` windowSteps() steps back, or the oldest sample since the last
 * restart() when fewer were recorded. This is the acceleration the integrator
 * applied, which MuJoCo's qacc is not where the joint damping is integrated
 * implicitly; and over a window of one control period it is the change of
 * velocity a reader that takes the state once per period sees, without the
 * aliasing of a single step's acceleration read once every many steps.
 *
 * Usage: restart() after the state was set by hand (a reset, a teleport), then
 * record() after every step. Storage is sized at construction, so record(),
 * restart() and the read allocate nothing. Not thread-safe.
 */
class BaseAccelerometer {
 public:
  /** Returns the steps of a window of `window` [s] at the physics step `dt`
   * [s]: round(window / dt), at least 1 (one step for a window of 0). */
  static int windowStepsFor(double window, double dt);

  /** An accelerometer over `windowSteps` steps; fewer than 1 counts as 1. It
   * reads zero until it has two samples. */
  explicit BaseAccelerometer(int windowSteps);

  BaseAccelerometer(const BaseAccelerometer&) = default;
  BaseAccelerometer& operator=(const BaseAccelerometer&) = default;
  BaseAccelerometer(BaseAccelerometer&&) noexcept = default;
  BaseAccelerometer& operator=(BaseAccelerometer&&) noexcept = default;
  ~BaseAccelerometer() = default;

  /** Forgets every sample and records the base velocity
   * `linvelBaseRtWorldEwrtWorld` [m/s] at `time` [s] as the first. */
  void restart(double time, const vector3_t& linvelBaseRtWorldEwrtWorld);

  /** Records the base velocity [m/s] at the end of a step at `time` [s]. A
   * sample at or before the newest one's time replaces the newest: the state
   * was published again without a step. */
  void record(double time, const vector3_t& linvelBaseRtWorldEwrtWorld);

  /** [m/s^2] The mean acceleration of the base in the world over the window;
   * zero with fewer than two samples. */
  vector3_t linaccBaseRtWorldEwrtWorld() const;

  int windowSteps() const { return static_cast<int>(samples_.size()) - 1; }

 private:
  struct Sample {
    double time = 0.0;
    vector3_t linvelBaseRtWorldEwrtWorld = vector3_t::Zero();
  };

  /// A ring of windowSteps() + 1 samples; newest_ indexes the newest of the
  /// count_ recorded ones, the others are older in reverse ring order.
  std::vector<Sample> samples_;
  size_t newest_ = 0;
  size_t count_ = 0;
};

}  // namespace robot::mujoco_sim_interface
