/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <mujoco/mujoco.h>

namespace robot::mujoco_sim_interface {

struct Metrics {
  /// FPS of Simulation::step
  double fpsSim;
  /// Real time factor for current sim step: RTF = dt_sim / dt_real
  double rtfTick;
  /// Smoothed RTF (exponential moving average)
  double rtfSmoothed;
  /// Time drift per-tick.
  double driftTick;
  /// Total time drift since starting the sim.
  double driftCumulative;

  void reset() {
    fpsSim = 0.0;
    rtfTick = 0.0;
    rtfSmoothed = 1.0;  // Start at ideal value
    driftTick = 0.0;
    driftCumulative = 0.0;
  }
};

struct MjState {
  explicit MjState(const mjModel* model);

  // Deep-copy: allocate new mjData and copy contents
  MjState(const MjState& other);
  MjState& operator=(const MjState& other);

  // Move: transfer ownership of mjData
  MjState(MjState&& other) noexcept;
  MjState& operator=(MjState&& other) noexcept;

  ~MjState();

  const mjModel* model{nullptr};
  int64_t timestamp{0};
  mjData* data{nullptr};
  Metrics metrics;
};
}  // namespace robot::mujoco_sim_interface
