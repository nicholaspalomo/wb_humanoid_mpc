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

#include "mujoco_sim_interface/visualization/MetricsOverlay.h"

#include <iomanip>
#include <sstream>

namespace robot::mujoco_sim_interface {

void MetricsOverlay::renderOverlay(const VisualizationFrame& frame) {
  if (frame.state == nullptr || frame.state->data == nullptr || frame.context == nullptr) return;
  const MjState& state = *frame.state;
  std::ostringstream metrics;

  // FPS (Simulation & Renderer)
  metrics << "Render FPS: " << static_cast<int>(frame.renderFps) << "\n";
  metrics << "Sim FPS: " << static_cast<int>(state.metrics.fpsSim) << "\n";

  // The actual amount of time elapsed in simulation.
  metrics << "Real Time[s]: " << std::fixed << std::setprecision(3) << frame.elapsedRealTime << "\n";
  metrics << "Sim  Time[s]: " << std::fixed << std::setprecision(3) << state.data->time << "\n\n";

  // Real-time tracking
  metrics << "RTF: " << std::fixed << std::setprecision(3) << state.metrics.rtfSmoothed << "\n";
  metrics << "Drift[ms]: " << std::fixed << std::setprecision(3) << state.metrics.driftTick * 1e3 << "\n";
  metrics << "Cummulative Drift[ms]: " << std::fixed << std::setprecision(3) << state.metrics.driftCumulative * 1e3;

  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, frame.viewport, metrics.str().c_str(), nullptr, frame.context);
}

}  // namespace robot::mujoco_sim_interface
