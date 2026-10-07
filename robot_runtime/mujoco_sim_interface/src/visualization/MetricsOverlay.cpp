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

#include <string>

#include "absl/strings/str_format.h"

namespace robot::mujoco_sim_interface {

std::string MetricsOverlay::text(const Metrics& metrics, double renderFps, double elapsedRealTime, double simTime) {
  return absl::StrFormat(
      "Render FPS: %d\n"
      "Sim FPS: %d\n"
      // The actual amount of time elapsed in simulation.
      "Real Time[s]: %.3f\n"
      "Sim  Time[s]: %.3f\n\n"
      // Real-time tracking
      "RTF: %.3f\n"
      "Drift[ms]: %.3f\n"
      "Cumulative Drift[ms]: %.3f",
      static_cast<int>(renderFps), static_cast<int>(metrics.fpsSim), elapsedRealTime, simTime, metrics.rtfSmoothed,
      metrics.driftTick * 1.0e3, metrics.driftCumulative * 1.0e3);
}

void MetricsOverlay::renderOverlay(const VisualizationFrame& frame) {
  if (frame.state == nullptr || frame.state->data == nullptr || frame.context == nullptr) return;
  const MjState& state = *frame.state;
  const std::string metrics = text(state.metrics, frame.renderFps, frame.elapsedRealTime, state.data->time);
  mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, frame.viewport, metrics.c_str(), /*overlay2=*/nullptr, frame.context);
}

}  // namespace robot::mujoco_sim_interface
