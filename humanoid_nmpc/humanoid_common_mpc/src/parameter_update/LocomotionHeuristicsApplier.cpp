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

#include "humanoid_common_mpc/parameter_update/LocomotionHeuristicsApplier.h"

#include <array>
#include <optional>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"

namespace ocs2::humanoid {

namespace {

// LINT.IfChange(locomotion_heuristics_fields)
constexpr std::array<absl::string_view, 10> kFields = {
    "locomotion_heuristics.orientation_compensation",
    "locomotion_heuristics.periodic_orientation",
    "locomotion_heuristics.height_compensation",
    "locomotion_heuristics.hip_centered_stepping",
    "locomotion_heuristics.capture_point",
    "locomotion_heuristics.translational_stepping",
    "locomotion_heuristics.in_place_turning",
    "locomotion_heuristics.high_speed_turning",
    "locomotion_heuristics.impulse_scaling",
    "locomotion_heuristics.centripetal_acceleration",
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/task_file.proto, //humanoid_nmpc/humanoid_mpc_config/locomotion_heuristics_config.proto)
// clang-format on

}  // namespace

absl::Span<const absl::string_view> LocomotionHeuristicsApplier::staticFields() {
  return absl::MakeConstSpan(kFields);
}

void LocomotionHeuristicsApplier::apply(HotUpdateTarget& target) {
  if (layer_ == nullptr) return;
  // A half-typed coefficient in the tuning GUI must not take the controller down: the layer keeps the values it has.
  const std::optional<LocomotionHeuristicConfig> config = convertedOrReported(
      locomotionHeuristicConfigFromConfig(target.task().locomotion_heuristics), target.source(), "locomotion_heuristics");
  if (!config.has_value()) return;
  if (const absl::Status status = layer_->reconfigure(*config); !status.ok()) {
    target.reportNotApplied("locomotion_heuristics", status);
  } else if (!layer_->empty()) {
    LOG(INFO) << "[MpcParameterUpdaterModule] Applied the locomotion_heuristics coefficients from " << target.source() << ":\n"
              << layer_->summary();
  }
}

}  // namespace ocs2::humanoid
