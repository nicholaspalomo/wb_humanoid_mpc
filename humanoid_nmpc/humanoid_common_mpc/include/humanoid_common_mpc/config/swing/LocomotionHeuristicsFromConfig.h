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

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_mpc_config/locomotion_heuristics_config.nproto.h"

namespace ocs2::humanoid {

/**
 * The locomotion heuristic layer's configuration from the task file's locomotion_heuristics block.
 *
 * The three lists are copied as written, and each parameter block becomes the parameter struct of its heuristic, field
 * by field (capture_point.com_height_override is capturePoint.comHeightOverride). The schema's defaults are the
 * layer's own, so an absent block gives LocomotionHeuristicConfig{}, whose lists are empty: an exact no-op. Nothing is logged: a verbose
 * start-up prints the file, and formulation.summary() the lists.
 *
 * @param config The locomotion_heuristics block.
 * @return The configuration; InvalidArgument when a value is not finite (inf or nan), and every error of
 * LocomotionHeuristicConfig::validate(): a name that is unknown, in the wrong list or listed twice, and a parameter outside its range.
 */
absl::StatusOr<LocomotionHeuristicConfig> locomotionHeuristicConfigFromConfig(const mpc_config::LocomotionHeuristicsConfig& config);

}  // namespace ocs2::humanoid
