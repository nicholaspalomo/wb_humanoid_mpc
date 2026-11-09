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

#include "humanoid_common_mpc/reference_manager/GaitSwitchingTable.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigValueChecks.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/gait_switching_table_config.nproto.h"

namespace ocs2::humanoid {
namespace {

// The gait the first row names: the one the reference file's initial STANCE
// schedule runs.
constexpr char kStartGait[] = "stance";

/** The threshold `field` of the row at `rowPath`; InvalidArgument naming it
 * when it is absent or not finite. */
absl::StatusOr<scalar_t> requiredThreshold(absl::string_view rowPath,
                                           absl::string_view field,
                                           std::optional<double> value) {
  const std::string path = absl::StrCat(rowPath, ".", field);
  if (!value.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat(path,
                     " is absent; every threshold of a gait_switching row is "
                     "required"));
  }
  RETURN_IF_ERROR(requireFinite(path, *value));
  return *value;
}

/** The row `row` at `rowPath` as a GaitModeStateConfig, its thresholds present,
 * finite and each cmd_min_* bound at most its cmd_max_* bound. */
absl::StatusOr<GaitModeStateConfig> gaitModeStateFromConfig(
    const mpc_config::GaitSwitchingTableConfig::Row& row,
    absl::string_view rowPath) {
  if (row.gait.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        rowPath, ".gait is empty; it names a gait of the gait file"));
  }
  GaitModeStateConfig state{.gaitCommand = row.gait};
  ASSIGN_OR_RETURN(
      state.cmdMinLinvelBaseRtWorldEwrtHeadingXy,
      requiredThreshold(rowPath, "cmd_min_linvel_base_rt_world_ewrt_heading_xy",
                        row.cmd_min_linvel_base_rt_world_ewrt_heading_xy));
  ASSIGN_OR_RETURN(
      state.cmdMaxLinvelBaseRtWorldEwrtHeadingXy,
      requiredThreshold(rowPath, "cmd_max_linvel_base_rt_world_ewrt_heading_xy",
                        row.cmd_max_linvel_base_rt_world_ewrt_heading_xy));
  ASSIGN_OR_RETURN(
      state.cmdMinAngvelBaseRtWorldEwrtWorldZ,
      requiredThreshold(rowPath, "cmd_min_angvel_base_rt_world_ewrt_world_z",
                        row.cmd_min_angvel_base_rt_world_ewrt_world_z));
  ASSIGN_OR_RETURN(
      state.cmdMaxAngvelBaseRtWorldEwrtWorldZ,
      requiredThreshold(rowPath, "cmd_max_angvel_base_rt_world_ewrt_world_z",
                        row.cmd_max_angvel_base_rt_world_ewrt_world_z));
  ASSIGN_OR_RETURN(
      state.maxErrLinvelBaseRtWorldEwrtHeadingXy,
      requiredThreshold(rowPath, "max_err_linvel_base_rt_world_ewrt_heading_xy",
                        row.max_err_linvel_base_rt_world_ewrt_heading_xy));
  ASSIGN_OR_RETURN(
      state.maxErrAngvelBaseRtWorldEwrtWorldZ,
      requiredThreshold(rowPath, "max_err_angvel_base_rt_world_ewrt_world_z",
                        row.max_err_angvel_base_rt_world_ewrt_world_z));
  RETURN_IF_ERROR(requireOrdered(
      absl::StrCat(rowPath, ".cmd_min_linvel_base_rt_world_ewrt_heading_xy"),
      state.cmdMinLinvelBaseRtWorldEwrtHeadingXy,
      absl::StrCat(rowPath, ".cmd_max_linvel_base_rt_world_ewrt_heading_xy"),
      state.cmdMaxLinvelBaseRtWorldEwrtHeadingXy));
  RETURN_IF_ERROR(requireOrdered(
      absl::StrCat(rowPath, ".cmd_min_angvel_base_rt_world_ewrt_world_z"),
      state.cmdMinAngvelBaseRtWorldEwrtWorldZ,
      absl::StrCat(rowPath, ".cmd_max_angvel_base_rt_world_ewrt_world_z"),
      state.cmdMaxAngvelBaseRtWorldEwrtWorldZ));
  return state;
}

}  // namespace

std::vector<GaitModeStateConfig> defaultGaitModeStates() {
  // The tuned thresholds of the biped gaits, one row per gait, in the order of
  // GaitModeStateConfig's fields. tools/locomotion_heuristics/
  // test_derive_parameters.py reads the gait names out of this initializer.
  // LINT.IfChange(default_gait_switching_table)
  // NOLINTBEGIN(magic-number): the tuned thresholds of the gait table, one row per gait (the comment above).
  // clang-format off
  return {
    { "stance",       -0.1,  0.1, -0.1,  0.1,  10.0,   10.0 }, // Large threshold allows switching away from stance purely command based.
    { "slow_walk",     0.05,  0.3,  0.05,  0.2,    0.05,  0.05},
    { "walk",          0.25,  0.5, 0.15,  0.35,    0.05,  0.05},
    { "slower_trot",   0.45, 0.7, 0.3,  0.55,      0.1,  0.1},
    { "slow_trot",     0.65, 0.9,  0.5,  0.7,      0.2,  0.2},
    { "trot",          0.8,  1.3,  0.65,  10.0,    0.2,  0.2},
    { "run",           1.2,  10.0,  0.65,  10.0,   0.2,  0.2}
  };  // clang-format on
  // NOLINTEND(magic-number)
  // LINT.ThenChange(//tools/locomotion_heuristics/test_derive_parameters.py:speed_ladder, //humanoid_nmpc/humanoid_mpc_config/gait_file.proto:default_gait_switching_table)
}

absl::StatusOr<std::vector<GaitModeStateConfig>> gaitModeStatesFromConfig(
    const mpc_config::GaitFile& gaitFile) {
  if (!gaitFile.gait_switching.has_value()) return defaultGaitModeStates();
  const std::vector<mpc_config::GaitSwitchingTableConfig::Row>& rows =
      gaitFile.gait_switching->rows;
  if (rows.empty()) {
    return absl::InvalidArgumentError(
        "gait_switching.rows is empty; the table needs at least one gait (or "
        "leave gait_switching out for the default table)");
  }
  // ProceduralMpcMotionManager::reset() takes row 0 as the gait the schedule
  // runs at start, and the reference file starts every schedule in STANCE.
  if (rows[0].gait != kStartGait) {
    return absl::InvalidArgumentError(
        absl::StrCat("gait_switching.rows[0].gait is '", rows[0].gait,
                     "'; the first row is the gait the robot starts in, '",
                     kStartGait, "'"));
  }
  std::vector<GaitModeStateConfig> states;
  states.reserve(rows.size());
  absl::flat_hash_set<std::string> named;
  for (size_t i = 0; i < rows.size(); ++i) {
    const std::string rowPath = absl::StrCat("gait_switching.rows[", i, "]");
    ASSIGN_OR_RETURN(GaitModeStateConfig state,
                     gaitModeStateFromConfig(rows[i], rowPath));
    if (!named.insert(state.gaitCommand).second) {
      return absl::InvalidArgumentError(
          absl::StrCat(rowPath, ".gait '", state.gaitCommand,
                       "' is named by an earlier row; each gait is one row"));
    }
    states.push_back(std::move(state));
  }
  return states;
}

}  // namespace ocs2::humanoid
