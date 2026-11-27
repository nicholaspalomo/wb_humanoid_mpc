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

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"

#include "legged_mpc_config/gait_file.nproto.h"

namespace ocs2::legged {

/**
 * One row of the gait switching table of ProceduralMpcMotionManager: a gait of
 * the gait file and the thresholds of the walking velocity command and of the
 * stride-mean motion at which the manager switches to the row after (faster)
 * or the row before (slower). The fields mirror GaitSwitchingTableConfig.Row
 * (gait_switching_table_config.proto), which says what each threshold means.
 * Passive data.
 */
struct GaitModeStateConfig {
  std::string gaitCommand = "stance";
  // [m/s]
  scalar_t cmdMinLinvelBaseRtWorldEwrtHeadingXy = 0.0;
  // [m/s]
  scalar_t cmdMaxLinvelBaseRtWorldEwrtHeadingXy = 0.0;
  // [rad/s]
  scalar_t cmdMinAngvelBaseRtWorldEwrtWorldZ = 0.0;
  // [rad/s]
  scalar_t cmdMaxAngvelBaseRtWorldEwrtWorldZ = 0.0;
  // [m/s]
  scalar_t maxErrLinvelBaseRtWorldEwrtHeadingXy = 0.0;
  // [rad/s]
  scalar_t maxErrAngvelBaseRtWorldEwrtWorldZ = 0.0;
};

/**
 * Returns the table a gait file without gait_switching gets: the biped gaits of
 * legged_common_mpc/config/command/gait.textproto, slowest first (stance,
 * slow_walk, walk, slower_trot, slow_trot, trot, run). Allocates.
 */
std::vector<GaitModeStateConfig> defaultGaitModeStates();

/**
 * Returns the gait switching table of `gaitFile`: defaultGaitModeStates() when
 * its gait_switching block is absent, its rows in order otherwise.
 *
 * @return InvalidArgument naming gait_switching.rows[i] (and the field) for an
 * empty table, a first row whose gait is not "stance" (the gait the robot
 * starts in, which ProceduralMpcMotionManager::reset() takes row 0 to be), a
 * row without a gait name, a gait two rows name, a threshold
 * that is absent or not finite, and a cmd_min_* bound above its cmd_max_*
 * bound. Whether the gait file has each gait is checked by
 * ProceduralMpcMotionManager::checkEveryGaitIsLoaded().
 */
absl::StatusOr<std::vector<GaitModeStateConfig>> gaitModeStatesFromConfig(
    const mpc_config::GaitFile& gaitFile);

}  // namespace ocs2::legged
