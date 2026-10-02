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
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/node/MpcFiles.h"

namespace ocs2::humanoid {

/**
 * The retired --mpc_link flag. The robot process reaches its MPC over the bus only (RemoteMpcLink <-> MpcServer, the MPC
 * node of its formulation), in simulation as on the robot; the in-process MPC (`--mpc_link=in_process`) was removed
 * from the robot binaries, and InProcessMpcLink is for unit tests. OK when the flag is not given; FailedPrecondition,
 * naming the replacement, for any value, so that a stale command line fails at start-up instead of being ignored.
 */
absl::Status checkRetiredMpcLinkFlag(absl::string_view value);

/**
 * The cores a --realtime_cores or --backend_cores value names: `default` is `defaultCores` (ThreadAffinity.h's
 * allocation for this machine, as the ROS sims pinned), `none` (or empty) pins nothing, otherwise a comma-separated list
 * of CPU numbers ("4,5"). InvalidArgument for anything else.
 */
absl::StatusOr<std::vector<int>> parseCoreList(absl::string_view value, const std::vector<int>& defaultCores);

/** Everything the robot binaries take from their command line (RobotAppFlags.h). */
struct RobotAppOptions {
  std::string robotName;
  /** --task_file, --reference_file, --urdf_file (--gait_file is the MPC node's). */
  node::MpcFiles files;
  std::string mjcfFile;
  std::string networkConfig;
  std::string ipcNode;
  std::string backend;
  int realtimePriority = 0;
  std::vector<int> realtimeCores;
  std::vector<int> backendCores;
  bool headless = false;
};

}  // namespace ocs2::humanoid
