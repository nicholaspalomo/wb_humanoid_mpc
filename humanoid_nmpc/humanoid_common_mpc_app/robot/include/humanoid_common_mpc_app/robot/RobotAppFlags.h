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

#include "absl/flags/declare.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc_app/node/MpcAppFlags.h"
#include "humanoid_common_mpc_app/robot/RobotAppOptions.h"

// The flags of the robot binaries besides those they share with the MPC node (--robot_name, --task_file,
// --reference_file, --urdf_file, --gait_file, --network_config; node/MpcAppFlags.h). Defined by :robot_app_flags, for
// binaries only.
ABSL_DECLARE_FLAG(std::string, mjcf_file);
ABSL_DECLARE_FLAG(std::string, ipc_node);
ABSL_DECLARE_FLAG(std::string, backend);
ABSL_DECLARE_FLAG(int, realtime_priority);
ABSL_DECLARE_FLAG(std::string, realtime_cores);
ABSL_DECLARE_FLAG(std::string, backend_cores);
ABSL_DECLARE_FLAG(std::string, mpc_link);
ABSL_DECLARE_FLAG(bool, headless);
ABSL_DECLARE_FLAG(std::string, config_store_dir);
ABSL_DECLARE_FLAG(std::string, config_seed);

namespace ocs2::humanoid {

/**
 * The robot binary's options from its flags, checked: the files exist, the core lists parse, the realtime priority is
 * 0-99, the retired --mpc_link is not given, --config_seed names a seed policy. `defaultRealtimeCores` and `defaultBackendCores` are what
 * `default` stands for.
 */
absl::StatusOr<RobotAppOptions> robotAppOptionsFromFlags(const std::vector<int>& defaultRealtimeCores,
                                                         const std::vector<int>& defaultBackendCores);

}  // namespace ocs2::humanoid
