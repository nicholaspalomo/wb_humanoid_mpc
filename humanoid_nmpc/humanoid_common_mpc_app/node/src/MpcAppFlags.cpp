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

#include "humanoid_common_mpc_app/node/MpcAppFlags.h"

#include "absl/flags/flag.h"

// LINT.IfChange(mpc_app_flags)
ABSL_FLAG(std::string, robot_name, "", "The robot, as its logs name it, e.g. drc_atlas.");
ABSL_FLAG(std::string, task_file, "", "The robot's MPC task file (config/mpc/task.yaml). Required.");
ABSL_FLAG(std::string, reference_file, "", "The robot's reference file (config/command/reference.yaml). Required.");
ABSL_FLAG(std::string, urdf_file, "", "The robot's URDF. Required.");
ABSL_FLAG(std::string, gait_file, "", "The gait file (humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml). Required.");
ABSL_FLAG(std::string,
          network_config,
          "",
          "The network file of the bus (a robot_ipc_proto.NetworkConfig textproto such as config/ipc/network.textproto); "
          "empty: the shipped single-machine network.");
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/node/include/humanoid_common_mpc_app/node/MpcAppFlags.h:mpc_app_flags)

namespace ocs2::humanoid::node {

MpcFiles mpcFilesFromFlags() {
  MpcFiles files;
  files.taskFile = absl::GetFlag(FLAGS_task_file);
  files.referenceFile = absl::GetFlag(FLAGS_reference_file);
  files.urdfFile = absl::GetFlag(FLAGS_urdf_file);
  files.gaitFile = absl::GetFlag(FLAGS_gait_file);
  return files;
}

}  // namespace ocs2::humanoid::node
