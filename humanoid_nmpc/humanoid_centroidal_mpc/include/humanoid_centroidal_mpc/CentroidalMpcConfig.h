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

#include <optional>
#include <string>

#include "absl/status/statusor.h"

#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/**
 * The configuration files a centroidal MPC is built from, as the typed structs of humanoid_nmpc/humanoid_mpc_config:
 * what CentroidalMpcInterface::Create() and its parameter updater read. Passive data; loadCentroidalMpcConfig() reads it
 * from a robot's files.
 */
struct CentroidalMpcConfig {
  // The task file, config/mpc/task.textproto: the model, the formulation, the weights and the solver.
  mpc_config::TaskFile task;
  // The reference file, config/command/reference.textproto: the default posture, the gait schedule's start and the
  // command limits.
  mpc_config::ReferenceFile reference;
  // The contact planner's file, config/mpc/contact_planning.textproto beside the task file. Absent for a robot that ships
  // none: it runs the planner's library defaults (ContactPlanningConfig{}) where its contact_schedule_source names the
  // planner, never the conversion of an empty file.
  std::optional<mpc_config::ContactPlanningFile> contactPlanning;
};

/**
 * The configuration of the robot whose task file is at `taskFile` and whose reference file is at `referenceFile`
 * (loadTaskFile(), loadReferenceFile()), with the contact planner's file beside the task file when there is one
 * (loadContactPlanningFileBeside()).
 *
 * @return The loaders' errors: NotFound naming a file that does not exist, and InvalidArgument naming the file, the line
 *         and the column of a file that does not parse strictly (a file in another format, such as the YAML of the old
 *         task files, among them).
 */
absl::StatusOr<CentroidalMpcConfig> loadCentroidalMpcConfig(const std::string& taskFile, const std::string& referenceFile);

}  // namespace ocs2::humanoid
