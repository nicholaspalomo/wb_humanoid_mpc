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

#include "absl/flags/declare.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"

// The flags the centroidal MPC's node and dummy simulator add to the shared
// ones (MpcAppFlags.h): the scenario and the plan they track under
// contact_schedule_source: "multi_contact_plan", and how the plan plays, in
// place of the task file's multi_contact_plan.scenario_file, plan_file and
// playback. Defined in
// CentroidalAppFlags.cpp, so link :centroidal_app_flags into binaries only.
// LINT.IfChange(centroidal_app_flags)
ABSL_DECLARE_FLAG(std::string, multi_contact_scenario);
ABSL_DECLARE_FLAG(std::string, multi_contact_plan);
ABSL_DECLARE_FLAG(std::string, multi_contact_playback);
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalAppFlags.cpp:centroidal_app_flags, //humanoid_nmpc/docs/distributed_runtime/README.md:command_line_flags)
// clang-format on

namespace ocs2::humanoid {

/** The files of --multi_contact_scenario and --multi_contact_plan, and the
 * playback of --multi_contact_playback. */
MultiContactPlanFiles multiContactPlanFilesFromFlags();

}  // namespace ocs2::humanoid
