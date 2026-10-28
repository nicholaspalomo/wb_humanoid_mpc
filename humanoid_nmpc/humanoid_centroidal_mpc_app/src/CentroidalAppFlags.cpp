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

#include "humanoid_centroidal_mpc_app/CentroidalAppFlags.h"

#include <string>

#include "absl/flags/flag.h"

// LINT.IfChange(centroidal_app_flags)
ABSL_FLAG(std::string,
          multi_contact_scenario,
          "",
          "The multi-contact scenario whose plan the MPC tracks under "
          "contact_schedule_source: \"multi_contact_plan\" "
          "(humanoid_nmpc/humanoid_multi_contact/scenarios/config/"
          "<scenario>.textproto); empty: the task file's "
          "multi_contact_plan.scenario_file.");
ABSL_FLAG(std::string,
          multi_contact_plan,
          "",
          "The plan the MPC tracks (a .binpb or a textproto made for the "
          "scenario); empty: the task file's multi_contact_plan.plan_file.");
ABSL_FLAG(std::string,
          multi_contact_playback,
          "",
          "How the tracked plan plays: \"once\", or \"forward_and_reverse\" "
          "(forward, a dwell, backward to its start, a dwell, forever); "
          "empty: the task file's multi_contact_plan.playback.");
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/include/humanoid_centroidal_mpc_app/CentroidalAppFlags.h:centroidal_app_flags, //humanoid_nmpc/docs/distributed_runtime/README.md:command_line_flags)
// clang-format on

namespace ocs2::humanoid {

MultiContactPlanFiles multiContactPlanFilesFromFlags() {
  return MultiContactPlanFiles{
      .scenarioFile = absl::GetFlag(FLAGS_multi_contact_scenario),
      .planFile = absl::GetFlag(FLAGS_multi_contact_plan),
      .playback = absl::GetFlag(FLAGS_multi_contact_playback)};
}

}  // namespace ocs2::humanoid
