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

// plan_multi_contact: plans a multi-contact scenario offline (design section
// 5), prints the certifier's report and a summary, and writes the plan.
//
//   bazel run //humanoid_nmpc/humanoid_multi_contact/app:plan_multi_contact --
//   \
//       --scenario=humanoid_nmpc/humanoid_multi_contact/scenarios/config/stairs.textproto
//       \
//       --output=<plan.binpb> [--planner=stance_graph]
//
// Exits 0 with a certified plan, 1 when no plan is found or the certifier
// refuses it, 2 on an error.

#include <string>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

#include "humanoid_multi_contact/app/CertifyPlan.h"
#include "humanoid_multi_contact/app/CommandLinePaths.h"
#include "humanoid_multi_contact/app/PlanMultiContact.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/planner/MultiContactPlannerRegistry.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

ABSL_FLAG(std::string,
          scenario,
          "",
          "the scenario textproto (relative to the directory bazel run was "
          "started from, or to the repository)");
ABSL_FLAG(std::string,
          output,
          "",
          "where to write the plan (a .binpb, with its summary beside it); "
          "empty: print the report only");
ABSL_FLAG(std::string,
          planner,
          "",
          "the planner by registry name; empty: the scenario's own");

namespace {

constexpr int kExitPlanned = 0;
constexpr int kExitNoPlan = 1;
constexpr int kExitError = 2;

}  // namespace

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  namespace multi_contact = ::ocs2::humanoid::multi_contact;
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  if (absl::GetFlag(FLAGS_scenario).empty()) {
    LOG(ERROR) << "plan_multi_contact needs --scenario";
    return kExitError;
  }
  const std::string scenarioPath =
      multi_contact::resolveInputPath(absl::GetFlag(FLAGS_scenario));
  const multi_contact::MultiContactPlannerRegistry registry;
  const absl::StatusOr<multi_contact::config::MultiContactPlan> plan =
      multi_contact::planScenarioFile(scenarioPath,
                                      absl::GetFlag(FLAGS_planner), registry);
  if (!plan.ok()) {
    LOG(ERROR) << plan.status();
    return plan.status().code() == absl::StatusCode::kNotFound ||
                   plan.status().code() == absl::StatusCode::kFailedPrecondition
               ? kExitNoPlan
               : kExitError;
  }
  absl::PrintF("%s", multi_contact::formatReport(*plan->report));
  const absl::StatusOr<multi_contact::LoadedScenario> loaded =
      multi_contact::loadScenario(scenarioPath);
  if (!loaded.ok()) {
    LOG(ERROR) << loaded.status();
    return kExitError;
  }
  const absl::StatusOr<multi_contact::PlanSummary> summary =
      multi_contact::summarizePlan(*plan, loaded->contactSet);
  if (!summary.ok()) {
    LOG(ERROR) << summary.status();
    return kExitError;
  }
  absl::PrintF("%s: %s\n", plan->scenario,
               multi_contact::formatPlanSummary(*summary));
  const std::string output = absl::GetFlag(FLAGS_output);
  if (!output.empty()) {
    const std::string target = multi_contact::resolveOutputPath(output);
    if (const absl::Status status = multi_contact::savePlanFile(*plan, target);
        !status.ok()) {
      LOG(ERROR) << status;
      return kExitError;
    }
    LOG(INFO) << "wrote " << target << " and "
              << multi_contact::planSummaryPath(target);
  }
  return plan->report->passed ? kExitPlanned : kExitNoPlan;
}
