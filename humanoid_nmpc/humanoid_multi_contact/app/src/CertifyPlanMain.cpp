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

// certify_plan: certifies a multi-contact plan against its scenario (design
// section 6), prints the report, and writes the certified plan.
//
//   bazel run //humanoid_nmpc/humanoid_multi_contact/app:certify_plan -- \
//       --scenario=humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto
//       \
//       --plan=<plan.binpb or plan.textproto> [--output=<certified.binpb>]
//       [--between_knots=<configurations certified between two knots>]
//
// Exits 0 when every knot passes, 1 when a knot fails, 2 on an error.

#include <string>
#include <vector>

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
#include "humanoid_multi_contact/plan/PlanFile.h"

ABSL_FLAG(std::string,
          scenario,
          "",
          "the scenario textproto (relative to the directory bazel run was "
          "started from, or to the repository)");
ABSL_FLAG(std::string,
          plan,
          "",
          "the plan: a binary .binpb or a textproto MultiContactPlan");
ABSL_FLAG(int,
          between_knots,
          0,
          "how many configurations to certify between every two consecutive "
          "knots, interpolated as the MPC's reference interpolates the plan; "
          "0: the knots only. Not with --output");
ABSL_FLAG(std::string,
          output,
          "",
          "where to write the certified plan (a .binpb, with its summary "
          "beside it); empty: print the report only");

namespace {

constexpr int kExitPassed = 0;
constexpr int kExitFailed = 1;
constexpr int kExitError = 2;

}  // namespace

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  if (absl::GetFlag(FLAGS_scenario).empty() ||
      absl::GetFlag(FLAGS_plan).empty()) {
    LOG(ERROR) << "certify_plan needs --scenario and --plan";
    return kExitError;
  }
  const int betweenKnots = absl::GetFlag(FLAGS_between_knots);
  if (betweenKnots != 0 && !absl::GetFlag(FLAGS_output).empty()) {
    LOG(ERROR) << "certify_plan writes no plan certified between its knots: "
                  "--between_knots and --output exclude each other";
    return kExitError;
  }
  const absl::StatusOr<ocs2::humanoid::multi_contact::config::MultiContactPlan>
      plan = ocs2::humanoid::multi_contact::certifyPlanFile(
          ocs2::humanoid::multi_contact::resolveInputPath(
              absl::GetFlag(FLAGS_scenario)),
          ocs2::humanoid::multi_contact::resolveInputPath(
              absl::GetFlag(FLAGS_plan)),
          betweenKnots);
  if (!plan.ok()) {
    LOG(ERROR) << plan.status();
    return kExitError;
  }
  absl::PrintF("%s",
               ocs2::humanoid::multi_contact::formatReport(*plan->report));
  const std::string output = absl::GetFlag(FLAGS_output);
  if (!output.empty()) {
    const std::string target =
        ocs2::humanoid::multi_contact::resolveOutputPath(output);
    if (const absl::Status status =
            ocs2::humanoid::multi_contact::savePlanFile(*plan, target);
        !status.ok()) {
      LOG(ERROR) << status;
      return kExitError;
    }
    LOG(INFO) << "wrote " << target << " and "
              << ocs2::humanoid::multi_contact::planSummaryPath(target);
  }
  return plan->report->passed ? kExitPassed : kExitFailed;
}
