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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/MultiContactTestFiles.h"
#include "support/StairStepPlan.h"

/*
 * The hand-written feet-only stair-step plan of the tracking tests
 * (support/StairStepPlan.h) on its test scenario: it certifies at every knot,
 * stands on the step at its end, and is the plan of the scenario's contact set
 * and terrain (its hashes are theirs, so a plan file of it loads).
 */
namespace ocs2::humanoid {
namespace {

constexpr char kStairStepScenario[] =
    "humanoid_nmpc/humanoid_centroidal_mpc/test/testdata/multi_contact/"
    "stair_step_scenario.textproto";
/** [m] The height of the scenario's step. */
constexpr scalar_t kStepHeight = 0.10;

TEST(StairStepPlan, certifiesAndEndsOnTheStep) {
  const CentroidalRobotFiles files = atlasMultiContactFiles();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CentroidalMpcInterface> models,
      CentroidalMpcInterface::CreateControllerModels(
          files.taskFile, files.urdfFile, files.referenceFile));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::LoadedScenario scenario,
      multi_contact::loadScenario(kStairStepScenario));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::config::MultiContactPlan plan,
      makeStairStepPlan(models->getPinocchioInterface(),
                        models->getMpcRobotModel(), scenario));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<multi_contact::StabilityCertifier> certifier,
      multi_contact::makeCertifier(scenario));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::config::CertifierReport report,
      certifier->certify(plan));
  std::vector<std::string> failures;
  for (const multi_contact::config::CertifierReport::Knot& knot :
       report.knots) {
    if (!knot.passed) {
      failures.push_back(absl::StrCat("t = ", knot.time, ": ",
                                      absl::StrJoin(knot.failures, "; ")));
    }
  }
  EXPECT_TRUE(report.passed) << absl::StrJoin(failures, "\n");
  // Both feet on the step at the end.
  ASSERT_FALSE(plan.knots.empty());
  EXPECT_EQ(plan.knots.back().active_contacts, 0b11U);
  for (const multi_contact::config::MultiContactPlan::ContactPhase& phase :
       plan.phases) {
    if (phase.primitive != "step_1") continue;
    EXPECT_NEAR(phase.target_pad_pose(2), kStepHeight, 1.0e-9) << phase.contact;
  }
  // A plan file of it loads against the scenario's contact set and terrain.
  const std::string path =
      (std::filesystem::path(::testing::TempDir()) / "stair_step.binpb")
          .string();
  HUMANOID_ASSERT_OK(multi_contact::savePlanFile(plan, path));
  EXPECT_TRUE(multi_contact::loadPlanFile(path, scenario.contactSetConfig,
                                          scenario.scenario.terrain_scene)
                  .ok());
}

}  // namespace
}  // namespace ocs2::humanoid
