/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-12;

std::string writeTemp(const std::string& name, const std::string& content) {
  const std::string file = testing::TempDir() + "/" + name;
  std::ofstream out(file);
  out << content;
  return file;
}

}  // namespace

/*============================================ names ======================================================*/

TEST(ContactPlanningFormulation, NamesAreMatchedLikeTheTaskFileLists) {
  EXPECT_TRUE(sameTermName("velocity_tracking", "velocityTracking"));
  EXPECT_TRUE(sameTermName("Velocity Tracking", "velocity-tracking"));
  EXPECT_FALSE(sameTermName("velocity_tracking", "heading_tracking"));
  EXPECT_EQ(canonicalTermName(TermKind::COST, "VelocityTracking"), term::kVelocityTracking);
  EXPECT_EQ(canonicalTermName(TermKind::COST, "no_such_cost"), "");
  EXPECT_EQ(canonicalTermName(TermKind::LOGIC_RULE, "noFlight"), term::kNoFlight);
  EXPECT_EQ(canonicalTermName(TermKind::HARD_CONSTRAINT, "noFlight"), term::kNoFlight) << "the same name in two lists";
  EXPECT_EQ(canonicalTermName(TermKind::COST, "noFlight"), "") << "but not in a list it does not belong to";
}

TEST(ContactPlanningFormulation, DefaultIsThePointMassPlannerWithoutHeuristics) {
  const ContactPlanningFormulation f;
  EXPECT_NO_THROW(f.validate());
  EXPECT_FALSE(f.usesHeadingModel());
  EXPECT_TRUE(f.hasDynamics(term::kLipCom));
  EXPECT_TRUE(f.hasDynamics(term::kFootholdIntegrator));
  EXPECT_TRUE(f.hasCost(term::kVelocityTracking));
  EXPECT_TRUE(f.hasCost(term::kRegularization)) << "the regularisation is a visible term";
  EXPECT_FALSE(f.hasCost(term::kHeadingTracking));
  EXPECT_TRUE(f.hasLogicRule(term::kAlternatingFeet));
  EXPECT_TRUE(f.hasSearchStage(term::kDiving));
  EXPECT_TRUE(f.execution.empty()) << "every execution rule is opt-in";
  EXPECT_FALSE(f.needsPredictedTrajectory());
}

TEST(ContactPlanningFormulation, HeadingModelIsAddedAndRemovedAsAWhole) {
  ContactPlanningFormulation f;
  f.setHeadingModel(true);
  EXPECT_NO_THROW(f.validate());
  EXPECT_TRUE(f.usesHeadingModel());
  for (const char* name : {term::kHeadingRateTracking, term::kHeadingTracking, term::kFootYawTracking, term::kYawTorqueRegularization,
                           term::kFootYawRegularization}) {
    EXPECT_TRUE(f.hasCost(name)) << name;
  }
  EXPECT_TRUE(f.hasSoftConstraint(term::kHipYawRange));
  EXPECT_TRUE(f.hasHardConstraint(term::kYawTorqueBudget));
  EXPECT_TRUE(f.hasHardConstraint(term::kFootYawPinnedInContact));
  EXPECT_TRUE(f.hasSearchStage(term::kHeadingRelinearisation));
  EXPECT_TRUE(f.hasExecutionRule(term::kPlannedHeadingOverride));
  // The heading costs sit before zmp_regularization: the accumulation order of the previous planner.
  size_t zmp = 0, footYawReg = 0;
  for (size_t i = 0; i < f.costs.size(); ++i) {
    if (sameTermName(f.costs[i], term::kZmpRegularization)) zmp = i;
    if (sameTermName(f.costs[i], term::kFootYawRegularization)) footYawReg = i;
  }
  EXPECT_LT(footYawReg, zmp);
  f.setHeadingModel(false);
  EXPECT_EQ(f, ContactPlanningFormulation{}) << "removing the heading model restores the default lists";
}

TEST(ContactPlanningFormulation, ValidationRejectsUnknownDuplicateAndUnsupportedTerms) {
  ContactPlanningFormulation f;
  f.costs.push_back("no_such_cost");
  EXPECT_THROW(f.validate(), std::invalid_argument);
  f = ContactPlanningFormulation{};
  f.costs.push_back(term::kVelocityTracking);
  EXPECT_THROW(f.validate(), std::invalid_argument) << "a duplicate";
  f = ContactPlanningFormulation{};
  f.costs.push_back(term::kHeadingTracking);
  EXPECT_THROW(f.validate(), std::invalid_argument) << "a heading cost without the heading block";
  f = ContactPlanningFormulation{};
  f.dynamics = {term::kFootholdIntegrator, term::kLipCom};
  EXPECT_THROW(f.validate(), std::invalid_argument) << "the LIP block must come first";
  f = ContactPlanningFormulation{};
  f.execution = {term::kEnergyCadenceModulation, term::kPhaseResetting};
  EXPECT_THROW(f.validate(), std::invalid_argument) << "cadence before phase resetting cannot honour an early touch-down";
  f.execution = {term::kPhaseResetting, term::kEnergyCadenceModulation};
  EXPECT_NO_THROW(f.validate());
  EXPECT_TRUE(f.needsPredictedTrajectory());
}

TEST(ContactPlanningFormulation, SummaryListsEveryCollection) {
  const std::string summary = ContactPlanningFormulation{}.summary();
  EXPECT_NE(summary.find("dynamics (2): lip_com, foothold_integrator"), std::string::npos) << summary;
  EXPECT_NE(summary.find("execution (0): (none)"), std::string::npos) << summary;
}

/*============================================ the structured file =========================================*/

TEST(ContactPlanningConfigFile, StructuredLayoutLoadsListsAndTermBlocks) {
  const std::string file = writeTemp("structured_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  planner:\n"
                                     "    dt: 0.05\n"
                                     "    numNodes: 20\n"
                                     "    commitTime: 0.2\n"
                                     "    runInBackgroundThread: false\n"
                                     "  shared:\n"
                                     "    comHeight: 0.9\n"
                                     "    bigM: 2.0\n"
                                     "    slack_penalty:\n"
                                     "      quadratic: 500.0\n"
                                     "      linear: 7.0\n"
                                     "    gait_limits:\n"
                                     "      minSwingDuration: 0.25\n"
                                     "      maxSwingDuration: 0.45\n"
                                     "  costs:\n"
                                     "    - regularization\n"
                                     "    - velocity_tracking\n"
                                     "    - zmp_regularization\n"
                                     "  execution:\n"
                                     "    - phase_resetting\n"
                                     "    - dcm_step_adjustment\n"
                                     "  velocity_tracking:\n"
                                     "    weight: 33.0\n"
                                     "  zmp_support_region:\n"
                                     "    halfWidthX: 0.1\n"
                                     "    slack:\n"
                                     "      quadratic: 42.0\n"
                                     "      linear: 1.0\n"
                                     "  dcm_step_adjustment:\n"
                                     "    gain: 0.9\n"
                                     "    maxOffset: 0.02\n");
  const ContactPlanningConfig loaded = loadContactPlanningConfig(file, "contact_planning.", false);
  std::remove(file.c_str());
  EXPECT_NEAR(loaded.planner.dt, 0.05, kTol);
  EXPECT_EQ(loaded.planner.numNodes, 20);
  EXPECT_NEAR(loaded.planner.commitTime, 0.2, kTol);
  EXPECT_FALSE(loaded.planner.runInBackgroundThread);
  EXPECT_NEAR(loaded.shared.comHeight, 0.9, kTol);
  EXPECT_NEAR(loaded.shared.bigM, 2.0, kTol);
  EXPECT_NEAR(loaded.shared.slackPenalty.quadratic, 500.0, kTol);
  EXPECT_NEAR(loaded.shared.slackPenalty.linear, 7.0, kTol);
  EXPECT_NEAR(loaded.shared.gaitLimits.minSwingDuration, 0.25, kTol);
  EXPECT_NEAR(loaded.shared.gaitLimits.maxSwingDuration, 0.45, kTol);
  EXPECT_NEAR(loaded.shared.gaitLimits.minContactDuration, ContactPlanningConfig{}.shared.gaitLimits.minContactDuration, kTol)
      << "missing keys keep their defaults";
  // A listed collection replaces the default list, an unlisted one keeps it.
  ASSERT_EQ(loaded.formulation.costs.size(), 3u);
  EXPECT_TRUE(loaded.formulation.hasCost(term::kVelocityTracking));
  EXPECT_FALSE(loaded.formulation.hasCost(term::kStepWidth));
  EXPECT_EQ(loaded.formulation.dynamics, ContactPlanningFormulation{}.dynamics);
  ASSERT_EQ(loaded.formulation.execution.size(), 2u);
  EXPECT_TRUE(loaded.formulation.hasExecutionRule(term::kPhaseResetting));
  EXPECT_TRUE(loaded.formulation.hasExecutionRule(term::kDcmStepAdjustment));
  EXPECT_NEAR(loaded.velocityTracking.weight, 33.0, kTol);
  EXPECT_NEAR(loaded.zmpSupportRegion.halfWidthX, 0.1, kTol);
  ASSERT_TRUE(loaded.zmpSupportRegion.slack.has_value());
  EXPECT_NEAR(loaded.zmpSupportRegion.slack->quadratic, 42.0, kTol);
  EXPECT_FALSE(loaded.reachability.slack.has_value()) << "no slack block: the shared default applies";
  EXPECT_NEAR(loaded.dcmStepAdjustment.gain, 0.9, kTol);
  EXPECT_NEAR(loaded.dcmStepAdjustment.maxOffset, 0.02, kTol);
}

TEST(ContactPlanningConfigFile, StructuredLayoutRejectsAnUnknownTermInAList) {
  const std::string file = writeTemp("bad_list_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  costs:\n"
                                     "    - regularization\n"
                                     "    - gravity_compensation\n");
  EXPECT_THROW(loadContactPlanningConfig(file, "contact_planning.", false), std::invalid_argument);
  std::remove(file.c_str());
}

/*============================================ the flat file of the previous planner =======================*/

TEST(ContactPlanningConfigFile, FlatLayoutIsRejectedWithAMigrationHint) {
  const std::string file = writeTemp("flat_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  dt: 0.1\n"
                                     "  numNodes: 14\n"
                                     "  velocityTrackingWeight: 12.0\n"
                                     "  useAcomDynamics: true\n"
                                     "  enablePhaseResetting: true\n");
  try {
    loadContactPlanningConfig(file, "contact_planning.", false, /*validate=*/false);
    FAIL() << "the flat layout of the previous planner is no longer read";
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("flat layout"), std::string::npos) << what;
    EXPECT_NE(what.find("phase_resetting"), std::string::npos) << "the message says how to migrate: " << what;
  }
  std::remove(file.c_str());
}

TEST(ContactPlanningConfigFile, MixingTheTwoLayoutsIsAnError) {
  const std::string file = writeTemp("mixed_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  planner:\n"
                                     "    dt: 0.1\n"
                                     "  velocityTrackingWeight: 12.0\n");
  EXPECT_THROW(loadContactPlanningConfig(file, "contact_planning.", false), std::invalid_argument);
  std::remove(file.c_str());
}

TEST(ContactPlanningConfigFile, AMissingBlockGivesTheDefaults) {
  const std::string file = writeTemp("empty_contact_planning.yaml", "useContactPlanning: true\n");
  const ContactPlanningConfig loaded = loadContactPlanningConfig(file, "contact_planning.", false);
  std::remove(file.c_str());
  EXPECT_NEAR(loaded.planner.dt, ContactPlanningConfig{}.planner.dt, kTol);
  EXPECT_EQ(loaded.formulation, ContactPlanningFormulation{});
}

}  // namespace ocs2::humanoid
