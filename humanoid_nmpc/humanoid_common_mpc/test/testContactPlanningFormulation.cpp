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
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

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

/*============================================ configuration validation ====================================*/

TEST(ContactPlanningConfigValidation, AnUnknownPlannerTypeIsRejectedNamingThePlannersThatExist) {
  // The whole reload path - loadContactPlanningConfig, ContactPlanningReferenceManager::setConfig, the parameter
  // updater - reads "validate() did not throw" as "this configuration can be applied", so a name only the factory
  // rejects is caught after the new parameters have already been stored and the background worker started or stopped.
  ContactPlanningConfig config;
  config.planner.type = "lipmiqp2";
  try {
    config.validate();
    FAIL() << "an unknown planner.type must not survive validation";
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("lipmiqp2"), std::string::npos) << "the message must quote what was written: " << what;
    for (const std::string& name : knownPlannerNames()) {
      EXPECT_NE(what.find(name), std::string::npos) << "the message must name " << name << ": " << what;
    }
  }
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig known;
    known.planner.type = name;
    EXPECT_NO_THROW(known.validate()) << name;
  }
  ContactPlanningConfig spelledDifferently;
  spelledDifferently.planner.type = "LIP-MIQP";  // the factory matches names without case or separators
  EXPECT_NO_THROW(spelledDifferently.validate());
}

TEST(ContactPlanningConfigValidation, BigMMustCoverTheLateralFootSeparation) {
  // In double support both contact binaries are one, the +-M terms of a single-support ZMP box cancel and the box
  // relaxes to |e_y'(zmp - p_i)| <= halfWidthY + M, so a big-M below the lateral separation of the feet clips the
  // double-support region instead of switching off with it. The guard against maxStepLength alone did not see this.
  ContactPlanningConfig config;
  config.footSeparation.maxStepLength = 0.30;
  config.footSeparation.maxStepWidth = 0.45;
  config.shared.bigM = 0.35;  // above maxStepLength, below maxStepWidth
  EXPECT_THROW(config.validate(), std::invalid_argument);
  config.shared.bigM = 0.45;  // exactly the widest separation the feet may reach is enough
  EXPECT_NO_THROW(config.validate());

  // Both shipped robots stay well clear of the new bound and must keep loading.
  ContactPlanningConfig atlas;  // drc_atlas contact_planning.yaml
  atlas.shared.bigM = 1.5;
  atlas.footSeparation.maxStepLength = 0.5;
  atlas.footSeparation.minStepWidth = 0.15;
  atlas.footSeparation.maxStepWidth = 0.45;
  EXPECT_NO_THROW(atlas.validate());
  ContactPlanningConfig sa01;  // engineai_sa01 contact_planning.yaml
  sa01.shared.bigM = 0.7;
  sa01.footSeparation.maxStepLength = 0.58;
  sa01.footSeparation.minStepWidth = 0.13;
  sa01.footSeparation.maxStepWidth = 0.32;
  EXPECT_NO_THROW(sa01.validate());
}

TEST(ContactPlanningConfigWarnings, TheBackgroundThreadWarningIsReportedForTheClosedFormPlanner) {
  const ContactPlanningConfig config;  // planner.type hlip and runInBackgroundThread true are both defaults
  const std::vector<std::string> warnings = config.warnings();
  ASSERT_EQ(warnings.size(), 1u) << (warnings.empty() ? std::string() : warnings.front());
  EXPECT_NE(warnings.front().find("runInBackgroundThread"), std::string::npos) << warnings.front();
}

TEST(ContactPlanningConfigWarnings, ASustainedSidestepMustFitInsideTheStepWidthClips) {
  // deadbeatStep plans the period-two orbit at +-hlip.stepWidth + v_y (sspDuration + dspDuration) and then clips the
  // placed foot into [minStepWidth, maxStepWidth]. Only the nominal orbit at a zero lateral command was ever checked
  // against those clips, so a configuration could ask, at full stick, for a step it clips on every occurrence.
  ContactPlanningConfig config;
  config.planner.runInBackgroundThread = false;  // its own warning, covered above
  const std::vector<std::string> defaults = config.warnings();
  EXPECT_TRUE(defaults.empty()) << "the library defaults must be self-consistent, but: " << (defaults.empty() ? "" : defaults.front());

  ContactPlanningConfig narrow = config;
  narrow.hlip.blend.maxCommandedVelocityY = 0.3;  // 0.3 * 0.35 s = 0.105 m of drift against stepWidth - minStepWidth = 0.10
  const std::vector<std::string> narrowWarnings = narrow.warnings();
  ASSERT_EQ(narrowWarnings.size(), 1u);
  EXPECT_NE(narrowWarnings.front().find("hlip.minStepWidth"), std::string::npos) << narrowWarnings.front();
  EXPECT_NE(narrowWarnings.front().find("hlip.blend.maxCommandedVelocityY"), std::string::npos) << narrowWarnings.front();
  EXPECT_NO_THROW(narrow.validate()) << "a warning, not an error: a shipped file must not stop loading over this";

  ContactPlanningConfig wide = config;
  wide.hlip.maxStepWidth = 0.30;  // stepWidth + 0.25 * 0.35 = 0.3375 m does not fit
  const std::vector<std::string> wideWarnings = wide.warnings();
  ASSERT_EQ(wideWarnings.size(), 1u);
  EXPECT_NE(wideWarnings.front().find("hlip.maxStepWidth"), std::string::npos) << wideWarnings.front();

  ContactPlanningConfig miqp = narrow;
  miqp.planner.type = planner::kLipMiqp;
  EXPECT_TRUE(miqp.warnings().empty()) << "the hlip block is not read by the mixed-integer planner";
}

/*============================================ the variable layout =========================================*/

TEST(ContactPlanningLayout, PerFootHeadingIndicesAreAbsentWithoutTheHeadingBlock) {
  // The well-known indices are documented as -1 when their block is absent, and the per-foot accessors used to be
  // plain offset arithmetic: footYaw(1) came out as 0, an in-bounds index that aliases c_x, so a term following the
  // documented contract would silently read or write the centre-of-mass state on a heading-less formulation.
  const Layout none;
  EXPECT_FALSE(none.hasHeading);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    EXPECT_EQ(none.footYaw(foot), -1) << "foot " << foot;
    EXPECT_EQ(none.yawTorque(foot), -1) << "foot " << foot;
    EXPECT_EQ(none.footYawDelta(foot), -1) << "foot " << foot;
  }
  Layout heading;
  heading.hasHeading = true;
  heading.footYaw0 = 8;
  heading.yawTorque0 = 5;
  heading.footYawDelta0 = 7;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    EXPECT_EQ(heading.footYaw(foot), 8 + static_cast<int>(foot)) << "foot " << foot;
    EXPECT_EQ(heading.yawTorque(foot), 5 + static_cast<int>(foot)) << "foot " << foot;
    EXPECT_EQ(heading.footYawDelta(foot), 7 + static_cast<int>(foot)) << "foot " << foot;
  }
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

TEST(ContactPlanningConfigFile, APartialSlackBlockInheritsTheSharedDefaultForTheHalfItOmits) {
  // loadPtreeValue leaves its destination untouched when a key is absent, and the term's penalty used to be seeded
  // from a default-constructed SlackPenalty (1e4 / 100) rather than from shared.slack_penalty as the file wrote it.
  // A block that overrides one of the two numbers therefore ran on a hard-coded value nobody asked for.
  const std::string file = writeTemp("partial_slack_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  shared:\n"
                                     "    slack_penalty:\n"
                                     "      quadratic: 500.0\n"
                                     "      linear: 7.0\n"
                                     "  zmp_support_region:\n"
                                     "    slack:\n"
                                     "      quadratic: 42.0\n"
                                     "  reachability:\n"
                                     "    slack:\n"
                                     "      linear: 3.0\n");
  const ContactPlanningConfig loaded = loadContactPlanningConfig(file, "contact_planning.", false);
  std::remove(file.c_str());
  ASSERT_TRUE(loaded.zmpSupportRegion.slack.has_value());
  EXPECT_NEAR(loaded.zmpSupportRegion.slack->quadratic, 42.0, kTol) << "the key the file wrote";
  EXPECT_NEAR(loaded.zmpSupportRegion.slack->linear, 7.0, kTol) << "the key it omitted comes from shared.slack_penalty";
  ASSERT_TRUE(loaded.reachability.slack.has_value());
  EXPECT_NEAR(loaded.reachability.slack->quadratic, 500.0, kTol) << "the key it omitted comes from shared.slack_penalty";
  EXPECT_NEAR(loaded.reachability.slack->linear, 3.0, kTol) << "the key the file wrote";
  EXPECT_FALSE(loaded.footSeparation.slack.has_value()) << "no slack block at all still means the shared default";
}

TEST(ContactPlanningConfigFile, AnUnknownPlannerTypeStopsTheFileFromLoading) {
  const std::string file = writeTemp("unknown_planner_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  planner:\n"
                                     "    type: hilp\n"
                                     "    runInBackgroundThread: true\n");
  EXPECT_THROW(loadContactPlanningConfig(file, "contact_planning.", false), std::invalid_argument)
      << "a typo must fail the reload atomically, leaving the running planner and its configuration in force";
  std::remove(file.c_str());
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
