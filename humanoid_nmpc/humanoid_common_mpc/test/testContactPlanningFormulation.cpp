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

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kTol = 1e-12;

std::string writeTemp(const std::string& name, const std::string& content) {
  const std::string file = absl::StrCat(testing::TempDir(), "/", name);
  std::ofstream out(file);
  out << content;
  return file;
}

/** Whether any of `warnings` mentions `text`. */
bool anyMentions(const std::vector<std::string>& warnings, const std::string& text) {
  for (const std::string& warning : warnings) {
    if (absl::StrContains(warning, text)) return true;
  }
  return false;
}

std::string joinWarnings(const std::vector<std::string>& warnings) {
  std::string out;
  for (const std::string& warning : warnings) absl::StrAppend(&out, "\n  - ", warning);
  return out.empty() ? std::string(" (none)") : out;
}

/**
 * The pendulum a model-free configuration plans on. The library default of shared.comHeight is 0, "the model's", which
 * ContactPlanningModelParameters::applyTo fills in before validation; a configuration with no model to derive it from -
 * every configuration in this file - has to set a height itself. 0.85 m is the height the DRC Atlas was once hand-set
 * to, the pendulum the library's cadence defaults were validated on (ContactPlanningConfig.h, HlipParameters).
 */
constexpr scalar_t kModelFreeComHeight = 0.85;

/** The library defaults on the model-free pendulum above. */
ContactPlanningConfig modelFreeConfig() {
  ContactPlanningConfig config;
  config.shared.comHeight = kModelFreeComHeight;
  return config;
}

/**
 * The H-LIP planner configured the way its README says it has to be: the library defaults with planner.type hlip, the
 * one execution rule the gait cannot walk without, and planning in the pre-solve hook.
 */
ContactPlanningConfig makeWorkingHlipConfig() {
  ContactPlanningConfig config = modelFreeConfig();
  config.planner.type = planner::kHlip;
  config.planner.runInBackgroundThread = false;
  config.formulation.execution = {term::kPlannedComOverride};
  return config;
}

/**
 * The term `name` of `kind` built through the factory, or nothing for the rules the core cannot build (the two
 * planned_*_override rules need the reference manager's robot model; testHlipPlanningIntegration covers them).
 */
std::unique_ptr<ContactPlanningTerm> makeTerm(TermKind kind, const std::string& name) {
  switch (kind) {
    case TermKind::MODEL_BLOCK:
      return ContactPlanningTermFactory::makeModelBlock(name);
    case TermKind::COST:
      return ContactPlanningTermFactory::makeCost(name);
    case TermKind::SOFT_CONSTRAINT:
      return ContactPlanningTermFactory::makeSoftConstraint(name);
    case TermKind::HARD_CONSTRAINT:
      return ContactPlanningTermFactory::makeHardConstraint(name);
    case TermKind::LOGIC_RULE:
      return ContactPlanningTermFactory::makeLogicRule(name);
    case TermKind::ASSIGNMENT_COST:
      return ContactPlanningTermFactory::makeAssignmentCost(name);
    case TermKind::SEARCH_STAGE:
      return ContactPlanningTermFactory::makeSearchStage(name);
    case TermKind::EXECUTION_RULE:
      if (name == term::kPlannedHeadingOverride || name == term::kPlannedComOverride) return nullptr;
      return ContactPlanningTermFactory::makeExecutionRule(name);
  }
  return nullptr;
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
  EXPECT_TRUE(f.validateStatus().ok()) << f.validateStatus().message();
  EXPECT_FALSE(f.usesHeadingModel());
  EXPECT_TRUE(f.hasDynamics(term::kLipCom));
  EXPECT_TRUE(f.hasDynamics(term::kFootholdIntegrator));
  EXPECT_TRUE(f.hasCost(term::kVelocityTracking));
  EXPECT_TRUE(f.hasCost(term::kRegularization)) << "the regularization is a visible term";
  EXPECT_FALSE(f.hasCost(term::kHeadingTracking));
  EXPECT_TRUE(f.hasLogicRule(term::kAlternatingFeet));
  EXPECT_TRUE(f.hasSearchStage(term::kDiving));
  EXPECT_TRUE(f.execution.empty()) << "every execution rule is opt-in";
  EXPECT_FALSE(f.needsPredictedTrajectory());
}

TEST(ContactPlanningFormulation, HeadingModelIsAddedAndRemovedAsAWhole) {
  ContactPlanningFormulation f;
  f.setHeadingModel(true);
  EXPECT_TRUE(f.validateStatus().ok()) << f.validateStatus().message();
  EXPECT_TRUE(f.usesHeadingModel());
  for (const char* name : {term::kHeadingRateTracking, term::kHeadingTracking, term::kFootYawTracking, term::kYawTorqueRegularization,
                           term::kFootYawRegularization}) {
    EXPECT_TRUE(f.hasCost(name)) << name;
  }
  EXPECT_TRUE(f.hasSoftConstraint(term::kHipYawRange));
  EXPECT_TRUE(f.hasHardConstraint(term::kYawTorqueBudget));
  EXPECT_TRUE(f.hasHardConstraint(term::kFootYawPinnedInContact));
  EXPECT_TRUE(f.hasSearchStage(term::kHeadingRelinearization));
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

/**
 * "Which terms belong to the heading model" used to be written down three times - requiredBlockOf()'s set, the
 * add/remove lists of setHeadingModel() and each term's requiredBlocks() - with nothing keeping them in step. The first
 * two now read requiredModelBlock(); this holds the third to it, by building every term the core can build.
 */
TEST(ContactPlanningFormulation, TheHeadingModelTermsAreTheTermsThatRequireItsBlock) {
  size_t headingTerms = 0;
  size_t otherTerms = 0;
  for (const TermKind kind : allTermKinds()) {
    for (const std::string& name : knownTermNames(kind)) {
      const std::unique_ptr<ContactPlanningTerm> built = makeTerm(kind, name);
      if (built == nullptr) continue;
      const std::vector<std::string> fromTerm = built->requiredBlocks();
      const std::string fromRegistry = requiredModelBlock(kind, name);
      if (fromRegistry.empty()) {
        EXPECT_TRUE(fromTerm.empty()) << termKindName(kind) << " term " << name << " requires a block the registry does not know of";
        ++otherTerms;
      } else {
        ASSERT_EQ(fromTerm.size(), 1u) << termKindName(kind) << " term " << name;
        EXPECT_EQ(fromTerm.front(), fromRegistry) << termKindName(kind) << " term " << name;
        ++headingTerms;
      }
    }
  }
  // The comparison has both answers to disagree about: heading terms in four lists, and plain terms in every list.
  EXPECT_GE(headingTerms, 9u);
  EXPECT_GE(otherTerms, 20u);
}

TEST(ContactPlanningFormulation, SetHeadingModelAddsAndRemovesExactlyTheTermsThatRequireTheBlock) {
  ContactPlanningFormulation on;
  on.setHeadingModel(true);
  ContactPlanningFormulation off = on;
  off.setHeadingModel(false);
  size_t checked = 0;
  for (const TermKind kind : allTermKinds()) {
    if (kind == TermKind::MODEL_BLOCK) continue;
    for (const std::string& name : knownTermNames(kind)) {
      if (requiredModelBlock(kind, name).empty()) continue;
      EXPECT_TRUE(ContactPlanningFormulation::listed(on.list(kind), name)) << termKindName(kind) << " lacks " << name;
      EXPECT_FALSE(ContactPlanningFormulation::listed(off.list(kind), name)) << termKindName(kind) << " keeps " << name;
      ++checked;
    }
  }
  EXPECT_GE(checked, 9u);
}

TEST(ContactPlanningFormulation, ValidationRejectsUnknownDuplicateAndUnsupportedTermsNamingTheList) {
  ContactPlanningFormulation f;
  f.costs.push_back("no_such_cost");
  absl::Status status = f.validateStatus();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(status.message(), "costs")) << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "no_such_cost")) << status.message();

  f = ContactPlanningFormulation{};
  f.costs.push_back(term::kVelocityTracking);
  status = f.validateStatus();
  EXPECT_FALSE(status.ok()) << "a duplicate";
  EXPECT_TRUE(absl::StrContains(status.message(), "costs")) << status.message();

  f = ContactPlanningFormulation{};
  f.softConstraints.push_back(term::kHipYawRange);
  status = f.validateStatus();
  EXPECT_FALSE(status.ok()) << "a heading constraint without the heading block";
  EXPECT_TRUE(absl::StrContains(status.message(), "soft_constraints")) << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "dynamics")) << status.message();

  f = ContactPlanningFormulation{};
  f.dynamics = {term::kFootholdIntegrator, term::kLipCom};
  EXPECT_FALSE(f.validateStatus().ok()) << "the LIP block must come first";

  f = ContactPlanningFormulation{};
  f.execution = {term::kEnergyCadenceModulation, term::kPhaseResetting};
  status = f.validateStatus();
  EXPECT_FALSE(status.ok()) << "cadence before phase resetting cannot honor an early touch-down";
  EXPECT_TRUE(absl::StrContains(status.message(), "execution")) << status.message();
  f.execution = {term::kPhaseResetting, term::kEnergyCadenceModulation};
  EXPECT_TRUE(f.validateStatus().ok());
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
  // updater - reads "validation passed" as "this configuration can be applied", so a name only the factory rejects is
  // caught after the new parameters have already been stored and the background worker started or stopped.
  ContactPlanningConfig config = modelFreeConfig();
  config.planner.type = "lipmiqp2";
  const absl::Status status = config.validateStatus();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(status.message(), "planner.type")) << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "lipmiqp2")) << "the message must quote what was written: " << status.message();
  for (const std::string& name : knownPlannerNames()) {
    EXPECT_TRUE(absl::StrContains(status.message(), name)) << "the message must name " << name << ": " << status.message();
  }
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig known = modelFreeConfig();
    known.planner.type = name;
    EXPECT_TRUE(known.validateStatus().ok()) << name;
  }
  ContactPlanningConfig spelledDifferently = modelFreeConfig();
  spelledDifferently.planner.type = "LIP-MIQP";  // the factory matches names without case or separators
  EXPECT_TRUE(spelledDifferently.validateStatus().ok());
}

/**
 * Every rejection names the one key the operator has to change. Several used to share one message across up to six
 * keys ("cost weights must be non-negative", "invalid solver limits", "heading model weights must be >= 0", ...), so a
 * reload rejected over foothold_regularization.weight left the operator to guess which of six weights was wrong.
 */
TEST(ContactPlanningConfigValidation, EveryRejectionNamesTheKeyToChange) {
  struct Case {
    std::string key;
    std::function<void(ContactPlanningConfig&)> breakIt;
  };
  const std::vector<Case> cases{
      {"planner.dt", [](ContactPlanningConfig& c) { c.planner.dt = 0.0; }},
      {"planner.numNodes", [](ContactPlanningConfig& c) { c.planner.numNodes = 1; }},
      {"planner.commitTime", [](ContactPlanningConfig& c) { c.planner.commitTime = -0.1; }},
      {"planner.maxCommitExtension", [](ContactPlanningConfig& c) { c.planner.maxCommitExtension = 0.01; }},
      {"planner.maxBranchAndBoundNodes", [](ContactPlanningConfig& c) { c.planner.maxBranchAndBoundNodes = 0; }},
      {"planner.maxSolveTime", [](ContactPlanningConfig& c) { c.planner.maxSolveTime = 0.0; }},
      {"planner.maxQpIterations", [](ContactPlanningConfig& c) { c.planner.maxQpIterations = 0; }},
      {"planner.planningFrequency", [](ContactPlanningConfig& c) { c.planner.planningFrequency = 0.0; }},
      {"shared.gravity", [](ContactPlanningConfig& c) { c.shared.gravity = 0.0; }},
      {"shared.comHeight", [](ContactPlanningConfig& c) { c.shared.comHeight = 0.0; }},
      {"shared.gait_limits.minSwingDuration", [](ContactPlanningConfig& c) { c.shared.gaitLimits.minSwingDuration = 0.0; }},
      {"shared.gait_limits.maxSwingDuration", [](ContactPlanningConfig& c) { c.shared.gaitLimits.maxSwingDuration = 0.1; }},
      {"shared.gait_limits.minContactDuration", [](ContactPlanningConfig& c) { c.shared.gaitLimits.minContactDuration = 0.0; }},
      {"shared.gait_limits.maxContactDuration", [](ContactPlanningConfig& c) { c.shared.gaitLimits.maxContactDuration = 0.01; }},
      {"shared.gait_limits.minDoubleSupportDuration",
       [](ContactPlanningConfig& c) { c.shared.gaitLimits.minDoubleSupportDuration = -1.0; }},
      {"shared.slack_penalty.quadratic", [](ContactPlanningConfig& c) { c.shared.slackPenalty.quadratic = -1.0; }},
      {"shared.slack_penalty.linear", [](ContactPlanningConfig& c) { c.shared.slackPenalty.linear = -1.0; }},
      {"shared.bigM", [](ContactPlanningConfig& c) { c.shared.bigM = 0.1; }},
      {"zmp_support_region.halfWidthX", [](ContactPlanningConfig& c) { c.zmpSupportRegion.halfWidthX = 0.0; }},
      {"zmp_support_region.halfWidthY", [](ContactPlanningConfig& c) { c.zmpSupportRegion.halfWidthY = 0.0; }},
      {"zmp_support_region.slack.quadratic", [](ContactPlanningConfig& c) { c.zmpSupportRegion.slack = SlackPenalty{-1.0, 1.0}; }},
      {"reachability.slack.linear", [](ContactPlanningConfig& c) { c.reachability.slack = SlackPenalty{1.0, -1.0}; }},
      {"foot_separation.minStepWidth", [](ContactPlanningConfig& c) { c.footSeparation.minStepWidth = 0.0; }},
      {"foot_separation.maxStepWidth", [](ContactPlanningConfig& c) { c.footSeparation.maxStepWidth = 0.1; }},
      {"foot_separation.maxStepLength", [](ContactPlanningConfig& c) { c.footSeparation.maxStepLength = 0.0; }},
      {"step_width.nominalStepWidth", [](ContactPlanningConfig& c) { c.stepWidth.nominalStepWidth = 0.9; }},
      {"reachability.reachX", [](ContactPlanningConfig& c) { c.reachability.reachX = 0.0; }},
      {"reachability.reachYOuter", [](ContactPlanningConfig& c) { c.reachability.reachYOuter = 0.0; }},
      {"regularization.state", [](ContactPlanningConfig& c) { c.regularization.state = -1.0; }},
      {"regularization.input", [](ContactPlanningConfig& c) { c.regularization.input = -1.0; }},
      {"previous_foothold_consistency.weight", [](ContactPlanningConfig& c) { c.previousFootholdConsistency.weight = -1.0; }},
      {"velocity_tracking.weight", [](ContactPlanningConfig& c) { c.velocityTracking.weight = -1.0; }},
      {"step_width.weight", [](ContactPlanningConfig& c) { c.stepWidth.weight = -1.0; }},
      {"heading_rate_tracking.weight", [](ContactPlanningConfig& c) { c.headingRateTracking.weight = -1.0; }},
      {"heading_tracking.weight", [](ContactPlanningConfig& c) { c.headingTracking.weight = -1.0; }},
      {"foot_yaw_tracking.weight", [](ContactPlanningConfig& c) { c.footYawTracking.weight = -1.0; }},
      {"yaw_torque_regularization.weight", [](ContactPlanningConfig& c) { c.yawTorqueRegularization.weight = -1.0; }},
      {"foot_yaw_regularization.weight", [](ContactPlanningConfig& c) { c.footYawRegularization.weight = -1.0; }},
      {"zmp_regularization.weight", [](ContactPlanningConfig& c) { c.zmpRegularization.weight = -1.0; }},
      {"foothold_regularization.weight", [](ContactPlanningConfig& c) { c.footholdRegularization.weight = -1.0; }},
      {"step_length.weight", [](ContactPlanningConfig& c) { c.stepLength.weight = -1.0; }},
      {"terminal_dcm.weight", [](ContactPlanningConfig& c) { c.terminalDcm.weight = -1.0; }},
      {"contact_switch.cost", [](ContactPlanningConfig& c) { c.contactSwitch.cost = -1.0; }},
      {"plan_consistency.cost", [](ContactPlanningConfig& c) { c.planConsistency.cost = -1.0; }},
      {"double_support_penalty.cost", [](ContactPlanningConfig& c) { c.doubleSupportPenalty.cost = -1.0; }},
      {"event_shift_local_search.iterations", [](ContactPlanningConfig& c) { c.eventShiftLocalSearch.iterations = -1; }},
      {"event_shift_local_search.maxTime", [](ContactPlanningConfig& c) { c.eventShiftLocalSearch.maxTime = -1.0; }},
      {"diving.maxDiveIterations", [](ContactPlanningConfig& c) { c.diving.maxDiveIterations = 0; }},
      {"cadence_stretch.samples", [](ContactPlanningConfig& c) { c.cadenceStretch.samples = -1; }},
      {"cadence_stretch.maxStretch",
       [](ContactPlanningConfig& c) {
         c.cadenceStretch.samples = 3;
         c.cadenceStretch.maxStretch = 0.9;
       }},
      {"heading_relinearization.passes", [](ContactPlanningConfig& c) { c.headingRelinearization.passes = 6; }},
      {"phase_resetting.earlyTouchdownMinSwingRatio", [](ContactPlanningConfig& c) { c.phaseResetting.earlyTouchdownMinSwingRatio = 2.0; }},
      {"phase_resetting.lateTouchdownExtensionStep", [](ContactPlanningConfig& c) { c.phaseResetting.lateTouchdownExtensionStep = 0.0; }},
      {"dcm_step_adjustment.gain", [](ContactPlanningConfig& c) { c.dcmStepAdjustment.gain = -1.0; }},
      {"dcm_step_adjustment.maxOffset", [](ContactPlanningConfig& c) { c.dcmStepAdjustment.maxOffset = -1.0; }},
      {"energy_cadence_modulation.gain", [](ContactPlanningConfig& c) { c.energyCadenceModulation.gain = -1.0; }},
      {"energy_cadence_modulation.deadband", [](ContactPlanningConfig& c) { c.energyCadenceModulation.deadband = -1.0; }},
      {"hlip.sspDuration", [](ContactPlanningConfig& c) { c.hlip.sspDuration = 0.0; }},
      {"hlip.dspDuration", [](ContactPlanningConfig& c) { c.hlip.dspDuration = -0.1; }},
      {"hlip.maxStepLength", [](ContactPlanningConfig& c) { c.hlip.maxStepLength = 0.0; }},
      {"hlip.minStepWidth", [](ContactPlanningConfig& c) { c.hlip.minStepWidth = 0.0; }},
      {"hlip.maxStepWidth", [](ContactPlanningConfig& c) { c.hlip.maxStepWidth = 0.1; }},
      {"hlip.stepWidth", [](ContactPlanningConfig& c) { c.hlip.stepWidth = 0.9; }},
      {"hlip.blend.sharpness", [](ContactPlanningConfig& c) { c.hlip.blend.sharpness = 0.0; }},
      {"hlip.blend.threshold", [](ContactPlanningConfig& c) { c.hlip.blend.threshold = 0.0; }},
      {"hlip.blend.maxCommandedVelocityX", [](ContactPlanningConfig& c) { c.hlip.blend.maxCommandedVelocityX = 0.0; }},
      {"hlip.blend.maxCommandedVelocityY", [](ContactPlanningConfig& c) { c.hlip.blend.maxCommandedVelocityY = 0.0; }},
      {"hlip.blend.maxCommandedYawRate", [](ContactPlanningConfig& c) { c.hlip.blend.maxCommandedYawRate = 0.0; }},
      {"hlip.blend.maxComVelocityX", [](ContactPlanningConfig& c) { c.hlip.blend.maxComVelocityX = 0.0; }},
      {"hlip.blend.maxComVelocityY", [](ContactPlanningConfig& c) { c.hlip.blend.maxComVelocityY = 0.0; }},
  };
  const ContactPlanningConfig valid = modelFreeConfig();
  ASSERT_TRUE(valid.validateStatus().ok()) << "the defaults must validate, or the rejections below prove nothing: "
                                           << valid.validateStatus().message();
  for (const Case& testCase : cases) {
    ContactPlanningConfig broken = valid;
    testCase.breakIt(broken);
    const absl::Status status = broken.validateStatus();
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << testCase.key;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.key))
        << "the rejection must name " << testCase.key << ": " << status.message();
  }
}

TEST(ContactPlanningConfigValidation, BigMMustCoverTheLateralFootSeparation) {
  // In double support both contact binaries are one, the +-M terms of a single-support ZMP box cancel and the box
  // relaxes to |e_y'(zmp - p_i)| <= halfWidthY + M, so a big-M below the lateral separation of the feet clips the
  // double-support region instead of switching off with it. The guard against maxStepLength alone did not see this.
  ContactPlanningConfig config = modelFreeConfig();
  config.footSeparation.maxStepLength = 0.30;
  config.footSeparation.maxStepWidth = 0.45;
  config.shared.bigM = 0.35;  // above maxStepLength, below maxStepWidth
  const absl::Status status = config.validateStatus();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "foot_separation.maxStepWidth")) << status.message();
  config.shared.bigM = 0.45;  // exactly the widest separation the feet may reach is enough
  EXPECT_TRUE(config.validateStatus().ok());
}

/*============================================ warnings: what is documented to fall or block ================*/

/**
 * The library defaults are what a robot runs whose file omits keys, or that has no contact_planning block at all. The
 * type used to default to hlip while everything else stayed the mixed-integer planner's, which assembled every
 * configuration the H-LIP README documents as falling; the defaults are now one coherent lip_miqp configuration.
 */
TEST(ContactPlanningConfigWarnings, TheLibraryDefaultsAreACoherentConfigurationWithNothingToReport) {
  const ContactPlanningConfig config = modelFreeConfig();
  EXPECT_EQ(canonicalPlannerName(config.planner.type), planner::kLipMiqp);
  EXPECT_TRUE(config.validateStatus().ok()) << config.validateStatus().message();
  EXPECT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
}

/**
 * The library default pendulum is the model's: shared.comHeight defaults to 0, which ContactPlanningModelParameters
 * fills in from the robot before validation, at start-up and on every hot reload - the one the DCM terminal cost derives
 * too. It used to default to 0.85 m, the height the Atlas was once hand-set to, so a robot file that omitted the key
 * planned on a pendulum that belongs to no robot while its DCM cost used its own. A configuration with no model to fill
 * it from is refused, naming the key, rather than planned on a height nobody chose.
 */
TEST(ContactPlanningConfigValidation, TheDefaultPendulumIsTheModelsAndIsRefusedWhereNoModelFillsItIn) {
  const ContactPlanningConfig defaults;
  EXPECT_EQ(defaults.shared.comHeight, 0.0) << "the library default must be 0, 'derived from the model'";
  const absl::Status refused = defaults.validateStatus();
  ASSERT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "shared.comHeight")) << refused.message();
  // Positive control: the same defaults with a pendulum validate.
  EXPECT_TRUE(modelFreeConfig().validateStatus().ok()) << modelFreeConfig().validateStatus().message();
}

TEST(ContactPlanningConfigWarnings, TheDefaultHlipBlockIsSafeOnceTheRuleAndTheThreadingAreSet) {
  // Switching a default configuration to hlip, with the settings every hlip configuration needs, must not land on a
  // cadence whose first step does not fit, whose sidestep clips, or whose commit boundary stalls.
  const ContactPlanningConfig config = makeWorkingHlipConfig();
  EXPECT_TRUE(config.validateStatus().ok()) << config.validateStatus().message();
  EXPECT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  EXPECT_LE(HlipContactPlanner::startUpLateralStep(config), config.hlip.maxStepWidth);
}

TEST(ContactPlanningConfigWarnings, HlipWithoutPlannedComOverrideIsReported) {
  ContactPlanningConfig config = makeWorkingHlipConfig();
  ASSERT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  config.formulation.execution.clear();
  const std::vector<std::string> warnings = config.warnings();
  ASSERT_EQ(warnings.size(), 1u) << joinWarnings(warnings);
  EXPECT_TRUE(absl::StrContains(warnings.front(), term::kPlannedComOverride)) << warnings.front();
  EXPECT_TRUE(absl::StrContains(warnings.front(), "execution")) << "the message names the list to edit: " << warnings.front();
  EXPECT_TRUE(config.validateStatus().ok()) << "a warning, not an error";

  // Another rule in the list does not stand in for it.
  config.formulation.execution = {term::kPhaseResetting};
  EXPECT_TRUE(anyMentions(config.warnings(), term::kPlannedComOverride)) << joinWarnings(config.warnings());

  // And the mixed-integer planner does not need it.
  config.planner.type = planner::kLipMiqp;
  config.planner.runInBackgroundThread = true;
  EXPECT_FALSE(anyMentions(config.warnings(), term::kPlannedComOverride)) << joinWarnings(config.warnings());
}

TEST(ContactPlanningConfigWarnings, AFirstStepThatDoesNotFitTheStepWidthClipIsReported) {
  // Whatever the cadence, the warning appears exactly when the start-up demand exceeds hlip.maxStepWidth: a clip just
  // above the demand is silent and one just below it is reported.
  ContactPlanningConfig config = makeWorkingHlipConfig();
  config.hlip.sspDuration = 0.35;  // README 3b's "cadence that fell"
  const scalar_t demand = HlipContactPlanner::startUpLateralStep(config);
  ASSERT_GT(demand, config.hlip.stepWidth);

  config.hlip.maxStepWidth = demand + 0.005;
  EXPECT_FALSE(anyMentions(config.warnings(), "startUpLateralStep")) << joinWarnings(config.warnings());

  config.hlip.maxStepWidth = demand - 0.005;
  const std::vector<std::string> warnings = config.warnings();
  EXPECT_TRUE(anyMentions(warnings, "startUpLateralStep")) << joinWarnings(warnings);
  EXPECT_TRUE(anyMentions(warnings, "hlip.maxStepWidth")) << joinWarnings(warnings);
  EXPECT_TRUE(anyMentions(warnings, "hlip.sspDuration")) << "the message says which key shrinks the demand" << joinWarnings(warnings);
}

TEST(ContactPlanningConfigWarnings, TheBackgroundThreadIsReportedForTheClosedFormPlanner) {
  ContactPlanningConfig config = makeWorkingHlipConfig();
  config.planner.runInBackgroundThread = true;
  const std::vector<std::string> warnings = config.warnings();
  ASSERT_EQ(warnings.size(), 1u) << joinWarnings(warnings);
  EXPECT_TRUE(absl::StrContains(warnings.front(), "planner.runInBackgroundThread")) << warnings.front();
}

/**
 * The Atlas file promised that "ContactPlanningConfig::validate() warns about the mismatch" of a synchronous
 * mixed-integer planner, while warnings() returned before looking at anything but the hlip block.
 */
TEST(ContactPlanningConfigWarnings, ASynchronousMixedIntegerPlannerIsReported) {
  ContactPlanningConfig config = modelFreeConfig();
  ASSERT_EQ(canonicalPlannerName(config.planner.type), planner::kLipMiqp);
  ASSERT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  config.planner.runInBackgroundThread = false;
  const std::vector<std::string> warnings = config.warnings();
  ASSERT_EQ(warnings.size(), 1u) << joinWarnings(warnings);
  EXPECT_TRUE(absl::StrContains(warnings.front(), "planner.runInBackgroundThread")) << warnings.front();
  EXPECT_TRUE(absl::StrContains(warnings.front(), "planner.maxSolveTime")) << "the message states the blocking bound: " << warnings.front();
  EXPECT_TRUE(absl::StrContains(warnings.front(), "event_shift_local_search.maxTime")) << warnings.front();
  EXPECT_TRUE(absl::StrContains(warnings.front(), absl::StrCat(config.planner.maxSolveTime + config.eventShiftLocalSearch.maxTime)))
      << "the bound is the sum of the two budgets: " << warnings.front();
  EXPECT_TRUE(config.validateStatus().ok()) << "the integration tests plan synchronously on purpose, so this stays a warning";

  // Without the local search the bound is the branch-and-bound's budget alone, and the message names only its key.
  config.formulation.setSearchStage(term::kEventShiftLocalSearch, /*on=*/false);
  const std::vector<std::string> withoutLocalSearch = config.warnings();
  ASSERT_EQ(withoutLocalSearch.size(), 1u) << joinWarnings(withoutLocalSearch);
  EXPECT_FALSE(absl::StrContains(withoutLocalSearch.front(), "event_shift_local_search.maxTime")) << withoutLocalSearch.front();
  EXPECT_TRUE(absl::StrContains(withoutLocalSearch.front(), absl::StrCat("planner.maxSolveTime = ", config.planner.maxSolveTime, " s")))
      << withoutLocalSearch.front();
}

TEST(ContactPlanningConfigWarnings, AMixedIntegerCommitWindowShorterThanItsSolveBudgetIsReported) {
  ContactPlanningConfig config = modelFreeConfig();  // lip_miqp on the worker thread
  const scalar_t budget = config.planner.maxSolveTime + config.eventShiftLocalSearch.maxTime;
  config.planner.commitTime = budget + 0.01;
  EXPECT_FALSE(anyMentions(config.warnings(), "planner.commitTime")) << joinWarnings(config.warnings());
  config.planner.commitTime = budget - 0.01;
  EXPECT_TRUE(anyMentions(config.warnings(), "planner.commitTime")) << joinWarnings(config.warnings());
  // Without the local search the budget is the branch-and-bound's alone.
  config.formulation.setSearchStage(term::kEventShiftLocalSearch, /*on=*/false);
  EXPECT_FALSE(anyMentions(config.warnings(), "planner.commitTime")) << joinWarnings(config.warnings());
}

TEST(ContactPlanningConfigWarnings, AnInstantaneousSupportExchangeNeedsACappedCommitExtension) {
  // PlannerSettings::maxCommitExtension: with no double support nothing ends the extension walk, the boundary runs to
  // the end of the stepping region and the planner stops publishing. Both planners can reach it.
  ContactPlanningConfig hlip = makeWorkingHlipConfig();
  hlip.hlip.dspDuration = 0.0;
  hlip.planner.maxCommitExtension = 0.0;
  EXPECT_TRUE(anyMentions(hlip.warnings(), "planner.maxCommitExtension")) << joinWarnings(hlip.warnings());
  EXPECT_TRUE(anyMentions(hlip.warnings(), "hlip.dspDuration")) << joinWarnings(hlip.warnings());
  hlip.planner.maxCommitExtension = std::max(hlip.hlip.sspDuration, hlip.shared.gaitLimits.maxSwingDuration);
  EXPECT_FALSE(anyMentions(hlip.warnings(), "planner.maxCommitExtension")) << joinWarnings(hlip.warnings());

  ContactPlanningConfig miqp = modelFreeConfig();
  miqp.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  EXPECT_TRUE(anyMentions(miqp.warnings(), "shared.gait_limits.minDoubleSupportDuration")) << joinWarnings(miqp.warnings());
  miqp.planner.maxCommitExtension = miqp.shared.gaitLimits.maxSwingDuration;
  EXPECT_TRUE(miqp.warnings().empty()) << joinWarnings(miqp.warnings());
}

TEST(ContactPlanningConfigWarnings, ASustainedSidestepMustFitInsideTheStepWidthClips) {
  // deadbeatStep plans the period-two orbit at +-hlip.stepWidth + v_y (sspDuration + dspDuration) and then clips the
  // placed foot into [minStepWidth, maxStepWidth]. Only the nominal orbit at a zero lateral command was ever checked
  // against those clips, so a configuration could ask, at full stick, for a step it clips on every occurrence.
  const ContactPlanningConfig config = makeWorkingHlipConfig();
  ASSERT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  const scalar_t stepDuration = config.hlip.sspDuration + config.hlip.dspDuration;

  ContactPlanningConfig narrow = config;
  narrow.hlip.blend.maxCommandedVelocityY = (config.hlip.stepWidth - config.hlip.minStepWidth) / stepDuration + 0.01;
  const std::vector<std::string> narrowWarnings = narrow.warnings();
  ASSERT_EQ(narrowWarnings.size(), 1u) << joinWarnings(narrowWarnings);
  EXPECT_TRUE(absl::StrContains(narrowWarnings.front(), "hlip.minStepWidth")) << narrowWarnings.front();
  EXPECT_TRUE(absl::StrContains(narrowWarnings.front(), "hlip.blend.maxCommandedVelocityY")) << narrowWarnings.front();
  EXPECT_TRUE(narrow.validateStatus().ok()) << "a warning, not an error: a shipped file must not stop loading over this";

  ContactPlanningConfig wide = config;
  wide.hlip.maxStepWidth = config.hlip.stepWidth + config.hlip.blend.maxCommandedVelocityY * stepDuration - 0.01;
  wide.hlip.maxStepWidth = std::max(wide.hlip.maxStepWidth, config.hlip.stepWidth);
  const std::vector<std::string> wideWarnings = wide.warnings();
  EXPECT_TRUE(anyMentions(wideWarnings, "clipped to hlip.maxStepWidth")) << joinWarnings(wideWarnings);

  ContactPlanningConfig miqp = narrow;
  miqp.planner.type = planner::kLipMiqp;
  EXPECT_FALSE(anyMentions(miqp.warnings(), "hlip."))
      << "the hlip block is not read by the mixed-integer planner" << joinWarnings(miqp.warnings());
}

TEST(ContactPlanningConfigWarnings, ASwingTimeScaleLongerThanTheShortestPlannedSwingIsReported) {
  const ContactPlanningConfig hlip = makeWorkingHlipConfig();
  EXPECT_NEAR(hlip.shortestPlannedSwingDuration(), hlip.hlip.sspDuration, kTol) << "every hlip swing lasts sspDuration";
  EXPECT_FALSE(hlip.swingTimeScaleWarning(hlip.hlip.sspDuration).has_value());
  const std::optional<std::string> hlipWarning = hlip.swingTimeScaleWarning(hlip.hlip.sspDuration + 0.01);
  ASSERT_TRUE(hlipWarning.has_value());
  EXPECT_TRUE(absl::StrContains(*hlipWarning, "swingTimeScale")) << *hlipWarning;
  EXPECT_TRUE(absl::StrContains(*hlipWarning, "hlip.sspDuration")) << *hlipWarning;

  // Under lip_miqp the shortest swing is the minimum swing rounded UP to whole nodes, never shorter than the key.
  ContactPlanningConfig miqp = modelFreeConfig();
  miqp.planner.dt = 0.1;
  miqp.shared.gaitLimits.minSwingDuration = 0.25;
  EXPECT_NEAR(miqp.shortestPlannedSwingDuration(), 0.3, 1e-9);
  EXPECT_FALSE(miqp.swingTimeScaleWarning(0.3).has_value());
  const std::optional<std::string> miqpWarning = miqp.swingTimeScaleWarning(0.31);
  ASSERT_TRUE(miqpWarning.has_value());
  EXPECT_TRUE(absl::StrContains(*miqpWarning, "shared.gait_limits.minSwingDuration")) << *miqpWarning;
}

/*============================================ the variable layout =========================================*/

TEST(ContactPlanningLayout, PerFootHeadingIndicesAreAbsentWithoutTheHeadingBlock) {
  // The well-known indices are documented as -1 when their block is absent, and the per-foot accessors used to be
  // plain offset arithmetic: footYaw(1) came out as 0, an in-bounds index that aliases c_x, so a term following the
  // documented contract would silently read or write the center-of-mass state on a heading-less formulation.
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
  const absl::StatusOr<ContactPlanningConfig> result = loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false);
  std::remove(file.c_str());
  ASSERT_TRUE(result.ok()) << result.status().message();
  const ContactPlanningConfig& loaded = *result;
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
  // shared.comHeight: no model derives the pendulum here (the library default 0 means "from the model").
  const std::string file = writeTemp("partial_slack_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  shared:\n"
                                     "    comHeight: 0.85\n"
                                     "    slack_penalty:\n"
                                     "      quadratic: 500.0\n"
                                     "      linear: 7.0\n"
                                     "  zmp_support_region:\n"
                                     "    slack:\n"
                                     "      quadratic: 42.0\n"
                                     "  reachability:\n"
                                     "    slack:\n"
                                     "      linear: 3.0\n");
  const ContactPlanningConfig loaded = loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false).value();
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
  const absl::StatusOr<ContactPlanningConfig> loaded = loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false);
  EXPECT_EQ(loaded.status().code(), absl::StatusCode::kInvalidArgument)
      << "a typo must fail the reload atomically, leaving the running planner and its configuration in force";
  EXPECT_TRUE(absl::StrContains(loaded.status().message(), "planner.type")) << loaded.status().message();
  std::remove(file.c_str());
}

TEST(ContactPlanningConfigFile, StructuredLayoutRejectsAnUnknownTermInAList) {
  // shared.comHeight: no model derives the pendulum here, and the list must be the only thing refused.
  const std::string file = writeTemp("bad_list_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  shared:\n"
                                     "    comHeight: 0.85\n"
                                     "  costs:\n"
                                     "    - regularization\n"
                                     "    - gravity_compensation\n");
  const absl::StatusOr<ContactPlanningConfig> loaded = loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false);
  EXPECT_FALSE(loaded.ok());
  EXPECT_TRUE(absl::StrContains(loaded.status().message(), "costs")) << loaded.status().message();
  EXPECT_TRUE(absl::StrContains(loaded.status().message(), "gravity_compensation")) << loaded.status().message();
  std::remove(file.c_str());
}

TEST(ContactPlanningConfigFile, AValueOfTheWrongTypeIsRejectedNamingItsKey) {
  // loadPtreeValue reports only the type it failed to convert to; the loader has to add which key it was reading.
  const std::string file = writeTemp("bad_value_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  planner:\n"
                                     "    dt: fast\n");
  const absl::StatusOr<ContactPlanningConfig> loaded = loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false);
  std::remove(file.c_str());
  EXPECT_EQ(loaded.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(loaded.status().message(), "contact_planning.planner.dt")) << loaded.status().message();
}

TEST(ContactPlanningConfigFile, AnUnreadableFileIsAStatusNotAnException) {
  const std::string file = absl::StrCat(testing::TempDir(), "/no_such_contact_planning.yaml");
  std::remove(file.c_str());
  absl::StatusOr<ContactPlanningConfig> loaded(absl::UnknownError("not called"));
  EXPECT_NO_THROW(loaded = loadContactPlanningConfigStatus(file));
  EXPECT_EQ(loaded.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(loaded.status().message(), file)) << loaded.status().message();
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
  const absl::StatusOr<ContactPlanningConfig> loaded =
      loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false, /*validate=*/false);
  ASSERT_FALSE(loaded.ok()) << "the flat layout of the previous planner is no longer read";
  const std::string message(loaded.status().message());
  EXPECT_TRUE(absl::StrContains(message, "flat layout")) << message;
  EXPECT_TRUE(absl::StrContains(message, "phase_resetting")) << "the message says how to migrate: " << message;
  for (const char* key : {"velocityTrackingWeight", "useAcomDynamics", "enablePhaseResetting"}) {
    EXPECT_TRUE(absl::StrContains(message, key)) << "the message names the flat key " << key << ": " << message;
  }
  std::remove(file.c_str());
}

TEST(ContactPlanningConfigFile, MixingTheTwoLayoutsIsAnError) {
  const std::string file = writeTemp("mixed_contact_planning.yaml",
                                     "contact_planning:\n"
                                     "  planner:\n"
                                     "    dt: 0.1\n"
                                     "  velocityTrackingWeight: 12.0\n");
  const absl::StatusOr<ContactPlanningConfig> loaded = loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false);
  EXPECT_FALSE(loaded.ok());
  EXPECT_TRUE(absl::StrContains(loaded.status().message(), "velocityTrackingWeight")) << loaded.status().message();
  std::remove(file.c_str());
}

TEST(ContactPlanningConfigFile, AMissingBlockGivesTheDefaults) {
  // A task file that selects the planner and carries no contact_planning block of its own.
  const std::string file = writeTemp("empty_contact_planning.yaml", "contactScheduleSource: contact_planner\n");
  // Read the way the interface reads it: unvalidated, because shared.comHeight is still the model's 0 here.
  const absl::StatusOr<ContactPlanningConfig> loaded =
      loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false, /*validate=*/false);
  // Validated without a model to fill the pendulum in, it is refused by the key that needs one.
  const absl::StatusOr<ContactPlanningConfig> validated = loadContactPlanningConfigStatus(file, "contact_planning.", /*verbose=*/false);
  std::remove(file.c_str());
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();
  EXPECT_NEAR(loaded->planner.dt, ContactPlanningConfig{}.planner.dt, kTol);
  EXPECT_EQ(loaded->shared.comHeight, 0.0) << "an absent shared.comHeight is the model's";
  EXPECT_EQ(loaded->formulation, ContactPlanningFormulation{});
  ASSERT_FALSE(validated.ok());
  EXPECT_TRUE(absl::StrContains(validated.status().message(), "shared.comHeight")) << validated.status().message();
  // What such a robot runs, once its model has filled the pendulum in, has to be a configuration with nothing
  // documented to fall or block.
  ContactPlanningConfig filled = *loaded;
  filled.shared.comHeight = kModelFreeComHeight;
  EXPECT_TRUE(filled.validateStatus().ok()) << filled.validateStatus().message();
  EXPECT_TRUE(filled.warnings().empty()) << joinWarnings(filled.warnings());
}

}  // namespace ocs2::humanoid
