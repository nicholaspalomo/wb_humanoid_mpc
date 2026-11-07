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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningTerm.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"

namespace ocs2::humanoid {

namespace {

// The biped's two feet, which every case here is written for.
constexpr size_t kNumFeet = static_cast<size_t>(kBipedFootCount);

constexpr scalar_t kTol = 1.0e-12;

/** Whether any of `warnings` mentions `text`. */
bool anyMentions(const std::vector<std::string>& warnings,
                 const std::string& text) {
  for (const std::string& warning : warnings) {
    if (absl::StrContains(warning, text)) return true;
  }
  return false;
}

std::string joinWarnings(const std::vector<std::string>& warnings) {
  std::string out;
  for (const std::string& warning : warnings) {
    absl::StrAppend(&out, "\n  - ", warning);
  }
  return out.empty() ? std::string(" (none)") : out;
}

/**
 * The pendulum a model-free configuration plans on. The library default of
 * shared.nomPosComRtFeetmeanEwrtWorldZ is 0, "the model's", which
 * ContactPlanningModelParameters::applyTo fills in before validation; a
 * configuration with no model to derive it from - every configuration in this
 * file - has to set a height itself. 0.85 m is the height the DRC Atlas was
 * once hand-set to, the pendulum the library's cadence defaults were validated
 * on (ContactPlanningConfig.h, HlipParameters).
 */
constexpr scalar_t kNomPosComRtFeetmeanEwrtWorldZ = 0.85;

/** The library defaults on the model-free pendulum above. */
ContactPlanningConfig modelFreeConfig() {
  ContactPlanningConfig config;
  config.shared.nomPosComRtFeetmeanEwrtWorldZ = kNomPosComRtFeetmeanEwrtWorldZ;
  return config;
}

/**
 * The H-LIP planner configured the way its README says it has to be: the
 * library defaults with planner.type hlip, the one execution rule the gait
 * cannot walk without, and planning in the pre-solve hook.
 */
ContactPlanningConfig makeWorkingHlipConfig() {
  ContactPlanningConfig config = modelFreeConfig();
  config.planner.type = planner::kHlip;
  config.planner.runInBackgroundThread = false;
  config.formulation.execution = {term::kPlannedComOverride};
  return config;
}

/**
 * The term `name` of `kind` built through the factory, or nothing for the rules
 * the core cannot build (the two planned_*_override rules need the reference
 * manager's robot model; testHlipPlanningIntegration covers them).
 */
std::unique_ptr<ContactPlanningTerm> makeTerm(TermKind kind,
                                              const std::string& name) {
  switch (kind) {
    case TermKind::kModelBlock:
      return checkedValue(ContactPlanningTermFactory::makeModelBlock(name));
    case TermKind::kCost:
      return checkedValue(ContactPlanningTermFactory::makeCost(name));
    case TermKind::kSoftConstraint:
      return checkedValue(ContactPlanningTermFactory::makeSoftConstraint(name));
    case TermKind::kHardConstraint:
      return checkedValue(ContactPlanningTermFactory::makeHardConstraint(name));
    case TermKind::kLogicRule:
      return checkedValue(ContactPlanningTermFactory::makeLogicRule(name));
    case TermKind::kAssignmentCost:
      return checkedValue(ContactPlanningTermFactory::makeAssignmentCost(name));
    case TermKind::kSearchStage:
      return checkedValue(ContactPlanningTermFactory::makeSearchStage(name));
    case TermKind::kExecutionRule:
      if (name == term::kPlannedHeadingOverride ||
          name == term::kPlannedComOverride) {
        return nullptr;
      }
      return checkedValue(ContactPlanningTermFactory::makeExecutionRule(name));
  }
  return nullptr;
}

}  // namespace

/*============================================ names
 * ======================================================*/

TEST(ContactPlanningFormulation, NamesAreMatchedLikeTheTaskFileLists) {
  EXPECT_TRUE(sameTermName("velocity_tracking", "velocityTracking"));
  EXPECT_TRUE(sameTermName("Velocity Tracking", "velocity-tracking"));
  EXPECT_FALSE(sameTermName("velocity_tracking", "heading_tracking"));
  EXPECT_EQ(canonicalTermName(TermKind::kCost, "VelocityTracking"),
            term::kVelocityTracking);
  EXPECT_EQ(canonicalTermName(TermKind::kCost, "no_such_cost"), "");
  EXPECT_EQ(canonicalTermName(TermKind::kLogicRule, "noFlight"),
            term::kNoFlight);
  EXPECT_EQ(canonicalTermName(TermKind::kHardConstraint, "noFlight"),
            term::kNoFlight)
      << "the same name in two lists";
  EXPECT_EQ(canonicalTermName(TermKind::kCost, "noFlight"), "")
      << "but not in a list it does not belong to";
}

TEST(ContactPlanningFormulation,
     DefaultIsThePointMassPlannerWithoutHeuristics) {
  const ContactPlanningFormulation f;
  EXPECT_TRUE(f.validateStatus().ok()) << f.validateStatus().message();
  EXPECT_FALSE(f.usesHeadingModel());
  EXPECT_TRUE(f.hasDynamics(term::kLipCom));
  EXPECT_TRUE(f.hasDynamics(term::kFootholdIntegrator));
  EXPECT_TRUE(f.hasCost(term::kVelocityTracking));
  EXPECT_TRUE(f.hasCost(term::kRegularization))
      << "the regularization is a visible term";
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
  for (const char* absl_nonnull name :
       {term::kHeadingRateTracking, term::kHeadingTracking,
        term::kFootYawTracking, term::kYawTorqueRegularizationName,
        term::kFootYawRegularization}) {
    EXPECT_TRUE(f.hasCost(name)) << name;
  }
  EXPECT_TRUE(f.hasSoftConstraint(term::kHipYawRange));
  EXPECT_TRUE(f.hasHardConstraint(term::kYawTorqueBudgetName));
  EXPECT_TRUE(f.hasHardConstraint(term::kFootYawPinnedInContact));
  EXPECT_TRUE(f.hasSearchStage(term::kHeadingRelinearization));
  EXPECT_TRUE(f.hasExecutionRule(term::kPlannedHeadingOverride));
  // The heading costs sit before zmp_regularization: the accumulation order of
  // the previous planner.
  size_t zmp = 0;
  size_t footYawReg = 0;
  for (size_t i = 0; i < f.costs.size(); ++i) {
    if (sameTermName(f.costs[i], term::kZmpRegularization)) zmp = i;
    if (sameTermName(f.costs[i], term::kFootYawRegularization)) footYawReg = i;
  }
  EXPECT_LT(footYawReg, zmp);
  f.setHeadingModel(false);
  EXPECT_EQ(f, ContactPlanningFormulation{})
      << "removing the heading model restores the default lists";
}

/**
 * "Which terms belong to the heading model" used to be written down three times
 * - requiredBlockOf()'s set, the add/remove lists of setHeadingModel() and each
 * term's requiredBlocks() - with nothing keeping them in step. The first two
 * now read requiredModelBlock(); this holds the third to it, by building every
 * term the core can build.
 */
TEST(ContactPlanningFormulation,
     TheHeadingModelTermsAreTheTermsThatRequireItsBlock) {
  size_t headingTerms = 0;
  size_t otherTerms = 0;
  for (const TermKind kind : allTermKinds()) {
    for (const std::string& name : knownTermNames(kind)) {
      const std::unique_ptr<ContactPlanningTerm> built = makeTerm(kind, name);
      if (built == nullptr) continue;
      const std::vector<std::string> fromTerm = built->requiredBlocks();
      const std::string fromRegistry = requiredModelBlock(kind, name);
      if (fromRegistry.empty()) {
        EXPECT_TRUE(fromTerm.empty())
            << termKindName(kind) << " term " << name
            << " requires a block the registry does not know of";
        ++otherTerms;
      } else {
        ASSERT_EQ(fromTerm.size(), 1u)
            << termKindName(kind) << " term " << name;
        EXPECT_EQ(fromTerm.front(), fromRegistry)
            << termKindName(kind) << " term " << name;
        ++headingTerms;
      }
    }
  }
  // The comparison has both answers to disagree about: heading terms in four
  // lists, and plain terms in every list.
  EXPECT_GE(headingTerms, 9u);
  EXPECT_GE(otherTerms, 20u);
}

TEST(ContactPlanningFormulation,
     SetHeadingModelAddsAndRemovesExactlyTheTermsThatRequireTheBlock) {
  ContactPlanningFormulation on;
  on.setHeadingModel(true);
  ContactPlanningFormulation off = on;
  off.setHeadingModel(false);
  size_t checked = 0;
  for (const TermKind kind : allTermKinds()) {
    if (kind == TermKind::kModelBlock) continue;
    for (const std::string& name : knownTermNames(kind)) {
      if (requiredModelBlock(kind, name).empty()) continue;
      EXPECT_TRUE(ContactPlanningFormulation::listed(on.list(kind), name))
          << termKindName(kind) << " lacks " << name;
      EXPECT_FALSE(ContactPlanningFormulation::listed(off.list(kind), name))
          << termKindName(kind) << " keeps " << name;
      ++checked;
    }
  }
  EXPECT_GE(checked, 9u);
}

TEST(ContactPlanningFormulation,
     ValidationRejectsUnknownDuplicateAndUnsupportedTermsNamingTheList) {
  ContactPlanningFormulation f;
  f.costs.push_back("no_such_cost");
  absl::Status status = f.validateStatus();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(status.message(), "costs")) << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "no_such_cost"))
      << status.message();

  f = ContactPlanningFormulation{};
  f.costs.push_back(term::kVelocityTracking);
  status = f.validateStatus();
  EXPECT_FALSE(status.ok()) << "a duplicate";
  EXPECT_TRUE(absl::StrContains(status.message(), "costs")) << status.message();

  f = ContactPlanningFormulation{};
  f.softConstraints.push_back(term::kHipYawRange);
  status = f.validateStatus();
  EXPECT_FALSE(status.ok()) << "a heading constraint without the heading block";
  EXPECT_TRUE(absl::StrContains(status.message(), "soft_constraints"))
      << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "dynamics"))
      << status.message();

  f = ContactPlanningFormulation{};
  f.dynamics = {term::kFootholdIntegrator, term::kLipCom};
  EXPECT_FALSE(f.validateStatus().ok()) << "the LIP block must come first";

  f = ContactPlanningFormulation{};
  f.execution = {term::kEnergyCadenceModulation, term::kPhaseResetting};
  status = f.validateStatus();
  EXPECT_FALSE(status.ok())
      << "cadence before phase resetting cannot honor an early touch-down";
  EXPECT_TRUE(absl::StrContains(status.message(), "execution"))
      << status.message();
  f.execution = {term::kPhaseResetting, term::kEnergyCadenceModulation};
  EXPECT_TRUE(f.validateStatus().ok());
  EXPECT_TRUE(f.needsPredictedTrajectory());

  // planned_com_override reads the knot's base yaw that
  // planned_heading_override writes, so the heading rule has to run first.
  f = ContactPlanningFormulation{};
  f.setHeadingModel(true);
  f.execution = {term::kPlannedComOverride, term::kPlannedHeadingOverride};
  status = f.validateStatus();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
      << status.message();
  EXPECT_TRUE(
      absl::StrContains(status.message(), term::kPlannedHeadingOverride))
      << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), term::kPlannedComOverride))
      << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "execution"))
      << status.message();
  f.execution = {term::kPlannedHeadingOverride, term::kPlannedComOverride};
  EXPECT_EQ(f.validateStatus(), absl::OkStatus());
  f.execution = {term::kPlannedComOverride};
  EXPECT_EQ(f.validateStatus(), absl::OkStatus()) << "either rule alone";
  // Switching the heading model on puts its override before a listed
  // center-of-mass override, as the shipped files list planned_com_override.
  f = ContactPlanningFormulation{};
  f.execution = {term::kPhaseResetting, term::kPlannedComOverride};
  f.setHeadingModel(true);
  EXPECT_EQ(f.execution,
            (std::vector<std::string>{term::kPhaseResetting,
                                      term::kPlannedHeadingOverride,
                                      term::kPlannedComOverride}));
  EXPECT_EQ(f.validateStatus(), absl::OkStatus());
}

TEST(ContactPlanningFormulation, SummaryListsEveryCollection) {
  const std::string summary = ContactPlanningFormulation{}.summary();
  EXPECT_NE(summary.find("dynamics (2): lip_com, foothold_integrator"),
            std::string::npos)
      << summary;
  EXPECT_NE(summary.find("execution (0): (none)"), std::string::npos)
      << summary;
}

/*============================================ configuration validation
 * ====================================*/

TEST(ContactPlanningConfigValidation,
     AnUnknownPlannerTypeIsRejectedNamingThePlannersThatExist) {
  // The whole reload path - loadContactPlanningConfig,
  // ContactPlanningReferenceManager::setConfig, the parameter updater - reads
  // "validation passed" as "this configuration can be applied", so a name only
  // the factory rejects is caught after the new parameters have already been
  // stored and the background worker started or stopped.
  ContactPlanningConfig config = modelFreeConfig();
  config.planner.type = "lipmiqp2";
  const absl::Status status = config.validateStatus();
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(status.message(), "planner.type"))
      << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "lipmiqp2"))
      << "the message must quote what was written: " << status.message();
  for (const std::string& name : knownPlannerNames()) {
    EXPECT_TRUE(absl::StrContains(status.message(), name))
        << "the message must name " << name << ": " << status.message();
  }
  for (const std::string& name : knownPlannerNames()) {
    ContactPlanningConfig known = modelFreeConfig();
    known.planner.type = name;
    EXPECT_TRUE(known.validateStatus().ok()) << name;
  }
  ContactPlanningConfig spelledDifferently = modelFreeConfig();
  spelledDifferently.planner.type =
      "LIP-MIQP";  // the factory matches names without case or separators
  EXPECT_TRUE(spelledDifferently.validateStatus().ok());
}

/**
 * Every rejection names the one key the operator has to change. Several used to
 * share one message across up to six keys ("cost weights must be non-negative",
 * "invalid solver limits", "heading model weights must be >= 0", ...), so a
 * reload rejected over foothold_regularization.weight left the operator to
 * guess which of six weights was wrong.
 */
TEST(ContactPlanningConfigValidation, EveryRejectionNamesTheKeyToChange) {
  struct Case {
    std::string key;
    std::function<void(ContactPlanningConfig&)> breakIt;
  };
  const std::vector<Case> cases{
      {"planner.dt", [](ContactPlanningConfig& c) { c.planner.dt = 0.0; }},
      {"planner.num_nodes",
       [](ContactPlanningConfig& c) { c.planner.numNodes = 1; }},
      {"planner.commit_time",
       [](ContactPlanningConfig& c) { c.planner.commitTime = -0.1; }},
      {"planner.max_commit_extension",
       [](ContactPlanningConfig& c) { c.planner.maxCommitExtension = 0.01; }},
      {"planner.max_branch_and_bound_nodes",
       [](ContactPlanningConfig& c) { c.planner.maxBranchAndBoundNodes = 0; }},
      {"planner.max_solve_time",
       [](ContactPlanningConfig& c) { c.planner.maxSolveTime = 0.0; }},
      {"planner.max_qp_iterations",
       [](ContactPlanningConfig& c) { c.planner.maxQpIterations = 0; }},
      {"planner.planning_frequency",
       [](ContactPlanningConfig& c) { c.planner.planningFrequency = 0.0; }},
      {"shared.gravity",
       [](ContactPlanningConfig& c) { c.shared.gravity = 0.0; }},
      {"shared.nom_pos_com_rt_feetmean_ewrt_world_z",
       [](ContactPlanningConfig& c) {
         c.shared.nomPosComRtFeetmeanEwrtWorldZ = 0.0;
       }},
      {"shared.gait_limits.min_swing_duration",
       [](ContactPlanningConfig& c) {
         c.shared.gaitLimits.minSwingDuration = 0.0;
       }},
      {"shared.gait_limits.max_swing_duration",
       [](ContactPlanningConfig& c) {
         c.shared.gaitLimits.maxSwingDuration = 0.1;
       }},
      {"shared.gait_limits.min_contact_duration",
       [](ContactPlanningConfig& c) {
         c.shared.gaitLimits.minContactDuration = 0.0;
       }},
      {"shared.gait_limits.max_contact_duration",
       [](ContactPlanningConfig& c) {
         c.shared.gaitLimits.maxContactDuration = 0.01;
       }},
      {"shared.gait_limits.min_double_support_duration",
       [](ContactPlanningConfig& c) {
         c.shared.gaitLimits.minDoubleSupportDuration = -1.0;
       }},
      {"shared.slack_penalty.quadratic",
       [](ContactPlanningConfig& c) {
         c.shared.slackPenalty.quadratic = -1.0;
       }},
      {"shared.slack_penalty.linear",
       [](ContactPlanningConfig& c) { c.shared.slackPenalty.linear = -1.0; }},
      {"shared.big_m", [](ContactPlanningConfig& c) { c.shared.bigM = 0.1; }},
      {"zmp_support_region.max_norm_pos_zmp_rt_foot_ewrt_foot_x",
       [](ContactPlanningConfig& c) {
         c.zmpSupportRegion.maxNormPosZmpRtFootEwrtFootX = 0.0;
       }},
      {"zmp_support_region.max_norm_pos_zmp_rt_foot_ewrt_foot_y",
       [](ContactPlanningConfig& c) {
         c.zmpSupportRegion.maxNormPosZmpRtFootEwrtFootY = 0.0;
       }},
      {"zmp_support_region.slack.quadratic",
       [](ContactPlanningConfig& c) {
         c.zmpSupportRegion.slack =
             SlackPenalty{.quadratic = -1.0, .linear = 1.0};
       }},
      {"reachability.slack.linear",
       [](ContactPlanningConfig& c) {
         c.reachability.slack = SlackPenalty{.quadratic = 1.0, .linear = -1.0};
       }},
      {"foot_separation.min_pos_lfoot_rt_rfoot_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.footSeparation.minPosLfootRtRfootEwrtHeadingY = 0.0;
       }},
      {"foot_separation.max_pos_lfoot_rt_rfoot_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.footSeparation.maxPosLfootRtRfootEwrtHeadingY = 0.1;
       }},
      {"foot_separation.max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x",
       [](ContactPlanningConfig& c) {
         c.footSeparation.maxNormPosLfootRtRfootEwrtHeadingX = 0.0;
       }},
      {"step_width.nom_pos_lfoot_rt_rfoot_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.stepWidth.nomPosLfootRtRfootEwrtHeadingY = 0.9;
       }},
      {"reachability.max_norm_pos_foot_rt_com_ewrt_heading_x",
       [](ContactPlanningConfig& c) {
         c.reachability.maxNormPosFootRtComEwrtHeadingX = 0.0;
       }},
      {"reachability.max_pos_foot_rt_com_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.reachability.maxPosFootRtComEwrtHeadingY = 0.0;
       }},
      {"regularization.state",
       [](ContactPlanningConfig& c) { c.regularization.state = -1.0; }},
      {"regularization.input",
       [](ContactPlanningConfig& c) { c.regularization.input = -1.0; }},
      {"previous_foothold_consistency.weight",
       [](ContactPlanningConfig& c) {
         c.previousFootholdConsistency.weight = -1.0;
       }},
      {"velocity_tracking.weight",
       [](ContactPlanningConfig& c) { c.velocityTracking.weight = -1.0; }},
      {"step_width.weight",
       [](ContactPlanningConfig& c) { c.stepWidth.weight = -1.0; }},
      {"heading_rate_tracking.weight",
       [](ContactPlanningConfig& c) { c.headingRateTracking.weight = -1.0; }},
      {"heading_tracking.weight",
       [](ContactPlanningConfig& c) { c.headingTracking.weight = -1.0; }},
      {"foot_yaw_tracking.weight",
       [](ContactPlanningConfig& c) { c.footYawTracking.weight = -1.0; }},
      {"yaw_torque_regularization_parameters.weight",
       [](ContactPlanningConfig& c) {
         c.yawTorqueRegularizationParameters.weight = -1.0;
       }},
      {"foot_yaw_regularization.weight",
       [](ContactPlanningConfig& c) { c.footYawRegularization.weight = -1.0; }},
      {"zmp_regularization.weight",
       [](ContactPlanningConfig& c) { c.zmpRegularization.weight = -1.0; }},
      {"foothold_regularization.weight",
       [](ContactPlanningConfig& c) {
         c.footholdRegularization.weight = -1.0;
       }},
      {"step_length.weight",
       [](ContactPlanningConfig& c) { c.stepLength.weight = -1.0; }},
      {"terminal_dcm.weight",
       [](ContactPlanningConfig& c) { c.terminalDcm.weight = -1.0; }},
      {"contact_switch.cost",
       [](ContactPlanningConfig& c) { c.contactSwitch.cost = -1.0; }},
      {"plan_consistency.cost",
       [](ContactPlanningConfig& c) { c.planConsistency.cost = -1.0; }},
      {"double_support_penalty.cost",
       [](ContactPlanningConfig& c) { c.doubleSupportPenalty.cost = -1.0; }},
      {"event_shift_local_search.iterations",
       [](ContactPlanningConfig& c) {
         c.eventShiftLocalSearch.iterations = -1;
       }},
      {"event_shift_local_search.max_time",
       [](ContactPlanningConfig& c) {
         c.eventShiftLocalSearch.maxTime = -1.0;
       }},
      {"diving.max_dive_iterations",
       [](ContactPlanningConfig& c) { c.diving.maxDiveIterations = 0; }},
      {"cadence_stretch.samples",
       [](ContactPlanningConfig& c) { c.cadenceStretch.samples = -1; }},
      {"cadence_stretch.max_stretch",
       [](ContactPlanningConfig& c) {
         c.cadenceStretch.samples = 3;
         c.cadenceStretch.maxStretch = 0.9;
       }},
      {"heading_relinearization.passes",
       [](ContactPlanningConfig& c) { c.headingRelinearization.passes = 6; }},
      {"phase_resetting.early_touchdown_min_swing_ratio",
       [](ContactPlanningConfig& c) {
         c.phaseResetting.earlyTouchdownMinSwingRatio = 2.0;
       }},
      {"phase_resetting.late_touchdown_extension_step",
       [](ContactPlanningConfig& c) {
         c.phaseResetting.lateTouchdownExtensionStep = 0.0;
       }},
      {"dcm_step_adjustment.gain",
       [](ContactPlanningConfig& c) { c.dcmStepAdjustment.gain = -1.0; }},
      {"dcm_step_adjustment.max_norm_err_pos_foot_rt_world_ewrt_world_xy",
       [](ContactPlanningConfig& c) {
         c.dcmStepAdjustment.maxNormErrPosFootRtWorldEwrtWorldXy = -1.0;
       }},
      {"energy_cadence_modulation.gain",
       [](ContactPlanningConfig& c) { c.energyCadenceModulation.gain = -1.0; }},
      {"energy_cadence_modulation.deadband",
       [](ContactPlanningConfig& c) {
         c.energyCadenceModulation.deadband = -1.0;
       }},
      {"hlip.ssp_duration",
       [](ContactPlanningConfig& c) { c.hlip.sspDuration = 0.0; }},
      {"hlip.dsp_duration",
       [](ContactPlanningConfig& c) { c.hlip.dspDuration = -0.1; }},
      {"hlip.max_norm_pos_lfoot_rt_rfoot_ewrt_heading_x",
       [](ContactPlanningConfig& c) {
         c.hlip.maxNormPosLfootRtRfootEwrtHeadingX = 0.0;
       }},
      {"hlip.min_pos_lfoot_rt_rfoot_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.hlip.minPosLfootRtRfootEwrtHeadingY = 0.0;
       }},
      {"hlip.max_pos_lfoot_rt_rfoot_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.hlip.maxPosLfootRtRfootEwrtHeadingY = 0.1;
       }},
      {"hlip.nom_pos_lfoot_rt_rfoot_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.hlip.nomPosLfootRtRfootEwrtHeadingY = 0.9;
       }},
      {"hlip.blend.sharpness",
       [](ContactPlanningConfig& c) { c.hlip.blend.sharpness = 0.0; }},
      {"hlip.blend.threshold",
       [](ContactPlanningConfig& c) { c.hlip.blend.threshold = 0.0; }},
      {"hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_x",
       [](ContactPlanningConfig& c) {
         c.hlip.blend.cmdMaxLinvelBaseRtWorldEwrtHeadingX = 0.0;
       }},
      {"hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.hlip.blend.cmdMaxLinvelBaseRtWorldEwrtHeadingY = 0.0;
       }},
      {"hlip.blend.cmd_max_angvel_base_rt_world_ewrt_world_z",
       [](ContactPlanningConfig& c) {
         c.hlip.blend.cmdMaxAngvelBaseRtWorldEwrtWorldZ = 0.0;
       }},
      {"hlip.blend.max_proxy_linvel_com_rt_world_ewrt_heading_x",
       [](ContactPlanningConfig& c) {
         c.hlip.blend.maxProxyLinvelComRtWorldEwrtHeadingX = 0.0;
       }},
      {"hlip.blend.max_proxy_linvel_com_rt_world_ewrt_heading_y",
       [](ContactPlanningConfig& c) {
         c.hlip.blend.maxProxyLinvelComRtWorldEwrtHeadingY = 0.0;
       }},
  };
  const ContactPlanningConfig valid = modelFreeConfig();
  ASSERT_TRUE(valid.validateStatus().ok())
      << "the defaults must validate, or the rejections below prove nothing: "
      << valid.validateStatus().message();
  for (const Case& testCase : cases) {
    ContactPlanningConfig broken = valid;
    testCase.breakIt(broken);
    const absl::Status status = broken.validateStatus();
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument)
        << testCase.key;
    EXPECT_TRUE(absl::StrContains(status.message(), testCase.key))
        << "the rejection must name " << testCase.key << ": "
        << status.message();
  }
}

TEST(ContactPlanningConfigValidation, BigMMustCoverTheLateralFootSeparation) {
  // In double support both contact binaries are one, the +-M terms of a
  // single-support ZMP box cancel and the box relaxes to |e_y'(zmp - p_i)| <=
  // maxNormPosZmpRtFootEwrtFootY + M, so a big-M below the lateral separation
  // of the feet clips the double-support region instead of switching off with
  // it. The guard against maxNormPosLfootRtRfootEwrtHeadingX alone did not see
  // this.
  ContactPlanningConfig config = modelFreeConfig();
  config.footSeparation.maxNormPosLfootRtRfootEwrtHeadingX = 0.30;
  config.footSeparation.maxPosLfootRtRfootEwrtHeadingY = 0.45;
  config.shared.bigM = 0.35;  // above maxNormPosLfootRtRfootEwrtHeadingX, below
                              // maxPosLfootRtRfootEwrtHeadingY
  const absl::Status status = config.validateStatus();
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "foot_separation.max_pos_lfoot_rt_rfoot_ewrt_heading_y"))
      << status.message();
  config.shared.bigM =
      0.45;  // exactly the widest separation the feet may reach is enough
  EXPECT_TRUE(config.validateStatus().ok());
}

/*============================================ warnings: what is documented to
 * fall or block ================*/

/**
 * The library defaults are what a robot runs whose file omits keys, or that has
 * no contact_planning block at all. The type used to default to hlip while
 * everything else stayed the mixed-integer planner's, which assembled every
 * configuration the H-LIP README documents as falling; the defaults are now one
 * coherent lip_miqp configuration.
 */
TEST(ContactPlanningConfigWarnings,
     TheLibraryDefaultsAreACoherentConfigurationWithNothingToReport) {
  const ContactPlanningConfig config = modelFreeConfig();
  EXPECT_EQ(canonicalPlannerName(config.planner.type), planner::kLipMiqp);
  EXPECT_TRUE(config.validateStatus().ok())
      << config.validateStatus().message();
  EXPECT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
}

/**
 * The library default pendulum is the model's:
 * shared.nomPosComRtFeetmeanEwrtWorldZ defaults to unset, which
 * ContactPlanningModelParameters fills in from the robot before validation, at
 * start-up and on every hot reload - the one the DCM terminal cost derives too.
 * It used to default to 0.85 m, the height the Atlas was once hand-set to, so a
 * robot file that omitted the key planned on a pendulum that belongs to no
 * robot while its DCM cost used its own. A configuration with no model to fill
 * it from is refused, naming the key, rather than planned on a height nobody
 * chose.
 */
TEST(ContactPlanningConfigValidation,
     TheDefaultPendulumIsTheModelsAndIsRefusedWhereNoModelFillsItIn) {
  const ContactPlanningConfig defaults;
  EXPECT_FALSE(defaults.shared.nomPosComRtFeetmeanEwrtWorldZ.has_value())
      << "the library default must be unset, 'derived from the model'";
  const absl::Status refused = defaults.validateStatus();
  ASSERT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(),
                                "shared.nom_pos_com_rt_feetmean_ewrt_world_z"))
      << refused.message();
  // Positive control: the same defaults with a pendulum validate.
  EXPECT_TRUE(modelFreeConfig().validateStatus().ok())
      << modelFreeConfig().validateStatus().message();
}

TEST(ContactPlanningConfigWarnings,
     TheDefaultHlipBlockIsSafeOnceTheRuleAndTheThreadingAreSet) {
  // Switching a default configuration to hlip, with the settings every hlip
  // configuration needs, must not land on a cadence whose first step does not
  // fit, whose sidestep clips, or whose commit boundary stalls.
  const ContactPlanningConfig config = makeWorkingHlipConfig();
  EXPECT_TRUE(config.validateStatus().ok())
      << config.validateStatus().message();
  EXPECT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  HUMANOID_ASSERT_OK_AND_ASSIGN(const scalar_t startUpStep,
                                HlipContactPlanner::startUpLateralStep(config));
  EXPECT_LE(startUpStep, config.hlip.maxPosLfootRtRfootEwrtHeadingY);
}

TEST(ContactPlanningConfigWarnings, HlipWithoutPlannedComOverrideIsReported) {
  ContactPlanningConfig config = makeWorkingHlipConfig();
  ASSERT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  config.formulation.execution.clear();
  const std::vector<std::string> warnings = config.warnings();
  ASSERT_EQ(warnings.size(), 1u) << joinWarnings(warnings);
  EXPECT_TRUE(absl::StrContains(warnings.front(), term::kPlannedComOverride))
      << warnings.front();
  EXPECT_TRUE(absl::StrContains(warnings.front(), "execution"))
      << "the message names the list to edit: " << warnings.front();
  EXPECT_TRUE(config.validateStatus().ok()) << "a warning, not an error";

  // Another rule in the list does not stand in for it.
  config.formulation.execution = {term::kPhaseResetting};
  EXPECT_TRUE(anyMentions(config.warnings(), term::kPlannedComOverride))
      << joinWarnings(config.warnings());

  // And the mixed-integer planner does not need it.
  config.planner.type = planner::kLipMiqp;
  config.planner.runInBackgroundThread = true;
  EXPECT_FALSE(anyMentions(config.warnings(), term::kPlannedComOverride))
      << joinWarnings(config.warnings());
}

TEST(ContactPlanningConfigWarnings,
     AFirstStepThatDoesNotFitTheStepWidthClipIsReported) {
  // Whatever the cadence, the warning appears exactly when the start-up demand
  // exceeds hlip.maxPosLfootRtRfootEwrtHeadingY: a clip just above the demand
  // is silent and one just below it is reported.
  ContactPlanningConfig config = makeWorkingHlipConfig();
  config.hlip.sspDuration = 0.35;  // README 3b's "cadence that fell"
  HUMANOID_ASSERT_OK_AND_ASSIGN(const scalar_t demand,
                                HlipContactPlanner::startUpLateralStep(config));
  ASSERT_GT(demand, config.hlip.nomPosLfootRtRfootEwrtHeadingY);

  config.hlip.maxPosLfootRtRfootEwrtHeadingY = demand + 0.005;
  EXPECT_FALSE(anyMentions(config.warnings(), "startUpLateralStep"))
      << joinWarnings(config.warnings());

  config.hlip.maxPosLfootRtRfootEwrtHeadingY = demand - 0.005;
  const std::vector<std::string> warnings = config.warnings();
  EXPECT_TRUE(anyMentions(warnings, "startUpLateralStep"))
      << joinWarnings(warnings);
  EXPECT_TRUE(
      anyMentions(warnings, "hlip.max_pos_lfoot_rt_rfoot_ewrt_heading_y"))
      << joinWarnings(warnings);
  EXPECT_TRUE(anyMentions(warnings, "hlip.ssp_duration"))
      << "the message says which key shrinks the demand"
      << joinWarnings(warnings);
}

TEST(ContactPlanningConfigWarnings,
     TheBackgroundThreadIsReportedForTheClosedFormPlanner) {
  ContactPlanningConfig config = makeWorkingHlipConfig();
  config.planner.runInBackgroundThread = true;
  const std::vector<std::string> warnings = config.warnings();
  ASSERT_EQ(warnings.size(), 1u) << joinWarnings(warnings);
  EXPECT_TRUE(absl::StrContains(warnings.front(), "planner.threading"))
      << warnings.front();
}

/**
 * The Atlas file promised that "ContactPlanningConfig::validate() warns about
 * the mismatch" of a synchronous mixed-integer planner, while warnings()
 * returned before looking at anything but the hlip block.
 */
TEST(ContactPlanningConfigWarnings, ASynchronousMixedIntegerPlannerIsReported) {
  ContactPlanningConfig config = modelFreeConfig();
  ASSERT_EQ(canonicalPlannerName(config.planner.type), planner::kLipMiqp);
  ASSERT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  config.planner.runInBackgroundThread = false;
  const std::vector<std::string> warnings = config.warnings();
  ASSERT_EQ(warnings.size(), 1u) << joinWarnings(warnings);
  EXPECT_TRUE(absl::StrContains(warnings.front(), "planner.threading"))
      << warnings.front();
  EXPECT_TRUE(absl::StrContains(warnings.front(), "planner.max_solve_time"))
      << "the message states the blocking bound: " << warnings.front();
  EXPECT_TRUE(
      absl::StrContains(warnings.front(), "event_shift_local_search.max_time"))
      << warnings.front();
  EXPECT_TRUE(absl::StrContains(
      warnings.front(), absl::StrCat(config.planner.maxSolveTime +
                                     config.eventShiftLocalSearch.maxTime)))
      << "the bound is the sum of the two budgets: " << warnings.front();
  EXPECT_TRUE(config.validateStatus().ok())
      << "the integration tests plan synchronously on purpose, so this stays a "
         "warning";

  // Without the local search the bound is the branch-and-bound's budget alone,
  // and the message names only its key.
  config.formulation.setSearchStage(term::kEventShiftLocalSearch, /*on=*/false);
  const std::vector<std::string> withoutLocalSearch = config.warnings();
  ASSERT_EQ(withoutLocalSearch.size(), 1u) << joinWarnings(withoutLocalSearch);
  EXPECT_FALSE(absl::StrContains(withoutLocalSearch.front(),
                                 "event_shift_local_search.max_time"))
      << withoutLocalSearch.front();
  EXPECT_TRUE(absl::StrContains(
      withoutLocalSearch.front(),
      absl::StrCat("planner.max_solve_time = ", config.planner.maxSolveTime,
                   " s")))
      << withoutLocalSearch.front();
}

TEST(ContactPlanningConfigWarnings,
     AMixedIntegerCommitWindowShorterThanItsSolveBudgetIsReported) {
  ContactPlanningConfig config =
      modelFreeConfig();  // lip_miqp on the worker thread
  const scalar_t budget =
      config.planner.maxSolveTime + config.eventShiftLocalSearch.maxTime;
  config.planner.commitTime = budget + 0.01;
  EXPECT_FALSE(anyMentions(config.warnings(), "planner.commit_time"))
      << joinWarnings(config.warnings());
  config.planner.commitTime = budget - 0.01;
  EXPECT_TRUE(anyMentions(config.warnings(), "planner.commit_time"))
      << joinWarnings(config.warnings());
  // Without the local search the budget is the branch-and-bound's alone.
  config.formulation.setSearchStage(term::kEventShiftLocalSearch, /*on=*/false);
  EXPECT_FALSE(anyMentions(config.warnings(), "planner.commit_time"))
      << joinWarnings(config.warnings());
}

TEST(ContactPlanningConfigWarnings,
     AnInstantaneousSupportExchangeNeedsACappedCommitExtension) {
  // PlannerSettings::maxCommitExtension: with no double support nothing ends
  // the extension walk, the boundary runs to the end of the stepping region and
  // the planner stops publishing. Both planners can reach it.
  ContactPlanningConfig hlip = makeWorkingHlipConfig();
  hlip.hlip.dspDuration = 0.0;
  hlip.planner.maxCommitExtension = 0.0;
  EXPECT_TRUE(anyMentions(hlip.warnings(), "planner.max_commit_extension"))
      << joinWarnings(hlip.warnings());
  EXPECT_TRUE(anyMentions(hlip.warnings(), "hlip.dsp_duration"))
      << joinWarnings(hlip.warnings());
  hlip.planner.maxCommitExtension =
      std::max(hlip.hlip.sspDuration, hlip.shared.gaitLimits.maxSwingDuration);
  EXPECT_FALSE(anyMentions(hlip.warnings(), "planner.max_commit_extension"))
      << joinWarnings(hlip.warnings());

  ContactPlanningConfig miqp = modelFreeConfig();
  miqp.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  EXPECT_TRUE(anyMentions(miqp.warnings(),
                          "shared.gait_limits.min_double_support_duration"))
      << joinWarnings(miqp.warnings());
  miqp.planner.maxCommitExtension = miqp.shared.gaitLimits.maxSwingDuration;
  EXPECT_TRUE(miqp.warnings().empty()) << joinWarnings(miqp.warnings());
}

TEST(ContactPlanningConfigWarnings,
     ASustainedSidestepMustFitInsideTheStepWidthClips) {
  // deadbeatStep plans the period-two orbit at
  // +-hlip.nomPosLfootRtRfootEwrtHeadingY + v_y (sspDuration + dspDuration) and
  // then clips the placed foot into [minPosLfootRtRfootEwrtHeadingY,
  // maxPosLfootRtRfootEwrtHeadingY]. Only the nominal orbit at a zero lateral
  // command was ever checked against those clips, so a configuration could ask,
  // at full stick, for a step it clips on every occurrence.
  const ContactPlanningConfig config = makeWorkingHlipConfig();
  ASSERT_TRUE(config.warnings().empty()) << joinWarnings(config.warnings());
  const scalar_t stepDuration =
      config.hlip.sspDuration + config.hlip.dspDuration;

  ContactPlanningConfig narrow = config;
  narrow.hlip.blend.cmdMaxLinvelBaseRtWorldEwrtHeadingY =
      (config.hlip.nomPosLfootRtRfootEwrtHeadingY -
       config.hlip.minPosLfootRtRfootEwrtHeadingY) /
          stepDuration +
      0.01;
  const std::vector<std::string> narrowWarnings = narrow.warnings();
  ASSERT_EQ(narrowWarnings.size(), 1u) << joinWarnings(narrowWarnings);
  EXPECT_TRUE(absl::StrContains(narrowWarnings.front(),
                                "hlip.min_pos_lfoot_rt_rfoot_ewrt_heading_y"))
      << narrowWarnings.front();
  EXPECT_TRUE(absl::StrContains(
      narrowWarnings.front(),
      "hlip.blend.cmd_max_linvel_base_rt_world_ewrt_heading_y"))
      << narrowWarnings.front();
  EXPECT_TRUE(narrow.validateStatus().ok())
      << "a warning, not an error: a shipped file must not stop loading over "
         "this";

  ContactPlanningConfig wide = config;
  wide.hlip.maxPosLfootRtRfootEwrtHeadingY =
      config.hlip.nomPosLfootRtRfootEwrtHeadingY +
      config.hlip.blend.cmdMaxLinvelBaseRtWorldEwrtHeadingY * stepDuration -
      0.01;
  wide.hlip.maxPosLfootRtRfootEwrtHeadingY =
      std::max(wide.hlip.maxPosLfootRtRfootEwrtHeadingY,
               config.hlip.nomPosLfootRtRfootEwrtHeadingY);
  const std::vector<std::string> wideWarnings = wide.warnings();
  EXPECT_TRUE(anyMentions(
      wideWarnings, "clipped to hlip.max_pos_lfoot_rt_rfoot_ewrt_heading_y"))
      << joinWarnings(wideWarnings);

  ContactPlanningConfig miqp = narrow;
  miqp.planner.type = planner::kLipMiqp;
  EXPECT_FALSE(anyMentions(miqp.warnings(), "hlip."))
      << "the hlip block is not read by the mixed-integer planner"
      << joinWarnings(miqp.warnings());
}

TEST(ContactPlanningConfigWarnings,
     ASwingTimeScaleLongerThanTheShortestPlannedSwingIsReported) {
  const ContactPlanningConfig hlip = makeWorkingHlipConfig();
  EXPECT_NEAR(hlip.shortestPlannedSwingDuration(), hlip.hlip.sspDuration, kTol)
      << "every hlip swing lasts sspDuration";
  EXPECT_FALSE(hlip.swingTimeScaleWarning(hlip.hlip.sspDuration).has_value());
  const std::optional<std::string> hlipWarning =
      hlip.swingTimeScaleWarning(hlip.hlip.sspDuration + 0.01);
  if (!hlipWarning.has_value()) GTEST_FAIL();
  EXPECT_TRUE(absl::StrContains(*hlipWarning, "swing_time_scale"))
      << *hlipWarning;
  EXPECT_TRUE(absl::StrContains(*hlipWarning, "hlip.ssp_duration"))
      << *hlipWarning;

  // Under lip_miqp the shortest swing is the minimum swing rounded UP to whole
  // nodes, never shorter than the key.
  ContactPlanningConfig miqp = modelFreeConfig();
  miqp.planner.dt = 0.1;
  miqp.shared.gaitLimits.minSwingDuration = 0.25;
  EXPECT_NEAR(miqp.shortestPlannedSwingDuration(), 0.3, 1.0e-9);
  EXPECT_FALSE(miqp.swingTimeScaleWarning(0.3).has_value());
  const std::optional<std::string> miqpWarning =
      miqp.swingTimeScaleWarning(0.31);
  if (!miqpWarning.has_value()) GTEST_FAIL();
  EXPECT_TRUE(
      absl::StrContains(*miqpWarning, "shared.gait_limits.min_swing_duration"))
      << *miqpWarning;
}

/*============================================ the variable layout
 * =========================================*/

TEST(ContactPlanningLayout,
     PerFootHeadingIndicesAreAbsentWithoutTheHeadingBlock) {
  // The well-known indices are documented as -1 when their block is absent, and
  // the per-foot accessors used to be plain offset arithmetic:
  // eulerzyxFootRtWorldYaw(1) came out as 0, an in-bounds index that aliases
  // c_x, so a term following the documented contract would silently read or
  // write the center-of-mass state on a heading-less formulation.
  const Layout none;
  EXPECT_FALSE(none.hasHeading);
  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    EXPECT_EQ(none.eulerzyxFootRtWorldYaw(foot), -1) << "foot " << foot;
    EXPECT_EQ(none.torqueFootEwrtWorldZ(foot), -1) << "foot " << foot;
    EXPECT_EQ(none.errEulerzyxFootRtWorldYaw(foot), -1) << "foot " << foot;
  }
  Layout heading;
  heading.hasHeading = true;
  heading.startEulerzyxFeetRtWorldYaw = 8;
  heading.yawTorqueStart = 5;
  heading.startErrEulerzyxFeetRtWorldYaw = 7;
  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    EXPECT_EQ(heading.eulerzyxFootRtWorldYaw(foot), 8 + static_cast<int>(foot))
        << "foot " << foot;
    EXPECT_EQ(heading.torqueFootEwrtWorldZ(foot), 5 + static_cast<int>(foot))
        << "foot " << foot;
    EXPECT_EQ(heading.errEulerzyxFootRtWorldYaw(foot),
              7 + static_cast<int>(foot))
        << "foot " << foot;
  }
}

}  // namespace ocs2::humanoid
