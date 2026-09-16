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

#include <cmath>

#include <stdexcept>
#include <string>

#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/cost/StepLengthCost.h"
#include "humanoid_common_mpc/contact_planning/cost/TerminalDcmCost.h"
#include "humanoid_common_mpc/contact_planning/cost/VelocityTrackingCost.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/execution/ScheduleAdaptationPipeline.h"
#include "humanoid_common_mpc/contact_planning/model/LipBlockIndices.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

namespace {

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.dt = 0.1;
  config.planner.numNodes = 12;
  config.planner.commitTime = 0.0;
  config.planner.maxBranchAndBoundNodes = 3000;
  config.planner.maxSolveTime = 10.0;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.5;
  config.shared.gaitLimits.minContactDuration = 0.15;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.1;
  config.eventShiftLocalSearch.maxTime = 2.0;
  config.validate();
  return config;
}

ContactPlanningConfig makeHeadingConfig() {
  ContactPlanningConfig config = makeConfig();
  config.setHeadingModel(true);
  config.yawTorqueBudget.torsionalFrictionTorque = 20.0;
  config.yawTorqueBudget.doubleSupportYawCouple = 40.0;
  config.setSymmetricFootYawOffset(0.5);
  config.validate();
  return config;
}

ContactPlannerInput makeStandingInput() {
  ContactPlannerInput input;
  input.time = 3.0;
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(0.0, -0.125);
  input.contacts = {true, true};
  input.phaseElapsedTime = {5.0, 5.0};
  input.yawInertia = 10.0;
  return input;
}

ContactPlannerInput makeWalkingInput() {
  ContactPlannerInput input = makeStandingInput();
  input.velocityCommand = vector2_t(0.4, 0.0);
  return input;
}

}  // namespace

/*============================================ layout and rows ============================================*/

TEST(ContactPlanningTerms, LayoutIsComposedFromTheBlocksInOrder) {
  const Layout lip = LipContactPlanner::makeLayout(makeConfig());
  EXPECT_EQ(lip.nx, LIP_STATE_DIM);
  EXPECT_EQ(lip.nu, LIP_INPUT_DIM);
  EXPECT_FALSE(lip.hasHeading);
  EXPECT_EQ(lip.state(var::kComX), LIP_CX);
  EXPECT_EQ(lip.state(var::footX(1)), LIP_PRX);
  EXPECT_EQ(lip.input(var::contact(0)), LIP_CL);
  EXPECT_EQ(lip.input(var::contact(1)), LIP_CR);
  EXPECT_NE(lip.describe().find("x = [c_x c_y v_x v_y p_Lx p_Ly p_Rx p_Ry] (8)"), std::string::npos) << lip.describe();

  const Layout heading = LipContactPlanner::makeLayout(makeHeadingConfig());
  EXPECT_EQ(heading.nx, LIP_STATE_DIM + 2 + static_cast<int>(N_CONTACTS));
  EXPECT_EQ(heading.nu, LIP_INPUT_DIM + 2 * static_cast<int>(N_CONTACTS));
  EXPECT_TRUE(heading.hasHeading);
  EXPECT_EQ(heading.heading, LIP_STATE_DIM) << "the heading block appends, the LIP indices are unchanged";
  EXPECT_EQ(heading.state(var::kComX), LIP_CX);
  EXPECT_EQ(heading.input(var::contact(1)), LIP_CR);
  EXPECT_EQ(heading.yawTorque(0), LIP_INPUT_DIM);
  EXPECT_EQ(heading.footYawDelta(1), LIP_INPUT_DIM + 3);
}

TEST(ContactPlanningTerms, RowCountsPerNodeAreTheDocumentedOnes) {
  {
    LipContactPlanner planner(makeConfig());
    const OcpQpProblem problem = planner.buildProblem(makeStandingInput());
    EXPECT_EQ(problem.stages.front().numGeneralConstraints(), 27) << "1 no-flight + 12 support + 8 foot motion + 4 reach + 2 separation";
    EXPECT_EQ(problem.stages.back().numGeneralConstraints(), 6) << "the terminal node carries the kinematic rows only";
    EXPECT_EQ(static_cast<int>(problem.stages.front().softGeneralIndices.size()), 18);
    EXPECT_EQ(static_cast<int>(problem.stages.front().idxbu.size()), 6);
  }
  {
    LipContactPlanner planner(makeHeadingConfig());
    const OcpQpProblem problem = planner.buildProblem(makeStandingInput());
    EXPECT_EQ(problem.stages.front().numGeneralConstraints(), 37) << "+ 4 torque budget + 4 yaw pin + 2 hip range";
    EXPECT_EQ(problem.stages.back().numGeneralConstraints(), 8);
    EXPECT_EQ(static_cast<int>(problem.stages.front().idxbu.size()), 10);
  }
}

TEST(ContactPlanningTerms, TermsAreAssembledInListOrderAndOnTheirNodeSets) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kRegularization, term::kVelocityTracking};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {term::kNoFlight};
  config.validate();
  LipContactPlanner planner(config);
  const ContactPlanningProblem& problem = planner.getProblem();
  ASSERT_EQ(problem.costs.size(), 2u);
  EXPECT_EQ(problem.costs.nameAt(0), term::kRegularization);
  EXPECT_EQ(problem.costs.nameAt(1), term::kVelocityTracking);
  EXPECT_EQ(problem.costs.get(term::kVelocityTracking).nodeSet(), NodeSet::ALL);
  const OcpQpProblem qp = planner.buildProblem(makeWalkingInput());
  EXPECT_EQ(qp.stages.front().numGeneralConstraints(), 1);
  EXPECT_EQ(qp.stages.back().numGeneralConstraints(), 0);
  // The velocity tracking cost acts on the terminal node too: q_N carries -2 w v_cmd on the velocity entries.
  EXPECT_NEAR(qp.stages.back().q(LIP_VX), -2.0 * config.velocityTracking.weight * 0.4, 1e-12);
  EXPECT_NEAR(qp.stages.back().Q(LIP_CX, LIP_CX), config.regularization.state, 1e-15) << "only the regularisation touches the CoM";
}

// step_length: w ||dp - d_nom (1 - c)||^2 per foot and axis on the running nodes. The residual is affine in dp and the
// relaxed binary c, so the stage carries the cross terms of the two: R(dp, dp) = 2w, R(dp, c) = 2w d, R(c, c) = 2w d^2,
// r(dp) = -2w d, r(c) = -2w d^2 (0.5 u'Ru + r'u convention), summed over both axes for the c entries.
TEST(ContactPlanningTerms, StepLengthCostTiesTheSwingDisplacementToTheCommandedSpeed) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kStepLength};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.stepLength.weight = 20.0;
  config.validate();
  LipContactPlanner planner(config);
  const auto& cost = planner.getProblem().costs.get<StepLengthCost>(term::kStepLength);
  EXPECT_EQ(cost.nodeSet(), NodeSet::RUNNING);
  // T_stride / T_swing = 2 (0.3 + 0.1) / 0.3
  EXPECT_NEAR(cost.strideToSwingRatio(), 8.0 / 3.0, 1e-12);
  const vector2_t d = cost.nominalDisplacementPerNode(vector2_t(0.4, 0.0), 0.1);
  EXPECT_NEAR(d.x(), 0.4 * 0.1 * 8.0 / 3.0, 1e-12);
  EXPECT_NEAR(d.y(), 0.0, 1e-12);

  const scalar_t w = config.stepLength.weight;
  const OcpQpProblem qp = planner.buildProblem(makeWalkingInput());
  const OcpQpStage& stage = qp.stages.front();
  EXPECT_NEAR(stage.R(LIP_DLX, LIP_DLX), 2.0 * w, 1e-12);
  EXPECT_NEAR(stage.R(LIP_DLY, LIP_DLY), 2.0 * w, 1e-12);
  EXPECT_NEAR(stage.R(LIP_DLX, LIP_CL), 2.0 * w * d.x(), 1e-12);
  EXPECT_NEAR(stage.R(LIP_CL, LIP_DLX), 2.0 * w * d.x(), 1e-12);
  EXPECT_NEAR(stage.R(LIP_DLY, LIP_CL), 0.0, 1e-12) << "no lateral command, no lateral nominal";
  EXPECT_NEAR(stage.R(LIP_CL, LIP_CL), 2.0 * w * d.squaredNorm(), 1e-12);
  EXPECT_NEAR(stage.r(LIP_DLX), -2.0 * w * d.x(), 1e-12);
  EXPECT_NEAR(stage.r(LIP_CL), -2.0 * w * d.squaredNorm(), 1e-12);
  EXPECT_NEAR(stage.R(LIP_DRX, LIP_CR), 2.0 * w * d.x(), 1e-12) << "both feet";
  EXPECT_NEAR(stage.R(LIP_DLX, LIP_CR), 0.0, 1e-12) << "no coupling across feet";
  EXPECT_TRUE(stage.Q.isZero()) << "the term touches no state";
  EXPECT_EQ(qp.stages.back().numInputs(), 0) << "not on the terminal node";

  // Standing still: the nominal is zero and the term is a plain regulariser of the foot displacement.
  const OcpQpProblem standing = planner.buildProblem(makeStandingInput());
  EXPECT_NEAR(standing.stages.front().R(LIP_DLX, LIP_DLX), 2.0 * w, 1e-12);
  EXPECT_NEAR(standing.stages.front().R(LIP_DLX, LIP_CL), 0.0, 1e-12);
  EXPECT_NEAR(standing.stages.front().r(LIP_CL), 0.0, 1e-12);
  EXPECT_NE(cost.describe().find("d_nom = v_cmd dt T_stride / T_swing"), std::string::npos);
}

// terminal_dcm.trackCommandedVelocity moves the target of the DCM from the last ZMP (rest) to zmp + v_cmd / omega (a CoM
// over the foot that keeps moving at the commanded velocity): only the linear term of the last running node changes.
TEST(ContactPlanningTerms, TerminalDcmCanTrackTheCommandedVelocityInsteadOfComingToRest) {
  ContactPlanningConfig rest = makeConfig();
  rest.formulation.costs = {term::kTerminalDcm};
  rest.formulation.softConstraints = {};
  rest.formulation.hardConstraints = {};
  rest.validate();
  ContactPlanningConfig walking = rest;
  walking.terminalDcm.trackCommandedVelocity = true;
  LipContactPlanner restPlanner(rest);
  LipContactPlanner walkingPlanner(walking);
  EXPECT_FALSE(restPlanner.getProblem().costs.get<TerminalDcmCost>(term::kTerminalDcm).tracksCommandedVelocity());
  EXPECT_TRUE(walkingPlanner.getProblem().costs.get<TerminalDcmCost>(term::kTerminalDcm).tracksCommandedVelocity());
  EXPECT_NE(walkingPlanner.getProblem().costs.get(term::kTerminalDcm).describe().find("v_cmd / omega"), std::string::npos);

  const ContactPlannerInput input = makeWalkingInput();  // v_cmd = (0.4, 0)
  const OcpQpProblem qpRest = restPlanner.buildProblem(input);
  const OcpQpProblem qpWalking = walkingPlanner.buildProblem(input);
  const size_t last = static_cast<size_t>(rest.planner.numNodes - 1);
  const scalar_t omega = std::sqrt(rest.shared.gravity / rest.shared.comHeight);
  const scalar_t gain = std::exp(2.0 * omega * rest.planner.dt);
  const scalar_t w = rest.terminalDcm.weight;
  // Quadratic terms are identical, the linear ones differ by 2 w gain offset per coefficient, offset = -v_cmd / omega.
  EXPECT_TRUE(qpWalking.stages[last].Q.isApprox(qpRest.stages[last].Q));
  EXPECT_TRUE(qpWalking.stages[last].R.isApprox(qpRest.stages[last].R));
  EXPECT_TRUE(qpRest.stages[last].q.isZero());
  EXPECT_NEAR(qpWalking.stages[last].q(LIP_CX), -2.0 * w * gain * 0.4 / omega, 1e-9);
  EXPECT_NEAR(qpWalking.stages[last].q(LIP_VX), -2.0 * w * gain * 0.4 / omega / omega, 1e-9);
  EXPECT_NEAR(qpWalking.stages[last].r(LIP_ZX), 2.0 * w * gain * 0.4 / omega, 1e-9);
  EXPECT_NEAR(qpWalking.stages[last].q(LIP_CY), 0.0, 1e-12);
  for (size_t k = 0; k + 1 < last; ++k) EXPECT_TRUE(qpWalking.stages[k].q.isZero()) << "only the last running node";
}

// planner.logPlans prints one line per plan: search statistics, phase sequence with durations, step lengths.
TEST(ContactPlanningTerms, APlanDescribesItsPhasesAndSteps) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 2.0;
  plan.dt = 0.1;
  plan.objective = 1.5;
  plan.numBranchAndBoundNodes = 7;
  plan.solveTime = 0.012;
  plan.optimal = true;
  // Left: contact 0.2 s, swing 0.3 s (a 0.4 m step forward), contact 0.1 s. Right: contact throughout.
  plan.contacts = {{true, true}, {true, true}, {false, true}, {false, true}, {false, true}, {true, true}};
  plan.footholds.assign(7, {vector2_t(0.0, 0.1), vector2_t(0.0, -0.1)});
  for (size_t k = 5; k < 7; ++k) plan.footholds[k][0] = vector2_t(0.4, 0.1);
  plan.comVelocity = {vector2_t(0.3, 0.0), vector2_t(0.5, 0.0)};
  const std::string line = plan.describe();
  EXPECT_NE(line.find("plan t=2.000 valid J=1.500 relaxations=7 solve=12.000ms optimal"), std::string::npos) << line;
  EXPECT_NE(line.find("v0=[0.300 0.000] vN=[0.500 0.000]"), std::string::npos) << line;
  EXPECT_NE(line.find("| L: C0.200 S0.300(0.400,0.000) C0.100"), std::string::npos) << line;
  EXPECT_NE(line.find("| R: C0.600"), std::string::npos) << line;
  plan.valid = false;
  plan.timeLimitHit = true;
  plan.optimal = false;
  EXPECT_NE(plan.describe().find("INVALID"), std::string::npos);
  EXPECT_NE(plan.describe().find("TIME-LIMIT"), std::string::npos);
}

/*============================================ required blocks and names ==================================*/

TEST(ContactPlanningTerms, AHeadingTermWithoutTheHeadingBlockIsRejected) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs.push_back(term::kHeadingTracking);
  EXPECT_THROW(config.validate(), std::invalid_argument);
  // The problem checks it too, for a problem assembled by hand.
  ContactPlanningProblem problem;
  problem.model.add(term::kLipCom, ContactPlanningTermFactory::makeModelBlock(term::kLipCom));
  problem.model.add(term::kFootholdIntegrator, ContactPlanningTermFactory::makeModelBlock(term::kFootholdIntegrator));
  problem.costs.add(term::kHeadingTracking, ContactPlanningTermFactory::makeCost(term::kHeadingTracking));
  EXPECT_THROW(problem.finalize(makeConfig()), std::invalid_argument);
}

TEST(ContactPlanningTerms, TheFactoryNamesTheSupportedTermsInItsErrors) {
  try {
    ContactPlanningTermFactory::makeCost("gravity_compensation");
    FAIL() << "an unknown cost must throw";
  } catch (const std::invalid_argument& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("gravity_compensation"), std::string::npos);
    EXPECT_NE(what.find(term::kVelocityTracking), std::string::npos) << "the supported names are listed";
  }
  EXPECT_THROW(ContactPlanningTermFactory::makeExecutionRule(term::kPlannedHeadingOverride), std::invalid_argument)
      << "the heading override needs the reference manager's model";
  EXPECT_NO_THROW(ContactPlanningTermFactory::makeExecutionRule("phaseResetting"));
}

TEST(ContactPlanningTerms, EveryKnownTermCanBeBuiltAndDescribed) {
  const ContactPlanningConfig config = makeHeadingConfig();
  for (const std::string& name : knownTermNames(TermKind::COST)) {
    auto cost = ContactPlanningTermFactory::makeCost(name);
    cost->configure(config);
    EXPECT_FALSE(cost->describe().empty()) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::SOFT_CONSTRAINT)) {
    auto constraint = ContactPlanningTermFactory::makeSoftConstraint(name);
    constraint->configure(config);
    EXPECT_EQ(constraint->softness(), Softness::SOFT) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::HARD_CONSTRAINT)) {
    auto constraint = ContactPlanningTermFactory::makeHardConstraint(name);
    constraint->configure(config);
    EXPECT_EQ(constraint->softness(), Softness::HARD) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::LOGIC_RULE))
    EXPECT_NO_THROW(ContactPlanningTermFactory::makeLogicRule(name)) << name;
  for (const std::string& name : knownTermNames(TermKind::ASSIGNMENT_COST)) {
    EXPECT_NO_THROW(ContactPlanningTermFactory::makeAssignmentCost(name)) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::SEARCH_STAGE))
    EXPECT_NO_THROW(ContactPlanningTermFactory::makeSearchStage(name)) << name;
  const std::string summary = LipContactPlanner::formulationSummary(config);
  for (const std::string& name : config.formulation.costs) EXPECT_NE(summary.find(name), std::string::npos) << summary;
  EXPECT_NE(summary.find("x = [c_x c_y v_x v_y p_Lx p_Ly p_Rx p_Ry theta omega psi_L psi_R] (12)"), std::string::npos) << summary;
}

/*============================================ hot reload =================================================*/

TEST(ContactPlanningTerms, ParametersReloadByTermNameWithoutReassembly) {
  LipContactPlanner planner(makeConfig());
  const ContactPlannerInput input = makeWalkingInput();
  const OcpQpProblem before = planner.buildProblem(input);
  ContactPlanningConfig retuned = planner.getConfig();
  retuned.velocityTracking.weight *= 3.0;
  retuned.zmpSupportRegion.slack = SlackPenalty{123.0, 4.0};
  planner.setConfig(retuned);
  EXPECT_NEAR(planner.getProblem().costs.get<VelocityTrackingCost>(term::kVelocityTracking).weight(), retuned.velocityTracking.weight,
              1e-12);
  const OcpQpProblem after = planner.buildProblem(input);
  EXPECT_NEAR(after.stages.front().Q(LIP_VX, LIP_VX), 3.0 * before.stages.front().Q(LIP_VX, LIP_VX) - 2.0 * retuned.regularization.state,
              1e-9);
  // The ZMP rows are the first 12 soft rows of a running node; their penalty is the term's own, the rest keep the shared one.
  EXPECT_NEAR(after.stages.front().Zl(0), 123.0, 1e-12);
  EXPECT_NEAR(after.stages.front().zl(0), 4.0, 1e-12);
  EXPECT_NEAR(after.stages.front().Zl(12), retuned.shared.slackPenalty.quadratic, 1e-12);
  EXPECT_NEAR(before.stages.front().Zl(0), planner.getConfig().shared.slackPenalty.quadratic, 1e-12);
}

TEST(ContactPlanningTerms, ChangingATermListReassemblesTheProblemAndKeepsTheWarmStartWhileTheLayoutHolds) {
  LipContactPlanner planner(makeConfig());
  const ContactPlannerInput input = makeWalkingInput();
  const ContactPlan first = planner.plan(input);
  ASSERT_TRUE(first.valid);
  ContactPlannerInput next = input;
  next.time += 0.1;
  next.contacts = first.contactsAtTime(next.time);
  ASSERT_GE(planner.defaultNominal(next).com.size(), 1u);
  const bool warmBefore = planner.makeContext(next, planner.defaultNominal(next)).previousPlanShift >= 0;
  EXPECT_TRUE(warmBefore);

  ContactPlanningConfig fewerCosts = planner.getConfig();
  fewerCosts.formulation.setCost(term::kStepWidth, false);
  planner.setConfig(fewerCosts);
  EXPECT_FALSE(planner.getProblem().costs.has(term::kStepWidth));
  EXPECT_GE(planner.makeContext(next, planner.defaultNominal(next)).previousPlanShift, 0)
      << "same grid and layout: the warm start survives";

  ContactPlanningConfig heading = makeHeadingConfig();
  planner.setConfig(heading);
  EXPECT_TRUE(planner.getLayout().hasHeading);
  EXPECT_LT(planner.makeContext(next, planner.defaultNominal(next)).previousPlanShift, 0) << "a new layout drops the warm start";
}

TEST(ContactPlanningTerms, SearchStagesFollowTheList) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.search = {term::kWarmStartPreviousPlan};
  LipContactPlanner planner(config);
  EXPECT_EQ(planner.getSearchStages().size(), 1u);
  const ContactPlan plan = planner.plan(makeWalkingInput());
  EXPECT_TRUE(plan.valid);
  EXPECT_EQ(planner.getLastStatistics().numLocalSearchQps, 0) << "no local search stage";
  config.formulation.search = {term::kHeadingRelinearisation};
  EXPECT_THROW(LipContactPlanner{config}, std::invalid_argument) << "a stage that needs the heading block";
}

/*============================================ execution rules ============================================*/

TEST(ContactPlanningTerms, ExecutionRulesAreBuiltFromTheListAndReportWhatTheyNeed) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.execution = {term::kPhaseResetting, term::kEnergyCadenceModulation, term::kDcmStepAdjustment};
  config.validate();
  const TermCollection<ExecutionRule> rules = ContactPlanningTermFactory::buildExecutionRules(config);
  ASSERT_EQ(rules.size(), 3u);
  EXPECT_FALSE(rules.get(term::kPhaseResetting).needsPredictedTrajectory());
  EXPECT_TRUE(rules.get(term::kEnergyCadenceModulation).needsPredictedTrajectory());
  EXPECT_TRUE(rules.get(term::kDcmStepAdjustment).needsPredictedTrajectory());
  config.formulation.execution.push_back(term::kPlannedHeadingOverride);
  EXPECT_THROW(ContactPlanningTermFactory::buildExecutionRules(config), std::invalid_argument) << "needs the heading block and the model";
}

TEST(ContactPlanningTerms, WithoutRulesTheScheduleIsLeftAloneAndTheLatchTracksTheSwing) {
  // A swing of the left foot from 1.0 to 1.4 s, the right foot down throughout.
  ModeSchedule schedule({1.0, 1.4}, {ModeNumber::STANCE, ModeNumber::RF, ModeNumber::STANCE});
  feet_array_t<SwingTimingLatch> latches = makeFeetArray(SwingTimingLatch{});
  const ContactPlanningConfig config = makeConfig();
  const feet_array_t<scalar_t> noCadence = makeFeetArray(0.0);
  contact_flag_t measured = {true, true};  // the left foot reports contact mid-swing
  const auto reports = adaptScheduleToContactEvents(schedule, 1.2, measured, noCadence, config, latches);
  EXPECT_EQ(reports[0].type, ContactEventReport::Type::NONE) << "no phase_resetting rule listed";
  EXPECT_TRUE(latches[0].active);
  EXPECT_NEAR(latches[0].nominalTouchDownTime, 1.4, 1e-12);
  EXPECT_NEAR(schedule.eventTimes[1], 1.4, 1e-12);
  measured = {false, true};  // the foot is still in the air at its scheduled touch-down: nothing extends it either
  const auto late = adaptScheduleToContactEvents(schedule, 1.45, measured, noCadence, config, latches);
  EXPECT_EQ(late[0].type, ContactEventReport::Type::NONE);
  EXPECT_FALSE(latches[0].active) << "the contact phase proceeds and the latch is released";
}

}  // namespace ocs2::humanoid
