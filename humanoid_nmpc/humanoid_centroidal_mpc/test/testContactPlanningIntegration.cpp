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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <regex>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

/**
 * End-to-end test of the contact planning path on the DRC Atlas model: the interface builds the planning reference
 * manager and module from a task file with useContactPlanning: true, the module plans synchronously from the interface's
 * initial state, and the reference manager turns the plan into a mode schedule and swing-foot references.
 */
class ContactPlanningIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    const std::string taskFile = configDir + "/config/mpc/task.yaml";
    referenceFile_ = configDir + "/config/command/reference.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";

    // Temporary task file with contact planning on, planned synchronously so that the test controls the timing.
    std::ifstream in(taskFile);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    content = std::regex_replace(content, std::regex("useContactPlanning: *(true|false)"), "useContactPlanning: true");
    content = std::regex_replace(content, std::regex("runInBackgroundThread: *(true|false)"), "runInBackgroundThread: false");
    content = std::regex_replace(content, std::regex("maxSolveTime: *[0-9.]+"), "maxSolveTime: 5.0");
    content = std::regex_replace(content, std::regex("maxBranchAndBoundNodes: *[0-9]+"), "maxBranchAndBoundNodes: 2000");
    tmpTaskFile_ = testing::TempDir() + "/contact_planning_task.yaml";
    std::ofstream out(tmpTaskFile_);
    out << content;
    out.close();

    auto created = CentroidalMpcInterface::Create(tmpTaskFile_, urdfFile_, referenceFile_);
    ASSERT_TRUE(created.ok()) << created.status().message();
    interface_ = *std::move(created);
  }

  void TearDown() override { std::remove(tmpTaskFile_.c_str()); }

  std::string referenceFile_, urdfFile_, tmpTaskFile_;
  std::unique_ptr<CentroidalMpcInterface> interface_;
};

TEST_F(ContactPlanningIntegrationTest, PlansStandingAndWalkingSchedules) {
  ASSERT_TRUE(interface_->usesContactPlanning());
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);
  EXPECT_TRUE(referenceManager->usesContactPlanning());

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();

  // 1. No plan yet: the gait schedule (double support) is used.
  scalar_t t = 0.0;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  EXPECT_FALSE(referenceManager->hasActivePlan());
  EXPECT_TRUE(referenceManager->isInStancePhase(t + 0.5));
  EXPECT_FALSE(referenceManager->getSwingFootReference(0, t + 0.5).has_value());

  // 2. Standing command: the plan is all double support.
  vector_t standingTarget = vector_t::Zero(state.size());
  standingTarget.segment(6, 6) = state.segment(6, 6);
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {standingTarget}, {vector_t::Zero(inputDim)}));
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  const auto standingStats = module->getStatistics();
  EXPECT_TRUE(standingStats.lastPlanValid);
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager->hasActivePlan());
  for (scalar_t tau = t; tau < t + horizon; tau += 0.05) {
    EXPECT_TRUE(referenceManager->isInStancePhase(tau)) << "tau=" << tau;
  }

  // 3. Walking command: the plan contains single support phases and provides swing-foot references.
  vector_t walkingTarget = standingTarget;
  walkingTarget(0) = 0.4;  // commanded CoM velocity x
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);  // swaps in the target trajectories
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  const auto walkingStats = module->getStatistics();
  ASSERT_TRUE(walkingStats.lastPlanValid);
  std::cout << "walking plan: " << walkingStats.lastSolveTime * 1e3 << " ms, " << walkingStats.lastNumBranchAndBoundNodes
            << " relaxations, optimal=" << walkingStats.lastOptimal << std::endl;
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  const ModeSchedule& schedule = referenceManager->getModeSchedule();
  std::cout << "mode schedule: " << schedule;

  bool foundSwing = false;
  for (scalar_t tau = t; tau < t + horizon; tau += 0.02) {
    const contact_flag_t contacts = referenceManager->getContactFlags(tau);
    ASSERT_TRUE(contacts[0] || contacts[1]) << "no flight phase allowed at tau=" << tau;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const auto reference = referenceManager->getSwingFootReference(foot, tau);
      if (contacts[foot]) {
        EXPECT_FALSE(reference.has_value());
      } else {
        foundSwing = true;
        ASSERT_TRUE(reference.has_value()) << "swing foot " << foot << " at tau=" << tau << " has no reference";
        EXPECT_TRUE(reference->position.allFinite());
        EXPECT_TRUE(reference->linearVelocity.allFinite());
        EXPECT_GE(reference->position(2), -1e-6);  // never below the (flat) ground
      }
    }
  }
  EXPECT_TRUE(foundSwing) << "a walking command must produce a swing phase within the horizon";
  // The committed window right after the current time keeps the previous (double support) schedule.
  EXPECT_TRUE(referenceManager->isInStancePhase(t + 0.01));

  // 4. A swing that has started must survive a later plan: advance into the first swing, command standing (which on
  //    its own would plan no steps) and check that the swing keeps its touch-down time.
  scalar_t firstLiftOff = -1.0, firstTouchDown = -1.0;
  size_t swingFoot = 0;
  for (size_t i = 0; i < schedule.eventTimes.size() && firstLiftOff < 0.0; ++i) {
    const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[i]);
    const contact_flag_t after = modeNumber2StanceLeg(schedule.modeSequence[i + 1]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (before[foot] && !after[foot] && schedule.eventTimes[i] > t) {
        firstLiftOff = schedule.eventTimes[i];
        swingFoot = foot;
        for (size_t j = i + 1; j < schedule.eventTimes.size(); ++j) {
          if (modeNumber2StanceLeg(schedule.modeSequence[j + 1])[foot]) {
            firstTouchDown = schedule.eventTimes[j];
            break;
          }
        }
      }
    }
  }
  ASSERT_GT(firstLiftOff, 0.0);
  ASSERT_GT(firstTouchDown, firstLiftOff);
  const scalar_t midSwing = 0.5 * (firstLiftOff + firstTouchDown);
  referenceManager->setTargetTrajectories(TargetTrajectories({midSwing}, {standingTarget}, {vector_t::Zero(inputDim)}));
  // The measured mode reports the swing foot in flight; reporting it in contact would (correctly) trigger an early
  // touch-down, see AdaptsScheduleToContactEventsAndDcmError.
  contact_flag_t inFlight = makeFeetArray(true);
  inFlight[swingFoot] = false;
  referenceManager->preSolverRun(midSwing, midSwing + horizon, state, stanceLeg2ModeNumber(inFlight));
  EXPECT_GE(referenceManager->commitBoundary(midSwing), firstTouchDown - 1e-9);
  module->preSolverRun(midSwing, midSwing + horizon, state, *referenceManager);
  ASSERT_TRUE(module->getStatistics().lastPlanValid);
  referenceManager->preSolverRun(midSwing + 0.02, midSwing + 0.02 + horizon, state, stanceLeg2ModeNumber(inFlight));
  EXPECT_FALSE(referenceManager->isInContact(midSwing + 0.02, swingFoot));
  EXPECT_FALSE(referenceManager->isInContact(firstTouchDown - 0.01, swingFoot)) << "the in-flight swing was cut short";
  EXPECT_TRUE(referenceManager->isInContact(firstTouchDown + 0.01, swingFoot)) << "the in-flight swing was extended";
  EXPECT_TRUE(referenceManager->getSwingFootReference(swingFoot, midSwing + 0.02).has_value());
}

namespace {

/** Mode number in which every foot but `swingFoot` is in contact. */
size_t modeWithSwingFoot(size_t swingFoot) {
  contact_flag_t flags = makeFeetArray(true);
  flags[swingFoot] = false;
  return stanceLeg2ModeNumber(flags);
}

struct Swing {
  size_t foot = 0;
  scalar_t liftOff = -1.0;
  scalar_t touchDown = -1.0;
};

/** First swing of the schedule that lifts off after `after`. */
std::optional<Swing> firstSwingAfter(const ModeSchedule& schedule, scalar_t after) {
  for (size_t i = 0; i < schedule.eventTimes.size(); ++i) {
    const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[i]);
    const contact_flag_t afterFlags = modeNumber2StanceLeg(schedule.modeSequence[i + 1]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (before[foot] && !afterFlags[foot] && schedule.eventTimes[i] > after) {
        Swing swing;
        swing.foot = foot;
        swing.liftOff = schedule.eventTimes[i];
        for (size_t j = i + 1; j < schedule.eventTimes.size(); ++j) {
          if (modeNumber2StanceLeg(schedule.modeSequence[j + 1])[foot]) {
            swing.touchDown = schedule.eventTimes[j];
            return swing;
          }
        }
      }
    }
  }
  return std::nullopt;
}

}  // namespace

/**
 * The shipped configuration has to reproduce plain plan merging. The adaptive execution features change the closed
 * loop, so they are opt-in; if one of them were enabled by default it would silently alter the gait every robot is
 * tuned against, and the symptom would be a walking regression rather than a failing unit test, because the other
 * tests set the flags explicitly.
 */
TEST_F(ContactPlanningIntegrationTest, DefaultConfigurationStepsInTheCommandedDirection) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);

  const ContactPlanningConfig config = referenceManager->getConfig();
  EXPECT_FALSE(config.enablePhaseResetting) << "adaptive execution must be opt-in in the shipped task file";
  EXPECT_FALSE(config.enableDcmStepAdjustment) << "adaptive execution must be opt-in in the shipped task file";
  EXPECT_FALSE(config.enableEnergyCadenceModulation) << "adaptive execution must be opt-in in the shipped task file";

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();

  // Command a forward walk.
  const scalar_t commandedVelocityX = 0.4;
  vector_t walkingTarget = vector_t::Zero(state.size());
  walkingTarget.segment(6, 6) = state.segment(6, 6);
  walkingTarget(0) = commandedVelocityX;

  scalar_t t = 0.0;
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  ASSERT_TRUE(module->getStatistics().lastPlanValid);
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager->hasActivePlan());

  const auto swing = firstSwingAfter(referenceManager->getModeSchedule(), t);
  ASSERT_TRUE(swing.has_value());
  const size_t foot = swing->foot;
  const size_t inFlightMode = modeWithSwingFoot(foot);

  // The swing foot has to travel forward, which is the symptom a saturating step-adjustment feedback destroys.
  const scalar_t justAfterLiftOff = swing->liftOff + 1e-3;
  const scalar_t justBeforeTouchDown = swing->touchDown - 1e-3;
  referenceManager->preSolverRun(justAfterLiftOff, justAfterLiftOff + horizon, state, inFlightMode);
  const auto atLiftOff = referenceManager->getSwingFootReference(foot, justAfterLiftOff);
  const auto atTouchDown = referenceManager->getSwingFootReference(foot, justBeforeTouchDown);
  ASSERT_TRUE(atLiftOff.has_value());
  ASSERT_TRUE(atTouchDown.has_value());
  EXPECT_GT(atTouchDown->position(0) - atLiftOff->position(0), 0.0)
      << "a forward velocity command must move the swing foot forward, not backward";

  // No correction is applied to the planned landing target.
  EXPECT_TRUE(referenceManager->getDcmStepAdjustment()[foot].isZero())
      << "the step adjustment must be inactive in the default configuration";

  // A measured contact in mid-swing leaves the executed schedule untouched.
  const scalar_t midSwing = 0.5 * (swing->liftOff + swing->touchDown);
  const ModeSchedule beforeEvent = referenceManager->getModeSchedule();
  referenceManager->preSolverRun(midSwing, midSwing + horizon, state, ModeNumber::STANCE);
  EXPECT_EQ(referenceManager->getLastContactEvents()[foot].type, ContactEventReport::Type::NONE)
      << "phase resetting must be inactive in the default configuration";
  EXPECT_FALSE(referenceManager->isInContact(midSwing, foot)) << "the swing must run to its scheduled touch-down";
  EXPECT_FALSE(referenceManager->consumeReplanRequest());
  const ModeSchedule afterEvent = referenceManager->getModeSchedule();
  ASSERT_EQ(afterEvent.eventTimes.size(), beforeEvent.eventTimes.size());
  for (size_t i = 0; i < beforeEvent.eventTimes.size(); ++i) {
    EXPECT_NEAR(afterEvent.eventTimes[i], beforeEvent.eventTimes[i], 1e-9) << "event " << i << " was re-timed";
  }
}

/**
 * Adaptive execution on the full model: measured contact events re-time the executed schedule, and the DCM error with
 * respect to the plan moves the landing target of the swing foot.
 */
TEST_F(ContactPlanningIntegrationTest, AdaptsScheduleToContactEventsAndDcmError) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);
  ContactPlanningConfig config = referenceManager->getConfig();
  config.enablePhaseResetting = true;
  config.enableDcmStepAdjustment = true;
  config.enableEnergyCadenceModulation = false;
  config.earlyTouchdownMinSwingRatio = 0.25;
  config.maxLateTouchdownExtension = 0.15;
  config.lateTouchdownExtensionStep = 0.05;
  module->setConfig(config);

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  const auto solveAt = [&](scalar_t t, const vector_t& x, size_t measuredMode) {
    referenceManager->preSolverRun(t, t + horizon, x, measuredMode);
  };
  const auto planAt = [&](scalar_t t, const vector_t& x) {
    module->preSolverRun(t, t + horizon, x, *referenceManager);
    ASSERT_TRUE(module->getStatistics().lastPlanValid);
  };

  // Walk: a plan with a swing phase.
  vector_t walkingTarget = vector_t::Zero(state.size());
  walkingTarget.segment(6, 6) = state.segment(6, 6);
  walkingTarget(0) = 0.4;
  scalar_t t = 0.0;
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  solveAt(t, state, ModeNumber::STANCE);
  planAt(t, state);
  t += 0.02;
  solveAt(t, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager->hasActivePlan());
  const auto swing = firstSwingAfter(referenceManager->getModeSchedule(), t);
  ASSERT_TRUE(swing.has_value());
  const size_t foot = swing->foot;
  const size_t inFlightMode = modeWithSwingFoot(foot);

  // ---- 1. DCM step adjustment: a forward CoM velocity error moves the landing target forward, bounded. ----
  // The test state is frozen while the plan's LIP walks on, so the nominal DCM error is not small. The offset bound and
  // the reach limits are widened for the comparison so that neither adjustment saturates; the bound is checked separately.
  ContactPlanningConfig wide = config;
  wide.dcmAdjustmentMaxOffset = 5.0;
  wide.reachX = 5.0;
  wide.reachYOuter = 5.0;
  module->setConfig(wide);
  const scalar_t midSwing = 0.5 * (swing->liftOff + swing->touchDown);
  solveAt(midSwing, state, inFlightMode);
  ASSERT_EQ(referenceManager->getLastContactEvents()[foot].type, ContactEventReport::Type::NONE);
  const vector2_t nominalAdjustment = referenceManager->getDcmStepAdjustment()[foot];
  const auto nominalReference = referenceManager->getSwingFootReference(foot, swing->touchDown - 1e-3);
  ASSERT_TRUE(nominalReference.has_value());

  vector_t pushed = state;
  pushed(0) += 0.3;  // normalized linear momentum x = CoM velocity x
  solveAt(midSwing, pushed, inFlightMode);
  const vector2_t pushedAdjustment = referenceManager->getDcmStepAdjustment()[foot];
  const auto pushedReference = referenceManager->getSwingFootReference(foot, swing->touchDown - 1e-3);
  ASSERT_TRUE(pushedReference.has_value());
  const scalar_t omega = config.omega();
  // The measured CoM velocity error maps to a DCM error of dv / omega, which the closed-form law propagates to
  // touch-down and scales by the gain.
  const scalar_t expectedShift = wide.dcmAdjustmentGain * 0.3 / omega * std::exp(omega * (swing->touchDown - midSwing));
  EXPECT_NEAR(pushedAdjustment(0) - nominalAdjustment(0), expectedShift, 1e-6) << "closed-form LIP propagation of the DCM error";
  EXPECT_NEAR(pushedAdjustment(1) - nominalAdjustment(1), 0.0, 1e-6);
  EXPECT_NEAR((pushedReference->position - nominalReference->position).head<2>().norm(), (pushedAdjustment - nominalAdjustment).norm(),
              2e-3)
      << "close to touch-down the reference carries the full adjustment";
  EXPECT_TRUE(pushedReference->position.allFinite());
  EXPECT_TRUE(pushedReference->linearVelocity.allFinite());

  // With the configured bound the offset saturates but stays bounded.
  module->setConfig(config);
  solveAt(midSwing, pushed, inFlightMode);
  const vector2_t boundedAdjustment = referenceManager->getDcmStepAdjustment()[foot];
  EXPECT_LE(boundedAdjustment.norm(), config.dcmAdjustmentMaxOffset + 1e-9);
  EXPECT_GT(boundedAdjustment.norm(), 0.0);

  ContactPlanningConfig noDcm = config;
  noDcm.enableDcmStepAdjustment = false;
  module->setConfig(noDcm);
  solveAt(midSwing, pushed, inFlightMode);
  EXPECT_TRUE(referenceManager->getDcmStepAdjustment()[foot].isZero());
  module->setConfig(config);

  // ---- 2. Early touch-down: contact measured mid-swing switches the foot to contact at once, in place. ----
  const ModeSchedule beforeEarly = referenceManager->getModeSchedule();
  solveAt(midSwing, state, ModeNumber::STANCE);
  EXPECT_EQ(referenceManager->getLastContactEvents()[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
  EXPECT_TRUE(referenceManager->isInContact(midSwing + 1e-3, foot));
  EXPECT_FALSE(referenceManager->isInContact(midSwing - 1e-3, foot));
  EXPECT_FALSE(referenceManager->getSwingFootReference(foot, midSwing + 1e-3).has_value());
  EXPECT_TRUE(referenceManager->consumeReplanRequest());
  EXPECT_FALSE(referenceManager->consumeReplanRequest()) << "the request is consumed once";
  // Later events keep their timing: every event of the old schedule after the old touch-down is still there.
  const ModeSchedule afterEarly = referenceManager->getModeSchedule();
  for (scalar_t event : beforeEarly.eventTimes) {
    if (event <= swing->touchDown + 1e-9) continue;
    bool found = false;
    for (scalar_t e : afterEarly.eventTimes) found = found || std::abs(e - event) < 1e-9;
    EXPECT_TRUE(found) << "event at " << event << " moved";
  }
  for (scalar_t tau = midSwing; tau < midSwing + horizon; tau += 0.02) {
    const contact_flag_t contacts = referenceManager->getContactFlags(tau);
    bool any = false;
    for (size_t i = 0; i < N_CONTACTS; ++i) any = any || contacts[i];
    ASSERT_TRUE(any) << "no flight phase at tau=" << tau;
  }
  // The planner input right after the event sees the foot in contact with a fresh phase.
  const ContactPlannerInput input = referenceManager->makePlannerInput(midSwing, state, vector2_t(0.4, 0.0));
  EXPECT_TRUE(input.contacts[foot]);
  EXPECT_NEAR(input.phaseElapsedTime[foot], 0.0, 1e-9);

  // ---- 3. Late touch-down: a foot that misses the ground keeps swinging in steps, up to the extension budget. ----
  // The DCM adjustment is switched off here so that the xy reference depends on the schedule only.
  module->setConfig(noDcm);
  planAt(midSwing, state);  // re-plan from the new stance state
  scalar_t t2 = midSwing + 0.02;
  solveAt(t2, state, ModeNumber::STANCE);
  const auto next = firstSwingAfter(referenceManager->getModeSchedule(), t2);
  ASSERT_TRUE(next.has_value());
  const size_t foot2 = next->foot;
  const size_t inFlight2 = modeWithSwingFoot(foot2);
  const scalar_t mid2 = 0.5 * (next->liftOff + next->touchDown);
  solveAt(mid2, state, inFlight2);
  ASSERT_TRUE(referenceManager->getSwingTimingLatches()[foot2].active);
  const auto referenceBeforeExtension = referenceManager->getSwingFootReference(foot2, next->touchDown - 1e-3);
  ASSERT_TRUE(referenceBeforeExtension.has_value());

  const scalar_t late1 = next->touchDown + 0.005;
  solveAt(late1, state, inFlight2);
  EXPECT_EQ(referenceManager->getLastContactEvents()[foot2].type, ContactEventReport::Type::LATE_TOUCH_DOWN);
  EXPECT_NEAR(referenceManager->getLastContactEvents()[foot2].touchDownTime, late1 + config.lateTouchdownExtensionStep, 1e-9);
  EXPECT_FALSE(referenceManager->isInContact(late1 + 1e-3, foot2));
  EXPECT_TRUE(referenceManager->isInContact(late1 + config.lateTouchdownExtensionStep + 1e-3, foot2));
  EXPECT_TRUE(referenceManager->consumeReplanRequest());
  const auto extendedReference = referenceManager->getSwingFootReference(foot2, late1 + 0.02);
  ASSERT_TRUE(extendedReference.has_value());
  EXPECT_NEAR((extendedReference->position - referenceBeforeExtension->position).head<2>().norm(), 0.0, 2e-3)
      << "the xy target is held while the foot searches for the ground";
  const scalar_t extendedTouchDown = referenceManager->getLastContactEvents()[foot2].touchDownTime;
  const auto atExtendedTouchDown = referenceManager->getSwingFootReference(foot2, extendedTouchDown - 1e-4);
  ASSERT_TRUE(atExtendedTouchDown.has_value());
  const scalar_t extension = referenceManager->getSwingTimingLatches()[foot2].lateExtension;
  EXPECT_GT(extension, 0.0);
  EXPECT_LT(atExtendedTouchDown->position(2), referenceBeforeExtension->position(2) - 0.5 * config.lateTouchdownSearchVelocity * extension)
      << "the height target at the extended touch-down descends at the search velocity";
  EXPECT_TRUE(referenceManager->getActiveContactPlan()->startTime > 0.0);

  // Keep missing the ground: the total extension is capped, then the contact phase proceeds.
  scalar_t lastTouchDown = late1 + config.lateTouchdownExtensionStep;
  for (int i = 0; i < 6; ++i) {
    const scalar_t tl = lastTouchDown + 0.005;
    solveAt(tl, state, inFlight2);
    const ContactEventReport report = referenceManager->getLastContactEvents()[foot2];
    if (report.type == ContactEventReport::Type::NONE) break;
    ASSERT_EQ(report.type, ContactEventReport::Type::LATE_TOUCH_DOWN);
    lastTouchDown = report.touchDownTime;
  }
  EXPECT_LE(lastTouchDown, next->touchDown + config.maxLateTouchdownExtension + 1e-9);
  EXPECT_GT(lastTouchDown, next->touchDown + config.maxLateTouchdownExtension - config.lateTouchdownExtensionStep);
  EXPECT_TRUE(referenceManager->isInContact(lastTouchDown + 0.005 + 1e-3, foot2));
  EXPECT_FALSE(referenceManager->getSwingTimingLatches()[foot2].active);
  for (scalar_t tau = lastTouchDown; tau < lastTouchDown + horizon; tau += 0.02) {
    const contact_flag_t contacts = referenceManager->getContactFlags(tau);
    bool any = false;
    for (size_t i = 0; i < N_CONTACTS; ++i) any = any || contacts[i];
    ASSERT_TRUE(any) << "no flight phase at tau=" << tau;
  }

  module->setConfig(config);

  // ---- 4. Cadence modulation: extra forward energy brings the touch-down of the swing in flight forward. ----
  planAt(lastTouchDown + 0.01, state);
  const scalar_t t3 = lastTouchDown + 0.03;
  solveAt(t3, state, ModeNumber::STANCE);
  const auto third = firstSwingAfter(referenceManager->getModeSchedule(), t3);
  ASSERT_TRUE(third.has_value());
  ContactPlanningConfig cadence = config;
  cadence.enableEnergyCadenceModulation = true;
  cadence.energyCadenceGain = 0.01;
  module->setConfig(cadence);
  const scalar_t early3 = third->liftOff + 0.05;
  vector_t fast = state;
  fast(0) += 1.0;  // a velocity error that dominates the position term of the orbital energy
  solveAt(early3, fast, modeWithSwingFoot(third->foot));
  const ContactEventReport cadenceReport = referenceManager->getLastContactEvents()[third->foot];
  EXPECT_LT(referenceManager->getCadenceTouchDownShift()[third->foot], 0.0) << "more energy than planned shortens the swing";
  if (third->touchDown - third->liftOff > cadence.minSwingDuration + 1e-6) {
    EXPECT_EQ(cadenceReport.type, ContactEventReport::Type::CADENCE_SHIFT);
    EXPECT_LT(cadenceReport.touchDownTime, third->touchDown);
    EXPECT_GE(cadenceReport.touchDownTime, third->liftOff + cadence.minSwingDuration - 1e-9);
    EXPECT_GE(cadenceReport.touchDownTime, early3 + 0.02 - 1e-9);
    EXPECT_TRUE(referenceManager->isInContact(cadenceReport.touchDownTime + 1e-3, third->foot));
  } else {
    EXPECT_EQ(cadenceReport.type, ContactEventReport::Type::NONE) << "a swing at its minimum duration cannot be shortened";
  }
  module->setConfig(config);
}

}  // namespace ocs2::humanoid
