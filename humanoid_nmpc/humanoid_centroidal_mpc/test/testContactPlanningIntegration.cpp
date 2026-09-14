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

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <thread>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include <pinocchio/algorithm/center-of-mass.hpp>

#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"

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

  // Sampled between the possible event instants: the plan's events lie on its node grid, and at an event instant the
  // contact flags (ocs2 mode lookup, the event has not passed) and the swing queries of this manager (the event has
  // passed) disagree by design.
  bool foundSwing = false;
  for (scalar_t tau = t + 0.01; tau < t + horizon; tau += 0.02) {
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
        // Never below the flat ground, except for the configured touch-down offset (a slightly negative offset presses
        // the foot onto the ground at the end of the swing).
        EXPECT_GE(reference->position(2),
                  std::min(0.0, referenceManager->getSwingTrajectoryPlanner()->getConfig().touchDownHeightOffset) - 1e-6);
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
  config.earlyTouchdownMinContactDuration = 0.02;
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

  // ---- 1. DCM step adjustment, measured against the NMPC's own prediction. ----
  // A prediction equal to the measured state means the robot does exactly what the controller expects: no correction,
  // however far the planner's reduced model has walked on in the meantime. This is what keeps the loop from fighting
  // the planner and pulling the foot backwards.
  const scalar_t midSwing = 0.5 * (swing->liftOff + swing->touchDown);
  referenceManager->setPredictedTrajectory({0.0, midSwing + 10.0}, {state, state});
  solveAt(midSwing, state, inFlightMode);
  ASSERT_EQ(referenceManager->getLastContactEvents()[foot].type, ContactEventReport::Type::NONE);
  ASSERT_TRUE(referenceManager->hasPredictedComState());
  EXPECT_TRUE(referenceManager->getDcmStepAdjustment()[foot].isZero(1e-12)) << "no deviation from the prediction, no correction";
  const auto nominalReference = referenceManager->getSwingFootReference(foot, swing->touchDown - 1e-3);
  ASSERT_TRUE(nominalReference.has_value());
  const auto liftOffReference = referenceManager->getSwingFootReference(foot, swing->liftOff + 1e-3);
  ASSERT_TRUE(liftOffReference.has_value());
  EXPECT_GT(nominalReference->position(0) - liftOffReference->position(0), 0.0) << "the foot still steps forward with the loop on";

  // A small forward velocity error relative to the prediction is corrected by the closed-form law, below the bound.
  const scalar_t omega = config.omega();
  const scalar_t velocityError = 0.05;
  vector_t pushed = state;
  pushed(0) += velocityError;  // normalized linear momentum x = CoM velocity x
  solveAt(midSwing, pushed, inFlightMode);
  const vector2_t pushedAdjustment = referenceManager->getDcmStepAdjustment()[foot];
  const scalar_t expectedShift = config.dcmAdjustmentGain * velocityError / omega * std::exp(omega * (swing->touchDown - midSwing));
  ASSERT_LT(expectedShift, config.dcmAdjustmentMaxOffset) << "the test push must stay below the bound";
  EXPECT_NEAR(pushedAdjustment(0), expectedShift, 1e-6) << "closed-form LIP propagation of the DCM error";
  EXPECT_NEAR(pushedAdjustment(1), 0.0, 1e-6);
  const auto pushedReference = referenceManager->getSwingFootReference(foot, swing->touchDown - 1e-3);
  ASSERT_TRUE(pushedReference.has_value());
  EXPECT_NEAR((pushedReference->position - nominalReference->position).head<2>().norm(), pushedAdjustment.norm(), 2e-3)
      << "close to touch-down the reference carries the full adjustment";
  EXPECT_TRUE(pushedReference->position.allFinite());
  EXPECT_TRUE(pushedReference->linearVelocity.allFinite());

  // A large push saturates at the configured bound.
  vector_t shoved = state;
  shoved(0) += 1.0;
  solveAt(midSwing, shoved, inFlightMode);
  const vector2_t boundedAdjustment = referenceManager->getDcmStepAdjustment()[foot];
  EXPECT_LE(boundedAdjustment.norm(), config.dcmAdjustmentMaxOffset + 1e-9);
  EXPECT_GT(boundedAdjustment.norm(), 0.9 * config.dcmAdjustmentMaxOffset);

  // Without a prediction there is nothing to measure against and no correction is applied.
  referenceManager->setPredictedTrajectory({}, {});
  solveAt(midSwing, shoved, inFlightMode);
  EXPECT_FALSE(referenceManager->hasPredictedComState());
  EXPECT_TRUE(referenceManager->getDcmStepAdjustment()[foot].isZero());

  // A diverged solve hands over a non-finite trajectory: it must be ignored rather than poison the foot reference.
  vector_t poisoned = state;
  poisoned(0) = std::numeric_limits<scalar_t>::quiet_NaN();
  referenceManager->setPredictedTrajectory({0.0, midSwing + 10.0}, {poisoned, poisoned});
  solveAt(midSwing, shoved, inFlightMode);
  EXPECT_FALSE(referenceManager->hasPredictedComState());
  EXPECT_TRUE(referenceManager->getDcmStepAdjustment()[foot].isZero());
  EXPECT_TRUE(referenceManager->getSwingFootReference(foot, swing->touchDown - 1e-3)->position.allFinite());

  // The module hands the primal solution over after every solve, but only while a correction that needs it is enabled.
  PrimalSolution primal;
  primal.timeTrajectory_ = {0.0, midSwing + 10.0};
  primal.stateTrajectory_ = {state, state};
  module->postSolverRun(primal);
  solveAt(midSwing, shoved, inFlightMode);
  EXPECT_TRUE(referenceManager->hasPredictedComState()) << "the prediction must reach the reference manager through the module";
  EXPECT_GT(referenceManager->getDcmStepAdjustment()[foot].norm(), 0.0);
  referenceManager->setPredictedTrajectory({}, {});
  ContactPlanningConfig nothingEnabled = config;
  nothingEnabled.enableDcmStepAdjustment = false;
  nothingEnabled.enableEnergyCadenceModulation = false;
  module->setConfig(nothingEnabled);
  module->postSolverRun(primal);  // skipped: no consumer
  module->setConfig(config);
  solveAt(midSwing, shoved, inFlightMode);
  EXPECT_FALSE(referenceManager->hasPredictedComState()) << "no hand-over while every consumer is disabled";
  referenceManager->setPredictedTrajectory({0.0, midSwing + 10.0}, {state, state});

  ContactPlanningConfig noDcm = config;
  noDcm.enableDcmStepAdjustment = false;
  module->setConfig(noDcm);
  solveAt(midSwing, shoved, inFlightMode);
  EXPECT_TRUE(referenceManager->getDcmStepAdjustment()[foot].isZero());
  module->setConfig(config);

  // ---- 2. Early touch-down: contact that persists for the debounce duration switches the foot to contact, in place. ----
  const ModeSchedule beforeEarly = referenceManager->getModeSchedule();
  solveAt(midSwing, state, ModeNumber::STANCE);  // first contact sample: the debounce timer starts
  EXPECT_EQ(referenceManager->getLastContactEvents()[foot].type, ContactEventReport::Type::NONE)
      << "a single contact sample must not end the swing";
  EXPECT_FALSE(referenceManager->isInContact(midSwing + 1e-3, foot));
  EXPECT_FALSE(referenceManager->consumeReplanRequest());
  const scalar_t landed = midSwing + config.earlyTouchdownMinContactDuration;
  solveAt(landed, state, ModeNumber::STANCE);  // contact has persisted: the swing ends now
  EXPECT_EQ(referenceManager->getLastContactEvents()[foot].type, ContactEventReport::Type::EARLY_TOUCH_DOWN);
  EXPECT_TRUE(referenceManager->isInContact(landed + 1e-3, foot));
  EXPECT_FALSE(referenceManager->isInContact(landed - 1e-3, foot));
  EXPECT_FALSE(referenceManager->getSwingFootReference(foot, landed + 1e-3).has_value());
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
  for (scalar_t tau = landed; tau < landed + horizon; tau += 0.02) {
    const contact_flag_t contacts = referenceManager->getContactFlags(tau);
    bool any = false;
    for (size_t i = 0; i < N_CONTACTS; ++i) any = any || contacts[i];
    ASSERT_TRUE(any) << "no flight phase at tau=" << tau;
  }
  // The planner input right after the event sees the foot in contact with a fresh phase.
  const ContactPlannerInput input = referenceManager->makePlannerInput(landed, state, vector2_t(0.4, 0.0));
  EXPECT_TRUE(input.contacts[foot]);
  EXPECT_NEAR(input.phaseElapsedTime[foot], 0.0, 1e-9);

  // ---- 3. Late touch-down: a foot that misses the ground keeps swinging in steps, up to the extension budget. ----
  // The DCM adjustment is switched off here so that the xy reference depends on the schedule only.
  module->setConfig(noDcm);
  planAt(landed, state);  // re-plan from the new stance state
  scalar_t t2 = landed + 0.02;
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

  // ---- 4. Cadence modulation: more forward energy than the NMPC predicted brings the touch-down forward. ----
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
  referenceManager->setPredictedTrajectory({0.0, early3 + 10.0}, {state, state});

  // Prediction equal to the measurement: the planned timing stands. (A shift reported here with a zero request means
  // the executed swing violated the duration limits and the clamp repaired it, i.e. the merge cut the swing.)
  solveAt(early3, state, modeWithSwingFoot(third->foot));
  EXPECT_NEAR(referenceManager->getCadenceTouchDownShift()[third->foot], 0.0, 1e-12);
  {
    const ContactEventReport report = referenceManager->getLastContactEvents()[third->foot];
    EXPECT_EQ(report.type, ContactEventReport::Type::NONE)
        << "swing [" << third->liftOff << ", " << third->touchDown << ") of duration " << third->touchDown - third->liftOff
        << " s was re-timed to " << report.touchDownTime << " (shift " << report.timeShift << ")";
  }

  // Extra forward energy relative to the prediction shortens the swing, within the duration limits. With the same
  // position in both, only the velocity term of the orbital energy differs, so the sign is unambiguous.
  vector_t fast = state;
  fast(0) += 1.0;
  solveAt(early3, fast, modeWithSwingFoot(third->foot));
  const ContactEventReport cadenceReport = referenceManager->getLastContactEvents()[third->foot];
  EXPECT_LT(referenceManager->getCadenceTouchDownShift()[third->foot], 0.0) << "more energy than predicted shortens the swing";
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

/**
 * The xy swing reference of a later swing of the same foot inside the horizon starts from where that foot will stand
 * after its earlier step, not from the position latched while it stands now. With a plan long enough to hold two swings
 * of one foot the two differ by a step length.
 */
TEST_F(ContactPlanningIntegrationTest, LaterSwingOfTheSameFootStartsFromItsPlannedStance) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);

  ContactPlanningConfig config = module->getConfig();
  config.numNodes = 24;             // 2.4 s plan
  config.maxContactDuration = 0.6;  // keep stepping, so that the plan holds two swings of one foot
  module->setConfig(config);

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  vector_t walkingTarget = vector_t::Zero(state.size());
  walkingTarget.segment(6, 6) = state.segment(6, 6);
  walkingTarget(0) = 0.4;
  scalar_t t = 0.0;
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  // The schedule has to cover the long plan: the reference manager keeps the merged schedule over [t - H, t + 2H].
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  ASSERT_TRUE(module->getStatistics().lastPlanValid);
  t += 0.02;
  referenceManager->preSolverRun(t, t + 1.4, state, ModeNumber::STANCE);  // a 1.4 s horizon keeps [t - 1.4, t + 2.8]
  ASSERT_TRUE(referenceManager->hasActivePlan());
  const ModeSchedule& schedule = referenceManager->getModeSchedule();

  // Two swings of the same foot.
  std::optional<Swing> first, second;
  for (size_t i = 0; i + 1 < schedule.modeSequence.size(); ++i) {
    const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[i]);
    const contact_flag_t after = modeNumber2StanceLeg(schedule.modeSequence[i + 1]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!(before[foot] && !after[foot]) || schedule.eventTimes[i] <= t) continue;
      Swing swing;
      swing.foot = foot;
      swing.liftOff = schedule.eventTimes[i];
      for (size_t j = i + 1; j < schedule.eventTimes.size(); ++j) {
        if (modeNumber2StanceLeg(schedule.modeSequence[j + 1])[foot]) {
          swing.touchDown = schedule.eventTimes[j];
          break;
        }
      }
      if (swing.touchDown < 0.0) continue;
      if (!first.has_value()) {
        first = swing;
      } else if (!second.has_value() && swing.foot == first->foot) {
        second = swing;
      }
    }
  }
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value()) << "the 2.4 s plan should contain two swings of the same foot";
  const size_t foot = first->foot;
  const auto& plan = *referenceManager->getActiveContactPlan();

  // The first swing ends the foot's current contact phase: it starts from the latched (measured) position, which is also
  // where the plan has the foot standing.
  const auto firstReference = referenceManager->getSwingFootReference(foot, first->liftOff + 1e-4);
  ASSERT_TRUE(firstReference.has_value());
  const vector2_t measuredStance = *plan.footholdAtTime(foot, t);
  EXPECT_LT((firstReference->position.head<2>() - measuredStance).norm(), 2e-3);

  // The second swing starts from the planned landing spot of the first, not from where the foot stands now.
  const auto secondReference = referenceManager->getSwingFootReference(foot, second->liftOff + 1e-4);
  ASSERT_TRUE(secondReference.has_value());
  const vector2_t plannedStanceBeforeSecond = *plan.footholdAtTime(foot, second->liftOff - 0.5 * plan.dt + 1e-9);
  const vector2_t landingOfFirst = *plan.footholdAtTime(foot, first->touchDown);
  EXPECT_LT((plannedStanceBeforeSecond - landingOfFirst).norm(), 1e-9) << "the foot does not move between its steps";
  EXPECT_LT((secondReference->position.head<2>() - plannedStanceBeforeSecond).norm(), 2e-3)
      << "reference " << secondReference->position.head<2>().transpose() << " vs planned stance " << plannedStanceBeforeSecond.transpose();
  EXPECT_GT((plannedStanceBeforeSecond - measuredStance).norm(), 0.05)
      << "the two starts must differ by a step for this test to mean anything";

  module->setConfig(config);
}

/**
 * The planner's ground and body limits left at 0 in the task file are derived from the model and from the wrench cone,
 * so that the planner and the whole-body constraint agree on friction, torque and hip range.
 */
TEST_F(ContactPlanningIntegrationTest, DerivesHeadingModelParametersFromTheModel) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  const ContactPlanningConfig config = module->getConfig();
  const auto& model = interface_->getPinocchioInterface().getModel();
  const scalar_t weight = pinocchio::computeTotalMass(model) * config.gravity;

  boost::property_tree::ptree pt;
  loadData::readPropertyTree(tmpTaskFile_, pt);
  scalar_t mu = 0.0, muTorsion = 0.0;
  loadData::loadPtreeValue(pt, mu, "contacts.contactWrenchConeSoftConstraint.frictionCoefficient", false);
  loadData::loadPtreeValue(pt, muTorsion, "contacts.contactWrenchConeSoftConstraint.torsionalFrictionCoefficient", false);
  ASSERT_GT(mu, 0.0);
  EXPECT_TRUE(config.hasModelParameters());
  EXPECT_NEAR(config.torsionalFrictionTorque, muTorsion * weight, 1e-9) << "torsional friction times the weight";
  EXPECT_NEAR(config.doubleSupportYawCouple, mu * 0.5 * weight * config.nominalStepWidth, 1e-9) << "friction couple of two feet";
  EXPECT_GT(config.torsionalFrictionTorque, 0.0);

  // Hip yaw limits per leg: l_leg_hpz [-0.174, 0.787], r_leg_hpz [-0.787, 0.174] on the DRC Atlas.
  const auto [leftLower, leftUpper] = config.footYawOffsetBounds(0);
  const auto [rightLower, rightUpper] = config.footYawOffsetBounds(1);
  const int leftHip = model.joints[model.getJointId("l_leg_hpz")].idx_q();
  const int rightHip = model.joints[model.getJointId("r_leg_hpz")].idx_q();
  EXPECT_NEAR(leftLower, model.lowerPositionLimit(leftHip), 1e-9);
  EXPECT_NEAR(leftUpper, model.upperPositionLimit(leftHip), 1e-9);
  EXPECT_NEAR(rightLower, model.lowerPositionLimit(rightHip), 1e-9);
  EXPECT_NEAR(rightUpper, model.upperPositionLimit(rightHip), 1e-9);
  EXPECT_LT(leftLower, 0.0);
  EXPECT_GT(leftUpper, 0.0);
  EXPECT_NEAR(leftLower, -rightUpper, 1e-9) << "mirrored legs";

  // The derived values survive a configuration reload that does not carry them (the hot reload from the task file).
  ContactPlanningConfig reloaded = loadContactPlanningConfig(tmpTaskFile_, "contact_planning.", false);
  EXPECT_FALSE(reloaded.hasModelParameters()) << "the task file has no such keys";
  module->setConfig(reloaded);
  EXPECT_TRUE(module->getConfig().hasModelParameters());
  EXPECT_NEAR(module->getConfig().torsionalFrictionTorque, config.torsionalFrictionTorque, 1e-12);
  EXPECT_NEAR(module->getConfig().footYawOffsetBounds(1).first, rightLower, 1e-12);

  // comHeight and the ZMP box are only filled where the task file leaves them at 0.
  ContactPlanningGroundParameters ground;
  ground.frictionCoefficient = mu;
  ground.torsionalFrictionCoefficient = muTorsion;
  PinocchioInterface pinocchio = interface_->getPinocchioInterface();
  const ContactPlanningModelParameters derived = deriveContactPlanningModelParameters(
      pinocchio, interface_->getEffectiveMpcRobotModel(), interface_->getInitialState(),
      interface_->modelSettings().contactParentJointNames, ground, config.gravity, config.nominalStepWidth);
  EXPECT_GT(derived.comHeight, 0.6);
  EXPECT_LT(derived.comHeight, 1.2);
  ContactPlanningConfig explicitHeight = reloaded;
  explicitHeight.comHeight = 0.85;
  derived.applyTo(explicitHeight);
  EXPECT_NEAR(explicitHeight.comHeight, 0.85, 1e-12) << "an explicit height is kept";
  ContactPlanningConfig modelHeight = reloaded;
  modelHeight.comHeight = 0.0;
  derived.applyTo(modelHeight);
  EXPECT_NEAR(modelHeight.comHeight, derived.comHeight, 1e-12) << "0 means from the model";
  EXPECT_NO_THROW(modelHeight.validate());
  EXPECT_EQ(derived.hipYawJoints.size(), N_CONTACTS);
  EXPECT_EQ(derived.hipYawJoints[0], "l_leg_hpz");
  EXPECT_EQ(derived.hipYawJoints[1], "r_leg_hpz");
}

/**
 * The background planner thread can be switched on and off at runtime (the parameter updater applies task-file edits
 * on the solver thread). Each transition has to leave the module planning: switching on must start a worker that
 * consumes the very next snapshot, switching off must join it promptly (no lost wake-up) and drop any snapshot it had
 * not consumed, and synchronous planning must work right after.
 */
TEST_F(ContactPlanningIntegrationTest, BackgroundPlannerThreadCanBeToggledAtRuntime) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);

  ContactPlanningConfig quick = module->getConfig();
  quick.maxSolveTime = 0.2;  // keep every plan of this test short; its result is irrelevant here
  quick.maxBranchAndBoundNodes = 200;
  quick.localSearchMaxTime = 0.02;

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  vector_t walkingTarget = vector_t::Zero(state.size());
  walkingTarget.segment(6, 6) = state.segment(6, 6);
  walkingTarget(0) = 0.4;
  scalar_t t = 0.0;
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);

  const auto plansSoFar = [&]() { return module->getStatistics().numPlans; };
  const auto waitForPlan = [&](size_t before) {
    for (int i = 0; i < 1000 && plansSoFar() == before; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return plansSoFar() > before;
  };

  for (int cycle = 0; cycle < 3; ++cycle) {
    // On: the first snapshot after (re)starting must reach the worker, whatever the throttle state of the previous run.
    ContactPlanningConfig background = quick;
    background.runInBackgroundThread = true;
    module->setConfig(background);
    const size_t beforeBackground = plansSoFar();
    t += 0.02;
    module->preSolverRun(t, t + horizon, state, *referenceManager);
    EXPECT_TRUE(waitForPlan(beforeBackground)) << "cycle " << cycle << ": the worker did not consume the posted snapshot";

    // Off: setConfig joins the worker and must return promptly.
    const auto stopStart = std::chrono::steady_clock::now();
    ContactPlanningConfig synchronous = quick;
    synchronous.runInBackgroundThread = false;
    module->setConfig(synchronous);
    const scalar_t stopSeconds = std::chrono::duration<scalar_t>(std::chrono::steady_clock::now() - stopStart).count();
    EXPECT_LT(stopSeconds, 5.0) << "cycle " << cycle << ": stopping the worker hung";

    // Synchronous planning works right away, once per pre-solve hook.
    const size_t beforeSynchronous = plansSoFar();
    t += 0.02;
    module->preSolverRun(t, t + horizon, state, *referenceManager);
    EXPECT_EQ(plansSoFar(), beforeSynchronous + 1) << "cycle " << cycle << ": no synchronous plan";
    t += 0.5;  // past the rate limiter's period, so the next background post is not throttled by this cycle's post
  }
  // Leave the module as the fixture configured it.
  ContactPlanningConfig restore = module->getConfig();
  restore.runInBackgroundThread = false;
  module->setConfig(restore);
}

}  // namespace ocs2::humanoid
