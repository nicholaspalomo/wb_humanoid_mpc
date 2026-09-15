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
#include <pinocchio/algorithm/centroidal.hpp>

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

    // Temporary task file with contact planning on, and next to it a temporary copy of the planner's own file
    // (contact_planning.yaml, found by its name in the task file's directory) with the test's solver limits, planned
    // synchronously so that the test controls the timing, and the heading model on.
    const auto readFile = [](const std::string& path) {
      std::ifstream in(path);
      return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    std::string content = readFile(taskFile);
    content = std::regex_replace(content, std::regex("useContactPlanning: *(true|false)"), "useContactPlanning: true");
    tmpTaskFile_ = testing::TempDir() + "/contact_planning_task.yaml";
    std::ofstream out(tmpTaskFile_);
    out << content;
    out.close();

    std::string planning = readFile(resolveContactPlanningConfigFile(taskFile));
    ASSERT_NE(planning.find("contact_planning:"), std::string::npos) << "the shipped planner configuration was not found";
    planning = std::regex_replace(planning, std::regex("runInBackgroundThread: *(true|false)"), "runInBackgroundThread: false");
    planning = std::regex_replace(planning, std::regex("maxSolveTime: *[0-9.]+"), "maxSolveTime: 5.0");
    planning = std::regex_replace(planning, std::regex("maxBranchAndBoundNodes: *[0-9]+"), "maxBranchAndBoundNodes: 2000");
    planning = std::regex_replace(planning, std::regex("useAcomDynamics: *(true|false)"), "useAcomDynamics: true");
    planning = std::regex_replace(planning, std::regex("planHeadingOverridesTarget: *(true|false)"), "planHeadingOverridesTarget: true");
    tmpContactPlanningFile_ = testing::TempDir() + "/" + kContactPlanningConfigFileName;
    std::ofstream planningOut(tmpContactPlanningFile_);
    planningOut << planning;
    planningOut.close();
    ASSERT_EQ(resolveContactPlanningConfigFile(tmpTaskFile_), tmpContactPlanningFile_);

    auto created = CentroidalMpcInterface::Create(tmpTaskFile_, urdfFile_, referenceFile_);
    ASSERT_TRUE(created.ok()) << created.status().message();
    interface_ = *std::move(created);
  }

  void TearDown() override {
    std::remove(tmpTaskFile_.c_str());
    std::remove(tmpContactPlanningFile_.c_str());
  }

  std::string referenceFile_, urdfFile_, tmpTaskFile_, tmpContactPlanningFile_;
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

  // The target contact poses (drawn by the MuJoCo viewer) are the landing poses of the upcoming swings: for a foot with
  // a swing in the schedule the pose is where the swing-foot reference ends up at its touch-down.
  {
    const feet_array_t<TargetContactPose> targets = referenceManager->getTargetContactPoses();
    bool foundLandingTarget = false;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const TargetContactPose& target = targets[foot];
      ASSERT_TRUE(target.valid) << "foot " << foot;
      EXPECT_TRUE(target.position.allFinite());
      EXPECT_TRUE(std::isfinite(target.yaw));
      if (target.kind == TargetContactPose::Kind::STANCE) continue;
      foundLandingTarget = true;
      EXPECT_GT(target.touchDownTime, t);
      EXPECT_NEAR(target.height, 0.0, 0.05) << "flat ground";
      const auto reference = referenceManager->getSwingFootReference(foot, target.touchDownTime - 1e-4);
      ASSERT_TRUE(reference.has_value()) << "foot " << foot << " has no swing reference just before its touch-down";
      EXPECT_NEAR(target.position(0), reference->position(0), 1e-3);
      EXPECT_NEAR(target.position(1), reference->position(1), 1e-3);
      if (reference->yaw.has_value()) {
        EXPECT_TRUE(target.yawPlanned);
        EXPECT_NEAR(std::remainder(target.yaw - *reference->yaw, 2.0 * M_PI), 0.0, 1e-3);
      }
    }
    EXPECT_TRUE(foundLandingTarget) << "a walking plan must give at least one foot a landing target";
  }

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
  // The height reference continues down from the planned touch-down at the search velocity: it never rises at an
  // extension (re-fitting the spline over the extended swing lifted it by several millimetres) and its slope is the
  // configured one, not the spline's.
  const auto atDetection = referenceManager->getSwingFootReference(foot2, late1 + 1e-4);
  ASSERT_TRUE(atDetection.has_value());
  EXPECT_LE(atDetection->position(2), referenceBeforeExtension->position(2) + 1e-6) << "the reference must not jump up";
  EXPECT_NEAR(extendedReference->position(2),
              referenceBeforeExtension->position(2) - config.lateTouchdownSearchVelocity * (late1 + 0.02 - next->touchDown), 2e-4);
  EXPECT_NEAR(extendedReference->linearVelocity(2), -config.lateTouchdownSearchVelocity, 1e-9);
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
    // The solver runs the reference manager's hook before the module's every cycle, which activates the plan the
    // previous cycle produced; without it the module would (rightly) hold its snapshot for a plan that awaits activation.
    ContactPlanningConfig background = quick;
    background.runInBackgroundThread = true;
    module->setConfig(background);
    const size_t beforeBackground = plansSoFar();
    t += 0.02;
    referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
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
    referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
    module->preSolverRun(t, t + horizon, state, *referenceManager);
    EXPECT_EQ(plansSoFar(), beforeSynchronous + 1) << "cycle " << cycle << ": no synchronous plan";
    t += 0.5;  // past the rate limiter's period, so the next background post is not throttled by this cycle's post
  }
  // Leave the module as the fixture configured it.
  ContactPlanningConfig restore = module->getConfig();
  restore.runInBackgroundThread = false;
  module->setConfig(restore);
}

/**
 * The plan's heading is written into the base yaw of the target trajectory (planHeadingOverridesTarget). The yaw rate
 * command handed to the planner used to be differentiated from that same base yaw, which fed the previous plan's heading
 * rate back in as the command: whenever a plan fell short of the command, e.g. through the hip yaw range of the stance
 * foot, the shortfall became the next command and the turn decayed to zero over a few plans. The command is now read
 * off the target's momentum channel, h_z = I_zz omega / m, which the override never touches.
 *
 * The override must still reach the MPC, the command must survive a cycle in which no fresh target arrives (the stored
 * target is then the rewritten one) and follow a change of command. The plan is hand-built with a deliberate shortfall
 * so that the two rates are distinguishable.
 */
TEST_F(ContactPlanningIntegrationTest, PlannedHeadingOverrideDoesNotFeedBackIntoTheYawRateCommand) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);
  const ContactPlanningConfig config = module->getConfig();
  ASSERT_TRUE(config.useAcomDynamics && config.planHeadingOverridesTarget);

  const MpcRobotModelBase<scalar_t>& robotModel = interface_->getEffectiveMpcRobotModel();
  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = robotModel.getInputDim();
  const scalar_t commandedYawRate = 0.5;  // [rad/s] what the operator asks for
  const scalar_t plannedYawRate = 0.15;   // [rad/s] what the (hand-built) plan delivers: a clear shortfall

  // The locked inertia about the CoM, as the target trajectories calculator builds the momentum target from it.
  const matrix3_t lockedInertia = [&]() {
    PinocchioInterface& pinocchio = interface_->getPinocchioInterface();
    const auto& model = pinocchio.getModel();
    pinocchio::ccrba(model, pinocchio.getData(), robotModel.getGeneralizedCoordinates(state), vector_t::Zero(model.nv));
    return matrix3_t(pinocchio.getData().Ig.inertia().matrix());
  }();
  const scalar_t mass = pinocchio::computeTotalMass(interface_->getPinocchioInterface().getModel());
  // A target that carries the commanded yaw rate as the momentum of a rigid turn and integrates it in the base yaw over
  // the horizon, as the motion manager produces from a turn command.
  const auto turningTarget = [&](scalar_t t0, scalar_t yawRate) {
    vector_t start = state;
    start.segment<3>(3) = lockedInertia.col(2) * (yawRate / mass);
    vector_t end = start;
    vector3_t euler = robotModel.getBaseOrientationEulerZYX(state);
    euler(0) += yawRate * horizon;
    robotModel.setBaseOrientationEulerZYX(end, euler);
    return TargetTrajectories({t0, t0 + horizon}, {start, end}, {vector_t::Zero(inputDim), vector_t::Zero(inputDim)});
  };
  // Yaw rate of whatever target the reference manager currently holds, differentiated the way the planner used to read it.
  const auto storedTargetYawRate = [&](scalar_t t0) {
    const TargetTrajectories& target = referenceManager->getTargetTrajectories();
    const scalar_t yaw0 = robotModel.getBaseOrientationEulerZYX(target.getDesiredState(t0))(0);
    const scalar_t yaw1 = robotModel.getBaseOrientationEulerZYX(target.getDesiredState(t0 + 0.2))(0);
    return (yaw1 - yaw0) / 0.2;
  };

  // 1. Before any plan exists the command is read straight off the target.
  scalar_t t = 0.0;
  referenceManager->setTargetTrajectories(turningTarget(t, commandedYawRate));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  ContactPlannerInput input = referenceManager->makePlannerInput(t, state, vector2_t::Zero());
  EXPECT_NEAR(input.headingRateCommand, commandedYawRate, 1e-9);
  EXPECT_NEAR(storedTargetYawRate(t), commandedYawRate, 1e-9);

  // 2. A plan with a heading model that turns slower than commanded, standing on both feet throughout.
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = t;
  plan.dt = config.dt;
  plan.committedUntil = t + config.commitTime;
  plan.yaw = input.heading;
  const int numNodes = config.numNodes;
  plan.contacts.assign(numNodes, makeFeetArray(true));
  plan.footholds.assign(numNodes + 1, input.footPositions);
  plan.comPosition.assign(numNodes + 1, input.comPosition);
  plan.comVelocity.assign(numNodes + 1, vector2_t::Zero());
  plan.zmp.assign(numNodes, input.comPosition);
  plan.heading.resize(numNodes + 1);
  plan.headingRate.assign(numNodes + 1, plannedYawRate);
  plan.footYaws.assign(numNodes + 1, makeFeetArray(input.heading));
  for (int k = 0; k <= numNodes; ++k) plan.heading[k] = input.heading + plannedYawRate * config.dt * static_cast<scalar_t>(k);
  referenceManager->setContactPlan(plan);

  // The motion manager buffers a fresh target every cycle; the reference manager activates the plan and rewrites it.
  t += 0.02;
  referenceManager->setTargetTrajectories(turningTarget(t, commandedYawRate));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager->hasActivePlan());
  EXPECT_NEAR(storedTargetYawRate(t), plannedYawRate, 1e-6) << "the override must still reach the MPC's target";
  input = referenceManager->makePlannerInput(t, state, vector2_t::Zero());
  EXPECT_NEAR(input.headingRateCommand, commandedYawRate, 1e-9) << "the planner must be asked for the operator's rate, not its own";

  // 3. No fresh target this cycle: the stored one is the rewritten one, and the captured command has to stand.
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager->hasActivePlan());
  EXPECT_NEAR(storedTargetYawRate(t), plannedYawRate, 1e-6);
  input = referenceManager->makePlannerInput(t, state, vector2_t::Zero());
  EXPECT_NEAR(input.headingRateCommand, commandedYawRate, 1e-9) << "a stale target must not be mistaken for a new command";

  // 4. The operator changes the command: the new rate is picked up on the next fresh target.
  t += 0.02;
  referenceManager->setTargetTrajectories(turningTarget(t, 2.0 * commandedYawRate));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  input = referenceManager->makePlannerInput(t, state, vector2_t::Zero());
  EXPECT_NEAR(input.headingRateCommand, 2.0 * commandedYawRate, 1e-9);
  EXPECT_NEAR(storedTargetYawRate(t), plannedYawRate, 1e-6) << "the override keeps following the plan";
}

namespace {

/** The first two swings of one foot that lift off after `after`. */
std::pair<std::optional<Swing>, std::optional<Swing>> twoSwingsOfOneFoot(const ModeSchedule& schedule, scalar_t after) {
  std::optional<Swing> first, second;
  for (size_t i = 0; i + 1 < schedule.modeSequence.size(); ++i) {
    const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[i]);
    const contact_flag_t afterFlags = modeNumber2StanceLeg(schedule.modeSequence[i + 1]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!(before[foot] && !afterFlags[foot]) || schedule.eventTimes[i] <= after) continue;
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
  return {first, second};
}

}  // namespace

/**
 * The DCM step adjustment is computed for the swing in flight, propagated to that swing's touch-down. It belongs to that
 * landing alone: a later swing of the same foot inside the horizon lands on its planned foothold. Before, the offset was
 * added to every swing of the foot, so a saturated 5 cm correction also displaced the second landing for no reason.
 */
TEST_F(ContactPlanningIntegrationTest, DcmStepAdjustmentAppliesOnlyToTheSwingInFlight) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);

  ContactPlanningConfig config = module->getConfig();
  config.numNodes = 24;             // 2.4 s plan
  config.maxContactDuration = 0.6;  // keep stepping, so that the plan holds two swings of one foot
  config.enablePhaseResetting = false;
  config.enableEnergyCadenceModulation = false;
  config.enableDcmStepAdjustment = true;
  module->setConfig(config);

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  vector_t walkingTarget = vector_t::Zero(state.size());
  walkingTarget.segment(6, 6) = state.segment(6, 6);
  walkingTarget(0) = 0.4;
  scalar_t t = 0.0;
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  ASSERT_TRUE(module->getStatistics().lastPlanValid);
  t += 0.02;
  referenceManager->preSolverRun(t, t + 1.4, state, ModeNumber::STANCE);  // a 1.4 s horizon keeps [t - 1.4, t + 2.8]
  ASSERT_TRUE(referenceManager->hasActivePlan());
  const auto [first, second] = twoSwingsOfOneFoot(referenceManager->getModeSchedule(), t);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value()) << "the 2.4 s plan should contain two swings of the same foot";
  const size_t foot = first->foot;
  const ContactPlan plan = *referenceManager->getActiveContactPlan();
  const vector2_t firstLanding = *plan.footholdAtTime(foot, first->touchDown);
  const vector2_t secondLanding = *plan.footholdAtTime(foot, second->touchDown);

  // Mid-flight in the first swing, the CoM velocity ahead of what the controller predicted: the first landing moves.
  const scalar_t mid = 0.5 * (first->liftOff + first->touchDown);
  referenceManager->setPredictedTrajectory({0.0, mid + 10.0}, {state, state});
  vector_t pushed = state;
  pushed(0) += 0.05;
  referenceManager->preSolverRun(mid, mid + 1.4, pushed, modeWithSwingFoot(foot));
  const vector2_t adjustment = referenceManager->getDcmStepAdjustment()[foot];
  ASSERT_GT(adjustment.norm(), 5e-3) << "the push must produce a visible offset for this test to mean anything";
  const auto atFirstTouchDown = referenceManager->getSwingFootReference(foot, first->touchDown - 1e-3);
  ASSERT_TRUE(atFirstTouchDown.has_value());
  EXPECT_LT((atFirstTouchDown->position.head<2>() - (firstLanding + adjustment)).norm(), 2e-3) << "the swing in flight carries the offset";

  // The same foot's next swing lands where the plan put it.
  const auto atSecondTouchDown = referenceManager->getSwingFootReference(foot, second->touchDown - 1e-3);
  ASSERT_TRUE(atSecondTouchDown.has_value());
  EXPECT_LT((atSecondTouchDown->position.head<2>() - secondLanding).norm(), 2e-3)
      << "reference " << atSecondTouchDown->position.head<2>().transpose() << " vs planned landing " << secondLanding.transpose()
      << " (offset " << adjustment.transpose() << " leaked into the second swing)";

  referenceManager->setPredictedTrajectory({}, {});
  module->setConfig(config);
}

/**
 * Closed forms of the two corrections, against the quantities the reference manager itself exposes: the cadence shift
 * is minus the gain times the orbital energy increment along the plan's heading (with the full model mass, relative to
 * the plan's ZMP); the DCM adjustment is the gain times the DCM increment propagated to touch-down, is the same on two
 * consecutive cycles with the same increment (no accumulation), and vanishes once the prediction has caught up with the
 * measurement, which is what makes the loop a one-period increment rather than a persistent-error feedback.
 */
TEST_F(ContactPlanningIntegrationTest, CadenceAndDcmCorrectionsMatchTheirClosedForms) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);
  ContactPlanningConfig config = module->getConfig();
  config.enablePhaseResetting = false;
  config.enableDcmStepAdjustment = false;
  config.enableEnergyCadenceModulation = true;
  config.energyCadenceGain = 0.01;
  module->setConfig(config);
  const scalar_t omega = config.omega();
  const scalar_t mass = pinocchio::computeTotalMass(interface_->getPinocchioInterface().getModel());

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  vector_t walkingTarget = vector_t::Zero(state.size());
  walkingTarget.segment(6, 6) = state.segment(6, 6);
  walkingTarget(0) = 0.4;
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
  const size_t inFlight = modeWithSwingFoot(foot);
  referenceManager->setPredictedTrajectory({0.0, t + 100.0}, {state, state});

  const auto orbitalEnergy = [&](scalar_t x, scalar_t v) { return 0.5 * mass * (v * v - omega * omega * x * x); };
  // The CoM state along the plan's heading relative to the plan's ZMP at `time`, as the manager computes it from a state.
  const auto lipState = [&](const vector_t& x, scalar_t time) {
    const ContactPlan plan = *referenceManager->getActiveContactPlan();
    const vector2_t heading(std::cos(plan.yaw), std::sin(plan.yaw));
    const std::optional<LipState> reference = lipReferenceState(plan, omega, time);
    EXPECT_TRUE(reference.has_value());
    const ContactPlannerInput input = referenceManager->makePlannerInput(time, x, vector2_t::Zero());
    return std::make_pair(heading.dot(input.comPosition - reference->zmp), heading.dot(input.comVelocity));
  };

  // ---- Cadence: a velocity increment along the heading. ----
  scalar_t time = swing->liftOff + 0.05;
  {
    const ContactPlan plan = *referenceManager->getActiveContactPlan();
    const vector2_t heading(std::cos(plan.yaw), std::sin(plan.yaw));
    vector_t faster = state;
    faster.segment<2>(0) += 0.05 * heading;  // normalized linear momentum = CoM velocity
    const auto [xPredicted, vPredicted] = lipState(state, time);
    const auto [xMeasured, vMeasured] = lipState(faster, time);
    EXPECT_NEAR(xMeasured, xPredicted, 1e-12);
    referenceManager->preSolverRun(time, time + horizon, faster, inFlight);
    const scalar_t expected = -config.energyCadenceGain * (orbitalEnergy(xMeasured, vMeasured) - orbitalEnergy(xPredicted, vPredicted));
    EXPECT_NEAR(referenceManager->getCadenceTouchDownShift()[foot], expected, 1e-9);
    EXPECT_LT(expected, 0.0) << "more energy brings the step forward";
  }
  // ---- Cadence: a lateral velocity increment carries no energy along the heading. ----
  time += 0.02;
  {
    const ContactPlan plan = *referenceManager->getActiveContactPlan();
    const vector2_t lateral(-std::sin(plan.yaw), std::cos(plan.yaw));
    vector_t sideways = state;
    sideways.segment<2>(0) += 0.05 * lateral;
    referenceManager->preSolverRun(time, time + horizon, sideways, inFlight);
    EXPECT_NEAR(referenceManager->getCadenceTouchDownShift()[foot], 0.0, 1e-12);
  }
  // ---- Cadence: a position increment enters through the -omega^2 x^2 term, measured from the plan's ZMP. ----
  time += 0.02;
  {
    const ContactPlan plan = *referenceManager->getActiveContactPlan();
    const vector2_t heading(std::cos(plan.yaw), std::sin(plan.yaw));
    vector_t ahead = state;
    ahead.segment<2>(6) += 0.02 * heading;  // the base, and with it the CoM, 2 cm further along the heading
    const auto [xPredicted, vPredicted] = lipState(state, time);
    const auto [xMeasured, vMeasured] = lipState(ahead, time);
    EXPECT_NEAR(xMeasured - xPredicted, 0.02, 1e-9);
    EXPECT_NEAR(vMeasured, vPredicted, 1e-12);
    referenceManager->preSolverRun(time, time + horizon, ahead, inFlight);
    const scalar_t expected = -config.energyCadenceGain * (orbitalEnergy(xMeasured, vMeasured) - orbitalEnergy(xPredicted, vPredicted));
    EXPECT_NEAR(referenceManager->getCadenceTouchDownShift()[foot], expected, 1e-9);
  }

  // ---- Cadence: a deadband swallows small deviations and measures the shift from its edge. ----
  time += 0.02;
  {
    ContactPlanningConfig banded = config;
    banded.energyCadenceDeadband = 0.1;
    module->setConfig(banded);
    const ContactPlan plan = *referenceManager->getActiveContactPlan();
    const vector2_t heading(std::cos(plan.yaw), std::sin(plan.yaw));
    vector_t faster = state;
    faster.segment<2>(0) += 0.05 * heading;
    const auto [xPredicted, vPredicted] = lipState(state, time);
    const auto [xMeasured, vMeasured] = lipState(faster, time);
    const scalar_t deviation = orbitalEnergy(xMeasured, vMeasured) - orbitalEnergy(xPredicted, vPredicted);
    ASSERT_GT(deviation, banded.energyCadenceDeadband) << "the push must exceed the band for this check to mean anything";
    referenceManager->preSolverRun(time, time + horizon, faster, inFlight);
    EXPECT_NEAR(referenceManager->getCadenceTouchDownShift()[foot], -banded.energyCadenceGain * (deviation - banded.energyCadenceDeadband),
                1e-9);
    // Inside the band nothing is re-timed.
    time += 0.02;
    vector_t slightlyFaster = state;
    slightlyFaster.segment<2>(0) += 0.01 * heading;
    const auto [xSlight, vSlight] = lipState(slightlyFaster, time);
    ASSERT_LT(std::abs(orbitalEnergy(xSlight, vSlight) - orbitalEnergy(xPredicted, vPredicted)), banded.energyCadenceDeadband);
    referenceManager->preSolverRun(time, time + horizon, slightlyFaster, inFlight);
    EXPECT_NEAR(referenceManager->getCadenceTouchDownShift()[foot], 0.0, 1e-12);
    module->setConfig(config);
  }

  // ---- DCM: a position increment, propagated to touch-down; no accumulation; gone once the prediction agrees. ----
  ContactPlanningConfig dcm = config;
  dcm.enableEnergyCadenceModulation = false;
  dcm.enableDcmStepAdjustment = true;
  module->setConfig(dcm);
  time += 0.02;
  const auto phase = swingPhaseAtTime(referenceManager->getModeSchedule(), foot, time);
  ASSERT_TRUE(phase.has_value());
  const scalar_t touchDown = phase->second;
  const scalar_t propagation = std::exp(omega * (touchDown - time));
  vector_t pushed = state;
  pushed(6) += 0.01;  // base x, and with it the CoM x and the DCM x
  pushed(7) -= 0.004;
  const vector2_t expectedAdjustment = dcm.dcmAdjustmentGain * propagation * vector2_t(0.01, -0.004);
  ASSERT_LT(expectedAdjustment.norm(), dcm.dcmAdjustmentMaxOffset) << "the push must stay below the bound";
  referenceManager->preSolverRun(time, time + horizon, pushed, inFlight);
  const vector2_t adjustment = referenceManager->getDcmStepAdjustment()[foot];
  EXPECT_NEAR(adjustment(0), expectedAdjustment(0), 1e-6);
  EXPECT_NEAR(adjustment(1), expectedAdjustment(1), 1e-6);
  // The same increment on the next cycle gives the same offset (the prediction has not moved): nothing accumulates.
  referenceManager->preSolverRun(time + 0.02, time + 0.02 + horizon, pushed, inFlight);
  const vector2_t again = referenceManager->getDcmStepAdjustment()[foot];
  EXPECT_NEAR(again(0), dcm.dcmAdjustmentGain * std::exp(omega * (touchDown - time - 0.02)) * 0.01, 1e-6);
  EXPECT_NEAR(again(1), dcm.dcmAdjustmentGain * std::exp(omega * (touchDown - time - 0.02)) * -0.004, 1e-6);
  // Once the controller's prediction has caught up with the displaced state the correction is gone, although the CoM
  // is still displaced: the loop acts on the one-period increment, not on a persistent error.
  referenceManager->setPredictedTrajectory({0.0, t + 100.0}, {pushed, pushed});
  referenceManager->preSolverRun(time + 0.04, time + 0.04 + horizon, pushed, inFlight);
  EXPECT_TRUE(referenceManager->getDcmStepAdjustment()[foot].isZero(1e-12));

  referenceManager->setPredictedTrajectory({}, {});
  module->setConfig(config);
}

/**
 * A plan the worker hands over between the reference manager's and this module's pre-solve hooks of the same cycle is
 * not in the schedule the module's snapshot would be taken from. Posting that snapshot had the idle worker plan from a
 * schedule without the pending plan; the result was dropped as inconsistent one cycle later, or merged around swings it
 * did not know about. The snapshot waits for the cycle that activates the plan.
 */
TEST_F(ContactPlanningIntegrationTest, SnapshotIsNotPostedWhileAPlanAwaitsActivation) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);

  ContactPlanningConfig background = module->getConfig();
  background.maxSolveTime = 0.2;
  background.maxBranchAndBoundNodes = 200;
  background.localSearchMaxTime = 0.02;
  background.runInBackgroundThread = true;
  module->setConfig(background);

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

  const size_t before = plansSoFar();
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  ASSERT_TRUE(waitForPlan(before)) << "the worker did not consume the posted snapshot";
  EXPECT_TRUE(referenceManager->hasPendingPlan()) << "the worker's plan waits for the solver thread to activate it";

  // The plan is pending and the planning period has passed: the snapshot must still not be posted.
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  const size_t guarded = plansSoFar();
  t += 0.02;
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  EXPECT_EQ(plansSoFar(), guarded) << "a snapshot was planned from a schedule that did not contain the pending plan";
  EXPECT_TRUE(referenceManager->hasPendingPlan());

  // The cycle that activates the plan posts the next snapshot.
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  EXPECT_FALSE(referenceManager->hasPendingPlan());
  ASSERT_TRUE(referenceManager->hasActivePlan());
  const size_t activated = plansSoFar();
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  EXPECT_TRUE(waitForPlan(activated)) << "the snapshot after the activation was not planned";

  ContactPlanningConfig restore = module->getConfig();
  restore.runInBackgroundThread = false;
  module->setConfig(restore);
}

/**
 * A plan whose commit boundary has already passed when it arrives (the planner took longer than commitTime) is dropped
 * and the active plan and schedule stay as they are; the drop is counted where the telemetry can see it. While plans
 * keep being dropped the executed schedule runs out and the robot stops walking, so this must not stay silent.
 */
TEST_F(ContactPlanningIntegrationTest, StalePlanIsDroppedAndCounted) {
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  vector_t walkingTarget = vector_t::Zero(state.size());
  walkingTarget.segment(6, 6) = state.segment(6, 6);
  walkingTarget(0) = 0.4;
  scalar_t t = 0.0;
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  ASSERT_TRUE(module->getStatistics().lastPlanValid);
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager->hasActivePlan());
  EXPECT_EQ(module->getStatistics().numStalePlansDropped, 0u);
  EXPECT_EQ(module->getStatistics().numInconsistentPlansDropped, 0u);
  const scalar_t activeStart = referenceManager->getActiveContactPlan()->startTime;
  const ModeSchedule before = referenceManager->getModeSchedule();

  ContactPlan stale = *referenceManager->getActiveContactPlan();
  stale.startTime = t - 0.5;
  stale.committedUntil = t - 0.01;  // its boundary passed while it was being computed
  referenceManager->setContactPlan(stale);
  EXPECT_TRUE(referenceManager->hasPendingPlan());
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  EXPECT_FALSE(referenceManager->hasPendingPlan());
  EXPECT_EQ(module->getStatistics().numStalePlansDropped, 1u);
  EXPECT_NEAR(referenceManager->getActiveContactPlan()->startTime, activeStart, 1e-12) << "the previous plan stays active";
  const ModeSchedule after = referenceManager->getModeSchedule();
  ASSERT_EQ(after.eventTimes.size(), before.eventTimes.size());
  for (size_t i = 0; i < before.eventTimes.size(); ++i) {
    EXPECT_NEAR(after.eventTimes[i], before.eventTimes[i], 1e-12) << "event " << i;
    EXPECT_EQ(after.modeSequence[i], before.modeSequence[i]) << "mode " << i;
  }
}

}  // namespace ocs2::humanoid
