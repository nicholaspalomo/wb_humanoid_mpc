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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <system_error>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

struct Swing {
  size_t foot = 0;
  scalar_t liftOff = 0.0;
  scalar_t touchDown = 0.0;
};

std::optional<Swing> firstSwingAfter(const ModeSchedule& schedule, scalar_t after) {
  for (size_t i = 0; i + 1 < schedule.modeSequence.size() && i < schedule.eventTimes.size(); ++i) {
    const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[i]);
    const contact_flag_t afterFlags = modeNumber2StanceLeg(schedule.modeSequence[i + 1]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (before[foot] && !afterFlags[foot] && schedule.eventTimes[i] > after) {
        Swing swing;
        swing.foot = foot;
        swing.liftOff = schedule.eventTimes[i];
        for (size_t j = i + 1; j + 1 < schedule.modeSequence.size() && j < schedule.eventTimes.size(); ++j) {
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
 * End-to-end test of the shipped contact planning default, the closed-form H-LIP planner of arXiv:2502.15630: the
 * interface builds it from the shipped configuration, it stands at rest and walks when commanded, and its footholds
 * reach the swing-foot references of the whole-body cost. The mixed-integer planner and the execution heuristics are
 * covered separately by testContactPlanningIntegration.cpp.
 */
class HlipPlanningIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    const std::string taskFile = configDir + "/config/mpc/task.yaml";
    referenceFile_ = configDir + "/config/command/reference.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";

    const std::function<std::string(const std::string&)> readFile = [](const std::string& path) {
      std::ifstream in(path);
      return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    // Each of these fixtures needs its planner configuration to sit beside its task file under the one name
    // resolveContactPlanningConfigFile looks for, so the file name cannot distinguish them and the directory has to.
    // Sharing one directory with testContactPlanningIntegration meant both wrote contact_planning.yaml to the same
    // path - this one selecting the H-LIP planner, that one the MIQP planner - and whichever ran second decided what
    // the other one loaded.
    tmpDir_ = (std::filesystem::path(testing::TempDir()) / "hlip_planning_integration").string();
    std::filesystem::create_directories(tmpDir_);

    std::string content = readFile(taskFile);
    content = std::regex_replace(content, std::regex("useContactPlanning: *(true|false)"), "useContactPlanning: true");
    tmpTaskFile_ = (std::filesystem::path(tmpDir_) / "hlip_planning_task.yaml").string();
    std::ofstream out(tmpTaskFile_);
    out << content;
    out.close();

    // The shipped planner configuration, planned synchronously so the test controls the timing. Nothing else is
    // changed: what is exercised here is the default the robot ships with.
    std::string planning = readFile(resolveContactPlanningConfigFile(taskFile));
    ASSERT_NE(planning.find("contact_planning:"), std::string::npos) << "the shipped planner configuration was not found";
    ASSERT_NE(planning.find("type: hlip"), std::string::npos) << "the shipped configuration no longer selects the H-LIP planner";
    planning = std::regex_replace(planning, std::regex("runInBackgroundThread: *(true|false)"), "runInBackgroundThread: false");
    tmpContactPlanningFile_ = (std::filesystem::path(tmpDir_) / kContactPlanningConfigFileName).string();
    std::ofstream planningOut(tmpContactPlanningFile_);
    planningOut << planning;
    planningOut.close();
    ASSERT_EQ(resolveContactPlanningConfigFile(tmpTaskFile_), tmpContactPlanningFile_);

    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created =
        CentroidalMpcInterface::Create(tmpTaskFile_, urdfFile_, referenceFile_);
    ASSERT_TRUE(created.ok()) << created.status().message();
    interface_ = *std::move(created);
    referenceManager_ = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
    ASSERT_NE(referenceManager_, nullptr);
    module_ = interface_->getContactPlannerModulePtr();
    ASSERT_NE(module_, nullptr);
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(tmpDir_, ignored);
  }

  /** Runs one planning cycle at `time` with a commanded forward velocity and activates the plan. */
  void planAt(scalar_t time, scalar_t commandedVelocityX) {
    const vector_t state = interface_->getInitialState();
    const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
    const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
    vector_t target = vector_t::Zero(state.size());
    target.segment(6, 6) = state.segment(6, 6);
    target(0) = commandedVelocityX;
    referenceManager_->setTargetTrajectories(TargetTrajectories({time}, {target}, {vector_t::Zero(inputDim)}));
    referenceManager_->preSolverRun(time, time + horizon, state, ModeNumber::STANCE);
    module_->preSolverRun(time, time + horizon, state, *referenceManager_);
    referenceManager_->preSolverRun(time + 0.02, time + 0.02 + horizon, state, ModeNumber::STANCE);
  }

  std::string referenceFile_, urdfFile_, tmpDir_, tmpTaskFile_, tmpContactPlanningFile_;
  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::shared_ptr<ContactPlanningReferenceManager> referenceManager_;
  std::shared_ptr<ContactPlannerModule> module_;
};

/**
 * Audit finding B6. The operator's commanded yaw rate used to be copied into the planner's input only inside the
 * `if (config.usesHeadingModel())` branch of makePlannerInput(). It is not part of the heading MODEL, it is part of
 * the COMMAND, and HlipStandingBlend reads it to decide whether the robot should be stepping at all. Left in that
 * branch, a robot without the heading model contributed nothing from the yaw stick to the blend's activity, so alpha
 * never crossed its half point on yaw alone and the robot would not start stepping to turn in place however hard it
 * was asked.
 */
TEST_F(HlipPlanningIntegrationTest, TheCommandedYawRateReachesThePlannerWithoutTheHeadingModel) {
  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();

  // A pure yaw command: no linear velocity at all, so the blend has nothing but the yaw rate to go on.
  const scalar_t commandedYawRate = 0.5;  // [rad/s], well above the blend's half point on its own
  const scalar_t yawInertia = 40.0;       // any positive value; the target carries m * I_zz * omega / m
  vector_t target = vector_t::Zero(state.size());
  target.segment(6, 6) = state.segment(6, 6);
  target(5) = yawInertia * commandedYawRate / interface_->getCentroidalModelInfo().robotMass;

  for (const bool headingModel : {true, false}) {
    ContactPlanningConfig config = referenceManager_->getConfig();
    config.setHeadingModel(headingModel);
    referenceManager_->setConfig(config);
    referenceManager_->setTargetTrajectories(TargetTrajectories({0.0}, {target}, {vector_t::Zero(inputDim)}));
    referenceManager_->preSolverRun(0.0, horizon, state, ModeNumber::STANCE);

    const ContactPlannerInput input = referenceManager_->makePlannerInput(0.0, state, referenceManager_->commandedVelocity());
    EXPECT_NEAR(input.headingRateCommand, referenceManager_->commandedYawRate(), 1e-9) << "heading model " << (headingModel ? "on" : "off");
    EXPECT_GT(std::abs(input.headingRateCommand), 1e-6)
        << "the yaw command must reach the planner with the heading model " << (headingModel ? "on" : "off");
  }
}

TEST_F(HlipPlanningIntegrationTest, TheShippedPlannerIsTheClosedFormHlip) {
  const ContactPlanningConfig config = referenceManager_->getConfig();
  EXPECT_EQ(canonicalPlannerName(config.planner.type), planner::kHlip);
  // The only listed rule is the paper's reference plumbing; every heuristic stays off.
  EXPECT_EQ(config.formulation.execution, std::vector<std::string>{term::kPlannedComOverride});
  const absl::StatusOr<std::string> summary = contactPlannerSummary(config);
  ASSERT_TRUE(summary.ok());
  EXPECT_NE(summary->find("H-LIP"), std::string::npos);
}

TEST_F(HlipPlanningIntegrationTest, StandsStillAtZeroCommand) {
  planAt(0.0, 0.0);
  ASSERT_TRUE(module_->getStatistics().lastPlanValid);
  ASSERT_TRUE(referenceManager_->hasActivePlan());

  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();
  for (const contact_flag_t& contacts : plan.contacts) {
    EXPECT_TRUE(contacts[CONTACT_LEFT_INDEX] && contacts[CONTACT_RIGHT_INDEX]) << "a resting robot must not step in place";
  }
}

TEST_F(HlipPlanningIntegrationTest, WalksAndAlternatesFeetAtACommand) {
  planAt(0.0, 0.5);
  ASSERT_TRUE(module_->getStatistics().lastPlanValid);
  ASSERT_TRUE(referenceManager_->hasActivePlan());

  const ModeSchedule& schedule = referenceManager_->getModeSchedule();
  const std::optional<Swing> first = firstSwingAfter(schedule, 0.02);
  ASSERT_TRUE(first.has_value()) << "a commanded walk must produce a swing";
  const std::optional<Swing> second = firstSwingAfter(schedule, first->liftOff + 1e-6);
  ASSERT_TRUE(second.has_value());
  EXPECT_NE(first->foot, second->foot) << "the feet must alternate";

  // The cadence is the configured single support duration, not a searched one. The first swing is measured from the
  // commit boundary rather than from its own lift-off - the schedule the robot is already executing wins inside the
  // commit window - so the second one is the first whole phase of the nominal gait.
  const ContactPlanningConfig config = referenceManager_->getConfig();
  EXPECT_NEAR(second->touchDown - second->liftOff, config.hlip.sspDuration, 1.5 * config.planner.dt);
}

TEST_F(HlipPlanningIntegrationTest, TheGaitAdvancesAtTheCommandedVelocity) {
  const scalar_t commandedVelocityX = 0.5;
  planAt(0.0, commandedVelocityX);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();
  const ContactPlanningConfig config = referenceManager_->getConfig();

  // Both feet end the horizon ahead of where they started.
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    EXPECT_GT(plan.footholds.back()[foot].x(), plan.footholds.front()[foot].x()) << "foot " << foot << " must travel forward";
  }

  // And so does the reduced model. The average falls short of the command over this horizon because the first step of
  // the deadbeat law is placed behind the centre of mass, which is how the H-LIP accelerates out of a standstill; what
  // matters is that the plan is walking forward at a speed of the commanded order.
  const scalar_t horizon = config.horizon();
  const scalar_t averageVelocity = (plan.comPosition.back().x() - plan.comPosition.front().x()) / horizon;
  EXPECT_GT(averageVelocity, 0.3 * commandedVelocityX);
  EXPECT_LT(averageVelocity, 1.2 * commandedVelocityX);
}

TEST_F(HlipPlanningIntegrationTest, ThePlannedComReplacesTheTargetComReference) {
  // The lateral orbit the deadbeat step regulates to needs the centre of mass to fall towards the swing foot. The
  // target trajectory built from the operator's command asks for a straight line with no lateral velocity, so unless
  // the plan's centre of mass is written into the reference the whole-body MPC is asked for the opposite motion and
  // the planner narrows the step until the robot falls sideways.
  const ContactPlanningConfig config = referenceManager_->getConfig();
  ASSERT_TRUE(config.formulation.hasExecutionRule(term::kPlannedComOverride)) << "the override must be enabled by default";

  planAt(0.0, 0.5);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();

  const TargetTrajectories& target = referenceManager_->getTargetTrajectories();
  ASSERT_FALSE(target.timeTrajectory.empty());
  const MpcRobotModelBase<scalar_t>& robotModel = interface_->getEffectiveMpcRobotModel();

  size_t comparedPoints = 0;
  for (size_t i = 0; i < target.timeTrajectory.size(); ++i) {
    const scalar_t time = target.timeTrajectory[i];
    if (time < plan.startTime || time > plan.endTime()) continue;
    const std::optional<vector2_t> plannedVelocity = plan.comVelocityAtTime(time);
    ASSERT_TRUE(plannedVelocity.has_value());
    const vector3_t referenceVelocity = robotModel.getBaseComLinearVelocity(target.stateTrajectory[i]);
    EXPECT_NEAR(referenceVelocity.x(), plannedVelocity->x(), 1e-9) << "at t=" << time;
    EXPECT_NEAR(referenceVelocity.y(), plannedVelocity->y(), 1e-9) << "at t=" << time;
    ++comparedPoints;
  }
  EXPECT_GT(comparedPoints, 0U) << "the target must have a point inside the plan's horizon";

  // And what it asks for is genuinely not a straight line: the orbit the footholds were placed for swings the centre
  // of mass from side to side, which is the motion the reference has to carry for the two layers to agree.
  scalar_t largestLateralVelocity = 0.0;
  for (const vector2_t& velocity : plan.comVelocity) largestLateralVelocity = std::max(largestLateralVelocity, std::abs(velocity.y()));
  EXPECT_GT(largestLateralVelocity, 0.05) << "the H-LIP reference must ask the controller for lateral motion";
}

TEST_F(HlipPlanningIntegrationTest, KeepsTheOperatorCommandWhenTheTargetIsNotRepublished) {
  // The regression this guards: planned_com_override rewrites the momentum channel of the live target, and a target is
  // only replaced when a new one is published. A planner that read its command back off that channel would be fed its
  // own output one cycle later, which at rest is a zero command - the blend never crosses its half point and the robot
  // never steps, however far the operator pushes the velocity slider. The command is therefore read from the copy the
  // operator published, so it must survive any number of cycles without a new one.
  const scalar_t commandedVelocityX = 0.5;
  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  vector_t target = vector_t::Zero(state.size());
  target.segment(6, 6) = state.segment(6, 6);
  target(0) = commandedVelocityX;
  referenceManager_->setTargetTrajectories(TargetTrajectories({0.0}, {target}, {vector_t::Zero(inputDim)}));

  scalar_t time = 0.0;
  for (int cycle = 0; cycle < 5; ++cycle) {
    referenceManager_->preSolverRun(time, time + horizon, state, ModeNumber::STANCE);
    module_->preSolverRun(time, time + horizon, state, *referenceManager_);
    EXPECT_NEAR(referenceManager_->commandedVelocity().x(), commandedVelocityX, 1e-9)
        << "the operator's command was lost at cycle " << cycle;
    time += 0.02;
  }

  referenceManager_->preSolverRun(time, time + horizon, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  EXPECT_TRUE(firstSwingAfter(referenceManager_->getModeSchedule(), time).has_value())
      << "the robot must still be stepping after several cycles without a new target";
}

TEST_F(HlipPlanningIntegrationTest, TheTerminalDcmReferenceFollowsThePlan) {
  // The lateral orbit needs the DCM beyond the stance foot, towards the foot about to land. The terminal cost's own
  // reference is the centre of the terminal support - the stance foot itself during single support - and at its weight
  // it wins: the centre of mass is held over the foot, the planner reads a state with no lateral velocity and narrows
  // the next step to minStepWidth, and the robot falls sideways. With a plan active the reference must be the plan's.
  planAt(0.0, 0.5);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();
  const ContactPlanningConfig config = referenceManager_->getConfig();
  const scalar_t omega = config.omega();
  const scalar_t time = plan.startTime + 0.5 * config.horizon();

  const std::optional<vector2_t> plannedDcm = referenceManager_->getPlannedDcm(time, omega);
  ASSERT_TRUE(plannedDcm.has_value());
  const std::optional<vector2_t> comPosition = plan.comPositionAtTime(time);
  const std::optional<vector2_t> comVelocity = plan.comVelocityAtTime(time);
  ASSERT_TRUE(comPosition.has_value() && comVelocity.has_value());
  const vector2_t expected = *comPosition + *comVelocity / omega;
  EXPECT_NEAR(plannedDcm->x(), expected.x(), 1e-12);
  EXPECT_NEAR(plannedDcm->y(), expected.y(), 1e-12);

  // And the cost carries it: the blend weight is on, and the reference it blends in is that DCM.
  const DcmTerminalCost& cost = interface_->getOptimalControlProblem().finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost");
  const vector_t parameters = cost.getParameters(time, referenceManager_->getTargetTrajectories());
  ASSERT_GE(parameters.size(), 11);
  EXPECT_NEAR(parameters(8), 1.0, 1e-12) << "the planned reference must be selected while a plan is active";
  EXPECT_NEAR(parameters(9), plannedDcm->x(), 1e-12);
  EXPECT_NEAR(parameters(10), plannedDcm->y(), 1e-12);
}

}  // namespace ocs2::humanoid
