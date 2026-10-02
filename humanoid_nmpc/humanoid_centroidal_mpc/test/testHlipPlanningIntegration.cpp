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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedComOverride.h"
#include "humanoid_common_mpc/contact_planning/execution/PlannedHeadingOverride.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "robot_core/ResourcePaths.h"

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

/** Every log message of any severity issued on any thread while it is alive. */
class MessageLog {
 public:
  MessageLog() {
    EXPECT_CALL(log_, Log(testing::_, testing::_, testing::_))
        .WillRepeatedly([this](absl::LogSeverity severity, const std::string& /*filePath*/, const std::string& message) {
          messages_.emplace_back(severity, message);
        });
    log_.StartCapturingLogs();
  }

  /** How many captured messages of `severity` contain `needle`. */
  size_t count(absl::LogSeverity severity, const std::string& needle) const {
    size_t found = 0;
    for (const std::pair<absl::LogSeverity, std::string>& message : messages_) {
      if (message.first == severity && absl::StrContains(message.second, needle)) ++found;
    }
    return found;
  }

 private:
  // Declared before the mock so that it outlives it: the mock stops capturing in its destructor.
  std::vector<std::pair<absl::LogSeverity, std::string>> messages_;
  absl::ScopedMockLog log_{absl::MockLogDefault::kIgnoreUnexpected};
};

/** Every LOG(WARNING) issued on any thread while it is alive. */
class WarningLog {
 public:
  WarningLog() {
    EXPECT_CALL(log_, Log(testing::_, testing::_, testing::_))
        .WillRepeatedly([this](absl::LogSeverity severity, const std::string& /*filePath*/, const std::string& message) {
          if (severity == absl::LogSeverity::kWarning) warnings_.push_back(message);
        });
    log_.StartCapturingLogs();
  }

  /** How many of the captured warnings contain `needle`. */
  size_t count(const std::string& needle) const {
    size_t found = 0;
    for (const std::string& warning : warnings_) {
      if (absl::StrContains(warning, needle)) ++found;
    }
    return found;
  }

 private:
  // Declared before the mock so that it outlives it: the mock stops capturing in its destructor.
  std::vector<std::string> warnings_;
  absl::ScopedMockLog log_{absl::MockLogDefault::kIgnoreUnexpected};
};

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
    const std::string taskFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
    referenceFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml").value();
    urdfFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();

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
    // The shipped Atlas takes its schedule from the gait schedule; the planner is switched on by name. ASSERTed rather
    // than substituted blindly, so that a renamed key cannot leave this fixture testing the gait schedule.
    const std::string shippedSource = "\ncontactScheduleSource: gait_schedule\n";
    ASSERT_NE(content.find(shippedSource), std::string::npos) << "the shipped task file no longer selects the gait schedule";
    content.replace(content.find(shippedSource), shippedSource.size(), "\ncontactScheduleSource: contact_planner\n");
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
    vector_t target = vector_t::Zero(state.size());
    target.segment(6, 6) = state.segment(6, 6);
    target(0) = commandedVelocityX;
    planWithTarget(time, target);
  }

  /** The same cycle for a whole published target state (held constant over the horizon). */
  void planWithTarget(scalar_t time, const vector_t& target) {
    const vector_t state = interface_->getInitialState();
    const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
    const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
    referenceManager_->setTargetTrajectories(TargetTrajectories({time}, {target}, {vector_t::Zero(inputDim)}));
    referenceManager_->preSolverRun(time, time + horizon, state, ModeNumber::STANCE);
    module_->preSolverRun(time, time + horizon, state, *referenceManager_);
    referenceManager_->preSolverRun(time + 0.02, time + 0.02 + horizon, state, ModeNumber::STANCE);
  }

  /** Horizontal center of mass of the full model at the generalized coordinates of `state`. */
  vector2_t centerOfMass(const vector_t& state) {
    PinocchioInterface& pinocchioInterface = interface_->getPinocchioInterface();
    const vector_t q = interface_->getEffectiveMpcRobotModel().getGeneralizedCoordinates(state);
    pinocchio::centerOfMass(pinocchioInterface.getModel(), pinocchioInterface.getData(), q, /*computeSubtreeComs=*/false);
    return pinocchioInterface.getData().com[0].head<2>();
  }

  /**
   * The fixture's task and planner files, edited by `editTask` and `editPlanning`, in a directory of their own (the
   * planner file is found by its fixed name beside the task file); returns the task file.
   */
  std::string writeVariant(const std::string& name,
                           const std::function<void(std::string&)>& editTask,
                           const std::function<void(std::string&)>& editPlanning) const {
    const std::function<std::string(const std::string&)> readFile = [](const std::string& path) {
      std::ifstream in(path);
      return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const std::filesystem::path dir = std::filesystem::path(tmpDir_) / name;
    std::filesystem::create_directories(dir);
    std::string task = readFile(tmpTaskFile_);
    std::string planning = readFile(tmpContactPlanningFile_);
    editTask(task);
    editPlanning(planning);
    const std::string taskFile = (dir / "task.yaml").string();
    std::ofstream(taskFile) << task;
    std::ofstream((dir / kContactPlanningConfigFileName).string()) << planning;
    return taskFile;
  }

  /**
   * The model's center of mass above the mean height of its soles at the initial state, computed here from Pinocchio
   * directly - forward kinematics and the contact frames looked up by name - rather than by the interface.
   */
  scalar_t independentComHeight() const {
    PinocchioInterface pinocchio(interface_->getPinocchioInterface());
    const pinocchio::ModelTpl<scalar_t>& model = pinocchio.getModel();
    pinocchio::DataTpl<scalar_t>& data = pinocchio.getData();
    const vector_t q = interface_->getEffectiveMpcRobotModel().getGeneralizedCoordinates(interface_->getInitialState());
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    const scalar_t comZ = pinocchio::centerOfMass(model, data, q)(2);
    const std::vector<std::string>& feet = interface_->modelSettings().contactNames;
    scalar_t footZ = 0.0;
    for (const std::string& foot : feet) {
      footZ += data.oMf[model.getFrameId(foot)].translation()(2) / static_cast<scalar_t>(feet.size());
    }
    return comZ - footZ;
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

  // Each iteration plans on a time base of its own, so that a plan left active by the first cannot answer for the second.
  scalar_t time = 0.0;
  for (const bool headingModel : {true, false}) {
    ContactPlanningConfig config = module_->getConfig();
    config.setHeadingModel(headingModel);
    ASSERT_EQ(module_->setConfig(config), absl::OkStatus());
    referenceManager_->setTargetTrajectories(TargetTrajectories({time}, {target}, {vector_t::Zero(inputDim)}));
    referenceManager_->preSolverRun(time, time + horizon, state, ModeNumber::STANCE);

    const ContactPlannerInput input = referenceManager_->makePlannerInput(time, state, referenceManager_->commandedVelocity());
    EXPECT_NEAR(input.headingRateCommand, referenceManager_->commandedYawRate(), 1e-9) << "heading model " << (headingModel ? "on" : "off");
    EXPECT_GT(std::abs(input.headingRateCommand), 1e-6)
        << "the yaw command must reach the planner with the heading model " << (headingModel ? "on" : "off");

    // And reaching the input is not the point: the yaw command alone has to start the gait. The plan made in THIS
    // iteration is activated and must contain a swing.
    const size_t plansBefore = module_->getStatistics().numPlans;
    planWithTarget(time, target);
    ASSERT_GT(module_->getStatistics().numPlans, plansBefore) << "no plan was made";
    ASSERT_TRUE(referenceManager_->hasActivePlan());
    const ContactPlan& plan = *referenceManager_->getActiveContactPlan();
    EXPECT_NEAR(plan.startTime, time, 1e-9) << "the active plan is not the one made in this iteration";
    bool swings = false;
    for (const contact_flag_t& contacts : plan.contacts) {
      swings = swings || !contacts[0] || !contacts[1];
    }
    EXPECT_TRUE(swings) << "a yaw command alone must start the gait, heading model " << (headingModel ? "on" : "off");
    time += 10.0;
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
  const std::optional<Swing> first = firstSwingAfter(schedule, /*after=*/0.02);
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

/**
 * The last swung foot outlives the executed schedule's history window. The reference manager keeps the applied schedule
 * only from one horizon before the solver time on, so about a horizon into a stand the schedule no longer holds the
 * robot's last lift-off; the foot used to be read off that schedule alone and became -1 (unknown), which cost the H-LIP
 * blend its zero-command orbit and made the first step out of the stand ignore the alternation. makePlannerInput() reads
 * it from the manager's LiftOffHistory instead: walk, stand for longer than a horizon, and it must still be the foot of the
 * last lift-off the robot executed.
 */
TEST_F(HlipPlanningIntegrationTest, TheLastSwungFootOutlivesTheSchedulesHistoryWindow) {
  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();
  const scalar_t cyclePeriod = 0.02;

  // The last lift-off of the schedule the robot executed, recorded independently of the manager at every cycle.
  int lastLiftOffFoot = -1;
  scalar_t lastLiftOffTime = -std::numeric_limits<scalar_t>::infinity();
  const std::function<void(scalar_t)> recordExecutedLiftOffs = [&](scalar_t now) {
    const ModeSchedule& schedule = referenceManager_->getModeSchedule();
    for (size_t event = 0; event < schedule.eventTimes.size() && event + 1 < schedule.modeSequence.size(); ++event) {
      if (schedule.eventTimes[event] > now) break;
      const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[event]);
      const contact_flag_t after = modeNumber2StanceLeg(schedule.modeSequence[event + 1]);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        if (before[foot] && !after[foot] && schedule.eventTimes[event] >= lastLiftOffTime) {
          lastLiftOffTime = schedule.eventTimes[event];
          lastLiftOffFoot = static_cast<int>(foot);
        }
      }
    }
  };
  vector_t target = vector_t::Zero(state.size());
  target.segment(6, 6) = state.segment(6, 6);
  scalar_t time = 0.0;
  const std::function<void()> runCycle = [&]() {
    referenceManager_->preSolverRun(time, time + horizon, state, ModeNumber::STANCE);
    module_->preSolverRun(time, time + horizon, state, *referenceManager_);
    recordExecutedLiftOffs(time);
    time += cyclePeriod;
  };

  // Walk.
  target(0) = 0.5;
  referenceManager_->setTargetTrajectories(TargetTrajectories({time}, {target}, {vector_t::Zero(inputDim)}));
  while (time < 1.5) runCycle();
  ASSERT_GE(lastLiftOffFoot, 0) << "the commanded walk must have lifted a foot";

  // Stand, until the last lift-off lies more than two horizons in the past.
  target(0) = 0.0;
  referenceManager_->setTargetTrajectories(TargetTrajectories({time}, {target}, {vector_t::Zero(inputDim)}));
  const scalar_t giveUpAt = time + 10.0;
  while (time < lastLiftOffTime + 2.0 * horizon && time < giveUpAt) runCycle();
  ASSERT_LT(time, giveUpAt) << "the robot never stopped stepping at a zero command";
  referenceManager_->preSolverRun(time, time + horizon, state, ModeNumber::STANCE);

  // Positive control: the schedule itself no longer holds a lift-off, so read off it alone the foot is unknown.
  const ModeSchedule& schedule = referenceManager_->getModeSchedule();
  EXPECT_FALSE(firstSwingAfter(schedule, -std::numeric_limits<scalar_t>::infinity()).has_value())
      << "the stand must outlast the schedule's history window for this test to mean anything";
  const ContactPlanningConfig config = referenceManager_->getConfig();
  ContactPlannerInput fromTheScheduleAlone;
  fromTheScheduleAlone.time = time;
  fromTheScheduleAlone.committedUntil = time;
  LiftOffHistory freshHistory;
  fillPlannerInputFromSchedule(schedule, config.planner.dt, std::max(0, config.planner.numNodes - 1), freshHistory, fromTheScheduleAlone);
  EXPECT_EQ(fromTheScheduleAlone.lastSwungFoot, -1);

  const ContactPlannerInput input = referenceManager_->makePlannerInput(time, state, vector2_t::Zero());
  EXPECT_EQ(input.lastSwungFoot, lastLiftOffFoot)
      << "the last lift-off was at t = " << lastLiftOffTime << ", the input made at t = " << time;
}

TEST_F(HlipPlanningIntegrationTest, TheGaitAdvancesAtTheCommandedVelocity) {
  const scalar_t commandedVelocityX = 0.5;
  planAt(/*time=*/0.0, commandedVelocityX);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();
  const ContactPlanningConfig config = referenceManager_->getConfig();

  // Both feet end the horizon ahead of where they started.
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    EXPECT_GT(plan.footholds.back()[foot].x(), plan.footholds.front()[foot].x()) << "foot " << foot << " must travel forward";
  }

  // And so does the reduced model. The average falls short of the command over this horizon because the first step of
  // the deadbeat law is placed behind the center of mass, which is how the H-LIP accelerates out of a standstill; what
  // matters is that the plan is walking forward at a speed of the commanded order.
  const scalar_t horizon = config.horizon();
  const scalar_t averageVelocity = (plan.comPosition.back().x() - plan.comPosition.front().x()) / horizon;
  EXPECT_GT(averageVelocity, 0.3 * commandedVelocityX);
  EXPECT_LT(averageVelocity, 1.2 * commandedVelocityX);
}

TEST_F(HlipPlanningIntegrationTest, ThePlannedComReplacesTheTargetComReference) {
  // The lateral orbit the deadbeat step regulates to needs the center of mass to fall towards the swing foot. The
  // target trajectory built from the operator's command asks for a straight line with no lateral velocity, so unless
  // the plan's center of mass is written into the reference the whole-body MPC is asked for the opposite motion and
  // the planner narrows the step until the robot falls sideways.
  const ContactPlanningConfig config = referenceManager_->getConfig();
  ASSERT_TRUE(config.formulation.hasExecutionRule(term::kPlannedComOverride)) << "the override must be enabled by default";

  planAt(0.0, 0.5);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();

  const TargetTrajectories& target = referenceManager_->getTargetTrajectories();
  ASSERT_FALSE(target.timeTrajectory.empty());
  const MpcRobotModelBase<scalar_t>& robotModel = interface_->getEffectiveMpcRobotModel();

  // The rule writes two channels. The velocity is the plan's CoM velocity; the position is carried through the base,
  // as the planned CoM minus the CoM-base offset measured at the cycle that wrote it (the initial state here).
  const vector_t state = interface_->getInitialState();
  const vector2_t comOffsetFromBase = centerOfMass(state) - robotModel.getBasePosition(state).head<2>();
  size_t comparedPoints = 0;
  for (size_t i = 0; i < target.timeTrajectory.size(); ++i) {
    const scalar_t time = target.timeTrajectory[i];
    if (time < plan.startTime || time > plan.endTime()) continue;
    const std::optional<vector2_t> plannedVelocity = plan.comVelocityAtTime(time);
    const std::optional<vector2_t> plannedPosition = plan.comPositionAtTime(time);
    ASSERT_TRUE(plannedVelocity.has_value() && plannedPosition.has_value());
    const vector3_t referenceVelocity = robotModel.getBaseComLinearVelocity(target.stateTrajectory[i]);
    EXPECT_NEAR(referenceVelocity.x(), plannedVelocity->x(), 1e-9) << "at t=" << time;
    EXPECT_NEAR(referenceVelocity.y(), plannedVelocity->y(), 1e-9) << "at t=" << time;
    const vector2_t referenceBase = robotModel.getBasePosition(target.stateTrajectory[i]).head<2>();
    EXPECT_NEAR(referenceBase.x() + comOffsetFromBase.x(), plannedPosition->x(), 1e-9) << "at t=" << time;
    EXPECT_NEAR(referenceBase.y() + comOffsetFromBase.y(), plannedPosition->y(), 1e-9) << "at t=" << time;
    ++comparedPoints;
  }
  EXPECT_GT(comparedPoints, 0U) << "the target must have a point inside the plan's horizon";

  // And what it asks for is genuinely not a straight line: the orbit the footholds were placed for swings the center
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
  // reference is the center of the terminal support - the stance foot itself during single support - and at its weight
  // it wins: the center of mass is held over the foot, the planner reads a state with no lateral velocity and narrows
  // the next step to minStepWidth, and the robot falls sideways. With a plan active the reference must be the plan's.
  planAt(0.0, 0.5);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();
  const ContactPlanningConfig config = referenceManager_->getConfig();
  const scalar_t omega = config.omega();
  ASSERT_NEAR(plan.omega, omega, 1e-12) << "a plan records the pendulum it was made on";
  const scalar_t time = plan.startTime + 0.5 * config.horizon();

  const std::optional<SwitchedModelReferenceManager::PlannedDcm> plannedDcm = referenceManager_->getPlannedDcm(time);
  ASSERT_TRUE(plannedDcm.has_value());
  EXPECT_NEAR(plannedDcm->omega, omega, 1e-12);
  const std::optional<vector2_t> comPosition = plan.comPositionAtTime(time);
  const std::optional<vector2_t> comVelocity = plan.comVelocityAtTime(time);
  ASSERT_TRUE(comPosition.has_value() && comVelocity.has_value());
  const vector2_t expected = *comPosition + *comVelocity / omega;
  EXPECT_NEAR(plannedDcm->dcm.x(), expected.x(), 1e-12);
  EXPECT_NEAR(plannedDcm->dcm.y(), expected.y(), 1e-12);

  // And the cost carries it: the blend weight is on, and the reference it blends in is that DCM, on its pendulum.
  const DcmTerminalCost& cost = interface_->getOptimalControlProblem().finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost");
  const vector_t parameters = cost.getParameters(time, referenceManager_->getTargetTrajectories());
  ASSERT_GE(parameters.size(), 11);
  EXPECT_NEAR(parameters(2), omega, 1e-12) << "the robot's DCM is taken on the plan's pendulum";
  EXPECT_NEAR(parameters(8), 1.0, 1e-12) << "the planned reference must be selected while a plan is active";
  EXPECT_NEAR(parameters(9), plannedDcm->dcm.x(), 1e-12);
  EXPECT_NEAR(parameters(10), plannedDcm->dcm.y(), 1e-12);
}

/**
 * Audit finding AC8. On the shipped Atlas configuration (com_and_acom_tracking_cost) the base-position block of Q is zero and
 * the CoM the MPC actually tracks is ComAndAcomTrackingCost's CoM of the REFERENCE configuration, which depends on
 * planned_com_override's base-position write alone - a channel no test compared with the plan. With the operator's
 * target in the measured posture the CoM of every rewritten reference configuration must be the planned CoM.
 */
TEST_F(HlipPlanningIntegrationTest, TheReferenceConfigurationCarriesThePlannedCenterOfMass) {
  const vector_t state = interface_->getInitialState();
  vector_t target = state;
  target.head(6).setZero();
  target(0) = 0.5;
  planWithTarget(/*time=*/0.0, target);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan& plan = *referenceManager_->getActiveContactPlan();
  const TargetTrajectories& reference = referenceManager_->getTargetTrajectories();

  size_t comparedPoints = 0;
  scalar_t largestDeparture = 0.0;
  for (size_t i = 0; i < reference.timeTrajectory.size(); ++i) {
    const scalar_t time = reference.timeTrajectory[i];
    if (time < plan.startTime || time > plan.endTime()) continue;
    const std::optional<vector2_t> planned = plan.comPositionAtTime(time);
    ASSERT_TRUE(planned.has_value());
    const vector2_t referenceCom = centerOfMass(reference.stateTrajectory[i]);
    EXPECT_NEAR(referenceCom.x(), planned->x(), 1e-9) << "at t=" << time;
    EXPECT_NEAR(referenceCom.y(), planned->y(), 1e-9) << "at t=" << time;
    largestDeparture = std::max(largestDeparture, (*planned - centerOfMass(target)).norm());
    ++comparedPoints;
  }
  EXPECT_GT(comparedPoints, 0U);
  // Control: the operator's own reference CoM stands still, so the comparison above is not satisfied trivially.
  EXPECT_GT(largestDeparture, 0.1) << "the plan walks away from the operator's straight-line reference";
}

/**
 * Audit findings A31 / A52 / A63. Once the active plan is older than (plan horizon - MPC horizon) it ends inside the
 * solver horizon. ContactPlan's lookups clamp to the last node, so the overrides used to rewrite every knot past the
 * plan's end with the plan's last position and last, non-zero velocity, and the terminal DCM cost aimed at the last
 * node's DCM. Past the end the reference must be the operator's and the planned DCM must be absent.
 */
TEST_F(HlipPlanningIntegrationTest, PastThePlansEndTheReferenceIsTheOperatorsAndThePlannedDcmIsAbsent) {
  planAt(0.0, 0.5);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  const ContactPlan plan = *referenceManager_->getActiveContactPlan();
  const MpcRobotModelBase<scalar_t>& robotModel = interface_->getEffectiveMpcRobotModel();
  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  vector_t operatorState = vector_t::Zero(state.size());
  operatorState.segment(6, 6) = state.segment(6, 6);
  operatorState(0) = 0.5;

  // No new plan: a cycle whose solver horizon runs half a horizon past the plan's end, the plan itself still usable.
  const scalar_t initTime = plan.endTime() - 0.5 * horizon;
  const scalar_t finalTime = initTime + horizon;
  referenceManager_->preSolverRun(initTime, finalTime, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager_->planReferencesUsable()) << "the plan must still drive the references inside its horizon";
  ASSERT_GT(finalTime, plan.endTime());

  const TargetTrajectories& reference = referenceManager_->getTargetTrajectories();
  size_t tailKnots = 0;
  size_t planKnots = 0;
  for (size_t i = 0; i < reference.timeTrajectory.size(); ++i) {
    const scalar_t time = reference.timeTrajectory[i];
    const vector_t& knot = reference.stateTrajectory[i];
    if (time > plan.endTime() + 1e-9) {
      ++tailKnots;
      EXPECT_NEAR(robotModel.getBasePosition(knot).x(), robotModel.getBasePosition(operatorState).x(), 1e-9) << "at t=" << time;
      EXPECT_NEAR(robotModel.getBasePosition(knot).y(), robotModel.getBasePosition(operatorState).y(), 1e-9) << "at t=" << time;
      EXPECT_NEAR(robotModel.getBaseComLinearVelocity(knot).x(), 0.5, 1e-9) << "at t=" << time;
      EXPECT_NEAR(robotModel.getBaseComLinearVelocity(knot).y(), 0.0, 1e-9) << "at t=" << time;
    } else if (time >= initTime) {
      ++planKnots;
      const std::optional<vector2_t> plannedVelocity = plan.comVelocityAtTime(time);
      ASSERT_TRUE(plannedVelocity.has_value());
      EXPECT_NEAR(robotModel.getBaseComLinearVelocity(knot).y(), plannedVelocity->y(), 1e-9)
          << "inside the plan it still rules, t=" << time;
    }
  }
  EXPECT_GT(tailKnots, 0U) << "the reference must extend past the plan's end to finalTime";
  EXPECT_GT(planKnots, 0U);
  // Control: what the clamped lookups would have written there is far from the operator's reference.
  const vector2_t operatorBase = robotModel.getBasePosition(operatorState).head<2>();
  EXPECT_GT((plan.comPosition.back() - operatorBase).norm(), 0.1);

  EXPECT_TRUE(referenceManager_->getPlannedDcm(initTime).has_value()) << "inside the plan the DCM is the plan's";
  EXPECT_TRUE(referenceManager_->getPlannedDcm(plan.endTime()).has_value()) << "the plan's last node is inside it";
  EXPECT_FALSE(referenceManager_->getPlannedDcm(finalTime).has_value()) << "past the plan's end there is no planned DCM";
  const DcmTerminalCost& cost = interface_->getOptimalControlProblem().finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost");
  EXPECT_NEAR(cost.getParameters(finalTime, reference)(8), 0.0, 1e-12) << "the terminal cost falls back to its own reference";
}

/**
 * The reload path returns a Status that names the key, leaves the running configuration in force when it rejects one,
 * and warns when the planner's shortest swing falls below the swing trajectory planner's swingTimeScale.
 */
TEST_F(HlipPlanningIntegrationTest, ARejectedReloadNamesTheKeyAndKeepsTheRunningConfiguration) {
  const ContactPlanningConfig running = referenceManager_->getConfig();
  ContactPlanningConfig broken = running;
  broken.footholdRegularization.weight = -1.0;
  const absl::Status status = referenceManager_->setConfigStatus(broken);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(status.message(), "foothold_regularization.weight")) << status.message();
  EXPECT_DOUBLE_EQ(referenceManager_->getConfig().footholdRegularization.weight, running.footholdRegularization.weight);
}

/**
 * The module's reload and its factory return the rejection that names the key, keep the running configuration of the
 * module and of its reference manager, and never build from a configuration that does not validate. Both used to throw,
 * and the planner's own reload did so on the worker thread, which cannot catch.
 */
TEST_F(HlipPlanningIntegrationTest, TheModuleRefusesByItsKeyAConfigurationItCannotRun) {
  const ContactPlanningConfig running = module_->getConfig();
  ContactPlanningConfig broken = running;
  broken.hlip.sspDuration = -0.25;

  const absl::Status reload = module_->setConfig(broken);
  EXPECT_EQ(reload.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(reload.message(), "hlip.sspDuration")) << reload;
  EXPECT_DOUBLE_EQ(module_->getConfig().hlip.sspDuration, running.hlip.sspDuration) << "the module took a refused configuration";
  EXPECT_DOUBLE_EQ(referenceManager_->getConfig().hlip.sspDuration, running.hlip.sspDuration)
      << "the reference manager took a refused configuration";
  const size_t plansBefore = module_->getStatistics().numPlans;
  planAt(0.0, 0.3);
  EXPECT_GT(module_->getStatistics().numPlans, plansBefore) << "the running configuration no longer plans";

  const absl::StatusOr<std::shared_ptr<ContactPlannerModule>> withoutManager =
      ContactPlannerModule::Create(/*referenceManagerPtr=*/nullptr, running);
  EXPECT_EQ(withoutManager.status().code(), absl::StatusCode::kInvalidArgument);
  const absl::StatusOr<std::shared_ptr<ContactPlannerModule>> fromBroken = ContactPlannerModule::Create(referenceManager_, broken);
  EXPECT_EQ(fromBroken.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(fromBroken.status().message(), "hlip.sspDuration")) << fromBroken.status();
  EXPECT_DOUBLE_EQ(referenceManager_->getConfig().hlip.sspDuration, running.hlip.sspDuration)
      << "a refused Create() reconfigured the reference manager it was handed";

  // Positive control: a configuration that validates is applied, to the module and to its reference manager.
  ContactPlanningConfig wider = running;
  wider.hlip.stepWidth = running.hlip.stepWidth + 0.02;
  ASSERT_EQ(module_->setConfig(wider), absl::OkStatus());
  EXPECT_DOUBLE_EQ(module_->getConfig().hlip.stepWidth, wider.hlip.stepWidth);
  EXPECT_DOUBLE_EQ(referenceManager_->getConfig().hlip.stepWidth, wider.hlip.stepWidth);
}

TEST_F(HlipPlanningIntegrationTest, ACadenceShorterThanTheSwingTimeScaleIsWarnedAboutOnReload) {
  const scalar_t swingTimeScale = referenceManager_->getSwingTrajectoryPlanner()->getConfig().swingTimeScale;
  ContactPlanningConfig config = referenceManager_->getConfig();
  ASSERT_EQ(canonicalPlannerName(config.planner.type), planner::kHlip);
  {
    WarningLog log;
    config.hlip.sspDuration = swingTimeScale;
    ASSERT_TRUE(referenceManager_->setConfigStatus(config).ok());
    EXPECT_EQ(log.count("swingTimeScale"), 0U) << "a swing as long as swingTimeScale is not scaled";
  }
  {
    WarningLog log;
    config.hlip.sspDuration = swingTimeScale - 0.05;
    ASSERT_TRUE(referenceManager_->setConfigStatus(config).ok());
    EXPECT_EQ(log.count("swingTimeScale"), 1U) << "every swing is now scaled down in height and velocity";
  }
}

/**
 * The other side of the same coupling: a task.yaml reload replaces the swing trajectory planner's configuration without
 * passing through the reference manager, so a swingTimeScale raised from the GUI above the planned swings has to be
 * noticed at the next solver run - and reported once per change, not once per control cycle.
 */
TEST_F(HlipPlanningIntegrationTest, ASwingTimeScaleRaisedByATaskFileReloadIsWarnedAboutOnce) {
  const std::shared_ptr<SwingTrajectoryPlanner>& swingPlanner = referenceManager_->getSwingTrajectoryPlanner();
  SwingTrajectoryPlanner::Config swingConfig = swingPlanner->getConfig();
  const scalar_t shortestSwing = referenceManager_->getConfig().shortestPlannedSwingDuration();
  ASSERT_LE(swingConfig.swingTimeScale, shortestSwing + 1e-9) << "the shipped swingTimeScale must fit the shipped cadence";
  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;

  WarningLog log;
  planAt(0.0, 0.0);
  EXPECT_EQ(log.count("swingTimeScale"), 0U) << "nothing changed, nothing to report";

  swingConfig.swingTimeScale = shortestSwing + 0.05;
  swingPlanner->setConfig(swingConfig);  // what MpcParameterUpdaterModule does when task.yaml is saved
  referenceManager_->preSolverRun(/*initTime=*/0.04, 0.04 + horizon, state, ModeNumber::STANCE);
  EXPECT_EQ(log.count("swingTimeScale"), 1U) << "the next solver run must notice that every swing is now scaled down";
  referenceManager_->preSolverRun(/*initTime=*/0.06, 0.06 + horizon, state, ModeNumber::STANCE);
  EXPECT_EQ(log.count("swingTimeScale"), 1U) << "once per change, not once per control cycle";

  swingConfig.swingTimeScale = shortestSwing;
  swingPlanner->setConfig(swingConfig);
  referenceManager_->preSolverRun(/*initTime=*/0.08, 0.08 + horizon, state, ModeNumber::STANCE);
  EXPECT_EQ(log.count("swingTimeScale"), 1U) << "a swingTimeScale that fits again is not reported";
}

/** The two rules the reference manager builds itself agree with the registry about the block they need. */
TEST_F(HlipPlanningIntegrationTest, TheModelRulesRequireWhatTheRegistrySays) {
  const MpcRobotModelBase<scalar_t>& robotModel = interface_->getEffectiveMpcRobotModel();
  const PlannedHeadingOverride heading(robotModel, /*acom=*/nullptr);
  const PlannedComOverride com(robotModel);
  EXPECT_EQ(heading.requiredBlocks(),
            std::vector<std::string>{requiredModelBlock(TermKind::EXECUTION_RULE, term::kPlannedHeadingOverride)});
  EXPECT_TRUE(com.requiredBlocks().empty());
  EXPECT_TRUE(requiredModelBlock(TermKind::EXECUTION_RULE, term::kPlannedComOverride).empty());
}

/**
 * The pendulum is the model's (AC1): with dcm_terminal_cost.comHeight and shared.comHeight both 0, as shipped, the DCM
 * terminal cost and the planner both run on omega = sqrt(g / h), h the model's center of mass above its soles at the
 * initial state, computed here independently of the interface - and a plan carries that omega. The control: explicit
 * positive heights are used as given, each by its own consumer.
 */
TEST_F(HlipPlanningIntegrationTest, TheDcmCostAndThePlannerRunOnTheModelsPendulum) {
  const scalar_t height = independentComHeight();
  ASSERT_GT(height, 0.9) << "Atlas's center of mass stands about 1.08 m above its soles";
  ASSERT_LT(height, 1.2);
  EXPECT_NEAR(interface_->getNominalComHeight(), height, 1e-9);

  const DcmTerminalCost& cost = interface_->getOptimalControlProblem().finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost");
  const scalar_t dcmOmega = std::sqrt(cost.getConfig().gravity / height);
  EXPECT_NEAR(cost.getConfig().comHeight, height, 1e-9) << "dcm_terminal_cost.comHeight: 0 must be the model's pendulum";
  EXPECT_NEAR(cost.getConfig().omega(), dcmOmega, 1e-9);
  const ContactPlanningConfig config = referenceManager_->getConfig();
  const scalar_t plannerOmega = std::sqrt(config.shared.gravity / height);
  EXPECT_NEAR(config.shared.comHeight, height, 1e-9) << "shared.comHeight: 0 must be the model's pendulum";
  EXPECT_NEAR(config.omega(), plannerOmega, 1e-9);
  EXPECT_NEAR(module_->getConfig().omega(), plannerOmega, 1e-9);
  planAt(0.0, 0.3);
  ASSERT_TRUE(referenceManager_->hasActivePlan());
  EXPECT_NEAR(referenceManager_->getActiveContactPlan()->omega, plannerOmega, 1e-9) << "the plan must carry the omega it was made on";
  // A reload that moves the planner's pendulum does not re-interpret the plan already active: its DCM stays the DCM of
  // the pendulum it was made on.
  const scalar_t time = referenceManager_->getActiveContactPlan()->startTime + 0.2;
  const std::optional<SwitchedModelReferenceManager::PlannedDcm> before = referenceManager_->getPlannedDcm(time);
  ASSERT_TRUE(before.has_value());
  ContactPlanningConfig lower = config;
  lower.shared.comHeight = 0.8;
  ASSERT_EQ(referenceManager_->setConfigStatus(lower), absl::OkStatus());
  ASSERT_GT(std::abs(referenceManager_->getConfig().omega() - plannerOmega), 0.3) << "the reload must move the pendulum";
  const std::optional<SwitchedModelReferenceManager::PlannedDcm> after = referenceManager_->getPlannedDcm(time);
  ASSERT_TRUE(after.has_value());
  EXPECT_NEAR(after->omega, plannerOmega, 1e-12) << "the active plan's DCM was re-taken on the reloaded pendulum";
  EXPECT_NEAR((after->dcm - before->dcm).norm(), 0.0, 1e-12);
  ASSERT_EQ(referenceManager_->setConfigStatus(config), absl::OkStatus());
  // Positive control on the magnitude: the hand-set 0.85 m this robot used to ship gives a clearly different omega.
  EXPECT_GT(std::sqrt(9.81 / 0.85) - plannerOmega, 0.3);

  // Explicit heights: each consumer takes its own as given. The foot position weights are raised here as well, which is
  // the control of the weightless-foothold warning of the next test: with them raised, nothing is reported.
  const std::string variant = writeVariant(
      "explicit_pendulum",
      [](std::string& task) {
        task = std::regex_replace(task, std::regex("\n  comHeight: [^\n]*"), "\n  comHeight: 0.9");
        const size_t block = task.find("\ntask_space_foot_cost_weights:");
        ASSERT_NE(block, std::string::npos);
        const std::string tail = std::regex_replace(task.substr(block), std::regex("\n  pos_([xy]): 0\n"), "\n  pos_$1: 30\n",
                                                    std::regex_constants::format_first_only);
        task = task.substr(0, block) +
               std::regex_replace(tail, std::regex("\n  pos_y: 0\n"), "\n  pos_y: 30\n", std::regex_constants::format_first_only);
      },
      [](std::string& planning) {
        planning = std::regex_replace(planning, std::regex("\n    comHeight: [^\n]*"), "\n    comHeight: 0.95");
      });
  MessageLog log;
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(variant, urdfFile_, referenceFile_);
  ASSERT_TRUE(created.ok()) << created.status();
  const std::unique_ptr<CentroidalMpcInterface> explicitInterface = *std::move(created);
  const DcmTerminalCost& explicitCost = explicitInterface->getOptimalControlProblem().finalCostPtr->get<DcmTerminalCost>("dcmTerminalCost");
  EXPECT_NEAR(explicitCost.getConfig().comHeight, 0.9, 1e-12);
  EXPECT_NEAR(explicitCost.getConfig().omega(), std::sqrt(9.81 / 0.9), 1e-12);
  EXPECT_NEAR(explicitInterface->getContactPlannerModulePtr()->getConfig().shared.comHeight, 0.95, 1e-12);
  EXPECT_NEAR(explicitInterface->getContactPlannerModulePtr()->getConfig().omega(), std::sqrt(9.81 / 0.95), 1e-12);
  EXPECT_EQ(log.count(absl::LogSeverity::kWarning, "the planned footholds carry no weight"), 0U)
      << "with task_space_foot_cost_weights.pos_x / pos_y raised, the planned footholds do reach the MPC";
}

/**
 * With the contact planner on, the planned footholds reach the whole-body MPC only through task_space_foot_cost's xy
 * weights, which the Atlas ships at 0 (AC2): the interface says so at start-up, naming the keys to raise. The start-up
 * log also names the planner planner.type actually selects rather than calling every planner the mixed-integer one
 * (A34). The control of the warning - no warning with the weights raised - is in the test above.
 */
TEST_F(HlipPlanningIntegrationTest, WeightlessFootholdsAreWarnedAboutAndTheLogNamesTheConfiguredPlanner) {
  MessageLog log;
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(tmpTaskFile_, urdfFile_, referenceFile_);
  ASSERT_TRUE(created.ok()) << created.status();
  EXPECT_EQ(log.count(absl::LogSeverity::kWarning, "task_space_foot_cost_weights.pos_x"), 1U)
      << "contactScheduleSource: contact_planner with pos_x = pos_y = 0 must be reported once, naming the keys to raise";
  EXPECT_EQ(log.count(absl::LogSeverity::kInfo, "planner.type 'hlip'"), 1U) << "the start-up log must name the configured planner";
  EXPECT_EQ(log.count(absl::LogSeverity::kInfo, "mixed-integer contact planning"), 0U)
      << "the shipped planner is the closed-form H-LIP, not the mixed-integer program";
}

}  // namespace ocs2::humanoid
