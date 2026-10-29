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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/constraint/SwingLandingAlignmentConstraint.h"
#include "humanoid_centroidal_mpc/constraint/TerrainNonPenetrationConstraint.h"
#include "humanoid_centroidal_mpc/multi_contact/PlannedMotion.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_multi_contact/plan/PlanFile.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_core/ResourcePaths.h"
#include "support/MultiContactPlanLoop.h"
#include "support/MultiContactTestFiles.h"
#include "support/StairStepPlan.h"
#include "support/TypedConfigFiles.h"

/*
 * The centroidal MPC tracking an offline multi-contact plan in the dummy
 * simulator's loop, faster than real time (design sections 8.2 and 9,
 * "Dummy sim", humanoid_nmpc/docs/multi_contact_planning/README.md): the DRC
 * Atlas's multi-contact task file - all seventeen contact candidates, the
 * wrists unlocked - on the planned-motion reference manager, with the
 * generalized swing planner's contact_swing terms, the terrain non-penetration
 * barrier and the patch-frame cones, its plant the projected rollout of the
 * policy.
 *
 * For every plan it tracks to the end: every solve succeeds, the robot ends on
 * the plan's final stance with every pad on its target patch, its base where
 * the plan's ends, and the stability certifier passes at every sample of the
 * TRACKED trajectory (the momentum rate in the contact wrench cone of the
 * stance, every active pad on the terrain within 5 mm).
 *
 * The plans: milestone 1's hand-written sway on the box_climb scene, a
 * hand-written feet-only step onto a 0.10 m stair (support/StairStepPlan.h),
 * the stairs scenario's shipped plan (plans/stairs.binpb, the planner's),
 * once and played "forward_and_reverse" - up and backward down, three times -
 * as the stairs dummy sim plays it, and the box_climb scenario's shipped plan
 * (plans/box_climb.binpb, planner B's), hands, knees and feet onto the box.
 * Generates the multi-contact model's CppAD libraries on a cold cache, once
 * for every case.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kBoxClimbScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto";
constexpr char kSwayPlan[] =
    "humanoid_nmpc/humanoid_multi_contact/app/testdata/"
    "box_climb_sway.textproto";
constexpr char kStairsScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/stairs.textproto";
constexpr char kStairsPlan[] =
    "humanoid_nmpc/humanoid_multi_contact/plans/stairs.binpb";
constexpr char kBoxClimbPlan[] =
    "humanoid_nmpc/humanoid_multi_contact/plans/box_climb.binpb";
constexpr char kGetUpSupineScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/"
    "get_up_supine.textproto";
constexpr char kGetUpSupinePlan[] =
    "humanoid_nmpc/humanoid_multi_contact/plans/get_up_supine.binpb";
constexpr char kGetUpProneScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/"
    "get_up_prone.textproto";
constexpr char kGetUpPronePlan[] =
    "humanoid_nmpc/humanoid_multi_contact/plans/get_up_prone.binpb";
constexpr char kStairStepScenario[] =
    "humanoid_nmpc/humanoid_centroidal_mpc/test/testdata/multi_contact/"
    "stair_step_scenario.textproto";
/** [s] How long past the plan's end the loop runs, and how long the clock may
 * hold in all. */
constexpr scalar_t kSettleTime = 1.0;
constexpr scalar_t kLongestHold = 5.0;
/** [m] How far a final pad may be from its target, along the patch and along
 * its normal, and the final base from the plan's. */
constexpr scalar_t kPadTangentialTolerance = 0.03;
constexpr scalar_t kPadNormalTolerance = 0.005;
constexpr scalar_t kBaseTolerance = 0.05;

/** The tracked trajectory and how the solves went. */
struct TrackingRun {
  std::vector<scalar_t> planTimes;
  std::vector<scalar_t> times;
  std::vector<vector_t> states;
  std::vector<size_t> modes;
  scalar_t heldTime = 0.0;  // [s]
  // The largest cost of a solve; std::nullopt before the first. A cost can be
  // negative: the relaxed log barriers of the cones and limits, -mu ln(h), are
  // below zero wherever the constraint value h exceeds 1, so a 0 start would
  // hide every solve of a plan that stands still (the sway's).
  std::optional<scalar_t> worstCost;
};

/** The multi-contact task file tracking `planFile` of `scenarioFile`. */
CentroidalMpcConfig trackingConfig(const std::string& scenarioFile,
                                   const std::string& planFile) {
  CentroidalMpcConfig config =
      checkedValue(loadConfigOf(atlasMultiContactFiles()));
  config.task.multi_contact_plan.scenario_file = scenarioFile;
  config.task.multi_contact_plan.plan_file = planFile;
  return config;
}

/** Runs `loop` until its plan time has reached `planEnd` [s] and kSettleTime
 * after; fails the test on a failed solve or past the time `limit` [s]. */
TrackingRun runUntil(MultiContactPlanLoop& loop,
                     scalar_t planEnd,
                     scalar_t limit) {
  TrackingRun run;
  std::optional<scalar_t> endedAt;
  while (!endedAt.has_value() ||
         loop.observation().time < *endedAt + kSettleTime) {
    const scalar_t time = loop.observation().time;
    if (time > limit) {
      ADD_FAILURE() << "the plan did not reach " << planEnd << " s within "
                    << limit << " s: it is at plan time "
                    << loop.referenceManager().planTimeAt(time) << " s";
      break;
    }
    run.times.push_back(time);
    run.states.push_back(loop.observation().state);
    const absl::StatusOr<MultiContactPlanLoopStep> step = loop.step();
    if (!step.ok()) {
      ADD_FAILURE() << "t = " << time << ": " << step.status();
      break;
    }
    // The mode the step was rolled out in: the schedule's in the middle of the
    // step, since the stances change at multiples of the period (at a change
    // itself the schedule answers the mode before it).
    run.modes.push_back(loop.referenceManager().getModeSchedule().modeAtTime(
        time + 0.5 * loop.period()));
    run.planTimes.push_back(step->planTime);
    if (step->holding) run.heldTime += loop.period();
    run.worstCost = std::max(run.worstCost.value_or(step->performance.cost),
                             step->performance.cost);
    if (!endedAt.has_value() && step->planTime >= planEnd) endedAt = time;
  }
  return run;
}

/** Runs `loop` until the plan has ended and kSettleTime after; fails the test
 * on a failed solve or a clock held longer than kLongestHold. */
TrackingRun runToTheEnd(MultiContactPlanLoop& loop) {
  const PlannedMotion& motion = loop.referenceManager().plannedMotion();
  return runUntil(
      loop, motion.endTime(),
      motion.endTime() - motion.startTime() + kSettleTime + kLongestHold);
}

/** The configuration of the MPC state `state` by joint name. */
multi_contact::RobotConfiguration configurationOf(
    const MpcRobotModelBase<scalar_t>& model, const vector_t& state) {
  multi_contact::RobotConfiguration configuration;
  configuration.basePosition = model.getBasePosition(state);
  configuration.baseOrientation.coeffs() = model.getBaseOrientation(state);
  configuration.jointNames = model.modelSettings.mpcModelJointNames;
  configuration.jointPositions = model.getJointAngles(state);
  return configuration;
}

/** The tracked trajectory of `run` as the certifier's knots. The momentum
 * rate of a sample is the forward difference of the momenta over the step the
 * simulator took from it (of the last, the backward one): the mean rate of a
 * rollout of the policy in that sample's mode, since the plan's stances change
 * at multiples of the loop's period. A central difference would average two
 * modes at every change. */
std::vector<multi_contact::CertifierKnot> certifierKnotsOf(
    const TrackingRun& run, const CentroidalMpcInterface& interface) {
  const MpcRobotModelBase<scalar_t>& model = interface.getMpcRobotModel();
  const scalar_t mass = interface.getCentroidalModelInfo().robotMass;
  const PlanContactOrder& order =
      interface.getPlannedMotionReferenceManagerPtr()->plannedMotion().order();
  std::vector<multi_contact::CertifierKnot> knots;
  const size_t n = std::min(run.states.size(), run.modes.size());
  if (n < 2) return knots;
  for (size_t k = 0; k < n; ++k) {
    const size_t previous = k + 1 < n ? k : k - 1;
    const size_t next = previous + 1;
    const scalar_t dt = run.times[next] - run.times[previous];
    const vector6_t momentumChange =
        mass * (run.states[next].head<6>() - run.states[previous].head<6>());
    multi_contact::CertifierKnot knot;
    knot.time = run.times[k];
    knot.configuration = configurationOf(model, run.states[k]);
    knot.activeContacts = checkedValue(order.maskOf(run.modes[k]));
    knot.momentumRate.linear = momentumChange.head<3>() / dt;
    knot.momentumRate.angular = momentumChange.tail<3>() / dt;
    knots.push_back(knot);
  }
  return knots;
}

/** How often one kind of certifier failure occurs, and when first and last. */
struct FailureKind {
  int count = 0;
  double first = std::numeric_limits<double>::infinity();
  double last = -std::numeric_limits<double>::infinity();
};

/** The kind of a certifier failure: its text up to the first number, with the
 * collision sphere's link kept. */
std::string failureKindOf(const std::string& failure) {
  const size_t number = failure.find_first_of("0123456789");
  return failure.substr(0, number);
}

/** Certifies the tracked trajectory of `run` on `scenarioFile` and expects
 * every knot to pass; records the worst margins. */
void expectTheTrackedTrajectoryCertifies(
    const TrackingRun& run,
    const CentroidalMpcInterface& interface,
    const std::string& scenarioFile) {
  const multi_contact::LoadedScenario scenario =
      checkedValue(multi_contact::loadScenario(scenarioFile));
  std::unique_ptr<multi_contact::StabilityCertifier> certifier =
      checkedValue(multi_contact::makeCertifier(scenario));
  const multi_contact::config::CertifierReport report =
      checkedValue(certifier->certifyKnots(certifierKnotsOf(run, interface)));
  double worstResidual = 0.0;
  double worstPadDistance = 0.0;
  std::optional<double> smallestStaticMargin;
  std::vector<std::string> failures;
  std::map<std::string, FailureKind> kinds;
  for (const multi_contact::config::CertifierReport::Knot& knot :
       report.knots) {
    worstResidual = std::max(worstResidual, knot.momentum_rate_residual);
    worstPadDistance =
        std::max(worstPadDistance, knot.max_active_contact_distance);
    if (knot.static_margin.has_value()) {
      smallestStaticMargin =
          std::min(smallestStaticMargin.value_or(*knot.static_margin),
                   *knot.static_margin);
    }
    if (!knot.passed && failures.size() < 10) {
      failures.push_back(absl::StrCat("t = ", knot.time, ": ",
                                      absl::StrJoin(knot.failures, "; ")));
    }
    for (const std::string& failure : knot.failures) {
      FailureKind& kind = kinds[failureKindOf(failure)];
      ++kind.count;
      kind.first = std::min(kind.first, knot.time);
      kind.last = std::max(kind.last, knot.time);
    }
  }
  std::vector<std::string> kindLines;
  for (const std::pair<const std::string, FailureKind>& kind : kinds) {
    kindLines.push_back(absl::StrCat(kind.second.count, " x '", kind.first,
                                     "' from t = ", kind.second.first,
                                     " to t = ", kind.second.last));
  }
  EXPECT_TRUE(report.passed) << report.failed_knots << " of "
                             << report.knots.size() << " tracked knots fail:\n"
                             << absl::StrJoin(failures, "\n") << "\nby kind:\n"
                             << absl::StrJoin(kindLines, "\n");
  ::testing::Test::RecordProperty("failed_tracked_knots",
                                  absl::StrCat(report.failed_knots));
  ::testing::Test::RecordProperty("worst_momentum_rate_residual",
                                  absl::StrCat(worstResidual));
  ::testing::Test::RecordProperty("worst_active_pad_distance_m",
                                  absl::StrCat(worstPadDistance));
  if (smallestStaticMargin.has_value()) {
    ::testing::Test::RecordProperty("smallest_static_margin_m",
                                    absl::StrCat(*smallestStaticMargin));
  }
}

/** Expects the robot of `loop` to end on the plan's final stance, every pad on
 * its target patch, the base at the plan's. */
void expectTheGoalIsReached(MultiContactPlanLoop& loop,
                            const TrackingRun& run) {
  CentroidalMpcInterface& interface = loop.interface();
  const PlannedMotion& motion =
      interface.getPlannedMotionReferenceManagerPtr()->plannedMotion();
  const MpcRobotModelBase<scalar_t>& model = interface.getMpcRobotModel();
  const vector_t& finalState = loop.observation().state;
  const size_t finalMode = motion.modeSequence().back();
  ASSERT_FALSE(run.modes.empty());
  EXPECT_EQ(run.modes.back(), finalMode);
  EXPECT_LT((model.getBasePosition(finalState) -
             model.getBasePosition(motion.targets().stateTrajectory.back()))
                .norm(),
            kBaseTolerance)
      << model.getBasePosition(finalState).transpose();
  PinocchioInterface pinocchio = interface.getPinocchioInterface();
  const vector_t q = model.getGeneralizedCoordinates(finalState);
  pinocchio::forwardKinematics(pinocchio.getModel(), pinocchio.getData(), q);
  pinocchio::updateFramePlacements(pinocchio.getModel(), pinocchio.getData());
  const size_t n = interface.modelSettings().contactNames.size();
  for (size_t i = 0; i < n; ++i) {
    if (!contactIsActiveInMode(finalMode, i, n)) continue;
    const std::string& name = interface.modelSettings().contactNames[i];
    const std::optional<PlannedContactPhase> phase =
        motion.phaseAt(i, motion.endTime());
    ASSERT_TRUE(phase.has_value()) << name;
    const vector3_t pad = pinocchio.getData()
                              .oMf[pinocchio.getModel().getFrameId(name)]
                              .translation();
    const vector3_t error = pad - phase->targetPosition;
    const scalar_t normal = phase->normal.dot(error);
    EXPECT_LT(std::abs(normal), kPadNormalTolerance) << name;
    EXPECT_LT((error - normal * phase->normal).norm(), kPadTangentialTolerance)
        << name << ": " << pad.transpose() << " against "
        << phase->targetPosition.transpose();
  }
  // The base followed the plan's, at the plan time of every solve.
  scalar_t worstBaseError = 0.0;
  for (size_t k = 0; k < run.planTimes.size(); ++k) {
    const vector3_t planned = model.getBasePosition(
        model.getDesiredState(motion.targets(), run.planTimes[k]));
    worstBaseError =
        std::max(worstBaseError,
                 (model.getBasePosition(run.states[k]) - planned).norm());
  }
  EXPECT_LT(worstBaseError, kBaseTolerance);
  ::testing::Test::RecordProperty("worst_base_tracking_error_m",
                                  absl::StrCat(worstBaseError));
  ::testing::Test::RecordProperty("held_s", absl::StrCat(run.heldTime));
  ASSERT_TRUE(run.worstCost.has_value());
  ::testing::Test::RecordProperty("worst_cost", absl::StrCat(*run.worstCost));
}

TEST(MultiContactPlanTracking, tracksTheSwayOnTheBoxClimbScene) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(trackingConfig(kBoxClimbScenario, kSwayPlan),
                                   atlasMultiContactFiles().urdfFile));
  const TrackingRun run = runToTheEnd(*loop);
  expectTheGoalIsReached(*loop, run);
  expectTheTrackedTrajectoryCertifies(run, loop->interface(),
                                      kBoxClimbScenario);
}

TEST(MultiContactPlanTracking, aStandingSolveCostsLessThanZero) {
  // A solve's cost is its last iteration's PerformanceIndex::cost: the costs
  // and the soft constraints' penalties. Standing still, the tracking costs are
  // small and the relaxed log barriers of the wrench cones and the joint
  // limits, -mu ln(h) with h in newtons or radians from the bound, are below
  // zero, so the sum is too: a worst cost that starts at 0 records 0 for every
  // run that stands (TrackingRun::worstCost).
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(trackingConfig(kBoxClimbScenario, kSwayPlan),
                                   atlasMultiContactFiles().urdfFile));
  HUMANOID_ASSERT_OK_AND_ASSIGN(const MultiContactPlanLoopStep step,
                                loop->step());
  EXPECT_TRUE(std::isfinite(step.performance.cost));
  EXPECT_LT(step.performance.cost, 0.0);
  // No augmented Lagrangian: the merit is the cost.
  EXPECT_EQ(step.performance.merit, step.performance.cost);
}

TEST(MultiContactPlanTracking, stepsOntoTheStairOnAHandWrittenFeetOnlyPlan) {
  // The plan, made on the multi-contact model, certified, and written where
  // the MPC reads it from.
  const CentroidalRobotFiles files = atlasMultiContactFiles();
  std::unique_ptr<CentroidalMpcInterface> models =
      checkedValue(CentroidalMpcInterface::CreateControllerModels(
          files.taskFile, files.urdfFile, files.referenceFile));
  const multi_contact::LoadedScenario scenario =
      checkedValue(multi_contact::loadScenario(kStairStepScenario));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::config::MultiContactPlan plan,
      makeStairStepPlan(models->getPinocchioInterface(),
                        models->getMpcRobotModel(), scenario));
  std::unique_ptr<multi_contact::StabilityCertifier> certifier =
      checkedValue(multi_contact::makeCertifier(scenario));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::config::CertifierReport planned,
      certifier->certify(plan));
  std::vector<std::string> plannedFailures;
  for (const multi_contact::config::CertifierReport::Knot& knot :
       planned.knots) {
    if (knot.passed) continue;
    plannedFailures.push_back(absl::StrCat("t = ", knot.time, ": ",
                                           absl::StrJoin(knot.failures, "; ")));
  }
  EXPECT_TRUE(planned.passed)
      << planned.failed_knots << " knots of the hand-written plan fail:\n"
      << absl::StrJoin(plannedFailures, "\n");
  const std::string planFile =
      (std::filesystem::path(::testing::TempDir()) / "stair_step.binpb")
          .string();
  HUMANOID_ASSERT_OK(multi_contact::savePlanFile(plan, planFile));

  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(trackingConfig(kStairStepScenario, planFile),
                                   files.urdfFile));
  const TrackingRun run = runToTheEnd(*loop);
  expectTheGoalIsReached(*loop, run);
  expectTheTrackedTrajectoryCertifies(run, loop->interface(),
                                      kStairStepScenario);
}

TEST(MultiContactPlanTracking, climbsTheStairsOfTheShippedPlan) {
  // The stairs scenario's own plan (make plan-drc-atlas-stairs), which the
  // stairs dummy sim tracks.
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(trackingConfig(kStairsScenario, kStairsPlan),
                                   atlasMultiContactFiles().urdfFile));
  const TrackingRun run = runToTheEnd(*loop);
  expectTheGoalIsReached(*loop, run);
  expectTheTrackedTrajectoryCertifies(run, loop->interface(), kStairsScenario);
}

TEST(MultiContactPlanTracking, climbsOntoTheBoxOfTheShippedPlan) {
  // The box_climb scenario's own plan (make plan-drc-atlas-box-climb): hands,
  // knees, then feet onto the box, which the box_climb dummy sim tracks. Every
  // sample certifies because planner B checked what lies between its knots
  // too: the configurations of the MPC's reference interpolation and the
  // straight paths of the free pads (design section 5.5).
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(
          trackingConfig(kBoxClimbScenario, kBoxClimbPlan),
          atlasMultiContactFiles().urdfFile));
  const TrackingRun run = runToTheEnd(*loop);
  expectTheGoalIsReached(*loop, run);
  expectTheTrackedTrajectoryCertifies(run, loop->interface(),
                                      kBoxClimbScenario);
}

TEST(MultiContactPlanTracking, getsUpFromSupineOfTheShippedPlan) {
  // The get_up_supine scenario's own plan (make plan-drc-atlas-get-up-supine):
  // from the back, a roll onto the right edge of the torso, the right foot and
  // hand down, onto the feet and the hands, a squat, up.
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(
          trackingConfig(kGetUpSupineScenario, kGetUpSupinePlan),
          atlasMultiContactFiles().urdfFile));
  const TrackingRun run = runToTheEnd(*loop);
  expectTheGoalIsReached(*loop, run);
  expectTheTrackedTrajectoryCertifies(run, loop->interface(),
                                      kGetUpSupineScenario);
}

TEST(MultiContactPlanTracking, getsUpFromProneOfTheShippedPlan) {
  // The get_up_prone scenario's own plan (make plan-drc-atlas-get-up-prone):
  // from the hands and knees, a foot forward, the other, the hands off, up.
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(
          trackingConfig(kGetUpProneScenario, kGetUpPronePlan),
          atlasMultiContactFiles().urdfFile));
  const TrackingRun run = runToTheEnd(*loop);
  expectTheGoalIsReached(*loop, run);
  expectTheTrackedTrajectoryCertifies(run, loop->interface(),
                                      kGetUpProneScenario);
}

TEST(MultiContactPlanTracking, aNonPenetrationTermIsItsSpheresTerrainDistance) {
  // After the cases above, the interface's libraries are in this process's
  // CppAD folder: the term below loads its tape rather than generating it.
  const CentroidalRobotFiles files = atlasMultiContactFiles();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CentroidalMpcInterface> interface,
      CentroidalMpcInterface::Create(
          trackingConfig(kBoxClimbScenario, kSwayPlan), files.urdfFile));
  const std::shared_ptr<PlannedMotionReferenceManager>& manager =
      interface->getPlannedMotionReferenceManagerPtr();
  ASSERT_NE(manager, nullptr);
  const vector_t& state = interface->getInitialState();
  manager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state,
                        manager->plannedMotion().modeSequence().front());
  const ModelSettings& settings = interface->modelSettings();
  const std::vector<std::string>& names = settings.contactNames;
  const size_t knee = static_cast<size_t>(
      std::find(names.begin(), names.end(), "knee_l_contact") - names.begin());
  const size_t foot = static_cast<size_t>(
      std::find(names.begin(), names.end(), "foot_l_contact") - names.begin());
  ASSERT_LT(knee, names.size());
  ASSERT_LT(foot, names.size());
  const std::optional<multi_contact::ContactCandidate> candidate =
      manager->candidateOf(knee);
  ASSERT_TRUE(candidate.has_value());
  // The knee's sphere: the geom radius inside its pad, against the pad normal,
  // in the knee frame's orientation (its parent joint's).
  const TerrainNonPenetrationConstraint::Sphere sphere{
      .centerOffset = -candidate->geomRadius * candidate->padNormal,
      .radius = candidate->geomRadius};
  const scalar_t margin = manager->settings().nonPenetrationMargin;
  const multi_contact::TerrainModel& terrain = manager->scenario().terrain;
  const TerrainNonPenetrationConstraint term(
      *manager, interface->getPinocchioInterface(),
      interface->getEffectiveMpcRobotModelAD(), terrain, knee, sphere, margin,
      absl::StrCat("terrain_non_penetration_knee_l_contact_",
                   multi_contact::terrainSceneHash(
                       manager->scenario().scenario.terrain_scene)),
      settings);
  EXPECT_TRUE(term.isActive(/*time=*/0.5)) << "the knee is not in contact";
  const TerrainNonPenetrationConstraint footTerm(
      *manager, interface->getPinocchioInterface(),
      interface->getEffectiveMpcRobotModelAD(), terrain, foot,
      TerrainNonPenetrationConstraint::Sphere{}, margin,
      absl::StrCat("terrain_non_penetration_foot_l_contact_",
                   multi_contact::terrainSceneHash(
                       manager->scenario().scenario.terrain_scene)),
      settings);
  EXPECT_FALSE(footTerm.isActive(/*time=*/0.5)) << "the foot stands";

  PinocchioInterface pinocchio = interface->getPinocchioInterface();
  const vector_t q =
      interface->getMpcRobotModel().getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(pinocchio.getModel(), pinocchio.getData(), q);
  pinocchio::updateFramePlacements(pinocchio.getModel(), pinocchio.getData());
  const pinocchio::SE3& frame =
      pinocchio.getData()
          .oMf[pinocchio.getModel().getFrameId("knee_l_contact")];
  const vector3_t center =
      frame.translation() + frame.rotation() * sphere.centerOffset;
  const PreComputation preComputation;
  const vector_t value = term.getValue(/*time=*/0.5, state, preComputation);
  ASSERT_EQ(value.size(), 1);
  EXPECT_NEAR(value(0),
              terrain.sphereDistance<scalar_t>(center, sphere.radius) - margin,
              1.0e-9);
  EXPECT_GT(value(0), 0.0) << "a standing robot's knee is off the floor";
}

/** [s] How long the robot stands on the top and at the bottom of the stairs
 * walked up and backward down. */
constexpr scalar_t kStairsDwell = 1.5;
/** The periods of the stairs' round trip the loop tracks. */
constexpr int kStairsPeriods = 3;

TEST(MultiContactPlanTracking, walksUpTheStairsAndBackwardDownOverAndOver) {
  // The stairs dummy sim's playback (design section 7.3): the shipped plan up
  // the stairs, a dwell on the top, the plan in reverse time back down, the
  // robot facing up the stairs, a dwell at the bottom, and again: three whole
  // periods (57 s), through both seams of each.
  CentroidalMpcConfig config = trackingConfig(kStairsScenario, kStairsPlan);
  config.task.multi_contact_plan.playback = "forward_and_reverse";
  config.task.multi_contact_plan.playback_dwell = kStairsDwell;
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiContactPlanLoop> loop,
      MultiContactPlanLoop::Create(std::move(config),
                                   atlasMultiContactFiles().urdfFile));
  const PlannedMotion& motion = loop->referenceManager().plannedMotion();
  ASSERT_TRUE(motion.period().has_value());
  const scalar_t period = *motion.period();
  const scalar_t planEnd =
      motion.startTime() + static_cast<scalar_t>(kStairsPeriods) * period;
  const TrackingRun run =
      runUntil(*loop, planEnd,
               planEnd - motion.startTime() + kSettleTime +
                   static_cast<scalar_t>(kStairsPeriods) * kLongestHold);

  // The base followed the plan's, up and down, at the plan time of every
  // solve; on the top in every period and back at the bottom at the end.
  const MpcRobotModelBase<scalar_t>& model =
      loop->interface().getMpcRobotModel();
  // The ascent's end: a period is the ascent, the dwell, the descent and the
  // dwell.
  const scalar_t top = motion.startTime() + 0.5 * (period - 2.0 * kStairsDwell);
  const vector3_t topBase =
      model.getBasePosition(model.getDesiredState(motion.targets(), top));
  const vector3_t bottomBase = model.getBasePosition(motion.initialState());
  ASSERT_GT(topBase.z() - bottomBase.z(), 0.5) << "the plan climbs the stairs";
  scalar_t worstBaseError = 0.0;
  scalar_t worstBottomError = 0.0;
  std::vector<bool> onTheTop(kStairsPeriods, false);
  std::vector<bool> atTheBottom(kStairsPeriods, false);
  for (size_t k = 0; k < run.planTimes.size(); ++k) {
    const vector3_t base = model.getBasePosition(run.states[k]);
    const vector3_t planned = model.getBasePosition(model.getDesiredState(
        motion.targets(), motion.cycleTime(run.planTimes[k])));
    worstBaseError = std::max(worstBaseError, (base - planned).norm());
    const int64_t cycle = motion.cycleIndex(run.planTimes[k]);
    const scalar_t inCycle = motion.cycleTime(run.planTimes[k]);
    if (cycle < 0 || cycle >= kStairsPeriods) continue;
    if (inCycle >= top && inCycle <= top + kStairsDwell &&
        (base - topBase).norm() < kBaseTolerance) {
      onTheTop[static_cast<size_t>(cycle)] = true;
    }
    // Standing at the bottom again, in the bottom dwell that ends the period.
    if (inCycle >= motion.endTime() - kStairsDwell) {
      atTheBottom[static_cast<size_t>(cycle)] = true;
      worstBottomError = std::max(worstBottomError, (base - bottomBase).norm());
      EXPECT_EQ(run.modes[k], motion.modeSequence().front())
          << "at plan time " << run.planTimes[k];
    }
  }
  EXPECT_LT(worstBaseError, kBaseTolerance);
  for (int cycle = 0; cycle < kStairsPeriods; ++cycle) {
    EXPECT_TRUE(onTheTop[static_cast<size_t>(cycle)])
        << "not on the top step in period " << cycle;
    EXPECT_TRUE(atTheBottom[static_cast<size_t>(cycle)])
        << "not back at the bottom in period " << cycle;
  }
  EXPECT_LT(worstBottomError, kBaseTolerance);
  ::testing::Test::RecordProperty("periods", absl::StrCat(kStairsPeriods));
  ::testing::Test::RecordProperty("tracked_s",
                                  absl::StrCat(loop->observation().time));
  ::testing::Test::RecordProperty("worst_base_tracking_error_m",
                                  absl::StrCat(worstBaseError));
  ::testing::Test::RecordProperty("worst_bottom_base_error_m",
                                  absl::StrCat(worstBottomError));
  ::testing::Test::RecordProperty("held_s", absl::StrCat(run.heldTime));
  expectTheTrackedTrajectoryCertifies(run, loop->interface(), kStairsScenario);
}

/** The landing alignment term's z_c x n at `state`, by Pinocchio. */
vector3_t padTiltOf(CentroidalMpcInterface& interface,
                    const vector_t& state,
                    const std::string& contactName,
                    const vector3_t& patchNormal) {
  PinocchioInterface pinocchio = interface.getPinocchioInterface();
  const vector_t q =
      interface.getMpcRobotModel().getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(pinocchio.getModel(), pinocchio.getData(), q);
  pinocchio::updateFramePlacements(pinocchio.getModel(), pinocchio.getData());
  const vector3_t padNormal =
      pinocchio.getData()
          .oMf[pinocchio.getModel().getFrameId(contactName)]
          .rotation()
          .col(kZAxisIndex);
  return padNormal.cross(patchNormal);
}

TEST(MultiContactPlanTracking, aLandingAlignmentTermTurnsTheSoleToItsPatch) {
  // The first swing of the right sole in the stairs plan, onto a step, with a
  // window of the term's own; its library is the interface's, so the term
  // below loads it.
  const CentroidalRobotFiles files = atlasMultiContactFiles();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CentroidalMpcInterface> interface,
      CentroidalMpcInterface::Create(
          trackingConfig(kStairsScenario, kStairsPlan), files.urdfFile));
  const std::shared_ptr<PlannedMotionReferenceManager>& manager =
      interface->getPlannedMotionReferenceManagerPtr();
  ASSERT_NE(manager, nullptr);
  const PlannedMotion& motion = manager->plannedMotion();
  constexpr scalar_t kWindow = 0.2;  // [s]
  manager->preSolverRun(/*initTime=*/0.0, motion.endTime() - motion.startTime(),
                        motion.initialState(), motion.modeSequence().front());
  const ModelSettings& settings = interface->modelSettings();
  const std::vector<std::string>& names = settings.contactNames;
  const size_t foot = static_cast<size_t>(
      std::find(names.begin(), names.end(), "foot_r_contact") - names.begin());
  ASSERT_LT(foot, names.size());
  ASSERT_FALSE(motion.swingWindows(foot).empty());
  const PlannedSwingWindow& swing = motion.swingWindows(foot).front();
  ASSERT_GT(swing.touchDownTime - swing.liftOffTime, kWindow);
  // Wall time of a plan time, at the first solve's offset.
  const scalar_t offset = -manager->planTimeAt(/*time=*/0.0);
  const scalar_t touchDown = swing.touchDownTime + offset;
  const scalar_t liftOff = swing.liftOffTime + offset;
  const SwingLandingAlignmentConstraint term(
      *manager, interface->getPinocchioInterface(),
      interface->getEffectiveMpcRobotModelAD(), foot, kWindow,
      "swing_landing_alignment_foot_r_contact", settings);

  // Active over the window before the touch-down, not in the stance before
  // the swing or early in it.
  const scalar_t landing = touchDown - 0.5 * kWindow;
  EXPECT_TRUE(term.isActive(landing));
  ASSERT_TRUE(term.timeToTouchDown(landing).has_value());
  EXPECT_NEAR(*term.timeToTouchDown(landing), 0.5 * kWindow, 1.0e-9);
  EXPECT_FALSE(term.isActive(liftOff - 0.05));
  EXPECT_FALSE(term.timeToTouchDown(liftOff - 0.05).has_value());
  EXPECT_FALSE(term.isActive(touchDown - 1.5 * kWindow));
  EXPECT_FALSE(term.isActive(touchDown + 0.05));

  // Its value is z_c x n for the normal of the patch it lands on.
  const std::optional<PlannedContactPhase> target =
      motion.nextPhase(foot, swing.liftOffTime);
  ASSERT_TRUE(target.has_value());
  const PreComputation preComputation;
  const vector_t normal = term.getParameters(landing, preComputation);
  EXPECT_TRUE(normal.isApprox(target->normal, 1.0e-9)) << normal.transpose();
  for (const scalar_t planTime :
       {swing.liftOffTime + 0.05, swing.touchDownTime - 0.5 * kWindow}) {
    const vector_t state = interface->getMpcRobotModel().getDesiredState(
        motion.targets(), planTime);
    const vector_t value = term.getValue(landing, state, preComputation);
    const vector3_t expected =
        padTiltOf(*interface, state, "foot_r_contact", target->normal);
    ASSERT_EQ(value.size(), 3);
    EXPECT_LT((value - expected).norm(), 1.0e-9)
        << "at plan time " << planTime << ": " << value.transpose()
        << " against " << expected.transpose();
  }
}

}  // namespace
}  // namespace ocs2::humanoid
