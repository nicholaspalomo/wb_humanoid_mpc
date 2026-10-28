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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/multi_contact/MultiContactPlanSource.h"
#include "humanoid_centroidal_mpc/multi_contact/PlanContactOrder.h"
#include "humanoid_centroidal_mpc/multi_contact/PlanPlayback.h"
#include "humanoid_centroidal_mpc/multi_contact/PlannedMotion.h"
#include "humanoid_centroidal_mpc/multi_contact/PlannedMotionReferenceManager.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/StabilityCertifier.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/MultiContactTestFiles.h"
#include "support/TypedConfigFiles.h"

/*
 * The "forward_and_reverse" playback of a tracked plan (design section 7.3,
 * humanoid_nmpc/docs/multi_contact_planning/README.md) on the stairs
 * scenario's shipped plan: the cycle forwardAndReverseCycle() builds - the
 * ascent, a dwell on the top, the ascent in reverse time, a dwell at the
 * bottom - certifies at every knot with the ascent's own momentum rates; the
 * periodic PlannedMotion of it answers every plan time through the period,
 * continuous at both seams, its velocities negated in the reversed half, its
 * modes a palindrome and its reversed swings stepping down; and the
 * planned-motion reference manager runs its clock, hold included, its mode
 * schedule and its targets across the wrap of the period. Builds no CppAD
 * library: the models are CreateControllerModels()'.
 */
namespace ocs2::humanoid {
namespace {

using Plan = multi_contact::config::MultiContactPlan;

constexpr char kStairsScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/stairs.textproto";
constexpr char kStairsPlan[] =
    "humanoid_nmpc/humanoid_multi_contact/plans/stairs.binpb";
constexpr scalar_t kDwell = 1.5;  // [s]
constexpr scalar_t kTolerance = 1.0e-9;
constexpr scalar_t kHorizon = 1.0;  // [s]

/** The multi-contact models of the Atlas, without an optimal control problem.
 */
std::unique_ptr<CentroidalMpcInterface> multiContactModels() {
  const CentroidalRobotFiles files = atlasMultiContactFiles();
  return checkedValue(CentroidalMpcInterface::CreateControllerModels(
      files.taskFile, files.urdfFile, files.referenceFile));
}

/** The stairs scenario and its shipped plan, the ascent. */
MultiContactPlanSource stairsSource() {
  return checkedValue(loadMultiContactPlanSource(kStairsScenario, kStairsPlan));
}

/** The motion of `plan` on `models`, repeated as `repetition` says. */
PlannedMotion motionOf(const CentroidalMpcInterface& models,
                       const MultiContactPlanSource& source,
                       const Plan& plan,
                       PlanRepetition repetition) {
  const PlanContactOrder order = checkedValue(PlanContactOrder::Create(
      source.scenario.contactSet, models.modelSettings().contactNames));
  return checkedValue(PlannedMotion::Create(
      plan, order, models.getMpcRobotModel(),
      models.getCentroidalModelInfo().robotMass, repetition));
}

/** The index of the contact `name` of `models`. */
size_t contactIndex(const CentroidalMpcInterface& models,
                    absl::string_view name) {
  const std::vector<std::string>& names = models.modelSettings().contactNames;
  for (size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return i;
  }
  ADD_FAILURE() << name << " is not a contact";
  return 0;
}

/** [s] tau(t) = mirror - t: the reversed half's time of the ascent's t. */
scalar_t mirrorOf(const Plan& ascent) {
  return 2.0 * ascent.knots.back().time + kDwell;
}

TEST(PlanPlayback, isSelectedByName) {
  EXPECT_EQ(checkedValue(planPlaybackFromName("once")), PlanPlayback::kOnce);
  EXPECT_EQ(checkedValue(planPlaybackFromName("forward_and_reverse")),
            PlanPlayback::kForwardAndReverse);
  const absl::StatusOr<PlanPlayback> unknown = planPlaybackFromName("loop");
  ASSERT_FALSE(unknown.ok());
  EXPECT_TRUE(absl::StrContains(unknown.status().message(),
                                "once, forward_and_reverse"))
      << unknown.status();

  // Off by default: the task file's default plays once, its dwell unread.
  mpc_config::MultiContactPlanConfig config;
  EXPECT_EQ(config.playback, "once");
  EXPECT_EQ(checkedValue(plannedMotionSettingsFromConfig(config)).playback,
            PlanPlayback::kOnce);
  config.playback_dwell = 0.0;
  EXPECT_TRUE(plannedMotionSettingsFromConfig(config).ok());
  config.playback = "forward_and_reverse";
  const absl::StatusOr<PlannedMotionSettings> noDwell =
      plannedMotionSettingsFromConfig(config);
  ASSERT_FALSE(noDwell.ok());
  EXPECT_TRUE(absl::StrContains(noDwell.status().message(), "playback_dwell"))
      << noDwell.status();
  config.playback_dwell = kDwell;
  const PlannedMotionSettings settings =
      checkedValue(plannedMotionSettingsFromConfig(config));
  EXPECT_EQ(settings.playback, PlanPlayback::kForwardAndReverse);
  EXPECT_EQ(settings.playbackDwell, kDwell);
  config.playback = "loop";
  EXPECT_FALSE(plannedMotionSettingsFromConfig(config).ok());
  EXPECT_FALSE(forwardAndReverseCycle(stairsSource().plan, /*dwell=*/0.0).ok());
}

TEST(ForwardAndReverseCycle, mirrorsEveryKnotInTimeWithItsVelocitiesNegated) {
  const Plan ascent = stairsSource().plan;
  const Plan cycle = checkedValue(forwardAndReverseCycle(ascent, kDwell));
  const size_t n = ascent.knots.size();
  const scalar_t start = ascent.knots.front().time;
  const scalar_t end = ascent.knots.back().time;
  const scalar_t mirror = mirrorOf(ascent);
  // The ascent, a resting knot, the reversed ascent, a resting knot, and the
  // first knot again a period later.
  ASSERT_EQ(cycle.knots.size(), 2 * n + 3);
  EXPECT_NEAR(cycle.knots.back().time,
              start + 2.0 * (end - start) + 2.0 * kDwell, kTolerance);
  EXPECT_FALSE(cycle.report.has_value());
  EXPECT_EQ(cycle.contact_set_hash, ascent.contact_set_hash);
  EXPECT_EQ(cycle.terrain_scene_hash, ascent.terrain_scene_hash);
  for (size_t i = 0; i < n; ++i) {
    const Plan::Knot& forward = ascent.knots[i];
    EXPECT_EQ(cycle.knots[i].time, forward.time);
    const Plan::Knot& reversed = cycle.knots[2 * n - i];
    EXPECT_NEAR(reversed.time, mirror - forward.time, kTolerance);
    EXPECT_EQ(reversed.base_position, forward.base_position);
    EXPECT_EQ(reversed.base_orientation_xyzw, forward.base_orientation_xyzw);
    EXPECT_EQ(reversed.joint_positions, forward.joint_positions);
    // The stance and wrenches that held over [t_(i-1), t_i] hold over its
    // mirror: those of the knot before.
    const Plan::Knot& before = ascent.knots[i == 0 ? 0 : i - 1];
    EXPECT_EQ(reversed.active_contacts, before.active_contacts);
    EXPECT_EQ(reversed.joint_velocities, -forward.joint_velocities);
    EXPECT_EQ(reversed.linear_momentum, -forward.linear_momentum);
    EXPECT_EQ(reversed.angular_momentum, -forward.angular_momentum);
    ASSERT_EQ(reversed.contact_wrenches.size(), before.contact_wrenches.size());
    for (size_t w = 0; w < before.contact_wrenches.size(); ++w) {
      EXPECT_EQ(reversed.contact_wrenches[w].force,
                before.contact_wrenches[w].force);
      EXPECT_EQ(reversed.contact_wrenches[w].torque,
                before.contact_wrenches[w].torque);
    }
  }
  // The dwells stand at rest in their middles.
  for (const size_t resting : {n, 2 * n + 1}) {
    EXPECT_TRUE(cycle.knots[resting].linear_momentum.isZero()) << resting;
    EXPECT_TRUE(cycle.knots[resting].angular_momentum.isZero()) << resting;
  }
  EXPECT_NEAR(cycle.knots[n].time, end + 0.5 * kDwell, kTolerance);
  EXPECT_EQ(cycle.knots[n].base_position, ascent.knots.back().base_position);
  EXPECT_EQ(cycle.knots[2 * n + 1].base_position,
            ascent.knots.front().base_position);
  // The plan starts at rest (its first knot's momenta are zero), so the seam
  // of the period is continuous in the momenta too; it does not end at rest,
  // which the top dwell's resting knot takes up.
  EXPECT_TRUE(ascent.knots.front().linear_momentum.isZero());
  EXPECT_TRUE(ascent.knots.front().angular_momentum.isZero());
  EXPECT_EQ(cycle.knots.back().base_position,
            ascent.knots.front().base_position);
  EXPECT_EQ(cycle.knots.back().joint_positions,
            ascent.knots.front().joint_positions);

  // Every phase and its mirror: one through the top dwell for a phase that
  // lasts to the end, one through the bottom dwell for one from the start.
  const scalar_t cycleEnd = cycle.knots.back().time;
  size_t merged = 0;
  for (const Plan::ContactPhase& phase : ascent.phases) {
    const bool toEnd = phase.end_time >= end - kTolerance;
    const scalar_t mirroredEnd = phase.start_time <= start + kTolerance
                                     ? cycleEnd
                                     : mirror - phase.start_time;
    const scalar_t mirroredStart =
        toEnd ? phase.start_time : mirror - phase.end_time;
    merged += toEnd ? 1 : 0;
    const bool found = std::any_of(
        cycle.phases.begin(), cycle.phases.end(),
        [&](const Plan::ContactPhase& candidate) {
          return candidate.contact == phase.contact &&
                 candidate.primitive == phase.primitive &&
                 std::abs(candidate.start_time - mirroredStart) < kTolerance &&
                 std::abs(candidate.end_time - mirroredEnd) < kTolerance &&
                 candidate.target_pad_pose == phase.target_pad_pose;
        });
    EXPECT_TRUE(found) << phase.contact << " on " << phase.primitive << " ["
                       << phase.start_time << ", " << phase.end_time << "]";
  }
  EXPECT_EQ(merged, 2U) << "both feet end on the top step";
  EXPECT_EQ(cycle.phases.size(), 2 * ascent.phases.size() - merged);
}

TEST(ForwardAndReverseCycle, theReversedStairsCertifyWithTheAscentsRates) {
  const MultiContactPlanSource source = stairsSource();
  const Plan& ascent = source.plan;
  const Plan cycle = checkedValue(forwardAndReverseCycle(ascent, kDwell));
  std::unique_ptr<multi_contact::StabilityCertifier> certifier =
      checkedValue(multi_contact::makeCertifier(source.scenario));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::config::CertifierReport report,
      certifier->certify(cycle));
  EXPECT_TRUE(report.passed)
      << report.failed_knots << " of " << report.knots.size() << " knots fail";
  EXPECT_EQ(report.failed_knots, 0);
  ASSERT_EQ(report.knots.size(), cycle.knots.size());
  const multi_contact::config::CertifierReport ascentReport =
      checkedValue(certifier->certify(ascent));

  // The momentum rate of a reversed knot is its forward knot's: the central
  // difference of the negated momenta over the mirrored neighbors.
  const std::vector<multi_contact::CertifierKnot> forward =
      checkedValue(multi_contact::certifierKnotsOf(ascent));
  const std::vector<multi_contact::CertifierKnot> cycled =
      checkedValue(multi_contact::certifierKnotsOf(cycle));
  const size_t n = ascent.knots.size();
  scalar_t worstRateMismatch = 0.0;
  size_t changes = 0;
  for (size_t i = 1; i + 1 < n; ++i) {
    const multi_contact::CertifierKnot& reversed = cycled[2 * n - i];
    const scalar_t scale = 1.0 + forward[i].momentumRate.linear.norm() +
                           forward[i].momentumRate.angular.norm();
    worstRateMismatch = std::max(
        {worstRateMismatch,
         (reversed.momentumRate.linear - forward[i].momentumRate.linear)
                 .norm() /
             scale,
         (reversed.momentumRate.angular - forward[i].momentumRate.angular)
                 .norm() /
             scale});
    // The same stance at the same configuration with the same rate: the same
    // measures.
    const multi_contact::config::CertifierReport::Knot& mirrored =
        report.knots[2 * n - i];
    const multi_contact::config::CertifierReport::Knot& original =
        ascentReport.knots[i];
    // At a change of stance the reversed knot stands in the stance before
    // it, which the ascent's knot does not.
    if (ascent.knots[i].active_contacts !=
        ascent.knots[i - 1].active_contacts) {
      ++changes;
      continue;
    }
    EXPECT_EQ(mirrored.active_contact_mask, original.active_contact_mask);
    EXPECT_NEAR(mirrored.momentum_rate_residual,
                original.momentum_rate_residual, 1.0e-7)
        << "knot " << i;
    EXPECT_NEAR(mirrored.max_active_contact_distance,
                original.max_active_contact_distance, kTolerance);
    ASSERT_EQ(mirrored.min_sphere_clearance.has_value(),
              original.min_sphere_clearance.has_value());
    if (original.min_sphere_clearance.has_value()) {
      EXPECT_NEAR(*mirrored.min_sphere_clearance,
                  *original.min_sphere_clearance, kTolerance);
    }
    ASSERT_EQ(mirrored.static_margin.has_value(),
              original.static_margin.has_value());
    if (original.static_margin.has_value()) {
      EXPECT_NEAR(*mirrored.static_margin, *original.static_margin, kTolerance);
    }
  }
  EXPECT_LT(worstRateMismatch, 1.0e-9);
  EXPECT_GT(changes, 0U);
  ::testing::Test::RecordProperty("worst_relative_momentum_rate_mismatch",
                                  absl::StrCat(worstRateMismatch));
}

TEST(PlannedMotion, everyContactInContactRestsOnAPhaseThroughTheCycle) {
  // The reversed mode schedule and the reversed phases agree at every time: a
  // contact the schedule puts down has a phase there, and so the patch of its
  // stance constraints is the pad it stands on (getContactPatchFrame()), not
  // the next one. With the reversed knots in their own stances instead of
  // the stances of the knots before them, the reversed schedule lagged the
  // mirrored phases by a knot interval, and the stance foot's zero-velocity
  // constraint was written on the pad a step below.
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const MultiContactPlanSource source = stairsSource();
  const Plan cycle = checkedValue(forwardAndReverseCycle(source.plan, kDwell));
  const PlannedMotion motion =
      motionOf(*models, source, cycle, PlanRepetition::kPeriodic);
  const size_t n = models->modelSettings().contactNames.size();
  const scalar_t period = *motion.period();
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  PinocchioInterface pinocchio = models->getPinocchioInterface();
  constexpr scalar_t kStep = 1.0e-3;  // [s]
  size_t inContact = 0;
  scalar_t worstDistance = 0.0;
  for (scalar_t time = 0.0; time < 2.0 * period; time += kStep) {
    const size_t mode = motion.modeAt(time);
    for (size_t i = 0; i < n; ++i) {
      if (!contactIsActiveInMode(mode, i, n)) continue;
      ++inContact;
      const std::optional<PlannedContactPhase> phase = motion.phaseAt(i, time);
      ASSERT_TRUE(phase.has_value())
          << models->modelSettings().contactNames[i] << " is in contact at "
          << time << " s with no phase";
      // The planned pad rests on that phase's patch.
      const vector_t q = model.getGeneralizedCoordinates(
          model.getDesiredState(motion.targets(), motion.cycleTime(time)));
      pinocchio::forwardKinematics(pinocchio.getModel(), pinocchio.getData(),
                                   q);
      const vector3_t pad = pinocchio::updateFramePlacement(
                                pinocchio.getModel(), pinocchio.getData(),
                                pinocchio.getModel().getFrameId(
                                    models->modelSettings().contactNames[i]))
                                .translation();
      worstDistance = std::max(
          worstDistance,
          std::abs(phase->normal.dot(pad) - phase->patch.normalOffset));
    }
  }
  EXPECT_GT(inContact, 0U);
  // Within the clock hold's contact tolerance.
  EXPECT_LT(worstDistance, 0.01);
  ::testing::Test::RecordProperty("worst_planned_pad_distance_m",
                                  absl::StrCat(worstDistance));
}

TEST(PlannedMotion, aPeriodicMotionLooksUpEveryPeriodAlike) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const MultiContactPlanSource source = stairsSource();
  const Plan cycle = checkedValue(forwardAndReverseCycle(source.plan, kDwell));
  const PlannedMotion motion =
      motionOf(*models, source, cycle, PlanRepetition::kPeriodic);
  ASSERT_TRUE(motion.period().has_value());
  const scalar_t period = *motion.period();
  EXPECT_NEAR(period, cycle.knots.back().time - cycle.knots.front().time,
              kTolerance);
  const size_t foot = contactIndex(*models, "foot_l_contact");
  // Between knots (0.025 s apart), where a time a few periods away rounds to
  // the same knot interval.
  for (const scalar_t time :
       {0.01, 0.4123, 3.0123, 8.7123, 12.3123, period - 0.01}) {
    for (const int64_t cycles : {-2, -1, 1, 2, 5}) {
      const scalar_t later = time + static_cast<scalar_t>(cycles) * period;
      EXPECT_EQ(motion.cycleIndex(later), cycles) << time;
      EXPECT_NEAR(motion.cycleTime(later), time, 1.0e-9) << time;
      EXPECT_EQ(motion.modeAt(later), motion.modeAt(time)) << time;
      EXPECT_EQ(motion.plannedWrenchesAt(later), motion.plannedWrenchesAt(time))
          << time;
      const std::optional<PlannedContactPhase> phase =
          motion.phaseAt(foot, time);
      const std::optional<PlannedContactPhase> shifted =
          motion.phaseAt(foot, later);
      ASSERT_EQ(phase.has_value(), shifted.has_value()) << time;
      if (phase.has_value()) {
        EXPECT_NEAR(shifted->startTime - phase->startTime,
                    static_cast<scalar_t>(cycles) * period, 1.0e-9);
        EXPECT_TRUE(shifted->targetPosition.isApprox(phase->targetPosition));
      }
      const std::optional<PlannedContactPhase> next =
          motion.nextPhase(foot, later);
      ASSERT_TRUE(next.has_value()) << "a periodic contact lands again";
      EXPECT_GT(next->startTime, later);
      const std::optional<PlannedContactPhase> previous =
          motion.previousPhase(foot, later);
      ASSERT_TRUE(previous.has_value());
      EXPECT_LE(previous->endTime, later);
    }
  }
  // The clock's events past the first period come back to the cycle's modes.
  const size_t events = motion.eventTimes().size();
  for (size_t k = 0; k < events; ++k) {
    EXPECT_EQ(motion.modeBeforeEvent(k + 3 * events), motion.modeSequence()[k]);
  }
}

TEST(PlannedMotion, thePeriodicMotionIsContinuousAtBothSeams) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const MultiContactPlanSource source = stairsSource();
  const Plan cycle = checkedValue(forwardAndReverseCycle(source.plan, kDwell));
  const PlannedMotion motion =
      motionOf(*models, source, cycle, PlanRepetition::kPeriodic);
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  const TargetTrajectories& knots = motion.targets();
  // The steepest slope of the knots' interpolation: no step between samples
  // dt apart may be larger than it allows.
  scalar_t slope = 0.0;
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    slope = std::max(
        slope, (knots.stateTrajectory[k + 1] - knots.stateTrajectory[k])
                       .lpNorm<Eigen::Infinity>() /
                   (knots.timeTrajectory[k + 1] - knots.timeTrajectory[k]));
  }
  const scalar_t period = *motion.period();
  const scalar_t end = source.plan.knots.back().time;
  constexpr scalar_t kStep = 1.0e-3;  // [s]
  constexpr scalar_t kWindow = 0.1;   // [s]
  // The ascent's end, the reversed half's start and end, and the period's
  // wrap, in the first period and in the third.
  for (const scalar_t seam : {end, end + kDwell, period - kDwell, period,
                              end + 2.0 * period, 3.0 * period}) {
    const TargetTrajectories over =
        motion.targetsOver(seam - 2.0 * kWindow, seam + 2.0 * kWindow);
    vector_t previous = model.getDesiredState(over, seam - kWindow);
    for (scalar_t time = seam - kWindow + kStep; time <= seam + kWindow;
         time += kStep) {
      const vector_t state = model.getDesiredState(over, time);
      EXPECT_LE((state - previous).lpNorm<Eigen::Infinity>(),
                1.01 * slope * kStep + kTolerance)
          << "a jump at " << time << " s, by the seam at " << seam << " s";
      previous = state;
    }
  }
  // Across the wrap, the period starts again at the plan's first state.
  const TargetTrajectories wrap =
      motion.targetsOver(period - kWindow, period + kWindow);
  EXPECT_TRUE(model.getDesiredState(wrap, period)
                  .isApprox(motion.initialState(), kTolerance));
  constexpr scalar_t kIntoThePeriod = 0.0567;  // [s] between two knots
  EXPECT_TRUE(
      model.getDesiredState(wrap, period + kIntoThePeriod)
          .isApprox(model.getDesiredState(knots, kIntoThePeriod), kTolerance));
}

TEST(PlannedMotion, theReversedHalfRetracesTheAscentWithNegatedMomenta) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const MultiContactPlanSource source = stairsSource();
  const Plan cycle = checkedValue(forwardAndReverseCycle(source.plan, kDwell));
  const PlannedMotion motion =
      motionOf(*models, source, cycle, PlanRepetition::kPeriodic);
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  const scalar_t mirror = mirrorOf(source.plan);
  for (const scalar_t time : {0.1234, 1.5, 2.71, 4.0001, 6.66, 7.9}) {
    const vector_t forward = model.getDesiredState(motion.targets(), time);
    const vector_t reversed =
        model.getDesiredState(motion.targets(), mirror - time);
    EXPECT_TRUE(model.getBasePosition(reversed).isApprox(
        model.getBasePosition(forward), kTolerance))
        << time;
    EXPECT_TRUE(model.getBaseOrientation(reversed).isApprox(
        model.getBaseOrientation(forward), kTolerance))
        << time;
    EXPECT_TRUE(model.getJointAngles(reversed).isApprox(
        model.getJointAngles(forward), kTolerance))
        << time;
    // The normalized centroidal momentum, the head of the state, negated.
    EXPECT_LT((reversed.head<6>() + forward.head<6>()).norm(), 1.0e-9)
        << time << ": " << forward.head<6>().transpose() << " against "
        << reversed.head<6>().transpose();
  }
  // Between every two knots the reversed half is in the stance of the
  // forward one and pulls toward its wrenches: the exact mirror.
  for (size_t k = 0; k + 1 < source.plan.knots.size(); ++k) {
    const scalar_t time =
        0.5 * (source.plan.knots[k].time + source.plan.knots[k + 1].time);
    EXPECT_EQ(motion.modeAt(mirror - time), motion.modeAt(time)) << time;
    EXPECT_EQ(motion.plannedWrenchesAt(mirror - time),
              motion.plannedWrenchesAt(time))
        << time;
  }
  // Every change of stance mirrored: at tau(t) of the ascent's change at t.
  const PlannedMotion ascent =
      motionOf(*models, source, source.plan, PlanRepetition::kOnce);
  for (const scalar_t event : ascent.eventTimes()) {
    EXPECT_TRUE(
        std::any_of(motion.eventTimes().begin(), motion.eventTimes().end(),
                    [&](scalar_t reversed) {
                      return std::abs(reversed - (mirror - event)) < kTolerance;
                    }))
        << "no reversed change at the mirror of " << event << " s";
  }
  // The stances of the cycle are the ascent's, then the ascent's backward.
  const std::vector<size_t>& modes = motion.modeSequence();
  const PlannedMotion once =
      motionOf(*models, source, source.plan, PlanRepetition::kOnce);
  ASSERT_EQ(modes.size(), 2 * once.modeSequence().size() - 1);
  ASSERT_EQ(motion.eventTimes().size(), 2 * once.eventTimes().size());
  for (size_t k = 0; k < modes.size(); ++k) {
    EXPECT_EQ(modes[k], modes[modes.size() - 1 - k]) << k;
  }
  EXPECT_TRUE(std::equal(once.modeSequence().begin(), once.modeSequence().end(),
                         modes.begin()));
}

TEST(PlannedMotion, theReversedSwingsStepDownTheStairs) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const MultiContactPlanSource source = stairsSource();
  const Plan cycle = checkedValue(forwardAndReverseCycle(source.plan, kDwell));
  const PlannedMotion motion =
      motionOf(*models, source, cycle, PlanRepetition::kPeriodic);
  const scalar_t end = source.plan.knots.back().time;
  const scalar_t period = *motion.period();
  for (const absl::string_view name : {"foot_l_contact", "foot_r_contact"}) {
    const size_t foot = contactIndex(*models, name);
    size_t up = 0;
    size_t down = 0;
    for (const PlannedSwingWindow& window : motion.swingWindows(foot)) {
      EXPECT_FALSE(window.fromStart);
      // The pad it leaves and the one it lands on, as the reference manager
      // finds them.
      const std::optional<PlannedContactPhase> leaving =
          motion.previousPhase(foot, window.liftOffTime);
      const std::optional<PlannedContactPhase> landing =
          motion.phaseAt(foot, window.touchDownTime);
      ASSERT_TRUE(leaving.has_value() && landing.has_value()) << name;
      const scalar_t rise =
          landing->targetPosition.z() - leaving->targetPosition.z();
      if (window.touchDownTime <= end) {
        EXPECT_GT(rise, 0.1)
            << name << ": a step up at " << window.touchDownTime << " s";
        ++up;
      } else {
        EXPECT_GE(window.liftOffTime, end + kDwell);
        EXPECT_LT(rise, -0.1)
            << name << ": a step down at " << window.touchDownTime << " s";
        ++down;
      }
    }
    EXPECT_EQ(up, down) << name;
    EXPECT_GT(up, 0U) << name;
    // Two periods later: the same swings, shifted.
    const std::vector<PlannedSwingWindow> later =
        motion.swingWindowsOver(foot, 2.0 * period, 3.0 * period - kTolerance);
    ASSERT_EQ(later.size(), motion.swingWindows(foot).size()) << name;
    for (size_t k = 0; k < later.size(); ++k) {
      EXPECT_NEAR(later[k].liftOffTime,
                  motion.swingWindows(foot)[k].liftOffTime + 2.0 * period,
                  1.0e-9);
      EXPECT_NEAR(later[k].touchDownTime,
                  motion.swingWindows(foot)[k].touchDownTime + 2.0 * period,
                  1.0e-9);
    }
  }
}

TEST(PlannedMotion, aMotionThatPlaysOnceIsTheShippedOne) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const MultiContactPlanSource source = stairsSource();
  const PlannedMotion motion =
      motionOf(*models, source, source.plan, PlanRepetition::kOnce);
  EXPECT_FALSE(motion.period().has_value());
  EXPECT_EQ(motion.cycleTime(42.0), 42.0);
  EXPECT_EQ(motion.cycleIndex(42.0), 0);
  EXPECT_EQ(motion.modeAt(42.0), motion.modeSequence().back());
  const ModeSchedule schedule =
      motion.modeScheduleOver(/*from=*/3.0, /*to=*/4.0);
  EXPECT_EQ(schedule.eventTimes, motion.eventTimes());
  EXPECT_EQ(schedule.modeSequence, motion.modeSequence());
  const size_t foot = contactIndex(*models, "foot_l_contact");
  EXPECT_EQ(motion.swingWindowsOver(foot, /*from=*/3.0, /*to=*/4.0).size(),
            motion.swingWindows(foot).size());
  EXPECT_FALSE(motion.nextPhase(foot, /*planTime=*/42.0).has_value());
  // Past its end, its last knot.
  const TargetTrajectories past =
      motion.targetsOver(/*from=*/42.0, /*to=*/43.0);
  ASSERT_EQ(past.size(), 1U);
  EXPECT_EQ(past.timeTrajectory.front(), motion.endTime());
  const TargetTrajectories slice =
      motion.targetsOver(/*from=*/1.01, /*to=*/2.0);
  ASSERT_FALSE(slice.empty());
  EXPECT_LE(slice.timeTrajectory.front(), 1.01);
  EXPECT_GE(slice.timeTrajectory.back(), 2.0);
  // A plan that does not end where it starts cannot repeat.
  const PlanContactOrder order = checkedValue(PlanContactOrder::Create(
      source.scenario.contactSet, models->modelSettings().contactNames));
  const absl::StatusOr<PlannedMotion> refused = PlannedMotion::Create(
      source.plan, order, models->getMpcRobotModel(),
      models->getCentroidalModelInfo().robotMass, PlanRepetition::kPeriodic);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
}

/** The manager of the stairs plan played "forward_and_reverse" on `models`. */
std::shared_ptr<PlannedMotionReferenceManager> forwardAndReverseManagerOf(
    CentroidalMpcInterface& models) {
  const CentroidalMpcConfig& config = models.config();
  PlannedMotionSettings settings = checkedValue(
      plannedMotionSettingsFromConfig(config.task.multi_contact_plan));
  settings.playback = PlanPlayback::kForwardAndReverse;
  settings.playbackDwell = kDwell;
  return checkedValue(PlannedMotionReferenceManager::Create(
      checkedValue(
          GaitSchedule::Create(config.reference, models.modelSettings())),
      std::make_unique<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(),
                                               kNumContacts),
      models.getPinocchioInterface(), models.getMpcRobotModel(), stairsSource(),
      settings));
}

TEST(PlannedMotionReferenceManager, runsTheClockAndTheScheduleAcrossTheWrap) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  std::shared_ptr<PlannedMotionReferenceManager> manager =
      forwardAndReverseManagerOf(*models);
  const PlannedMotion& motion = manager->plannedMotion();
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  ASSERT_TRUE(motion.period().has_value());
  const scalar_t period = *motion.period();
  const size_t events = motion.eventTimes().size();
  ASSERT_GT(events, 0U);
  const scalar_t firstEvent = motion.eventTimes().front();
  /** The plan's own state at plan time `time`: every stance reached. */
  const std::function<vector_t(scalar_t)> planned = [&](scalar_t time) {
    return model.getDesiredState(motion.targets(), motion.cycleTime(time));
  };
  constexpr scalar_t kStep = 0.05;  // [s]
  // Tracking the plan's own states through a whole period, the clock never
  // holds and every change is released, the wrap included.
  bool everHeld = false;
  scalar_t time = 0.0;
  for (; time < period + 0.5 * firstEvent; time += kStep) {
    manager->preSolverRun(time, time + kHorizon, planned(time),
                          motion.modeAt(time));
    everHeld = everHeld || manager->clock().isHolding();
    // Over the horizon, the next period's stances and states follow the
    // wrap.
    if (time > period - 0.5 * kHorizon && time < period) {
      const scalar_t after = period + 0.5 * firstEvent + firstEvent;
      EXPECT_EQ(manager->getModeSchedule().modeAtTime(after),
                motion.modeAt(after - period))
          << time;
      EXPECT_TRUE(
          model.getDesiredState(manager->getTargetTrajectories(), after)
              .isApprox(model.getDesiredState(motion.targets(), after - period),
                        kTolerance))
          << time;
    }
  }
  EXPECT_FALSE(everHeld);
  EXPECT_NEAR(manager->planTimeAt(time), time, kTolerance);
  EXPECT_EQ(manager->clock().releasedEvents(), events);
  // The bottom stance after the wrap: the robot stays 20 cm behind it, and
  // the clock holds `lead` before the period's first change, which stays in
  // the horizon.
  const scalar_t lead = manager->settings().clockHoldLead;
  vector_t behind = planned(period);
  model.setBasePosition(
      behind, model.getBasePosition(behind) - vector3_t(0.2, 0.0, 0.0));
  for (const scalar_t late : {period + firstEvent, period + 3.0}) {
    manager->preSolverRun(late, late + kHorizon, behind, motion.modeAt(0.0));
    EXPECT_TRUE(manager->clock().isHolding()) << late;
    EXPECT_NEAR(manager->planTimeAt(late), period + firstEvent - lead,
                kTolerance);
    EXPECT_EQ(manager->getModeSchedule().modeAtTime(late + lead + 0.01),
              motion.modeSequence()[1]);
  }
  EXPECT_EQ(manager->clock().releasedEvents(), events);
  // Reached: released.
  manager->preSolverRun(period + 3.05, period + 3.05 + kHorizon,
                        planned(firstEvent - lead), motion.modeAt(0.0));
  EXPECT_FALSE(manager->clock().isHolding());
  EXPECT_EQ(manager->clock().releasedEvents(), events + 1);
  // The swings of the next period are rebuilt for it: the first one starts
  // from the bottom.
  const ContactSwingPlanner* absl_nullable planner =
      manager->getContactSwingPlanner();
  ASSERT_NE(planner, nullptr);
  size_t swings = 0;
  for (size_t i = 0; i < planner->numContacts(); ++i) {
    for (const ContactSwingTrajectory& swing : planner->swings(i)) {
      ++swings;
      EXPECT_GT(manager->planTimeAt(swing.touchDownTime()), period);
      EXPECT_LT(swing.reference(swing.liftOffTime()).position.z(), 0.05);
    }
  }
  EXPECT_GT(swings, 0U);
}

}  // namespace
}  // namespace ocs2::humanoid
