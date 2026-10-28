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

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/multi_contact/MultiContactPlanSource.h"
#include "humanoid_centroidal_mpc/multi_contact/PlanContactOrder.h"
#include "humanoid_centroidal_mpc/multi_contact/PlannedMotion.h"
#include "humanoid_centroidal_mpc/multi_contact/PlannedMotionReferenceManager.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_core/ResourcePaths.h"
#include "support/MultiContactTestFiles.h"
#include "support/TypedConfigFiles.h"

/*
 * The planned-motion reference manager (contact_schedule_source:
 * "multi_contact_plan"), on the DRC Atlas's multi-contact models and the
 * hand-written plan of milestone 1 (Atlas swaying on both feet in the box_climb
 * scenario), and on a variant of it that lifts the right foot: the mode
 * schedule is the plan's stances mapped by contact name, the targets are the
 * plan's states, a plan made for another scene is refused, the clock holds
 * before a contact change until the stance is reached, the swings run from the
 * measured lift-off point to the planned pad, and the patches are the plan's.
 * Builds no CppAD library: the models are CreateControllerModels()'.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kBoxClimbScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto";
constexpr char kStairsScenario[] =
    "humanoid_nmpc/humanoid_multi_contact/scenarios/config/stairs.textproto";
constexpr char kSwayPlan[] =
    "humanoid_nmpc/humanoid_multi_contact/app/testdata/"
    "box_climb_sway.textproto";
constexpr scalar_t kTolerance = 1.0e-9;
constexpr scalar_t kHorizon = 1.0;  // [s]

/** The multi-contact models of the Atlas, without an optimal control problem.
 */
std::unique_ptr<CentroidalMpcInterface> multiContactModels() {
  const CentroidalRobotFiles files = atlasMultiContactFiles();
  return checkedValue(CentroidalMpcInterface::CreateControllerModels(
      files.taskFile, files.urdfFile, files.referenceFile));
}

/** The tracked sway plan and its scenario. */
MultiContactPlanSource swaySource() {
  return checkedValue(loadMultiContactPlanSource(kBoxClimbScenario, kSwayPlan));
}

/**
 * The sway plan with the right foot lifted from 0.5 s to 1.0 s: the stance
 * changes at 0.5 s (a lift-off) and at 1.0 s (a touch-down onto the pad it
 * left).
 */
MultiContactPlanSource rightFootStepSource() {
  MultiContactPlanSource source = swaySource();
  multi_contact::config::MultiContactPlan& plan = source.plan;
  constexpr multi_contact::ContactMask kLeftFootOnly = 1;
  plan.knots[2].active_contacts = kLeftFootOnly;
  plan.knots[3].active_contacts = kLeftFootOnly;
  std::vector<multi_contact::config::MultiContactPlan::ContactPhase> phases;
  for (const multi_contact::config::MultiContactPlan::ContactPhase& phase :
       plan.phases) {
    if (phase.contact != "foot_r_contact") {
      phases.push_back(phase);
      continue;
    }
    multi_contact::config::MultiContactPlan::ContactPhase before = phase;
    before.end_time = 0.5;
    multi_contact::config::MultiContactPlan::ContactPhase after = phase;
    after.start_time = 1.0;
    phases.push_back(before);
    phases.push_back(after);
  }
  plan.phases = std::move(phases);
  return source;
}

/** The settings of the multi-contact task file. */
PlannedMotionSettings settingsOf(const CentroidalMpcInterface& models) {
  return checkedValue(
      plannedMotionSettingsFromConfig(models.config().task.multi_contact_plan));
}

/** The manager of `source` on `models` with `settings`. */
std::shared_ptr<PlannedMotionReferenceManager> managerOf(
    CentroidalMpcInterface& models,
    MultiContactPlanSource source,
    const PlannedMotionSettings& settings) {
  const CentroidalMpcConfig& config = models.config();
  return checkedValue(PlannedMotionReferenceManager::Create(
      checkedValue(
          GaitSchedule::Create(config.reference, models.modelSettings())),
      std::make_unique<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(),
                                               kNumContacts),
      models.getPinocchioInterface(), models.getMpcRobotModel(),
      std::move(source), settings));
}

/** The manager of `source` on `models`. */
std::shared_ptr<PlannedMotionReferenceManager> managerOf(
    CentroidalMpcInterface& models, MultiContactPlanSource source) {
  const CentroidalMpcConfig& config = models.config();
  return checkedValue(PlannedMotionReferenceManager::Create(
      checkedValue(
          GaitSchedule::Create(config.reference, models.modelSettings())),
      std::make_unique<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(),
                                               kNumContacts),
      models.getPinocchioInterface(), models.getMpcRobotModel(),
      std::move(source), settingsOf(models)));
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

TEST(PlannedMotionReferenceManager, theModeScheduleIsThePlansStancesByName) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  std::shared_ptr<PlannedMotionReferenceManager> manager =
      managerOf(*models, swaySource());
  const vector_t initial = manager->plannedMotion().initialState();
  manager->preSolverRun(/*initTime=*/10.0, /*finalTime=*/10.0 + kHorizon,
                        initial, ModeNumber::kStance);
  // Both feet, the MPC's last two contacts: the biped's STANCE in the larger
  // set, every other contact inactive.
  EXPECT_EQ(manager->getModeSchedule().modeAtTime(10.5),
            static_cast<size_t>(ModeNumber::kStance));
  const size_t n = models->modelSettings().contactNames.size();
  for (size_t i = 0; i < n; ++i) {
    const std::string& name = models->modelSettings().contactNames[i];
    EXPECT_EQ(manager->isInContact(/*time=*/10.5, i),
              name == "foot_l_contact" || name == "foot_r_contact")
        << name;
  }
}

TEST(PlannedMotionReferenceManager, theTargetsAreThePlansStatesFromTheStart) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  std::shared_ptr<PlannedMotionReferenceManager> manager =
      managerOf(*models, swaySource());
  const PlannedMotion& motion = manager->plannedMotion();
  manager->preSolverRun(/*initTime=*/10.0, /*finalTime=*/10.0 + kHorizon,
                        motion.initialState(), ModeNumber::kStance);
  const TargetTrajectories& targets = manager->getTargetTrajectories();
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  // The plan starts at the first solve: its knot at 0.25 s is the target at
  // 10.25 s, its base 3 cm forward with the momentum of the sway.
  const vector_t target = model.getDesiredState(targets, /*time=*/10.25);
  EXPECT_TRUE(target.isApprox(motion.targets().stateTrajectory[1], kTolerance));
  EXPECT_NEAR(model.getBasePosition(target).x(), 0.03, kTolerance);
  EXPECT_NEAR(model.getBaseComLinearVelocity(target).x(),
              19.164 / models->getCentroidalModelInfo().robotMass, kTolerance);
  // Past the plan's end it holds the last knot.
  manager->preSolverRun(/*initTime=*/20.0, /*finalTime=*/20.0 + kHorizon,
                        motion.initialState(), ModeNumber::kStance);
  EXPECT_TRUE(
      model.getDesiredState(manager->getTargetTrajectories(), /*time=*/20.5)
          .isApprox(motion.targets().stateTrajectory.back(), kTolerance));
  // The MPC's initial state is the plan's first knot.
  EXPECT_NEAR(model.getBasePosition(motion.initialState()).z(), 0.9232,
              kTolerance);
}

TEST(PlannedMotionReferenceManager, theForceReferenceIsThePlansWrenches) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const MultiContactPlanSource source = swaySource();
  PlannedMotionSettings settings = settingsOf(*models);
  ASSERT_EQ(settings.forceReference, ForceReference::kPlannedWrenches)
      << "the multi-contact task file names planned_wrenches";
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  std::shared_ptr<PlannedMotionReferenceManager> planned =
      managerOf(*models, source, settings);
  settings.forceReference = ForceReference::kWeightShare;
  std::shared_ptr<PlannedMotionReferenceManager> shared =
      managerOf(*models, source, settings);
  const vector_t initial = planned->plannedMotion().initialState();
  for (const std::shared_ptr<PlannedMotionReferenceManager>& manager :
       {planned, shared}) {
    manager->preSolverRun(/*initTime=*/10.0, /*finalTime=*/10.0 + kHorizon,
                          initial, ModeNumber::kStance);
  }
  // At 10.3 s the plan is at 0.3 s: the knot of 0.25 s.
  const multi_contact::config::MultiContactPlan::Knot& knot =
      source.plan.knots[1];
  ASSERT_NEAR(knot.time, 0.25, kTolerance);
  const vector_t plannedInput = planned->getDesiredInput(
      planned->getTargetTrajectories(), initial, /*time=*/10.3);
  for (const multi_contact::config::MultiContactPlan::ContactWrench& wrench :
       knot.contact_wrenches) {
    const size_t i = contactIndex(*models, wrench.contact);
    EXPECT_TRUE(model.getContactForce(plannedInput, i)
                    .isApprox(vector3_t(wrench.force), kTolerance))
        << wrench.contact;
  }
  // Weight share: the base's reference.
  const vector_t sharedInput = shared->getDesiredInput(
      shared->getTargetTrajectories(), initial, /*time=*/10.3);
  const scalar_t weight =
      models->getCentroidalModelInfo().robotMass * kGravitationalAcceleration;
  const size_t left = contactIndex(*models, "foot_l_contact");
  EXPECT_NEAR(model.getContactForce(sharedInput, left).z(), 0.5 * weight,
              1.0e-6 * weight);
}

TEST(PlannedMotionReferenceManager, readsTheLandingAlignmentOfTheSwings) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  const PlannedMotionSettings settings = settingsOf(*models);
  // The multi-contact task file's: soles swing flat, over the whole swing.
  EXPECT_GT(settings.landingAlignmentWeight, 0.0);
  EXPECT_NEAR(settings.landingWindow, 2.0, kTolerance);
  // Off by default, and refused when negative or with an empty window.
  EXPECT_EQ(mpc_config::MultiContactPlanConfig::ContactSwing()
                .landing_alignment_weight,
            0.0);
  mpc_config::MultiContactPlanConfig config =
      models->config().task.multi_contact_plan;
  config.contact_swing.landing_alignment_weight = -1.0;
  const absl::StatusOr<PlannedMotionSettings> negative =
      plannedMotionSettingsFromConfig(config);
  ASSERT_FALSE(negative.ok());
  EXPECT_TRUE(absl::StrContains(negative.status().message(),
                                "landing_alignment_weight"))
      << negative.status();
  config.contact_swing.landing_alignment_weight = 0.0;
  config.contact_swing.landing_window = 0.0;
  const absl::StatusOr<PlannedMotionSettings> empty =
      plannedMotionSettingsFromConfig(config);
  ASSERT_FALSE(empty.ok());
  EXPECT_TRUE(absl::StrContains(empty.status().message(), "landing_window"))
      << empty.status();
}

TEST(PlannedMotionReferenceManager, refusesAPlanMadeForAnotherScene) {
  const absl::StatusOr<MultiContactPlanSource> refused =
      loadMultiContactPlanSource(kStairsScenario, kSwayPlan);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kFailedPrecondition)
      << refused.status();
  const absl::StatusOr<MultiContactPlanSource> missing =
      loadMultiContactPlanSource(kBoxClimbScenario, "no/such/plan.binpb");
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound)
      << missing.status();
  const absl::StatusOr<MultiContactPlanSource> unnamed =
      loadMultiContactPlanSource(kBoxClimbScenario, /*planFile=*/"");
  EXPECT_EQ(unnamed.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(PlannedMotionReferenceManager, holdsTheClockUntilTheStanceIsReached) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  std::shared_ptr<PlannedMotionReferenceManager> manager =
      managerOf(*models, rightFootStepSource());
  const PlannedMotion& motion = manager->plannedMotion();
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  ASSERT_EQ(motion.eventTimes().size(), 2U);
  EXPECT_NEAR(motion.eventTimes()[0], 0.5, kTolerance);
  const scalar_t lead = manager->settings().clockHoldLead;
  // The robot stays where the plan started, 6 cm behind the plan's CoM at the
  // hold point: the clock stops there.
  const vector_t standing = motion.initialState();
  manager->preSolverRun(/*initTime=*/0.0, kHorizon, standing,
                        ModeNumber::kStance);
  manager->preSolverRun(/*initTime=*/0.45, 0.45 + kHorizon, standing,
                        ModeNumber::kStance);
  EXPECT_TRUE(manager->clock().isHolding());
  EXPECT_NEAR(manager->planTimeAt(0.45), 0.5 - lead, kTolerance);
  manager->preSolverRun(/*initTime=*/2.0, 2.0 + kHorizon, standing,
                        ModeNumber::kStance);
  EXPECT_TRUE(manager->clock().isHolding());
  EXPECT_NEAR(manager->planTimeAt(2.0), 0.5 - lead, kTolerance);
  // The lift-off stays in the horizon, `lead` ahead: never skipped.
  EXPECT_EQ(manager->getModeSchedule().modeAtTime(2.0),
            static_cast<size_t>(ModeNumber::kStance));
  EXPECT_EQ(manager->getModeSchedule().modeAtTime(2.0 + lead + 0.01),
            static_cast<size_t>(ModeNumber::kLf));
  // The robot reaches the plan's state at the hold point: released.
  const vector_t reached = model.getDesiredState(motion.targets(), 0.5 - lead);
  manager->preSolverRun(/*initTime=*/2.02, 2.02 + kHorizon, reached,
                        ModeNumber::kStance);
  EXPECT_FALSE(manager->clock().isHolding());
  EXPECT_EQ(manager->clock().releasedEvents(), 1U);
  EXPECT_NEAR(manager->planTimeAt(2.02), 0.5 - lead + 0.02, kTolerance);
}

TEST(PlannedMotionReferenceManager, swingsAContactFromWhereItLiftedOff) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  std::shared_ptr<PlannedMotionReferenceManager> manager =
      managerOf(*models, rightFootStepSource());
  const PlannedMotion& motion = manager->plannedMotion();
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  const size_t right = contactIndex(*models, "foot_r_contact");
  const size_t left = contactIndex(*models, "foot_l_contact");
  // Track the plan's own states, so that the clock never holds. The last
  // solve with the right foot in contact is at 0.4 s.
  vector3_t lastInContact = vector3_t::Zero();
  for (const scalar_t time : {0.0, 0.2, 0.4, 0.6}) {
    manager->preSolverRun(time, time + kHorizon,
                          model.getDesiredState(motion.targets(), time),
                          ModeNumber::kStance);
    if (manager->isInContact(time, right)) {
      lastInContact = manager->measuredContactPositions()[right];
    }
  }
  EXPECT_FALSE(manager->clock().isHolding());
  const ContactSwingPlanner* absl_nullable planner =
      manager->getContactSwingPlanner();
  ASSERT_NE(planner, nullptr);
  ASSERT_EQ(planner->swings(right).size(), 1U);
  const ContactSwingTrajectory& swing = planner->swings(right).front();
  EXPECT_NEAR(manager->planTimeAt(swing.liftOffTime()), 0.5, kTolerance);
  EXPECT_NEAR(manager->planTimeAt(swing.touchDownTime()), 1.0, kTolerance);
  // From where the foot was in contact, along the floor's normal, to its pad.
  const std::optional<PlannedContactPhase> pad =
      motion.nextPhase(right, /*planTime=*/0.6);
  ASSERT_TRUE(pad.has_value());
  EXPECT_TRUE(swing.reference(swing.touchDownTime())
                  .position.isApprox(pad->targetPosition, kTolerance));
  EXPECT_TRUE(swing.reference(swing.liftOffTime())
                  .position.isApprox(lastInContact, kTolerance))
      << swing.reference(swing.liftOffTime()).position.transpose()
      << " against " << lastInContact.transpose();
  EXPECT_TRUE(swing.clearanceSatisfied());
  EXPECT_GT(swing.reference(0.5 * (swing.liftOffTime() + swing.touchDownTime()))
                .position.z(),
            manager->settings().swing.clearance);
  // The swinging foot has a reference; the stance foot has none.
  EXPECT_TRUE(planner->reference(right, /*time=*/0.6).has_value());
  EXPECT_FALSE(planner->reference(left, /*time=*/0.6).has_value());
}

/** [m] Where the plan of `motion` has the frame `frame` at plan time `time`. */
vector3_t plannedFramePosition(CentroidalMpcInterface& models,
                               const PlannedMotion& motion,
                               const std::string& frame,
                               scalar_t time) {
  const MpcRobotModelBase<scalar_t>& model = models.getMpcRobotModel();
  PinocchioInterface pinocchio = models.getPinocchioInterface();
  pinocchio::forwardKinematics(
      pinocchio.getModel(), pinocchio.getData(),
      model.getGeneralizedCoordinates(
          model.getDesiredState(motion.targets(), time)));
  return pinocchio::updateFramePlacement(pinocchio.getModel(),
                                         pinocchio.getData(),
                                         pinocchio.getModel().getFrameId(frame))
      .translation();
}

TEST(PlannedMotionReferenceManager, aSwingLongerThanTheArcLimitFollowsThePlan) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  PlannedMotionSettings settings = settingsOf(*models);
  // The right foot is in the air from 0.5 s to 1.0 s, longer than 0.2 s: it
  // follows the plan's foot from knot to knot (0.5, 0.75 and 1.0 s) on two
  // straight segments.
  settings.arcSwingDurationLimit = 0.2;
  std::shared_ptr<PlannedMotionReferenceManager> manager =
      managerOf(*models, rightFootStepSource(), settings);
  const PlannedMotion& motion = manager->plannedMotion();
  const MpcRobotModelBase<scalar_t>& model = models->getMpcRobotModel();
  const size_t right = contactIndex(*models, "foot_r_contact");
  for (const scalar_t time : {0.0, 0.2, 0.4, 0.6}) {
    manager->preSolverRun(time, time + kHorizon,
                          model.getDesiredState(motion.targets(), time),
                          ModeNumber::kStance);
  }
  const ContactSwingPlanner* absl_nullable planner =
      manager->getContactSwingPlanner();
  ASSERT_NE(planner, nullptr);
  const std::vector<ContactSwingTrajectory>& segments = planner->swings(right);
  ASSERT_EQ(segments.size(), 2U);
  const std::vector<scalar_t> knots = {0.5, 0.75, 1.0};
  // The path's tolerance: a segment's inner control points lie 1 mm along it.
  constexpr scalar_t kPathTolerance = 1.0e-3;
  for (size_t k = 0; k < segments.size(); ++k) {
    const ContactSwingTrajectory& segment = segments[k];
    EXPECT_NEAR(manager->planTimeAt(segment.liftOffTime()), knots[k],
                kTolerance);
    EXPECT_NEAR(manager->planTimeAt(segment.touchDownTime()), knots[k + 1],
                kTolerance);
    const vector3_t start =
        plannedFramePosition(*models, motion, "foot_r_contact", knots[k]);
    const vector3_t end =
        plannedFramePosition(*models, motion, "foot_r_contact", knots[k + 1]);
    EXPECT_TRUE(segment.reference(segment.liftOffTime())
                    .position.isApprox(start, kTolerance))
        << segment.reference(segment.liftOffTime()).position.transpose()
        << " against " << start.transpose();
    EXPECT_TRUE(segment.reference(segment.touchDownTime())
                    .position.isApprox(end, kTolerance));
    // Half way in time, on the straight line between the knots' feet.
    const vector3_t middle =
        segment
            .reference(0.5 * (segment.liftOffTime() + segment.touchDownTime()))
            .position;
    const vector3_t way = end - start;
    const scalar_t along = (middle - start).dot(way) / way.squaredNorm();
    EXPECT_LT((middle - start - along * way).norm(), kPathTolerance);
  }
  // At the arc limit's default (0) the same step is one arc.
  PlannedMotionSettings arcSettings = settingsOf(*models);
  arcSettings.arcSwingDurationLimit = 0.0;
  std::shared_ptr<PlannedMotionReferenceManager> oneArc =
      managerOf(*models, rightFootStepSource(), arcSettings);
  for (const scalar_t time : {0.0, 0.2, 0.4, 0.6}) {
    oneArc->preSolverRun(time, time + kHorizon,
                         model.getDesiredState(motion.targets(), time),
                         ModeNumber::kStance);
  }
  ASSERT_NE(oneArc->getContactSwingPlanner(), nullptr);
  EXPECT_EQ(oneArc->getContactSwingPlanner()->swings(right).size(), 1U);
  // A negative limit is refused.
  mpc_config::MultiContactPlanConfig config =
      models->config().task.multi_contact_plan;
  config.arc_swing_duration_limit = -1.0;
  EXPECT_FALSE(plannedMotionSettingsFromConfig(config).ok());
}

TEST(PlannedMotionReferenceManager, thePatchesAreThePlansPhases) {
  std::unique_ptr<CentroidalMpcInterface> models = multiContactModels();
  std::shared_ptr<PlannedMotionReferenceManager> manager =
      managerOf(*models, swaySource());
  manager->preSolverRun(/*initTime=*/0.0, kHorizon,
                        manager->plannedMotion().initialState(),
                        ModeNumber::kStance);
  // The feet rest on the floor's top face: the world-aligned patch at z = 0,
  // exactly the arithmetic of the shipped terms.
  const ContactPatchFrame foot = manager->getContactPatchFrame(
      contactIndex(*models, "foot_l_contact"), /*time=*/0.5);
  EXPECT_TRUE(isWorldAligned(foot));
  EXPECT_NEAR(foot.normalOffset, 0.0, kTolerance);
  // A contact the plan never puts down falls back to the base's patch.
  const ContactPatchFrame knee = manager->getContactPatchFrame(
      contactIndex(*models, "knee_l_contact"), /*time=*/0.5);
  EXPECT_TRUE(isWorldAligned(knee));
}

TEST(PlannedMotion, aPatchFrameIsOrthonormalWithItsNormalAsZ) {
  const vector3_t normal = vector3_t(0.3, -0.5, 0.8).normalized();
  const vector3_t point(1.0, 2.0, 0.5);
  const ContactPatchFrame patch = PlannedMotion::patchFrameOf(point, normal);
  const matrix3_t& rotation = patch.worldToPatchRotation;
  EXPECT_TRUE((rotation * rotation.transpose())
                  .isApprox(matrix3_t::Identity(), kTolerance));
  EXPECT_NEAR(rotation.determinant(), 1.0, kTolerance);
  EXPECT_TRUE((rotation * normal).isApprox(vector3_t::UnitZ(), kTolerance));
  EXPECT_TRUE(patchNormal(patch).isApprox(normal, kTolerance));
  EXPECT_NEAR(patch.normalOffset, normal.dot(point), kTolerance);
  // The world's z axis is the identity, exactly.
  EXPECT_TRUE(
      isWorldAligned(PlannedMotion::patchFrameOf(point, vector3_t::UnitZ())));
  // A wall facing x takes the world's y axis as its x direction.
  const ContactPatchFrame wall =
      PlannedMotion::patchFrameOf(point, -vector3_t::UnitX());
  EXPECT_TRUE((wall.worldToPatchRotation * -vector3_t::UnitX())
                  .isApprox(vector3_t::UnitZ(), kTolerance));
}

/** The shipped (biped) Atlas's models: its wrists are fixed and its contacts
 * are the two feet. */
std::unique_ptr<CentroidalMpcInterface> bipedModels() {
  const CentroidalRobotFiles files = atlasFiles();
  return checkedValue(CentroidalMpcInterface::CreateControllerModels(
      files.taskFile, files.urdfFile, files.referenceFile));
}

/** PlannedMotion::Create() of `plan` on the biped's models. */
absl::Status motionStatusOn(const CentroidalMpcInterface& models,
                            const MultiContactPlanSource& source) {
  const PlanContactOrder order = checkedValue(PlanContactOrder::Create(
      source.scenario.contactSet, models.modelSettings().contactNames));
  return PlannedMotion::Create(source.plan, order, models.getMpcRobotModel(),
                               models.getCentroidalModelInfo().robotMass)
      .status();
}

TEST(PlannedMotion, refusesWhatTheMpcCannotTrack) {
  std::unique_ptr<CentroidalMpcInterface> biped = bipedModels();
  // The plan on the feet alone is the biped's.
  EXPECT_TRUE(motionStatusOn(*biped, swaySource()).ok());
  // A joint the biped's MPC fixes, moved.
  MultiContactPlanSource wrist = swaySource();
  wrist.plan.joint_names.push_back("l_arm_wrx");
  for (multi_contact::config::MultiContactPlan::Knot& knot : wrist.plan.knots) {
    knot.joint_positions.conservativeResize(knot.joint_positions.size() + 1);
    knot.joint_positions(knot.joint_positions.size() - 1) = 0.3;
  }
  const absl::Status fixed = motionStatusOn(*biped, wrist);
  EXPECT_EQ(fixed.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(fixed.message(), "l_arm_wrx")) << fixed;
  // A joint of no URDF.
  MultiContactPlanSource unknown = swaySource();
  unknown.plan.joint_names[0] = "no_such_joint";
  EXPECT_TRUE(absl::StrContains(motionStatusOn(*biped, unknown).message(),
                                "no_such_joint"));
  // A knee the biped does not have, active.
  MultiContactPlanSource kneeling = swaySource();
  constexpr multi_contact::ContactMask kFeetAndLeftKnee = 0b111;
  kneeling.plan.knots[1].active_contacts = kFeetAndLeftKnee;
  const absl::Status knee = motionStatusOn(*biped, kneeling);
  EXPECT_EQ(knee.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(knee.message(), "knee_l_contact")) << knee;
  // An active contact no phase covers.
  MultiContactPlanSource uncovered = swaySource();
  uncovered.plan.phases.erase(uncovered.plan.phases.begin());
  EXPECT_FALSE(motionStatusOn(*biped, uncovered).ok());
}

}  // namespace
}  // namespace ocs2::humanoid
