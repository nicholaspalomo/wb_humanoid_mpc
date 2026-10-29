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

// StanceMotion on the DRC Atlas: a step forward on the floor, as the planner
// takes it: shift onto the right foot, swing the left foot forward, put it
// down; and the overlap of pads on one surface.

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/planner/StanceInverseKinematics.h"
#include "humanoid_multi_contact/planner/StanceMotion.h"
#include "humanoid_multi_contact/stability/JointLimits.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kAtlasContactSet[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "contact_set.textproto";
constexpr char kAtlasCollisionModel[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "collision_model.textproto";

/** The floor and, with `wall`, a wall 0.1 m thick and 3 m tall across the x
 * axis at x = 0. */
TerrainModel terrainOf(bool wall) {
  config::TerrainScene scene;
  scene.name = "floor";
  config::TerrainScene::Primitive floor;
  floor.name = "ground";
  floor.friction = 0.8;
  config::TerrainScene::Box box;
  box.center = Eigen::Vector3d(0.0, 0.0, -0.5);
  box.half_extents = Eigen::Vector3d(20.0, 20.0, 0.5);
  floor.shape = box;
  scene.primitives.push_back(floor);
  if (wall) {
    config::TerrainScene::Primitive panel = floor;
    panel.name = "wall";
    box.center = Eigen::Vector3d(0.0, 0.0, 1.5);
    box.half_extents = Eigen::Vector3d(0.05, 20.0, 1.5);
    panel.shape = box;
    scene.primitives.push_back(panel);
  }
  absl::StatusOr<TerrainModel> terrain = TerrainModel::Create(scene);
  EXPECT_TRUE(terrain.ok()) << terrain.status();
  return *std::move(terrain);
}

TerrainModel floorTerrain() {
  return terrainOf(/*wall=*/false);
}

/** The planner's settings of the tests: a 3 cm CoM margin, 90 % of the
 * effort limits. */
StanceMotionOptions defaultOptions() {
  StanceMotionOptions options;
  options.comMargin = 0.03;
  options.torqueMargin = 0.9;
  return options;
}

/** The stance IK of Atlas on `terrain`. */
std::unique_ptr<StanceInverseKinematics> atlasIk(TerrainModel terrain) {
  const absl::StatusOr<std::string> contactSetPath =
      robot::resolveResourcePath(kAtlasContactSet);
  EXPECT_TRUE(contactSetPath.ok()) << contactSetPath.status();
  absl::StatusOr<config::ContactSet> contactSetConfig =
      loadContactSetFile(*contactSetPath);
  EXPECT_TRUE(contactSetConfig.ok()) << contactSetConfig.status();
  absl::StatusOr<ContactSet> contactSet = ContactSet::Create(*contactSetConfig);
  EXPECT_TRUE(contactSet.ok()) << contactSet.status();
  const absl::StatusOr<std::string> urdf =
      robot::resolveResourcePath(kAtlasUrdf);
  EXPECT_TRUE(urdf.ok()) << urdf.status();
  absl::StatusOr<std::unique_ptr<RobotStanceModel>> model =
      RobotStanceModel::Create(*urdf, *contactSet);
  EXPECT_TRUE(model.ok()) << model.status();
  const absl::StatusOr<std::string> path =
      robot::resolveResourcePath(kAtlasCollisionModel);
  EXPECT_TRUE(path.ok()) << path.status();
  absl::StatusOr<config::CollisionModel> config = loadCollisionModelFile(*path);
  EXPECT_TRUE(config.ok()) << config.status();
  absl::StatusOr<CollisionModel> collision =
      CollisionModel::Create(*config, **model);
  EXPECT_TRUE(collision.ok()) << collision.status();
  absl::StatusOr<std::unique_ptr<StanceInverseKinematics>> ik =
      StanceInverseKinematics::Create(*std::move(model), std::move(terrain),
                                      *std::move(collision), StanceIkOptions());
  EXPECT_TRUE(ik.ok()) << ik.status();
  return *std::move(ik);
}

/** The stance motions of Atlas on `terrain` with `options`. */
std::unique_ptr<StanceMotion> atlasMotion(const StanceMotionOptions& options,
                                          TerrainModel terrain) {
  absl::StatusOr<std::unique_ptr<StanceMotion>> motion =
      StanceMotion::Create(atlasIk(std::move(terrain)), options);
  EXPECT_TRUE(motion.ok()) << motion.status();
  return *std::move(motion);
}

/** The stance motions of Atlas on the floor with `options`. */
std::unique_ptr<StanceMotion> atlasMotion(const StanceMotionOptions& options) {
  return atlasMotion(options, floorTerrain());
}

/** atlasMotion() with defaultOptions(). */
std::unique_ptr<StanceMotion> atlasMotion() {
  return atlasMotion(defaultOptions());
}

Eigen::VectorXd standing(const RobotStanceModel& model) {
  RobotConfiguration configuration;
  configuration.basePosition = Eigen::Vector3d(0.0, 0.0, 0.9232);
  configuration.jointNames = {"l_leg_hpy", "r_leg_hpy", "l_leg_kny",
                              "r_leg_kny", "l_leg_aky", "r_leg_aky",
                              "l_arm_shx", "r_arm_shx"};
  configuration.jointPositions.resize(8);
  configuration.jointPositions << -0.3, -0.3, 0.6, 0.6, -0.3, -0.3, -1.3, 1.3;
  absl::StatusOr<Eigen::VectorXd> q =
      model.generalizedPositionOf(configuration);
  EXPECT_TRUE(q.ok()) << q.status();
  return *q;
}

ContactPlacement onFloor(int candidate, const Eigen::Vector3d& position) {
  return ContactPlacement{.candidate = candidate,
                          .primitive = 0,
                          .face = BoxFace::kPositiveZ,
                          .position = position,
                          .normal = Eigen::Vector3d::UnitZ(),
                          .xAxis = Eigen::Vector3d::UnitX()};
}

TEST(StanceMotion, StepsForwardOnTheFloor) {
  std::unique_ptr<StanceMotion> motion = atlasMotion();
  RobotStanceModel& model = motion->model();
  const Eigen::VectorXd q0 = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q0).ok());
  const int left = *model.contactSet().indexOf("foot_l_contact");
  const int right = *model.contactSet().indexOf("foot_r_contact");
  const StancePlacements both = {onFloor(left, model.padPosition(left)),
                                 onFloor(right, model.padPosition(right))};
  const StancePlacements rightOnly = withoutCandidate(both, left);

  // Shift onto the right foot.
  const absl::StatusOr<MotionCheck> shifted =
      motion->transition(both, rightOnly, q0, {q0});
  ASSERT_TRUE(shifted.ok()) << shifted.status();
  ASSERT_TRUE(shifted->feasible) << shifted->failure;
  const absl::StatusOr<SegmentCheck> toShift =
      motion->segment(both, q0, shifted->q, SegmentContacts());
  ASSERT_TRUE(toShift.ok()) << toShift.status();
  ASSERT_TRUE(toShift->feasible) << toShift->failure;

  // Put the left foot down 0.3 m ahead, with the CoM still over the right.
  const ContactPlacement ahead =
      onFloor(left, both.front().position + Eigen::Vector3d(0.3, 0.0, 0.0));
  const StancePlacements stepped = withPlacement(rightOnly, ahead);
  const absl::StatusOr<MotionCheck> landing =
      motion->transition(stepped, rightOnly, shifted->q, {shifted->q});
  ASSERT_TRUE(landing.ok()) << landing.status();
  ASSERT_TRUE(landing->feasible) << landing->failure;
  const absl::StatusOr<SegmentCheck> swing =
      motion->segment(rightOnly, shifted->q, landing->q,
                      SegmentContacts{.liftOff = left, .touchdown = ahead});
  ASSERT_TRUE(swing.ok()) << swing.status();
  ASSERT_TRUE(swing->feasible) << swing->failure;
  ASSERT_GE(swing->knots.size(), 2U);
  EXPECT_EQ(swing->knots.back(), landing->q);
  // The configuration that breaks the left foot's contact keeps every sphere
  // out of the floor but the left foot's own, which rest on it with the sole
  // (CollisionModel::terrainClearance()'s resting pads), as the certifier will
  // judge it once the contact is broken.
  ASSERT_TRUE(model.setGeneralizedPosition(shifted->q).ok());
  const absl::StatusOr<std::optional<std::string>> clear =
      motion->checkSphereClearance(shifted->q, rightOnly);
  ASSERT_TRUE(clear.ok()) << clear.status();
  EXPECT_FALSE(clear->has_value()) << clear->value_or("");
  // The swinging foot clears the floor in the middle of the swing.
  ASSERT_TRUE(
      model.setGeneralizedPosition(swing->knots[swing->knots.size() / 2]).ok());
  EXPECT_GT(model.padPosition(left).z(), 0.02);
}

TEST(StanceMotion, TheStaticsCheckJudgesTheTorquesOnlyUnderPositionAndEffort) {
  // A torque margin no stance keeps: standing takes more than 1 % of some
  // joint's effort limit.
  StanceMotionOptions options = defaultOptions();
  options.torqueMargin = 0.01;
  for (const JointLimits limits :
       {JointLimits::kPositionAndEffort, JointLimits::kPositionOnly}) {
    options.jointLimits = limits;
    std::unique_ptr<StanceMotion> motion = atlasMotion(options);
    RobotStanceModel& model = motion->model();
    const Eigen::VectorXd q0 = standing(model);
    ASSERT_TRUE(model.setGeneralizedPosition(q0).ok());
    const int left = *model.contactSet().indexOf("foot_l_contact");
    const int right = *model.contactSet().indexOf("foot_r_contact");
    const StancePlacements both = {onFloor(left, model.padPosition(left)),
                                   onFloor(right, model.padPosition(right))};
    const absl::StatusOr<std::optional<std::string>> failure =
        motion->checkStatics(q0, both);
    ASSERT_TRUE(failure.ok()) << failure.status();
    if (limits == JointLimits::kPositionAndEffort) {
      ASSERT_TRUE(failure->has_value());
      EXPECT_NE((*failure)->find("torque utilization"), std::string::npos)
          << **failure;
    } else {
      EXPECT_FALSE(failure->has_value()) << failure->value_or("");
    }
    // Without equilibrium the check fails under both: the left foot alone
    // does not hold the CoM between the feet.
    const absl::StatusOr<std::optional<std::string>> oneFoot =
        motion->checkStatics(q0, withoutCandidate(both, right));
    ASSERT_TRUE(oneFoot.ok()) << oneFoot.status();
    EXPECT_TRUE(oneFoot->has_value());
  }
}

TEST(StanceMotion, ASphereInTheTerrainCountsOutsideTheStancesPadsOnly) {
  std::unique_ptr<StanceMotion> motion = atlasMotion();
  RobotStanceModel& model = motion->model();
  const Eigen::VectorXd q0 = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q0).ok());
  const int left = *model.contactSet().indexOf("foot_l_contact");
  const int right = *model.contactSet().indexOf("foot_r_contact");
  const StancePlacements both = {onFloor(left, model.padPosition(left)),
                                 onFloor(right, model.padPosition(right))};
  const StancePlacements rightOnly = withoutCandidate(both, left);
  // Standing: the soles rest on the floor, within the tolerance.
  const absl::StatusOr<std::optional<std::string>> standingClear =
      motion->checkSphereClearance(q0, rightOnly);
  ASSERT_TRUE(standingClear.ok()) << standingClear.status();
  EXPECT_FALSE(standingClear->has_value()) << standingClear->value_or("");
  // Sunk 3 cm into the floor: the left sole's spheres count when the left
  // foot is not in the stance, and are exempt when it is.
  Eigen::VectorXd sunk = q0;
  sunk(2) -= 0.03;
  const absl::StatusOr<std::optional<std::string>> swinging =
      motion->checkSphereClearance(sunk, rightOnly);
  ASSERT_TRUE(swinging.ok()) << swinging.status();
  ASSERT_TRUE(swinging->has_value());
  EXPECT_NE((*swinging)->find("l_foot"), std::string::npos) << **swinging;
  const absl::StatusOr<std::optional<std::string>> held =
      motion->checkSphereClearance(sunk, both);
  ASSERT_TRUE(held.ok()) << held.status();
  EXPECT_FALSE(held->has_value()) << held->value_or("");
}

TEST(StanceMotion, CreateRefusesABadSweepOrPostureLead) {
  for (const double lead : {0.0, 1.5}) {
    StanceMotionOptions options = defaultOptions();
    options.pointTouchdownPostureLead = lead;
    EXPECT_EQ(
        StanceMotion::Create(atlasIk(floorTerrain()), options).status().code(),
        absl::StatusCode::kInvalidArgument)
        << lead;
  }
  for (const SweepResolution& sweep :
       {SweepResolution{.joint = 0.0}, SweepResolution{.position = -0.01},
        SweepResolution{.minIntervals = 0},
        SweepResolution{.tolerance = -0.001}}) {
    StanceMotionOptions options = defaultOptions();
    options.sweep = sweep;
    EXPECT_EQ(
        StanceMotion::Create(atlasIk(floorTerrain()), options).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
}

TEST(StanceMotion, KnotsInterpolateAsTheMpcsReference) {
  std::unique_ptr<StanceMotion> motion = atlasMotion();
  const Eigen::VectorXd from = standing(motion->model());
  Eigen::VectorXd to = from;
  to.head<3>() += Eigen::Vector3d(0.4, -0.2, 0.1);
  // A turn of 1 rad about z; the base's quaternion is (x, y, z, w).
  const Eigen::Quaterniond turned(
      Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitZ()));
  to.segment<4>(3) = turned.coeffs();
  to.tail(3) += Eigen::Vector3d(0.3, -0.6, 0.9);
  EXPECT_TRUE(knotInterpolation(from, to, /*fraction=*/0.0).isApprox(from));
  EXPECT_TRUE(knotInterpolation(from, to, /*fraction=*/1.0).isApprox(to));
  const Eigen::VectorXd middle = knotInterpolation(from, to, /*fraction=*/0.25);
  // The position and the joints linearly ...
  EXPECT_TRUE(middle.head<3>().isApprox(
      from.head<3>() + 0.25 * (to.head<3>() - from.head<3>())));
  EXPECT_TRUE(middle.tail(3).isApprox(from.tail(3) +
                                      0.25 * (to.tail(3) - from.tail(3))));
  // ... and the orientation a quarter of the way along the turn.
  const Eigen::Quaterniond quarter(
      Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitZ()));
  EXPECT_TRUE(middle.segment<4>(3).isApprox(quarter.coeffs()));
}

TEST(StanceMotion, ASweepThroughTheTerrainBetweenClearKnotsIsFound) {
  // Atlas in the air on either side of a wall: both knots clear the terrain,
  // the motion between them passes through the wall.
  StanceMotionOptions options = defaultOptions();
  options.sweep = SweepResolution();
  std::unique_ptr<StanceMotion> motion =
      atlasMotion(options, terrainOf(/*wall=*/true));
  Eigen::VectorXd before = standing(motion->model());
  before.head<3>() = Eigen::Vector3d(-0.6, 0.0, 1.5);
  Eigen::VectorXd after = before;
  after.x() = 0.6;
  const StancePlacements none;
  for (const Eigen::VectorXd& knot : {before, after}) {
    const absl::StatusOr<std::optional<std::string>> clear =
        motion->checkSphereClearance(knot, none);
    ASSERT_TRUE(clear.ok()) << clear.status();
    EXPECT_FALSE(clear->has_value()) << clear->value_or("");
  }
  const absl::StatusOr<std::optional<std::string>> swept =
      motion->sweptClearance(before, after, none);
  ASSERT_TRUE(swept.ok()) << swept.status();
  ASSERT_TRUE(swept->has_value());
  EXPECT_NE((*swept)->find("between two knots"), std::string::npos) << **swept;
  // Without a sweep resolution, only the knots are checked.
  std::unique_ptr<StanceMotion> knotsOnly =
      atlasMotion(defaultOptions(), terrainOf(/*wall=*/true));
  const absl::StatusOr<std::optional<std::string>> unchecked =
      knotsOnly->sweptClearance(before, after, none);
  ASSERT_TRUE(unchecked.ok()) << unchecked.status();
  EXPECT_FALSE(unchecked->has_value());
}

TEST(StanceMotion, AFreePadsStraightPathThroughTheTerrainIsFound) {
  // The same knots, with the left knee among the pads kept out of the
  // terrain: the straight path of its sphere, which the MPC has a long swing
  // follow from knot to knot, goes through the wall.
  StanceMotionOptions options = defaultOptions();
  options.sweep = SweepResolution();
  std::unique_ptr<StanceMotion> motion =
      atlasMotion(options, terrainOf(/*wall=*/true));
  options.clearedPads = {
      *motion->model().contactSet().indexOf("knee_l_contact")};
  motion = atlasMotion(options, terrainOf(/*wall=*/true));
  Eigen::VectorXd before = standing(motion->model());
  before.head<3>() = Eigen::Vector3d(-0.6, 0.0, 1.5);
  Eigen::VectorXd after = before;
  after.x() = 0.6;
  const absl::StatusOr<std::optional<std::string>> swept =
      motion->sweptClearance(before, after, StancePlacements());
  ASSERT_TRUE(swept.ok()) << swept.status();
  ASSERT_TRUE(swept->has_value());
  EXPECT_NE((*swept)->find("straight path of the pad of knee_l_contact"),
            std::string::npos)
      << **swept;
}

TEST(StanceMotion, ASegmentsStepsClearTheTerrainBetweenItsKnots) {
  // The step of StepsForwardOnTheFloor with the sweeps checked: every step
  // between consecutive knots, the start's included, clears the floor.
  StanceMotionOptions options = defaultOptions();
  options.sweep = SweepResolution();
  std::unique_ptr<StanceMotion> motion = atlasMotion(options);
  RobotStanceModel& model = motion->model();
  const Eigen::VectorXd q0 = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q0).ok());
  const int left = *model.contactSet().indexOf("foot_l_contact");
  const int right = *model.contactSet().indexOf("foot_r_contact");
  const StancePlacements both = {onFloor(left, model.padPosition(left)),
                                 onFloor(right, model.padPosition(right))};
  const StancePlacements rightOnly = withoutCandidate(both, left);
  const absl::StatusOr<MotionCheck> shifted =
      motion->transition(both, rightOnly, q0, {q0});
  ASSERT_TRUE(shifted.ok()) << shifted.status();
  ASSERT_TRUE(shifted->feasible) << shifted->failure;
  const ContactPlacement ahead =
      onFloor(left, both.front().position + Eigen::Vector3d(0.3, 0.0, 0.0));
  const absl::StatusOr<MotionCheck> landing = motion->transition(
      withPlacement(rightOnly, ahead), rightOnly, shifted->q, {shifted->q});
  ASSERT_TRUE(landing.ok()) << landing.status();
  ASSERT_TRUE(landing->feasible) << landing->failure;
  const absl::StatusOr<SegmentCheck> swing =
      motion->segment(rightOnly, shifted->q, landing->q,
                      SegmentContacts{.liftOff = left, .touchdown = ahead});
  ASSERT_TRUE(swing.ok()) << swing.status();
  ASSERT_TRUE(swing->feasible) << swing->failure;
  Eigen::VectorXd previous = shifted->q;
  for (const Eigen::VectorXd& knot : swing->knots) {
    const absl::StatusOr<std::optional<std::string>> swept =
        motion->sweptClearance(previous, knot, rightOnly);
    ASSERT_TRUE(swept.ok()) << swept.status();
    EXPECT_FALSE(swept->has_value()) << swept->value_or("");
    previous = knot;
  }
}

TEST(StanceMotion, AJumpThatTakesTheCoMIntoTheMarginIsFound) {
  // From standing over both feet to the weight on the right foot: on the
  // right foot alone, the CoM between the two knots is outside the margin
  // until the end; over both feet it keeps it.
  StanceMotionOptions options = defaultOptions();
  options.sweep = SweepResolution();
  std::unique_ptr<StanceMotion> motion = atlasMotion(options);
  RobotStanceModel& model = motion->model();
  const Eigen::VectorXd q0 = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q0).ok());
  const int left = *model.contactSet().indexOf("foot_l_contact");
  const int right = *model.contactSet().indexOf("foot_r_contact");
  const StancePlacements both = {onFloor(left, model.padPosition(left)),
                                 onFloor(right, model.padPosition(right))};
  const StancePlacements rightOnly = withoutCandidate(both, left);
  const absl::StatusOr<MotionCheck> shifted =
      motion->transition(both, rightOnly, q0, {q0});
  ASSERT_TRUE(shifted.ok()) << shifted.status();
  ASSERT_TRUE(shifted->feasible) << shifted->failure;
  const absl::StatusOr<std::optional<std::string>> outside =
      motion->jumpComDefect(q0, shifted->q, rightOnly);
  ASSERT_TRUE(outside.ok()) << outside.status();
  ASSERT_TRUE(outside->has_value());
  EXPECT_TRUE(absl::StrContains(**outside, "through a jump, the CoM"))
      << **outside;
  const absl::StatusOr<std::optional<std::string>> inside =
      motion->jumpComDefect(q0, shifted->q, both);
  ASSERT_TRUE(inside.ok()) << inside.status();
  EXPECT_FALSE(inside->has_value()) << inside->value_or("");
}

TEST(StanceMotion, PadsOverlap) {
  std::unique_ptr<StanceMotion> motion = atlasMotion();
  const ContactSet& contactSet = motion->model().contactSet();
  const int left = *contactSet.indexOf("foot_l_contact");
  const int right = *contactSet.indexOf("foot_r_contact");
  const int knee = *contactSet.indexOf("knee_l_contact");
  const ContactPlacement leftFoot = onFloor(left, Eigen::Vector3d(0, 0.11, 0));
  // Side by side at the nominal width: apart.
  EXPECT_FALSE(placementsOverlap(
      leftFoot, onFloor(right, Eigen::Vector3d(0, -0.11, 0)), contactSet,
      /*margin=*/0.03));
  // Heel to toe on one line: overlapping.
  EXPECT_TRUE(placementsOverlap(
      leftFoot, onFloor(right, Eigen::Vector3d(0.2, 0.11, 0)), contactSet,
      /*margin=*/0.03));
  // A knee on the sole: overlapping; beside it: apart.
  EXPECT_TRUE(placementsOverlap(leftFoot,
                                onFloor(knee, Eigen::Vector3d(0.05, 0.11, 0)),
                                contactSet, /*margin=*/0.03));
  EXPECT_FALSE(placementsOverlap(leftFoot,
                                 onFloor(knee, Eigen::Vector3d(0.05, 0.3, 0)),
                                 contactSet, /*margin=*/0.03));
  // On another surface: apart.
  ContactPlacement raised = onFloor(right, Eigen::Vector3d(0.0, 0.11, 0.3));
  EXPECT_FALSE(placementsOverlap(leftFoot, raised, contactSet,
                                 /*margin=*/0.03));
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
