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

// StanceInverseKinematics on the DRC Atlas: holding a stance, shifting the CoM
// onto one foot, reaching a hand onto a box, refusing an unreachable target,
// keeping the collision spheres off the floor and staying in the joint limits.

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/joint-configuration.hpp"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/planner/StanceInverseKinematics.h"
#include "humanoid_multi_contact/planner/StanceMotion.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"
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

std::unique_ptr<RobotStanceModel> atlasModel() {
  const absl::StatusOr<std::string> contactSetPath =
      robot::resolveResourcePath(kAtlasContactSet);
  EXPECT_TRUE(contactSetPath.ok()) << contactSetPath.status();
  absl::StatusOr<config::ContactSet> config =
      loadContactSetFile(*contactSetPath);
  EXPECT_TRUE(config.ok()) << config.status();
  absl::StatusOr<ContactSet> contactSet = ContactSet::Create(*config);
  EXPECT_TRUE(contactSet.ok()) << contactSet.status();
  const absl::StatusOr<std::string> urdf =
      robot::resolveResourcePath(kAtlasUrdf);
  EXPECT_TRUE(urdf.ok()) << urdf.status();
  absl::StatusOr<std::unique_ptr<RobotStanceModel>> model =
      RobotStanceModel::Create(*urdf, *contactSet);
  EXPECT_TRUE(model.ok()) << model.status();
  return *std::move(model);
}

/** The floor and a box of `center` and `halfExtents` on it. */
TerrainModel floorAndBox(const Eigen::Vector3d& center,
                         const Eigen::Vector3d& halfExtents) {
  config::TerrainScene scene;
  scene.name = "box";
  for (const bool box : {false, true}) {
    config::TerrainScene::Primitive primitive;
    primitive.name = box ? "box" : "ground";
    primitive.friction = 0.8;
    config::TerrainScene::Box shape;
    shape.center = box ? center : Eigen::Vector3d(0.0, 0.0, -0.5);
    shape.half_extents = box ? halfExtents : Eigen::Vector3d(20.0, 20.0, 0.5);
    primitive.shape = shape;
    scene.primitives.push_back(primitive);
  }
  absl::StatusOr<TerrainModel> terrain = TerrainModel::Create(scene);
  EXPECT_TRUE(terrain.ok()) << terrain.status();
  return *std::move(terrain);
}

/** A 0.6 m box 0.4 m in front of the robot, as box_climb. */
TerrainModel boxTerrain() {
  return floorAndBox(Eigen::Vector3d(0.9, 0.0, 0.3),
                     Eigen::Vector3d(0.5, 0.5, 0.3));
}

/** The first stair of the stairs scenario: 0.18 m high, 0.5 m ahead. */
TerrainModel stepTerrain() {
  return floorAndBox(Eigen::Vector3d(0.64, 0.0, 0.09),
                     Eigen::Vector3d(0.14, 0.6, 0.09));
}

std::unique_ptr<StanceInverseKinematics> atlasIk(
    TerrainModel terrain = boxTerrain()) {
  std::unique_ptr<RobotStanceModel> model = atlasModel();
  const absl::StatusOr<std::string> path =
      robot::resolveResourcePath(kAtlasCollisionModel);
  EXPECT_TRUE(path.ok()) << path.status();
  absl::StatusOr<config::CollisionModel> config = loadCollisionModelFile(*path);
  EXPECT_TRUE(config.ok()) << config.status();
  absl::StatusOr<CollisionModel> collision =
      CollisionModel::Create(*config, *model);
  EXPECT_TRUE(collision.ok()) << collision.status();
  absl::StatusOr<std::unique_ptr<StanceInverseKinematics>> ik =
      StanceInverseKinematics::Create(std::move(model), std::move(terrain),
                                      *std::move(collision), StanceIkOptions());
  EXPECT_TRUE(ik.ok()) << ik.status();
  return *std::move(ik);
}

/** The scenarios' nominal standing pose, as a generalized position. */
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

int candidateOf(const RobotStanceModel& model, const std::string& name) {
  const absl::StatusOr<int> index = model.contactSet().indexOf(name);
  EXPECT_TRUE(index.ok()) << index.status();
  return *index;
}

/** The placement of a pad where the model has it, on a horizontal surface. */
ContactPlacement placed(const RobotStanceModel& model, int candidate) {
  return ContactPlacement{.candidate = candidate,
                          .primitive = 0,
                          .face = BoxFace::kPositiveZ,
                          .position = model.padPosition(candidate),
                          .normal = Eigen::Vector3d::UnitZ(),
                          .xAxis = Eigen::Vector3d::UnitX()};
}

/** The IK problem of holding `stance` with the CoM in `polygon`. */
StanceIkProblem holding(const RobotStanceModel& model,
                        const StancePlacements& stance,
                        std::vector<Eigen::Vector2d> polygon,
                        double margin) {
  StanceIkProblem problem;
  for (const ContactPlacement& placement : stance) {
    problem.contacts.push_back(ContactTarget{.candidate = placement.candidate,
                                             .position = placement.position,
                                             .normal = placement.normal,
                                             .xAxis = placement.xAxis});
    problem.exemptPads.push_back(padRegionOf(placement, model.contactSet()));
  }
  problem.comPolygon = std::move(polygon);
  problem.comMargin = margin;
  return problem;
}

/** The rectangle of a foot pad at `center`, counterclockwise. */
std::vector<Eigen::Vector2d> footPolygon(const Eigen::Vector3d& center) {
  const Eigen::Vector2d c = center.head<2>();
  return {c + Eigen::Vector2d(-0.12, -0.055), c + Eigen::Vector2d(0.12, -0.055),
          c + Eigen::Vector2d(0.12, 0.055), c + Eigen::Vector2d(-0.12, 0.055)};
}

TEST(StanceInverseKinematics, HoldsTheStandingStance) {
  std::unique_ptr<StanceInverseKinematics> ik = atlasIk();
  RobotStanceModel& model = ik->model();
  const Eigen::VectorXd q = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q).ok());
  const StancePlacements feet = {
      placed(model, candidateOf(model, "foot_l_contact")),
      placed(model, candidateOf(model, "foot_r_contact"))};
  StanceIkProblem problem =
      holding(model, feet, /*polygon=*/{}, /*margin=*/0.0);
  problem.postureReference = q;
  const absl::StatusOr<StanceIkResult> result = ik->solve(problem, q);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->feasible) << result->failure;
  EXPECT_LT(result->maxReachError, 1.0e-4);
  // The joints at a limit (the elbows and wrists of this pose) move the IK's
  // margin inside it; the rest stays.
  EXPECT_LT(
      pinocchio::difference(model.model(), ik->clampToLimits(q), result->q)
          .norm(),
      1.0e-3);
}

TEST(StanceInverseKinematics, ShiftsTheComOntoOneFoot) {
  std::unique_ptr<StanceInverseKinematics> ik = atlasIk();
  RobotStanceModel& model = ik->model();
  const Eigen::VectorXd q = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q).ok());
  const int right = candidateOf(model, "foot_r_contact");
  const StancePlacements feet = {
      placed(model, candidateOf(model, "foot_l_contact")),
      placed(model, right)};
  constexpr double kMargin = 0.03;
  StanceIkProblem problem =
      holding(model, feet, footPolygon(model.padPosition(right)), kMargin);
  problem.postureReference = q;
  const absl::StatusOr<StanceIkResult> result = ik->solve(problem, q);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->feasible)
      << result->failure << " after " << result->iterations << " iterations ("
      << result->termination << ")";
  ASSERT_TRUE(result->comMargin.has_value());
  EXPECT_GE(*result->comMargin, kMargin - 1.0e-3);
  EXPECT_LE(result->maxReachError, kDefaultReachTolerance);
  for (const ActuatedJoint& joint : model.actuatedJoints()) {
    EXPECT_GE(result->q(joint.positionIndex), joint.lowerLimit) << joint.name;
    EXPECT_LE(result->q(joint.positionIndex), joint.upperLimit) << joint.name;
  }
}

TEST(StanceInverseKinematics, ReachesAHandOntoTheBox) {
  std::unique_ptr<StanceInverseKinematics> ik = atlasIk();
  RobotStanceModel& model = ik->model();
  const Eigen::VectorXd q = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q).ok());
  const int left = candidateOf(model, "foot_l_contact");
  const int right = candidateOf(model, "foot_r_contact");
  const int hand = candidateOf(model, "hand_r_contact");
  const StancePlacements feet = {placed(model, left), placed(model, right)};
  const ContactPlacement onBox{.candidate = hand,
                               .primitive = 1,
                               .face = BoxFace::kPositiveZ,
                               .position = Eigen::Vector3d(0.5, -0.25, 0.6),
                               .normal = Eigen::Vector3d::UnitZ(),
                               .xAxis = Eigen::Vector3d::UnitX()};
  StanceIkProblem problem =
      holding(model, withPlacement(feet, onBox), /*polygon=*/{},
              /*margin=*/0.0);
  problem.postureReference = q;
  const absl::StatusOr<StanceIkResult> result = ik->solve(problem, q);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_TRUE(result->feasible)
      << result->failure << " after " << result->iterations << " iterations ("
      << result->termination << ")";
  EXPECT_LE(result->maxReachError, kDefaultReachTolerance);
  EXPECT_LE(result->maxPointTilt, kDefaultMaxPadTilt);
  ASSERT_TRUE(result->terrainClearance.has_value());
  EXPECT_GE(*result->terrainClearance, 0.0);
}

TEST(StanceInverseKinematics, RefusesAnUnreachableTarget) {
  std::unique_ptr<StanceInverseKinematics> ik = atlasIk();
  RobotStanceModel& model = ik->model();
  const Eigen::VectorXd q = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q).ok());
  const StancePlacements feet = {
      placed(model, candidateOf(model, "foot_l_contact")),
      placed(model, candidateOf(model, "foot_r_contact"))};
  // A hand 2.5 m in front of the feet: beyond the arm, the leaning torso and
  // the legs together.
  const ContactPlacement farAway{
      .candidate = candidateOf(model, "hand_l_contact"),
      .primitive = 1,
      .position = Eigen::Vector3d(2.5, 0.2, 0.6),
      .normal = Eigen::Vector3d::UnitZ(),
      .xAxis = Eigen::Vector3d::UnitX()};
  StanceIkProblem problem =
      holding(model, withPlacement(feet, farAway), /*polygon=*/{},
              /*margin=*/0.0);
  problem.postureReference = q;
  const absl::StatusOr<StanceIkResult> result = ik->solve(problem, q);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->feasible);
  EXPECT_GT(result->maxReachError, kDefaultReachTolerance);
}

TEST(StanceInverseKinematics, KeepsTheShinsOffTheFloorWhenCrouching) {
  std::unique_ptr<StanceInverseKinematics> ik = atlasIk();
  RobotStanceModel& model = ik->model();
  const Eigen::VectorXd q = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q).ok());
  const StancePlacements feet = {
      placed(model, candidateOf(model, "foot_l_contact")),
      placed(model, candidateOf(model, "foot_r_contact"))};
  StanceIkProblem problem =
      holding(model, feet, /*polygon=*/{}, /*margin=*/0.0);
  // A reference with the base 0.6 m lower: a deep squat, which the knee and
  // ankle limits and the floor end before the base gets there.
  problem.postureReference = q;
  problem.postureReference.z() -= 0.6;
  problem.comTarget = model.centerOfMass() - Eigen::Vector3d(0.0, 0.0, 0.6);
  const absl::StatusOr<StanceIkResult> result = ik->solve(problem, q);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_LE(result->maxReachError, kDefaultReachTolerance);
  ASSERT_TRUE(result->terrainClearance.has_value());
  EXPECT_GE(*result->terrainClearance, 0.0) << result->terrainClearanceLink;
  EXPECT_LT(result->q.z(), q.z() - 0.2);
}

TEST(StanceInverseKinematics, ShiftsOntoAFootOnAStep) {
  std::unique_ptr<StanceInverseKinematics> ik = atlasIk(stepTerrain());
  RobotStanceModel& model = ik->model();
  const Eigen::VectorXd q = standing(model);
  ASSERT_TRUE(model.setGeneralizedPosition(q).ok());
  const int left = candidateOf(model, "foot_l_contact");
  const int right = candidateOf(model, "foot_r_contact");
  // The left foot on a 0.18 m step 0.6 m ahead, the right one 0.3 m ahead on
  // the floor (the box of this terrain stands in for the step).
  ContactPlacement onStep = placed(model, left);
  onStep.position = Eigen::Vector3d(0.64, 0.2, 0.18);
  ContactPlacement onFloor = placed(model, right);
  onFloor.position = Eigen::Vector3d(0.3, -0.1, 0.0);
  const StancePlacements feet = {onStep, onFloor};
  StanceIkProblem both = holding(model, feet, /*polygon=*/{}, /*margin=*/0.0);
  both.postureReference = q;
  const absl::StatusOr<StanceIkResult> placedFeet = ik->solve(both, q);
  ASSERT_TRUE(placedFeet.ok()) << placedFeet.status();
  std::cerr << "both feet: " << placedFeet->failure << " "
            << placedFeet->iterations << " " << placedFeet->termination << "\n";
  StanceIkProblem shift =
      holding(model, feet, footPolygon(onStep.position), /*margin=*/0.02);
  shift.postureReference = placedFeet->q;
  const absl::StatusOr<StanceIkResult> shifted =
      ik->solve(shift, placedFeet->q);
  ASSERT_TRUE(shifted.ok()) << shifted.status();
  std::cerr << "shifted: " << shifted->failure << " " << shifted->iterations
            << " " << shifted->termination << "\n";
}

TEST(StanceInverseKinematics, RefusesBadInput) {
  std::unique_ptr<StanceInverseKinematics> ik = atlasIk();
  StanceIkProblem problem;
  problem.postureReference = Eigen::VectorXd::Zero(3);
  EXPECT_EQ(ik->solve(problem, Eigen::VectorXd::Zero(3)).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
