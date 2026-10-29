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

// CollisionModel: the spheres of capsules, boxes and given spheres, the
// validation of a model, the neighborhood of the active pads, and the terrain
// and self clearances of the DRC Atlas's model.

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
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
constexpr double kTolerance = 1.0e-9;

std::unique_ptr<RobotStanceModel> atlasModel() {
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
  return *std::move(model);
}

CollisionModel atlasCollisionModel(const RobotStanceModel& model) {
  const absl::StatusOr<std::string> path =
      robot::resolveResourcePath(kAtlasCollisionModel);
  EXPECT_TRUE(path.ok()) << path.status();
  absl::StatusOr<config::CollisionModel> config = loadCollisionModelFile(*path);
  EXPECT_TRUE(config.ok()) << config.status();
  absl::StatusOr<CollisionModel> collision =
      CollisionModel::Create(*config, model);
  EXPECT_TRUE(collision.ok()) << collision.status();
  return *std::move(collision);
}

TerrainModel floorTerrain() {
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
  absl::StatusOr<TerrainModel> terrain = TerrainModel::Create(scene);
  EXPECT_TRUE(terrain.ok()) << terrain.status();
  return *std::move(terrain);
}

/** Standing on the floor at the scenarios' nominal pose. */
RobotConfiguration standing() {
  RobotConfiguration configuration;
  configuration.basePosition = Eigen::Vector3d(0.0, 0.0, 0.9232);
  configuration.jointNames = {"l_leg_hpy", "r_leg_hpy", "l_leg_kny",
                              "r_leg_kny", "l_leg_aky", "r_leg_aky",
                              "l_arm_shx", "r_arm_shx"};
  configuration.jointPositions.resize(8);
  configuration.jointPositions << -0.3, -0.3, 0.6, 0.6, -0.3, -0.3, -1.3, 1.3;
  return configuration;
}

config::CollisionModel::Link capsuleLink(const std::string& name,
                                         const std::string& joint,
                                         const Eigen::Vector3d& end,
                                         double radius) {
  config::CollisionModel::Link link;
  link.name = name;
  link.parent_joint = joint;
  config::CollisionModel::Capsule capsule;
  capsule.start = Eigen::Vector3d::Zero();
  capsule.end = end;
  capsule.radius = radius;
  link.shape = capsule;
  return link;
}

TEST(CollisionModel, CapsuleSpheresAreAtMostOneRadiusApart) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  config::CollisionModel config;
  config.robot = "test";
  config.links.push_back(capsuleLink(
      "shin", "l_leg_kny", Eigen::Vector3d(0.0, 0.0, -0.42), /*radius=*/0.055));
  const absl::StatusOr<CollisionModel> collision =
      CollisionModel::Create(config, *model);
  ASSERT_TRUE(collision.ok()) << collision.status();
  // 0.42 / 0.055 = 7.6: eight intervals, nine spheres.
  ASSERT_EQ(collision->spheres().size(), 9U);
  EXPECT_NEAR(collision->spheres().front().center.z(), 0.0, kTolerance);
  EXPECT_NEAR(collision->spheres().back().center.z(), -0.42, kTolerance);
  for (size_t i = 1; i < collision->spheres().size(); ++i) {
    const double spacing =
        (collision->spheres()[i].center - collision->spheres()[i - 1].center)
            .norm();
    EXPECT_LE(spacing, 0.055 + kTolerance);
    EXPECT_EQ(collision->spheres()[i].radius, 0.055);
  }
}

TEST(CollisionModel, BoxSpheresLieInsideTheBox) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  config::CollisionModel config;
  config.robot = "test";
  config::CollisionModel::Link link;
  link.name = "pelvis";
  link.parent_joint = "root_joint";
  config::CollisionModel::Box box;
  box.center = Eigen::Vector3d(-0.02, 0.0, 0.01);
  box.half_extents = Eigen::Vector3d(0.12, 0.15, 0.08);
  link.shape = box;
  config.links.push_back(link);
  const absl::StatusOr<CollisionModel> collision =
      CollisionModel::Create(config, *model);
  ASSERT_TRUE(collision.ok()) << collision.status();
  // x: -0.04 and 0.04; y: -0.07, 0 and 0.07; z: the center plane.
  ASSERT_EQ(collision->spheres().size(), 6U);
  for (const CollisionSphere& sphere : collision->spheres()) {
    EXPECT_EQ(sphere.radius, 0.08);
    const Eigen::Vector3d offset = (sphere.center - box.center).cwiseAbs();
    EXPECT_LE(offset.x() + sphere.radius, 0.12 + kTolerance);
    EXPECT_LE(offset.y() + sphere.radius, 0.15 + kTolerance);
    EXPECT_LE(offset.z() + sphere.radius, 0.08 + kTolerance);
  }
}

TEST(CollisionModel, RefusesBadModels) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  config::CollisionModel empty;
  EXPECT_EQ(CollisionModel::Create(empty, *model).status().code(),
            absl::StatusCode::kInvalidArgument);

  config::CollisionModel twice;
  twice.links.push_back(
      capsuleLink("a", "l_leg_kny", Eigen::Vector3d::UnitZ(), /*radius=*/0.05));
  twice.links.push_back(
      capsuleLink("a", "r_leg_kny", Eigen::Vector3d::UnitZ(), /*radius=*/0.05));
  EXPECT_EQ(CollisionModel::Create(twice, *model).status().code(),
            absl::StatusCode::kInvalidArgument);

  config::CollisionModel unknownJoint;
  unknownJoint.links.push_back(capsuleLink(
      "a", "no_such_joint", Eigen::Vector3d::UnitZ(), /*radius=*/0.05));
  EXPECT_EQ(CollisionModel::Create(unknownJoint, *model).status().code(),
            absl::StatusCode::kNotFound);

  config::CollisionModel badRadius;
  badRadius.links.push_back(
      capsuleLink("a", "l_leg_kny", Eigen::Vector3d::UnitZ(), /*radius=*/0.0));
  EXPECT_EQ(CollisionModel::Create(badRadius, *model).status().code(),
            absl::StatusCode::kInvalidArgument);

  config::CollisionModel noShape;
  config::CollisionModel::Link shapeless;
  shapeless.name = "a";
  shapeless.parent_joint = "l_leg_kny";
  noShape.links.push_back(shapeless);
  EXPECT_EQ(CollisionModel::Create(noShape, *model).status().code(),
            absl::StatusCode::kInvalidArgument);

  config::CollisionModel badPair = twice;
  badPair.links[1].name = "b";
  config::CollisionModel::SelfCollisionPair pair;
  pair.first = "a";
  pair.second = "c";
  badPair.self_collision_pairs.push_back(pair);
  EXPECT_EQ(CollisionModel::Create(badPair, *model).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CollisionModel, DistanceToPad) {
  PadRegion point{.center = Eigen::Vector3d(1.0, 0.0, 0.0),
                  .rotation = Eigen::Matrix3d::Identity(),
                  .rectangle = std::nullopt};
  EXPECT_NEAR(distanceToPad(point, Eigen::Vector3d(1.0, 0.0, 0.3)), 0.3,
              kTolerance);
  PadRegion rectangle{
      .center = Eigen::Vector3d::Zero(),
      .rotation = Eigen::Matrix3d::Identity(),
      .rectangle = ContactRectangle{
          .xMin = -0.1, .xMax = 0.1, .yMin = -0.05, .yMax = 0.05}};
  // Above the rectangle: the height; beside it: the distance to its edge.
  EXPECT_NEAR(distanceToPad(rectangle, Eigen::Vector3d(0.05, 0.0, 0.2)), 0.2,
              kTolerance);
  EXPECT_NEAR(distanceToPad(rectangle, Eigen::Vector3d(0.4, 0.0, 0.0)), 0.3,
              kTolerance);
}

TEST(CollisionModel, AtlasStandingClearsTheFloor) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  const CollisionModel collision = atlasCollisionModel(*model);
  const TerrainModel floor = floorTerrain();
  ASSERT_TRUE(model->setConfiguration(standing()).ok());
  const absl::StatusOr<ContactMask> feet = model->contactSet().maskOf(
      std::vector<std::string>{"foot_l_contact", "foot_r_contact"});
  ASSERT_TRUE(feet.ok()) << feet.status();

  // With the feet active, their corner spheres are exempt: the lowest sphere
  // is a shin's, above the floor.
  const SphereClearance active =
      collision.terrainClearance(*model, floor, *feet, kDefaultPadNeighborhood);
  ASSERT_TRUE(active.clearance.has_value());
  EXPECT_GT(*active.clearance, 0.0);
  EXPECT_NE(active.link, "l_foot");
  EXPECT_NE(active.link, "r_foot");

  // Without them, the soles' corner spheres touch the floor.
  const SphereClearance inactive = collision.terrainClearance(
      *model, floor, /*mask=*/0, kDefaultPadNeighborhood);
  ASSERT_TRUE(inactive.clearance.has_value());
  EXPECT_NEAR(*inactive.clearance, 0.0, 0.002);

  // Inactive but resting on the floor, the soles exempt their own spheres
  // (not the shins'): the lowest counted sphere is a shin's again.
  const SphereClearance resting = collision.terrainClearance(
      *model, floor, /*mask=*/0, kDefaultPadNeighborhood,
      /*restingTolerance=*/0.005);
  ASSERT_TRUE(resting.clearance.has_value());
  EXPECT_GT(*resting.clearance, 0.0);
  EXPECT_NE(resting.link, "l_foot");
  EXPECT_NE(resting.link, "r_foot");

  // Two centimeters lower, they penetrate by two centimeters.
  RobotConfiguration lower = standing();
  lower.basePosition.z() -= 0.02;
  ASSERT_TRUE(model->setConfiguration(lower).ok());
  const SphereClearance penetrating = collision.terrainClearance(
      *model, floor, /*mask=*/0, kDefaultPadNeighborhood);
  ASSERT_TRUE(penetrating.clearance.has_value());
  EXPECT_NEAR(*penetrating.clearance, -0.02, 0.002);
  // Two centimeters into the floor, a sole no longer rests on it.
  const SphereClearance sunk = collision.terrainClearance(
      *model, floor, /*mask=*/0, kDefaultPadNeighborhood,
      /*restingTolerance=*/0.005);
  ASSERT_TRUE(sunk.clearance.has_value());
  EXPECT_NEAR(*sunk.clearance, -0.02, 0.002);
}

TEST(CollisionModel, APointPadExemptsItsLegBelowAndTheLinkAboveAsNamed) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  const ContactSet& contacts = model->contactSet();
  const int knee = *contacts.indexOf("knee_l_contact");
  const int foot = *contacts.indexOf("foot_l_contact");
  const pinocchio::JointIndex shin = *model->jointIndex("l_leg_kny");
  const pinocchio::JointIndex ankle = *model->jointIndex("l_leg_akx");
  const pinocchio::JointIndex thigh = *model->jointIndex("l_leg_hpy");
  const pinocchio::JointIndex otherShin = *model->jointIndex("r_leg_kny");
  // The knee: the shin and the foot below it, not the thigh above, whose
  // spheres beside the pad may rest kRestingLinkDepth in the terrain.
  const PadRegion kneeRegion{
      .joints = exemptJointsOf(*model, knee, LinkAbove::kRests),
      .restingJoints = restingJointsOf(*model, knee, LinkAbove::kRests)};
  EXPECT_TRUE(exemptsJoint(kneeRegion, shin));
  EXPECT_TRUE(exemptsJoint(kneeRegion, ankle));
  EXPECT_FALSE(exemptsJoint(kneeRegion, thigh));
  EXPECT_FALSE(exemptsJoint(kneeRegion, otherShin));
  EXPECT_EQ(kneeRegion.restingJoints,
            std::vector<pinocchio::JointIndex>{thigh});
  // A sole exempts every link near its rectangle; its own joints are the
  // foot's, and none rests.
  EXPECT_TRUE(exemptJointsOf(*model, foot, LinkAbove::kRests).empty());
  EXPECT_TRUE(restingJointsOf(*model, foot, LinkAbove::kRests).empty());
  // Under kExempt the knee exempts the thigh's end beside it, and none rests.
  const std::vector<pinocchio::JointIndex> exempting =
      exemptJointsOf(*model, knee, LinkAbove::kExempt);
  EXPECT_NE(std::find(exempting.begin(), exempting.end(), thigh),
            exempting.end());
  EXPECT_TRUE(restingJointsOf(*model, knee, LinkAbove::kExempt).empty());
  EXPECT_EQ(*linkAboveOf("rests"), LinkAbove::kRests);
  EXPECT_EQ(linkAboveOf("whole").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(exemptsJoint(PadRegion{}, thigh));
  const std::vector<pinocchio::JointIndex> footJoints =
      ownJointsOf(*model, foot);
  EXPECT_EQ(footJoints, std::vector<pinocchio::JointIndex>{ankle});
  // A sphere of the thigh beside the knee's pad counts, but may rest: one up
  // the thigh may not, nor one of the other leg beside it.
  const Eigen::Vector3d nearPad = Eigen::Vector3d::Zero();
  EXPECT_FALSE(CollisionModel::isExempt({kneeRegion}, thigh, nearPad,
                                        /*radius=*/0.06,
                                        kDefaultPadNeighborhood));
  EXPECT_EQ(CollisionModel::restingAllowance({kneeRegion}, thigh, nearPad,
                                             /*radius=*/0.06,
                                             kDefaultPadNeighborhood),
            kRestingLinkDepth);
  EXPECT_EQ(CollisionModel::restingAllowance(
                {kneeRegion}, thigh, Eigen::Vector3d(0.0, 0.0, 0.15),
                /*radius=*/0.06, kDefaultPadNeighborhood),
            0.0);
  EXPECT_EQ(CollisionModel::restingAllowance({kneeRegion}, otherShin, nearPad,
                                             /*radius=*/0.06,
                                             kDefaultPadNeighborhood),
            0.0);
  EXPECT_TRUE(CollisionModel::isExempt({kneeRegion}, shin, nearPad,
                                       /*radius=*/0.06,
                                       kDefaultPadNeighborhood));
}

TEST(CollisionModel, CrossedLegsCollide) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  const CollisionModel collision = atlasCollisionModel(*model);
  ASSERT_TRUE(model->setConfiguration(standing()).ok());
  const SphereClearance apart = collision.selfClearance(*model);
  ASSERT_TRUE(apart.clearance.has_value());
  std::cerr << "DEBUG standing self clearance " << *apart.clearance << " "
            << apart.link << "\n";
  EXPECT_GT(*apart.clearance, 0.0);

  // Both hips rolled inward by 0.5 rad cross the shins.
  RobotConfiguration crossed = standing();
  crossed.jointNames.push_back("l_leg_hpx");
  crossed.jointNames.push_back("r_leg_hpx");
  crossed.jointPositions.conservativeResize(crossed.jointPositions.size() + 2);
  crossed.jointPositions.tail(2) << -0.5, 0.5;
  ASSERT_TRUE(model->setConfiguration(crossed).ok());
  const SphereClearance overlapping = collision.selfClearance(*model);
  ASSERT_TRUE(overlapping.clearance.has_value());
  EXPECT_LT(*overlapping.clearance, 0.0);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
