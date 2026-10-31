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

// RobotStanceModel and the torque feasibility LP on the DRC Atlas: the contact
// set's parent joints, the pad kinematics, the point Jacobians, the statics of
// the torque LP's forces, and the infeasibility of scaled-down limits.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/PhysicalConstants.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/stability/Stance.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"
#include "humanoid_multi_contact/stability/TorqueFeasibility.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kAtlasContactSet[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "contact_set.textproto";

ContactSet atlasContactSet() {
  const absl::StatusOr<std::string> path =
      robot::resolveResourcePath(kAtlasContactSet);
  EXPECT_TRUE(path.ok()) << path.status();
  absl::StatusOr<config::ContactSet> config = loadContactSetFile(*path);
  EXPECT_TRUE(config.ok()) << config.status();
  absl::StatusOr<ContactSet> set = ContactSet::Create(*config);
  EXPECT_TRUE(set.ok()) << set.status();
  return *std::move(set);
}

std::unique_ptr<RobotStanceModel> atlasModel() {
  const absl::StatusOr<std::string> urdf =
      robot::resolveResourcePath(kAtlasUrdf);
  EXPECT_TRUE(urdf.ok()) << urdf.status();
  absl::StatusOr<std::unique_ptr<RobotStanceModel>> model =
      RobotStanceModel::Create(*urdf, atlasContactSet());
  EXPECT_TRUE(model.ok()) << model.status();
  return *std::move(model);
}

/** Standing on flat ground with bent knees and the arms down: the soles at z =
 * 0. */
RobotConfiguration standing() {
  RobotConfiguration configuration;
  configuration.basePosition = Eigen::Vector3d(0.0, 0.0, 0.9232239);
  configuration.jointNames = {"l_leg_hpy", "r_leg_hpy", "l_leg_kny",
                              "r_leg_kny", "l_leg_aky", "r_leg_aky",
                              "l_arm_shx", "r_arm_shx"};
  configuration.jointPositions.resize(8);
  configuration.jointPositions << -0.3, -0.3, 0.6, 0.6, -0.3, -0.3, -1.3, 1.3;
  return configuration;
}

/** The stance of the named candidates at the model's configuration, on flat
 * ground. */
Stance flatStance(const RobotStanceModel& model,
                  const std::vector<std::string>& names) {
  Stance stance;
  for (const std::string& name : names) {
    const int index = *model.contactSet().indexOf(name);
    const ContactCandidate& candidate = model.contactSet().candidate(index);
    stance.push_back(StanceContact{.name = name,
                                   .candidateIndex = index,
                                   .kind = candidate.kind,
                                   .position = model.padPosition(index),
                                   .rotation = model.padRotation(index),
                                   .friction = 0.8,
                                   .rectangle = candidate.rectangle});
  }
  return stance;
}

TEST(RobotStanceModelTest, EveryParentJointIsAJointOfTheUrdf) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  ASSERT_NE(model, nullptr);
  for (const ContactCandidate& candidate : model->contactSet().candidates()) {
    EXPECT_TRUE(model->hasJoint(candidate.parentJoint))
        << candidate.name << ": " << candidate.parentJoint;
  }
  // A candidate on an unknown joint is refused, naming it.
  config::ContactSet config;
  config.robot = "drc_atlas";
  config::ContactSet::Candidate candidate;
  candidate.name = "tail";
  candidate.kind = config::ContactSet::Kind::kPoint;
  candidate.parent_joint = "tail_joint";
  candidate.offset = Eigen::Vector3d::Zero();
  candidate.pad_normal = Eigen::Vector3d::UnitZ();
  config.candidates = {candidate};
  const absl::StatusOr<std::string> urdf =
      robot::resolveResourcePath(kAtlasUrdf);
  ASSERT_TRUE(urdf.ok());
  const absl::StatusOr<std::unique_ptr<RobotStanceModel>> refused =
      RobotStanceModel::Create(*urdf, *ContactSet::Create(config));
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kNotFound);
}

TEST(RobotStanceModelTest, PadsAtTheZeroPose) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  RobotConfiguration zero;
  zero.basePosition = Eigen::Vector3d(0.0, 0.0, 0.944);
  ASSERT_TRUE(model->setConfiguration(zero).ok());
  const ContactSet& set = model->contactSet();
  // The soles: 0.944 m below the base with straight legs (0.066 + 0.374 +
  // 0.422 + 0.082), 0.04 m ahead of the ankles.
  EXPECT_LT((model->padPosition(*set.indexOf("foot_l_contact")) -
             Eigen::Vector3d(0.04, 0.1115, 0.0))
                .norm(),
            1.0e-9);
  EXPECT_TRUE(
      model->padRotation(*set.indexOf("foot_l_contact")).isIdentity(1.0e-12));
  EXPECT_LT((model->padPosition(*set.indexOf("torso_back_contact")) -
             Eigen::Vector3d(-0.2325, 0.0, 1.456))
                .norm(),
            1.0e-9);
  // The torso pad's z axis points into the body, forward.
  EXPECT_LT((model->padRotation(*set.indexOf("torso_back_contact")).col(2) -
             Eigen::Vector3d::UnitX())
                .norm(),
            1.0e-12);
  EXPECT_GT(model->mass(), 100.0);
  EXPECT_FALSE(model->actuatedJoints().empty());
}

TEST(RobotStanceModelTest, PointJacobiansMatchDifferences) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  RobotConfiguration configuration = standing();
  ASSERT_TRUE(model->setConfiguration(configuration).ok());
  constexpr double kStep = 1.0e-6;
  for (const std::string& name :
       {std::string("hand_l_contact"), std::string("knee_r_contact"),
        std::string("thigh_l_contact")}) {
    const int index = *model->contactSet().indexOf(name);
    ASSERT_TRUE(model->setConfiguration(configuration).ok());
    const Eigen::Vector3d point = model->padPosition(index);
    const Eigen::Matrix<double, 3, Eigen::Dynamic> jacobian =
        model->pointJacobian(index, point);
    for (const ActuatedJoint& joint : model->actuatedJoints()) {
      RobotConfiguration plus = configuration;
      RobotConfiguration minus = configuration;
      plus.jointNames.push_back(joint.name);
      minus.jointNames.push_back(joint.name);
      const Eigen::Index last = configuration.jointPositions.size();
      plus.jointPositions.conservativeResize(last + 1);
      minus.jointPositions.conservativeResize(last + 1);
      // A joint already named keeps its position; only the others are probed.
      bool named = false;
      for (const std::string& existing : configuration.jointNames) {
        named = named || existing == joint.name;
      }
      if (named) continue;
      plus.jointPositions(last) = kStep;
      minus.jointPositions(last) = -kStep;
      ASSERT_TRUE(model->setConfiguration(plus).ok());
      const Eigen::Vector3d after = model->padPosition(index);
      ASSERT_TRUE(model->setConfiguration(minus).ok());
      const Eigen::Vector3d before = model->padPosition(index);
      EXPECT_LT(
          (jacobian.col(joint.velocityIndex) - (after - before) / (2.0 * kStep))
              .norm(),
          1.0e-6)
          << name << " / " << joint.name;
    }
  }
}

TEST(TorqueFeasibilityTest, TheForcesBalanceTheBaseAndScaledLimitsFail) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  ASSERT_TRUE(model->setConfiguration(standing()).ok());
  const absl::StatusOr<StaticEquilibrium> equilibrium =
      StaticEquilibrium::Create(
          flatStance(*model, {"foot_l_contact", "foot_r_contact"}),
          StaticEquilibriumOptions());
  ASSERT_TRUE(equilibrium.ok()) << equilibrium.status();
  const Eigen::Vector2d com = model->centerOfMass().head<2>();
  ASSERT_LT(*equilibrium->equilibriumResidual(com), 1.0e-6);
  const absl::StatusOr<TorqueFeasibilityResult> result =
      minimumTorqueUtilization(*model, *equilibrium);
  ASSERT_TRUE(result.ok()) << result.status();

  // g(q) - sum_g J_g^T f_g: zero on the floating base, the joint torques on the
  // joints.
  const WrenchGenerators& generators = equilibrium->generators();
  const double weight = model->mass() * kGravityMagnitude;
  Eigen::VectorXd generalized = model->generalizedGravity();
  for (int g = 0; g < generators.size(); ++g) {
    const int candidate =
        equilibrium->stance()[generators.contactIndices[g]].candidateIndex;
    generalized -=
        weight * result->scalings(g) *
        model->pointJacobian(candidate, generators.points[g]).transpose() *
        generators.directions[g];
  }
  EXPECT_LT(generalized.head<6>().norm(), 1.0e-6 * weight)
      << generalized.head<6>().transpose();
  for (int j = 0; j < static_cast<int>(model->actuatedJoints().size()); ++j) {
    EXPECT_NEAR(result->jointTorques(j),
                generalized(model->actuatedJoints()[j].velocityIndex), 1.0e-6);
  }
  EXPECT_TRUE((result->scalings.array() >= -1.0e-9).all());

  // Standing is within Atlas's limits, and not within 1% of them.
  EXPECT_GT(result->utilization, 0.0);
  EXPECT_LT(result->utilization, 1.0);
  EXPECT_FALSE(result->maxTorqueJoint.empty());
  EXPECT_TRUE(isTorqueFeasible(*result, /*torqueMargin=*/1.0));
  EXPECT_FALSE(isTorqueFeasible(*result, /*torqueMargin=*/0.01));
}

TEST(RobotStanceModelTest, RelaxingThePositionLimitsWidensBothSides) {
  const std::unique_ptr<RobotStanceModel> model = atlasModel();
  ASSERT_NE(model, nullptr);
  const std::vector<ActuatedJoint> urdf = model->actuatedJoints();
  constexpr double kRelaxation = 0.1;
  ASSERT_TRUE(model->relaxPositionLimits(kRelaxation).ok());
  ASSERT_EQ(model->actuatedJoints().size(), urdf.size());
  for (size_t i = 0; i < urdf.size(); ++i) {
    EXPECT_DOUBLE_EQ(model->actuatedJoints()[i].lowerLimit,
                     urdf[i].lowerLimit - kRelaxation);
    EXPECT_DOUBLE_EQ(model->actuatedJoints()[i].upperLimit,
                     urdf[i].upperLimit + kRelaxation);
    EXPECT_EQ(model->actuatedJoints()[i].effortLimit, urdf[i].effortLimit);
  }
  EXPECT_EQ(model->relaxPositionLimits(-kRelaxation).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
