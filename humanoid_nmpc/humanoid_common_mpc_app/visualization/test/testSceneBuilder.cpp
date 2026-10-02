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

// Pinocchio forward declarations must be included first.
#include <pinocchio/fwd.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"

#include "VisualizationTestRobot.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/SceneBuilder.h"
#include "humanoid_common_mpc_app/visualization/SceneContract.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.pb.h"

namespace ocs2::humanoid::visualization {
namespace {

using Model = PinocchioInterface::Model;
using Data = PinocchioInterface::Data;

constexpr scalar_t kTolerance = 1e-9;
constexpr size_t kNodes = 21;
constexpr scalar_t kNormalForce = 300.0;

vector3_t toVector(const humanoid_mpc_msgs::Vector3& vector) {
  return vector3_t(vector.x(), vector.y(), vector.z());
}

matrix3_t toRotation(const humanoid_mpc_msgs::Quaternion& quaternion) {
  return quaternion_t(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()).normalized().toRotationMatrix();
}

const humanoid_mpc_msgs::RobotModelInstance* findInstance(const humanoid_mpc_msgs::VisualizationScene& scene, absl::string_view name) {
  for (const humanoid_mpc_msgs::RobotModelInstance& instance : scene.robots()) {
    if (instance.name() == name) {
      return &instance;
    }
  }
  return nullptr;
}

const humanoid_mpc_msgs::Arrows& arrowsAt(const humanoid_mpc_msgs::VisualizationScene& scene, absl::string_view path) {
  for (const humanoid_mpc_msgs::Arrows& arrows : scene.arrows()) {
    if (arrows.path() == path) {
      return arrows;
    }
  }
  ADD_FAILURE() << "no arrows at " << path;
  return humanoid_mpc_msgs::Arrows::default_instance();
}

const humanoid_mpc_msgs::Spheres& spheresAt(const humanoid_mpc_msgs::VisualizationScene& scene, absl::string_view path) {
  for (const humanoid_mpc_msgs::Spheres& spheres : scene.spheres()) {
    if (spheres.path() == path) {
      return spheres;
    }
  }
  ADD_FAILURE() << "no spheres at " << path;
  return humanoid_mpc_msgs::Spheres::default_instance();
}

const humanoid_mpc_msgs::LineStrips& lineStripsAt(const humanoid_mpc_msgs::VisualizationScene& scene, absl::string_view path) {
  for (const humanoid_mpc_msgs::LineStrips& strips : scene.line_strips()) {
    if (strips.path() == path) {
      return strips;
    }
  }
  ADD_FAILURE() << "no line strips at " << path;
  return humanoid_mpc_msgs::LineStrips::default_instance();
}

/** Forward kinematics and frame placements of `interface` at `q`. */
void forwardKinematics(PinocchioInterface& interface, const vector_t& q) {
  pinocchio::forwardKinematics(interface.getModel(), interface.getData(), q);
  pinocchio::updateFramePlacements(interface.getModel(), interface.getData());
}

/** Every link pose of `instance` equals the frame of the same name in `interface`'s data. */
void expectLinkPoses(const humanoid_mpc_msgs::RobotModelInstance& instance, const PinocchioInterface& interface) {
  ASSERT_EQ(instance.link_names_size(), instance.link_poses_size());
  ASSERT_GT(instance.link_names_size(), 0);
  for (int link = 0; link < instance.link_names_size(); ++link) {
    const std::string& name = instance.link_names(link);
    ASSERT_TRUE(interface.getModel().existFrame(name)) << name;
    const pinocchio::SE3& expected = interface.getData().oMf[interface.getModel().getFrameId(name)];
    EXPECT_LT((toVector(instance.link_poses(link).position()) - expected.translation()).norm(), kTolerance)
        << instance.name() << " " << name;
    EXPECT_TRUE(toRotation(instance.link_poses(link).orientation()).isApprox(expected.rotation(), 1e-9)) << instance.name() << " " << name;
  }
}

std::string formulationName(const ::testing::TestParamInfo<test::Formulation>& info) {
  switch (info.param) {
    case test::Formulation::kCentroidal:
      return "G1Centroidal";
    case test::Formulation::kCentroidalBasisVectors:
      return "AtlasBasisVectors";
    case test::Formulation::kWholeBody:
      return "G1WholeBody";
  }
  return "Unknown";
}

test::RobotFiles filesOf(test::Formulation formulation) {
  switch (formulation) {
    case test::Formulation::kCentroidal:
      return test::g1CentroidalFiles();
    case test::Formulation::kCentroidalBasisVectors:
      return test::atlasFiles();
    case test::Formulation::kWholeBody:
      return test::g1WholeBodyFiles();
  }
  return test::g1CentroidalFiles();
}

class SceneBuilderTest : public ::testing::TestWithParam<test::Formulation> {
 protected:
  void SetUp() override {
    robot_ = test::TestRobot::load(filesOf(GetParam()), GetParam());
    const absl::StatusOr<VisualizationConfig> config = loadVisualizationConfig(robot_->taskFile(), robot_->modelSettings());
    ASSERT_TRUE(config.ok()) << config.status();
    config_ = *config;
    absl::StatusOr<std::unique_ptr<SceneBuilder>> builder = SceneBuilder::Create(robot_->model(), config_);
    ASSERT_TRUE(builder.ok()) << builder.status();
    builder_ = std::move(*builder);

    observation_ = robot_->observation(/*time=*/0.0, ModeNumber::STANCE);
    makePolicy(kNormalForce);
    const humanoid_mpc_msgs::RobotStateSample proto =
        robot_->robotState(/*time=*/0.02, vector3_t(0.1, -0.2, 0.75), vector3_t(0.05, -0.1, 0.4));
    ASSERT_TRUE(msgs::FromProto(proto, &sample_).ok());
  }

  void makePolicy(scalar_t normalForce) {
    CommandData command;
    PrimalSolution solution;
    robot_->makePolicy(/*startTime=*/0.0, kNodes, normalForce, vector2_t(0.02, -0.01), &command, &solution);
    // The reference at the plan's end is the nominal state, not the plan's last state.
    command.mpcTargetTrajectories_.stateTrajectory.back() = robot_->nominalState();
    policy_.assign(command, solution);
  }

  humanoid_mpc_msgs::VisualizationScene build(const SceneInputs& inputs) {
    humanoid_mpc_msgs::VisualizationScene scene;
    const absl::Status status = builder_->build(inputs, &scene);
    EXPECT_TRUE(status.ok()) << status;
    return scene;
  }

  SceneInputs allInputs() const {
    SceneInputs inputs;
    inputs.observation = &observation_;
    inputs.policy = &policy_;
    inputs.policyVersion = 1;
    inputs.robotState = &sample_;
    return inputs;
  }

  /** The MPC model's input of the policy at the observation, as the scene evaluates it. */
  vector_t inputAtObservation() const {
    vector_t state;
    vector_t input;
    samplePlan(policy_, observation_.time, &state, &input);
    return input;
  }

  std::unique_ptr<test::TestRobot> robot_;
  VisualizationConfig config_;
  std::unique_ptr<SceneBuilder> builder_;
  SystemObservation observation_;
  PolicySnapshot policy_;
  msgs::RobotStateSample sample_;
};

TEST_P(SceneBuilderTest, EveryMarkerPathIsInTheSceneOnceAndTheThreeRobotsAre) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  EXPECT_EQ(scene.time(), sample_.time);
  absl::flat_hash_map<std::string, int> paths;
  for (const humanoid_mpc_msgs::Arrows& arrows : scene.arrows()) {
    ++paths[arrows.path()];
  }
  for (const humanoid_mpc_msgs::Spheres& spheres : scene.spheres()) {
    ++paths[spheres.path()];
  }
  for (const humanoid_mpc_msgs::LineStrips& strips : scene.line_strips()) {
    ++paths[strips.path()];
  }
  const std::vector<absl::string_view> expected = {scene::kContactForces,    scene::kCenterOfPressure, scene::kCornerForces,
                                                   scene::kPlanEndEffectors, scene::kPlanBase,         scene::kPlanCom,
                                                   scene::kPlanFootholds,    scene::kCollisionSpheres};
  EXPECT_EQ(paths.size(), expected.size());
  for (const absl::string_view path : expected) {
    EXPECT_EQ(paths[std::string(path)], 1) << path;
  }
  ASSERT_EQ(scene.robots_size(), 3);
  EXPECT_NE(findInstance(scene, scene::kMeasuredInstance), nullptr);
  EXPECT_NE(findInstance(scene, scene::kTerminalStateInstance), nullptr);
  EXPECT_NE(findInstance(scene, scene::kTerminalTargetInstance), nullptr);
}

TEST_P(SceneBuilderTest, TheMeasuredRobotHasEveryJointOfTheUrdf) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  const humanoid_mpc_msgs::RobotModelInstance* measured = findInstance(scene, scene::kMeasuredInstance);
  ASSERT_NE(measured, nullptr);

  // An independent model of the whole URDF with the MPC's root (Euler angles instead of a quaternion), every joint set
  // by name from the sample.
  PinocchioInterface reference = createDefaultPinocchioInterface(robot_->urdfFile());
  const Model& model = reference.getModel();
  vector_t q = vector_t::Zero(model.nq);
  q.head<3>() = vector3_t(sample_.base_position_world.x, sample_.base_position_world.y, sample_.base_position_world.z);
  q.segment<3>(3) = vector3_t(0.4, -0.1, 0.05);  // yaw, pitch, roll
  absl::flat_hash_map<std::string, scalar_t> positions;
  for (size_t joint = 0; joint < sample_.joint_names.size(); ++joint) {
    positions[sample_.joint_names[joint]] = sample_.joint_positions[static_cast<Eigen::Index>(joint)];
  }
  size_t jointsSet = 0;
  for (pinocchio::JointIndex joint = 2; joint < static_cast<pinocchio::JointIndex>(model.njoints); ++joint) {
    const absl::flat_hash_map<std::string, scalar_t>::const_iterator found = positions.find(model.names[joint]);
    if (found != positions.end()) {
      ASSERT_EQ(model.joints[joint].nq(), 1);
      q[model.joints[joint].idx_q()] = found->second;
      ++jointsSet;
    }
  }
  EXPECT_EQ(jointsSet, robot_->modelSettings().fullJointNames.size());
  forwardKinematics(reference, q);
  expectLinkPoses(*measured, reference);

  size_t links = 0;
  for (const pinocchio::Frame& frame : model.frames) {
    links += frame.type == pinocchio::BODY && frame.name != "universe" ? 1 : 0;
  }
  EXPECT_EQ(static_cast<size_t>(measured->link_names_size()), links);
}

TEST_P(SceneBuilderTest, TheTerminalRobotsAreThePlansLastNodeAndTheReferenceThere) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(policy_.state.back()));
  const humanoid_mpc_msgs::RobotModelInstance* terminalState = findInstance(scene, scene::kTerminalStateInstance);
  ASSERT_NE(terminalState, nullptr);
  expectLinkPoses(*terminalState, reference);

  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(robot_->nominalState()));
  const humanoid_mpc_msgs::RobotModelInstance* terminalTarget = findInstance(scene, scene::kTerminalTargetInstance);
  ASSERT_NE(terminalTarget, nullptr);
  expectLinkPoses(*terminalTarget, reference);
  EXPECT_EQ(terminalTarget->link_names_size(), terminalState->link_names_size());
}

TEST_P(SceneBuilderTest, ContactForcesEndAtTheCentersOfPressure) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  const humanoid_mpc_msgs::Arrows& forces = arrowsAt(scene, scene::kContactForces);
  ASSERT_EQ(forces.vectors_size(), static_cast<int>(N_CONTACTS));
  ASSERT_EQ(forces.origins_size(), static_cast<int>(N_CONTACTS));

  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(observation_.state));
  const vector_t input = inputAtObservation();
  vector3_t weighted = vector3_t::Zero();
  scalar_t normalForceSum = 0.0;
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    const pinocchio::SE3& frame = reference.getData().oMf[reference.getModel().getFrameId(robot_->modelSettings().contactNames[contact])];
    const vector6_t wrench = robot_->robotModel().getContactWrenchInWorldFrame(observation_.state, input, contact);
    const vector3_t localForce = frame.rotation().transpose() * wrench.head<3>();
    const vector3_t localTorque = frame.rotation().transpose() * wrench.tail<3>();
    const vector3_t cop = frame.act(vector3_t(-localTorque.y() / localForce.z(), localTorque.x() / localForce.z(), 0.0));
    const vector3_t vector = toVector(forces.vectors(static_cast<int>(contact)));
    const vector3_t tip = toVector(forces.origins(static_cast<int>(contact))) + vector;
    EXPECT_TRUE((tip - cop).norm() < 1e-9) << "contact " << contact << ": " << tip.transpose() << " vs " << cop.transpose();
    EXPECT_TRUE((vector - wrench.head<3>() / scene::kForceScale).norm() < 1e-9) << contact;
    weighted += wrench.z() * cop;
    normalForceSum += wrench.z();
    if (GetParam() != test::Formulation::kCentroidalBasisVectors) {
      // The world wrench the policy was written with: 300 N up, 15 N forward.
      EXPECT_NEAR(vector.z(), kNormalForce / scene::kForceScale, 1e-9);
      EXPECT_NEAR(vector.x(), 0.05 * kNormalForce / scene::kForceScale, 1e-9);
    }
  }
  const humanoid_mpc_msgs::Spheres& centerOfPressure = spheresAt(scene, scene::kCenterOfPressure);
  ASSERT_EQ(centerOfPressure.centers_size(), 1);
  EXPECT_TRUE((toVector(centerOfPressure.centers(0)) - weighted / normalForceSum).norm() < 1e-9);
}

TEST_P(SceneBuilderTest, TheCornerForcesAreEquivalentToTheContactWrench) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  const humanoid_mpc_msgs::Arrows& corners = arrowsAt(scene, scene::kCornerForces);
  ASSERT_EQ(corners.vectors_size(), static_cast<int>(4 * N_CONTACTS));

  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(observation_.state));
  const vector_t input = inputAtObservation();
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    const Model& model = reference.getModel();
    const vector3_t origin = reference.getData().oMf[model.getFrameId(robot_->modelSettings().contactNames[contact])].translation();
    const ContactRectangle rectangle =
        ContactRectangle::loadContactRectangle(robot_->taskFile(), robot_->modelSettings(), static_cast<int>(contact), /*verbose=*/false);
    vector3_t force = vector3_t::Zero();
    vector3_t torque = vector3_t::Zero();
    for (size_t corner = 0; corner < 4; ++corner) {
      const int arrow = static_cast<int>(4 * contact + corner);
      const vector3_t cornerForce = toVector(corners.vectors(arrow)) * scene::kForceScale;
      const vector3_t tip = toVector(corners.origins(arrow)) + toVector(corners.vectors(arrow));
      const vector3_t cornerPosition =
          reference.getData().oMf[model.getFrameId(rectangle.getPolygonPointFrameName(static_cast<int>(corner)))].translation();
      EXPECT_TRUE((tip - cornerPosition).norm() < 1e-9) << contact << " " << corner;
      force += cornerForce;
      torque += (cornerPosition - origin).cross(cornerForce);
    }
    const vector6_t wrench = robot_->robotModel().getContactWrenchInWorldFrame(observation_.state, input, contact);
    EXPECT_TRUE((force - wrench.head<3>()).norm() < 1e-6) << force.transpose() << " vs " << wrench.head<3>().transpose();
    EXPECT_TRUE((torque - wrench.tail<3>()).norm() < 1e-6) << torque.transpose() << " vs " << wrench.tail<3>().transpose();
  }
}

TEST_P(SceneBuilderTest, ASwingFootHasNoContactMarkers) {
  observation_.mode = ModeNumber::RF;
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  EXPECT_EQ(arrowsAt(scene, scene::kContactForces).vectors_size(), 1);
  EXPECT_EQ(arrowsAt(scene, scene::kCornerForces).vectors_size(), 4);
}

TEST_P(SceneBuilderTest, ACenterOfPressureWithoutForceIsTheContactFrameNotNan) {
  makePolicy(/*normalForce=*/0.0);
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(observation_.state));
  const humanoid_mpc_msgs::Arrows& forces = arrowsAt(scene, scene::kContactForces);
  ASSERT_EQ(forces.vectors_size(), static_cast<int>(N_CONTACTS));
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    const vector3_t tip = toVector(forces.origins(static_cast<int>(contact))) + toVector(forces.vectors(static_cast<int>(contact)));
    ASSERT_TRUE(tip.allFinite());
    const vector3_t frame =
        reference.getData().oMf[reference.getModel().getFrameId(robot_->modelSettings().contactNames[contact])].translation();
    EXPECT_TRUE((tip - frame).norm() < 1e-9);
  }
  EXPECT_EQ(spheresAt(scene, scene::kCenterOfPressure).centers_size(), 0);
  for (const humanoid_mpc_msgs::Arrows& arrows : scene.arrows()) {
    for (const humanoid_mpc_msgs::Vector3& vector : arrows.vectors()) {
      EXPECT_TRUE(toVector(vector).allFinite()) << arrows.path();
    }
  }
}

TEST_P(SceneBuilderTest, TheCollisionSpheresAreAtTheirFrames) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(observation_.state));
  const FootCollisionConstraint::Config collision = FootCollisionConstraint::loadFootCollisionConstraintConfig(robot_->taskFile());
  std::vector<std::pair<std::string, scalar_t>> expected;
  for (const std::string* frame :
       {&collision.leftAnkleFrame, &collision.rightAnkleFrame, &collision.leftFootCenterFrame, &collision.rightFootCenterFrame,
        &collision.leftFootFrame1, &collision.rightFootFrame1, &collision.leftFootFrame2, &collision.rightFootFrame2}) {
    if (!frame->empty() && reference.getModel().existFrame(*frame)) {
      expected.emplace_back(*frame, collision.footCollisionSphereRadius);
    }
  }
  for (const std::string* frame : {&collision.leftKneeFrame, &collision.rightKneeFrame}) {
    if (!frame->empty() && reference.getModel().existFrame(*frame)) {
      expected.emplace_back(*frame, collision.kneeCollisionSphereRadius);
    }
  }
  const humanoid_mpc_msgs::Spheres& spheres = spheresAt(scene, scene::kCollisionSpheres);
  ASSERT_EQ(spheres.centers_size(), static_cast<int>(expected.size()));
  ASSERT_EQ(spheres.radii_size(), static_cast<int>(expected.size()));
  EXPECT_GE(expected.size(), 6u);
  for (size_t sphere = 0; sphere < expected.size(); ++sphere) {
    const vector3_t center = reference.getData().oMf[reference.getModel().getFrameId(expected[sphere].first)].translation();
    EXPECT_TRUE((toVector(spheres.centers(static_cast<int>(sphere))) - center).norm() < 1e-9) << expected[sphere].first;
    EXPECT_FLOAT_EQ(spheres.radii(static_cast<int>(sphere)), static_cast<float>(expected[sphere].second));
  }
}

TEST_P(SceneBuilderTest, WithoutSphereRadiiThereAreNoCollisionSpheres) {
  std::ifstream stream(robot_->taskFile());
  std::stringstream text;
  text << stream.rdbuf();
  const std::string withoutRadii = absl::StrReplaceAll(
      text.str(), {{"footCollisionSphereRadius", "renamedFootRadius"}, {"kneeCollisionSphereRadius", "renamedKneeRadius"}});
  ASSERT_NE(withoutRadii, text.str());
  const char* directory = std::getenv("TEST_TMPDIR");
  const std::string taskFile = absl::StrCat(directory != nullptr ? directory : "/tmp", "/task_without_radii.yaml");
  std::ofstream(taskFile) << withoutRadii;
  VisualizationModel model = robot_->model();
  model.taskFile = taskFile;
  absl::StatusOr<std::unique_ptr<SceneBuilder>> builder = SceneBuilder::Create(model, config_);
  ASSERT_TRUE(builder.ok()) << builder.status();
  humanoid_mpc_msgs::VisualizationScene scene;
  ASSERT_TRUE((*builder)->build(allInputs(), &scene).ok());
  EXPECT_EQ(spheresAt(scene, scene::kCollisionSpheres).centers_size(), 0);
}

TEST_P(SceneBuilderTest, ThePlanIsTheFramesBaseAndGroundProjectedCom) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  const Model& model = reference.getModel();

  // The ground is the stance feet's mean height at the observation.
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(observation_.state));
  scalar_t groundHeight = 0.0;
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    groundHeight += reference.getData().oMf[model.getFrameId(robot_->modelSettings().contactNames[contact])].translation().z() / N_CONTACTS;
  }

  const humanoid_mpc_msgs::LineStrips& endEffectors = lineStripsAt(scene, scene::kPlanEndEffectors);
  const humanoid_mpc_msgs::LineStrips& base = lineStripsAt(scene, scene::kPlanBase);
  const humanoid_mpc_msgs::LineStrips& com = lineStripsAt(scene, scene::kPlanCom);
  ASSERT_EQ(endEffectors.strips_size(), static_cast<int>(config_.planFrames.size()));
  ASSERT_EQ(base.strips_size(), 1);
  ASSERT_EQ(com.strips_size(), 1);
  ASSERT_EQ(base.strips(0).points_size(), static_cast<int>(kNodes));
  ASSERT_EQ(com.strips(0).points_size(), static_cast<int>(kNodes));
  for (size_t node = 0; node < kNodes; ++node) {
    const vector_t q = robot_->robotModel().getGeneralizedCoordinates(policy_.state[node]);
    forwardKinematics(reference, q);
    const int point = static_cast<int>(node);
    for (size_t frame = 0; frame < config_.planFrames.size(); ++frame) {
      ASSERT_EQ(endEffectors.strips(static_cast<int>(frame)).points_size(), static_cast<int>(kNodes));
      const vector3_t expected = reference.getData().oMf[model.getFrameId(config_.planFrames[frame])].translation();
      EXPECT_TRUE((toVector(endEffectors.strips(static_cast<int>(frame)).points(point)) - expected).norm() < 1e-9)
          << config_.planFrames[frame] << " node " << node;
    }
    // The base is the root link: the MPC model's first link.
    const vector3_t root = reference.getData().oMf[model.getFrameId(builder_->mpcLinkNames().front())].translation();
    EXPECT_TRUE((toVector(base.strips(0).points(point)) - root).norm() < 1e-9) << node;
    vector3_t centerOfMass = pinocchio::centerOfMass(model, reference.getData(), q, /*computeSubtreeComs=*/false);
    centerOfMass.z() = groundHeight;
    EXPECT_TRUE((toVector(com.strips(0).points(point)) - centerOfMass).norm() < 1e-9) << node;
  }
}

TEST_P(SceneBuilderTest, AFootholdIsWhereAFootLandsInTheHorizon) {
  const humanoid_mpc_msgs::VisualizationScene scene = build(allInputs());
  // The left foot swings from 0.3 s to 0.6 s and lands at 0.6 s; nothing else lands.
  const humanoid_mpc_msgs::Spheres& footholds = spheresAt(scene, scene::kPlanFootholds);
  ASSERT_EQ(footholds.centers_size(), 1);
  ASSERT_EQ(footholds.colors_size(), 1);
  vector_t state;
  vector_t input;
  samplePlan(policy_, /*time=*/0.6, &state, &input);
  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(state));
  const vector3_t foot =
      reference.getData().oMf[reference.getModel().getFrameId(robot_->modelSettings().contactNames[CONTACT_LEFT_INDEX])].translation();
  EXPECT_TRUE((toVector(footholds.centers(0)) - foot).norm() < 1e-9);
  EXPECT_FLOAT_EQ(footholds.colors(0).r(), scene::kPurple.r);
  EXPECT_FLOAT_EQ(footholds.colors(0).a(), 1.0f);
}

TEST_P(SceneBuilderTest, ASampleAloneDrawsTheMeasuredRobotAndEmptyMarkers) {
  SceneInputs inputs;
  inputs.robotState = &sample_;
  const humanoid_mpc_msgs::VisualizationScene scene = build(inputs);
  EXPECT_EQ(scene.time(), sample_.time);
  ASSERT_EQ(scene.robots_size(), 1);
  EXPECT_EQ(scene.robots(0).name(), scene::kMeasuredInstance);
  EXPECT_EQ(scene.arrows_size() + scene.spheres_size() + scene.line_strips_size(), 8);
  for (const humanoid_mpc_msgs::Arrows& arrows : scene.arrows()) {
    EXPECT_EQ(arrows.vectors_size(), 0) << arrows.path();
  }
  for (const humanoid_mpc_msgs::Spheres& spheres : scene.spheres()) {
    EXPECT_EQ(spheres.centers_size(), 0) << spheres.path();
  }
  for (const humanoid_mpc_msgs::LineStrips& strips : scene.line_strips()) {
    EXPECT_EQ(strips.strips_size(), 0) << strips.path();
  }
}

TEST_P(SceneBuilderTest, WithoutASampleTheMeasuredRobotIsTheObservation) {
  SceneInputs inputs = allInputs();
  inputs.robotState = nullptr;
  const humanoid_mpc_msgs::VisualizationScene scene = build(inputs);
  EXPECT_EQ(scene.time(), observation_.time);
  const humanoid_mpc_msgs::RobotModelInstance* measured = findInstance(scene, scene::kMeasuredInstance);
  ASSERT_NE(measured, nullptr);
  PinocchioInterface reference = robot_->makeReferencePinocchioInterface();
  forwardKinematics(reference, robot_->robotModel().getGeneralizedCoordinates(observation_.state));
  expectLinkPoses(*measured, reference);
}

TEST_P(SceneBuilderTest, WithNothingToDrawThereIsNoScene) {
  humanoid_mpc_msgs::VisualizationScene scene;
  SceneInputs inputs;
  inputs.policy = &policy_;
  EXPECT_EQ(builder_->build(inputs, &scene).code(), absl::StatusCode::kFailedPrecondition);
  // An observation of other dimensions than the model's does not count.
  SystemObservation wrong = observation_;
  wrong.state.conservativeResize(wrong.state.size() - 1);
  inputs.observation = &wrong;
  EXPECT_EQ(builder_->build(inputs, &scene).code(), absl::StatusCode::kFailedPrecondition);
}

TEST_P(SceneBuilderTest, ThePlanIsComputedAgainOnlyForANewPolicy) {
  SceneInputs inputs = allInputs();
  const humanoid_mpc_msgs::VisualizationScene first = build(inputs);
  for (vector_t& state : policy_.state) {
    vector6_t basePose = robot_->robotModel().getBasePose(state);
    basePose[1] += 0.5;
    robot_->robotModel().setBasePose(state, basePose);
  }
  const humanoid_mpc_msgs::VisualizationScene same = build(inputs);
  EXPECT_EQ(lineStripsAt(same, scene::kPlanBase).SerializeAsString(), lineStripsAt(first, scene::kPlanBase).SerializeAsString());
  inputs.policyVersion = 2;
  const humanoid_mpc_msgs::VisualizationScene changed = build(inputs);
  EXPECT_NEAR(
      lineStripsAt(changed, scene::kPlanBase).strips(0).points(0).y() - lineStripsAt(first, scene::kPlanBase).strips(0).points(0).y(), 0.5,
      1e-9);
  // A policy of other dimensions is no policy: no plan and no terminal robots.
  policy_.input.front().resize(1);
  inputs.policyVersion = 3;
  const humanoid_mpc_msgs::VisualizationScene without = build(inputs);
  EXPECT_EQ(lineStripsAt(without, scene::kPlanBase).strips_size(), 0);
  EXPECT_EQ(findInstance(without, scene::kTerminalStateInstance), nullptr);
  EXPECT_EQ(arrowsAt(without, scene::kContactForces).vectors_size(), 0);
}

INSTANTIATE_TEST_SUITE_P(Formulations,
                         SceneBuilderTest,
                         ::testing::Values(test::Formulation::kCentroidal,
                                           test::Formulation::kCentroidalBasisVectors,
                                           test::Formulation::kWholeBody),
                         formulationName);

}  // namespace
}  // namespace ocs2::humanoid::visualization
