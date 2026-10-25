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
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"

/*
 * The whole-body MPC's (state, input) overloads of computeJointTorques and
 * computeBaseHeldJointTorques: the joint torques of the generalized
 * coordinates, velocities, joint accelerations and world-frame sole wrenches
 * that the state and the input carry. The (q, v, qdd_j, wrenches) functions
 * themselves are pinned against RNEA in humanoid_common_mpc's
 * testJointTorqueInverseDynamics; this checks that the overloads hand them the
 * right pieces, and that the base-held one - the gantry feedforward of
 * WBMpcMrtJointController - is not the floating-base one.
 */

namespace ocs2::humanoid {
namespace {

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR"))
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** A deterministic, well spread sequence in [-1, 1]. */
scalar_t spread(Eigen::Index i, scalar_t phase) {
  return std::sin(1.7 * static_cast<scalar_t>(i) + phase);
}

scalar_t maxAbs(const vector_t& v) {
  return v.cwiseAbs().maxCoeff();
}

/** RNEA at the base acceleration that balances its own base rows, the oracle of
 * the floating-base dynamics. */
struct RneaReference {
  vector6_t baseAcceleration;
  vector_t jointTorques;
};

/**
 * Returns RNEA's base acceleration and joint torques for world-frame wrenches
 * applied at the origins of `soleFrames`, passed to RNEA as external forces on
 * the frames' parent joints - not through the frame Jacobians the code under
 * test uses. RNEA is affine in the acceleration, so its unactuated base rows
 * vanish at a_b = -M_bb^-1 tau_b(a_b = 0).
 */
RneaReference floatingBaseRnea(
    const pinocchio::Model& model,
    const vector_t& q,
    const vector_t& v,
    const vector_t& qddJoints,
    const std::array<vector6_t, 2>& wrenches,
    const feet_array_t<pinocchio::FrameIndex>& soleFrames) {
  pinocchio::Data data(model);
  pinocchio::framesForwardKinematics(model, data, q);
  std::vector<pinocchio::Force> fext(static_cast<size_t>(model.njoints),
                                     pinocchio::Force::Zero());
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const pinocchio::JointIndex joint =
        model.frames[soleFrames[foot]].parentJoint;
    const matrix3_t rotation = data.oMi[joint].rotation();
    const vector3_t lever = data.oMf[soleFrames[foot]].translation() -
                            data.oMi[joint].translation();
    const vector3_t force = wrenches[foot].head<3>();
    const vector3_t moment = wrenches[foot].tail<3>() + lever.cross(force);
    fext[joint] += pinocchio::Force(rotation.transpose() * force,
                                    rotation.transpose() * moment);
  }
  vector_t a = vector_t::Zero(model.nv);
  a.tail(qddJoints.size()) = qddJoints;
  const vector6_t baseForceAtZeroBaseAcceleration =
      pinocchio::rnea(model, data, q, v, a, fext).head<6>();
  pinocchio::crba(model, data, q);
  const matrix6_t baseMassMatrix =
      data.M.topLeftCorner<6, 6>().selfadjointView<Eigen::Upper>();
  a.head<6>() = -baseMassMatrix.ldlt().solve(baseForceAtZeroBaseAcceleration);
  const vector_t tau = pinocchio::rnea(model, data, q, v, a, fext);
  return {.baseAcceleration = a.head<6>(),
          .jointTorques = tau.tail(qddJoints.size())};
}

class JointTorqueStateInputOverloadsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // LINT.IfChange(robot_files)
    const std::string taskFile = runfilePath(
        "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
    const std::string urdfFile = runfilePath(
        "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
    // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:joint_torque_overload_test_data)
    ASSERT_FALSE(taskFile.empty() || urdfFile.empty())
        << "the G1 whole-body files are not in the runfiles";
    taskFile_ = taskFile;
    urdfFile_ = urdfFile;
    modelSettings_ =
        std::make_unique<ModelSettings>(checkedValue(ModelSettings::Create(
            taskFile, urdfFile, "wb_mpc_", /*verbose=*/false)));
    absl::StatusOr<PinocchioInterface> pinocchioInterface =
        loadCustomPinocchioInterface(taskFile, urdfFile, *modelSettings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ =
        std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);

    // A moving, tilted robot with every joint accelerating and a different
    // wrench under each sole. The configuration is a point of Pinocchio's
    // configuration space, seeded: the joints within their limits and the base
    // orientation a uniformly random unit quaternion
    // (pinocchio::randomConfiguration, which draws from std::rand), the base
    // within a box.
    const pinocchio::Model& model = pinocchioInterface_->getModel();
    vector_t lower = model.lowerPositionLimit;
    vector_t upper = model.upperPositionLimit;
    lower.head<3>() = vector3_t(-1.0, -1.0, 0.6);
    upper.head<3>() = vector3_t(1.0, 1.0, 1.0);
    std::srand(20'261'009);
    q_ = pinocchio::randomConfiguration(model, lower, upper);
    ASSERT_NEAR(q_.segment<4>(3).norm(), 1.0, 1.0e-12)
        << "the base orientation of the sample is not a unit quaternion";
    v_ = vector_t(model.nv);
    qddJoints_ = vector_t(model.nv - 6);
    for (Eigen::Index i = 0; i < v_.size(); ++i)
      v_(i) = 0.8 * spread(i, /*phase=*/1.9);
    for (Eigen::Index i = 0; i < qddJoints_.size(); ++i)
      qddJoints_(i) = 3.0 * spread(i, /*phase=*/0.4);
    wrenches_[0] << 30.0, -12.0, 320.0, 4.0, -6.0, 1.5;
    wrenches_[1] << -18.0, 25.0, 210.0, -3.0, 5.0, -0.8;
    for (size_t i = 0; i < kNumContacts; ++i)
      soleFrames_[i] = model.getFrameId(modelSettings_->contactNames[i]);
  }

  /** The whole-body MPC state [q, v] and input [W_left, W_right, qdd_j] that
   * carry the given pieces. */
  std::pair<vector_t, vector_t> stateAndInput(
      const vector_t& q,
      const vector_t& v,
      const vector_t& qddJoints,
      const std::array<vector6_t, 2>& wrenches) const {
    vector_t state = model_->getNeutralState();
    model_->setGeneralizedCoordinates(state, q);
    state.tail(v.size()) = v;
    vector_t input = vector_t::Zero(model_->getInputDim());
    model_->setContactWrench(input, wrenches[0], /*contactIndex=*/0);
    model_->setContactWrench(input, wrenches[1], /*contactIndex=*/1);
    input.tail(qddJoints.size()) = qddJoints;
    return {state, input};
  }

  std::string taskFile_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<WBAccelMpcRobotModel<scalar_t>> model_;
  vector_t q_;
  vector_t v_;
  vector_t qddJoints_;
  std::array<vector6_t, 2> wrenches_;
  feet_array_t<pinocchio::FrameIndex> soleFrames_{};
};

TEST_F(JointTorqueStateInputOverloadsTest,
       theOverloadsReadTheStateAndTheInputTheirPiecesCameFrom) {
  const std::pair<vector_t, vector_t> stateInput =
      stateAndInput(q_, v_, qddJoints_, wrenches_);
  ASSERT_TRUE(model_->getGeneralizedVelocities(stateInput.first,
                                               stateInput.second) == v_)
      << "the state layout of the test is stale";
  ASSERT_TRUE(model_->getJointAccelerations(stateInput.second) == qddJoints_)
      << "the input layout of the test is stale";

  PinocchioInterface overloadInterface = *pinocchioInterface_;
  PinocchioInterface directInterface = *pinocchioInterface_;
  const vector_t baseHeld = computeBaseHeldJointTorques<scalar_t>(
      stateInput.first, stateInput.second, overloadInterface, *model_);
  const vector_t baseHeldDirect = computeBaseHeldJointTorques<scalar_t>(
      q_, v_, qddJoints_, wrenches_, soleFrames_, directInterface);
  const scalar_t scale = std::max(1.0, maxAbs(baseHeldDirect));
  EXPECT_LT(maxAbs(baseHeld - baseHeldDirect), 1.0e-12 * scale);

  const vector_t floating = computeJointTorques<scalar_t>(
      stateInput.first, stateInput.second, overloadInterface, *model_);
  const vector_t floatingDirect = computeJointTorques<scalar_t>(
      q_, v_, qddJoints_, wrenches_, soleFrames_, directInterface);
  EXPECT_LT(maxAbs(floating - floatingDirect),
            1.0e-12 * std::max(1.0, maxAbs(floatingDirect)));

  // Positive controls. The base-held overload is not the floating-base one:
  // this sample accelerates the base.
  EXPECT_GT(maxAbs(baseHeld - floating), 1.0);
  // And the soles are told apart: the same wrenches under the other feet are
  // other torques.
  const std::array<vector6_t, 2> exchanged = {wrenches_[1], wrenches_[0]};
  EXPECT_GT(
      maxAbs(computeBaseHeldJointTorques<scalar_t>(
                 q_, v_, qddJoints_, exchanged, soleFrames_, directInterface) -
             baseHeld),
      1.0);
}

TEST_F(JointTorqueStateInputOverloadsTest,
       theBaseHeldTorqueOfARobotHangingAtRestIsItsGravityTorque) {
  // The gantry: at rest, no sole loaded and no joint accelerating, the torque
  // that holds the posture is g_j(q) - an oracle computed by Pinocchio alone.
  const vector_t atRest = vector_t::Zero(v_.size());
  const std::pair<vector_t, vector_t> stateInput =
      stateAndInput(q_, atRest, vector_t::Zero(qddJoints_.size()),
                    {vector6_t::Zero(), vector6_t::Zero()});
  PinocchioInterface pinocchioInterface = *pinocchioInterface_;
  const vector_t baseHeld = computeBaseHeldJointTorques<scalar_t>(
      stateInput.first, stateInput.second, pinocchioInterface, *model_);

  pinocchio::Data data(pinocchioInterface_->getModel());
  const vector_t gravity = pinocchio::computeGeneralizedGravity(
                               pinocchioInterface_->getModel(), data, q_)
                               .tail(qddJoints_.size());
  ASSERT_GT(maxAbs(gravity), 1.0)
      << "positive control: gravity loads the joints of this posture";
  EXPECT_LT(maxAbs(baseHeld - gravity), 1.0e-9 * maxAbs(gravity));
  // The floating base, by contrast, falls freely without a wrench, and gravity
  // then loads no joint.
  EXPECT_GT(
      maxAbs(computeJointTorques<scalar_t>(stateInput.first, stateInput.second,
                                           pinocchioInterface, *model_) -
             gravity),
      1.0);
}

TEST_F(
    JointTorqueStateInputOverloadsTest,
    theDynamicsApplyTheWrenchesAtTheConfiguredContactFramesWhateverTheirNames) {
  // The base acceleration and the inverse dynamics used to look the soles up as
  // "foot_l_contact" and "foot_r_contact". A robot whose
  // model_settings.contact_names name other frames - here the G1's soles
  // renamed - must have its wrenches applied at those frames: the RNEA oracle
  // of the same frames.
  absl::StatusOr<ModelSettings> created =
      ModelSettings::Create(taskFile_, urdfFile_, "wb_mpc_", /*verbose=*/false);
  ASSERT_TRUE(created.ok()) << created.status();
  ModelSettings renamedSettings = *std::move(created);
  renamedSettings.contactNames = {"left_sole_renamed", "right_sole_renamed"};
  pinocchio::Model renamedModel = pinocchioInterface_->getModel();
  for (size_t i = 0; i < kNumContacts; ++i) {
    renamedModel
        .frames[renamedModel.getFrameId(modelSettings_->contactNames[i])]
        .name = renamedSettings.contactNames[i];
  }
  ASSERT_FALSE(renamedModel.existFrame(modelSettings_->contactNames[0]) ||
               renamedModel.existFrame(modelSettings_->contactNames[1]));
  ASSERT_TRUE(
      checkContactFramesExist(renamedModel, renamedSettings.contactNames).ok());
  PinocchioInterface renamedInterface(renamedModel);
  WBAccelMpcRobotModel<scalar_t> renamedMpcModel(renamedSettings);

  const std::pair<vector_t, vector_t> stateInput =
      stateAndInput(q_, v_, qddJoints_, wrenches_);
  const RneaReference reference = floatingBaseRnea(
      renamedModel, q_, v_, qddJoints_, wrenches_, soleFrames_);
  ASSERT_GT(reference.baseAcceleration.norm(), 0.1)
      << "the sample accelerates the base";

  const vector_t accelerations = computeGeneralizedAccelerations<scalar_t>(
      stateInput.first, stateInput.second, renamedInterface, renamedMpcModel);
  EXPECT_LT(maxAbs(accelerations.head<6>() - reference.baseAcceleration),
            1.0e-9 * std::max(1.0, maxAbs(reference.baseAcceleration)));
  EXPECT_LT(maxAbs(accelerations.tail(qddJoints_.size()) - qddJoints_),
            1.0e-15);

  const vector_t torques = computeJointTorques<scalar_t>(
      stateInput.first, stateInput.second, renamedInterface, renamedMpcModel);
  EXPECT_LT(maxAbs(torques - reference.jointTorques),
            1.0e-9 * std::max(1.0, maxAbs(reference.jointTorques)));

  // Positive control: the oracle tells the frames apart - the same wrenches on
  // the exchanged soles accelerate the base otherwise.
  const RneaReference exchanged =
      floatingBaseRnea(renamedModel, q_, v_, qddJoints_, wrenches_,
                       {soleFrames_[1], soleFrames_[0]});
  EXPECT_GT(maxAbs(exchanged.baseAcceleration - reference.baseAcceleration),
            1.0e-2);
}

}  // namespace
}  // namespace ocs2::humanoid
