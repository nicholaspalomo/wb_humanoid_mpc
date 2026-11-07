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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/spatial/fwd.hpp"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"

/*
 * computeJointTorques() and computeBaseHeldJointTorques() against Pinocchio's
 * recursive Newton-Euler algorithm.
 *
 * The reference is built without anything computeJointTorques uses: the contact
 * wrenches reach RNEA as external forces on the foot joints rather than through
 * the frame Jacobians, and the base acceleration the wrenches produce is found
 * from RNEA's own base rows (RNEA is affine in the acceleration, so the
 * unactuated base rows vanish at one base acceleration). computeJointTorques
 * used to multiply the joint rows of crba()'s mass matrix, of which Pinocchio
 * fills only the upper triangle: it left out the joint-base coupling M_jb a_b
 * and, with a joint acceleration, the lower triangle of M_jj. The legacy
 * formula is kept here as the positive control that the comparison sees that
 * difference.
 */

namespace ocs2::humanoid {
namespace {

using ad_fun_t = CppAD::ADFun<ad_base_t>;
using ForceVector = std::vector<pinocchio::Force>;

constexpr scalar_t kGravityMagnitude = 9.81;

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** A deterministic, well spread sequence in [-1, 1]. */
scalar_t spread(size_t i, scalar_t phase) {
  return std::sin(1.7 * static_cast<scalar_t>(i) + phase);
}

scalar_t maxAbs(const vector_t& v) {
  return v.cwiseAbs().maxCoeff();
}

/** A robot of the repository, loaded as its MPC loads it. */
struct Robot {
  mpc_config::TaskFile task;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;

  const pinocchio::Model& model() const {
    return pinocchioInterface->getModel();
  }
  Eigen::Index numJoints() const {
    return model().nv - static_cast<Eigen::Index>(kBaseVelocityDim);
  }
  /** The sole frames of the task file's model_settings.contact_names, in the
   * order of the contacts. */
  ContactArray<pinocchio::FrameIndex> soleFrames() const {
    ContactArray<pinocchio::FrameIndex> frames(settings->contactNames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
      frames[i] = model().getFrameId(settings->contactNames[i]);
    }
    return frames;
  }
};

std::unique_ptr<Robot> loadRobot(absl::string_view taskFile,
                                 absl::string_view urdfFile,
                                 absl::string_view mpcName) {
  std::unique_ptr<Robot> robot = std::make_unique<Robot>();
  const std::string task = runfilePath(taskFile);
  const std::string urdf = runfilePath(urdfFile);
  if (task.empty() || urdf.empty()) return nullptr;
  robot->task = checkedValue(loadTaskFile(task));
  robot->settings =
      std::make_unique<ModelSettings>(checkedValue(ModelSettings::Create(
          robot->task, urdf, std::string(mpcName), /*verbose=*/false)));
  robot->pinocchioInterface = std::make_unique<PinocchioInterface>(checkedValue(
      loadCustomPinocchioInterface(robot->task, urdf, *robot->settings)));
  return robot;
}

// LINT.IfChange(robot_files)
std::unique_ptr<Robot> loadG1WholeBody() {
  return loadRobot(
      "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
      "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf", "wb_mpc_");
}

std::unique_ptr<Robot> loadAtlasCentroidal() {
  return loadRobot(
      "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/"
      "task.textproto",
      "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
      "centroidal_mpc_");
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:joint_torque_test_data)

/** A state, joint accelerations and sole wrenches [W_left, W_right] (world
 * frame, at the sole frames). */
struct Sample {
  vector_t q;
  vector_t v;
  vector_t qddJoints;
  std::array<vector6_t, 2> wrenchFeetEwrtWorld;
};

/** The wrenches `wrenchFeetEwrtWorld` of a sample as the inverse dynamics take
 * them, one per contact. */
ContactArray<vector6_t> toWrenchContactsEwrtWorld(
    const std::array<vector6_t, 2>& wrenchFeetEwrtWorld) {
  ContactArray<vector6_t> wrenchContactsEwrtWorld(wrenchFeetEwrtWorld.size());
  for (size_t i = 0; i < wrenchFeetEwrtWorld.size(); ++i) {
    wrenchContactsEwrtWorld[i] = wrenchFeetEwrtWorld[i];
  }
  return wrenchContactsEwrtWorld;
}

/**
 * Random-looking but deterministic: every generalized coordinate, velocity and
 * joint acceleration non-zero. The root's quaternion q[3:7] = (x, y, z, w) of
 * Composite(Translation, Spherical) is normalized, as every configuration the
 * MPC hands Pinocchio is (MpcRobotModelBase::getGeneralizedCoordinates()).
 */
Sample makeSample(const pinocchio::Model& model, size_t seed) {
  const scalar_t phase = 0.37 * static_cast<scalar_t>(seed);
  Sample sample;
  sample.q = vector_t(model.nq);
  sample.v = vector_t(model.nv);
  sample.qddJoints =
      vector_t(model.nv - static_cast<Eigen::Index>(kBaseVelocityDim));
  for (Eigen::Index i = 0; i < model.nq; ++i) {
    sample.q(i) = 0.4 * spread(static_cast<size_t>(i), phase + 0.7);
  }
  sample.q(2) = 0.9;
  sample.q.segment<4>(kBaseTranslationDim).normalize();
  for (Eigen::Index i = 0; i < model.nv; ++i) {
    sample.v(i) = 0.8 * spread(static_cast<size_t>(i), phase + 1.9);
  }
  for (Eigen::Index i = 0; i < sample.qddJoints.size(); ++i) {
    sample.qddJoints(i) = 3.0 * spread(static_cast<size_t>(i), phase + 0.4);
  }
  for (size_t foot = 0; foot < 2; ++foot) {
    for (Eigen::Index i = 0; i < 6; ++i) {
      const scalar_t magnitude = i == 2 ? 150.0 : (i < 3 ? 40.0 : 15.0);
      sample.wrenchFeetEwrtWorld[foot](i) =
          magnitude *
          spread(static_cast<size_t>(6 * foot) + static_cast<size_t>(i),
                 phase + 2.6);
    }
    sample.wrenchFeetEwrtWorld[foot](2) += 300.0;
  }
  return sample;
}

/** Pinocchio external forces - joint frame, about the joint origin - of
 * world-frame wrenches at the sole frames' origins. */
ForceVector computeWrenchLinksEwrtLink(
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const vector_t& q,
    const std::array<vector6_t, 2>& wrenchFeetEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& soleFrames) {
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  ForceVector fext(static_cast<size_t>(model.njoints),
                   pinocchio::Force::Zero());
  for (size_t foot = 0; foot < soleFrames.size(); ++foot) {
    const pinocchio::FrameIndex frame = soleFrames[foot];
    const pinocchio::JointIndex joint = model.frames[frame].parentJoint;
    const matrix3_t rotmatAnkleRtWorld = data.oMi[joint].rotation();
    const vector3_t posFootRtAnkleEwrtWorld =
        data.oMf[frame].translation() - data.oMi[joint].translation();
    const vector3_t forceFootEwrtWorld = wrenchFeetEwrtWorld[foot].head<3>();
    const vector3_t torqueAnkleEwrtWorld =
        wrenchFeetEwrtWorld[foot].tail<3>() +
        posFootRtAnkleEwrtWorld.cross(forceFootEwrtWorld);
    fext[joint] +=
        pinocchio::Force(rotmatAnkleRtWorld.transpose() * forceFootEwrtWorld,
                         rotmatAnkleRtWorld.transpose() * torqueAnkleEwrtWorld);
  }
  return fext;
}

/** RNEA at the base acceleration the wrenches produce, found from RNEA's own
 * base rows. */
struct RneaReference {
  vector6_t baseGeneralizedAcceleration;
  vector6_t baseResidual;
  vector_t jointTorques;
};

RneaReference floatingBaseRnea(
    const pinocchio::Model& model,
    const Sample& sample,
    const ContactArray<pinocchio::FrameIndex>& soleFrames) {
  pinocchio::Data data(model);
  const ForceVector fext = computeWrenchLinksEwrtLink(
      model, data, sample.q, sample.wrenchFeetEwrtWorld, soleFrames);
  vector_t a = vector_t::Zero(model.nv);
  a.tail(sample.qddJoints.size()) = sample.qddJoints;
  // tau(a) = M a + b(q, v, fext), so the base rows vanish at a_b = -M_bb^-1
  // tau_b(a_b = 0).
  const vector6_t baseGeneralizedForceAtZeroBaseGeneralizedAcceleration =
      pinocchio::rnea(model, data, sample.q, sample.v, a, fext).head<6>();
  pinocchio::crba(model, data, sample.q);
  const matrix6_t baseMassMatrix =
      data.M.topLeftCorner<6, 6>().selfadjointView<Eigen::Upper>();
  a.head<6>() = -baseMassMatrix.ldlt().solve(
      baseGeneralizedForceAtZeroBaseGeneralizedAcceleration);
  const vector_t tau =
      pinocchio::rnea(model, data, sample.q, sample.v, a, fext);
  return {.baseGeneralizedAcceleration = a.head<6>(),
          .baseResidual = tau.head<6>(),
          .jointTorques = tau.tail(sample.qddJoints.size())};
}

/** RNEA with the base held still: its joint rows at a_b = 0. */
vector_t baseHeldRnea(const pinocchio::Model& model,
                      const Sample& sample,
                      const ContactArray<pinocchio::FrameIndex>& soleFrames) {
  pinocchio::Data data(model);
  const ForceVector fext = computeWrenchLinksEwrtLink(
      model, data, sample.q, sample.wrenchFeetEwrtWorld, soleFrames);
  vector_t a = vector_t::Zero(model.nv);
  a.tail(sample.qddJoints.size()) = sample.qddJoints;
  return pinocchio::rnea(model, data, sample.q, sample.v, a, fext)
      .tail(sample.qddJoints.size());
}

/** computeJointTorques as it was: the joint rows of crba()'s upper-triangular
 * mass matrix. The positive control. */
vector_t legacyJointTorques(
    const pinocchio::Model& model,
    const Sample& sample,
    const ContactArray<pinocchio::FrameIndex>& soleFrames) {
  pinocchio::Data data(model);
  pinocchio::crba(model, data, sample.q);
  pinocchio::nonLinearEffects(model, data, sample.q, sample.v);
  vector_t generalizedForce = vector_t::Zero(model.nv);
  for (size_t foot = 0; foot < soleFrames.size(); ++foot) {
    matrix_t jacobian = matrix_t::Zero(6, model.nv);
    pinocchio::computeFrameJacobian(
        model, data, sample.q, soleFrames[foot],
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian);
    generalizedForce += jacobian.transpose() * sample.wrenchFeetEwrtWorld[foot];
  }
  const vector6_t baseGeneralizedAcceleration =
      computeBaseGeneralizedAcceleration<scalar_t>(
          data.M, data.nle, sample.qddJoints, generalizedForce);
  vector_t a(model.nv);
  a << baseGeneralizedAcceleration, sample.qddJoints;
  const Eigen::Index numJoints = sample.qddJoints.size();
  return data.M.bottomRows(numJoints) * a + data.nle.tail(numJoints) -
         generalizedForce.tail(numJoints);
}

class JointTorqueInverseDynamicsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g1_ = loadG1WholeBody();
    atlas_ = loadAtlasCentroidal();
    ASSERT_NE(g1_, nullptr)
        << "the G1 whole-body files are not in the runfiles";
    ASSERT_NE(atlas_, nullptr)
        << "the DRC Atlas centroidal files are not in the runfiles";
    // Both models are on the quaternion root, Composite(Translation,
    // Spherical): q holds one entry more than v.
    for (const Robot* absl_nonnull robot : {g1_.get(), atlas_.get()}) {
      ASSERT_EQ(robot->model().nq, robot->model().nv + 1)
          << robot->settings->robotName;
    }
  }

  std::unique_ptr<Robot> g1_;
  std::unique_ptr<Robot> atlas_;
};

TEST_F(JointTorqueInverseDynamicsTest,
       theJointTorquesAreRneaJointRowsAtTheBaseAccelerationTheWrenchesProduce) {
  for (const Robot* absl_nonnull robot : {g1_.get(), atlas_.get()}) {
    for (size_t seed = 0; seed < 5; ++seed) {
      SCOPED_TRACE(absl::StrCat(robot->settings->robotName, ", sample ", seed));
      const Sample sample = makeSample(robot->model(), seed);
      const RneaReference reference =
          floatingBaseRnea(robot->model(), sample, robot->soleFrames());
      const scalar_t scale = std::max(1.0, maxAbs(reference.jointTorques));
      // The reference balances the base, and the case under test has both a
      // base and a joint acceleration.
      ASSERT_LT(maxAbs(reference.baseResidual), 1.0e-9 * scale);
      ASSERT_GT(reference.baseGeneralizedAcceleration.norm(), 0.1);
      ASSERT_GT(maxAbs(sample.qddJoints), 0.1);

      PinocchioInterface pinocchioInterface = *robot->pinocchioInterface;
      const vector_t jointTorques = computeJointTorques<scalar_t>(
          sample.q, sample.v, sample.qddJoints,
          toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
          robot->soleFrames(), pinocchioInterface);
      EXPECT_LT(maxAbs(jointTorques - reference.jointTorques), 1.0e-9 * scale)
          << "computeJointTorques: " << jointTorques.transpose()
          << "\nRNEA:                " << reference.jointTorques.transpose();

      // Positive control: the upper-triangular joint rows are measurably wrong
      // on the same sample.
      EXPECT_GT(maxAbs(legacyJointTorques(robot->model(), sample,
                                          robot->soleFrames()) -
                       reference.jointTorques),
                1.0e-2);
    }
  }
}

TEST_F(JointTorqueInverseDynamicsTest,
       theJointTorquesDoNotDependOnWhatTheDataHeld) {
  const Sample sample = makeSample(g1_->model(), /*seed=*/3);
  PinocchioInterface fresh = *g1_->pinocchioInterface;
  const vector_t expected = computeJointTorques<scalar_t>(
      sample.q, sample.v, sample.qddJoints,
      toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld), g1_->soleFrames(),
      fresh);

  // crba() writes each joint's rows over its own subtree only, and leaves every
  // other entry of M - the lower triangle, and the two legs' coupling - at
  // whatever an earlier user of the data left there. None of it may reach the
  // result. Positive control: crba() on such a data does leave entries of the
  // joint rows untouched, so the case is exercised.
  pinocchio::Data probe(g1_->model());
  probe.M.setConstant(1.0e3);
  pinocchio::crba(g1_->model(), probe, sample.q);
  ASSERT_TRUE((probe.M.bottomRows(g1_->numJoints()).array() == 1.0e3).any())
      << "crba() now writes every entry of M";

  PinocchioInterface used = *g1_->pinocchioInterface;
  used.getData().M.setConstant(1.0e3);
  const vector_t jointTorques = computeJointTorques<scalar_t>(
      sample.q, sample.v, sample.qddJoints,
      toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld), g1_->soleFrames(),
      used);
  EXPECT_LT(maxAbs(jointTorques - expected),
            1.0e-12 * std::max(1.0, maxAbs(expected)));
  EXPECT_TRUE(used.getData().M.isApprox(used.getData().M.transpose(), 1.0e-12))
      << "the mass matrix is left symmetric";
}

TEST_F(JointTorqueInverseDynamicsTest,
       theBaseHeldJointTorquesAreRneaJointRowsAtRest) {
  for (const Robot* absl_nonnull robot : {g1_.get(), atlas_.get()}) {
    for (size_t seed = 0; seed < 3; ++seed) {
      SCOPED_TRACE(absl::StrCat(robot->settings->robotName, ", sample ", seed));
      Sample sample = makeSample(robot->model(), seed);
      PinocchioInterface pinocchioInterface = *robot->pinocchioInterface;
      const vector_t expected =
          baseHeldRnea(robot->model(), sample, robot->soleFrames());
      const scalar_t scale = std::max(1.0, maxAbs(expected));
      EXPECT_LT(
          maxAbs(computeBaseHeldJointTorques<scalar_t>(
                     sample.q, sample.v, sample.qddJoints,
                     toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
                     robot->soleFrames(), pinocchioInterface) -
                 expected),
          1.0e-9 * scale);

      // At zero joint acceleration it is exactly what computeJointTorques
      // returned before it carried the base acceleration - which is what the
      // gantry's weight-compensating feedforward was tuned on.
      sample.qddJoints.setZero();
      const vector_t legacy =
          legacyJointTorques(robot->model(), sample, robot->soleFrames());
      EXPECT_LT(
          maxAbs(computeBaseHeldJointTorques<scalar_t>(
                     sample.q, sample.v, sample.qddJoints,
                     toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
                     robot->soleFrames(), pinocchioInterface) -
                 legacy),
          1.0e-9 * scale);

      // With no wrench and the robot at rest: the gravity torques of the free
      // legs, g_j(q).
      sample.v.setZero();
      sample.wrenchFeetEwrtWorld = {vector6_t::Zero(), vector6_t::Zero()};
      pinocchio::Data data(robot->model());
      pinocchio::nonLinearEffects(robot->model(), data, sample.q, sample.v);
      const vector_t gravityJointTorques = data.nle.tail(robot->numJoints());
      EXPECT_LT(
          maxAbs(computeBaseHeldJointTorques<scalar_t>(
                     sample.q, sample.v, sample.qddJoints,
                     toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
                     robot->soleFrames(), pinocchioInterface) -
                 gravityJointTorques),
          1.0e-9 * std::max(1.0, maxAbs(gravityJointTorques)));
      // ...whereas the floating base falls freely without a wrench, and gravity
      // then loads no joint.
      EXPECT_GT(
          maxAbs(computeJointTorques<scalar_t>(
                     sample.q, sample.v, sample.qddJoints,
                     toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
                     robot->soleFrames(), pinocchioInterface) -
                 gravityJointTorques),
          1.0);
    }
  }
}

/** A copy of `robot`'s Pinocchio interface whose contact frames are renamed
 * `newNames`, which must be new frame names. */
PinocchioInterface withRenamedContactFrames(
    const Robot& robot, const BipedFeetArray<std::string>& newNames) {
  pinocchio::Model model = robot.model();
  for (size_t i = 0; i < newNames.size(); ++i) {
    model.frames[model.getFrameId(robot.settings->contactNames[i])].name =
        newNames[i];
  }
  return PinocchioInterface(model);
}

TEST_F(JointTorqueInverseDynamicsTest,
       theWrenchesAreAppliedAtTheConfiguredContactFramesWhateverTheirNames) {
  // The inverse dynamics used to look the soles up as "foot_l_contact" and
  // "foot_r_contact"; a robot whose contact frames are named otherwise read
  // frame nframes, out of bounds. Renamed, the frames are found by the
  // configured names, and the torques are RNEA's with the wrenches on those
  // frames.
  const BipedFeetArray<std::string> renamed = {"left_sole_renamed",
                                               "right_sole_renamed"};
  const PinocchioInterface renamedInterface =
      withRenamedContactFrames(*g1_, renamed);
  const pinocchio::Model& model = renamedInterface.getModel();
  ASSERT_FALSE(model.existFrame("foot_l_contact") ||
               model.existFrame("foot_r_contact"))
      << "the frames were not renamed";
  EXPECT_TRUE(checkContactFramesExist(model, renamed).ok());
  EXPECT_EQ(checkContactFramesExist(model, g1_->settings->contactNames).code(),
            absl::StatusCode::kNotFound);

  ContactArray<pinocchio::FrameIndex> soleFrames(renamed.size());
  soleFrames[0] = model.getFrameId(renamed[0]);
  soleFrames[1] = model.getFrameId(renamed[1]);
  EXPECT_EQ(soleFrames, g1_->soleFrames())
      << "renaming keeps the frames' indices";
  for (size_t seed = 0; seed < 3; ++seed) {
    SCOPED_TRACE(absl::StrCat("sample ", seed));
    const Sample sample = makeSample(model, seed);
    const RneaReference reference = floatingBaseRnea(model, sample, soleFrames);
    const scalar_t scale = std::max(1.0, maxAbs(reference.jointTorques));
    PinocchioInterface pinocchioInterface = renamedInterface;
    EXPECT_LT(maxAbs(computeJointTorques<scalar_t>(
                         sample.q, sample.v, sample.qddJoints,
                         toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
                         soleFrames, pinocchioInterface) -
                     reference.jointTorques),
              1.0e-9 * scale);
    EXPECT_LT(maxAbs(computeBaseHeldJointTorques<scalar_t>(
                         sample.q, sample.v, sample.qddJoints,
                         toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
                         soleFrames, pinocchioInterface) -
                     baseHeldRnea(model, sample, soleFrames)),
              1.0e-9 * scale);
    // Positive control: the frames matter - the wrenches on the exchanged soles
    // are other torques.
    ContactArray<pinocchio::FrameIndex> exchanged = soleFrames;
    exchanged[0] = soleFrames[1];
    exchanged[1] = soleFrames[0];
    EXPECT_GT(maxAbs(computeJointTorques<scalar_t>(
                         sample.q, sample.v, sample.qddJoints,
                         toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
                         exchanged, pinocchioInterface) -
                     reference.jointTorques),
              1.0);
  }
}

/*
 * The inverse dynamics take any number of contacts, in contact order. With four
 * contacts - two point contacts at the joint frames of the sole joints' parents
 * ahead of the two soles, contact order putting the 3-DoF contacts first - the
 * free-base torques are M a + nle - J^T W with all four wrenches, and the
 * base-held ones
 * differ from the two soles' alone by exactly -J_p^T f_p of the point
 * contacts.
 */
TEST_F(JointTorqueInverseDynamicsTest, coverEveryContactOfAFourContactSet) {
  const pinocchio::Model& model = g1_->model();
  const ContactArray<pinocchio::FrameIndex> soleFrames = g1_->soleFrames();
  constexpr size_t kNumPointContacts = 2;
  ContactArray<pinocchio::FrameIndex> frames(/*numContacts=*/4, /*value=*/0);
  for (size_t i = 0; i < kNumPointContacts; ++i) {
    const pinocchio::JointIndex soleJoint =
        model.frames[soleFrames[i]].parentJoint;
    frames[i] = model.getFrameId(model.names[model.parents[soleJoint]]);
    ASSERT_LT(frames[i], model.frames.size());
    frames[kNumPointContacts + i] = soleFrames[i];
  }
  for (size_t seed = 0; seed < 3; ++seed) {
    SCOPED_TRACE(absl::StrCat("sample ", seed));
    const Sample sample = makeSample(model, seed);
    ContactArray<vector6_t> wrenchContactsEwrtWorld(/*numContacts=*/4,
                                                    vector6_t::Zero());
    for (size_t i = 0; i < kNumPointContacts; ++i) {
      // A point contact carries a force alone.
      wrenchContactsEwrtWorld[i].head<3>() =
          sample.wrenchFeetEwrtWorld[1 - i].head<3>();
      wrenchContactsEwrtWorld[kNumPointContacts + i] =
          sample.wrenchFeetEwrtWorld[i];
    }

    // Free base: the joint rows of M a + nle - J^T W at the base acceleration
    // the four wrenches produce, built here from Pinocchio's terms.
    PinocchioInterface pinocchioInterface = *g1_->pinocchioInterface;
    const vector_t freeBase = computeJointTorques<scalar_t>(
        sample.q, sample.v, sample.qddJoints, wrenchContactsEwrtWorld, frames,
        pinocchioInterface);
    pinocchio::Data data(model);
    pinocchio::forwardKinematics(model, data, sample.q);
    pinocchio::updateFramePlacements(model, data);
    vector_t generalizedForce = vector_t::Zero(model.nv);
    for (size_t i = 0; i < frames.size(); ++i) {
      matrix_t jacobian = matrix_t::Zero(6, model.nv);
      pinocchio::computeFrameJacobian(
          model, data, sample.q, frames[i],
          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian);
      generalizedForce += jacobian.transpose() * wrenchContactsEwrtWorld[i];
    }
    pinocchio::Data massData(model);
    pinocchio::crba(model, massData, sample.q);
    massData.M.triangularView<Eigen::StrictlyLower>() =
        massData.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::nonLinearEffects(model, massData, sample.q, sample.v);
    const Eigen::Index numJoints = sample.qddJoints.size();
    vector_t generalizedAcceleration(model.nv);
    generalizedAcceleration << computeBaseGeneralizedAcceleration<scalar_t>(
        massData.M, massData.nle, sample.qddJoints, generalizedForce),
        sample.qddJoints;
    const vector_t expectedFreeBase =
        massData.M.bottomRows(numJoints) * generalizedAcceleration +
        massData.nle.tail(numJoints) - generalizedForce.tail(numJoints);
    const scalar_t scale = std::max(1.0, maxAbs(expectedFreeBase));
    EXPECT_LT(maxAbs(freeBase - expectedFreeBase), 1.0e-9 * scale);

    // Base held: linear in the wrenches, so the point contacts add -J_p^T f_p.
    PinocchioInterface heldInterface = *g1_->pinocchioInterface;
    const vector_t heldBase = computeBaseHeldJointTorques<scalar_t>(
        sample.q, sample.v, sample.qddJoints, wrenchContactsEwrtWorld, frames,
        heldInterface);
    ContactArray<vector6_t> wrenchFeetEwrtWorld(
        /*numContacts=*/2, vector6_t::Zero());
    wrenchFeetEwrtWorld[0] = wrenchContactsEwrtWorld[kNumPointContacts];
    wrenchFeetEwrtWorld[1] = wrenchContactsEwrtWorld[kNumPointContacts + 1];
    const vector_t heldBaseSoles = computeBaseHeldJointTorques<scalar_t>(
        sample.q, sample.v, sample.qddJoints, wrenchFeetEwrtWorld, soleFrames,
        heldInterface);
    vector_t generalizedForceOfPoints = vector_t::Zero(model.nv);
    for (size_t i = 0; i < kNumPointContacts; ++i) {
      matrix_t jacobian = matrix_t::Zero(6, model.nv);
      pinocchio::computeFrameJacobian(
          model, data, sample.q, frames[i],
          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian);
      generalizedForceOfPoints +=
          jacobian.transpose() * wrenchContactsEwrtWorld[i];
    }
    EXPECT_LT(maxAbs(heldBase - (heldBaseSoles -
                                 generalizedForceOfPoints.tail(numJoints))),
              1.0e-9 * std::max(1.0, maxAbs(heldBase)));
    // Positive control: the point contacts change the torques.
    EXPECT_GT(maxAbs(heldBase - heldBaseSoles), 1.0);
  }
}

TEST(CheckContactFramesExist, RefusesNoNamesAndNamesTheMissingFrame) {
  pinocchio::Model model;
  model.addFrame(pinocchio::Frame("left_sole", /*parentJoint=*/0,
                                  pinocchio::SE3::Identity(),
                                  pinocchio::OP_FRAME));
  model.addFrame(pinocchio::Frame("right_sole", /*parentJoint=*/0,
                                  pinocchio::SE3::Identity(),
                                  pinocchio::OP_FRAME));
  EXPECT_TRUE(checkContactFramesExist(
                  model, std::vector<std::string>{"left_sole", "right_sole"})
                  .ok());
  // Any number of contacts is a contact set; none is not.
  EXPECT_TRUE(
      checkContactFramesExist(model, std::vector<std::string>{"left_sole"})
          .ok());
  EXPECT_EQ(checkContactFramesExist(model, std::vector<std::string>{}).code(),
            absl::StatusCode::kInvalidArgument);
  // Every name is checked, not only the first two.
  EXPECT_EQ(checkContactFramesExist(
                model, std::vector<std::string>{"left_sole", "right_sole",
                                                "no_such_point"})
                .code(),
            absl::StatusCode::kNotFound);
  const absl::Status missing = checkContactFramesExist(
      model, std::vector<std::string>{"left_sole", "no_such_sole"});
  EXPECT_EQ(missing.code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(missing.message(), "no_such_sole")) << missing;
}

TEST_F(JointTorqueInverseDynamicsTest,
       theTapedJointTorquesMatchAndCarryNoComparisonBetweenVariables) {
  // JointTorqueCostCppAd tapes computeJointTorques<ad_scalar_t>; the taped
  // function must be the corrected one, and CppADCodeGen must be able to
  // generate code for it.
  const pinocchio::Model& model = g1_->model();
  const Eigen::Index nv = model.nv;
  const Eigen::Index nj = nv - static_cast<Eigen::Index>(kBaseVelocityDim);
  const Eigen::Index numInputs = model.nq + nv + nj + 12;
  const Sample sample = makeSample(model, /*seed=*/1);
  vector_t values(numInputs);
  values << sample.q, sample.v, sample.qddJoints, sample.wrenchFeetEwrtWorld[0],
      sample.wrenchFeetEwrtWorld[1];

  ad_vector_t x(numInputs);
  for (Eigen::Index i = 0; i < numInputs; ++i) x(i) = values(i);
  CppAD::Independent(x);
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd =
      g1_->pinocchioInterface->toCppAd();
  ContactArray<ad_vector6_t> wrenchFeetEwrtWorld(/*numContacts=*/2);
  wrenchFeetEwrtWorld[0] = ad_vector6_t(x.segment(model.nq + nv + nj, 6));
  wrenchFeetEwrtWorld[1] = ad_vector6_t(x.tail(6));
  ad_vector_t y = computeJointTorques<ad_scalar_t>(
      x.head(model.nq), x.segment(model.nq, nv), x.segment(model.nq + nv, nj),
      wrenchFeetEwrtWorld, g1_->soleFrames(), pinocchioInterfaceCppAd);
  ad_fun_t fun(x, y);

  CppAD::vector<ad_base_t> numericInput(static_cast<size_t>(numInputs));
  for (Eigen::Index i = 0; i < numInputs; ++i) {
    numericInput[static_cast<size_t>(i)] = ad_base_t(values(i));
  }
  const CppAD::vector<ad_base_t> numericOutput =
      fun.Forward(/*q=*/0, numericInput);
  vector_t taped(nj);
  for (Eigen::Index i = 0; i < nj; ++i) {
    taped(i) = numericOutput[static_cast<size_t>(i)].getValue();
  }
  const RneaReference reference =
      floatingBaseRnea(model, sample, g1_->soleFrames());
  EXPECT_LT(maxAbs(taped - reference.jointTorques),
            1.0e-9 * std::max(1.0, maxAbs(reference.jointTorques)));

  CppAD::cg::CodeHandler<scalar_t> handler;
  CppAD::vector<ad_base_t> variables(static_cast<size_t>(numInputs));
  handler.makeVariables(variables);
  EXPECT_NO_THROW(fun.Forward(/*q=*/0, variables));
}

/*
 * What the correction changes on the shipped DRC Atlas, the robot being tuned:
 * the centroidal controller commands computeJointTorques at zero joint
 * acceleration, so the change is exactly the joint-base coupling M_jb a_b. The
 * magnitudes are logged for the record; the assertions are the properties.
 */
TEST_F(JointTorqueInverseDynamicsTest,
       theCorrectionOnTheShippedAtlasStanceIsTheBaseCouplingAlone) {
  const pinocchio::Model& model = atlas_->model();
  const absl::StatusOr<vector_t> initial = stateValuesFromConfig(
      atlas_->task.initial_state,
      stateInputLayout(*atlas_->settings, StateInputLayout::Mpc::kCentroidal),
      "initial_state");
  ASSERT_TRUE(initial.ok()) << initial.status();
  // The task file's initial_state is written in the tuning layout (ZYX Euler
  // angles); the state stores the quaternion.
  const absl::StatusOr<vector_t> initialState = stateFromTuningLayout(
      *initial, placeTuningLayout(tuningLayoutOf(stateInputLayout(
                    *atlas_->settings, StateInputLayout::Mpc::kCentroidal))));
  ASSERT_TRUE(initialState.ok()) << initialState.status();
  Sample standing;
  standing.q = initialState->tail(model.nq);
  ASSERT_NEAR(standing.q.segment<4>(kBaseTranslationDim).norm(), 1.0, 1.0e-15);
  standing.v = vector_t::Zero(model.nv);
  standing.qddJoints = vector_t::Zero(atlas_->numJoints());
  const scalar_t weight =
      kGravityMagnitude * pinocchio::computeTotalMass(model);
  vector6_t halfWeightWrenchFootEwrtWorld = vector6_t::Zero();
  halfWeightWrenchFootEwrtWorld(2) = 0.5 * weight;
  standing.wrenchFeetEwrtWorld = {halfWeightWrenchFootEwrtWorld,
                                  halfWeightWrenchFootEwrtWorld};

  // Single support on the left foot: the weight straight up, and the weight
  // along the line from the sole to the center of mass as the linear inverted
  // pendulum carries it (no moment about the center of mass).
  Sample singleSupport = standing;
  singleSupport.wrenchFeetEwrtWorld = {vector6_t::Zero(), vector6_t::Zero()};
  singleSupport.wrenchFeetEwrtWorld[0](2) = weight;
  Sample pendulum = singleSupport;
  pinocchio::Data data(model);
  const vector3_t posComRtWorldEwrtWorld =
      pinocchio::centerOfMass(model, data, standing.q);
  pinocchio::framesForwardKinematics(model, data, standing.q);
  const vector3_t posComRtLfootEwrtWorld =
      posComRtWorldEwrtWorld - data.oMf[atlas_->soleFrames()[0]].translation();
  pendulum.wrenchFeetEwrtWorld[0].head<3>() =
      weight * posComRtLfootEwrtWorld / posComRtLfootEwrtWorld.z();

  const std::array<
      std::pair<const char* absl_nonnull, const Sample* absl_nonnull>, 3>
      cases = {{{"double support, half the weight on each sole", &standing},
                {"single support, the weight straight up", &singleSupport},
                {"single support, the weight toward the center of mass",
                 &pendulum}}};
  for (size_t c = 0; c < cases.size(); ++c) {
    const char* absl_nonnull description = cases[c].first;
    SCOPED_TRACE(description);
    const Sample& sample = *cases[c].second;
    PinocchioInterface pinocchioInterface = *atlas_->pinocchioInterface;
    const vector_t corrected = computeJointTorques<scalar_t>(
        sample.q, sample.v, sample.qddJoints,
        toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
        atlas_->soleFrames(), pinocchioInterface);
    const RneaReference reference =
        floatingBaseRnea(model, sample, atlas_->soleFrames());
    EXPECT_LT(maxAbs(corrected - reference.jointTorques),
              1.0e-9 * std::max(1.0, maxAbs(reference.jointTorques)));

    // At zero joint acceleration the correction is the base coupling: the
    // base-held torques plus M_jb a_b.
    PinocchioInterface baseHeldInterface = *atlas_->pinocchioInterface;
    const vector_t baseHeld = computeBaseHeldJointTorques<scalar_t>(
        sample.q, sample.v, sample.qddJoints,
        toWrenchContactsEwrtWorld(sample.wrenchFeetEwrtWorld),
        atlas_->soleFrames(), baseHeldInterface);
    const matrix_t& M = pinocchioInterface.getData().M;
    const vector_t coupling = M.bottomLeftCorner(atlas_->numJoints(), 6) *
                              reference.baseGeneralizedAcceleration;
    EXPECT_LT(maxAbs(corrected - baseHeld - coupling),
              1.0e-9 * std::max(1.0, maxAbs(corrected)));

    const vector_t change =
        corrected - legacyJointTorques(model, sample, atlas_->soleFrames());
    Eigen::Index joint = 0;
    const scalar_t largest = change.cwiseAbs().maxCoeff(&joint);
    LOG(INFO) << "[DRC Atlas, " << description
              << "] a_b = " << reference.baseGeneralizedAcceleration.transpose()
              << "; largest torque change " << largest << " Nm at "
              << model.names[static_cast<size_t>(joint) + 2]
              << ", RMS over the joints "
              << change.norm() / std::sqrt(static_cast<scalar_t>(change.size()))
              << " Nm, largest torque " << maxAbs(corrected) << " Nm";
    RecordProperty(absl::StrCat("largest_change_nm_", c),
                   absl::StrCat(largest));
  }
}

}  // namespace
}  // namespace ocs2::humanoid
