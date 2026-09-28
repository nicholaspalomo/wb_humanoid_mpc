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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include <ocs2_core/automatic_differentiation/Types.h>
#include <ocs2_core/misc/LoadData.h>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

/*
 * computeJointTorques() and computeBaseHeldJointTorques() against Pinocchio's recursive Newton-Euler algorithm.
 *
 * The reference is built without anything computeJointTorques uses: the contact wrenches reach RNEA as external forces
 * on the foot joints rather than through the frame Jacobians, and the base acceleration the wrenches produce is found
 * from RNEA's own base rows (RNEA is affine in the acceleration, so the unactuated base rows vanish at one base
 * acceleration). computeJointTorques used to multiply the joint rows of crba()'s mass matrix, of which Pinocchio fills
 * only the upper triangle: it left out the joint-base coupling M_jb a_b and, with a joint acceleration, the lower
 * triangle of M_jj. The legacy formula is kept here as the positive control that the comparison sees that difference.
 */

namespace ocs2::humanoid {
namespace {

using ad_fun_t = CppAD::ADFun<ad_base_t>;
using ForceVector = pinocchio::container::aligned_vector<pinocchio::Force>;

constexpr std::array<const char*, 2> kSoleFrames = {"foot_l_contact", "foot_r_contact"};
constexpr scalar_t kGravity = 9.81;

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) roots.emplace_back(std::filesystem::path(srcDir) / "_main");
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
  std::string taskFile;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;

  const pinocchio::Model& model() const { return pinocchioInterface->getModel(); }
  Eigen::Index numJoints() const { return model().nv - 6; }
};

std::unique_ptr<Robot> loadRobot(absl::string_view taskFile, absl::string_view urdfFile, absl::string_view mpcName) {
  std::unique_ptr<Robot> robot = std::make_unique<Robot>();
  robot->taskFile = runfilePath(taskFile);
  const std::string urdf = runfilePath(urdfFile);
  if (robot->taskFile.empty() || urdf.empty()) return nullptr;
  robot->settings = std::make_unique<ModelSettings>(robot->taskFile, urdf, std::string(mpcName), /*verbose=*/false);
  robot->pinocchioInterface = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(robot->taskFile, urdf, *robot->settings));
  return robot;
}

// LINT.IfChange(robot_files)
std::unique_ptr<Robot> loadG1WholeBody() {
  return loadRobot("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml", "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
                   "wb_mpc_");
}

std::unique_ptr<Robot> loadAtlasCentroidal() {
  return loadRobot("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
                   "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf", "centroidal_mpc_");
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:joint_torque_test_data)

/** A state, joint accelerations and sole wrenches [W_left, W_right] (world frame, at the sole frames). */
struct Sample {
  vector_t q;
  vector_t v;
  vector_t qddJoints;
  std::array<vector6_t, 2> wrenches;
};

/** Random-looking but deterministic: every generalized coordinate, velocity and joint acceleration non-zero. */
Sample makeSample(const pinocchio::Model& model, size_t seed) {
  const scalar_t phase = 0.37 * static_cast<scalar_t>(seed);
  Sample sample;
  sample.q = vector_t(model.nq);
  sample.v = vector_t(model.nv);
  sample.qddJoints = vector_t(model.nv - 6);
  for (Eigen::Index i = 0; i < model.nq; ++i) sample.q(i) = 0.4 * spread(static_cast<size_t>(i), phase + 0.7);
  sample.q(2) = 0.9;
  for (Eigen::Index i = 0; i < model.nv; ++i) sample.v(i) = 0.8 * spread(static_cast<size_t>(i), phase + 1.9);
  for (Eigen::Index i = 0; i < sample.qddJoints.size(); ++i) sample.qddJoints(i) = 3.0 * spread(static_cast<size_t>(i), phase + 0.4);
  for (size_t foot = 0; foot < 2; ++foot) {
    for (Eigen::Index i = 0; i < 6; ++i) {
      const scalar_t magnitude = i == 2 ? 150.0 : (i < 3 ? 40.0 : 15.0);
      sample.wrenches[foot](i) = magnitude * spread(static_cast<size_t>(6 * foot) + static_cast<size_t>(i), phase + 2.6);
    }
    sample.wrenches[foot](2) += 300.0;
  }
  return sample;
}

/** Pinocchio external forces - joint frame, about the joint origin - of world-frame wrenches at the sole frames' origins. */
ForceVector soleWrenchesAsJointForces(const pinocchio::Model& model,
                                      pinocchio::Data& data,
                                      const vector_t& q,
                                      const std::array<vector6_t, 2>& wrenches) {
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  ForceVector fext(static_cast<size_t>(model.njoints), pinocchio::Force::Zero());
  for (size_t foot = 0; foot < kSoleFrames.size(); ++foot) {
    const pinocchio::FrameIndex frame = model.getFrameId(kSoleFrames[foot]);
    const pinocchio::JointIndex joint = model.frames[frame].parentJoint;
    const matrix3_t rotation = data.oMi[joint].rotation();
    const vector3_t lever = data.oMf[frame].translation() - data.oMi[joint].translation();
    const vector3_t force = wrenches[foot].head<3>();
    const vector3_t moment = wrenches[foot].tail<3>() + lever.cross(force);
    fext[joint] += pinocchio::Force(rotation.transpose() * force, rotation.transpose() * moment);
  }
  return fext;
}

/** RNEA at the base acceleration the wrenches produce, found from RNEA's own base rows. */
struct RneaReference {
  vector6_t baseAcceleration;
  vector6_t baseResidual;
  vector_t jointTorques;
};

RneaReference floatingBaseRnea(const pinocchio::Model& model, const Sample& sample) {
  pinocchio::Data data(model);
  const ForceVector fext = soleWrenchesAsJointForces(model, data, sample.q, sample.wrenches);
  vector_t a = vector_t::Zero(model.nv);
  a.tail(sample.qddJoints.size()) = sample.qddJoints;
  // tau(a) = M a + b(q, v, fext), so the base rows vanish at a_b = -M_bb^-1 tau_b(a_b = 0).
  const vector6_t baseForceAtZeroBaseAcceleration = pinocchio::rnea(model, data, sample.q, sample.v, a, fext).head<6>();
  pinocchio::crba(model, data, sample.q);
  const matrix6_t baseMassMatrix = data.M.topLeftCorner<6, 6>().selfadjointView<Eigen::Upper>();
  a.head<6>() = -baseMassMatrix.ldlt().solve(baseForceAtZeroBaseAcceleration);
  const vector_t tau = pinocchio::rnea(model, data, sample.q, sample.v, a, fext);
  return {a.head<6>(), tau.head<6>(), tau.tail(sample.qddJoints.size())};
}

/** RNEA with the base held still: its joint rows at a_b = 0. */
vector_t baseHeldRnea(const pinocchio::Model& model, const Sample& sample) {
  pinocchio::Data data(model);
  const ForceVector fext = soleWrenchesAsJointForces(model, data, sample.q, sample.wrenches);
  vector_t a = vector_t::Zero(model.nv);
  a.tail(sample.qddJoints.size()) = sample.qddJoints;
  return pinocchio::rnea(model, data, sample.q, sample.v, a, fext).tail(sample.qddJoints.size());
}

/** computeJointTorques as it was: the joint rows of crba()'s upper-triangular mass matrix. The positive control. */
vector_t legacyJointTorques(const pinocchio::Model& model, const Sample& sample) {
  pinocchio::Data data(model);
  pinocchio::crba(model, data, sample.q);
  pinocchio::nonLinearEffects(model, data, sample.q, sample.v);
  vector_t generalizedForce = vector_t::Zero(model.nv);
  for (size_t foot = 0; foot < kSoleFrames.size(); ++foot) {
    matrix_t jacobian = matrix_t::Zero(6, model.nv);
    pinocchio::computeFrameJacobian(model, data, sample.q, model.getFrameId(kSoleFrames[foot]),
                                    pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian);
    generalizedForce += jacobian.transpose() * sample.wrenches[foot];
  }
  const vector6_t baseAcceleration = computeBaseAcceleration<scalar_t>(data.M, data.nle, sample.qddJoints, generalizedForce);
  vector_t a(model.nv);
  a << baseAcceleration, sample.qddJoints;
  const Eigen::Index numJoints = sample.qddJoints.size();
  return data.M.bottomRows(numJoints) * a + data.nle.tail(numJoints) - generalizedForce.tail(numJoints);
}

class JointTorqueInverseDynamicsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g1_ = loadG1WholeBody();
    atlas_ = loadAtlasCentroidal();
    ASSERT_NE(g1_, nullptr) << "the G1 whole-body files are not in the runfiles";
    ASSERT_NE(atlas_, nullptr) << "the DRC Atlas centroidal files are not in the runfiles";
  }

  std::unique_ptr<Robot> g1_;
  std::unique_ptr<Robot> atlas_;
};

TEST_F(JointTorqueInverseDynamicsTest, theJointTorquesAreRneaJointRowsAtTheBaseAccelerationTheWrenchesProduce) {
  for (const Robot* robot : {g1_.get(), atlas_.get()}) {
    for (size_t seed = 0; seed < 5; ++seed) {
      SCOPED_TRACE(absl::StrCat(robot->taskFile, ", sample ", seed));
      const Sample sample = makeSample(robot->model(), seed);
      const RneaReference reference = floatingBaseRnea(robot->model(), sample);
      const scalar_t scale = std::max(1.0, maxAbs(reference.jointTorques));
      // The reference balances the base, and the case under test has both a base and a joint acceleration.
      ASSERT_LT(maxAbs(reference.baseResidual), 1e-9 * scale);
      ASSERT_GT(reference.baseAcceleration.norm(), 0.1);
      ASSERT_GT(maxAbs(sample.qddJoints), 0.1);

      PinocchioInterface pinocchioInterface = *robot->pinocchioInterface;
      const vector_t torques = computeJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, pinocchioInterface);
      EXPECT_LT(maxAbs(torques - reference.jointTorques), 1e-9 * scale)
          << "computeJointTorques: " << torques.transpose() << "\nRNEA:                " << reference.jointTorques.transpose();

      // Positive control: the upper-triangular joint rows are measurably wrong on the same sample.
      EXPECT_GT(maxAbs(legacyJointTorques(robot->model(), sample) - reference.jointTorques), 1e-2);
    }
  }
}

TEST_F(JointTorqueInverseDynamicsTest, theJointTorquesDoNotDependOnWhatTheDataHeld) {
  const Sample sample = makeSample(g1_->model(), /*seed=*/3);
  PinocchioInterface fresh = *g1_->pinocchioInterface;
  const vector_t expected = computeJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, fresh);

  // crba() writes each joint's rows over its own subtree only, and leaves every other entry of M - the lower triangle,
  // and the two legs' coupling - at whatever an earlier user of the data left there. None of it may reach the result.
  // Positive control: crba() on such a data does leave entries of the joint rows untouched, so the case is exercised.
  pinocchio::Data probe(g1_->model());
  probe.M.setConstant(1.0e3);
  pinocchio::crba(g1_->model(), probe, sample.q);
  ASSERT_TRUE((probe.M.bottomRows(g1_->numJoints()).array() == 1.0e3).any()) << "crba() now writes every entry of M";

  PinocchioInterface used = *g1_->pinocchioInterface;
  used.getData().M.setConstant(1.0e3);
  const vector_t torques = computeJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, used);
  EXPECT_LT(maxAbs(torques - expected), 1e-12 * std::max(1.0, maxAbs(expected)));
  EXPECT_TRUE(used.getData().M.isApprox(used.getData().M.transpose(), 1e-12)) << "the mass matrix is left symmetric";
}

TEST_F(JointTorqueInverseDynamicsTest, theBaseHeldJointTorquesAreRneaJointRowsAtRest) {
  for (const Robot* robot : {g1_.get(), atlas_.get()}) {
    for (size_t seed = 0; seed < 3; ++seed) {
      SCOPED_TRACE(absl::StrCat(robot->taskFile, ", sample ", seed));
      Sample sample = makeSample(robot->model(), seed);
      PinocchioInterface pinocchioInterface = *robot->pinocchioInterface;
      const vector_t expected = baseHeldRnea(robot->model(), sample);
      const scalar_t scale = std::max(1.0, maxAbs(expected));
      EXPECT_LT(maxAbs(computeBaseHeldJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, pinocchioInterface) -
                       expected),
                1e-9 * scale);

      // At zero joint acceleration it is exactly what computeJointTorques returned before it carried the base
      // acceleration - which is what the gantry's weight-compensating feedforward was tuned on.
      sample.qddJoints.setZero();
      const vector_t legacy = legacyJointTorques(robot->model(), sample);
      EXPECT_LT(
          maxAbs(computeBaseHeldJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, pinocchioInterface) - legacy),
          1e-9 * scale);

      // With no wrench and the robot at rest: the gravity torques of the free legs, g_j(q).
      sample.v.setZero();
      sample.wrenches = {vector6_t::Zero(), vector6_t::Zero()};
      pinocchio::Data data(robot->model());
      pinocchio::nonLinearEffects(robot->model(), data, sample.q, sample.v);
      const vector_t gravity = data.nle.tail(robot->numJoints());
      EXPECT_LT(maxAbs(computeBaseHeldJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, pinocchioInterface) -
                       gravity),
                1e-9 * std::max(1.0, maxAbs(gravity)));
      // ...whereas the floating base falls freely without a wrench, and gravity then loads no joint.
      EXPECT_GT(maxAbs(computeJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, pinocchioInterface) - gravity),
                1.0);
    }
  }
}

TEST_F(JointTorqueInverseDynamicsTest, theTapedJointTorquesMatchAndCarryNoComparisonBetweenVariables) {
  // JointTorqueCostCppAd tapes computeJointTorques<ad_scalar_t>; the taped function must be the corrected one, and
  // CppADCodeGen must be able to generate code for it.
  const pinocchio::Model& model = g1_->model();
  const Eigen::Index nv = model.nv;
  const Eigen::Index nj = nv - 6;
  const Eigen::Index numInputs = model.nq + nv + nj + 12;
  const Sample sample = makeSample(model, /*seed=*/1);
  vector_t values(numInputs);
  values << sample.q, sample.v, sample.qddJoints, sample.wrenches[0], sample.wrenches[1];

  ad_vector_t x(numInputs);
  for (Eigen::Index i = 0; i < numInputs; ++i) x(i) = values(i);
  CppAD::Independent(x);
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd = g1_->pinocchioInterface->toCppAd();
  const std::array<ad_vector6_t, 2> wrenches = {ad_vector6_t(x.segment(model.nq + nv + nj, 6)), ad_vector6_t(x.tail(6))};
  ad_vector_t y = computeJointTorques<ad_scalar_t>(x.head(model.nq), x.segment(model.nq, nv), x.segment(model.nq + nv, nj), wrenches,
                                                   pinocchioInterfaceCppAd);
  ad_fun_t fun(x, y);

  CppAD::vector<ad_base_t> numericInput(static_cast<size_t>(numInputs));
  for (Eigen::Index i = 0; i < numInputs; ++i) numericInput[static_cast<size_t>(i)] = ad_base_t(values(i));
  const CppAD::vector<ad_base_t> numericOutput = fun.Forward(/*q=*/0, numericInput);
  vector_t taped(nj);
  for (Eigen::Index i = 0; i < nj; ++i) taped(i) = numericOutput[static_cast<size_t>(i)].getValue();
  const RneaReference reference = floatingBaseRnea(model, sample);
  EXPECT_LT(maxAbs(taped - reference.jointTorques), 1e-9 * std::max(1.0, maxAbs(reference.jointTorques)));

  CppAD::cg::CodeHandler<scalar_t> handler;
  CppAD::vector<ad_base_t> variables(static_cast<size_t>(numInputs));
  handler.makeVariables(variables);
  EXPECT_NO_THROW(fun.Forward(/*q=*/0, variables));
}

/*
 * What the correction changes on the shipped DRC Atlas, the robot being tuned: the centroidal controller commands
 * computeJointTorques at zero joint acceleration, so the change is exactly the joint-base coupling M_jb a_b. The
 * magnitudes are logged for the record; the assertions are the properties.
 */
TEST_F(JointTorqueInverseDynamicsTest, theCorrectionOnTheShippedAtlasStanceIsTheBaseCouplingAlone) {
  const pinocchio::Model& model = atlas_->model();
  vector_t initialState = vector_t::Zero(12 + atlas_->numJoints());
  loadData::loadEigenMatrix(atlas_->taskFile, "initialState", initialState);
  Sample standing;
  standing.q = initialState.tail(model.nq);
  standing.v = vector_t::Zero(model.nv);
  standing.qddJoints = vector_t::Zero(atlas_->numJoints());
  const scalar_t weight = kGravity * pinocchio::computeTotalMass(model);
  vector6_t halfWeight = vector6_t::Zero();
  halfWeight(2) = 0.5 * weight;
  standing.wrenches = {halfWeight, halfWeight};

  // Single support on the left foot: the weight straight up, and the weight along the line from the sole to the center of
  // mass as the linear inverted pendulum carries it (no moment about the center of mass).
  Sample singleSupport = standing;
  singleSupport.wrenches = {vector6_t::Zero(), vector6_t::Zero()};
  singleSupport.wrenches[0](2) = weight;
  Sample pendulum = singleSupport;
  pinocchio::Data data(model);
  const vector3_t com = pinocchio::centerOfMass(model, data, standing.q);
  pinocchio::framesForwardKinematics(model, data, standing.q);
  const vector3_t soleToCom = com - data.oMf[model.getFrameId(kSoleFrames[0])].translation();
  pendulum.wrenches[0].head<3>() = weight * soleToCom / soleToCom.z();

  const std::array<std::pair<const char*, const Sample*>, 3> cases = {
      {{"double support, half the weight on each sole", &standing},
       {"single support, the weight straight up", &singleSupport},
       {"single support, the weight toward the center of mass", &pendulum}}};
  for (size_t c = 0; c < cases.size(); ++c) {
    const char* description = cases[c].first;
    SCOPED_TRACE(description);
    const Sample& sample = *cases[c].second;
    PinocchioInterface pinocchioInterface = *atlas_->pinocchioInterface;
    const vector_t corrected = computeJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, pinocchioInterface);
    const RneaReference reference = floatingBaseRnea(model, sample);
    EXPECT_LT(maxAbs(corrected - reference.jointTorques), 1e-9 * std::max(1.0, maxAbs(reference.jointTorques)));

    // At zero joint acceleration the correction is the base coupling: the base-held torques plus M_jb a_b.
    PinocchioInterface baseHeldInterface = *atlas_->pinocchioInterface;
    const vector_t baseHeld =
        computeBaseHeldJointTorques<scalar_t>(sample.q, sample.v, sample.qddJoints, sample.wrenches, baseHeldInterface);
    const matrix_t& M = pinocchioInterface.getData().M;
    const vector_t coupling = M.bottomLeftCorner(atlas_->numJoints(), 6) * reference.baseAcceleration;
    EXPECT_LT(maxAbs(corrected - baseHeld - coupling), 1e-9 * std::max(1.0, maxAbs(corrected)));

    const vector_t change = corrected - legacyJointTorques(model, sample);
    Eigen::Index joint = 0;
    const scalar_t largest = change.cwiseAbs().maxCoeff(&joint);
    LOG(INFO) << "[DRC Atlas, " << description << "] a_b = " << reference.baseAcceleration.transpose() << "; largest torque change "
              << largest << " Nm at " << model.names[static_cast<size_t>(joint) + 2] << ", RMS over the joints "
              << change.norm() / std::sqrt(static_cast<scalar_t>(change.size())) << " Nm, largest torque " << maxAbs(corrected) << " Nm";
    RecordProperty(absl::StrCat("largest_change_nm_", c), absl::StrCat(largest));
  }
}

}  // namespace
}  // namespace ocs2::humanoid
