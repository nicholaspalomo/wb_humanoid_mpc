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

#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/RealtimeRigidBodyDynamics.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "robot_core/ResourcePaths.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * RealtimeRigidBodyDynamics runs Pinocchio on a twin of the MPC model with a
 * FreeFlyer root in place of its Composite(Translation, Spherical) one, in the
 * MPC model's coordinates. Every result is checked against the MPC model
 * itself, which shares nothing with the twin but the URDF: the frame
 * placements, the generalized gravity, the centroidal momentum of
 * computeCentroidalMap(), the joint torques of computeJointTorques() and
 * computeBaseHeldJointTorques(), and, independently of both, Pinocchio's RNEA
 * with the contact wrenches as external forces - at the reported base
 * acceleration it needs no base wrench and gives the joint torques. Random
 * configurations (attitude included), velocities, joint accelerations and
 * wrenches, on the whole-body and the centroidal G1, the same object reused so
 * that a workspace left over from one call cannot reach the next. And none of
 * its functions allocates, which is what it is for.
 */

namespace ocs2::humanoid {
namespace {

/** An MPC package of the G1: its task file and the robot's URDF, by their paths
 * in the repository. */
struct MpcPackage {
  const char* absl_nonnull name;
  const char* absl_nonnull taskFile;
  const char* absl_nonnull urdfFile;
};

void PrintTo(const MpcPackage& package, std::ostream* absl_nonnull os) {
  *os << package.name;
}

constexpr MpcPackage kPackages[] = {
    {.name = "g1_wb_mpc",
     .taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"},
    {.name = "g1_centroidal_mpc",
     .taskFile =
         "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"},
};

/** The external forces of RNEA, one per joint, in the joint's local frame. */
using ForceVector = std::vector<pinocchio::Force>;

/** The external forces of RNEA for world-frame wrenches at the origins of the
 * sole frames `frames`, after kinematics. */
ForceVector externalForces(const pinocchio::Model& model,
                           const pinocchio::Data& data,
                           const feet_array_t<pinocchio::FrameIndex>& frames,
                           const feet_array_t<vector6_t>& wrenches) {
  ForceVector forces(model.njoints, pinocchio::Force::Zero());
  for (size_t i = 0; i < kNumContacts; ++i) {
    const pinocchio::Frame& frame = model.frames[frames[i]];
    const matrix3_t rotation = data.oMf[frames[i]].rotation();
    // The wrench in the frame's own axes, about its origin, then in the parent
    // joint's frame.
    const pinocchio::Force local(rotation.transpose() * wrenches[i].head<3>(),
                                 rotation.transpose() * wrenches[i].tail<3>());
    forces[frame.parentJoint] += frame.placement.act(local);
  }
  return forces;
}

/** A random state of the MPC model: q with a unit quaternion, v, joint
 * accelerations and contact wrenches. */
struct Sample {
  vector_t q;
  vector_t v;
  vector_t qddJoints;
  feet_array_t<vector6_t> wrenches;
};

Sample randomSample(const pinocchio::Model& model, std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  Sample sample;
  sample.q.resize(model.nq);
  for (Eigen::Index i = 0; i < sample.q.size(); ++i)
    sample.q(i) = 0.5 * normal(generator);
  sample.q.segment<4>(3).normalize();  // (x, y, z, w)
  sample.v.resize(model.nv);
  for (Eigen::Index i = 0; i < sample.v.size(); ++i)
    sample.v(i) = normal(generator);
  sample.qddJoints.resize(model.nv - 6);
  for (Eigen::Index i = 0; i < sample.qddJoints.size(); ++i)
    sample.qddJoints(i) = 3.0 * normal(generator);
  for (vector6_t& wrench : sample.wrenches) {
    for (Eigen::Index i = 0; i < wrench.size(); ++i)
      wrench(i) = 100.0 * normal(generator);
  }
  return sample;
}

class RealtimeRigidBodyDynamicsTest
    : public ::testing::TestWithParam<MpcPackage> {
 protected:
  void SetUp() override {
    const std::string taskFile =
        robot::resolveResourcePath(GetParam().taskFile).value();
    const std::string urdfFile =
        robot::resolveResourcePath(GetParam().urdfFile).value();
    const ModelSettings modelSettings =
        ModelSettings::Create(taskFile, urdfFile, "realtime_dynamics_",
                              /*verbose=*/false)
            .value();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(
        loadCustomPinocchioInterface(taskFile, urdfFile, modelSettings)
            .value());
    const pinocchio::Model& model = pinocchioInterface_->getModel();
    for (size_t i = 0; i < kNumContacts; ++i)
      frames_[i] = model.getFrameId(modelSettings.contactNames[i]);
    absl::StatusOr<std::unique_ptr<RealtimeRigidBodyDynamics>> dynamics =
        RealtimeRigidBodyDynamics::Create(model, frames_);
    ASSERT_TRUE(dynamics.ok()) << dynamics.status();
    dynamics_ = *std::move(dynamics);
  }

  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  feet_array_t<pinocchio::FrameIndex> frames_{};
  std::unique_ptr<RealtimeRigidBodyDynamics> dynamics_;
};

}  // namespace

TEST_P(RealtimeRigidBodyDynamicsTest,
       RefusesAModelWhoseRootIsNotTheMpcModelsComposite) {
  // The twin's own model has a FreeFlyer root.
  const absl::StatusOr<std::unique_ptr<RealtimeRigidBodyDynamics>> ofTwin =
      RealtimeRigidBodyDynamics::Create(dynamics_->getModel(), frames_);
  EXPECT_EQ(ofTwin.status().code(), absl::StatusCode::kInvalidArgument)
      << ofTwin.status();
  feet_array_t<pinocchio::FrameIndex> missing = frames_;
  missing[1] = pinocchioInterface_->getModel().frames.size();
  EXPECT_EQ(RealtimeRigidBodyDynamics::Create(pinocchioInterface_->getModel(),
                                              missing)
                .status()
                .code(),
            absl::StatusCode::kNotFound);
}

TEST_P(RealtimeRigidBodyDynamicsTest,
       TheTwinPlacesEveryFrameWhereTheMpcModelDoes) {
  const pinocchio::Model& model = pinocchioInterface_->getModel();
  const pinocchio::Model& twin = dynamics_->getModel();
  ASSERT_EQ(twin.nq, model.nq);
  ASSERT_EQ(twin.nv, model.nv);
  ASSERT_EQ(twin.nframes, model.nframes);
  pinocchio::Data modelData(model);
  pinocchio::Data twinData(twin);
  std::mt19937 generator(/*sd=*/43);
  for (int trial = 0; trial < 20; ++trial) {
    const Sample sample = randomSample(model, generator);
    pinocchio::framesForwardKinematics(model, modelData, sample.q);
    pinocchio::framesForwardKinematics(twin, twinData, sample.q);
    for (pinocchio::FrameIndex frame = 0; frame < model.frames.size();
         ++frame) {
      ASSERT_EQ(twin.frames[frame].name, model.frames[frame].name);
      EXPECT_TRUE(twinData.oMf[frame].isApprox(modelData.oMf[frame], 1.0e-12))
          << model.frames[frame].name;
    }
    const feet_array_t<matrix3_t>& rotations =
        dynamics_->contactFrameRotations(sample.q);
    for (size_t i = 0; i < kNumContacts; ++i) {
      EXPECT_LT((rotations[i] - modelData.oMf[frames_[i]].rotation())
                    .cwiseAbs()
                    .maxCoeff(),
                1.0e-12)
          << "contact " << i;
    }
  }
}

TEST_P(RealtimeRigidBodyDynamicsTest, TheGravityAndTheMomentumAreTheMpcModels) {
  PinocchioInterface& mpc = *pinocchioInterface_;
  const pinocchio::Model& model = mpc.getModel();
  std::mt19937 generator(/*sd=*/47);
  for (int trial = 0; trial < 20; ++trial) {
    const Sample sample = randomSample(model, generator);
    const Eigen::Index numJoints = model.nv - 6;
    pinocchio::nonLinearEffects(model, mpc.getData(), sample.q,
                                vector_t::Zero(model.nv));
    const vector_t gravity = mpc.getData().nle.tail(numJoints);
    EXPECT_LT(
        (dynamics_->gravityTorques(sample.q) - gravity).cwiseAbs().maxCoeff(),
        1.0e-9 * (1.0 + gravity.cwiseAbs().maxCoeff()));

    pinocchio::computeCentroidalMap(model, mpc.getData(), sample.q);
    const vector6_t momentum = mpc.getData().Ag * sample.v;
    EXPECT_LT((dynamics_->centroidalMomentum(sample.q, sample.v) - momentum)
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-10 * (1.0 + momentum.cwiseAbs().maxCoeff()));
  }
}

TEST_P(RealtimeRigidBodyDynamicsTest,
       TheFreeBaseTorquesNeedNoBaseWrenchAndAreComputeJointTorques) {
  PinocchioInterface& mpc = *pinocchioInterface_;
  const pinocchio::Model& model = mpc.getModel();
  pinocchio::Data reference(model);
  std::mt19937 generator(/*sd=*/31);
  for (int trial = 0; trial < 50; ++trial) {
    const Sample sample = randomSample(model, generator);
    const vector_t torques = dynamics_->freeBaseJointTorques(
        sample.q, sample.v, sample.qddJoints, sample.wrenches);
    vector_t a(model.nv);
    a << dynamics_->getBaseAcceleration(), sample.qddJoints;

    pinocchio::forwardKinematics(model, reference, sample.q);
    pinocchio::updateFramePlacements(model, reference);
    const ForceVector forces =
        externalForces(model, reference, frames_, sample.wrenches);
    const vector_t rnea =
        pinocchio::rnea(model, reference, sample.q, sample.v, a, forces);
    const scalar_t scale = 1.0 + rnea.cwiseAbs().maxCoeff();
    EXPECT_LT(rnea.head<6>().cwiseAbs().maxCoeff(), 1.0e-9 * scale)
        << "trial " << trial << ": the base needs a wrench";
    EXPECT_LT(
        (rnea.tail(sample.qddJoints.size()) - torques).cwiseAbs().maxCoeff(),
        1.0e-9 * scale)
        << "trial " << trial;

    const vector_t helper = computeJointTorques<scalar_t>(
        sample.q, sample.v, sample.qddJoints, sample.wrenches, frames_, mpc);
    EXPECT_LT((helper - torques).cwiseAbs().maxCoeff(), 1.0e-9 * scale)
        << "trial " << trial;
  }
}

TEST_P(RealtimeRigidBodyDynamicsTest,
       TheHeldBaseTorquesAreRneaAtZeroBaseAcceleration) {
  PinocchioInterface& mpc = *pinocchioInterface_;
  const pinocchio::Model& model = mpc.getModel();
  pinocchio::Data reference(model);
  std::mt19937 generator(/*sd=*/53);
  for (int trial = 0; trial < 50; ++trial) {
    const Sample sample = randomSample(model, generator);
    const vector_t torques = dynamics_->heldBaseJointTorques(
        sample.q, sample.v, sample.qddJoints, sample.wrenches);
    vector_t a = vector_t::Zero(model.nv);
    a.tail(sample.qddJoints.size()) = sample.qddJoints;

    pinocchio::forwardKinematics(model, reference, sample.q);
    pinocchio::updateFramePlacements(model, reference);
    const ForceVector forces =
        externalForces(model, reference, frames_, sample.wrenches);
    const vector_t rnea =
        pinocchio::rnea(model, reference, sample.q, sample.v, a, forces);
    const scalar_t scale = 1.0 + rnea.cwiseAbs().maxCoeff();
    EXPECT_LT(
        (rnea.tail(sample.qddJoints.size()) - torques).cwiseAbs().maxCoeff(),
        1.0e-9 * scale)
        << "trial " << trial;

    const vector_t helper = computeBaseHeldJointTorques<scalar_t>(
        sample.q, sample.v, sample.qddJoints, sample.wrenches, frames_, mpc);
    EXPECT_LT((helper - torques).cwiseAbs().maxCoeff(), 1.0e-9 * scale)
        << "trial " << trial;
  }
}

TEST_P(RealtimeRigidBodyDynamicsTest, NoFunctionAllocates) {
  std::mt19937 generator(/*sd=*/59);
  const Sample sample =
      randomSample(pinocchioInterface_->getModel(), generator);
  for (int pass = 0; pass < 2;
       ++pass) {  // the first pass sizes what sizes itself on first use
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    dynamics_->gravityTorques(sample.q);
    dynamics_->freeBaseJointTorques(sample.q, sample.v, sample.qddJoints,
                                    sample.wrenches);
    dynamics_->heldBaseJointTorques(sample.q, sample.v, sample.qddJoints,
                                    sample.wrenches);
    dynamics_->centroidalMomentum(sample.q, sample.v);
    dynamics_->contactFrameRotations(sample.q);
    const size_t allocations =
        robot::realtime::heapAllocationCountOnThisThread() - before;
    if (pass == 1) EXPECT_EQ(allocations, 0u);
  }

  // Positive control: the same passes on the MPC model's composite root do
  // allocate, which is why the twin exists.
  PinocchioInterface& mpc = *pinocchioInterface_;
  const pinocchio::Model& model = mpc.getModel();
  pinocchio::crba(model, mpc.getData(), sample.q);
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  pinocchio::crba(model, mpc.getData(), sample.q);
  pinocchio::nonLinearEffects(model, mpc.getData(), sample.q, sample.v);
  pinocchio::computeCentroidalMap(model, mpc.getData(), sample.q);
  const size_t compositeAllocations =
      robot::realtime::heapAllocationCountOnThisThread() - before;
  RecordProperty("pinocchio_allocations_on_the_composite_root",
                 static_cast<int>(compositeAllocations));
  EXPECT_GT(compositeAllocations, 0u);
}

INSTANTIATE_TEST_SUITE_P(G1,
                         RealtimeRigidBodyDynamicsTest,
                         ::testing::ValuesIn(kPackages));

}  // namespace ocs2::humanoid
