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

#include "pinocchio/fwd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <string>

#include "Eigen/LU"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/multibody/joint/fwd.hpp"

#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/orientation/EulerBoundary.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "legged_nmpc/legged_common_mpc/test/equivalence/RootJointModelPair.h"
#include "robot_core/ResourcePaths.h"

/*
 * The MRT joint controllers' inverse dynamics on the quaternion root against
 * the Euler root's, written here again from the equations of motion of the
 * Composite(Translation, SphericalZYX) twin of the MPC's own Pinocchio model
 * (the model loadCustomPinocchioInterface() builds, with its contact frames):
 * the feedforward torques both MRTs send, with a world-frame wrench at every
 * contact of the configuration (a force alone at a 3-DoF contact).
 *
 *   free base   computeJointTorques(): the base rows of M a + n = J' W + S' tau
 * balance the base acceleration, the joint rows give tau_j. Coordinate-free
 * physics: tau_e = tau_t at the same motion (q_e = Phi^-1(q), v_e = J_v^-1 v),
 * the same joint accelerations and the same world-frame wrenches. base held
 * computeBaseHeldJointTorques(): the base acceleration is zero - Theta_ddot = 0
 * on the Euler root, the body rate's dw_B/dt = 0 on the quaternion root, which
 * differ by dw_B/dt = (dT_B/dt) dTheta = C(Theta, dTheta) dTheta when
 * Theta_ddot = 0 - so tau_e - tau_t = M_jw (dT_B/dt) dTheta, M_jw the joint
 * rows of the quaternion root's mass matrix in its angular base columns (design
 * section 4.4's base-held row).
 *
 * Tolerance: design section 4.4's 1e-9 relative (floor 1) for both: the solve
 * of the free base's 6 x 6 block with a condition number of at most 1e3, and
 * the (T1) round-off of the two trees (RootJointModelPair.h).
 */

namespace ocs2::legged {
namespace {

constexpr int kSamples = 40;
constexpr scalar_t kJointTorqueTolerance = 1.0e-9;

/** `model` with its root joint replaced by Composite(Translation,
 * SphericalZYX): every other joint, inertia and frame kept. */
pinocchio::Model eulerRootTwin(const pinocchio::Model& model) {
  pinocchio::JointModelComposite root(2);
  root.addJoint(pinocchio::JointModelTranslation());
  // NOLINTNEXTLINE(spherical-zyx): the Euler root of the oracle the quaternion root's torques are compared with
  root.addJoint(pinocchio::JointModelSphericalZYX());
  pinocchio::Model twin;
  twin.name = model.name;
  twin.gravity = model.gravity;
  for (pinocchio::JointIndex joint = 1;
       joint < static_cast<pinocchio::JointIndex>(model.njoints); ++joint) {
    if (joint == 1) {
      twin.addJoint(/*parent=*/0, pinocchio::JointModel(root),
                    model.jointPlacements[joint], model.names[joint]);
    } else {
      twin.addJoint(model.parents[joint], model.joints[joint],
                    model.jointPlacements[joint], model.names[joint]);
    }
    twin.inertias[joint] = model.inertias[joint];
  }
  for (pinocchio::FrameIndex frame = 1;
       frame < static_cast<pinocchio::FrameIndex>(model.nframes); ++frame) {
    twin.addFrame(model.frames[frame], /*append_inertia=*/false);
  }
  return twin;
}

/** The terms of M a + n = J' W + S' tau of the Euler root at (q_e, v_e).
 * Passive data. */
struct EquationsOfMotion {
  matrix_t massMatrix;
  vector_t nonlinearEffects;
  /** J' W of the world-frame wrenches at the contact frames. */
  vector_t externalGeneralizedForces;
};

EquationsOfMotion eulerEquationsOf(
    const pinocchio::Model& model,
    const vector_t& q,
    const vector_t& v,
    const ContactArray<pinocchio::FrameIndex>& contactFrames,
    const ContactArray<vector6_t>& wrenchContactsEwrtWorld) {
  pinocchio::Data data(model);
  EquationsOfMotion equations;
  equations.massMatrix = symmetricFromUpper(pinocchio::crba(model, data, q));
  equations.nonlinearEffects = pinocchio::nonLinearEffects(model, data, q, v);
  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  equations.externalGeneralizedForces = vector_t::Zero(model.nv);
  for (size_t contact = 0; contact < contactFrames.size(); ++contact) {
    matrix_t jacobian = matrix_t::Zero(6, model.nv);
    pinocchio::getFrameJacobian(model, data, contactFrames[contact],
                                pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
    equations.externalGeneralizedForces +=
        jacobian.transpose() * wrenchContactsEwrtWorld[contact];
  }
  return equations;
}

class MrtTorqueChartEquivalence
    : public ::testing::TestWithParam<EquivalenceRobot> {};

TEST_P(MrtTorqueChartEquivalence,
       TheFeedforwardTorquesAreTheEulerRootsAndTheBaseHoldDiffersByTheRateMap) {
  const EquivalenceRobot& robot = GetParam();
  const absl::StatusOr<std::string> taskFile =
      robot::resolveResourcePath(robot.taskFile);
  const absl::StatusOr<std::string> urdfFile =
      robot::resolveResourcePath(robot.urdfFile);
  ASSERT_TRUE(taskFile.ok() && urdfFile.ok())
      << taskFile.status() << urdfFile.status();
  const absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
      *taskFile, *urdfFile, "mrt_torque_chart", /*verbose=*/false);
  ASSERT_TRUE(settings.ok()) << settings.status();
  absl::StatusOr<PinocchioInterface> interface =
      loadCustomPinocchioInterface(*taskFile, *urdfFile, *settings);
  ASSERT_TRUE(interface.ok()) << interface.status();
  const pinocchio::Model model = interface->getModel();
  const size_t numContacts = settings->contactNames.size();
  ContactArray<pinocchio::FrameIndex> contactFrames(numContacts);
  for (size_t contact = 0; contact < numContacts; ++contact) {
    const std::string& name = settings->contactNames[contact];
    ASSERT_TRUE(model.existFrame(name)) << name;
    contactFrames[contact] = model.getFrameId(name);
  }
  const pinocchio::Model eulerModel = eulerRootTwin(model);
  ASSERT_EQ(eulerModel.nv, model.nv);
  const Eigen::Index joints = model.nv - 6;
  std::mt19937 generator(/*sd=*/1011);
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  scalar_t largestHoldDifference = 0.0;
  for (int sample = 0; sample < kSamples; ++sample) {
    SCOPED_TRACE(sample);
    const vector3_t angles(std::numbers::pi * unit(generator),
                           1.2 * unit(generator),
                           std::numbers::pi * unit(generator));
    const vector3_t eulerRates =
        1.5 * vector3_t(unit(generator), unit(generator), unit(generator));
    vector_t jointPositions(joints);
    for (Eigen::Index j = 0; j < joints; ++j) {
      // NOLINTBEGIN(frame-naming): Pinocchio's joint limits (joint space)
      const scalar_t low = model.lowerPositionLimit(7 + j);
      const scalar_t high = model.upperPositionLimit(7 + j);
      // NOLINTEND(frame-naming)
      // Within the middle 40 % of the joint's range, or within 0.5 rad of zero
      // for a joint without finite limits.
      jointPositions(j) =
          std::isfinite(low) && std::isfinite(high)
              ? 0.5 * (low + high) + 0.2 * (high - low) * unit(generator)
              : 0.5 * unit(generator);
    }
    const vector3_t posBaseRtWorldEwrtWorld(unit(generator), unit(generator),
                                            0.8 + 0.1 * unit(generator));
    const vector3_t linvelBaseRtWorldEwrtWorld =
        0.5 * vector3_t(unit(generator), unit(generator), unit(generator));
    const vector_t jointVelocities =
        uniformVector(joints, /*halfWidth=*/1.0, generator);
    const vector_t jointAccelerations =
        uniformVector(joints, /*halfWidth=*/5.0, generator);
    ContactArray<vector6_t> wrenchContactsEwrtWorld(numContacts);
    for (size_t contact = 0; contact < numContacts; ++contact) {
      vector6_t& wrenchContactEwrtWorld = wrenchContactsEwrtWorld[contact];
      wrenchContactEwrtWorld << 50.0 * unit(generator), 50.0 * unit(generator),
          200.0 + 100.0 * unit(generator), 10.0 * unit(generator),
          10.0 * unit(generator), 5.0 * unit(generator);
      // A point contact carries a force and no moment.
      if (settings->isPointContact(contact)) {
        wrenchContactEwrtWorld.tail<3>().setZero();
      }
    }
    const matrix3_t rateMap = eulerZyxRateToBodyRateMatrix(angles);

    // The quaternion root's motion and the Euler root's, the same physical
    // motion.
    vector_t q(model.nq);
    q << posBaseRtWorldEwrtWorld, quaternionFromEulerZyx(angles),
        jointPositions;
    vector_t v(model.nv);
    v << linvelBaseRtWorldEwrtWorld, rateMap * eulerRates, jointVelocities;
    vector_t eulerQ(eulerModel.nq);
    eulerQ << posBaseRtWorldEwrtWorld, angles, jointPositions;
    vector_t eulerV(eulerModel.nv);
    eulerV << linvelBaseRtWorldEwrtWorld, eulerRates, jointVelocities;

    const EquationsOfMotion euler = eulerEquationsOf(
        eulerModel, eulerQ, eulerV, contactFrames, wrenchContactsEwrtWorld);
    const matrix_t& m = euler.massMatrix;
    const vector_t residual =
        euler.externalGeneralizedForces - euler.nonlinearEffects;
    // Free base: M_bb a_b = (J'W - n)_b - M_bj qdd_j, then tau_j = M_jb a_b +
    // M_jj qdd_j - (J'W - n)_j.
    const vector_t baseGeneralizedAcceleration =
        m.topLeftCorner(6, 6).partialPivLu().solve(residual.head<6>() -
                                                   m.topRightCorner(6, joints) *
                                                       jointAccelerations);
    const vector_t eulerModelJointTorques =
        m.bottomLeftCorner(joints, 6) * baseGeneralizedAcceleration +
        m.bottomRightCorner(joints, joints) * jointAccelerations -
        residual.tail(joints);
    const vector_t eulerModelHeldJointTorques =
        m.bottomRightCorner(joints, joints) * jointAccelerations -
        residual.tail(joints);

    const vector_t jointTorques = computeJointTorques<scalar_t>(
        q, v, jointAccelerations, wrenchContactsEwrtWorld, contactFrames,
        *interface);
    const vector_t heldJointTorques = computeBaseHeldJointTorques<scalar_t>(
        q, v, jointAccelerations, wrenchContactsEwrtWorld, contactFrames,
        *interface);
    ASSERT_TRUE(isCloseRelative(jointTorques, eulerModelJointTorques,
                                kJointTorqueTolerance))
        << "the free base's torques";

    // Base held: tau_e - tau_t = M_jw (dT_B/dt) dTheta, with (dT_B/dt) dTheta =
    // sum_k (dT_B / dTheta_k) dTheta_k dTheta by central differences of T_B
    // (step 1e-6: truncation 1e-12 times the rates squared, round-off 1e-10).
    constexpr scalar_t kStep = 1.0e-6;
    vector3_t rateMapRate = vector3_t::Zero();
    for (Eigen::Index k = 0; k < 3; ++k) {
      vector3_t up = angles;
      vector3_t down = angles;
      up(k) += kStep;
      down(k) -= kStep;
      rateMapRate += eulerRates(k) *
                     (eulerZyxRateToBodyRateMatrix(up) -
                      eulerZyxRateToBodyRateMatrix(down)) *
                     eulerRates / (2.0 * kStep);
    }
    pinocchio::Data data(model);
    const matrix_t quaternionMassMatrix =
        symmetricFromUpper(pinocchio::crba(model, data, q));
    const vector_t expectedDifference =
        quaternionMassMatrix.block(6, 3, joints, 3) * rateMapRate;
    ASSERT_TRUE(isCloseAbsolute(
        eulerModelHeldJointTorques - heldJointTorques, expectedDifference,
        kJointTorqueTolerance *
                std::max(1.0,
                         eulerModelHeldJointTorques.cwiseAbs().maxCoeff()) +
            1.0e-8))
        << "the base hold: Theta_ddot = 0 against dw_B/dt = 0";
    largestHoldDifference = std::max(largestHoldDifference,
                                     expectedDifference.cwiseAbs().maxCoeff());
  }
  // The two holds do differ: the rate map's rate is not a round-off effect.
  EXPECT_GT(largestHoldDifference, 1.0e-3);
  RecordProperty("largest_base_hold_difference",
                 absl::StrCat(largestHoldDifference));
}

INSTANTIATE_TEST_SUITE_P(Shipped,
                         MrtTorqueChartEquivalence,
                         ::testing::ValuesIn(equivalenceRobots()),
                         equivalenceRobotName);

}  // namespace
}  // namespace ocs2::legged
