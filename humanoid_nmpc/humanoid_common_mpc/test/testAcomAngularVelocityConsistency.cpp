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
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/joint/joint-free-flyer.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "urdf_parser/urdf_parser.h"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

/** The floating base: the Pinocchio model's first joint after the universe,
 * whatever joints it is composed of. */
constexpr pinocchio::JointIndex kRootJoint = 1;
/** How many random configurations the statistical assertions are averaged over.
 */
constexpr int kNumSamples = 200;
/**
 * Half-width of the range a joint without position limits is sampled over.
 * dataset_generator.py samples such a joint over the same range, so both sides
 * grade the network on the distribution it was fit to. No shipped robot has
 * one.
 */
// LINT.IfChange(acom_unlimited_joint_range)
constexpr scalar_t kUnlimitedJointHalfRange = M_PI;
// LINT.ThenChange(//humanoid_learning/acom/dataset_generator.py:unlimited_joint_range)

/**
 * Acceptance numbers of one SHIPPED weight header, in units of the norm of the
 * target. See the class comment below for what the two quantities are.
 */
struct AcceptanceNumbers {
  scalar_t meanRelativeError;
  scalar_t worstRelativeError;
  scalar_t meanRelativeRateError;
  scalar_t worstRelativeRateError;
};

/** One robot with a compiled-in aCOM network, and what its network has to
 * achieve. */
struct AcomRobotCase {
  /// model_settings.robot_name, which is also the name the network is
  /// registered under.
  const char* absl_nonnull robotName;
  /// Robot config package, relative to the repository root, holding
  /// config/mpc/task.textproto (and contact_planning.textproto where the robot
  /// has one).
  const char* absl_nonnull configPackage;
  /// The robot's URDF, relative to the repository root.
  const char* absl_nonnull urdfFile;
  /**
   * Whether the network is good enough to run in closed loop. A network that is
   * not must stay switched off in the robot's shipped configuration, which
   * theUnvalidatedNetworksAreSwitchedOffInTheShippedConfiguration enforces.
   */
  bool validatedForClosedLoop;
  /// What the shipped header measured when its bounds were set, reported next
  /// to the fresh measurement.
  AcceptanceNumbers measured;
  /// The acceptance bounds the header must stay under.
  AcceptanceNumbers bounds;
};

std::ostream& operator<<(std::ostream& stream, const AcomRobotCase& robotCase) {
  return stream << robotCase.robotName;
}

/**
 * Every robot with a network in AngularCenterOfMass's registry, and its
 * acceptance bounds.
 *
 * The bounds are PER ROBOT and MEASURED, and they do not all mean "good".
 *
 * All three headers were trained with the composed objective J_l(Delta_theta)
 * J_Delta_theta ~ Abar_omega,j by `train_main --install_header` on 80 000
 * samples over 300 epochs, and record that recipe in their banners.
 *
 *  - atlas is the reference, and the only network validated for closed loop.
 * Its bounds sit about 40 % above its measurement on purpose: retrains on more
 * data or another seed move the mean by about 2 %, so bounds pinned to the last
 * measurement would fail on a reseed rather than on a regression. They are set
 * where a MEANINGFUL regression lives - a wrong architecture, a bad export, a
 * network trained for another robot or on the uncomposed objective - and well
 * below both the constant-Jacobian baseline (0.378) and the same network fed
 * the wrong joint order (1.37).
 *
 *  - g1 and engineai_sa01 do NOT meet atlas's bounds, and are NOT VALIDATED: on
 * average their rate map is 34 % and 44 % off the target, against atlas's 16 %,
 * and at some configurations of both robots the network is further from the
 *    target than no network at all (worst relative error above 1). Their bounds
 * are their measurements plus about 20 %: loose enough that a reseeded retrain
 * of the same recipe still passes, tight enough that a permuted, transposed or
 *    wrong-robot header fails (the positive control
 * theBoundsRejectANetworkFedTheWrongJointOrder shows it does), and below each
 * robot's constant-Jacobian baseline. These bounds pin "no worse than what
 * ships", NOT "good enough to use"; a better network is how they come down, and
 * meeting atlas's bounds is what validating one would take.
 *
 * "measured" is what this file's samples give on the shipped header (mean and
 * worst relative Frobenius error of J_l(Delta_theta) J_Delta_theta, then mean
 * and worst relative rate error of the composed attitude xi (x)
 * Exp(Delta_theta) at a tilted base); the test logs the fresh numbers next to
 * it.
 */
// LINT.IfChange(acom_acceptance_robots)
const AcomRobotCase kAcomRobotCases[] = {
    {.robotName = "atlas",
     .configPackage = "robot_models/drc_atlas/drc_atlas_centroidal_mpc",
     .urdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
     /*validatedForClosedLoop=*/.validatedForClosedLoop = true,
     /*measured=*/.measured = {.meanRelativeError = 0.158,
                               .worstRelativeError = 0.363,
                               .meanRelativeRateError = 0.031,
                               .worstRelativeRateError = 0.130},
     /*bounds=*/.bounds = {.meanRelativeError = 0.22,
                           .worstRelativeError = 0.45,
                           .meanRelativeRateError = 0.05,
                           .worstRelativeRateError = 0.20}},
    {.robotName = "engineai_sa01",
     .configPackage = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
     .urdfFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
                 "zq_sa01.urdf",
     /*validatedForClosedLoop=*/.validatedForClosedLoop = false,
     /*measured=*/.measured = {.meanRelativeError = 0.445,
                               .worstRelativeError = 1.70,
                               .meanRelativeRateError = 0.105,
                               .worstRelativeRateError = 0.642},
     /*bounds=*/.bounds = {.meanRelativeError = 0.53,
                           .worstRelativeError = 2.05,
                           .meanRelativeRateError = 0.13,
                           .worstRelativeRateError = 0.77}},
    {.robotName = "g1",
     .configPackage = "robot_models/unitree_g1/g1_centroidal_mpc",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     /*validatedForClosedLoop=*/.validatedForClosedLoop = false,
     /*measured=*/.measured = {.meanRelativeError = 0.342,
                               .worstRelativeError = 1.14,
                               .meanRelativeRateError = 0.059,
                               .worstRelativeRateError = 0.225},
     /*bounds=*/.bounds = {.meanRelativeError = 0.41,
                           .worstRelativeError = 1.37,
                           .meanRelativeRateError = 0.07,
                           .worstRelativeRateError = 0.27}},
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/acom/AngularCenterOfMass.cpp:acom_robot_dispatch, //humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:acom_acceptance_data, //humanoid_learning/acom/README.md:acom_acceptance_numbers, //robot_models/drc_atlas/README.md:acom_status, //robot_models/unitree_g1/README.md:acom_status, //robot_models/engineai_sa01/README.md:acom_status)
// clang-format on

/** The robot's task file, from the test's runfiles. */
std::string taskFileOf(const AcomRobotCase& robotCase) {
  return checkedValue(robot::resolveResourcePath(
      absl::StrCat(robotCase.configPackage, "/config/mpc/task.textproto")));
}

/** The robot's URDF, from the test's runfiles. */
std::string urdfFileOf(const AcomRobotCase& robotCase) {
  return checkedValue(robot::resolveResourcePath(robotCase.urdfFile));
}

/**
 * Does the aCOM actually do the one thing it exists to do?
 *
 * Every other test of this subsystem checks its STRUCTURE - shapes,
 * floating-base equivariance, the analytic Jacobian against finite differences,
 * the attitude's tangent Jacobian, the joint ordering recorded in the generated
 * header. All of those would pass just as happily with an untrained network, or
 * with weights trained for a different robot's mass distribution. None of them
 * looks at the quantity the network was fit to.
 *
 * The defining property (Chen, Nelson, Griffin, Posa and Pratt, IROS 2023) is
 * that the aCOM is an INTEGRABLE approximation of the whole-body orientation
 * whose angular velocity is the centroidal angular velocity:
 *
 *   omega_aCOM  ~=  omega_locked  =  I_G(q)^-1 L_G(q, qdot),
 *
 * with I_G the locked rotational inertia at the center of mass and L_G the
 * centroidal angular momentum, both in world axes. The ACoM attitude is xi_a =
 * xi (x) Exp(Delta_theta(q_j)) (AngularCenterOfMass::acomAttitude()), so its
 * world angular velocity is omega_W + R(xi) J_l(Delta_theta) J_Delta_theta
 * qdot_j, with omega_W the base's; omega_locked is omega_W + R(xi) Abar_omega,j
 * qdot_j with the connection in base axes. The base term is exact by
 * construction, which reduces the claim to the joint block alone, exactly and
 * not only to first order in |Delta_theta|:
 *
 *   J_l(Delta_theta) d(Delta_theta)/d(q_j)  ~=  Abar_omega,j(q)  =  I_G(q)^-1
 * A_omega,j(q),
 *
 * with J_l the left Jacobian of SO(3). The network's offsets are tens of
 * degrees (up to 80 on G1), where J_l is far from the identity, so the Jacobian
 * alone would be the wrong thing to grade: this composed map
 * (acomJointRateJacobian()) is what is graded, and it is exactly the Frobenius
 * objective train_acom.py minimizes, on the connection dataset_generator.py
 * samples in base axes at the identity base (a MuJoCo free joint, whose angular
 * velocity is the body angular velocity omega_B). So this file is the C++-side
 * acceptance test for the TRAINING, evaluated through the exported weight
 * header that the robot actually runs - not through the JAX parameters the
 * Python tests see - for every robot that has one. A retrained or re-exported
 * network that regressed, a transposed weight matrix, a row permutation applied
 * in the wrong direction, or a header exported for the wrong robot all show up
 * here and nowhere else.
 *
 * There is a second thing this file gets for free, and it is worth stating
 * because nothing else in the repository has it. The target here is recomputed
 * with Pinocchio's ccrba on the reduced MPC model, which is an INDEPENDENT
 * implementation of the one that produced the training data -
 * dataset_generator.py reads MuJoCo's subtree_angmom under unit joint
 * velocities, on a model built from a different file, with a different joint
 * ordering that a permutation has to reconcile. A frame convention applied in
 * the wrong direction, a permutation off by one, or a momentum read about the
 * wrong subtree would leave every Python test green (they check MuJoCo against
 * itself) and every structural C++ test green, and would show up here as an
 * error far larger than any fit residual.
 *
 * The rate tests compare angular velocities, in world axes, at a tilted base:
 * the base is driven through its root joint's own body velocity, whatever
 * coordinates the root joint has, so nothing here depends on how the MPC's
 * Pinocchio model parameterizes the base orientation.
 *
 * Configurations are drawn uniformly from the model's joint-limit box, which is
 * the box dataset_generator.py samples and the shipped headers were trained on.
 *
 * The relation is an approximation and cannot be otherwise: the connection
 * Abar_omega,j has non-zero curvature, so no exact integrable whole-body
 * orientation exists at all - that is the entire premise of the paper. The
 * thresholds are therefore acceptance bounds on the fit quality, not tolerances
 * on a numerical identity; they are stated relative to the magnitude of the
 * target so that a number means the same thing on every robot, and set per
 * robot as described at kAcomRobotCases.
 */
class AcomAngularVelocityConsistencyTest
    : public ::testing::TestWithParam<AcomRobotCase> {
 protected:
  void SetUp() override {
    const AcomRobotCase& robotCase = GetParam();
    const std::string taskFile = taskFileOf(robotCase);
    const std::string urdfFile = urdfFileOf(robotCase);

    // The REDUCED model the MPC builds, not the full URDF: the trained network
    // is indexed by the MPC's joints, and on the full model the joint count
    // would not even match.
    modelSettingsPtr_ =
        std::make_unique<ModelSettings>(checkedValue(ModelSettings::Create(
            taskFile, urdfFile, "testAcomAngularVelocityConsistency",
            /*verbose=*/false)));
    ASSERT_EQ(modelSettingsPtr_->robotName, robotCase.robotName)
        << "the case's task file belongs to another robot";
    pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(checkedValue(
        loadCustomPinocchioInterface(taskFile, urdfFile, *modelSettingsPtr_)));
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    numJoints_ = model.nv - model.joints[kRootJoint].nv();

    // The checked factory, as the MPC calls it: a joint list that differs from
    // the one the header records fails here.
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create(modelSettingsPtr_->robotName,
                                    modelSettingsPtr_->mpcModelJointNames);
    ASSERT_TRUE(acom.ok()) << acom.status();
    acomPtr_ = *std::move(acom);
    ASSERT_EQ(acomPtr_->getInputDim(), static_cast<size_t>(numJoints_))
        << "the exported weights are indexed by a different joint set than the "
           "MPC model";
  }

  /**
   * The joint block of the connection, Abar_omega,j = I_G^-1 A_omega,j, in
   * world axes, from Pinocchio.
   *
   * Only the joint columns of ccrba's centroidal map are read: a joint velocity
   * is the same number whatever coordinates the root joint has, while the base
   * columns are with respect to the root's own velocity coordinates (the Euler
   * rates of an Euler-angle root, the body angular velocity of a quaternion
   * one). The rate tests therefore drive the base through rootVelocityOf(),
   * never through those columns.
   */
  matrix_t connectionJointBlock(const vector_t& q) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    const vector_t zeroVelocity = vector_t::Zero(model.nv);
    pinocchio::ccrba(model, data, q, zeroVelocity);
    const matrix3_t lockedInertia = data.Ig.inertia().matrix();
    const matrix_t jointAngularMomentumMap =
        data.Ag.bottomRows(3).rightCols(numJoints_);
    return lockedInertia.ldlt().solve(jointAngularMomentumMap);
  }

  /** omega_locked = I_G^-1 L_G, the centroidal angular velocity, in world axes.
   */
  vector3_t lockedAngularVelocity(const vector_t& q, const vector_t& v) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    pinocchio::ccrba(model, data, q, v);
    const matrix3_t lockedInertia = data.Ig.inertia().matrix();
    return lockedInertia.ldlt().solve(vector3_t(data.hg.angular()));
  }

  /**
   * A configuration drawn uniformly from the model's joint limits, with the
   * base at the identity orientation, which is where dataset_generator.py
   * samples the connection. The joints are the last numJoints_ entries of q,
   * after the root joint's coordinates.
   */
  vector_t sampleConfiguration(std::mt19937& generator) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    vector_t q = pinocchio::neutral(model);
    q(2) = 0.9;  // a plausible base height; the CoM and the inertia do not
                 // depend on it, but a sane pose is easier to read
    for (Eigen::Index joint = 0; joint < numJoints_; ++joint) {
      const Eigen::Index index = model.nq - numJoints_ + joint;
      const scalar_t lower = std::isfinite(model.lowerPositionLimit(index))
                                 ? model.lowerPositionLimit(index)
                                 : -kUnlimitedJointHalfRange;
      const scalar_t upper = std::isfinite(model.upperPositionLimit(index))
                                 ? model.upperPositionLimit(index)
                                 : kUnlimitedJointHalfRange;
      std::uniform_real_distribution<scalar_t> distribution(lower, upper);
      q(index) = distribution(generator);
    }
    return q;
  }

  /** `q` with its base rotated: Pinocchio's integration of the root joint by
   * the velocity coordinates `rootAngular`. */
  vector_t withRotatedBase(const vector_t& q,
                           const vector3_t& rootAngular) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    vector_t step = vector_t::Zero(model.nv);
    step.segment<3>(3) = rootAngular;
    return pinocchio::integrate(model, q, step);
  }

  /** The base orientation xi of `q`, quaternion coefficients (x, y, z, w), from
   * the root joint's placement. */
  vector4_t baseOrientationOf(const vector_t& q) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    pinocchio::forwardKinematics(model, data, q);
    return Eigen::Quaternion<scalar_t>(data.oMi[kRootJoint].rotation())
        .coeffs();
  }

  /**
   * The velocity v at `q` that rotates the whole robot, its joints locked, with
   * the body angular velocity `angularVelocityLocal` about the base origin: the
   * root joint's velocity coordinates whose local spatial velocity is (0,
   * angularVelocityLocal), whatever coordinates the root joint has.
   */
  vector_t rootVelocityOf(const vector_t& q,
                          const vector3_t& angularVelocityLocal) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    pinocchio::computeJointJacobians(model, data, q);
    matrix_t jacobian = matrix_t::Zero(6, model.nv);
    pinocchio::getJointJacobian(model, data, kRootJoint, pinocchio::LOCAL,
                                jacobian);
    const Eigen::Index rootVelocityDim = model.joints[kRootJoint].nv();
    vector6_t twist = vector6_t::Zero();
    twist.tail<3>() = angularVelocityLocal;
    vector_t v = vector_t::Zero(model.nv);
    v.head(rootVelocityDim) =
        jacobian.leftCols(rootVelocityDim).fullPivLu().solve(twist);
    return v;
  }

  /**
   * The world angular velocity of the ACoM attitude xi_a = xi (x)
   * Exp(Delta_theta(q_j)) at `q` under the body angular velocity
   * `angularVelocityLocal` of the base and the joint velocities
   * `jointVelocities`: R(xi_a) (R(Delta_theta)^T omega_B + J_r(Delta_theta)
   * J_Delta_theta qdot_j), through the tangent Jacobian the MPC cost assembles.
   */
  vector3_t acomAngularVelocity(const vector_t& q,
                                const vector3_t& angularVelocityLocal,
                                const vector_t& jointVelocities) const {
    const vector_t qJoints = q.tail(numJoints_);
    const vector4_t attitude =
        acomPtr_->acomAttitude(baseOrientationOf(q), qJoints);
    const AcomAttitudeTangentJacobian tangent =
        acomPtr_->acomAttitudeTangentJacobian(qJoints);
    const vector3_t bodyRate = tangent.baseOrientation * angularVelocityLocal +
                               tangent.jointAngles * jointVelocities;
    return quaternionRotationMatrix(attitude) * bodyRate;
  }

  /**
   * The map from joint velocities to the ACoM's angular velocity in base axes,
   * J_l(Delta_theta) J_Delta_theta, at the joint positions `qJoints`:
   * R(Delta_theta) - the attitude at the identity base, Exp(Delta_theta) -
   * times the tangent Jacobian's joint block J_r(Delta_theta) J_Delta_theta,
   * which carries that block from the attitude's axes into the base's (J_l = R
   * J_r). This, not J_Delta_theta alone, is what the composition xi (x)
   * Exp(Delta_theta) makes the network responsible for.
   */
  matrix_t acomJointRateJacobian(const vector_t& qJoints) const {
    const matrix3_t offsetRotation = quaternionRotationMatrix(
        acomPtr_->acomAttitude(vector4_t::UnitW(), qJoints));
    return offsetRotation *
           acomPtr_->acomAttitudeTangentJacobian(qJoints).jointAngles;
  }

  /**
   * Mean and worst relative Frobenius error of `jacobian` against the
   * connection over kNumSamples configurations. `jacobian` maps the Pinocchio
   * configuration to a (3 x numJoints) joint Jacobian.
   */
  template <typename JacobianFunction>
  std::pair<scalar_t, scalar_t> relativeJacobianError(
      std::mt19937& generator, const JacobianFunction& jacobian) const {
    scalar_t worstRelativeError = 0.0;
    scalar_t summedRelativeError = 0.0;
    for (int sample = 0; sample < kNumSamples; ++sample) {
      const vector_t q = sampleConfiguration(generator);
      const matrix_t target = connectionJointBlock(q);
      const matrix_t predicted = jacobian(q);
      EXPECT_EQ(predicted.rows(), target.rows());
      EXPECT_EQ(predicted.cols(), target.cols());
      EXPECT_GT(target.norm(), 1.0e-6)
          << "the connection must not be identically zero, or this test is "
             "vacuous";
      const scalar_t relativeError =
          (predicted - target).norm() / target.norm();
      worstRelativeError = std::max(worstRelativeError, relativeError);
      summedRelativeError += relativeError;
    }
    return {summedRelativeError / static_cast<scalar_t>(kNumSamples),
            worstRelativeError};
  }

  std::unique_ptr<ModelSettings> modelSettingsPtr_;
  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  std::unique_ptr<AngularCenterOfMass> acomPtr_;
  Eigen::Index numJoints_ = 0;
};

TEST_P(AcomAngularVelocityConsistencyTest,
       theJointRateMapApproximatesTheCentroidalConnection) {
  // The training objective itself, J_l(Delta_theta) J_Delta_theta against the
  // connection, evaluated on the exported weights over the joint-limit box.
  // Reported relative to the norm of the target, so the number means the same
  // thing on another robot.
  const AcomRobotCase& robotCase = GetParam();
  std::mt19937 generator(20260919);
  const std::pair<scalar_t, scalar_t> error =
      relativeJacobianError(generator, [this](const vector_t& q) -> matrix_t {
        return acomJointRateJacobian(q.tail(numJoints_));
      });
  RecordProperty("meanRelativeError", absl::StrCat(error.first));
  RecordProperty("worstRelativeError", absl::StrCat(error.second));
  LOG(INFO) << "[aCOM " << robotCase.robotName
            << "] mean relative Frobenius error " << error.first << " (shipped "
            << robotCase.measured.meanRelativeError << "), worst "
            << error.second << " (shipped "
            << robotCase.measured.worstRelativeError << ")";

  EXPECT_LT(error.first, robotCase.bounds.meanRelativeError)
      << "the exported aCOM network's J_l J no longer approximates I_G^-1 "
         "A_omega,j as well as the shipped one; retrain or re-export";
  EXPECT_LT(error.second, robotCase.bounds.worstRelativeError)
      << "some configuration in the joint-limit box is badly approximated";
}

TEST_P(AcomAngularVelocityConsistencyTest,
       theBoundsRejectANetworkFedTheWrongJointOrder) {
  // Positive control for the bounds above: are they tight enough to fail the
  // regression they exist to catch? A header trained on a different joint order
  // evaluates N(P q) where the MPC expects N(q), so its rate map is J_l(N(P q))
  // J_N(P q) P. The same network, fed its joints reversed, is exactly such a
  // header - and it has to land above the robot's bound.
  const AcomRobotCase& robotCase = GetParam();
  std::vector<int> reversed(static_cast<size_t>(numJoints_));
  std::iota(reversed.rbegin(), reversed.rend(), 0);
  const Eigen::PermutationMatrix<Eigen::Dynamic> permutation(
      Eigen::Map<const Eigen::VectorXi>(reversed.data(), numJoints_));

  std::mt19937 generator(20260919);
  const std::pair<scalar_t, scalar_t> error = relativeJacobianError(
      generator, [this, &permutation](const vector_t& q) -> matrix_t {
        const vector_t permutedJoints = permutation * q.tail(numJoints_);
        return acomJointRateJacobian(permutedJoints) * matrix_t(permutation);
      });
  LOG(INFO) << "[aCOM " << robotCase.robotName
            << "] joint-reversed network: mean relative Frobenius error "
            << error.first;
  EXPECT_GT(error.first, robotCase.bounds.meanRelativeError)
      << "the acceptance bound is too loose to tell this robot's network from "
         "the same network fed the wrong joint order";
}

TEST_P(AcomAngularVelocityConsistencyTest,
       theAcomRateIsTheCentroidalAngularVelocity) {
  // The property stated directly, end to end through the ACoM attitude and the
  // tangent Jacobian the MPC cost assembles, in angular velocities: the world
  // angular velocity of xi_a against omega_locked. The base is tilted but
  // motionless, so omega_locked is R(xi) Abar_omega,j qdot_j and the ACoM's
  // R(xi) J_l(Delta_theta) J_Delta_theta qdot_j: what is compared is what the
  // network was fit to, contracted with a velocity, and carried through the
  // multiplicative composition. Both sides rotate with the base, so the number
  // is that of the identity base.
  //
  // The error is measured against ||Abar||_F ||qdot||, not against
  // ||omega_locked||. Dividing by the latter looks more natural and is a trap:
  // omega_locked passes arbitrarily close to zero for velocity directions near
  // the connection's null space, so that ratio is unbounded for a fixed,
  // perfectly good network and the test would fail on the seed rather than on
  // the weights. This scale is the operator norm bound, so it is stable and
  // comparable with the Frobenius numbers above. Contracting a matrix error
  // with a velocity averages over its directions, so the rate numbers are much
  // smaller than the Frobenius ones and have bounds of their own.
  const AcomRobotCase& robotCase = GetParam();
  std::mt19937 generator(981);
  std::mt19937 baseGenerator(1729);
  std::uniform_real_distribution<scalar_t> velocityDistribution(-1.0, 1.0);
  std::uniform_real_distribution<scalar_t> baseRotationDistribution(-0.6, 0.6);
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();

  scalar_t worstRelativeError = 0.0;
  scalar_t summedRelativeError = 0.0;
  for (int sample = 0; sample < kNumSamples; ++sample) {
    const vector_t atIdentity = sampleConfiguration(generator);
    vector_t jointVelocities(numJoints_);
    for (Eigen::Index joint = 0; joint < numJoints_; ++joint) {
      jointVelocities(joint) = velocityDistribution(generator);
    }
    const vector3_t baseRotation(baseRotationDistribution(baseGenerator),
                                 baseRotationDistribution(baseGenerator),
                                 baseRotationDistribution(baseGenerator));
    const vector_t q = withRotatedBase(atIdentity, baseRotation);
    vector_t v = vector_t::Zero(model.nv);
    v.tail(numJoints_) = jointVelocities;

    const vector3_t acomRate =
        acomAngularVelocity(q, vector3_t::Zero(), jointVelocities);
    const vector3_t locked = lockedAngularVelocity(q, v);

    const scalar_t scale =
        connectionJointBlock(q).norm() * jointVelocities.norm();
    ASSERT_GT(scale, 1.0e-6) << "the sampled motion must actually swing the "
                                "robot for this to mean anything";
    const scalar_t relativeError = (acomRate - locked).norm() / scale;
    worstRelativeError = std::max(worstRelativeError, relativeError);
    summedRelativeError += relativeError;
  }
  const scalar_t meanRelativeError =
      summedRelativeError / static_cast<scalar_t>(kNumSamples);
  RecordProperty("meanRelativeRateError", absl::StrCat(meanRelativeError));
  RecordProperty("worstRelativeRateError", absl::StrCat(worstRelativeError));
  LOG(INFO) << "[aCOM " << robotCase.robotName << "] mean relative rate error "
            << meanRelativeError << " (shipped "
            << robotCase.measured.meanRelativeRateError << "), worst "
            << worstRelativeError << " (shipped "
            << robotCase.measured.worstRelativeRateError << ")";

  EXPECT_LT(meanRelativeError, robotCase.bounds.meanRelativeRateError)
      << "the ACoM's angular velocity has stopped tracking I_G^-1 L_G";
  EXPECT_LT(worstRelativeError, robotCase.bounds.worstRelativeRateError)
      << "some sampled motion is badly tracked";
}

TEST_P(AcomAngularVelocityConsistencyTest,
       theBaseContributionToTheRateIsExactNotApproximated) {
  // The equivariant half of the claim, and the half that must hold to machine
  // precision rather than to a training tolerance: a rigid rotation of the
  // whole robot rotates the ACoM one for one. The joints are locked and the
  // base, at a tilted attitude, spins with a body angular velocity omega_B, so
  // omega_locked is the base's world angular velocity R(xi) omega_B exactly,
  // and so is the ACoM's R(xi_a) R(Delta_theta)^T omega_B: any discrepancy is a
  // structural error in the attitude or its tangent Jacobian rather than a fit
  // error.
  std::mt19937 generator(4242);
  const vector3_t angularVelocityLocal(0.3, -0.2, 0.5);
  for (int sample = 0; sample < 20; ++sample) {
    const vector_t q = withRotatedBase(sampleConfiguration(generator),
                                       vector3_t(0.25, -0.35, 0.4));
    const vector3_t baseRate =
        quaternionRotationMatrix(baseOrientationOf(q)) * angularVelocityLocal;

    const vector3_t locked =
        lockedAngularVelocity(q, rootVelocityOf(q, angularVelocityLocal));
    EXPECT_LT((locked - baseRate).norm(), 1.0e-9)
        << "sample " << sample << ": the base was not driven at omega_B";
    const vector3_t acomRate = acomAngularVelocity(q, angularVelocityLocal,
                                                   vector_t::Zero(numJoints_));
    EXPECT_LT((acomRate - locked).norm(), 1.0e-9) << "sample " << sample;
    // And it is the base's rate itself, unchanged by the configuration of the
    // joints.
    EXPECT_LT((acomRate - baseRate).norm(), 1.0e-12) << "sample " << sample;
  }
}

TEST_P(AcomAngularVelocityConsistencyTest,
       theNetworkBeatsTheTwoTrivialBaselines) {
  // A guard against a threshold that passes vacuously. Two predictors need no
  // training at all:
  //
  //   - Delta_theta == 0, the base orientation used as the whole-body
  //   orientation. Its relative error is exactly one.
  //   - the constant rate map E[Abar_omega,j], the best a single matrix can do
  //   (a linear Delta_theta, to first order in
  //     its offsets). This one is the real baseline: it captures whatever part
  //     of the connection does not vary with configuration. If the network's
  //     J_l(Delta_theta) J_Delta_theta were barely better than this, the SIREN
  //     and the whole training pipeline would be buying almost nothing, and the
  //     honest fix would be to ship the constant matrix.
  //
  // A network that collapsed to zero, was exported with zeroed weights, or was
  // trained for a different robot passes every structural test in this package,
  // and is caught here.
  const AcomRobotCase& robotCase = GetParam();
  std::mt19937 generator(7);
  std::vector<matrix_t> targets;
  std::vector<matrix_t> predictions;
  targets.reserve(kNumSamples);
  predictions.reserve(kNumSamples);
  matrix_t meanConnection = matrix_t::Zero(3, numJoints_);
  for (int sample = 0; sample < kNumSamples; ++sample) {
    const vector_t q = sampleConfiguration(generator);
    targets.push_back(connectionJointBlock(q));
    predictions.push_back(acomJointRateJacobian(q.tail(numJoints_)));
    meanConnection += targets.back();
  }
  meanConnection /= static_cast<scalar_t>(kNumSamples);

  scalar_t networkError = 0.0;
  scalar_t constantJacobianError = 0.0;
  for (int sample = 0; sample < kNumSamples; ++sample) {
    const scalar_t scale = targets[sample].norm();
    networkError += (predictions[sample] - targets[sample]).norm() / scale;
    constantJacobianError += (meanConnection - targets[sample]).norm() / scale;
  }
  networkError /= static_cast<scalar_t>(kNumSamples);
  constantJacobianError /= static_cast<scalar_t>(kNumSamples);
  RecordProperty("constantJacobianBaselineError",
                 absl::StrCat(constantJacobianError));
  LOG(INFO) << "[aCOM " << robotCase.robotName << "] network " << networkError
            << " vs constant-Jacobian baseline " << constantJacobianError
            << " vs zero-offset baseline 1";

  EXPECT_LT(networkError, 1.0) << "the trained network is no better than "
                                  "assuming the joints do not move the aCOM";
  EXPECT_LT(networkError, constantJacobianError)
      << "the trained network is no better than a single constant Jacobian, "
         "which needs no network at all";
  // The zero-offset baseline scores exactly one, so a mean bound at or above
  // one would accept it.
  EXPECT_LT(robotCase.bounds.meanRelativeError, 1.0)
      << "this robot's mean bound would accept a network that does nothing";
}

TEST_P(AcomAngularVelocityConsistencyTest,
       theExportedJointNamesMatchTheMpcModelJointForJoint) {
  // The network is a function of a VECTOR of joint angles, so a permutation
  // between the order it was trained in and the order the MPC hands it is not
  // an error that degrades gracefully - it evaluates a different robot. The
  // shapes still match, the analytic Jacobian still matches finite differences
  // and the equivariance still holds, so only a comparison of names catches it.
  // Checked against both the model settings and the reduced Pinocchio model the
  // cost is built on, because either could drift from the header.
  const std::vector<std::string>& trainedJointNames = acomPtr_->getJointNames();
  ASSERT_EQ(trainedJointNames.size(),
            modelSettingsPtr_->mpcModelJointNames.size());
  for (size_t joint = 0; joint < trainedJointNames.size(); ++joint) {
    EXPECT_EQ(trainedJointNames[joint],
              modelSettingsPtr_->mpcModelJointNames[joint])
        << "joint " << joint
        << ": regenerate the weight header against the reduced MPC model";
  }

  // The model info as the MPC builds it for the same reduced model.
  const CentroidalModelInfo info = centroidal_model::createCentroidalModelInfo(
      *pinocchioInterfacePtr_, CentroidalModelType::FullCentroidalDynamics,
      vector_t::Zero(numJoints_), modelSettingsPtr_->contactNames3DoF,
      modelSettingsPtr_->contactNames6DoF);
  ASSERT_EQ(info.actuatedDofNum, static_cast<size_t>(numJoints_));
  const absl::StatusOr<std::vector<std::string>> pinocchioJointNames =
      ComAndAcomTrackingCost::actuatedJointNames(*pinocchioInterfacePtr_, info);
  ASSERT_TRUE(pinocchioJointNames.ok()) << pinocchioJointNames.status();
  EXPECT_EQ(*pinocchioJointNames, trainedJointNames);
}

TEST_P(AcomAngularVelocityConsistencyTest,
       createRejectsAJointListTheNetworkWasNotTrainedOn) {
  // What the runtime check does when the configuration drifts: a joint list of
  // the right LENGTH whose order differs - here the last two joints swapped, as
  // a changed fixed_joint_names or a re-ordered URDF would produce.
  const AcomRobotCase& robotCase = GetParam();
  std::vector<std::string> swapped = modelSettingsPtr_->mpcModelJointNames;
  ASSERT_GE(swapped.size(), 2u);
  const size_t firstDifference = swapped.size() - 2;
  std::swap(swapped[firstDifference], swapped[firstDifference + 1]);

  const absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
      AngularCenterOfMass::Create(robotCase.robotName, swapped);
  ASSERT_FALSE(acom.ok()) << "a permuted joint list was accepted";
  EXPECT_EQ(acom.status().code(), absl::StatusCode::kFailedPrecondition);
  const std::string message(acom.status().message());
  EXPECT_TRUE(
      absl::StrContains(message, absl::StrCat("'", robotCase.robotName, "'")))
      << message;
  EXPECT_TRUE(
      absl::StrContains(message, absl::StrCat("joint ", firstDifference, " ")))
      << message;
  EXPECT_TRUE(absl::StrContains(
      message,
      absl::StrCat("'", swapped[firstDifference], "' in the MPC model")))
      << message;
  EXPECT_TRUE(absl::StrContains(message, "model_settings.fixed_joint_names"))
      << message;

  // A list of the wrong length is refused too, rather than reaching the
  // evaluator's size check at solve time.
  swapped.pop_back();
  EXPECT_EQ(
      AngularCenterOfMass::Create(robotCase.robotName, swapped).status().code(),
      absl::StatusCode::kFailedPrecondition);
}

INSTANTIATE_TEST_SUITE_P(
    EveryRegisteredNetwork,
    AcomAngularVelocityConsistencyTest,
    ::testing::ValuesIn(kAcomRobotCases),
    [](const ::testing::TestParamInfo<AcomRobotCase>& info) {
      return std::string(info.param.robotName);
    });

TEST(AcomRegistryTest, everyRegisteredNetworkHasAnAcceptanceCase) {
  // A robot added to the registry without a case above would ship a network no
  // test has graded - which is how the G1 and SA01 headers went untested. This
  // makes the omission a failure rather than a LINT reminder.
  std::vector<std::string> cased;
  for (const AcomRobotCase& robotCase : kAcomRobotCases) {
    cased.emplace_back(robotCase.robotName);
  }
  std::sort(cased.begin(), cased.end());
  EXPECT_EQ(cased, AngularCenterOfMass::registeredRobotNames());
}

TEST(AcomRegistryTest, createRejectsAnUnknownRobotAndSaysWhatToChange) {
  const absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
      AngularCenterOfMass::Create("r1", /*modelJointNames=*/{});
  ASSERT_FALSE(acom.ok());
  EXPECT_EQ(acom.status().code(), absl::StatusCode::kNotFound);
  const std::string message(acom.status().message());
  // The keys to change, and every name that would have worked.
  EXPECT_TRUE(absl::StrContains(message, "model_settings.robot_name 'r1'"))
      << message;
  EXPECT_TRUE(absl::StrContains(message, "com_and_acom_tracking_cost"))
      << message;
  EXPECT_TRUE(absl::StrContains(message, "heading_double_integrator"))
      << message;
  for (const std::string& name : AngularCenterOfMass::registeredRobotNames()) {
    EXPECT_TRUE(absl::StrContains(message, name))
        << name << " is missing from: " << message;
  }
}

TEST(AcomRegistryTest,
     theUnvalidatedNetworksAreSwitchedOffInTheShippedConfiguration) {
  // A network that has not met the reference acceptance bounds must not be
  // switched on where it ships. Two places load it: ComAndAcomTrackingCost
  // (com_and_acom_tracking_cost in task.textproto's costs list) and the contact
  // planner's heading model (heading_double_integrator in
  // contact_planning.textproto's dynamics list).
  int unvalidated = 0;
  for (const AcomRobotCase& robotCase : kAcomRobotCases) {
    // Through the loader and the formulation conversion the MPC itself uses, so
    // that a file it would refuse fails here too.
    const absl::StatusOr<mpc_config::TaskFile> task =
        loadTaskFile(taskFileOf(robotCase));
    ASSERT_TRUE(task.ok()) << robotCase.robotName << ": " << task.status();
    const absl::StatusOr<MpcFormulationTasks> formulation =
        mpcFormulationTasksFromConfig(*task, FormulationLogging::kQuiet);
    ASSERT_TRUE(formulation.ok())
        << robotCase.robotName << ": " << formulation.status();
    if (robotCase.validatedForClosedLoop) {
      continue;
    }
    ++unvalidated;
    EXPECT_FALSE(formulation->hasCost(MpcCostType::kComAndAcomTrackingCost))
        << robotCase.robotName
        << "'s ACoM network is NOT VALIDATED (see kAcomRobotCases), but its "
           "task.textproto lists com_and_acom_tracking_cost in costs";

    const absl::StatusOr<std::optional<mpc_config::ContactPlanningFile>>
        contactPlanning = loadContactPlanningFileBeside(taskFileOf(robotCase));
    ASSERT_TRUE(contactPlanning.ok())
        << robotCase.robotName << ": " << contactPlanning.status();
    if (!contactPlanning->has_value()) {
      continue;
    }
    for (const std::string& block : (*contactPlanning)->dynamics) {
      EXPECT_NE(block, "heading_double_integrator")
          << robotCase.robotName
          << "'s ACoM network is NOT VALIDATED (see kAcomRobotCases), but its "
          << contactPlanningFileBeside(taskFileOf(robotCase))
          << " lists heading_double_integrator, which runs it";
    }
  }
  EXPECT_GT(unvalidated, 0) << "every network is validated; this test no "
                               "longer checks anything and can go";
}

TEST(AcomJointOrderTest, pinocchioOrdersSiblingJointsByName) {
  // dataset_generator.py has to reproduce the joint order the MPC's Pinocchio
  // model uses, and it cannot ask Pinocchio: the training runs in a hermetic
  // Python without it. So the rule it implements is checked HERE, against the
  // parser the MPC uses (urdfdom, then pinocchio::urdf::buildModel, as
  // loadCustomPinocchioInterface calls them): a depth-first walk of the
  // kinematic tree that visits the children of a link sorted by JOINT NAME,
  // because urdfdom builds each link's child list by iterating its std::map of
  // joints - not in the order the URDF lists them. test_acom.py asserts that
  // the generator produces this same list from this same URDF.
  // LINT.IfChange(sibling_order_fixture)
  const std::string urdfXml = R"(<robot name="sibling_order">
  <link name="base"/>
  <link name="z_link"/>
  <link name="z_child_link"/>
  <link name="a_link"/>
  <link name="m_link"/>
  <joint name="zeta_joint" type="revolute"><parent link="base"/><child link="z_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="alpha_joint" type="revolute"><parent link="base"/><child link="a_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="mid_joint" type="revolute"><parent link="base"/><child link="m_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="zeta_child_joint" type="revolute"><parent link="z_link"/><child link="z_child_link"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>)";
  const std::vector<std::string> expectedOrder = {
      "alpha_joint", "mid_joint", "zeta_joint", "zeta_child_joint"};
  // LINT.ThenChange(//humanoid_learning/acom/tests/test_acom.py:sibling_order_fixture)

  const urdf::ModelInterfaceSharedPtr urdfTree = urdf::parseURDF(urdfXml);
  ASSERT_NE(urdfTree, nullptr);
  pinocchio::Model model;
  pinocchio::urdf::buildModel(urdfTree, pinocchio::JointModelFreeFlyer(),
                              model);
  // names[0] is the universe and names[1] the floating base.
  const std::vector<std::string> jointOrder(model.names.begin() + 2,
                                            model.names.end());
  EXPECT_EQ(jointOrder, expectedOrder);
  // The fixture only proves anything because its document order is NOT the
  // answer.
  EXPECT_NE(jointOrder,
            (std::vector<std::string>{"zeta_joint", "zeta_child_joint",
                                      "alpha_joint", "mid_joint"}));
}

}  // namespace
}  // namespace ocs2::humanoid
