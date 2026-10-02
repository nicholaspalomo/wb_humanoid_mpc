/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <urdf_parser/urdf_parser.h>
#include <yaml-cpp/yaml.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

/** Number of generalized coordinates the floating base occupies: 3 translations, then ZYX Euler angles. */
constexpr Eigen::Index kGeneralizedBaseDim = 6;
/** Index of the first base orientation coordinate within the Pinocchio q vector. */
constexpr Eigen::Index kBaseOrientationOffset = 3;
/** How many random configurations the statistical assertions are averaged over. */
constexpr int kNumSamples = 200;
/**
 * Half-width of the range a joint without position limits is sampled over. dataset_generator.py samples such a joint
 * over the same range, so both sides grade the network on the distribution it was fit to. No shipped robot has one.
 */
// LINT.IfChange(acom_unlimited_joint_range)
constexpr scalar_t kUnlimitedJointHalfRange = M_PI;
// LINT.ThenChange(//humanoid_learning/acom/dataset_generator.py:unlimited_joint_range)

/**
 * Acceptance numbers of one SHIPPED weight header, in units of the norm of the target. See the class comment below for
 * what the two quantities are.
 */
struct AcceptanceNumbers {
  scalar_t meanRelativeError;
  scalar_t worstRelativeError;
  scalar_t meanRelativeRateError;
  scalar_t worstRelativeRateError;
};

/** One robot with a compiled-in aCOM network, and what its network has to achieve. */
struct AcomRobotCase {
  /// model_settings.robotName, which is also the name the network is registered under.
  const char* robotName;
  /// Robot config package, relative to the repository root, holding config/mpc/task.yaml (and contact_planning.yaml
  /// where the robot has one).
  const char* configPackage;
  /// The robot's URDF, relative to the repository root.
  const char* urdfFile;
  /**
   * Whether the network is good enough to run in closed loop. A network that is not must stay switched off in the
   * robot's shipped configuration, which theUnvalidatedNetworksAreSwitchedOffInTheShippedConfiguration enforces.
   */
  bool validatedForClosedLoop;
  /// What the shipped header measured when its bounds were set, reported next to the fresh measurement.
  AcceptanceNumbers measured;
  /// The acceptance bounds the header must stay under.
  AcceptanceNumbers bounds;
};

std::ostream& operator<<(std::ostream& stream, const AcomRobotCase& robotCase) {
  return stream << robotCase.robotName;
}

/**
 * Every robot with a network in AngularCenterOfMass's registry, and its acceptance bounds.
 *
 * The bounds are PER ROBOT and MEASURED, and they do not all mean "good".
 *
 *  - atlas is the reference, and the only network validated for closed loop. Its bounds sit well above its
 *    measurement on purpose: a retrain on four times the data moved the mean by 2 %, the same order as the 1.8 % spread
 *    across training seeds, so bounds pinned to the last measurement would fail on a reseed rather than on a
 *    regression. They are set where a MEANINGFUL regression lives - a wrong architecture, a bad export, a network
 *    trained for another robot.
 *
 *  - g1 and engineai_sa01 do NOT meet atlas's bounds, and are NOT VALIDATED: on average their joint Jacobian is 40 %
 *    and 51 % off the target, against atlas's 22 %, and at some configurations of both robots the network is further
 *    from the target than no network at all (worst relative error above 1). They were measured and kept rather than
 *    retrained, and their bounds are those measurements plus about 20 %: loose enough that a reseeded retrain of the
 *    same recipe still passes, tight enough that a permuted, transposed or wrong-robot header fails (the positive
 *    control theBoundsRejectANetworkFedTheWrongJointOrder shows it does). These bounds pin "no worse than what ships",
 *    NOT "good enough to use"; a better network is how they come down, and meeting atlas's bounds is what validating
 *    one would take.
 *
 * "measured" is what this file's samples give on the shipped header (mean and worst relative Frobenius error, then
 * mean and worst relative rate error); the test logs the fresh numbers next to it.
 */
// LINT.IfChange(acom_acceptance_robots)
const AcomRobotCase kAcomRobotCases[] = {
    {"atlas", "robot_models/drc_atlas/drc_atlas_centroidal_mpc", "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
     /*validatedForClosedLoop=*/true,
     /*measured=*/{0.219, 0.375, 0.042, 0.138}, /*bounds=*/{0.30, 0.50, 0.08, 0.25}},
    {"engineai_sa01", "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
     "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf", /*validatedForClosedLoop=*/false,
     /*measured=*/{0.508, 1.62, 0.127, 0.647}, /*bounds=*/{0.60, 1.95, 0.15, 0.80}},
    {"g1", "robot_models/unitree_g1/g1_centroidal_mpc", "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     /*validatedForClosedLoop=*/false,
     /*measured=*/{0.402, 1.33, 0.072, 0.238}, /*bounds=*/{0.48, 1.60, 0.09, 0.30}},
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/acom/AngularCenterOfMass.cpp:acom_robot_dispatch, //humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:acom_acceptance_data, //humanoid_learning/acom/README.md:acom_acceptance_numbers, //robot_models/drc_atlas/README.md:acom_status, //robot_models/unitree_g1/README.md:acom_status, //robot_models/engineai_sa01/README.md:acom_status)
// clang-format on

/** The robot's task file, from the test's runfiles. */
std::string taskFileOf(const AcomRobotCase& robotCase) {
  return robot::resolveResourcePath(absl::StrCat(robotCase.configPackage, "/config/mpc/task.yaml")).value();
}

/** The robot's URDF, from the test's runfiles. */
std::string urdfFileOf(const AcomRobotCase& robotCase) {
  return robot::resolveResourcePath(robotCase.urdfFile).value();
}

/**
 * Does the aCOM actually do the one thing it exists to do?
 *
 * Every other test of this subsystem checks its STRUCTURE - shapes, floating-base equivariance, the analytic Jacobian
 * against finite differences, the XYZ-to-ZYX row permutation, the joint ordering recorded in the generated header.
 * All of those would pass just as happily with an untrained network, or with weights trained for a different robot's
 * mass distribution. None of them looks at the quantity the network was fit to.
 *
 * The defining property (Chen, Nelson, Griffin, Posa and Pratt, IROS 2023) is that the aCOM is an INTEGRABLE
 * approximation of the whole-body orientation whose rate is the centroidal angular velocity:
 *
 *   d/dt theta_aCOM  ~=  omega_locked  =  I_G(q)^-1 L_G(q, qdot),
 *
 * with I_G the locked rotational inertia at the center of mass and L_G the centroidal angular momentum. Substituting
 * the equivariant decomposition theta_aCOM = theta_base + Delta_theta(q_j), whose base block is exact by construction,
 * reduces the claim to the joint block alone:
 *
 *   d(Delta_theta)/d(q_j)  ~=  Abar_omega,j(q)  =  I_G(q)^-1 A_omega,j(q),
 *
 * which is exactly the Frobenius objective train_acom.py minimizes. So this file is the C++-side acceptance test for
 * the TRAINING, evaluated through the exported weight header that the robot actually runs - not through the JAX
 * parameters the Python tests see - for every robot that has one. A retrained or re-exported network that regressed, a
 * transposed weight matrix, a row permutation applied in the wrong direction, or a header exported for the wrong robot
 * all show up here and nowhere else.
 *
 * There is a second thing this file gets for free, and it is worth stating because nothing else in the repository has
 * it. The target here is recomputed with Pinocchio's ccrba on the reduced MPC model, which is an INDEPENDENT
 * implementation of the one that produced the training data - dataset_generator.py reads MuJoCo's subtree_angmom under
 * unit joint velocities, on a model built from a different file, with a different joint ordering that a permutation
 * has to reconcile. A frame convention applied in the wrong direction, a permutation off by one, or a momentum read
 * about the wrong subtree would leave every Python test green (they check MuJoCo against itself) and every structural
 * C++ test green, and would show up here as an error far larger than any fit residual.
 *
 * Configurations are drawn uniformly from the model's joint-limit box, which is the box dataset_generator.py samples.
 * (The shipped headers were trained on that box trimmed by 10 % of each range at both ends, a margin the generator no
 * longer applies because it excluded G1's nominal knee angle; the numbers below therefore include the trimmed margins.)
 *
 * The relation is an approximation and cannot be otherwise: the connection Abar_omega,j has non-zero curvature, so no
 * exact integrable whole-body orientation exists at all - that is the entire premise of the paper. The thresholds are
 * therefore acceptance bounds on the fit quality, not tolerances on a numerical identity; they are stated relative to
 * the magnitude of the target so that a number means the same thing on every robot, and set per robot as described at
 * kAcomRobotCases.
 */
class AcomAngularVelocityConsistencyTest : public ::testing::TestWithParam<AcomRobotCase> {
 protected:
  void SetUp() override {
    const AcomRobotCase& robotCase = GetParam();
    const std::string taskFile = taskFileOf(robotCase);
    const std::string urdfFile = urdfFileOf(robotCase);

    // The REDUCED model the MPC builds, not the full URDF: the trained network is indexed by the MPC's joints, and on
    // the full model the joint count would not even match.
    modelSettingsPtr_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "testAcomAngularVelocityConsistency", /*verbose=*/false);
    ASSERT_EQ(modelSettingsPtr_->robotName, robotCase.robotName) << "the case's task file belongs to another robot";
    pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile, urdfFile, *modelSettingsPtr_));
    numJoints_ = pinocchioInterfacePtr_->getModel().nq - kGeneralizedBaseDim;

    // The checked factory, as the MPC calls it: a joint list that differs from the one the header records fails here.
    absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
        AngularCenterOfMass::Create(modelSettingsPtr_->robotName, modelSettingsPtr_->mpcModelJointNames);
    ASSERT_TRUE(acom.ok()) << acom.status();
    acomPtr_ = *std::move(acom);
    ASSERT_EQ(acomPtr_->getInputDim(), static_cast<std::size_t>(numJoints_))
        << "the exported weights are indexed by a different joint set than the MPC model";
  }

  /**
   * The joint block of the connection, Abar_omega,j = I_G^-1 A_omega,j, and the locked inertia, from Pinocchio.
   *
   * This model's floating base is a Translation joint composed with a SphericalZYX joint, so the tangent vector v is
   * literally dq/dt and ccrba's base columns are with respect to the EULER RATES rather than an angular velocity.
   * That distinction does not touch the joint columns this method returns - a joint velocity is the same number in
   * either convention - but it is why the rate tests below keep the base level or motionless.
   */
  matrix_t connectionJointBlock(const vector_t& q) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    const vector_t zeroVelocity = vector_t::Zero(model.nv);
    pinocchio::ccrba(model, data, q, zeroVelocity);
    const matrix3_t lockedInertia = data.Ig.inertia().matrix();
    const matrix_t jointAngularMomentumMap = data.Ag.bottomRows(3).rightCols(numJoints_);
    return lockedInertia.ldlt().solve(jointAngularMomentumMap);
  }

  /** omega_locked = I_G^-1 L_G, the centroidal angular velocity, in world axes. */
  vector3_t lockedAngularVelocity(const vector_t& q, const vector_t& v) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    pinocchio::Data data(model);
    pinocchio::ccrba(model, data, q, v);
    const matrix3_t lockedInertia = data.Ig.inertia().matrix();
    return lockedInertia.ldlt().solve(vector3_t(data.hg.angular()));
  }

  /** A configuration drawn uniformly from the model's joint limits, with the base wherever the caller wants it. */
  vector_t sampleConfiguration(std::mt19937& generator, const vector3_t& baseEulerZyx) const {
    const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
    vector_t q = vector_t::Zero(model.nq);
    q(2) = 0.9;  // a plausible base height; the CoM and the inertia do not depend on it, but a sane pose is easier to read
    q.segment<3>(kBaseOrientationOffset) = baseEulerZyx;
    for (Eigen::Index joint = 0; joint < numJoints_; ++joint) {
      const Eigen::Index index = kGeneralizedBaseDim + joint;
      const scalar_t lower = std::isfinite(model.lowerPositionLimit(index)) ? model.lowerPositionLimit(index) : -kUnlimitedJointHalfRange;
      const scalar_t upper = std::isfinite(model.upperPositionLimit(index)) ? model.upperPositionLimit(index) : kUnlimitedJointHalfRange;
      std::uniform_real_distribution<scalar_t> distribution(lower, upper);
      q(index) = distribution(generator);
    }
    return q;
  }

  /**
   * Mean and worst relative Frobenius error of `jacobian` against the connection over kNumSamples configurations.
   * `jacobian` maps the Pinocchio configuration to a (3 x numJoints) joint Jacobian.
   */
  template <typename JacobianFunction>
  std::pair<scalar_t, scalar_t> relativeJacobianError(std::mt19937& generator, const JacobianFunction& jacobian) const {
    scalar_t worstRelativeError = 0.0;
    scalar_t summedRelativeError = 0.0;
    for (int sample = 0; sample < kNumSamples; ++sample) {
      const vector_t q = sampleConfiguration(generator, vector3_t::Zero());
      const matrix_t target = connectionJointBlock(q);
      const matrix_t predicted = jacobian(q);
      EXPECT_EQ(predicted.rows(), target.rows());
      EXPECT_EQ(predicted.cols(), target.cols());
      EXPECT_GT(target.norm(), 1e-6) << "the connection must not be identically zero, or this test is vacuous";
      const scalar_t relativeError = (predicted - target).norm() / target.norm();
      worstRelativeError = std::max(worstRelativeError, relativeError);
      summedRelativeError += relativeError;
    }
    return {summedRelativeError / static_cast<scalar_t>(kNumSamples), worstRelativeError};
  }

  std::unique_ptr<ModelSettings> modelSettingsPtr_;
  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  std::unique_ptr<AngularCenterOfMass> acomPtr_;
  Eigen::Index numJoints_ = 0;
};

TEST_P(AcomAngularVelocityConsistencyTest, theJointJacobianApproximatesTheCentroidalConnection) {
  // The training objective itself, evaluated on the exported weights over the joint-limit box. Reported relative to the
  // norm of the target, so the number means the same thing on another robot.
  const AcomRobotCase& robotCase = GetParam();
  std::mt19937 generator(20260919);
  const std::pair<scalar_t, scalar_t> error = relativeJacobianError(
      generator, [this](const vector_t& q) -> matrix_t { return acomPtr_->computeJointOffsetJacobian(q.tail(numJoints_)); });
  RecordProperty("meanRelativeError", absl::StrCat(error.first));
  RecordProperty("worstRelativeError", absl::StrCat(error.second));
  LOG(INFO) << "[aCOM " << robotCase.robotName << "] mean relative Frobenius error " << error.first << " (shipped "
            << robotCase.measured.meanRelativeError << "), worst " << error.second << " (shipped " << robotCase.measured.worstRelativeError
            << ")";

  EXPECT_LT(error.first, robotCase.bounds.meanRelativeError)
      << "the exported aCOM network no longer approximates I_G^-1 A_omega,j as well as the shipped one; retrain or re-export";
  EXPECT_LT(error.second, robotCase.bounds.worstRelativeError) << "some configuration in the joint-limit box is badly approximated";
}

TEST_P(AcomAngularVelocityConsistencyTest, theBoundsRejectANetworkFedTheWrongJointOrder) {
  // Positive control for the bounds above: are they tight enough to fail the regression they exist to catch? A header
  // trained on a different joint order evaluates N(P q) where the MPC expects N(q), so its Jacobian is J_N(P q) P. The
  // same network, fed its joints reversed, is exactly such a header - and it has to land above the robot's bound.
  const AcomRobotCase& robotCase = GetParam();
  std::vector<int> reversed(static_cast<std::size_t>(numJoints_));
  std::iota(reversed.rbegin(), reversed.rend(), 0);
  const Eigen::PermutationMatrix<Eigen::Dynamic> permutation(Eigen::Map<const Eigen::VectorXi>(reversed.data(), numJoints_));

  std::mt19937 generator(20260919);
  const std::pair<scalar_t, scalar_t> error = relativeJacobianError(generator, [this, &permutation](const vector_t& q) -> matrix_t {
    const vector_t permutedJoints = permutation * q.tail(numJoints_);
    return acomPtr_->computeJointOffsetJacobian(permutedJoints) * matrix_t(permutation);
  });
  LOG(INFO) << "[aCOM " << robotCase.robotName << "] joint-reversed network: mean relative Frobenius error " << error.first;
  EXPECT_GT(error.first, robotCase.bounds.meanRelativeError)
      << "the acceptance bound is too loose to tell this robot's network from the same network fed the wrong joint order";
}

TEST_P(AcomAngularVelocityConsistencyTest, theAcomRateIsTheCentroidalAngularVelocity) {
  // The property stated directly, end to end through the aCOM Jacobian the MPC cost actually assembles. The base is
  // left level and motionless, so omega_locked reduces to the joint term Abar_omega,j qdot_j and the aCOM rate to
  // P J_Delta_theta qdot_j: no Euler-rate to angular-velocity mapping enters, and what is compared is exactly what the
  // network was fit to, contracted with a velocity.
  //
  // The error is measured against ||Abar||_F ||qdot||, not against ||omega_locked||. Dividing by the latter looks more
  // natural and is a trap: omega_locked passes arbitrarily close to zero for velocity directions near the connection's
  // null space, so that ratio is unbounded for a fixed, perfectly good network and the test would fail on the seed
  // rather than on the weights. This scale is the operator norm bound, so it is stable and comparable with the
  // Frobenius numbers above. Contracting a matrix error with a velocity averages over its directions, so the rate
  // numbers are much smaller than the Frobenius ones and have bounds of their own.
  const AcomRobotCase& robotCase = GetParam();
  std::mt19937 generator(981);
  std::uniform_real_distribution<scalar_t> velocityDistribution(-1.0, 1.0);
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();

  scalar_t worstRelativeError = 0.0;
  scalar_t summedRelativeError = 0.0;
  for (int sample = 0; sample < kNumSamples; ++sample) {
    const vector_t q = sampleConfiguration(generator, vector3_t::Zero());
    vector_t v = vector_t::Zero(model.nv);
    for (Eigen::Index joint = 0; joint < numJoints_; ++joint) {
      v(kGeneralizedBaseDim + joint) = velocityDistribution(generator);
    }

    // d/dt theta_aCOM in the centroidal state's ZYX order, from the full aCOM Jacobian the MPC cost uses...
    const vector3_t acomRateZyx = acomPtr_->computeAcomJacobian(q) * v;
    // ...against the centroidal angular velocity, permuted into the same order.
    const vector3_t lockedZyx = acomXyzToZyx(lockedAngularVelocity(q, v));

    const scalar_t scale = connectionJointBlock(q).norm() * v.norm();
    ASSERT_GT(scale, 1e-6) << "the sampled motion must actually swing the robot for this to mean anything";
    const scalar_t relativeError = (acomRateZyx - lockedZyx).norm() / scale;
    worstRelativeError = std::max(worstRelativeError, relativeError);
    summedRelativeError += relativeError;
  }
  const scalar_t meanRelativeError = summedRelativeError / static_cast<scalar_t>(kNumSamples);
  RecordProperty("meanRelativeRateError", absl::StrCat(meanRelativeError));
  RecordProperty("worstRelativeRateError", absl::StrCat(worstRelativeError));
  LOG(INFO) << "[aCOM " << robotCase.robotName << "] mean relative rate error " << meanRelativeError << " (shipped "
            << robotCase.measured.meanRelativeRateError << "), worst " << worstRelativeError << " (shipped "
            << robotCase.measured.worstRelativeRateError << ")";

  EXPECT_LT(meanRelativeError, robotCase.bounds.meanRelativeRateError) << "d/dt theta_aCOM has stopped tracking I_G^-1 L_G";
  EXPECT_LT(worstRelativeError, robotCase.bounds.worstRelativeRateError) << "some sampled motion is badly tracked";
}

TEST_P(AcomAngularVelocityConsistencyTest, theBaseContributionToTheRateIsExactNotApproximated) {
  // The equivariant half of the claim, and the half that must hold to machine precision rather than to a training
  // tolerance: a rigid rotation of the whole robot rotates the aCOM one-for-one. Keeping the base level makes the
  // Euler rates and the angular velocity the same vector, so omega_locked picks up the base rate exactly and any
  // discrepancy is a structural error in the Jacobian rather than a fit error.
  std::mt19937 generator(4242);
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  for (int sample = 0; sample < 20; ++sample) {
    const vector_t q = sampleConfiguration(generator, vector3_t::Zero());

    vector_t baseOnly = vector_t::Zero(model.nv);
    baseOnly.segment<3>(kBaseOrientationOffset) = vector3_t(0.3, -0.2, 0.5);  // ZYX Euler rates == omega at theta = 0

    const vector3_t acomRateZyx = acomPtr_->computeAcomJacobian(q) * baseOnly;
    const vector3_t lockedZyx = acomXyzToZyx(lockedAngularVelocity(q, baseOnly));
    EXPECT_LT((acomRateZyx - lockedZyx).norm(), 1e-9) << "sample " << sample;
    // And it is the base rate itself, unchanged by the configuration of the joints.
    EXPECT_LT((acomRateZyx - vector3_t(0.3, -0.2, 0.5)).norm(), 1e-12) << "sample " << sample;
  }
}

TEST_P(AcomAngularVelocityConsistencyTest, theNetworkBeatsTheTwoTrivialBaselines) {
  // A guard against a threshold that passes vacuously. Two predictors need no training at all:
  //
  //   - Delta_theta == 0, the base orientation used as the whole-body orientation. Its relative error is exactly one.
  //   - a LINEAR Delta_theta, i.e. the constant Jacobian E[Abar_omega,j]. This one is the real baseline: it costs a
  //     single matrix and captures whatever part of the connection does not vary with configuration. If the network
  //     were barely better than this, the SIREN and the whole training pipeline would be buying almost nothing, and
  //     the honest fix would be to ship the constant matrix.
  //
  // A network that collapsed to zero, was exported with zeroed weights, or was trained for a different robot passes
  // every structural test in this package, and is caught here.
  const AcomRobotCase& robotCase = GetParam();
  std::mt19937 generator(7);
  std::vector<matrix_t> targets;
  std::vector<matrix_t> predictions;
  targets.reserve(kNumSamples);
  predictions.reserve(kNumSamples);
  matrix_t meanConnection = matrix_t::Zero(3, numJoints_);
  for (int sample = 0; sample < kNumSamples; ++sample) {
    const vector_t q = sampleConfiguration(generator, vector3_t::Zero());
    targets.push_back(connectionJointBlock(q));
    predictions.push_back(acomPtr_->computeJointOffsetJacobian(q.tail(numJoints_)));
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
  RecordProperty("constantJacobianBaselineError", absl::StrCat(constantJacobianError));
  LOG(INFO) << "[aCOM " << robotCase.robotName << "] network " << networkError << " vs constant-Jacobian baseline " << constantJacobianError
            << " vs zero-offset baseline 1";

  EXPECT_LT(networkError, 1.0) << "the trained network is no better than assuming the joints do not move the aCOM";
  EXPECT_LT(networkError, constantJacobianError)
      << "the trained network is no better than a single constant Jacobian, which needs no network at all";
  // The zero-offset baseline scores exactly one, so a mean bound at or above one would accept it.
  EXPECT_LT(robotCase.bounds.meanRelativeError, 1.0) << "this robot's mean bound would accept a network that does nothing";
}

TEST_P(AcomAngularVelocityConsistencyTest, theExportedJointNamesMatchTheMpcModelJointForJoint) {
  // The network is a function of a VECTOR of joint angles, so a permutation between the order it was trained in and the
  // order the MPC hands it is not an error that degrades gracefully - it evaluates a different robot. The shapes still
  // match, the analytic Jacobian still matches finite differences and the equivariance still holds, so only a
  // comparison of names catches it. Checked against both the model settings and the reduced Pinocchio model the cost
  // is built on, because either could drift from the header.
  const std::vector<std::string>& trainedJointNames = acomPtr_->getJointNames();
  ASSERT_EQ(trainedJointNames.size(), modelSettingsPtr_->mpcModelJointNames.size());
  for (std::size_t joint = 0; joint < trainedJointNames.size(); ++joint) {
    EXPECT_EQ(trainedJointNames[joint], modelSettingsPtr_->mpcModelJointNames[joint])
        << "joint " << joint << ": regenerate the weight header against the reduced MPC model";
  }

  CentroidalModelInfo info;
  info.actuatedDofNum = static_cast<std::size_t>(numJoints_);
  const absl::StatusOr<std::vector<std::string>> pinocchioJointNames =
      ComAndAcomTrackingCost::actuatedJointNames(*pinocchioInterfacePtr_, info);
  ASSERT_TRUE(pinocchioJointNames.ok()) << pinocchioJointNames.status();
  EXPECT_EQ(*pinocchioJointNames, trainedJointNames);
}

TEST_P(AcomAngularVelocityConsistencyTest, createRejectsAJointListTheNetworkWasNotTrainedOn) {
  // What the runtime check does when the configuration drifts: a joint list of the right LENGTH whose order differs -
  // here the last two joints swapped, as a changed fixedJointNames or a re-ordered URDF would produce.
  const AcomRobotCase& robotCase = GetParam();
  std::vector<std::string> swapped = modelSettingsPtr_->mpcModelJointNames;
  ASSERT_GE(swapped.size(), 2u);
  const std::size_t firstDifference = swapped.size() - 2;
  std::swap(swapped[firstDifference], swapped[firstDifference + 1]);

  const absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom = AngularCenterOfMass::Create(robotCase.robotName, swapped);
  ASSERT_FALSE(acom.ok()) << "a permuted joint list was accepted";
  EXPECT_EQ(acom.status().code(), absl::StatusCode::kFailedPrecondition);
  const std::string message(acom.status().message());
  EXPECT_TRUE(absl::StrContains(message, absl::StrCat("'", robotCase.robotName, "'"))) << message;
  EXPECT_TRUE(absl::StrContains(message, absl::StrCat("joint ", firstDifference, " "))) << message;
  EXPECT_TRUE(absl::StrContains(message, absl::StrCat("'", swapped[firstDifference], "' in the MPC model"))) << message;
  EXPECT_TRUE(absl::StrContains(message, "model_settings.fixedJointNames")) << message;

  // A list of the wrong length is refused too, rather than reaching the evaluator's size check at solve time.
  swapped.pop_back();
  EXPECT_EQ(AngularCenterOfMass::Create(robotCase.robotName, swapped).status().code(), absl::StatusCode::kFailedPrecondition);
}

INSTANTIATE_TEST_SUITE_P(EveryRegisteredNetwork,
                         AcomAngularVelocityConsistencyTest,
                         ::testing::ValuesIn(kAcomRobotCases),
                         [](const ::testing::TestParamInfo<AcomRobotCase>& info) { return std::string(info.param.robotName); });

TEST(AcomRegistryTest, everyRegisteredNetworkHasAnAcceptanceCase) {
  // A robot added to the registry without a case above would ship a network no test has graded - which is how the G1
  // and SA01 headers went untested. This makes the omission a failure rather than a LINT reminder.
  std::vector<std::string> cased;
  for (const AcomRobotCase& robotCase : kAcomRobotCases) {
    cased.emplace_back(robotCase.robotName);
  }
  std::sort(cased.begin(), cased.end());
  EXPECT_EQ(cased, AngularCenterOfMass::registeredRobotNames());
}

TEST(AcomRegistryTest, createRejectsAnUnknownRobotAndSaysWhatToChange) {
  const absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom = AngularCenterOfMass::Create("r1", {});
  ASSERT_FALSE(acom.ok());
  EXPECT_EQ(acom.status().code(), absl::StatusCode::kNotFound);
  const std::string message(acom.status().message());
  // The keys to change, and every name that would have worked.
  EXPECT_TRUE(absl::StrContains(message, "model_settings.robotName 'r1'")) << message;
  EXPECT_TRUE(absl::StrContains(message, "com_and_acom_tracking_cost")) << message;
  EXPECT_TRUE(absl::StrContains(message, "heading_double_integrator")) << message;
  for (const std::string& name : AngularCenterOfMass::registeredRobotNames()) {
    EXPECT_TRUE(absl::StrContains(message, name)) << name << " is missing from: " << message;
  }
}

TEST(AcomRegistryTest, theUnvalidatedNetworksAreSwitchedOffInTheShippedConfiguration) {
  // A network that has not met the reference acceptance bounds must not be switched on where it ships. Two places load
  // it: ComAndAcomTrackingCost (com_and_acom_tracking_cost in task.yaml's costs list) and the contact planner's heading
  // model (heading_double_integrator in contact_planning.yaml's dynamics list).
  int unvalidated = 0;
  for (const AcomRobotCase& robotCase : kAcomRobotCases) {
    // Through the formulation loader the MPC itself uses, so that a file it would refuse fails here too.
    const absl::StatusOr<MpcFormulationTasks> formulation = loadMpcFormulationTasks(taskFileOf(robotCase));
    ASSERT_TRUE(formulation.ok()) << robotCase.robotName << ": " << formulation.status();
    if (robotCase.validatedForClosedLoop) {
      continue;
    }
    ++unvalidated;
    EXPECT_FALSE(formulation->hasCost(MpcCostType::ComAndAcomTrackingCost))
        << robotCase.robotName
        << "'s ACoM network is NOT VALIDATED (see kAcomRobotCases), but its task.yaml lists com_and_acom_tracking_cost in costs";

    const std::filesystem::path contactPlanningFile = std::filesystem::path(taskFileOf(robotCase)).parent_path() / "contact_planning.yaml";
    if (!std::filesystem::exists(contactPlanningFile)) {
      continue;
    }
    const YAML::Node dynamics = YAML::LoadFile(contactPlanningFile.string())["contact_planning"]["dynamics"];
    ASSERT_TRUE(dynamics.IsSequence()) << contactPlanningFile << " has no contact_planning.dynamics list";
    for (const YAML::Node& block : dynamics) {
      EXPECT_NE(block.as<std::string>(), "heading_double_integrator")
          << robotCase.robotName << "'s ACoM network is NOT VALIDATED (see kAcomRobotCases), but " << contactPlanningFile
          << " lists heading_double_integrator, which runs it";
    }
  }
  EXPECT_GT(unvalidated, 0) << "every network is validated; this test no longer checks anything and can go";
}

TEST(AcomJointOrderTest, pinocchioOrdersSiblingJointsByName) {
  // dataset_generator.py has to reproduce the joint order the MPC's Pinocchio model uses, and it cannot ask Pinocchio:
  // the training runs in a hermetic Python without it. So the rule it implements is checked HERE, against the parser
  // the MPC uses (urdfdom, then pinocchio::urdf::buildModel, as createCustomPinocchioInterface calls them): a
  // depth-first walk of the kinematic tree that visits the children of a link sorted by JOINT NAME, because urdfdom
  // builds each link's child list by iterating its std::map of joints - not in the order the URDF lists them.
  // test_acom.py asserts that the generator produces this same list from this same URDF.
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
  const std::vector<std::string> expectedOrder = {"alpha_joint", "mid_joint", "zeta_joint", "zeta_child_joint"};
  // LINT.ThenChange(//humanoid_learning/acom/tests/test_acom.py:sibling_order_fixture)

  const urdf::ModelInterfaceSharedPtr urdfTree = urdf::parseURDF(urdfXml);
  ASSERT_NE(urdfTree, nullptr);
  pinocchio::Model model;
  pinocchio::urdf::buildModel(urdfTree, pinocchio::JointModelFreeFlyer(), model);
  // names[0] is the universe and names[1] the floating base.
  const std::vector<std::string> jointOrder(model.names.begin() + 2, model.names.end());
  EXPECT_EQ(jointOrder, expectedOrder);
  // The fixture only proves anything because its document order is NOT the answer.
  EXPECT_NE(jointOrder, (std::vector<std::string>{"zeta_joint", "zeta_child_joint", "alpha_joint", "mid_joint"}));
}

}  // namespace
}  // namespace ocs2::humanoid
