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
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_common_mpc/acom/AcomSirenWeightsAtlas.h"
#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

#include "absl/log/log.h"

namespace ocs2::humanoid {
namespace {

/** Number of generalized coordinates the floating base occupies: 3 translations, then ZYX Euler angles. */
constexpr Eigen::Index kGeneralizedBaseDim = 6;
/** Index of the first base orientation coordinate within the Pinocchio q vector. */
constexpr Eigen::Index kBaseOrientationOffset = 3;
/** How many random configurations the statistical assertions are averaged over. */
constexpr int kNumSamples = 200;
/**
 * Acceptance bounds on the fit quality of the SHIPPED weight header, in units of the norm of the target.
 *
 * These are not tolerances on a numerical identity. No exact integrable whole-body orientation exists - the connection
 * Abar_omega,j has non-zero curvature, which is the premise of the paper - so the residual has a floor that no amount
 * of training removes, and the only honest threshold is one measured against the network the robot actually runs.
 *
 * As measured on the Atlas weights currently in AcomSirenWeightsAtlas.h, over configurations drawn uniformly from the
 * joint-limit box (the same distribution dataset_generator.py samples): mean relative Frobenius error 0.219, worst
 * 0.375. The bounds below sit well above that, and deliberately so. They are not tracking the measurement: a retrain
 * on four times the data moved the mean by 2 %, which is the same order as the 1.8 % spread across training seeds, so
 * bounds pinned to the last measurement would fail on a reseed rather than on a regression. They are set where a
 * MEANINGFUL regression lives - a wrong architecture, a bad export, a network trained for another robot.
 */
constexpr scalar_t kMeanRelativeErrorBound = 0.30;
constexpr scalar_t kWorstRelativeErrorBound = 0.50;
/**
 * The same acceptance, on the rate rather than on the Jacobian, and measured against ||Abar||_F ||qdot|| so that the
 * scale cannot collapse; see the rate test for why. Contracting a matrix error with a velocity averages over its
 * directions, so these are much smaller than the Frobenius numbers above and deserve their own bounds rather than
 * borrowing the loose ones. Measured on the shipped Atlas weights: mean 0.042, worst 0.14.
 */
constexpr scalar_t kMeanRelativeRateErrorBound = 0.08;
constexpr scalar_t kWorstRelativeRateErrorBound = 0.25;

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
 * with I_G the locked rotational inertia at the centre of mass and L_G the centroidal angular momentum. Substituting
 * the equivariant decomposition theta_aCOM = theta_base + Delta_theta(q_j), whose base block is exact by construction,
 * reduces the claim to the joint block alone:
 *
 *   d(Delta_theta)/d(q_j)  ~=  Abar_omega,j(q)  =  I_G(q)^-1 A_omega,j(q),
 *
 * which is exactly the Frobenius objective train_acom.py minimises. So this file is the C++-side acceptance test for
 * the TRAINING, evaluated through the exported weight header that the robot actually runs - not through the JAX
 * parameters the Python tests see. A retrained or re-exported network that regressed, a transposed weight matrix, a
 * row permutation applied in the wrong direction, or a header exported for the wrong robot all show up here and
 * nowhere else.
 *
 * There is a second thing this file gets for free, and it is worth stating because nothing else in the repository has
 * it. The target here is recomputed with Pinocchio's ccrba on the reduced MPC model, which is an INDEPENDENT
 * implementation of the one that produced the training data - dataset_generator.py reads MuJoCo's subtree_angmom under
 * unit joint velocities, on a model built from a different file, with a different joint ordering that a permutation
 * has to reconcile. A frame convention applied in the wrong direction, a permutation off by one, or a momentum read
 * about the wrong subtree would leave every Python test green (they check MuJoCo against itself) and every structural
 * C++ test green, and would show up here as an error far larger than any fit residual.
 *
 * The relation is an approximation and cannot be otherwise: the connection Abar_omega,j has non-zero curvature, so no
 * exact integrable whole-body orientation exists at all - that is the entire premise of the paper. The thresholds
 * below are therefore acceptance bounds on the fit quality, not tolerances on a numerical identity, and they are
 * stated relative to the magnitude of the target so that they mean the same thing on a different robot.
 */
class AcomAngularVelocityConsistencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    const std::string taskFile = configDir + "/config/mpc/task.yaml";
    const std::string urdfFile = descriptionDir + "/urdf/atlas.urdf";

    // The REDUCED model the MPC builds, not the full URDF: the trained network is indexed by the MPC's joints, and on
    // the full model the joint count would not even match.
    modelSettingsPtr_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "testAcomAngularVelocityConsistency", false);
    pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile, urdfFile, *modelSettingsPtr_));
    numJoints_ = pinocchioInterfacePtr_->getModel().nq - kGeneralizedBaseDim;

    acomPtr_ = AngularCenterOfMass::createForRobot(modelSettingsPtr_->robotName);
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
      // The limits of a continuous joint come back as +/-infinity; fall back to a radian either way so the sample
      // stays inside the region the network was trained on rather than running off to a numerically silly angle.
      const scalar_t lower = std::isfinite(model.lowerPositionLimit(index)) ? model.lowerPositionLimit(index) : -1.0;
      const scalar_t upper = std::isfinite(model.upperPositionLimit(index)) ? model.upperPositionLimit(index) : 1.0;
      std::uniform_real_distribution<scalar_t> distribution(lower, upper);
      q(index) = distribution(generator);
    }
    return q;
  }

  std::unique_ptr<ModelSettings> modelSettingsPtr_;
  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  std::unique_ptr<AngularCenterOfMass> acomPtr_;
  Eigen::Index numJoints_ = 0;
};

TEST_F(AcomAngularVelocityConsistencyTest, theJointJacobianApproximatesTheCentroidalConnection) {
  // The training objective itself, evaluated on the exported weights over the configuration space the network was fit
  // on. Reported relative to the norm of the target, so the number means the same thing on another robot.
  std::mt19937 generator(20260919);
  scalar_t worstRelativeError = 0.0;
  scalar_t summedRelativeError = 0.0;
  for (int sample = 0; sample < kNumSamples; ++sample) {
    const vector_t q = sampleConfiguration(generator, vector3_t::Zero());
    const matrix_t target = connectionJointBlock(q);
    const matrix_t predicted = acomPtr_->computeJointOffsetJacobian(q.tail(numJoints_));
    ASSERT_EQ(predicted.rows(), target.rows());
    ASSERT_EQ(predicted.cols(), target.cols());

    ASSERT_GT(target.norm(), 1e-6) << "the connection must not be identically zero, or this test is vacuous";
    const scalar_t relativeError = (predicted - target).norm() / target.norm();
    worstRelativeError = std::max(worstRelativeError, relativeError);
    summedRelativeError += relativeError;
  }
  const scalar_t meanRelativeError = summedRelativeError / static_cast<scalar_t>(kNumSamples);
  RecordProperty("meanRelativeError", std::to_string(meanRelativeError));
  RecordProperty("worstRelativeError", std::to_string(worstRelativeError));
  LOG(INFO) << "[aCOM] mean relative Frobenius error " << meanRelativeError << ", worst " << worstRelativeError;

  EXPECT_LT(meanRelativeError, kMeanRelativeErrorBound)
      << "the exported aCOM network no longer approximates I_G^-1 A_omega,j; retrain or re-export";
  EXPECT_LT(worstRelativeError, kWorstRelativeErrorBound) << "some configuration in the joint-limit box is badly approximated";
}

TEST_F(AcomAngularVelocityConsistencyTest, theAcomRateIsTheCentroidalAngularVelocity) {
  // The property stated directly, end to end through the aCOM Jacobian the MPC cost actually assembles. The base is
  // left level and motionless, so omega_locked reduces to the joint term Abar_omega,j qdot_j and the aCOM rate to
  // P J_Delta_theta qdot_j: no Euler-rate to angular-velocity mapping enters, and what is compared is exactly what the
  // network was fit to, contracted with a velocity.
  //
  // The error is measured against ||Abar||_F ||qdot||, not against ||omega_locked||. Dividing by the latter looks more
  // natural and is a trap: omega_locked passes arbitrarily close to zero for velocity directions near the connection's
  // null space, so that ratio is unbounded for a fixed, perfectly good network and the test would fail on the seed
  // rather than on the weights. This scale is the operator norm bound, so it is stable and comparable with the
  // Frobenius numbers above.
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
  RecordProperty("meanRelativeRateError", std::to_string(meanRelativeError));
  RecordProperty("worstRelativeRateError", std::to_string(worstRelativeError));
  LOG(INFO) << "[aCOM] mean relative rate error " << meanRelativeError << ", worst " << worstRelativeError;

  EXPECT_LT(meanRelativeError, kMeanRelativeRateErrorBound) << "d/dt theta_aCOM has stopped tracking I_G^-1 L_G";
  EXPECT_LT(worstRelativeError, kWorstRelativeRateErrorBound) << "some sampled motion is badly tracked";
}

TEST_F(AcomAngularVelocityConsistencyTest, theBaseContributionToTheRateIsExactNotApproximated) {
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

TEST_F(AcomAngularVelocityConsistencyTest, theNetworkBeatsTheTwoTrivialBaselines) {
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
  RecordProperty("constantJacobianBaselineError", std::to_string(constantJacobianError));
  LOG(INFO) << "[aCOM] network " << networkError << " vs constant-Jacobian baseline " << constantJacobianError
            << " vs zero-offset baseline 1";

  EXPECT_LT(networkError, 1.0) << "the trained network is no better than assuming the joints do not move the aCOM";
  EXPECT_LT(networkError, constantJacobianError)
      << "the trained network is no better than a single constant Jacobian, which needs no network at all";
}

TEST_F(AcomAngularVelocityConsistencyTest, theExportedJointNamesMatchTheMpcModelJointForJoint) {
  // The network is a function of a VECTOR of joint angles, so a permutation between the order it was trained in and the
  // order the MPC hands it is not an error that degrades gracefully - it evaluates a different robot. Nothing else
  // catches it: the shapes still match, the analytic Jacobian still matches finite differences, the equivariance still
  // holds, and the existing spot check only compares the first three names.
  //
  // It is checked here rather than beside those tests because this file already has the reduced MPC model, and because
  // a permutation would also inflate the error measured above - so when both assertions are in one binary, a failure
  // here explains a failure there.
  ASSERT_EQ(modelSettingsPtr_->robotName, "atlas") << "this test reads the Atlas weight header directly";
  ASSERT_EQ(acom::AcomSirenWeightsAtlas::input_dim, modelSettingsPtr_->mpcModelJointNames.size());
  for (std::size_t joint = 0; joint < acom::AcomSirenWeightsAtlas::input_dim; ++joint) {
    EXPECT_EQ(std::string(acom::AcomSirenWeightsAtlas::joint_names[joint]), modelSettingsPtr_->mpcModelJointNames[joint])
        << "joint " << joint << ": regenerate the weight header against the reduced MPC model";
  }
}

}  // namespace
}  // namespace ocs2::humanoid
