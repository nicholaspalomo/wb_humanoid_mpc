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
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/spatial/explog.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The configuration tangent of the quaternion state
 * (humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.2): the
 * MPC's state manifold (stateManifoldOf()) moves the configuration slots q =
 * [p_W, xi, q_j] of the state as Pinocchio moves the configuration of the root
 * joint Composite(Translation, Spherical) the MPC models are built on
 * (loadCustomPinocchioInterface()), so the state's tangent rows of the
 * configuration are Pinocchio's velocity v = [v_W, w_B, dq_j] and every
 * Pinocchio v-Jacobian is the derivative along the retraction:
 *   - pinocchio::integrate(q, v) is the retraction q (+) v, and
 * pinocchio::difference(q0, q1) the manifold's difference;
 *   - finite differences of the contact frames' placements and of the CoM along
 * q (+) dv are Pinocchio's frame Jacobians (LOCAL_WORLD_ALIGNED: the
 * world-frame linear and angular velocity) and its CoM Jacobian, column for
 * column;
 *   - the centroidal flow map's configuration rate
 * (computeConfigurationDerivative(), [dp_W, 1/2 G(xi) w_B, dq_j]) is the rate
 * of pinocchio::integrate(q, t v) at t = 0, and on a stored quaternion of
 * another length its quaternion rows scale with it and stay orthogonal to it,
 * so |xi| is constant to first order (design decision D5); on the shipped DRC
 * Atlas (centroidal) and whole-body G1 at random attitudes, a pitch of 90
 * degrees among them, and for both quaternions of each rotation. Builds no
 * CppAD model.
 */
namespace ocs2::humanoid {
namespace {

/** A shipped MPC, from the repository root. */
struct ShippedRobot {
  const char* absl_nonnull name;
  const char* absl_nonnull taskFile;
  const char* absl_nonnull urdfFile;
  StateInputLayout::Mpc mpc;
};

// LINT.IfChange(configuration_tangent_robots)
constexpr std::array<ShippedRobot, 2> kRobots = {{
    {.name = "drc_atlas_centroidal_mpc",
     .taskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/"
                 "task.textproto",
     .urdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal},
    {.name = "g1_wb_mpc",
     .taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .mpc = StateInputLayout::Mpc::kWholeBody},
}};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:configuration_tangent_data)

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kFiniteDifferenceTolerance = 1.0e-7;

/** A vector of `size` entries uniform in [-halfWidth, halfWidth]. */
vector_t uniformVector(Eigen::Index size,
                       scalar_t halfWidth,
                       std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> distribution(-halfWidth, halfWidth);
  vector_t vector(size);
  for (Eigen::Index i = 0; i < size; ++i) vector(i) = distribution(generator);
  return vector;
}

/** The largest magnitude of the entries of `m`. */
scalar_t maxAbs(const matrix_t& m) {
  return m.size() == 0 ? 0.0 : m.cwiseAbs().maxCoeff();
}

/** The fixture: a shipped robot's settings, its production Pinocchio model, its
 * quaternion-layout model and manifold. */
class ConfigurationTangentTest : public ::testing::TestWithParam<ShippedRobot> {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> taskFile =
        robot::resolveResourcePath(GetParam().taskFile);
    const absl::StatusOr<std::string> urdfFile =
        robot::resolveResourcePath(GetParam().urdfFile);
    ASSERT_TRUE(taskFile.ok() && urdfFile.ok())
        << "the shipped files are not in the runfiles";
    absl::StatusOr<ModelSettings> settings = ModelSettings::Create(
        *taskFile, *urdfFile, "testConfigurationTangent_", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    settings_ = std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface =
        loadCustomPinocchioInterface(*taskFile, *urdfFile, *settings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ =
        std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<test::QuaternionLayoutRobotModel>(*settings_,
                                                                GetParam().mpc);
    absl::StatusOr<std::shared_ptr<const StateManifold>> manifold =
        stateManifoldOf(*model_);
    ASSERT_TRUE(manifold.ok()) << manifold.status();
    manifold_ = *std::move(manifold);
    ASSERT_EQ(static_cast<size_t>(pinocchio().nq),
              model_->getConfigurationDim());
    ASSERT_EQ(static_cast<size_t>(pinocchio().nv), model_->getVelocityDim());
    ASSERT_EQ(pinocchio().nq, pinocchio().nv + 1)
        << "the root is not Composite(Translation, Spherical)";
  }

  const pinocchio::Model& pinocchio() const {
    return pinocchioInterface_->getModel();
  }

  /** The first tangent row of the configuration: its first state slot, as every
   * block before it is Euclidean. */
  Eigen::Index configurationTangentStart() const {
    return static_cast<Eigen::Index>(model_->getBaseStartindex());
  }

  /** A neutral state whose configuration slots are `q`. */
  vector_t stateOf(const vector_t& q) const {
    vector_t state = model_->getNeutralState();
    model_->setGeneralizedCoordinates(state, q);
    return state;
  }

  /** The configuration of state(q) (+) dv: the state's retraction along the
   * configuration's tangent rows by `dv`. */
  vector_t retracted(const vector_t& q, const vector_t& dv) const {
    vector_t direction =
        vector_t::Zero(static_cast<Eigen::Index>(manifold_->getTangentDim()));
    direction.segment(configurationTangentStart(), dv.size()) = dv;
    vector_t state;
    manifold_->retract(stateOf(q), direction, /*alpha=*/1.0, state);
    return state.segment(static_cast<Eigen::Index>(model_->getBaseStartindex()),
                         pinocchio().nq);
  }

  /** A configuration of uniform joints and position, its quaternion `xi`. */
  vector_t configurationAt(const vector4_t& xi, std::mt19937& generator) const {
    vector_t q = uniformVector(pinocchio().nq, /*halfWidth=*/0.5, generator);
    q.segment<4>(kBaseTranslationDim) = xi;
    return q;
  }

  /** Random unit quaternions, a pitch of 90 degrees and both quaternions of
   * each rotation. */
  std::vector<vector4_t> attitudes(std::mt19937& generator) const {
    std::vector<vector4_t> result;
    for (size_t i = 0; i < 4; ++i)
      result.emplace_back(
          uniformVector(/*size=*/4, /*halfWidth=*/1.0, generator).normalized());
    result.push_back(
        quaternionFromEulerZyx(vector3_t(0.7, 0.5 * std::numbers::pi, -0.4)));
    const size_t count = result.size();
    for (size_t i = 0; i < count; ++i) result.emplace_back(-result[i]);
    return result;
  }

  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<test::QuaternionLayoutRobotModel> model_;
  std::shared_ptr<const StateManifold> manifold_;
};

TEST_P(
    ConfigurationTangentTest,
    PinocchioIntegrateIsTheRetractionAndPinocchioDifferenceTheManifoldsDifference) {
  std::mt19937 generator(3);
  for (const vector4_t& xi : attitudes(generator)) {
    const vector_t q = configurationAt(xi, generator);
    // Rotations below pi, where the difference is the shortest path back.
    for (const scalar_t magnitude : {1.0e-3, 0.3, 1.0}) {
      const vector_t dv = uniformVector(pinocchio().nv, magnitude, generator);
      const vector_t integrated = pinocchio::integrate(pinocchio(), q, dv);
      const vector_t moved = retracted(q, dv);
      EXPECT_LE(maxAbs(integrated - moved), 1.0e-14)
          << "|dv| " << magnitude << ", xi " << xi.transpose();
      // The retraction keeps the quaternion's sign, as Pinocchio does: xi (x)
      // Exp(dtheta).
      EXPECT_GT(integrated.segment<4>(kBaseTranslationDim)
                    .dot(moved.segment<4>(kBaseTranslationDim)),
                0.0);

      const vector_t pinocchioDifference =
          pinocchio::difference(pinocchio(), q, integrated);
      const vector_t manifoldDifference =
          manifold_->difference(stateOf(q), stateOf(integrated))
              .segment(configurationTangentStart(), pinocchio().nv);
      EXPECT_LE(maxAbs(pinocchioDifference - manifoldDifference), 1.0e-12)
          << "|dv| " << magnitude << ", xi " << xi.transpose();
      EXPECT_LE(maxAbs(manifoldDifference - dv),
                1.0e-12 * std::max(1.0, maxAbs(dv)))
          << "the difference undoes the retraction";
    }
  }
}

TEST_P(
    ConfigurationTangentTest,
    FiniteDifferencesOfTheFramesAndTheComAlongTheRetractionArePinocchiosJacobians) {
  const pinocchio::Model& model = pinocchio();
  pinocchio::Data data(model);
  pinocchio::Data plusData(model);
  pinocchio::Data minusData(model);
  std::mt19937 generator(5);
  for (const vector4_t& xi : attitudes(generator)) {
    const vector_t q = configurationAt(xi, generator);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    const matrix_t comJacobian =
        pinocchio::jacobianCenterOfMass(model, data, q);
    std::vector<matrix_t> frameJacobians;
    for (const std::string& contact : settings_->contactNames) {
      matrix_t jacobian = matrix_t::Zero(6, model.nv);
      pinocchio::getFrameJacobian(model, data, model.getFrameId(contact),
                                  pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
      frameJacobians.push_back(jacobian);
    }

    matrix_t comFiniteDifferences(3, model.nv);
    std::vector<matrix_t> frameFiniteDifferences(frameJacobians.size(),
                                                 matrix_t(6, model.nv));
    for (Eigen::Index column = 0; column < model.nv; ++column) {
      const vector_t step =
          kFiniteDifferenceStep * vector_t::Unit(model.nv, column);
      const vector_t plus = retracted(q, step);
      const vector_t minus = retracted(q, -step);
      comFiniteDifferences.col(column) =
          (pinocchio::centerOfMass(model, plusData, plus) -
           pinocchio::centerOfMass(model, minusData, minus)) /
          (2.0 * kFiniteDifferenceStep);
      pinocchio::framesForwardKinematics(model, plusData, plus);
      pinocchio::framesForwardKinematics(model, minusData, minus);
      for (size_t contact = 0; contact < frameJacobians.size(); ++contact) {
        const pinocchio::FrameIndex frame =
            model.getFrameId(settings_->contactNames[contact]);
        const pinocchio::SE3& plusPlacement = plusData.oMf[frame];
        const pinocchio::SE3& minusPlacement = minusData.oMf[frame];
        frameFiniteDifferences[contact].col(column).head<3>() =
            (plusPlacement.translation() - minusPlacement.translation()) /
            (2.0 * kFiniteDifferenceStep);
        // The world-frame rotation between the two placements, as
        // LOCAL_WORLD_ALIGNED's angular rows read it.
        frameFiniteDifferences[contact].col(column).tail<3>() =
            pinocchio::log3(matrix3_t(plusPlacement.rotation() *
                                      minusPlacement.rotation().transpose())) /
            (2.0 * kFiniteDifferenceStep);
      }
    }
    EXPECT_LE(maxAbs(comFiniteDifferences - comJacobian),
              kFiniteDifferenceTolerance * std::max(1.0, maxAbs(comJacobian)))
        << "xi " << xi.transpose();
    for (size_t contact = 0; contact < frameJacobians.size(); ++contact) {
      EXPECT_LE(
          maxAbs(frameFiniteDifferences[contact] - frameJacobians[contact]),
          kFiniteDifferenceTolerance *
              std::max(1.0, maxAbs(frameJacobians[contact])))
          << settings_->contactNames[contact] << ", xi " << xi.transpose();
    }
  }
}

TEST_P(ConfigurationTangentTest,
       TheCentroidalFlowMapsConfigurationRateIsTheRateOfPinocchioIntegrate) {
  if (GetParam().mpc != StateInputLayout::Mpc::kCentroidal) {
    GTEST_SKIP() << "computeConfigurationDerivative() is the configuration "
                    "rate of the centroidal flow map";
  }
  const CentroidalModelInfo info = centroidal_model::createCentroidalModelInfo(
      *pinocchioInterface_, CentroidalModelType::FullCentroidalDynamics,
      vector_t::Zero(static_cast<Eigen::Index>(settings_->mpc_joint_dim)),
      settings_->contactNames3DoF, settings_->contactNames6DoF);
  ASSERT_EQ(info.stateDim, model_->getStateDim());
  std::mt19937 generator(9);
  for (const vector4_t& xi : attitudes(generator)) {
    const vector_t q = configurationAt(xi, generator);
    const vector_t v =
        uniformVector(pinocchio().nv, /*halfWidth=*/1.0, generator);
    const vector_t rate =
        computeConfigurationDerivative<scalar_t>(info, stateOf(q), v);
    ASSERT_EQ(rate.size(), pinocchio().nq);
    const vector_t finiteDifferences =
        (pinocchio::integrate(pinocchio(), q, kFiniteDifferenceStep * v) -
         pinocchio::integrate(pinocchio(), q, -kFiniteDifferenceStep * v)) /
        (2.0 * kFiniteDifferenceStep);
    EXPECT_LE(maxAbs(rate - finiteDifferences),
              kFiniteDifferenceTolerance * std::max(1.0, maxAbs(rate)))
        << "xi " << xi.transpose();

    // The raw xi of the state: twice as long, twice the quaternion rate, still
    // orthogonal to xi.
    vector_t longer = q;
    longer.segment<4>(kBaseTranslationDim) *= 2.0;
    const vector_t longerRate =
        computeConfigurationDerivative<scalar_t>(info, stateOf(longer), v);
    EXPECT_LE(maxAbs(vector_t(longerRate.segment<4>(kBaseTranslationDim) -
                              2.0 * rate.segment<4>(kBaseTranslationDim))),
              1.0e-15);
    EXPECT_LE(std::abs(longerRate.segment<4>(kBaseTranslationDim)
                           .dot(longer.segment<4>(kBaseTranslationDim))),
              1.0e-14);
    EXPECT_EQ(vector_t(longerRate.head<3>()), vector_t(rate.head<3>()));
    EXPECT_EQ(vector_t(longerRate.tail(
                  static_cast<Eigen::Index>(settings_->mpc_joint_dim))),
              vector_t(rate.tail(
                  static_cast<Eigen::Index>(settings_->mpc_joint_dim))));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ShippedRobots,
    ConfigurationTangentTest,
    ::testing::ValuesIn(kRobots),
    [](const ::testing::TestParamInfo<ShippedRobot>& info) {
      return std::string(info.param.name);
    });

}  // namespace
}  // namespace ocs2::humanoid
