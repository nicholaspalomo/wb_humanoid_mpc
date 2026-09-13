#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <cstddef>
#include <memory>
#include <string>

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

namespace ocs2::humanoid {

namespace {

/// Generalized coordinates start after the 6D normalized momentum in the
/// centroidal state.
constexpr std::size_t kGeneralizedCoordinatesStartIndex = 6;

/// Index of the first base orientation coordinate within the Pinocchio q vector.
constexpr Eigen::Index kBaseOrientationOffset = 3;

/// Number of generalized coordinates occupied by the floating base.
constexpr Eigen::Index kGeneralizedBaseDim = 6;

}  // namespace

/**
 * Tests the pieces of ComAndAcomTrackingCost that can be exercised without
 * standing up a full MPC: the reduced Pinocchio model the cost is built against,
 * and the two Jacobians the Gauss-Newton approximation is assembled from.
 *
 * The fixture deliberately builds the same *reduced* model the MPC uses, via
 * ModelSettings and createCustomPinocchioInterface, rather than the full URDF
 * model. Using the full model would make the joint count disagree with the
 * trained aCOM weights and quietly turn every test below into a skip.
 */
class ComAndAcomTrackingCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    taskFile_ = configDir + "/config/mpc/task.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";

    modelSettingsPtr_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "testComAndAcomTrackingCost", false);
    pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettingsPtr_));

    const auto& model = pinocchioInterfacePtr_->getModel();
    nJoints_ = model.nq - kGeneralizedBaseDim;

    info_.generalizedCoordinatesNum = model.nq;
    info_.actuatedDofNum = nJoints_;
    info_.stateDim = kGeneralizedCoordinatesStartIndex + model.nq;
    info_.inputDim = nJoints_;
    info_.robotMass = pinocchio::computeTotalMass(model);

    acomPtr_ = AngularCenterOfMass::createForRobot(modelSettingsPtr_->robotName);
  }

  /** A configuration with the base slightly off level, so T(theta) != identity. */
  vector_t makeTestConfiguration() const {
    vector_t q = vector_t::Zero(info_.generalizedCoordinatesNum);
    q(2) = 0.8;                             // Base height.
    q(kBaseOrientationOffset + 0) = 0.30;   // Yaw.
    q(kBaseOrientationOffset + 1) = 0.20;   // Pitch.
    q(kBaseOrientationOffset + 2) = -0.15;  // Roll.
    q.tail(nJoints_) = 0.2 * vector_t::Ones(nJoints_);
    return q;
  }

  std::string taskFile_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> modelSettingsPtr_;
  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  std::unique_ptr<AngularCenterOfMass> acomPtr_;
  CentroidalModelInfo info_;
  int nJoints_ = 0;
};

/**
 * The trained weights must be indexed by exactly the joints the MPC optimizes
 * over. A mismatch here means the network is being fed a different robot.
 */
TEST_F(ComAndAcomTrackingCostTest, AcomInputDimMatchesReducedModel) {
  EXPECT_EQ(acomPtr_->getInputDim(), static_cast<std::size_t>(nJoints_))
      << "Regenerate AcomSirenWeights" << modelSettingsPtr_->robotName << ".h against the reduced MPC model.";
  EXPECT_EQ(modelSettingsPtr_->mpcModelJointNames.size(), static_cast<std::size_t>(nJoints_));
}

/**
 * Regression test for the CoM Jacobian used by getQuadraticApproximation.
 *
 * The floating base of this model is a Translation joint composed with a
 * SphericalZYX joint, so Pinocchio's tangent vector v is literally dq/dt and
 * jacobianCenterOfMass already returns d(com)/dq. An earlier version of the cost
 * chained an Euler-rate to angular-velocity mapping T(theta) onto the base
 * orientation columns, which double-counts that mapping and, at theta = 0, even
 * swaps the yaw and roll columns. Finite differences pin the correct convention.
 */
TEST_F(ComAndAcomTrackingCostTest, ComJacobianIsWithRespectToEulerAnglesNotAngularVelocity) {
  const vector_t q = makeTestConfiguration();
  auto& data = pinocchioInterfacePtr_->getData();
  const auto& model = pinocchioInterfacePtr_->getModel();

  pinocchio::jacobianCenterOfMass(model, data, q);
  const matrix_t J_com = data.Jcom;
  ASSERT_EQ(J_com.cols(), q.size()) << "Jcom must have one column per generalized coordinate (nq == nv for this model).";

  const scalar_t eps = 1e-7;
  matrix_t J_fd = matrix_t::Zero(3, q.size());
  for (Eigen::Index i = 0; i < q.size(); ++i) {
    vector_t qPlus = q;
    vector_t qMinus = q;
    qPlus(i) += eps;
    qMinus(i) -= eps;
    const vector3_t comPlus = pinocchio::centerOfMass(model, data, qPlus);
    const vector3_t comMinus = pinocchio::centerOfMass(model, data, qMinus);
    J_fd.col(i) = (comPlus - comMinus) / (2.0 * eps);
  }

  EXPECT_TRUE(J_com.isApprox(J_fd, 1e-5)) << "Max abs error: " << (J_com - J_fd).cwiseAbs().maxCoeff();
}

/**
 * The aCOM Jacobian rows must follow the centroidal state's ZYX Euler ordering,
 * while the network natively emits XYZ. Getting this backwards silently swaps
 * the roll and yaw tracking weights.
 */
TEST_F(ComAndAcomTrackingCostTest, AcomJacobianRowsAreInZyxOrder) {
  const vector_t q = makeTestConfiguration();
  const vector_t qJoints = q.tail(nJoints_);

  const matrix_t J_xyz = acomPtr_->computeJointOffsetJacobian(qJoints);
  const matrix_t J_zyx = acomJacobianXyzToZyx(J_xyz);

  EXPECT_TRUE(J_zyx.row(0).isApprox(J_xyz.row(2)));
  EXPECT_TRUE(J_zyx.row(1).isApprox(J_xyz.row(1)));
  EXPECT_TRUE(J_zyx.row(2).isApprox(J_xyz.row(0)));

  // computeAcomJacobian must apply the same reordering to its joint block.
  const matrix_t J_acom = acomPtr_->computeAcomJacobian(q);
  ASSERT_EQ(J_acom.cols(), q.size());
  EXPECT_TRUE(J_acom.leftCols<3>().isZero(1e-15));
  EXPECT_TRUE(J_acom.block<3, 3>(0, kBaseOrientationOffset).isApprox(matrix_t::Identity(3, 3), 1e-15));
  EXPECT_TRUE(J_acom.rightCols(nJoints_).isApprox(J_zyx));
}

/**
 * The whole-body aCOM orientation must reduce to the base Euler triple plus the
 * reordered joint offset, in the state's ZYX convention.
 */
TEST_F(ComAndAcomTrackingCostTest, AcomOrientationIsBaseEulerPlusReorderedOffset) {
  const vector_t q = makeTestConfiguration();
  const vector3_t expected = q.segment<3>(kBaseOrientationOffset) + acomXyzToZyx(acomPtr_->computeJointOrientationOffset(q.tail(nJoints_)));
  EXPECT_TRUE(acomPtr_->computeAcomOrientation(q).isApprox(expected, 1e-12));
}

/**
 * The reduced model must place the CoM somewhere physically sensible, which
 * catches a mis-built or mis-scaled Pinocchio model before it reaches the cost.
 */
TEST_F(ComAndAcomTrackingCostTest, CenterOfMassIsPhysicallyPlausible) {
  vector_t state = vector_t::Zero(info_.stateDim);
  state(kGeneralizedCoordinatesStartIndex + 2) = 0.8;

  CentroidalModelPinocchioMapping mapping(info_);
  const vector_t q = mapping.getPinocchioJointPosition(state);
  ASSERT_EQ(q.size(), info_.generalizedCoordinatesNum);

  const vector3_t com = pinocchio::centerOfMass(pinocchioInterfacePtr_->getModel(), pinocchioInterfacePtr_->getData(), q);
  EXPECT_NEAR(com[0], 0.0, 0.5);
  EXPECT_NEAR(com[1], 0.0, 0.5);
  EXPECT_GT(com[2], 0.3);
  EXPECT_GT(info_.robotMass, 0.0);
}

}  // namespace ocs2::humanoid
