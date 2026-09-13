#include <gtest/gtest.h>

#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_core/cost/CostFunctionLinearApproximation.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

using namespace ocs2;
using namespace ocs2::humanoid;

class ComAndAcomTrackingCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Generate dummy URDF model path
    std::string urdfPath = "/home/nico-palomo/workspace/wb_humanoid_mpc/robot_models/drc_atlas/drc_atlas_description/urdf/drc_atlas.urdf";

    // We can just use the createDefaultPinocchioInterface if possible, but actually we just need a dummy interface
    try {
      pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(createDefaultPinocchioInterface(urdfPath));
    } catch (...) {
      GTEST_SKIP() << "URDF not found, skipping test.";
      return;
    }

    info_.stateDim = 36;
    info_.inputDim = 24;
    info_.generalizedCoordinatesNum = 37;
    info_.actuatedDofNum = 30;
    info_.generalizedCoordinatesIndex = 6;
    info_.robotMass = 100.0;

    referenceManagerPtr_ = std::make_unique<SwitchedModelReferenceManager>(nullptr, nullptr);

    matrix_t Q_com = matrix_t::Identity(3, 3);
    matrix_t Q_acom = matrix_t::Identity(3, 3);

    costPtr_ = std::make_unique<ComAndAcomTrackingCost>(Q_com, Q_acom, *pinocchioInterfacePtr_, info_, *referenceManagerPtr_);
  }

  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
  CentroidalModelInfo info_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManagerPtr_;
  std::unique_ptr<ComAndAcomTrackingCost> costPtr_;
};

TEST_F(ComAndAcomTrackingCostTest, testQuadraticApproximation) {
  if (!costPtr_) return;

  vector_t state = vector_t::Random(info_.stateDim);
  vector_t stateRef = vector_t::Random(info_.stateDim);

  // Normalize quaternion in state if needed? Centroidal uses euler angles, so state is fine.

  TargetTrajectories targetTrajectories;
  targetTrajectories.timeTrajectory.push_back(0.0);
  targetTrajectories.stateTrajectory.push_back(stateRef);
  targetTrajectories.inputTrajectory.push_back(vector_t::Random(info_.inputDim));

  PreComputation preComp;

  auto approx = costPtr_->getQuadraticApproximation(0.0, state, targetTrajectories, preComp);

  EXPECT_EQ(approx.dfdx.size(), info_.stateDim);
  EXPECT_EQ(approx.dfdxx.rows(), info_.stateDim);
  EXPECT_EQ(approx.dfdxx.cols(), info_.stateDim);

  // Finite difference test
  const scalar_t eps = 1e-5;
  vector_t dfdx_fd = vector_t::Zero(info_.stateDim);

  for (size_t i = 0; i < info_.stateDim; ++i) {
    vector_t state_plus = state;
    state_plus(i) += eps;
    scalar_t cost_plus = costPtr_->getValue(0.0, state_plus, targetTrajectories, preComp);

    vector_t state_minus = state;
    state_minus(i) -= eps;
    scalar_t cost_minus = costPtr_->getValue(0.0, state_minus, targetTrajectories, preComp);

    dfdx_fd(i) = (cost_plus - cost_minus) / (2.0 * eps);
  }

  EXPECT_TRUE(approx.dfdx.isApprox(dfdx_fd, 1e-3)) << "Analytical: " << approx.dfdx.transpose() << "\nFD: " << dfdx_fd.transpose();
}
