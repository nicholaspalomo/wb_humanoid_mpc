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

#include <cmath>
#include <memory>
#include <string>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1e-6;
constexpr scalar_t kDerivativeTol = 1e-5;

}  // namespace

/**
 * The relaxed complementarity conditions of contact, on the DRC Atlas model: their values, and their analytic linear
 * approximations against finite differences of the values.
 */
class RelaxedContactConstraintsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    const std::string taskFile = configDir + "/config/mpc/task.yaml";
    const std::string referenceFile = configDir + "/config/command/reference.yaml";
    const std::string urdfFile = descriptionDir + "/urdf/atlas.urdf";

    modelSettings_ = std::make_unique<ModelSettings>(taskFile, urdfFile, "testRelaxedContactConstraints_", false);
    modelSettings_->recompileLibrariesCppAd = false;
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(createCustomPinocchioInterface(taskFile, urdfFile, *modelSettings_, false));
    info_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile), modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF);
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);

    const CentroidalModelInfoCppAd infoCppAd = info_.toCppAd();
    mappingCppAd_ = std::make_unique<CentroidalModelPinocchioMappingCppAd>(infoCppAd);
    const PinocchioEndEffectorKinematicsCppAd::update_pinocchio_interface_callback velocityUpdateCallback =
        [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
          const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
          updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd, q);
        };
    const std::string& footName = modelSettings_->contactNames[CONTACT_LEFT_INDEX];
    eeKinematics_ = std::make_unique<PinocchioEndEffectorKinematicsCppAd>(
        *pinocchioInterface_, *mappingCppAd_, std::vector<std::string>{footName}, info_.stateDim, robotModel_->getInputDim(),
        velocityUpdateCallback, "testRelaxedContact_" + footName, modelSettings_->modelFolderCppAd, false, false);

    state_.setZero(info_.stateDim);
    loadData::loadEigenMatrix(taskFile, "initialState", state_);
    input_.setZero(robotModel_->getInputDim());
    // A plausible working point: the foot carries about half the weight and the joints are moving.
    vector6_t wrench = vector6_t::Zero();
    wrench(WRENCH_FORCE_Z_INDEX) = 400.0;
    robotModel_->setContactWrench(input_, wrench, CONTACT_LEFT_INDEX);
    for (size_t index = 0; index < robotModel_->getJointDim(); ++index) {
      input_(robotModel_->getJointVelocitiesStartindex() + index) = 0.05 * static_cast<scalar_t>(index % 5) - 0.1;
    }
  }

  /** Numerical dfdx / dfdu of a state-input term, for comparison with its analytic approximation. */
  template <typename TERM_T>
  void expectDerivativesMatchFiniteDifferences(const TERM_T& term) const {
    const PreComputation preComp;
    const VectorFunctionLinearApproximation approximation = term.getLinearApproximation(0.0, state_, input_, preComp);
    EXPECT_TRUE(approximation.f.isApprox(term.getValue(0.0, state_, input_, preComp), 1e-12));

    for (long index = 0; index < state_.size(); ++index) {
      vector_t perturbed = state_;
      perturbed(index) += kFiniteDifferenceStep;
      const vector_t forward = term.getValue(0.0, perturbed, input_, preComp);
      perturbed(index) -= 2.0 * kFiniteDifferenceStep;
      const vector_t backward = term.getValue(0.0, perturbed, input_, preComp);
      const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
      for (long row = 0; row < numerical.size(); ++row) {
        EXPECT_NEAR(approximation.dfdx(row, index), numerical(row), kDerivativeTol) << "dfdx(" << row << ", " << index << ")";
      }
    }
    for (long index = 0; index < input_.size(); ++index) {
      vector_t perturbed = input_;
      perturbed(index) += kFiniteDifferenceStep;
      const vector_t forward = term.getValue(0.0, state_, perturbed, preComp);
      perturbed(index) -= 2.0 * kFiniteDifferenceStep;
      const vector_t backward = term.getValue(0.0, state_, perturbed, preComp);
      const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
      for (long row = 0; row < numerical.size(); ++row) {
        EXPECT_NEAR(approximation.dfdu(row, index), numerical(row), kDerivativeTol) << "dfdu(" << row << ", " << index << ")";
      }
    }
  }

  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> robotModel_;
  std::unique_ptr<CentroidalModelPinocchioMappingCppAd> mappingCppAd_;
  std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> eeKinematics_;
  vector_t state_;
  vector_t input_;
};

TEST_F(RelaxedContactConstraintsTest, NormalForceRowReadsTheContactBlock) {
  const vector_t row = normalContactForceRow(*robotModel_, CONTACT_LEFT_INDEX);
  ASSERT_EQ(row.size(), static_cast<long>(robotModel_->getInputDim()));
  EXPECT_NEAR(row.dot(input_), robotModel_->getContactForce(input_, CONTACT_LEFT_INDEX)(2), 1e-12);
  // The other foot's block and the joint velocities have no bearing on this foot's normal force.
  vector_t otherFootOnly = vector_t::Zero(robotModel_->getInputDim());
  vector6_t wrench = vector6_t::Zero();
  wrench(WRENCH_FORCE_Z_INDEX) = 500.0;
  robotModel_->setContactWrench(otherFootOnly, wrench, CONTACT_RIGHT_INDEX);
  EXPECT_NEAR(row.dot(otherFootOnly), 0.0, 1e-12);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityIsTheForceTimesTheHeight) {
  const scalar_t footHeight = eeKinematics_->getPosition(state_).front()(2);
  const scalar_t terrainHeight = footHeight - 0.03;  // the foot is 3 cm above the ground
  const ContactComplementarityConstraint term(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, terrainHeight);

  const PreComputation preComp;
  const vector_t value = term.getValue(0.0, state_, input_, preComp);
  ASSERT_EQ(value.size(), 1);
  EXPECT_NEAR(value(0), robotModel_->getContactForce(input_, CONTACT_LEFT_INDEX)(2) * 0.03, 1e-9);

  // A foot on the ground may carry any load, and a foot in the air may carry none: both are free of penalty.
  const ContactComplementarityConstraint onTheGround(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, footHeight);
  EXPECT_NEAR(onTheGround.getValue(0.0, state_, input_, preComp)(0), 0.0, 1e-9);
  const vector_t noForce = vector_t::Zero(robotModel_->getInputDim());
  EXPECT_NEAR(term.getValue(0.0, state_, noForce, preComp)(0), 0.0, 1e-12);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityDerivativesMatchFiniteDifferences) {
  const scalar_t terrainHeight = eeKinematics_->getPosition(state_).front()(2) - 0.03;
  expectDerivativesMatchFiniteDifferences(
      ContactComplementarityConstraint(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, terrainHeight));
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityResidualIsOneAtTheReferences) {
  // The anchor that makes the weight interpretable: a foot carrying exactly the reference force at exactly the
  // reference height produces a residual of one, so the configured weight is the cost of that worst case and can be
  // compared directly with the task-space weights the term competes against. If this stops holding, the weight in
  // task.yaml silently changes meaning.
  const scalar_t footHeight = eeKinematics_->getPosition(state_).front()(2);
  const scalar_t heightReference = 0.08;
  const scalar_t terrainHeight = footHeight - heightReference;
  const scalar_t forceReference = robotModel_->getContactForce(input_, CONTACT_LEFT_INDEX)(2);
  ASSERT_GT(forceReference, 0.0) << "the fixture's input must load this foot for the test to mean anything";

  const ContactComplementarityConstraint term(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, terrainHeight, forceReference,
                                              heightReference);
  const PreComputation preComp;
  EXPECT_NEAR(term.getValue(0.0, state_, input_, preComp)(0), 1.0, 1e-9);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityResidualIsNormalisedByBothReferences) {
  const scalar_t footHeight = eeKinematics_->getPosition(state_).front()(2);
  const scalar_t terrainHeight = footHeight - 0.03;
  const scalar_t forceReference = 1500.0;
  const scalar_t heightReference = 0.08;
  const ContactComplementarityConstraint term(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, terrainHeight, forceReference,
                                              heightReference);

  const PreComputation preComp;
  const scalar_t normalForce = robotModel_->getContactForce(input_, CONTACT_LEFT_INDEX)(2);
  EXPECT_NEAR(term.getValue(0.0, state_, input_, preComp)(0), (normalForce / forceReference) * (0.03 / heightReference), 1e-9);
  EXPECT_NEAR(term.getForceReference(), forceReference, 1e-9);
  EXPECT_NEAR(term.getHeightReference(), heightReference, 1e-9);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityDerivativesMatchFiniteDifferencesUnderNormalisation) {
  // The scales multiply both derivative blocks, so a normalisation applied to the value but not to the Jacobian would
  // pass every value test above and quietly hand the solver a wrong gradient.
  const scalar_t terrainHeight = eeKinematics_->getPosition(state_).front()(2) - 0.03;
  expectDerivativesMatchFiniteDifferences(
      ContactComplementarityConstraint(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, terrainHeight, 1500.0, 0.08));
}

TEST_F(RelaxedContactConstraintsTest, SlipIsTheForceTimesTheConstrainedTwist) {
  const ForceWeightedSlipConstraint term(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX);
  const PreComputation preComp;
  const vector_t value = term.getValue(0.0, state_, input_, preComp);
  ASSERT_EQ(value.size(), 3) << "the two tangential velocities and the spin about the contact normal";

  const vector3_t velocity = eeKinematics_->getVelocity(state_, input_).front();
  const vector3_t angularVelocity = eeKinematics_->getAngularVelocity(state_, input_).front();
  const scalar_t normalForce = robotModel_->getContactForce(input_, CONTACT_LEFT_INDEX)(2);
  EXPECT_NEAR(value(0), normalForce * velocity(0), 1e-9);
  EXPECT_NEAR(value(1), normalForce * velocity(1), 1e-9);
  // The pivot row is what the schedule-gated zero_velocity constraint used to provide; without it a loaded foot can
  // yaw freely and walks the robot sideways.
  EXPECT_NEAR(value(2), normalForce * angularVelocity(2), 1e-9);

  // An unloaded foot may slide and pivot as it likes: the penalty is on the product, not on the motion.
  const vector_t noForce = vector_t::Zero(robotModel_->getInputDim());
  EXPECT_NEAR(term.getValue(0.0, state_, noForce, preComp).norm(), 0.0, 1e-12);
}

TEST_F(RelaxedContactConstraintsTest, SlipLeavesTheRockingRatesFree) {
  // Rolling the foot about its heel and toe edges under load is how a heel-to-toe strike happens, so those two rates
  // must not be priced: only three of the six twist components are constrained.
  const ForceWeightedSlipConstraint term(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX);
  EXPECT_EQ(term.getNumConstraints(0.0), 3U);
}

TEST_F(RelaxedContactConstraintsTest, SlipDerivativesMatchFiniteDifferences) {
  expectDerivativesMatchFiniteDifferences(ForceWeightedSlipConstraint(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX));
}

TEST_F(RelaxedContactConstraintsTest, SlipScalesEachRowInItsOwnUnits) {
  // Two rows are linear velocities and the third is a yaw rate. Penalising them together without separate references
  // declares one radian per second to be exactly as bad as one metre per second, which is not a statement about the
  // robot but about the choice of SI units.
  const scalar_t forceReference = 1500.0;
  const scalar_t velocityReference = 0.3;
  const scalar_t angularVelocityReference = 1.0;
  const ForceWeightedSlipConstraint term(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, forceReference, velocityReference,
                                         angularVelocityReference);

  const PreComputation preComp;
  const vector_t value = term.getValue(0.0, state_, input_, preComp);
  const vector3_t velocity = eeKinematics_->getVelocity(state_, input_).front();
  const vector3_t angularVelocity = eeKinematics_->getAngularVelocity(state_, input_).front();
  const scalar_t normalisedForce = robotModel_->getContactForce(input_, CONTACT_LEFT_INDEX)(2) / forceReference;

  EXPECT_NEAR(value(0), normalisedForce * velocity(0) / velocityReference, 1e-9);
  EXPECT_NEAR(value(1), normalisedForce * velocity(1) / velocityReference, 1e-9);
  EXPECT_NEAR(value(2), normalisedForce * angularVelocity(2) / angularVelocityReference, 1e-9);

  // And the yaw row must not pick up the linear scale: with the two references different, swapping them would change
  // the third component, which is exactly the bug this guards.
  EXPECT_GT(std::abs(1.0 / velocityReference - 1.0 / angularVelocityReference), 1e-9)
      << "choose different references or this assertion proves nothing";
}

TEST_F(RelaxedContactConstraintsTest, SlipDerivativesMatchFiniteDifferencesUnderNormalisation) {
  expectDerivativesMatchFiniteDifferences(ForceWeightedSlipConstraint(*eeKinematics_, *robotModel_, CONTACT_LEFT_INDEX, 1500.0, 0.3, 1.0));
}

TEST_F(RelaxedContactConstraintsTest, PenetrationIsTheHeightAboveTheTerrain) {
  const scalar_t footHeight = eeKinematics_->getPosition(state_).front()(2);
  const GroundPenetrationConstraint term(*eeKinematics_, footHeight - 0.05);
  const PreComputation preComp;
  EXPECT_NEAR(term.getValue(0.0, state_, preComp)(0), 0.05, 1e-9);

  // Below the terrain the constraint is violated, which is what the relaxed barrier around it prices.
  const GroundPenetrationConstraint sunken(*eeKinematics_, footHeight + 0.02);
  EXPECT_LT(sunken.getValue(0.0, state_, preComp)(0), 0.0);

  const VectorFunctionLinearApproximation approximation = term.getLinearApproximation(0.0, state_, preComp);
  for (long index = 0; index < state_.size(); ++index) {
    vector_t perturbed = state_;
    perturbed(index) += kFiniteDifferenceStep;
    const scalar_t forward = term.getValue(0.0, perturbed, preComp)(0);
    perturbed(index) -= 2.0 * kFiniteDifferenceStep;
    const scalar_t backward = term.getValue(0.0, perturbed, preComp)(0);
    EXPECT_NEAR(approximation.dfdx(0, index), (forward - backward) / (2.0 * kFiniteDifferenceStep), kDerivativeTol) << "dfdx " << index;
  }
}

}  // namespace ocs2::humanoid
