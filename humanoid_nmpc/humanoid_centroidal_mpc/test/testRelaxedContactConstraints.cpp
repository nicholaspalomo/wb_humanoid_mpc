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
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/contact/ContactInputJacobian.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kDerivativeTol = 1.0e-5;

}  // namespace

/**
 * The relaxed complementarity conditions of contact, on the DRC Atlas model: their values, and their analytic linear
 * approximations against finite differences of the values.
 */
class RelaxedContactConstraintsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const CentroidalRobotFiles files = atlasFiles();
    absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files);
    ASSERT_TRUE(config.ok()) << config.status();
    atlas_ = *std::move(config);

    modelSettings_ = std::make_unique<ModelSettings>(
        ModelSettings::Create(atlas_.task, files.urdfFile, "testRelaxedContactConstraints_", /*verbose=*/false).value());
    modelSettings_->recompileLibrariesCppAd = false;
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(
        loadCustomPinocchioInterface(atlas_.task, files.urdfFile, *modelSettings_, /*scaleTotalMass=*/false).value());
    info_ = centroidalModelInfoOf(atlas_, *pinocchioInterface_, *modelSettings_).value();
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
    adRobotModel_ =
        std::make_unique<CentroidalMpcRobotModel<ad_scalar_t>>(*modelSettings_, pinocchioInterface_->toCppAd(), info_.toCppAd());

    const CentroidalModelInfoCppAd infoCppAd = info_.toCppAd();
    mappingCppAd_ = std::make_unique<CentroidalModelPinocchioMappingCppAd>(infoCppAd);
    const PinocchioEndEffectorKinematicsCppAd::update_pinocchio_interface_callback velocityUpdateCallback =
        [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
          const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
          updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd, q);
        };
    const std::string& footName = modelSettings_->contactNames[kContactLeftIndex];
    eeKinematics_ = std::make_unique<PinocchioEndEffectorKinematicsCppAd>(
        *pinocchioInterface_, *mappingCppAd_, std::vector<std::string>{footName}, info_.stateDim, robotModel_->getInputDim(),
        velocityUpdateCallback, absl::StrCat("testRelaxedContact_", footName), modelSettings_->modelFolderCppAd,
        /*recompileLibraries=*/false, /*verbose=*/false);

    // The footprint corners: the geometry the complementarity product and the penetration hinge share. Both terms are
    // built on this one object in CentroidalMpcInterface, so the tests build them that way too.
    const ContactRectangle footprint = contactRectangleFromConfig(atlas_.task.contacts, *modelSettings_, kContactLeftIndex).value();
    std::vector<std::string> cornerFrames;
    cornerFrames.reserve(footprint.getNumberOfContactPoints());
    for (size_t corner = 0; corner < footprint.getNumberOfContactPoints(); ++corner) {
      cornerFrames.push_back(footprint.getPolygonPointFrameName(static_cast<int>(corner)));
    }
    cornerHeights_ =
        std::make_unique<FootprintCornerHeights>(*pinocchioInterface_, *adRobotModel_, std::move(cornerFrames),
                                                 absl::StrCat("testRelaxedContact_", footName, "_footprintCorners"), *modelSettings_);

    state_ = initialStateOf(atlas_.task, *modelSettings_).value();
    input_.setZero(robotModel_->getInputDim());
    // A plausible working point: the foot carries about half the weight and the joints are moving.
    vector6_t wrench = vector6_t::Zero();
    wrench(kWrenchForceZIndex) = 400.0;
    robotModel_->setContactWrench(input_, wrench, kContactLeftIndex);
    for (size_t index = 0; index < robotModel_->getJointDim(); ++index) {
      input_(robotModel_->getJointVelocitiesStartindex() + index) = 0.05 * static_cast<scalar_t>(index % 5) - 0.1;
    }
  }

  /** Numerical dfdx / dfdu of a state-input term at the fixture's working point. */
  template <typename TERM_T>
  void expectDerivativesMatchFiniteDifferences(const TERM_T& term) const {
    expectDerivativesMatchFiniteDifferencesAt(term, state_);
  }

  /** Numerical dfdx / dfdu of a state-input term at an arbitrary state, for comparison with its approximation. */
  template <typename TERM_T>
  void expectDerivativesMatchFiniteDifferencesAt(const TERM_T& term, const vector_t& state) const {
    const PreComputation preComp;
    const VectorFunctionLinearApproximation approximation = term.getLinearApproximation(/*time=*/0.0, state, input_, preComp);
    EXPECT_TRUE(approximation.f.isApprox(term.getValue(/*time=*/0.0, state, input_, preComp), 1.0e-12));

    for (Eigen::Index index = 0; index < state.size(); ++index) {
      vector_t perturbed = state;
      perturbed(index) += kFiniteDifferenceStep;
      const vector_t forward = term.getValue(/*time=*/0.0, perturbed, input_, preComp);
      perturbed(index) -= 2.0 * kFiniteDifferenceStep;
      const vector_t backward = term.getValue(/*time=*/0.0, perturbed, input_, preComp);
      const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
      for (Eigen::Index row = 0; row < numerical.size(); ++row) {
        EXPECT_NEAR(approximation.dfdx(row, index), numerical(row), kDerivativeTol) << "dfdx(" << row << ", " << index << ")";
      }
    }
    for (Eigen::Index index = 0; index < input_.size(); ++index) {
      vector_t perturbed = input_;
      perturbed(index) += kFiniteDifferenceStep;
      const vector_t forward = term.getValue(/*time=*/0.0, state, perturbed, preComp);
      perturbed(index) -= 2.0 * kFiniteDifferenceStep;
      const vector_t backward = term.getValue(/*time=*/0.0, state, perturbed, preComp);
      const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
      for (Eigen::Index row = 0; row < numerical.size(); ++row) {
        EXPECT_NEAR(approximation.dfdu(row, index), numerical(row), kDerivativeTol) << "dfdu(" << row << ", " << index << ")";
      }
    }
  }

  /**
   * State index of the left foot's ankle pitch joint, so a test can tilt the sole.
   *
   * Looked up by joint NAME rather than hard-coded, because the index depends on which joints the task file fixes; a
   * test that pitched the wrong joint would still pass its own assertions while proving nothing about the foot.
   */
  Eigen::Index anklePitchStateIndex() const {
    const std::vector<std::string>& jointNames = modelSettings_->mpcModelJointNames;
    const std::vector<std::string>::const_iterator found = std::find(jointNames.begin(), jointNames.end(), "l_leg_aky");
    EXPECT_TRUE(found != jointNames.end()) << "no joint named l_leg_aky in the MPC model";
    return static_cast<Eigen::Index>(robotModel_->getJointStartindex() + static_cast<size_t>(std::distance(jointNames.begin(), found)));
  }

  // The typed DRC Atlas files.
  CentroidalMpcConfig atlas_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> robotModel_;
  std::unique_ptr<CentroidalMpcRobotModel<ad_scalar_t>> adRobotModel_;
  std::unique_ptr<CentroidalModelPinocchioMappingCppAd> mappingCppAd_;
  std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> eeKinematics_;
  std::unique_ptr<FootprintCornerHeights> cornerHeights_;
  vector_t state_;
  vector_t input_;
};

TEST_F(RelaxedContactConstraintsTest, NormalForceRowReadsTheContactBlock) {
  const vector_t row = normalContactForceRow(*robotModel_, kContactLeftIndex);
  ASSERT_EQ(row.size(), static_cast<Eigen::Index>(robotModel_->getInputDim()));
  EXPECT_NEAR(row.dot(input_), robotModel_->getContactForce(input_, kContactLeftIndex)(2), 1.0e-12);
  // The other foot's block and the joint velocities have no bearing on this foot's normal force.
  vector_t otherFootOnly = vector_t::Zero(robotModel_->getInputDim());
  vector6_t wrench = vector6_t::Zero();
  wrench(kWrenchForceZIndex) = 500.0;
  robotModel_->setContactWrench(otherFootOnly, wrench, kContactRightIndex);
  EXPECT_NEAR(row.dot(otherFootOnly), 0.0, 1.0e-12);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityIsTheForceTimesTheGap) {
  const scalar_t lowestCorner = cornerHeights_->getHeights(state_).minCoeff();
  const scalar_t terrainHeight = lowestCorner - 0.03;  // the lowest corner of the foot is 3 cm above the ground
  const ContactComplementarityConstraint term(*cornerHeights_, *robotModel_, kContactLeftIndex, terrainHeight);

  // The gap is a smoothed minimum of the corner heights, so it is bracketed rather than exact: never less than the
  // true clearance, and never more than log(N) * gapSmoothing above it.
  const scalar_t gap = term.getGap(state_);
  const scalar_t smoothingBound = std::log(static_cast<scalar_t>(cornerHeights_->numCorners())) * term.getGapSmoothing();
  EXPECT_GE(gap, 0.03 - 1.0e-12);
  EXPECT_LE(gap, 0.03 + smoothingBound + 1.0e-12);

  const PreComputation preComp;
  const vector_t value = term.getValue(/*time=*/0.0, state_, input_, preComp);
  ASSERT_EQ(value.size(), 1);
  EXPECT_NEAR(value(0), robotModel_->getContactForce(input_, kContactLeftIndex)(2) * gap, 1.0e-9);

  // A foot on the ground may carry any load, and a foot in the air may carry none: both are free of penalty.
  const ContactComplementarityConstraint onTheGround(*cornerHeights_, *robotModel_, kContactLeftIndex, lowestCorner);
  EXPECT_NEAR(onTheGround.getValue(/*time=*/0.0, state_, input_, preComp)(0), 0.0, smoothingBound * 1.0e3);
  const vector_t noForce = vector_t::Zero(robotModel_->getInputDim());
  EXPECT_NEAR(term.getValue(/*time=*/0.0, state_, noForce, preComp)(0), 0.0, 1.0e-12);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityMeasuresTheLowestCornerNotTheSoleCenter) {
  // The bug this guards: measured at the contact frame, a foot rocked onto one edge reads a positive height while it
  // is carrying load, and the term then charges for a contact that physically exists - and disagrees with the
  // penetration hinge, which had already been moved to the corners.
  vector_t pitched = state_;
  pitched(anklePitchStateIndex()) += 0.25;  // [rad] roll the sole up onto one edge
  const vector_t heights = cornerHeights_->getHeights(pitched);
  const scalar_t spread = heights.maxCoeff() - heights.minCoeff();
  ASSERT_GT(spread, 0.02) << "the ankle pitch must actually tilt the sole or this test proves nothing";

  const scalar_t soleCenter = eeKinematics_->getPosition(pitched).front()(2);
  ASSERT_GT(soleCenter - heights.minCoeff(), 0.01) << "the sole center must sit well above the lowest corner";

  // Ground under the lowest corner: the foot is touching, so the gap - and with it the whole residual - is zero, even
  // though the sole center is a centimeter up.
  const ContactComplementarityConstraint term(*cornerHeights_, *robotModel_, kContactLeftIndex, heights.minCoeff());
  const PreComputation preComp;
  const scalar_t smoothingBound = std::log(static_cast<scalar_t>(cornerHeights_->numCorners())) * term.getGapSmoothing();
  EXPECT_LE(term.getGap(pitched), smoothingBound + 1.0e-12);
  EXPECT_GE(term.getGap(pitched), -1.0e-12);
  const scalar_t normalForce = robotModel_->getContactForce(input_, kContactLeftIndex)(2);
  EXPECT_LE(std::abs(term.getValue(/*time=*/0.0, pitched, input_, preComp)(0)), normalForce * smoothingBound + 1.0e-9);

  // Measured at the sole center the same configuration would have reported a full centimeter of clearance.
  EXPECT_GT((soleCenter - heights.minCoeff()) * normalForce, 1.0);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityDerivativesMatchFiniteDifferences) {
  const scalar_t terrainHeight = cornerHeights_->getHeights(state_).minCoeff() - 0.03;
  expectDerivativesMatchFiniteDifferences(
      ContactComplementarityConstraint(*cornerHeights_, *robotModel_, kContactLeftIndex, terrainHeight));
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityDerivativesMatchFiniteDifferencesOnATiltedFoot) {
  // On a flat foot every softmin weight is 1/N and the gradient is the plain average of the corners', which would
  // hide a wrong contraction. Tilted, the weights are lopsided and the state Jacobian is a genuine convex combination.
  vector_t pitched = state_;
  pitched(anklePitchStateIndex()) += 0.25;
  const scalar_t terrainHeight = cornerHeights_->getHeights(pitched).minCoeff() - 0.03;
  const ContactComplementarityConstraint term(*cornerHeights_, *robotModel_, kContactLeftIndex, terrainHeight);
  expectDerivativesMatchFiniteDifferencesAt(term, pitched);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityResidualIsOneAtTheReferences) {
  // The anchor that makes the weight interpretable: a foot carrying exactly the reference force at exactly the
  // reference height produces a residual of one, so the configured weight is the cost of that worst case - comparable
  // with the slip weight, whose residual is normalized the same way, though NOT with task_space_foot_cost.weights, whose
  // residuals are in meters (humanoid_nmpc/docs/contact_implicit_mpc/README.md, section 4). If this stops holding, the
  // weight in task.textproto silently changes meaning.
  const scalar_t heightReference = 0.08;
  const scalar_t terrainHeight = cornerHeights_->getHeights(state_).minCoeff() - heightReference;
  const scalar_t forceReference = robotModel_->getContactForce(input_, kContactLeftIndex)(2);
  ASSERT_GT(forceReference, 0.0) << "the fixture's input must load this foot for the test to mean anything";

  const ContactComplementarityConstraint term(*cornerHeights_, *robotModel_, kContactLeftIndex, terrainHeight, forceReference,
                                              heightReference);
  const PreComputation preComp;
  EXPECT_NEAR(term.getValue(/*time=*/0.0, state_, input_, preComp)(0), 1.0, 1.0e-2);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityResidualIsNormalizedByBothReferences) {
  const scalar_t terrainHeight = cornerHeights_->getHeights(state_).minCoeff() - 0.03;
  const scalar_t forceReference = 1500.0;
  const scalar_t heightReference = 0.08;
  const ContactComplementarityConstraint term(*cornerHeights_, *robotModel_, kContactLeftIndex, terrainHeight, forceReference,
                                              heightReference);

  const PreComputation preComp;
  const scalar_t normalForce = robotModel_->getContactForce(input_, kContactLeftIndex)(2);
  EXPECT_NEAR(term.getValue(/*time=*/0.0, state_, input_, preComp)(0),
              (normalForce / forceReference) * (term.getGap(state_) / heightReference), 1.0e-9);
  EXPECT_NEAR(term.getForceReference(), forceReference, 1.0e-9);
  EXPECT_NEAR(term.getHeightReference(), heightReference, 1.0e-9);
}

TEST_F(RelaxedContactConstraintsTest, ComplementarityDerivativesMatchFiniteDifferencesUnderNormalization) {
  // The scales multiply both derivative blocks, so a normalization applied to the value but not to the Jacobian would
  // pass every value test above and quietly hand the solver a wrong gradient.
  const scalar_t terrainHeight = cornerHeights_->getHeights(state_).minCoeff() - 0.03;
  expectDerivativesMatchFiniteDifferences(ContactComplementarityConstraint(*cornerHeights_, *robotModel_, kContactLeftIndex, terrainHeight,
                                                                           /*forceReference=*/1500.0, /*heightReference=*/0.08));
}

TEST_F(RelaxedContactConstraintsTest, SlipIsTheForceTimesTheConstrainedTwist) {
  const ForceWeightedSlipConstraint term(*eeKinematics_, *robotModel_, kContactLeftIndex);
  const PreComputation preComp;
  const vector_t value = term.getValue(/*time=*/0.0, state_, input_, preComp);
  ASSERT_EQ(value.size(), 3) << "the two tangential velocities and the spin about the contact normal";

  const vector3_t velocity = eeKinematics_->getVelocity(state_, input_).front();
  const vector3_t angularVelocity = eeKinematics_->getAngularVelocity(state_, input_).front();
  const scalar_t normalForce = robotModel_->getContactForce(input_, kContactLeftIndex)(2);
  EXPECT_NEAR(value(0), normalForce * velocity(0), 1.0e-9);
  EXPECT_NEAR(value(1), normalForce * velocity(1), 1.0e-9);
  // The pivot row is what the schedule-gated zero_velocity constraint used to provide; without it a loaded foot can
  // yaw freely and walks the robot sideways.
  EXPECT_NEAR(value(2), normalForce * angularVelocity(2), 1.0e-9);

  // An unloaded foot may slide and pivot as it likes: the penalty is on the product, not on the motion.
  const vector_t noForce = vector_t::Zero(robotModel_->getInputDim());
  EXPECT_NEAR(term.getValue(/*time=*/0.0, state_, noForce, preComp).norm(), 0.0, 1.0e-12);
}

/** A state and input at which the left foot rocks about one edge of its footprint, and what that motion is. */
struct EdgeRock {
  vector_t state;
  vector_t input;
  /** [rad/s] world-frame angular velocity of the foot, about its own lateral axis. */
  vector3_t angularVelocity;
  /** [m] world-frame position of the contact frame relative to the pivot corner. */
  vector3_t pivotToContactFrame;
  /** [m] fore-aft distance from the pivot corner to the contact frame, in the contact frame. */
  scalar_t leverArm = 0.0;
  /** The x component of the foot's z axis in the world, sin(pitch) for a foot that is only pitched. */
  scalar_t soleTilt = 0.0;
};

TEST_F(RelaxedContactConstraintsTest, SlipMeasuredAtTheSoleCenterChargesEdgeRockingByTheLeverArm) {
  // A reported "rocking foot paradox": the twist is read at the contact frame, so when the foot rocks about an edge the
  // sole center moves even though the contact line does not, and the term charges for slip that is not happening.
  // ForceWeightedSlipConstraint documents the size of it as v_x = omega * a * sin(theta), with a the lever arm from
  // the edge to the contact frame: zero at a flat foot, because the contact frame lies in the plane of the corners,
  // and growing with the tilt. This test drives exactly that motion - a pure rock about the TOE edge, the toe corner
  // held still and the base held still - and checks the term against the formula at a flat foot and at two tilts.
  //
  // A pure ankle-pitch rate would not do: it rotates the foot about the ankle axis, 8 cm above the sole, which slides
  // the sole center even at a flat foot and says nothing about the edge-rocking lever arm.
  const pinocchio::Model& model = pinocchioInterface_->getModel();
  const std::string& footName = modelSettings_->contactNames[kContactLeftIndex];
  const pinocchio::FrameIndex contactFrame = model.getFrameId(footName);
  const std::vector<std::string>& jointNames = modelSettings_->mpcModelJointNames;
  const std::vector<std::string> legJoints = {"l_leg_hpz", "l_leg_hpx", "l_leg_hpy", "l_leg_kny", "l_leg_aky", "l_leg_akx"};
  const ContactRectangle footprint = contactRectangleFromConfig(atlas_.task.contacts, *modelSettings_, kContactLeftIndex).value();
  constexpr scalar_t kRockRate = 1.0;  // [rad/s]

  for (const scalar_t tilt : {0.0, 0.08, 0.3}) {
    EdgeRock rock;
    rock.state = state_;
    rock.state(anklePitchStateIndex()) += tilt;
    const vector_t q = robotModel_->getGeneralizedCoordinates(rock.state);
    pinocchio::Data data(model);
    pinocchio::framesForwardKinematics(model, data, q);

    // The pivot is the lowest corner, or at a flat foot the one furthest forward: the toe edge either way.
    std::string pivotName;
    scalar_t pivotHeight = 0.0;
    scalar_t pivotForward = 0.0;
    for (const std::string& corner : cornerHeights_->frameNames()) {
      const pinocchio::SE3& placement = data.oMf[model.getFrameId(corner)];
      const scalar_t forward = (data.oMf[contactFrame].actInv(placement)).translation().x();
      if (pivotName.empty() || placement.translation().z() < pivotHeight - 1.0e-9 ||
          (std::abs(placement.translation().z() - pivotHeight) <= 1.0e-9 && forward > pivotForward)) {
        pivotName = corner;
        pivotHeight = placement.translation().z();
        pivotForward = forward;
      }
    }
    const pinocchio::FrameIndex pivotFrame = model.getFrameId(pivotName);
    const matrix3_t footRotation = data.oMf[contactFrame].rotation();
    rock.angularVelocity = kRockRate * footRotation.col(1);
    rock.pivotToContactFrame = data.oMf[contactFrame].translation() - data.oMf[pivotFrame].translation();
    rock.leverArm = std::abs((data.oMf[contactFrame].actInv(data.oMf[pivotFrame])).translation().x());
    rock.soleTilt = footRotation(0, 2);

    // Leg joint rates that give the pivot corner zero linear velocity and the foot the rocking angular velocity.
    matrix_t pivotJacobian = matrix_t::Zero(6, model.nv);
    pinocchio::computeFrameJacobian(model, data, q, pivotFrame, pinocchio::LOCAL_WORLD_ALIGNED, pivotJacobian);
    matrix_t legJacobian(6, static_cast<Eigen::Index>(legJoints.size()));
    std::vector<Eigen::Index> legVelocityIndices;
    for (const std::string& joint : legJoints) {
      const Eigen::Index velocityIndex = static_cast<Eigen::Index>(model.joints[model.getJointId(joint)].idx_v());
      ASSERT_EQ(jointNames.at(static_cast<size_t>(velocityIndex) - 6), joint) << "the MPC joint order must be the Pinocchio one";
      legJacobian.col(static_cast<Eigen::Index>(legVelocityIndices.size())) = pivotJacobian.col(velocityIndex);
      legVelocityIndices.push_back(velocityIndex);
    }
    vector6_t desiredTwist;
    desiredTwist << vector3_t::Zero(), rock.angularVelocity;
    const vector_t legRates = legJacobian.fullPivLu().solve(desiredTwist);

    rock.input = vector_t::Zero(robotModel_->getInputDim());
    vector6_t wrench = vector6_t::Zero();
    wrench(kWrenchForceZIndex) = 800.0;
    robotModel_->setContactWrench(rock.input, wrench, kContactLeftIndex);
    vector_t jointRates = vector_t::Zero(static_cast<Eigen::Index>(robotModel_->getJointDim()));
    for (size_t leg = 0; leg < legVelocityIndices.size(); ++leg) {
      jointRates(legVelocityIndices[leg] - 6) = legRates(static_cast<Eigen::Index>(leg));
      rock.input(static_cast<Eigen::Index>(robotModel_->getJointVelocitiesStartindex()) + legVelocityIndices[leg] - 6) =
          legRates(static_cast<Eigen::Index>(leg));
    }

    // Hold the base still: the centroidal momentum is exactly what the moving leg carries.
    PinocchioInterface workingInterface = *pinocchioInterface_;
    updateCentroidalDynamics(workingInterface, info_, q);
    rock.state.head<6>() = getCentroidalMomentumMatrix(workingInterface).rightCols(jointRates.size()) * jointRates / info_.robotMass;

    // The motion really is an edge rock: base still, pivot corner still, the foot turning about its lateral axis.
    CentroidalModelPinocchioMapping mapping(info_);
    mapping.setPinocchioInterface(workingInterface);
    updateCentroidalDynamics(workingInterface, info_, q);
    const vector_t v = mapping.getPinocchioJointVelocity(rock.state, rock.input);
    ASSERT_LT(v.head<6>().norm(), 1.0e-9) << "tilt " << tilt << ": the base must be still";
    pinocchio::forwardKinematics(model, data, q, v);
    const pinocchio::Motion pivotTwist = pinocchio::getFrameVelocity(model, data, pivotFrame, pinocchio::LOCAL_WORLD_ALIGNED);
    ASSERT_LT(pivotTwist.linear().norm(), 1.0e-9) << "tilt " << tilt << ": the pivot corner must be still";
    ASSERT_LT((pivotTwist.angular() - rock.angularVelocity).norm(), 1.0e-9) << "tilt " << tilt;

    // The lever arm is the configured footprint's: the toe edge sits at x_max of the contact rectangle.
    EXPECT_NEAR(rock.leverArm, footprint.getBounds().x_max, 1.0e-9) << "tilt " << tilt;

    // The term, with the load and the references at one, reads the contact frame's tangential velocity directly...
    const ForceWeightedSlipConstraint term(*eeKinematics_, *robotModel_, kContactLeftIndex, /*forceReference=*/800.0,
                                           /*velocityReference=*/1.0, /*angularVelocityReference=*/1.0);
    const PreComputation preComp;
    const vector_t value = term.getValue(/*time=*/0.0, rock.state, rock.input, preComp);
    const vector3_t rigidBodyVelocity = rock.angularVelocity.cross(rock.pivotToContactFrame);
    EXPECT_NEAR(value(0), rigidBodyVelocity.x(), 1.0e-8) << "tilt " << tilt;
    EXPECT_NEAR(value(1), rigidBodyVelocity.y(), 1.0e-8) << "tilt " << tilt;

    // ...which is omega * a * sin(theta): the magnitude the class comment quotes, zero at a flat foot.
    EXPECT_NEAR(std::abs(value(0)), kRockRate * rock.leverArm * std::abs(rock.soleTilt), 1.0e-8) << "tilt " << tilt;
    EXPECT_NEAR(std::abs(rock.soleTilt), std::abs(std::sin(tilt)), 1.0e-3) << "the nominal foot is expected to be flat";
    if (tilt == 0.0) {
      EXPECT_LT(std::abs(value(0)), 1.0e-8) << "a contact frame off the plane of the sole would slide under a flat edge rock";
    } else {
      EXPECT_GT(std::abs(value(0)), 0.5 * kRockRate * rock.leverArm * std::sin(tilt)) << "tilt " << tilt;
    }
  }
}

TEST_F(RelaxedContactConstraintsTest, CornerHeightsFollowTheGeometryTheyWereBuiltOnDespiteACachedLibrary) {
  // Audit finding A26. The corner placements are constants inside the CppAD tape, the library name is the cache key,
  // and every robot ships recompileLibrariesCppAd: false - so a footprint edited under an unchanged name used to load
  // the library of the OLD corners, and both contact-implicit terms kept evaluating them without a word.
  ASSERT_FALSE(modelSettings_->recompileLibrariesCppAd) << "the cache is what is under test";
  const std::string& footName = modelSettings_->contactNames[kContactLeftIndex];
  pinocchio::Model moved = pinocchioInterface_->getModel();
  const pinocchio::FrameIndex movedCorner = moved.getFrameId(cornerHeights_->frameNames().front());
  moved.frames[movedCorner].placement.translation() += vector3_t(0.0, 0.0, 0.01);
  const PinocchioInterface movedInterface(moved);

  // The same name prefix the fixture's object was built with, whose library is therefore on disk.
  const FootprintCornerHeights movedHeights(movedInterface, *adRobotModel_, cornerHeights_->frameNames(),
                                            absl::StrCat("testRelaxedContact_", footName, "_footprintCorners"), *modelSettings_);
  EXPECT_NE(movedHeights.modelName(), cornerHeights_->modelName()) << "the geometry must be part of the library's name";

  const vector_t q = robotModel_->getGeneralizedCoordinates(state_);
  pinocchio::Data data(moved);
  pinocchio::framesForwardKinematics(moved, data, q);
  const vector_t heights = movedHeights.getHeights(state_);
  for (size_t corner = 0; corner < movedHeights.numCorners(); ++corner) {
    EXPECT_NEAR(heights(static_cast<Eigen::Index>(corner)), data.oMf[moved.getFrameId(movedHeights.frameNames()[corner])].translation().z(),
                1.0e-9)
        << "corner " << corner << " is not where the geometry it was built on puts it";
  }
  // Positive control: the edit is visible, so an object reading the old library would fail the loop above.
  EXPECT_GT(std::abs(heights(0) - cornerHeights_->getHeights(state_)(0)), 0.005);

  // And an unchanged geometry keeps its name, so the cache still works for everything that did not change.
  const FootprintCornerHeights sameGeometry(*pinocchioInterface_, *adRobotModel_, cornerHeights_->frameNames(),
                                            absl::StrCat("testRelaxedContact_", footName, "_footprintCorners"), *modelSettings_);
  EXPECT_EQ(sameGeometry.modelName(), cornerHeights_->modelName());
}

TEST_F(RelaxedContactConstraintsTest, SlipLeavesTheRockingRatesFree) {
  // Rolling the foot about its heel and toe edges under load is how a heel-to-toe strike happens, so those two rates
  // must not be priced: only three of the six twist components are constrained.
  const ForceWeightedSlipConstraint term(*eeKinematics_, *robotModel_, kContactLeftIndex);
  EXPECT_EQ(term.getNumConstraints(0.0), 3U);
}

TEST_F(RelaxedContactConstraintsTest, SlipDerivativesMatchFiniteDifferences) {
  expectDerivativesMatchFiniteDifferences(ForceWeightedSlipConstraint(*eeKinematics_, *robotModel_, kContactLeftIndex));
}

TEST_F(RelaxedContactConstraintsTest, SlipScalesEachRowInItsOwnUnits) {
  // Two rows are linear velocities and the third is a yaw rate. Penalizing them together without separate references
  // declares one radian per second to be exactly as bad as one meter per second, which is not a statement about the
  // robot but about the choice of SI units.
  const scalar_t forceReference = 1500.0;
  const scalar_t velocityReference = 0.3;
  const scalar_t angularVelocityReference = 1.0;
  const ForceWeightedSlipConstraint term(*eeKinematics_, *robotModel_, kContactLeftIndex, forceReference, velocityReference,
                                         angularVelocityReference);

  const PreComputation preComp;
  const vector_t value = term.getValue(/*time=*/0.0, state_, input_, preComp);
  const vector3_t velocity = eeKinematics_->getVelocity(state_, input_).front();
  const vector3_t angularVelocity = eeKinematics_->getAngularVelocity(state_, input_).front();
  const scalar_t normalizedForce = robotModel_->getContactForce(input_, kContactLeftIndex)(2) / forceReference;

  EXPECT_NEAR(value(0), normalizedForce * velocity(0) / velocityReference, 1.0e-9);
  EXPECT_NEAR(value(1), normalizedForce * velocity(1) / velocityReference, 1.0e-9);
  EXPECT_NEAR(value(2), normalizedForce * angularVelocity(2) / angularVelocityReference, 1.0e-9);

  // And the yaw row must not pick up the linear scale: with the two references different, swapping them would change
  // the third component, which is exactly the bug this guards.
  EXPECT_GT(std::abs(1.0 / velocityReference - 1.0 / angularVelocityReference), 1.0e-9)
      << "choose different references or this assertion proves nothing";
}

TEST_F(RelaxedContactConstraintsTest, SlipDerivativesMatchFiniteDifferencesUnderNormalization) {
  expectDerivativesMatchFiniteDifferences(ForceWeightedSlipConstraint(*eeKinematics_, *robotModel_, kContactLeftIndex,
                                                                      /*forceReference=*/1500.0, /*velocityReference=*/0.3,
                                                                      /*angularVelocityReference=*/1.0));
}

TEST_F(RelaxedContactConstraintsTest, PenetrationIsTheHeightOfEveryCornerAboveTheTerrain) {
  const vector_t heights = cornerHeights_->getHeights(state_);
  const GroundPenetrationConstraint term(*cornerHeights_, heights.minCoeff() - 0.05);
  const PreComputation preComp;
  ASSERT_EQ(term.getNumConstraints(0.0), cornerHeights_->numCorners());
  ASSERT_GT(cornerHeights_->numCorners(), 1U) << "one row per footprint corner, not one for the sole center";

  const vector_t value = term.getValue(/*time=*/0.0, state_, preComp);
  for (Eigen::Index corner = 0; corner < value.size(); ++corner) {
    EXPECT_NEAR(value(corner), heights(corner) - (heights.minCoeff() - 0.05), 1.0e-9) << "corner " << corner;
  }

  // Below the terrain the constraint is violated, which is what the one-sided hinge around it prices.
  const GroundPenetrationConstraint sunken(*cornerHeights_, heights.maxCoeff() + 0.02);
  EXPECT_LT(sunken.getValue(/*time=*/0.0, state_, preComp).maxCoeff(), 0.0);

  const VectorFunctionLinearApproximation approximation = term.getLinearApproximation(/*time=*/0.0, state_, preComp);
  for (Eigen::Index index = 0; index < state_.size(); ++index) {
    vector_t perturbed = state_;
    perturbed(index) += kFiniteDifferenceStep;
    const vector_t forward = term.getValue(/*time=*/0.0, perturbed, preComp);
    perturbed(index) -= 2.0 * kFiniteDifferenceStep;
    const vector_t backward = term.getValue(/*time=*/0.0, perturbed, preComp);
    const vector_t numerical = (forward - backward) / (2.0 * kFiniteDifferenceStep);
    for (Eigen::Index corner = 0; corner < numerical.size(); ++corner) {
      EXPECT_NEAR(approximation.dfdx(corner, index), numerical(corner), kDerivativeTol) << "dfdx(" << corner << ", " << index << ")";
    }
  }
}

TEST_F(RelaxedContactConstraintsTest, PenetrationAndComplementarityShareOneGeometry) {
  // The two terms are the two halves of one condition - no load above the ground, no foot below it - so they have to
  // agree on where the foot is. They disagreed for a while: the hinge read the corners while the product read the sole
  // center, and a foot up on its heel was therefore "airborne" and "touching" at the same time.
  vector_t pitched = state_;
  pitched(anklePitchStateIndex()) += 0.25;
  const vector_t heights = cornerHeights_->getHeights(pitched);
  const scalar_t terrainHeight = heights.minCoeff();

  const GroundPenetrationConstraint penetration(*cornerHeights_, terrainHeight);
  const ContactComplementarityConstraint complementarity(*cornerHeights_, *robotModel_, kContactLeftIndex, terrainHeight);
  const PreComputation preComp;

  // The lowest corner is exactly on the ground: the hinge is at its boundary and the gap has closed.
  EXPECT_NEAR(penetration.getValue(/*time=*/0.0, pitched, preComp).minCoeff(), 0.0, 1.0e-9);
  const scalar_t smoothingBound = std::log(static_cast<scalar_t>(cornerHeights_->numCorners())) * complementarity.getGapSmoothing();
  EXPECT_LE(complementarity.getGap(pitched), smoothingBound + 1.0e-12);
}

}  // namespace ocs2::humanoid
