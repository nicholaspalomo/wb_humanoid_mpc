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

#include <ocs2_core/PreComputation.h>

#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/contact/ContactInputJacobian.h"
#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"
#include "support/DrcAtlasContactTestModel.h"
#include "support/FiniteDifferenceChecks.h"

namespace ocs2::humanoid {
namespace {

constexpr size_t kFoot = CONTACT_LEFT_INDEX;
constexpr scalar_t kForceReference = 1600.0;  // [N] about the DRC Atlas's weight
constexpr scalar_t kHeightReference = 0.08;   // [m]
constexpr scalar_t kVelocityReference = 0.3;  // [m/s]
constexpr scalar_t kAngularReference = 1.0;   // [rad/s]

/**
 * The contact-implicit terms against the BASIS-VECTOR input parameterization - the one the shipped DRC Atlas actually
 * runs (`useContactBasisVectorInputs: true`).
 *
 * testRelaxedContactConstraints.cpp builds these terms on the wrench-space CentroidalMpcRobotModel only. That model
 * stores the contact wrench in the WORLD frame, so `getContactForce(input, i)(2)` there is the world-vertical force;
 * under BasisInputsModelDecorator the input stores non-negative scalings of a LOCAL contact-frame wrench-cone basis,
 * and the same accessor returns the SOLE-normal force instead. The terms were therefore never exercised in the
 * parameterization they ship in, and the frame they use was decided by accident rather than by design.
 *
 * What makes both frames correct is that these terms use the normal force as a LOAD INDICATOR rather than as a
 * physical force: a number that vanishes exactly when the foot carries no wrench. These tests pin that property, which
 * is the one a future input parameterization could silently break.
 */
class ContactImplicitTermsOnBasisInputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = std::make_unique<DrcAtlasContactTestModel>("testContactImplicitBasis_");
    basisKinematics_ = model_->makeEndEffectorKinematics(kFoot, model_->basisModel().getInputDim());
    cornerHeights_ = model_->makeCornerHeights(kFoot, "_footprintCorners");
    state_ = model_->nominalState();
    // A plausible working point: the foot carries about half the body weight through its basis scalings, and the
    // joints are moving so the slip term's velocity rows are non-trivial.
    input_ = model_->makeInput(model_->basisModel(), kFoot, 800.0);
  }

  std::unique_ptr<DrcAtlasContactTestModel> model_;
  std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> basisKinematics_;
  std::unique_ptr<FootprintCornerHeights> cornerHeights_;
  vector_t state_;
  vector_t input_;
};

// ---------------------------------------------------------------------------------------------------------------
// The load indicator.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactImplicitTermsOnBasisInputsTest, theNormalForceRowIsANonNegativeLoadIndicatorOnTheBasisModel) {
  const vector_t row = normalContactForceRow(model_->basisModel(), kFoot);
  ASSERT_EQ(row.size(), static_cast<long>(model_->basisModel().getInputDim()));

  // Every generator of ContactWrenchConeBasisMatrix is built with a local normal force of exactly 1, so the row is all
  // ones over this foot's block. That makes the indicator the SUM of the foot's scalings, which is zero if and only if
  // every one of them is - i.e. if and only if the whole wrench is zero. This is an exact "does this foot carry load"
  // test, which is precisely what the complementarity and slip products need.
  const size_t start = model_->basisModel().getContactWrenchStartIndices(kFoot);
  const size_t numBasis = model_->numBasisPerFoot();
  for (size_t index = 0; index < numBasis; ++index) {
    EXPECT_NEAR(row(static_cast<long>(start + index)), 1.0, 1e-12) << "basis " << index;
  }
  // Everything else - the joint velocities and the other foot - contributes nothing.
  EXPECT_NEAR(row.sum(), static_cast<scalar_t>(numBasis), 1e-12);
  EXPECT_TRUE((row.array() >= -1e-12).all()) << "a negative entry would let a positive scaling reduce the indicator";
}

TEST_F(ContactImplicitTermsOnBasisInputsTest, theForceJacobianIsTheParameterizationAndTheNormalRowIsItsThirdRow) {
  // normalContactForceRow() is one row of contactForceInputJacobian(), and the friction cone linearises through the
  // other two. Pinning the relationship keeps the three terms reading one description of the parameterization: a
  // future input layout that broke it would otherwise break them one at a time and in different ways.
  const matrix_t jacobian = contactForceInputJacobian(model_->basisModel(), kFoot);
  const vector_t row = normalContactForceRow(model_->basisModel(), kFoot);

  ASSERT_EQ(jacobian.rows(), 3);
  ASSERT_EQ(jacobian.cols(), static_cast<long>(model_->basisModel().getInputDim()));
  EXPECT_TRUE(jacobian.row(2).transpose().isApprox(row, 1e-12)) << "the normal row must be the third row of the Jacobian";

  // Over this foot's block it is the force rows of B_local: every generator is a unit normal force applied somewhere
  // on the footprint, and the friction-pyramid generators tilt it.
  const long start = static_cast<long>(model_->basisModel().getContactWrenchStartIndices(kFoot));
  const long width = static_cast<long>(model_->numBasisPerFoot());
  EXPECT_TRUE(jacobian.block(2, start, 1, width).isOnes(1e-12)) << jacobian.block(2, start, 1, width);
  EXPECT_GT(jacobian.block(0, start, 2, width).cwiseAbs().maxCoeff(), 0.1) << "the tangential rows must not be dead";

  // And nothing outside the block moves this foot's force - not the joint velocities, not the other foot.
  matrix_t outsideTheBlock = jacobian;
  outsideTheBlock.middleCols(start, width).setZero();
  EXPECT_TRUE(outsideTheBlock.isZero(1e-12)) << outsideTheBlock;
}

TEST_F(ContactImplicitTermsOnBasisInputsTest, theForceJacobianIsTheIdentityBlockOnTheWrenchModel) {
  // The same probe on the wrench-space model returns the identity that the hand-written 3x3 assumed - which is why
  // that assumption survived: it is right in exactly one of the two parameterizations this repository ships.
  const matrix_t jacobian = contactForceInputJacobian(model_->wrenchModel(), kFoot);
  const long start = static_cast<long>(model_->wrenchModel().getContactForceStartIndices(kFoot));

  ASSERT_EQ(jacobian.rows(), 3);
  EXPECT_TRUE(jacobian.middleCols(start, 3).isIdentity(1e-12)) << jacobian.middleCols(start, 3);
  matrix_t outsideTheForce = jacobian;
  outsideTheForce.middleCols(start, 3).setZero();
  EXPECT_TRUE(outsideTheForce.isZero(1e-12)) << "the moment and the joint velocities cannot move the force";
}

TEST_F(ContactImplicitTermsOnBasisInputsTest, theIndicatorVanishesExactlyWhenTheFootCarriesNoWrench) {
  const vector_t row = normalContactForceRow(model_->basisModel(), kFoot);
  const size_t start = model_->basisModel().getContactWrenchStartIndices(kFoot);

  vector_t noLoad = vector_t::Zero(model_->basisModel().getInputDim());
  EXPECT_NEAR(row.dot(noLoad), 0.0, 1e-12);
  EXPECT_TRUE(model_->basisModel().getContactWrench(noLoad, kFoot).isZero(1e-12));

  // Any single non-zero scaling makes both the indicator and the wrench non-zero.
  for (size_t index = 0; index < model_->numBasisPerFoot(); ++index) {
    vector_t oneRay = vector_t::Zero(model_->basisModel().getInputDim());
    oneRay(static_cast<long>(start + index)) = 0.5;
    EXPECT_GT(row.dot(oneRay), 0.0) << "basis " << index;
    EXPECT_FALSE(model_->basisModel().getContactWrench(oneRay, kFoot).isZero(1e-12)) << "basis " << index;
  }

  // The other foot's scalings do not register as load on this one.
  vector_t otherFootLoaded =
      model_->makeInput(model_->basisModel(), kFoot == CONTACT_LEFT_INDEX ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX, 800.0);
  EXPECT_NEAR(row.dot(otherFootLoaded), 0.0, 1e-9);
}

// ---------------------------------------------------------------------------------------------------------------
// The terms themselves, on the basis model. Values, and analytic derivatives against finite differences - the check
// that catches a scale or a frame applied to the value but not to the Jacobian.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactImplicitTermsOnBasisInputsTest, complementarityIsTheNormalisedProductOnTheBasisModel) {
  const scalar_t lowestCorner = cornerHeights_->getHeights(state_).minCoeff();
  const scalar_t terrainHeight = lowestCorner - 0.03;  // the lowest corner of the foot is 3 cm above the ground
  const ContactComplementarityConstraint term(*cornerHeights_, model_->basisModel(), kFoot, terrainHeight, kForceReference,
                                              kHeightReference);
  const PreComputation preComp;

  const vector_t row = normalContactForceRow(model_->basisModel(), kFoot);
  const scalar_t expected = (row.dot(input_) / kForceReference) * (term.getGap(state_) / kHeightReference);
  EXPECT_NEAR(term.getValue(0.0, state_, input_, preComp)(0), expected, 1e-9);
  // The gap the term reports is the clearance of the lowest corner, up to the smoothing of the minimum.
  EXPECT_GE(term.getGap(state_), 0.03 - 1e-12);
  EXPECT_LE(term.getGap(state_), 0.03 + std::log(4.0) * term.getGapSmoothing() + 1e-12);

  // A foot on the ground costs nothing however hard it presses...
  const ContactComplementarityConstraint onGround(*cornerHeights_, model_->basisModel(), kFoot, lowestCorner, kForceReference,
                                                  kHeightReference);
  EXPECT_NEAR(onGround.getValue(0.0, state_, input_, preComp)(0), 0.0, 1e-2);
  // ...and a foot carrying nothing costs nothing however high it is.
  const vector_t noLoad = vector_t::Zero(model_->basisModel().getInputDim());
  EXPECT_NEAR(term.getValue(0.0, state_, noLoad, preComp)(0), 0.0, 1e-12);
}

TEST_F(ContactImplicitTermsOnBasisInputsTest, complementarityDerivativesMatchFiniteDifferencesOnTheBasisModel) {
  const scalar_t terrainHeight = cornerHeights_->getHeights(state_).minCoeff() - 0.03;
  const ContactComplementarityConstraint term(*cornerHeights_, model_->basisModel(), kFoot, terrainHeight, kForceReference,
                                              kHeightReference);
  expectStateInputDerivativesMatchFiniteDifferences(term, state_, input_);

  // And on a tilted foot, where the softmin weights are lopsided rather than the uniform 1/N a flat sole produces.
  vector_t pitchedState = state_;
  pitchedState(model_->anklePitchStateIndex(kFoot)) += 0.15;
  const ContactComplementarityConstraint pitchedTerm(*cornerHeights_, model_->basisModel(), kFoot,
                                                     cornerHeights_->getHeights(pitchedState).minCoeff() - 0.03, kForceReference,
                                                     kHeightReference);
  expectStateInputDerivativesMatchFiniteDifferences(pitchedTerm, pitchedState, input_);
}

TEST_F(ContactImplicitTermsOnBasisInputsTest, slipConstrainsThreeTwistRowsOnTheBasisModel) {
  const ForceWeightedSlipConstraint term(*basisKinematics_, model_->basisModel(), kFoot, kForceReference, kVelocityReference,
                                         kAngularReference);
  const PreComputation preComp;
  const vector_t value = term.getValue(0.0, state_, input_, preComp);
  ASSERT_EQ(value.size(), 3);

  const vector3_t velocity = basisKinematics_->getVelocity(state_, input_).front();
  const vector3_t angularVelocity = basisKinematics_->getAngularVelocity(state_, input_).front();
  const scalar_t load = normalContactForceRow(model_->basisModel(), kFoot).dot(input_) / kForceReference;
  EXPECT_NEAR(value(0), load * velocity(0) / kVelocityReference, 1e-9);
  EXPECT_NEAR(value(1), load * velocity(1) / kVelocityReference, 1e-9);
  EXPECT_NEAR(value(2), load * angularVelocity(2) / kAngularReference, 1e-9);

  // A foot carrying nothing is free to move however it likes.
  const vector_t noLoad = vector_t::Zero(model_->basisModel().getInputDim());
  EXPECT_TRUE(term.getValue(0.0, state_, noLoad, preComp).isZero(1e-12));
}

TEST_F(ContactImplicitTermsOnBasisInputsTest, slipDerivativesMatchFiniteDifferencesOnTheBasisModel) {
  const ForceWeightedSlipConstraint term(*basisKinematics_, model_->basisModel(), kFoot, kForceReference, kVelocityReference,
                                         kAngularReference);
  expectStateInputDerivativesMatchFiniteDifferences(term, state_, input_);
}

// ---------------------------------------------------------------------------------------------------------------
// Ground penetration at the footprint corners, which is what makes a rocking foot safe.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactImplicitTermsOnBasisInputsTest, penetrationIsCheckedAtEveryCornerAndItsDerivativesAreRight) {
  const GroundPenetrationConstraint term(*cornerHeights_, 0.0);
  ASSERT_EQ(cornerHeights_->numCorners(), 4U);
  EXPECT_EQ(term.getNumPoints(), 4U);
  EXPECT_EQ(term.getNumConstraints(0.0), 4U);

  const PreComputation preComp;
  const vector_t value = term.getValue(0.0, state_, preComp);
  ASSERT_EQ(value.size(), 4);
  const vector_t heights = cornerHeights_->getHeights(state_);
  for (long corner = 0; corner < 4; ++corner) {
    EXPECT_NEAR(value(corner), heights(corner), 1e-12);
  }

  // At the nominal state the foot is flat, so every corner sits at the sole centre's height and a centre-only term
  // would look perfectly adequate. PITCH the foot and the whole point of finding A6 appears: the lowest corner drops
  // well below the centre, so a term watching only the centre lets the toe through the floor. The Atlas footprint is
  // 0.12 m fore and aft, so 0.15 rad of ankle pitch should move a corner by about 0.12 * sin(0.15) = 18 mm.
  const std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> soleCentre =
      model_->makeEndEffectorKinematics(kFoot, model_->basisModel().getInputDim());
  EXPECT_NEAR(value.maxCoeff() - value.minCoeff(), 0.0, 1e-6) << "the nominal foot is expected to be flat";

  vector_t pitchedState = state_;
  pitchedState(model_->anklePitchStateIndex(kFoot)) += 0.15;
  const vector_t pitchedHeights = term.getValue(0.0, pitchedState, preComp);
  const scalar_t pitchedCentreHeight = soleCentre->getPosition(pitchedState).front()(2);
  EXPECT_GT(pitchedHeights.maxCoeff() - pitchedHeights.minCoeff(), 0.01)
      << "pitching the foot must separate the corner heights, or the corners are not being read";
  EXPECT_LT(pitchedHeights.minCoeff(), pitchedCentreHeight - 0.005)
      << "the lowest corner must sit below the sole centre: that gap is exactly what a centre-only term missed";

  expectStateDerivativesMatchFiniteDifferences(term, state_);
  expectStateDerivativesMatchFiniteDifferences(term, pitchedState);
}

TEST_F(ContactImplicitTermsOnBasisInputsTest, theGapAndThePenetrationRowsComeFromTheSamePoints) {
  // One FootprintCornerHeights feeds both terms, so there is no way for them to end up measuring different geometry -
  // which is what happened when the hinge was moved to the corners and the product was left on the sole centre.
  vector_t pitchedState = state_;
  pitchedState(model_->anklePitchStateIndex(kFoot)) += 0.15;
  const vector_t heights = cornerHeights_->getHeights(pitchedState);

  const GroundPenetrationConstraint penetration(*cornerHeights_, heights.minCoeff());
  const ContactComplementarityConstraint complementarity(*cornerHeights_, model_->basisModel(), kFoot, heights.minCoeff(), kForceReference,
                                                         kHeightReference);
  const PreComputation preComp;

  // The lowest corner is on the ground, so the hinge sits exactly at its boundary...
  EXPECT_NEAR(penetration.getValue(0.0, pitchedState, preComp).minCoeff(), 0.0, 1e-9);
  // ...and the product agrees that the foot is touching, to within the smoothing of the minimum.
  EXPECT_LE(complementarity.getGap(pitchedState), std::log(4.0) * complementarity.getGapSmoothing() + 1e-12);
  EXPECT_GE(complementarity.getGap(pitchedState), -1e-12);

  // Measured at the sole centre instead, the same configuration would have claimed a centimetre of clearance while
  // the foot was carrying load - the term would have charged full price for a contact that physically exists.
  const std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> soleCentre =
      model_->makeEndEffectorKinematics(kFoot, model_->basisModel().getInputDim());
  EXPECT_GT(soleCentre->getPosition(pitchedState).front()(2) - heights.minCoeff(), 0.005);
}

}  // namespace
}  // namespace ocs2::humanoid
