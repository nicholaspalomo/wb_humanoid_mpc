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

#include <memory>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>

#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactMomentXYConstraintCppAd.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactInputJacobian.h"
#include "humanoid_common_mpc/cost/StateInputQuadraticCost.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "support/DrcAtlasContactTestModel.h"
#include "support/FiniteDifferenceChecks.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTime = DrcAtlasContactTestModel::kQueryTime;
constexpr size_t kSwingFoot = CONTACT_LEFT_INDEX;

/**
 * Audit finding A1. Every contact cone in this repository switched itself off while the mode schedule called a foot a
 * swing foot. That gate was sound for exactly one reason: the hard `zero_wrench` constraint had already pinned the
 * swinging foot's wrench to zero, so there was nothing for a cone to bound. The contact-implicit formulation removes
 * `zero_wrench` - loadMpcFormulationTasks() insists on it - and the gate then leaves a foot the schedule calls a swing
 * foot with an unbounded wrench: adhesion, unlimited friction, a centre of pressure anywhere. On the shipped DRC Atlas
 * it is worse still, because `useContactBasisVectorInputs: true` makes the non-negativity of the basis scalings the
 * only bound there is, and that term was gated too.
 *
 * Every test below fails on the pre-fix code.
 */
class ContactConstraintScheduleGatingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = std::make_unique<DrcAtlasContactTestModel>("testContactConstraintScheduleGating_");
    // The schedule calls the left foot a swing foot; the solver, under the contact-implicit formulation, may disagree.
    model_->setSwing(kSwingFoot);
  }

  std::unique_ptr<ContactWrenchConeConstraint> makeWrenchCone(bool scheduleGated) const {
    return std::make_unique<ContactWrenchConeConstraint>(model_->referenceManager(), model_->contactRectangle(kSwingFoot), kSwingFoot,
                                                         model_->pinocchioInterface(), model_->wrenchModel(),
                                                         ContactWrenchConeConstraint::Config(), scheduleGated);
  }

  std::unique_ptr<FrictionForceConeConstraint> makeFrictionCone(bool scheduleGated) const {
    return std::make_unique<FrictionForceConeConstraint>(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                                         model_->wrenchModel(), scheduleGated);
  }

  /**
   * The centre-of-pressure constraint. It is the one of the four whose rows are already homogeneous in the wrench, so
   * only its gate and its penalty change - which is exactly why it is the one most easily left untested.
   */
  std::unique_ptr<ContactMomentXYConstraintCppAd> makeCentreOfPressure(bool scheduleGated) const {
    return std::make_unique<ContactMomentXYConstraintCppAd>(
        model_->referenceManager(), model_->contactRectangle(kSwingFoot), kSwingFoot, model_->pinocchioInterface(), model_->adWrenchModel(),
        "testContactConstraintScheduleGating_cop", model_->modelSettings(), scheduleGated);
  }

  std::unique_ptr<BasisScalingNonNegativityConstraint> makeLambdaBarrier(bool scheduleGated) const {
    return std::make_unique<BasisScalingNonNegativityConstraint>(
        model_->referenceManager(), kSwingFoot, model_->basisModel().getContactWrenchStartIndices(kSwingFoot), model_->numBasisPerFoot(),
        PieceWisePolynomialBarrierPenalty::Config(1.0e-2, 1.0e-3), scheduleGated);
  }

  std::unique_ptr<DrcAtlasContactTestModel> model_;
};

// ---------------------------------------------------------------------------------------------------------------
// The predicate that decides the gate.
// ---------------------------------------------------------------------------------------------------------------

TEST(ContactConstraintScheduleGatingPredicate, followsZeroWrenchRatherThanTheContactImplicitTerms) {
  MpcFormulationTasks tasks;
  // Nothing listed at all: no zero_wrench, so nothing may gate itself. A task file may drop zero_wrench without
  // listing the contact-implicit terms, and keying the gate off those terms would leave exactly that case unguarded.
  EXPECT_FALSE(contactConstraintsAreScheduleGated(tasks));
  EXPECT_FALSE(usesContactImplicitFormulation(tasks));

  tasks.hardConstraints.insert(MpcHardConstraintType::ZeroWrench);
  EXPECT_TRUE(contactConstraintsAreScheduleGated(tasks));

  tasks.hardConstraints.erase(MpcHardConstraintType::ZeroWrench);
  tasks.softConstraints.insert(MpcSoftConstraintType::ContactComplementarity);
  EXPECT_FALSE(contactConstraintsAreScheduleGated(tasks));
  EXPECT_TRUE(usesContactImplicitFormulation(tasks));

  // Any one of the three names the formulation.
  MpcFormulationTasks slipOnly;
  slipOnly.softConstraints.insert(MpcSoftConstraintType::ForceWeightedSlip);
  EXPECT_TRUE(usesContactImplicitFormulation(slipOnly));
  MpcFormulationTasks penetrationOnly;
  penetrationOnly.softConstraints.insert(MpcSoftConstraintType::GroundPenetration);
  EXPECT_TRUE(usesContactImplicitFormulation(penetrationOnly));
}

// ---------------------------------------------------------------------------------------------------------------
// The gate itself. Each of these is the audit finding stated as an assertion.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactConstraintScheduleGatingTest, gatedTermsSwitchThemselvesOffDuringAScheduledSwing) {
  // The historical behaviour, which must survive unchanged for a task file that still lists zero_wrench.
  EXPECT_FALSE(makeWrenchCone(true)->isActive(kTime));
  EXPECT_FALSE(makeFrictionCone(true)->isActive(kTime));
  EXPECT_FALSE(makeCentreOfPressure(true)->isActive(kTime));
  EXPECT_FALSE(makeLambdaBarrier(true)->isActive(kTime));

  model_->setStance();
  EXPECT_TRUE(makeWrenchCone(true)->isActive(kTime));
  EXPECT_TRUE(makeFrictionCone(true)->isActive(kTime));
  EXPECT_TRUE(makeCentreOfPressure(true)->isActive(kTime));
  EXPECT_TRUE(makeLambdaBarrier(true)->isActive(kTime));
}

TEST_F(ContactConstraintScheduleGatingTest, ungatedTermsStayActiveDuringAScheduledSwing) {
  // The fix. Without it every one of these is false, and the swing foot's wrench is bounded by nothing at all.
  EXPECT_TRUE(makeWrenchCone(false)->isActive(kTime));
  EXPECT_TRUE(makeFrictionCone(false)->isActive(kTime));
  EXPECT_TRUE(makeCentreOfPressure(false)->isActive(kTime));
  EXPECT_TRUE(makeLambdaBarrier(false)->isActive(kTime));

  // ...and in flight, with neither foot on the ground.
  model_->setContactFlags(makeFeetArray(false));
  EXPECT_TRUE(makeWrenchCone(false)->isActive(kTime));
  EXPECT_TRUE(makeFrictionCone(false)->isActive(kTime));
  EXPECT_TRUE(makeCentreOfPressure(false)->isActive(kTime));
  EXPECT_TRUE(makeLambdaBarrier(false)->isActive(kTime));
}

TEST_F(ContactConstraintScheduleGatingTest, setActiveStillSwitchesAnUngatedTermOffEntirely) {
  // Dropping the schedule gate must not make a term impossible to disable: the runtime tuning path uses setActive().
  std::unique_ptr<ContactWrenchConeConstraint> cone = makeWrenchCone(false);
  cone->setActive(false);
  EXPECT_FALSE(cone->isActive(kTime));
  std::unique_ptr<FrictionForceConeConstraint> friction = makeFrictionCone(false);
  friction->setActive(false);
  EXPECT_FALSE(friction->isActive(kTime));
}

// ---------------------------------------------------------------------------------------------------------------
// The affine offsets. An always-active cone is evaluated on a foot at zero wrench; if the cone is not satisfied
// there, the penalty buys the violation off by inventing a contact force on a foot in the air.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactConstraintScheduleGatingTest, theGatedWrenchConeIsViolatedByAFootAtZeroWrench) {
  // The reason the offsets have to go: this is what an un-gated cone would be asked to enforce in flight.
  // Atlas ships minNormalForce = 5 N, so the normal-force row reads -5 at the zero wrench.
  ContactWrenchConeConstraint::Config config;
  config.minNormalForce = 5.0;
  const ContactWrenchConeConstraint gated(model_->referenceManager(), model_->contactRectangle(kSwingFoot), kSwingFoot,
                                          model_->pinocchioInterface(), model_->wrenchModel(), config, /*scheduleGated=*/true);
  const PreComputation preComp;
  const vector_t zeroInput = vector_t::Zero(model_->wrenchModel().getInputDim());
  const vector_t value = gated.getValue(kTime, model_->nominalState(), zeroInput, preComp);
  EXPECT_LT(value.minCoeff(), -1.0) << "a cone carrying minNormalForce must be violated at the zero wrench";
}

TEST_F(ContactConstraintScheduleGatingTest, theUngatedWrenchConeIsSatisfiedExactlyByAFootAtZeroWrench) {
  ContactWrenchConeConstraint::Config config;
  config.minNormalForce = 5.0;
  config.gripperForce = 3.0;
  const ContactWrenchConeConstraint ungated(model_->referenceManager(), model_->contactRectangle(kSwingFoot), kSwingFoot,
                                            model_->pinocchioInterface(), model_->wrenchModel(), config, /*scheduleGated=*/false);
  EXPECT_DOUBLE_EQ(ungated.getConfig().minNormalForce, 0.0);
  EXPECT_DOUBLE_EQ(ungated.getConfig().gripperForce, 0.0);
  EXPECT_FALSE(ungated.isScheduleGated());

  const PreComputation preComp;
  const vector_t zeroInput = vector_t::Zero(model_->wrenchModel().getInputDim());
  const vector_t value = ungated.getValue(kTime, model_->nominalState(), zeroInput, preComp);
  // Every row of the homogeneous cone is exactly zero at the zero wrench: the zero wrench is on its boundary.
  EXPECT_NEAR(value.cwiseAbs().maxCoeff(), 0.0, 1e-12);

  // The cone must still bound a real wrench: a purely tangential force is outside it.
  vector_t slidingInput = vector_t::Zero(model_->wrenchModel().getInputDim());
  vector6_t wrench = vector6_t::Zero();
  wrench(WRENCH_FORCE_X_INDEX) = 50.0;
  model_->wrenchModel().setContactWrench(slidingInput, wrench, kSwingFoot);
  EXPECT_LT(ungated.getValue(kTime, model_->nominalState(), slidingInput, preComp).minCoeff(), 0.0);
}

TEST_F(ContactConstraintScheduleGatingTest, theUngatedFrictionConeIsExactlyZeroAtTheZeroForce) {
  const PreComputation preComp;
  const vector_t zeroInput = vector_t::Zero(model_->wrenchModel().getInputDim());

  // Gated: the parabolic safety margin puts the zero force sqrt(regularization) INSIDE the cone, i.e. the term
  // reports -sqrt(regularization) = -5 with the default regularization of 25.
  const FrictionForceConeConstraint gated(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                          model_->wrenchModel(), /*scheduleGated=*/true);
  EXPECT_NEAR(gated.getValue(kTime, model_->nominalState(), zeroInput, preComp)(0), -5.0, 1e-9);

  // Un-gated: the same constant is added back, so the zero force sits exactly on the cone.
  const FrictionForceConeConstraint ungated(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                            model_->wrenchModel(), /*scheduleGated=*/false);
  EXPECT_NEAR(ungated.getValue(kTime, model_->nominalState(), zeroInput, preComp)(0), 0.0, 1e-12);
  EXPECT_FALSE(ungated.isScheduleGated());
}

TEST_F(ContactConstraintScheduleGatingTest, theUngatedFrictionConeKeepsItsGradientAndStillBoundsSliding) {
  const PreComputation preComp;
  const vector_t input = model_->makeInput(model_->wrenchModel(), kSwingFoot, 400.0);
  const FrictionForceConeConstraint gated(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                          model_->wrenchModel(), /*scheduleGated=*/true);
  const FrictionForceConeConstraint ungated(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                            model_->wrenchModel(), /*scheduleGated=*/false);

  // The offset is a constant: the values differ by exactly sqrt(regularization) and the gradients are identical, so
  // the smoothing the regularization exists for is untouched.
  const VectorFunctionQuadraticApproximation gatedApproximation =
      gated.getQuadraticApproximation(kTime, model_->nominalState(), input, preComp);
  const VectorFunctionQuadraticApproximation ungatedApproximation =
      ungated.getQuadraticApproximation(kTime, model_->nominalState(), input, preComp);
  EXPECT_NEAR(ungatedApproximation.f(0) - gatedApproximation.f(0), 5.0, 1e-9);
  EXPECT_TRUE(ungatedApproximation.dfdu.isApprox(gatedApproximation.dfdu, 1e-12));
  EXPECT_TRUE(ungatedApproximation.dfdx.isApprox(gatedApproximation.dfdx, 1e-12));

  // A foot sliding under load is still outside the un-gated cone.
  vector_t slidingInput = model_->makeInput(model_->wrenchModel(), kSwingFoot, 100.0);
  vector6_t wrench = vector6_t::Zero();
  wrench(WRENCH_FORCE_Z_INDEX) = 100.0;
  wrench(WRENCH_FORCE_X_INDEX) = 500.0;
  model_->wrenchModel().setContactWrench(slidingInput, wrench, kSwingFoot);
  EXPECT_LT(ungated.getValue(kTime, model_->nominalState(), slidingInput, preComp)(0), 0.0);
}

TEST_F(ContactConstraintScheduleGatingTest, theCentreOfPressureRowsAreHomogeneousSoTheZeroWrenchSatisfiesThem) {
  // Unlike the two cones, these four rows carry no constant term, which is why the gate is all that had to change.
  // If they did, an always-active term would be violated on every foot in flight.
  const PreComputation preComp;
  const vector_t zeroInput = vector_t::Zero(model_->wrenchModel().getInputDim());
  const std::unique_ptr<ContactMomentXYConstraintCppAd> ungated = makeCentreOfPressure(false);
  const vector_t value = ungated->getValue(kTime, model_->nominalState(), zeroInput, preComp);
  ASSERT_EQ(value.size(), 4);
  EXPECT_NEAR(value.cwiseAbs().maxCoeff(), 0.0, 1e-9);

  // And a centre of pressure outside the footprint is still rejected: a moment with no normal force to support it.
  vector_t offFootprint = vector_t::Zero(model_->wrenchModel().getInputDim());
  vector6_t wrench = vector6_t::Zero();
  wrench(WRENCH_FORCE_Z_INDEX) = 100.0;
  wrench(WRENCH_TORQUE_Y_INDEX) = 100.0;  // a CoP far ahead of the toe
  model_->wrenchModel().setContactWrench(offFootprint, wrench, kSwingFoot);
  EXPECT_LT(ungated->getValue(kTime, model_->nominalState(), offFootprint, preComp).minCoeff(), 0.0);
}

// ---------------------------------------------------------------------------------------------------------------
// The penalty. A relaxed log barrier never reaches zero, so on a cone that is tight at the origin it pays the solver
// to leave the origin - which is the force floor the offsets above were removed to avoid.
// ---------------------------------------------------------------------------------------------------------------

TEST(ContactConeGatingPenalty, aRelaxedBarrierPushesAtZeroSlackButASquaredHingeDoesNot) {
  // Atlas's shipped centre-of-pressure barrier.
  const RelaxedBarrierPenalty barrier(RelaxedBarrierPenalty::Config(0.6, 0.03));
  EXPECT_NEAR(barrier.getDerivative(0.0, 0.0), -2.0 * 0.6 / 0.03, 1e-9);
  EXPECT_LT(barrier.getDerivative(0.0, 0.0), -1.0) << "a log barrier pays the solver to leave the boundary";

  // The squared hinge with delta = 0 is zero in value AND in gradient at zero slack, and quadratic below it.
  const SquaredHingePenalty hinge(SquaredHingePenalty::Config(0.6, 0.0));
  EXPECT_DOUBLE_EQ(hinge.getValue(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(hinge.getDerivative(0.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(hinge.getValue(0.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(hinge.getDerivative(0.0, 1.0), 0.0);
  EXPECT_NEAR(hinge.getValue(0.0, -2.0), 0.5 * 0.6 * 4.0, 1e-12);
  EXPECT_NEAR(hinge.getDerivative(0.0, -2.0), 0.6 * -2.0, 1e-12);
}

TEST(ContactConeGatingPenalty, theSquaredHingeIsHotReloadable) {
  // A penalty whose setParameters() is not overridden silently swallows every update the parameter updater writes.
  // The hinge is new to this formulation, so it has to be checked the moment it is introduced.
  SquaredHingePenalty hinge(SquaredHingePenalty::Config(1.0, 0.0));
  vector_t parameters(2);
  parameters << 7.0, 0.5;
  hinge.setParameters(parameters);
  vector_t readBack;
  hinge.getParameters(readBack);
  ASSERT_EQ(readBack.size(), 2);
  EXPECT_DOUBLE_EQ(readBack(0), 7.0);
  EXPECT_DOUBLE_EQ(readBack(1), 0.5);
  EXPECT_NEAR(hinge.getSecondDerivative(0.0, 0.0), 7.0, 1e-12);

  vector_t wrongSize(3);
  wrongSize << 1.0, 2.0, 3.0;
  EXPECT_THROW(hinge.setParameters(wrongSize), std::runtime_error);
}

// ---------------------------------------------------------------------------------------------------------------
// The copy the SQP solver makes of the whole problem, once per worker thread.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactConstraintScheduleGatingTest, theGateAndTheActiveFlagSurviveACloneOfEveryTerm) {
  // FrictionForceConeConstraint and ContactMomentXYConstraintCppAd both dropped isActive_ in their copy constructors,
  // so a deactivated term silently came back to life in every worker thread.
  std::unique_ptr<FrictionForceConeConstraint> friction = makeFrictionCone(false);
  friction->setActive(false);
  const std::unique_ptr<FrictionForceConeConstraint> frictionClone(friction->clone());
  EXPECT_FALSE(frictionClone->getActive()) << "isActive_ was dropped by the copy constructor";
  EXPECT_FALSE(frictionClone->isScheduleGated());
  EXPECT_FALSE(frictionClone->isActive(kTime));

  std::unique_ptr<ContactWrenchConeConstraint> cone = makeWrenchCone(false);
  cone->setActive(false);
  const std::unique_ptr<ContactWrenchConeConstraint> coneClone(cone->clone());
  EXPECT_FALSE(coneClone->getActive());
  EXPECT_FALSE(coneClone->isScheduleGated());

  std::unique_ptr<ContactMomentXYConstraintCppAd> centreOfPressure = makeCentreOfPressure(false);
  centreOfPressure->setActive(false);
  const std::unique_ptr<ContactMomentXYConstraintCppAd> centreOfPressureClone(centreOfPressure->clone());
  EXPECT_FALSE(centreOfPressureClone->getActive()) << "isActive_ was dropped by the copy constructor";
  EXPECT_FALSE(centreOfPressureClone->isScheduleGated());

  std::unique_ptr<BasisScalingNonNegativityConstraint> barrier = makeLambdaBarrier(false);
  const std::unique_ptr<BasisScalingNonNegativityConstraint> barrierClone(barrier->clone());
  EXPECT_FALSE(barrierClone->isScheduleGated());
  EXPECT_TRUE(barrierClone->isActive(kTime));

  // And an un-gated cone's dropped offsets survive the clone too - otherwise a worker thread would enforce a
  // different cone from the one the main thread was configured with.
  const std::unique_ptr<ContactWrenchConeConstraint> ungatedClone(makeWrenchCone(false)->clone());
  EXPECT_DOUBLE_EQ(ungatedClone->getConfig().minNormalForce, 0.0);
}

// ---------------------------------------------------------------------------------------------------------------
// The basis scalings: on the shipped Atlas this is the only bound there is.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactConstraintScheduleGatingTest, theUngatedLambdaBoundIsSilentAtZeroLoadWhereTheBarrierPaidToLeaveIt) {
  // The bribe. PieceWisePolynomialBarrierPenalty's derivative at h = 0 is -mu*delta/2, strictly NEGATIVE, so the
  // objective FALLS as a basis scaling grows away from zero - the solver is paid to invent contact force on a foot in
  // flight. Gated that is unreachable, because the swing-phase scalings are pinned by the zero_wrench equality and the
  // term is off anyway; un-gated this is the only bound there is and it must not lean on the solution.
  //
  // It is the same defect as the ground-penetration log barrier, and it takes the same cure: a one-sided hinge whose
  // zero sits on the boundary in value AND gradient.
  const PreComputation preComp;
  const TargetTrajectories emptyTarget;
  const std::unique_ptr<BasisScalingNonNegativityConstraint> ungated = makeLambdaBarrier(false);
  const std::unique_ptr<BasisScalingNonNegativityConstraint> gated = makeLambdaBarrier(true);

  const long lambdaStart = static_cast<long>(model_->basisModel().getContactWrenchStartIndices(kSwingFoot));
  const vector_t noLoad = vector_t::Zero(model_->basisModel().getInputDim());

  // Un-gated: zero cost and zero gradient at zero load, so a foot carrying nothing is charged nothing and pulled
  // nowhere.
  EXPECT_DOUBLE_EQ(ungated->getValue(kTime, model_->nominalState(), noLoad, emptyTarget, preComp), 0.0);
  const ScalarFunctionQuadraticApproximation ungatedApprox =
      ungated->getQuadraticApproximation(kTime, model_->nominalState(), noLoad, emptyTarget, preComp);
  for (size_t basis = 0; basis < model_->numBasisPerFoot(); ++basis) {
    EXPECT_DOUBLE_EQ(ungatedApprox.dfdu(lambdaStart + static_cast<long>(basis)), 0.0) << "basis " << basis;
  }
  EXPECT_EQ(ungated->getPenaltyName(), "SquaredHingePenalty");

  // The gated term still carries the barrier, and that barrier really does have the negative gradient described
  // above - so this assertion is what makes the one before it mean something rather than restating a tautology.
  const ScalarFunctionQuadraticApproximation gatedApprox =
      gated->getQuadraticApproximation(kTime, model_->nominalState(), noLoad, emptyTarget, preComp);
  EXPECT_LT(gatedApprox.dfdu(lambdaStart), 0.0) << "the barrier is expected to pay the solver to leave zero";
  EXPECT_EQ(gated->getPenaltyName(), "PieceWisePolynomialBarrierPenalty");

  // Both still price a negative scaling, and to within the smoothing width they price it the same: swapping the
  // penalty removes the dead zone, it does not retune the bound.
  vector_t negativeLambda = vector_t::Zero(model_->basisModel().getInputDim());
  negativeLambda(lambdaStart) = -1.0;
  const scalar_t ungatedCost = ungated->getValue(kTime, model_->nominalState(), negativeLambda, emptyTarget, preComp);
  const scalar_t gatedCost = gated->getValue(kTime, model_->nominalState(), negativeLambda, emptyTarget, preComp);
  EXPECT_GT(ungatedCost, 0.0);
  EXPECT_NEAR(ungatedCost, gatedCost, 1.0e-2 * gatedCost) << "the two penalties must agree away from the boundary";
}

TEST_F(ContactConstraintScheduleGatingTest, hotReloadingTheUngatedLambdaBoundKeepsItsZeroOnTheBoundary) {
  // The tuning dashboard writes (mu, delta) from the task file. Delta belongs to the barrier; carrying it into the
  // hinge would move the hinge's zero into the interior and reinstate exactly the bribe the hinge removes - which is
  // how the equivalent hot-reload path for the cone terms was caught doing it.
  const PreComputation preComp;
  const TargetTrajectories emptyTarget;
  const std::unique_ptr<BasisScalingNonNegativityConstraint> ungated = makeLambdaBarrier(false);
  ungated->setBarrierPenalty(PieceWisePolynomialBarrierPenalty::Config(0.5, 0.25));

  // The configured values round-trip, so the dashboard still shows what the operator typed...
  EXPECT_DOUBLE_EQ(ungated->getBarrierConfig().mu, 0.5);
  EXPECT_DOUBLE_EQ(ungated->getBarrierConfig().delta, 0.25);

  // ...but the penalty's zero has not moved off the boundary.
  const long lambdaStart = static_cast<long>(model_->basisModel().getContactWrenchStartIndices(kSwingFoot));
  const vector_t noLoad = vector_t::Zero(model_->basisModel().getInputDim());
  EXPECT_DOUBLE_EQ(ungated->getValue(kTime, model_->nominalState(), noLoad, emptyTarget, preComp), 0.0);
  const ScalarFunctionQuadraticApproximation approx =
      ungated->getQuadraticApproximation(kTime, model_->nominalState(), noLoad, emptyTarget, preComp);
  EXPECT_DOUBLE_EQ(approx.dfdu(lambdaStart), 0.0);

  // And the new stiffness did take effect: 0.5 * mu * 1^2 with mu = 0.5.
  vector_t negativeLambda = vector_t::Zero(model_->basisModel().getInputDim());
  negativeLambda(lambdaStart) = -1.0;
  EXPECT_NEAR(ungated->getValue(kTime, model_->nominalState(), negativeLambda, emptyTarget, preComp), 0.25, 1e-12);
}

TEST_F(ContactConstraintScheduleGatingTest, theUngatedLambdaBarrierPricesANegativeScalingDuringASwing) {
  const PreComputation preComp;
  const TargetTrajectories emptyTarget;
  const std::unique_ptr<BasisScalingNonNegativityConstraint> ungated = makeLambdaBarrier(false);
  const std::unique_ptr<BasisScalingNonNegativityConstraint> gated = makeLambdaBarrier(true);

  // A negative basis scaling is an adhesive, outside-the-cone wrench. The complementarity penalty (f_n h)^2 is
  // sign-blind and would not object to it, so this term is the only thing that does.
  vector_t negativeLambda = vector_t::Zero(model_->basisModel().getInputDim());
  negativeLambda(model_->basisModel().getContactWrenchStartIndices(kSwingFoot)) = -1.0;
  const scalar_t cost = ungated->getValue(kTime, model_->nominalState(), negativeLambda, emptyTarget, preComp);
  EXPECT_GT(cost, 0.0);

  // The gated term is simply not evaluated during a scheduled swing, which is the whole defect.
  EXPECT_FALSE(gated->isActive(kTime));
  EXPECT_TRUE(ungated->isActive(kTime));

  // A non-negative scaling is (nearly) free either way: the barrier's value at zero is mu*delta^2/6.
  const vector_t zeroLambda = vector_t::Zero(model_->basisModel().getInputDim());
  EXPECT_LT(ungated->getValue(kTime, model_->nominalState(), zeroLambda, emptyTarget, preComp), 1e-6);
  EXPECT_GT(cost, ungated->getValue(kTime, model_->nominalState(), zeroLambda, emptyTarget, preComp));
}

// ---------------------------------------------------------------------------------------------------------------
// The input regularization's nominal input is a schedule-derived REFERENCE, and it stays that way. It is what gives
// the solver a reason to unload the foot the plan wants in the air, and it is the mechanism by which a reduced-order
// plan guides the whole-body MPC at all. Spreading the weight over both feet instead - on the theory that the schedule
// should not decide contact - leaves the problem symmetric and pulls the swing foot's force up to half body weight,
// against which the complementarity term pins the foot to within a few millimetres of the ground.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactConstraintScheduleGatingTest, theNominalInputLeavesTheScheduledSwingFootUnloaded) {
  // SetUp has the schedule calling the left foot a swing foot.
  const vector_t duringSwing =
      weightCompensatingInput(model_->pinocchioInterface(), model_->referenceManager().getContactFlags(kTime), model_->wrenchModel());
  const vector_t row = normalContactForceRow(model_->wrenchModel(), kSwingFoot);
  const vector_t stanceRow =
      normalContactForceRow(model_->wrenchModel(), kSwingFoot == CONTACT_LEFT_INDEX ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX);

  EXPECT_NEAR(row.dot(duringSwing), 0.0, 1e-9)
      << "the nominal input must give the scheduled swing foot no load: that is the only thing telling the solver "
         "which foot the plan wants in the air";
  EXPECT_GT(stanceRow.dot(duringSwing), 100.0) << "and the whole weight to the stance foot";

  // In double support it is shared, so neither foot is singled out.
  model_->setStance();
  const vector_t duringStance =
      weightCompensatingInput(model_->pinocchioInterface(), model_->referenceManager().getContactFlags(kTime), model_->wrenchModel());
  EXPECT_NEAR(row.dot(duringStance), stanceRow.dot(duringStance), 1e-6);
  EXPECT_GT(row.dot(duringStance), 100.0);
}

TEST_F(ContactConstraintScheduleGatingTest, theQuadraticCostPricesALoadedSwingFootAboveAnUnloadedOne) {
  const size_t stateDim = model_->wrenchModel().getStateDim();
  const size_t inputDim = model_->wrenchModel().getInputDim();
  const matrix_t Q = matrix_t::Identity(stateDim, stateDim);
  const matrix_t R = matrix_t::Identity(inputDim, inputDim);
  const PreComputation preComp;
  const TargetTrajectories target({kTime}, {model_->nominalState()}, {vector_t::Zero(inputDim)});
  const StateInputQuadraticCost cost(Q, R, model_->referenceManager(), model_->pinocchioInterface(), model_->wrenchModel());

  // The schedule says the left foot is swinging. An input that loads it must cost more than one that does not, or
  // nothing in the problem prefers the gait the plan is proposing.
  const size_t stanceFoot = kSwingFoot == CONTACT_LEFT_INDEX ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
  const vector_t stanceCarries = model_->makeInput(model_->wrenchModel(), stanceFoot, 1600.0);
  vector_t swingAlsoCarries = model_->makeInput(model_->wrenchModel(), stanceFoot, 800.0);
  vector6_t swingWrench = vector6_t::Zero();
  swingWrench(WRENCH_FORCE_Z_INDEX) = 800.0;
  model_->wrenchModel().setContactWrench(swingAlsoCarries, swingWrench, kSwingFoot);

  EXPECT_LT(cost.getValue(kTime, model_->nominalState(), stanceCarries, target, preComp),
            cost.getValue(kTime, model_->nominalState(), swingAlsoCarries, target, preComp))
      << "loading the scheduled swing foot must be the more expensive of the two";
}

// ---------------------------------------------------------------------------------------------------------------
// The friction cone's INPUT JACOBIAN, which has to follow the input parameterization the model actually uses.
//
// Section 2a of the contact-implicit README names this term as one of the four that must be un-gated, so a task file
// that follows it may list `friction_force_cone` alongside `useContactBasisVectorInputs: true`. The cone wrote its
// derivative as a fixed 3x3 at getContactForceStartIndices(), i.e. it assumed the input stores the three force
// components there. Under BasisInputsModelDecorator that index is the start of an 11-wide block of basis scalings and
// the force is `B_local * lambda`, so the write landed on the first three scalings and the solver was handed the
// Jacobian of a function the term was not evaluating. getValue() was right throughout, which is exactly why nothing
// caught it.
// ---------------------------------------------------------------------------------------------------------------

/** A working point with a genuinely OBLIQUE contact force, so the tangential rows of dCone_dF do not vanish. */
vector_t loadedInputWithTangentialForce(const DrcAtlasContactTestModel& model, const MpcRobotModelBase<scalar_t>& robotModel) {
  vector_t input = model.makeInput(robotModel, kSwingFoot, 800.0);
  const size_t start = robotModel.getContactWrenchStartIndices(kSwingFoot);
  if (robotModel.getContactInputDim(kSwingFoot) == CONTACT_WRENCH_DIM) {
    // Wrench-space: tilt the force directly.
    input(static_cast<long>(start + WRENCH_FORCE_X_INDEX)) = 120.0;
    input(static_cast<long>(start + WRENCH_FORCE_Y_INDEX)) = -80.0;
  } else {
    // Basis-vector: lean on two friction-pyramid edge generators, which are the first numBasisVectors columns of
    // B_local and the only ones with a tangential component.
    input(static_cast<long>(start + 0)) += 150.0;
    input(static_cast<long>(start + 1)) += 60.0;
  }
  const vector3_t force = robotModel.getContactForce(input, kSwingFoot);
  EXPECT_GT(force.head<2>().norm(), 10.0) << "the working point must have a tangential force, or the test proves nothing "
                                             "about the x and y rows of the force Jacobian";
  return input;
}

TEST_F(ContactConstraintScheduleGatingTest, theForceJacobianIsTheIdentityBlockOnTheWrenchModel) {
  const std::unique_ptr<FrictionForceConeConstraint> cone = makeFrictionCone(/*scheduleGated=*/false);
  const matrix_t jacobian = cone->getContactForceInputJacobian();

  // The wrench model stores the force in the first three entries of a six-wide wrench block, so d(force)/d(block) is
  // [I3 | 0]. This is the case the hard-coded 3x3 happened to be right for.
  ASSERT_EQ(jacobian.rows(), 3);
  ASSERT_EQ(jacobian.cols(), static_cast<long>(CONTACT_WRENCH_DIM));
  EXPECT_TRUE(jacobian.leftCols<3>().isIdentity(1e-12)) << jacobian;
  EXPECT_TRUE(jacobian.rightCols<3>().isZero(1e-12)) << "the moment cannot move the force";
}

TEST_F(ContactConstraintScheduleGatingTest, theForceJacobianIsTheBasisGeneratorsOnTheBasisModel) {
  const FrictionForceConeConstraint cone(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                         model_->basisModel(), /*scheduleGated=*/false);
  const matrix_t jacobian = cone.getContactForceInputJacobian();

  ASSERT_EQ(jacobian.rows(), 3);
  ASSERT_EQ(jacobian.cols(), static_cast<long>(model_->numBasisPerFoot()))
      << "the cone must linearise through the whole scaling block, not through three of it";

  // It is the force rows of B_local. Every generator of ContactWrenchConeBasisMatrix is a unit normal force applied
  // somewhere on the footprint, so the third row is all ones - the same property the load indicator rests on.
  EXPECT_TRUE(jacobian.row(2).isOnes(1e-12)) << jacobian.row(2);
  EXPECT_GT(jacobian.topRows<2>().cwiseAbs().maxCoeff(), 0.1)
      << "the friction-pyramid generators must contribute tangential force, or the x and y columns are dead";
  // And it is emphatically NOT the identity the pre-fix code wrote there.
  EXPECT_FALSE(jacobian.leftCols<3>().isIdentity(1e-9));
}

TEST_F(ContactConstraintScheduleGatingTest, theConeDerivativesMatchFiniteDifferencesOnTheWrenchModel) {
  const std::unique_ptr<FrictionForceConeConstraint> cone = makeFrictionCone(/*scheduleGated=*/false);
  const vector_t input = loadedInputWithTangentialForce(*model_, model_->wrenchModel());
  expectStateInputDerivativesMatchFiniteDifferences(*cone, model_->nominalState(), input);
}

TEST_F(ContactConstraintScheduleGatingTest, theConeDerivativesMatchFiniteDifferencesOnTheBasisModel) {
  // The regression: this fails on the pre-fix code, for every column of the scaling block.
  const FrictionForceConeConstraint cone(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                         model_->basisModel(), /*scheduleGated=*/false);
  const vector_t input = loadedInputWithTangentialForce(*model_, model_->basisModel());
  expectStateInputDerivativesMatchFiniteDifferences(cone, model_->nominalState(), input);
}

TEST_F(ContactConstraintScheduleGatingTest, theConeHessianMatchesFiniteDifferencesOfItsGradientOnTheBasisModel) {
  // The second-order block moved with the first, and the SQP solver reads it: getQuadraticApproximation() is what
  // ConstraintOrder::Quadratic exists for. Differentiating the analytic gradient isolates d2Cone_du2 from dCone_du.
  const FrictionForceConeConstraint::Config config;
  const FrictionForceConeConstraint cone(model_->referenceManager(), config, kSwingFoot, model_->basisModel(),
                                         /*scheduleGated=*/false);
  const vector_t input = loadedInputWithTangentialForce(*model_, model_->basisModel());
  const vector_t state = model_->nominalState();
  const PreComputation preComp;

  const VectorFunctionQuadraticApproximation quadratic = cone.getQuadraticApproximation(kTime, state, input, preComp);
  ASSERT_EQ(quadratic.dfduu.size(), 1U);
  // getQuadraticApproximation() subtracts the Hessian shift from the whole diagonal; add it back before comparing.
  matrix_t analytic = quadratic.dfduu.front();
  analytic.diagonal().array() += config.hessianDiagonalShift;

  for (long column = 0; column < input.size(); ++column) {
    vector_t perturbed = input;
    perturbed(column) += kFiniteDifferenceStep;
    const matrix_t forward = cone.getLinearApproximation(kTime, state, perturbed, preComp).dfdu;
    perturbed(column) -= 2.0 * kFiniteDifferenceStep;
    const matrix_t backward = cone.getLinearApproximation(kTime, state, perturbed, preComp).dfdu;
    const vector_t numerical = ((forward - backward) / (2.0 * kFiniteDifferenceStep)).row(0).transpose();
    for (long row = 0; row < numerical.size(); ++row) {
      EXPECT_NEAR(analytic(row, column), numerical(row), kDerivativeTol) << "dfduu(" << row << ", " << column << ")";
    }
  }
}

TEST_F(ContactConstraintScheduleGatingTest, theConeTouchesOnlyItsOwnContactsInputs) {
  // A Jacobian written at the wrong offset does not merely lose its own columns, it writes into a neighbour's. On the
  // basis model the two feet's blocks are adjacent, so the left foot's cone reaching three columns past its start is
  // how this would show up in the QP.
  const FrictionForceConeConstraint cone(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                         model_->basisModel(), /*scheduleGated=*/false);
  const vector_t input = loadedInputWithTangentialForce(*model_, model_->basisModel());
  const PreComputation preComp;
  const matrix_t dfdu = cone.getLinearApproximation(kTime, model_->nominalState(), input, preComp).dfdu;

  const long start = static_cast<long>(model_->basisModel().getContactWrenchStartIndices(kSwingFoot));
  const long width = static_cast<long>(model_->numBasisPerFoot());
  matrix_t outsideTheBlock = dfdu;
  outsideTheBlock.middleCols(start, width).setZero();
  EXPECT_TRUE(outsideTheBlock.isZero(1e-12)) << "the cone of one foot must not put gradient on another input: " << outsideTheBlock;
  EXPECT_GT(dfdu.middleCols(start, width).cwiseAbs().maxCoeff(), 1e-6) << "and it must put some on its own";
}

}  // namespace
}  // namespace ocs2::humanoid
