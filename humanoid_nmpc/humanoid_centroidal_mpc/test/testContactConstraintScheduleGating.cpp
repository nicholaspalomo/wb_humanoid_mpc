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

#include <pinocchio/fwd.hpp>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/rnea.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_sqp/SqpSettings.h>
#include <ocs2_sqp/SqpSolver.h>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/HumanoidCostConstraintFactory.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"

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
 * foot with an unbounded wrench: adhesion, unlimited friction, a center of pressure anywhere. On the shipped DRC Atlas
 * it is worse still, because `contactInputParameterization: basis_vectors` makes the non-negativity of the basis scalings the
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
   * The center-of-pressure constraint. It is the one of the four whose rows are already homogeneous in the wrench, so
   * only its gate and its penalty change - which is exactly why it is the one most easily left untested.
   */
  std::unique_ptr<ContactMomentXYConstraintCppAd> makeCenterOfPressure(bool scheduleGated) const {
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
  // The historical behavior, which must survive unchanged for a task file that still lists zero_wrench.
  EXPECT_FALSE(makeWrenchCone(true)->isActive(kTime));
  EXPECT_FALSE(makeFrictionCone(true)->isActive(kTime));
  EXPECT_FALSE(makeCenterOfPressure(true)->isActive(kTime));
  EXPECT_FALSE(makeLambdaBarrier(true)->isActive(kTime));

  model_->setStance();
  EXPECT_TRUE(makeWrenchCone(true)->isActive(kTime));
  EXPECT_TRUE(makeFrictionCone(true)->isActive(kTime));
  EXPECT_TRUE(makeCenterOfPressure(true)->isActive(kTime));
  EXPECT_TRUE(makeLambdaBarrier(true)->isActive(kTime));
}

TEST_F(ContactConstraintScheduleGatingTest, ungatedTermsStayActiveDuringAScheduledSwing) {
  // The fix. Without it every one of these is false, and the swing foot's wrench is bounded by nothing at all.
  EXPECT_TRUE(makeWrenchCone(false)->isActive(kTime));
  EXPECT_TRUE(makeFrictionCone(false)->isActive(kTime));
  EXPECT_TRUE(makeCenterOfPressure(false)->isActive(kTime));
  EXPECT_TRUE(makeLambdaBarrier(false)->isActive(kTime));

  // ...and in flight, with neither foot on the ground.
  model_->setContactFlags(makeFeetArray(false));
  EXPECT_TRUE(makeWrenchCone(false)->isActive(kTime));
  EXPECT_TRUE(makeFrictionCone(false)->isActive(kTime));
  EXPECT_TRUE(makeCenterOfPressure(false)->isActive(kTime));
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

  // Un-gated: every row is exactly zero at the zero force, so a foot in flight is on the boundary and a squared hinge
  // charges it nothing.
  const FrictionForceConeConstraint ungated(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                            model_->wrenchModel(), /*scheduleGated=*/false);
  const vector_t value = ungated.getValue(kTime, model_->nominalState(), zeroInput, preComp);
  ASSERT_EQ(value.size(), 2) << "the friction row and Fz >= 0";
  EXPECT_NEAR(value.cwiseAbs().maxCoeff(), 0.0, 1e-12);
  EXPECT_FALSE(ungated.isScheduleGated());
}

TEST_F(ContactConstraintScheduleGatingTest, theUngatedFrictionConeIsExactlyTheCoulombCone) {
  // Audit finding A3. The un-gated cone used to be the gated row with sqrt(regularization) added back. That put the
  // zero force on the boundary, but it also made the cone an OUTER approximation: |F_t| <= sqrt(mu^2 Fz^2 + 2 mu Fz
  // sqrt(r)), up to sqrt(r) = 5 N more friction than mu Fz, which on a lightly loaded foot is several times mu. The two
  // rows it has now accept a force if and only if Fz >= 0 and |F_t| <= mu Fz, at every load.
  const PreComputation preComp;
  const FrictionForceConeConstraint::Config config;  // mu = 0.7, regularization = 25
  const FrictionForceConeConstraint ungated(model_->referenceManager(), config, kSwingFoot, model_->wrenchModel(),
                                            /*scheduleGated=*/false);
  const scalar_t mu = ungated.getConfig().frictionCoefficient;

  size_t accepted = 0;
  size_t rejected = 0;
  for (const scalar_t normalForce : {-100.0, -1.0, 0.0, 0.25, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0}) {
    for (const scalar_t ratio : {0.0, 0.3, 0.9, 0.999, 1.001, 1.1, 2.0, 5.0}) {
      // |F_t| as a multiple of mu |Fz|, pointing along a skewed direction so both tangential components are exercised.
      const scalar_t tangential = ratio * mu * std::abs(normalForce) + (normalForce == 0.0 ? ratio : 0.0);
      vector_t input = vector_t::Zero(model_->wrenchModel().getInputDim());
      vector6_t wrench = vector6_t::Zero();
      wrench(WRENCH_FORCE_X_INDEX) = 0.6 * tangential;
      wrench(WRENCH_FORCE_Y_INDEX) = -0.8 * tangential;
      wrench(WRENCH_FORCE_Z_INDEX) = normalForce;
      model_->wrenchModel().setContactWrench(input, wrench, kSwingFoot);

      const bool insideCoulomb = normalForce >= 0.0 && tangential <= mu * normalForce;
      const bool acceptedByTerm = (ungated.getValue(kTime, model_->nominalState(), input, preComp).array() >= 0.0).all();
      EXPECT_EQ(acceptedByTerm, insideCoulomb) << "Fz = " << normalForce << " N, |F_t| = " << tangential << " N";
      (insideCoulomb ? accepted : rejected) += 1;
    }
  }
  ASSERT_GT(accepted, 10U);
  ASSERT_GT(rejected, 10U);

  // The case the old form got wrong: 1 N of load cannot hold 2 N of friction at mu = 0.7, and the outer approximation
  // accepted it (0.7 - sqrt(4 + 25) + 5 = 0.31 >= 0).
  vector_t lightlyLoaded = vector_t::Zero(model_->wrenchModel().getInputDim());
  vector6_t wrench = vector6_t::Zero();
  wrench(WRENCH_FORCE_X_INDEX) = 2.0;
  wrench(WRENCH_FORCE_Z_INDEX) = 1.0;
  model_->wrenchModel().setContactWrench(lightlyLoaded, wrench, kSwingFoot);
  EXPECT_LT(ungated.getValue(kTime, model_->nominalState(), lightlyLoaded, preComp)(0), 0.0);
  const scalar_t oldOuterRow = mu * 1.0 - std::sqrt(4.0 + config.regularization) + std::sqrt(config.regularization);
  EXPECT_GT(oldOuterRow, 0.0) << "positive control: the form this replaces accepted the point";
}

TEST_F(ContactConstraintScheduleGatingTest, onlyTheGatedFrictionConeIsHandedToThePenaltyAtSecondOrder) {
  // The exact un-gated friction row is convex in Fz, so its Hessian is indefinite and would make the penalty's Hessian
  // indefinite too. It is declared Linear, so the soft-constraint wrapper linearizes it; the gated row is concave and
  // keeps its second-order treatment unchanged.
  EXPECT_EQ(makeFrictionCone(/*scheduleGated=*/true)->getOrder(), ConstraintOrder::Quadratic);
  EXPECT_EQ(makeFrictionCone(/*scheduleGated=*/false)->getOrder(), ConstraintOrder::Linear);
  EXPECT_EQ(std::unique_ptr<FrictionForceConeConstraint>(makeFrictionCone(false)->clone())->getOrder(), ConstraintOrder::Linear);
}

TEST_F(ContactConstraintScheduleGatingTest, theUngatedFrictionConeStillBoundsSlidingUnderLoad) {
  const PreComputation preComp;
  const FrictionForceConeConstraint ungated(model_->referenceManager(), FrictionForceConeConstraint::Config(), kSwingFoot,
                                            model_->wrenchModel(), /*scheduleGated=*/false);
  // A foot sliding under load is outside the un-gated cone...
  vector_t slidingInput = model_->makeInput(model_->wrenchModel(), kSwingFoot, /*normalForce=*/100.0);
  vector6_t wrench = vector6_t::Zero();
  wrench(WRENCH_FORCE_Z_INDEX) = 100.0;
  wrench(WRENCH_FORCE_X_INDEX) = 500.0;
  model_->wrenchModel().setContactWrench(slidingInput, wrench, kSwingFoot);
  EXPECT_LT(ungated.getValue(kTime, model_->nominalState(), slidingInput, preComp)(0), 0.0);

  // ...and a foot pulling on the ground is refused by the second row even with no tangential force at all.
  wrench.setZero();
  wrench(WRENCH_FORCE_Z_INDEX) = -50.0;
  model_->wrenchModel().setContactWrench(slidingInput, wrench, kSwingFoot);
  const vector_t pulling = ungated.getValue(kTime, model_->nominalState(), slidingInput, preComp);
  EXPECT_GE(pulling(0), 0.0) << "the friction row alone cannot tell adhesion from load";
  EXPECT_LT(pulling(1), 0.0) << "Fz >= 0 is what refuses it";
}

TEST_F(ContactConstraintScheduleGatingTest, theCenterOfPressureRowsAreHomogeneousSoTheZeroWrenchSatisfiesThem) {
  // Unlike the two cones, these four rows carry no constant term, which is why the gate is all that had to change.
  // If they did, an always-active term would be violated on every foot in flight.
  const PreComputation preComp;
  const vector_t zeroInput = vector_t::Zero(model_->wrenchModel().getInputDim());
  const std::unique_ptr<ContactMomentXYConstraintCppAd> ungated = makeCenterOfPressure(false);
  const vector_t value = ungated->getValue(kTime, model_->nominalState(), zeroInput, preComp);
  ASSERT_EQ(value.size(), 4);
  EXPECT_NEAR(value.cwiseAbs().maxCoeff(), 0.0, 1e-9);

  // And a center of pressure outside the footprint is still rejected: a moment with no normal force to support it.
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
  // Atlas's shipped center-of-pressure barrier.
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

  std::unique_ptr<ContactMomentXYConstraintCppAd> centerOfPressure = makeCenterOfPressure(false);
  centerOfPressure->setActive(false);
  const std::unique_ptr<ContactMomentXYConstraintCppAd> centerOfPressureClone(centerOfPressure->clone());
  EXPECT_FALSE(centerOfPressureClone->getActive()) << "isActive_ was dropped by the copy constructor";
  EXPECT_FALSE(centerOfPressureClone->isScheduleGated());

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
// against which the complementarity term pins the foot to within a few millimeters of the ground.
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
  const StateInputQuadraticCost cost(Q, R, model_->referenceManager(), model_->wrenchModel());

  // The schedule says the left foot is swinging. An input that loads it must cost more than one that does not, or
  // nothing in the problem prefers the gait the plan is proposing.
  const size_t stanceFoot = kSwingFoot == CONTACT_LEFT_INDEX ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
  const vector_t stanceCarries = model_->makeInput(model_->wrenchModel(), stanceFoot, /*normalForce=*/1600.0);
  vector_t swingAlsoCarries = model_->makeInput(model_->wrenchModel(), stanceFoot, /*normalForce=*/800.0);
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
// that follows it may list `friction_force_cone` alongside `contactInputParameterization: basis_vectors`. The cone wrote its
// derivative as a fixed 3x3 at getContactForceStartIndices(), i.e. it assumed the input stores the three force
// components there. Under BasisInputsModelDecorator that index is the start of an 11-wide block of basis scalings and
// the force is `B_local * lambda`, so the write landed on the first three scalings and the solver was handed the
// Jacobian of a function the term was not evaluating. getValue() was right throughout, which is exactly why nothing
// caught it.
// ---------------------------------------------------------------------------------------------------------------

/** A working point with a genuinely OBLIQUE contact force, so the tangential rows of dCone_dF do not vanish. */
vector_t loadedInputWithTangentialForce(const DrcAtlasContactTestModel& model, const MpcRobotModelBase<scalar_t>& robotModel) {
  vector_t input = model.makeInput(robotModel, kSwingFoot, /*normalForce=*/800.0);
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
      << "the cone must linearize through the whole scaling block, not through three of it";

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
  // The second-order block moved with the first, and for the gated term the SQP solver reads it:
  // getQuadraticApproximation() is what ConstraintOrder::Quadratic exists for. The un-gated term is linearized by the
  // solver, but its second derivatives are still reported and must still be exact, row by row. Differentiating the
  // analytic gradient isolates d2Cone_du2 from dCone_du.
  const FrictionForceConeConstraint::Config config;
  const vector_t input = loadedInputWithTangentialForce(*model_, model_->basisModel());
  const vector_t state = model_->nominalState();
  const PreComputation preComp;

  for (const bool scheduleGated : {true, false}) {
    const FrictionForceConeConstraint term(model_->referenceManager(), config, kSwingFoot, model_->basisModel(), scheduleGated);
    const VectorFunctionQuadraticApproximation quadratic = term.getQuadraticApproximation(kTime, state, input, preComp);
    ASSERT_EQ(quadratic.dfduu.size(), term.getNumConstraints(kTime)) << "gated: " << scheduleGated;
    for (size_t constraintRow = 0; constraintRow < quadratic.dfduu.size(); ++constraintRow) {
      // getQuadraticApproximation() subtracts the Hessian shift from the whole diagonal; add it back before comparing.
      matrix_t analytic = quadratic.dfduu[constraintRow];
      analytic.diagonal().array() += config.hessianDiagonalShift;

      for (long column = 0; column < input.size(); ++column) {
        vector_t perturbed = input;
        perturbed(column) += kFiniteDifferenceStep;
        const matrix_t forward = term.getLinearApproximation(kTime, state, perturbed, preComp).dfdu;
        perturbed(column) -= 2.0 * kFiniteDifferenceStep;
        const matrix_t backward = term.getLinearApproximation(kTime, state, perturbed, preComp).dfdu;
        const vector_t numerical = ((forward - backward) / (2.0 * kFiniteDifferenceStep)).row(static_cast<long>(constraintRow)).transpose();
        for (long row = 0; row < numerical.size(); ++row) {
          EXPECT_NEAR(analytic(row, column), numerical(row), kDerivativeTol)
              << "gated: " << scheduleGated << ", constraint " << constraintRow << ", dfduu(" << row << ", " << column << ")";
        }
      }
    }
  }
}

TEST_F(ContactConstraintScheduleGatingTest, theUngatedFrictionConeDerivativesAreExactNearTheApex) {
  // Under body weight the exact friction row is indistinguishable from mu Fz - |F_t| to first and second order: its
  // slope in Fz is mu to within mu r / (2 mu^2 Fz^2) and its curvature in Fz is about 1e-7 at 800 N, so the checks at
  // the loaded working point above cannot tell a right normal half of the row from a wrong one. The row differs from
  // the gated form only near the apex - on the lightly loaded touch-down and lift-off feet it was rewritten for - so
  // that is where its derivatives are checked, on both parameterizations.
  const FrictionForceConeConstraint::Config config;  // mu = 0.7, regularization = 25
  const vector_t state = model_->nominalState();
  const PreComputation preComp;

  for (const MpcRobotModelBase<scalar_t>* robotModel : {static_cast<const MpcRobotModelBase<scalar_t>*>(&model_->wrenchModel()),
                                                        static_cast<const MpcRobotModelBase<scalar_t>*>(&model_->basisModel())}) {
    const bool onBasis = robotModel == &model_->basisModel();
    // About 5 N of load and 2 N of friction: mu^2 Fz^2 is of the order of the regularization, where the two halves of the
    // friction row bend the most.
    vector_t input = model_->makeInput(*robotModel, kSwingFoot, /*normalForce=*/5.0);
    const long start = static_cast<long>(robotModel->getContactWrenchStartIndices(kSwingFoot));
    if (onBasis) {
      input(start + 0) += 1.5;
      input(start + 1) += 0.5;
    } else {
      input(start + WRENCH_FORCE_X_INDEX) = 1.6;
      input(start + WRENCH_FORCE_Y_INDEX) = -1.2;
    }
    const vector3_t force = robotModel->getContactForce(input, kSwingFoot);
    ASSERT_GT(force.head<2>().norm(), 0.5) << "basis: " << onBasis << ": the working point needs a tangential force";
    // Positive control: here the curvature of the normal half of the friction row, mu^2 r / (mu^2 Fz^2 + r)^(3/2), is
    // three orders of magnitude above the tolerance, so a wrong one - or the gated row's zero - fails the loop below.
    const scalar_t muSquareFz = config.frictionCoefficient * config.frictionCoefficient * force.z() * force.z();
    const scalar_t normalCurvature =
        config.frictionCoefficient * config.frictionCoefficient * config.regularization / std::pow(muSquareFz + config.regularization, 1.5);
    ASSERT_GT(normalCurvature, 1.0e3 * kDerivativeTol) << "basis: " << onBasis << ": the working point must be near the apex";

    const FrictionForceConeConstraint ungated(model_->referenceManager(), config, kSwingFoot, *robotModel, /*scheduleGated=*/false);
    expectStateInputDerivativesMatchFiniteDifferences(ungated, state, input);

    const VectorFunctionQuadraticApproximation quadratic = ungated.getQuadraticApproximation(kTime, state, input, preComp);
    ASSERT_EQ(quadratic.dfduu.size(), 2U) << "basis: " << onBasis;
    for (size_t constraintRow = 0; constraintRow < quadratic.dfduu.size(); ++constraintRow) {
      matrix_t analytic = quadratic.dfduu[constraintRow];
      analytic.diagonal().array() += config.hessianDiagonalShift;
      for (long column = 0; column < input.size(); ++column) {
        vector_t perturbed = input;
        perturbed(column) += kFiniteDifferenceStep;
        const matrix_t forward = ungated.getLinearApproximation(kTime, state, perturbed, preComp).dfdu;
        perturbed(column) -= 2.0 * kFiniteDifferenceStep;
        const matrix_t backward = ungated.getLinearApproximation(kTime, state, perturbed, preComp).dfdu;
        const vector_t numerical = ((forward - backward) / (2.0 * kFiniteDifferenceStep)).row(static_cast<long>(constraintRow)).transpose();
        for (long row = 0; row < numerical.size(); ++row) {
          EXPECT_NEAR(analytic(row, column), numerical(row), kDerivativeTol)
              << "basis: " << onBasis << ", constraint " << constraintRow << ", dfduu(" << row << ", " << column << ")";
        }
      }
    }
  }
}

TEST_F(ContactConstraintScheduleGatingTest, theWrenchConeDerivativesMatchFiniteDifferencesOnAPitchedFoot) {
  // The un-gated wrench cone is evaluated on a foot whose rocking rates the contact-implicit formulation leaves free,
  // so its dependence on the foot's orientation is the derivative the solver needs. It used to report dfdx = 0.
  vector_t pitched = model_->nominalState();
  pitched(model_->anklePitchStateIndex(kSwingFoot)) += 0.25;
  vector_t input = loadedInputWithTangentialForce(*model_, model_->wrenchModel());
  // A moment as well, so the moment rows' rotation is differentiated too, not only the force rows'.
  model_->wrenchModel().setContactMoment(input, vector3_t(6.0, -4.0, 3.0), kSwingFoot);
  expectStateInputDerivativesMatchFiniteDifferences(*makeWrenchCone(/*scheduleGated=*/false), pitched, input);
  expectStateInputDerivativesMatchFiniteDifferences(*makeWrenchCone(/*scheduleGated=*/true), pitched, input);
}

TEST_F(ContactConstraintScheduleGatingTest, theConeTouchesOnlyItsOwnContactsInputs) {
  // A Jacobian written at the wrong offset does not merely lose its own columns, it writes into a neighbor's. On the
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

// ---------------------------------------------------------------------------------------------------------------
// The property GRAVITY_COMP depends on: a loaded foot changes the joint torque, and by more than g_j(q) is.
//
// For a floating base, static equilibrium is g(q) = S^T tau + J_c^T f, so the joint rows are tau = g_j(q) - J_c,j^T f.
// computeBaseHeldJointTorques supplies both halves (computeJointTorques, the floating-base inverse dynamics, lets a base
// without contact fall freely, so it is not the gantry's torque); pinocchio's nonLinearEffects at zero velocity supplies
// only the first.
// CentroidalMpcMrtJointController::fillGravityCompAction used to command the first alone with kp = 0, and at a bent
// knee that is not a small error - the contact term is several times g_j(q) and points the other way, so the
// commanded torque drove the knee into deeper flexion, which lengthens its own moment arm. The robot folded up.
//
// The two assertions are the two regimes the mode runs in: feet loaded (standing) and feet clear (on the gantry).
// The second is what lets one code path serve both, and is why the fix needed no caller to decide which it was in.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactConstraintScheduleGatingTest, theContactWrenchDominatesTheJointTorqueAtABentKnee) {
  const PinocchioInterface& pinocchioInterface = model_->pinocchioInterface();
  const vector_t q = model_->wrenchModel().getGeneralizedCoordinates(model_->nominalState());
  const size_t jointDim = model_->wrenchModel().getJointDim();
  const vector_t qd = vector_t::Zero(q.size());
  const vector_t qdd_j = vector_t::Zero(jointDim);

  // Feet clear of the ground: the result must be exactly the base-held gravity torques, which is the gantry case and
  // the behavior GRAVITY_COMP had in every regime.
  PinocchioInterface workingInterface = pinocchioInterface;
  const std::array<vector6_t, 2> noContact{vector6_t::Zero(), vector6_t::Zero()};
  const vector_t unloaded = computeBaseHeldJointTorques<scalar_t>(q, qd, qdd_j, noContact, workingInterface);

  const pinocchio::Model& model = workingInterface.getModel();
  pinocchio::Data data = workingInterface.getData();
  pinocchio::nonLinearEffects(model, data, q, qd);
  const vector_t gravityOnly = data.nle.tail(jointDim);
  EXPECT_TRUE(unloaded.isApprox(gravityOnly, 1e-9))
      << "with no foot loaded the contact-aware torque must degrade to g_j(q), or the gantry case regresses";

  // Feet loaded: half the robot's weight up through each foot, as weightCompensatingInput produces in double support.
  const scalar_t weight = 9.81 * pinocchio::computeTotalMass(model);
  vector6_t halfWeight = vector6_t::Zero();
  halfWeight(2) = 0.5 * weight;
  const std::array<vector6_t, 2> standing{halfWeight, halfWeight};
  const vector_t loaded = computeBaseHeldJointTorques<scalar_t>(q, qd, qdd_j, standing, workingInterface);

  const scalar_t contactContribution = (loaded - gravityOnly).cwiseAbs().maxCoeff();
  const scalar_t gravityMagnitude = gravityOnly.cwiseAbs().maxCoeff();
  EXPECT_GT(contactContribution, gravityMagnitude)
      << "the contact term is the DOMINANT one when the feet carry the robot: |J_c,j^T f|_max = " << contactContribution
      << " Nm against |g_j|_max = " << gravityMagnitude << " Nm. A feedforward that omits it is not a small error.";
}

// ---------------------------------------------------------------------------------------------------------------
// The penalty the factory wraps each cone in, and what a hot reload writes into it (audit finding A15). The two used to
// be two separate pieces of code - HumanoidCostConstraintFactory's makeContactConePenalty() and a lambda in
// MpcParameterUpdaterModule - that had to agree on the un-gated hinge's delta of 0, and neither was exercised: the
// end-to-end fixtures run the basis-vector Atlas, which never asks the factory for a contact_wrench_cone.
// ---------------------------------------------------------------------------------------------------------------

/** `content` with `<key>:` of the `contacts.<block>` section (keys at four spaces) set to `value`. */
std::string withContactsSectionValue(const std::string& content,
                                     const std::string& block,
                                     const std::string& key,
                                     const std::string& value) {
  const std::string blockLine = absl::StrCat("\n  ", block, ":");
  const std::string::size_type blockPos = content.find(blockLine);
  EXPECT_NE(blockPos, std::string::npos) << "the task file has no contacts." << block;
  if (blockPos == std::string::npos) return content;
  // The section ends at the next line indented by two spaces or fewer that is not a comment.
  std::string::size_type blockEnd = content.size();
  for (std::string::size_type lineStart = content.find('\n', blockPos + 1); lineStart != std::string::npos;
       lineStart = content.find('\n', lineStart + 1)) {
    const std::string::size_type firstChar = content.find_first_not_of(' ', lineStart + 1);
    if (firstChar == std::string::npos) break;
    if (content[firstChar] != '\n' && content[firstChar] != '#' && firstChar - (lineStart + 1) <= 2) {
      blockEnd = lineStart;
      break;
    }
  }
  const std::string keyLine = absl::StrCat("\n    ", key, ":");
  const std::string::size_type keyPos = content.find(keyLine, blockPos);
  EXPECT_TRUE(keyPos != std::string::npos && keyPos < blockEnd) << "contacts." << block << "." << key << " is not in the task file";
  if (keyPos == std::string::npos || keyPos >= blockEnd) return content;
  const std::string::size_type valueStart = keyPos + keyLine.size();
  std::string result = content;
  result.replace(valueStart, result.find('\n', valueStart) - valueStart, absl::StrCat(" ", value));
  return result;
}

/** The (mu, delta) parameters and the name of the penalty a cone's soft constraint carries. */
struct InstalledPenalty {
  std::string name;
  vector_t parameters;
  scalar_t valueAtZeroSlack = 0.0;
  scalar_t derivativeAtZeroSlack = 0.0;
  scalar_t derivativeAtNegativeSlack = 0.0;
};

InstalledPenalty installedPenalty(StateInputSoftConstraint& softConstraint) {
  InstalledPenalty installed;
  const std::vector<std::unique_ptr<augmented::AugmentedPenaltyBase>>& penalties = softConstraint.getPenalty().getPenaltyPtrArray();
  EXPECT_EQ(penalties.size(), 1U) << "one penalty for every row of the cone";
  if (penalties.empty()) return installed;
  installed.name = penalties.front()->name();
  penalties.front()->getParameters(installed.parameters);
  installed.valueAtZeroSlack = penalties.front()->getValue(kTime, /*l=*/0.0, /*h=*/0.0);
  installed.derivativeAtZeroSlack = penalties.front()->getDerivative(kTime, /*l=*/0.0, /*h=*/0.0);
  installed.derivativeAtNegativeSlack = penalties.front()->getDerivative(kTime, /*l=*/0.0, /*h=*/-1.0);
  return installed;
}

TEST_F(ContactConstraintScheduleGatingTest, theFactoryAndTheHotReloadKeepAnUngatedConesHingeZeroOnTheCone) {
  const std::string reloadedTaskFile = absl::StrCat(testing::TempDir(), "/testContactConstraintScheduleGating_conePenalties.yaml");
  std::string shipped;
  {
    std::ifstream in(model_->taskFile());
    shipped.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  ASSERT_FALSE(shipped.empty());

  // The three cones, their task-file sections, and the (mu, delta) a reload writes: values the shipped file has for none
  // of them, so that a parameter read back after the reload can only have come from it.
  struct Cone {
    absl::string_view termSuffix;
    std::string section;
    scalar_t reloadedMu;
    scalar_t reloadedDelta;
  };
  const std::vector<Cone> cones = {
      {contact_term::kContactWrenchCone, "contactWrenchConeSoftConstraint", 0.37, 4.5},
      {contact_term::kFrictionForceCone, "frictionForceConeSoftConstraint", 0.29, 3.5},
      {contact_term::kContactMomentXY, "contactMomentXYSoftConstraint", 0.71, 0.045},
  };
  std::string reloaded = shipped;
  for (const Cone& cone : cones) {
    reloaded = withContactsSectionValue(reloaded, cone.section, "mu", absl::StrCat(cone.reloadedMu));
    reloaded = withContactsSectionValue(reloaded, cone.section, "delta", absl::StrCat(cone.reloadedDelta));
  }

  for (const bool scheduleGated : {false, true}) {
    SCOPED_TRACE(scheduleGated ? "schedule-gated (zero_wrench listed)" : "un-gated (the contact-implicit formulation)");
    {
      std::ofstream out(reloadedTaskFile, std::ios::trunc);
      out << shipped;
    }

    // The cones exactly as the MPC interfaces build them: through the factory, on the wrench-space model.
    const HumanoidCostConstraintFactory factory(model_->taskFile(), model_->referenceFile(), model_->referenceManager(),
                                                model_->pinocchioInterface(), model_->wrenchModel(), model_->adWrenchModel(),
                                                model_->modelSettings(), /*verbose=*/false, scheduleGated);
    OptimalControlProblem problem;
    const std::vector<std::string>& contactNames = model_->modelSettings().contactNames;
    for (size_t foot = 0; foot < contactNames.size(); ++foot) {
      absl::StatusOr<std::unique_ptr<StateInputCost>> wrenchCone = factory.getContactWrenchConeConstraint(foot);
      ASSERT_TRUE(wrenchCone.ok()) << wrenchCone.status();
      problem.softConstraintPtr->add(contact_term::name(contactNames[foot], contact_term::kContactWrenchCone), *std::move(wrenchCone));
      problem.softConstraintPtr->add(contact_term::name(contactNames[foot], contact_term::kFrictionForceCone),
                                     factory.getFrictionForceConeConstraint(foot));
      problem.softConstraintPtr->add(
          contact_term::name(contactNames[foot], contact_term::kContactMomentXY),
          factory.getContactMomentXYConstraint(foot, absl::StrCat("testContactConstraintScheduleGating_cop_", foot)));
    }

    // Built: an un-gated cone is a hinge whose zero is on the cone, a gated one the file's relaxed barrier. The gated
    // case is the positive control - its derivative at zero slack is what the hinge exists to remove.
    for (const std::string& footName : contactNames) {
      for (const Cone& cone : cones) {
        SCOPED_TRACE(contact_term::name(footName, cone.termSuffix));
        const InstalledPenalty built =
            installedPenalty(problem.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, cone.termSuffix)));
        ASSERT_EQ(built.parameters.size(), 2);
        if (scheduleGated) {
          EXPECT_EQ(built.name, "RelaxedBarrierPenalty");
          EXPECT_GT(built.parameters(1), 0.0);
          EXPECT_LT(built.derivativeAtZeroSlack, 0.0) << "the barrier pays the solver to leave the boundary";
        } else {
          EXPECT_EQ(built.name, "SquaredHingePenalty");
          EXPECT_DOUBLE_EQ(built.parameters(1), 0.0) << "the hinge's zero must sit on the cone";
          EXPECT_DOUBLE_EQ(built.valueAtZeroSlack, 0.0);
          EXPECT_DOUBLE_EQ(built.derivativeAtZeroSlack, 0.0);
          EXPECT_LT(built.derivativeAtNegativeSlack, 0.0) << "and it still prices a violation";
        }
      }
    }

    // Reloaded: the updater walks every worker's copy of the problem and rewrites each cone's (mu, delta) from the file.
    const size_t inputDim = model_->wrenchModel().getInputDim();
    const mpc::Settings mpcSettings = mpc::loadSettings(model_->taskFile(), "mpc", /*verbose=*/false);
    const sqp::Settings sqpSettings = sqp::loadSettings(model_->taskFile(), "multiple_shooting", /*verbose=*/false);
    const DefaultInitializer initializer(inputDim);
    SqpMpc mpc(mpcSettings, sqpSettings, problem, initializer);
    absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created = MpcParameterUpdaterModule::Create(
        &mpc, reloadedTaskFile, /*urdfFile=*/"", model_->referenceFile(), model_->wrenchModel().getStateDim(), inputDim, contactNames);
    ASSERT_TRUE(created.ok()) << created.status();
    MpcParameterUpdaterModule& updater = **created;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    {
      std::ofstream out(reloadedTaskFile, std::ios::trunc);
      out << reloaded;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (size_t i = 0; i < 101; ++i) {  // the file is stat-ed once every 100 pre-solve hooks
      updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, model_->nominalState(), model_->referenceManager());
    }

    SqpSolver& solver = dynamic_cast<SqpSolver&>(*mpc.getSolverPtr());
    ASSERT_GT(solver.getOcpDefinitions().size(), 0U);
    for (OptimalControlProblem& ocp : solver.getOcpDefinitions()) {
      for (const std::string& footName : contactNames) {
        for (const Cone& cone : cones) {
          SCOPED_TRACE(contact_term::name(footName, cone.termSuffix));
          const InstalledPenalty reloadedPenalty =
              installedPenalty(ocp.softConstraintPtr->get<StateInputSoftConstraint>(contact_term::name(footName, cone.termSuffix)));
          ASSERT_EQ(reloadedPenalty.parameters.size(), 2);
          EXPECT_DOUBLE_EQ(reloadedPenalty.parameters(0), cone.reloadedMu) << "the reload did not reach the cone";
          if (scheduleGated) {
            EXPECT_DOUBLE_EQ(reloadedPenalty.parameters(1), cone.reloadedDelta);
          } else {
            // The file's barrier delta must not have been written into the hinge.
            EXPECT_DOUBLE_EQ(reloadedPenalty.parameters(1), 0.0) << "the hot reload moved the hinge's zero off the cone";
            EXPECT_DOUBLE_EQ(reloadedPenalty.valueAtZeroSlack, 0.0);
            EXPECT_DOUBLE_EQ(reloadedPenalty.derivativeAtZeroSlack, 0.0);
          }
        }
      }
    }
  }
  std::remove(reloadedTaskFile.c_str());
}

}  // namespace
}  // namespace ocs2::humanoid
