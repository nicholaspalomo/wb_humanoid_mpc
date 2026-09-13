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

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <memory>

#include <ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h>

#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

// ==================== Test Constants ====================

/// Number of basis-vector scalings per foot (mimics N=4 → 4+7=11 basis).
static constexpr size_t kNumBasisPerFoot = 11;
/// Two-foot layout: [λ_left(11), λ_right(11), joint_vel(24)] = 46.
static constexpr size_t kJointDim = 24;
static constexpr size_t kInputDim = kNumBasisPerFoot * N_CONTACTS + kJointDim;
/// Arbitrary state dimension (constraint math is independent of state).
static constexpr size_t kStateDim = 36;

static constexpr size_t kLeftFootIndex = 0;
static constexpr size_t kRightFootIndex = 1;
static constexpr size_t kLeftLambdaStart = 0;
static constexpr size_t kRightLambdaStart = kNumBasisPerFoot;

/// Barrier penalty parameters. PieceWisePolynomialBarrierPenalty returns
/// non-zero values only for h < delta; for h >= delta, value/grad/hess = 0.
static constexpr scalar_t kBarrierMu = 1e-2;
static constexpr scalar_t kBarrierDelta = 1e-3;

/// Test tolerances.
static constexpr scalar_t kTolerance = 1e-10;
static constexpr scalar_t kFiniteDiffEps = 1e-7;
static constexpr scalar_t kFiniteDiffTolerance = 1e-4;

/// Mode numbers from MotionPhaseDefinition.h.
static constexpr size_t kModeDoubleStance = 3;  // Both feet in contact

/// Test time.
static constexpr scalar_t kTestTime = 0.5;

// ==================== Test Fixture ====================
// Uses a SwitchedModelReferenceManager with the double-stance mode schedule
// set via the constructor's initial ModeSchedule (active value).
// NOTE: We do NOT call preSolverRun() — isActive() tests are deferred
// to integration tests (testActiveInStance) since preSolverRun requires
// a fully valid state vector and swing trajectory planner.

class BasisScalingNonNegativityConstraintTest : public ::testing::Test {
 protected:
  void SetUp() override {
    testingModelInterface_ = std::make_unique<CentroidalTestingModelInterface>();

    barrierConfig_ = PieceWisePolynomialBarrierPenalty::Config(kBarrierMu, kBarrierDelta);

    // Create a reference manager. The mode schedule is set via setModeSchedule
    // but NOT activated via preSolverRun (which would require a valid state).
    // The constraint's getValue/getQuadraticApproximation don't use the
    // reference manager — only isActive() does.
    ModeSchedule doubleStanceModeSchedule({0.0, 1.0}, {kModeDoubleStance});
    ModeSequenceTemplate doubleStanceModeSequenceTemplate({0.5, 0.5}, {kModeDoubleStance, kModeDoubleStance});
    std::shared_ptr<GaitSchedule> gaitSchedule =
        std::make_shared<GaitSchedule>(doubleStanceModeSchedule, doubleStanceModeSequenceTemplate, 0.0);

    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        gaitSchedule, nullptr, testingModelInterface_->getPinocchioInterface(), testingModelInterface_->getMpcRobotModel());

    // Create constraints for left and right feet.
    leftFootConstraint_ = std::make_unique<BasisScalingNonNegativityConstraint>(*referenceManager_, kLeftFootIndex, kLeftLambdaStart,
                                                                                kNumBasisPerFoot, barrierConfig_);
    rightFootConstraint_ = std::make_unique<BasisScalingNonNegativityConstraint>(*referenceManager_, kRightFootIndex, kRightLambdaStart,
                                                                                 kNumBasisPerFoot, barrierConfig_);
  }

  /// Creates an input vector with all λ set to the given value.
  vector_t makeInput(scalar_t lambdaValue) const {
    vector_t input = vector_t::Zero(kInputDim);
    input.head(kNumBasisPerFoot * N_CONTACTS).setConstant(lambdaValue);
    return input;
  }

  /// Creates an input vector with per-foot λ values.
  vector_t makeInputPerFoot(scalar_t leftLambda, scalar_t rightLambda) const {
    vector_t input = vector_t::Zero(kInputDim);
    input.segment(kLeftLambdaStart, kNumBasisPerFoot).setConstant(leftLambda);
    input.segment(kRightLambdaStart, kNumBasisPerFoot).setConstant(rightLambda);
    return input;
  }

  vector_t makeState() const { return vector_t::Zero(kStateDim); }
  TargetTrajectories makeEmptyTargets() const { return TargetTrajectories({}, {}, {}); }
  PreComputation makePreComp() const { return PreComputation(); }

  PieceWisePolynomialBarrierPenalty::Config barrierConfig_;
  std::unique_ptr<CentroidalTestingModelInterface> testingModelInterface_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
  std::unique_ptr<BasisScalingNonNegativityConstraint> leftFootConstraint_;
  std::unique_ptr<BasisScalingNonNegativityConstraint> rightFootConstraint_;
};

// ==================== Clone ====================

TEST_F(BasisScalingNonNegativityConstraintTest, CloneProducesSameValue) {
  std::unique_ptr<StateInputCost> clone = std::unique_ptr<StateInputCost>(leftFootConstraint_->clone());
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);  // In active penalty region (h < delta).
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t originalValue = leftFootConstraint_->getValue(kTestTime, state, input, targets, preComp);
  scalar_t cloneValue = clone->getValue(kTestTime, state, input, targets, preComp);
  EXPECT_NEAR(originalValue, cloneValue, kTolerance);
  EXPECT_GT(originalValue, 0.0) << "Cost should be positive in the active penalty region";
}

// ==================== getValue ====================

TEST_F(BasisScalingNonNegativityConstraintTest, PositiveLambdaInActiveRegion) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t value = leftFootConstraint_->getValue(kTestTime, state, input, targets, preComp);
  EXPECT_TRUE(std::isfinite(value));
  EXPECT_GT(value, 0.0) << "Penalty should be positive for 0 < λ < delta";
}

TEST_F(BasisScalingNonNegativityConstraintTest, PositiveLambdaOutsideActiveRegion) {
  const vector_t state = makeState();
  const vector_t input = makeInput(1.0);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t value = leftFootConstraint_->getValue(kTestTime, state, input, targets, preComp);
  EXPECT_NEAR(value, 0.0, kTolerance) << "Penalty should be zero when λ >> delta";
}

TEST_F(BasisScalingNonNegativityConstraintTest, NegativeLambdaProducesHighCost) {
  const vector_t state = makeState();
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t valueNeg = leftFootConstraint_->getValue(kTestTime, state, makeInput(-0.01), targets, preComp);
  scalar_t valuePos = leftFootConstraint_->getValue(kTestTime, state, makeInput(5e-4), targets, preComp);

  EXPECT_TRUE(std::isfinite(valueNeg));
  EXPECT_GT(valueNeg, valuePos) << "Negative λ should produce higher cost than positive λ";
}

TEST_F(BasisScalingNonNegativityConstraintTest, CostMonotonicallyDecreases) {
  const vector_t state = makeState();
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t prevValue = std::numeric_limits<scalar_t>::max();
  for (scalar_t lambda : {-0.01, -0.001, 0.0, 1e-4, 5e-4, 9e-4}) {
    scalar_t value = leftFootConstraint_->getValue(kTestTime, state, makeInput(lambda), targets, preComp);
    EXPECT_LT(value, prevValue) << "Cost should decrease monotonically, failed at λ=" << lambda;
    prevValue = value;
  }
}

TEST_F(BasisScalingNonNegativityConstraintTest, ValueScalesWithNumBasis) {
  const vector_t state = makeState();
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t leftValue = leftFootConstraint_->getValue(kTestTime, state, makeInput(5e-4), targets, preComp);

  PieceWisePolynomialBarrierPenalty penalty(barrierConfig_);
  scalar_t expectedValue = kNumBasisPerFoot * penalty.getValue(0.0, 5e-4);

  EXPECT_NEAR(leftValue, expectedValue, kTolerance);
}

TEST_F(BasisScalingNonNegativityConstraintTest, LeftAndRightConstraintsAreIndependent) {
  const vector_t state = makeState();
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();
  const vector_t input = makeInputPerFoot(1.0, 5e-4);

  scalar_t leftValue = leftFootConstraint_->getValue(kTestTime, state, input, targets, preComp);
  scalar_t rightValue = rightFootConstraint_->getValue(kTestTime, state, input, targets, preComp);

  EXPECT_NEAR(leftValue, 0.0, kTolerance) << "Left foot λ >> delta → zero cost";
  EXPECT_GT(rightValue, 0.0) << "Right foot λ in active region → positive cost";
}

// ==================== getQuadraticApproximation ====================

TEST_F(BasisScalingNonNegativityConstraintTest, QuadraticApproximationDimensions) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  EXPECT_EQ(approx.dfdx.size(), static_cast<Eigen::Index>(kStateDim));
  EXPECT_EQ(approx.dfdu.size(), static_cast<Eigen::Index>(kInputDim));
  EXPECT_EQ(approx.dfdxx.rows(), static_cast<Eigen::Index>(kStateDim));
  EXPECT_EQ(approx.dfdxx.cols(), static_cast<Eigen::Index>(kStateDim));
  EXPECT_EQ(approx.dfduu.rows(), static_cast<Eigen::Index>(kInputDim));
  EXPECT_EQ(approx.dfduu.cols(), static_cast<Eigen::Index>(kInputDim));
  EXPECT_EQ(approx.dfdux.rows(), static_cast<Eigen::Index>(kInputDim));
  EXPECT_EQ(approx.dfdux.cols(), static_cast<Eigen::Index>(kStateDim));
}

TEST_F(BasisScalingNonNegativityConstraintTest, ApproximationValueMatchesGetValue) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t value = leftFootConstraint_->getValue(kTestTime, state, input, targets, preComp);
  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  EXPECT_NEAR(approx.f, value, kTolerance);
}

TEST_F(BasisScalingNonNegativityConstraintTest, StateGradientIsZero) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  EXPECT_NEAR(approx.dfdx.norm(), 0.0, kTolerance);
  EXPECT_NEAR(approx.dfdxx.norm(), 0.0, kTolerance);
  EXPECT_NEAR(approx.dfdux.norm(), 0.0, kTolerance);
}

TEST_F(BasisScalingNonNegativityConstraintTest, InputGradientNonZeroOnlyInLambdaSegment) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  for (size_t i = 0; i < kInputDim; ++i) {
    if (i < kNumBasisPerFoot) {
      EXPECT_NE(approx.dfdu(i), 0.0) << "dfdu[" << i << "] should be non-zero (left λ segment)";
    } else {
      EXPECT_NEAR(approx.dfdu(i), 0.0, kTolerance) << "dfdu[" << i << "] should be zero";
    }
  }
}

TEST_F(BasisScalingNonNegativityConstraintTest, RightFootGradientInCorrectSegment) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = rightFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  for (size_t i = 0; i < kInputDim; ++i) {
    if (i >= kRightLambdaStart && i < kRightLambdaStart + kNumBasisPerFoot) {
      EXPECT_NE(approx.dfdu(i), 0.0) << "dfdu[" << i << "] should be non-zero (right λ segment)";
    } else {
      EXPECT_NEAR(approx.dfdu(i), 0.0, kTolerance) << "dfdu[" << i << "] should be zero";
    }
  }
}

TEST_F(BasisScalingNonNegativityConstraintTest, HessianIsDiagonalInActiveRegion) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  for (int i = 0; i < approx.dfduu.rows(); ++i) {
    for (int j = 0; j < approx.dfduu.cols(); ++j) {
      if (i != j) {
        EXPECT_NEAR(approx.dfduu(i, j), 0.0, kTolerance) << "Off-diagonal dfduu(" << i << "," << j << ") should be zero";
      }
    }
  }

  for (size_t i = 0; i < kNumBasisPerFoot; ++i) {
    EXPECT_GT(approx.dfduu(i, i), 0.0) << "Diagonal dfduu(" << i << "," << i << ") should be positive";
  }

  for (size_t i = kNumBasisPerFoot; i < kInputDim; ++i) {
    EXPECT_NEAR(approx.dfduu(i, i), 0.0, kTolerance);
  }
}

TEST_F(BasisScalingNonNegativityConstraintTest, HessianIsZeroOutsideActiveRegion) {
  const vector_t state = makeState();
  const vector_t input = makeInput(1.0);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  EXPECT_NEAR(approx.f, 0.0, kTolerance);
  EXPECT_NEAR(approx.dfdu.norm(), 0.0, kTolerance);
  EXPECT_NEAR(approx.dfduu.norm(), 0.0, kTolerance);
}

// ==================== Finite-difference verification ====================

TEST_F(BasisScalingNonNegativityConstraintTest, GradientMatchesFiniteDifference) {
  const vector_t state = makeState();
  const vector_t input = makeInput(-0.005);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);
  scalar_t f0 = approx.f;

  for (size_t i = 0; i < kNumBasisPerFoot; ++i) {
    vector_t input_perturbed = input;
    input_perturbed(i) += kFiniteDiffEps;
    scalar_t f_perturbed = leftFootConstraint_->getValue(kTestTime, state, input_perturbed, targets, preComp);
    scalar_t fd_gradient = (f_perturbed - f0) / kFiniteDiffEps;

    EXPECT_NEAR(approx.dfdu(i), fd_gradient, kFiniteDiffTolerance) << "Analytical gradient dfdu[" << i << "] != finite-diff";
  }
}

TEST_F(BasisScalingNonNegativityConstraintTest, HessianMatchesFiniteDifference) {
  const vector_t state = makeState();
  const vector_t input = makeInput(-0.005);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  for (size_t i = 0; i < kNumBasisPerFoot; ++i) {
    vector_t input_plus = input;
    vector_t input_minus = input;
    input_plus(i) += kFiniteDiffEps;
    input_minus(i) -= kFiniteDiffEps;

    ScalarFunctionQuadraticApproximation approx_plus =
        leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input_plus, targets, preComp);
    ScalarFunctionQuadraticApproximation approx_minus =
        leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input_minus, targets, preComp);

    scalar_t fd_hessian = (approx_plus.dfdu(i) - approx_minus.dfdu(i)) / (2.0 * kFiniteDiffEps);

    EXPECT_NEAR(approx.dfduu(i, i), fd_hessian, kFiniteDiffTolerance) << "Analytical Hessian dfduu(" << i << "," << i << ") != finite-diff";
  }
}

// ==================== Penalty boundary ====================

TEST_F(BasisScalingNonNegativityConstraintTest, PenaltyTransitionAtDelta) {
  const vector_t state = makeState();
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  scalar_t valueBelow = leftFootConstraint_->getValue(kTestTime, state, makeInput(kBarrierDelta * 0.9), targets, preComp);
  EXPECT_GT(valueBelow, 0.0);

  scalar_t valueAbove = leftFootConstraint_->getValue(kTestTime, state, makeInput(kBarrierDelta * 1.1), targets, preComp);
  EXPECT_NEAR(valueAbove, 0.0, kTolerance);
}

TEST_F(BasisScalingNonNegativityConstraintTest, MixedLambdaValues) {
  const vector_t state = makeState();
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  vector_t input = vector_t::Zero(kInputDim);
  for (size_t i = 0; i < kNumBasisPerFoot; ++i) {
    input(kLeftLambdaStart + i) = (i % 2 == 0) ? 1.0 : 5e-4;
  }

  scalar_t value = leftFootConstraint_->getValue(kTestTime, state, input, targets, preComp);
  EXPECT_GT(value, 0.0);

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);
  for (size_t i = 0; i < kNumBasisPerFoot; ++i) {
    if (i % 2 != 0) {
      EXPECT_NE(approx.dfdu(kLeftLambdaStart + i), 0.0) << "dfdu[" << i << "] should be non-zero (λ in active region)";
    } else {
      EXPECT_NEAR(approx.dfdu(kLeftLambdaStart + i), 0.0, kTolerance) << "dfdu[" << i << "] should be zero (λ >> delta)";
    }
  }
}

TEST_F(BasisScalingNonNegativityConstraintTest, UniformLambdaGivesUniformGradient) {
  const vector_t state = makeState();
  const vector_t input = makeInput(5e-4);
  const TargetTrajectories targets = makeEmptyTargets();
  const PreComputation preComp = makePreComp();

  ScalarFunctionQuadraticApproximation approx = leftFootConstraint_->getQuadraticApproximation(kTestTime, state, input, targets, preComp);

  scalar_t firstGrad = approx.dfdu(kLeftLambdaStart);
  EXPECT_NE(firstGrad, 0.0);
  for (size_t i = 1; i < kNumBasisPerFoot; ++i) {
    EXPECT_NEAR(approx.dfdu(kLeftLambdaStart + i), firstGrad, kTolerance);
  }
}

}  // namespace ocs2::humanoid
