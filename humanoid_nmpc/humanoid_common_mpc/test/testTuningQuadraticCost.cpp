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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/cost/TuningGaussNewton.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/TuningQuadraticStateInputCost.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The Tuning* quadratic costs and their block-wise Gauss-Newton assembly
 * (cost/TuningGaussNewton.h), on the quaternion layouts of both MPCs, whose
 * robot models QuaternionLayoutRobotModel reproduces slot for slot:
 *   - at a level state at rest a Tuning cost is the Euler-difference cost -
 * OCS2's QuadraticStateInputCost / QuadraticStateCost of the Euler-written
 * states' difference - to 1e-12, value and every derivative, its state
 *     derivatives pulled back to the tangent (E' g, E' H E) and read in body
 * coordinates (P' g, P' H P); on a layout of differences only it is OCS2's cost
 * of x - x_ref bit for bit;
 *   - its value and gradient agree with finite differences of its value in the
 * stored slots and along the state manifold's retraction, and its Hessian is
 * the Gauss-Newton J' Q J of the residual's finite-difference Jacobian;
 *   - the block assembly of J' Q r and J' Q J equals the dense products on the
 * quaternion layouts' rotation blocks and on a synthetic quaternion layout,
 * whose orientation and whole-body angular-velocity blocks share state columns;
 *   - the negated quaternion of the same rotation gives the same value and
 * tangent derivatives, bit for bit;
 *   - the weight setters refuse a wrong size, an asymmetric and a non-finite
 * matrix and keep the running weights.
 */
namespace ocs2::humanoid {
namespace {

/** A shipped robot configuration, from the repository root. */
struct ShippedRobot {
  const char* absl_nonnull name;
  const char* absl_nonnull taskFile;
  const char* absl_nonnull urdfFile;
  StateInputLayout::Mpc mpc;
};

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

constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;

/** The model settings of `robot`'s shipped files; a refusal fails the test. */
std::unique_ptr<ModelSettings> settingsOf(const ShippedRobot& robot) {
  const absl::StatusOr<std::string> taskFile =
      robot::resolveResourcePath(robot.taskFile);
  const absl::StatusOr<std::string> urdfFile =
      robot::resolveResourcePath(robot.urdfFile);
  EXPECT_TRUE(taskFile.ok()) << taskFile.status();
  EXPECT_TRUE(urdfFile.ok()) << urdfFile.status();
  if (!taskFile.ok() || !urdfFile.ok()) return nullptr;
  absl::StatusOr<ModelSettings> settings =
      ModelSettings::Create(*taskFile, *urdfFile, "testTuningQuadraticCost");
  EXPECT_TRUE(settings.ok()) << settings.status();
  return settings.ok() ? std::make_unique<ModelSettings>(*std::move(settings))
                       : nullptr;
}

using test::QuaternionLayoutRobotModel;
using test::stateStartOf;

/** A vector of `size` entries uniform in [-1, 1]. */
vector_t randomVector(Eigen::Index size, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> distribution(-1.0, 1.0);
  vector_t vector(size);
  for (Eigen::Index i = 0; i < size; ++i) vector(i) = distribution(generator);
  return vector;
}

/** A dense symmetric positive definite `size` x `size` matrix: A A' + I. */
matrix_t randomWeights(Eigen::Index size, std::mt19937& generator) {
  matrix_t a(size, size);
  for (Eigen::Index col = 0; col < size; ++col)
    a.col(col) = randomVector(size, generator);
  matrix_t weights = a * a.transpose() + matrix_t::Identity(size, size);
  // Symmetric bit for bit, as the converted task-file weights are.
  return 0.5 * (weights + weights.transpose());
}

/** ZYX Euler angles (yaw, pitch, roll): a yaw in [-4, 4], partly outside (-pi,
 * pi] as knot yaws are, and a tilt <= 0.3. */
vector3_t randomEulerZyx(std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> yaw(-4.0, 4.0);
  std::uniform_real_distribution<scalar_t> tilt(-0.3, 0.3);
  return vector3_t(yaw(generator), tilt(generator), tilt(generator));
}

/** The largest magnitude of the entries of `m`. */
scalar_t maxAbs(const matrix_t& m) {
  return m.size() == 0 ? 0.0 : m.cwiseAbs().maxCoeff();
}

/**
 * A layout of one block of differences over `stateDim` state rows, placed:
 * OCS2's x - x_ref, the identity Jacobian. It has no orientation block, so it
 * places the same on any orientation state dimension.
 */
TuningLayout differencesLayout(size_t stateDim) {
  return placeTuningLayout(TuningLayout{
      .name = "differences",
      .segments = {TuningSegment{.name = "state",
                                 .kind = TuningSegmentKind::kEuclidean,
                                 .tuningDim = stateDim}}});
}

/**
 * P of the tuning layout `layout`: the tangent Jacobian of the tuning rows at a
 * level state at rest, the identity but on the rotation blocks, whose rows
 * (yaw, pitch, roll) and (z, y, x) are the tangent's body (x, y, z) reversed.
 */
matrix_t levelTangentJacobian(const TuningLayout& layout) {
  const Eigen::Index tuningDim =
      static_cast<Eigen::Index>(getTuningLayoutDim(layout));
  matrix_t jacobian = matrix_t::Identity(tuningDim, tuningDim);
  Eigen::Index row = 0;
  for (const TuningSegment& segment : layout.segments) {
    if (segment.kind != TuningSegmentKind::kEuclidean)
      jacobian.block<3, 3>(row, row) =
          matrix3_t::Identity().rowwise().reverse();
    row += static_cast<Eigen::Index>(segment.tuningDim);
  }
  return jacobian;
}

/** Whether `a` and `b` have the same size and the same bits. */
bool bitwiseEqual(const matrix_t& a, const matrix_t& b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(),
                     static_cast<size_t>(a.size()) * sizeof(scalar_t)) == 0;
}

bool bitwiseEqual(scalar_t a, scalar_t b) {
  return std::memcmp(&a, &b, sizeof(scalar_t)) == 0;
}

/** A TuningQuadraticStateInputCost whose references are fixed. */
class FixedReferenceStateInputCost final
    : public TuningQuadraticStateInputCost {
 public:
  FixedReferenceStateInputCost(
      const MpcRobotModelBase<scalar_t>* absl_nonnull model,
      StateInputReference reference)
      : TuningQuadraticStateInputCost(model),
        reference_(std::move(reference)) {}
  ~FixedReferenceStateInputCost() override = default;
  FixedReferenceStateInputCost& operator=(const FixedReferenceStateInputCost&) =
      delete;
  FixedReferenceStateInputCost(FixedReferenceStateInputCost&&) = delete;
  FixedReferenceStateInputCost& operator=(FixedReferenceStateInputCost&&) =
      delete;
  FixedReferenceStateInputCost* absl_nonnull clone() const override {
    return new FixedReferenceStateInputCost(*this);
  }

 private:
  FixedReferenceStateInputCost(const FixedReferenceStateInputCost& other) =
      default;
  StateInputReference getReference(
      scalar_t /*time*/,
      const vector_t& /*state*/,
      const vector_t& /*input*/,
      const TargetTrajectories& /*targetTrajectories*/,
      const PreComputation& /*preComputation*/) const override {
    return reference_;
  }

  StateInputReference reference_;
};

/** A TuningQuadraticStateCost whose reference is fixed. */
class FixedReferenceStateCost final : public TuningQuadraticStateCost {
 public:
  FixedReferenceStateCost(const MpcRobotModelBase<scalar_t>* absl_nonnull model,
                          vector_t reference)
      : TuningQuadraticStateCost(model), reference_(std::move(reference)) {}
  ~FixedReferenceStateCost() override = default;
  FixedReferenceStateCost& operator=(const FixedReferenceStateCost&) = delete;
  FixedReferenceStateCost(FixedReferenceStateCost&&) = delete;
  FixedReferenceStateCost& operator=(FixedReferenceStateCost&&) = delete;
  FixedReferenceStateCost* absl_nonnull clone() const override {
    return new FixedReferenceStateCost(*this);
  }

 private:
  FixedReferenceStateCost(const FixedReferenceStateCost& other) = default;
  vector_t getReference(
      scalar_t /*time*/,
      const vector_t& /*state*/,
      const TargetTrajectories& /*targetTrajectories*/,
      const PreComputation& /*preComputation*/) const override {
    return reference_;
  }

  vector_t reference_;
};

/** OCS2's QuadraticStateInputCost of the same fixed references: the arithmetic
 * the Tuning cost must reproduce. */
class FixedReferenceOcs2StateInputCost final : public QuadraticStateInputCost {
 public:
  FixedReferenceOcs2StateInputCost(matrix_t Q,
                                   matrix_t R,
                                   StateInputReference reference)
      : QuadraticStateInputCost(std::move(Q), std::move(R)),
        reference_(std::move(reference)) {}
  ~FixedReferenceOcs2StateInputCost() override = default;
  FixedReferenceOcs2StateInputCost& operator=(
      const FixedReferenceOcs2StateInputCost&) = delete;
  FixedReferenceOcs2StateInputCost(FixedReferenceOcs2StateInputCost&&) = delete;
  FixedReferenceOcs2StateInputCost& operator=(
      FixedReferenceOcs2StateInputCost&&) = delete;
  FixedReferenceOcs2StateInputCost* absl_nonnull clone() const override {
    return new FixedReferenceOcs2StateInputCost(*this);
  }

 private:
  FixedReferenceOcs2StateInputCost(
      const FixedReferenceOcs2StateInputCost& other) = default;
  std::pair<vector_t, vector_t> getStateInputDeviation(
      scalar_t /*time*/,
      const vector_t& state,
      const vector_t& input,
      const TargetTrajectories& /*targetTrajectories*/) const override {
    return {state - reference_.state, input - reference_.input};
  }

  StateInputReference reference_;
};

/** OCS2's QuadraticStateCost of the same fixed reference. */
class FixedReferenceOcs2StateCost final : public QuadraticStateCost {
 public:
  FixedReferenceOcs2StateCost(matrix_t Q, vector_t reference)
      : QuadraticStateCost(std::move(Q)), reference_(std::move(reference)) {}
  ~FixedReferenceOcs2StateCost() override = default;
  FixedReferenceOcs2StateCost& operator=(const FixedReferenceOcs2StateCost&) =
      delete;
  FixedReferenceOcs2StateCost(FixedReferenceOcs2StateCost&&) = delete;
  FixedReferenceOcs2StateCost& operator=(FixedReferenceOcs2StateCost&&) =
      delete;
  FixedReferenceOcs2StateCost* absl_nonnull clone() const override {
    return new FixedReferenceOcs2StateCost(*this);
  }

 private:
  FixedReferenceOcs2StateCost(const FixedReferenceOcs2StateCost& other) =
      default;
  vector_t getStateDeviation(
      scalar_t /*time*/,
      const vector_t& state,
      const TargetTrajectories& /*targetTrajectories*/) const override {
    return state - reference_;
  }

  vector_t reference_;
};

/**
 * The fixture: the quaternion-layout model of each shipped MPC and its state
 * manifold, a model of the same state size whose layout is differences only,
 * random weights (Q of the tuning layout, Q_differences of the state's size),
 * references and a point at a tilted attitude with, on the whole body, a
 * nonzero body angular velocity.
 */
class TuningQuadraticCostTest : public ::testing::TestWithParam<ShippedRobot> {
 protected:
  void SetUp() override {
    settings_ = settingsOf(GetParam());
    ASSERT_NE(settings_, nullptr);
    model_ = std::make_unique<QuaternionLayoutRobotModel>(*settings_,
                                                          GetParam().mpc);
    absl::StatusOr<std::shared_ptr<const StateManifold>> manifold =
        stateManifoldOf(*model_);
    ASSERT_TRUE(manifold.ok()) << manifold.status();
    manifold_ = *std::move(manifold);
    differencesModel_ = std::make_unique<QuaternionLayoutRobotModel>(
        *settings_, GetParam().mpc, differencesLayout(model_->getStateDim()));
    std::mt19937 generator(17);
    stateDim_ = static_cast<Eigen::Index>(model_->getStateDim());
    inputDim_ = static_cast<Eigen::Index>(model_->getInputDim());
    ASSERT_EQ(static_cast<size_t>(stateDim_), model_->getStateTangentDim() + 1);
    Q_ = randomWeights(static_cast<Eigen::Index>(model_->getStateTangentDim()),
                       generator);
    differencesQ_ = randomWeights(stateDim_, generator);
    R_ = randomWeights(inputDim_, generator);
    reference_ =
        StateInputReference{.state = randomVector(stateDim_, generator),
                            .input = randomVector(inputDim_, generator)};
    state_ = randomVector(stateDim_, generator);
    input_ = randomVector(inputDim_, generator);
    // A yaw error within 2.5 rad, away from +-pi, where the heading row jumps.
    const vector3_t eulerZyx = randomEulerZyx(generator);
    std::uniform_real_distribution<scalar_t> yawError(-2.5, 2.5);
    model_->setEulerzyxBaseRtWorld(state_, eulerZyx);
    const vector3_t referenceTilt = randomEulerZyx(generator);
    model_->setEulerzyxBaseRtWorld(
        reference_.state, vector3_t(eulerZyx(0) + yawError(generator),
                                    referenceTilt(1), referenceTilt(2)));
  }

  /** `state` (+) step e_i on the fixture's state manifold. */
  vector_t retracted(const vector_t& state,
                     Eigen::Index i,
                     scalar_t step) const {
    vector_t result;
    manifold_->retract(
        state,
        vector_t::Unit(static_cast<Eigen::Index>(manifold_->getTangentDim()),
                       i),
        step, result);
    return result;
  }

  /** The first state slot of the base orientation. */
  Eigen::Index orientationStart() const {
    return static_cast<Eigen::Index>(model_->getBaseOrientationStartIndex());
  }

  /** The state-input cost of the fixture's weights and references on its
   * quaternion-layout model. */
  std::unique_ptr<FixedReferenceStateInputCost> stateInputCost() const {
    return stateInputCostOn(*model_, Q_);
  }

  /** The state-input cost of the state weights `Q`, the fixture's R and
   * references, on `model`. */
  std::unique_ptr<FixedReferenceStateInputCost> stateInputCostOn(
      const QuaternionLayoutRobotModel& model, const matrix_t& Q) const {
    std::unique_ptr<FixedReferenceStateInputCost> cost =
        std::make_unique<FixedReferenceStateInputCost>(&model, reference_);
    EXPECT_TRUE(cost->setStateWeights(Q).ok());
    EXPECT_TRUE(cost->setInputWeights(R_).ok());
    return cost;
  }

  /** The state cost of the fixture's state weights and state reference on its
   * quaternion-layout model. */
  std::unique_ptr<FixedReferenceStateCost> stateCost() const {
    return stateCostOn(*model_, Q_);
  }

  /** The state cost of the state weights `Q` and the fixture's state reference,
   * on `model`. */
  std::unique_ptr<FixedReferenceStateCost> stateCostOn(
      const QuaternionLayoutRobotModel& model, const matrix_t& Q) const {
    std::unique_ptr<FixedReferenceStateCost> cost =
        std::make_unique<FixedReferenceStateCost>(&model, reference_.state);
    EXPECT_TRUE(cost->setStateWeights(Q).ok());
    return cost;
  }

  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<QuaternionLayoutRobotModel> model_;
  std::shared_ptr<const StateManifold> manifold_;
  std::unique_ptr<QuaternionLayoutRobotModel> differencesModel_;
  Eigen::Index stateDim_ = 0;
  Eigen::Index inputDim_ = 0;
  matrix_t Q_;
  // The state weights of the layout of differences, which has as many rows as
  // the state.
  matrix_t differencesQ_;
  matrix_t R_;
  StateInputReference reference_;
  vector_t state_;
  vector_t input_;
  const TargetTrajectories targets_;
  const PreComputation preComputation_;
};

/** dr/dstate of `deviation` at (`state`, `reference`) by central finite
 * differences of evaluate(). */
matrix_t finiteDifferenceJacobian(const TuningLayoutDeviation& deviation,
                                  const vector_t& state,
                                  const vector_t& reference) {
  matrix_t jacobian(static_cast<Eigen::Index>(deviation.getTuningDim()),
                    state.size());
  for (Eigen::Index column = 0; column < state.size(); ++column) {
    vector_t plus = state;
    vector_t minus = state;
    plus(column) += kFiniteDifferenceStep;
    minus(column) -= kFiniteDifferenceStep;
    jacobian.col(column) = (deviation.evaluate(plus, reference) -
                            deviation.evaluate(minus, reference)) /
                           (2.0 * kFiniteDifferenceStep);
  }
  return jacobian;
}

/** J of `approximation` as a dense matrix of `stateDim` columns, the sum of its
 * blocks. */
matrix_t denseJacobian(const TuningDeviationApproximation& approximation,
                       Eigen::Index stateDim) {
  matrix_t jacobian = matrix_t::Zero(approximation.residual.size(), stateDim);
  for (const TuningJacobianIdentityRun& run : approximation.identityRuns) {
    const Eigen::Index size = static_cast<Eigen::Index>(run.size);
    jacobian.block(static_cast<Eigen::Index>(run.tuningRow),
                   static_cast<Eigen::Index>(run.stateColumn), size, size) +=
        matrix_t::Identity(size, size);
  }
  for (const TuningJacobianDenseBlock& block : approximation.denseBlocks) {
    jacobian.block(static_cast<Eigen::Index>(block.tuningRow),
                   static_cast<Eigen::Index>(block.stateColumn),
                   block.value.rows(), block.value.cols()) += block.value;
  }
  return jacobian;
}

/** Whether `actual` is `expected` to `tolerance` relative to max(1, its largest
 * entry). */
::testing::AssertionResult relativelyNear(const matrix_t& actual,
                                          const matrix_t& expected,
                                          scalar_t tolerance) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
    return ::testing::AssertionFailure()
           << actual.rows() << " x " << actual.cols() << " against "
           << expected.rows() << " x " << expected.cols();
  }
  const scalar_t difference = maxAbs(actual - expected);
  const scalar_t bound = tolerance * std::max(1.0, maxAbs(expected));
  if (difference <= bound) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "max |difference| " << difference << " > " << bound;
}

TEST_P(
    TuningQuadraticCostTest,
    TheRotationRowsAreDenseBlocksAndOnlyALayoutOfDifferencesTakesTheIdentityPath) {
  const TuningLayoutDeviation deviation(model_.get());
  EXPECT_FALSE(isIdentityTuningJacobian(
      deviation.linearize(state_, reference_.state), model_->getStateDim()));
  const TuningLayoutDeviation differences(differencesModel_.get());
  EXPECT_TRUE(
      isIdentityTuningJacobian(differences.linearize(state_, reference_.state),
                               differencesModel_->getStateDim()));
}

TEST_P(
    TuningQuadraticCostTest,
    OnALayoutOfDifferencesTheStateInputCostIsOcs2sQuadraticStateInputCostBitForBit) {
  const std::unique_ptr<FixedReferenceStateInputCost> tuning =
      stateInputCostOn(*differencesModel_, differencesQ_);
  const FixedReferenceOcs2StateInputCost ocs2(differencesQ_, R_, reference_);
  EXPECT_TRUE(bitwiseEqual(
      tuning->getValue(/*time=*/0.3, state_, input_, targets_, preComputation_),
      ocs2.getValue(/*time=*/0.3, state_, input_, targets_, preComputation_)));
  const ScalarFunctionQuadraticApproximation expected =
      ocs2.getQuadraticApproximation(/*time=*/0.3, state_, input_, targets_,
                                     preComputation_);
  const ScalarFunctionQuadraticApproximation actual =
      tuning->getQuadraticApproximation(/*time=*/0.3, state_, input_, targets_,
                                        preComputation_);
  EXPECT_TRUE(bitwiseEqual(actual.f, expected.f));
  EXPECT_TRUE(bitwiseEqual(actual.dfdx, expected.dfdx));
  EXPECT_TRUE(bitwiseEqual(actual.dfdu, expected.dfdu));
  EXPECT_TRUE(bitwiseEqual(actual.dfdxx, expected.dfdxx));
  EXPECT_TRUE(bitwiseEqual(actual.dfduu, expected.dfduu));
  EXPECT_TRUE(bitwiseEqual(actual.dfdux, expected.dfdux));
}

TEST_P(TuningQuadraticCostTest,
       OnALayoutOfDifferencesTheStateCostIsOcs2sQuadraticStateCostBitForBit) {
  const std::unique_ptr<FixedReferenceStateCost> tuning =
      stateCostOn(*differencesModel_, differencesQ_);
  const FixedReferenceOcs2StateCost ocs2(differencesQ_, reference_.state);
  EXPECT_TRUE(bitwiseEqual(
      tuning->getValue(/*time=*/0.3, state_, targets_, preComputation_),
      ocs2.getValue(/*time=*/0.3, state_, targets_, preComputation_)));
  const ScalarFunctionQuadraticApproximation expected =
      ocs2.getQuadraticApproximation(/*time=*/0.3, state_, targets_,
                                     preComputation_);
  const ScalarFunctionQuadraticApproximation actual =
      tuning->getQuadraticApproximation(/*time=*/0.3, state_, targets_,
                                        preComputation_);
  EXPECT_TRUE(bitwiseEqual(actual.f, expected.f));
  EXPECT_TRUE(bitwiseEqual(actual.dfdx, expected.dfdx));
  EXPECT_TRUE(bitwiseEqual(actual.dfdxx, expected.dfdxx));
}

TEST_P(TuningQuadraticCostTest,
       AtALevelStateAtRestTheCostsAreTheEulerDifferenceCosts) {
  // Zero pitch and roll and, on the whole body, no base rotation; a level or
  // single-axis reference within pi of its yaw. The Euler-difference cost is
  // OCS2's of the Euler-written states (tuningFromState(), n_t rows), the yaw
  // difference wrapped as the heading row is; the Tuning cost's state
  // derivatives pulled back to the tangent are its derivatives in body
  // coordinates, P' g and P' H P.
  std::mt19937 generator(19);
  std::uniform_real_distribution<scalar_t> yawError(-2.5, 2.5);
  std::uniform_real_distribution<scalar_t> tilt(-0.3, 0.3);
  const std::optional<size_t> angularVelocityStart = stateStartOf(
      model_->getTuningLayout(), TuningSegmentKind::kZyxOrderedAngularVelocity);
  const absl::StatusOr<size_t> yawRow =
      tuningSegmentStart(model_->getTuningLayout(), "eulerzyx_base_rt_world");
  ASSERT_TRUE(yawRow.ok()) << yawRow.status();
  const matrix_t P = levelTangentJacobian(model_->getTuningLayout());
  for (size_t trial = 0; trial < 12; ++trial) {
    vector_t state = state_;
    vector_t referenceState = reference_.state;
    const scalar_t yaw = randomEulerZyx(generator)(0);
    const scalar_t referenceYaw = yaw + yawError(generator);
    const std::array<vector3_t, 3> referenceAttitudes = {
        vector3_t(referenceYaw, 0.0, 0.0),
        vector3_t(referenceYaw, tilt(generator), 0.0),
        vector3_t(referenceYaw, 0.0, tilt(generator))};
    model_->setEulerzyxBaseRtWorld(state, vector3_t(yaw, 0.0, 0.0));
    model_->setEulerzyxBaseRtWorld(
        referenceState, referenceAttitudes[trial % referenceAttitudes.size()]);
    if (angularVelocityStart.has_value()) {
      state.segment<3>(static_cast<Eigen::Index>(*angularVelocityStart))
          .setZero();
      referenceState
          .segment<3>(static_cast<Eigen::Index>(*angularVelocityStart))
          .setZero();
    }
    const StateInputReference reference{.state = referenceState,
                                        .input = reference_.input};
    const absl::StatusOr<vector_t> eulerState = model_->tuningFromState(state);
    absl::StatusOr<vector_t> eulerReference =
        model_->tuningFromState(referenceState);
    ASSERT_TRUE(eulerState.ok() && eulerReference.ok());
    const Eigen::Index yawIndex = static_cast<Eigen::Index>(*yawRow);
    (*eulerReference)(yawIndex) = (*eulerState)(yawIndex)-std::remainder(
        (*eulerState)(yawIndex) - (*eulerReference)(yawIndex),
        2.0 * std::numbers::pi);
    const StateInputReference eulerDifferenceReference{
        .state = *eulerReference, .input = reference_.input};

    FixedReferenceStateInputCost tuning(model_.get(), reference);
    ASSERT_TRUE(tuning.setStateWeights(Q_).ok());
    ASSERT_TRUE(tuning.setInputWeights(R_).ok());
    const FixedReferenceOcs2StateInputCost eulerDifferenceCost(
        Q_, R_, eulerDifferenceReference);
    const ScalarFunctionQuadraticApproximation expected =
        eulerDifferenceCost.getQuadraticApproximation(
            /*time=*/0.0, *eulerState, input_, targets_, preComputation_);
    ScalarFunctionQuadraticApproximation actual =
        tuning.getQuadraticApproximation(/*time=*/0.0, state, input_, targets_,
                                         preComputation_);
    manifold_->pullBackGradient(state, actual.dfdx);
    manifold_->pullBackHessian(state, actual.dfdxx);
    EXPECT_NEAR(
        tuning.getValue(/*time=*/0.0, state, input_, targets_, preComputation_),
        expected.f, 1.0e-12 * std::max(1.0, expected.f))
        << "trial " << trial;
    EXPECT_NEAR(actual.f, expected.f, 1.0e-12 * std::max(1.0, expected.f))
        << "trial " << trial;
    EXPECT_TRUE(relativelyNear(actual.dfdx, P.transpose() * expected.dfdx,
                               /*tolerance=*/1.0e-12))
        << "trial " << trial;
    EXPECT_TRUE(relativelyNear(actual.dfdxx, P.transpose() * expected.dfdxx * P,
                               /*tolerance=*/1.0e-12))
        << "trial " << trial;
    EXPECT_TRUE(bitwiseEqual(actual.dfdu, expected.dfdu)) << "trial " << trial;
    EXPECT_TRUE(bitwiseEqual(actual.dfduu, expected.dfduu))
        << "trial " << trial;
    EXPECT_TRUE(actual.dfdux.isZero(0.0)) << "trial " << trial;
    EXPECT_EQ(actual.dfdux.cols(), stateDim_)
        << "the cross term has the ambient state's columns, which the solver "
           "pulls back";

    FixedReferenceStateCost terminal(model_.get(), referenceState);
    ASSERT_TRUE(terminal.setStateWeights(Q_).ok());
    const FixedReferenceOcs2StateCost terminalEulerDifferenceCost(
        Q_, *eulerReference);
    const ScalarFunctionQuadraticApproximation expectedTerminal =
        terminalEulerDifferenceCost.getQuadraticApproximation(
            /*time=*/0.0, *eulerState, targets_, preComputation_);
    ScalarFunctionQuadraticApproximation actualTerminal =
        terminal.getQuadraticApproximation(/*time=*/0.0, state, targets_,
                                           preComputation_);
    manifold_->pullBackGradient(state, actualTerminal.dfdx);
    manifold_->pullBackHessian(state, actualTerminal.dfdxx);
    EXPECT_NEAR(actualTerminal.f, expectedTerminal.f,
                1.0e-12 * std::max(1.0, expectedTerminal.f))
        << "trial " << trial;
    EXPECT_TRUE(relativelyNear(actualTerminal.dfdx,
                               P.transpose() * expectedTerminal.dfdx,
                               /*tolerance=*/1.0e-12))
        << "trial " << trial;
    EXPECT_TRUE(relativelyNear(actualTerminal.dfdxx,
                               P.transpose() * expectedTerminal.dfdxx * P,
                               /*tolerance=*/1.0e-12))
        << "trial " << trial;
  }
}

TEST_P(TuningQuadraticCostTest,
       ValueGradientAndHessianAgreeWithFiniteDifferences) {
  const std::unique_ptr<FixedReferenceStateInputCost> cost = stateInputCost();
  const ScalarFunctionQuadraticApproximation approximation =
      cost->getQuadraticApproximation(/*time=*/0.0, state_, input_, targets_,
                                      preComputation_);
  EXPECT_DOUBLE_EQ(approximation.f, cost->getValue(/*time=*/0.0, state_, input_,
                                                   targets_, preComputation_));
  for (Eigen::Index i = 0; i < stateDim_; ++i) {
    vector_t plus = state_;
    vector_t minus = state_;
    plus(i) += kFiniteDifferenceStep;
    minus(i) -= kFiniteDifferenceStep;
    const scalar_t derivative =
        (cost->getValue(/*time=*/0.0, plus, input_, targets_, preComputation_) -
         cost->getValue(/*time=*/0.0, minus, input_, targets_,
                        preComputation_)) /
        (2.0 * kFiniteDifferenceStep);
    EXPECT_NEAR(approximation.dfdx(i), derivative,
                1.0e-6 * std::max(1.0, std::abs(derivative)))
        << "state " << i;
  }
  // The Hessian is the Gauss-Newton one, J' Q J of the residual's Jacobian (the
  // rotation rows are not linear in x).
  const matrix_t jacobian = finiteDifferenceJacobian(
      TuningLayoutDeviation(model_.get()), state_, reference_.state);
  EXPECT_TRUE(relativelyNear(approximation.dfdxx,
                             jacobian.transpose() * Q_ * jacobian,
                             /*tolerance=*/1.0e-6));
  for (Eigen::Index i = 0; i < inputDim_; ++i) {
    vector_t plus = input_;
    vector_t minus = input_;
    plus(i) += kFiniteDifferenceStep;
    minus(i) -= kFiniteDifferenceStep;
    const scalar_t derivative =
        (cost->getValue(/*time=*/0.0, state_, plus, targets_, preComputation_) -
         cost->getValue(/*time=*/0.0, state_, minus, targets_,
                        preComputation_)) /
        (2.0 * kFiniteDifferenceStep);
    EXPECT_NEAR(approximation.dfdu(i), derivative,
                1.0e-6 * std::max(1.0, std::abs(derivative)))
        << "input " << i;
  }
  EXPECT_TRUE(approximation.dfdux.isZero(0.0));

  const std::unique_ptr<FixedReferenceStateCost> terminal = stateCost();
  const ScalarFunctionQuadraticApproximation terminalApproximation =
      terminal->getQuadraticApproximation(/*time=*/0.0, state_, targets_,
                                          preComputation_);
  EXPECT_DOUBLE_EQ(
      terminalApproximation.f,
      terminal->getValue(/*time=*/0.0, state_, targets_, preComputation_));
  EXPECT_TRUE(relativelyNear(terminalApproximation.dfdx,
                             jacobian.transpose() * Q_ *
                                 TuningLayoutDeviation(model_.get())
                                     .evaluate(state_, reference_.state),
                             /*tolerance=*/1.0e-6));
  EXPECT_TRUE(relativelyNear(terminalApproximation.dfdxx,
                             jacobian.transpose() * Q_ * jacobian,
                             /*tolerance=*/1.0e-6));
}

TEST_P(TuningQuadraticCostTest,
       ThePulledBackGradientIsTheDerivativeAlongTheRetraction) {
  // The solver's coordinates: E(x)' dfdx against central differences of the
  // value at x (+) (+-h e_i).
  const std::unique_ptr<FixedReferenceStateInputCost> cost = stateInputCost();
  vector_t gradient =
      cost->getQuadraticApproximation(/*time=*/0.0, state_, input_, targets_,
                                      preComputation_)
          .dfdx;
  manifold_->pullBackGradient(state_, gradient);
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold_->getTangentDim());
  ASSERT_EQ(gradient.size(), tangentDim);
  for (Eigen::Index i = 0; i < tangentDim; ++i) {
    const scalar_t derivative =
        (cost->getValue(/*time=*/0.0,
                        retracted(state_, i, kFiniteDifferenceStep), input_,
                        targets_, preComputation_) -
         cost->getValue(/*time=*/0.0,
                        retracted(state_, i, -kFiniteDifferenceStep), input_,
                        targets_, preComputation_)) /
        (2.0 * kFiniteDifferenceStep);
    EXPECT_NEAR(gradient(i), derivative,
                1.0e-6 * std::max(1.0, std::abs(derivative)))
        << "tangent " << i;
  }
}

TEST_P(TuningQuadraticCostTest,
       TheBlockAssemblyIsTheDenseProductOnTheQuaternionLayoutsRotationBlocks) {
  const TuningDeviationApproximation approximation =
      TuningLayoutDeviation(model_.get()).linearize(state_, reference_.state);
  ASSERT_FALSE(approximation.denseBlocks.empty());
  const matrix_t J = denseJacobian(approximation, stateDim_);
  const vector_t expectedGradient = J.transpose() * Q_ * approximation.residual;
  const matrix_t expectedHessian = J.transpose() * Q_ * J;
  EXPECT_TRUE(
      relativelyNear(tuningGradient(approximation, Q_, model_->getStateDim()),
                     expectedGradient, /*tolerance=*/1.0e-12));
  EXPECT_TRUE(
      relativelyNear(tuningHessian(approximation, Q_, model_->getStateDim()),
                     expectedHessian, /*tolerance=*/1.0e-12));
}

TEST_P(TuningQuadraticCostTest,
       TheNegatedQuaternionOfTheSameRotationGivesTheSameCostBitForBit) {
  // -xi stored in the state: the same value, and the same tangent gradient and
  // Hessian, bit for bit (the ambient ones change sign on the quaternion's rows
  // and columns, E(-xi) = -E(xi)).
  const std::unique_ptr<FixedReferenceStateInputCost> cost = stateInputCost();
  vector_t negated = state_;
  negated.segment<4>(orientationStart()) *= -1.0;
  EXPECT_TRUE(bitwiseEqual(
      cost->getValue(/*time=*/0.0, negated, input_, targets_, preComputation_),
      cost->getValue(/*time=*/0.0, state_, input_, targets_, preComputation_)));
  ScalarFunctionQuadraticApproximation expected =
      cost->getQuadraticApproximation(/*time=*/0.0, state_, input_, targets_,
                                      preComputation_);
  ScalarFunctionQuadraticApproximation actual = cost->getQuadraticApproximation(
      /*time=*/0.0, negated, input_, targets_, preComputation_);
  EXPECT_TRUE(bitwiseEqual(actual.f, expected.f));
  manifold_->pullBackGradient(state_, expected.dfdx);
  manifold_->pullBackHessian(state_, expected.dfdxx);
  manifold_->pullBackGradient(negated, actual.dfdx);
  manifold_->pullBackHessian(negated, actual.dfdxx);
  EXPECT_TRUE(bitwiseEqual(actual.dfdx, expected.dfdx));
  EXPECT_TRUE(bitwiseEqual(actual.dfdxx, expected.dfdxx));
  EXPECT_TRUE(bitwiseEqual(actual.dfdu, expected.dfdu));
}

TEST_P(TuningQuadraticCostTest, ACloneEvaluatesAsTheOriginal) {
  const std::unique_ptr<FixedReferenceStateInputCost> cost = stateInputCost();
  const std::unique_ptr<FixedReferenceStateInputCost> clone(cost->clone());
  EXPECT_TRUE(bitwiseEqual(clone->getStateWeights(), Q_));
  EXPECT_TRUE(bitwiseEqual(clone->getInputWeights(), R_));
  EXPECT_TRUE(bitwiseEqual(
      clone->getValue(/*time=*/0.0, state_, input_, targets_, preComputation_),
      cost->getValue(/*time=*/0.0, state_, input_, targets_, preComputation_)));
  EXPECT_EQ(cost->getTuningDim(), model_->getStateTangentDim());
}

TEST_P(TuningQuadraticCostTest,
       TheSettersRefuseWeightsTheyCannotWeighWithAndKeepTheRunningOnes) {
  const std::unique_ptr<FixedReferenceStateInputCost> cost = stateInputCost();
  const Eigen::Index tuningDim =
      static_cast<Eigen::Index>(cost->getTuningDim());

  const absl::Status wrongSize =
      cost->setStateWeights(matrix_t::Identity(tuningDim + 1, tuningDim + 1));
  EXPECT_EQ(wrongSize.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(wrongSize.message(), "state weights"))
      << wrongSize;
  matrix_t asymmetric = Q_;
  asymmetric(0, 1) += 1.0e-3;
  EXPECT_EQ(cost->setStateWeights(asymmetric).code(),
            absl::StatusCode::kInvalidArgument);
  matrix_t notFinite = Q_;
  notFinite(1, 1) = std::numeric_limits<scalar_t>::quiet_NaN();
  EXPECT_EQ(cost->setStateWeights(notFinite).code(),
            absl::StatusCode::kInvalidArgument);
  notFinite(1, 1) = std::numeric_limits<scalar_t>::infinity();
  EXPECT_EQ(cost->setStateWeights(notFinite).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      cost->setInputWeights(matrix_t::Identity(inputDim_ - 1, inputDim_ - 1))
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(bitwiseEqual(cost->getStateWeights(), Q_))
      << "a refused Q replaced the running one";
  EXPECT_TRUE(bitwiseEqual(cost->getInputWeights(), R_))
      << "a refused R replaced the running one";

  // Round-off asymmetry, as M' R_wrench M of the basis-vector inputs has, is
  // accepted.
  matrix_t roundOff = R_;
  roundOff(0, 1) *= 1.0 + 1.0e-14;
  EXPECT_TRUE(cost->setInputWeights(roundOff).ok());
  // An accepted reload replaces the weights the cost evaluates with.
  const matrix_t doubled = 2.0 * Q_;
  ASSERT_TRUE(cost->setStateWeights(doubled).ok());
  EXPECT_TRUE(bitwiseEqual(cost->getStateWeights(), doubled));
  EXPECT_TRUE(
      bitwiseEqual(cost->getQuadraticApproximation(/*time=*/0.0, state_, input_,
                                                   targets_, preComputation_)
                       .dfdxx,
                   tuningHessian(TuningLayoutDeviation(model_.get())
                                     .linearize(state_, reference_.state),
                                 doubled, model_->getStateDim())));

  const std::unique_ptr<FixedReferenceStateCost> terminal = stateCost();
  EXPECT_EQ(
      terminal
          ->setStateWeights(matrix_t::Identity(tuningDim - 1, tuningDim - 1))
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(bitwiseEqual(terminal->getStateWeights(), Q_));
}

INSTANTIATE_TEST_SUITE_P(
    ShippedRobots,
    TuningQuadraticCostTest,
    ::testing::ValuesIn(kRobots),
    [](const ::testing::TestParamInfo<ShippedRobot>& info) {
      return std::string(info.param.name);
    });

/******************************************************************************************************/
/*                     The block assembly against the dense products, on dense
 * blocks                 */
/******************************************************************************************************/

/** A dense `rows` x `cols` block of random entries. */
matrix_t randomBlock(Eigen::Index rows,
                     Eigen::Index cols,
                     std::mt19937& generator) {
  matrix_t block(rows, cols);
  for (Eigen::Index col = 0; col < cols; ++col)
    block.col(col) = randomVector(rows, generator);
  return block;
}

/**
 * The blocks of the whole-body Jacobian on a quaternion layout, n joints: base
 * position (identity), orientation (3 x 4 at the quaternion), joints, base
 * linear velocity (identities shifted by the quaternion's extra row), the
 * angular velocity rows (one 3 x 3 block at the angular-velocity slots and one
 * 3 x 4 at the quaternion, sharing the orientation block's columns) and the
 * joint velocities.
 */
TuningDeviationApproximation unitQuaternionWholeBodyShape(
    size_t joints, std::mt19937& generator) {
  const size_t tuningDim = 2 * (6 + joints);
  TuningDeviationApproximation approximation;
  approximation.residual =
      randomVector(static_cast<Eigen::Index>(tuningDim), generator);
  approximation.identityRuns = {
      TuningJacobianIdentityRun{.tuningRow = 0, .stateColumn = 0, .size = 3},
      TuningJacobianIdentityRun{
          .tuningRow = 6, .stateColumn = 7, .size = joints},
      TuningJacobianIdentityRun{
          .tuningRow = 6 + joints, .stateColumn = 7 + joints, .size = 3},
      TuningJacobianIdentityRun{
          .tuningRow = 12 + joints, .stateColumn = 13 + joints, .size = joints},
  };
  approximation.denseBlocks = {
      TuningJacobianDenseBlock{
          .tuningRow = 3,
          .stateColumn = 3,
          .value = randomBlock(/*rows=*/3, /*cols=*/4, generator)},
      TuningJacobianDenseBlock{
          .tuningRow = 9 + joints,
          .stateColumn = 10 + joints,
          .value = randomBlock(/*rows=*/3, /*cols=*/3, generator)},
      TuningJacobianDenseBlock{
          .tuningRow = 9 + joints,
          .stateColumn = 3,
          .value = randomBlock(/*rows=*/3, /*cols=*/4, generator)},
  };
  return approximation;
}

TEST(TuningGaussNewtonTest,
     TheBlockAssemblyIsTheDenseProductOnALayoutWithDenseBlocks) {
  std::mt19937 generator(23);
  constexpr size_t kJoints = 7;
  const TuningDeviationApproximation approximation =
      unitQuaternionWholeBodyShape(kJoints, generator);
  const Eigen::Index tuningDim = approximation.residual.size();
  const Eigen::Index stateDim = tuningDim + 1;
  const matrix_t Q = randomWeights(tuningDim, generator);
  ASSERT_FALSE(
      isIdentityTuningJacobian(approximation, static_cast<size_t>(stateDim)));

  const matrix_t J = denseJacobian(approximation, stateDim);
  const vector_t expectedGradient = J.transpose() * Q * approximation.residual;
  const matrix_t expectedHessian = J.transpose() * Q * J;
  const vector_t gradient =
      tuningGradient(approximation, Q, static_cast<size_t>(stateDim));
  const matrix_t hessian =
      tuningHessian(approximation, Q, static_cast<size_t>(stateDim));
  ASSERT_EQ(gradient.size(), stateDim);
  ASSERT_EQ(hessian.rows(), stateDim);
  ASSERT_EQ(hessian.cols(), stateDim);
  EXPECT_LE((gradient - expectedGradient).cwiseAbs().maxCoeff(),
            1.0e-12 * std::max(1.0, expectedGradient.cwiseAbs().maxCoeff()));
  EXPECT_LE((hessian - expectedHessian).cwiseAbs().maxCoeff(),
            1.0e-12 * std::max(1.0, expectedHessian.cwiseAbs().maxCoeff()));
}

TEST(TuningGaussNewtonTest, SplitIdentityRunsAssembleAsTheIdentity) {
  std::mt19937 generator(29);
  TuningDeviationApproximation approximation;
  approximation.residual = randomVector(/*size=*/9, generator);
  approximation.identityRuns = {
      TuningJacobianIdentityRun{.tuningRow = 0, .stateColumn = 0, .size = 4},
      TuningJacobianIdentityRun{.tuningRow = 4, .stateColumn = 4, .size = 5},
  };
  const matrix_t Q = randomWeights(/*size=*/9, generator);
  ASSERT_FALSE(isIdentityTuningJacobian(approximation, /*stateDim=*/9))
      << "two runs take the general path";
  EXPECT_LE((tuningGradient(approximation, Q, /*stateDim=*/9) -
             Q * approximation.residual)
                .cwiseAbs()
                .maxCoeff(),
            1.0e-12);
  EXPECT_LE((tuningHessian(approximation, Q, /*stateDim=*/9) - Q)
                .cwiseAbs()
                .maxCoeff(),
            0.0);
}

TEST(TuningGaussNewtonTest, OnlyOneRunOverTheWholeStateIsTheIdentity) {
  TuningDeviationApproximation identity;
  identity.residual = vector_t::Zero(6);
  identity.identityRuns = {
      TuningJacobianIdentityRun{.tuningRow = 0, .stateColumn = 0, .size = 6}};
  EXPECT_TRUE(isIdentityTuningJacobian(identity, /*stateDim=*/6));
  EXPECT_FALSE(isIdentityTuningJacobian(identity, /*stateDim=*/7))
      << "a state with a column no run reaches";

  TuningDeviationApproximation shifted = identity;
  shifted.identityRuns.front().stateColumn = 1;
  EXPECT_FALSE(isIdentityTuningJacobian(shifted, /*stateDim=*/6));

  TuningDeviationApproximation withBlock = identity;
  withBlock.denseBlocks = {TuningJacobianDenseBlock{
      .tuningRow = 0, .stateColumn = 0, .value = matrix_t::Zero(1, 1)}};
  EXPECT_FALSE(isIdentityTuningJacobian(withBlock, /*stateDim=*/6));
}

TEST(TuningGaussNewtonTest, ValidateQuadraticWeightsNamesWhatIsWrong) {
  EXPECT_TRUE(
      validateQuadraticWeights(matrix_t::Identity(3, 3), /*dimension=*/3, "Q")
          .ok());
  EXPECT_TRUE(validateQuadraticWeights(matrix_t(), /*dimension=*/0, "Q").ok());
  const absl::Status wrongSize = validateQuadraticWeights(
      matrix_t::Identity(3, 3), /*dimension=*/4, "the weights");
  EXPECT_EQ(wrongSize.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(wrongSize.message(), "the weights is 3 x 3"))
      << wrongSize;
  matrix_t asymmetric = matrix_t::Identity(3, 3);
  asymmetric(2, 0) = 0.5;
  EXPECT_TRUE(absl::StrContains(
      validateQuadraticWeights(asymmetric, /*dimension=*/3, "Q").message(),
      "not symmetric"));
  matrix_t notFinite = matrix_t::Identity(3, 3);
  notFinite(0, 0) = -std::numeric_limits<scalar_t>::infinity();
  EXPECT_TRUE(absl::StrContains(
      validateQuadraticWeights(notFinite, /*dimension=*/3, "Q").message(),
      "not a finite number"));
}

TEST(TuningGaussNewtonTest, ValidateQuadraticWeightsRefusesAnIndefiniteMatrix) {
  // A negative diagonal entry.
  matrix_t negative = matrix_t::Identity(3, 3);
  negative(1, 1) = -1.0;
  EXPECT_TRUE(absl::StrContains(
      validateQuadraticWeights(negative, /*dimension=*/3, "Q").message(),
      "not positive semidefinite"));
  // A non-negative diagonal with an off-diagonal coupling that makes it
  // indefinite: [[1, 2], [2, 1]] has eigenvalue -1.
  matrix_t coupled(2, 2);
  coupled << 1.0, 2.0, 2.0, 1.0;
  EXPECT_FALSE(validateQuadraticWeights(coupled, /*dimension=*/2, "R").ok());
  // A random M' D M with D >= 0, singular included, is accepted (the
  // basis-space R).
  std::srand(/*seed=*/7);
  for (int trial = 0; trial < 20; ++trial) {
    const matrix_t basis = matrix_t::Random(6, 4);
    vector_t diagonal = vector_t::Random(6).cwiseAbs();
    diagonal(trial % 6) = 0.0;
    const matrix_t projected =
        basis.transpose() * diagonal.asDiagonal() * basis;
    EXPECT_TRUE(validateQuadraticWeights(projected, /*dimension=*/4, "R").ok())
        << projected;
  }
  // Zero weighs nothing and is accepted.
  EXPECT_TRUE(
      validateQuadraticWeights(matrix_t::Zero(3, 3), /*dimension=*/3, "Q")
          .ok());
}

}  // namespace
}  // namespace ocs2::humanoid
