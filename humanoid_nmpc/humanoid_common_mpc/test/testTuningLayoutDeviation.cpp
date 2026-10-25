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
#include <cstring>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/cost/TuningGaussNewton.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/orientation/TuningLayoutDeviation.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * TuningLayoutDeviation
 * (humanoid_nmpc/docs/quaternion_base_orientation/README.md, design
 * sections 2.8.1, 2.8.2 and 4.4) on the quaternion layouts of both MPCs, whose
 * robot models QuaternionLayoutRobotModel reproduces slot for slot: the base
 * orientation is four slots xi read through xi / |xi|, the whole body's base
 * angular velocity the body rate w_B.
 *   - the blocks of differences are identity runs and their rows x - x_ref bit
 * for bit, the runs after the quaternion one state column further on than their
 * tuning row; the orientation rows are the heading-tilt residual and the whole
 *     body's angular-velocity rows P_3 (w_B - R^T R_r w_B,r), with dense
 * blocks, 3 x 4 at the quaternion;
 *   - the blocks agree with finite differences of the residual in the stored
 * slots, and their pull-back to the tangent (J E(x), the solver's coordinates)
 * with finite differences along the state manifold's retraction x (+) dx; the
 *     residual reads the quaternion through its direction only (any length,
 * ambient Jacobian orthogonal to xi);
 *   - at a level state the orientation rows' tangent Jacobian is P_3 =
 * antidiag(1, 1, 1) - the rows (yaw, pitch, roll) of the body rotation (x, y,
 * z) - whatever the reference, and the whole body's angular-velocity rows' only
 * when the state and the reference rotate the same (at rest here), so the
 * tangent Gauss-Newton Hessian and gradient are the Euler differences' in body
 * coordinates to 1e-12 against a reference tilted about at most one axis; a
 *     mixed-tilt reference shifts the yaw row by the twist-vs-ZYX identity, and
 * a turning one moves the angular-velocity rows' block at the orientation off
 * zero, by -P_3 [R^T R_r w_B,r]x; through the Euler chart a tilted state keeps
 *     the bounds of design section 4.4;
 *   - the results are invariant under xi -> -xi: the residual and the tangent
 * Jacobian bit for bit (the 3 x 4 blocks change sign, as the ambient derivative
 * of a function even in xi must), and to 1e-12 for a yaw written 2 pi further
 *     on through the Euler boundary;
 *   - a malformed layout stops the program in the constructor.
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
      ModelSettings::Create(*taskFile, *urdfFile, "testTuningLayoutDeviation");
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

/** A uniform draw from [low, high]. */
scalar_t uniform(scalar_t low, scalar_t high, std::mt19937& generator) {
  return std::uniform_real_distribution<scalar_t>(low, high)(generator);
}

/** A yaw in [-4, 4], partly outside (-pi, pi], as the knots' yaws written
 * through the Euler boundary are. */
scalar_t randomYaw(std::mt19937& generator) {
  return uniform(/*low=*/-4.0, /*high=*/4.0, generator);
}

/** ZYX Euler angles (yaw, pitch, roll): a randomYaw(), and pitch and roll in
 * [-maxTilt, maxTilt]. */
vector3_t randomEulerZyx(scalar_t maxTilt, std::mt19937& generator) {
  return vector3_t(randomYaw(generator), uniform(-maxTilt, maxTilt, generator),
                   uniform(-maxTilt, maxTilt, generator));
}

/**
 * ZYX Euler angles of a reference for the attitude `eulerZyx`: a yaw within 2.5
 * rad of its yaw, so that the yaw error stays away from +-pi, where the heading
 * row jumps, and pitch and roll in [-0.3, 0.3].
 */
vector3_t referenceEulerZyxNear(const vector3_t& eulerZyx,
                                std::mt19937& generator) {
  constexpr scalar_t kMaxTilt = 0.3;
  return vector3_t(eulerZyx(0) + uniform(/*low=*/-2.5, /*high=*/2.5, generator),
                   uniform(-kMaxTilt, kMaxTilt, generator),
                   uniform(-kMaxTilt, kMaxTilt, generator));
}

/** Euler rates (or any 3-vector) with entries uniform in [-1, 1]. */
vector3_t randomRates(std::mt19937& generator) {
  return randomVector(/*size=*/3, generator);
}

/** The largest magnitude of the entries of `m`. */
scalar_t maxAbs(const matrix_t& m) {
  return m.size() == 0 ? 0.0 : m.cwiseAbs().maxCoeff();
}

/** Whether `a` and `b` have the same size and the same bits. */
bool bitwiseEqual(const matrix_t& a, const matrix_t& b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(),
                     static_cast<size_t>(a.size()) * sizeof(scalar_t)) == 0;
}

/**
 * Whether `a` and `b` have the same size and exactly the same values: bit for
 * bit but for the sign of a zero, which a product 0 * x carries from the sign
 * of x (-xi gives -0 where xi gives 0).
 */
bool exactlyEqual(const matrix_t& a, const matrix_t& b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         (a.array() == b.array()).all();
}

/** Whether two approximations have the same identity runs, the same dense
 * blocks and the same residual, bit for bit. */
bool bitwiseEqual(const TuningDeviationApproximation& a,
                  const TuningDeviationApproximation& b) {
  if (!bitwiseEqual(a.residual, b.residual) ||
      a.identityRuns.size() != b.identityRuns.size() ||
      a.denseBlocks.size() != b.denseBlocks.size()) {
    return false;
  }
  for (size_t i = 0; i < a.identityRuns.size(); ++i) {
    const TuningJacobianIdentityRun& runA = a.identityRuns[i];
    const TuningJacobianIdentityRun& runB = b.identityRuns[i];
    if (runA.tuningRow != runB.tuningRow ||
        runA.stateColumn != runB.stateColumn || runA.size != runB.size)
      return false;
  }
  for (size_t i = 0; i < a.denseBlocks.size(); ++i) {
    const TuningJacobianDenseBlock& blockA = a.denseBlocks[i];
    const TuningJacobianDenseBlock& blockB = b.denseBlocks[i];
    if (blockA.tuningRow != blockB.tuningRow ||
        blockA.stateColumn != blockB.stateColumn ||
        !bitwiseEqual(blockA.value, blockB.value)) {
      return false;
    }
  }
  return true;
}

/** J of `approximation` as a dense matrix of `stateDim` columns: the sum of its
 * identity runs and dense blocks. */
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

/** A copy of `deviation`, made by its copy constructor. */
TuningLayoutDeviation copyOf(const TuningLayoutDeviation& deviation) {
  return deviation;
}

/** A diagonal 3 x 3 weight with entries in [0.5, 100], as the task file's
 * per-row state weights are. */
matrix_t randomDiagonalWeights(std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> weight(0.5, 100.0);
  return vector3_t(weight(generator), weight(generator), weight(generator))
      .asDiagonal();
}

/** P_3 = antidiag(1, 1, 1): the tuning rows (yaw, pitch, roll) or (z, y, x) of
 * a body vector (x, y, z). */
matrix3_t headingTiltPermutation() {
  return matrix3_t::Identity().rowwise().reverse();
}

/** [a]x, the matrix of the cross product a x. */
matrix3_t crossProductMatrix(const vector3_t& a) {
  matrix3_t matrix;
  matrix << 0.0, -a(2), a(1), a(2), 0.0, -a(0), -a(1), a(0), 0.0;
  return matrix;
}

/**
 * P of the tuning layout `layout`: the tangent Jacobian of the tuning rows at a
 * level state at rest, the identity but on the rotation blocks, whose rows
 * (yaw, pitch, roll) and (z, y, x) are the tangent's body (x, y, z) reversed,
 * P_3. The tangent of the quaternion layout has the tuning layout's rows, block
 * for block.
 */
matrix_t levelTangentJacobian(const TuningLayout& layout) {
  const Eigen::Index tuningDim =
      static_cast<Eigen::Index>(getTuningLayoutDim(layout));
  matrix_t jacobian = matrix_t::Identity(tuningDim, tuningDim);
  Eigen::Index row = 0;
  for (const TuningSegment& segment : layout.segments) {
    if (segment.kind != TuningSegmentKind::kEuclidean) {
      jacobian.block<3, 3>(row, row) = headingTiltPermutation();
    }
    row += static_cast<Eigen::Index>(segment.tuningDim);
  }
  return jacobian;
}

/** A dense symmetric positive definite `size` x `size` matrix: A A' + I,
 * symmetric bit for bit. */
matrix_t randomWeights(Eigen::Index size, std::mt19937& generator) {
  matrix_t a(size, size);
  for (Eigen::Index col = 0; col < size; ++col)
    a.col(col) = randomVector(size, generator);
  const matrix_t weights = a * a.transpose() + matrix_t::Identity(size, size);
  return 0.5 * (weights + weights.transpose());
}

/** Where a dense block of a TuningDeviationApproximation sits. */
struct DenseBlockPosition {
  size_t tuningRow = 0;
  size_t stateColumn = 0;
};

/** The fixture: the quaternion-layout model of each shipped MPC and its state
 * manifold. */
class TuningLayoutDeviationTest
    : public ::testing::TestWithParam<ShippedRobot> {
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
    const absl::StatusOr<size_t> orientationRow =
        tuningSegmentStart(model_->getTuningLayout(), "base_orientation");
    ASSERT_TRUE(orientationRow.ok()) << orientationRow.status();
    orientationRow_ = *orientationRow;
    orientationStart_ = model_->getBaseOrientationStartIndex();
    angularVelocityStart_ =
        stateStartOf(model_->getTuningLayout(),
                     TuningSegmentKind::kZyxOrderedAngularVelocity);
    if (angularVelocityStart_.has_value()) {
      const absl::StatusOr<size_t> angularVelocityRow = tuningSegmentStart(
          model_->getTuningLayout(), "base_angular_velocity");
      ASSERT_TRUE(angularVelocityRow.ok()) << angularVelocityRow.status();
      angularVelocityRow_ = *angularVelocityRow;
    }
  }

  /**
   * A state with every entry uniform in [-1, 1] but the base orientation, which
   * is the quaternion of the ZYX Euler angles `eulerZyx`, and on the whole body
   * the body angular velocity, which is that of the Euler rates `eulerZyxRates`
   * there.
   */
  vector_t stateAt(const vector3_t& eulerZyx,
                   const vector3_t& eulerZyxRates,
                   std::mt19937& generator) const {
    vector_t state = randomVector(
        static_cast<Eigen::Index>(model_->getStateDim()), generator);
    model_->setBaseOrientationEulerZyx(state, eulerZyx);
    if (angularVelocityStart_.has_value()) {
      model_->setBaseAngularVelocityLocal(
          state,
          localAngularVelocityFromEulerZyxRates(eulerZyx, eulerZyxRates));
    }
    return state;
  }

  /** The tuning row of the whole body's angular velocity block. */
  size_t angularVelocityRow() const { return angularVelocityRow_; }

  /** J E(x): the Jacobian of `approximation` at `state` pulled back to the
   * tangent, the solver's coordinates. */
  matrix_t tangentJacobian(const TuningDeviationApproximation& approximation,
                           const vector_t& state) const {
    matrix_t jacobian = denseJacobian(approximation, state.size());
    manifold_->pullBackStateColumns(state, jacobian);
    return jacobian;
  }

  /** dr/d(dx) of `deviation` at (`state`, `reference`) by central finite
   * differences along the retraction state (+) dx. */
  matrix_t tangentFiniteDifferenceJacobian(
      const TuningLayoutDeviation& deviation,
      const vector_t& state,
      const vector_t& reference) const {
    const Eigen::Index tangentDim =
        static_cast<Eigen::Index>(manifold_->getTangentDim());
    matrix_t jacobian(static_cast<Eigen::Index>(deviation.getTuningDim()),
                      tangentDim);
    vector_t plus;
    vector_t minus;
    for (Eigen::Index column = 0; column < tangentDim; ++column) {
      const vector_t direction = vector_t::Unit(tangentDim, column);
      manifold_->retract(state, direction, kFiniteDifferenceStep, plus);
      manifold_->retract(state, direction, -kFiniteDifferenceStep, minus);
      jacobian.col(column) = (deviation.evaluate(plus, reference) -
                              deviation.evaluate(minus, reference)) /
                             (2.0 * kFiniteDifferenceStep);
    }
    return jacobian;
  }

  /** `state` with its stored quaternion multiplied by `scale`. */
  vector_t withScaledQuaternion(vector_t state, scalar_t scale) const {
    state.segment<4>(static_cast<Eigen::Index>(orientationStart_)) *= scale;
    return state;
  }

  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<QuaternionLayoutRobotModel> model_;
  std::shared_ptr<const StateManifold> manifold_;
  // The tuning row of the orientation block, which is also its first tangent
  // row.
  size_t orientationRow_ = 0;
  // The first state slot of the quaternion.
  size_t orientationStart_ = 0;
  // The whole body's angular velocity block: its first state slot (nullopt on
  // the centroidal layout) and tuning row.
  std::optional<size_t> angularVelocityStart_;
  size_t angularVelocityRow_ = 0;
};

TEST_P(TuningLayoutDeviationTest, ItsDimensionsAreTheModelsTangentAndState) {
  const TuningLayoutDeviation deviation(model_.get());
  EXPECT_EQ(deviation.getTuningDim(), model_->getStateTangentDim());
  EXPECT_EQ(deviation.getStateDim(), model_->getStateDim());
  EXPECT_EQ(deviation.getTuningDim(),
            tuningDimension(stateInputLayout(*settings_, GetParam().mpc)));
  EXPECT_EQ(deviation.getStateDim(), deviation.getTuningDim() + 1)
      << "the quaternion's four slots have a three-row tangent";
  EXPECT_EQ(deviation.getTuningDim(), manifold_->getTangentDim());
}

TEST_P(TuningLayoutDeviationTest,
       TheRotationBlocksAreDenseAndTheOtherBlocksIdentityRuns) {
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(7);
  const TuningDeviationApproximation approximation =
      deviation.linearize(stateAt(randomEulerZyx(/*maxTilt=*/0.3, generator),
                                  randomRates(generator), generator),
                          stateAt(randomEulerZyx(/*maxTilt=*/0.3, generator),
                                  randomRates(generator), generator));
  const TuningLayout& layout = model_->getTuningLayout();
  const absl::StatusOr<size_t> joints =
      tuningSegmentStart(layout, "joint_positions");
  ASSERT_TRUE(joints.ok()) << joints.status();
  const size_t jointDim = model_->getJointDim();
  // The quaternion has one state slot more than tuning rows, so every block
  // after it sits one state column further on than its tuning row; the blocks
  // before it at their own rows.
  ASSERT_EQ(orientationStart_, orientationRow_);
  std::vector<TuningJacobianIdentityRun> expectedRuns;
  std::vector<DenseBlockPosition> expectedBlocks;
  std::vector<Eigen::Index>
      expectedColumns;  // of each dense block: 4 at the quaternion, 3 at the
                        // angular velocity
  if (GetParam().mpc == StateInputLayout::Mpc::kCentroidal) {
    // [normalized momentum (6), base position (3)] [orientation] [joints].
    expectedRuns = {
        TuningJacobianIdentityRun{
            .tuningRow = 0, .stateColumn = 0, .size = orientationRow_},
        TuningJacobianIdentityRun{.tuningRow = *joints,
                                  .stateColumn = *joints + 1,
                                  .size = jointDim}};
    expectedBlocks = {DenseBlockPosition{.tuningRow = orientationRow_,
                                         .stateColumn = orientationStart_}};
    expectedColumns = {4};
  } else {
    const absl::StatusOr<size_t> jointVelocities =
        tuningSegmentStart(layout, "joint_velocities");
    ASSERT_TRUE(jointVelocities.ok()) << jointVelocities.status();
    // An if, not ASSERT_TRUE: bugprone-unchecked-optional-access sees the check
    // only through an if.
    if (!angularVelocityStart_.has_value())
      FAIL() << "the whole-body layout has no angular-velocity block";
    const size_t angularVelocityStart = *angularVelocityStart_;
    ASSERT_EQ(angularVelocityStart, angularVelocityRow() + 1);
    // [base position] [orientation] [joints, base linear velocity] [angular
    // velocity] [joint velocities].
    expectedRuns = {
        TuningJacobianIdentityRun{
            .tuningRow = 0, .stateColumn = 0, .size = orientationRow_},
        TuningJacobianIdentityRun{.tuningRow = *joints,
                                  .stateColumn = *joints + 1,
                                  .size = angularVelocityRow() - *joints},
        TuningJacobianIdentityRun{.tuningRow = *jointVelocities,
                                  .stateColumn = *jointVelocities + 1,
                                  .size = jointDim}};
    expectedBlocks = {DenseBlockPosition{.tuningRow = orientationRow_,
                                         .stateColumn = orientationStart_},
                      DenseBlockPosition{.tuningRow = angularVelocityRow(),
                                         .stateColumn = angularVelocityStart},
                      DenseBlockPosition{.tuningRow = angularVelocityRow(),
                                         .stateColumn = orientationStart_}};
    expectedColumns = {4, 3, 4};
  }
  ASSERT_EQ(approximation.identityRuns.size(), expectedRuns.size());
  for (size_t i = 0; i < expectedRuns.size(); ++i) {
    EXPECT_EQ(approximation.identityRuns[i].tuningRow,
              expectedRuns[i].tuningRow)
        << "run " << i;
    EXPECT_EQ(approximation.identityRuns[i].stateColumn,
              expectedRuns[i].stateColumn)
        << "run " << i;
    EXPECT_EQ(approximation.identityRuns[i].size, expectedRuns[i].size)
        << "run " << i;
  }
  ASSERT_EQ(approximation.denseBlocks.size(), expectedBlocks.size());
  for (size_t i = 0; i < expectedBlocks.size(); ++i) {
    EXPECT_EQ(approximation.denseBlocks[i].tuningRow,
              expectedBlocks[i].tuningRow)
        << "block " << i;
    EXPECT_EQ(approximation.denseBlocks[i].stateColumn,
              expectedBlocks[i].stateColumn)
        << "block " << i;
    EXPECT_EQ(approximation.denseBlocks[i].value.rows(), 3) << "block " << i;
    EXPECT_EQ(approximation.denseBlocks[i].value.cols(), expectedColumns[i])
        << "block " << i;
  }
  EXPECT_FALSE(isIdentityTuningJacobian(approximation, model_->getStateDim()))
      << "the rotation rows are no identity";
}

TEST_P(TuningLayoutDeviationTest,
       TheRowsOfTheBlocksOfDifferencesAreTheStateDifferenceBitForBit) {
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(11);
  for (size_t trial = 0; trial < 20; ++trial) {
    const vector_t state = stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                                   randomRates(generator), generator);
    const vector_t reference =
        stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                randomRates(generator), generator);
    const vector_t residual = deviation.evaluate(state, reference);
    size_t tuningRow = 0;
    for (const TuningSegment& segment : model_->getTuningLayout().segments) {
      if (segment.kind == TuningSegmentKind::kEuclidean) {
        const Eigen::Index size = static_cast<Eigen::Index>(segment.tuningDim);
        const Eigen::Index stateStart =
            static_cast<Eigen::Index>(segment.stateStart);
        const vector_t expected = state.segment(stateStart, size) -
                                  reference.segment(stateStart, size);
        EXPECT_TRUE(bitwiseEqual(
            residual.segment(static_cast<Eigen::Index>(tuningRow), size),
            expected))
            << segment.name << ", trial " << trial;
      }
      tuningRow += segment.tuningDim;
    }
  }
}

TEST_P(TuningLayoutDeviationTest, TheOrientationRowsAreTheHeadingTiltResidual) {
  const TuningLayoutDeviation deviation(model_.get());
  const Eigen::Index row = static_cast<Eigen::Index>(orientationRow_);
  std::mt19937 generator(13);
  for (size_t trial = 0; trial < 20; ++trial) {
    const vector_t state = stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                                   randomRates(generator), generator);
    const vector_t reference =
        stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                randomRates(generator), generator);
    const vector3_t expected =
        headingTiltResidual(model_->getBaseOrientation(state),
                            model_->getBaseOrientation(reference));
    EXPECT_TRUE(bitwiseEqual(
        deviation.evaluate(state, reference).segment<3>(row), expected))
        << "trial " << trial;
  }

  const vector3_t noRates = vector3_t::Zero();
  // The heading error takes the short way round: yaw 3 against -3 is 6 - 2 pi,
  // where the Euler difference was 6.
  const vector3_t acrossPi =
      deviation
          .evaluate(stateAt(vector3_t(3.0, 0.0, 0.0), noRates, generator),
                    stateAt(vector3_t(-3.0, 0.0, 0.0), noRates, generator))
          .segment<3>(row);
  EXPECT_NEAR(acrossPi(0), 6.0 - 2.0 * std::numbers::pi, 1.0e-12);
  EXPECT_EQ(acrossPi(1), 0.0);
  EXPECT_EQ(acrossPi(2), 0.0);

  // At a single-axis tilt against a level reference the rows are the Euler
  // differences, also for an unwrapped yaw.
  const vector_t level = stateAt(vector3_t(0.4, 0.0, 0.0), noRates, generator);
  for (const scalar_t yaw : {1.2, 1.2 + 4.0 * std::numbers::pi}) {
    const vector3_t pitched =
        deviation
            .evaluate(stateAt(vector3_t(yaw, 0.25, 0.0), noRates, generator),
                      level)
            .segment<3>(row);
    EXPECT_LE((pitched - vector3_t(0.8, 0.25, 0.0)).cwiseAbs().maxCoeff(),
              1.0e-12)
        << pitched.transpose();
    const vector3_t rolled =
        deviation
            .evaluate(stateAt(vector3_t(yaw, 0.0, -0.3), noRates, generator),
                      level)
            .segment<3>(row);
    EXPECT_LE((rolled - vector3_t(0.8, 0.0, -0.3)).cwiseAbs().maxCoeff(),
              1.0e-12)
        << rolled.transpose();
  }

  // The tilt rows do not depend on the yaw error.
  const vector_t tilted =
      stateAt(vector3_t(0.7, 0.2, -0.15), noRates, generator);
  const vector3_t atZeroYaw =
      deviation
          .evaluate(tilted,
                    stateAt(vector3_t(0.0, 0.05, 0.1), noRates, generator))
          .segment<3>(row);
  for (const scalar_t referenceYaw : {-2.0, 1.0, 2.5}) {
    const vector3_t atYaw =
        deviation
            .evaluate(tilted, stateAt(vector3_t(referenceYaw, 0.05, 0.1),
                                      noRates, generator))
            .segment<3>(row);
    EXPECT_NEAR(atYaw(1), atZeroYaw(1), 1.0e-12)
        << "reference yaw " << referenceYaw;
    EXPECT_NEAR(atYaw(2), atZeroYaw(2), 1.0e-12)
        << "reference yaw " << referenceYaw;
  }
}

TEST_P(TuningLayoutDeviationTest,
       TheAngularVelocityRowsAreTheBodyAngularVelocityError) {
  if (!angularVelocityStart_.has_value())
    GTEST_SKIP() << "the centroidal state holds no base angular velocity";
  const TuningLayoutDeviation deviation(model_.get());
  const Eigen::Index row = static_cast<Eigen::Index>(angularVelocityRow());
  std::mt19937 generator(17);
  for (size_t trial = 0; trial < 20; ++trial) {
    const vector3_t eulerZyx = randomEulerZyx(/*maxTilt=*/0.5, generator);
    const vector3_t referenceEulerZyx =
        randomEulerZyx(/*maxTilt=*/0.5, generator);
    const vector3_t rates = randomRates(generator);
    const vector3_t referenceRates = randomRates(generator);
    const vector_t state = stateAt(eulerZyx, rates, generator);
    const vector_t reference =
        stateAt(referenceEulerZyx, referenceRates, generator);
    // P_3 (w_B - R^T R_r w_B,r), with Eigen's rotation matrices.
    const matrix3_t rotation =
        quaternion_t(quaternionFromEulerZyx(eulerZyx)).toRotationMatrix();
    const matrix3_t referenceRotation =
        quaternion_t(quaternionFromEulerZyx(referenceEulerZyx))
            .toRotationMatrix();
    const vector3_t difference =
        localAngularVelocityFromEulerZyxRates(eulerZyx, rates) -
        rotation.transpose() * referenceRotation *
            localAngularVelocityFromEulerZyxRates(referenceEulerZyx,
                                                  referenceRates);
    const vector3_t expected(difference(2), difference(1), difference(0));
    EXPECT_LE((deviation.evaluate(state, reference).segment<3>(row) - expected)
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-12)
        << "trial " << trial;
  }

  // At a level state against a level reference of the same yaw the rows are the
  // Euler-rate differences.
  for (const scalar_t yaw : {0.0, 2.0, 3.5}) {
    const vector3_t rates(0.4, -0.7, 0.9);
    const vector3_t referenceRates(-0.2, 0.3, 0.5);
    const vector3_t rows =
        deviation
            .evaluate(
                stateAt(vector3_t(yaw, 0.0, 0.0), rates, generator),
                stateAt(vector3_t(yaw, 0.0, 0.0), referenceRates, generator))
            .segment<3>(row);
    EXPECT_LE((rows - (rates - referenceRates)).cwiseAbs().maxCoeff(), 1.0e-12)
        << "yaw " << yaw;
  }

  // Design section 4.4, E2: against a reference at rest the rows differ from
  // the Euler rates by at most |tilt| |rates|.
  for (size_t trial = 0; trial < 50; ++trial) {
    const vector3_t eulerZyx = randomEulerZyx(/*maxTilt=*/0.3, generator);
    const vector3_t rates = randomRates(generator);
    const vector3_t rows =
        deviation
            .evaluate(stateAt(eulerZyx, rates, generator),
                      stateAt(randomEulerZyx(/*maxTilt=*/0.3, generator),
                              vector3_t::Zero(), generator))
            .segment<3>(row);
    EXPECT_LE((rows - rates).norm(),
              std::hypot(eulerZyx(1), eulerZyx(2)) * rates.norm() + 1.0e-12)
        << "trial " << trial;
  }
}

TEST_P(TuningLayoutDeviationTest, EvaluateIsTheResidualOfLinearizeBitForBit) {
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(19);
  for (size_t trial = 0; trial < 20; ++trial) {
    const vector_t state = stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                                   randomRates(generator), generator);
    const vector_t reference =
        stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                randomRates(generator), generator);
    EXPECT_TRUE(bitwiseEqual(deviation.evaluate(state, reference),
                             deviation.linearize(state, reference).residual))
        << "trial " << trial;
  }
}

TEST_P(TuningLayoutDeviationTest, TheBlocksAgreeWithFiniteDifferences) {
  const TuningLayoutDeviation deviation(model_.get());
  const Eigen::Index stateDim =
      static_cast<Eigen::Index>(model_->getStateDim());
  std::mt19937 generator(23);
  for (size_t trial = 0; trial < 8; ++trial) {
    const vector3_t eulerZyx = randomEulerZyx(/*maxTilt=*/0.3, generator);
    const vector3_t referenceEulerZyx =
        referenceEulerZyxNear(eulerZyx, generator);
    const vector_t state = stateAt(eulerZyx, randomRates(generator), generator);
    const vector_t reference =
        stateAt(referenceEulerZyx, randomRates(generator), generator);
    const matrix_t jacobian =
        denseJacobian(deviation.linearize(state, reference), stateDim);
    const matrix_t finiteDifferences =
        finiteDifferenceJacobian(deviation, state, reference);
    EXPECT_LE(maxAbs(jacobian - finiteDifferences),
              1.0e-6 * std::max(1.0, maxAbs(jacobian)))
        << "trial " << trial;
  }
}

TEST_P(TuningLayoutDeviationTest,
       TheTangentJacobianAgreesWithFiniteDifferencesAlongTheRetraction) {
  // The solver's coordinates: J E(x), against the residual of x (+) dx, the
  // quaternion moved by xi (x) Exp(dx_rot).
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(53);
  for (size_t trial = 0; trial < 8; ++trial) {
    const vector3_t eulerZyx = randomEulerZyx(/*maxTilt=*/0.3, generator);
    const vector_t state = stateAt(eulerZyx, randomRates(generator), generator);
    const vector_t reference =
        stateAt(referenceEulerZyxNear(eulerZyx, generator),
                randomRates(generator), generator);
    const matrix_t jacobian =
        tangentJacobian(deviation.linearize(state, reference), state);
    ASSERT_EQ(static_cast<size_t>(jacobian.cols()), manifold_->getTangentDim());
    const matrix_t finiteDifferences =
        tangentFiniteDifferenceJacobian(deviation, state, reference);
    EXPECT_LE(maxAbs(jacobian - finiteDifferences),
              1.0e-6 * std::max(1.0, maxAbs(jacobian)))
        << "trial " << trial;
  }
}

TEST_P(TuningLayoutDeviationTest,
       TheResidualReadsTheStoredQuaternionThroughItsDirectionOnly) {
  // xi and c xi (c > 0) are the same attitude: the same residual, and an
  // ambient Jacobian orthogonal to xi (E+ xi = 0), so a solver step along xi
  // changes nothing.
  const TuningLayoutDeviation deviation(model_.get());
  const Eigen::Index orientation = static_cast<Eigen::Index>(orientationStart_);
  std::mt19937 generator(59);
  for (size_t trial = 0; trial < 10; ++trial) {
    const vector_t state = stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                                   randomRates(generator), generator);
    const vector_t reference =
        stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                randomRates(generator), generator);
    const vector_t residual = deviation.evaluate(state, reference);
    for (const scalar_t scale : {0.5, 2.0, 1.0e-3}) {
      EXPECT_LE(maxAbs(deviation.evaluate(withScaledQuaternion(state, scale),
                                          reference) -
                       residual),
                1.0e-14 * std::max(1.0, maxAbs(residual)))
          << "scale " << scale << ", trial " << trial;
      EXPECT_LE(maxAbs(deviation.evaluate(
                           state, withScaledQuaternion(reference, scale)) -
                       residual),
                1.0e-14 * std::max(1.0, maxAbs(residual)))
          << "reference scale " << scale << ", trial " << trial;
    }
    const matrix_t jacobian =
        denseJacobian(deviation.linearize(state, reference), state.size());
    const vector_t radial =
        jacobian.middleCols<4>(orientation) * state.segment<4>(orientation);
    EXPECT_LE(maxAbs(radial), 1.0e-12 * std::max(1.0, maxAbs(jacobian)))
        << "trial " << trial;
  }
}

TEST_P(TuningLayoutDeviationTest,
       AtALevelStateTheTangentGaussNewtonModelIsTheEulerDifferences) {
  // Zero pitch and roll and, on the whole body, no base rotation of the state
  // or the reference: the tangent Jacobian is P, the identity with the rotation
  // blocks' body (x, y, z) reversed into the tuning rows' (yaw, pitch, roll)
  // and (z, y, x), whatever the reference's attitude. So the tangent
  // Gauss-Newton Hessian is P' Q P, the Euler differences' Q in body
  // coordinates, and against a level or single-axis reference within pi of the
  // state's yaw the gradient is P' Q e of the Euler differences e of the
  // Euler-written states (tuningFromState(), the yaw difference wrapped).
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(29);
  const matrix_t Q = randomWeights(
      static_cast<Eigen::Index>(deviation.getTuningDim()), generator);
  const matrix_t P = levelTangentJacobian(model_->getTuningLayout());
  const vector3_t atRest = vector3_t::Zero();
  for (size_t trial = 0; trial < 12; ++trial) {
    const scalar_t yaw = randomYaw(generator);
    const vector_t state = stateAt(vector3_t(yaw, 0.0, 0.0), atRest, generator);
    const TuningDeviationApproximation anyReference = deviation.linearize(
        state,
        stateAt(randomEulerZyx(/*maxTilt=*/0.3, generator), atRest, generator));
    EXPECT_LE(maxAbs(tangentJacobian(anyReference, state) - P), 1.0e-12)
        << "trial " << trial;
    matrix_t hessian = tuningHessian(anyReference, Q, model_->getStateDim());
    manifold_->pullBackHessian(state, hessian);
    EXPECT_LE(maxAbs(hessian - P.transpose() * Q * P), 1.0e-12 * maxAbs(Q))
        << "trial " << trial;

    const vector3_t nearby =
        referenceEulerZyxNear(vector3_t(yaw, 0.0, 0.0), generator);
    const std::array<vector3_t, 3> referenceAttitudes = {
        vector3_t(nearby(0), 0.0, 0.0), vector3_t(nearby(0), nearby(1), 0.0),
        vector3_t(nearby(0), 0.0, nearby(2))};
    for (const vector3_t& referenceAttitude : referenceAttitudes) {
      const vector_t reference = stateAt(referenceAttitude, atRest, generator);
      const absl::StatusOr<vector_t> eulerState =
          model_->tuningFromState(state);
      const absl::StatusOr<vector_t> eulerReference =
          model_->tuningFromState(reference);
      ASSERT_TRUE(eulerState.ok() && eulerReference.ok());
      vector_t eulerDifferences = *eulerState - *eulerReference;
      eulerDifferences(static_cast<Eigen::Index>(orientationRow_)) =
          std::remainder(
              eulerDifferences(static_cast<Eigen::Index>(orientationRow_)),
              2.0 * std::numbers::pi);
      const vector_t eulerDifferenceGradient =
          P.transpose() * Q * eulerDifferences;
      vector_t gradient = tuningGradient(deviation.linearize(state, reference),
                                         Q, model_->getStateDim());
      manifold_->pullBackGradient(state, gradient);
      EXPECT_LE(maxAbs(gradient - eulerDifferenceGradient),
                1.0e-12 * std::max(1.0, maxAbs(eulerDifferenceGradient)))
          << "trial " << trial << ", reference "
          << referenceAttitude.transpose();
    }
  }
}

TEST_P(TuningLayoutDeviationTest,
       AtALevelStateAMixedTiltReferenceShiftsTheYawRowByTheTwistIdentity) {
  // The value and the gradient at a level state are the Euler differences' only
  // against a reference tilted about at most one axis (previous test). Against
  // a mixed tilt (theta_r, phi_r) the yaw row is wrap(psi - psi_r) plus 2
  // atan(tan(theta_r / 2) tan(phi_r / 2)) exactly, and the tilt rows differ
  // from -(theta_r, phi_r) at third order (0.13 |tilt_r|^3 at most up to 0.3
  // rad).
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(41);
  for (size_t trial = 0; trial < 30; ++trial) {
    const vector3_t levelState(randomYaw(generator), 0.0, 0.0);
    const vector3_t referenceEulerZyx =
        referenceEulerZyxNear(levelState, generator);
    const vector3_t rows =
        deviation
            .evaluate(stateAt(levelState, vector3_t::Zero(), generator),
                      stateAt(referenceEulerZyx, vector3_t::Zero(), generator))
            .segment<3>(orientationRow_);
    const scalar_t yawShift =
        2.0 * std::atan(std::tan(0.5 * referenceEulerZyx(1)) *
                        std::tan(0.5 * referenceEulerZyx(2)));
    EXPECT_NEAR(std::remainder(
                    rows(0) - (levelState(0) - referenceEulerZyx(0)) - yawShift,
                    2.0 * std::numbers::pi),
                0.0, 1.0e-12)
        << "trial " << trial;
    const scalar_t referenceTilt =
        std::hypot(referenceEulerZyx(1), referenceEulerZyx(2));
    EXPECT_LE(std::abs(rows(1) + referenceEulerZyx(1)),
              0.2 * referenceTilt * referenceTilt * referenceTilt)
        << "trial " << trial;
    EXPECT_LE(std::abs(rows(2) + referenceEulerZyx(2)),
              0.2 * referenceTilt * referenceTilt * referenceTilt)
        << "trial " << trial;
  }
}

TEST_P(
    TuningLayoutDeviationTest,
    AngularVelocityRowsOfAWorldYawRateReferenceAreTheTransportedEulerRateError) {
  // Design decision D8 and section 4.4: for a reference whose Euler rate is a
  // world yaw rate, dTheta_r = (r, 0, 0) (every shipped reference), e_w = P_3
  // T_B(Theta) v exactly at any attitude, v = dTheta - dTheta_r the Euler-rate
  // error, so |e_w - v| <= |tilt| |v|. The state stores w_B, which a rotation
  // of the base does not change, so the rows' tangent block at the orientation
  // is -P_3 [a]x, a = R^T R_r w_B,r the reference rate in the state's body
  // frame, at any attitude: zero only for a reference at rest, not for a state
  // at rest against a turning reference.
  if (!angularVelocityStart_.has_value())
    GTEST_SKIP() << "the centroidal layout has no angular-velocity rows";
  const TuningLayoutDeviation deviation(model_.get());
  const size_t row = angularVelocityRow();
  std::mt19937 generator(37);
  for (size_t trial = 0; trial < 30; ++trial) {
    const vector3_t eulerZyx = randomEulerZyx(/*maxTilt=*/0.6, generator);
    const vector3_t rates = randomRates(generator);
    const vector3_t referenceRates(
        uniform(/*low=*/-1.0, /*high=*/1.0, generator), 0.0, 0.0);
    const vector_t reference = stateAt(
        randomEulerZyx(/*maxTilt=*/0.6, generator), referenceRates, generator);
    const vector3_t rateError = rates - referenceRates;
    const vector3_t bodyRateError =
        localAngularVelocityFromEulerZyxRates(eulerZyx, rateError);
    const vector3_t rows =
        deviation.evaluate(stateAt(eulerZyx, rates, generator), reference)
            .segment<3>(row);
    EXPECT_LE(
        (rows - vector3_t(bodyRateError(2), bodyRateError(1), bodyRateError(0)))
            .cwiseAbs()
            .maxCoeff(),
        1.0e-12)
        << "trial " << trial;
    EXPECT_LE((rows - rateError).norm(),
              std::hypot(eulerZyx(1), eulerZyx(2)) * rateError.norm() + 1.0e-12)
        << "trial " << trial;

    const vector_t state = stateAt(eulerZyx, rates, generator);
    const matrix_t tangentBlock =
        tangentJacobian(deviation.linearize(state, reference), state)
            .block(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(orientationRow_), 3, 3);
    const vector3_t a =
        quaternionRotationMatrix<scalar_t>(model_->getBaseOrientation(state))
            .transpose() *
        quaternionRotationMatrix<scalar_t>(
            model_->getBaseOrientation(reference)) *
        model_->getBaseAngularVelocityLocal(reference);
    const matrix3_t expectedBlock =
        -headingTiltPermutation() * crossProductMatrix(a);
    EXPECT_LE(maxAbs(tangentBlock - expectedBlock), 1.0e-12)
        << "trial " << trial << "\n"
        << tangentBlock;
  }
  // At rest against a level reference of the same yaw turning at 0.5 rad/s: a =
  // (0, 0, 0.5), and J' Q J - Q gains 3 * 0.5^2 = 0.75 on the roll and pitch
  // diagonal (body x and y) with G1-WB's angular-velocity weights (3, 3, 3).
  const vector_t atRest =
      stateAt(vector3_t(0.3, 0.0, 0.0), vector3_t::Zero(), generator);
  const matrix_t atRestBlock =
      tangentJacobian(deviation.linearize(
                          atRest, stateAt(vector3_t(0.3, 0.0, 0.0),
                                          vector3_t(0.5, 0.0, 0.0), generator)),
                      atRest)
          .block(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(orientationRow_), 3, 3);
  const matrix_t gainedHessian =
      atRestBlock.transpose() * (3.0 * matrix_t::Identity(3, 3)) * atRestBlock;
  EXPECT_NEAR(gainedHessian(0, 0), 0.75, 1.0e-12);
  EXPECT_NEAR(gainedHessian(1, 1), 0.75, 1.0e-12);
  EXPECT_NEAR(gainedHessian(2, 2), 0.0, 1.0e-12);
}

TEST_P(
    TuningLayoutDeviationTest,
    AngularVelocityRowsOfAnotherReferenceRateDifferFromTheEulerRateErrorAtFirstOrderInItsTilt) {
  // The limit of D8's equivalence (design section 4.4, E2 whole-body rows): it
  // holds for a world-yaw-rate reference only. A reference pitched by theta_r
  // and rolling at 1 rad/s, dTheta_r = (0, 0, 1), turns about its body x axis,
  // w_B,r = T_B(Theta_r) dTheta_r = (1, 0, 0); a state at rest and level sees
  // it as a = R_y(theta_r) (1, 0, 0) = (cos, 0, -sin), so e_w = P_3 (0 - a) =
  // (sin theta_r, 0, -cos theta_r), while the Euler-rate error is (0, 0, -1):
  // they differ by 2 sin(theta_r / 2), first order in the reference's tilt.
  if (!angularVelocityStart_.has_value())
    GTEST_SKIP() << "the centroidal layout has no angular-velocity rows";
  const TuningLayoutDeviation deviation(model_.get());
  const size_t row = angularVelocityRow();
  std::mt19937 generator(41);
  std::vector<scalar_t> differences;
  for (const scalar_t referencePitch : {0.2, 0.1, 0.05}) {
    SCOPED_TRACE(referencePitch);
    const vector3_t referenceRates(0.0, 0.0, 1.0);
    const vector_t reference =
        stateAt(vector3_t(0.0, referencePitch, 0.0), referenceRates, generator);
    const vector_t atRest =
        stateAt(vector3_t::Zero(), vector3_t::Zero(), generator);
    const vector3_t rows =
        deviation.evaluate(atRest, reference).segment<3>(row);
    EXPECT_LE((rows - vector3_t(std::sin(referencePitch), 0.0,
                                -std::cos(referencePitch)))
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-12);
    const vector3_t eulerRateError = vector3_t::Zero() - referenceRates;
    differences.push_back((rows - eulerRateError).norm());
    EXPECT_NEAR(differences.back(), 2.0 * std::sin(0.5 * referencePitch),
                1.0e-12);
  }
  // First order: halving the reference's tilt halves the difference (to within
  // the sine's curvature).
  EXPECT_NEAR(differences[0] / differences[1], 2.0, 0.01);
  EXPECT_NEAR(differences[1] / differences[2], 2.0, 0.01);
  EXPECT_GT(differences[0], 0.19) << "e_w at a reference pitch of 0.2 is "
                                     "(0.199, 0, -0.980) against (0, 0, -1)";
}

TEST_P(TuningLayoutDeviationTest,
       ThroughTheEulerChartATiltedStateKeepsTheBoundsOfDesignSectionFourFour) {
  // The Euler formulation's Jacobian is the tangent one times the chart's,
  // d(body rotation)/dTheta = T_B(Theta) (and dw_B/d(dTheta/dt) = T_B(Theta) on
  // the whole body's rates): J_e = J_t T_B, the identity at a level state.
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(31);
  const matrix_t identity = matrix_t::Identity(3, 3);
  const Eigen::Index orientationRow =
      static_cast<Eigen::Index>(orientationRow_);
  const Eigen::Index rateRow = static_cast<Eigen::Index>(angularVelocityRow());
  for (const scalar_t tilt : {0.02, 0.05, 0.1, 0.2, 0.3}) {
    for (size_t direction = 0; direction < 8; ++direction) {
      const scalar_t angle =
          0.1 + static_cast<scalar_t>(direction) * std::numbers::pi / 4.0;
      const vector3_t eulerZyx(randomYaw(generator), tilt * std::cos(angle),
                               tilt * std::sin(angle));
      const vector_t state = stateAt(eulerZyx, vector3_t::Zero(), generator);
      const vector_t reference =
          stateAt(randomEulerZyx(/*maxTilt=*/0.3, generator), vector3_t::Zero(),
                  generator);
      const matrix_t jacobian =
          tangentJacobian(deviation.linearize(state, reference), state);
      const matrix3_t chart =
          eulerZyxRateToLocalAngularVelocityMatrix(eulerZyx);
      const matrix_t orientationBlock =
          jacobian.block(orientationRow, orientationRow, 3, 3) * chart;
      // ||J_e - I||_max <= 0.6 |tilt|, and the tilt does move it: the yaw row
      // picks up tan(theta / 2) and tan(phi / 2).
      const scalar_t jacobianDeviation = maxAbs(orientationBlock - identity);
      EXPECT_LE(jacobianDeviation, 0.6 * tilt)
          << "tilt " << tilt << ", direction " << direction;
      EXPECT_GE(jacobianDeviation, 0.3 * tilt)
          << "tilt " << tilt << ", direction " << direction;
      // ||J_e' Q J_e - Q||_max <= |tilt| ||Q||_max for the task file's diagonal
      // weights.
      const matrix_t Q = randomDiagonalWeights(generator);
      EXPECT_LE(maxAbs(orientationBlock.transpose() * Q * orientationBlock - Q),
                tilt * maxAbs(Q))
          << "tilt " << tilt << ", direction " << direction;
      if (angularVelocityStart_.has_value()) {
        // The angular-velocity rows' block at their own slots is P_3, through
        // the chart P_3 T_B(Theta): off the identity by at most |tilt|.
        const matrix_t rateBlock = jacobian.block(rateRow, rateRow, 3, 3);
        EXPECT_EQ(rateBlock, matrix_t(headingTiltPermutation()))
            << "tilt " << tilt << ", direction " << direction;
        EXPECT_LE(maxAbs(rateBlock * chart - identity), tilt)
            << "tilt " << tilt << ", direction " << direction;
      }
    }
  }
}

TEST_P(TuningLayoutDeviationTest,
       TheNegatedQuaternionGivesTheSameResidualAndTangentJacobianExactly) {
  // -xi is the same rotation. The residual reads xi through BaseOrientation's
  // double-cover-invariant functions, bit for bit; the ambient 3 x 4 blocks are
  // tangent Jacobians times E+(xi), which changes sign with xi, and E(-xi) =
  // -E(xi), so the pull-back J E(x) has exactly the same values (the zeros of
  // the 3 x 4 blocks change sign with them).
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(37);
  for (size_t trial = 0; trial < 20; ++trial) {
    const vector_t state = stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                                   randomRates(generator), generator);
    const vector_t reference =
        stateAt(randomEulerZyx(/*maxTilt=*/0.5, generator),
                randomRates(generator), generator);
    const vector_t negated = withScaledQuaternion(state, /*scale=*/-1.0);
    const vector_t negatedReference =
        withScaledQuaternion(reference, /*scale=*/-1.0);
    ASSERT_TRUE(bitwiseEqual(model_->getBaseOrientation(negated),
                             -model_->getBaseOrientation(state)));
    const vector_t residual = deviation.evaluate(state, reference);
    EXPECT_TRUE(bitwiseEqual(deviation.evaluate(negated, reference), residual))
        << "trial " << trial;
    EXPECT_TRUE(
        bitwiseEqual(deviation.evaluate(state, negatedReference), residual))
        << "trial " << trial;
    EXPECT_TRUE(
        bitwiseEqual(deviation.evaluate(negated, negatedReference), residual))
        << "trial " << trial;

    const TuningDeviationApproximation approximation =
        deviation.linearize(state, reference);
    const TuningDeviationApproximation negatedApproximation =
        deviation.linearize(negated, reference);
    EXPECT_TRUE(bitwiseEqual(negatedApproximation.residual, residual))
        << "trial " << trial;
    ASSERT_EQ(negatedApproximation.denseBlocks.size(),
              approximation.denseBlocks.size());
    for (size_t block = 0; block < approximation.denseBlocks.size(); ++block) {
      const bool atTheQuaternion =
          approximation.denseBlocks[block].stateColumn == orientationStart_;
      const matrix_t expected =
          atTheQuaternion ? matrix_t(-approximation.denseBlocks[block].value)
                          : approximation.denseBlocks[block].value;
      EXPECT_TRUE(
          exactlyEqual(negatedApproximation.denseBlocks[block].value, expected))
          << "block " << block << ", trial " << trial << ": max difference "
          << maxAbs(negatedApproximation.denseBlocks[block].value - expected);
    }
    const matrix_t negatedTangentJacobian =
        tangentJacobian(negatedApproximation, negated);
    const matrix_t tangent = tangentJacobian(approximation, state);
    EXPECT_TRUE(exactlyEqual(negatedTangentJacobian, tangent))
        << "trial " << trial << ": max difference "
        << maxAbs(negatedTangentJacobian - tangent);
  }
}

TEST_P(TuningLayoutDeviationTest,
       AYawWrittenTwoPiFurtherOnThroughTheEulerBoundaryGivesTheSameResults) {
  const TuningLayoutDeviation deviation(model_.get());
  std::mt19937 generator(41);
  for (size_t trial = 0; trial < 10; ++trial) {
    const vector3_t eulerZyx = randomEulerZyx(/*maxTilt=*/0.3, generator);
    const vector3_t referenceEulerZyx =
        referenceEulerZyxNear(eulerZyx, generator);
    const vector3_t rates = randomRates(generator);
    const vector_t state = stateAt(eulerZyx, rates, generator);
    const vector_t reference =
        stateAt(referenceEulerZyx, randomRates(generator), generator);
    const vector_t residual = deviation.evaluate(state, reference);
    const matrix_t jacobian =
        tangentJacobian(deviation.linearize(state, reference), state);
    for (const scalar_t turns : {-1.0, 1.0, 2.0}) {
      vector_t turned = state;
      model_->setBaseOrientationEulerZyx(
          turned, vector3_t(eulerZyx(0) + 2.0 * std::numbers::pi * turns,
                            eulerZyx(1), eulerZyx(2)));
      // An odd number of turns gives the other quaternion of the same rotation,
      // an even number the same quaternion.
      const scalar_t alignment = model_->getBaseOrientation(turned).dot(
          model_->getBaseOrientation(state));
      EXPECT_NEAR(alignment, std::fmod(turns, 2.0) == 0.0 ? 1.0 : -1.0, 1.0e-12)
          << "turns " << turns;
      for (const bool turnTheReference : {false, true}) {
        vector_t turnedReference = reference;
        if (turnTheReference) {
          model_->setBaseOrientationEulerZyx(
              turnedReference,
              vector3_t(referenceEulerZyx(0) + 2.0 * std::numbers::pi * turns,
                        referenceEulerZyx(1), referenceEulerZyx(2)));
        }
        EXPECT_LE(
            maxAbs(deviation.evaluate(turned, turnedReference) - residual),
            1.0e-12)
            << "turns " << turns << ", trial " << trial;
        EXPECT_LE(
            maxAbs(tangentJacobian(deviation.linearize(turned, turnedReference),
                                   turned) -
                   jacobian),
            1.0e-12)
            << "turns " << turns << ", trial " << trial;
      }
    }
  }
}

TEST_P(TuningLayoutDeviationTest, ACopyEvaluatesAsTheOriginal) {
  const TuningLayoutDeviation deviation(model_.get());
  const TuningLayoutDeviation copy = copyOf(deviation);
  TuningLayoutDeviation assigned(model_.get());
  assigned = copy;
  std::mt19937 generator(43);
  const vector_t state = stateAt(randomEulerZyx(/*maxTilt=*/0.3, generator),
                                 randomRates(generator), generator);
  const vector_t reference = stateAt(randomEulerZyx(/*maxTilt=*/0.3, generator),
                                     randomRates(generator), generator);
  EXPECT_TRUE(bitwiseEqual(copy.evaluate(state, reference),
                           deviation.evaluate(state, reference)));
  EXPECT_TRUE(bitwiseEqual(assigned.linearize(state, reference),
                           deviation.linearize(state, reference)));
}

INSTANTIATE_TEST_SUITE_P(
    ShippedRobots,
    TuningLayoutDeviationTest,
    ::testing::ValuesIn(kRobots),
    [](const ::testing::TestParamInfo<ShippedRobot>& info) {
      return std::string(info.param.name);
    });

/** A model of `segments` placed on a quaternion state, with the whole-body G1's
 * settings (retained in `settings`). */
std::unique_ptr<QuaternionLayoutRobotModel> modelOfSegments(
    const ModelSettings& settings, std::vector<TuningSegment> segments) {
  return std::make_unique<QuaternionLayoutRobotModel>(
      settings, StateInputLayout::Mpc::kWholeBody,
      placeTuningLayout(
          TuningLayout{.name = "malformed", .segments = std::move(segments)}));
}

TEST(TuningLayoutDeviationDeathTest,
     AnAngularVelocityBlockWithoutAnOrientationBlockStopsTheProgram) {
  const std::unique_ptr<ModelSettings> settings = settingsOf(kRobots[1]);
  ASSERT_NE(settings, nullptr);
  const std::unique_ptr<QuaternionLayoutRobotModel> model = modelOfSegments(
      *settings,
      {TuningSegment{.name = "base_angular_velocity",
                     .kind = TuningSegmentKind::kZyxOrderedAngularVelocity,
                     .tuningDim = 3}});
  EXPECT_DEATH(TuningLayoutDeviation{model.get()}.getTuningDim(),
               "no base orientation block");
}

TEST(TuningLayoutDeviationDeathTest, ASecondOrientationBlockStopsTheProgram) {
  const std::unique_ptr<ModelSettings> settings = settingsOf(kRobots[1]);
  ASSERT_NE(settings, nullptr);
  const std::unique_ptr<QuaternionLayoutRobotModel> model = modelOfSegments(
      *settings, {TuningSegment{.name = "base_orientation",
                                .kind = TuningSegmentKind::kEulerZyxOrientation,
                                .tuningDim = 3},
                  TuningSegment{.name = "head_orientation",
                                .kind = TuningSegmentKind::kEulerZyxOrientation,
                                .tuningDim = 3}});
  EXPECT_DEATH(TuningLayoutDeviation{model.get()}.getTuningDim(),
               "second base orientation block, 'head_orientation'");
}

}  // namespace
}  // namespace ocs2::humanoid
