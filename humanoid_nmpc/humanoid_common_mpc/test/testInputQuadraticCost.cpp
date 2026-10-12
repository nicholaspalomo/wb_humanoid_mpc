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

#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/cost/InputQuadraticCost.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/LayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * InputQuadraticCost, the input-only quadratic cost: it is OCS2's QuadraticStateInputCost with a zero Q, bit for bit,
 * without holding a state weight - its state derivatives are zeros of the evaluated state's size, whichever that is
 * (the zeroQ hazard: a hot reload wrote a zero Q of the tuning layout's size, which a quaternion state's ambient size
 * does not have) - and a hot reload of input_weights replaces only R.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto";
constexpr char kUrdfFile[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** Whether `a` and `b` have the same size and the same bits. */
bool bitwiseEqual(const matrix_t& a, const matrix_t& b) {
  return a.rows() == b.rows() && a.cols() == b.cols() &&
         std::memcmp(a.data(), b.data(), static_cast<size_t>(a.size()) * sizeof(scalar_t)) == 0;
}

bool bitwiseEqual(scalar_t a, scalar_t b) {
  return std::memcmp(&a, &b, sizeof(scalar_t)) == 0;
}

/** The input cost of before: OCS2's QuadraticStateInputCost of a zero Q, the state deviation zero. */
class ZeroStateWeightOcs2Cost final : public QuadraticStateInputCost {
 public:
  ZeroStateWeightOcs2Cost(size_t stateDim, matrix_t R, const SwitchedModelReferenceManager* absl_nonnull referenceManager)
      : QuadraticStateInputCost(matrix_t::Zero(static_cast<Eigen::Index>(stateDim), static_cast<Eigen::Index>(stateDim)), std::move(R)),
        stateDim_(stateDim),
        referenceManagerPtr_(referenceManager) {}
  ~ZeroStateWeightOcs2Cost() override = default;
  ZeroStateWeightOcs2Cost& operator=(const ZeroStateWeightOcs2Cost&) = delete;
  ZeroStateWeightOcs2Cost(ZeroStateWeightOcs2Cost&&) = delete;
  ZeroStateWeightOcs2Cost& operator=(ZeroStateWeightOcs2Cost&&) = delete;
  ZeroStateWeightOcs2Cost* absl_nonnull clone() const override { return new ZeroStateWeightOcs2Cost(*this); }

 private:
  ZeroStateWeightOcs2Cost(const ZeroStateWeightOcs2Cost& other) = default;
  std::pair<vector_t, vector_t> getStateInputDeviation(scalar_t time,
                                                       const vector_t& state,
                                                       const vector_t& input,
                                                       const TargetTrajectories& targetTrajectories) const override {
    return {vector_t::Zero(static_cast<Eigen::Index>(stateDim_)),
            input - referenceManagerPtr_->getDesiredInput(targetTrajectories, state, time)};
  }

  size_t stateDim_ = 0;
  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
};

/** The shipped DRC Atlas's centroidal layout and a reference manager on it, without a robot model package. */
class InputQuadraticCostTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> taskFile = robot::resolveResourcePath(kTaskFile);
    const absl::StatusOr<std::string> referenceFile = robot::resolveResourcePath(kReferenceFile);
    const absl::StatusOr<std::string> urdfFile = robot::resolveResourcePath(kUrdfFile);
    ASSERT_TRUE(taskFile.ok() && referenceFile.ok() && urdfFile.ok()) << "the shipped DRC Atlas files are not in the runfiles";
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(*taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    absl::StatusOr<ModelSettings> settings = ModelSettings::Create(*task, *urdfFile, "testInputQuadraticCost", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    settings_ = std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(*task, *urdfFile, *settings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));
    model_ = std::make_unique<test::LayoutRobotModel<scalar_t>>(*settings_, StateInputLayout::Mpc::kCentroidal);
    absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule = GaitSchedule::Create(*referenceFile, *settings_);
    ASSERT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
    referenceManager_ = std::make_unique<SwitchedModelReferenceManager>(
        *gaitSchedule, std::make_shared<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(), /*numFeet=*/2), *pinocchioInterface_,
        *model_);

    std::mt19937 generator(31);
    std::uniform_real_distribution<scalar_t> distribution(-1.0, 1.0);
    const Eigen::Index inputDim = static_cast<Eigen::Index>(model_->getInputDim());
    R_ = matrix_t::Zero(inputDim, inputDim);
    for (Eigen::Index i = 0; i < inputDim; ++i) R_(i, i) = 2.0 + distribution(generator);
    R_(0, 1) = 0.25;
    R_(1, 0) = 0.25;
    input_ = vector_t(inputDim);
    for (Eigen::Index i = 0; i < inputDim; ++i) input_(i) = 100.0 * distribution(generator);
    state_ = vector_t::Zero(static_cast<Eigen::Index>(model_->getStateDim()));
  }

  /** The cost of `R` on the fixture's reference manager and model; a refusal fails the test. */
  std::unique_ptr<InputQuadraticCost> makeCost(const matrix_t& R) const {
    absl::StatusOr<std::unique_ptr<InputQuadraticCost>> cost = InputQuadraticCost::Create(R, referenceManager_.get(), *model_);
    EXPECT_TRUE(cost.ok()) << cost.status();
    return cost.ok() ? *std::move(cost) : nullptr;
  }

  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  std::unique_ptr<test::LayoutRobotModel<scalar_t>> model_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
  matrix_t R_;
  vector_t input_;
  vector_t state_;
  const TargetTrajectories targets_;
  const PreComputation preComputation_;
};

TEST_F(InputQuadraticCostTest, ItIsOcs2sQuadraticStateInputCostOfAZeroQBitForBit) {
  const std::unique_ptr<InputQuadraticCost> cost = makeCost(R_);
  ASSERT_NE(cost, nullptr);
  const ZeroStateWeightOcs2Cost before(model_->getStateDim(), R_, referenceManager_.get());
  EXPECT_TRUE(bitwiseEqual(cost->getValue(/*time=*/0.4, state_, input_, targets_, preComputation_),
                           before.getValue(/*time=*/0.4, state_, input_, targets_, preComputation_)));
  const ScalarFunctionQuadraticApproximation expected =
      before.getQuadraticApproximation(/*time=*/0.4, state_, input_, targets_, preComputation_);
  const ScalarFunctionQuadraticApproximation actual =
      cost->getQuadraticApproximation(/*time=*/0.4, state_, input_, targets_, preComputation_);
  EXPECT_TRUE(bitwiseEqual(actual.f, expected.f));
  EXPECT_TRUE(bitwiseEqual(actual.dfdx, expected.dfdx));
  EXPECT_TRUE(bitwiseEqual(actual.dfdu, expected.dfdu));
  EXPECT_TRUE(bitwiseEqual(actual.dfdxx, expected.dfdxx));
  EXPECT_TRUE(bitwiseEqual(actual.dfduu, expected.dfduu));
  EXPECT_TRUE(bitwiseEqual(actual.dfdux, expected.dfdux));
}

TEST_F(InputQuadraticCostTest, ItsStateDerivativesAreZerosOfTheEvaluatedStatesSize) {
  const std::unique_ptr<InputQuadraticCost> cost = makeCost(R_);
  ASSERT_NE(cost, nullptr);
  // The Euler state, and a state one entry longer: the ambient size of a quaternion layout, which a zero Q of the tuning
  // layout's size would not fit.
  for (const Eigen::Index stateDim : {state_.size(), state_.size() + 1}) {
    const vector_t state = vector_t::Zero(stateDim);
    const ScalarFunctionQuadraticApproximation approximation =
        cost->getQuadraticApproximation(/*time=*/0.4, state, input_, targets_, preComputation_);
    EXPECT_EQ(approximation.dfdx.size(), stateDim);
    EXPECT_TRUE(approximation.dfdx.isZero(0.0));
    EXPECT_EQ(approximation.dfdxx.rows(), stateDim);
    EXPECT_EQ(approximation.dfdxx.cols(), stateDim);
    EXPECT_TRUE(approximation.dfdxx.isZero(0.0));
    EXPECT_EQ(approximation.dfdux.rows(), input_.size());
    EXPECT_EQ(approximation.dfdux.cols(), stateDim);
    EXPECT_TRUE(approximation.dfdux.isZero(0.0));
    EXPECT_TRUE(bitwiseEqual(approximation.f, cost->getValue(/*time=*/0.4, state, input_, targets_, preComputation_)));
  }
}

TEST_F(InputQuadraticCostTest, AHotReloadOfTheInputWeightsReplacesROnly) {
  const std::unique_ptr<InputQuadraticCost> cost = makeCost(R_);
  ASSERT_NE(cost, nullptr);
  const matrix_t reloaded = 3.0 * R_;
  ASSERT_TRUE(cost->setInputWeights(reloaded).ok());
  EXPECT_TRUE(bitwiseEqual(cost->getInputWeights(), reloaded));
  // The same cost as one created with the reloaded weights: a reload is a fresh start of the term.
  const std::unique_ptr<InputQuadraticCost> fresh = makeCost(reloaded);
  ASSERT_NE(fresh, nullptr);
  const ScalarFunctionQuadraticApproximation live =
      cost->getQuadraticApproximation(/*time=*/0.4, state_, input_, targets_, preComputation_);
  const ScalarFunctionQuadraticApproximation started =
      fresh->getQuadraticApproximation(/*time=*/0.4, state_, input_, targets_, preComputation_);
  EXPECT_TRUE(bitwiseEqual(live.f, started.f));
  EXPECT_TRUE(bitwiseEqual(live.dfdu, started.dfdu));
  EXPECT_TRUE(bitwiseEqual(live.dfduu, reloaded));

  // A clone carries the reloaded weights, as every worker's copy of the problem must.
  const std::unique_ptr<InputQuadraticCost> clone(cost->clone());
  EXPECT_TRUE(bitwiseEqual(clone->getInputWeights(), reloaded));
}

TEST_F(InputQuadraticCostTest, RefusedInputWeightsKeepTheRunningOnes) {
  const std::unique_ptr<InputQuadraticCost> cost = makeCost(R_);
  ASSERT_NE(cost, nullptr);
  const Eigen::Index inputDim = R_.rows();
  const absl::Status wrongSize = cost->setInputWeights(matrix_t::Identity(inputDim + 1, inputDim + 1));
  EXPECT_EQ(wrongSize.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(wrongSize.message(), "input weights")) << wrongSize;
  matrix_t asymmetric = R_;
  asymmetric(1, 0) = 1.0;
  EXPECT_EQ(cost->setInputWeights(asymmetric).code(), absl::StatusCode::kInvalidArgument);
  matrix_t notFinite = R_;
  notFinite(2, 2) = std::numeric_limits<scalar_t>::quiet_NaN();
  EXPECT_EQ(cost->setInputWeights(notFinite).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(bitwiseEqual(cost->getInputWeights(), R_));

  const absl::StatusOr<std::unique_ptr<InputQuadraticCost>> refused =
      InputQuadraticCost::Create(matrix_t::Identity(inputDim - 1, inputDim - 1), referenceManager_.get(), *model_);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid
