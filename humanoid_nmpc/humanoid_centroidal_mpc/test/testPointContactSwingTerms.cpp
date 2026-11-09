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

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/model_data/ModelData.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/approximate_model/LinearQuadraticApproximator.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_centroidal_mpc/CentroidalContactSet.h"
#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/parameter_update/CentroidalCostWeightsApplier.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/FourPointContactSet.h"
#include "support/TypedConfigFiles.h"

/**
 * The swing terms of the centroidal MPC on the four point contacts of
 * FourPointContactSet.h, described by their legs LF, RF, LH and RH: the
 * swing-foot cost and the hard normal-velocity servo run on every contact.
 *  - checkCentroidalContactSet() accepts them on the legged set.
 *  - A point contact's foot cost carries no orientation or angular-velocity
 *    weight, after construction and after a hot reload of
 *    task_space_foot_cost.
 *  - On a trot, each contact's swing height rises in its own swing and stays
 *    on the ground in stance, its normal-velocity servo is active in its swing
 *    alone, and the LQ approximation is finite.
 */
namespace ocs2::humanoid {
namespace {

using four_point_set::kAllFour;
using four_point_set::kFrontLeftHindRight;
using four_point_set::kFrontRightHindLeft;

constexpr size_t kNumPoints = four_point_set::kNumContacts;
constexpr scalar_t kWeightTolerance = 1.0e-9;
// [m] A swing height above the ground.
constexpr scalar_t kMinSwingClearance = 1.0e-3;

// The trot: all four, one diagonal, the other, all four.
constexpr scalar_t kFirstLiftOffTime = 0.2;
constexpr scalar_t kDiagonalSwitchTime = 0.5;
constexpr scalar_t kLastTouchDownTime = 0.8;
constexpr scalar_t kFirstDiagonalTime = 0.35;
constexpr scalar_t kSecondDiagonalTime = 0.65;

// The rotation vector and the angular velocity in the foot cost's weights
// (EndEffectorKinematicsWeights::toVector()).
constexpr int kRotvecStart = kPositionDim;
constexpr int kAngvelStart = kPoseTangentDim + kPositionDim;

/** The legged four-point configuration with the swing-foot cost and the hard
 * normal-velocity servo. */
CentroidalMpcConfig swingTermsConfig() {
  CentroidalMpcConfig config = four_point_set::leggedFourPointConfig(
      checkedValue(loadConfigOf(atlasFiles())));
  config.task.costs.emplace_back("task_space_foot_cost");
  config.task.hard_constraints.emplace_back("normal_velocity");
  return config;
}

/** The interface, built once for the whole suite: it generates the CppAD
 * libraries of the dynamics, the contact frames and the four foot costs. */
class PointContactSwingTermsTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ = checkedValue(CentroidalMpcInterface::Create(
                                  swingTermsConfig(), atlasFiles().urdfFile))
                     .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  static CentroidalMpcInterface& interface() { return *interface_; }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable PointContactSwingTermsTest::interface_ =
    nullptr;

/** The trot, every swing ending in a contact. */
ModeSchedule trot() {
  return ModeSchedule(
      {kFirstLiftOffTime, kDiagonalSwitchTime, kLastTouchDownTime},
      {kAllFour, kFrontLeftHindRight, kFrontRightHindLeft, kAllFour});
}

/** Makes `schedule` the reference manager's, with a standing target, the way
 * the MPC loop prepares a solve, and points the problem at that target. */
void prepareSolve(CentroidalMpcInterface& mpc, const ModeSchedule& schedule) {
  SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  const vector_t& state = mpc.getInitialState();
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc.getMpcRobotModel().getInputDim()));
  referenceManager.setTargetTrajectories(
      TargetTrajectories({0.0, 2.0}, {state, state}, {input, input}));
  referenceManager.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state,
                                schedule.modeAtTime(0.0));
  mpc.getOptimalControlProblemRef().targetTrajectoriesPtr =
      &referenceManager.getTargetTrajectories();
}

/** The foot cost of every contact in `problem`, in contact order. */
std::vector<CentroidalMpcEndEffectorFootCost* absl_nonnull> footCostsOf(
    OptimalControlProblem& problem, const ModelSettings& settings) {
  std::vector<CentroidalMpcEndEffectorFootCost* absl_nonnull> costs;
  for (const std::string& name : settings.contactNames) {
    costs.push_back(&problem.costPtr->get<CentroidalMpcEndEffectorFootCost>(
        taskSpaceKinematicsCostName(name)));
  }
  return costs;
}

/** Expects `cost`'s weights to be `expected` with the orientation and angular
 * velocity entries zero. */
void expectPointContactWeights(const CentroidalMpcEndEffectorFootCost& cost,
                               const vector12_t& expected) {
  vector12_t weights;
  cost.getWeights(weights);
  EXPECT_TRUE(weights.segment<kRotationTangentDim>(kRotvecStart).isZero(0.0))
      << weights.transpose();
  EXPECT_TRUE(weights.segment<kRotationTangentDim>(kAngvelStart).isZero(0.0))
      << weights.transpose();
  EXPECT_TRUE(weights.head<kPositionDim>().isApprox(
      expected.head<kPositionDim>(), kWeightTolerance))
      << weights.transpose();
  EXPECT_TRUE(weights.segment<kPositionDim>(kPoseTangentDim)
                  .isApprox(expected.segment<kPositionDim>(kPoseTangentDim),
                            kWeightTolerance))
      << weights.transpose();
}

/** task_space_foot_cost.weights of `task` as the cost's 12 weights. */
vector12_t footWeightsOf(const mpc_config::TaskFile& task) {
  const TaskSpaceFootCostSettings settings =
      checkedValue(taskSpaceFootCostFromConfig(task.task_space_foot_cost));
  return settings.weights.toVector();
}

TEST(PointContactSwingTerms, theLeggedSetRunsTheSwingTerms) {
  const CentroidalMpcConfig config = swingTermsConfig();
  const ModelSettings settings = checkedValue(ModelSettings::Create(
      config.task, atlasFiles().urdfFile, "point_contact_swing_terms_",
      /*verbose=*/false));
  ASSERT_TRUE(settings.isLeggedContactSet());
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, config.task));
}

TEST_F(PointContactSwingTermsTest, aPointFootCostHasNoOrientationWeights) {
  CentroidalMpcInterface& mpc = interface();
  const vector12_t configured = footWeightsOf(mpc.config().task);
  ASSERT_FALSE(configured.segment<kRotationTangentDim>(kRotvecStart).isZero())
      << "the shipped Atlas weighs the foot's orientation";
  for (const CentroidalMpcEndEffectorFootCost* absl_nonnull cost :
       footCostsOf(mpc.getOptimalControlProblemRef(), mpc.modelSettings())) {
    expectPointContactWeights(*cost, configured);
  }
}

// A hot reload writes every weight of task_space_foot_cost into each worker's
// foot costs, and a point contact's orientation weights stay zero.
TEST_F(PointContactSwingTermsTest, aReloadKeepsThePointOrientationWeightsZero) {
  CentroidalMpcInterface& mpc = interface();
  SqpSolver solver(mpc.sqpSettings(), mpc.getOptimalControlProblem(),
                   mpc.getInitializer());
  mpc_config::TaskFile task = mpc.config().task;
  // NOLINTBEGIN(frame-naming): a per-component weight field
  task.task_space_foot_cost.weights.pos_z =
      2.0 * task.task_space_foot_cost.weights.pos_z;
  // NOLINTEND(frame-naming)
  task.task_space_foot_cost.weights.rotvec_x = 300.0;
  task.task_space_foot_cost.weights.angvel_z = 2.0;
  const StateInputLayout layout =
      stateInputLayout(mpc.modelSettings(), StateInputLayout::Mpc::kCentroidal);
  HotUpdateTarget target(&task, &solver, /*referenceManager=*/nullptr, &layout,
                         mpc.getEffectiveMpcRobotModel().getInputDim(),
                         /*source=*/"testPointContactSwingTerms");
  CentroidalCostWeightsApplier applier;
  applier.apply(target);
  const vector12_t reloaded = footWeightsOf(task);
  ASSERT_FALSE(solver.getOcpDefinitions().empty());
  for (OptimalControlProblem& problem : solver.getOcpDefinitions()) {
    for (const CentroidalMpcEndEffectorFootCost* absl_nonnull cost :
         footCostsOf(problem, mpc.modelSettings())) {
      expectPointContactWeights(*cost, reloaded);
    }
  }
}

// Each contact swings on its own schedule: its height reference rises above
// the ground in its own swing only, and its normal-velocity servo is active
// then and only then.
TEST_F(PointContactSwingTermsTest, eachContactSwingsInItsOwnSwing) {
  CentroidalMpcInterface& mpc = interface();
  prepareSolve(mpc, trot());
  const SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  const SwingTrajectoryPlanner& swingPlanner =
      *referenceManager.getSwingTrajectoryPlanner();
  ASSERT_EQ(swingPlanner.numFeet(), kNumPoints);
  const scalar_t posSurfaceRtWorldEwrtWorldZ =
      referenceManager.getAppliedPosSurfaceRtWorldEwrtWorldZ();
  const OptimalControlProblem& problem = mpc.getOptimalControlProblemRef();
  for (const scalar_t time : {kFirstDiagonalTime, kSecondDiagonalTime}) {
    const ContactMask stance = referenceManager.getContactFlags(time);
    ASSERT_EQ(stance.numActive(), 2U);
    for (size_t i = 0; i < kNumPoints; ++i) {
      const std::string& name = mpc.modelSettings().contactNames[i];
      SCOPED_TRACE(absl::StrCat(name, " at ", time));
      const scalar_t desPosFootRtWorldEwrtWorldZ =
          swingPlanner.getDesPosFootRtWorldEwrtWorldZ(i, time);
      const NormalVelocityConstraintCppAd& swingServo =
          problem.equalityConstraintPtr->get<NormalVelocityConstraintCppAd>(
              contact_term::name(name, contact_term::kNormalVelocityName));
      if (stance[i]) {
        EXPECT_NEAR(desPosFootRtWorldEwrtWorldZ, posSurfaceRtWorldEwrtWorldZ,
                    kWeightTolerance);
        EXPECT_FALSE(swingServo.isActive(time));
      } else {
        EXPECT_GT(desPosFootRtWorldEwrtWorldZ,
                  posSurfaceRtWorldEwrtWorldZ + kMinSwingClearance);
        EXPECT_TRUE(swingServo.isActive(time));
      }
    }
  }
}

TEST_F(PointContactSwingTermsTest, theLqApproximationOfATrotIsFinite) {
  CentroidalMpcInterface& mpc = interface();
  prepareSolve(mpc, trot());
  OptimalControlProblem& problem = mpc.getOptimalControlProblemRef();
  const SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  const vector_t input = referenceManager.getDesiredInput(
      referenceManager.getTargetTrajectories(), mpc.getInitialState(),
      kFirstDiagonalTime);
  const ModelData lq = approximateIntermediateLQ(problem, kFirstDiagonalTime,
                                                 mpc.getInitialState(), input);
  EXPECT_TRUE(std::isfinite(lq.cost.f));
  EXPECT_TRUE(lq.cost.dfdx.allFinite());
  EXPECT_TRUE(lq.cost.dfdu.allFinite());
  EXPECT_TRUE(lq.cost.dfdxx.allFinite());
  EXPECT_TRUE(lq.cost.dfduu.allFinite());
  EXPECT_TRUE(lq.stateInputEqConstraint.f.allFinite());
  EXPECT_TRUE(lq.stateInputEqConstraint.dfdx.allFinite());
  EXPECT_TRUE(lq.stateInputEqConstraint.dfdu.allFinite());
}

}  // namespace
}  // namespace ocs2::humanoid
