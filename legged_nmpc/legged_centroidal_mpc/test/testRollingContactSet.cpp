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
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/ComputationRequest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/soft_constraint/StateInputSoftConstraint.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_sqp/SqpSolver.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "legged_centroidal_mpc/CentroidalContactSet.h"
#include "legged_centroidal_mpc/CentroidalMpcConfig.h"
#include "legged_centroidal_mpc/CentroidalMpcInterface.h"
#include "legged_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "legged_centroidal_mpc/constraint/RollingContactConstraintCppAd.h"
#include "legged_centroidal_mpc/parameter_update/ZeroVelocityGainsApplier.h"
#include "legged_common_mpc/common/ContactTermNames.h"
#include "legged_common_mpc/common/CostTermNames.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/weights/StateInputLayout.h"
#include "legged_common_mpc/constraint/JointLimitsSoftConstraint.h"
#include "legged_common_mpc/parameter_update/FootConstraintSoftWeightsApplier.h"
#include "legged_common_mpc/parameter_update/HotUpdateTarget.h"
#include "legged_mpc_config/task_file.nproto.h"
#include "legged_nmpc/legged_common_mpc/test/support/StatusTestMacros.h"
#include "support/RollingContactTestSupport.h"
#include "support/TypedConfigFiles.h"

/**
 * The centroidal MPC on Tron1's two rolling contacts, from the shipped
 * configuration:
 *  - the input is a force per wheel and the eight joint velocities;
 *  - zero_velocity builds the rolling no-slip constraint on each wheel, beside
 *    the normal-velocity servo and the friction cone, and no stance-foot
 *    constraint;
 *  - the joint-limit barrier does not depend on a wheel's angle;
 *  - every term a rolling contact cannot run is refused by name, with the
 *    rolling contacts;
 *  - the dynamics take the moment of each wheel's force at its contact point,
 *    a wheel radius below the wheel center.
 */
namespace ocs2::legged {
namespace {

using ::testing::HasSubstr;
using ::testing::IsSupersetOf;

constexpr size_t kNumWheels = 2;
constexpr size_t kNumJoints = 8;
constexpr scalar_t kTolerance = 1.0e-9;
// [rad] Wheel angles the joint-limit barrier must not see.
constexpr scalar_t kWheelTurns = 100.0;
// The reloaded foot gains and the factor of the reloaded soft weight.
constexpr scalar_t kReloadedVelocityGain = 3.0;
constexpr scalar_t kReloadedPositionGain = 7.0;
constexpr scalar_t kReloadedWeightFactor = 2.0;
// [m/s] A sideways slide of the mass-normalized linear momentum.
constexpr scalar_t kSlideSpeed = 0.3;

/** The interface, built once for the suite. */
class RollingContactSetTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ = checkedValue(CentroidalMpcInterface::Create(
                                  rolling_contact_test::tron1Config(),
                                  tron1Files().urdfFile))
                     .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  static CentroidalMpcInterface& mpc() { return *interface_; }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable RollingContactSetTest::interface_ =
    nullptr;

TEST_F(RollingContactSetTest, theInputIsAForcePerWheelAndTheJointVelocities) {
  const ModelSettings& settings = mpc().modelSettings();
  ASSERT_EQ(settings.contactNames,
            (std::vector<std::string>{"wheel_L_contact", "wheel_R_contact"}));
  for (size_t i = 0; i < kNumWheels; ++i) {
    EXPECT_TRUE(settings.isRollingContact(i)) << i;
  }
  EXPECT_EQ(mpc().getMpcRobotModel().getInputDim(),
            kNumWheels * kPositionDim + kNumJoints);
  EXPECT_EQ(mpc().getMpcRobotModel().getJointDim(), kNumJoints);
}

TEST_F(RollingContactSetTest, eachWheelHasTheRollingNoSlipAndItsSwingTerms) {
  const std::vector<std::string> equalities =
      mpc().getEqualityConstraintNames();
  const std::vector<std::string> softConstraints =
      mpc().getSoftConstraintNames();
  for (const std::string& name : mpc().modelSettings().contactNames) {
    SCOPED_TRACE(name);
    EXPECT_THAT(
        equalities,
        IsSupersetOf(
            {contact_term::name(name, contact_term::kRollingNoSlipName),
             contact_term::name(name, contact_term::kNormalVelocityName),
             absl::StrCat(name, "_zeroWrench")}));
    EXPECT_EQ(std::count(equalities.begin(), equalities.end(),
                         zeroVelocityTermName(name)),
              0)
        << "a wheel has no stance-foot constraint on its center";
    EXPECT_THAT(softConstraints, IsSupersetOf({contact_term::name(
                                     name, contact_term::kFrictionForceCone)}));
  }
}

TEST_F(RollingContactSetTest, theJointLimitsDoNotSeeAWheelsAngle) {
  const OptimalControlProblem& problem = mpc().getOptimalControlProblemRef();
  const JointLimitsSoftConstraint& jointLimits =
      problem.stateSoftConstraintPtr->get<JointLimitsSoftConstraint>(
          kJointLimitsTerm);
  const ModelSettings& settings = mpc().modelSettings();
  const vector_t standing =
      mpc().getMpcRobotModel().getJointAngles(mpc().getInitialState());
  vector_t turned = standing;
  for (const std::string& wheelJoint : settings.rollingContactJointNames()) {
    turned(static_cast<Eigen::Index>(settings.jointIndexMap.at(wheelJoint))) =
        kWheelTurns;
  }
  EXPECT_EQ(jointLimits.getValue(turned), jointLimits.getValue(standing));
  const ScalarFunctionQuadraticApproximation approximation =
      jointLimits.getQuadraticApproximation(turned);
  // The approximation is over the whole state; the joints start at this row.
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(mpc().getMpcRobotModel().getJointStartindex());
  for (const std::string& wheelJoint : settings.rollingContactJointNames()) {
    const Eigen::Index row =
        jointStart +
        static_cast<Eigen::Index>(settings.jointIndexMap.at(wheelJoint));
    EXPECT_EQ(approximation.dfdx(row), 0.0) << wheelJoint;
    EXPECT_EQ(approximation.dfdxx(row, row), 0.0) << wheelJoint;
  }
}

/** Returns the soft rolling no-slip term of wheel `contactIndex` of `problem`,
 * whose task file lists zero_velocity as a soft constraint. */
StateInputSoftConstraint& softRollingTermOf(OptimalControlProblem& problem,
                                            const ModelSettings& settings,
                                            size_t contactIndex) {
  return problem.softConstraintPtr->get<StateInputSoftConstraint>(
      contact_term::name(settings.contactNames[contactIndex],
                         contact_term::kRollingNoSlipName));
}

// A reload of model_settings.foot_constraint reaches the rolling no-slip
// constraint, hard (ZeroVelocityGainsApplier) and soft (its gains, and its
// weight through FootConstraintSoftWeightsApplier), in every worker's problem.
TEST_F(RollingContactSetTest, aReloadAppliesTheFootGainsToTheRollingRows) {
  SqpSolver solver(mpc().sqpSettings(), mpc().getOptimalControlProblem(),
                   mpc().getInitializer());
  mpc_config::TaskFile task = mpc().config().task;
  task.model_settings.foot_constraint.linear_velocity_error_gain_xy =
      kReloadedVelocityGain;
  task.model_settings.foot_constraint.position_error_gain_z =
      kReloadedPositionGain;
  const StateInputLayout layout = stateInputLayout(
      mpc().modelSettings(), StateInputLayout::Mpc::kCentroidal);
  HotUpdateTarget target(&task, &solver, /*referenceManager=*/nullptr, &layout,
                         mpc().getEffectiveMpcRobotModel().getInputDim(),
                         /*source=*/"testRollingContactSet");
  ZeroVelocityGainsApplier applier;
  applier.apply(target);
  ASSERT_FALSE(solver.getOcpDefinitions().empty());
  for (OptimalControlProblem& problem : solver.getOcpDefinitions()) {
    for (const std::string& name : mpc().modelSettings().contactNames) {
      SCOPED_TRACE(name);
      const RollingContactConstraintCppAd& rolling =
          problem.equalityConstraintPtr->get<RollingContactConstraintCppAd>(
              contact_term::name(name, contact_term::kRollingNoSlipName));
      EXPECT_EQ(rolling.getVelocityErrorGains()(kXAxisIndex, kXAxisIndex),
                kReloadedVelocityGain);
      EXPECT_EQ(rolling.getVelocityErrorGains()(kYAxisIndex, kYAxisIndex),
                kReloadedVelocityGain);
      EXPECT_EQ(rolling.getPositionErrorGains()(kZAxisIndex, kZAxisIndex),
                kReloadedPositionGain);
    }
  }
}

TEST(RollingContactSet, theSoftNoSlipRowsTakeTheReloadedGainsAndWeight) {
  CentroidalMpcConfig config = rolling_contact_test::tron1Config();
  std::vector<std::string>& hard = config.task.hard_constraints;
  hard.erase(std::remove(hard.begin(), hard.end(), "zero_velocity"),
             hard.end());
  config.task.soft_constraints.emplace_back("zero_velocity");
  std::unique_ptr<CentroidalMpcInterface> mpc = checkedValue(
      CentroidalMpcInterface::Create(config, tron1Files().urdfFile));
  const ModelSettings& settings = mpc->modelSettings();
  for (const std::string& name : settings.contactNames) {
    const std::vector<std::string> softConstraints =
        mpc->getSoftConstraintNames();
    EXPECT_EQ(
        std::count(softConstraints.begin(), softConstraints.end(),
                   contact_term::name(name, contact_term::kRollingNoSlipName)),
        1)
        << name;
  }
  rolling_contact_test::prepareSolve(*mpc,
                                     rolling_contact_test::bothWheelsStand());

  // A sideways slide, which the rows price.
  vector_t state = mpc->getInitialState();
  state(1) = kSlideSpeed;
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc->getMpcRobotModel().getInputDim()));
  OptimalControlProblem& running = mpc->getOptimalControlProblemRef();
  running.preComputationPtr->request(
      Request::SoftConstraint, rolling_contact_test::kStanceTime, state, input);
  const scalar_t pricedBefore =
      softRollingTermOf(running, settings, rolling_contact_test::kLeftWheel)
          .getValue(rolling_contact_test::kStanceTime, state, input,
                    *running.targetTrajectoriesPtr, *running.preComputationPtr);
  ASSERT_GT(pricedBefore, 0.0);

  SqpSolver solver(mpc->sqpSettings(), mpc->getOptimalControlProblem(),
                   mpc->getInitializer());
  mpc_config::TaskFile task = mpc->config().task;
  task.model_settings.foot_constraint.linear_velocity_error_gain_xy =
      kReloadedVelocityGain;
  task.model_settings.foot_constraint.soft_constraint_weight =
      kReloadedWeightFactor *
      settings.footConstraintConfig.softConstraintWeight;
  const StateInputLayout layout =
      stateInputLayout(settings, StateInputLayout::Mpc::kCentroidal);
  HotUpdateTarget target(&task, &solver, /*referenceManager=*/nullptr, &layout,
                         mpc->getEffectiveMpcRobotModel().getInputDim(),
                         /*source=*/"testRollingContactSet");
  ZeroVelocityGainsApplier gainsApplier;
  gainsApplier.apply(target);
  FootConstraintSoftWeightsApplier weightsApplier;
  weightsApplier.apply(target);
  ASSERT_FALSE(solver.getOcpDefinitions().empty());
  for (OptimalControlProblem& problem : solver.getOcpDefinitions()) {
    StateInputSoftConstraint& soft =
        softRollingTermOf(problem, settings, rolling_contact_test::kLeftWheel);
    EXPECT_EQ(soft.get<RollingContactConstraintCppAd>().getVelocityErrorGains()(
                  kYAxisIndex, kYAxisIndex),
              kReloadedVelocityGain);
    problem.preComputationPtr->request(Request::SoftConstraint,
                                       rolling_contact_test::kStanceTime, state,
                                       input);
    // The penalty is quadratic in the rows: the y row grows with its gain, and
    // the price with the weight.
    const scalar_t pricedAfter = soft.getValue(
        rolling_contact_test::kStanceTime, state, input,
        *running.targetTrajectoriesPtr, *problem.preComputationPtr);
    EXPECT_GT(pricedAfter, kReloadedWeightFactor * pricedBefore);
  }
}

// LINT.IfChange(rolling_refused_terms)
/** Lists the contact-implicit formulation in place of the schedule-gated
 * stance constraints, as checkMpcFormulationTasks() accepts it. */
void selectContactImplicitTerms(mpc_config::TaskFile& task) {
  task.hard_constraints = {};
  task.soft_constraints = {
      "joint_limits",        "friction_force_cone", "contact_complementarity",
      "force_weighted_slip", "ground_penetration",  "normal_velocity"};
}

TEST(RollingContactSet, everyTermARollingContactCannotRunIsRefused) {
  const mpc_config::TaskFile shipped = rolling_contact_test::tron1Config().task;
  const ModelSettings settings = checkedValue(
      ModelSettings::Create(shipped, tron1Files().urdfFile,
                            "testRollingContactSet_", /*verbose=*/false));
  LEGGED_ASSERT_OK(checkCentroidalContactSet(settings, shipped));

  const std::vector<
      std::pair<std::string, void (*absl_nonnull)(mpc_config::TaskFile&)>>
      selections = {
          {"soft_constraints 'contact_complementarity', soft_constraints "
           "'force_weighted_slip', soft_constraints 'ground_penetration'",
           &selectContactImplicitTerms},
          {"soft_constraints 'foot_collision'",
           [](mpc_config::TaskFile& task) {
             task.soft_constraints.emplace_back("foot_collision");
           }},
          {"soft_constraints 'foot_collision_cbf'",
           [](mpc_config::TaskFile& task) {
             task.soft_constraints.emplace_back("foot_collision_cbf");
           }},
          {"soft_constraints 'contact_swing'",
           [](mpc_config::TaskFile& task) {
             task.soft_constraints.emplace_back("contact_swing");
           }},
          {"soft_constraints 'terrain_non_penetration'",
           [](mpc_config::TaskFile& task) {
             task.soft_constraints.emplace_back("terrain_non_penetration");
           }},
          {"soft_constraints 'contact_moment_xy'",
           [](mpc_config::TaskFile& task) {
             task.soft_constraints.emplace_back("contact_moment_xy");
           }},
          {"contact_input_parameterization 'basis_vectors'",
           [](mpc_config::TaskFile& task) {
             task.contact_input_parameterization = "basis_vectors";
           }},
          {"contact_schedule_source 'contact_planner'",
           [](mpc_config::TaskFile& task) {
             task.contact_schedule_source = "contact_planner";
           }},
          {"contact_schedule_source 'multi_contact_plan'",
           [](mpc_config::TaskFile& task) {
             task.contact_schedule_source = "multi_contact_plan";
           }},
          {"leg_joint_reference 'foot_motion'",
           [](mpc_config::TaskFile& task) {
             task.leg_joint_reference = "foot_motion";
           }},
      };
  for (const std::pair<std::string,
                       void (*absl_nonnull)(mpc_config::TaskFile&)>& selection :
       selections) {
    SCOPED_TRACE(selection.first);
    mpc_config::TaskFile task = shipped;
    selection.second(task);
    const absl::Status refused = checkCentroidalContactSet(settings, task);
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
    EXPECT_THAT(refused.message(),
                HasSubstr("the rolling contacts wheel_L_contact, "
                          "wheel_R_contact (model_settings.contacts_rolling)"));
    EXPECT_THAT(refused.message(), HasSubstr(selection.first));
    EXPECT_THAT(refused.message(),
                HasSubstr("legged_nmpc/docs/rolling_contact/README.md"));
  }
}
// LINT.ThenChange(//legged_nmpc/legged_centroidal_mpc/src/CentroidalContactSet.cpp:rolling_refused_terms)

TEST_F(RollingContactSetTest, theDynamicsTakeEachWheelsMomentAtItsContact) {
  const CentroidalMpcRobotModel<scalar_t>& model = mpc().getMpcRobotModel();
  const ModelSettings& settings = mpc().modelSettings();
  const vector_t& state = mpc().getInitialState();
  vector_t input =
      vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  // Traction forward on the left wheel and backward on the right, a lateral
  // push on both, and about half the weight on each.
  const std::vector<vector3_t> forceWheelsEwrtWorld = {
      vector3_t(20.0, 5.0, 110.0), vector3_t(-10.0, 3.0, 100.0)};
  for (size_t i = 0; i < kNumWheels; ++i) {
    input.segment<kPositionDim>(static_cast<Eigen::Index>(
        model.getContactForceStartIndices(i))) = forceWheelsEwrtWorld[i];
  }

  // The contact points and the center of mass, from Pinocchio alone.
  const pinocchio::Model& pinocchioModel =
      mpc().getPinocchioInterface().getModel();
  pinocchio::Data data(pinocchioModel);
  const vector_t q = model.getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(pinocchioModel, data, q);
  pinocchio::updateFramePlacements(pinocchioModel, data);
  const vector3_t posComRtWorldEwrtWorld =
      pinocchio::centerOfMass(pinocchioModel, data, q);
  const scalar_t mass = mpc().getCentroidalModelInfo().robotMass;
  vector3_t torqueComEwrtWorld = vector3_t::Zero();
  vector3_t leverArmShift = vector3_t::Zero();
  for (size_t i = 0; i < kNumWheels; ++i) {
    const vector3_t posWheelRtWorldEwrtWorld =
        data.oMf[pinocchioModel.getFrameId(settings.contactNames[i])]
            .translation();
    const vector3_t posContactRtWorldEwrtWorld =
        posWheelRtWorldEwrtWorld - settings.wheelRadius(i) * vector3_t::UnitZ();
    torqueComEwrtWorld += (posContactRtWorldEwrtWorld - posComRtWorldEwrtWorld)
                              .cross(forceWheelsEwrtWorld[i]);
    leverArmShift += (settings.wheelRadius(i) * vector3_t::UnitZ())
                         .cross(forceWheelsEwrtWorld[i]);
  }

  const PreComputation preComputation;
  const vector_t flowMap =
      mpc().getOptimalControlProblem().dynamicsPtr->computeFlowMap(
          /*time=*/0.0, state, input, preComputation);
  const vector3_t mnormRateAngmomComRtWorldEwrtWorld =
      flowMap.segment<kRotationTangentDim>(kPositionDim);
  EXPECT_TRUE(mnormRateAngmomComRtWorldEwrtWorld.isApprox(
      torqueComEwrtWorld / mass, kTolerance))
      << mnormRateAngmomComRtWorldEwrtWorld.transpose() << " against "
      << (torqueComEwrtWorld / mass).transpose();
  // The wheel center as the lever arm is off by r e_z x f per wheel.
  EXPECT_GT((leverArmShift / mass).norm(), 0.05);
  // The linear rows are the forceWheelsEwrtWorld and gravity.
  const vector3_t linaccComRtWorldEwrtWorld =
      (forceWheelsEwrtWorld[0] + forceWheelsEwrtWorld[1]) / mass -
      kGravityMagnitude * vector3_t::UnitZ();
  EXPECT_TRUE(flowMap.head<kPositionDim>().isApprox(linaccComRtWorldEwrtWorld,
                                                    kTolerance));
}

}  // namespace
}  // namespace ocs2::legged
