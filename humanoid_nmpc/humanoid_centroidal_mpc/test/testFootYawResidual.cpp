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
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/frames.hpp"

#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/DrcAtlasContactTestModel.h"

/*
 * The yaw residual of the swing-foot tracking cost. It was atan2(sin(yaw -
 * ref), cos(yaw - ref)) with yaw = atan2(R10, R00), masked by hasYawReference.
 * CppAD evaluates both branches of an atan2, one of which divides by its first
 * argument, so at a foot yaw of exactly zero its derivative was 0 * inf = NaN,
 * and 0 * NaN is NaN, mask or no mask. Exactly zero is the yaw of every foot of
 * a robot put back in its initial pose by the simulator, and every QP of a
 * swing of that foot then failed.
 */

namespace ocs2::humanoid {
namespace {

using Cost = CentroidalMpcEndEffectorFootCost;

/** The residual as it was, as the cost taped it. */
ad_scalar_t legacyFootYawError(const ad_matrix3_t& orientation,
                               const ad_scalar_t& yawReference,
                               const ad_scalar_t& hasYawReference) {
  const ad_scalar_t footYaw =
      CppAD::atan2(orientation(1, 0), orientation(0, 0));
  return hasYawReference * CppAD::atan2(CppAD::sin(footYaw - yawReference),
                                        CppAD::cos(footYaw - yawReference));
}

/** The residual as it was, evaluated. */
scalar_t legacyFootYawError(const matrix3_t& orientation,
                            scalar_t yawReference,
                            scalar_t hasYawReference) {
  const scalar_t footYaw = std::atan2(orientation(1, 0), orientation(0, 0));
  return hasYawReference * std::atan2(std::sin(footYaw - yawReference),
                                      std::cos(footYaw - yawReference));
}

using AdResidual = std::function<ad_scalar_t(
    const ad_matrix3_t&, const ad_scalar_t&, const ad_scalar_t&)>;

/**
 * `residual` of a foot whose orientation has the ZYX Euler angles x, with the
 * parameters p = (yawReference, hasYawReference), as a generated CppAD library:
 * the code path of the cost, where the derivatives of both branches of a
 * conditional expression are evaluated and 0 * NaN is NaN.
 */
std::unique_ptr<CppAdInterface> generate(const AdResidual& residual,
                                         const std::string& name) {
  const CppAdInterface::ad_parameterized_function_t function =
      [residual](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
        const ad_matrix3_t rotation =
            getRotationMatrixFromZyxEulerAngles<ad_scalar_t>(
                Eigen::Matrix<ad_scalar_t, 3, 1>(x(0), x(1), x(2)));
        y.resize(1);
        y(0) = residual(rotation, p(0), p(1));
      };
  std::unique_ptr<CppAdInterface> library = std::make_unique<CppAdInterface>(
      function, /*variableDim=*/3, /*parameterDim=*/2,
      "testFootYawResidual_" + name, testing::TempDir() + "/cppad_foot_yaw");
  library->createModels(CppAdInterface::ApproximationOrder::First,
                        /*verbose=*/false);
  return library;
}

vector_t jacobianAt(const CppAdInterface& library,
                    const vector3_t& eulerZyx,
                    scalar_t yawReference,
                    scalar_t hasYawReference) {
  const matrix_t jacobian = library.getJacobian(
      eulerZyx, (vector_t(2) << yawReference, hasYawReference).finished());
  return jacobian.row(0).transpose();
}

TEST(FootYawResidual, IsTheWrappedYawErrorAtEveryNonSingularOrientation) {
  std::mt19937 generator(7);
  std::uniform_real_distribution<scalar_t> yawDistribution(-M_PI, M_PI);
  std::uniform_real_distribution<scalar_t> tiltDistribution(-0.4, 0.4);
  for (int sample = 0; sample < 2000; ++sample) {
    const vector3_t euler(yawDistribution(generator),
                          tiltDistribution(generator),
                          tiltDistribution(generator));
    const matrix3_t rotation =
        getRotationMatrixFromZyxEulerAngles<scalar_t>(euler);
    const scalar_t reference = yawDistribution(generator);
    const scalar_t legacy =
        legacyFootYawError(rotation, reference, /*hasYawReference=*/1.0);
    if (std::abs(std::abs(legacy) - M_PI) < 1.0e-3)
      continue;  // the wrap itself, where neither is continuous
    EXPECT_NEAR(Cost::footYawError<scalar_t>(rotation, reference,
                                             /*hasYawReference=*/1.0),
                legacy, 1.0e-9)
        << "sample " << sample;
    // Masked, it is exactly zero, whatever the orientation and the reference.
    EXPECT_EQ(Cost::footYawError<scalar_t>(rotation, reference,
                                           /*hasYawReference=*/0.0),
              0.0)
        << "sample " << sample;
  }
}

TEST(FootYawResidual, IsTheFootYawMinusItsReference) {
  // The sign: s = r x h = sin(yaw - ref), so the residual is yaw - ref wrapped
  // to (-pi, pi), positive when the foot has turned past its reference. The
  // orientation is built about the world z axis directly, not through the Euler
  // helpers.
  std::mt19937 generator(11);
  std::uniform_real_distribution<scalar_t> yawDistribution(-M_PI, M_PI);
  for (int sample = 0; sample < 2000; ++sample) {
    const scalar_t yaw = yawDistribution(generator);
    const scalar_t reference = yawDistribution(generator);
    const scalar_t wrapped = std::remainder(yaw - reference, 2.0 * M_PI);
    if (std::abs(std::abs(wrapped) - M_PI) < 1.0e-6)
      continue;  // the wrap itself
    const matrix3_t rotation =
        Eigen::AngleAxis<scalar_t>(yaw, vector3_t::UnitZ()).toRotationMatrix();
    EXPECT_NEAR(Cost::footYawError<scalar_t>(rotation, reference,
                                             /*hasYawReference=*/1.0),
                wrapped, 1.0e-9)
        << "sample " << sample;
  }
  const matrix3_t turnedLeft =
      Eigen::AngleAxis<scalar_t>(0.1, vector3_t::UnitZ()).toRotationMatrix();
  EXPECT_GT(Cost::footYawError<scalar_t>(turnedLeft, /*yawReference=*/0.0,
                                         /*hasYawReference=*/1.0),
            0.0);
}

TEST(FootYawResidual, HasFiniteDerivativesAtAYawOfExactlyZero) {
  const std::unique_ptr<CppAdInterface> library = generate(
      [](const ad_matrix3_t& rotation, const ad_scalar_t& reference,
         const ad_scalar_t& has) {
        return Cost::footYawError<ad_scalar_t>(rotation, reference, has);
      },
      "halfAngle");
  for (const scalar_t hasYawReference : {0.0, 1.0}) {
    for (const vector3_t& euler :
         {vector3_t(0.0, 0.0, 0.0), vector3_t(0.0, 0.1, -0.05),
          vector3_t(1.0e-9, 0.0, 0.0)}) {
      const vector_t jacobian =
          jacobianAt(*library, euler, /*yawReference=*/0.0, hasYawReference);
      EXPECT_TRUE(jacobian.allFinite())
          << "hasYawReference " << hasYawReference << ", euler "
          << euler.transpose() << ": " << jacobian.transpose();
    }
  }
}

TEST(FootYawResidual, TheLegacyResidualHadNoFiniteDerivativeThere) {
  // Positive control: generated the same way, the residual this replaced is not
  // differentiable at a yaw of exactly zero, masked or not, and is at a yaw of
  // 1e-9.
  const std::unique_ptr<CppAdInterface> library = generate(
      [](const ad_matrix3_t& rotation, const ad_scalar_t& reference,
         const ad_scalar_t& has) {
        return legacyFootYawError(rotation, reference, has);
      },
      "legacy");
  EXPECT_FALSE(jacobianAt(*library, vector3_t::Zero(), /*yawReference=*/0.0,
                          /*hasYawReference=*/0.0)
                   .allFinite());
  EXPECT_FALSE(jacobianAt(*library, vector3_t::Zero(), /*yawReference=*/0.0,
                          /*hasYawReference=*/1.0)
                   .allFinite());
  EXPECT_TRUE(jacobianAt(*library, vector3_t(1.0e-9, 0.0, 0.0),
                         /*yawReference=*/0.0, /*hasYawReference=*/1.0)
                  .allFinite());
}

TEST(FootYawResidual,
     TheSwingFootCostIsDifferentiableAtTheSimulatorsResetPose) {
  // The shipped Atlas initial state: base yaw 0 and the hip yaw joints at 0, so
  // both feet have a yaw of exactly zero - the pose the simulator puts the
  // robot back in. With a swing of the left foot scheduled the cost is active,
  // without a yaw reference (no contact planner).
  DrcAtlasContactTestModel atlas("testFootYawResidual_");
  const vector_t& state = atlas.nominalState();
  const vector_t input = atlas.makeInput(atlas.wrenchModel(), /*loadedFoot=*/1,
                                         /*normalForce=*/800.0);
  const TargetTrajectories target({DrcAtlasContactTestModel::kQueryTime},
                                  {state}, {input});
  atlas.referenceManager().setTargetTrajectories(target);
  atlas.setSwing(0);
  Cost cost(atlas.referenceManager(), TaskSpaceFootCostSettings(),
            atlas.pinocchioInterface(), atlas.adWrenchModel(),
            /*contactIndex=*/0, "testFootYawResidual_foot_l",
            atlas.modelSettings());
  const scalar_t time = DrcAtlasContactTestModel::kQueryTime;
  ASSERT_TRUE(cost.isActive(time))
      << "the swing foot cost is not active, so nothing was checked";
  ASSERT_EQ(atlas.wrenchModel().getBaseOrientationEulerZyx(state)(0), 0.0)
      << "the case under test: a base yaw of exactly zero";
  // The foot's own yaw is what the residual sees: R10 exactly zero is where the
  // legacy residual has no derivative
  // (TheLegacyResidualHadNoFiniteDerivativeThere), so this pose is one the old
  // cost failed at.
  PinocchioInterface pinocchioInterface = atlas.pinocchioInterface();
  const pinocchio::FrameIndex footFrame =
      pinocchioInterface.getModel().getFrameId(
          atlas.modelSettings().contactNames[0]);
  pinocchio::framesForwardKinematics(
      pinocchioInterface.getModel(), pinocchioInterface.getData(),
      atlas.wrenchModel().getGeneralizedCoordinates(state));
  const matrix3_t footRotation =
      pinocchioInterface.getData().oMf[footFrame].rotation();
  ASSERT_EQ(footRotation(1, 0), 0.0)
      << "the case under test: a foot yaw of exactly zero";
  ASSERT_GT(footRotation(0, 0), 0.0);
  const ScalarFunctionQuadraticApproximation approximation =
      cost.getQuadraticApproximation(time, state, input, target,
                                     PreComputation());
  EXPECT_TRUE(std::isfinite(approximation.f));
  EXPECT_TRUE(approximation.dfdx.allFinite()) << approximation.dfdx.transpose();
  EXPECT_TRUE(approximation.dfdu.allFinite());
  EXPECT_TRUE(approximation.dfdxx.allFinite());
}

TEST(FootCostParameters, AreTheReferenceManagersWhateverTargetTheSolverPasses) {
  // The cost's parameters - the swing reference, the plane normal, the impact
  // proximity - come from the reference manager; the target the solver passes
  // is not read. It used to be interpolated into two values nobody used, which
  // also made an empty target throw here. Built as above, so the library that
  // test generated is loaded, not rebuilt.
  DrcAtlasContactTestModel atlas("testFootYawResidual_");
  const vector_t& state = atlas.nominalState();
  const vector_t input = atlas.makeInput(atlas.wrenchModel(), /*loadedFoot=*/1,
                                         /*normalForce=*/800.0);
  const TargetTrajectories target({DrcAtlasContactTestModel::kQueryTime},
                                  {state}, {input});
  atlas.referenceManager().setTargetTrajectories(target);
  atlas.setSwing(0);
  const Cost cost(atlas.referenceManager(), TaskSpaceFootCostSettings(),
                  atlas.pinocchioInterface(), atlas.adWrenchModel(),
                  /*contactIndex=*/0, "testFootYawResidual_foot_l",
                  atlas.modelSettings());
  const scalar_t time = DrcAtlasContactTestModel::kQueryTime;
  const vector_t parameters =
      cost.getParameters(time, target, PreComputation());
  ASSERT_TRUE(parameters.allFinite()) << parameters.transpose();

  // A target somewhere else entirely, with other inputs, and a target with no
  // knots at all.
  vector_t movedState = state;
  movedState.head(6).setConstant(0.3);
  const TargetTrajectories moved({time - 1.0, time + 1.0},
                                 {movedState, 2.0 * movedState},
                                 {2.0 * input, -input});
  EXPECT_TRUE(cost.getParameters(time, moved, PreComputation()) == parameters);
  vector_t emptyTargetParameters;
  ASSERT_NO_THROW(emptyTargetParameters = cost.getParameters(
                      time, TargetTrajectories(), PreComputation()));
  EXPECT_TRUE(emptyTargetParameters == parameters);
}

// The layout of the cost's parameters
// (CentroidalMpcEndEffectorFootCost::getParameters()): the 12-vector reference,
// then the 12 sqrt weights of the residual, whose linear velocity errors follow
// the pose's, then the impact proximity.
using PlanarReference = PlanarEndEffectorKinematicsPlanarReference<scalar_t>;
constexpr int kReferenceZIndex = 2;
constexpr int kReferenceZVelocityIndex =
    PlanarReference::kLinearVelocityStart + 2;
constexpr int kSqrtWeightsStart = PlanarReference::kDim;
constexpr int kLinearVelocitySqrtWeightStart = kSqrtWeightsStart + 6;
constexpr int kImpactProximityIndex = kSqrtWeightsStart + 12;
/** [s] The half-width of the swing the tests schedule around 0 (the stance
 * brackets stand outside it). */
constexpr scalar_t kSwingHalfWidth = 10.0;

/**
 * The reference manager of the Atlas fixture with no landing target for any
 * foot (no contact planner, no nominal foothold): the case in which the foot
 * cost falls back to the swing planner's height alone and to the commanded
 * velocity.
 */
class NoLandingTargetReferenceManager final
    : public SwitchedModelReferenceManager {
 public:
  explicit NoLandingTargetReferenceManager(
      const DrcAtlasContactTestModel& atlas)
      : SwitchedModelReferenceManager(
            checkedValue(GaitSchedule::Create(atlas.config().reference,
                                              atlas.modelSettings(),
                                              /*verbose=*/false)),
            std::make_unique<SwingTrajectoryPlanner>(
                checkedValue(swingTrajectorySettingsFromConfig(
                    atlas.config().task.swing_trajectory_config)),
                kNumContacts),
            atlas.pinocchioInterface(),
            atlas.wrenchModel()) {}

  std::optional<SwingFootReference> getSwingFootReference(
      size_t /*contactIndex*/, scalar_t /*time*/) const override {
    return std::nullopt;
  }
};

/**
 * Schedules `contacts` over (-kSwingHalfWidth, kSwingHalfWidth), between stance
 * phases, on `referenceManager` and runs its preSolverRun() from `state`, which
 * builds the swing trajectories on its terrain height (as
 * DrcAtlasContactTestModel::setContactFlags() does).
 */
void schedule(SwitchedModelReferenceManager& referenceManager,
              const contact_flag_t& contacts,
              const vector_t& state) {
  const ModeSchedule modeSchedule(
      {-kSwingHalfWidth, kSwingHalfWidth},
      {ModeNumber::kStance, stanceLeg2ModeNumber(contacts),
       ModeNumber::kStance});
  referenceManager.getGaitSchedule()->updateModeSchedule(modeSchedule);
  referenceManager.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state,
                                stanceLeg2ModeNumber(contacts));
}

/** The Atlas foot cost's settings as shipped, with
 * `swingVelocityReferenceWeight` and in every phase. */
TaskSpaceFootCostSettings atlasFootCost(const DrcAtlasContactTestModel& atlas,
                                        scalar_t swingVelocityReferenceWeight) {
  TaskSpaceFootCostSettings settings = checkedValue(
      taskSpaceFootCostFromConfig(atlas.config().task.task_space_foot_cost));
  settings.swingVelocityReferenceWeight = swingVelocityReferenceWeight;
  settings.activeInStance = true;
  return settings;
}

/**
 * The foot cost of `swingFoot` on `referenceManager` that weighs only the swing
 * velocity reference, with `weight`: every configured weight is 0, so that the
 * cost is the two rows under test alone.
 */
Cost velocityOnlyFootCost(const DrcAtlasContactTestModel& atlas,
                          const SwitchedModelReferenceManager& referenceManager,
                          size_t swingFoot,
                          scalar_t weight) {
  TaskSpaceFootCostSettings settings = atlasFootCost(atlas, weight);
  settings.weights.contactPositionErrorWeight.setZero();
  settings.weights.contactOrientationErrorWeight.setZero();
  settings.weights.contactLinearVelocityErrorWeight.setZero();
  settings.weights.contactAngularVelocityErrorWeight.setZero();
  return Cost(referenceManager, settings, atlas.pinocchioInterface(),
              atlas.adWrenchModel(), swingFoot, "testFootYawResidual_foot_l",
              atlas.modelSettings());
}

TEST(FootCostParameters, TheHeightReferenceIsTheSwingPlannersOnAnyGround) {
  // Audit C3: without a landing target the height reference was the world's z =
  // 0, whatever the ground and the swing height. It is the swing planner's, in
  // swing and in stance, under both reference managers: translating the ground
  // translates the reference, and in stance it is the ground itself.
  DrcAtlasContactTestModel atlas("testFootYawResidual_");
  NoLandingTargetReferenceManager noTarget(atlas);
  const vector_t& state = atlas.nominalState();
  const vector_t input = atlas.makeInput(atlas.wrenchModel(), /*loadedFoot=*/1,
                                         /*normalForce=*/800.0);
  const TargetTrajectories target({DrcAtlasContactTestModel::kQueryTime},
                                  {state}, {input});
  constexpr size_t kSwingFoot = 0;
  const std::vector<SwitchedModelReferenceManager* absl_nonnull>
      referenceManagers = {&noTarget, &atlas.referenceManager()};
  for (SwitchedModelReferenceManager* absl_nonnull referenceManager :
       referenceManagers) {
    referenceManager->setTargetTrajectories(target);
    const Cost cost(*referenceManager,
                    atlasFootCost(atlas, kDefaultSwingVelocityReferenceWeight),
                    atlas.pinocchioInterface(), atlas.adWrenchModel(),
                    kSwingFoot, "testFootYawResidual_foot_l",
                    atlas.modelSettings());
    std::optional<vector_t> heightsOnFlatGround;
    for (const scalar_t terrainHeight : {0.0, 0.37, -0.21}) {
      SCOPED_TRACE(absl::StrCat("terrain height ", terrainHeight));
      referenceManager->setTerrainHeight(terrainHeight);
      contact_flag_t swing = makeFeetArray(true);
      swing[kSwingFoot] = false;
      schedule(*referenceManager, swing, state);
      const SwingTrajectoryPlanner& planner =
          *referenceManager->getSwingTrajectoryPlanner();
      const std::vector<scalar_t> times = {-0.9 * kSwingHalfWidth,
                                           -0.3 * kSwingHalfWidth, 0.0,
                                           0.6 * kSwingHalfWidth};
      vector_t heights(static_cast<Eigen::Index>(times.size()));
      for (size_t i = 0; i < times.size(); ++i) {
        const vector_t parameters =
            cost.getParameters(times[i], target, PreComputation());
        EXPECT_EQ(parameters(kReferenceZIndex),
                  planner.getZpositionConstraint(kSwingFoot, times[i]))
            << "t = " << times[i];
        EXPECT_EQ(parameters(kReferenceZVelocityIndex),
                  planner.getZvelocityConstraint(kSwingFoot, times[i]))
            << "t = " << times[i];
        EXPECT_GT(parameters(kReferenceZIndex), terrainHeight)
            << "a swing foot is referenced above the ground, t = " << times[i];
        heights(static_cast<Eigen::Index>(i)) =
            parameters(kReferenceZIndex) - terrainHeight;
      }
      if (!heightsOnFlatGround.has_value()) heightsOnFlatGround = heights;
      EXPECT_TRUE(heights.isApprox(*heightsOnFlatGround, /*prec=*/1.0e-12))
          << heights.transpose() << " | " << heightsOnFlatGround->transpose();

      // In stance (active_phases "swing_and_stance") the foot is referenced on
      // the ground, at rest.
      schedule(*referenceManager, makeFeetArray(true), state);
      const vector_t stance =
          cost.getParameters(/*time=*/0.0, target, PreComputation());
      EXPECT_NEAR(stance(kReferenceZIndex), terrainHeight, 1.0e-12);
      EXPECT_EQ(stance(kReferenceZVelocityIndex), 0.0);
    }
  }
}

TEST(FootCostParameters,
     TheSwingVelocityReferenceIsWeighedWithTheConfiguredWeight) {
  // Audit C2: a swing without a landing target weighed its xy velocity error
  // with a hard-coded sqrt weight of 1.5. It is
  // task_space_foot_cost.swing_velocity_reference_weight, at construction and
  // on a hot reload, and the swing-phase cost scales with it.
  DrcAtlasContactTestModel atlas("testFootYawResidual_");
  NoLandingTargetReferenceManager noTarget(atlas);
  const vector_t& state = atlas.nominalState();
  const vector_t input = atlas.makeInput(atlas.wrenchModel(), /*loadedFoot=*/1,
                                         /*normalForce=*/800.0);
  vector_t movingState = state;
  movingState(0) = 0.4;  // a commanded forward velocity [m/s], the normalized
                         // linear momentum of the target
  const TargetTrajectories target({DrcAtlasContactTestModel::kQueryTime},
                                  {movingState}, {input});
  noTarget.setTargetTrajectories(target);
  constexpr size_t kSwingFoot = 0;
  contact_flag_t swing = makeFeetArray(true);
  swing[kSwingFoot] = false;
  schedule(noTarget, swing, state);
  // Near lift-off, where the impact proximity has not yet ramped the velocity
  // error away.
  const scalar_t time = -0.95 * kSwingHalfWidth;

  const std::vector<scalar_t> weights = {
      0.0, kDefaultSwingVelocityReferenceWeight, 9.0};
  std::vector<scalar_t> values;
  for (const scalar_t weight : weights) {
    Cost cost = velocityOnlyFootCost(atlas, noTarget, kSwingFoot, weight);
    ASSERT_TRUE(cost.isActive(time));
    const vector_t parameters =
        cost.getParameters(time, target, PreComputation());
    ASSERT_GT(parameters(kImpactProximityIndex), 0.5)
        << "the velocity error is ramped away at the time under test";
    EXPECT_NEAR(parameters(kLinearVelocitySqrtWeightStart), std::sqrt(weight),
                1.0e-15);
    EXPECT_NEAR(parameters(kLinearVelocitySqrtWeightStart + 1),
                std::sqrt(weight), 1.0e-15);
    EXPECT_EQ(parameters(kLinearVelocitySqrtWeightStart + 2), 0.0)
        << "lin_velocity_z keeps its configured weight";
    // A hot reload writes the same parameters as a cost built with the weight.
    Cost reloaded = velocityOnlyFootCost(atlas, noTarget, kSwingFoot,
                                         kDefaultSwingVelocityReferenceWeight);
    reloaded.setSwingVelocityReferenceWeight(weight);
    EXPECT_EQ(reloaded.getSwingVelocityReferenceWeight(),
              cost.getSwingVelocityReferenceWeight());
    EXPECT_TRUE(reloaded.getParameters(time, target, PreComputation()) ==
                parameters);
    values.push_back(
        cost.getValue(time, state, input, target, PreComputation()));
  }
  EXPECT_EQ(values[0], 0.0);
  ASSERT_GT(values[1], 0.0)
      << "the foot's velocity equals its reference, so nothing was checked";
  // The cost is quadratic in the residual, whose two rows carry the sqrt
  // weight: linear in the weight.
  EXPECT_NEAR(values[2] / values[1], weights[2] / weights[1], 1.0e-9);
}

}  // namespace
}  // namespace ocs2::humanoid
