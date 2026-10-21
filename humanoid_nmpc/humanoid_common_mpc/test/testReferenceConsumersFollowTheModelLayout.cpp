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
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The common consumers of the robot model's state - the reference managers, the target calculator's base, the
 * locomotion heuristics' model parameters and the contact wrench cone - read and interpolate the state only through the
 * layout-agnostic accessors of MpcRobotModelBase, never through its slots or TargetTrajectories::getDesiredState. A probe
 * model on the shipped DRC Atlas's centroidal quaternion layout gives those accessors answers no slot read and no linear
 * interpolation reproduces - a heading off the stored yaw, a zero-order hold between knots, a scaled tangent map - and
 * every consumer must follow them. On the shipped models (the identity of these probes) the same calls are the reads of
 * before bit for bit, which the formulation tests pin.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto";
constexpr char kUrdfFile[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

// The centroidal quaternion layout: [h / m (6), base position (3), base orientation xi = (x, y, z, w) (4), joint
// positions (nj)].
constexpr Eigen::Index kBasePositionStart = 6;
constexpr Eigen::Index kBaseOrientationStart = 9;
constexpr Eigen::Index kJointStart = 13;
constexpr Eigen::Index kGeneralizedCoordinatesStart = kBasePositionStart;

/** What the probe model answers differently from the slots it stores. */
struct ProbeAnswers {
  /** [rad] getBaseHeading() is the ZYX yaw of the stored orientation plus this, not its twist heading. */
  scalar_t headingOffset = 0.0;
  /** getPinocchioTangentJacobian() is CentroidalMpcRobotModel's (the selection of the slots, E^+ of the quaternion) times this. */
  scalar_t tangentScale = 1.0;
  /** getVelocityDim() is the configuration's dimension plus this. */
  size_t extraVelocityEntries = 0;
};

/**
 * The centroidal quaternion layout of a robot's ModelSettings, read and written slot for slot as CentroidalMpcRobotModel
 * does, except for three probes (ProbeAnswers) and its interpolation, a zero-order hold: the state of the last knot at
 * or before the time. Not thread-safe to construct; its const methods are.
 */
class ProbeRobotModel final : public MpcRobotModelBase<scalar_t> {
 public:
  ProbeRobotModel(const ModelSettings& modelSettings, ProbeAnswers answers)
      : MpcRobotModelBase<scalar_t>(
            modelSettings,
            MpcRobotModelDimensions{.stateDim = tuningDimension(stateInputLayout(modelSettings, StateInputLayout::Mpc::kCentroidal)) + 1,
                                    .inputDim = inputDimension(stateInputLayout(modelSettings, StateInputLayout::Mpc::kCentroidal)),
                                    .configurationDim = kBaseConfigurationDim + modelSettings.mpc_joint_dim,
                                    .velocityDim = kBaseVelocityDim + modelSettings.mpc_joint_dim + answers.extraVelocityEntries},
            placeTuningLayout(tuningLayoutOf(stateInputLayout(modelSettings, StateInputLayout::Mpc::kCentroidal)))),
        answers_(answers) {}
  ~ProbeRobotModel() override = default;
  ProbeRobotModel& operator=(const ProbeRobotModel&) = delete;
  ProbeRobotModel(ProbeRobotModel&&) = delete;
  ProbeRobotModel& operator=(ProbeRobotModel&&) = delete;

  ProbeRobotModel* absl_nonnull clone() const override { return new ProbeRobotModel(*this); }

  size_t getBaseStartindex() const override { return kBasePositionStart; }
  size_t getBaseOrientationStartIndex() const override { return kBaseOrientationStart; }
  size_t getJointStartindex() const override { return kJointStart; }
  size_t getJointVelocitiesStartindex() const override { return 6 * modelSettings.contactNames.size(); }
  size_t getContactWrenchStartIndices(size_t contactIndex) const override { return 6 * contactIndex; }

  vector_t getGeneralizedCoordinates(const vector_t& state) const override {
    vector_t q = state.segment(kGeneralizedCoordinatesStart, static_cast<Eigen::Index>(getConfigurationDim()));
    q.segment<4>(kBaseOrientationStart - kGeneralizedCoordinatesStart) = getBaseOrientation(state);
    return q;
  }
  vector3_t getBasePosition(const vector_t& state) const override { return state.segment<3>(kBasePositionStart); }
  vector3_t getBaseComLinearVelocity(const vector_t& state) const override { return state.head<3>(); }
  vector6_t getBaseComVelocity(const vector_t& state) const override { return state.head<6>(); }
  vector_t getJointAngles(const vector_t& state) const override { return state.segment(kJointStart, jointDim()); }
  vector_t getJointVelocities(const vector_t& /*state*/, const vector_t& input) const override {
    return input.segment(static_cast<Eigen::Index>(getJointVelocitiesStartindex()), jointDim());
  }
  vector_t getGeneralizedVelocities(const vector_t& state, const vector_t& input) override {
    vector_t velocities = vector_t::Zero(static_cast<Eigen::Index>(getVelocityDim()));
    velocities.segment(kBaseVelocityDim, jointDim()) = getJointVelocities(state, input);
    return velocities;
  }

  void setGeneralizedCoordinates(vector_t& state, const vector_t& generalizedCorrdinates) const override {
    state.segment(kGeneralizedCoordinatesStart, generalizedCorrdinates.size()) = generalizedCorrdinates;
  }
  void setBasePosition(vector_t& state, const vector3_t& position) const override { state.segment<3>(kBasePositionStart) = position; }
  void setBaseComLinearVelocity(vector_t& state, const vector3_t& velocity) const override { state.head<3>() = velocity; }
  void setJointAngles(vector_t& state, const vector_t& jointAngles) const override { state.segment(kJointStart, jointDim()) = jointAngles; }
  void setJointVelocities(vector_t& /*state*/, vector_t& input, const vector_t& jointVelocities) const override {
    input.segment(static_cast<Eigen::Index>(getJointVelocitiesStartindex()), jointDim()) = jointVelocities;
  }
  void adaptBasePoseHeight(vector_t& state, scalar_t heightChange) const override { state(kBasePositionStart + 2) += heightChange; }

  vector4_t getBaseOrientation(const vector_t& state) const override {
    return safelyNormalizedQuaternion(vector4_t(state.segment<4>(kBaseOrientationStart)));
  }
  void setBaseOrientation(vector_t& state, const vector4_t& orientation) const override {
    state.segment<4>(kBaseOrientationStart) = safelyNormalizedQuaternion(orientation);
  }
  scalar_t getBaseHeading(const vector_t& state) const override { return getBaseOrientationEulerZyx(state)(0) + answers_.headingOffset; }
  vector6_t getBasePoseEulerZyx(const vector_t& state) const override {
    vector6_t basePose;
    basePose << getBasePosition(state), getBaseOrientationEulerZyx(state);
    return basePose;
  }
  void setBasePoseEulerZyx(vector_t& state, const vector6_t& basePose) const override {
    setBasePosition(state, basePose.head<3>());
    setBaseOrientationEulerZyx(state, basePose.tail<3>());
  }
  vector3_t getBaseOrientationEulerZyx(const vector_t& state) const override { return eulerZyxFromQuaternion(getBaseOrientation(state)); }
  void setBaseOrientationEulerZyx(vector_t& state, const vector3_t& eulerZyx) const override {
    state.segment<4>(kBaseOrientationStart) = quaternionFromEulerZyx(eulerZyx);
  }

  // The centroidal state holds no base velocity (MpcRobotModelBase); nothing tested here reads one.
  vector3_t getBaseAngularVelocityLocal(const vector_t& /*state*/) const override { return vector3_t::Zero(); }
  void setBaseAngularVelocityLocal(vector_t& /*state*/, const vector3_t& /*angularVelocityLocal*/) const override {}
  void setBaseAngularVelocityFromWorldYawRate(vector_t& /*state*/, scalar_t /*yawRate*/) const override {}
  BaseAngularVelocityJacobian getBaseAngularVelocityLocalJacobian(const vector_t& /*state*/) const override {
    return BaseAngularVelocityJacobian{.orientationSlots = matrix_t::Zero(3, kQuaternionOrientationStateDim),
                                       .angularVelocitySlots = matrix3_t::Zero()};
  }
  // Nothing tested here reads the base twist, or a knot's yaw rate; the commanded velocity is the knot's CoM velocity,
  // as CentroidalMpcRobotModel reads it.
  vector6_t getBaseTwistInWorldFrame(const vector_t& /*state*/, const vector_t& /*jointVelocities*/) const override {
    return vector6_t::Zero();
  }
  vector3_t getTargetPlanarCommand(const vector_t& targetState, scalar_t /*compositeYawInertia*/) const override {
    return vector3_t(targetState(0), targetState(1), 0.0);
  }

  absl::StatusOr<vector_t> stateFromTuning(const vector_t& tuningVector) const override {
    return stateFromTuningLayout(tuningVector, getTuningLayout());
  }
  absl::StatusOr<vector_t> tuningFromState(const vector_t& state) const override { return tuningLayoutFromState(state, getTuningLayout()); }

  /** The state of the last knot at or before `time`, the first knot's before them. */
  vector_t interpolateStateTrajectory(scalar_t time, const scalar_array_t& times, const vector_array_t& states) const override {
    const scalar_array_t::const_iterator next = std::upper_bound(times.begin(), times.end(), time);
    if (next == times.begin()) return states.front();
    return states[static_cast<size_t>(std::distance(times.begin(), next)) - 1];
  }
  vector_t getNeutralState() const override {
    vector_t state = vector_t::Zero(static_cast<Eigen::Index>(getStateDim()));
    state.segment<4>(kBaseOrientationStart) = vector4_t(0.0, 0.0, 0.0, 1.0);
    return state;
  }

  matrix_t getBaseRotationTangentJacobian(const vector_t& state) const override {
    return configurationTangentLift(vector4_t(state.segment<4>(kBaseOrientationStart)));
  }
  /** CentroidalMpcRobotModel's map times the probe's scale; the extra velocity entries, if any, have no columns. */
  matrix_t getPinocchioTangentJacobian(const vector_t& state) const override {
    matrix_t jacobian = matrix_t::Zero(static_cast<Eigen::Index>(getVelocityDim()), static_cast<Eigen::Index>(getStateDim()));
    jacobian.block<3, 3>(0, kBasePositionStart).setIdentity();
    jacobian.block<3, 4>(3, kBaseOrientationStart) = getBaseRotationTangentJacobian(state);
    jacobian.block(kBaseVelocityDim, kJointStart, jointDim(), jointDim()).setIdentity();
    return answers_.tangentScale * jacobian;
  }

  void fillGeneralizedCoordinatesFromMeasurement(const vector3_t& /*positionWorld*/,
                                                 const vector4_t& /*orientation*/,
                                                 const vector_t& /*jointPositions*/,
                                                 Eigen::Ref<vector_t> q) const override {
    q.setZero();
  }
  void fillGeneralizedVelocitiesFromMeasurement(const vector4_t& /*orientation*/,
                                                const vector3_t& /*linearVelocityLocal*/,
                                                const vector3_t& /*angularVelocityLocal*/,
                                                const vector_t& /*jointVelocities*/,
                                                Eigen::Ref<vector_t> v) const override {
    v.setZero();
  }

  vector6_t getContactWrench(const vector_t& input, size_t contactIndex) const override {
    return input.segment<6>(wrenchStart(contactIndex));
  }
  vector3_t getContactForce(const vector_t& input, size_t contactIndex) const override {
    return input.segment<3>(wrenchStart(contactIndex));
  }
  vector3_t getContactMoment(const vector_t& input, size_t contactIndex) const override {
    return input.segment<3>(wrenchStart(contactIndex) + 3);
  }
  void setContactWrench(vector_t& input, const vector6_t& wrench, size_t contactIndex) const override {
    input.segment<6>(wrenchStart(contactIndex)) = wrench;
  }
  void setContactForce(vector_t& input, const vector3_t& force, size_t contactIndex) const override {
    input.segment<3>(wrenchStart(contactIndex)) = force;
  }
  void setContactMoment(vector_t& input, const vector3_t& moment, size_t contactIndex) const override {
    input.segment<3>(wrenchStart(contactIndex) + 3) = moment;
  }

 private:
  // For clone().
  ProbeRobotModel(const ProbeRobotModel& other) = default;

  Eigen::Index jointDim() const { return static_cast<Eigen::Index>(getJointDim()); }
  Eigen::Index wrenchStart(size_t contactIndex) const { return static_cast<Eigen::Index>(getContactWrenchStartIndices(contactIndex)); }

  ProbeAnswers answers_;
};

/** A calculator that builds no targets, for the base class's level start pose. */
class LevelPoseCalculator final : public TargetTrajectoriesCalculatorBase {
 public:
  explicit LevelPoseCalculator(const MpcRobotModelBase<scalar_t>& model)
      : TargetTrajectoriesCalculatorBase(ReferenceSettings(),
                                         vector_t::Zero(static_cast<Eigen::Index>(model.getJointDim())),
                                         model,
                                         /*mpcHorizon=*/1.0) {}

  TargetTrajectories commandedPositionToTargetTrajectories(const vector4_t& /*commandedVelocities*/,
                                                           scalar_t /*initTime*/,
                                                           const vector_t& /*initState*/) override {
    return TargetTrajectories();
  }
  TargetTrajectories commandedVelocityToTargetTrajectories(const vector4_t& /*commandedVelocities*/,
                                                           scalar_t /*initTime*/,
                                                           const vector_t& /*initState*/,
                                                           const BasePlanarMotion& /*measuredMotion*/) override {
    return TargetTrajectories();
  }

  using TargetTrajectoriesCalculatorBase::getCurrentBasePoseTarget;
  using TargetTrajectoriesCalculatorBase::VelocityCommandKnot;
  using TargetTrajectoriesCalculatorBase::velocityCommandKnots;
};

/** Whether `a` and `b` have the same size and the same bits. */
bool bitwiseEqual(const vector_t& a, const vector_t& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), static_cast<size_t>(a.size()) * sizeof(scalar_t)) == 0;
}

/** A plan that stands on both feet for `numNodes` nodes of `dt` from `startTime`, the center of mass at rest. */
ContactPlan standingPlan(scalar_t startTime, scalar_t dt, size_t numNodes) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = startTime;
  plan.dt = dt;
  plan.committedUntil = startTime + dt;
  plan.contacts.assign(numNodes, makeFeetArray(true));
  plan.footholds.assign(numNodes + 1, makeFeetArray(vector2_t(vector2_t::Zero())));
  plan.comPosition.assign(numNodes + 1, vector2_t::Zero());
  plan.comVelocity.assign(numNodes + 1, vector2_t::Zero());
  plan.zmp.assign(numNodes, vector2_t::Zero());
  return plan;
}

/** The shipped DRC Atlas's settings, Pinocchio model and gait, and a state and two target knots on its layout. */
class ReferenceConsumersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> taskFile = robot::resolveResourcePath(kTaskFile);
    const absl::StatusOr<std::string> referenceFile = robot::resolveResourcePath(kReferenceFile);
    const absl::StatusOr<std::string> urdfFile = robot::resolveResourcePath(kUrdfFile);
    ASSERT_TRUE(taskFile.ok() && referenceFile.ok() && urdfFile.ok()) << "the shipped DRC Atlas files are not in the runfiles";
    referenceFile_ = *referenceFile;
    absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(*taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    absl::StatusOr<ModelSettings> settings =
        ModelSettings::Create(*task, *urdfFile, "testReferenceConsumersFollowTheModelLayout", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    settings_ = std::make_unique<ModelSettings>(*std::move(settings));
    absl::StatusOr<PinocchioInterface> pinocchioInterface = loadCustomPinocchioInterface(*task, *urdfFile, *settings_);
    ASSERT_TRUE(pinocchioInterface.ok()) << pinocchioInterface.status();
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(*std::move(pinocchioInterface));

    // A tilted, turned, moving robot with bent joints, and two target knots that differ in every block.
    const ProbeRobotModel layout(*settings_, ProbeAnswers());
    const Eigen::Index jointDim = static_cast<Eigen::Index>(layout.getJointDim());
    state_ = layout.getNeutralState();
    layout.setBaseComLinearVelocity(state_, vector3_t(0.3, -0.1, 0.0));
    layout.setBasePosition(state_, vector3_t(1.0, 2.0, 0.9));
    layout.setBaseOrientationEulerZyx(state_, vector3_t(0.4, 0.1, -0.2));
    vector_t joints(jointDim);
    for (Eigen::Index joint = 0; joint < jointDim; ++joint) joints(joint) = 0.05 * std::sin(static_cast<scalar_t>(joint + 1));
    layout.setJointAngles(state_, joints);

    firstKnot_ = state_;
    layout.setBaseComLinearVelocity(firstKnot_, vector3_t(0.2, 0.1, 0.0));
    lastKnot_ = state_;
    layout.setBaseComLinearVelocity(lastKnot_, vector3_t(0.6, 0.5, 0.0));
    layout.setBasePosition(lastKnot_, vector3_t(3.0, 1.0, 0.8));
    layout.setBaseOrientationEulerZyx(lastKnot_, vector3_t(1.4, 0.0, 0.0));
    layout.setJointAngles(lastKnot_, vector_t(joints + vector_t::Constant(jointDim, 0.3)));
    zeroInput_ = vector_t::Zero(static_cast<Eigen::Index>(layout.getInputDim()));
  }

  /** The reference manager of `model` on the fixture's gait and Pinocchio model. */
  std::unique_ptr<SwitchedModelReferenceManager> referenceManagerOf(const MpcRobotModelBase<scalar_t>& model) const {
    absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule = GaitSchedule::Create(referenceFile_, *settings_);
    EXPECT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
    if (!gaitSchedule.ok()) return nullptr;
    return std::make_unique<SwitchedModelReferenceManager>(
        *std::move(gaitSchedule), std::make_unique<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(), /*numFeet=*/2),
        *pinocchioInterface_, model);
  }

  /** The contact planner's reference manager of `model` with `config`, which must be valid. */
  std::shared_ptr<ContactPlanningReferenceManager> contactPlanningReferenceManagerOf(const MpcRobotModelBase<scalar_t>& model,
                                                                                     const ContactPlanningConfig& config) const {
    absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule = GaitSchedule::Create(referenceFile_, *settings_);
    EXPECT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
    if (!gaitSchedule.ok()) return nullptr;
    absl::StatusOr<std::shared_ptr<ContactPlanningReferenceManager>> manager = ContactPlanningReferenceManager::Create(
        *std::move(gaitSchedule), std::make_unique<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(), /*numFeet=*/2),
        *pinocchioInterface_, model, config);
    EXPECT_TRUE(manager.ok()) << manager.status();
    return manager.ok() ? *std::move(manager) : nullptr;
  }

  /** The two-knot target from firstKnot_ at time 0 to lastKnot_ at time 1. */
  TargetTrajectories twoKnotTarget() const { return TargetTrajectories({0.0, 1.0}, {firstKnot_, lastKnot_}, {zeroInput_, zeroInput_}); }

  std::string referenceFile_;
  std::unique_ptr<ModelSettings> settings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  vector_t state_;
  vector_t firstKnot_;
  vector_t lastKnot_;
  vector_t zeroInput_;
};

TEST_F(ReferenceConsumersTest, TheStateReferenceIsTheModelsInterpolationOfTheTarget) {
  const ProbeRobotModel model(*settings_, ProbeAnswers());
  const std::unique_ptr<SwitchedModelReferenceManager> referenceManager = referenceManagerOf(model);
  ASSERT_NE(referenceManager, nullptr);
  const TargetTrajectories target = twoKnotTarget();
  // Between the knots the hold answers the first knot, which the linear interpolation of the knots never does.
  EXPECT_TRUE(bitwiseEqual(referenceManager->getDesiredState(target, state_, /*time=*/0.5), firstKnot_));
  // NOLINTNEXTLINE(state-interpolation): the linear interpolation the probe model's must differ from
  EXPECT_FALSE(bitwiseEqual(target.getDesiredState(0.5), firstKnot_)) << "the probe does not tell the two interpolations apart";
  EXPECT_TRUE(bitwiseEqual(referenceManager->getDesiredState(target, state_, /*time=*/1.5), lastKnot_));
}

TEST_F(ReferenceConsumersTest, TheCommandedVelocityIsReadThroughTheModelsInterpolation) {
  const ProbeRobotModel model(*settings_, ProbeAnswers());
  const std::unique_ptr<SwitchedModelReferenceManager> referenceManager = referenceManagerOf(model);
  ASSERT_NE(referenceManager, nullptr);
  referenceManager->setTargetTrajectories(twoKnotTarget());
  referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state_, ModeNumber::kStance);
  const vector2_t commanded = referenceManager->getCommandedVelocity(/*time=*/0.5);
  EXPECT_EQ(commanded, vector2_t(0.2, 0.1)) << "the linear interpolation would answer (0.4, 0.3)";
}

TEST_F(ReferenceConsumersTest, ATargetStartsFromTheModelsHeadingWithTheBaseLevel) {
  constexpr scalar_t kHeadingOffset = 0.25;
  const ProbeRobotModel model(*settings_, ProbeAnswers{.headingOffset = kHeadingOffset});
  const LevelPoseCalculator calculator(model);
  // The probe's heading: the stored attitude's ZYX yaw, 0.4 to round-off, plus the offset - not its twist heading.
  const scalar_t heading = model.getBaseHeading(state_);
  ASSERT_NEAR(heading, 0.4 + kHeadingOffset, 1.0e-12);
  ASSERT_GT(std::abs(heading - kHeadingOffset - twistHeading(model.getBaseOrientation(state_))), 1.0e-3)
      << "at this mixed tilt the twist heading would answer differently";
  vector6_t expected;
  expected << 1.0, 2.0, 0.9, heading, 0.0, 0.0;
  EXPECT_EQ(calculator.getCurrentBasePoseTarget(state_), expected) << "the base position, the heading and no roll or pitch";
}

TEST_F(ReferenceConsumersTest, TheVelocityCommandKnotsDoNotDependOnTheSolveRate) {
  // A step in the command at t = 1 s, solved at 100 Hz by one calculator and at 25 Hz by another: at t = 1.2 s, a solve
  // time of both, the knots must be the same, and those of a fresh calculator. A per-call low-pass filter of the command
  // (alpha = 0.8, as the calculator once had) let 1 - 0.8^21 of the step through at 100 Hz and 1 - 0.8^6 at 25 Hz.
  const ProbeRobotModel model(*settings_, ProbeAnswers());
  const vector4_t step(0.5, -0.1, 0.0, 0.4);
  const BasePlanarMotion measured{.velocityInHeadingFrame = vector2_t(0.1, 0.0), .headingRate = 0.05};
  constexpr scalar_t kStepTime = 1.0;
  constexpr scalar_t kQueryTime = 1.2;
  const std::vector<std::vector<LevelPoseCalculator::VelocityCommandKnot>> answers = [&]() {
    std::vector<std::vector<LevelPoseCalculator::VelocityCommandKnot>> result;
    for (const int solvesPerSecond : {100, 25, 0}) {
      LevelPoseCalculator calculator(model);
      if (solvesPerSecond > 0) {
        const scalar_t period = 1.0 / solvesPerSecond;
        for (int k = 0; k * period < kQueryTime - 1.0e-9; ++k) {
          const scalar_t time = k * period;
          calculator.velocityCommandKnots(time < kStepTime ? vector4_t::Zero() : step, time, state_, measured);
        }
      }
      result.push_back(calculator.velocityCommandKnots(step, kQueryTime, state_, measured));
    }
    return result;
  }();
  for (size_t run = 1; run < answers.size(); ++run) {
    ASSERT_EQ(answers[run].size(), answers[0].size());
    for (size_t k = 0; k < answers[0].size(); ++k) {
      EXPECT_EQ(answers[run][k].time, answers[0][k].time);
      EXPECT_EQ(answers[run][k].basePose, answers[0][k].basePose) << "run " << run << " knot " << k;
      EXPECT_EQ(answers[run][k].commandedVelocityWorld, answers[0][k].commandedVelocityWorld) << "run " << run << " knot " << k;
      EXPECT_EQ(answers[run][k].commandedYawRate, answers[0][k].commandedYawRate) << "run " << run << " knot " << k;
    }
  }
  // And the knot's command is the step itself, turned into the world at the start heading.
  const scalar_t heading = model.getBaseHeading(state_);
  const vector2_t expectedWorld(std::cos(heading) * step(0) - std::sin(heading) * step(1),
                                std::sin(heading) * step(0) + std::cos(heading) * step(1));
  EXPECT_LT((answers[0].front().commandedVelocityWorld - expectedWorld).norm(), 1.0e-12);
  EXPECT_EQ(answers[0].front().commandedYawRate, step(3));
}

TEST_F(ReferenceConsumersTest, ThePlannersHeadingIsTheModelsHeadingWithoutAnAcom) {
  constexpr scalar_t kHeadingOffset = 0.25;
  const ProbeRobotModel model(*settings_, ProbeAnswers{.headingOffset = kHeadingOffset});
  ContactPlanningConfig config;
  config.shared.comHeight = 0.85;  // no model derives the pendulum here (unset means "from the model")
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  const std::shared_ptr<ContactPlanningReferenceManager> referenceManager = contactPlanningReferenceManagerOf(model, config);
  ASSERT_NE(referenceManager, nullptr);
  const scalar_t heading = model.getBaseHeading(state_);
  ASSERT_NEAR(heading, 0.4 + kHeadingOffset, 1.0e-12);
  EXPECT_EQ(referenceManager->computeHeading(state_), heading);
  referenceManager->setTargetTrajectories(twoKnotTarget());
  referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state_, ModeNumber::kStance);
  EXPECT_EQ(referenceManager->makePlannerInput(/*initTime=*/0.0, state_, vector2_t::Zero()).yaw, heading);
}

TEST_F(ReferenceConsumersTest, ThePlannersResampledTargetIsTheModelsInterpolation) {
  const ProbeRobotModel model(*settings_, ProbeAnswers());
  ContactPlanningConfig config;
  config.shared.comHeight = 0.85;  // no model derives the pendulum here (unset means "from the model")
  config.formulation.setExecutionRule(term::kPlannedComOverride, /*on=*/true);
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  const std::shared_ptr<ContactPlanningReferenceManager> referenceManager = contactPlanningReferenceManagerOf(model, config);
  ASSERT_NE(referenceManager, nullptr);
  // The operator's knots ten seconds apart, and a standing plan over the first half second: the target in use is then
  // resampled at the plan's nodes. planned_com_override rewrites only the base position and the CoM velocity of those
  // samples, so their joint angles are the resampling's.
  constexpr size_t kPlanNodes = 5;
  referenceManager->setTargetTrajectories(TargetTrajectories({0.0, 10.0}, {firstKnot_, lastKnot_}, {zeroInput_, zeroInput_}));
  referenceManager->setContactPlan(standingPlan(/*startTime=*/0.0, config.planner.dt, kPlanNodes));
  referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state_, ModeNumber::kStance);

  const TargetTrajectories& resampled = referenceManager->getTargetTrajectories();
  ASSERT_GT(resampled.timeTrajectory.size(), kPlanNodes) << "the plan's nodes were not resampled";
  for (size_t knot = 1; knot <= kPlanNodes; ++knot) {
    ASSERT_LT(resampled.timeTrajectory[knot], 10.0);
    EXPECT_TRUE(bitwiseEqual(model.getJointAngles(resampled.stateTrajectory[knot]), model.getJointAngles(firstKnot_)))
        << "knot " << knot << " at " << resampled.timeTrajectory[knot] << " s is not the hold of the operator's first knot";
  }
}

TEST_F(ReferenceConsumersTest, TheHipLandmarksAreTurnedIntoTheModelsHeading) {
  constexpr scalar_t kHeadingOffset = 0.25;
  const ProbeRobotModel unturned(*settings_, ProbeAnswers());
  const ProbeRobotModel turned(*settings_, ProbeAnswers{.headingOffset = kHeadingOffset});
  const absl::StatusOr<LocomotionHeuristicModelParameters> reference =
      deriveLocomotionHeuristicModelParameters(*pinocchioInterface_, unturned, state_);
  const absl::StatusOr<LocomotionHeuristicModelParameters> probed =
      deriveLocomotionHeuristicModelParameters(*pinocchioInterface_, turned, state_);
  ASSERT_TRUE(reference.ok()) << reference.status();
  ASSERT_TRUE(probed.ok()) << probed.status();
  // The same hips in the world, read in a base frame turned by the offset.
  const scalar_t cosOffset = std::cos(kHeadingOffset);
  const scalar_t sinOffset = std::sin(kHeadingOffset);
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const vector2_t& hip = reference->hipPositionInBaseFrame[foot];
    const vector2_t expected(cosOffset * hip.x() + sinOffset * hip.y(), -sinOffset * hip.x() + cosOffset * hip.y());
    EXPECT_LT((probed->hipPositionInBaseFrame[foot] - expected).norm(), 1.0e-12) << "foot " << foot;
  }
}

/**
 * The contact planner's dense resample of a turning target (ContactPlanningReferenceManager::preSolverRun): sampled at
 * the plan's nodes alone, the shortest-path slerp between two nodes more than a quarter turn apart would take the other
 * way round, so the resample adds the target's own knots and cuts between them until consecutive knots are at most
 * pi / 2 apart - for a turn faster than pi / 2 per node (63 rad/s at dt = 0.025 s; 15.7 rad/s at the default 0.1 s) -
 * and adds nothing below that. The model is the centroidal quaternion layout, which slerps its interpolation.
 */
TEST_F(ReferenceConsumersTest, TheDenseResampleOfAFastTurnKeepsConsecutiveKnotsAQuarterTurnApart) {
  const test::QuaternionLayoutRobotModel model(*settings_, StateInputLayout::Mpc::kCentroidal);
  ContactPlanningConfig config;
  config.shared.comHeight = 0.85;  // no model derives the pendulum here (unset means "from the model")
  config.formulation.setExecutionRule(term::kPlannedComOverride, /*on=*/true);
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  constexpr size_t kPlanNodes = 5;
  const scalar_t dt = config.planner.dt;
  const scalar_t planEnd = static_cast<scalar_t>(kPlanNodes) * dt;
  const scalar_t knotSpacing = 0.25 * dt;

  // The operator's knots a quarter of a node apart for 1.2 plan lengths, the base turning `yawPerKnot` between them.
  const std::function<TargetTrajectories(scalar_t)> turningTarget = [&](scalar_t yawPerKnot) {
    TargetTrajectories target;
    for (size_t knot = 0; static_cast<scalar_t>(knot) * knotSpacing <= 1.2 * planEnd; ++knot) {
      vector_t state = firstKnot_;
      model.setBaseOrientationEulerZyx(state, vector3_t(yawPerKnot * static_cast<scalar_t>(knot), 0.0, 0.0));
      target.timeTrajectory.push_back(static_cast<scalar_t>(knot) * knotSpacing);
      target.stateTrajectory.push_back(state);
      target.inputTrajectory.push_back(zeroInput_);
    }
    return target;
  };
  // The resampled target's knots inside the plan, after a solver run at time 0.
  const std::function<TargetTrajectories(const TargetTrajectories&)> resampledInsideThePlan = [&](const TargetTrajectories& target) {
    const std::shared_ptr<ContactPlanningReferenceManager> referenceManager = contactPlanningReferenceManagerOf(model, config);
    TargetTrajectories inside;
    if (referenceManager == nullptr) return inside;
    referenceManager->setTargetTrajectories(target);
    referenceManager->setContactPlan(standingPlan(/*startTime=*/0.0, dt, kPlanNodes));
    referenceManager->preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state_, ModeNumber::kStance);
    const TargetTrajectories& resampled = referenceManager->getTargetTrajectories();
    for (size_t knot = 0; knot < resampled.timeTrajectory.size(); ++knot) {
      if (resampled.timeTrajectory[knot] > planEnd + 1.0e-12) break;
      inside.timeTrajectory.push_back(resampled.timeTrajectory[knot]);
      inside.stateTrajectory.push_back(resampled.stateTrajectory[knot]);
    }
    return inside;
  };

  // 1.2 rad a quarter node apart, 4.8 rad per node: every knot of the operator's inside the plan is kept, and the
  // resampled base follows the turn the way it turns.
  constexpr scalar_t kFastYawPerKnot = 1.2;
  const TargetTrajectories fast = resampledInsideThePlan(turningTarget(kFastYawPerKnot));
  ASSERT_EQ(fast.timeTrajectory.size(), 4 * kPlanNodes + 1) << "the plan's nodes and the operator's knots between them";
  for (size_t knot = 0; knot < fast.timeTrajectory.size(); ++knot) {
    const scalar_t expectedYaw = kFastYawPerKnot * fast.timeTrajectory[knot] / knotSpacing;
    EXPECT_LT(std::abs(std::remainder(model.getBaseHeading(fast.stateTrajectory[knot]) - expectedYaw, 2.0 * std::numbers::pi)), 1.0e-9)
        << "knot " << knot << " at " << fast.timeTrajectory[knot] << " s";
    if (knot == 0) continue;
    const vector4_t relative =
        quaternionProduct<scalar_t>(quaternionConjugate<scalar_t>(model.getBaseOrientation(fast.stateTrajectory[knot - 1])),
                                    model.getBaseOrientation(fast.stateTrajectory[knot]));
    EXPECT_LE(quaternionLog<scalar_t>(relative).norm(), 0.5 * std::numbers::pi + 1.0e-12) << "knots " << knot - 1 << " and " << knot;
  }

  // Positive control: 0.2 rad/s turns 0.02 rad per node, and the resample is the plan's nodes alone.
  const TargetTrajectories slow = resampledInsideThePlan(turningTarget(/*yawPerKnot=*/0.2 * knotSpacing));
  ASSERT_EQ(slow.timeTrajectory.size(), kPlanNodes + 1);
  for (size_t node = 0; node <= kPlanNodes; ++node) {
    EXPECT_NEAR(slow.timeTrajectory[node], static_cast<scalar_t>(node) * dt, 1.0e-12) << "node " << node;
  }
}

/** The left foot's wrench cone of `model`, on `referenceManager`, or null after a failed expectation. */
std::unique_ptr<ContactWrenchConeConstraint> leftFootCone(const SwitchedModelReferenceManager& referenceManager,
                                                          const PinocchioInterface& pinocchioInterface,
                                                          const MpcRobotModelBase<scalar_t>& model) {
  const ContactRectangle rectangle(PolygonBounds(/*x_min=*/-0.1, /*x_max=*/0.15, /*y_min=*/-0.05, /*y_max=*/0.05),
                                   ContactCenterPoint("foot_l_contact", "l_leg_akx", vector3_t::Zero()));
  absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> cone =
      ContactWrenchConeConstraint::Create(referenceManager, rectangle, /*contactPointIndex=*/0, pinocchioInterface, model);
  EXPECT_TRUE(cone.ok()) << cone.status();
  return cone.ok() ? *std::move(cone) : nullptr;
}

TEST_F(ReferenceConsumersTest, TheWrenchConesStateJacobianGoesThroughTheModelsTangentMap) {
  const ProbeRobotModel selection(*settings_, ProbeAnswers());
  const ProbeRobotModel scaled(*settings_, ProbeAnswers{.tangentScale = 2.0});
  const std::unique_ptr<SwitchedModelReferenceManager> referenceManager = referenceManagerOf(selection);
  ASSERT_NE(referenceManager, nullptr);
  const std::unique_ptr<ContactWrenchConeConstraint> selectionCone = leftFootCone(*referenceManager, *pinocchioInterface_, selection);
  const std::unique_ptr<ContactWrenchConeConstraint> scaledCone = leftFootCone(*referenceManager, *pinocchioInterface_, scaled);
  ASSERT_NE(selectionCone, nullptr);
  ASSERT_NE(scaledCone, nullptr);

  vector_t input = zeroInput_;
  selection.setContactForce(input, vector3_t(10.0, -20.0, 400.0), /*contactIndex=*/0);
  selection.setContactMoment(input, vector3_t(3.0, -2.0, 1.0), /*contactIndex=*/0);
  const PreComputation preComputation;
  const matrix_t dfdx = selectionCone->getLinearApproximation(/*time=*/0.0, state_, input, preComputation).dfdx;
  const matrix_t scaledDfdx = scaledCone->getLinearApproximation(/*time=*/0.0, state_, input, preComputation).dfdx;
  ASSERT_GT(dfdx.norm(), 1.0) << "the tilted foot's loaded cone must depend on the state";
  EXPECT_LE((scaledDfdx - 2.0 * dfdx).cwiseAbs().maxCoeff(), 1.0e-12 * dfdx.cwiseAbs().maxCoeff());
  // The momenta are no configuration: the map has no columns there, so neither has the cone.
  EXPECT_TRUE(dfdx.leftCols(kGeneralizedCoordinatesStart).isZero(0.0));
}

TEST_F(ReferenceConsumersTest, TheWrenchConeRefusesAModelWhoseVelocityIsNotPinocchios) {
  const ProbeRobotModel model(*settings_, ProbeAnswers{.extraVelocityEntries = 1});
  const std::unique_ptr<SwitchedModelReferenceManager> referenceManager = referenceManagerOf(model);
  ASSERT_NE(referenceManager, nullptr);
  const ContactRectangle rectangle(PolygonBounds(/*x_min=*/-0.1, /*x_max=*/0.15, /*y_min=*/-0.05, /*y_max=*/0.05),
                                   ContactCenterPoint("foot_l_contact", "l_leg_akx", vector3_t::Zero()));
  const absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> cone =
      ContactWrenchConeConstraint::Create(*referenceManager, rectangle, /*contactPointIndex=*/0, *pinocchioInterface_, model);
  ASSERT_FALSE(cone.ok());
  EXPECT_EQ(cone.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(cone.status().message(), "Pinocchio velocity")) << cone.status();
  EXPECT_EQ(ContactWrenchConeConstraint::checkPinocchioVelocityDim(*pinocchioInterface_, ProbeRobotModel(*settings_, ProbeAnswers())),
            absl::OkStatus());
}

}  // namespace
}  // namespace ocs2::humanoid
