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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/model/ModelSettingsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The reference manager per leg: the biped keeps its sides, hips and gait
 * phase; on the Atlas sole-corner set with the legs LF, RF, LH and RH each leg
 * has its own swing phase, the nominal_stance anchor lands each contact at its
 * nominal stance point relative to the predicted base turned by the predicted
 * yaw, and the foothold heuristics read each leg's side and hip. Also the
 * anchor's name registry and the refusals of ModelSettings::Create().
 */
namespace ocs2::humanoid {
namespace {

constexpr char kTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

// The sole corners of the Atlas, in a quadruped's leg order.
constexpr char kFrontLeft[] = "sole_l_toe_contact";
constexpr char kFrontRight[] = "sole_r_toe_contact";
constexpr char kHindLeft[] = "sole_l_heel_contact";
constexpr char kHindRight[] = "sole_r_heel_contact";
constexpr size_t kNumLegs = 4;

constexpr scalar_t kTolerance = 1.0e-9;
constexpr scalar_t kCmdLinvelBaseRtWorldEwrtWorldX = 0.4;  // [m/s]
constexpr scalar_t kMeasYawBaseRtWorld = 0.3;              // [rad]
constexpr scalar_t kSolveTime = 0.0;                       // [s]

// The trot of the four-leg tests: all four until 0.2 s, LF and RH stand while
// RF and LH swing until 0.5 s, RF and LH stand while LF and RH swing until
// 0.8 s, all four after.
constexpr scalar_t kFirstLiftOffTime = 0.2;
constexpr scalar_t kDiagonalSwitchTime = 0.5;
constexpr scalar_t kLastTouchDownTime = 0.8;

/** The reference manager with its foothold context exposed. */
class ReferenceManagerPeer : public SwitchedModelReferenceManager {
 public:
  using SwitchedModelReferenceManager::footholdContext;
  using SwitchedModelReferenceManager::SwitchedModelReferenceManager;
};

/** A robot loaded for the reference manager. Not copyable: the model keeps a
 * reference to the settings. */
struct LoadedRobot {
  std::string referenceFile;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<test::QuaternionLayoutRobotModel> model;
};

absl::StatusOr<mpc_config::TaskFile> atlasTask() {
  ASSIGN_OR_RETURN(const std::string taskFile,
                   robot::resolveResourcePath(kTaskFile));
  return loadTaskFile(taskFile);
}

absl::StatusOr<LoadedRobot> loadRobot(const mpc_config::TaskFile& task) {
  LoadedRobot loaded;
  ASSIGN_OR_RETURN(loaded.referenceFile,
                   robot::resolveResourcePath(kReferenceFile));
  ASSIGN_OR_RETURN(const std::string urdfFile,
                   robot::resolveResourcePath(kUrdfFile));
  ASSIGN_OR_RETURN(ModelSettings settings,
                   ModelSettings::Create(task, urdfFile, "testPerLegFootholds",
                                         /*verbose=*/false));
  loaded.settings = std::make_unique<ModelSettings>(std::move(settings));
  ASSIGN_OR_RETURN(
      PinocchioInterface pinocchioInterface,
      loadCustomPinocchioInterface(task, urdfFile, *loaded.settings));
  loaded.pinocchioInterface =
      std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  // The whole-body layout: its knots carry the base velocity the command is
  // read from (getRefPlanarMotion()).
  loaded.model = std::make_unique<test::QuaternionLayoutRobotModel>(
      *loaded.settings, StateInputLayout::Mpc::kWholeBody);
  return loaded;
}

/** A 3-DoF contact on the Atlas sole of the ankle `ankle`. */
mpc_config::ModelSettingsConfig::PointContact soleCorner(
    absl::string_view name,
    absl::string_view ankle,
    scalar_t posContactRtLinkEwrtLinkX) {
  mpc_config::ModelSettingsConfig::PointContact contact;
  contact.name = std::string(name);
  contact.parent_joint_name = std::string(ankle);
  contact.pos_contact_rt_link_ewrt_link.x = posContactRtLinkEwrtLinkX;
  contact.pos_contact_rt_link_ewrt_link.z = -0.082;
  return contact;
}

mpc_config::ModelSettingsConfig::Leg leg(absl::string_view name,
                                         absl::string_view contact) {
  mpc_config::ModelSettingsConfig::Leg entry;
  entry.name = std::string(name);
  entry.contacts = {std::string(contact)};
  return entry;
}

/** The Atlas task on its four sole corners with the legs LF, RF, LH, RH and
 * the nominal_stance anchor. */
absl::StatusOr<mpc_config::TaskFile> soleCornerTask() {
  ASSIGN_OR_RETURN(mpc_config::TaskFile task, atlasTask());
  mpc_config::ModelSettingsConfig& settings = task.model_settings;
  settings.contacts_3dof = {
      soleCorner(kFrontLeft, "l_leg_akx", /*posContactRtLinkEwrtLinkX=*/0.16),
      soleCorner(kFrontRight, "r_leg_akx", /*posContactRtLinkEwrtLinkX=*/0.16),
      soleCorner(kHindLeft, "l_leg_akx", /*posContactRtLinkEwrtLinkX=*/-0.08),
      soleCorner(kHindRight, "r_leg_akx", /*posContactRtLinkEwrtLinkX=*/-0.08),
  };
  settings.contact_names_6dof = {};
  settings.contact_parent_joint_names = {};
  settings.legs = {leg("LF", kFrontLeft), leg("RF", kFrontRight),
                   leg("LH", kHindLeft), leg("RH", kHindRight)};
  task.nominal_foothold.anchor = "nominal_stance";
  return task;
}

ModeSchedule quadrupedTrot() {
  return ModeSchedule(
      /*eventTimesInput=*/{kFirstLiftOffTime, kDiagonalSwitchTime,
                           kLastTouchDownTime},
      /*modeSequenceInput=*/{0b1111, 0b1001, 0b0110, 0b1111});
}

/** The nominal standing state of `model`: the neutral posture, its base 0.9 m
 * up. */
vector_t nominalState(const test::QuaternionLayoutRobotModel& model) {
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.0, 0.0, 0.9));
  return state;
}

/** The measured state of the solve: the nominal posture moved to (1.0, 0.2)
 * and turned to kMeasYawBaseRtWorld. */
vector_t measuredState(const test::QuaternionLayoutRobotModel& model) {
  vector_t state = nominalState(model);
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(1.0, 0.2, 0.9));
  model.setEulerzyxBaseRtWorld(state, vector3_t(kMeasYawBaseRtWorld, 0.0, 0.0));
  return state;
}

/** A reference manager on `robot`, its swing planner for every contact. */
absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> makeManager(
    const LoadedRobot& robot) {
  ASSIGN_OR_RETURN(std::shared_ptr<GaitSchedule> gaitSchedule,
                   GaitSchedule::Create(robot.referenceFile, *robot.settings));
  return std::make_unique<ReferenceManagerPeer>(
      std::move(gaitSchedule),
      std::make_unique<SwingTrajectoryPlanner>(
          SwingTrajectoryPlanner::Config(),
          /*numFeet=*/robot.settings->contactNames.size()),
      *robot.pinocchioInterface, *robot.model);
}

/** Runs the one solve of a test: `schedule`, a target commanding
 * kCmdLinvelBaseRtWorldEwrtWorldX along the world x, the measured state. */
void solveOnce(const LoadedRobot& robot,
               const ModeSchedule& schedule,
               SwitchedModelReferenceManager& manager) {
  const test::QuaternionLayoutRobotModel& model = *robot.model;
  const vector_t state = measuredState(model);
  vector_t knot = state;
  model.setLinvelBaseRtWorldEwrtWorld(
      knot, vector3_t(kCmdLinvelBaseRtWorldEwrtWorldX, 0.0, 0.0));
  const vector_t zeroInput =
      vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  manager.getGaitSchedule()->updateModeSchedule(schedule);
  manager.setTargetTrajectories(TargetTrajectories(
      {-100.0, 100.0}, {knot, knot}, {zeroInput, zeroInput}));
  manager.preSolverRun(kSolveTime, /*finalTime=*/kSolveTime + 1.0, state,
                       schedule.modeAtTime(kSolveTime));
}

/** [m] Where each contact stands relative to the base at the nominal posture
 * of `robot`, in the heading frame (the nominal yaw is 0): computed here from
 * the forward kinematics. */
std::vector<vector2_t> expectedNominalStance(const LoadedRobot& robot) {
  PinocchioInterface pinocchioInterface = *robot.pinocchioInterface;
  const vector_t state = nominalState(*robot.model);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(
          robot.model->getGeneralizedCoordinates(state), pinocchioInterface,
          *robot.model);
  std::vector<vector2_t> offsets;
  for (const vector3_t& posContactRtWorldEwrtWorld :
       posContactsRtWorldEwrtWorld) {
    offsets.emplace_back(posContactRtWorldEwrtWorld.head<2>());
  }
  return offsets;
}

/** `offset` turned about the vertical by kMeasYawBaseRtWorld. */
vector2_t turned(const vector2_t& offset) {
  const scalar_t cosYaw = std::cos(kMeasYawBaseRtWorld);
  const scalar_t sinYaw = std::sin(kMeasYawBaseRtWorld);
  return {cosYaw * offset.x() - sinYaw * offset.y(),
          sinYaw * offset.x() + cosYaw * offset.y()};
}

/** [m] The base predicted at `time`: the measured (1.0, 0.2) carried forward by
 * the constant command. */
vector2_t predictedBase(scalar_t time) {
  return {1.0 + kCmdLinvelBaseRtWorldEwrtWorldX * (time - kSolveTime), 0.2};
}

/** The touch-down time of contact `contact` (LF, RF, LH, RH) in
 * quadrupedTrot(). */
scalar_t touchDownTime(size_t contact) {
  // RF (1) and LH (2) swing first.
  return (contact == 1 || contact == 2) ? kDiagonalSwitchTime
                                        : kLastTouchDownTime;
}

TEST(PerLegFootholds, theBipedKeepsItsSidesAndHips) {
  const absl::StatusOr<mpc_config::TaskFile> task = atlasTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> atlas = loadRobot(*task);
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> manager =
      makeManager(*atlas);
  ASSERT_TRUE(manager.ok()) << manager.status();
  const ModeSchedule walk(
      /*eventTimesInput=*/{0.1, 0.5, 0.6, 1.0},
      /*modeSequenceInput=*/{kBipedModeStance, kBipedModeRf, kBipedModeStance,
                             kBipedModeLf, kBipedModeStance});
  solveOnce(*atlas, walk, **manager);
  const FootholdHeuristicContext left =
      (*manager)->footholdContext(kContactLeftIndex, /*time=*/0.5);
  const FootholdHeuristicContext right =
      (*manager)->footholdContext(kContactRightIndex, /*time=*/1.0);
  EXPECT_EQ(left.side, 1.0);
  EXPECT_EQ(right.side, -1.0);
  EXPECT_EQ(left.legIndex, kContactLeftIndex);
  EXPECT_EQ(right.legIndex, kContactRightIndex);

  // The hips: the last joint before the floating base above each foot's
  // parent joint, at the nominal posture, turned into the heading frame - as
  // they were derived when they were a BipedFeetArray.
  const test::QuaternionLayoutRobotModel& model = *atlas->model;
  const vector_t state = nominalState(model);
  PinocchioInterface pinocchioInterface = *atlas->pinocchioInterface;
  const absl::StatusOr<LocomotionHeuristicModelParameters> derived =
      deriveLocomotionHeuristicModelParameters(pinocchioInterface, model,
                                               state);
  ASSERT_TRUE(derived.ok()) << derived.status();
  ASSERT_EQ(derived->posHipsRtBaseEwrtHeadingXy.size(), 2U);
  PinocchioInterface walked = *atlas->pinocchioInterface;
  computePosContactsRtWorldEwrtWorld<scalar_t>(
      model.getGeneralizedCoordinates(state), walked, model);
  const pinocchio::Model& pinocchioModel = walked.getModel();
  for (size_t foot = 0; foot < 2; ++foot) {
    const pinocchio::JointIndex hip = hipJointOf(
        pinocchioModel, pinocchioModel.getJointId(
                            atlas->settings->contactParentJointNames[foot]));
    const vector2_t expected =
        walked.getData().oMi[hip].translation().head<2>() -
        model.getPosBaseRtWorldEwrtWorld(state).head<2>();
    EXPECT_EQ(derived->posHipsRtBaseEwrtHeadingXy[foot], expected)
        << "foot " << foot;
  }
  EXPECT_GT(derived->posHipsRtBaseEwrtHeadingXy[kContactLeftIndex].y(), 0.0);
  EXPECT_LT(derived->posHipsRtBaseEwrtHeadingXy[kContactRightIndex].y(), 0.0);
}

TEST(PerLegFootholds, theBipedsGaitPhaseIsUnchanged) {
  const absl::StatusOr<mpc_config::TaskFile> task = atlasTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> atlas = loadRobot(*task);
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> manager =
      makeManager(*atlas);
  ASSERT_TRUE(manager.ok()) << manager.status();
  // Left stance (LF) [0.1, 0.5), double support, right stance (RF)
  // [0.6, 1.0).
  const ModeSchedule walk(
      /*eventTimesInput=*/{0.1, 0.5, 0.6, 1.0},
      /*modeSequenceInput=*/{kBipedModeStance, kBipedModeLf, kBipedModeStance,
                             kBipedModeRf, kBipedModeStance});
  solveOnce(*atlas, walk, **manager);
  const ReferenceManagerPeer& reference = **manager;
  EXPECT_EQ(reference.getPhaseVariable(0.3), 0.5 * (0.3 - 0.1) / (0.5 - 0.1));
  EXPECT_EQ(reference.getPhaseVariable(0.55), 0.5);
  EXPECT_EQ(reference.getPhaseVariable(0.8),
            0.5 + 0.5 * (0.8 - 0.6) / (1.0 - 0.6));
  EXPECT_EQ(reference.getPhaseVariableRate(0.3), 0.5 / (0.5 - 0.1));
  EXPECT_EQ(reference.getPhaseVariableRate(0.55), 0.0);
  // The right leg swings in left stance, the left leg in right stance.
  EXPECT_NEAR(reference.getSwingPhaseVariable(kContactRightIndex, /*time=*/0.3),
              0.5, kTolerance);
  EXPECT_EQ(reference.getSwingPhaseVariable(kContactLeftIndex, /*time=*/0.3),
            0.0);
  EXPECT_NEAR(reference.getSwingPhaseVariable(kContactLeftIndex, /*time=*/0.8),
              0.5, kTolerance);
  EXPECT_NEAR(
      reference.getSwingPhaseVariableRate(kContactLeftIndex, /*time=*/0.8),
      1.0 / (1.0 - 0.6), kTolerance);
}

TEST(PerLegFootholds, eachLegOfTheTrotHasItsSwingPhase) {
  const absl::StatusOr<mpc_config::TaskFile> task = soleCornerTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> corners = loadRobot(*task);
  ASSERT_TRUE(corners.ok()) << corners.status();
  ASSERT_EQ(corners->settings->legs.size(), kNumLegs);
  absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> manager =
      makeManager(*corners);
  ASSERT_TRUE(manager.ok()) << manager.status();
  solveOnce(*corners, quadrupedTrot(), **manager);
  const ReferenceManagerPeer& reference = **manager;
  for (size_t legIndex = 0; legIndex < kNumLegs; ++legIndex) {
    SCOPED_TRACE(legIndex);
    const scalar_t touchDown = touchDownTime(legIndex);
    const scalar_t liftOff = touchDown == kDiagonalSwitchTime
                                 ? kFirstLiftOffTime
                                 : kDiagonalSwitchTime;
    const scalar_t stanceTime =
        touchDown == kDiagonalSwitchTime
            ? 0.5 * (kDiagonalSwitchTime + kLastTouchDownTime)
            : 0.5 * (kFirstLiftOffTime + kDiagonalSwitchTime);
    EXPECT_EQ(reference.getSwingPhaseVariable(legIndex, liftOff), 0.0);
    EXPECT_NEAR(
        reference.getSwingPhaseVariable(legIndex, 0.5 * (liftOff + touchDown)),
        0.5, kTolerance);
    EXPECT_NEAR(reference.getSwingPhaseVariableRate(legIndex, liftOff + 0.01),
                1.0 / (touchDown - liftOff), kTolerance);
    EXPECT_NEAR(reference.getSwingPhaseVariableRate(legIndex, touchDown - 0.01),
                1.0 / (touchDown - liftOff), kTolerance);
    EXPECT_EQ(reference.getSwingPhaseVariable(legIndex, stanceTime), 0.0);
    EXPECT_EQ(reference.getSwingPhaseVariableRate(legIndex, stanceTime), 0.0);
  }
  EXPECT_EQ(reference.getSwingPhaseVariable(kNumLegs, /*time=*/0.35), 0.0)
      << "a leg the settings do not have";
  // The biped's gait phase has no biped mode to read on this set.
  EXPECT_EQ(reference.getPhaseVariable(0.35), 0.0);
}

TEST(PerLegFootholds, theNominalStanceLandsEachContactAtItsStance) {
  const absl::StatusOr<mpc_config::TaskFile> task = soleCornerTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> corners = loadRobot(*task);
  ASSERT_TRUE(corners.ok()) << corners.status();
  absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> manager =
      makeManager(*corners);
  ASSERT_TRUE(manager.ok()) << manager.status();
  ReferenceManagerPeer& reference = **manager;
  solveOnce(*corners, quadrupedTrot(), reference);
  EXPECT_FALSE(reference
                   .getNomPosFootRtWorldEwrtWorldXy(/*contactIndex=*/0,
                                                    kLastTouchDownTime)
                   .has_value())
      << "no nominal posture yet";

  reference.configureNominalStance(nominalState(*corners->model));
  solveOnce(*corners, quadrupedTrot(), reference);
  const std::vector<vector2_t> nominal = expectedNominalStance(*corners);
  ASSERT_EQ(nominal.size(), kNumLegs);
  // The left legs stand to the left of the base, the front ones ahead of the
  // hind ones.
  EXPECT_GT(nominal[0].y(), 0.0);
  EXPECT_LT(nominal[1].y(), 0.0);
  EXPECT_GT(nominal[2].y(), 0.0);
  EXPECT_LT(nominal[3].y(), 0.0);
  EXPECT_GT(nominal[0].x(), nominal[2].x());
  for (size_t contact = 0; contact < kNumLegs; ++contact) {
    SCOPED_TRACE(contact);
    const scalar_t touchDown = touchDownTime(contact);
    const std::optional<vector2_t> foothold =
        reference.getNomPosFootRtWorldEwrtWorldXy(contact, touchDown);
    if (!foothold.has_value()) {
      GTEST_FAIL() << "no foothold";
    }
    const vector2_t expected =
        predictedBase(touchDown) + turned(nominal[contact]);
    EXPECT_NEAR(foothold->x(), expected.x(), kTolerance);
    EXPECT_NEAR(foothold->y(), expected.y(), kTolerance);
    const FootholdHeuristicContext context =
        reference.footholdContext(contact, touchDown);
    EXPECT_EQ(context.legIndex, contact);
    EXPECT_EQ(context.side, corners->settings->legs[contact].lateralSign);
    EXPECT_NEAR(
        (context.nomPosContactRtBaseEwrtHeadingXy - nominal[contact]).norm(),
        0.0, kTolerance);
  }
  // A swing reference comes with the foothold.
  EXPECT_TRUE(reference
                  .getSwingFootReference(/*contactIndex=*/1,
                                         /*time=*/0.35)
                  .has_value());
}

/** The layer of the foothold heuristics `names` on `robot`, with
 * `translationalStepping` and the nominal_stance anchor. */
absl::StatusOr<std::shared_ptr<LocomotionHeuristicLayer>> footholdLayer(
    const LoadedRobot& robot,
    std::vector<std::string> names,
    const TranslationalSteppingParameters& translationalStepping) {
  LocomotionHeuristicConfig config;
  config.formulation.footholdHeuristics = std::move(names);
  config.translationalStepping = translationalStepping;
  PinocchioInterface pinocchioInterface = *robot.pinocchioInterface;
  ASSIGN_OR_RETURN(
      const LocomotionHeuristicModelParameters model,
      deriveLocomotionHeuristicModelParameters(pinocchioInterface, *robot.model,
                                               nominalState(*robot.model)));
  LocomotionHeuristicEnvironment environment;
  environment.nominalFootholdAnchor =
      ModelSettings::NominalFootholdAnchor::kNominalStance;
  ASSIGN_OR_RETURN(
      std::unique_ptr<LocomotionHeuristicLayer> layer,
      LocomotionHeuristicLayer::Create(config, model, environment));
  return std::shared_ptr<LocomotionHeuristicLayer>(std::move(layer));
}

TEST(PerLegFootholds, theLateralHeuristicTermsUseEachLegsSide) {
  const absl::StatusOr<mpc_config::TaskFile> task = soleCornerTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> corners = loadRobot(*task);
  ASSERT_TRUE(corners.ok()) << corners.status();
  absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> manager =
      makeManager(*corners);
  ASSERT_TRUE(manager.ok()) << manager.status();
  ReferenceManagerPeer& reference = **manager;
  reference.configureNominalStance(nominalState(*corners->model));
  // a0 = 0.05 m to each leg's own side, nothing else; with no hip-centered
  // stepping the anchor is the nominal stance point.
  constexpr scalar_t kOffsetPosFootRtWorldEwrtHeadingY = 0.05;
  TranslationalSteppingParameters widen;
  widen.offsetPosFootRtWorldEwrtHeadingY = kOffsetPosFootRtWorldEwrtHeadingY;
  absl::StatusOr<std::shared_ptr<LocomotionHeuristicLayer>> layer =
      footholdLayer(*corners, {"translational_stepping"}, widen);
  ASSERT_TRUE(layer.ok()) << layer.status();
  reference.setLocomotionHeuristicLayer(*layer);
  solveOnce(*corners, quadrupedTrot(), reference);
  const std::vector<vector2_t> nominal = expectedNominalStance(*corners);
  for (size_t contact = 0; contact < kNumLegs; ++contact) {
    SCOPED_TRACE(contact);
    const scalar_t touchDown = touchDownTime(contact);
    const std::optional<vector2_t> foothold =
        reference.getNomPosFootRtWorldEwrtWorldXy(contact, touchDown);
    if (!foothold.has_value()) {
      GTEST_FAIL() << "no foothold";
    }
    const scalar_t side = corners->settings->legs[contact].lateralSign;
    const vector2_t expected =
        predictedBase(touchDown) + turned(nominal[contact]) +
        turned(vector2_t(0.0, side * kOffsetPosFootRtWorldEwrtHeadingY));
    EXPECT_NEAR(foothold->x(), expected.x(), kTolerance);
    EXPECT_NEAR(foothold->y(), expected.y(), kTolerance);
  }
}

TEST(PerLegFootholds, hipCenteredSteppingPlacesEachLegUnderItsHip) {
  const absl::StatusOr<mpc_config::TaskFile> task = soleCornerTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> corners = loadRobot(*task);
  ASSERT_TRUE(corners.ok()) << corners.status();
  absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> manager =
      makeManager(*corners);
  ASSERT_TRUE(manager.ok()) << manager.status();
  ReferenceManagerPeer& reference = **manager;
  reference.configureNominalStance(nominalState(*corners->model));
  absl::StatusOr<std::shared_ptr<LocomotionHeuristicLayer>> layer =
      footholdLayer(*corners, {"hip_centered_stepping"},
                    TranslationalSteppingParameters());
  ASSERT_TRUE(layer.ok()) << layer.status();
  reference.setLocomotionHeuristicLayer(*layer);
  solveOnce(*corners, quadrupedTrot(), reference);

  PinocchioInterface pinocchioInterface = *corners->pinocchioInterface;
  const absl::StatusOr<LocomotionHeuristicModelParameters> model =
      deriveLocomotionHeuristicModelParameters(
          pinocchioInterface, *corners->model, nominalState(*corners->model));
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ(model->posHipsRtBaseEwrtHeadingXy.size(), kNumLegs);
  ASSERT_EQ(model->nomPosContactsRtBaseEwrtHeadingXy.size(), kNumLegs);
  // The toe and the heel of one sole share its hip.
  EXPECT_EQ(model->posHipsRtBaseEwrtHeadingXy[0],
            model->posHipsRtBaseEwrtHeadingXy[2]);
  EXPECT_EQ(model->posHipsRtBaseEwrtHeadingXy[1],
            model->posHipsRtBaseEwrtHeadingXy[3]);
  EXPECT_GT(model->posHipsRtBaseEwrtHeadingXy[0].y(), 0.0);
  EXPECT_LT(model->posHipsRtBaseEwrtHeadingXy[1].y(), 0.0);
  for (size_t contact = 0; contact < kNumLegs; ++contact) {
    SCOPED_TRACE(contact);
    const scalar_t touchDown = touchDownTime(contact);
    const std::optional<vector2_t> foothold =
        reference.getNomPosFootRtWorldEwrtWorldXy(contact, touchDown);
    if (!foothold.has_value()) {
      GTEST_FAIL() << "no foothold";
    }
    const vector2_t expected =
        predictedBase(touchDown) +
        turned(model->posHipsRtBaseEwrtHeadingXy[contact]);
    EXPECT_NEAR(foothold->x(), expected.x(), kTolerance);
    EXPECT_NEAR(foothold->y(), expected.y(), kTolerance);
  }
}

TEST(PerLegFootholds, hipCenteredSteppingUsesTheHipFrameALegNames) {
  absl::StatusOr<mpc_config::TaskFile> task = soleCornerTask();
  ASSERT_TRUE(task.ok()) << task.status();
  // The hip pitch joints of the front legs, 0.05 m ahead of the hip yaw joints
  // the tree walk derives; the hind legs keep the derived ones.
  constexpr char kFrontLeftHip[] = "l_leg_hpy";
  constexpr char kFrontRightHip[] = "r_leg_hpy";
  task->model_settings.legs[0].hip_frame_name = kFrontLeftHip;
  task->model_settings.legs[1].hip_frame_name = kFrontRightHip;
  absl::StatusOr<LoadedRobot> corners = loadRobot(*task);
  ASSERT_TRUE(corners.ok()) << corners.status();
  absl::StatusOr<std::unique_ptr<ReferenceManagerPeer>> manager =
      makeManager(*corners);
  ASSERT_TRUE(manager.ok()) << manager.status();
  ReferenceManagerPeer& reference = **manager;
  reference.configureNominalStance(nominalState(*corners->model));
  absl::StatusOr<std::shared_ptr<LocomotionHeuristicLayer>> layer =
      footholdLayer(*corners, {"hip_centered_stepping"},
                    TranslationalSteppingParameters());
  ASSERT_TRUE(layer.ok()) << layer.status();
  reference.setLocomotionHeuristicLayer(*layer);
  solveOnce(*corners, quadrupedTrot(), reference);

  // Each hip frame at the nominal posture, relative to the base (the nominal
  // yaw is 0, so the heading frame is the world's).
  const test::QuaternionLayoutRobotModel& model = *corners->model;
  const vector_t state = nominalState(model);
  PinocchioInterface walked = *corners->pinocchioInterface;
  computePosContactsRtWorldEwrtWorld<scalar_t>(
      model.getGeneralizedCoordinates(state), walked, model);
  const pinocchio::Model& pinocchioModel = walked.getModel();
  std::vector<vector2_t> posHipsRtBaseEwrtHeadingXy;
  for (const ModelSettings::Leg& entry : corners->settings->legs) {
    posHipsRtBaseEwrtHeadingXy.emplace_back(
        walked.getData()
            .oMf[pinocchioModel.getFrameId(entry.hipFrameName)]
            .translation()
            .head<2>() -
        model.getPosBaseRtWorldEwrtWorld(state).head<2>());
  }
  ASSERT_EQ(corners->settings->legs[0].hipFrameName, kFrontLeftHip);
  ASSERT_EQ(corners->settings->legs[1].hipFrameName, kFrontRightHip);
  EXPECT_GT(
      posHipsRtBaseEwrtHeadingXy[0].x() - posHipsRtBaseEwrtHeadingXy[2].x(),
      0.02)
      << "the named hip is not the derived one";

  PinocchioInterface pinocchioInterface = *corners->pinocchioInterface;
  const absl::StatusOr<LocomotionHeuristicModelParameters> derived =
      deriveLocomotionHeuristicModelParameters(pinocchioInterface, model,
                                               state);
  ASSERT_TRUE(derived.ok()) << derived.status();
  ASSERT_EQ(derived->posHipsRtBaseEwrtHeadingXy.size(), kNumLegs);
  for (size_t contact = 0; contact < kNumLegs; ++contact) {
    SCOPED_TRACE(contact);
    EXPECT_NEAR((derived->posHipsRtBaseEwrtHeadingXy[contact] -
                 posHipsRtBaseEwrtHeadingXy[contact])
                    .norm(),
                0.0, kTolerance);
    const scalar_t touchDown = touchDownTime(contact);
    const std::optional<vector2_t> foothold =
        reference.getNomPosFootRtWorldEwrtWorldXy(contact, touchDown);
    if (!foothold.has_value()) {
      GTEST_FAIL() << "no foothold";
    }
    const vector2_t expected =
        predictedBase(touchDown) + turned(posHipsRtBaseEwrtHeadingXy[contact]);
    EXPECT_NEAR(foothold->x(), expected.x(), kTolerance);
    EXPECT_NEAR(foothold->y(), expected.y(), kTolerance);
  }
}

TEST(PerLegFootholds, aHipFrameTheMpcModelLacksIsRefused) {
  const absl::StatusOr<mpc_config::TaskFile> task = soleCornerTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> corners = loadRobot(*task);
  ASSERT_TRUE(corners.ok()) << corners.status();
  corners->settings->legs[0].hipFrameName = "no_such_frame";
  PinocchioInterface pinocchioInterface = *corners->pinocchioInterface;
  const absl::StatusOr<LocomotionHeuristicModelParameters> derived =
      deriveLocomotionHeuristicModelParameters(
          pinocchioInterface, *corners->model, nominalState(*corners->model));
  ASSERT_FALSE(derived.ok());
  EXPECT_EQ(derived.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(absl::StrContains(derived.status().message(), "no_such_frame"))
      << derived.status();
}

TEST(PerLegFootholds, theAnchorIsSelectedByName) {
  EXPECT_EQ(nominalFootholdAnchorNames(),
            (std::vector<std::string>{"step_width", "nominal_stance"}));
  for (const std::string& name : nominalFootholdAnchorNames()) {
    const absl::StatusOr<ModelSettings::NominalFootholdAnchor> anchor =
        nominalFootholdAnchorFromName(name);
    ASSERT_TRUE(anchor.ok()) << anchor.status();
    EXPECT_EQ(nominalFootholdAnchorName(*anchor), name);
  }
  const absl::StatusOr<ModelSettings::NominalFootholdAnchor> unknown =
      nominalFootholdAnchorFromName("under_the_hip");
  ASSERT_FALSE(unknown.ok());
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknown.status().message(),
                                "nominal_foothold.anchor is 'under_the_hip'"))
      << unknown.status();
  EXPECT_TRUE(absl::StrContains(unknown.status().message(),
                                "step_width, nominal_stance"))
      << unknown.status();

  // The task file's default is step_width, and every shipped biped keeps it.
  const absl::StatusOr<mpc_config::TaskFile> task = atlasTask();
  ASSERT_TRUE(task.ok()) << task.status();
  absl::StatusOr<LoadedRobot> atlas = loadRobot(*task);
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  EXPECT_EQ(atlas->settings->nominalFootholdConfig.anchor,
            ModelSettings::NominalFootholdAnchor::kStepWidth);

  mpc_config::TaskFile misspelled = *task;
  misspelled.nominal_foothold.anchor = "under_the_hip";
  const absl::StatusOr<LoadedRobot> refused = loadRobot(misspelled);
  ASSERT_FALSE(refused.ok());
  EXPECT_TRUE(
      absl::StrContains(refused.status().message(), "nominal_foothold.anchor"))
      << refused.status();
}

TEST(PerLegFootholds, theNominalStanceNeedsLegs) {
  // The sole corners without legs: no leg to place.
  absl::StatusOr<mpc_config::TaskFile> task = soleCornerTask();
  ASSERT_TRUE(task.ok()) << task.status();
  task->model_settings.legs = {};
  const absl::StatusOr<LoadedRobot> refused = loadRobot(*task);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(),
                                "nominal_foothold.anchor 'nominal_stance'"))
      << refused.status();
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "has 4 contacts"))
      << refused.status();
  // The biped may opt in.
  absl::StatusOr<mpc_config::TaskFile> biped = atlasTask();
  ASSERT_TRUE(biped.ok()) << biped.status();
  biped->nominal_foothold.anchor = "nominal_stance";
  const absl::StatusOr<LoadedRobot> atlas = loadRobot(*biped);
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  EXPECT_EQ(atlas->settings->nominalFootholdConfig.anchor,
            ModelSettings::NominalFootholdAnchor::kNominalStance);
}

TEST(PerLegFootholds, theNominalStanceKeepsTheFeetApartForTheLayer) {
  LocomotionHeuristicConfig config;
  config.formulation.footholdHeuristics = {"translational_stepping"};
  LocomotionHeuristicEnvironment environment;
  // No step width and no hip_centered_stepping: refused on step_width.
  EXPECT_FALSE(LocomotionHeuristicLayer::Create(
                   config, LocomotionHeuristicModelParameters(), environment)
                   .ok());
  environment.nominalFootholdAnchor =
      ModelSettings::NominalFootholdAnchor::kNominalStance;
  EXPECT_TRUE(LocomotionHeuristicLayer::Create(
                  config, LocomotionHeuristicModelParameters(), environment)
                  .ok());
}

}  // namespace
}  // namespace ocs2::humanoid
