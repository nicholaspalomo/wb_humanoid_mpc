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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/config/weights/StateInputLayout.h"
#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_common_mpc/gait/GaitSchedule.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "legged_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "legged_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "legged_mpc_config/task_file.nproto.h"
#include "legged_nmpc/legged_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The nominal step (SwitchedModelReferenceManager::
 * getNomPosFootRtWorldEwrtWorldXy() with no foothold heuristic listed) places a
 * foot one step width across from the stance foot, carried forward over one
 * step. For a later step in the horizon the stance foot has not landed yet at
 * the last solve, so the position latched for it then is where it lifted off,
 * not where it will stand: the step must start from the stance foot's own
 * nominal target at its touch-down. The same holds for the start of a swing
 * whose previous touch-down lies after the last solve.
 */
namespace ocs2::legged {
namespace {

constexpr char kTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

constexpr scalar_t kCmdLinvelBaseRtWorldEwrtWorldX = 0.4;  // [m/s]
constexpr scalar_t kLatchTime = 0.0;                       // [s]
// The schedule: the left foot swings [-0.05, 0.35) (mode RF), double support,
// the right foot swings [0.45, 0.8) (mode LF), double support, the left foot
// swings again [0.9, 1.25), and stance after.
constexpr scalar_t kLfootLiftoffTime = -0.05;
constexpr scalar_t kLfootTouchdownTime = 0.35;
constexpr scalar_t kRfootLiftoffTime = 0.45;
constexpr scalar_t kRfootTouchdownTime = 0.8;
constexpr scalar_t kSecondLfootLiftoffTime = 0.9;
constexpr scalar_t kSecondLfootTouchdownTime = 1.25;

/** The shipped DRC Atlas loaded for the reference manager. */
struct LoadedAtlas {
  std::string referenceFile;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<test::QuaternionLayoutRobotModel> model;
};

absl::StatusOr<LoadedAtlas> loadAtlas() {
  LoadedAtlas loaded;
  ASSIGN_OR_RETURN(const std::string taskFile,
                   robot::resolveResourcePath(kTaskFile));
  ASSIGN_OR_RETURN(loaded.referenceFile,
                   robot::resolveResourcePath(kReferenceFile));
  ASSIGN_OR_RETURN(const std::string urdfFile,
                   robot::resolveResourcePath(kUrdfFile));
  ASSIGN_OR_RETURN(const mpc_config::TaskFile task, loadTaskFile(taskFile));
  ASSIGN_OR_RETURN(
      ModelSettings settings,
      ModelSettings::Create(task, urdfFile, "testNominalStepAnchor",
                            /*verbose=*/false));
  loaded.settings = std::make_unique<ModelSettings>(std::move(settings));
  ASSIGN_OR_RETURN(
      PinocchioInterface pinocchioInterface,
      loadCustomPinocchioInterface(task, urdfFile, *loaded.settings));
  loaded.pinocchioInterface =
      std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  // The whole-body layout: its knots carry the base velocity the commanded
  // velocity is read from (getRefPlanarMotion()).
  loaded.model = std::make_unique<test::QuaternionLayoutRobotModel>(
      *loaded.settings, StateInputLayout::Mpc::kWholeBody);
  return loaded;
}

ModeSchedule stepSchedule() {
  const std::vector<scalar_t> eventTimes = {
      kLfootLiftoffTime,   kLfootTouchdownTime,     kRfootLiftoffTime,
      kRfootTouchdownTime, kSecondLfootLiftoffTime, kSecondLfootTouchdownTime};
  const std::vector<size_t> modeSequence = {
      kBipedModeStance, kBipedModeRf, kBipedModeStance, kBipedModeLf,
      kBipedModeStance, kBipedModeRf, kBipedModeStance};
  return ModeSchedule(eventTimes, modeSequence);
}

/** The cubic swing blend -tau^3 + tau^2 + tau of the swing [liftOff,
 * touchDown) at `time`. */
scalar_t swingBlend(scalar_t liftOff, scalar_t touchDown, scalar_t time) {
  const scalar_t tau = (time - liftOff) / (touchDown - liftOff);
  return -tau * tau * tau + tau * tau + tau;
}

TEST(NominalStepAnchor, LaterStepsStartFromTheStanceFootsNominalTarget) {
  absl::StatusOr<LoadedAtlas> atlas = loadAtlas();
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  const scalar_t nomPosLfootRtRfootEwrtHeadingY =
      atlas->settings->nominalFootholdConfig.nomPosLfootRtRfootEwrtHeadingY;
  ASSERT_GT(nomPosLfootRtRfootEwrtHeadingY, 0.0)
      << "the nominal step needs a positive step width";

  absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule =
      GaitSchedule::Create(atlas->referenceFile, *atlas->settings);
  ASSERT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
  SwitchedModelReferenceManager referenceManager(
      *std::move(gaitSchedule),
      std::make_unique<SwingTrajectoryPlanner>(
          SwingTrajectoryPlanner::Config(),
          /*numFeet=*/static_cast<size_t>(kBipedFootCount)),
      *atlas->pinocchioInterface, *atlas->model);
  // The default heuristic layer lists none: the nominal step.

  const test::QuaternionLayoutRobotModel& model = *atlas->model;
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(1.0, 0.2, 0.9));
  vector_t knot = state;
  model.setLinvelBaseRtWorldEwrtWorld(
      knot, vector3_t(kCmdLinvelBaseRtWorldEwrtWorldX, 0.0, 0.0));
  const vector_t zeroInput =
      vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));

  const ModeSchedule schedule = stepSchedule();
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  referenceManager.setTargetTrajectories(TargetTrajectories(
      {-100.0, 100.0}, {knot, knot}, {zeroInput, zeroInput}));
  // The only solve, with the left foot in flight: its latched position is
  // never measured, and the right foot's is where it stands now.
  referenceManager.preSolverRun(kLatchTime, /*finalTime=*/kLatchTime + 1.0,
                                state, schedule.modeAtTime(kLatchTime));

  const std::optional<vector2_t> desPosLfootRtWorldEwrtWorldXy =
      referenceManager.getNomPosFootRtWorldEwrtWorldXy(kContactLeftIndex,
                                                       kLfootTouchdownTime);
  const std::optional<vector2_t> desPosRfootRtWorldEwrtWorldXy =
      referenceManager.getNomPosFootRtWorldEwrtWorldXy(kContactRightIndex,
                                                       kRfootTouchdownTime);
  const std::optional<vector2_t> secondDesPosLfootRtWorldEwrtWorldXy =
      referenceManager.getNomPosFootRtWorldEwrtWorldXy(
          kContactLeftIndex, kSecondLfootTouchdownTime);
  if (!desPosLfootRtWorldEwrtWorldXy.has_value()) {
    GTEST_FAIL() << "no desPosLfootRtWorldEwrtWorldXy";
  }
  if (!desPosRfootRtWorldEwrtWorldXy.has_value()) {
    GTEST_FAIL() << "no desPosRfootRtWorldEwrtWorldXy";
  }
  if (!secondDesPosLfootRtWorldEwrtWorldXy.has_value()) {
    GTEST_FAIL() << "no secondDesPosLfootRtWorldEwrtWorldXy";
  }

  // The right foot's step starts where the left foot lands, one step of travel
  // ahead and one step width to the right.
  EXPECT_NEAR(desPosRfootRtWorldEwrtWorldXy->x(),
              desPosLfootRtWorldEwrtWorldXy->x() +
                  kCmdLinvelBaseRtWorldEwrtWorldX *
                      (kRfootTouchdownTime - kLfootTouchdownTime),
              1.0e-9);
  EXPECT_NEAR(
      desPosRfootRtWorldEwrtWorldXy->y(),
      desPosLfootRtWorldEwrtWorldXy->y() - nomPosLfootRtRfootEwrtHeadingY,
      1.0e-9);
  // And the left foot's second step from where the right foot lands.
  EXPECT_NEAR(secondDesPosLfootRtWorldEwrtWorldXy->x(),
              desPosRfootRtWorldEwrtWorldXy->x() +
                  kCmdLinvelBaseRtWorldEwrtWorldX *
                      (kSecondLfootTouchdownTime - kRfootTouchdownTime),
              1.0e-9);
  EXPECT_NEAR(
      secondDesPosLfootRtWorldEwrtWorldXy->y(),
      desPosRfootRtWorldEwrtWorldXy->y() + nomPosLfootRtRfootEwrtHeadingY,
      1.0e-9);

  // The right foot has not touched down since the solve, so its swing starts
  // from where it was measured standing.
  const vector_t q = model.getGeneralizedCoordinates(state);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(
          q, *atlas->pinocchioInterface, model);
  const vector2_t liftoffPosRfootRtWorldEwrtWorldXy =
      posContactsRtWorldEwrtWorld[kContactRightIndex].head<2>();
  const scalar_t rfootSampleTime = 0.6;
  const std::optional<SwingFootReference> rfootSwing =
      referenceManager.getSwingFootReference(kContactRightIndex,
                                             rfootSampleTime);
  if (!rfootSwing.has_value()) {
    GTEST_FAIL() << "no rfootSwing";
  }
  EXPECT_NEAR(
      rfootSwing->desPosFootRtWorldEwrtWorld.x(),
      liftoffPosRfootRtWorldEwrtWorldXy.x() +
          swingBlend(kRfootLiftoffTime, kRfootTouchdownTime, rfootSampleTime) *
              (desPosRfootRtWorldEwrtWorldXy->x() -
               liftoffPosRfootRtWorldEwrtWorldXy.x()),
      1.0e-9);

  // The left foot touches down at 0.35, after the solve: its second swing
  // starts from that nominal touch-down, not from a position never measured.
  const scalar_t lfootSampleTime = 1.0;
  const std::optional<SwingFootReference> lfootSwing =
      referenceManager.getSwingFootReference(kContactLeftIndex,
                                             lfootSampleTime);
  if (!lfootSwing.has_value()) {
    GTEST_FAIL() << "no lfootSwing";
  }
  const scalar_t lfootBlend = swingBlend(
      kSecondLfootLiftoffTime, kSecondLfootTouchdownTime, lfootSampleTime);
  for (Eigen::Index axis = 0; axis < 2; ++axis) {
    EXPECT_NEAR(lfootSwing->desPosFootRtWorldEwrtWorld(axis),
                (*desPosLfootRtWorldEwrtWorldXy)(axis) +
                    lfootBlend * ((*secondDesPosLfootRtWorldEwrtWorldXy)(axis) -
                                  (*desPosLfootRtWorldEwrtWorldXy)(axis)),
                1.0e-9)
        << "axis " << axis;
  }
}

}  // namespace
}  // namespace ocs2::legged
