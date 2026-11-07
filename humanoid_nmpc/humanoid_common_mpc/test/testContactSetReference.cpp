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
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/data.hpp"  // IWYU pragma: keep
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/multibody/model.hpp"  // IWYU pragma: keep

#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/QuaternionLayoutRobotModel.h"
#include "robot_core/ResourcePaths.h"

/**
 * The swing trajectory planner and SwitchedModelReferenceManager::
 * getContactFlags() on contact sets other than the biped's: every contact of
 * the set gets its own contact flag and swing height from the mode number,
 * contact i at bit numContacts - 1 - i.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTolerance = 1.0e-12;

// A trot-like schedule of four contacts: all in contact, the diagonal 0 and 3
// in contact while 1 and 2 swing over [1.0, 1.5), all in contact, the other
// diagonal in contact while 0 and 3 swing over [2.0, 2.5), all in contact.
constexpr size_t kFourContacts = 4;
constexpr size_t kAllFourInContact = 0b1111;
constexpr size_t kDiagonalZeroThreeInContact = 0b1001;
constexpr size_t kDiagonalOneTwoInContact = 0b0110;
constexpr scalar_t kFirstLiftoffTime = 1.0;
constexpr scalar_t kFirstTouchdownTime = 1.5;
constexpr scalar_t kSecondLiftoffTime = 2.0;
constexpr scalar_t kSecondTouchdownTime = 2.5;
constexpr scalar_t kFirstMidSwingTime = 1.25;
constexpr scalar_t kSecondMidSwingTime = 2.25;

ModeSchedule trotSchedule() {
  return ModeSchedule(
      {kFirstLiftoffTime, kFirstTouchdownTime, kSecondLiftoffTime,
       kSecondTouchdownTime},
      {kAllFourInContact, kDiagonalZeroThreeInContact, kAllFourInContact,
       kDiagonalOneTwoInContact, kAllFourInContact});
}

/** Whether contact `contact` of the trot swings at `time`. */
bool swingsInTrot(size_t contact, scalar_t time) {
  const bool firstSwing = time > kFirstLiftoffTime &&
                          time < kFirstTouchdownTime &&
                          (contact == 1 || contact == 2);
  const bool secondSwing = time > kSecondLiftoffTime &&
                           time < kSecondTouchdownTime &&
                           (contact == 0 || contact == 3);
  return firstSwing || secondSwing;
}

/** Whether foot `contact` (2 left, 3 right) swings at `time` in the walk of the
 * reference manager test: the right foot over the first window (LF), the left
 * over the second (RF). */
bool swingsInWalk(size_t contact, scalar_t time) {
  const bool firstSwing =
      time > kFirstLiftoffTime && time < kFirstTouchdownTime && contact == 3;
  const bool secondSwing =
      time > kSecondLiftoffTime && time < kSecondTouchdownTime && contact == 2;
  return firstSwing || secondSwing;
}

TEST(SwingTrajectoryPlannerContactSet, plansTheSwingOfEachOfFourContacts) {
  const SwingTrajectoryPlanner::Config config;
  SwingTrajectoryPlanner planner(config, kFourContacts);
  ASSERT_EQ(planner.numFeet(), kFourContacts);
  planner.update(trotSchedule(), /*posSurfaceRtWorldEwrtWorldZ=*/0.0);

  // The swings last longer than swingTimeScale, so each reaches the full apex
  // at its middle.
  for (size_t contact = 0; contact < kFourContacts; ++contact) {
    for (const scalar_t time : {kFirstMidSwingTime, kSecondMidSwingTime}) {
      const scalar_t desPosContactRtWorldEwrtWorldZ =
          planner.getDesPosFootRtWorldEwrtWorldZ(contact, time);
      if (swingsInTrot(contact, time)) {
        EXPECT_NEAR(desPosContactRtWorldEwrtWorldZ,
                    config.apexPosFootRtSwingendEwrtWorldZ, kTolerance)
            << "contact " << contact << " at t = " << time;
        EXPECT_LT(planner.getImpactProximityFactor(contact, time), 1.0);
      } else {
        EXPECT_NEAR(desPosContactRtWorldEwrtWorldZ, 0.0, kTolerance)
            << "contact " << contact << " at t = " << time;
        EXPECT_NEAR(planner.getImpactProximityFactor(contact, time), 1.0,
                    kTolerance);
      }
    }
  }
}

TEST(SwingTrajectoryPlannerContactSet, plansTheSwingOfEachOfSixContacts) {
  // Six contacts, contact 0 the most significant bit: contacts 2 and 3 swing
  // over [1.0, 1.5) while the other four stay in contact.
  constexpr size_t kSixContacts = 6;
  constexpr size_t kAllSixInContact = 0b111111;
  constexpr size_t kTwoAndThreeSwing = 0b110011;
  const SwingTrajectoryPlanner::Config config;
  SwingTrajectoryPlanner planner(config, kSixContacts);
  planner.update(
      ModeSchedule({kFirstLiftoffTime, kFirstTouchdownTime},
                   {kAllSixInContact, kTwoAndThreeSwing, kAllSixInContact}),
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0);
  for (size_t contact = 0; contact < kSixContacts; ++contact) {
    const scalar_t expected = (contact == 2 || contact == 3)
                                  ? config.apexPosFootRtSwingendEwrtWorldZ
                                  : 0.0;
    EXPECT_NEAR(
        planner.getDesPosFootRtWorldEwrtWorldZ(contact, kFirstMidSwingTime),
        expected, kTolerance)
        << "contact " << contact;
  }
}

TEST(SwingTrajectoryPlannerContactSet, aSwingOutsideTheScheduleIsNotPlanned) {
  // Contacts 0 and 1 are in no phase; the last two walk. The first two read as
  // in stance rather than as a swing without a lift-off or a touch-down.
  const SwingTrajectoryPlanner::Config config;
  SwingTrajectoryPlanner planner(config, kFourContacts);
  planner.update(
      ModeSchedule({kFirstLiftoffTime, kFirstTouchdownTime},
                   {kBipedModeStance, kBipedModeLf, kBipedModeStance}),
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0);
  for (size_t contact = 0; contact < 2; ++contact) {
    EXPECT_NEAR(
        planner.getDesPosFootRtWorldEwrtWorldZ(contact, kFirstMidSwingTime),
        0.0, kTolerance);
    EXPECT_NEAR(planner.getImpactProximityFactor(contact, kFirstMidSwingTime),
                1.0, kTolerance);
  }
  // LF: contact 2 (the left foot) in contact, contact 3 swinging.
  EXPECT_NEAR(
      planner.getDesPosFootRtWorldEwrtWorldZ(/*leg=*/2, kFirstMidSwingTime),
      0.0, kTolerance);
  EXPECT_NEAR(
      planner.getDesPosFootRtWorldEwrtWorldZ(/*leg=*/3, kFirstMidSwingTime),
      config.apexPosFootRtSwingendEwrtWorldZ, kTolerance);

  // The hands placed for one phase only: out of contact before and after it,
  // swings without a lift-off or a touch-down in the schedule, so none is
  // planned either.
  planner.update(
      ModeSchedule({kFirstLiftoffTime, kFirstTouchdownTime},
                   {kBipedModeStance, kAllFourInContact, kBipedModeStance}),
      /*posSurfaceRtWorldEwrtWorldZ=*/0.0);
  for (size_t contact = 0; contact < kFourContacts; ++contact) {
    for (const scalar_t time : {0.5, kFirstMidSwingTime, 2.0}) {
      EXPECT_NEAR(planner.getDesPosFootRtWorldEwrtWorldZ(contact, time), 0.0,
                  kTolerance)
          << "contact " << contact << " at t = " << time;
    }
  }
}

TEST(SwingTrajectoryPlannerContactSet, aTwoFootPlannerSeesNoModeAsFlight) {
  // A biped planner reads a mode number past STANCE as both feet in swing, as
  // bipedFootContactsOfMode() does.
  const SwingTrajectoryPlanner::Config config;
  SwingTrajectoryPlanner planner(config, /*numFeet=*/2);
  planner.update(ModeSchedule({kFirstLiftoffTime, kFirstTouchdownTime},
                              {kBipedModeStance, kDiagonalZeroThreeInContact,
                               kBipedModeStance}),
                 /*posSurfaceRtWorldEwrtWorldZ=*/0.0);
  for (size_t foot = 0; foot < 2; ++foot) {
    EXPECT_NEAR(
        planner.getDesPosFootRtWorldEwrtWorldZ(foot, kFirstMidSwingTime),
        config.apexPosFootRtSwingendEwrtWorldZ, kTolerance);
  }
}

constexpr char kTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kReferenceFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/"
    "reference.textproto";
constexpr char kUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** A 3-DoF contact of the task file, 0.1 m to the side of a wrist. */
mpc_config::ModelSettingsConfig::PointContact handContact(
    absl::string_view name,
    absl::string_view wrist,
    scalar_t posContactRtLinkEwrtLinkY) {
  mpc_config::ModelSettingsConfig::PointContact contact;
  contact.name = std::string(name);
  contact.parent_joint_name = std::string(wrist);
  contact.pos_contact_rt_link_ewrt_link.y = posContactRtLinkEwrtLinkY;
  return contact;
}

/** The shipped DRC Atlas with a 3-DoF contact on each hand: the contact set
 * {left hand, right hand, left foot, right foot}. */
struct FourContactAtlas {
  std::string referenceFile;
  std::unique_ptr<ModelSettings> settings;
  std::unique_ptr<PinocchioInterface> pinocchioInterface;
  std::unique_ptr<test::QuaternionLayoutRobotModel> model;
};

absl::StatusOr<FourContactAtlas> loadFourContactAtlas() {
  FourContactAtlas loaded;
  ASSIGN_OR_RETURN(const std::string taskFile,
                   robot::resolveResourcePath(kTaskFile));
  ASSIGN_OR_RETURN(loaded.referenceFile,
                   robot::resolveResourcePath(kReferenceFile));
  ASSIGN_OR_RETURN(const std::string urdfFile,
                   robot::resolveResourcePath(kUrdfFile));
  ASSIGN_OR_RETURN(mpc_config::TaskFile task, loadTaskFile(taskFile));
  task.model_settings.contacts_3dof = {
      handContact("hand_l_contact", "l_arm_wrx",
                  /*posContactRtLinkEwrtLinkY=*/0.1),
      handContact("hand_r_contact", "r_arm_wrx",
                  /*posContactRtLinkEwrtLinkY=*/-0.1)};
  ASSIGN_OR_RETURN(
      ModelSettings settings,
      ModelSettings::Create(task, urdfFile, "testContactSetReference",
                            /*verbose=*/false));
  loaded.settings = std::make_unique<ModelSettings>(std::move(settings));
  ASSIGN_OR_RETURN(
      PinocchioInterface pinocchioInterface,
      loadCustomPinocchioInterface(task, urdfFile, *loaded.settings));
  loaded.pinocchioInterface =
      std::make_unique<PinocchioInterface>(std::move(pinocchioInterface));
  loaded.model = std::make_unique<test::QuaternionLayoutRobotModel>(
      *loaded.settings, StateInputLayout::Mpc::kCentroidal);
  return loaded;
}

TEST(ContactPositionsContactSet, coverEveryContactInContactOrder) {
  absl::StatusOr<FourContactAtlas> atlas = loadFourContactAtlas();
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  const test::QuaternionLayoutRobotModel& model = *atlas->model;
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.0, 0.0, 0.9));
  const vector_t q = model.getGeneralizedCoordinates(state);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(
          q, *atlas->pinocchioInterface, model);
  ASSERT_EQ(posContactsRtWorldEwrtWorld.size(), kFourContacts);
  const pinocchio::Model& pinocchioModel =
      atlas->pinocchioInterface->getModel();
  const pinocchio::Data& data = atlas->pinocchioInterface->getData();
  for (size_t contact = 0; contact < kFourContacts; ++contact) {
    const std::string& name = atlas->settings->contactNames[contact];
    EXPECT_TRUE(posContactsRtWorldEwrtWorld[contact].isApprox(
        data.oMf[pinocchioModel.getFrameId(name)].translation()))
        << name;
  }
}

TEST(SwitchedModelReferenceManagerContactSet, readsEveryContactOfTheMode) {
  absl::StatusOr<FourContactAtlas> atlas = loadFourContactAtlas();
  ASSERT_TRUE(atlas.ok()) << atlas.status();
  ASSERT_EQ(atlas->settings->numContacts(), kFourContacts);
  ASSERT_FALSE(atlas->settings->isBipedContactSet());

  absl::StatusOr<std::shared_ptr<GaitSchedule>> gaitSchedule =
      GaitSchedule::Create(atlas->referenceFile, *atlas->settings);
  ASSERT_TRUE(gaitSchedule.ok()) << gaitSchedule.status();
  SwitchedModelReferenceManager referenceManager(
      *std::move(gaitSchedule),
      std::make_unique<SwingTrajectoryPlanner>(SwingTrajectoryPlanner::Config(),
                                               atlas->settings->numContacts()),
      *atlas->pinocchioInterface, *atlas->model);

  const test::QuaternionLayoutRobotModel& model = *atlas->model;
  vector_t state = model.getNeutralState();
  model.setPosBaseRtWorldEwrtWorld(state, vector3_t(0.0, 0.0, 0.9));
  const vector_t zeroInput =
      vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
  const TargetTrajectories target({-100.0, 100.0}, {state, state},
                                  {zeroInput, zeroInput});
  // The gait files' modes on the four contacts: the feet are the last two, so
  // STANCE, LF and RF keep their biped numbers and the hands are in contact in
  // none of them.
  const ModeSchedule schedule({kFirstLiftoffTime, kFirstTouchdownTime,
                               kSecondLiftoffTime, kSecondTouchdownTime},
                              {kBipedModeStance, kBipedModeLf, kBipedModeStance,
                               kBipedModeRf, kBipedModeStance});
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  referenceManager.setTargetTrajectories(target);
  referenceManager.preSolverRun(/*initTime=*/0.0, /*finalTime=*/3.0, state,
                                schedule.modeAtTime(0.0));

  const scalar_t weight = computeRobotWeight(*atlas->pinocchioInterface);
  const SwingTrajectoryPlanner& planner =
      *referenceManager.getSwingTrajectoryPlanner();
  for (const scalar_t time : {0.5, kFirstMidSwingTime, kSecondMidSwingTime}) {
    const ContactMask flags = referenceManager.getContactFlags(time);
    ASSERT_EQ(flags.size(), kFourContacts);
    EXPECT_EQ(flags, ContactMask::fromModeNumber(schedule.modeAtTime(time),
                                                 kFourContacts));
    const vector_t desiredInput =
        referenceManager.getDesiredInput(target, state, time);
    for (size_t contact = 0; contact < kFourContacts; ++contact) {
      const bool isHand = contact < 2;
      const bool swings = !isHand && swingsInWalk(contact, time);
      EXPECT_EQ(flags[contact], !isHand && !swings)
          << "contact " << contact << " at t = " << time;
      EXPECT_EQ(referenceManager.isInContact(time, contact), flags[contact]);
      const scalar_t desPosContactRtWorldEwrtWorldZ =
          planner.getDesPosFootRtWorldEwrtWorldZ(contact, time);
      if (swings) {
        EXPECT_GT(desPosContactRtWorldEwrtWorldZ, 0.0) << "contact " << contact;
      } else {
        EXPECT_NEAR(desPosContactRtWorldEwrtWorldZ, 0.0, kTolerance)
            << "contact " << contact;
      }
      // The weight is shared by the contacts in contact.
      const scalar_t expectedForceContactEwrtParamZ =
          flags[contact] ? weight / static_cast<scalar_t>(flags.numActive())
                         : 0.0;
      EXPECT_NEAR(model.getForceContactEwrtParam(desiredInput, contact).z(),
                  expectedForceContactEwrtParamZ, 1.0e-9 * weight)
          << "contact " << contact << " at t = " << time;
      // The nominal step places a left and a right foot: no contact of this
      // set has a foothold or a swing reference from it.
      EXPECT_FALSE(
          referenceManager.getNomPosFootRtWorldEwrtWorldXy(contact, time)
              .has_value());
      EXPECT_FALSE(
          referenceManager.getSwingFootReference(contact, time).has_value());
    }
  }
  // STANCE has both feet in contact but not the hands.
  EXPECT_FALSE(referenceManager.isInStancePhase(0.5));
}

}  // namespace
}  // namespace ocs2::humanoid
