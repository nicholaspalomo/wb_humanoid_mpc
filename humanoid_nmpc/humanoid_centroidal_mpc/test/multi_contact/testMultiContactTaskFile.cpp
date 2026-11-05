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
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/spatial/fwd.hpp"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/MultiContactTestFiles.h"
#include "support/TypedConfigFiles.h"

/*
 * The DRC Atlas's multi-contact task file
 * (robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/
 * task.textproto) against the contact set beside it, the one description of
 * the candidates the plans are made for: every candidate is a contact of the
 * MPC model, of its kind, on its parent joint, at its offset, with its pad
 * frame and rectangle; the MPC model's contact frames are the stability
 * model's pads at any configuration; the feet are the last two contacts, so
 * the biped's mode numbers keep their meaning; and the wrists are unlocked.
 */
namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTolerance = 1.0e-9;
/** [rad] The range of the random joint positions of the frame check. */
constexpr scalar_t kJointRange = 0.6;

/** The model settings of the multi-contact task file. */
struct TaskUnderTest {
  CentroidalRobotFiles files;
  mpc_config::TaskFile task;
};

TaskUnderTest loadTask() {
  TaskUnderTest loaded{.files = atlasMultiContactFiles(), .task = {}};
  loaded.task = checkedValue(loadTaskFile(loaded.files.taskFile));
  return loaded;
}

ModelSettings modelSettingsOf(const TaskUnderTest& loaded) {
  return checkedValue(ModelSettings::Create(loaded.task, loaded.files.urdfFile,
                                            "multi_contact_task_file_",
                                            /*verbose=*/false));
}

/** The index of `name` among `settings`' contacts; fails the test without one.
 */
size_t contactIndexOf(const ModelSettings& settings, const std::string& name) {
  const std::vector<std::string>& names = settings.contactNames;
  const std::vector<std::string>::const_iterator found =
      std::find(names.begin(), names.end(), name);
  EXPECT_NE(found, names.end()) << name << " is not a contact of the task";
  return static_cast<size_t>(found - names.begin());
}

TEST(MultiContactTaskFile, everyCandidateIsAContactWithItsGeometry) {
  const TaskUnderTest loaded = loadTask();
  const ModelSettings settings = modelSettingsOf(loaded);
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactSet contactSet,
                                loadAtlasContactSet());
  ASSERT_EQ(settings.contactNames.size(),
            static_cast<size_t>(contactSet.size()));
  for (const multi_contact::ContactCandidate& candidate :
       contactSet.candidates()) {
    SCOPED_TRACE(candidate.name);
    const size_t i = contactIndexOf(settings, candidate.name);
    ASSERT_LT(i, settings.contactNames.size());
    const bool point = candidate.kind == multi_contact::ContactKind::kPoint;
    ASSERT_EQ(settings.isPointContact(i), point);
    if (point) {
      const ModelSettings::PointContact& contact = settings.pointContacts[i];
      EXPECT_EQ(contact.parentJointName, candidate.parentJoint);
      EXPECT_TRUE(contact.posContactRtLinkEwrtLink.isApprox(
          candidate.posPadRtLinkEwrtLink, kTolerance))
          << contact.posContactRtLinkEwrtLink.transpose();
      continue;
    }
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const ContactRectangle rectangle,
        contactRectangleFromConfig(loaded.task.contacts, settings,
                                   static_cast<int>(i)));
    const ContactCenterPoint& center = rectangle.getContactCenterPoint();
    EXPECT_EQ(center.parentJointName, candidate.parentJoint);
    EXPECT_TRUE(center.posContactRtLinkEwrtLink.isApprox(
        candidate.posPadRtLinkEwrtLink, kTolerance))
        << center.posContactRtLinkEwrtLink.transpose();
    EXPECT_TRUE(center.rotmatContactRtLink.isApprox(candidate.rotmatPadRtLink,
                                                    kTolerance))
        << center.rotmatContactRtLink;
    if (!candidate.rectangle.has_value()) GTEST_FAIL();
    EXPECT_DOUBLE_EQ(
        rectangle.getBounds().min_pos_corners_rt_contact_ewrt_contact_x,
        candidate.rectangle->minPosCornersRtPadEwrtPadX);
    EXPECT_DOUBLE_EQ(
        rectangle.getBounds().max_pos_corners_rt_contact_ewrt_contact_x,
        candidate.rectangle->maxPosCornersRtPadEwrtPadX);
    EXPECT_DOUBLE_EQ(
        rectangle.getBounds().min_pos_corners_rt_contact_ewrt_contact_y,
        candidate.rectangle->minPosCornersRtPadEwrtPadY);
    EXPECT_DOUBLE_EQ(
        rectangle.getBounds().max_pos_corners_rt_contact_ewrt_contact_y,
        candidate.rectangle->maxPosCornersRtPadEwrtPadY);
  }
}

TEST(MultiContactTaskFile, theFeetAreLastSoTheBipedModesKeepTheirMeaning) {
  const ModelSettings settings = modelSettingsOf(loadTask());
  const size_t n = settings.contactNames.size();
  ASSERT_GT(n, static_cast<size_t>(kBipedFootCount));
  EXPECT_EQ(settings.contactNames[n - kBipedFootCount], "foot_l_contact");
  EXPECT_EQ(settings.contactNames[n - 1], "foot_r_contact");
  EXPECT_TRUE(contactIsActiveInMode(ModeNumber::kLf, n - kBipedFootCount, n));
  EXPECT_FALSE(contactIsActiveInMode(ModeNumber::kLf, n - 1, n));
  EXPECT_EQ(numModeNumbersOf(n), size_t{1} << n);
}

TEST(MultiContactTaskFile, theWristsAreUnlocked) {
  const ModelSettings settings = modelSettingsOf(loadTask());
  EXPECT_TRUE(settings.fixedJointNames.empty());
  const std::vector<std::string> wrists = {"l_arm_wry", "l_arm_wrx",
                                           "r_arm_wry", "r_arm_wrx"};
  for (const std::string& wrist : wrists) {
    EXPECT_NE(std::find(settings.mpcModelJointNames.begin(),
                        settings.mpcModelJointNames.end(), wrist),
              settings.mpcModelJointNames.end())
        << wrist;
  }
}

TEST(MultiContactTaskFile, theModelsContactFramesAreTheStanceModelsPads) {
  const TaskUnderTest loaded = loadTask();
  const ModelSettings settings = modelSettingsOf(loaded);
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactSet contactSet,
                                loadAtlasContactSet());
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      PinocchioInterface mpc,
      loadCustomPinocchioInterface(loaded.task, loaded.files.urdfFile, settings,
                                   /*scaleTotalMass=*/false));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<multi_contact::RobotStanceModel> stance,
      multi_contact::RobotStanceModel::Create(loaded.files.urdfFile,
                                              contactSet));
  // A pose of the base and of every joint away from zero.
  const Eigen::Quaterniond quatBaseRtWorld(
      Eigen::AngleAxisd(0.7, Eigen::Vector3d(0.3, -1.0, 0.4).normalized()));
  const vector3_t posBaseRtWorldEwrtWorld(0.2, -0.3, 0.9);
  multi_contact::RobotConfiguration configuration;
  configuration.posBaseRtWorldEwrtWorld = posBaseRtWorldEwrtWorld;
  configuration.quatBaseRtWorld = quatBaseRtWorld;
  configuration.jointNames = settings.mpcModelJointNames;
  configuration.jointPositions.resize(
      static_cast<Eigen::Index>(settings.mpcModelJointNames.size()));
  for (Eigen::Index j = 0; j < configuration.jointPositions.size(); ++j) {
    configuration.jointPositions(j) =
        kJointRange * std::sin(1.3 * static_cast<scalar_t>(j) + 0.4);
  }
  HUMANOID_ASSERT_OK(stance->setConfiguration(configuration));
  // The MPC model's q = [p_W, xi (x, y, z, w), q_j] in its joint order.
  vector_t q(mpc.getModel().nq);
  q << posBaseRtWorldEwrtWorld, quatBaseRtWorld.coeffs(),
      configuration.jointPositions;
  pinocchio::forwardKinematics(mpc.getModel(), mpc.getData(), q);
  pinocchio::updateFramePlacements(mpc.getModel(), mpc.getData());
  for (int k = 0; k < contactSet.size(); ++k) {
    const multi_contact::ContactCandidate& candidate = contactSet.candidate(k);
    SCOPED_TRACE(candidate.name);
    const pinocchio::SE3& posePadRtWorld =
        mpc.getData().oMf[mpc.getModel().getFrameId(candidate.name)];
    EXPECT_TRUE(posePadRtWorld.translation().isApprox(
        stance->getPosPadRtWorldEwrtWorld(k), kTolerance))
        << posePadRtWorld.translation().transpose() << " against "
        << stance->getPosPadRtWorldEwrtWorld(k).transpose();
    if (candidate.kind == multi_contact::ContactKind::kSurface) {
      EXPECT_TRUE(posePadRtWorld.rotation().isApprox(
          stance->getRotmatPadRtWorld(k), kTolerance))
          << posePadRtWorld.rotation() << "\nagainst\n"
          << stance->getRotmatPadRtWorld(k);
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid
