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

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_core/ResourcePaths.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

/*
 * The joint PD gains of WBMpcMrtJointController, set from other threads (see testMrtJointControllerPdGains for the
 * centroidal controller): resolved on the caller's thread, taken into use by the next control cycle without an allocation
 * of its own, refused gains refused without changing the gains, and the gains file (a joint_pd_gains.textproto) watched
 * by pollPdGainsFile() rather than by every 500th control cycle. This binary links the allocation counter, which
 * interposes malloc for the whole process; the solver thread is never started.
 */

namespace ocs2::humanoid {
namespace {

using ::ocs2::humanoid::estimation::heapAllocationCount;

constexpr scalar_t kControlPeriod = 0.01;  // [s]

/** The action of joint `index`, which every joint of the description holds: a test failure and a zero action if not. */
const robot::model::JointAction& actionOf(const robot::model::RobotJointAction& action, size_t index) {
  const std::optional<robot::model::JointAction>& entry = action.at(index);
  if (!entry.has_value()) {
    ADD_FAILURE() << "no action for joint " << index;
    static const robot::model::JointAction kNoAction;
    return kNoAction;
  }
  return *entry;
}

class WholeBodyPdGainsHarness {
 public:
  explicit WholeBodyPdGainsHarness(const std::string& pdGainsFile = "")
      // A file missing from the runfiles throws here, naming the data dependency to add (robot::resolveResourcePath).
      : task_(loadTaskFile(robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto").value()).value()),
        urdfFile_(robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf").value()),
        modelSettings_(ModelSettings::Create(task_, urdfFile_, "wb_mpc_", /*verbose=*/false).value()),
        pinocchioInterface_(loadCustomPinocchioInterface(task_, urdfFile_, modelSettings_).value()),
        model_(modelSettings_),
        mpc_(toMpcSettings(task_.mpc), model_.getInputDim()),
        description_(robot::model::RobotDescription::Create(urdfFile_).value()),
        robotState_(description_),
        action_(description_) {
    const vector_t initialState =
        stateValuesFromConfig(task_.initial_state, stateInputLayout(modelSettings_, StateInputLayout::Mpc::kWholeBody), "initial_state")
            .value();
    mpcJointIndices_ = description_.getJointIndices(modelSettings_.mpcModelJointNames);
    otherJointIndices_ = description_.getJointIndices(modelSettings_.fixedJointNames);
    robotState_.setRootPositionInWorldFrame(model_.getBasePosition(initialState));
    robotState_.setRootRotationLocalToWorldFrame(
        getQuaternionFromEulerAnglesZyx(vector3_t(model_.getBaseOrientationEulerZYX(initialState))));
    const vector_t joints = model_.getJointAngles(initialState);
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) robotState_.setJointPosition(mpcJointIndices_[i], joints[i]);
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/true);
    absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created = WBMpcMrtJointController::Create(
        description_, modelSettings_, mpc_, pinocchioInterface_, /*mpcDesiredFrequency=*/1000.0, pdGainsFile);
    if (created.ok()) {
      controller_ = *std::move(created);
    } else {
      creationStatus_ = created.status();
    }
  }

  /** What WBMpcMrtJointController::Create() returned, when it made no controller. */
  const absl::Status& creationStatus() const { return creationStatus_; }
  WBMpcMrtJointController& controller() {
    ABSL_CHECK(controller_ != nullptr) << "the controller was not created: " << creationStatus_;
    return *controller_;
  }
  const std::vector<std::string>& mpcJointNames() const { return modelSettings_.mpcModelJointNames; }
  const std::vector<size_t>& mpcJointIndices() const { return mpcJointIndices_; }
  const std::vector<size_t>& otherJointIndices() const { return otherJointIndices_; }

  const robot::model::RobotJointAction& cycle() {
    robotState_.setTime(time_);
    controller_->computeJointControlAction(time_, robotState_, action_);
    time_ += kControlPeriod;
    return action_;
  }

  size_t allocationsOfOneCycle() {
    const size_t before = heapAllocationCount();
    cycle();
    return heapAllocationCount() - before;
  }

 private:
  mpc_config::TaskFile task_;
  std::string urdfFile_;
  ModelSettings modelSettings_;
  PinocchioInterface pinocchioInterface_;
  WBAccelMpcRobotModel<scalar_t> model_;
  mpc_test::ScriptedMpc mpc_;
  robot::model::RobotDescription description_;
  robot::model::RobotState robotState_;
  robot::model::RobotJointAction action_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> otherJointIndices_;
  scalar_t time_ = 1.0;
  absl::Status creationStatus_;
  std::unique_ptr<WBMpcMrtJointController> controller_;
};

/** A gains file: `kp` as the default, and `firstJointKp` for the joint `firstJoint`. */
mpc_config::JointPdGainsFile gainsFile(scalar_t kp, const std::string& firstJoint, scalar_t firstJointKp) {
  mpc_config::JointPdGainsFile file;
  file.default_gains.kp = kp;
  file.default_gains.kd = 3.0;
  file.joint_gains.push_back({.joint = firstJoint, .kp = firstJointKp, .kd = std::nullopt, .torque_limit = std::nullopt});
  return file;
}

/** gainsFile() as the text of a joint_pd_gains.textproto. */
std::string gainsText(scalar_t kp, const std::string& firstJoint, scalar_t firstJointKp) {
  return absl::StrCat("default_gains {\n  kp: ", kp, "\n  kd: 3.0\n}\njoint_gains { joint: \"", firstJoint, "\" kp: ", firstJointKp,
                      " }\n");
}

absl::Status setFromAnotherThread(WBMpcMrtJointController& controller, const mpc_config::JointPdGainsFile& gains) {
  absl::Status status;
  std::thread caller([&]() { status = controller.setPdGains(gains); });
  caller.join();
  return status;
}

void writeFile(const std::string& file, const std::string& content) {
  std::ofstream out(file, std::ios::trunc);
  out << content;
}

void touchForward(const std::string& file) {
  std::filesystem::last_write_time(file, std::filesystem::last_write_time(file) + std::chrono::seconds(1));
}

TEST(WBMpcMrtJointControllerPdGains, GainsSetFromAnotherThreadReachTheNextCycleWithoutAnAllocationOfTheirOwn) {
  WholeBodyPdGainsHarness harness;
  harness.controller().setControlMode("ZERO_TORQUE");
  for (int k = 0; k < 5; ++k) harness.cycle();
  // Positive control: the counter sees an allocation of this thread. (A ZERO_TORQUE cycle may make none of its own.)
  const size_t beforeProbe = heapAllocationCount();
  const std::unique_ptr<std::vector<double>> probe = std::make_unique<std::vector<double>>(16);
  ASSERT_GT(heapAllocationCount() - beforeProbe, 0u) << "the allocation counter is not linked into this binary";
  const size_t baseline = harness.allocationsOfOneCycle();
  ASSERT_EQ(harness.allocationsOfOneCycle(), baseline) << "a control cycle does not allocate the same every time; nothing to compare";

  ASSERT_TRUE(
      setFromAnotherThread(harness.controller(), gainsFile(/*kp=*/123.0, harness.mpcJointNames().front(), /*firstJointKp=*/77.0)).ok());
  EXPECT_EQ(harness.controller().getPdGains().defaults.kp, 150.0) << "the gains changed before a control cycle took them";
  EXPECT_EQ(harness.allocationsOfOneCycle(), baseline) << "taking the new gains into use allocated on the control thread";
  EXPECT_EQ(harness.controller().getPdGains().defaults.kp, 123.0) << "the cycle did not take the new gains into use";

  harness.controller().setControlMode("JOINT_PD");
  const robot::model::RobotJointAction& action = harness.cycle();
  EXPECT_EQ(actionOf(action, harness.mpcJointIndices()[0]).kp, 77.0);
  for (size_t i = 1; i < harness.mpcJointIndices().size(); ++i) EXPECT_EQ(actionOf(action, harness.mpcJointIndices()[i]).kp, 123.0);
  for (size_t index : harness.otherJointIndices()) EXPECT_EQ(actionOf(action, index).kp, 123.0 * kOtherJointDefaultGainScale);
}

TEST(WBMpcMrtJointControllerPdGains, RefusedGainsAreRefusedOnTheCallersThreadAndTheGainsStay) {
  WholeBodyPdGainsHarness harness;
  harness.controller().setControlMode("JOINT_PD");
  const std::string firstJoint = harness.mpcJointNames().front();
  ASSERT_TRUE(setFromAnotherThread(harness.controller(), gainsFile(/*kp=*/123.0, firstJoint, /*firstJointKp=*/77.0)).ok());
  ASSERT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[1]).kp, 123.0);

  const mpc_config::JointPdGainsFile valid = gainsFile(/*kp=*/1.0, firstJoint, /*firstJointKp=*/1.0);
  std::vector<mpc_config::JointPdGainsFile> refused = {valid, valid, valid, valid};
  refused[0].default_gains.kd = -3.0;
  refused[1].default_gains.kp = std::numeric_limits<double>::quiet_NaN();
  refused[2].joint_gains.push_back(refused[2].joint_gains.front());  // the same joint twice
  refused[3].joint_gains.front().joint.clear();                      // an entry that names no joint
  for (size_t k = 0; k < refused.size(); ++k) {
    const absl::Status status = setFromAnotherThread(harness.controller(), refused[k]);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "accepted the refused gains " << k;
    const robot::model::RobotJointAction& action = harness.cycle();
    EXPECT_EQ(actionOf(action, harness.mpcJointIndices()[0]).kp, 77.0) << "refused gains " << k << " changed the gains";
    EXPECT_EQ(actionOf(action, harness.mpcJointIndices()[1]).kp, 123.0) << "refused gains " << k << " changed the gains";
    EXPECT_EQ(actionOf(action, harness.mpcJointIndices()[1]).kd, 3.0) << "refused gains " << k << " changed the gains";
  }

  // The payload is the file: one that sets nothing is the controller's defaults.
  ASSERT_TRUE(setFromAnotherThread(harness.controller(), mpc_config::JointPdGainsFile{}).ok());
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[0]).kp, 150.0);
}

TEST(WBMpcMrtJointControllerPdGains, TheGainsFileIsWatchedByPollPdGainsFileAndNotByTheControlCycle) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/wb_joint_pd_gains.textproto");
  writeFile(file, "default_gains {\n  kp: 111.0\n  kd: 3.0\n}\n");
  WholeBodyPdGainsHarness harness(file);
  harness.controller().setControlMode("JOINT_PD");
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[1]).kp, 111.0) << "the constructor did not load the gains file";

  writeFile(file, gainsText(/*kp=*/222.0, harness.mpcJointNames().front(), /*firstJointKp=*/22.0));
  touchForward(file);
  for (int k = 0; k < 1000; ++k) harness.cycle();
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[1]).kp, 111.0) << "the control cycle reloaded the gains file";

  std::thread watcher([&harness]() { harness.controller().pollPdGainsFile(); });
  watcher.join();
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[1]).kp, 222.0) << "pollPdGainsFile() did not reload the changed file";
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[0]).kp, 22.0);

  writeFile(file, "default_gains {\n  kp: 3OO\n");
  touchForward(file);
  harness.controller().pollPdGainsFile();
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[1]).kp, 222.0) << "a malformed gains file changed the gains";

  std::filesystem::remove(file);
}

// At start-up there are no gains in use to keep: a gains file that is refused stops the controller from starting
// rather than leaving every joint on the hard-coded defaults.
TEST(WBMpcMrtJointControllerPdGains, AGainsFileThatIsRefusedStopsTheControllerFromStarting) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/wb_refused_joint_pd_gains.textproto");
  writeFile(file, "default_gains { kp: 150.0 }\njoint_gains { joint: \"left_knee_joint\" kd: -1.0 }\n");
  const WholeBodyPdGainsHarness harness(file);
  const absl::Status& refused = harness.creationStatus();
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << "the controller started on a gains file it refused: " << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), file)) << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "joint_gains[joint=left_knee_joint].kd")) << refused;
  std::filesystem::remove(file);
}

// The whole-body controller commands no torque limit, and its loader never read the field: a value there, whatever it
// is, refuses neither the gains file nor the gains of the GUI.
TEST(WBMpcMrtJointControllerPdGains, TheTorqueLimitKeysAreNotRead) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/wb_torque_limit_joint_pd_gains.textproto");
  writeFile(file, "default_gains {\n  kp: 111.0\n  kd: 3.0\n  torque_limit: -5.0\n}\n");
  WholeBodyPdGainsHarness harness(file);
  harness.controller().setControlMode("JOINT_PD");
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[1]).kp, 111.0);

  mpc_config::JointPdGainsFile unlimited;
  unlimited.default_gains = {.kp = 123.0, .kd = 3.0, .torque_limit = std::numeric_limits<double>::infinity()};
  ASSERT_TRUE(setFromAnotherThread(harness.controller(), unlimited).ok());
  EXPECT_EQ(actionOf(harness.cycle(), harness.mpcJointIndices()[1]).kp, 123.0);
  std::filesystem::remove(file);
}

}  // namespace
}  // namespace ocs2::humanoid
