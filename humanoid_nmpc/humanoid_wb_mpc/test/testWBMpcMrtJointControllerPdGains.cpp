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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_mpc_test/ScriptedMpc.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_core/ResourcePaths.h"

/*
 * The joint PD gains of WBMpcMrtJointController, set from other threads (see testMrtJointControllerPdGains for the
 * centroidal controller): parsed on the caller's thread, taken into use by the next control cycle without an allocation
 * of its own, a malformed document refused without changing the gains, and the gains file watched by
 * pollPdGainsFile() rather than by every 500th control cycle. This binary links the allocation counter, which
 * interposes malloc for the whole process; the solver thread is never started.
 */

namespace ocs2::humanoid {
namespace {

using ::ocs2::humanoid::estimation::heapAllocationCount;

constexpr scalar_t kControlPeriod = 0.01;  // [s]

class WholeBodyPdGainsHarness {
 public:
  explicit WholeBodyPdGainsHarness(const std::string& pdGainsFile = "")
      // A file missing from the runfiles throws here, naming the data dependency to add (robot::resolveResourcePath).
      : taskFile_(robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml").value()),
        urdfFile_(robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf").value()),
        modelSettings_(taskFile_, urdfFile_, "wb_mpc_", /*verbose=*/false),
        pinocchioInterface_(createCustomPinocchioInterface(taskFile_, urdfFile_, modelSettings_)),
        model_(modelSettings_),
        mpc_(mpc::loadSettings(taskFile_, "mpc", /*verbose=*/false), model_.getInputDim()),
        description_(urdfFile_),
        robotState_(description_),
        action_(description_) {
    vector_t initialState = vector_t::Zero(model_.getStateDim());
    loadData::loadEigenMatrix(taskFile_, "initialState", initialState);
    mpcJointIndices_ = description_.getJointIndices(modelSettings_.mpcModelJointNames);
    otherJointIndices_ = description_.getJointIndices(modelSettings_.fixedJointNames);
    robotState_.setRootPositionInWorldFrame(model_.getBasePosition(initialState));
    robotState_.setRootRotationLocalToWorldFrame(
        getQuaternionFromEulerAnglesZyx(vector3_t(model_.getBaseOrientationEulerZYX(initialState))));
    const vector_t joints = model_.getJointAngles(initialState);
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) robotState_.setJointPosition(mpcJointIndices_[i], joints[i]);
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/true);
    controller_ = std::make_unique<WBMpcMrtJointController>(description_, modelSettings_, mpc_, pinocchioInterface_,
                                                            /*mpcDesiredFrequency=*/1000.0, pdGainsFile);
  }

  WBMpcMrtJointController& controller() { return *controller_; }
  const std::vector<std::string>& mpcJointNames() const { return modelSettings_.mpcModelJointNames; }
  const std::vector<size_t>& mpcJointIndices() const { return mpcJointIndices_; }
  const std::vector<size_t>& otherJointIndices() const { return otherJointIndices_; }

  const robot::model::RobotJointAction& cycle() {
    robotState_.setTime(time_);
    controller_->computeJointControlAction(time_, robotState_, action_);
    time_ += kControlPeriod;
    return action_;
  }

  std::size_t allocationsOfOneCycle() {
    const std::size_t before = heapAllocationCount();
    cycle();
    return heapAllocationCount() - before;
  }

 private:
  std::string taskFile_;
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
  std::unique_ptr<WBMpcMrtJointController> controller_;
};

std::string gainsDocument(scalar_t kp, const std::string& firstJoint, scalar_t firstJointKp) {
  return absl::StrCat("default_gains:\n  kp: ", kp, "\n  kd: 3.0\njoint_gains:\n  ", firstJoint, ":\n    kp: ", firstJointKp, "\n");
}

absl::Status setFromAnotherThread(WBMpcMrtJointController& controller, const std::string& yaml) {
  absl::Status status;
  std::thread caller([&]() { status = controller.setPdGainsYaml(yaml); });
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
  const std::size_t beforeProbe = heapAllocationCount();
  const std::unique_ptr<std::vector<double>> probe = std::make_unique<std::vector<double>>(16);
  ASSERT_GT(heapAllocationCount() - beforeProbe, 0u) << "the allocation counter is not linked into this binary";
  const std::size_t baseline = harness.allocationsOfOneCycle();
  ASSERT_EQ(harness.allocationsOfOneCycle(), baseline) << "a control cycle does not allocate the same every time; nothing to compare";

  ASSERT_TRUE(
      setFromAnotherThread(harness.controller(), gainsDocument(/*kp=*/123.0, harness.mpcJointNames().front(), /*firstJointKp=*/77.0)).ok());
  EXPECT_EQ(harness.controller().getPdGains().defaults.kp, 150.0) << "the gains changed before a control cycle took them";
  EXPECT_EQ(harness.allocationsOfOneCycle(), baseline) << "taking the new gains into use allocated on the control thread";
  EXPECT_EQ(harness.controller().getPdGains().defaults.kp, 123.0) << "the cycle did not take the new gains into use";

  harness.controller().setControlMode("JOINT_PD");
  const robot::model::RobotJointAction& action = harness.cycle();
  EXPECT_EQ(action.at(harness.mpcJointIndices()[0])->kp, 77.0);
  for (size_t i = 1; i < harness.mpcJointIndices().size(); ++i) EXPECT_EQ(action.at(harness.mpcJointIndices()[i])->kp, 123.0);
  for (size_t index : harness.otherJointIndices()) EXPECT_EQ(action.at(index)->kp, 123.0 * kOtherJointDefaultGainScale);
}

TEST(WBMpcMrtJointControllerPdGains, AMalformedDocumentIsRefusedOnTheCallersThreadAndTheGainsStay) {
  WholeBodyPdGainsHarness harness;
  harness.controller().setControlMode("JOINT_PD");
  ASSERT_TRUE(
      setFromAnotherThread(harness.controller(), gainsDocument(/*kp=*/123.0, harness.mpcJointNames().front(), /*firstJointKp=*/77.0)).ok());
  ASSERT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 123.0);

  for (const char* malformed :
       {"default_gains: {kp: stiff}\n", "joint_gains: [a, b]\n", "default_gains: {kd: -3.0}\n", "{unclosed: [\n", ""}) {
    const absl::Status status = setFromAnotherThread(harness.controller(), malformed);
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << "accepted: " << malformed;
    const robot::model::RobotJointAction& action = harness.cycle();
    EXPECT_EQ(action.at(harness.mpcJointIndices()[0])->kp, 77.0) << "a refused document changed the gains: " << malformed;
    EXPECT_EQ(action.at(harness.mpcJointIndices()[1])->kp, 123.0) << "a refused document changed the gains: " << malformed;
    EXPECT_EQ(action.at(harness.mpcJointIndices()[1])->kd, 3.0) << "a refused document changed the gains: " << malformed;
  }
}

TEST(WBMpcMrtJointControllerPdGains, TheGainsFileIsWatchedByPollPdGainsFileAndNotByTheControlCycle) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/wb_joint_pd_gains.yaml");
  writeFile(file, "default_gains:\n  kp: 111.0\n  kd: 3.0\n");
  WholeBodyPdGainsHarness harness(file);
  harness.controller().setControlMode("JOINT_PD");
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 111.0) << "the constructor did not load the gains file";

  writeFile(file, gainsDocument(/*kp=*/222.0, harness.mpcJointNames().front(), /*firstJointKp=*/22.0));
  touchForward(file);
  for (int k = 0; k < 1000; ++k) harness.cycle();
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 111.0) << "the control cycle reloaded the gains file";

  std::thread watcher([&harness]() { harness.controller().pollPdGainsFile(); });
  watcher.join();
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 222.0) << "pollPdGainsFile() did not reload the changed file";
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[0])->kp, 22.0);

  writeFile(file, "default_gains:\n  kp: 3OO\n");
  touchForward(file);
  harness.controller().pollPdGainsFile();
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 222.0) << "a malformed gains file changed the gains";

  std::filesystem::remove(file);
}

// At start-up there are no gains in use to keep: a gains file that is refused stops the controller from starting
// rather than leaving every joint on the hard-coded defaults.
TEST(WBMpcMrtJointControllerPdGains, AGainsFileThatIsRefusedStopsTheControllerFromStarting) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/wb_refused_joint_pd_gains.yaml");
  writeFile(file, "default_gains:\n  kp: 150.0\njoint_gains:\n  left_knee_joint: {kd: fast}\n");
  try {
    WholeBodyPdGainsHarness harness(file);
    ADD_FAILURE() << "the controller started on a gains file it refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_TRUE(absl::StrContains(e.what(), file)) << e.what();
    EXPECT_TRUE(absl::StrContains(e.what(), "joint_gains.left_knee_joint.kd")) << e.what();
  }
  std::filesystem::remove(file);
}

// The whole-body controller commands no torque limit, and its loader never read the key: a value there, whatever it
// is, refuses neither the gains file nor a document of the GUI.
TEST(WBMpcMrtJointControllerPdGains, TheTorqueLimitKeysAreNotRead) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/wb_torque_limit_joint_pd_gains.yaml");
  writeFile(file, "default_gains:\n  kp: 111.0\n  kd: 3.0\n  torque_limit: -5.0\n");
  WholeBodyPdGainsHarness harness(file);
  harness.controller().setControlMode("JOINT_PD");
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 111.0);

  ASSERT_TRUE(setFromAnotherThread(harness.controller(), "default_gains:\n  kp: 123.0\n  kd: 3.0\n  torque_limit: .inf\n").ok());
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 123.0);
  std::filesystem::remove(file);
}

}  // namespace
}  // namespace ocs2::humanoid
