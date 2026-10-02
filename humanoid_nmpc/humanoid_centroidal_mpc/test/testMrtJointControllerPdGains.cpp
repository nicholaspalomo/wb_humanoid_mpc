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

#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <robot_model/RobotDescription.h>
#include <robot_model/RobotJointAction.h>
#include <robot_model/RobotState.h>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_common_mpc/mrt/JointPdGains.h"
#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"
#include "support/AtlasReferenceStack.h"

/*
 * The joint PD gains of CentroidalMpcMrtJointController, set from other threads. computeJointControlAction() used to take
 * the mutex of the ROS callback, write the YAML it received to <gains file>.live.yaml, parse that with YAML::LoadFile and
 * log every joint's gains, and every 100 cycles stat the gains file - all on the realtime thread. The text is now
 * parsed on the caller's thread (setPdGainsYaml(), pollPdGainsFile()) and the control cycle takes the gains into use
 * without a lock or an allocation of its own. This binary links the allocation counter, which interposes malloc for the
 * whole process; the controller's solver thread is never started, so the control thread is the only one allocating.
 */

namespace ocs2::humanoid {
namespace {

using ::ocs2::humanoid::estimation::heapAllocationCount;

constexpr scalar_t kControlPeriod = 0.01;  // [s]

class PdGainsHarness {
 public:
  explicit PdGainsHarness(const std::string& pdGainsFile = "")
      : description_(stack_.urdfFile()), robotState_(description_), action_(description_) {
    controller_ =
        std::make_unique<CentroidalMpcMrtJointController>(description_, stack_.modelSettings(), stack_.model(), stack_.mpc(),
                                                          stack_.pinocchioInterface(), /*mpcDesiredFrequency=*/1000.0, pdGainsFile);
    mpcJointIndices_ = description_.getJointIndices(stack_.modelSettings().mpcModelJointNames);
    otherJointIndices_ = description_.getJointIndices(stack_.modelSettings().fixedJointNames);
    const CentroidalMpcRobotModel<scalar_t>& model = stack_.model();
    const vector_t& state = stack_.initialState();
    robotState_.setRootPositionInWorldFrame(model.getBasePosition(state));
    robotState_.setRootRotationLocalToWorldFrame(getQuaternionFromEulerAnglesZyx(vector3_t(model.getBaseOrientationEulerZYX(state))));
    const vector_t joints = model.getJointAngles(state);
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) robotState_.setJointPosition(mpcJointIndices_[i], joints[i]);
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  }

  CentroidalMpcMrtJointController& controller() { return *controller_; }
  const std::vector<std::string>& mpcJointNames() const { return stack_.modelSettings().mpcModelJointNames; }
  const std::vector<size_t>& mpcJointIndices() const { return mpcJointIndices_; }
  const std::vector<size_t>& otherJointIndices() const { return otherJointIndices_; }

  const robot::model::RobotJointAction& cycle() {
    robotState_.setTime(time_);
    controller_->computeJointControlAction(time_, robotState_, action_);
    time_ += kControlPeriod;
    return action_;
  }

  /** The heap allocations of one control cycle. */
  std::size_t allocationsOfOneCycle() {
    const std::size_t before = heapAllocationCount();
    cycle();
    return heapAllocationCount() - before;
  }

 private:
  AtlasReferenceStack stack_;
  robot::model::RobotDescription description_;
  robot::model::RobotState robotState_;
  robot::model::RobotJointAction action_;
  std::vector<size_t> mpcJointIndices_;
  std::vector<size_t> otherJointIndices_;
  scalar_t time_ = 1.0;
  std::unique_ptr<CentroidalMpcMrtJointController> controller_;
};

/** A gains document: `kp` as the default, and `firstJointKp` for the joint `firstJoint`. */
std::string gainsDocument(scalar_t kp, const std::string& firstJoint, scalar_t firstJointKp) {
  return absl::StrCat("default_gains:\n  kp: ", kp, "\n  kd: 3.0\njoint_gains:\n  ", firstJoint, ":\n    kp: ", firstJointKp, "\n");
}

absl::Status setFromAnotherThread(CentroidalMpcMrtJointController& controller, const std::string& yaml) {
  absl::Status status;
  std::thread caller([&]() { status = controller.setPdGainsYaml(yaml); });
  caller.join();
  return status;
}

void writeFile(const std::string& file, const std::string& content) {
  std::ofstream out(file, std::ios::trunc);
  out << content;
}

/** Moves the file's modification time on by a second, so that a rewrite inside the file system's clock tick counts. */
void touchForward(const std::string& file) {
  std::filesystem::last_write_time(file, std::filesystem::last_write_time(file) + std::chrono::seconds(1));
}

TEST(MrtJointControllerPdGains, GainsSetFromAnotherThreadReachTheNextCycleWithoutAnAllocationOfTheirOwn) {
  PdGainsHarness harness;
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
  EXPECT_EQ(harness.controller().getPdGains().defaults.kp, 250.0) << "the gains changed before a control cycle took them";
  EXPECT_EQ(harness.allocationsOfOneCycle(), baseline) << "taking the new gains into use allocated on the control thread";
  EXPECT_EQ(harness.controller().getPdGains().defaults.kp, 123.0) << "the cycle did not take the new gains into use";

  // And they are what the joints are commanded with.
  harness.controller().setControlMode("JOINT_PD");
  const robot::model::RobotJointAction& action = harness.cycle();
  EXPECT_EQ(action.at(harness.mpcJointIndices()[0])->kp, 77.0);
  for (size_t i = 1; i < harness.mpcJointIndices().size(); ++i) EXPECT_EQ(action.at(harness.mpcJointIndices()[i])->kp, 123.0);
  for (size_t index : harness.otherJointIndices()) EXPECT_EQ(action.at(index)->kp, 123.0 * kOtherJointDefaultGainScale);
}

TEST(MrtJointControllerPdGains, AMalformedDocumentIsRefusedOnTheCallersThreadAndTheGainsStay) {
  PdGainsHarness harness;
  harness.controller().setControlMode("JOINT_PD");
  ASSERT_TRUE(
      setFromAnotherThread(harness.controller(), gainsDocument(/*kp=*/123.0, harness.mpcJointNames().front(), /*firstJointKp=*/77.0)).ok());
  ASSERT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 123.0);

  // The old path wrote these to a file, failed to parse it on the control thread and reset every joint to 250.
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

TEST(MrtJointControllerPdGains, TheGainsFileIsWatchedByPollPdGainsFileAndNotByTheControlCycle) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/centroidal_joint_pd_gains.yaml");
  writeFile(file, "default_gains:\n  kp: 111.0\n  kd: 3.0\n");
  PdGainsHarness harness(file);
  harness.controller().setControlMode("JOINT_PD");
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 111.0) << "the constructor did not load the gains file";

  // A rewrite is not seen by the control cycle, however many run: it no longer stats the file.
  writeFile(file, gainsDocument(/*kp=*/222.0, harness.mpcJointNames().front(), /*firstJointKp=*/22.0));
  touchForward(file);
  for (int k = 0; k < 500; ++k) harness.cycle();
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 111.0) << "the control cycle reloaded the gains file";

  // The watcher, on a thread of its own, reloads it for the next cycle.
  std::thread watcher([&harness]() { harness.controller().pollPdGainsFile(); });
  watcher.join();
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 222.0) << "pollPdGainsFile() did not reload the changed file";
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[0])->kp, 22.0);

  // A file saved half-written or with a typo is reported and leaves the gains in use, where it used to apply whatever
  // had parsed before the error and the hard-coded defaults for the rest.
  writeFile(file, "default_gains:\n  kp: 3OO\n");
  touchForward(file);
  harness.controller().pollPdGainsFile();
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 222.0) << "a malformed gains file changed the gains";

  std::filesystem::remove(file);
}

// At start-up there are no gains in use to keep: a gains file that is refused stops the controller from starting. It
// used to leave every joint the file did not reach on the hard-coded 500 N*m torque limit.
TEST(MrtJointControllerPdGains, AGainsFileThatIsRefusedStopsTheControllerFromStarting) {
  const std::string file = absl::StrCat(::testing::TempDir(), "/centroidal_refused_joint_pd_gains.yaml");
  writeFile(file, "default_gains:\n  kp: 111.0\n  torque_limit: 24.0\njoint_gains:\n  l_leg_kny:\n    torque_limit: -24.0\n");
  try {
    PdGainsHarness harness(file);
    ADD_FAILURE() << "the controller started on a gains file it refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_TRUE(absl::StrContains(e.what(), file)) << e.what();
    EXPECT_TRUE(absl::StrContains(e.what(), "joint_gains.l_leg_kny.torque_limit")) << e.what();
  }

  // Positive control: the same file without the mistake starts the controller on its gains.
  writeFile(file, "default_gains:\n  kp: 111.0\n  torque_limit: 24.0\n");
  PdGainsHarness harness(file);
  harness.controller().setControlMode("JOINT_PD");
  EXPECT_EQ(harness.cycle().at(harness.mpcJointIndices()[1])->kp, 111.0);
  EXPECT_EQ(harness.controller().getPdGains().mpcJointTorqueLimit[1], 24.0);
  std::filesystem::remove(file);
}

}  // namespace
}  // namespace ocs2::humanoid
