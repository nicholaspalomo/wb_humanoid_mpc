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
#include <filesystem>
#include <fstream>
#include <thread>

#include <humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h>
#include <ocs2_dummy_loop/DummyObserver.h>
#include <ocs2_mpc/MPC_BASE.h>
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"

using namespace ocs2;
using namespace ocs2::humanoid;

class MockMpc : public MPC_BASE {
 public:
  MockMpc() {
    SystemObservation initialObservation;
    initialObservation.time = 0.0;
    initialObservation.state = vector_t::Zero(10);
    initialObservation.input = vector_t::Zero(10);
    mrtObservationQueue_.push(initialObservation);
  }

  bool run(double currentTime, const vector_t& currentState) override { return true; }
  void calculateController(double initTime, const vector_t& initState, double finalTime) override {}
  const OptimalControlProblem& getOptimalControlProblem() const override { throw std::runtime_error("Not implemented"); }
};

class CentroidalMpcMrtJointControllerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tempPdGainsFile_ = std::filesystem::temp_directory_path() / "test_pd_gains.yaml";

    // Create a dummy PD gains file
    std::ofstream ofs(tempPdGainsFile_);
    ofs << "joints:\n"
        << "  LeftKneePitch:\n"
        << "    kp: 100.0\n"
        << "    kd: 10.0\n"
        << "    torque_limit: 150.0\n";
    ofs.close();
  }

  void TearDown() override {
    if (std::filesystem::exists(tempPdGainsFile_)) {
      std::filesystem::remove(tempPdGainsFile_);
    }
  }

  CentroidalTestingModelInterface testingModelInterface;
  std::filesystem::path tempPdGainsFile_;
};

TEST_F(CentroidalMpcMrtJointControllerTest, testPdGainsHotReloading) {
  MockMpc mockMpc;

  // Set up robot state and joint action
  robot::model::RobotState robotState;
  robotState.t = 0.0;
  robot::model::RobotJointAction jointAction;
  auto jointNames = testingModelInterface.getPinocchioInterface().getModel().names;
  for (const auto& name : jointNames) {
    if (name != "universe") {
      jointAction.insert(name, robot::model::JointAction{});
    }
  }

  // Create controller
  CentroidalMpcMrtJointController controller(testingModelInterface.getPinocchioInterface().getModel(),
                                             testingModelInterface.getModelSettings(), testingModelInterface.getMpcRobotModel(), mockMpc,
                                             testingModelInterface.getPinocchioInterface(), 400.0, nullptr, tempPdGainsFile_.string());

  controller.setControlMode("JOINT_PD");

  // Trigger first compute loop
  EXPECT_NO_THROW({ controller.computeJointControlAction(0.01, robotState, jointAction); });

  // Modify file
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Ensure timestamp difference
  std::ofstream ofs(tempPdGainsFile_, std::ios::app);
  ofs << "  RightKneePitch:\n"
      << "    kp: 200.0\n"
      << "    kd: 20.0\n"
      << "    torque_limit: 150.0\n";
  ofs.close();

  // Trigger again
  EXPECT_NO_THROW({ controller.computeJointControlAction(0.02, robotState, jointAction); });
}
