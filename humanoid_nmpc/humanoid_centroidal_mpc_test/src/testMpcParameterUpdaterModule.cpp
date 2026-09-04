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

#include <humanoid_centroidal_mpc/mrt/MpcParameterUpdaterModule.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"

using namespace ocs2;
using namespace ocs2::humanoid;

class MpcParameterUpdaterModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // We create a temporary copy of the task.yaml to safely modify it
    tempTaskFile_ = std::filesystem::temp_directory_path() / "test_task.yaml";
    std::filesystem::copy_file(testingModelInterface.taskFile, tempTaskFile_, std::filesystem::copy_options::overwrite_existing);
  }

  void TearDown() override {
    if (std::filesystem::exists(tempTaskFile_)) {
      std::filesystem::remove(tempTaskFile_);
    }
  }

  CentroidalTestingModelInterface testingModelInterface;
  std::filesystem::path tempTaskFile_;
};

TEST_F(MpcParameterUpdaterModuleTest, testFileWatcher) {
  // Pass a nullptr for MPC_BASE. The module should safely handle this.
  MpcParameterUpdaterModule updater(nullptr, tempTaskFile_.string(), testingModelInterface.urdfFile, testingModelInterface.referenceFile);

  ReferenceManager referenceManager;
  vector_t state = vector_t::Zero(testingModelInterface.getMpcRobotModel().getStateDim());

  // Call it a few times, it shouldn't trigger anything since file hasn't changed.
  for (int i = 0; i < 150; ++i) {
    updater.preSolverRun(0.0, 0.01, state, referenceManager);
  }

  // Now modify the file
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Ensure timestamp difference

  std::ofstream ofs(tempTaskFile_, std::ios::app);
  ofs << "\n# Test Modification\n";
  ofs.close();

  // Call it enough times to trigger the modulo counter (checkCounter_ % 100 == 0)
  // It should parse the file (and log it) without crashing because of the nullptr check we added.
  EXPECT_NO_THROW({
    for (int i = 0; i < 150; ++i) {
      updater.preSolverRun(0.0, 0.01, state, referenceManager);
    }
  });
}
