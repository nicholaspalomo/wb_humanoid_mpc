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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/initialization/DefaultInitializer.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_oc/synchronized_module/ReferenceManager.h"
#include "ocs2_sqp/SqpMpc.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_centroidal_mpc/parameter_update/BasisInputsCostApplier.h"
#include "humanoid_centroidal_mpc_test/CentroidalTestingModelInterface.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_common_mpc/parameter_update/SqpSettingsApplier.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

class MpcParameterUpdaterModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // A temporary copy of the task file, to modify safely.
    tempTaskFile_ = std::filesystem::path(testing::TempDir()) / "test_task.textproto";
    std::error_code error;
    std::filesystem::copy_file(testingModelInterface_.taskFile, tempTaskFile_, std::filesystem::copy_options::overwrite_existing, error);
    ASSERT_FALSE(error) << error.message();
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove(tempTaskFile_, ignored);
  }

  /**
   * The updater of the test model watching the temporary task file, on `mpc` (nullptr: none, which the tests below expect
   * Create() to accept) with `appliers`.
   */
  std::unique_ptr<MpcParameterUpdaterModule> createUpdater(MPC_BASE* absl_nullable mpc = nullptr,
                                                           std::vector<std::unique_ptr<HotFieldApplier>> appliers = {}) {
    MpcParameterUpdaterModule::Options options;
    options.taskFile = tempTaskFile_.string();
    options.referenceFile = testingModelInterface_.referenceFile;
    options.runningTask = testingModelInterface_.config.task;
    options.layout = stateInputLayout(testingModelInterface_.getModelSettings(), StateInputLayout::Mpc::kCentroidal);
    options.inputDim = testingModelInterface_.getMpcRobotModel().getInputDim();
    options.appliers = std::move(appliers);
    absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> created = MpcParameterUpdaterModule::Create(mpc, std::move(options));
    EXPECT_TRUE(created.ok()) << created.status();
    return created.ok() ? *std::move(created) : nullptr;
  }

  /** Replaces the watched task file with `text`, a modification time later than any before it, and polls it. */
  void writeAndPoll(MpcParameterUpdaterModule& updater, absl::string_view text) {
    {
      std::ofstream out(tempTaskFile_, std::ios::trunc);
      out << text;
    }
    ++writeCount_;
    std::error_code error;
    std::filesystem::last_write_time(tempTaskFile_, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(writeCount_),
                                     error);
    ASSERT_FALSE(error) << error.message();
    ReferenceManager referenceManager;
    const vector_t state = vector_t::Zero(testingModelInterface_.getMpcRobotModel().getStateDim());
    // The file is polled once every hundred pre-solve hooks.
    for (int i = 0; i < 150; ++i) updater.preSolverRun(/*initTime=*/0.0, /*finalTime=*/0.01, state, referenceManager);
  }

  CentroidalTestingModelInterface testingModelInterface_;
  std::filesystem::path tempTaskFile_;
  int writeCount_ = 0;
};

TEST_F(MpcParameterUpdaterModuleTest, testFileWatcher) {
  // Pass a nullptr for MPC_BASE. The module should safely handle this.
  std::unique_ptr<MpcParameterUpdaterModule> updater = createUpdater();
  ASSERT_NE(updater, nullptr);

  ReferenceManager referenceManager;
  vector_t state = vector_t::Zero(testingModelInterface_.getMpcRobotModel().getStateDim());

  // Call it a few times, it shouldn't trigger anything since file hasn't changed.
  for (int i = 0; i < 150; ++i) {
    updater->preSolverRun(/*initTime=*/0.0, /*finalTime=*/0.01, state, referenceManager);
  }

  // Now modify the file: it is parsed (and logged) without crashing, although there is no solver to update.
  std::ifstream in(tempTaskFile_);
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NO_FATAL_FAILURE(writeAndPoll(*updater, text + "\n# Test Modification\n"));
}

// The solver settings of an applied task file reach the running solver, whichever pathway delivered it (the task file
// here). A refused block keeps the running settings, a file that does not parse applies nothing, and the file is the
// payload: a file without the block applies the block's defaults. (The contact estimator and the contact wrench gate of
// a file are the robot's, which takes them from the bus and from its own task file.)
TEST_F(MpcParameterUpdaterModuleTest, theSolverSettingsOfAnAppliedFileReachTheRunningSolver) {
  const mpc_config::TaskFile& shipped = testingModelInterface_.config.task;
  const absl::StatusOr<SolverSettings> settings = solverSettingsFromConfig(shipped);
  ASSERT_TRUE(settings.ok()) << settings.status();
  // A solver of the shipped settings on a problem without terms: the settings are all a reload reaches here.
  sqp::Settings sqpSettings = settings->sqpSettings;
  sqpSettings.enableLogging = false;
  const OptimalControlProblem problem;
  const DefaultInitializer initializer(testingModelInterface_.getMpcRobotModel().getInputDim());
  SqpMpc mpc(settings->mpcSettings, sqpSettings, problem, initializer);
  const SqpSolver& solver = dynamic_cast<const SqpSolver&>(*mpc.getSolverPtr());
  std::vector<std::unique_ptr<HotFieldApplier>> appliers;
  appliers.push_back(std::make_unique<SqpSettingsApplier>());
  std::unique_ptr<MpcParameterUpdaterModule> updater = createUpdater(&mpc, std::move(appliers));
  ASSERT_NE(updater, nullptr);
  const size_t running = solver.getSettings().sqpIteration;

  mpc_config::TaskFile edited = shipped;
  edited.multiple_shooting.sqp_iteration = static_cast<int32_t>(running) + 3;
  writeAndPoll(*updater, taskFileText(edited));
  EXPECT_EQ(solver.getSettings().sqpIteration, running + 3) << "the edited task file did not reach the solver";

  // A negative iteration count is refused by its field and the running settings are kept.
  edited.multiple_shooting.sqp_iteration = -1;
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::AllOf(testing::HasSubstr("multiple_shooting"), testing::HasSubstr("was not applied"))))
        .Times(1);
    log.StartCapturingLogs();
    writeAndPoll(*updater, taskFileText(edited));
    log.StopCapturingLogs();
  }
  EXPECT_EQ(solver.getSettings().sqpIteration, running + 3);

  // A value that does not parse refuses the whole file, with an error that names the file and the position, where a
  // `get` with a default used to apply the default in its place without a word.
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                         testing::AllOf(testing::HasSubstr("was not reloaded"), testing::HasSubstr("test_task.textproto:2:"))))
        .Times(1);
    log.StartCapturingLogs();
    writeAndPoll(*updater, "multiple_shooting {\n  sqp_iteration: many\n}\n");
    log.StopCapturingLogs();
  }
  EXPECT_EQ(solver.getSettings().sqpIteration, running + 3) << "a file that does not parse was applied";

  // A file without the block applies its defaults.
  writeAndPoll(*updater, "state_weights { scaling: 1.0 }\n");
  EXPECT_EQ(solver.getSettings().sqpIteration, static_cast<size_t>(mpc_config::SqpSettingsConfig{}.sqp_iteration));
}

TEST_F(MpcParameterUpdaterModuleTest, basisCostTransformRequiresBasisSpaceInputDim) {
  // The task file's input_weights is indexed in wrench space, so the only sane inputDim for a module carrying a basis-space transform is
  // the transform's basis-space dimension. Passing the wrench-space dimension must be rejected at construction.
  const size_t wrenchInputDim = testingModelInterface_.getMpcRobotModel().getInputDim();
  const size_t numJoints = testingModelInterface_.getModelSettings().mpc_joint_dim;

  BasisInputsCostTransformConfig config;
  config.wrenchInputDim = wrenchInputDim;
  config.numBasisInputs = 8 * kNumContacts;
  config.lambdaRegularization = 1.0e-3;
  config.basisToWrenchMap = matrix_t::Random(wrenchInputDim, config.numBasisInputs + numJoints);
  ASSERT_NE(config.basisInputDim(), wrenchInputDim);

  const absl::StatusOr<std::unique_ptr<BasisInputsCostApplier>> wrench = BasisInputsCostApplier::Create(config, wrenchInputDim);
  ASSERT_FALSE(wrench.ok()) << "the wrench-space input dimension was accepted beside a basis-space transform";
  EXPECT_EQ(wrench.status().code(), absl::StatusCode::kInvalidArgument) << wrench.status();
  EXPECT_NE(wrench.status().message().find("inputDim"), absl::string_view::npos) << wrench.status();
  const absl::StatusOr<std::unique_ptr<BasisInputsCostApplier>> basis = BasisInputsCostApplier::Create(config, config.basisInputDim());
  EXPECT_TRUE(basis.ok()) << basis.status();
}

}  // namespace ocs2::humanoid
