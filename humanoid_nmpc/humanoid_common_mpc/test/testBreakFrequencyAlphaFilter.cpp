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
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/reference_manager/BreakFrequencyAlphaFilter.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"

/*
 * The command filter of the procedural motion manager: a first-order low-pass filter on the solver time, configured by
 * reference.yaml's velocityCommandFilterBreakFrequency and shipped off. It used to blend its input with its initial
 * output by the wall-clock time since construction and never update either, so the "5 Hz" filter was a scale factor
 * t / (t + 0.032 s) that became the identity after start-up.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStart = 300.0;  // [s] a clock far from zero: the filter runs on the time it is given

vector_t constant(scalar_t value) {
  return vector_t::Constant(1, value);
}

scalar_t timeConstant(scalar_t breakFrequency) {
  return 1.0 / (2.0 * M_PI * breakFrequency);
}

BreakFrequencyAlphaFilter filterAt(scalar_t breakFrequency, scalar_t initialOutput = 0.0) {
  BreakFrequencyAlphaFilter filter(constant(initialOutput));
  EXPECT_TRUE(filter.setBreakFrequency(breakFrequency).ok());
  return filter;
}

TEST(BreakFrequencyAlphaFilter, aFilterThatIsOffIsTheIdentityFromTheFirstSample) {
  BreakFrequencyAlphaFilter filter(vector4_t::Zero());
  EXPECT_FALSE(filter.isEnabled());
  scalar_t time = kStart;
  for (const vector4_t& command : {vector4_t(1.2, 0.0, 0.0, 0.0), vector4_t(-0.3, 0.25, 0.1, 1.0), vector4_t(vector4_t::Zero())}) {
    EXPECT_EQ(filter.update(time, command), command);
    time += 0.013;
  }
}

TEST(BreakFrequencyAlphaFilter, theStepResponseIsTheContinuousFilterAtEverySampleHoweverTheyAreSpaced) {
  const std::vector<scalar_t> steps = {0.004, 0.011, 0.0005, 0.02, 0.0071, 0.03, 0.0123, 0.001};
  for (const scalar_t breakFrequency : {0.5, 2.0, 5.0}) {
    SCOPED_TRACE(absl::StrCat("f_c = ", breakFrequency, " Hz"));
    const scalar_t tau = timeConstant(breakFrequency);
    BreakFrequencyAlphaFilter filter = filterAt(breakFrequency);
    EXPECT_EQ(filter.update(kStart, constant(1.0))(0), 0.0) << "the first sample starts the clock";
    scalar_t time = kStart;
    for (size_t k = 0; time - kStart < 5.0 * tau; ++k) {
      time += steps[k % steps.size()];
      // The time the filter sees since the step, (kStart + t) - kStart, rather than t: the two differ in the last digit.
      const scalar_t elapsed = time - kStart;
      EXPECT_NEAR(filter.update(time, constant(1.0))(0), 1.0 - std::exp(-elapsed / tau), 1e-12) << "t = " << elapsed;
    }
    EXPECT_GT(filter.getOutput()(0), 0.99);
  }
}

TEST(BreakFrequencyAlphaFilter, theOutputCoversOneMinusOneOverEOfAStepInOneTimeConstant) {
  for (const scalar_t breakFrequency : {1.0, 5.0, 20.0}) {
    const scalar_t tau = timeConstant(breakFrequency);
    // A clock at 1 s, where one time constant is represented to the last digit that the checks below resolve.
    const scalar_t start = 1.0;
    const scalar_t expected = 0.4 + 2.0 * (1.0 - std::exp(-((start + tau) - start) / tau));
    EXPECT_NEAR(expected, 0.4 + 2.0 * (1.0 - std::exp(-1.0)), 1e-12);
    BreakFrequencyAlphaFilter oneSample = filterAt(breakFrequency, /*initialOutput=*/0.4);
    oneSample.update(start, constant(2.4));
    EXPECT_NEAR(oneSample.update(start + tau, constant(2.4))(0), expected, 1e-12);

    BreakFrequencyAlphaFilter manySamples = filterAt(breakFrequency, /*initialOutput=*/0.4);
    manySamples.update(start, constant(2.4));
    for (int k = 1; k < 100; ++k) manySamples.update(start + tau * k / 100.0, constant(2.4));
    EXPECT_NEAR(manySamples.update(start + tau, constant(2.4))(0), expected, 1e-12);
  }
}

TEST(BreakFrequencyAlphaFilter, itRunsOnTheTimeItIsGivenAndKeepsItsState) {
  BreakFrequencyAlphaFilter filter = filterAt(/*breakFrequency=*/5.0);
  const scalar_t tau = timeConstant(5.0);
  filter.update(kStart, constant(1.0));
  // Wall-clock time passes, solver time does not: the output does not move. The filter this replaces grew with the
  // wall clock here, to 0.39 of the input after the 20 ms below.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (int k = 0; k < 100; ++k) EXPECT_EQ(filter.update(kStart, constant(1.0))(0), 0.0);

  // The state carries over: after ten time constants at 1, a step back to 0 decays from there, not from the seed.
  const scalar_t settled = filter.update(kStart + 10.0 * tau, constant(1.0))(0);
  EXPECT_NEAR(settled, 1.0 - std::exp(-10.0), 1e-9);
  const scalar_t oneMore = (kStart + 11.0 * tau) - (kStart + 10.0 * tau);
  EXPECT_NEAR(filter.update(kStart + 11.0 * tau, constant(0.0))(0), settled * std::exp(-oneMore / tau), 1e-12);
}

TEST(BreakFrequencyAlphaFilter, aResetReturnsToTheStateAfterConstructionAndKeepsTheBreakFrequency) {
  BreakFrequencyAlphaFilter used = filterAt(/*breakFrequency=*/2.0, /*initialOutput=*/0.1);
  used.update(kStart, constant(1.0));
  used.update(kStart + 0.3, constant(-0.5));
  ASSERT_NE(used.getOutput()(0), 0.1) << "positive control: the used filter has moved";
  used.reset();
  EXPECT_EQ(used.getOutput()(0), 0.1);
  EXPECT_EQ(used.getBreakFrequency(), 2.0);

  BreakFrequencyAlphaFilter fresh = filterAt(/*breakFrequency=*/2.0, /*initialOutput=*/0.1);
  // After the reset the clock starts again at the next sample, wherever that is.
  for (const scalar_t time : {1.0, 1.02, 1.05, 1.3}) {
    EXPECT_EQ(used.update(time, constant(0.7))(0), fresh.update(time, constant(0.7))(0)) << "t = " << time;
  }
}

TEST(BreakFrequencyAlphaFilter, aClockThatRunsBackwardsRestartsTheClockAndHoldsTheOutput) {
  const scalar_t tau = timeConstant(2.0);
  BreakFrequencyAlphaFilter filter = filterAt(/*breakFrequency=*/2.0);
  filter.update(/*time=*/5.0, constant(1.0));
  const scalar_t before = filter.update(/*time=*/5.1, constant(1.0))(0);
  ASSERT_GT(before, 0.0);
  EXPECT_EQ(filter.update(/*time=*/2.0, constant(1.0))(0), before);
  const scalar_t elapsed = (2.0 + tau) - 2.0;
  EXPECT_NEAR(filter.update(2.0 + tau, constant(1.0))(0), before + (1.0 - before) * (1.0 - std::exp(-elapsed / tau)), 1e-12);
}

TEST(BreakFrequencyAlphaFilter, switchingTheFilterOnContinuesFromTheLastInput) {
  BreakFrequencyAlphaFilter filter(constant(0.0));
  EXPECT_EQ(filter.update(/*time=*/1.0, constant(0.8))(0), 0.8);
  ASSERT_TRUE(filter.setBreakFrequency(2.0).ok());
  EXPECT_TRUE(filter.isEnabled());
  EXPECT_EQ(filter.update(/*time=*/1.1, constant(0.8))(0), 0.8) << "no jump back to the initial output";
  EXPECT_LT(filter.update(/*time=*/1.2, constant(0.0))(0), 0.8);
  EXPECT_GT(filter.getOutput()(0), 0.0);
  // And off again: the identity from the next sample.
  ASSERT_TRUE(filter.setBreakFrequency(0.0).ok());
  EXPECT_EQ(filter.update(/*time=*/1.3, constant(-0.2))(0), -0.2);
}

TEST(BreakFrequencyAlphaFilter, anInvalidBreakFrequencyIsRejectedAndChangesNothing) {
  BreakFrequencyAlphaFilter filter = filterAt(/*breakFrequency=*/2.0);
  for (const scalar_t invalid : {-1.0, std::numeric_limits<scalar_t>::quiet_NaN(), std::numeric_limits<scalar_t>::infinity()}) {
    EXPECT_EQ(filter.setBreakFrequency(invalid).code(), absl::StatusCode::kInvalidArgument) << invalid;
    EXPECT_EQ(filter.getBreakFrequency(), 2.0);
  }
  EXPECT_TRUE(BreakFrequencyAlphaFilter::validateBreakFrequency(0.0).ok());
}

/******************************************************************************************************/
/*                         The reference.yaml key ProceduralMpcMotionManager reads                          */
/******************************************************************************************************/

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

TEST(VelocityCommandFilterKey, everyShippedRobotConfiguresTheFilterAndShipsItOff) {
  // LINT.IfChange(shipped_reference_files)
  const std::vector<std::string> referenceFiles = {
      "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml",
      "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/command/reference.yaml",
      "robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.yaml",
      "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml",
      "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/command/reference.yaml",
  };
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:filter_test_data)
  for (const std::string& relativePath : referenceFiles) {
    SCOPED_TRACE(relativePath);
    const std::string file = runfilePath(relativePath);
    ASSERT_FALSE(file.empty()) << "not in the runfiles";
    EXPECT_TRUE(
        absl::StrContains(readFile(file), absl::StrCat("\n", ProceduralMpcMotionManager::kVelocityCommandFilterBreakFrequencyKey, ":")))
        << "the key is set explicitly, next to its documentation";
    const absl::StatusOr<scalar_t> breakFrequency = ProceduralMpcMotionManager::loadVelocityCommandFilterBreakFrequency(file);
    ASSERT_TRUE(breakFrequency.ok()) << breakFrequency.status();
    EXPECT_EQ(*breakFrequency, 0.0) << "a robot ships the command filter off until it has been tried in simulation";
  }
}

class VelocityCommandFilterKeyFileTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const std::string& file : files_) std::filesystem::remove(file);
  }

  std::string referenceFileWith(absl::string_view line) {
    const std::string file = (std::filesystem::path(::testing::TempDir()) / absl::StrCat("reference_", files_.size(), ".yaml")).string();
    std::ofstream(file) << "maxDisplacementVelocityX: 1.0\n" << line << "\n";
    files_.push_back(file);
    return file;
  }

  std::vector<std::string> files_;
};

TEST_F(VelocityCommandFilterKeyFileTest, anAbsentKeyIsOffAndAValidOneIsRead) {
  const absl::StatusOr<scalar_t> absent = ProceduralMpcMotionManager::loadVelocityCommandFilterBreakFrequency(referenceFileWith(""));
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_EQ(*absent, 0.0);
  const absl::StatusOr<scalar_t> set =
      ProceduralMpcMotionManager::loadVelocityCommandFilterBreakFrequency(referenceFileWith("velocityCommandFilterBreakFrequency: 2.5"));
  ASSERT_TRUE(set.ok()) << set.status();
  EXPECT_EQ(*set, 2.5);
}

TEST_F(VelocityCommandFilterKeyFileTest, anInvalidValueIsRejectedWithAMessageThatNamesTheKey) {
  for (const absl::string_view value : {"-1.0", "fast", ".inf"}) {
    SCOPED_TRACE(value);
    const absl::StatusOr<scalar_t> loaded = ProceduralMpcMotionManager::loadVelocityCommandFilterBreakFrequency(
        referenceFileWith(absl::StrCat("velocityCommandFilterBreakFrequency: ", value)));
    ASSERT_FALSE(loaded.ok());
    EXPECT_EQ(loaded.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(loaded.status().message(), "velocityCommandFilterBreakFrequency")) << loaded.status();
  }
}

}  // namespace
}  // namespace ocs2::humanoid
