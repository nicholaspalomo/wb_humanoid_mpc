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

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "support/AtlasReferenceStack.h"

/*
 * The command filter of ProceduralMpcMotionManager as the DRC Atlas runs it: configured by reference.yaml's
 * velocityCommandFilterBreakFrequency, hot-reloaded with the command limits, run on the solver clock, reset with the
 * manager - and shipped off, so that the reference the MPC follows is the operator's command as before.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStart = 300.0;       // [s] far from zero: the filter runs on the solver time, not since start-up
constexpr scalar_t kSolvePeriod = 0.01;  // [s]
constexpr scalar_t kRawCommand = 0.5;    // of the forward command limit

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

/** A copy of the shipped reference.yaml with the given top-level `key: value` lines rewritten. */
std::string shippedReferenceWith(const std::string& shipped,
                                 const std::vector<std::pair<std::string, std::string>>& values,
                                 const std::string& name) {
  std::string content = readFile(shipped);
  for (const std::pair<std::string, std::string>& entry : values) {
    const std::string keyLine = absl::StrCat("\n", entry.first, ":");
    const std::string::size_type position = content.find(keyLine);
    EXPECT_NE(position, std::string::npos) << entry.first << " is not in " << shipped;
    if (position == std::string::npos) continue;
    const std::string::size_type valueStart = position + keyLine.size();
    content.replace(valueStart, content.find('\n', valueStart) - valueStart, absl::StrCat(" ", entry.second));
  }
  const std::string file = (std::filesystem::path(::testing::TempDir()) / name).string();
  std::ofstream(file) << content;
  return file;
}

class VelocityCommandFilterWiringTest : public ::testing::Test {
 protected:
  /** Solves at kStart, kStart + kSolvePeriod, ... and returns the forward reference of every solve. */
  std::vector<scalar_t> forwardReferences(size_t numSolves, scalar_t start = kStart) {
    vector_t state = stack_.initialState();
    stack_.referenceManager().setTargetTrajectories(stack_.resetTarget(start, state));
    std::vector<scalar_t> references;
    for (size_t k = 0; k < numSolves; ++k) {
      stack_.mpc().run(start + static_cast<scalar_t>(k) * kSolvePeriod, state, ModeNumber::STANCE);
      references.push_back(stack_.motionManager().getRampedVelocityCommand()(0));
    }
    return references;
  }

  scalar_t scaledCommand() const { return kRawCommand * maxDisplacementVelocityX_; }

  void SetUp() override {
    PropertyTree pt;
    loadData::readPropertyTree(stack_.referenceFile(), pt);
    maxDisplacementVelocityX_ = pt.get<scalar_t>("maxDisplacementVelocityX");
    stack_.command(kRawCommand);
  }

  AtlasReferenceStack stack_;
  scalar_t maxDisplacementVelocityX_ = 0.0;
};

TEST_F(VelocityCommandFilterWiringTest, theShippedFilterIsOffAndTheReferenceIsTheCommand) {
  EXPECT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 0.0);
  // The acceleration ramp after the filter is switched off here, so that the reference shows the filter alone.
  stack_.motionManager().reloadCommandLimits(
      shippedReferenceWith(stack_.referenceFile(), {{"maxLinearAcceleration", "0.0"}}, "shipped_filter.yaml"));
  ASSERT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 0.0);
  for (const scalar_t reference : forwardReferences(/*numSolves=*/5)) EXPECT_EQ(reference, scaledCommand());
}

TEST_F(VelocityCommandFilterWiringTest, aReloadedBreakFrequencyLagsTheReferenceOnTheSolverClock) {
  const scalar_t breakFrequency = 2.0;
  const scalar_t tau = 1.0 / (2.0 * M_PI * breakFrequency);
  stack_.motionManager().reloadCommandLimits(shippedReferenceWith(
      stack_.referenceFile(), {{"maxLinearAcceleration", "0.0"}, {"velocityCommandFilterBreakFrequency", "2.0"}}, "filter_on.yaml"));
  ASSERT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), breakFrequency);

  const std::vector<scalar_t> references = forwardReferences(/*numSolves=*/60);
  for (size_t k = 0; k < references.size(); ++k) {
    const scalar_t elapsed = (kStart + static_cast<scalar_t>(k) * kSolvePeriod) - kStart;
    EXPECT_NEAR(references[k], scaledCommand() * (1.0 - std::exp(-elapsed / tau)), 1e-9) << "solve " << k;
  }

  // A reset of the manager restarts the filter from rest, as a fresh manager starts: the command is held, the
  // reference is back at zero.
  stack_.motionManager().reset();
  EXPECT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), breakFrequency) << "the configuration stays";
  const std::vector<scalar_t> afterReset = forwardReferences(/*numSolves=*/2, /*start=*/kStart + 1.0);
  EXPECT_EQ(afterReset[0], 0.0);
  EXPECT_GT(afterReset[1], 0.0);
}

TEST_F(VelocityCommandFilterWiringTest, anInvalidBreakFrequencyFailsTheReloadNamingTheKeyAndKeepsTheFilter) {
  stack_.motionManager().reloadCommandLimits(
      shippedReferenceWith(stack_.referenceFile(), {{"velocityCommandFilterBreakFrequency", "3.0"}}, "filter_valid.yaml"));
  ASSERT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 3.0);
  try {
    stack_.motionManager().reloadCommandLimits(
        shippedReferenceWith(stack_.referenceFile(), {{"velocityCommandFilterBreakFrequency", "-1.0"}}, "filter_invalid.yaml"));
    ADD_FAILURE() << "a negative break frequency was accepted";
  } catch (const std::invalid_argument& error) {
    EXPECT_TRUE(absl::StrContains(error.what(), "velocityCommandFilterBreakFrequency")) << error.what();
  }
  EXPECT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 3.0);
}

}  // namespace
}  // namespace ocs2::humanoid
