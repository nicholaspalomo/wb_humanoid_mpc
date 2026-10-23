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

#include <cmath>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "support/AtlasReferenceStack.h"

/*
 * The command filter of ProceduralMpcMotionManager as the DRC Atlas runs it: configured by the reference file's
 * velocity_command_filter_break_frequency, hot-reloaded with the command limits, run on the solver clock, reset with the
 * manager - and shipped off, so that the reference the MPC follows is the operator's command as before.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kStart = 300.0;       // [s] far from zero: the filter runs on the solver time, not since start-up
constexpr scalar_t kSolvePeriod = 0.01;  // [s]
constexpr scalar_t kRawCommand = 0.5;    // of the forward command limit

class VelocityCommandFilterWiringTest : public ::testing::Test {
 protected:
  /** Solves at kStart, kStart + kSolvePeriod, ... and returns the forward reference of every solve. */
  std::vector<scalar_t> forwardReferences(size_t numSolves, scalar_t start = kStart) {
    vector_t state = stack_.initialState();
    stack_.referenceManager().setTargetTrajectories(stack_.resetTarget(start, state));
    std::vector<scalar_t> references;
    for (size_t k = 0; k < numSolves; ++k) {
      stack_.mpc().run(start + static_cast<scalar_t>(k) * kSolvePeriod, state, ModeNumber::kStance);
      references.push_back(stack_.motionManager().getRampedVelocityCommand()(0));
    }
    return references;
  }

  scalar_t scaledCommand() const { return kRawCommand * maxDisplacementVelocityX_; }

  void SetUp() override {
    const absl::StatusOr<ReferenceSettings> shipped = referenceSettingsFromConfig(stack_.config().reference);
    ASSERT_TRUE(shipped.ok()) << shipped.status();
    maxDisplacementVelocityX_ = shipped->maxDisplacementVelocityX;
    stack_.command(kRawCommand);
  }

  /**
   * A hot reload of the shipped reference file edited by `edit`, as the parameter updater applies one: the file is
   * converted, and only a file that converts reaches the motion manager. The conversion's or the manager's refusal.
   */
  absl::Status reloadShippedReferenceWith(const std::function<void(mpc_config::ReferenceFile&)>& edit) {
    mpc_config::ReferenceFile reference = stack_.config().reference;
    edit(reference);
    const absl::StatusOr<ReferenceSettings> settings = referenceSettingsFromConfig(reference);
    if (!settings.ok()) return settings.status();
    return stack_.motionManager().applyCommandLimits(*settings);
  }

  AtlasReferenceStack stack_;
  scalar_t maxDisplacementVelocityX_ = 0.0;
};

TEST_F(VelocityCommandFilterWiringTest, theShippedFilterIsOffAndTheReferenceIsTheCommand) {
  EXPECT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 0.0);
  // The acceleration ramp after the filter is switched off here, so that the reference shows the filter alone.
  ASSERT_TRUE(reloadShippedReferenceWith([](mpc_config::ReferenceFile& reference) { reference.max_linear_acceleration = 0.0; }).ok());
  ASSERT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 0.0);
  for (const scalar_t reference : forwardReferences(/*numSolves=*/5)) EXPECT_EQ(reference, scaledCommand());
}

TEST_F(VelocityCommandFilterWiringTest, aReloadedBreakFrequencyLagsTheReferenceOnTheSolverClock) {
  const scalar_t breakFrequency = 2.0;
  const scalar_t tau = 1.0 / (2.0 * M_PI * breakFrequency);
  ASSERT_TRUE(reloadShippedReferenceWith([breakFrequency](mpc_config::ReferenceFile& reference) {
                reference.max_linear_acceleration = 0.0;
                reference.velocity_command_filter_break_frequency = breakFrequency;
              }).ok());
  ASSERT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), breakFrequency);

  const std::vector<scalar_t> references = forwardReferences(/*numSolves=*/60);
  for (size_t k = 0; k < references.size(); ++k) {
    const scalar_t elapsed = (kStart + static_cast<scalar_t>(k) * kSolvePeriod) - kStart;
    EXPECT_NEAR(references[k], scaledCommand() * (1.0 - std::exp(-elapsed / tau)), 1.0e-9) << "solve " << k;
  }

  // A reset of the manager restarts the filter from rest, as a fresh manager starts: the command is held, the
  // reference is back at zero.
  stack_.motionManager().reset();
  EXPECT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), breakFrequency) << "the configuration stays";
  const std::vector<scalar_t> afterReset = forwardReferences(/*numSolves=*/2, /*start=*/kStart + 1.0);
  EXPECT_EQ(afterReset[0], 0.0);
  EXPECT_GT(afterReset[1], 0.0);
}

TEST_F(VelocityCommandFilterWiringTest, anInvalidBreakFrequencyFailsTheReloadNamingTheFieldAndKeepsTheFilter) {
  ASSERT_TRUE(reloadShippedReferenceWith([](mpc_config::ReferenceFile& reference) {
                reference.velocity_command_filter_break_frequency = 3.0;
              }).ok());
  ASSERT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 3.0);
  const absl::Status refused =
      reloadShippedReferenceWith([](mpc_config::ReferenceFile& reference) { reference.velocity_command_filter_break_frequency = -1.0; });
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << "a negative break frequency was accepted";
  EXPECT_TRUE(absl::StrContains(refused.message(), "velocity_command_filter_break_frequency")) << refused;
  EXPECT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 3.0);

  // The manager refuses it as well when it is handed one, applying the limits before it and keeping the filter.
  absl::StatusOr<ReferenceSettings> settings = referenceSettingsFromConfig(stack_.config().reference);
  ASSERT_TRUE(settings.ok()) << settings.status();
  settings->velocityCommandFilterBreakFrequency = -1.0;
  EXPECT_EQ(stack_.motionManager().applyCommandLimits(*settings).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(stack_.motionManager().getVelocityCommandFilterBreakFrequency(), 3.0);
}

}  // namespace
}  // namespace ocs2::humanoid
