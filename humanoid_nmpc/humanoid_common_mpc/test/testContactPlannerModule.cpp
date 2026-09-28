/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"

namespace ocs2::humanoid {

namespace {

using Throttle = ContactPlannerModule::SnapshotThrottle;

/** A steady-clock instant `seconds` after the epoch of the test. */
Throttle::Clock::time_point at(scalar_t seconds) {
  return Throttle::Clock::time_point{} + std::chrono::duration_cast<Throttle::Clock::duration>(std::chrono::duration<scalar_t>(seconds));
}

constexpr std::chrono::duration<scalar_t> kPeriod(0.1);  // planningFrequency 10 Hz

}  // namespace

TEST(SnapshotThrottle, FirstPostIsAlwaysAllowedAndShortPlansAreCappedAtThePeriod) {
  Throttle throttle;
  EXPECT_TRUE(throttle.allows(at(0.0), kPeriod));
  throttle.posted(at(0.0));
  throttle.taken();  // the worker starts at once and finishes well within the period
  EXPECT_FALSE(throttle.allows(at(0.03), kPeriod)) << "the period is a cap on the planning rate";
  EXPECT_FALSE(throttle.allows(at(0.09), kPeriod));
  EXPECT_TRUE(throttle.allows(at(0.1), kPeriod));
}

/**
 * A snapshot posted while the worker is busy is dropped when the worker hands its plan over (the next plan has to start
 * from a schedule that contains it). Counting the period from that dropped post held the next snapshot back for a
 * whole period while the worker sat idle, so with plans of 0.1-0.2 s the planner ran at 5 Hz instead of 10. The period
 * counts from the snapshot the worker actually took, or from the one still waiting for it.
 */
TEST(SnapshotThrottle, CountsThePeriodFromTheSnapshotTheWorkerTookNotFromADroppedPost) {
  Throttle throttle;
  throttle.posted(at(0.0));
  throttle.taken();  // a plan that takes 0.15 s
  EXPECT_TRUE(throttle.allows(at(0.1), kPeriod));
  throttle.posted(at(0.1));  // posted while the worker is busy: it waits
  EXPECT_FALSE(throttle.allows(at(0.12), kPeriod)) << "a waiting snapshot holds the period";
  throttle.dropped();  // the hand-over at 0.15 drops it
  EXPECT_TRUE(throttle.allows(at(0.15), kPeriod)) << "the last snapshot the worker took is 0.15 s old: post at once";
  throttle.posted(at(0.15));
  throttle.taken();
  EXPECT_FALSE(throttle.allows(at(0.2), kPeriod)) << "and the cap holds again from the new snapshot";
  EXPECT_TRUE(throttle.allows(at(0.25), kPeriod));
}

TEST(SnapshotThrottle, ResetForgetsTheHistory) {
  Throttle throttle;
  throttle.posted(at(5.0));
  throttle.taken();
  EXPECT_FALSE(throttle.allows(at(5.01), kPeriod));
  throttle = Throttle{};
  EXPECT_TRUE(throttle.allows(at(5.01), kPeriod)) << "a restarted worker takes the first snapshot regardless of the old timestamps";
}

namespace {

/** The shipped H-LIP cadence, whose first step out of a standstill fits the reach. */
ContactPlanningConfig makeHlipConfig() {
  ContactPlanningConfig config;
  config.planner.type = "hlip";
  config.planner.dt = 0.025;
  config.planner.numNodes = 56;
  config.planner.commitTime = 0.05;
  config.planner.runInBackgroundThread = false;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.25;
  config.shared.gaitLimits.maxSwingDuration = 0.35;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.hlip.sspDuration = 0.25;
  config.hlip.dspDuration = 0.05;
  config.hlip.stepWidth = 0.25;
  config.hlip.maxStepWidth = 0.45;
  return config;
}

}  // namespace

TEST(ContactPlannerModuleReload, AReloadThatChangesNothingTheSummaryReportsIsSilent) {
  const ContactPlanningConfig config = makeHlipConfig();
  EXPECT_FALSE(ContactPlannerModule::reloadSummary(config, config).has_value()) << "the GUI re-sends the file on every slider move";
  ContactPlanningConfig faster = config;
  faster.planner.planningFrequency = 2.0 * config.planner.planningFrequency;
  EXPECT_FALSE(ContactPlannerModule::reloadSummary(config, faster).has_value());
}

TEST(ContactPlannerModuleReload, RetuningTheCadenceReprintsTheSummaryWithItsStartUpCheck) {
  // The H-LIP summary ends with the start-up check - does the first step out of a standstill fit hlip.maxStepWidth? -
  // and every key it depends on is a hot-reloadable slider. The summary used to be re-printed only for a change of the
  // formulation, the grid or the planner type, so a cadence retuned from the GUI was never checked: raising
  // hlip.sspDuration from 0.25 s to 0.35 s makes the first step need 0.52 m against a 0.45 m reach, and nothing said so.
  const ContactPlanningConfig fits = makeHlipConfig();
  ContactPlanningConfig longSwing = fits;
  longSwing.hlip.sspDuration = 0.35;

  const std::optional<std::string> reloaded = ContactPlannerModule::reloadSummary(fits, longSwing);
  ASSERT_TRUE(reloaded.has_value()) << "a cadence change must be re-checked and re-printed";
  EXPECT_NE(reloaded->find("single support 0.35"), std::string::npos) << *reloaded;
  EXPECT_NE(reloaded->find("DOES NOT FIT"), std::string::npos) << *reloaded;
  // It is the same summary the start-up prints, not a partial one.
  const absl::StatusOr<std::string> startUp = contactPlannerSummary(longSwing);
  ASSERT_TRUE(startUp.ok());
  EXPECT_EQ(*reloaded, *startUp);

  // And back: the fitting cadence reports that it fits.
  const std::optional<std::string> restored = ContactPlannerModule::reloadSummary(longSwing, fits);
  ASSERT_TRUE(restored.has_value());
  EXPECT_NE(restored->find("(fits)"), std::string::npos) << *restored;

  // Each of the other keys the check reads re-prints it too.
  ContactPlanningConfig narrowReach = fits;
  narrowReach.hlip.maxStepWidth = 0.35;
  EXPECT_TRUE(ContactPlannerModule::reloadSummary(fits, narrowReach).has_value()) << "hlip.maxStepWidth";
  ContactPlanningConfig longerTransfer = fits;
  longerTransfer.hlip.dspDuration = 0.1;
  EXPECT_TRUE(ContactPlannerModule::reloadSummary(fits, longerTransfer).has_value()) << "hlip.dspDuration";
  ContactPlanningConfig wider = fits;
  wider.hlip.stepWidth = 0.3;
  EXPECT_TRUE(ContactPlannerModule::reloadSummary(fits, wider).has_value()) << "hlip.stepWidth";
}

TEST(ContactPlannerModuleReload, AStructuralChangeAlwaysReprints) {
  const ContactPlanningConfig config = makeHlipConfig();
  ContactPlanningConfig finerGrid = config;
  finerGrid.planner.dt = 0.02;
  EXPECT_TRUE(ContactPlannerModule::reloadSummary(config, finerGrid).has_value());
  ContactPlanningConfig otherPlanner = config;
  otherPlanner.planner.type = "lip_miqp";
  EXPECT_TRUE(ContactPlannerModule::reloadSummary(config, otherPlanner).has_value());
}

}  // namespace ocs2::humanoid
