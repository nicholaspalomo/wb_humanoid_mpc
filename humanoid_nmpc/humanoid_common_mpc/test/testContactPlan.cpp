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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kTol = 1e-6;

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.dt = 0.1;
  config.numNodes = 12;
  config.commitTime = 0.25;
  config.comHeight = 0.85;
  config.minSwingDuration = 0.3;
  config.maxSwingDuration = 0.5;
  config.minContactDuration = 0.15;
  config.maxContactDuration = 0.0;
  config.minDoubleSupportDuration = 0.1;
  config.maxBranchAndBoundNodes = 3000;
  config.maxSolveTime = 10.0;
  config.localSearchMaxTime = 2.0;
  config.verbose = false;
  config.validate();
  return config;
}

size_t modeWithSwinging(size_t foot) {
  contact_flag_t flags = makeFeetArray(true);
  flags[foot] = false;
  return stanceLeg2ModeNumber(flags);
}

/** The planner input the reference manager builds from the executed schedule (ContactPlanningReferenceManager::makePlannerInput). */
ContactPlannerInput makeInput(const ModeSchedule& schedule, scalar_t time, const ContactPlanningConfig& config) {
  ContactPlannerInput input;
  input.time = time;
  input.velocityCommand = vector2_t(0.3, 0.0);
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(0.05, -0.125);
  input.contacts = contactFlagsAtTime(schedule, time);
  const feet_array_t<scalar_t> phaseStarts = contactPhaseStartTimes(schedule, time);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    input.phaseElapsedTime[foot] = std::isfinite(phaseStarts[foot]) ? time - phaseStarts[foot] : 10.0;
  }
  input.committedUntil = commitBoundaryForSchedule(schedule, time, config.commitTime);
  const int maxCommitted = config.numNodes - 1;
  input.committedContacts = committedContactsForPlanner(schedule, time, config.dt, maxCommitted, input.committedUntil);
  input.committedPhaseStartTimes = committedPhaseStartsForPlanner(schedule, time, config.dt, maxCommitted, input.committedUntil);
  return input;
}

struct Phase {
  size_t foot;
  bool inContact;
  scalar_t start;
  scalar_t end;
};

/** Contact and swing phases of every foot in `schedule` that begin and end inside [from, to]. */
std::vector<Phase> phasesInside(const ModeSchedule& schedule, scalar_t from, scalar_t to) {
  std::vector<Phase> phases;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    std::optional<Phase> current;
    for (size_t i = 0; i < schedule.eventTimes.size(); ++i) {
      const scalar_t t = schedule.eventTimes[i];
      const bool before = modeNumber2StanceLeg(schedule.modeSequence[i])[foot];
      const bool after = modeNumber2StanceLeg(schedule.modeSequence[i + 1])[foot];
      if (before == after) continue;
      if (current.has_value()) {
        current->end = t;
        if (current->start >= from - kTol && current->end <= to + kTol) phases.push_back(*current);
      }
      current = Phase{foot, after, t, t};
    }
  }
  return phases;
}

/** Double supports of `schedule` that begin and end inside [from, to] as (start, end). */
std::vector<std::pair<scalar_t, scalar_t>> doubleSupportsInside(const ModeSchedule& schedule, scalar_t from, scalar_t to) {
  std::vector<std::pair<scalar_t, scalar_t>> supports;
  for (size_t i = 0; i + 1 < schedule.modeSequence.size(); ++i) {
    if (schedule.modeSequence[i + 1] != ModeNumber::STANCE || i + 1 >= schedule.eventTimes.size()) continue;
    const scalar_t start = schedule.eventTimes[i];
    const scalar_t end = schedule.eventTimes[i + 1];
    if (start >= from - kTol && end <= to + kTol) supports.emplace_back(start, end);
  }
  return supports;
}

void printSchedule(const ModeSchedule& schedule) {
  std::cout << "  schedule:";
  for (size_t i = 0; i < schedule.eventTimes.size(); ++i) {
    const contact_flag_t c = modeNumber2StanceLeg(schedule.modeSequence[i]);
    std::cout << " [" << c[0] << c[1] << "] " << schedule.eventTimes[i];
  }
  const contact_flag_t c = modeNumber2StanceLeg(schedule.modeSequence.back());
  std::cout << " [" << c[0] << c[1] << "]\n";
}

/**
 * Plans from the executed schedule at `time`, merges the plan at its commit boundary as the reference manager does, and
 * checks that every phase the merge produces honours the configured minima in real time.
 */
void expectMergedScheduleHonoursMinimumDurations(const ModeSchedule& applied, scalar_t time, const ContactPlanningConfig& config) {
  LipContactPlanner planner(config);
  const ContactPlannerInput input = makeInput(applied, time, config);
  const ContactPlan plan = planner.plan(input);
  ASSERT_TRUE(plan.valid);
  const scalar_t commitTime = std::max(time, plan.committedUntil);
  const ModeSchedule merged = mergeModeSchedules(applied, plan.toModeSchedule(), commitTime, time - 1.2, plan.endTime() + 1.2);
  printSchedule(merged);

  // Only phases the horizon does not cut off can be judged; the plan may not lift a foot it cannot land in time anyway.
  for (const Phase& phase : phasesInside(merged, time, plan.endTime())) {
    const scalar_t duration = phase.end - phase.start;
    if (phase.inContact) {
      EXPECT_GE(duration, config.minContactDuration - kTol) << "foot " << phase.foot << " contact from " << phase.start;
    } else {
      EXPECT_GE(duration, config.minSwingDuration - kTol) << "foot " << phase.foot << " swing from " << phase.start;
      EXPECT_LE(duration, config.maxSwingDuration + kTol) << "foot " << phase.foot << " swing from " << phase.start;
    }
  }
  const auto supports = doubleSupportsInside(merged, time, plan.endTime());
  EXPECT_FALSE(supports.empty()) << "a walking plan transfers the weight at least once inside the horizon";
  for (const auto& [start, end] : supports) {
    EXPECT_GE(end - start, config.minDoubleSupportDuration - kTol) << "double support from " << start << " to " << end;
  }
}

}  // namespace

TEST(ContactPlanMerge, TouchDownAtTheCommitBoundaryDoesNotShortenTheFollowingPhases) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // The right foot lands at 0.97 s, 0.07 s into the planner node [0.9, 1.0). The commit boundary of a plan made at
  // 0.7 s is that touch-down, so the node straddles it and is committed with the foot in contact. Counted from the node
  // start, the double support was one node old at 1.0 s and the left foot lifted there: 0.03 s of double support in
  // the merged schedule, against the 0.1 s minimum, and a contact of the right foot that could end 0.13 s after it
  // landed against the 0.15 s minimum.
  const ContactPlanningConfig config = makeConfig();
  const ModeSchedule applied({0.57, 0.97}, {ModeNumber::STANCE, modeWithSwinging(1), ModeNumber::STANCE});
  expectMergedScheduleHonoursMinimumDurations(applied, 0.7, config);
}

TEST(ContactPlanMerge, TouchDownInsideTheCommitWindowDoesNotShortenTheFollowingPhases) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // The touch-down at 0.92 s lies inside the commit window of a plan made at 0.7 s (boundary 0.95 s): the node
  // [0.9, 1.0) straddles the boundary and reports the landed state, the landing itself is 0.02 s into the node.
  const ContactPlanningConfig config = makeConfig();
  const ModeSchedule applied({0.52, 0.92}, {ModeNumber::STANCE, modeWithSwinging(1), ModeNumber::STANCE});
  expectMergedScheduleHonoursMinimumDurations(applied, 0.7, config);
}

TEST(ContactPlanMerge, TouchDownAfterTheMidpointOfACommittedNodeIsCountedFromItsEvent) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  // The touch-down at 0.88 s lies after the midpoint of the node [0.8, 0.9), which is committed as "swinging"; the next
  // node reports the contact, 0.02 s before its start. Counting from that node's start would make the contact appear
  // 0.02 s younger than it is, which is harmless for the minima but shows that the event time, not the node, is used.
  const ContactPlanningConfig config = makeConfig();
  const ModeSchedule applied({0.48, 0.88}, {ModeNumber::STANCE, modeWithSwinging(1), ModeNumber::STANCE});
  expectMergedScheduleHonoursMinimumDurations(applied, 0.7, config);
}

TEST(ContactPlanMerge, GridAlignedTouchDownIsUnchanged) {
  if (N_CONTACTS < 2) GTEST_SKIP() << "needs a second foot";
  const ContactPlanningConfig config = makeConfig();
  const ModeSchedule applied({0.6, 1.0}, {ModeNumber::STANCE, modeWithSwinging(1), ModeNumber::STANCE});
  expectMergedScheduleHonoursMinimumDurations(applied, 0.7, config);
}

}  // namespace ocs2::humanoid
