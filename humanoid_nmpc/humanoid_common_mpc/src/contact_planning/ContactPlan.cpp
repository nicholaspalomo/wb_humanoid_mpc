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

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/gait/BipedModes.h"

namespace ocs2::humanoid {

namespace {
// [s] a continuous-time phase shorter than this is not a phase of the mode
// schedule (it has no duration to execute).
constexpr scalar_t kMinModeDuration = 1.0e-6;
// [s] how far after the last event of a merged schedule its closing STANCE
// begins at the earliest.
constexpr scalar_t kClosingStanceDelay = 1.0e-3;
}  // namespace

std::string ContactPlan::describe() const {
  std::string out = absl::StrFormat(
      "plan t=%.3f%s J=%.3f relaxations=%d solve=%.3fms%s%s%s", startTime,
      valid ? " valid" : " INVALID", objective, numBranchAndBoundNodes,
      solveTime * kMillisecondsPerSecond, optimal ? " optimal" : "",
      nodeLimitHit ? " NODE-LIMIT" : "", timeLimitHit ? " TIME-LIMIT" : "");
  // A step the reach clip had to cut is no longer the deadbeat step, so report
  // it in the log.
  if (numClippedSteps > 0) {
    absl::StrAppend(&out, " CLIPPED-STEPS=", numClippedSteps);
  }
  if (!linvelComRtWorldEwrtWorldXyTraj.empty()) {
    absl::StrAppendFormat(&out, " v0=[%.3f %.3f] vN=[%.3f %.3f]",
                          linvelComRtWorldEwrtWorldXyTraj.front().x(),
                          linvelComRtWorldEwrtWorldXyTraj.front().y(),
                          linvelComRtWorldEwrtWorldXyTraj.back().x(),
                          linvelComRtWorldEwrtWorldXyTraj.back().y());
  }
  const int N = numIntervals();
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    absl::StrAppend(&out, " | ", foot == 0 ? "L" : "R", ":");
    int k = 0;
    while (k < N) {
      const bool inContact = contacts[static_cast<size_t>(k)][foot];
      int end = k;
      while (end < N && contacts[static_cast<size_t>(end)][foot] == inContact) {
        ++end;
      }
      absl::StrAppendFormat(&out, " %s%.3f", inContact ? "C" : "S",
                            dt * static_cast<scalar_t>(end - k));
      // The step of a swing: the foothold at its touch-down node measured
      // against the last node the foot was still STANDING on, which is node k -
      // 1, and not against the lift-off node k itself.
      //
      // This used to read desPosFeetRtWorldEwrtWorldXyTraj[k], and that printed
      // a fabricated (0.000,0.000) for every stepping swing of an H-LIP plan -
      // the default planner in both shipped robots - because the two planners
      // do not agree about what the lift-off node holds. Under `lip_miqp` the
      // foothold state has not jumped yet at node k: FootholdIntegrator
      // integrates p_{k+1} = p_k + dp_k and FootMotionInSwingOnlyConstraint
      // pins dp to zero on contact intervals, so node k still carries the
      // stance the foot is about to leave and the old expression was right
      // there. Under `hlip` HlipContactPlanner computes the landing spot of a
      // swing BEFORE stamping plan.desPosFeetRtWorldEwrtWorldXyTraj over every
      // node of that single-support phase, so node k already carries the
      // LANDING position - and so does node `end`, since the double support
      // that follows leaves the feet where the swing put them. The difference
      // was then identically zero and the log said nothing at all about the
      // step, which defeats the purpose of the line: numClippedSteps tells the
      // operator that the reach clip had to cut a step, and this is where they
      // look to see which one and by how much. It is the same lift-off
      // off-by-one that getPrevDesPosFootRtWorldEwrtWorldXy() exists to keep
      // callers out of.
      //
      // Node k - 1 is a contact node for this foot under BOTH conventions -
      // under `lip_miqp` desPosFeetRtWorldEwrtWorldXyTraj[k-1] equals
      // desPosFeetRtWorldEwrtWorldXyTraj[k] because the foot was in contact
      // through interval k - 1 - so the one expression serves both planners and
      // the existing expectation on a hand-built mixed-integer plan is
      // unchanged. A plan that already begins mid-swing has no pre-lift-off
      // node to measure against; that swing prints its duration and no
      // displacement, because printing a zero there would be exactly the silent
      // lie this is fixing.
      if (!inContact && k > 0 &&
          static_cast<size_t>(end) < desPosFeetRtWorldEwrtWorldXyTraj.size()) {
        // The landing minus the foothold before lift-off.
        const vector2_t desErrPosFootRtWorldEwrtWorldXy =
            desPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(end)][foot] -
            desPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(k - 1)][foot];
        absl::StrAppendFormat(&out, "(%.3f,%.3f)",
                              desErrPosFootRtWorldEwrtWorldXy.x(),
                              desErrPosFootRtWorldEwrtWorldXy.y());
      }
      k = end;
    }
  }
  return out;
}

int ContactPlan::intervalIndex(scalar_t time) const {
  if (contacts.empty()) return 0;
  const int k = static_cast<int>(std::floor(
      (time - startTime) / dt + ContactPlanningConfig::kNodeRoundingTolerance));
  return std::clamp(k, 0, numIntervals() - 1);
}

BipedFeetArray<bool> ContactPlan::contactsAtTime(scalar_t time) const {
  if (hasContinuousPhases()) {
    // The last phase that has begun by `time`; an event at `time` itself has
    // passed, as in intervalIndex().
    size_t phase = 0;
    while (phase + 1 < phaseStartTimes.size() &&
           phaseStartTimes[phase + 1] <= time + kEventTimeTolerance) {
      ++phase;
    }
    return phaseContacts[phase];
  }
  if (contacts.empty()) return makeBipedFeetArray(true);
  return contacts[intervalIndex(time)];
}

void ContactPlan::shiftInTime(scalar_t shift) {
  startTime += shift;
  committedUntil += shift;
  for (scalar_t& phaseStart : phaseStartTimes) phaseStart += shift;
}

namespace {
template <typename VALUE_T>
std::optional<VALUE_T> interpolateNodes(absl::Span<const VALUE_T> values,
                                        scalar_t startTime,
                                        scalar_t dt,
                                        scalar_t time) {
  if (values.empty()) return std::nullopt;
  const scalar_t s = std::clamp((time - startTime) / dt, 0.0,
                                static_cast<scalar_t>(values.size() - 1));
  const int k = static_cast<int>(std::floor(s));
  const int k1 = std::min(k + 1, static_cast<int>(values.size()) - 1);
  const scalar_t a = s - static_cast<scalar_t>(k);
  return VALUE_T((1.0 - a) * values[k] + a * values[k1]);
}
}  // namespace

std::optional<scalar_t> ContactPlan::headingModelYawAtTime(
    scalar_t time) const {
  if (!hasHeading()) return std::nullopt;
  return interpolateNodes<scalar_t>(headingModelYawTraj, startTime, dt, time);
}

scalar_t ContactPlan::getYawHeadingRtWorld(scalar_t time) const {
  if (const std::optional<scalar_t> headingModelYaw =
          headingModelYawAtTime(time);
      headingModelYaw.has_value()) {
    return *headingModelYaw;
  }
  const scalar_t clampedTime =
      std::clamp(time, startTime, std::max(startTime, endTime()));
  return yawHeadingRtWorld + rateYawHeadingRtWorld * (clampedTime - startTime);
}

std::optional<scalar_t> ContactPlan::headingModelYawRateAtTime(
    scalar_t time) const {
  if (!hasHeading() || headingModelYawRateTraj.empty()) return std::nullopt;
  return interpolateNodes<scalar_t>(headingModelYawRateTraj, startTime, dt,
                                    time);
}

std::optional<scalar_t> ContactPlan::getDesEulerzyxFootRtWorldYaw(
    size_t contactIndex, scalar_t time) const {
  if (!hasHeading() || desEulerzyxFeetRtWorldYawTraj.empty()) {
    return std::nullopt;
  }
  const int node = static_cast<int>(std::lround((time - startTime) / dt));
  const int clamped = std::clamp(
      node, 0, static_cast<int>(desEulerzyxFeetRtWorldYawTraj.size()) - 1);
  return desEulerzyxFeetRtWorldYawTraj[clamped][contactIndex];
}

std::optional<scalar_t> ContactPlan::getPrevDesEulerzyxFootRtWorldYaw(
    size_t contactIndex, scalar_t time) const {
  if (!hasHeading() || desEulerzyxFeetRtWorldYawTraj.empty()) {
    return std::nullopt;
  }
  const int node = static_cast<int>(std::lround((time - startTime) / dt)) - 1;
  const int clamped = std::clamp(
      node, 0, static_cast<int>(desEulerzyxFeetRtWorldYawTraj.size()) - 1);
  return desEulerzyxFeetRtWorldYawTraj[clamped][contactIndex];
}

namespace {
// The plan's own reduced model at `time`, clamped to the plan; empty when the
// plan records no pendulum or its per-interval data is inconsistent
// (lipReferenceState's checks).
std::optional<LipState> planReducedModelState(const ContactPlan& plan,
                                              scalar_t time) {
  if (!(plan.omega > 0.0)) return std::nullopt;
  return lipReferenceState(plan, plan.omega,
                           std::clamp(time, plan.startTime, plan.endTime()));
}
}  // namespace

std::optional<vector2_t> ContactPlan::evaluatePosComRtWorldEwrtWorldXy(
    scalar_t time) const {
  const std::optional<LipState> state = planReducedModelState(*this, time);
  if (state.has_value()) return state->posComRtWorldEwrtWorldXy;
  return interpolateNodes<vector2_t>(posComRtWorldEwrtWorldXyTraj, startTime,
                                     dt, valid ? time : startTime);
}

std::optional<vector2_t> ContactPlan::evaluateLinvelComRtWorldEwrtWorldXy(
    scalar_t time) const {
  const std::optional<LipState> state = planReducedModelState(*this, time);
  if (state.has_value()) return state->linvelComRtWorldEwrtWorldXy;
  return interpolateNodes<vector2_t>(linvelComRtWorldEwrtWorldXyTraj, startTime,
                                     dt, valid ? time : startTime);
}

std::optional<vector2_t> ContactPlan::getDesPosFootRtWorldEwrtWorldXy(
    size_t contactIndex, scalar_t time) const {
  if (!valid || desPosFeetRtWorldEwrtWorldXyTraj.empty()) return std::nullopt;
  const int node = static_cast<int>(std::lround((time - startTime) / dt));
  const int clamped = std::clamp(
      node, 0, static_cast<int>(desPosFeetRtWorldEwrtWorldXyTraj.size()) - 1);
  return desPosFeetRtWorldEwrtWorldXyTraj[clamped][contactIndex];
}

std::optional<vector2_t> ContactPlan::getPrevDesPosFootRtWorldEwrtWorldXy(
    size_t contactIndex, scalar_t time) const {
  if (!valid || desPosFeetRtWorldEwrtWorldXyTraj.empty()) return std::nullopt;
  const int node = static_cast<int>(std::lround((time - startTime) / dt)) - 1;
  const int clamped = std::clamp(
      node, 0, static_cast<int>(desPosFeetRtWorldEwrtWorldXyTraj.size()) - 1);
  return desPosFeetRtWorldEwrtWorldXyTraj[clamped][contactIndex];
}

ModeSchedule ContactPlan::toModeSchedule() const {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence;
  if (hasContinuousPhases()) {
    // The phases that last, each with its own start time; one that does not
    // (two feet switching at the same instant) only leaves its neighbors to be
    // merged when they are the same mode.
    for (size_t phase = 0; phase < phaseContacts.size(); ++phase) {
      const scalar_t phaseEnd =
          phase + 1 < phaseStartTimes.size()
              ? phaseStartTimes[phase + 1]
              : std::max(endTime(), phaseStartTimes[phase]);
      const bool isLast = phase + 1 == phaseContacts.size();
      if (!isLast && phaseEnd - phaseStartTimes[phase] < kMinModeDuration) {
        continue;
      }
      const size_t mode = bipedModeOfFootContacts(phaseContacts[phase]);
      if (modeSequence.empty()) {
        modeSequence.push_back(mode);
      } else if (mode != modeSequence.back()) {
        eventTimes.push_back(phaseStartTimes[phase]);
        modeSequence.push_back(mode);
      }
    }
    if (modeSequence.back() != kBipedModeStance) {
      eventTimes.push_back(
          std::max(endTime(), eventTimes.empty()
                                  ? startTime
                                  : eventTimes.back() + kMinModeDuration));
      modeSequence.push_back(kBipedModeStance);
    }
    return ModeSchedule(eventTimes, modeSequence);
  }
  if (contacts.empty()) {
    modeSequence.push_back(kBipedModeStance);
    return ModeSchedule(eventTimes, modeSequence);
  }
  modeSequence.push_back(bipedModeOfFootContacts(contacts.front()));
  for (int k = 1; k < numIntervals(); ++k) {
    const size_t mode = bipedModeOfFootContacts(contacts[k]);
    if (mode != modeSequence.back()) {
      eventTimes.push_back(startTime + dt * static_cast<scalar_t>(k));
      modeSequence.push_back(mode);
    }
  }
  if (modeSequence.back() != kBipedModeStance) {
    eventTimes.push_back(endTime());
    modeSequence.push_back(kBipedModeStance);
  }
  return ModeSchedule(eventTimes, modeSequence);
}

ModeSchedule mergeModeSchedules(const ModeSchedule& applied,
                                const ModeSchedule& plan,
                                scalar_t commitTime,
                                scalar_t lowerBoundTime,
                                scalar_t upperBoundTime) {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence;

  // Part 1: the applied schedule up to commitTime. The leading mode uses the
  // same convention as the loop below (an event at lowerBoundTime itself has
  // passed); ocs2's modeAtTime treats it as not yet passed, which dropped the
  // phase that started exactly there.
  modeSequence.push_back(
      applied.modeSequence[modeIndexAtTime(applied, lowerBoundTime)]);
  for (size_t i = 0; i < applied.eventTimes.size(); ++i) {
    const scalar_t t = applied.eventTimes[i];
    if (t <= lowerBoundTime) continue;
    if (t >= commitTime) break;
    const size_t mode = applied.modeSequence[i + 1];
    if (mode != modeSequence.back()) {
      eventTimes.push_back(t);
      modeSequence.push_back(mode);
    }
  }

  // Part 2: the plan from commitTime on.
  const size_t planModeAtCommit =
      plan.modeAtTime(commitTime + kEventTimeTolerance);
  if (planModeAtCommit != modeSequence.back()) {
    eventTimes.push_back(commitTime);
    modeSequence.push_back(planModeAtCommit);
  }
  for (size_t i = 0; i < plan.eventTimes.size(); ++i) {
    const scalar_t t = plan.eventTimes[i];
    if (t <= commitTime) continue;
    const size_t mode = plan.modeSequence[i + 1];
    if (mode != modeSequence.back()) {
      eventTimes.push_back(t);
      modeSequence.push_back(mode);
    }
  }

  // The swing planner needs every swing phase to be preceded and followed by a
  // contact phase of the same foot. A leading STANCE placed before the first
  // event guarantees a lift-off, a trailing STANCE guarantees a touch-down.
  if (modeSequence.front() != kBipedModeStance) {
    const scalar_t firstTime =
        eventTimes.empty() ? lowerBoundTime
                           : std::min(lowerBoundTime, eventTimes.front());
    eventTimes.insert(eventTimes.begin(), firstTime - 1.0);
    modeSequence.insert(modeSequence.begin(), kBipedModeStance);
  }
  if (modeSequence.back() != kBipedModeStance) {
    const scalar_t lastTime =
        eventTimes.empty()
            ? upperBoundTime
            : std::max(upperBoundTime, eventTimes.back() + kClosingStanceDelay);
    eventTimes.push_back(lastTime);
    modeSequence.push_back(kBipedModeStance);
  }
  // The swing trajectory planner rejects a schedule without events (it needs a
  // preceding and a following phase for every subsystem), so an all-stance
  // schedule is expressed as two stance phases, like the initial schedule of
  // the reference file.
  if (eventTimes.empty()) {
    eventTimes.push_back(upperBoundTime);
    modeSequence.push_back(kBipedModeStance);
  }
  return ModeSchedule(eventTimes, modeSequence);
}

}  // namespace ocs2::humanoid
