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

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"

#include <algorithm>
#include <cmath>

#include <sstream>

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

std::string ContactPlan::describe() const {
  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(3);
  out << "plan t=" << startTime << (valid ? " valid" : " INVALID") << " J=" << objective << " relaxations=" << numBranchAndBoundNodes
      << " solve=" << solveTime * 1e3 << "ms" << (optimal ? " optimal" : "") << (nodeLimitHit ? " NODE-LIMIT" : "")
      << (timeLimitHit ? " TIME-LIMIT" : "");
  // A step the reach clip had to cut is no longer the deadbeat step, so it is worth seeing in the log.
  if (numClippedSteps > 0) out << " CLIPPED-STEPS=" << numClippedSteps;
  if (!comVelocity.empty()) {
    out << " v0=[" << comVelocity.front().x() << " " << comVelocity.front().y() << "] vN=[" << comVelocity.back().x() << " "
        << comVelocity.back().y() << "]";
  }
  const int N = numIntervals();
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    out << " | " << (foot == 0 ? "L" : "R") << ":";
    int k = 0;
    while (k < N) {
      const bool inContact = contacts[static_cast<size_t>(k)][foot];
      int end = k;
      while (end < N && contacts[static_cast<size_t>(end)][foot] == inContact) ++end;
      out << " " << (inContact ? "C" : "S") << dt * static_cast<scalar_t>(end - k);
      // The step of a swing: the foothold at its touch-down node against the one at its lift-off node.
      if (!inContact && static_cast<size_t>(end) < footholds.size() && static_cast<size_t>(k) < footholds.size()) {
        const vector2_t step = footholds[static_cast<size_t>(end)][foot] - footholds[static_cast<size_t>(k)][foot];
        out << "(" << step.x() << "," << step.y() << ")";
      }
      k = end;
    }
  }
  return out.str();
}

int ContactPlan::intervalIndex(scalar_t time) const {
  if (contacts.empty()) return 0;
  const int k = static_cast<int>(std::floor((time - startTime) / dt + 1e-9));
  return std::clamp(k, 0, numIntervals() - 1);
}

contact_flag_t ContactPlan::contactsAtTime(scalar_t time) const {
  if (contacts.empty()) return makeFeetArray(true);
  return contacts[intervalIndex(time)];
}

void ContactPlan::shiftInTime(scalar_t shift) {
  startTime += shift;
  committedUntil += shift;
}

namespace {
template <typename VALUE_T>
std::optional<VALUE_T> interpolateNodes(const std::vector<VALUE_T>& values, scalar_t startTime, scalar_t dt, scalar_t time) {
  if (values.empty()) return std::nullopt;
  const scalar_t s = std::clamp((time - startTime) / dt, 0.0, static_cast<scalar_t>(values.size() - 1));
  const int k = static_cast<int>(std::floor(s));
  const int k1 = std::min(k + 1, static_cast<int>(values.size()) - 1);
  const scalar_t a = s - static_cast<scalar_t>(k);
  return VALUE_T((1.0 - a) * values[k] + a * values[k1]);
}
}  // namespace

std::optional<scalar_t> ContactPlan::headingAtTime(scalar_t time) const {
  if (!hasHeading()) return std::nullopt;
  return interpolateNodes(heading, startTime, dt, time);
}

std::optional<scalar_t> ContactPlan::headingRateAtTime(scalar_t time) const {
  if (!hasHeading() || headingRate.empty()) return std::nullopt;
  return interpolateNodes(headingRate, startTime, dt, time);
}

std::optional<scalar_t> ContactPlan::footYawAtTime(size_t contactIndex, scalar_t time) const {
  if (!hasHeading() || footYaws.empty()) return std::nullopt;
  const int node = static_cast<int>(std::lround((time - startTime) / dt));
  const int clamped = std::clamp(node, 0, static_cast<int>(footYaws.size()) - 1);
  return footYaws[clamped][contactIndex];
}

std::optional<vector2_t> ContactPlan::comPositionAtTime(scalar_t time) const {
  return interpolateNodes(comPosition, startTime, dt, valid ? time : startTime);
}

std::optional<vector2_t> ContactPlan::comVelocityAtTime(scalar_t time) const {
  return interpolateNodes(comVelocity, startTime, dt, valid ? time : startTime);
}

std::optional<vector2_t> ContactPlan::footholdAtTime(size_t contactIndex, scalar_t time) const {
  if (!valid || footholds.empty()) return std::nullopt;
  const int node = static_cast<int>(std::lround((time - startTime) / dt));
  const int clamped = std::clamp(node, 0, static_cast<int>(footholds.size()) - 1);
  return footholds[clamped][contactIndex];
}

ModeSchedule ContactPlan::toModeSchedule() const {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence;
  if (contacts.empty()) {
    modeSequence.push_back(ModeNumber::STANCE);
    return ModeSchedule(eventTimes, modeSequence);
  }
  modeSequence.push_back(stanceLeg2ModeNumber(contacts.front()));
  for (int k = 1; k < numIntervals(); ++k) {
    const size_t mode = stanceLeg2ModeNumber(contacts[k]);
    if (mode != modeSequence.back()) {
      eventTimes.push_back(startTime + dt * static_cast<scalar_t>(k));
      modeSequence.push_back(mode);
    }
  }
  if (modeSequence.back() != ModeNumber::STANCE) {
    eventTimes.push_back(endTime());
    modeSequence.push_back(ModeNumber::STANCE);
  }
  return ModeSchedule(eventTimes, modeSequence);
}

ModeSchedule mergeModeSchedules(
    const ModeSchedule& applied, const ModeSchedule& plan, scalar_t commitTime, scalar_t lowerBoundTime, scalar_t upperBoundTime) {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence;

  // Part 1: the applied schedule up to commitTime. The leading mode uses the same convention as the loop below (an event
  // at lowerBoundTime itself has passed); ocs2's modeAtTime treats it as not yet passed, which dropped the phase that
  // started exactly there.
  modeSequence.push_back(applied.modeSequence[modeIndexAtTime(applied, lowerBoundTime)]);
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
  const size_t planModeAtCommit = plan.modeAtTime(commitTime + 1e-9);
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

  // The swing planner needs every swing phase to be preceded and followed by a contact phase of the same foot. A leading
  // STANCE placed before the first event guarantees a lift-off, a trailing STANCE guarantees a touch-down.
  if (modeSequence.front() != ModeNumber::STANCE) {
    const scalar_t firstTime = eventTimes.empty() ? lowerBoundTime : std::min(lowerBoundTime, eventTimes.front());
    eventTimes.insert(eventTimes.begin(), firstTime - 1.0);
    modeSequence.insert(modeSequence.begin(), ModeNumber::STANCE);
  }
  if (modeSequence.back() != ModeNumber::STANCE) {
    const scalar_t lastTime = eventTimes.empty() ? upperBoundTime : std::max(upperBoundTime, eventTimes.back() + 1e-3);
    eventTimes.push_back(lastTime);
    modeSequence.push_back(ModeNumber::STANCE);
  }
  // The swing trajectory planner rejects a schedule without events (it needs a preceding and a following phase for every
  // subsystem), so an all-stance schedule is expressed as two stance phases, like the initial schedule of the reference file.
  if (eventTimes.empty()) {
    eventTimes.push_back(upperBoundTime);
    modeSequence.push_back(ModeNumber::STANCE);
  }
  return ModeSchedule(eventTimes, modeSequence);
}

}  // namespace ocs2::humanoid
