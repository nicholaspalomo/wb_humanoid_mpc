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

#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"

namespace ocs2::humanoid {
namespace {

using Rotation2 = Eigen::Matrix<scalar_t, 2, 2>;

/**
 * [s] a phase shorter than this that covers no interval is not emitted: a swing whose touch-down is due within it has
 * landed where the foot is, and the next phase starts at once.
 */
constexpr scalar_t kMinPhaseDuration = 1.0e-6;

/** [m] a step cut by less than this is not counted as clipped. */
constexpr scalar_t kStepClipTolerance = 1.0e-9;

/** [s] slack on the window an executed event time must fall in to be taken as the switch between two committed nodes. */
constexpr scalar_t kSwitchTimeTolerance = 1.0e-9;

/** Steps the start-up check rolls the reduced model forward for, and the tolerance [m] it calls a width settled at. */
constexpr int kStartUpSteps = 12;
constexpr scalar_t kStartUpSettledTolerance = 1.0e-3;

Rotation2 rotation(scalar_t yaw) {
  const scalar_t cosine = std::cos(yaw);
  const scalar_t sine = std::sin(yaw);
  Rotation2 matrix;
  matrix << cosine, -sine, sine, cosine;
  return matrix;
}

contact_flag_t stanceOnly(size_t stanceFoot) {
  contact_flag_t contacts = makeFeetArray(false);
  contacts[stanceFoot] = true;
  return contacts;
}

size_t otherFoot(size_t foot) {
  return foot == kContactLeftIndex ? kContactRightIndex : kContactLeftIndex;
}

bool bothFeetDown(const contact_flag_t& contacts) {
  return contacts[kContactLeftIndex] && contacts[kContactRightIndex];
}

/** The foot in flight in a contact state, or -1 when both feet are down (and when neither is). */
int swingFootOf(const contact_flag_t& contacts) {
  if (bothFeetDown(contacts)) return -1;
  if (!contacts[kContactLeftIndex] && !contacts[kContactRightIndex]) return -1;
  return contacts[kContactLeftIndex] ? static_cast<int>(kContactRightIndex) : static_cast<int>(kContactLeftIndex);
}

/**
 * [s] how long the robot has already been in the contact state it is in at `input.time`.
 *
 * In double support the newly landed foot is the one that dates the phase, so the smaller of the two elapsed times is
 * the age of the double support; in single support it is how long the swinging foot has been in flight.
 */
scalar_t elapsedInCurrentPhase(const ContactPlannerInput& input) {
  const int swingFoot = swingFootOf(input.contacts);
  if (swingFoot < 0) {
    return std::min(input.phaseElapsedTime[kContactLeftIndex], input.phaseElapsedTime[kContactRightIndex]);
  }
  return input.phaseElapsedTime[static_cast<size_t>(swingFoot)];
}

/**
 * [s] when the executed schedule switched `foot` between the samples of committed nodes `node - 1` and `node` (for
 * node 0, between `input.time` and its sample).
 *
 * The committed contacts are samples of the executed schedule, one per node, so they say in which node a switch
 * happened but not when. The time is in `input.committedPhaseStartTimes`: the start of the contact phase the foot is in
 * at the node's sample, which is exactly that switch when it lies between the two samples. When it does not, or the
 * input carries no times, the switch is placed at the node start, which is all the grid itself can say.
 */
scalar_t committedSwitchTime(
    const ContactPlannerInput& input, const std::vector<scalar_t>& sampleTimes, int node, size_t foot, scalar_t dt) {
  const scalar_t nodeStart = input.time + dt * static_cast<scalar_t>(node);
  const size_t index = static_cast<size_t>(node);
  if (index >= input.committedPhaseStartTimes.size() || index >= sampleTimes.size()) return nodeStart;
  const scalar_t switchTime = input.committedPhaseStartTimes[index][foot];
  const scalar_t previousSample = node == 0 ? input.time : sampleTimes[index - 1];
  const bool betweenTheSamples = std::isfinite(switchTime) && switchTime > previousSample - kSwitchTimeTolerance &&
                                 switchTime <= sampleTimes[index] + kSwitchTimeTolerance;
  return betweenTheSamples ? switchTime : nodeStart;
}

/** The first step from which every width in `widths` is `stepWidth`, or -1 when the widths never settle there. */
int settlingStep(const std::vector<scalar_t>& widths, scalar_t stepWidth) {
  int settled = -1;
  for (int step = static_cast<int>(widths.size()) - 1; step >= 0; --step) {
    if (std::abs(widths[static_cast<size_t>(step)] - stepWidth) > kStartUpSettledTolerance) break;
    settled = step;
  }
  // A width that is right only on the very last step has not been shown to stay there.
  return settled >= 0 && settled + 2 <= static_cast<int>(widths.size()) ? settled : -1;
}

/** The start-up line of the formulation summary: the first-step demand and what the reduced model makes of it. */
std::string startUpVerdict(const ContactPlanningConfig& config, scalar_t demand) {
  const HlipParameters& params = config.hlip;
  if (demand <= params.maxStepWidth) return " (fits)\n";
  const std::vector<scalar_t> widths = HlipContactPlanner::startUpLateralWidths(config, kStartUpSteps);
  const int settled = settlingStep(widths, params.stepWidth);
  const std::vector<scalar_t> firstWidths(widths.begin(), widths.begin() + std::min<ptrdiff_t>(6, static_cast<ptrdiff_t>(widths.size())));
  const std::string rolledOut = absl::StrJoin(firstWidths, ", ");
  if (settled >= 0) {
    return absl::StrCat(
        " <-- DOES NOT FIT: the first step is clipped, which costs the deadbeat property. The reduced model still recovers the "
        "nominal width after ",
        settled, " steps (widths ", rolledOut, ", ... m); shorten hlip.ssp_duration or raise hlip.max_step_width for margin.\n");
  }
  return absl::StrCat(
      " <-- DOES NOT FIT: the first step is clipped, and the reduced model then locks into an alternating wide/narrow limit "
      "cycle it never leaves (widths ",
      rolledOut, ", ... m). Shorten hlip.ssp_duration or raise hlip.max_step_width.\n");
}

}  // namespace

HlipModel HlipContactPlanner::makeModel(const ContactPlanningConfig& config) {
  return HlipModel(config.hlip.sspDuration, config.hlip.dspDuration, config.pendulumHeight(), config.shared.gravity);
}

HlipContactPlanner::HlipContactPlanner(ContactPlanningConfig config)
    : config_(std::move(config)), model_(makeModel(config_)), blend_(config_.hlip.blend) {}

absl::Status HlipContactPlanner::setConfig(const ContactPlanningConfig& config) {
  // The H-LIP model and the blend CHECK-fail on the values validateStatus() rejects, so nothing unvalidated is built.
  RETURN_IF_ERROR(config.validateStatus());
  config_ = config;
  model_ = makeModel(config_);
  blend_ = HlipStandingBlend(config_.hlip.blend);
  return absl::OkStatus();
}

size_t HlipContactPlanner::nextSwingFoot(const ContactPlannerInput& input) {
  if (input.lastSwungFoot >= 0) return otherFoot(static_cast<size_t>(input.lastSwungFoot));
  // Nothing has swung yet: the foot that has been in contact the longest has had the most time to be unloaded.
  return input.phaseElapsedTime[kContactLeftIndex] >= input.phaseElapsedTime[kContactRightIndex] ? kContactLeftIndex : kContactRightIndex;
}

std::optional<scalar_t> HlipContactPlanner::zeroCommandOrbitLateralVelocity(const ContactPlannerInput& input) const {
  const scalar_t stepWidth = config_.hlip.stepWidth;
  // `first` precedes the step that places the left foot (standing on the right), `second` the one that places the right.
  const std::pair<HlipModel::State, HlipModel::State> orbit = model_.periodTwoOrbit(stepWidth, -stepWidth);
  const int swingFoot = swingFootOf(input.contacts);
  if (swingFoot >= 0) {
    // The single support began at the lift-off after the impact that placed the stance foot: that step's pre-impact
    // state, carried through the impact and the double support, then flowed for as long as the swing has been in flight.
    const bool placesLeftFoot = swingFoot == static_cast<int>(kContactLeftIndex);
    const HlipModel::State& beforeStanceStep = placesLeftFoot ? orbit.second : orbit.first;
    const scalar_t stanceStep = placesLeftFoot ? -stepWidth : stepWidth;
    const HlipModel::State atLiftOff =
        HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(beforeStanceStep, stanceStep), model_.dspDuration());
    const scalar_t inFlight = std::clamp(input.phaseElapsedTime[static_cast<size_t>(swingFoot)], 0.0, model_.sspDuration());
    return model_.flowSingleSupport(atLiftOff, inFlight)(1);
  }
  if (bothFeetDown(input.contacts) && input.lastSwungFoot >= 0) {
    // After the touch-down of the last swing the orbit drifts at that step's pre-impact velocity.
    return (input.lastSwungFoot == static_cast<int>(kContactLeftIndex) ? orbit.first : orbit.second)(1);
  }
  return std::nullopt;
}

vector2_t HlipContactPlanner::blendVelocity(const ContactPlannerInput& input) const {
  const Rotation2 world_R_heading = rotation(input.yaw);
  vector2_t velocity = world_R_heading.transpose() * input.comVelocity;
  const std::optional<scalar_t> orbitVelocity = zeroCommandOrbitLateralVelocity(input);
  if (orbitVelocity.has_value()) {
    // Only what lies outside the range between rest and the sway of stepping in place counts.
    const scalar_t low = std::min(0.0, *orbitVelocity);
    const scalar_t high = std::max(0.0, *orbitVelocity);
    velocity(1) = velocity(1) > high ? velocity(1) - high : (velocity(1) < low ? velocity(1) - low : 0.0);
  }
  return velocity;
}

scalar_t HlipContactPlanner::blendWeight(const ContactPlannerInput& input) const {
  const Rotation2 world_R_heading = rotation(input.yaw);
  const vector2_t commandInHeading = world_R_heading.transpose() * input.velocityCommand;
  return blend_.weight(commandInHeading, input.headingRateCommand, blendVelocity(input));
}

bool HlipContactPlanner::isWalking(const ContactPlannerInput& input) const {
  const Rotation2 world_R_heading = rotation(input.yaw);
  const vector2_t commandInHeading = world_R_heading.transpose() * input.velocityCommand;
  return blend_.isWalking(commandInHeading, input.headingRateCommand, blendVelocity(input));
}

scalar_t HlipContactPlanner::headingAt(const ContactPlannerInput& input, scalar_t time, scalar_t alpha) const {
  // `yaw` is the frame the geometry is planned in: the whole-body heading with the heading model, the base yaw
  // without it (ContactPlanningReferenceManager::makePlannerInput). `heading` is only filled with the model, so the
  // frame must come from `yaw` or a robot without the model would plan its footholds in the world frame.
  return input.yaw + alpha * input.headingRateCommand * (time - input.time);
}

std::vector<HlipContactPlanner::GaitPhase> HlipContactPlanner::buildGait(const ContactPlannerInput& input, bool walking) const {
  const scalar_t dt = config_.planner.dt;
  const int numNodes = config_.planner.numNodes;
  const scalar_t horizonEnd = input.time + config_.horizon();
  const scalar_t sspDuration = model_.sspDuration();
  const scalar_t dspDuration = model_.dspDuration();
  std::vector<GaitPhase> gait;

  // NOTE that standing is NOT handled by returning here. It used to be: `!walking` produced one all-stance phase over
  // the whole horizon and returned before the committed window below, which made the standing plan the one plan this
  // planner produces that is not built from input.committedContacts - in flat contradiction of the comment on that
  // window, which calls those intervals "not the planner's to choose".
  //
  // The consequence was not cosmetic. A lone stance phase has swingFoot == -1, so plan() never enters its
  // single-support branch, `feet` is never advanced past its initialization from input.footPositions, and
  // plan.footholds[node] therefore published the measured position of whichever foot was IN THE AIR as that swing's
  // landing target - a target in mid-flight. The blend crosses into standing on a released stick, and single support
  // is most of a stride, so a foot is airborne at that moment more often than not. The plan is still accepted,
  // because activatePendingPlan only checks agreement with the swings in flight at mergeTime, by which point the
  // commit boundary has advanced past that touch-down.
  //
  // So the committed window is replayed first for standing exactly as for walking, and only the REMAINDER of the
  // horizon becomes the stance phase. A swing in flight is then a real single-support phase, plan() rolls it out and
  // deadbeatStep() places its landing spot under the step-width clip like any other.

  // Every boundary after the committed window is assigned to the nearest node, once, here.
  const std::function<int(scalar_t)> nearestNode = [&input, dt, numNodes](scalar_t time) {
    return std::clamp(static_cast<int>(std::lround((time - input.time) / dt)), 0, numNodes);
  };
  const std::function<scalar_t(scalar_t, scalar_t, int, const contact_flag_t&, int)> append =
      [&gait](scalar_t startTime, scalar_t endTime, int endNode, const contact_flag_t& contacts, int swingFoot) {
        const int startNode = gait.empty() ? 0 : gait.back().endNode;
        endNode = std::max(endNode, startNode);
        // A phase with no duration is dropped, unless it covers intervals: the grid must stay covered.
        if (endTime - startTime <= kMinPhaseDuration && endNode == startNode) return startTime;
        // A phase that merely continues the one before it is the SAME phase. This matters beyond tidiness: the
        // deadbeat step is evaluated once per single-support phase, at the pre-impact state that phase ends in, so a
        // swing split into two adjacent phases would have its landing spot computed twice - the first time from a
        // pre-impact state flowed only part of the way.
        if (!gait.empty() && gait.back().contacts == contacts && gait.back().swingFoot == swingFoot) {
          gait.back().endTime = std::max(gait.back().endTime, endTime);
          gait.back().endNode = endNode;
          return gait.back().endTime;
        }
        GaitPhase phase;
        phase.startTime = startTime;
        phase.endTime = std::max(startTime, endTime);
        phase.endNode = endNode;
        phase.contacts = contacts;
        phase.swingFoot = swingFoot;
        gait.push_back(phase);
        return phase.endTime;
      };

  // 1. The committed window, replayed from the executed schedule.
  //
  // These intervals are not the planner's to choose: the reference manager merges the applied schedule over them
  // anyway, and a plan that disagreed with it there would be dropped as inconsistent. They used to be stamped over
  // `plan.contacts` at the very END of plan(), AFTER the center of mass and the footholds had been rolled out against
  // a gait built as if the window were free. The two then described different gaits. The worst case is the one the
  // robot starts every walk from: out of a long stance, `dspDuration - elapsed` is negative, so the nominal cadence
  // lifted a foot at node 0 while the commit window held both feet down for the first two intervals - a lift-off 0.05 s
  // out of a 0.25 s single support away from where the footholds assumed it.
  //
  // The window is replayed in continuous time. Each switch of the committed contacts is placed at the executed
  // schedule's own event time, and only the node it is assigned to comes from the grid. Replaying whole nodes instead
  // ended the swing in flight at the start of the node its touch-down falls in: the reference manager extends the
  // commit boundary to that touch-down and samples the last committed node AT the boundary, where the foot is already
  // down, so the swing lost up to a whole node. The deadbeat step was then evaluated up to 25 ms before the impact the
  // robot executes - on the shipped Atlas cadence that asked a robot exactly on the orbit for a 0.230 m lateral step
  // instead of 0.25 m, one-sidedly early for every swing - and the following double support was counted from the node
  // start, so the next lift-off came up to a node early as well.
  const int numCommitted = std::min(static_cast<int>(input.committedContacts.size()), numNodes);
  const scalar_t windowEnd = input.time + dt * static_cast<scalar_t>(numCommitted);
  const std::vector<scalar_t> sampleTimes = committedSampleTimes(input.time, dt, numCommitted, input.committedUntil);
  scalar_t time = input.time;                                       // start of the phase in progress, continuous
  contact_flag_t contacts = input.contacts;                         // its contact state
  scalar_t phaseBegan = input.time - elapsedInCurrentPhase(input);  // when it began, which the cadence counts from
  int lastSwungFoot = input.lastSwungFoot;
  if (swingFootOf(contacts) >= 0) lastSwungFoot = swingFootOf(contacts);

  for (int node = 0; node < numCommitted; ++node) {
    const contact_flag_t& next = input.committedContacts[static_cast<size_t>(node)];
    if (next == contacts) continue;
    // One switch per foot that changed, in the order they happened. Two feet switching between the same two samples
    // (a double support shorter than the node that holds it) keep the brief state between them as a phase of its own
    // that covers no interval.
    std::vector<std::pair<scalar_t, size_t>> switches;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      if (next[foot] != contacts[foot]) switches.emplace_back(committedSwitchTime(input, sampleTimes, node, foot, dt), foot);
    }
    std::sort(switches.begin(), switches.end());
    for (const std::pair<scalar_t, size_t>& footSwitch : switches) {
      const scalar_t switchTime = std::max(time, footSwitch.first);
      time = append(time, switchTime, node, contacts, swingFootOf(contacts));
      time = switchTime;
      contacts[footSwitch.second] = !contacts[footSwitch.second];
      phaseBegan = switchTime;
      if (swingFootOf(contacts) >= 0) lastSwungFoot = swingFootOf(contacts);
    }
  }

  // Where the nominal cadence picks up: the phase the window leaves in progress, which began at `phaseBegan` and has
  // to last at least to the end of the window. A double support is served out from its touch-down, a swing in flight
  // lands at its nominal touch-down, both counted from the executed events rather than from the node grid.
  if (!bothFeetDown(contacts) && swingFootOf(contacts) < 0) {
    // In flight (not a gait this planner makes): hold it to the end of the window, then carry on as double support.
    time = append(time, std::max(time, windowEnd), numCommitted, contacts, /*swingFoot=*/-1);
    contacts = makeFeetArray(true);
  }
  const int currentSwingFoot = swingFootOf(contacts);
  if (currentSwingFoot >= 0) {
    // A swing in flight finishes whether or not the robot keeps walking; the commit boundary normally reaches its
    // touch-down, so the window already covers it.
    const scalar_t touchDown = std::max(windowEnd, phaseBegan + sspDuration);
    time = append(time, touchDown, nearestNode(touchDown), contacts, currentSwingFoot);
  }

  // 2a. Standing: the rest of the horizon is double support.
  if (!walking) {
    append(time, horizonEnd, numNodes, makeFeetArray(true), /*swingFoot=*/-1);
    if (gait.empty()) {
      // Nothing was committed and the horizon is degenerate; keep the previous behavior of one covering phase.
      GaitPhase stance;
      stance.startTime = input.time;
      stance.endTime = horizonEnd;
      stance.endNode = numNodes;
      stance.contacts = makeFeetArray(true);
      gait.push_back(stance);
    }
    return gait;
  }

  // 2b. Walking: serve out the double support in progress (or the one after the swing that just landed), then lift
  // the foot whose turn it is.
  size_t swingFoot = 0;
  if (currentSwingFoot < 0) {
    const scalar_t liftOff = std::max(windowEnd, phaseBegan + dspDuration);
    time = append(time, liftOff, nearestNode(liftOff), makeFeetArray(true), /*swingFoot=*/-1);
    swingFoot = lastSwungFoot >= 0 ? otherFoot(static_cast<size_t>(lastSwungFoot)) : nextSwingFoot(input);
  } else {
    const scalar_t liftOff = time + dspDuration;
    time = append(time, liftOff, nearestNode(liftOff), makeFeetArray(true), /*swingFoot=*/-1);
    swingFoot = otherFoot(static_cast<size_t>(currentSwingFoot));
  }

  // From here the cadence is nominal: single supports of sspDuration alternating between the feet, separated by
  // double supports of dspDuration. The horizon is covered with whole phases, so the last one may reach past its end.
  while (time < horizonEnd - kMinPhaseDuration) {
    const scalar_t touchDown = time + sspDuration;
    time = append(time, touchDown, nearestNode(touchDown), stanceOnly(otherFoot(swingFoot)), static_cast<int>(swingFoot));
    const scalar_t liftOff = time + dspDuration;
    time = append(time, liftOff, nearestNode(liftOff), makeFeetArray(true), /*swingFoot=*/-1);
    swingFoot = otherFoot(swingFoot);
  }
  return gait;
}

scalar_t HlipContactPlanner::clipLateralStep(const HlipParameters& params, scalar_t lateralStep, bool placesLeftFoot) {
  // The placed foot must stay on its own side of the stance foot, between the self-collision margin and the reach.
  return placesLeftFoot ? std::clamp(lateralStep, params.minStepWidth, params.maxStepWidth)
                        : std::clamp(lateralStep, -params.maxStepWidth, -params.minStepWidth);
}

vector2_t HlipContactPlanner::deadbeatStep(const HlipModel::State& preImpactX,
                                           const HlipModel::State& preImpactY,
                                           const vector2_t& commandedVelocity,
                                           size_t swingFoot,
                                           bool& clipped) const {
  const HlipParameters& params = config_.hlip;
  const scalar_t stepDuration = model_.stepDuration();

  // Along the heading the nominal gait is the period-one orbit that advances by the commanded velocity every step.
  const scalar_t nominalStepX = commandedVelocity(0) * stepDuration;
  const HlipModel::State orbitX = model_.periodOneOrbit(nominalStepX);

  // Laterally it is the period-two orbit that alternates the two feet around the nominal step width.
  const scalar_t lateralDrift = commandedVelocity(1) * stepDuration;
  const scalar_t leftStep = params.stepWidth + lateralDrift;    // the step that places the left foot
  const scalar_t rightStep = -params.stepWidth + lateralDrift;  // the step that places the right foot
  const std::pair<HlipModel::State, HlipModel::State> orbitY = model_.periodTwoOrbit(leftStep, rightStep);
  const bool placesLeftFoot = swingFoot == kContactLeftIndex;
  const scalar_t nominalStepY = placesLeftFoot ? leftStep : rightStep;
  const HlipModel::State orbitStateY = placesLeftFoot ? orbitY.first : orbitY.second;

  const scalar_t sagittalStep = model_.deadbeatStepLength(preImpactX, orbitX, nominalStepX);
  const scalar_t lateralStep = model_.deadbeatStepLength(preImpactY, orbitStateY, nominalStepY);

  vector2_t step;
  step(0) = std::clamp(sagittalStep, -params.maxStepLength, params.maxStepLength);
  step(1) = clipLateralStep(params, lateralStep, placesLeftFoot);
  // A clipped step is not the deadbeat step: the part of the error it was asked to cancel survives into the next one.
  // The caller counts these onto the plan, because a gait that keeps clipping is a gait that will diverge.
  clipped = std::abs(step(0) - sagittalStep) > kStepClipTolerance || std::abs(step(1) - lateralStep) > kStepClipTolerance;
  return step;
}

ContactPlan HlipContactPlanner::plan(const ContactPlannerInput& input) {
  const std::chrono::steady_clock::time_point startedAt = std::chrono::steady_clock::now();
  const PlannerSettings& settings = config_.planner;
  const int numIntervals = settings.numNodes;
  const int numNodes = numIntervals + 1;
  const scalar_t dt = settings.dt;
  const bool withHeading = config_.usesHeadingModel();

  ContactPlan plan;
  plan.valid = true;
  plan.optimal = true;
  plan.startTime = input.time;
  plan.dt = dt;
  plan.committedUntil = input.committedUntil;
  plan.yaw = input.yaw;
  plan.omega = config_.omega();
  plan.contacts.assign(numIntervals, makeFeetArray(true));
  plan.footholds.assign(numNodes, input.footPositions);
  plan.comPosition.assign(numNodes, input.comPosition);
  plan.comVelocity.assign(numNodes, input.comVelocity);
  plan.zmp.assign(numIntervals, input.comPosition);
  if (withHeading) {
    plan.heading.assign(numNodes, input.heading);
    plan.headingRate.assign(numNodes, 0.0);
    plan.footYaws.assign(numNodes, input.footYaws);
  }

  const Rotation2 world_R_planning = rotation(input.yaw);
  const vector2_t commandInHeading = world_R_planning.transpose() * input.velocityCommand;
  const scalar_t alpha = blendWeight(input);
  const bool walking = alpha >= 0.5;
  // Below the blend's half point the planner stands; above it the stride grows with alpha out of standing.
  const vector2_t plannedCommand = alpha * commandInHeading;

  const std::vector<GaitPhase> gait = buildGait(input, walking);

  // The reduced model at the start of every phase, and the feet during it, rolled forward through the gait in
  // continuous time: the deadbeat step of a swing is evaluated at the pre-impact state of the phase's own end time,
  // which inside the committed window is the executed touch-down.
  struct PhaseRollOut {
    vector2_t comPosition;  // world, at the start of the phase
    vector2_t comVelocity;
    size_t stanceFoot = kContactLeftIndex;  // single support only
    Rotation2 world_R_phase = Rotation2::Identity();
    HlipModel::State stateX = HlipModel::State::Zero();  // relative to the stance foot, in the phase's heading frame
    HlipModel::State stateY = HlipModel::State::Zero();
    feet_array_t<vector2_t> feet;  // during the phase: a swinging foot at its landing spot
    feet_array_t<scalar_t> footYaws = makeFeetArray(0.0);
  };
  std::vector<PhaseRollOut> rollOut;
  rollOut.reserve(gait.size());
  {
    vector2_t comPosition = input.comPosition;
    vector2_t comVelocity = input.comVelocity;
    feet_array_t<vector2_t> feet = input.footPositions;
    feet_array_t<scalar_t> footYaws = input.footYaws;
    for (const GaitPhase& phase : gait) {
      PhaseRollOut entry;
      entry.comPosition = comPosition;
      entry.comVelocity = comVelocity;
      if (phase.isSingleSupport()) {
        const scalar_t phaseHeading = headingAt(input, phase.endTime, alpha);
        entry.world_R_phase = rotation(phaseHeading);
        entry.stanceFoot = otherFoot(static_cast<size_t>(phase.swingFoot));
        const vector2_t positionInHeading = entry.world_R_phase.transpose() * (comPosition - feet[entry.stanceFoot]);
        const vector2_t velocityInPhase = entry.world_R_phase.transpose() * comVelocity;
        entry.stateX << positionInHeading(0), velocityInPhase(0);
        entry.stateY << positionInHeading(1), velocityInPhase(1);

        // The landing spot of this swing: the deadbeat step evaluated at the pre-impact state the phase ends in.
        const HlipModel::State preImpactX = model_.flowSingleSupport(entry.stateX, phase.duration());
        const HlipModel::State preImpactY = model_.flowSingleSupport(entry.stateY, phase.duration());
        bool clipped = false;
        const vector2_t step = deadbeatStep(preImpactX, preImpactY, plannedCommand, static_cast<size_t>(phase.swingFoot), clipped);
        if (clipped) ++plan.numClippedSteps;
        feet[phase.swingFoot] = feet[entry.stanceFoot] + entry.world_R_phase * step;
        footYaws[phase.swingFoot] = phaseHeading;

        comPosition = feet[entry.stanceFoot] + entry.world_R_phase * vector2_t(preImpactX(0), preImpactY(0));
        comVelocity = entry.world_R_phase * vector2_t(preImpactX(1), preImpactY(1));
      } else {
        // Double support and standing: the reduced model drifts at constant velocity, as the H-LIP assumes.
        comPosition += phase.duration() * comVelocity;
      }
      entry.feet = feet;
      entry.footYaws = footYaws;
      rollOut.push_back(entry);
    }
  }

  // The center of mass of a phase's roll-out at `time`.
  const std::function<std::pair<vector2_t, vector2_t>(size_t, scalar_t)> comAt = [&](size_t phaseIndex, scalar_t time) {
    const GaitPhase& phase = gait[phaseIndex];
    const PhaseRollOut& entry = rollOut[phaseIndex];
    const scalar_t elapsed = std::max(0.0, time - phase.startTime);
    if (!phase.isSingleSupport()) {
      return std::make_pair(vector2_t(entry.comPosition + elapsed * entry.comVelocity), entry.comVelocity);
    }
    const HlipModel::State flowedX = model_.flowSingleSupport(entry.stateX, elapsed);
    const HlipModel::State flowedY = model_.flowSingleSupport(entry.stateY, elapsed);
    const vector2_t position = entry.feet[entry.stanceFoot] + entry.world_R_phase * vector2_t(flowedX(0), flowedY(0));
    const vector2_t velocity = entry.world_R_phase * vector2_t(flowedX(1), flowedY(1));
    return std::make_pair(position, velocity);
  };

  // Every node. The per-node foot quantities follow the phase the node is assigned to (GaitPhase::endNode), the same
  // integer boundaries the per-interval contacts below are filled from, so a foothold changes on exactly the node its
  // contact does. They used to be assigned by two different rules - nodes compared the node's own time against the
  // boundary, intervals compared their midpoint - which put interval k in the next phase while node k was still in the
  // current one at every phase boundary in the plan. The center of mass follows the continuous roll-out instead: it is
  // the reduced model at the node's own time, whichever side of a rounded boundary that time falls on.
  size_t gridPhase = 0;
  size_t timePhase = 0;
  for (int node = 0; node < numNodes; ++node) {
    while (gridPhase + 1 < gait.size() && node >= gait[gridPhase].endNode) ++gridPhase;
    const scalar_t nodeTime = input.time + dt * static_cast<scalar_t>(node);
    while (timePhase + 1 < gait.size() && nodeTime >= gait[timePhase].endTime) ++timePhase;
    const std::pair<vector2_t, vector2_t> com = comAt(timePhase, nodeTime);
    plan.comPosition[node] = com.first;
    plan.comVelocity[node] = com.second;
    plan.footholds[node] = rollOut[gridPhase].feet;
    if (withHeading) {
      plan.heading[node] = headingAt(input, nodeTime, alpha);
      plan.headingRate[node] = alpha * input.headingRateCommand;
      plan.footYaws[node] = rollOut[gridPhase].footYaws;
    }
  }

  // The standing / walking blend of the reference itself, equation (14) of the paper: the rolled-out reduced model is
  // blended against a static standing reference, the center of the support. Without it a standing plan would hand the
  // controller the measured drift as its center-of-mass reference and ask it to keep drifting, since the reduced
  // model's double support has no way to decelerate; with it, standing asks for a center of mass at rest over the feet.
  // The support is the one the robot will stand on: the feet after the last phase, with a swing that was still in
  // flight at the planning instant at its landing spot, not at the measured position of a foot in the air.
  //
  // It applies ONLY while standing. A stepping plan must publish the trajectory its own footholds were placed for:
  // planned_com_override writes this into the whole-body MPC's reference, and the deadbeat step was computed against
  // the unblended roll-out. Blending a stepping plan asked the controller for a smaller lateral sway than the step
  // assumed, which is the conflict section 3c of the README calls fatal, at reduced amplitude - and with the shipped
  // blend it bit exactly in the 0.10-0.17 m/s band a first cautious walk is commanded in. There is a step in the
  // reference across alpha = 0.5, but the gait itself already steps there (one long double support on one side, a
  // stepping cadence on the other) and so does the roll-out that produced it.
  if (!walking) {
    const feet_array_t<vector2_t>& standingFeet = rollOut.back().feet;
    const vector2_t supportCenter = 0.5 * (standingFeet[kContactLeftIndex] + standingFeet[kContactRightIndex]);
    for (int node = 0; node < numNodes; ++node) {
      plan.comPosition[node] = alpha * plan.comPosition[node] + (1.0 - alpha) * supportCenter;
      plan.comVelocity[node] *= alpha;
    }
  }

  // Contacts and ZMP per interval, from the same integer phase boundaries the nodes were filled from.
  size_t phaseIndex = 0;
  for (int interval = 0; interval < numIntervals; ++interval) {
    while (phaseIndex + 1 < gait.size() && interval >= gait[phaseIndex].endNode) ++phaseIndex;
    const GaitPhase& phase = gait[phaseIndex];
    const feet_array_t<vector2_t>& feet = rollOut[phaseIndex].feet;
    plan.contacts[interval] = phase.contacts;
    if (phase.isSingleSupport()) {
      plan.zmp[interval] = feet[otherFoot(static_cast<size_t>(phase.swingFoot))];
    } else {
      plan.zmp[interval] = 0.5 * (feet[kContactLeftIndex] + feet[kContactRightIndex]);
    }
  }

  // The committed window is no longer stamped over the contacts here: buildGait() now starts FROM it, so the contact
  // sequence, the footholds and the center-of-mass roll-out all describe the one gait.

  // The same gait in continuous time, which the mode schedule is built from (ContactPlan::toModeSchedule). The
  // intervals above round every event to its nearest node, and a schedule built from them executed each double
  // support for whatever the grid rounded it to, while the deadbeat step of the swing before it had been computed for
  // hlip.dspDuration exactly. The events decided here are continuous - a touch-down sspDuration after its lift-off, a
  // lift-off dspDuration after its touch-down - and the committed window is replayed at the executed event times, so
  // every later plan re-deciding a lift-off keeps it where the deadbeat gain assumes it.
  plan.phaseContacts.reserve(gait.size());
  plan.phaseStartTimes.reserve(gait.size());
  for (const GaitPhase& phase : gait) {
    plan.phaseContacts.push_back(phase.contacts);
    plan.phaseStartTimes.push_back(phase.startTime);
  }

  plan.solveTime = std::chrono::duration<scalar_t>(std::chrono::steady_clock::now() - startedAt).count();
  return plan;
}

scalar_t HlipContactPlanner::startUpLateralStep(const ContactPlanningConfig& config) {
  const HlipModel model = makeModel(config);
  const HlipParameters& params = config.hlip;
  // Standing: the center of mass sits half a step width from the stance foot with no lateral velocity, which is the
  // worst lateral state the gait ever starts from.
  const HlipModel::State atLiftOff(0.5 * params.stepWidth, 0.0);  // NOLINT(argument-comment): a vector2_t [p, v]
  const HlipModel::State preImpact = model.flowSingleSupport(atLiftOff, model.sspDuration());
  const std::pair<HlipModel::State, HlipModel::State> orbit = model.periodTwoOrbit(params.stepWidth, -params.stepWidth);
  return std::abs(model.deadbeatStepLength(preImpact, orbit.first, params.stepWidth));
}

std::vector<scalar_t> HlipContactPlanner::startUpLateralWidths(const ContactPlanningConfig& config, int numSteps) {
  const HlipModel model = makeModel(config);
  const HlipParameters& params = config.hlip;
  const std::pair<HlipModel::State, HlipModel::State> orbit = model.periodTwoOrbit(params.stepWidth, -params.stepWidth);
  // Standing on the right foot with the left about to swing, the center of mass half a step width to its left, at
  // rest: the state startUpLateralStep() starts from. Each step is the deadbeat step under the planner's clip.
  HlipModel::State state(0.5 * params.stepWidth, 0.0);  // NOLINT(argument-comment): a vector2_t [p, v]
  bool placesLeftFoot = true;
  std::vector<scalar_t> widths;
  widths.reserve(static_cast<size_t>(std::max(0, numSteps)));
  for (int step = 0; step < numSteps; ++step) {
    const HlipModel::State preImpact = model.flowSingleSupport(state, model.sspDuration());
    const scalar_t nominalStep = placesLeftFoot ? params.stepWidth : -params.stepWidth;
    const scalar_t demanded = model.deadbeatStepLength(preImpact, placesLeftFoot ? orbit.first : orbit.second, nominalStep);
    const scalar_t placed = clipLateralStep(params, demanded, placesLeftFoot);
    widths.push_back(std::abs(placed));
    state = HlipModel::flowDoubleSupport(HlipModel::applyStepTransition(preImpact, placed), model.dspDuration());
    placesLeftFoot = !placesLeftFoot;
  }
  return widths;
}

std::string HlipContactPlanner::formulationSummary(const ContactPlanningConfig& config) {
  const HlipParameters& params = config.hlip;
  const HlipModel model = makeModel(config);
  const scalar_t stepDuration = model.stepDuration();
  const scalar_t startUpDemand = startUpLateralStep(config);
  const std::vector<std::string>& execution = config.formulation.execution;
  return absl::StrCat(
      "[HlipContactPlanner] closed-form H-LIP contact planner (arXiv:2502.15630)\n", "  cadence      : single support ", params.sspDuration,
      " s, double support ", params.dspDuration, " s, stride ", 2.0 * stepDuration, " s\n", "  pendulum     : height ",
      config.pendulumHeight(), " m, omega ", model.naturalFrequency(), " 1/s\n", "  step law     : deadbeat, K = [",
      model.deadbeatGain()(0), ", ", model.deadbeatGain()(1), "] (closed form, nothing tuned)\n",
      "  nominal orbit: period one along the heading, ", "period two laterally at a step width of ", params.stepWidth, " m\n",
      "  step clip    : |dx| <= ", params.maxStepLength, " m, step width in [", params.minStepWidth, ", ", params.maxStepWidth, "] m\n",
      "  stand / walk : alpha = tanh(", params.blend.sharpness, " (phi - ", params.blend.threshold,
      ")) / 2 + 1/2, lateral velocity measured against the zero-command stepping orbit\n", "  grid         : ", config.planner.numNodes,
      " intervals x ", config.planner.dt, " s = ", config.horizon(), " s\n", "  heading model: ", config.usesHeadingModel() ? "on" : "off",
      "\n", "  start-up     : the first step out of a standstill needs ", startUpDemand,
      " m of lateral step, against an hlip.max_step_width of ", params.maxStepWidth, startUpVerdict(config, startUpDemand),
      "  no optimization is performed: no costs, no constraints, no search (the formulation's cost, constraint, logic and search "
      "lists are not read)\n",
      "  execution    : ", execution.empty() ? std::string("none") : absl::StrJoin(execution, ", "),
      " (applied by the reference manager under either planner; planned_com_override is required here)");
}

}  // namespace ocs2::humanoid
