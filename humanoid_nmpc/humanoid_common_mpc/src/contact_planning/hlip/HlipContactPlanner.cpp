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
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipModel.h"

namespace ocs2::humanoid {
namespace {

using Rotation2 =
    Eigen::Matrix<scalar_t, kHorizontalPositionDim, kHorizontalPositionDim>;

/**
 * [s] a phase shorter than this that covers no interval is not emitted: a swing
 * whose touch-down is due within it has landed where the foot is, and the next
 * phase starts at once.
 */
constexpr scalar_t kMinPhaseDuration = 1.0e-6;

/** [m] a step cut by less than this is not counted as clipped. */
constexpr scalar_t kStepClipTolerance = 1.0e-9;

/** Steps the start-up check rolls the reduced model forward for, and the
 * tolerance [m] it calls a width settled at. */
constexpr int kStartUpSteps = 12;
constexpr scalar_t kStartUpSettledTolerance = 1.0e-3;
/** The fewest last steps at the nominal width that count as settled, and the
 * widths the start-up line prints. */
constexpr int kMinSettledSteps = 2;
constexpr int kStartUpWidthsPrinted = 6;

Rotation2 rotmatFromYaw(scalar_t yaw) {
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

/** The foot in flight in a contact state, or -1 when both feet are down (and
 * when neither is). */
int swingFootOf(const contact_flag_t& contacts) {
  if (bothFeetDown(contacts)) return -1;
  if (!contacts[kContactLeftIndex] && !contacts[kContactRightIndex]) return -1;
  return contacts[kContactLeftIndex] ? static_cast<int>(kContactRightIndex)
                                     : static_cast<int>(kContactLeftIndex);
}

/**
 * [s] how long the robot has already been in the contact state it is in at
 * `input.time`.
 *
 * In double support the newly landed foot is the one that dates the phase, so
 * the smaller of the two elapsed times is the age of the double support; in
 * single support it is how long the swinging foot has been in flight.
 */
scalar_t elapsedInCurrentPhase(const ContactPlannerInput& input) {
  const int swingFoot = swingFootOf(input.contacts);
  if (swingFoot < 0) {
    return std::min(input.phaseElapsedTime[kContactLeftIndex],
                    input.phaseElapsedTime[kContactRightIndex]);
  }
  return input.phaseElapsedTime[static_cast<size_t>(swingFoot)];
}

/**
 * [s] when the executed schedule switched `foot` between the samples of
 * committed nodes `node - 1` and `node` (for node 0, between `input.time` and
 * its sample).
 *
 * The committed contacts are samples of the executed schedule, one per node, so
 * they say in which node a switch happened but not when. The time is in
 * `input.committedPhaseStartTimes`: the start of the contact phase the foot is
 * in at the node's sample, which is exactly that switch when it lies between
 * the two samples. When it does not, or the input carries no times, the switch
 * is placed at the node start, which is all the grid itself can say.
 */
scalar_t committedSwitchTime(const ContactPlannerInput& input,
                             const std::vector<scalar_t>& sampleTimes,
                             int node,
                             size_t foot,
                             scalar_t dt) {
  const scalar_t nodeStart = input.time + dt * static_cast<scalar_t>(node);
  const size_t index = static_cast<size_t>(node);
  if (index >= input.committedPhaseStartTimes.size() ||
      index >= sampleTimes.size()) {
    return nodeStart;
  }
  const scalar_t switchTime = input.committedPhaseStartTimes[index][foot];
  const scalar_t previousSample =
      node == 0 ? input.time : sampleTimes[index - 1];
  const bool betweenTheSamples =
      std::isfinite(switchTime) &&
      switchTime > previousSample - kSwitchTimeTolerance &&
      switchTime <= sampleTimes[index] + kSwitchTimeTolerance;
  return betweenTheSamples ? switchTime : nodeStart;
}

/** The first step from which every width in `widths` is
 * `nomPosLfootRtRfootEwrtHeadingY`, or -1 when the widths never settle there.
 */
int settlingStep(const std::vector<scalar_t>& widths,
                 scalar_t nomPosLfootRtRfootEwrtHeadingY) {
  int settled = -1;
  for (int step = static_cast<int>(widths.size()) - 1; step >= 0; --step) {
    if (std::abs(widths[static_cast<size_t>(step)] -
                 nomPosLfootRtRfootEwrtHeadingY) > kStartUpSettledTolerance) {
      break;
    }
    settled = step;
  }
  // A width that is right only on the very last step has not been shown to stay
  // there.
  return settled >= 0 &&
                 settled + kMinSettledSteps <= static_cast<int>(widths.size())
             ? settled
             : -1;
}

/** The start-up line of the formulation summary: the first-step demand and what
 * the reduced model makes of it. */
std::string startUpVerdict(const ContactPlanningConfig& config,
                           scalar_t demand,
                           const std::vector<scalar_t>& widths) {
  const HlipParameters& params = config.hlip;
  if (demand <= params.maxPosLfootRtRfootEwrtHeadingY) return " (fits)\n";
  const int settled =
      settlingStep(widths, params.nomPosLfootRtRfootEwrtHeadingY);
  const std::vector<scalar_t> firstNormPosLfootRtRfootEwrtHeadingYTraj(
      widths.begin(),
      widths.begin() +
          std::min<ptrdiff_t>(kStartUpWidthsPrinted,
                              static_cast<ptrdiff_t>(widths.size())));
  const std::string rolledOut =
      absl::StrJoin(firstNormPosLfootRtRfootEwrtHeadingYTraj, ", ");
  if (settled >= 0) {
    return absl::StrCat(
        " <-- DOES NOT FIT: the first step is clipped, which costs the "
        "deadbeat property. The reduced model still recovers the "
        "nominal width after ",
        settled, " steps (widths ", rolledOut,
        ", ... m); shorten hlip.ssp_duration or raise "
        "hlip.max_pos_lfoot_rt_rfoot_ewrt_heading_y for "
        "margin.\n");
  }
  return absl::StrCat(
      " <-- DOES NOT FIT: the first step is clipped, and the reduced model "
      "then locks into an alternating wide/narrow limit "
      "cycle it never leaves (widths ",
      rolledOut,
      ", ... m). Shorten hlip.ssp_duration or raise "
      "hlip.max_pos_lfoot_rt_rfoot_ewrt_heading_y.\n");
}

using GaitPhase = HlipContactPlanner::GaitPhase;

/** The node nearest to `time` on the grid of `numNodes` intervals of `dt` that
 * starts at `input.time`. */
int nearestNode(const ContactPlannerInput& input,
                scalar_t dt,
                int numNodes,
                scalar_t time) {
  return std::clamp(static_cast<int>(std::lround((time - input.time) / dt)), 0,
                    numNodes);
}

/**
 * Appends the phase [startTime, endTime) that ends on node `endNode` to `gait`
 * and returns the time the gait then ends at. A phase with no duration that
 * covers no interval is dropped, and a phase that continues the last one is
 * merged into it.
 */
scalar_t appendPhase(std::vector<GaitPhase>& gait,
                     scalar_t startTime,
                     scalar_t endTime,
                     int endNode,
                     const contact_flag_t& contacts,
                     int swingFoot) {
  const int startNode = gait.empty() ? 0 : gait.back().endNode;
  endNode = std::max(endNode, startNode);
  // A phase with no duration is dropped, unless it covers intervals: the grid
  // must stay covered.
  if (endTime - startTime <= kMinPhaseDuration && endNode == startNode) {
    return startTime;
  }
  // A phase that merely continues the one before it is the SAME phase. This
  // matters beyond tidiness: the deadbeat step is evaluated once per
  // single-support phase, at the pre-impact state that phase ends in, so a
  // swing split into two adjacent phases would have its landing spot computed
  // twice - the first time from a pre-impact state flowed only part of the way.
  if (!gait.empty() && gait.back().contacts == contacts &&
      gait.back().swingFoot == swingFoot) {
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
}

/** Where the replay of the committed window leaves the gait
 * (replayCommittedWindow). */
struct CommittedWindowEnd {
  scalar_t time = 0.0;  // start of the phase in progress, continuous
  contact_flag_t contacts = makeFeetArray(true);  // its contact state
  scalar_t phaseBegan = 0.0;  // when it began, which the cadence counts from
  int lastSwungFoot = -1;     // the last foot that swung, -1 when none is known
};

/**
 * Appends the committed window of `input`, its first `numCommitted` nodes, to
 * `gait` in continuous time, and returns the phase it leaves in progress.
 */
CommittedWindowEnd replayCommittedWindow(const ContactPlannerInput& input,
                                         scalar_t dt,
                                         int numCommitted,
                                         std::vector<GaitPhase>& gait) {
  const std::vector<scalar_t> sampleTimes =
      committedSampleTimes(input.time, dt, numCommitted, input.committedUntil);
  scalar_t time = input.time;
  contact_flag_t contacts = input.contacts;
  scalar_t phaseBegan = input.time - elapsedInCurrentPhase(input);
  int lastSwungFoot = input.lastSwungFoot;
  if (swingFootOf(contacts) >= 0) lastSwungFoot = swingFootOf(contacts);

  for (int node = 0; node < numCommitted; ++node) {
    const contact_flag_t& next =
        input.committedContacts[static_cast<size_t>(node)];
    if (next == contacts) continue;
    // One switch per foot that changed, in the order they happened. Two feet
    // switching between the same two samples (a double support shorter than the
    // node that holds it) keep the brief state between them as a phase of its
    // own that covers no interval.
    std::vector<std::pair<scalar_t, size_t>> switches;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      if (next[foot] != contacts[foot]) {
        switches.emplace_back(
            committedSwitchTime(input, sampleTimes, node, foot, dt), foot);
      }
    }
    std::sort(switches.begin(), switches.end());
    for (const std::pair<scalar_t, size_t>& footSwitch : switches) {
      const scalar_t switchTime = std::max(time, footSwitch.first);
      time = appendPhase(gait, time, switchTime, node, contacts,
                         swingFootOf(contacts));
      time = switchTime;
      contacts[footSwitch.second] = !contacts[footSwitch.second];
      phaseBegan = switchTime;
      if (swingFootOf(contacts) >= 0) lastSwungFoot = swingFootOf(contacts);
    }
  }
  return {.time = time,
          .contacts = contacts,
          .phaseBegan = phaseBegan,
          .lastSwungFoot = lastSwungFoot};
}

}  // namespace

absl::StatusOr<HlipModel> HlipContactPlanner::makeModel(
    const ContactPlanningConfig& config) {
  return HlipModel::Create(config.hlip.sspDuration, config.hlip.dspDuration,
                           config.getNomPosComRtFeetmeanEwrtWorldZ(),
                           config.shared.gravity);
}

absl::StatusOr<std::unique_ptr<HlipContactPlanner>> HlipContactPlanner::Create(
    ContactPlanningConfig config) {
  RETURN_IF_ERROR(config.validateStatus());
  ASSIGN_OR_RETURN(const HlipModel model, makeModel(config));
  // WrapUnique: the constructor is private (ToTW #134).
  return absl::WrapUnique(new HlipContactPlanner(std::move(config), model));
}

HlipContactPlanner::HlipContactPlanner(ContactPlanningConfig config,
                                       const HlipModel& model)
    : config_(std::move(config)), model_(model), blend_(config_.hlip.blend) {}

absl::Status HlipContactPlanner::setConfig(
    const ContactPlanningConfig& config) {
  // Validated first, so that nothing unvalidated is built and the running
  // configuration is kept on a refusal.
  RETURN_IF_ERROR(config.validateStatus());
  ASSIGN_OR_RETURN(HlipModel model, makeModel(config));
  config_ = config;
  model_ = std::move(model);
  blend_ = HlipStandingBlend(config_.hlip.blend);
  return absl::OkStatus();
}

size_t HlipContactPlanner::nextSwingFoot(const ContactPlannerInput& input) {
  if (input.lastSwungFoot >= 0) {
    return otherFoot(static_cast<size_t>(input.lastSwungFoot));
  }
  // Nothing has swung yet: the foot that has been in contact the longest has
  // had the most time to be unloaded.
  return input.phaseElapsedTime[kContactLeftIndex] >=
                 input.phaseElapsedTime[kContactRightIndex]
             ? kContactLeftIndex
             : kContactRightIndex;
}

std::optional<scalar_t>
HlipContactPlanner::computeZeroCommandOrbitLinvelComRtWorldEwrtHeadingY(
    const ContactPlannerInput& input) const {
  const scalar_t nomPosLfootRtRfootEwrtHeadingY =
      config_.hlip.nomPosLfootRtRfootEwrtHeadingY;
  // `first` precedes the step that places the left foot (standing on the
  // right), `second` the one that places the right.
  const HlipModel::PeriodTwoOrbit orbit = model_.periodTwoOrbit(
      nomPosLfootRtRfootEwrtHeadingY, -nomPosLfootRtRfootEwrtHeadingY);
  const int swingFoot = swingFootOf(input.contacts);
  if (swingFoot >= 0) {
    // The single support began at the lift-off after the impact that placed the
    // stance foot: that step's pre-impact state, carried through the impact and
    // the double support, then flowed for as long as the swing has been in
    // flight.
    const bool placesLeftFoot =
        swingFoot == static_cast<int>(kContactLeftIndex);
    const HlipModel::State& beforeStanceStep =
        placesLeftFoot ? orbit.beforeSecondStep : orbit.beforeFirstStep;
    const scalar_t stanceErrPosFootRtWorldEwrtHeadingY =
        placesLeftFoot ? -nomPosLfootRtRfootEwrtHeadingY
                       : nomPosLfootRtRfootEwrtHeadingY;
    const HlipModel::State atLiftOff = HlipModel::flowDoubleSupport(
        HlipModel::applyStepTransition(beforeStanceStep,
                                       stanceErrPosFootRtWorldEwrtHeadingY),
        model_.dspDuration());
    const scalar_t inFlight =
        std::clamp(input.phaseElapsedTime[static_cast<size_t>(swingFoot)], 0.0,
                   model_.sspDuration());
    return model_.flowSingleSupport(atLiftOff, inFlight)(1);
  }
  if (bothFeetDown(input.contacts) && input.lastSwungFoot >= 0) {
    // After the touch-down of the last swing the orbit drifts at that step's
    // pre-impact velocity.
    return (input.lastSwungFoot == static_cast<int>(kContactLeftIndex)
                ? orbit.beforeFirstStep
                : orbit.beforeSecondStep)(1);
  }
  return std::nullopt;
}

vector2_t HlipContactPlanner::computeBlendLinvelComRtWorldEwrtHeadingXy(
    const ContactPlannerInput& input) const {
  const Rotation2 rotmatHeadingRtWorld = rotmatFromYaw(input.yawHeadingRtWorld);
  vector2_t linvelComRtWorldEwrtHeadingXy =
      rotmatHeadingRtWorld.transpose() * input.proxyLinvelComRtWorldEwrtWorldXy;
  const std::optional<scalar_t> orbitLinvelComRtWorldEwrtHeadingY =
      computeZeroCommandOrbitLinvelComRtWorldEwrtHeadingY(input);
  if (orbitLinvelComRtWorldEwrtHeadingY.has_value()) {
    // Only what lies outside the range between rest and the sway of stepping in
    // place counts.
    const scalar_t low = std::min(0.0, *orbitLinvelComRtWorldEwrtHeadingY);
    const scalar_t high = std::max(0.0, *orbitLinvelComRtWorldEwrtHeadingY);
    linvelComRtWorldEwrtHeadingXy(1) =
        linvelComRtWorldEwrtHeadingXy(1) > high
            ? linvelComRtWorldEwrtHeadingXy(1) - high
            : (linvelComRtWorldEwrtHeadingXy(1) < low
                   ? linvelComRtWorldEwrtHeadingXy(1) - low
                   : 0.0);
  }
  return linvelComRtWorldEwrtHeadingXy;
}

scalar_t HlipContactPlanner::blendWeight(
    const ContactPlannerInput& input) const {
  const Rotation2 rotmatHeadingRtWorld = rotmatFromYaw(input.yawHeadingRtWorld);
  const vector2_t cmdLinvelBaseRtWorldEwrtHeadingXy =
      rotmatHeadingRtWorld.transpose() * input.cmdLinvelBaseRtWorldEwrtWorldXy;
  return blend_.weight(cmdLinvelBaseRtWorldEwrtHeadingXy,
                       input.cmdRateYawBaseRtWorld,
                       computeBlendLinvelComRtWorldEwrtHeadingXy(input));
}

bool HlipContactPlanner::isWalking(const ContactPlannerInput& input) const {
  const Rotation2 rotmatHeadingRtWorld = rotmatFromYaw(input.yawHeadingRtWorld);
  const vector2_t cmdLinvelBaseRtWorldEwrtHeadingXy =
      rotmatHeadingRtWorld.transpose() * input.cmdLinvelBaseRtWorldEwrtWorldXy;
  return blend_.isWalking(cmdLinvelBaseRtWorldEwrtHeadingXy,
                          input.cmdRateYawBaseRtWorld,
                          computeBlendLinvelComRtWorldEwrtHeadingXy(input));
}

scalar_t HlipContactPlanner::computeYawHeadingRtWorld(
    const ContactPlannerInput& input, scalar_t time, scalar_t alpha) const {
  // `yaw` is the frame the geometry is planned in: the whole-body heading with
  // the heading model, the base yaw without it
  // (ContactPlanningReferenceManager::makePlannerInput). `heading` is only
  // filled with the model, so the frame must come from `yaw` or a robot without
  // the model would plan its footholds in the world frame.
  return input.yawHeadingRtWorld +
         alpha * input.cmdRateYawBaseRtWorld * (time - input.time);
}

std::vector<HlipContactPlanner::GaitPhase> HlipContactPlanner::buildGait(
    const ContactPlannerInput& input, GaitKind kind) const {
  const scalar_t dt = config_.planner.dt;
  const int numNodes = config_.planner.numNodes;
  const scalar_t horizonEnd = input.time + config_.horizon();
  const scalar_t sspDuration = model_.sspDuration();
  const scalar_t dspDuration = model_.dspDuration();
  std::vector<GaitPhase> gait;

  // NOTE that standing is NOT handled by returning here. It used to be:
  // `!walking` produced one all-stance phase over the whole horizon and
  // returned before the committed window below, which made the standing plan
  // the one plan this planner produces that is not built from
  // input.committedContacts - in flat contradiction of the comment on that
  // window, which calls those intervals "not the planner's to choose".
  //
  // The consequence was not cosmetic. A lone stance phase has swingFoot == -1,
  // so plan() never enters its single-support branch, `feet` is never advanced
  // past its initialization from input.posFeetRtWorldEwrtWorldXy, and
  // plan.desPosFeetRtWorldEwrtWorldXyTraj[node] therefore published the
  // measured position of whichever foot was IN THE AIR as that swing's landing
  // target - a target in mid-flight. The blend crosses into standing on a
  // released stick, and single support is most of a stride, so a foot is
  // airborne at that moment more often than not. The plan is still accepted,
  // because activatePendingPlan only checks agreement with the swings in flight
  // at mergeTime, by which point the commit boundary has advanced past that
  // touch-down.
  //
  // So the committed window is replayed first for standing exactly as for
  // walking, and only the REMAINDER of the horizon becomes the stance phase. A
  // swing in flight is then a real single-support phase, plan() rolls it out
  // and deadbeatStep() places its landing spot under the step-width clip like
  // any other.

  // 1. The committed window, replayed from the executed schedule.
  //
  // These intervals are not the planner's to choose: the reference manager
  // merges the applied schedule over them anyway, and a plan that disagreed
  // with it there would be dropped as inconsistent. They used to be stamped
  // over `plan.contacts` at the very END of plan(), AFTER the center of mass
  // and the footholds had been rolled out against a gait built as if the window
  // were free. The two then described different gaits. The worst case is the
  // one the robot starts every walk from: out of a long stance, `dspDuration -
  // elapsed` is negative, so the nominal cadence lifted a foot at node 0 while
  // the commit window held both feet down for the first two intervals - a
  // lift-off 0.05 s out of a 0.25 s single support away from where the
  // footholds assumed it.
  //
  // The window is replayed in continuous time. Each switch of the committed
  // contacts is placed at the executed schedule's own event time, and only the
  // node it is assigned to comes from the grid. Replaying whole nodes instead
  // ended the swing in flight at the start of the node its touch-down falls in:
  // the reference manager extends the commit boundary to that touch-down and
  // samples the last committed node AT the boundary, where the foot is already
  // down, so the swing lost up to a whole node. The deadbeat step was then
  // evaluated up to 25 ms before the impact the robot executes - on the shipped
  // Atlas cadence that asked a robot exactly on the orbit for a 0.230 m lateral
  // step instead of 0.25 m, one-sidedly early for every swing - and the
  // following double support was counted from the node start, so the next
  // lift-off came up to a node early as well.
  const int numCommitted =
      std::min(static_cast<int>(input.committedContacts.size()), numNodes);
  const scalar_t windowEnd =
      input.time + dt * static_cast<scalar_t>(numCommitted);
  const CommittedWindowEnd windowState =
      replayCommittedWindow(input, dt, numCommitted, gait);
  scalar_t time = windowState.time;
  contact_flag_t contacts = windowState.contacts;
  const scalar_t phaseBegan = windowState.phaseBegan;
  const int lastSwungFoot = windowState.lastSwungFoot;

  // Where the nominal cadence picks up: the phase the window leaves in
  // progress, which began at `phaseBegan` and has to last at least to the end
  // of the window. A double support is served out from its touch-down, a swing
  // in flight lands at its nominal touch-down, both counted from the executed
  // events rather than from the node grid.
  if (!bothFeetDown(contacts) && swingFootOf(contacts) < 0) {
    // In flight (not a gait this planner makes): hold it to the end of the
    // window, then carry on as double support.
    time = appendPhase(gait, time, std::max(time, windowEnd), numCommitted,
                       contacts, /*swingFoot=*/-1);
    contacts = makeFeetArray(true);
  }
  const int currentSwingFoot = swingFootOf(contacts);
  if (currentSwingFoot >= 0) {
    // A swing in flight finishes whether or not the robot keeps walking; the
    // commit boundary normally reaches its touch-down, so the window already
    // covers it.
    const scalar_t touchDown = std::max(windowEnd, phaseBegan + sspDuration);
    time = appendPhase(gait, time, touchDown,
                       nearestNode(input, dt, numNodes, touchDown), contacts,
                       currentSwingFoot);
  }

  // 2a. Standing: the rest of the horizon is double support.
  if (kind == GaitKind::kStanding) {
    appendPhase(gait, time, horizonEnd, numNodes, makeFeetArray(true),
                /*swingFoot=*/-1);
    if (gait.empty()) {
      // Nothing was committed and the horizon is degenerate; keep the previous
      // behavior of one covering phase.
      GaitPhase stance;
      stance.startTime = input.time;
      stance.endTime = horizonEnd;
      stance.endNode = numNodes;
      stance.contacts = makeFeetArray(true);
      gait.push_back(stance);
    }
    return gait;
  }

  // 2b. Walking: serve out the double support in progress (or the one after the
  // swing that just landed), then lift the foot whose turn it is.
  size_t swingFoot = 0;
  if (currentSwingFoot < 0) {
    const scalar_t liftOff = std::max(windowEnd, phaseBegan + dspDuration);
    time = appendPhase(gait, time, liftOff,
                       nearestNode(input, dt, numNodes, liftOff),
                       makeFeetArray(true), /*swingFoot=*/-1);
    swingFoot = lastSwungFoot >= 0
                    ? otherFoot(static_cast<size_t>(lastSwungFoot))
                    : nextSwingFoot(input);
  } else {
    const scalar_t liftOff = time + dspDuration;
    time = appendPhase(gait, time, liftOff,
                       nearestNode(input, dt, numNodes, liftOff),
                       makeFeetArray(true), /*swingFoot=*/-1);
    swingFoot = otherFoot(static_cast<size_t>(currentSwingFoot));
  }

  // From here the cadence is nominal: single supports of sspDuration
  // alternating between the feet, separated by double supports of dspDuration.
  // The horizon is covered with whole phases, so the last one may reach past
  // its end.
  while (time < horizonEnd - kMinPhaseDuration) {
    const scalar_t touchDown = time + sspDuration;
    time = appendPhase(
        gait, time, touchDown, nearestNode(input, dt, numNodes, touchDown),
        stanceOnly(otherFoot(swingFoot)), static_cast<int>(swingFoot));
    const scalar_t liftOff = time + dspDuration;
    time = appendPhase(gait, time, liftOff,
                       nearestNode(input, dt, numNodes, liftOff),
                       makeFeetArray(true), /*swingFoot=*/-1);
    swingFoot = otherFoot(swingFoot);
  }
  return gait;
}

scalar_t HlipContactPlanner::clipDesErrPosFootRtWorldEwrtHeadingY(
    const HlipParameters& params,
    scalar_t errPosFootRtWorldEwrtHeadingY,
    bool placesLeftFoot) {
  // The placed foot must stay on its own side of the stance foot, between the
  // self-collision margin and the reach.
  return placesLeftFoot ? std::clamp(errPosFootRtWorldEwrtHeadingY,
                                     params.minPosLfootRtRfootEwrtHeadingY,
                                     params.maxPosLfootRtRfootEwrtHeadingY)
                        : std::clamp(errPosFootRtWorldEwrtHeadingY,
                                     -params.maxPosLfootRtRfootEwrtHeadingY,
                                     -params.minPosLfootRtRfootEwrtHeadingY);
}

HlipContactPlanner::DeadbeatStep HlipContactPlanner::deadbeatStep(
    const HlipModel::State& preImpactX,
    const HlipModel::State& preImpactY,
    const vector2_t& cmdLinvelBaseRtWorldEwrtHeadingXy,
    size_t swingFoot) const {
  const HlipParameters& params = config_.hlip;
  const scalar_t stepDuration = model_.stepDuration();

  // Along the heading the nominal gait is the period-one orbit that advances by
  // the commanded velocity every step.
  const scalar_t nomErrPosFootRtWorldEwrtHeadingX =
      cmdLinvelBaseRtWorldEwrtHeadingXy(0) * stepDuration;
  const HlipModel::State orbitX =
      model_.periodOneOrbit(nomErrPosFootRtWorldEwrtHeadingX);

  // Laterally it is the period-two orbit that alternates the two feet around
  // the nominal step width.
  const scalar_t driftErrPosFootRtWorldEwrtHeadingY =
      cmdLinvelBaseRtWorldEwrtHeadingXy(1) * stepDuration;
  const scalar_t leftErrPosFootRtWorldEwrtHeadingY =
      params.nomPosLfootRtRfootEwrtHeadingY +
      driftErrPosFootRtWorldEwrtHeadingY;  // the step that places the left foot
  const scalar_t rightErrPosFootRtWorldEwrtHeadingY =
      -params.nomPosLfootRtRfootEwrtHeadingY +
      driftErrPosFootRtWorldEwrtHeadingY;  // the step that places the right
                                           // foot
  const HlipModel::PeriodTwoOrbit orbitY = model_.periodTwoOrbit(
      leftErrPosFootRtWorldEwrtHeadingY, rightErrPosFootRtWorldEwrtHeadingY);
  const bool placesLeftFoot = swingFoot == kContactLeftIndex;
  const scalar_t nomErrPosFootRtWorldEwrtHeadingY =
      placesLeftFoot ? leftErrPosFootRtWorldEwrtHeadingY
                     : rightErrPosFootRtWorldEwrtHeadingY;
  const HlipModel::State orbitStateY =
      placesLeftFoot ? orbitY.beforeFirstStep : orbitY.beforeSecondStep;

  const scalar_t deadbeatErrPosFootRtWorldEwrtHeadingX =
      model_.deadbeatStepLength(preImpactX, orbitX,
                                nomErrPosFootRtWorldEwrtHeadingX);
  const scalar_t deadbeatErrPosFootRtWorldEwrtHeadingY =
      model_.deadbeatStepLength(preImpactY, orbitStateY,
                                nomErrPosFootRtWorldEwrtHeadingY);

  vector2_t desErrPosFootRtWorldEwrtHeadingXy;
  desErrPosFootRtWorldEwrtHeadingXy(0) =
      std::clamp(deadbeatErrPosFootRtWorldEwrtHeadingX,
                 -params.maxNormPosLfootRtRfootEwrtHeadingX,
                 params.maxNormPosLfootRtRfootEwrtHeadingX);
  desErrPosFootRtWorldEwrtHeadingXy(1) = clipDesErrPosFootRtWorldEwrtHeadingY(
      params, deadbeatErrPosFootRtWorldEwrtHeadingY, placesLeftFoot);
  // A clipped step is not the deadbeat step: the part of the error it was asked
  // to cancel survives into the next one. The caller counts these onto the
  // plan, because a gait that keeps clipping is a gait that will diverge.
  const bool clipped =
      std::abs(desErrPosFootRtWorldEwrtHeadingXy(0) -
               deadbeatErrPosFootRtWorldEwrtHeadingX) > kStepClipTolerance ||
      std::abs(desErrPosFootRtWorldEwrtHeadingXy(1) -
               deadbeatErrPosFootRtWorldEwrtHeadingY) > kStepClipTolerance;
  return {
      .desErrPosFootRtWorldEwrtHeadingXy = desErrPosFootRtWorldEwrtHeadingXy,
      .clipped = clipped};
}

ContactPlan HlipContactPlanner::plan(const ContactPlannerInput& input) {
  const std::chrono::steady_clock::time_point startedAt =
      std::chrono::steady_clock::now();
  ContactPlan plan = initialPlan(input);

  const Rotation2 rotmatHeadingRtWorld = rotmatFromYaw(input.yawHeadingRtWorld);
  const vector2_t cmdLinvelBaseRtWorldEwrtHeadingXy =
      rotmatHeadingRtWorld.transpose() * input.cmdLinvelBaseRtWorldEwrtWorldXy;
  const scalar_t alpha = blendWeight(input);
  const bool walking = alpha >= HlipStandingBlend::kHalfPointWeight;
  // Below the blend's half point the planner stands; above it the stride grows
  // with alpha out of standing.
  const vector2_t plannedCmdLinvelBaseRtWorldEwrtHeadingXy =
      alpha * cmdLinvelBaseRtWorldEwrtHeadingXy;

  const std::vector<GaitPhase> gait =
      buildGait(input, walking ? GaitKind::kWalking : GaitKind::kStanding);
  const GaitRollOut rollOut =
      rollOutGait(input, gait, alpha, plannedCmdLinvelBaseRtWorldEwrtHeadingXy);
  plan.numClippedSteps += rollOut.numClippedSteps;
  fillNodes(input, gait, rollOut.phases, alpha, plan);
  // The standing / walking blend of the reference itself, equation (14) of the
  // paper: the rolled-out reduced model is blended against a static standing
  // reference, the center of the support. Without it a standing plan would hand
  // the controller the measured drift as its center-of-mass reference and ask
  // it to keep drifting, since the reduced model's double support has no way to
  // decelerate; with it, standing asks for a center of mass at rest over the
  // feet. The support is the one the robot will stand on: the feet after the
  // last phase, with a swing that was still in flight at the planning instant
  // at its landing spot, not at the measured position of a foot in the air.
  //
  // It applies ONLY while standing. A stepping plan must publish the trajectory
  // its own footholds were placed for: planned_com_override writes this into
  // the whole-body MPC's reference, and the deadbeat step was computed against
  // the unblended roll-out. Blending a stepping plan asked the controller for a
  // smaller lateral sway than the step assumed, which is the conflict section
  // 3c of the README calls fatal, at reduced amplitude - and with the shipped
  // blend it bit exactly in the 0.10-0.17 m/s band a first cautious walk is
  // commanded in. There is a step in the reference across alpha = 0.5, but the
  // gait itself already steps there (one long double support on one side, a
  // stepping cadence on the other) and so does the roll-out that produced it.
  // The ZMP each continuous phase's center of mass falls about: the stance foot
  // of a single support, and none in double support or flight, where the
  // roll-out moves at constant velocity. The standing blend is affine, so a
  // blended single support is the LIP about the blended stance foot, and a
  // blended double support still moves at constant velocity.
  fillPhaseZmps(gait, rollOut.phases, plan);
  if (!walking) {
    blendTowardsStanding(rollOut.phases.back().desPosFeetRtWorldEwrtWorldXy,
                         alpha, plan);
  }
  fillContactsAndPhases(input, gait, plan);

  plan.solveTime = std::chrono::duration<scalar_t>(
                       std::chrono::steady_clock::now() - startedAt)
                       .count();
  return plan;
}

ContactPlan HlipContactPlanner::initialPlan(
    const ContactPlannerInput& input) const {
  const PlannerSettings& settings = config_.planner;
  const int numIntervals = settings.numNodes;
  const int numNodes = numIntervals + 1;
  ContactPlan plan;
  plan.valid = true;
  plan.optimal = true;
  plan.startTime = input.time;
  plan.dt = settings.dt;
  plan.committedUntil = input.committedUntil;
  plan.yawHeadingRtWorld = input.yawHeadingRtWorld;
  plan.omega = config_.omega();
  plan.contacts.assign(numIntervals, makeFeetArray(true));
  plan.desPosFeetRtWorldEwrtWorldXyTraj.assign(numNodes,
                                               input.posFeetRtWorldEwrtWorldXy);
  plan.posComRtWorldEwrtWorldXyTraj.assign(numNodes,
                                           input.posComRtWorldEwrtWorldXy);
  plan.linvelComRtWorldEwrtWorldXyTraj.assign(
      numNodes, input.proxyLinvelComRtWorldEwrtWorldXy);
  plan.posZmpRtWorldEwrtWorldXyTraj.assign(numIntervals,
                                           input.posComRtWorldEwrtWorldXy);
  if (config_.usesHeadingModel()) {
    plan.headingModelYawTraj.assign(numNodes, input.initYawHeadingRtWorld);
    plan.headingModelYawRateTraj.assign(numNodes, 0.0);
    plan.desEulerzyxFeetRtWorldYawTraj.assign(numNodes,
                                              input.eulerzyxFeetRtWorldYaw);
  }
  return plan;
}

HlipContactPlanner::GaitRollOut HlipContactPlanner::rollOutGait(
    const ContactPlannerInput& input,
    const std::vector<GaitPhase>& gait,
    scalar_t alpha,
    const vector2_t& plannedCmdLinvelBaseRtWorldEwrtHeadingXy) const {
  // The reduced model at the start of every phase, and the feet during it,
  // rolled forward through the gait in continuous time: the deadbeat step of a
  // swing is evaluated at the pre-impact state of the phase's own end time,
  // which inside the committed window is the executed touch-down.
  GaitRollOut rollOut;
  rollOut.phases.reserve(gait.size());
  vector2_t posComRtWorldEwrtWorldXy = input.posComRtWorldEwrtWorldXy;
  // The LIP starts from the input's CoM velocity.
  vector2_t linvelComRtWorldEwrtWorldXy =
      input.proxyLinvelComRtWorldEwrtWorldXy;
  feet_array_t<vector2_t> desPosFeetRtWorldEwrtWorldXy =
      input.posFeetRtWorldEwrtWorldXy;
  feet_array_t<scalar_t> desEulerzyxFeetRtWorldYaw =
      input.eulerzyxFeetRtWorldYaw;
  for (const GaitPhase& phase : gait) {
    PhaseRollOut entry;
    entry.posComRtWorldEwrtWorldXy = posComRtWorldEwrtWorldXy;
    entry.linvelComRtWorldEwrtWorldXy = linvelComRtWorldEwrtWorldXy;
    if (phase.isSingleSupport()) {
      const scalar_t yawHeadingRtWorld =
          computeYawHeadingRtWorld(input, phase.endTime, alpha);
      entry.rotmatHeadingRtWorld = rotmatFromYaw(yawHeadingRtWorld);
      entry.stanceFoot = otherFoot(static_cast<size_t>(phase.swingFoot));
      const vector2_t posComRtFootEwrtHeadingXy =
          entry.rotmatHeadingRtWorld.transpose() *
          (posComRtWorldEwrtWorldXy -
           desPosFeetRtWorldEwrtWorldXy[entry.stanceFoot]);
      const vector2_t linvelComRtWorldEwrtHeadingXy =
          entry.rotmatHeadingRtWorld.transpose() * linvelComRtWorldEwrtWorldXy;
      entry.stateX << posComRtFootEwrtHeadingXy(0),
          linvelComRtWorldEwrtHeadingXy(0);
      entry.stateY << posComRtFootEwrtHeadingXy(1),
          linvelComRtWorldEwrtHeadingXy(1);

      // The landing spot of this swing: the deadbeat step evaluated at the
      // pre-impact state the phase ends in.
      const HlipModel::State preImpactX =
          model_.flowSingleSupport(entry.stateX, phase.duration());
      const HlipModel::State preImpactY =
          model_.flowSingleSupport(entry.stateY, phase.duration());
      const DeadbeatStep step = deadbeatStep(
          preImpactX, preImpactY, plannedCmdLinvelBaseRtWorldEwrtHeadingXy,
          static_cast<size_t>(phase.swingFoot));
      if (step.clipped) ++rollOut.numClippedSteps;
      desPosFeetRtWorldEwrtWorldXy[phase.swingFoot] =
          desPosFeetRtWorldEwrtWorldXy[entry.stanceFoot] +
          entry.rotmatHeadingRtWorld * step.desErrPosFootRtWorldEwrtHeadingXy;
      desEulerzyxFeetRtWorldYaw[phase.swingFoot] = yawHeadingRtWorld;

      posComRtWorldEwrtWorldXy =
          desPosFeetRtWorldEwrtWorldXy[entry.stanceFoot] +
          entry.rotmatHeadingRtWorld * vector2_t(preImpactX(0), preImpactY(0));
      linvelComRtWorldEwrtWorldXy =
          entry.rotmatHeadingRtWorld * vector2_t(preImpactX(1), preImpactY(1));
    } else {
      // Double support and standing: the reduced model drifts at constant
      // velocity, as the H-LIP assumes.
      posComRtWorldEwrtWorldXy +=
          phase.duration() * linvelComRtWorldEwrtWorldXy;
    }
    entry.desPosFeetRtWorldEwrtWorldXy = desPosFeetRtWorldEwrtWorldXy;
    entry.desEulerzyxFeetRtWorldYaw = desEulerzyxFeetRtWorldYaw;
    rollOut.phases.push_back(entry);
  }
  return rollOut;
}

HlipContactPlanner::ComState HlipContactPlanner::comAt(
    const GaitPhase& phase, const PhaseRollOut& entry, scalar_t time) const {
  const scalar_t elapsed = std::max(0.0, time - phase.startTime);
  if (!phase.isSingleSupport()) {
    return {.posComRtWorldEwrtWorldXy =
                vector2_t(entry.posComRtWorldEwrtWorldXy +
                          elapsed * entry.linvelComRtWorldEwrtWorldXy),
            .linvelComRtWorldEwrtWorldXy = entry.linvelComRtWorldEwrtWorldXy};
  }
  const HlipModel::State flowedX =
      model_.flowSingleSupport(entry.stateX, elapsed);
  const HlipModel::State flowedY =
      model_.flowSingleSupport(entry.stateY, elapsed);
  return {.posComRtWorldEwrtWorldXy =
              entry.desPosFeetRtWorldEwrtWorldXy[entry.stanceFoot] +
              entry.rotmatHeadingRtWorld * vector2_t(flowedX(0), flowedY(0)),
          .linvelComRtWorldEwrtWorldXy =
              entry.rotmatHeadingRtWorld * vector2_t(flowedX(1), flowedY(1))};
}

void HlipContactPlanner::fillNodes(const ContactPlannerInput& input,
                                   const std::vector<GaitPhase>& gait,
                                   const std::vector<PhaseRollOut>& rollOut,
                                   scalar_t alpha,
                                   ContactPlan& plan) const {
  const int numNodes = config_.planner.numNodes + 1;
  const scalar_t dt = config_.planner.dt;
  const bool withHeading = config_.usesHeadingModel();
  // The planning heading turns at this rate (computeYawHeadingRtWorld()); a
  // consumer without the heading model rebuilds a node's frame from it.
  plan.rateYawHeadingRtWorld = alpha * input.cmdRateYawBaseRtWorld;
  // Every node. The per-node foot quantities follow the phase the node is
  // assigned to (GaitPhase::endNode), the same integer boundaries the
  // per-interval contacts below are filled from, so a foothold changes on
  // exactly the node its contact does. They used to be assigned by two
  // different rules - nodes compared the node's own time against the boundary,
  // intervals compared their midpoint - which put interval k in the next phase
  // while node k was still in the current one at every phase boundary in the
  // plan. The center of mass follows the continuous roll-out instead: it is the
  // reduced model at the node's own time, whichever side of a rounded boundary
  // that time falls on.
  size_t gridPhase = 0;
  size_t timePhase = 0;
  for (int node = 0; node < numNodes; ++node) {
    while (gridPhase + 1 < gait.size() && node >= gait[gridPhase].endNode) {
      ++gridPhase;
    }
    const scalar_t nodeTime = input.time + dt * static_cast<scalar_t>(node);
    while (timePhase + 1 < gait.size() && nodeTime >= gait[timePhase].endTime) {
      ++timePhase;
    }
    const ComState com = comAt(gait[timePhase], rollOut[timePhase], nodeTime);
    plan.posComRtWorldEwrtWorldXyTraj[node] = com.posComRtWorldEwrtWorldXy;
    plan.linvelComRtWorldEwrtWorldXyTraj[node] =
        com.linvelComRtWorldEwrtWorldXy;
    plan.desPosFeetRtWorldEwrtWorldXyTraj[node] =
        rollOut[gridPhase].desPosFeetRtWorldEwrtWorldXy;
    if (withHeading) {
      plan.headingModelYawTraj[node] =
          computeYawHeadingRtWorld(input, nodeTime, alpha);
      plan.headingModelYawRateTraj[node] = alpha * input.cmdRateYawBaseRtWorld;
      plan.desEulerzyxFeetRtWorldYawTraj[node] =
          rollOut[gridPhase].desEulerzyxFeetRtWorldYaw;
    }
  }
}

void HlipContactPlanner::fillPhaseZmps(const std::vector<GaitPhase>& gait,
                                       const std::vector<PhaseRollOut>& rollOut,
                                       ContactPlan& plan) {
  plan.phasePosZmpRtWorldEwrtWorldXyTraj.reserve(gait.size());
  for (size_t phase = 0; phase < gait.size(); ++phase) {
    if (gait[phase].isSingleSupport()) {
      plan.phasePosZmpRtWorldEwrtWorldXyTraj.emplace_back(
          rollOut[phase]
              .desPosFeetRtWorldEwrtWorldXy[rollOut[phase].stanceFoot]);
    } else {
      plan.phasePosZmpRtWorldEwrtWorldXyTraj.emplace_back(std::nullopt);
    }
  }
}

void HlipContactPlanner::blendTowardsStanding(
    const feet_array_t<vector2_t>& posFeetRtWorldEwrtWorldXy,
    scalar_t alpha,
    ContactPlan& plan) {
  const vector2_t posFeetmeanRtWorldEwrtWorldXy =
      0.5 * (posFeetRtWorldEwrtWorldXy[kContactLeftIndex] +
             posFeetRtWorldEwrtWorldXy[kContactRightIndex]);
  for (vector2_t& posComRtWorldEwrtWorldXy :
       plan.posComRtWorldEwrtWorldXyTraj) {
    posComRtWorldEwrtWorldXy = alpha * posComRtWorldEwrtWorldXy +
                               (1.0 - alpha) * posFeetmeanRtWorldEwrtWorldXy;
  }
  for (vector2_t& linvelComRtWorldEwrtWorldXy :
       plan.linvelComRtWorldEwrtWorldXyTraj) {
    linvelComRtWorldEwrtWorldXy *= alpha;
  }
  for (std::optional<vector2_t>& posZmpRtWorldEwrtWorldXy :
       plan.phasePosZmpRtWorldEwrtWorldXyTraj) {
    if (posZmpRtWorldEwrtWorldXy.has_value()) {
      posZmpRtWorldEwrtWorldXy =
          vector2_t(alpha * *posZmpRtWorldEwrtWorldXy +
                    (1.0 - alpha) * posFeetmeanRtWorldEwrtWorldXy);
    }
  }
}

void HlipContactPlanner::fillContactsAndPhases(
    const ContactPlannerInput& input,
    const std::vector<GaitPhase>& gait,
    ContactPlan& plan) const {
  const int numIntervals = config_.planner.numNodes;
  const scalar_t dt = config_.planner.dt;
  // Contacts per interval, from the same integer phase boundaries the nodes
  // were filled from.
  size_t phaseIndex = 0;
  for (int interval = 0; interval < numIntervals; ++interval) {
    while (phaseIndex + 1 < gait.size() &&
           interval >= gait[phaseIndex].endNode) {
      ++phaseIndex;
    }
    plan.contacts[interval] = gait[phaseIndex].contacts;
  }

  // The committed window is no longer stamped over the contacts here:
  // buildGait() now starts FROM it, so the contact sequence, the footholds and
  // the center-of-mass roll-out all describe the one gait.

  // The same gait in continuous time, which the mode schedule is built from
  // (ContactPlan::toModeSchedule). The intervals above round every event to its
  // nearest node, and a schedule built from them executed each double support
  // for whatever the grid rounded it to, while the deadbeat step of the swing
  // before it had been computed for hlip.dspDuration exactly. The events
  // decided here are continuous - a touch-down sspDuration after its lift-off,
  // a lift-off dspDuration after its touch-down - and the committed window is
  // replayed at the executed event times, so every later plan re-deciding a
  // lift-off keeps it where the deadbeat gain assumes it.
  plan.phaseContacts.reserve(gait.size());
  plan.phaseStartTimes.reserve(gait.size());
  for (const GaitPhase& phase : gait) {
    plan.phaseContacts.push_back(phase.contacts);
    plan.phaseStartTimes.push_back(phase.startTime);
  }

  // The per-interval ZMP, for display: the reduced model's own ZMP at the
  // interval's midpoint. It is not constant over an interval - in double
  // support the constant-velocity roll-out's ZMP is the CoM itself, and an
  // event can fall inside an interval - so lipReferenceState() reads the
  // phases, not this. It used to hold the midpoint of the feet in double
  // support, a ZMP inconsistent with the constant-velocity CoM the plan
  // publishes there.
  for (int interval = 0; interval < numIntervals; ++interval) {
    const scalar_t midpoint =
        input.time + dt * static_cast<scalar_t>(interval) + 0.5 * dt;
    const std::optional<LipState> state =
        lipReferenceState(plan, plan.omega, midpoint);
    if (state.has_value()) {
      plan.posZmpRtWorldEwrtWorldXyTraj[interval] =
          state->posZmpRtWorldEwrtWorldXy;
    }
  }
}

absl::StatusOr<scalar_t> HlipContactPlanner::startUpLateralStep(
    const ContactPlanningConfig& config) {
  ASSIGN_OR_RETURN(const HlipModel model, makeModel(config));
  return startUpLateralStepOf(config, model);
}

absl::StatusOr<std::vector<scalar_t>>
HlipContactPlanner::computeStartUpNormPosLfootRtRfootEwrtHeadingYTraj(
    const ContactPlanningConfig& config, int numSteps) {
  ASSIGN_OR_RETURN(const HlipModel model, makeModel(config));
  return rollOutStartUpNormPosLfootRtRfootEwrtHeadingYTraj(config, model,
                                                           numSteps);
}

absl::StatusOr<std::string> HlipContactPlanner::formulationSummary(
    const ContactPlanningConfig& config) {
  ASSIGN_OR_RETURN(const HlipModel model, makeModel(config));
  return summaryOf(config, model);
}

scalar_t HlipContactPlanner::startUpLateralStepOf(
    const ContactPlanningConfig& config, const HlipModel& model) {
  const HlipParameters& params = config.hlip;
  // Standing: the center of mass sits half a step width from the stance foot
  // with no lateral velocity, which is the worst lateral state the gait ever
  // starts from.
  const HlipModel::State atLiftOff(
      0.5 * params.nomPosLfootRtRfootEwrtHeadingY,
      0.0);  // NOLINT(argument-comment): a vector2_t [p, v]
  const HlipModel::State preImpact =
      model.flowSingleSupport(atLiftOff, model.sspDuration());
  const HlipModel::PeriodTwoOrbit orbit =
      model.periodTwoOrbit(params.nomPosLfootRtRfootEwrtHeadingY,
                           -params.nomPosLfootRtRfootEwrtHeadingY);
  return std::abs(model.deadbeatStepLength(
      preImpact, orbit.beforeFirstStep, params.nomPosLfootRtRfootEwrtHeadingY));
}

std::vector<scalar_t>
HlipContactPlanner::rollOutStartUpNormPosLfootRtRfootEwrtHeadingYTraj(
    const ContactPlanningConfig& config, const HlipModel& model, int numSteps) {
  const HlipParameters& params = config.hlip;
  const HlipModel::PeriodTwoOrbit orbit =
      model.periodTwoOrbit(params.nomPosLfootRtRfootEwrtHeadingY,
                           -params.nomPosLfootRtRfootEwrtHeadingY);
  // Standing on the right foot with the left about to swing, the center of mass
  // half a step width to its left, at rest: the state startUpLateralStep()
  // starts from. Each step is the deadbeat step under the planner's clip.
  HlipModel::State state(0.5 * params.nomPosLfootRtRfootEwrtHeadingY,
                         0.0);  // NOLINT(argument-comment): a vector2_t [p, v]
  bool placesLeftFoot = true;
  std::vector<scalar_t> normPosLfootRtRfootEwrtHeadingYTraj;
  normPosLfootRtRfootEwrtHeadingYTraj.reserve(
      static_cast<size_t>(std::max(0, numSteps)));
  for (int step = 0; step < numSteps; ++step) {
    const HlipModel::State preImpact =
        model.flowSingleSupport(state, model.sspDuration());
    const scalar_t nomErrPosFootRtWorldEwrtHeadingY =
        placesLeftFoot ? params.nomPosLfootRtRfootEwrtHeadingY
                       : -params.nomPosLfootRtRfootEwrtHeadingY;
    const scalar_t deadbeatErrPosFootRtWorldEwrtHeadingY =
        model.deadbeatStepLength(
            preImpact,
            placesLeftFoot ? orbit.beforeFirstStep : orbit.beforeSecondStep,
            nomErrPosFootRtWorldEwrtHeadingY);
    const scalar_t desErrPosFootRtWorldEwrtHeadingY =
        clipDesErrPosFootRtWorldEwrtHeadingY(
            params, deadbeatErrPosFootRtWorldEwrtHeadingY, placesLeftFoot);
    normPosLfootRtRfootEwrtHeadingYTraj.push_back(
        std::abs(desErrPosFootRtWorldEwrtHeadingY));
    state = HlipModel::flowDoubleSupport(
        HlipModel::applyStepTransition(preImpact,
                                       desErrPosFootRtWorldEwrtHeadingY),
        model.dspDuration());
    placesLeftFoot = !placesLeftFoot;
  }
  return normPosLfootRtRfootEwrtHeadingYTraj;
}

std::string HlipContactPlanner::summaryOf(const ContactPlanningConfig& config,
                                          const HlipModel& model) {
  const HlipParameters& params = config.hlip;
  const scalar_t stepDuration = model.stepDuration();
  const scalar_t startUpDemand = startUpLateralStepOf(config, model);
  const std::vector<scalar_t> startUpNormPosLfootRtRfootEwrtHeadingYTraj =
      rollOutStartUpNormPosLfootRtRfootEwrtHeadingYTraj(config, model,
                                                        kStartUpSteps);
  const std::vector<std::string>& execution = config.formulation.execution;
  return absl::StrCat(
      "[HlipContactPlanner] closed-form H-LIP contact planner "
      "(arXiv:2502.15630)\n",
      "  cadence      : single support ", params.sspDuration,
      " s, double support ", params.dspDuration, " s, stride ",
      2.0 * stepDuration, " s\n", "  pendulum     : height ",
      config.getNomPosComRtFeetmeanEwrtWorldZ(), " m, omega ",
      model.naturalFrequency(), " 1/s\n", "  step law     : deadbeat, K = [",
      model.deadbeatGain()(0), ", ", model.deadbeatGain()(1),
      "] (closed form, nothing tuned)\n",
      "  nominal orbit: period one along the heading, ",
      "period two laterally at a step width of ",
      params.nomPosLfootRtRfootEwrtHeadingY, " m\n",
      "  step clip    : |dx| <= ", params.maxNormPosLfootRtRfootEwrtHeadingX,
      " m, step width in [", params.minPosLfootRtRfootEwrtHeadingY, ", ",
      params.maxPosLfootRtRfootEwrtHeadingY, "] m\n",
      "  stand / walk : alpha = tanh(", params.blend.sharpness, " (phi - ",
      params.blend.threshold,
      ")) / 2 + 1/2, lateral velocity measured against the zero-command "
      "stepping orbit\n",
      "  grid         : ", config.planner.numNodes, " intervals x ",
      config.planner.dt, " s = ", config.horizon(), " s\n",
      "  heading model: ", config.usesHeadingModel() ? "on" : "off", "\n",
      "  start-up     : the first step out of a standstill needs ",
      startUpDemand,
      " m of lateral step, against an "
      "hlip.max_pos_lfoot_rt_rfoot_ewrt_heading_y of ",
      params.maxPosLfootRtRfootEwrtHeadingY,
      startUpVerdict(config, startUpDemand,
                     startUpNormPosLfootRtRfootEwrtHeadingYTraj),
      "  no optimization is performed: no costs, no constraints, no search "
      "(the formulation's cost, constraint, logic and search "
      "lists are not read)\n",
      "  execution    : ",
      execution.empty() ? std::string("none") : absl::StrJoin(execution, ", "),
      " (applied by the reference manager under either planner; "
      "planned_com_override is required here)");
}

}  // namespace ocs2::humanoid
