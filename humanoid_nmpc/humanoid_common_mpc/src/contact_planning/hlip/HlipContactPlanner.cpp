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

#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <utility>

#include "absl/strings/str_cat.h"

namespace ocs2::humanoid {
namespace {

using Rotation2 = Eigen::Matrix<scalar_t, 2, 2>;

/** Phases shorter than this are not emitted: a swing whose touch-down is already due starts the next phase at once. */
constexpr scalar_t kMinPhaseDuration = 1e-6;

/** [m] a step cut by less than this is not counted as clipped. */
constexpr scalar_t kStepClipTolerance = 1e-9;

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
  return foot == CONTACT_LEFT_INDEX ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
}

}  // namespace

HlipModel HlipContactPlanner::makeModel(const ContactPlanningConfig& config) {
  return HlipModel(config.hlip.sspDuration, config.hlip.dspDuration, config.shared.comHeight, config.shared.gravity);
}

HlipContactPlanner::HlipContactPlanner(ContactPlanningConfig config)
    : config_(std::move(config)), model_(makeModel(config_)), blend_(config_.hlip.blend) {}

void HlipContactPlanner::setConfig(const ContactPlanningConfig& config) {
  config_ = config;
  model_ = makeModel(config_);
  blend_ = HlipStandingBlend(config_.hlip.blend);
}

size_t HlipContactPlanner::nextSwingFoot(const ContactPlannerInput& input) {
  if (input.lastSwungFoot >= 0) return otherFoot(static_cast<size_t>(input.lastSwungFoot));
  // Nothing has swung yet: the foot that has been in contact the longest has had the most time to be unloaded.
  return input.phaseElapsedTime[CONTACT_LEFT_INDEX] >= input.phaseElapsedTime[CONTACT_RIGHT_INDEX] ? CONTACT_LEFT_INDEX
                                                                                                   : CONTACT_RIGHT_INDEX;
}

bool HlipContactPlanner::isWalking(const ContactPlannerInput& input) const {
  const Rotation2 world_R_heading = rotation(input.yaw);
  const vector2_t commandInHeading = world_R_heading.transpose() * input.velocityCommand;
  const vector2_t velocityInHeading = world_R_heading.transpose() * input.comVelocity;
  return blend_.isWalking(commandInHeading, input.headingRateCommand, velocityInHeading);
}

scalar_t HlipContactPlanner::headingAt(const ContactPlannerInput& input, scalar_t time, scalar_t blendWeight) const {
  // `yaw` is the frame the geometry is planned in: the whole-body heading with the heading model, the base yaw
  // without it (ContactPlanningReferenceManager::makePlannerInput). `heading` is only filled with the model, so the
  // frame must come from `yaw` or a robot without the model would plan its footholds in the world frame.
  return input.yaw + blendWeight * input.headingRateCommand * (time - input.time);
}

std::vector<HlipContactPlanner::GaitPhase> HlipContactPlanner::buildGait(const ContactPlannerInput& input, bool walking) const {
  const scalar_t horizonEnd = input.time + config_.horizon();
  std::vector<GaitPhase> gait;

  if (!walking) {
    GaitPhase stance;
    stance.startTime = input.time;
    stance.endTime = horizonEnd;
    stance.contacts = makeFeetArray(true);
    gait.push_back(stance);
    return gait;
  }

  const scalar_t sspDuration = model_.sspDuration();
  const scalar_t dspDuration = model_.dspDuration();
  const std::function<scalar_t(scalar_t, scalar_t, const contact_flag_t&, int)> append =
      [&gait](scalar_t startTime, scalar_t duration, const contact_flag_t& contacts, int swingFoot) {
        if (duration <= kMinPhaseDuration) return startTime;
        GaitPhase phase;
        phase.startTime = startTime;
        phase.endTime = startTime + duration;
        phase.contacts = contacts;
        phase.swingFoot = swingFoot;
        gait.push_back(phase);
        return phase.endTime;
      };

  scalar_t time = input.time;
  size_t swingFoot = 0;
  if (input.contacts[CONTACT_LEFT_INDEX] && input.contacts[CONTACT_RIGHT_INDEX]) {
    // Double support: serve out what is left of it, then lift the foot whose turn it is.
    const scalar_t elapsed = std::min(input.phaseElapsedTime[CONTACT_LEFT_INDEX], input.phaseElapsedTime[CONTACT_RIGHT_INDEX]);
    time = append(time, std::max(0.0, dspDuration - elapsed), makeFeetArray(true), -1);
    swingFoot = nextSwingFoot(input);
  } else {
    // Single support: finish the swing in flight, keeping the cadence of the executed schedule, then hand over.
    swingFoot = input.contacts[CONTACT_LEFT_INDEX] ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
    const scalar_t elapsed = input.phaseElapsedTime[swingFoot];
    time = append(time, std::max(0.0, sspDuration - elapsed), stanceOnly(otherFoot(swingFoot)), static_cast<int>(swingFoot));
    time = append(time, dspDuration, makeFeetArray(true), -1);
    swingFoot = otherFoot(swingFoot);
  }

  // From here the cadence is nominal: single supports of sspDuration alternating between the feet, separated by
  // double supports of dspDuration. The horizon is covered with whole phases, so the last one may reach past its end.
  while (time < horizonEnd - kMinPhaseDuration) {
    time = append(time, sspDuration, stanceOnly(otherFoot(swingFoot)), static_cast<int>(swingFoot));
    time = append(time, dspDuration, makeFeetArray(true), -1);
    swingFoot = otherFoot(swingFoot);
  }
  return gait;
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
  const bool placesLeftFoot = swingFoot == CONTACT_LEFT_INDEX;
  const scalar_t nominalStepY = placesLeftFoot ? leftStep : rightStep;
  const HlipModel::State orbitStateY = placesLeftFoot ? orbitY.first : orbitY.second;

  const scalar_t sagittalStep = model_.deadbeatStepLength(preImpactX, orbitX, nominalStepX);
  const scalar_t lateralStep = model_.deadbeatStepLength(preImpactY, orbitStateY, nominalStepY);

  vector2_t step;
  step(0) = std::clamp(sagittalStep, -params.maxStepLength, params.maxStepLength);
  // The placed foot must stay on its own side of the stance foot, between the self-collision margin and the reach.
  step(1) = placesLeftFoot ? std::clamp(lateralStep, params.minStepWidth, params.maxStepWidth)
                           : std::clamp(lateralStep, -params.maxStepWidth, -params.minStepWidth);
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
  const vector2_t velocityInHeading = world_R_planning.transpose() * input.comVelocity;
  const scalar_t blendWeight = blend_.weight(commandInHeading, input.headingRateCommand, velocityInHeading);
  const bool walking = blendWeight >= 0.5;
  // Below the blend's half point the planner stands; above it the stride grows with alpha out of standing.
  const vector2_t plannedCommand = blendWeight * commandInHeading;

  const std::vector<GaitPhase> gait = buildGait(input, walking);

  // Rolled forward through the gait: the world state of the reduced model and of the feet.
  vector2_t comPosition = input.comPosition;
  vector2_t comVelocity = input.comVelocity;
  feet_array_t<vector2_t> feet = input.footPositions;
  feet_array_t<scalar_t> footYaws = input.footYaws;

  // The foot positions and yaws during each phase, so that the per-interval contacts and ZMP can be filled afterwards
  // from the phase that covers the interval's midpoint.
  std::vector<feet_array_t<vector2_t>> phaseFeet;
  phaseFeet.reserve(gait.size());

  int node = 0;
  for (const GaitPhase& phase : gait) {
    const scalar_t phaseHeading = headingAt(input, phase.endTime, blendWeight);
    const Rotation2 world_R_phase = rotation(phaseHeading);

    HlipModel::State stateX = HlipModel::State::Zero();
    HlipModel::State stateY = HlipModel::State::Zero();
    size_t stanceFoot = CONTACT_LEFT_INDEX;
    if (phase.isSingleSupport()) {
      stanceFoot = otherFoot(static_cast<size_t>(phase.swingFoot));
      const vector2_t positionInHeading = world_R_phase.transpose() * (comPosition - feet[stanceFoot]);
      const vector2_t velocityInPhase = world_R_phase.transpose() * comVelocity;
      stateX << positionInHeading(0), velocityInPhase(0);
      stateY << positionInHeading(1), velocityInPhase(1);

      // The landing spot of this swing: the deadbeat step evaluated at the pre-impact state the phase ends in.
      const HlipModel::State preImpactX = model_.flowSingleSupport(stateX, phase.duration());
      const HlipModel::State preImpactY = model_.flowSingleSupport(stateY, phase.duration());
      bool clipped = false;
      const vector2_t step = deadbeatStep(preImpactX, preImpactY, plannedCommand, static_cast<size_t>(phase.swingFoot), clipped);
      if (clipped) ++plan.numClippedSteps;
      feet[phase.swingFoot] = feet[stanceFoot] + world_R_phase * step;
      footYaws[phase.swingFoot] = phaseHeading;
    }
    phaseFeet.push_back(feet);

    // Sample the nodes that fall inside this phase. The last phase takes whatever is left, so that a horizon the
    // whole phases overshoot or fall short of is always fully covered.
    const bool isLastPhase = &phase == &gait.back();
    for (; node < numNodes; ++node) {
      const scalar_t nodeTime = input.time + dt * static_cast<scalar_t>(node);
      if (!isLastPhase && nodeTime >= phase.endTime - kMinPhaseDuration) break;
      const scalar_t elapsed = std::max(0.0, nodeTime - phase.startTime);
      vector2_t nodePosition;
      vector2_t nodeVelocity;
      if (phase.isSingleSupport()) {
        const HlipModel::State flowedX = model_.flowSingleSupport(stateX, elapsed);
        const HlipModel::State flowedY = model_.flowSingleSupport(stateY, elapsed);
        nodePosition = feet[stanceFoot] + world_R_phase * vector2_t(flowedX(0), flowedY(0));
        nodeVelocity = world_R_phase * vector2_t(flowedX(1), flowedY(1));
      } else {
        // Double support and standing: the reduced model drifts at constant velocity, as the H-LIP assumes.
        nodePosition = comPosition + elapsed * comVelocity;
        nodeVelocity = comVelocity;
      }
      plan.comPosition[node] = nodePosition;
      plan.comVelocity[node] = nodeVelocity;
      plan.footholds[node] = feet;
      if (withHeading) {
        plan.heading[node] = headingAt(input, nodeTime, blendWeight);
        plan.headingRate[node] = blendWeight * input.headingRateCommand;
        plan.footYaws[node] = footYaws;
      }
    }

    // Carry the state to the end of the phase.
    if (phase.isSingleSupport()) {
      const HlipModel::State endX = model_.flowSingleSupport(stateX, phase.duration());
      const HlipModel::State endY = model_.flowSingleSupport(stateY, phase.duration());
      comPosition = feet[stanceFoot] + world_R_phase * vector2_t(endX(0), endY(0));
      comVelocity = world_R_phase * vector2_t(endX(1), endY(1));
    } else {
      comPosition += phase.duration() * comVelocity;
    }
  }

  // The standing / walking blend of the reference itself, equation (14) of the paper: the rolled-out reduced model is
  // blended against a static standing reference, the centre of the support. Without it a standing plan would hand the
  // controller the measured drift as its centre-of-mass reference and ask it to keep drifting, since the reduced
  // model's double support has no way to decelerate; with it, standing asks for a centre of mass at rest over the feet.
  const vector2_t supportCentre = 0.5 * (input.footPositions[CONTACT_LEFT_INDEX] + input.footPositions[CONTACT_RIGHT_INDEX]);
  for (int node = 0; node < numNodes; ++node) {
    plan.comPosition[node] = blendWeight * plan.comPosition[node] + (1.0 - blendWeight) * supportCentre;
    plan.comVelocity[node] *= blendWeight;
  }

  // Contacts and ZMP per interval, from the phase that covers the interval's midpoint.
  size_t phaseIndex = 0;
  for (int interval = 0; interval < numIntervals; ++interval) {
    const scalar_t midTime = input.time + dt * (static_cast<scalar_t>(interval) + 0.5);
    while (phaseIndex + 1 < gait.size() && midTime >= gait[phaseIndex].endTime) ++phaseIndex;
    const GaitPhase& phase = gait[phaseIndex];
    plan.contacts[interval] = phase.contacts;
    if (phase.isSingleSupport()) {
      plan.zmp[interval] = phaseFeet[phaseIndex][otherFoot(static_cast<size_t>(phase.swingFoot))];
    } else {
      plan.zmp[interval] = 0.5 * (phaseFeet[phaseIndex][CONTACT_LEFT_INDEX] + phaseFeet[phaseIndex][CONTACT_RIGHT_INDEX]);
    }
  }

  // The executed schedule wins inside the commit window: the reference manager merges it over the plan anyway, and a
  // plan that disagrees with it there would be dropped as inconsistent.
  const int numCommitted = std::min(static_cast<int>(input.committedContacts.size()), numIntervals);
  for (int interval = 0; interval < numCommitted; ++interval) {
    plan.contacts[interval] = input.committedContacts[interval];
  }

  plan.solveTime = std::chrono::duration<scalar_t>(std::chrono::steady_clock::now() - startedAt).count();
  return plan;
}

std::string HlipContactPlanner::formulationSummary(const ContactPlanningConfig& config) {
  const HlipParameters& params = config.hlip;
  const HlipModel model = makeModel(config);
  const scalar_t stepDuration = model.stepDuration();
  return absl::StrCat("[HlipContactPlanner] closed-form H-LIP contact planner (arXiv:2502.15630)\n", "  cadence      : single support ",
                      params.sspDuration, " s, double support ", params.dspDuration, " s, stride ", 2.0 * stepDuration, " s\n",
                      "  pendulum     : height ", config.shared.comHeight, " m, omega ", model.naturalFrequency(), " 1/s\n",
                      "  step law     : deadbeat, K = [", model.deadbeatGain()(0), ", ", model.deadbeatGain()(1),
                      "] (closed form, nothing tuned)\n", "  nominal orbit: period one along the heading, ",
                      "period two laterally at a step width of ", params.stepWidth, " m\n",
                      "  step clip    : |dx| <= ", params.maxStepLength, " m, step width in [", params.minStepWidth, ", ",
                      params.maxStepWidth, "] m\n", "  stand / walk : alpha = tanh(", params.blend.sharpness, " (phi - ",
                      params.blend.threshold, ")) / 2 + 1/2\n", "  grid         : ", config.planner.numNodes, " intervals x ",
                      config.planner.dt, " s = ", config.horizon(), " s\n", "  heading model: ", config.usesHeadingModel() ? "on" : "off",
                      "\n", "  no optimization is performed: no costs, no constraints, no search, no execution rules.");
}

}  // namespace ocs2::humanoid
