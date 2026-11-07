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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/contact_planning/execution/DcmStepAdjustmentRule.h"
#include "humanoid_common_mpc/contact_planning/execution/EnergyCadenceModulationRule.h"
#include "humanoid_common_mpc/contact_planning/execution/ExecutionContext.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipContactPlanner.h"
#include "humanoid_common_mpc/contact_planning/hlip/HlipModel.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"

namespace ocs2::humanoid {
namespace {

// The biped's two feet, which every case here is written for.
constexpr size_t kNumFeet = static_cast<size_t>(kBipedFootCount);

constexpr scalar_t kNomPosLfootRtRfootEwrtHeadingY = 0.25;

/** The value of `optional`, or a failed expectation and a value-initialized T
 * (bugprone-unchecked-optional-access does not see googletest's ASSERTs). */
template <typename T>
T valueOrFailure(std::optional<T> optional) {
  if (!optional.has_value()) {
    ADD_FAILURE() << "the optional has no value";
    return T();
  }
  return *std::move(optional);
}

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.type = "hlip";
  config.planner.dt = 0.025;
  config.planner.numNodes = 56;
  config.planner.commitTime = 0.05;
  config.shared.nomPosComRtFeetmeanEwrtWorldZ = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.25;
  config.shared.gaitLimits.maxSwingDuration = 0.35;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.0;
  config.hlip.sspDuration = 0.25;
  config.hlip.dspDuration = 0.05;
  config.hlip.nomPosLfootRtRfootEwrtHeadingY = kNomPosLfootRtRfootEwrtHeadingY;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  return config;
}

/** A robot standing in double support, with the feet symmetric about the
 * origin. */
ContactPlannerInput makeStandingInput(
    const vector2_t& cmdLinvelBaseRtWorldEwrtWorldXy) {
  ContactPlannerInput input;
  input.time = 0.0;
  input.posComRtWorldEwrtWorldXy = vector2_t(0.0, 0.0);
  input.proxyLinvelComRtWorldEwrtWorldXy = vector2_t::Zero();
  input.posFeetRtWorldEwrtWorldXy[kContactLeftIndex] =
      vector2_t(0.0, 0.5 * kNomPosLfootRtRfootEwrtHeadingY);
  input.posFeetRtWorldEwrtWorldXy[kContactRightIndex] =
      vector2_t(0.0, -0.5 * kNomPosLfootRtRfootEwrtHeadingY);
  input.contacts = makeBipedFeetArray(true);
  input.phaseElapsedTime = makeBipedFeetArray(1.0);
  input.cmdLinvelBaseRtWorldEwrtWorldXy = cmdLinvelBaseRtWorldEwrtWorldXy;
  input.committedUntil = input.time;
  return input;
}

/** The feet the planner placed over the horizon, in the order they land. */
std::vector<size_t> swingOrder(const ContactPlan& plan) {
  std::vector<size_t> order;
  BipedFeetArray<bool> previous = plan.contacts.front();
  for (const BipedFeetArray<bool>& contacts : plan.contacts) {
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      if (previous[foot] && !contacts[foot]) order.push_back(foot);
    }
    previous = contacts;
  }
  return order;
}

size_t otherFootOf(size_t foot) {
  return foot == kContactLeftIndex ? kContactRightIndex : kContactLeftIndex;
}

Eigen::Matrix<scalar_t, 2, 2> planarRotmatHeadingRtWorld(
    scalar_t yawHeadingRtWorld) {
  Eigen::Matrix<scalar_t, 2, 2> matrix;
  matrix << std::cos(yawHeadingRtWorld), -std::sin(yawHeadingRtWorld),
      std::sin(yawHeadingRtWorld), std::cos(yawHeadingRtWorld);
  return matrix;
}

/** A swing of the plan: which foot, the interval it lifts off and the interval
 * it is down again. */
struct PlannedSwing {
  size_t foot;
  int liftOffInterval;
  int touchDownInterval;
};

/** Every swing of the plan that lifts off and lands inside the horizon. */
std::vector<PlannedSwing> plannedSwings(const ContactPlan& plan) {
  std::vector<PlannedSwing> swings;
  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    int liftOff = -1;
    for (int interval = 1; interval < plan.numIntervals(); ++interval) {
      const bool wasDown =
          plan.contacts[static_cast<size_t>(interval - 1)][foot];
      const bool isDown = plan.contacts[static_cast<size_t>(interval)][foot];
      if (wasDown && !isDown) liftOff = interval;
      if (!wasDown && isDown && liftOff >= 0) {
        swings.push_back(PlannedSwing{.foot = foot,
                                      .liftOffInterval = liftOff,
                                      .touchDownInterval = interval});
        liftOff = -1;
      }
    }
  }
  std::sort(swings.begin(), swings.end(),
            [](const PlannedSwing& first, const PlannedSwing& second) {
              return first.liftOffInterval < second.liftOffInterval;
            });
  return swings;
}

/** The planner of a configuration the test knows to be valid; ends the test
 * binary otherwise. */
std::unique_ptr<HlipContactPlanner> createPlanner(
    const ContactPlanningConfig& config) {
  absl::StatusOr<std::unique_ptr<HlipContactPlanner>> planner =
      HlipContactPlanner::Create(config);
  ABSL_CHECK_OK(planner.status());
  return *std::move(planner);
}

/**
 * The planner closed around its own reduced model at the MPC rate, with the
 * executed schedule in between kept the way ContactPlanningReferenceManager
 * keeps it. Every cycle builds the planner input from that schedule exactly as
 * makePlannerInput does (commit boundary extended to the touch-down of a swing
 * in flight, committed contacts and their event times sampled with
 * committedContactsForPlanner / committedPhaseStartsForPlanner), plans, merges
 * the plan past the commit boundary with mergeModeSchedules, and flows the
 * H-LIP through the executed contacts to the next cycle, landing each swing
 * where the plan active at its touch-down put it. A foot in flight is reported
 * where its latest plan sends it, as if the swing reference were tracked
 * perfectly. The touch-downs of one plan are therefore off the node grid of the
 * next, as they are at runtime. What it leaves out is the whole-body MPC: the
 * center of mass follows the reduced model exactly, and its double support
 * drifts at constant velocity - except under a standing plan, which asks the
 * MPC to bring the center of mass to rest over the feet, and which the MPC can
 * do with both feet down; there the harness moves the center-of-mass velocity
 * towards the plan's reference with a first-order lag (kStandingTimeConstant).
 * Without that the reduced model would keep sliding sideways at the orbit's
 * 0.165 m/s for ever after the robot stood, which no standing controller
 * allows. The lag is deliberately not zero: a controller that stopped the
 * center of mass instantly would hide the double supports in which the robot is
 * still slowing down from the orbit's sway, and those are the ones the blend
 * has to read as rest
 * (HlipContactPlanner::computeBlendLinvelComRtWorldEwrtHeadingXy).
 */
class ReducedModelLoop {
 public:
  struct Event {
    size_t foot;
    scalar_t time;
    bool touchDown;
    // At a touch-down: the landed foot minus the stance foot.
    vector2_t errPosFootRtWorldEwrtWorldXy = vector2_t::Zero();
    // At a touch-down: the center of mass relative to the stance foot at the
    // impact, and the blend weight and command of the plan the foot was landed
    // from.
    HlipModel::State preImpactX = HlipModel::State::Zero();
    HlipModel::State preImpactY = HlipModel::State::Zero();
    scalar_t alpha = 0.0;
    vector2_t cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t::Zero();
  };

  ReducedModelLoop(const ContactPlanningConfig& config, scalar_t cyclePeriod)
      : config_(config),
        planner_(createPlanner(config)),
        cyclePeriod_(cyclePeriod),
        applied_(/*eventTimesInput=*/{}, {kBipedModeStance}) {
    posFeetRtWorldEwrtWorldXy_[kContactLeftIndex] =
        vector2_t(0.0, 0.5 * config.hlip.nomPosLfootRtRfootEwrtHeadingY);
    posFeetRtWorldEwrtWorldXy_[kContactRightIndex] =
        vector2_t(0.0, -0.5 * config.hlip.nomPosLfootRtRfootEwrtHeadingY);
    posComRtWorldEwrtWorldXy_ = vector2_t::Zero();
    linvelComRtWorldEwrtWorldXy_ = vector2_t::Zero();
    contacts_ = makeBipedFeetArray(true);
  }

  void setCmdLinvelBaseRtWorldEwrtWorldXy(
      const vector2_t& cmdLinvelBaseRtWorldEwrtWorldXy) {
    cmdLinvelBaseRtWorldEwrtWorldXy_ = cmdLinvelBaseRtWorldEwrtWorldXy;
  }

  /**
   * Builds the executed schedule from the plans' node grid alone, as it was
   * built before plans carried their gait in continuous time
   * (ContactPlan::phaseStartTimes): the positive control of the closed-loop
   * timing tests.
   */
  void useNodeGridSchedule() { useNodeGridSchedule_ = true; }

  /** One MPC cycle. Returns whether the plan made in it was a stepping plan. */
  bool cycle() {
    const ContactPlannerInput input = makeInput();
    ContactPlan plan = planner_->plan(input);
    EXPECT_TRUE(plan.valid);
    if (useNodeGridSchedule_) {
      plan.phaseContacts.clear();
      plan.phaseStartTimes.clear();
      plan.phasePosZmpRtWorldEwrtWorldXyTraj.clear();
    }
    const bool walking = planner_->isWalking(input);
    activeAlpha_ = planner_->blendWeight(input);
    applied_ =
        mergeModeSchedules(applied_, plan.toModeSchedule(),
                           input.committedUntil, time_ - 2.0, plan.endTime());
    activePlan_ = plan;
    activePlanWalks_ = walking;
    // A foot in flight follows its swing reference to the landing spot of the
    // plan now active.
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      if (contacts_[foot]) continue;
      const std::optional<size_t> touchDown =
          touchDownEventIndex(applied_, foot, time_);
      if (!touchDown.has_value()) continue;
      const std::optional<vector2_t> desPosFootRtWorldEwrtWorldXy =
          plan.getDesPosFootRtWorldEwrtWorldXy(foot,
                                               applied_.eventTimes[*touchDown]);
      if (desPosFootRtWorldEwrtWorldXy.has_value()) {
        posFeetRtWorldEwrtWorldXy_[foot] = *desPosFootRtWorldEwrtWorldXy;
      }
    }
    advanceTo(cycleStart_ + cyclePeriod_ * static_cast<scalar_t>(++numCycles_));
    return walking;
  }

  scalar_t time() const { return time_; }
  const std::vector<Event>& events() const { return events_; }
  /** The executed schedule; it keeps two seconds of history, as the reference
   * manager keeps one horizon. */
  const ModeSchedule& executedSchedule() const { return applied_; }
  const HlipContactPlanner& planner() const { return *planner_; }

  /** The planner input at `time_`, read off the executed schedule as
   * ContactPlanningReferenceManager::makePlannerInput does. */
  ContactPlannerInput makeInput() {
    ContactPlannerInput input;
    input.time = time_;
    input.posComRtWorldEwrtWorldXy = posComRtWorldEwrtWorldXy_;
    input.proxyLinvelComRtWorldEwrtWorldXy = linvelComRtWorldEwrtWorldXy_;
    input.posFeetRtWorldEwrtWorldXy = posFeetRtWorldEwrtWorldXy_;
    input.cmdLinvelBaseRtWorldEwrtWorldXy = cmdLinvelBaseRtWorldEwrtWorldXy_;
    input.committedUntil =
        commitBoundaryForSchedule(applied_, time_, config_.planner.commitTime,
                                  config_.planner.maxCommitExtension);
    fillPlannerInputFromSchedule(applied_, config_.planner.dt,
                                 std::max(0, config_.planner.numNodes - 1),
                                 liftOffHistory_, input);
    return input;
  }

 private:
  /** Applies the contact events of the executed schedule at `time_`: lands
   * swings, records lift-offs. */
  void applyEvents() {
    const BipedFeetArray<bool> now =
        contactFlagsAtTime(applied_, time_ + kEventTolerance);
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      if (contacts_[foot] == now[foot]) continue;
      Event event{.foot = foot, .time = time_, .touchDown = now[foot]};
      if (now[foot]) {
        const std::optional<vector2_t> desPosFootRtWorldEwrtWorldXy =
            activePlan_.getDesPosFootRtWorldEwrtWorldXy(foot, time_);
        EXPECT_TRUE(desPosFootRtWorldEwrtWorldXy.has_value());
        if (desPosFootRtWorldEwrtWorldXy.has_value()) {
          posFeetRtWorldEwrtWorldXy_[foot] = *desPosFootRtWorldEwrtWorldXy;
        }
        const vector2_t stancePosFootRtWorldEwrtWorldXy =
            posFeetRtWorldEwrtWorldXy_[otherFootOf(foot)];
        event.errPosFootRtWorldEwrtWorldXy =
            posFeetRtWorldEwrtWorldXy_[foot] - stancePosFootRtWorldEwrtWorldXy;
        event.preImpactX = HlipModel::State(
            posComRtWorldEwrtWorldXy_.x() - stancePosFootRtWorldEwrtWorldXy.x(),
            linvelComRtWorldEwrtWorldXy_.x());
        event.preImpactY = HlipModel::State(
            posComRtWorldEwrtWorldXy_.y() - stancePosFootRtWorldEwrtWorldXy.y(),
            linvelComRtWorldEwrtWorldXy_.y());
        event.alpha = activeAlpha_;
        event.cmdLinvelBaseRtWorldEwrtWorldXy =
            cmdLinvelBaseRtWorldEwrtWorldXy_;
      }
      events_.push_back(event);
    }
    contacts_ = now;
  }

  void advanceTo(scalar_t endTime) {
    applyEvents();
    while (time_ < endTime - kEventTolerance) {
      scalar_t next = endTime;
      for (const scalar_t eventTime : applied_.eventTimes) {
        if (eventTime > time_ + kEventTolerance) {
          next = std::min(next, eventTime);
          break;
        }
      }
      flow(next - time_);
      time_ = next;
      applyEvents();
    }
  }

  /** The H-LIP through `duration` of the current contact state: the pendulum in
   * single support, a drift otherwise. */
  void flow(scalar_t duration) {
    const bool left = contacts_[kContactLeftIndex];
    const bool right = contacts_[kContactRightIndex];
    if (left == right) {
      const std::optional<vector2_t> desLinvelComRtWorldEwrtWorldXy =
          activePlanWalks_
              ? std::nullopt
              : activePlan_.evaluateLinvelComRtWorldEwrtWorldXy(time_);
      if (desLinvelComRtWorldEwrtWorldXy.has_value()) {
        // A first-order approach to the standing reference, integrated exactly
        // over `duration`.
        const scalar_t decay = std::exp(-duration / kStandingTimeConstant);
        const vector2_t errLinvelComRtWorldEwrtWorldXy =
            linvelComRtWorldEwrtWorldXy_ - *desLinvelComRtWorldEwrtWorldXy;
        posComRtWorldEwrtWorldXy_ +=
            duration * *desLinvelComRtWorldEwrtWorldXy +
            kStandingTimeConstant * (1.0 - decay) *
                errLinvelComRtWorldEwrtWorldXy;
        linvelComRtWorldEwrtWorldXy_ = *desLinvelComRtWorldEwrtWorldXy +
                                       decay * errLinvelComRtWorldEwrtWorldXy;
        return;
      }
      posComRtWorldEwrtWorldXy_ += duration * linvelComRtWorldEwrtWorldXy_;
      return;
    }
    const vector2_t stancePosFootRtWorldEwrtWorldXy =
        posFeetRtWorldEwrtWorldXy_[left ? kContactLeftIndex
                                        : kContactRightIndex];
    const HlipModel& model = planner_->getModel();
    const HlipModel::State x = model.flowSingleSupport(
        HlipModel::State(
            posComRtWorldEwrtWorldXy_.x() - stancePosFootRtWorldEwrtWorldXy.x(),
            linvelComRtWorldEwrtWorldXy_.x()),
        duration);
    const HlipModel::State y = model.flowSingleSupport(
        HlipModel::State(
            posComRtWorldEwrtWorldXy_.y() - stancePosFootRtWorldEwrtWorldXy.y(),
            linvelComRtWorldEwrtWorldXy_.y()),
        duration);
    posComRtWorldEwrtWorldXy_ =
        stancePosFootRtWorldEwrtWorldXy + vector2_t(x(0), y(0));
    linvelComRtWorldEwrtWorldXy_ = vector2_t(x(1), y(1));
  }

  // [s] events this close to the current time have happened: the node times of
  // successive plans are sums of dt and land a few ulp either side of each
  // other.
  static constexpr scalar_t kEventTolerance = 1.0e-9;
  // [s] time constant of the harness's standing controller: how quickly the
  // center of mass approaches a standing plan's velocity reference with both
  // feet down. Deliberately not instantaneous (see the class comment).
  static constexpr scalar_t kStandingTimeConstant = 0.15;

  ContactPlanningConfig config_;
  std::unique_ptr<HlipContactPlanner> planner_;  // never null
  scalar_t cyclePeriod_;
  scalar_t cycleStart_ = 0.0;
  int numCycles_ = 0;
  ModeSchedule applied_;
  LiftOffHistory liftOffHistory_;
  bool useNodeGridSchedule_ = false;
  ContactPlan activePlan_;
  bool activePlanWalks_ = true;
  scalar_t activeAlpha_ = 0.0;
  scalar_t time_ = 0.0;
  vector2_t cmdLinvelBaseRtWorldEwrtWorldXy_ = vector2_t::Zero();
  vector2_t posComRtWorldEwrtWorldXy_;
  vector2_t linvelComRtWorldEwrtWorldXy_;
  BipedFeetArray<vector2_t> posFeetRtWorldEwrtWorldXy_;
  BipedFeetArray<bool> contacts_ = makeBipedFeetArray(true);
  std::vector<Event> events_;
};

// [s] about 58 Hz. A period that shares no small multiple with the planner's
// 0.025 s grid or the 0.3 s step, so the touch-downs of one plan fall at a
// different fraction of a node of every later plan.
constexpr scalar_t kMpcCyclePeriod = 0.0173;

// The factory validates: a configuration that does not validate, or whose H-LIP
// cannot be built, is refused as a Status instead of building a planner (or
// ending the process) from it.
TEST(HlipContactPlanner, createRefusesAConfigurationThatDoesNotValidate) {
  EXPECT_TRUE(HlipContactPlanner::Create(makeConfig()).ok());
  for (const scalar_t sspDuration :
       {0.0, -0.3, std::numeric_limits<scalar_t>::quiet_NaN()}) {
    ContactPlanningConfig config = makeConfig();
    config.hlip.sspDuration = sspDuration;
    EXPECT_EQ(HlipContactPlanner::Create(config).status().code(),
              absl::StatusCode::kInvalidArgument)
        << sspDuration;
    EXPECT_FALSE(HlipContactPlanner::startUpLateralStep(config).ok())
        << sspDuration;
    EXPECT_FALSE(HlipContactPlanner::formulationSummary(config).ok())
        << sspDuration;
  }
}

TEST(HlipContactPlanner, standsAtZeroCommand) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(makeConfig()));
  const ContactPlan plan = planner->plan(makeStandingInput(vector2_t::Zero()));

  ASSERT_TRUE(plan.valid);
  for (const BipedFeetArray<bool>& contacts : plan.contacts) {
    EXPECT_TRUE(contacts[kContactLeftIndex]);
    EXPECT_TRUE(contacts[kContactRightIndex]);
  }
  // Standing keeps the feet exactly where they are: there is no
  // stepping-in-place at rest.
  for (const BipedFeetArray<vector2_t>& desPosFeetRtWorldEwrtWorldXy :
       plan.desPosFeetRtWorldEwrtWorldXyTraj) {
    EXPECT_NEAR(desPosFeetRtWorldEwrtWorldXy[kContactLeftIndex].y(),
                0.5 * kNomPosLfootRtRfootEwrtHeadingY, 1.0e-12);
    EXPECT_NEAR(desPosFeetRtWorldEwrtWorldXy[kContactRightIndex].y(),
                -0.5 * kNomPosLfootRtRfootEwrtHeadingY, 1.0e-12);
  }
}

TEST(HlipContactPlanner, standingAsksForACenterOfMassAtRest) {
  // A standing plan must not hand the controller the measured drift as its
  // reference: the reduced model's double support drifts at constant velocity
  // and can never decelerate, so a plan that simply rolled it out would ask the
  // robot to keep sliding. The blend against the static standing reference (the
  // paper's equation 14) is what stops it.
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(makeConfig()));
  ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
  input.posComRtWorldEwrtWorldXy = vector2_t(0.03, 0.01);
  input.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(
      0.02, -0.01);  // a small drift, well below the blend's half point

  const ContactPlan plan = planner->plan(input);
  ASSERT_TRUE(plan.valid);
  ASSERT_FALSE(planner->isWalking(input));

  // The reference is the blend itself: alpha of the rolled-out drift and (1 -
  // alpha) of standing still. The reduced model's double support cannot
  // decelerate, so every bit of the drift that survives is alpha's doing and no
  // more.
  const scalar_t blendWeight = planner->blendWeight(input);
  ASSERT_LT(blendWeight, 0.5);
  const vector2_t expectedLinvelComRtWorldEwrtWorldXy =
      blendWeight * input.proxyLinvelComRtWorldEwrtWorldXy;
  EXPECT_NEAR(plan.linvelComRtWorldEwrtWorldXyTraj.back().x(),
              expectedLinvelComRtWorldEwrtWorldXy.x(), 1.0e-9)
      << "standing must ask for a stop, not a drift";
  EXPECT_NEAR(plan.linvelComRtWorldEwrtWorldXyTraj.back().y(),
              expectedLinvelComRtWorldEwrtWorldXy.y(), 1.0e-9)
      << "standing must ask for a stop, not a drift";

  const vector2_t posFeetmeanRtWorldEwrtWorldXy =
      0.5 * (input.posFeetRtWorldEwrtWorldXy[kContactLeftIndex] +
             input.posFeetRtWorldEwrtWorldXy[kContactRightIndex]);
  EXPECT_LT(
      (plan.posComRtWorldEwrtWorldXyTraj.back() - posFeetmeanRtWorldEwrtWorldXy)
          .norm(),
      0.02)
      << "standing must ask for the center of the support";
}

TEST(HlipContactPlanner, walksAtACommandAndAlternatesFeet) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(makeConfig()));
  const ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  ASSERT_TRUE(planner->isWalking(input));
  const ContactPlan plan = planner->plan(input);
  ASSERT_TRUE(plan.valid);

  const std::vector<size_t> order = swingOrder(plan);
  ASSERT_GE(order.size(), 2U);
  for (size_t index = 1; index < order.size(); ++index) {
    EXPECT_NE(order[index], order[index - 1])
        << "a foot may not swing twice in a row";
  }
  // Exactly one foot is ever off the ground: the nominal gait has no flight
  // phase.
  for (const BipedFeetArray<bool>& contacts : plan.contacts) {
    EXPECT_TRUE(contacts[kContactLeftIndex] || contacts[kContactRightIndex]);
  }
}

TEST(HlipContactPlanner, singleSupportPhasesLastTheConfiguredDuration) {
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  const std::vector<HlipContactPlanner::GaitPhase> gait =
      planner->buildGait(makeStandingInput(vector2_t(0.5, 0.0)),
                         HlipContactPlanner::GaitKind::kWalking);

  ASSERT_GE(gait.size(), 2U);
  for (const HlipContactPlanner::GaitPhase& phase : gait) {
    if (phase.isSingleSupport()) {
      EXPECT_NEAR(phase.duration(), config.hlip.sspDuration, 1.0e-12);
    }
  }
  // The gait is contiguous and covers the horizon, and every phase ends on a
  // node no earlier than the one before.
  for (size_t index = 1; index < gait.size(); ++index) {
    EXPECT_NEAR(gait[index].startTime, gait[index - 1].endTime, 1.0e-12);
    EXPECT_GE(gait[index].endNode, gait[index - 1].endNode);
  }
  EXPECT_GE(gait.back().endTime,
            gait.front().startTime + config.horizon() - 1.0e-12);
  EXPECT_EQ(gait.back().endNode, config.planner.numNodes);
}

TEST(HlipContactPlanner, continuesTheSwingInFlight) {
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  input.contacts[kContactLeftIndex] =
      false;  // the left foot is in flight, a third of the way through its
              // swing
  input.phaseElapsedTime[kContactLeftIndex] = 0.1;
  input.phaseElapsedTime[kContactRightIndex] = 0.1;

  const std::vector<HlipContactPlanner::GaitPhase> gait =
      planner->buildGait(input, HlipContactPlanner::GaitKind::kWalking);
  ASSERT_FALSE(gait.empty());
  EXPECT_EQ(gait.front().swingFoot, static_cast<int>(kContactLeftIndex));
  EXPECT_NEAR(gait.front().duration(), config.hlip.sspDuration - 0.1, 1.0e-12);
  // With a non-zero double support the next single support is not the next
  // phase: the hand-over comes first.
  ASSERT_GE(gait.size(), 2U);
  const std::vector<HlipContactPlanner::GaitPhase>::const_iterator
      nextSingleSupport =
          std::find_if(gait.begin() + 1, gait.end(),
                       [](const HlipContactPlanner::GaitPhase& phase) {
                         return phase.isSingleSupport();
                       });
  ASSERT_NE(nextSingleSupport, gait.end());
  EXPECT_EQ(nextSingleSupport->swingFoot, static_cast<int>(kContactRightIndex))
      << "the feet must alternate";
}

TEST(HlipContactPlanner, stepsAdvanceByTheCommandedVelocityOnTheOrbit) {
  // The steady state of the deadbeat law is the period-one orbit, whose step is
  // v * T. Start the reduced model exactly on that orbit and the planner must
  // ask for exactly that step.
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  const HlipModel& model = planner->getModel();
  const scalar_t cmdLinvelBaseRtWorldEwrtWorldX = 0.5;
  const scalar_t stepDuration = model.stepDuration();
  const scalar_t nomErrPosFootRtWorldEwrtLocal =
      cmdLinvelBaseRtWorldEwrtWorldX * stepDuration;
  const HlipModel::State orbit =
      model.periodOneOrbit(nomErrPosFootRtWorldEwrtLocal);
  const HlipModel::PeriodTwoOrbit lateralOrbit =
      model.periodTwoOrbit(config.hlip.nomPosLfootRtRfootEwrtHeadingY,
                           -config.hlip.nomPosLfootRtRfootEwrtHeadingY);

  // Place the robot at the post-impact state of that orbit, standing on the
  // right foot with the left about to swing.
  ContactPlannerInput input =
      makeStandingInput(vector2_t(cmdLinvelBaseRtWorldEwrtWorldX, 0.0));
  input.contacts[kContactLeftIndex] = false;
  input.phaseElapsedTime = makeBipedFeetArray(0.0);
  // The state a single support actually begins from is the post-impact state
  // carried through the double support, which the reduced model drifts at
  // constant velocity.
  const HlipModel::State postImpactX = HlipModel::flowDoubleSupport(
      HlipModel::applyStepTransition(orbit, nomErrPosFootRtWorldEwrtLocal),
      config.hlip.dspDuration);
  const HlipModel::State postImpactY = HlipModel::flowDoubleSupport(
      HlipModel::applyStepTransition(
          lateralOrbit.beforeSecondStep,
          -config.hlip.nomPosLfootRtRfootEwrtHeadingY),
      config.hlip.dspDuration);
  const vector2_t stancePosFootRtWorldEwrtWorldXy =
      input.posFeetRtWorldEwrtWorldXy[kContactRightIndex];
  input.posComRtWorldEwrtWorldXy = stancePosFootRtWorldEwrtWorldXy +
                                   vector2_t(postImpactX(0), postImpactY(0));
  input.proxyLinvelComRtWorldEwrtWorldXy =
      vector2_t(postImpactX(1), postImpactY(1));
  // The blend must not scale the command down, or the step would be shorter
  // than the orbit's.
  ASSERT_GE(planner->blendWeight(input), 0.999);

  const ContactPlan plan = planner->plan(input);
  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.numClippedSteps, 0);
  const std::optional<vector2_t> desPosLfootRtWorldEwrtWorldXy =
      plan.getDesPosFootRtWorldEwrtWorldXy(kContactLeftIndex,
                                           config.hlip.sspDuration);
  if (!desPosLfootRtWorldEwrtWorldXy.has_value()) GTEST_FAIL();
  EXPECT_NEAR(
      desPosLfootRtWorldEwrtWorldXy->x() - stancePosFootRtWorldEwrtWorldXy.x(),
      nomErrPosFootRtWorldEwrtLocal, 1.0e-6);
  EXPECT_NEAR(
      desPosLfootRtWorldEwrtWorldXy->y() - stancePosFootRtWorldEwrtWorldXy.y(),
      config.hlip.nomPosLfootRtRfootEwrtHeadingY, 1.0e-6);
}

TEST(HlipContactPlanner,
     aLateralCommandShiftsBothStepsOfTheLateralOrbitByTheCommandedDrift) {
  // Laterally the nominal gait is the period-two orbit of u_L =
  // +nomPosLfootRtRfootEwrtHeadingY + v_y T and u_R =
  // -nomPosLfootRtRfootEwrtHeadingY + v_y T, so on that orbit the two steps of
  // a stride sum to exactly 2 v_y T and the robot drifts sideways at the
  // commanded rate, in the direction of the command. Nothing else tested the
  // lateral drift term: every other test commands v_y = 0, or checks only an
  // equivariance that holds whatever its sign.
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  const HlipModel& model = planner->getModel();
  const scalar_t stepDuration = model.stepDuration();
  for (const scalar_t cmdLinvelBaseRtWorldEwrtWorldY : {0.2, -0.2}) {
    const scalar_t nomErrPosLfootRtWorldEwrtLocal =
        config.hlip.nomPosLfootRtRfootEwrtHeadingY +
        cmdLinvelBaseRtWorldEwrtWorldY * stepDuration;
    const scalar_t nomErrPosRfootRtWorldEwrtLocal =
        -config.hlip.nomPosLfootRtRfootEwrtHeadingY +
        cmdLinvelBaseRtWorldEwrtWorldY * stepDuration;
    const HlipModel::PeriodTwoOrbit orbit = model.periodTwoOrbit(
        nomErrPosLfootRtWorldEwrtLocal, nomErrPosRfootRtWorldEwrtLocal);

    // On the orbit at the lift-off of the left foot: the right step's impact
    // carried through the double support.
    ContactPlannerInput input =
        makeStandingInput(vector2_t(0.0, cmdLinvelBaseRtWorldEwrtWorldY));
    input.contacts[kContactLeftIndex] = false;
    input.phaseElapsedTime = makeBipedFeetArray(0.0);
    const HlipModel::State atLiftOff = HlipModel::flowDoubleSupport(
        HlipModel::applyStepTransition(orbit.beforeSecondStep,
                                       nomErrPosRfootRtWorldEwrtLocal),
        config.hlip.dspDuration);
    const vector2_t stancePosFootRtWorldEwrtWorldXy =
        input.posFeetRtWorldEwrtWorldXy[kContactRightIndex];
    input.posComRtWorldEwrtWorldXy =
        stancePosFootRtWorldEwrtWorldXy + vector2_t(0.0, atLiftOff(0));
    input.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(0.0, atLiftOff(1));
    ASSERT_DOUBLE_EQ(planner->blendWeight(input), 1.0)
        << "the command must not be scaled down, or this is not the commanded "
           "orbit";

    const ContactPlan plan = planner->plan(input);
    ASSERT_TRUE(plan.valid);
    EXPECT_EQ(plan.numClippedSteps, 0)
        << "the commanded orbit must fit inside the step-width clip";
    const std::vector<PlannedSwing> swings = plannedSwings(plan);
    ASSERT_GE(swings.size(), 2U);
    // The first swing is the left foot already in flight, so its touch-down is
    // the first left touch-down.
    const std::optional<vector2_t> desPosLfootRtWorldEwrtWorldXy =
        plan.getDesPosFootRtWorldEwrtWorldXy(kContactLeftIndex,
                                             config.hlip.sspDuration);
    if (!desPosLfootRtWorldEwrtWorldXy.has_value()) GTEST_FAIL();
    const scalar_t desPosLfootRtRfootEwrtWorldY =
        desPosLfootRtWorldEwrtWorldXy->y() -
        stancePosFootRtWorldEwrtWorldXy.y();
    EXPECT_NEAR(desPosLfootRtRfootEwrtWorldY, nomErrPosLfootRtWorldEwrtLocal,
                1.0e-9)
        << "v_y = " << cmdLinvelBaseRtWorldEwrtWorldY;

    // The next swing places the right foot from the left one.
    const PlannedSwing& rightSwing =
        swings.front().foot == kContactRightIndex ? swings.front() : swings[1];
    ASSERT_EQ(rightSwing.foot, kContactRightIndex);
    const vector2_t desPosRfootRtWorldEwrtWorldXy =
        plan.desPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(
            rightSwing.touchDownInterval)][kContactRightIndex];
    const scalar_t desPosRfootRtLfootEwrtWorldY =
        desPosRfootRtWorldEwrtWorldXy.y() - desPosLfootRtWorldEwrtWorldXy->y();
    EXPECT_NEAR(desPosRfootRtLfootEwrtWorldY, nomErrPosRfootRtWorldEwrtLocal,
                1.0e-9)
        << "v_y = " << cmdLinvelBaseRtWorldEwrtWorldY;

    EXPECT_NEAR(desPosLfootRtRfootEwrtWorldY + desPosRfootRtLfootEwrtWorldY,
                2.0 * cmdLinvelBaseRtWorldEwrtWorldY * stepDuration, 1.0e-9);
    EXPECT_GT((desPosLfootRtRfootEwrtWorldY + desPosRfootRtLfootEwrtWorldY) *
                  cmdLinvelBaseRtWorldEwrtWorldY,
              0.0)
        << "the robot must drift the way it is commanded to";
  }
}

TEST(HlipContactPlanner, aYawRateTurnsEveryFootholdByTheIntegratedHeading) {
  // With a yaw-rate command the heading of the gait advances by alpha * rate *
  // t, and each foothold is placed in the frame of the heading at its own
  // touch-down. Nothing exercised this with a non-zero rate: a sign flip of the
  // heading integration turned the robot the wrong way and every test still
  // passed.
  for (const bool headingModel : {false, true}) {
    ContactPlanningConfig config = makeConfig();
    config.setHeadingModel(headingModel);
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<HlipContactPlanner> planner,
        HlipContactPlanner::Create(config));
    ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
    input.cmdRateYawBaseRtWorld = 0.6;  // [rad/s] in place, no translation
    const scalar_t alpha = planner->blendWeight(input);
    ASSERT_GE(alpha, 0.5) << "a yaw command alone must start the gait";
    ASSERT_TRUE(planner->isWalking(input));

    const ContactPlan plan = planner->plan(input);
    ASSERT_TRUE(plan.valid);
    const std::vector<PlannedSwing> swings = plannedSwings(plan);
    ASSERT_GE(swings.size(), 3U) << "turning in place is a stepping gait";

    scalar_t prevYawHeadingRtWorld = input.yawHeadingRtWorld;
    for (const PlannedSwing& swing : swings) {
      const size_t node = static_cast<size_t>(swing.touchDownInterval);
      const scalar_t touchDown =
          plan.startTime + plan.dt * static_cast<scalar_t>(node);
      // The phase heading the foot was placed in, within the half node the
      // touch-down is rounded by.
      const scalar_t expectedYawHeadingRtWorld =
          input.yawHeadingRtWorld +
          alpha * input.cmdRateYawBaseRtWorld * (touchDown - input.time);
      // The placed foot minus the other foot.
      const vector2_t errPosFootRtWorldEwrtWorldXy =
          plan.desPosFeetRtWorldEwrtWorldXyTraj[node][swing.foot] -
          plan.desPosFeetRtWorldEwrtWorldXyTraj[node][otherFootOf(swing.foot)];
      const vector2_t errPosFootRtWorldEwrtHeadingXy =
          planarRotmatHeadingRtWorld(expectedYawHeadingRtWorld).transpose() *
          errPosFootRtWorldEwrtWorldXy;
      const scalar_t side = swing.foot == kContactLeftIndex ? 1.0 : -1.0;
      const scalar_t tolerance = alpha * input.cmdRateYawBaseRtWorld * 0.5 *
                                     plan.dt *
                                     errPosFootRtWorldEwrtWorldXy.norm() +
                                 1.0e-9;
      // In its own heading frame the placed foot is on its own side, at least
      // minPosLfootRtRfootEwrtHeadingY out: a foot placed in the frame of the
      // opposite rotation would be swung round by twice the turn and fail this
      // by a wide margin.
      EXPECT_GE(side * errPosFootRtWorldEwrtHeadingXy.y(),
                config.hlip.minPosLfootRtRfootEwrtHeadingY - tolerance)
          << "foot " << swing.foot << " at " << touchDown;
      EXPECT_LE(std::abs(errPosFootRtWorldEwrtHeadingXy.x()),
                config.hlip.maxNormPosLfootRtRfootEwrtHeadingX + tolerance)
          << "foot " << swing.foot << " at " << touchDown;
      EXPECT_GT(expectedYawHeadingRtWorld, prevYawHeadingRtWorld)
          << "a positive rate turns the gait counter-clockwise";
      prevYawHeadingRtWorld = expectedYawHeadingRtWorld;
    }
    ASSERT_GT(prevYawHeadingRtWorld - input.yawHeadingRtWorld, 0.5)
        << "the horizon must turn far enough for the frame to matter";
    // With or without the heading model, the plan reports the heading each
    // node was planned in.
    EXPECT_NEAR(plan.rateYawHeadingRtWorld, alpha * input.cmdRateYawBaseRtWorld,
                1.0e-12);
    for (size_t node = 0; node < plan.desPosFeetRtWorldEwrtWorldXyTraj.size();
         ++node) {
      const scalar_t nodeTime =
          plan.startTime + plan.dt * static_cast<scalar_t>(node);
      EXPECT_NEAR(plan.getYawHeadingRtWorld(nodeTime),
                  input.yawHeadingRtWorld + alpha *
                                                input.cmdRateYawBaseRtWorld *
                                                (nodeTime - input.time),
                  1.0e-12)
          << "node " << node << ", heading model " << headingModel;
    }

    if (headingModel) {
      ASSERT_EQ(plan.headingModelYawTraj.size(),
                plan.desPosFeetRtWorldEwrtWorldXyTraj.size());
      for (size_t node = 0; node < plan.headingModelYawTraj.size(); ++node) {
        const scalar_t nodeTime =
            plan.startTime + plan.dt * static_cast<scalar_t>(node);
        EXPECT_NEAR(plan.headingModelYawTraj[node],
                    input.yawHeadingRtWorld + alpha *
                                                  input.cmdRateYawBaseRtWorld *
                                                  (nodeTime - input.time),
                    1.0e-12);
        EXPECT_NEAR(plan.headingModelYawRateTraj[node],
                    alpha * input.cmdRateYawBaseRtWorld, 1.0e-12);
      }
      // A placed foot is yawed to the heading of its touch-down, so the landing
      // yaws advance with the turn.
      for (const PlannedSwing& swing : swings) {
        const size_t node = static_cast<size_t>(swing.touchDownInterval);
        const scalar_t touchDown =
            plan.startTime + plan.dt * static_cast<scalar_t>(node);
        const scalar_t expectedYawHeadingRtWorld =
            input.yawHeadingRtWorld +
            alpha * input.cmdRateYawBaseRtWorld * (touchDown - input.time);
        EXPECT_NEAR(
            plan.desEulerzyxFeetRtWorldYawTraj[node][swing.foot],
            expectedYawHeadingRtWorld,
            alpha * input.cmdRateYawBaseRtWorld * 0.5 * plan.dt + 1.0e-12);
      }
    } else {
      EXPECT_TRUE(plan.headingModelYawTraj.empty());
    }
  }
}

TEST(HlipContactPlanner, theStrideIsScaledByAlpha) {
  // "Above the half point the commanded velocity used for u* is scaled by
  // alpha, so the stride grows continuously out of standing." On the period-one
  // orbit of alpha * v the step is exactly alpha * v * T, not v * T. Every
  // other stride test runs at alpha = 1, where dropping the scaling changes
  // nothing.
  ContactPlanningConfig config = makeConfig();
  // Normalize the measured velocity away, so that alpha is set by the command
  // alone and the orbit below is not chased by an alpha that depends on it.
  config.hlip.blend.maxProxyLinvelComRtWorldEwrtHeadingX = 1.0e3;
  config.hlip.blend.maxProxyLinvelComRtWorldEwrtHeadingY = 1.0e3;
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  const HlipModel& model = planner->getModel();
  // makeStandingInput() leaves the base yaw at 0, so the heading frame is the
  // world here and the same value is the world-axes command below.
  const scalar_t cmdLinvelBaseRtWorldEwrtHeadingX =
      0.12;  // [m/s] a little above the half point of the 0.7 m/s range

  ContactPlannerInput input =
      makeStandingInput(vector2_t(cmdLinvelBaseRtWorldEwrtHeadingX, 0.0));
  input.contacts[kContactLeftIndex] = false;
  input.phaseElapsedTime = makeBipedFeetArray(0.0);
  const vector2_t stancePosFootRtWorldEwrtWorldXy =
      input.posFeetRtWorldEwrtWorldXy[kContactRightIndex];
  const HlipModel::PeriodTwoOrbit lateralOrbit =
      model.periodTwoOrbit(config.hlip.nomPosLfootRtRfootEwrtHeadingY,
                           -config.hlip.nomPosLfootRtRfootEwrtHeadingY);
  const HlipModel::State atLiftOffY = HlipModel::flowDoubleSupport(
      HlipModel::applyStepTransition(
          lateralOrbit.beforeSecondStep,
          -config.hlip.nomPosLfootRtRfootEwrtHeadingY),
      config.hlip.dspDuration);
  scalar_t alpha = planner->getBlend().weight(
      vector2_t(cmdLinvelBaseRtWorldEwrtHeadingX, 0.0),
      /*cmdRateYawBaseRtWorld=*/0.0, vector2_t::Zero());
  for (int iteration = 0; iteration < 5;
       ++iteration) {  // the measured velocity still enters alpha, if
                       // negligibly
    const scalar_t nomErrPosFootRtWorldEwrtLocal =
        alpha * cmdLinvelBaseRtWorldEwrtHeadingX * model.stepDuration();
    const HlipModel::State atLiftOffX = HlipModel::flowDoubleSupport(
        HlipModel::applyStepTransition(
            model.periodOneOrbit(nomErrPosFootRtWorldEwrtLocal),
            nomErrPosFootRtWorldEwrtLocal),
        config.hlip.dspDuration);
    input.posComRtWorldEwrtWorldXy = stancePosFootRtWorldEwrtWorldXy +
                                     vector2_t(atLiftOffX(0), atLiftOffY(0));
    input.proxyLinvelComRtWorldEwrtWorldXy =
        vector2_t(atLiftOffX(1), atLiftOffY(1));
    alpha = planner->blendWeight(input);
  }
  ASSERT_GT(alpha, 0.6)
      << "this test needs a stepping plan with alpha well inside (0.5, 1)";
  ASSERT_LT(alpha, 0.9);

  const ContactPlan plan = planner->plan(input);
  ASSERT_TRUE(plan.valid);
  const std::optional<vector2_t> desPosLfootRtWorldEwrtWorldXy =
      plan.getDesPosFootRtWorldEwrtWorldXy(kContactLeftIndex,
                                           config.hlip.sspDuration);
  if (!desPosLfootRtWorldEwrtWorldXy.has_value()) GTEST_FAIL();
  const scalar_t desPosLfootRtRfootEwrtWorldX =
      desPosLfootRtWorldEwrtWorldXy->x() - stancePosFootRtWorldEwrtWorldXy.x();
  EXPECT_NEAR(desPosLfootRtRfootEwrtWorldX,
              alpha * cmdLinvelBaseRtWorldEwrtHeadingX * model.stepDuration(),
              1.0e-9);
  ASSERT_GT(cmdLinvelBaseRtWorldEwrtHeadingX * model.stepDuration() -
                desPosLfootRtRfootEwrtWorldX,
            0.005)
      << "the unscaled stride must be distinguishable from the scaled one";
}

TEST(HlipContactPlanner, aYawCommandAloneStartsTheGait) {
  // Turning on the spot is walking: the yaw rate is one of the blend's five
  // terms, and a robot at rest asked only to turn must get a stepping plan, not
  // a standing one.
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(makeConfig()));
  ContactPlannerInput input = makeStandingInput(vector2_t::Zero());
  ASSERT_FALSE(planner->isWalking(input))
      << "positive control: at rest with no command the robot stands";
  input.cmdRateYawBaseRtWorld = 0.3;
  EXPECT_TRUE(planner->isWalking(input));
  const ContactPlan plan = planner->plan(input);
  EXPECT_FALSE(swingOrder(plan).empty())
      << "a yaw command must produce a swing";
}

TEST(HlipContactPlanner, honorsTheCommittedContacts) {
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  ContactPlannerInput input = makeStandingInput(vector2_t(0.5, 0.0));
  input.committedUntil = 0.1;
  input.committedContacts.assign(4, makeBipedFeetArray(true));
  input.committedContacts[2][kContactRightIndex] = false;

  const ContactPlan plan = planner->plan(input);
  ASSERT_TRUE(plan.valid);
  for (size_t interval = 0; interval < input.committedContacts.size();
       ++interval) {
    EXPECT_EQ(plan.contacts[interval], input.committedContacts[interval]);
  }
}

TEST(HlipContactPlanner, planIsWellFormed) {
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  const ContactPlan plan =
      planner->plan(makeStandingInput(vector2_t(0.4, 0.1)));

  ASSERT_TRUE(plan.valid);
  EXPECT_EQ(plan.numIntervals(), config.planner.numNodes);
  EXPECT_EQ(plan.desPosFeetRtWorldEwrtWorldXyTraj.size(),
            static_cast<size_t>(config.planner.numNodes) + 1);
  EXPECT_EQ(plan.posComRtWorldEwrtWorldXyTraj.size(),
            plan.desPosFeetRtWorldEwrtWorldXyTraj.size());
  EXPECT_EQ(plan.posZmpRtWorldEwrtWorldXyTraj.size(), plan.contacts.size());
  EXPECT_NEAR(plan.endTime(), config.horizon(), 1.0e-12);

  const ModeSchedule schedule = plan.toModeSchedule();
  EXPECT_FALSE(schedule.modeSequence.empty());
  EXPECT_TRUE(
      std::is_sorted(schedule.eventTimes.begin(), schedule.eventTimes.end()));
}

/** The inputs the reduced-model tests below plan from: a walk off a standstill,
 * and a swing in flight under a standing plan. */
std::vector<ContactPlannerInput> reducedModelInputs() {
  ContactPlannerInput walking = makeStandingInput(vector2_t(0.4, 0.1));
  walking.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(0.1, -0.05);
  ContactPlannerInput standingInFlight = makeStandingInput(vector2_t::Zero());
  standingInFlight.contacts[kContactLeftIndex] =
      false;  // a swing in flight, its touch-down off the node grid
  standingInFlight.phaseElapsedTime[kContactLeftIndex] = 0.087;
  standingInFlight.phaseElapsedTime[kContactRightIndex] = 0.087;
  standingInFlight.posComRtWorldEwrtWorldXy = vector2_t(0.02, -0.03);
  standingInFlight.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(0.03, 0.02);
  return {walking, standingInFlight};
}

/**
 * Audit finding CP-1. Between nodes the plan's center of mass is the planner's
 * continuous-time roll-out, not an interpolation of its nodes: a plan on a five
 * times finer grid samples the same roll-out at the coarse plan's off-grid
 * times, and lipReferenceState(), evaluatePosComRtWorldEwrtWorldXy() and
 * evaluateLinvelComRtWorldEwrtWorldXy() must all reproduce those samples.
 */
TEST(HlipContactPlanner, theReducedModelBetweenNodesIsTheRollOutItself) {
  constexpr int kRefinement = 5;
  const ContactPlanningConfig coarseConfig = makeConfig();
  ContactPlanningConfig fineConfig = coarseConfig;
  fineConfig.planner.dt = coarseConfig.planner.dt / kRefinement;
  fineConfig.planner.numNodes = coarseConfig.planner.numNodes * kRefinement;
  ASSERT_EQ(fineConfig.validateStatus(), absl::OkStatus());
  for (const ContactPlannerInput& input : reducedModelInputs()) {
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<HlipContactPlanner> coarsePlanner,
        HlipContactPlanner::Create(coarseConfig));
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<HlipContactPlanner> finePlanner,
        HlipContactPlanner::Create(fineConfig));
    const ContactPlan coarse = coarsePlanner->plan(input);
    const ContactPlan fine = finePlanner->plan(input);
    ASSERT_TRUE(coarse.valid && fine.valid);
    ASSERT_EQ(coarse.phasePosZmpRtWorldEwrtWorldXyTraj.size(),
              coarse.phaseStartTimes.size());
    for (size_t node = 0; node < fine.posComRtWorldEwrtWorldXyTraj.size();
         ++node) {
      const scalar_t time =
          fine.startTime + fine.dt * static_cast<scalar_t>(node);
      const std::optional<LipState> state =
          lipReferenceState(coarse, coarse.omega, time);
      const std::optional<vector2_t> posComRtWorldEwrtWorldXy =
          coarse.evaluatePosComRtWorldEwrtWorldXy(time);
      const std::optional<vector2_t> linvelComRtWorldEwrtWorldXy =
          coarse.evaluateLinvelComRtWorldEwrtWorldXy(time);
      if (!(state.has_value() && posComRtWorldEwrtWorldXy.has_value() &&
            linvelComRtWorldEwrtWorldXy.has_value())) {
        FAIL() << "t=" << time;
      }
      EXPECT_LT((state->posComRtWorldEwrtWorldXy -
                 fine.posComRtWorldEwrtWorldXyTraj[node])
                    .norm(),
                1.0e-9)
          << "t=" << time;
      EXPECT_LT((state->linvelComRtWorldEwrtWorldXy -
                 fine.linvelComRtWorldEwrtWorldXyTraj[node])
                    .norm(),
                1.0e-9)
          << "t=" << time;
      EXPECT_LT(
          (*posComRtWorldEwrtWorldXy - state->posComRtWorldEwrtWorldXy).norm(),
          1.0e-12)
          << "t=" << time;
      EXPECT_LT(
          (*linvelComRtWorldEwrtWorldXy - state->linvelComRtWorldEwrtWorldXy)
              .norm(),
          1.0e-12)
          << "t=" << time;
    }
  }
}

/**
 * Audit finding CP-1. The ZMP lipReferenceState() publishes is the one the
 * plan's CoM motion implies: in double support the roll-out moves at constant
 * velocity, so the ZMP is the CoM itself (it used to be the midpoint of the
 * feet), and everywhere the LIP equation c'' = omega^2 (c - zmp) holds against
 * a central difference of the published velocity.
 */
TEST(HlipContactPlanner,
     thePublishedZmpIsTheOneThePlannedCenterOfMassMotionImplies) {
  constexpr int kSamples = 400;
  constexpr scalar_t kDifferenceStep = 1.0e-5;  // [s]
  const ContactPlanningConfig config = makeConfig();
  for (const ContactPlannerInput& input : reducedModelInputs()) {
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<HlipContactPlanner> planner,
        HlipContactPlanner::Create(config));
    const ContactPlan plan = planner->plan(input);
    ASSERT_TRUE(plan.valid);
    const scalar_t omega = plan.omega;
    int doubleSupportSamples = 0;
    int singleSupportSamples = 0;
    for (int sample = 0; sample < kSamples; ++sample) {
      // A deterministic, irrational stride through the horizon, off the node
      // grid.
      const scalar_t fraction =
          std::fmod(0.5 + std::sqrt(2.0) * static_cast<scalar_t>(sample), 1.0);
      const scalar_t time =
          plan.startTime + kDifferenceStep +
          fraction * (plan.endTime() - plan.startTime - 2.0 * kDifferenceStep);
      // Keep the difference inside one phase.
      bool nearEvent = false;
      for (scalar_t start : plan.phaseStartTimes) {
        nearEvent = nearEvent || std::abs(time - start) < 2.0 * kDifferenceStep;
      }
      if (nearEvent) continue;
      const std::optional<LipState> state =
          lipReferenceState(plan, omega, time);
      const std::optional<LipState> before =
          lipReferenceState(plan, omega, time - kDifferenceStep);
      const std::optional<LipState> after =
          lipReferenceState(plan, omega, time + kDifferenceStep);
      ASSERT_TRUE(state.has_value() && before.has_value() && after.has_value());
      const vector2_t linaccComRtWorldEwrtWorldXy =
          (after->linvelComRtWorldEwrtWorldXy -
           before->linvelComRtWorldEwrtWorldXy) /
          (2.0 * kDifferenceStep);
      const vector2_t linvelComRtWorldEwrtWorldXy =
          (after->posComRtWorldEwrtWorldXy - before->posComRtWorldEwrtWorldXy) /
          (2.0 * kDifferenceStep);
      EXPECT_LT(
          (linvelComRtWorldEwrtWorldXy - state->linvelComRtWorldEwrtWorldXy)
              .norm(),
          1.0e-6)
          << "t=" << time;
      EXPECT_LT(
          (linaccComRtWorldEwrtWorldXy - omega * omega *
                                             (state->posComRtWorldEwrtWorldXy -
                                              state->posZmpRtWorldEwrtWorldXy))
              .norm(),
          1.0e-4)
          << "t=" << time;
      const BipedFeetArray<bool> contacts = plan.contactsAtTime(time);
      if (contacts[kContactLeftIndex] && contacts[kContactRightIndex]) {
        EXPECT_LT(
            (state->posZmpRtWorldEwrtWorldXy - state->posComRtWorldEwrtWorldXy)
                .norm(),
            1.0e-12)
            << "a constant-velocity double support has its ZMP at the CoM, t="
            << time;
        ++doubleSupportSamples;
      } else {
        ++singleSupportSamples;
      }
    }
    EXPECT_GT(doubleSupportSamples, 0);
    EXPECT_GT(singleSupportSamples, 0);
  }
}

TEST(HlipContactPlanner, closedLoopTracksTheCommandedVelocity) {
  // Roll the planner and its own reduced model forward: from standing, the
  // deadbeat steps must bring the average velocity of the model to the command
  // within a few steps, which is the property the whole planner rests on.
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  const HlipModel& model = planner->getModel();
  const scalar_t cmdLinvelBaseRtWorldEwrtWorldX = 0.5;

  ContactPlannerInput input =
      makeStandingInput(vector2_t(cmdLinvelBaseRtWorldEwrtWorldX, 0.0));
  input.contacts[kContactLeftIndex] = false;
  input.phaseElapsedTime = makeBipedFeetArray(0.0);

  scalar_t prevStancePosFootRtWorldEwrtWorldX =
      input.posFeetRtWorldEwrtWorldXy[kContactRightIndex].x();
  scalar_t prevErrPosFootRtWorldEwrtWorldX = 0.0;
  for (int step = 0; step < 8; ++step) {
    const size_t swingFoot = input.contacts[kContactLeftIndex]
                                 ? kContactRightIndex
                                 : kContactLeftIndex;
    const size_t stanceFoot =
        swingFoot == kContactLeftIndex ? kContactRightIndex : kContactLeftIndex;
    const ContactPlan plan = planner->plan(input);
    ASSERT_TRUE(plan.valid);
    const std::optional<vector2_t> desPosFootRtWorldEwrtWorldXy =
        plan.getDesPosFootRtWorldEwrtWorldXy(
            swingFoot, input.time + config.hlip.sspDuration);
    if (!desPosFootRtWorldEwrtWorldXy.has_value()) GTEST_FAIL();

    // Advance the reduced model to the touch-down of that step.
    HlipModel::State stateX(input.posComRtWorldEwrtWorldXy.x() -
                                input.posFeetRtWorldEwrtWorldXy[stanceFoot].x(),
                            input.proxyLinvelComRtWorldEwrtWorldXy.x());
    HlipModel::State stateY(input.posComRtWorldEwrtWorldXy.y() -
                                input.posFeetRtWorldEwrtWorldXy[stanceFoot].y(),
                            input.proxyLinvelComRtWorldEwrtWorldXy.y());
    stateX = model.flowSingleSupport(stateX, config.hlip.sspDuration);
    stateY = model.flowSingleSupport(stateY, config.hlip.sspDuration);

    prevErrPosFootRtWorldEwrtWorldX =
        desPosFootRtWorldEwrtWorldXy->x() - prevStancePosFootRtWorldEwrtWorldX;
    prevStancePosFootRtWorldEwrtWorldX = desPosFootRtWorldEwrtWorldXy->x();

    input.time += config.hlip.sspDuration;
    input.posFeetRtWorldEwrtWorldXy[swingFoot] = *desPosFootRtWorldEwrtWorldXy;
    input.posComRtWorldEwrtWorldXy =
        input.posFeetRtWorldEwrtWorldXy[stanceFoot] +
        vector2_t(stateX(0), stateY(0));
    input.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(stateX(1), stateY(1));
    // The double support that follows: the reduced model drifts through it at
    // constant velocity. Leaving it out makes the rolled-out step shorter than
    // the one the planner's own step duration is built on.
    input.posComRtWorldEwrtWorldXy +=
        config.hlip.dspDuration * input.proxyLinvelComRtWorldEwrtWorldXy;
    input.time += config.hlip.dspDuration;
    input.contacts = makeBipedFeetArray(true);
    input.contacts[stanceFoot] =
        false;  // the roles swap: the old stance foot swings next
    input.phaseElapsedTime = makeBipedFeetArray(0.0);
    input.lastSwungFoot = static_cast<int>(swingFoot);
    input.committedUntil = input.time;
  }

  EXPECT_NEAR(prevErrPosFootRtWorldEwrtWorldX / model.stepDuration(),
              cmdLinvelBaseRtWorldEwrtWorldX, 1.0e-3);
}

/**
 * The planner's own lateral start-up out of a standstill, rolled forward step
 * by step as the closed loop does: the lateral distance between the feet after
 * each step, and the clipped steps each plan reported.
 */
struct StartUpRollOut {
  std::vector<scalar_t> widths;
  std::vector<int> numClippedSteps;
};

StartUpRollOut rollOutTheStartUp(const ContactPlanningConfig& config,
                                 int numSteps) {
  const std::unique_ptr<HlipContactPlanner> planner = createPlanner(config);
  const HlipModel& model = planner->getModel();
  ContactPlannerInput input = makeStandingInput(vector2_t(0.3, 0.0));
  input.posFeetRtWorldEwrtWorldXy[kContactLeftIndex] =
      vector2_t(0.0, 0.5 * config.hlip.nomPosLfootRtRfootEwrtHeadingY);
  input.posFeetRtWorldEwrtWorldXy[kContactRightIndex] =
      vector2_t(0.0, -0.5 * config.hlip.nomPosLfootRtRfootEwrtHeadingY);
  input.contacts[kContactLeftIndex] =
      false;  // standing on the right, the left about to swing
  input.phaseElapsedTime = makeBipedFeetArray(0.0);

  StartUpRollOut result;
  for (int step = 0; step < numSteps; ++step) {
    const size_t swingFoot = input.contacts[kContactLeftIndex]
                                 ? kContactRightIndex
                                 : kContactLeftIndex;
    const size_t stanceFoot = otherFootOf(swingFoot);
    const ContactPlan plan = planner->plan(input);
    EXPECT_TRUE(plan.valid);
    result.numClippedSteps.push_back(plan.numClippedSteps);
    const std::optional<vector2_t> desPosFootRtWorldEwrtWorldXy =
        plan.getDesPosFootRtWorldEwrtWorldXy(
            swingFoot, input.time + config.hlip.sspDuration);
    EXPECT_TRUE(desPosFootRtWorldEwrtWorldXy.has_value());
    if (!desPosFootRtWorldEwrtWorldXy.has_value()) break;

    HlipModel::State stateX(input.posComRtWorldEwrtWorldXy.x() -
                                input.posFeetRtWorldEwrtWorldXy[stanceFoot].x(),
                            input.proxyLinvelComRtWorldEwrtWorldXy.x());
    HlipModel::State stateY(input.posComRtWorldEwrtWorldXy.y() -
                                input.posFeetRtWorldEwrtWorldXy[stanceFoot].y(),
                            input.proxyLinvelComRtWorldEwrtWorldXy.y());
    stateX = model.flowSingleSupport(stateX, config.hlip.sspDuration);
    stateY = model.flowSingleSupport(stateY, config.hlip.sspDuration);

    input.time += config.hlip.sspDuration;
    input.posFeetRtWorldEwrtWorldXy[swingFoot] = *desPosFootRtWorldEwrtWorldXy;
    input.posComRtWorldEwrtWorldXy =
        input.posFeetRtWorldEwrtWorldXy[stanceFoot] +
        vector2_t(stateX(0), stateY(0));
    input.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(stateX(1), stateY(1));
    // The double support: the reduced model drifts at constant velocity through
    // it.
    input.posComRtWorldEwrtWorldXy +=
        config.hlip.dspDuration * input.proxyLinvelComRtWorldEwrtWorldXy;
    input.time += config.hlip.dspDuration;
    input.contacts = makeBipedFeetArray(true);
    input.contacts[stanceFoot] = false;
    input.phaseElapsedTime = makeBipedFeetArray(0.0);
    input.lastSwungFoot = static_cast<int>(swingFoot);
    input.committedUntil = input.time;

    result.widths.push_back(
        input.posFeetRtWorldEwrtWorldXy[kContactLeftIndex].y() -
        input.posFeetRtWorldEwrtWorldXy[kContactRightIndex].y());
  }
  return result;
}

TEST(HlipContactPlanner,
     theLateralGaitConvergesFromAStandstillWithoutClipping) {
  // The regression this guards. Starting to walk from a standstill demands a
  // first lateral step far wider than the nominal one: the center of mass is
  // half a step width from the stance foot with no lateral velocity, so it
  // falls sideways fast once the foot lifts. If that step does not fit inside
  // maxPosLfootRtRfootEwrtHeadingY it is clipped, and a clipped step is not the
  // deadbeat step - at some cadences the widths then lock into an alternating
  // wide / narrow limit cycle: the feet come together and the robot walks
  // itself sideways.
  const ContactPlanningConfig config = makeConfig();

  HUMANOID_ASSERT_OK_AND_ASSIGN(const scalar_t startUpStep,
                                HlipContactPlanner::startUpLateralStep(config));
  EXPECT_LT(startUpStep, config.hlip.maxPosLfootRtRfootEwrtHeadingY)
      << "the first step out of a standstill must fit inside the reach, or the "
         "gait cannot start; shorten sspDuration";

  // Roll the planner and its own reduced model forward laterally, exactly as
  // the closed loop does. The widths alone cannot see a clipped first step - a
  // gait whose first step is clipped can still settle at the nominal width - so
  // every plan must also report that it clipped nothing.
  const StartUpRollOut rollOut = rollOutTheStartUp(config, /*numSteps=*/6);
  ASSERT_EQ(rollOut.widths.size(), 6U);
  for (size_t step = 0; step < rollOut.numClippedSteps.size(); ++step) {
    EXPECT_EQ(rollOut.numClippedSteps[step], 0)
        << "the plan made before step " << step << " clipped a step";
  }

  // The deadbeat law reaches the nominal width and stays there; the limit cycle
  // would alternate the two clips forever.
  EXPECT_NEAR(rollOut.widths.back(), config.hlip.nomPosLfootRtRfootEwrtHeadingY,
              1.0e-3)
      << "the lateral gait must settle at the nominal step width";
  EXPECT_NEAR(rollOut.widths[rollOut.widths.size() - 2],
              config.hlip.nomPosLfootRtRfootEwrtHeadingY, 1.0e-3)
      << "and stay there, rather than alternating";
}

TEST(HlipContactPlanner, theStartUpWidthsOfTheSummaryAreThePlannersOwn) {
  // formulationSummary decides whether a clipped first step recovers or locks
  // from computeStartUpNormPosLfootRtRfootEwrtHeadingYTraj(), a lateral
  // roll-out of the reduced model under the planner's step law and clip. That
  // verdict is only true of the planner if the roll-out is what the planner
  // itself does, so compare it with the planner closed around its own model, at
  // a cadence that fits, one that clips and recovers, and one that clips and
  // locks.
  struct Case {
    scalar_t sspDuration;
    scalar_t dspDuration;
    scalar_t maxPosLfootRtRfootEwrtHeadingY;
  };
  for (const Case& testCase : {Case{.sspDuration = 0.25,
                                    .dspDuration = 0.05,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.45},
                               Case{.sspDuration = 0.35,
                                    .dspDuration = 0.0,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.45},
                               Case{.sspDuration = 0.35,
                                    .dspDuration = 0.05,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.45}}) {
    ContactPlanningConfig config = makeConfig();
    config.hlip.sspDuration = testCase.sspDuration;
    config.hlip.dspDuration = testCase.dspDuration;
    config.hlip.maxPosLfootRtRfootEwrtHeadingY =
        testCase.maxPosLfootRtRfootEwrtHeadingY;
    const StartUpRollOut planned = rollOutTheStartUp(config, /*numSteps=*/8);
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const std::vector<scalar_t> modeled,
        HlipContactPlanner::computeStartUpNormPosLfootRtRfootEwrtHeadingYTraj(
            config, /*numSteps=*/8));
    ASSERT_EQ(planned.widths.size(), modeled.size());
    for (size_t step = 0; step < modeled.size(); ++step) {
      EXPECT_NEAR(planned.widths[step], modeled[step], 1.0e-9)
          << "ssp " << testCase.sspDuration << " dsp " << testCase.dspDuration
          << " step " << step;
    }
  }
}

TEST(HlipContactPlanner,
     theStartUpBannerSaysWhetherTheFirstStepFitsAndWhatAClipDoes) {
  // "DOES NOT FIT" exactly when the first-step demand exceeds
  // maxPosLfootRtRfootEwrtHeadingY, and "locks" exactly when the planner's own
  // start-up then never settles. The banner used to say that every clipped
  // first step locks the gait, which the library defaults themselves
  // contradict: at 0.35 s with no double support the first step is clipped
  // (0.49 m against 0.45 m) and the gait still settles within three steps, so
  // an operator reading the banner would retune a cadence that works.
  struct Case {
    scalar_t sspDuration;
    scalar_t dspDuration;
    scalar_t maxPosLfootRtRfootEwrtHeadingY;
  };
  int sawFits = 0;
  int sawRecovers = 0;
  int sawLocks = 0;
  for (const Case& testCase : {Case{.sspDuration = 0.25,
                                    .dspDuration = 0.05,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.45},
                               Case{.sspDuration = 0.35,
                                    .dspDuration = 0.0,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.45},
                               Case{.sspDuration = 0.35,
                                    .dspDuration = 0.05,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.45},
                               Case{.sspDuration = 0.5,
                                    .dspDuration = 0.05,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.45},
                               Case{.sspDuration = 0.25,
                                    .dspDuration = 0.05,
                                    .maxPosLfootRtRfootEwrtHeadingY = 0.35}}) {
    ContactPlanningConfig config = makeConfig();
    config.hlip.sspDuration = testCase.sspDuration;
    config.hlip.dspDuration = testCase.dspDuration;
    config.hlip.maxPosLfootRtRfootEwrtHeadingY =
        testCase.maxPosLfootRtRfootEwrtHeadingY;
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const std::string summary,
        HlipContactPlanner::formulationSummary(config));
    HUMANOID_ASSERT_OK_AND_ASSIGN(
        const scalar_t startUpStep,
        HlipContactPlanner::startUpLateralStep(config));
    const bool fits = startUpStep <= config.hlip.maxPosLfootRtRfootEwrtHeadingY;
    const StartUpRollOut planned = rollOutTheStartUp(config, /*numSteps=*/12);
    const bool settles =
        std::abs(planned.widths.back() -
                 config.hlip.nomPosLfootRtRfootEwrtHeadingY) < 1.0e-3 &&
        std::abs(planned.widths[planned.widths.size() - 2] -
                 config.hlip.nomPosLfootRtRfootEwrtHeadingY) < 1.0e-3;
    const std::string label = absl::StrCat(
        "ssp ", testCase.sspDuration, " dsp ", testCase.dspDuration, " max ",
        testCase.maxPosLfootRtRfootEwrtHeadingY);
    EXPECT_EQ(!absl::StrContains(summary, "DOES NOT FIT"), fits)
        << label << "\n"
        << summary;
    EXPECT_EQ(absl::StrContains(summary, "locks"), !settles) << label << "\n"
                                                             << summary;
    EXPECT_EQ(planned.numClippedSteps.front() > 0, !fits)
        << label << ": the first plan must count the clipped first step";
    sawFits += fits ? 1 : 0;
    sawRecovers += !fits && settles ? 1 : 0;
    sawLocks += !settles ? 1 : 0;
  }
  // The cases must cover all three outcomes, or the assertions above prove less
  // than they seem to.
  EXPECT_GT(sawFits, 0);
  EXPECT_GT(sawRecovers, 0);
  EXPECT_GT(sawLocks, 0);
}

TEST(HlipContactPlanner, clipsStepsToTheReachableRegion) {
  ContactPlanningConfig config = makeConfig();
  config.hlip.maxNormPosLfootRtRfootEwrtHeadingX = 0.2;
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));

  ContactPlannerInput input = makeStandingInput(
      vector2_t(1.5, 0.0));  // far beyond what one step can deliver
  input.contacts[kContactLeftIndex] = false;
  input.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(1.5, 0.0);
  input.phaseElapsedTime = makeBipedFeetArray(0.0);

  const ContactPlan plan = planner->plan(input);
  ASSERT_TRUE(plan.valid);
  const std::optional<vector2_t> desPosLfootRtWorldEwrtWorldXy =
      plan.getDesPosFootRtWorldEwrtWorldXy(kContactLeftIndex,
                                           config.hlip.sspDuration);
  if (!desPosLfootRtWorldEwrtWorldXy.has_value()) GTEST_FAIL();
  const scalar_t desPosLfootRtRfootEwrtWorldX =
      desPosLfootRtWorldEwrtWorldXy->x() -
      input.posFeetRtWorldEwrtWorldXy[kContactRightIndex].x();
  EXPECT_LE(desPosLfootRtRfootEwrtWorldX,
            config.hlip.maxNormPosLfootRtRfootEwrtHeadingX + 1.0e-9);
  const scalar_t desPosLfootRtRfootEwrtWorldY =
      desPosLfootRtWorldEwrtWorldXy->y() -
      input.posFeetRtWorldEwrtWorldXy[kContactRightIndex].y();
  EXPECT_GE(desPosLfootRtRfootEwrtWorldY,
            config.hlip.minPosLfootRtRfootEwrtHeadingY - 1.0e-9);
  EXPECT_LE(desPosLfootRtRfootEwrtWorldY,
            config.hlip.maxPosLfootRtRfootEwrtHeadingY + 1.0e-9);

  // And the clip is reported: a clipped step is no longer the deadbeat step,
  // which is what the counter and the log line exist to say.
  EXPECT_GT(plan.numClippedSteps, 0);
  EXPECT_NE(plan.describe().find("CLIPPED-STEPS"), std::string::npos)
      << plan.describe();

  // Positive control: the same input at a command one step can deliver clips
  // nothing.
  ContactPlanningConfig roomy = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> roomyPlanner,
      HlipContactPlanner::Create(roomy));
  ContactPlannerInput reachable = makeStandingInput(vector2_t(0.3, 0.0));
  reachable.contacts[kContactLeftIndex] = false;
  reachable.phaseElapsedTime = makeBipedFeetArray(0.0);
  const HlipModel::PeriodTwoOrbit lateralOrbit =
      roomyPlanner->getModel().periodTwoOrbit(
          roomy.hlip.nomPosLfootRtRfootEwrtHeadingY,
          -roomy.hlip.nomPosLfootRtRfootEwrtHeadingY);
  const HlipModel::State atLiftOffY = HlipModel::flowDoubleSupport(
      HlipModel::applyStepTransition(
          lateralOrbit.beforeSecondStep,
          -roomy.hlip.nomPosLfootRtRfootEwrtHeadingY),
      roomy.hlip.dspDuration);
  reachable.posComRtWorldEwrtWorldXy =
      reachable.posFeetRtWorldEwrtWorldXy[kContactRightIndex] +
      vector2_t(0.0, atLiftOffY(0));
  reachable.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(0.3, atLiftOffY(1));
  const ContactPlan reachablePlan = roomyPlanner->plan(reachable);
  EXPECT_EQ(reachablePlan.numClippedSteps, 0);
  EXPECT_EQ(reachablePlan.describe().find("CLIPPED-STEPS"), std::string::npos)
      << reachablePlan.describe();
}

TEST(HlipContactPlanner, isPlannedInTheHeadingFrame) {
  // Rotating the whole problem must rotate the whole plan: the geometry is
  // planned in the frame of `yawHeadingRtWorld`, and that frame has to come
  // from `yawHeadingRtWorld` even when the heading model is off and
  // `initYawHeadingRtWorld` is therefore never filled.
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  ASSERT_FALSE(config.usesHeadingModel());

  const scalar_t yawHeadingRtWorld = 0.5 * std::numbers::pi;
  const Eigen::Matrix<scalar_t, 2, 2> rotmatHeadingRtWorld =
      planarRotmatHeadingRtWorld(yawHeadingRtWorld);

  ContactPlannerInput straight = makeStandingInput(vector2_t(0.5, 0.1));
  straight.contacts[kContactLeftIndex] = false;
  straight.phaseElapsedTime = makeBipedFeetArray(0.0);
  straight.proxyLinvelComRtWorldEwrtWorldXy = vector2_t(0.3, -0.05);

  ContactPlannerInput turned = straight;
  turned.yawHeadingRtWorld = yawHeadingRtWorld;
  turned.cmdLinvelBaseRtWorldEwrtWorldXy =
      rotmatHeadingRtWorld * straight.cmdLinvelBaseRtWorldEwrtWorldXy;
  turned.proxyLinvelComRtWorldEwrtWorldXy =
      rotmatHeadingRtWorld * straight.proxyLinvelComRtWorldEwrtWorldXy;
  turned.posComRtWorldEwrtWorldXy =
      rotmatHeadingRtWorld * straight.posComRtWorldEwrtWorldXy;
  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    turned.posFeetRtWorldEwrtWorldXy[foot] =
        rotmatHeadingRtWorld * straight.posFeetRtWorldEwrtWorldXy[foot];
  }

  const ContactPlan straightPlan = planner->plan(straight);
  const ContactPlan turnedPlan = planner->plan(turned);
  ASSERT_TRUE(straightPlan.valid && turnedPlan.valid);
  ASSERT_EQ(straightPlan.desPosFeetRtWorldEwrtWorldXyTraj.size(),
            turnedPlan.desPosFeetRtWorldEwrtWorldXyTraj.size());
  for (size_t node = 0;
       node < straightPlan.desPosFeetRtWorldEwrtWorldXyTraj.size(); ++node) {
    EXPECT_EQ(
        straightPlan.contacts[std::min(node, straightPlan.contacts.size() - 1)],
        turnedPlan.contacts[std::min(node, turnedPlan.contacts.size() - 1)]);
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      const vector2_t expected =
          rotmatHeadingRtWorld *
          straightPlan.desPosFeetRtWorldEwrtWorldXyTraj[node][foot];
      EXPECT_NEAR(turnedPlan.desPosFeetRtWorldEwrtWorldXyTraj[node][foot].x(),
                  expected.x(), 1.0e-9)
          << "node " << node << " foot " << foot;
      EXPECT_NEAR(turnedPlan.desPosFeetRtWorldEwrtWorldXyTraj[node][foot].y(),
                  expected.y(), 1.0e-9)
          << "node " << node << " foot " << foot;
    }
  }
}

/** A walk that turns at kTurnCmdRateYawBaseRtWorld without the heading model,
 * planned from a snapshot at yaw kTurnYawHeadingRtWorld, and the first swing
 * of that plan. */
struct TurningWalk {
  ContactPlannerInput input;
  ContactPlan plan;
  scalar_t alpha = 0.0;
  size_t foot = 0;
  SwingWindow swing;
};

constexpr scalar_t kTurnYawHeadingRtWorld = 0.3;      // [rad]
constexpr scalar_t kTurnCmdRateYawBaseRtWorld = 1.5;  // [rad/s]

TurningWalk planTurningWalk(const ContactPlanningConfig& config) {
  const std::unique_ptr<HlipContactPlanner> planner = createPlanner(config);
  TurningWalk walk;
  walk.input = makeStandingInput(
      planarRotmatHeadingRtWorld(kTurnYawHeadingRtWorld) * vector2_t(0.5, 0.0));
  walk.input.yawHeadingRtWorld = kTurnYawHeadingRtWorld;
  walk.input.cmdRateYawBaseRtWorld = kTurnCmdRateYawBaseRtWorld;
  walk.alpha = planner->blendWeight(walk.input);
  walk.plan = planner->plan(walk.input);
  const ModeSchedule schedule = walk.plan.toModeSchedule();
  bool found = false;
  for (const scalar_t eventTime : schedule.eventTimes) {
    for (size_t foot = 0; foot < kNumFeet && !found; ++foot) {
      const std::optional<SwingWindow> swing =
          swingPhaseAtTime(schedule, foot, eventTime + 1.0e-3);
      if (!swing.has_value()) continue;
      walk.foot = foot;
      walk.swing = *swing;
      found = true;
    }
    if (found) break;
  }
  EXPECT_TRUE(found) << "the turning walk must swing a foot";
  return walk;
}

/** The heading the H-LIP planned a quantity at `time` in: the snapshot yaw
 * turned by the blended command. */
scalar_t plannedYawHeadingRtWorld(const TurningWalk& walk, scalar_t time) {
  return walk.input.yawHeadingRtWorld + walk.alpha *
                                            walk.input.cmdRateYawBaseRtWorld *
                                            (time - walk.input.time);
}

/** A context of `walk` at `time`: the NMPC predicted the plan's own reduced
 * model, and the robot measures it pushed by the given offsets. */
ExecutionContext makePushedContext(
    const ContactPlanningConfig& config,
    const TurningWalk& walk,
    scalar_t time,
    const vector2_t& errPosComRtWorldEwrtWorldXy,
    const vector2_t& errLinvelComRtWorldEwrtWorldXy) {
  ExecutionContext ctx(config);
  ctx.time = time;
  ctx.activePlan = &walk.plan;
  ctx.hasPredictedComState = true;
  ctx.totalMass = 1.0;
  const std::optional<LipState> predicted =
      lipReferenceState(walk.plan, config.omega(), time);
  EXPECT_TRUE(predicted.has_value());
  if (!predicted.has_value()) return ctx;
  ctx.predictedPosComRtWorldEwrtWorldXy = predicted->posComRtWorldEwrtWorldXy;
  ctx.predictedProxyLinvelComRtWorldEwrtWorldXy =
      predicted->linvelComRtWorldEwrtWorldXy;
  ctx.posComRtWorldEwrtWorldXy =
      predicted->posComRtWorldEwrtWorldXy + errPosComRtWorldEwrtWorldXy;
  ctx.proxyLinvelComRtWorldEwrtWorldXy =
      predicted->linvelComRtWorldEwrtWorldXy + errLinvelComRtWorldEwrtWorldXy;
  return ctx;
}

TEST(HlipContactPlanner,
     dcmStepAdjustmentClipsInTheTurnedHeadingTheFootholdWasPlacedIn) {
  // Without the heading model the H-LIP places every foothold in the snapshot
  // heading turned by alpha * rate up to its touch-down. The rule clipped the
  // adjusted foothold in the snapshot heading instead, a box rotated by the
  // whole turn, and so cut a pushed step along the wrong axes.
  ContactPlanningConfig config = makeConfig();
  ASSERT_FALSE(config.usesHeadingModel());
  config.reachability.maxNormPosFootRtComEwrtHeadingX = 0.05;
  config.reachability.maxPosFootRtComEwrtHeadingY = 0.15;
  config.dcmStepAdjustment.gain = 1.0;
  config.dcmStepAdjustment.maxNormErrPosFootRtWorldEwrtWorldXy = 0.3;
  const TurningWalk walk = planTurningWalk(config);
  ASSERT_TRUE(walk.plan.valid);
  ASSERT_GE(walk.alpha, 0.5);
  const scalar_t time = walk.swing.liftOff + 0.05;
  const ExecutionContext ctx = makePushedContext(
      config, walk, time, vector2_t(0.04, 0.03), vector2_t(0.1, -0.05));

  DcmStepAdjustmentRule rule;
  rule.configure(config);
  const BipedFeetArray<vector2_t> errPosFeetRtWorldEwrtWorldXy =
      rule.correctFootholds(ctx, walk.plan.toModeSchedule());

  const scalar_t omega = config.omega();
  const vector2_t plannedDesPosFootRtWorldEwrtWorldXy =
      valueOrFailure(walk.plan.getDesPosFootRtWorldEwrtWorldXy(
          walk.foot, walk.swing.touchDown));
  const vector2_t errPosDcmRtWorldEwrtWorldXy =
      computePosDcmRtWorldEwrtWorldXy(ctx.posComRtWorldEwrtWorldXy,
                                      ctx.proxyLinvelComRtWorldEwrtWorldXy,
                                      omega) -
      computePosDcmRtWorldEwrtWorldXy(
          ctx.predictedPosComRtWorldEwrtWorldXy,
          ctx.predictedProxyLinvelComRtWorldEwrtWorldXy, omega);
  const vector2_t adjustedDesPosFootRtWorldEwrtWorldXy =
      plannedDesPosFootRtWorldEwrtWorldXy +
      computeErrPosFootRtWorldEwrtWorldXy(
          errPosDcmRtWorldEwrtWorldXy, omega, walk.swing.touchDown - time,
          config.dcmStepAdjustment.gain,
          config.dcmStepAdjustment.maxNormErrPosFootRtWorldEwrtWorldXy);
  const vector2_t touchdownPosComRtWorldEwrtWorldXy =
      valueOrFailure(lipReferenceState(walk.plan, omega, walk.swing.touchDown))
          .posComRtWorldEwrtWorldXy;
  const scalar_t touchdownYawHeadingRtWorld =
      plannedYawHeadingRtWorld(walk, walk.swing.touchDown);
  ASSERT_GT(touchdownYawHeadingRtWorld - kTurnYawHeadingRtWorld, 0.3)
      << "the plan must turn far enough before the touch-down to matter";
  const vector2_t expected =
      clipFootholdToReach(adjustedDesPosFootRtWorldEwrtWorldXy, walk.foot,
                          touchdownPosComRtWorldEwrtWorldXy,
                          touchdownYawHeadingRtWorld, config) -
      plannedDesPosFootRtWorldEwrtWorldXy;
  const vector2_t inSnapshotHeading =
      clipFootholdToReach(adjustedDesPosFootRtWorldEwrtWorldXy, walk.foot,
                          touchdownPosComRtWorldEwrtWorldXy,
                          kTurnYawHeadingRtWorld, config) -
      plannedDesPosFootRtWorldEwrtWorldXy;
  ASSERT_GT((expected - inSnapshotHeading).norm(), 1.0e-2)
      << "the clip must bind differently in the two headings";
  EXPECT_TRUE(
      errPosFeetRtWorldEwrtWorldXy[walk.foot].isApprox(expected, 1.0e-9))
      << "got " << errPosFeetRtWorldEwrtWorldXy[walk.foot].transpose()
      << ", expected " << expected.transpose() << ", snapshot-heading clip "
      << inSnapshotHeading.transpose();
}

TEST(HlipContactPlanner,
     energyCadenceModulationMeasuresTheEnergyAlongTheTurnedHeading) {
  // The orbital energy is taken along the heading the plan has at the current
  // time, which without the heading model is the snapshot heading turned by
  // alpha * rate over the plan's age, not the snapshot heading itself.
  ContactPlanningConfig config = makeConfig();
  ASSERT_FALSE(config.usesHeadingModel());
  config.energyCadenceModulation.gain = 1.0;
  config.energyCadenceModulation.deadband = 0.0;
  const TurningWalk walk = planTurningWalk(config);
  ASSERT_TRUE(walk.plan.valid);
  const scalar_t time = walk.swing.liftOff + 0.05;
  ExecutionContext ctx = makePushedContext(
      config, walk, time, vector2_t::Zero(), vector2_t(0.0, 0.4));

  EnergyCadenceModulationRule rule;
  rule.configure(config);
  rule.beginCycle(ctx);

  const scalar_t omega = config.omega();
  const LipState reference =
      valueOrFailure(lipReferenceState(walk.plan, omega, time));
  const std::function<scalar_t(scalar_t)> shiftAlong =
      [&](scalar_t yawHeadingRtWorld) {
        const vector2_t dirHeadingEwrtWorldXy(std::cos(yawHeadingRtWorld),
                                              std::sin(yawHeadingRtWorld));
        const scalar_t measured = lipOrbitalEnergy(
            dirHeadingEwrtWorldXy.dot(ctx.posComRtWorldEwrtWorldXy -
                                      reference.posZmpRtWorldEwrtWorldXy),
            dirHeadingEwrtWorldXy.dot(ctx.proxyLinvelComRtWorldEwrtWorldXy),
            omega, ctx.totalMass);
        const scalar_t predicted = lipOrbitalEnergy(
            dirHeadingEwrtWorldXy.dot(ctx.predictedPosComRtWorldEwrtWorldXy -
                                      reference.posZmpRtWorldEwrtWorldXy),
            dirHeadingEwrtWorldXy.dot(
                ctx.predictedProxyLinvelComRtWorldEwrtWorldXy),
            omega, ctx.totalMass);
        return -config.energyCadenceModulation.gain * (measured - predicted);
      };
  const scalar_t expected = shiftAlong(plannedYawHeadingRtWorld(walk, time));
  const scalar_t inSnapshotHeading = shiftAlong(kTurnYawHeadingRtWorld);
  ASSERT_GT(std::abs(expected - inSnapshotHeading), 1.0e-3)
      << "the two headings must disagree for the test to tell them apart";
  for (size_t foot = 0; foot < kNumFeet; ++foot) {
    EXPECT_NEAR(ctx.cadenceTouchDownShift[foot], expected, 1.0e-9)
        << "foot " << foot << ", snapshot-heading shift " << inSnapshotHeading;
  }
}

TEST(HlipContactPlanner,
     summaryNamesThePlannerItsCadenceAndTheExecutionRulesThatStillRun) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.execution = {"planned_com_override"};
  HUMANOID_ASSERT_OK_AND_ASSIGN(const std::string summary,
                                HlipContactPlanner::formulationSummary(config));
  EXPECT_NE(summary.find("H-LIP"), std::string::npos);
  EXPECT_NE(summary.find("deadbeat"), std::string::npos);
  EXPECT_NE(summary.find("no optimization"), std::string::npos);
  // The execution list is not gated by planner.type, and planned_com_override
  // is required under hlip: the start-up print used to end "no execution
  // rules", which is how an operator comes to empty the list.
  EXPECT_EQ(summary.find("no execution rules"), std::string::npos) << summary;
  EXPECT_NE(summary.find("execution    : planned_com_override"),
            std::string::npos)
      << summary;

  config.formulation.execution.clear();
  HUMANOID_ASSERT_OK_AND_ASSIGN(const std::string withoutExecution,
                                HlipContactPlanner::formulationSummary(config));
  EXPECT_NE(withoutExecution.find("execution    : none"), std::string::npos);
}

// ---------------------------------------------------------------------------------------------------------------
// The planner closed around its reduced model at the MPC rate
// (ReducedModelLoop).
// ---------------------------------------------------------------------------------------------------------------

TEST(HlipContactPlanner, aWalkThatIsReleasedComesBackToStanding) {
  // After any walk the robot must be able to stop. At a zero command the
  // deadbeat law regulates the lateral motion onto the period-two orbit at
  // +-nomPosLfootRtRfootEwrtHeadingY, which crosses every touch-down at 0.165
  // m/s at this cadence - three times the blend's half point for the lateral
  // center-of-mass velocity. The blend used to measure that sway raw, so alpha
  // was one through every double support, the next lift-off was planned and
  // committed at every touch-down, and a released stick left the robot marching
  // in place for ever. The start-from-rest tests could not see it. Both phases
  // the blend measures the sway in are exercised: the last cycles of a swing,
  // whose plan decides whether the next lift-off is committed at the
  // touch-down, and the standing double support after it, in which the harness
  // only gradually slows the center of mass from the orbit's sway
  // (ReducedModelLoop), so a blend that read that sway as motion would step
  // again. Removing either phase from
  // HlipContactPlanner::computeZeroCommandOrbitLinvelComRtWorldEwrtHeadingY
  // fails this test.
  const ContactPlanningConfig config = makeConfig();
  ReducedModelLoop loop(config, kMpcCyclePeriod);
  loop.setCmdLinvelBaseRtWorldEwrtWorldXy(vector2_t(0.3, 0.0));
  while (loop.time() < 3.0) loop.cycle();
  int touchDowns = 0;
  for (const ReducedModelLoop::Event& event : loop.events()) {
    touchDowns += event.touchDown ? 1 : 0;
  }
  ASSERT_GE(touchDowns, 6) << "positive control: the robot must have been "
                              "walking before the release";

  const scalar_t releasedAt = loop.time();
  loop.setCmdLinvelBaseRtWorldEwrtWorldXy(vector2_t::Zero());
  std::optional<scalar_t> standingSince;
  while (loop.time() < releasedAt + 3.0) {
    const bool walking = loop.cycle();
    if (walking) {
      standingSince.reset();
    } else if (!standingSince.has_value()) {
      standingSince = loop.time();
    }
  }
  if (!standingSince.has_value()) {
    GTEST_FAIL()
        << "the planner was still stepping 3 s after the stick was released";
  }
  EXPECT_LT(*standingSince - releasedAt, 1.5)
      << "the robot must stand within a few steps of the release";

  // Standing means no further lift-off: only the swings already committed when
  // the plans turned to standing run out.
  for (const ReducedModelLoop::Event& event : loop.events()) {
    if (!event.touchDown) {
      EXPECT_LE(event.time, *standingSince + config.planner.commitTime + 1.0e-9)
          << "a lift-off at " << event.time << " after standing began";
    }
  }
}

TEST(HlipContactPlanner,
     everyExecutedStepIsTheDeadbeatStepForItsExecutedTouchDown) {
  // At runtime every touch-down comes from an earlier plan's grid, so it is off
  // the node grid of the plan that has to place it. The deadbeat step of the
  // swing in flight used to be evaluated at the start of the node the executed
  // touch-down falls in - up to a node early, always early - and the double
  // support after it was counted from that node start, so the next lift-off
  // came up to a node early too. The property that must hold instead: the step
  // the robot lands with is the deadbeat step, u = u* + K (x - x*), of the
  // state it is actually in at the impact it executes. Several MPC periods, so
  // that the touch-downs sample many fractions of a node.
  const ContactPlanningConfig config = makeConfig();
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const HlipModel model,
      HlipModel::Create(config.hlip.sspDuration, config.hlip.dspDuration,
                        config.getNomPosComRtFeetmeanEwrtWorldZ(),
                        config.shared.gravity));
  const scalar_t stepDuration = model.stepDuration();
  for (const scalar_t cyclePeriod : {0.01, 0.0173, 0.02, 0.021, 0.025, 0.033}) {
    ReducedModelLoop loop(config, cyclePeriod);
    loop.setCmdLinvelBaseRtWorldEwrtWorldXy(vector2_t(0.3, 0.05));
    while (loop.time() < 4.0) ASSERT_TRUE(loop.cycle());

    int checkedSteps = 0;
    int doubleSupports = 0;
    int swings = 0;
    std::optional<scalar_t> lastTouchDown;
    BipedFeetArray<std::optional<scalar_t>> liftOffs =
        makeBipedFeetArray(std::optional<scalar_t>());
    for (const ReducedModelLoop::Event& event : loop.events()) {
      if (!event.touchDown) {
        liftOffs[event.foot] = event.time;
        if (lastTouchDown.has_value()) {
          // The executed double support is hlip.dspDuration exactly - the
          // duration the deadbeat gain K = [1, T_dsp + coth(omega T_ssp) /
          // omega] assumes - whatever the MPC period. It used to be the node
          // grid of whichever plan last re-decided the lift-off that set it:
          // anywhere in dspDuration +- dt / 2, and one systematic value at a
          // fixed period
          // (theExecutedDoubleSupportWasRoundedToTheGridWithoutContinuousEvents
          // is the control).
          EXPECT_NEAR(event.time - *lastTouchDown, config.hlip.dspDuration,
                      1.0e-9)
              << "period " << cyclePeriod << ", at " << event.time;
          ++doubleSupports;
        }
        continue;
      }
      lastTouchDown = event.time;
      if (const std::optional<scalar_t>& liftOff = liftOffs[event.foot];
          liftOff.has_value()) {
        // And so is every single support, which the gain assumes as well.
        EXPECT_NEAR(event.time - *liftOff, config.hlip.sspDuration, 1.0e-9)
            << "period " << cyclePeriod << ", at " << event.time;
        ++swings;
      }
      // The deadbeat step of the planner's own law, computed here from the
      // executed pre-impact state.
      const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
          event.alpha * event.cmdLinvelBaseRtWorldEwrtWorldXy;
      const scalar_t nomErrPosFootRtWorldEwrtLocal =
          cmdLinvelBaseRtWorldEwrtWorldXy.x() * stepDuration;
      const scalar_t expectedX = model.deadbeatStepLength(
          event.preImpactX, model.periodOneOrbit(nomErrPosFootRtWorldEwrtLocal),
          nomErrPosFootRtWorldEwrtLocal);
      const scalar_t nomErrPosLfootRtWorldEwrtLocal =
          config.hlip.nomPosLfootRtRfootEwrtHeadingY +
          cmdLinvelBaseRtWorldEwrtWorldXy.y() * stepDuration;
      const scalar_t nomErrPosRfootRtWorldEwrtLocal =
          -config.hlip.nomPosLfootRtRfootEwrtHeadingY +
          cmdLinvelBaseRtWorldEwrtWorldXy.y() * stepDuration;
      const HlipModel::PeriodTwoOrbit orbitY = model.periodTwoOrbit(
          nomErrPosLfootRtWorldEwrtLocal, nomErrPosRfootRtWorldEwrtLocal);
      const bool left = event.foot == kContactLeftIndex;
      const scalar_t expectedY = model.deadbeatStepLength(
          event.preImpactY,
          left ? orbitY.beforeFirstStep : orbitY.beforeSecondStep,
          left ? nomErrPosLfootRtWorldEwrtLocal
               : nomErrPosRfootRtWorldEwrtLocal);
      if (std::abs(expectedX) >
              config.hlip.maxNormPosLfootRtRfootEwrtHeadingX ||
          (left ? expectedY : -expectedY) <
              config.hlip.minPosLfootRtRfootEwrtHeadingY ||
          (left ? expectedY : -expectedY) >
              config.hlip.maxPosLfootRtRfootEwrtHeadingY) {
        continue;  // clipped (the start-up): not the deadbeat step by design
      }
      EXPECT_NEAR(event.errPosFootRtWorldEwrtWorldXy.x(), expectedX, 1.0e-9)
          << "period " << cyclePeriod << ", touch-down at " << event.time;
      EXPECT_NEAR(event.errPosFootRtWorldEwrtWorldXy.y(), expectedY, 1.0e-9)
          << "period " << cyclePeriod << ", touch-down at " << event.time;
      ++checkedSteps;
    }
    EXPECT_GE(checkedSteps, 10) << "period " << cyclePeriod;
    EXPECT_GE(doubleSupports, 10) << "period " << cyclePeriod;
    EXPECT_GE(swings, 10) << "period " << cyclePeriod;
  }
}

/**
 * The positive control of the exact timing above: the same loop with the
 * executed schedule built from the plans' node grid alone, as it was before
 * plans carried their gait in continuous time. At an MPC period that shares no
 * multiple with the planner's grid the double supports it executes are not
 * hlip.dspDuration, though each stays within half a node of it.
 */
TEST(HlipContactPlanner,
     theExecutedDoubleSupportWasRoundedToTheGridWithoutContinuousEvents) {
  const ContactPlanningConfig config = makeConfig();
  ReducedModelLoop loop(config, kMpcCyclePeriod);
  loop.useNodeGridSchedule();
  loop.setCmdLinvelBaseRtWorldEwrtWorldXy(vector2_t(0.3, 0.05));
  while (loop.time() < 4.0) ASSERT_TRUE(loop.cycle());
  scalar_t largestError = 0.0;
  int doubleSupports = 0;
  std::optional<scalar_t> lastTouchDown;
  for (const ReducedModelLoop::Event& event : loop.events()) {
    if (event.touchDown) {
      lastTouchDown = event.time;
    } else if (lastTouchDown.has_value()) {
      const scalar_t error =
          std::abs(event.time - *lastTouchDown - config.hlip.dspDuration);
      EXPECT_LE(error, 0.5 * config.planner.dt + 1.0e-9) << "at " << event.time;
      largestError = std::max(largestError, error);
      ++doubleSupports;
    }
  }
  EXPECT_GE(doubleSupports, 10);
  EXPECT_GT(largestError, 1.0e-3)
      << "the node grid alone executed every double support exactly: the "
         "exact-timing test proves nothing";
}

/**
 * The last swung foot outlives the executed schedule's history. The schedule
 * keeps its past only one window deep (the reference manager one horizon, this
 * harness two seconds), so a robot that has stood for longer than that has no
 * lift-off left in it. The planner input used to read the last swung foot off
 * the schedule alone, and it became -1: the blend lost the zero-command orbit
 * it measures the lateral velocity against in double support, and the first
 * step of the next walk was chosen by phase timing instead of alternating with
 * the last one.
 */
TEST(HlipContactPlanner, theLastSwungFootOutlivesTheScheduleHistory) {
  const ContactPlanningConfig config = makeConfig();
  ReducedModelLoop loop(config, kMpcCyclePeriod);
  loop.setCmdLinvelBaseRtWorldEwrtWorldXy(vector2_t(0.3, 0.0));
  while (loop.time() < 3.0) loop.cycle();
  loop.setCmdLinvelBaseRtWorldEwrtWorldXy(vector2_t::Zero());
  while (loop.time() < 9.0) loop.cycle();

  int lastLiftOffFoot = -1;
  scalar_t lastLiftOffTime = 0.0;
  for (const ReducedModelLoop::Event& event : loop.events()) {
    if (event.touchDown) continue;
    lastLiftOffFoot = static_cast<int>(event.foot);
    lastLiftOffTime = event.time;
  }
  ASSERT_GE(lastLiftOffFoot, 0)
      << "positive control: the robot must have walked";
  ASSERT_LT(lastLiftOffTime, loop.time() - 3.0)
      << "positive control: the robot must have stood for longer than the "
         "schedule reaches back";
  // Positive control: the executed schedule alone no longer holds that
  // lift-off.
  LiftOffHistory scheduleOnly;
  scheduleOnly.record(loop.executedSchedule(), loop.time());
  EXPECT_EQ(scheduleOnly.lastSwungFoot(), -1);

  const ContactPlannerInput input = loop.makeInput();
  EXPECT_EQ(input.lastSwungFoot, lastLiftOffFoot);
  // And with it the zero-command orbit the blend reads the lateral velocity
  // against while standing.
  EXPECT_TRUE(loop.planner()
                  .computeZeroCommandOrbitLinvelComRtWorldEwrtHeadingY(input)
                  .has_value());
  // The next walk starts with the other foot.
  EXPECT_EQ(static_cast<int>(HlipContactPlanner::nextSwingFoot(input)),
            1 - lastLiftOffFoot);
}

/** The history forgets a lift-off later than the time it is asked at: time ran
 * backwards, as it does on a reset. */
TEST(HlipContactPlanner, theLiftOffHistoryForgetsTheFutureAfterAReset) {
  BipedFeetArray<bool> leftSwinging = makeBipedFeetArray(true);
  leftSwinging[kContactLeftIndex] = false;
  const ModeSchedule walked(
      {5.0, 5.25}, {kBipedModeStance, bipedModeOfFootContacts(leftSwinging),
                    kBipedModeStance});
  LiftOffHistory history;
  history.record(walked, /*time=*/4.9);
  EXPECT_EQ(history.lastSwungFoot(), -1)
      << "a lift-off after the query time has not happened";
  history.record(walked, /*time=*/6.0);
  EXPECT_EQ(history.lastSwungFoot(), static_cast<int>(kContactLeftIndex));
  history.record(ModeSchedule(/*eventTimesInput=*/{}, {kBipedModeStance}),
                 /*time=*/0.5);
  EXPECT_EQ(history.lastSwungFoot(), -1)
      << "after the reset the lift-off at 5 s lies in the future";
}

/**
 * A plan records the pendulum it was made on (ContactPlan::omega): the terminal
 * DCM cost takes the plan's DCM, and the robot's, with that omega, so that a
 * reload of shared.nomPosComRtFeetmeanEwrtWorldZ cannot re-interpret a plan
 * already made. And a configuration the H-LIP cannot run is refused by its key,
 * keeping the running one: the model CHECK-fails on it, which on the planner's
 * worker thread used to take the controller down.
 */
TEST(HlipContactPlanner,
     aPlanCarriesItsPendulumAndARefusedReloadKeepsTheRunningConfiguration) {
  ContactPlanningConfig config = makeConfig();
  config.shared.nomPosComRtFeetmeanEwrtWorldZ = 1.05;
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<HlipContactPlanner> planner,
      HlipContactPlanner::Create(config));
  const ContactPlan plan = planner->plan(makeStandingInput(vector2_t::Zero()));
  ASSERT_TRUE(plan.valid);
  EXPECT_NEAR(plan.omega, std::sqrt(config.shared.gravity / 1.05), 1.0e-12)
      << "a plan must record the pendulum it was made on";

  // A reload that moves the pendulum: the next plan is made, and recorded, on
  // the new one.
  ContactPlanningConfig lower = config;
  lower.shared.nomPosComRtFeetmeanEwrtWorldZ = 0.8;
  ASSERT_EQ(planner->setConfig(lower), absl::OkStatus());
  const scalar_t lowerOmega = std::sqrt(lower.shared.gravity / 0.8);
  ASSERT_GT(std::abs(lowerOmega - plan.omega), 0.3)
      << "the reload must move the pendulum, or the check below proves nothing";
  EXPECT_NEAR(planner->plan(makeStandingInput(vector2_t::Zero())).omega,
              lowerOmega, 1.0e-12);

  ContactPlanningConfig broken = lower;
  broken.shared.nomPosComRtFeetmeanEwrtWorldZ = -1.0;
  const absl::Status refused = planner->setConfig(broken);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.message(),
                                "shared.nom_pos_com_rt_feetmean_ewrt_world_z"))
      << refused;
  EXPECT_DOUBLE_EQ(planner->getConfig().getNomPosComRtFeetmeanEwrtWorldZ(), 0.8)
      << "a refused reload must keep the running configuration";
  EXPECT_NEAR(planner->getModel().naturalFrequency(), lowerOmega, 1.0e-12);
  EXPECT_NEAR(planner->plan(makeStandingInput(vector2_t::Zero())).omega,
              lowerOmega, 1.0e-12);
}

// The H-LIP gait is a left and a right foot stepping in turn: a model whose
// contact set is not two feet (shared.numContacts, written from the model by
// ContactPlanningModelParameters::applyTo()) is refused by name and count when
// the planner is built, reconfigured or summarized.
TEST(HlipContactPlanner, refusesAContactSetThatIsNotTwoFeet) {
  constexpr size_t kFourContacts = 4;
  ContactPlanningConfig fourContacts = makeConfig();
  fourContacts.shared.numContacts = kFourContacts;
  const absl::Status created =
      HlipContactPlanner::Create(fourContacts).status();
  const absl::Status summarized =
      HlipContactPlanner::formulationSummary(fourContacts).status();
  const std::unique_ptr<HlipContactPlanner> planner =
      createPlanner(makeConfig());
  const absl::Status reconfigured = planner->setConfig(fourContacts);
  const std::array<absl::Status, 3> statuses = {created, summarized,
                                                reconfigured};
  for (const absl::Status& status : statuses) {
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_TRUE(absl::StrContains(
        status.message(),
        "HlipContactPlanner is written for two contacts; the contact set has 4 "
        "contacts"))
        << status;
  }
  EXPECT_EQ(planner->getConfig().shared.numContacts,
            static_cast<size_t>(kBipedFootCount))
      << "a refused configuration keeps the running one";
}

}  // namespace
}  // namespace ocs2::humanoid
