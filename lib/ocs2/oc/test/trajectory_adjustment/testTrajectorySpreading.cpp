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
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/NumericTraits.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/initialization/DefaultInitializer.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_oc/multiple_shooting/Helpers.h"
#include "ocs2_oc/multiple_shooting/Initialization.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_data/TimeDiscretization.h"
#include "ocs2_oc/trajectory_adjustment/TrajectorySpreadingHelperFunctions.h"

/*
 * trajectorySpread() on the solutions the SQP publishes and warm-starts from: multiple_shooting::toPrimalSolution() on
 * the grid of timeDiscretizationWithEvents(), whose pre-event and post-event samples share their event's time. Node i of
 * a plan has the state (t_i, i) and the input (i, -i), so that every sample says where it came from.
 *
 * The plan's grid on [0, 1] with dt 0.1 and events at 0.35 (mode 0 to 1) and 0.72 (mode 1 to 2):
 *
 *   index  0    1    2    3    4      5       6     7     8     9      10      11    12    13
 *   time   0    0.1  0.2  0.3  0.35   0.35    0.45  0.55  0.65  0.72   0.72    0.82  0.92  1.0
 *   event                      pre    post                      pre    post
 */

namespace ocs2 {
namespace {

/** [s] The node spacing of every grid. */
constexpr scalar_t kTimeStep = 0.1;
/** [s] The horizon of every grid. */
constexpr scalar_t kHorizon = 1.0;
/** [s] The next solve's initial time: a fraction of a node after the plan's, 0. */
constexpr scalar_t kNextInitTime = 0.03;
/** [s] The plan's events, off the grid's regular nodes. */
constexpr scalar_t kLiftOffTime = 0.35;
constexpr scalar_t kTouchDownTime = 0.72;
/** The plan's samples of its events (the grid of the file comment). */
constexpr size_t kLiftOffPreEvent = 4;
constexpr size_t kLiftOffPostEvent = 5;
constexpr size_t kTouchDownPreEvent = 9;
constexpr size_t kTouchDownPostEvent = 10;
/** The plan's samples. */
constexpr size_t kPlanSamples = 14;
/** The input dimension of the plans, (i, -i). */
constexpr size_t kInputDim = 2;
/**
 * The weight of the node before a pre-event node in the state the warm start reads at the end of its interval
 * (getIntervalEnd(), weakEpsilon before the event): weakEpsilon over the 0.05 s interval, times the (0.05, 1) between
 * the two nodes' states. The defect this file guards against moves a value by 0.5 or more.
 */
constexpr scalar_t kPreEventReadTolerance = 1e-6;

vector_t stateOf(scalar_t time, size_t node) {
  return (vector_t(2) << time, static_cast<scalar_t>(node)).finished();
}

vector_t inputOf(size_t node) {
  return (vector_t(kInputDim) << static_cast<scalar_t>(node), -static_cast<scalar_t>(node)).finished();
}

ModeSchedule gaitSchedule(scalar_t liftOffTime, scalar_t touchDownTime) {
  return ModeSchedule(/*eventTimesInput=*/{liftOffTime, touchDownTime}, /*modeSequenceInput=*/{0, 1, 2});
}

std::vector<AnnotatedTime> gridOf(scalar_t initTime, const ModeSchedule& modeSchedule) {
  return timeDiscretizationWithEvents(initTime, initTime + kHorizon, kTimeStep, modeSchedule.eventTimes);
}

/**
 * Returns the solution the SQP publishes on `grid` (multiple_shooting::toPrimalSolution()): node i's state (t_i, i)
 * and input (i, -i), a pre-event node without an input (the SQP's), which toPrimalSolution() fills with the one before.
 */
PrimalSolution planOn(const std::vector<AnnotatedTime>& grid, const ModeSchedule& modeSchedule) {
  vector_array_t states;
  vector_array_t inputs;
  for (size_t node = 0; node < grid.size(); ++node) {
    states.push_back(stateOf(grid[node].time, node));
    if (node + 1 < grid.size()) inputs.push_back(grid[node].event == AnnotatedTime::Event::PreEvent ? vector_t() : inputOf(node));
  }
  return multiple_shooting::toPrimalSolution(grid, ModeSchedule(modeSchedule), std::move(states), std::move(inputs));
}

/** Returns the plan of the file comment, after checking its layout. */
PrimalSolution gaitPlan() {
  const ModeSchedule modeSchedule = gaitSchedule(kLiftOffTime, kTouchDownTime);
  PrimalSolution plan = planOn(gridOf(/*initTime=*/0.0, modeSchedule), modeSchedule);
  EXPECT_EQ(plan.timeTrajectory_.size(), kPlanSamples);
  EXPECT_EQ(plan.postEventIndices_, (size_array_t{kLiftOffPostEvent, kTouchDownPostEvent}));
  EXPECT_EQ(plan.timeTrajectory_[kLiftOffPostEvent], plan.timeTrajectory_[kLiftOffPreEvent]) << "toPrimalSolution(): one time per event";
  return plan;
}

/** The time of a post-event node at `eventTime`: the start of its interval, where the SQP reads its input. */
scalar_t postEventTime(scalar_t eventTime) {
  return getIntervalStart(AnnotatedTime(eventTime, AnnotatedTime::Event::PostEvent));
}

/** Expects the samples [begin, end) of `spread` to hold the values of `plan`'s sample `source`. */
void expectSpread(const PrimalSolution& spread, const PrimalSolution& plan, size_t begin, size_t end, size_t source) {
  for (size_t sample = begin; sample < end; ++sample) {
    SCOPED_TRACE(absl::StrCat("sample ", sample, ", spread from ", source));
    EXPECT_EQ(spread.stateTrajectory_[sample], plan.stateTrajectory_[source]);
    EXPECT_EQ(spread.inputTrajectory_[sample], plan.inputTrajectory_[source]);
  }
}

/** Expects the samples [begin, end) of `spread` to hold their own values of `plan`. */
void expectUntouched(const PrimalSolution& spread, const PrimalSolution& plan, size_t begin, size_t end) {
  for (size_t sample = begin; sample < end; ++sample) {
    SCOPED_TRACE(absl::StrCat("sample ", sample));
    EXPECT_EQ(spread.stateTrajectory_[sample], plan.stateTrajectory_[sample]);
    EXPECT_EQ(spread.inputTrajectory_[sample], plan.inputTrajectory_[sample]);
  }
}

/** Expects each post-event index of `spread` to name a sample at postEventTime() of the one before it. */
void expectPostEventSamplesAfterTheirEvents(const PrimalSolution& spread) {
  EXPECT_TRUE(std::is_sorted(spread.timeTrajectory_.begin(), spread.timeTrajectory_.end()));
  for (const size_t postEvent : spread.postEventIndices_) {
    SCOPED_TRACE(absl::StrCat("post-event sample ", postEvent));
    ASSERT_GT(postEvent, size_t{0});
    ASSERT_LT(postEvent, spread.timeTrajectory_.size());
    EXPECT_EQ(spread.timeTrajectory_[postEvent], postEventTime(spread.timeTrajectory_[postEvent - 1]));
  }
}

/** What happened to a LoggedSample. */
enum class SampleEvent { kConstructed, kRead, kAssigned, kDestroyed };

/** One entry of a SampleLog: what happened to the sample at `address`. */
struct SampleLogEntry {
  SampleEvent event = SampleEvent::kConstructed;
  std::uintptr_t address = 0;
};

/**
 * A trajectory sample that logs its construction, its destruction, each copy made of it and each assignment to it, so
 * that a test sees whether TrajectorySpreading::adjustTrajectory() reads or writes a sample after its erase destroyed it.
 * The sanitizer-free stand-in for a use-after-free check: an Eigen sample read after its erase holds the bytes of a freed
 * buffer, and one assigned after it writes into the allocator's free list, neither of which a value comparison sees
 * reliably. Copyable; it has no move operations, so a move copies. Not thread-safe.
 */
class LoggedSample {
 public:
  LoggedSample() = default;
  /** Logs to `log`, which must outlive the sample and every copy of it. */
  LoggedSample(size_t source, std::vector<SampleLogEntry>* absl_nonnull log) : source_(source), log_(log) {
    record(SampleEvent::kConstructed, *this);
  }
  LoggedSample(const LoggedSample& other) : source_(other.source_), log_(other.log_) {
    record(SampleEvent::kRead, other);
    record(SampleEvent::kConstructed, *this);
  }
  LoggedSample& operator=(const LoggedSample& other) {
    source_ = other.source_;
    log_ = other.log_;
    record(SampleEvent::kRead, other);
    record(SampleEvent::kAssigned, *this);
    return *this;
  }
  ~LoggedSample() { record(SampleEvent::kDestroyed, *this); }

  /** The index of the sample of the original trajectory whose value this sample holds. */
  size_t source() const { return source_; }

 private:
  void record(SampleEvent event, const LoggedSample& sample) const {
    if (log_ != nullptr) log_->push_back({.event = event, .address = reinterpret_cast<std::uintptr_t>(&sample)});
  }

  size_t source_ = 0;
  /** Null for a default-constructed sample, which logs nothing until a logged sample is assigned to it. */
  std::vector<SampleLogEntry>* absl_nullable log_ = nullptr;
};

/**
 * Spreads a trajectory of LoggedSample on the stamped times of `plan` as trajectorySpread() spreads the plan, and returns
 * the source of each kept sample. Expects no sample to be read or assigned after it was destroyed. The trajectory's
 * storage holds as many destroyed samples after its end as it has samples, so that an index past the end of the old
 * trajectory reaches a logged sample too.
 */
std::vector<size_t> spreadLoggedSamples(const ModeSchedule& oldModeSchedule, const ModeSchedule& newModeSchedule,
                                        const PrimalSolution& plan) {
  PrimalSolution stamped = plan;
  stampPostEventSamplesAtIntervalStart(stamped);
  TrajectorySpreading spreading;
  spreading.set(oldModeSchedule, newModeSchedule, stamped.timeTrajectory_);

  std::vector<SampleLogEntry> log;
  const size_t samples = plan.timeTrajectory_.size();
  std::vector<LoggedSample> trajectory;
  trajectory.reserve(2 * samples);
  for (size_t sample = 0; sample < 2 * samples; ++sample) trajectory.emplace_back(sample, &log);
  trajectory.erase(trajectory.begin() + samples, trajectory.end());
  const std::uintptr_t storageBegin = reinterpret_cast<std::uintptr_t>(trajectory.data());
  const size_t storageSlots = 2 * samples;

  spreading.adjustTrajectory(trajectory);

  std::vector<bool> destroyed(storageSlots, false);
  for (const SampleLogEntry& entry : log) {
    if (entry.address < storageBegin || (entry.address - storageBegin) % sizeof(LoggedSample) != 0) continue;
    const size_t slot = (entry.address - storageBegin) / sizeof(LoggedSample);
    if (slot >= storageSlots) continue;
    switch (entry.event) {
      case SampleEvent::kConstructed:
        destroyed[slot] = false;
        break;
      case SampleEvent::kDestroyed:
        destroyed[slot] = true;
        break;
      case SampleEvent::kRead:
        EXPECT_FALSE(destroyed[slot]) << "sample " << slot << " read after its destruction";
        break;
      case SampleEvent::kAssigned:
        EXPECT_FALSE(destroyed[slot]) << "sample " << slot << " assigned after its destruction";
        break;
    }
  }

  std::vector<size_t> sources;
  for (const LoggedSample& sample : trajectory) sources.push_back(sample.source());
  return sources;
}

/** Returns the plan of planOn() on the grid of `modeSchedule` from the plan's initial time, 0. */
PrimalSolution planOf(const ModeSchedule& modeSchedule) {
  return planOn(gridOf(/*initTime=*/0.0, modeSchedule), modeSchedule);
}

TEST(TrajectorySpreading, AnUnchangedScheduleKeepsThePlanWithItsPostEventSamplesAfterTheirEvents) {
  const PrimalSolution plan = gaitPlan();
  const ModeSchedule modeSchedule = plan.modeSchedule_;
  PrimalSolution spread = plan;
  trajectorySpread(modeSchedule, modeSchedule, spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), kPlanSamples);
  expectUntouched(spread, plan, /*begin=*/0, /*end=*/kPlanSamples);
  EXPECT_EQ(spread.postEventIndices_, plan.postEventIndices_) << "the post-event samples, not the nodes after them";
  expectPostEventSamplesAfterTheirEvents(spread);
  for (size_t sample = 0; sample < kPlanSamples; ++sample) {
    SCOPED_TRACE(absl::StrCat("sample ", sample));
    const bool postEvent = sample == kLiftOffPostEvent || sample == kTouchDownPostEvent;
    EXPECT_EQ(spread.timeTrajectory_[sample], postEvent ? postEventTime(plan.timeTrajectory_[sample]) : plan.timeTrajectory_[sample]);
  }
}

TEST(TrajectorySpreading, AnUnchangedScheduleWarmStartsTheNextSolveOnThePlanFromTheFirstEventOn) {
  const PrimalSolution plan = gaitPlan();
  const ModeSchedule modeSchedule = plan.modeSchedule_;
  const std::vector<AnnotatedTime> grid = gridOf(/*initTime=*/0.0, modeSchedule);
  PrimalSolution spread = plan;
  trajectorySpread(modeSchedule, modeSchedule, spread);

  // What SqpSolver::runImpl() does next: the next solve's grid, a fraction of a node later, initialized from the spread plan.
  const std::vector<AnnotatedTime> nextGrid = gridOf(kNextInitTime, modeSchedule);
  DefaultInitializer initializer(kInputDim);
  vector_array_t states;
  vector_array_t inputs;
  multiple_shooting::initializeStateInputTrajectories(stateOf(kNextInitTime, /*node=*/0), nextGrid, spread, initializer, states, inputs);

  // From the first event on the two grids are one, node for node (0.03, 0.13, 0.23, 0.33 and 0.35 against 0, 0.1, 0.2,
  // 0.3 and 0.35; then the event plus multiples of dt), up to the plan's final time. The node before it, 0.92, is the
  // plan's too, but its input is the initializer's: its interval ends at 1.02, after the plan's horizon.
  size_t comparedInputs = 0;
  for (size_t node = kLiftOffPreEvent; node + 1 < grid.size(); ++node) {
    SCOPED_TRACE(absl::StrCat("node ", node, " at ", grid[node].time));
    ASSERT_EQ(nextGrid[node].time, grid[node].time);
    ASSERT_EQ(nextGrid[node].event, grid[node].event);
    switch (grid[node].event) {
      case AnnotatedTime::Event::PreEvent:
        // Read weakEpsilon before the event, on the interval that ends there.
        EXPECT_LT((states[node] - plan.stateTrajectory_[node]).cwiseAbs().maxCoeff(), kPreEventReadTolerance);
        EXPECT_EQ(inputs[node].size(), 0) << "an event node has no input";
        break;
      case AnnotatedTime::Event::PostEvent:
        // initializeEventNode() maps the pre-event state through an identity jump.
        EXPECT_EQ(states[node], states[node - 1]);
        EXPECT_EQ(inputs[node], plan.inputTrajectory_[node]) << "the post-event input, read at the start of its interval";
        ++comparedInputs;
        break;
      case AnnotatedTime::Event::None:
        EXPECT_EQ(states[node], plan.stateTrajectory_[node]);
        if (node + 2 < grid.size()) {
          EXPECT_EQ(inputs[node], plan.inputTrajectory_[node]);
          ++comparedInputs;
        }
        break;
    }
  }
  EXPECT_EQ(comparedInputs, size_t{6}) << "0.35 post-event, 0.45, 0.55, 0.65, 0.72 post-event and 0.82";
}

TEST(TrajectorySpreading, AnEventMovedLaterByMoreThanANodeSpreadsThePreEventValuesForward) {
  const PrimalSolution plan = gaitPlan();
  // The lift-off 0.17 s (1.7 nodes) later: in the plan's grid, between 0.45 and 0.55.
  const scalar_t laterLiftOff = 0.52;
  PrimalSolution spread = plan;
  trajectorySpread(plan.modeSchedule_, gaitSchedule(laterLiftOff, kTouchDownTime), spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), kPlanSamples);
  const size_t newLiftOffPostEvent = 7;
  // Mode 0 lasts until the new lift-off: the old post-event sample and the node after it take the pre-event values.
  expectUntouched(spread, plan, /*begin=*/0, /*end=*/kLiftOffPostEvent);
  expectSpread(spread, plan, /*begin=*/kLiftOffPostEvent, /*end=*/newLiftOffPostEvent, /*source=*/kLiftOffPreEvent);
  // Forward spreading moves no sample of mode 1: the new post-event sample keeps the plan's mode-1 value at its node.
  expectUntouched(spread, plan, /*begin=*/newLiftOffPostEvent, /*end=*/kPlanSamples);
  EXPECT_EQ(spread.postEventIndices_, (size_array_t{newLiftOffPostEvent, kTouchDownPostEvent}));
  EXPECT_EQ(spread.timeTrajectory_[newLiftOffPostEvent - 1], laterLiftOff);
  expectPostEventSamplesAfterTheirEvents(spread);
}

TEST(TrajectorySpreading, AnEventMovedEarlierByMoreThanANodeSpreadsThePostEventValuesBackward) {
  const PrimalSolution plan = gaitPlan();
  // The lift-off 0.17 s (1.7 nodes) earlier: in the plan's grid, between 0.1 and 0.2.
  const scalar_t earlierLiftOff = 0.18;
  PrimalSolution spread = plan;
  trajectorySpread(plan.modeSchedule_, gaitSchedule(earlierLiftOff, kTouchDownTime), spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), kPlanSamples);
  const size_t newLiftOffPostEvent = 2;
  // Mode 1 starts at the new lift-off: from its post-event sample to the old one, the old post-event values.
  expectUntouched(spread, plan, /*begin=*/0, /*end=*/newLiftOffPostEvent);
  expectSpread(spread, plan, /*begin=*/newLiftOffPostEvent, /*end=*/kLiftOffPostEvent + 1, /*source=*/kLiftOffPostEvent);
  expectUntouched(spread, plan, /*begin=*/kLiftOffPostEvent + 1, /*end=*/kPlanSamples);
  EXPECT_EQ(spread.postEventIndices_, (size_array_t{newLiftOffPostEvent, kTouchDownPostEvent}));
  EXPECT_EQ(spread.timeTrajectory_[newLiftOffPostEvent - 1], earlierLiftOff);
  expectPostEventSamplesAfterTheirEvents(spread);
}

TEST(TrajectorySpreading, AnEventMovedBeforeThePlanSpreadsItsPostEventValuesToThePlansStart) {
  const PrimalSolution plan = gaitPlan();
  // The lift-off before the plan's initial time: mode 1 is the new schedule's first mode there (the triggering event).
  PrimalSolution spread = plan;
  trajectorySpread(plan.modeSchedule_, gaitSchedule(/*liftOffTime=*/-0.05, kTouchDownTime), spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), kPlanSamples);
  expectSpread(spread, plan, /*begin=*/0, /*end=*/kLiftOffPostEvent + 1, /*source=*/kLiftOffPostEvent);
  expectUntouched(spread, plan, /*begin=*/kLiftOffPostEvent + 1, /*end=*/kPlanSamples);
  EXPECT_EQ(spread.postEventIndices_, (size_array_t{kTouchDownPostEvent})) << "the lift-off left the plan";
  expectPostEventSamplesAfterTheirEvents(spread);
}

TEST(TrajectorySpreading, AnEventMovedPastThePlanSpreadsItsPreEventValuesToThePlansEnd) {
  const PrimalSolution plan = gaitPlan();
  // The touchdown after the plan's final time: mode 1 is the new schedule's last mode there (the end event).
  PrimalSolution spread = plan;
  trajectorySpread(plan.modeSchedule_, gaitSchedule(kLiftOffTime, /*touchDownTime=*/1.2), spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), kPlanSamples);
  expectUntouched(spread, plan, /*begin=*/0, /*end=*/kTouchDownPostEvent);
  expectSpread(spread, plan, /*begin=*/kTouchDownPostEvent, /*end=*/kPlanSamples, /*source=*/kTouchDownPreEvent);
  EXPECT_EQ(spread.postEventIndices_, (size_array_t{kLiftOffPostEvent})) << "the touchdown left the plan";
  expectPostEventSamplesAfterTheirEvents(spread);
}

TEST(TrajectorySpreading, AnotherModeAfterAnEventErasesThePlanFromThatEventOn) {
  const PrimalSolution plan = gaitPlan();
  // Mode 3 instead of mode 2 after the touchdown: no sample of mode 2 is kept, from the touchdown's pre-event sample on.
  PrimalSolution spread = plan;
  trajectorySpread(plan.modeSchedule_, ModeSchedule(/*eventTimesInput=*/{kLiftOffTime, kTouchDownTime}, /*modeSequenceInput=*/{0, 1, 3}),
                   spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), kTouchDownPreEvent);
  ASSERT_EQ(spread.stateTrajectory_.size(), kTouchDownPreEvent);
  ASSERT_EQ(spread.inputTrajectory_.size(), kTouchDownPreEvent);
  expectUntouched(spread, plan, /*begin=*/0, /*end=*/kTouchDownPreEvent);
  EXPECT_EQ(spread.postEventIndices_, (size_array_t{kLiftOffPostEvent}));
  expectPostEventSamplesAfterTheirEvents(spread);
}

TEST(TrajectorySpreading, AnEventMovedEarlierThanANewModeOfThePlanSpreadsThePostEventValuesItErases) {
  const PrimalSolution plan = gaitPlan();
  // The gait 0.1 s earlier with a mode 3 after a 0.08 s mode 2: lift-off at 0.25, touchdown at 0.62 and mode 3 from 0.70,
  // before the plan's touchdown samples at 0.72. The plan is erased from the touchdown's pre-event sample on, and the
  // touchdown's post-event values, which that erases, are the ones mode 2 starts from at 0.62.
  const ModeSchedule earlierGait(/*eventTimesInput=*/{0.25, 0.62, 0.70}, /*modeSequenceInput=*/{0, 1, 2, 3});
  PrimalSolution spread = plan;
  trajectorySpread(plan.modeSchedule_, earlierGait, spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), kTouchDownPreEvent);
  ASSERT_EQ(spread.stateTrajectory_.size(), kTouchDownPreEvent);
  ASSERT_EQ(spread.inputTrajectory_.size(), kTouchDownPreEvent);
  const size_t newLiftOffPostEvent = 3;
  const size_t newTouchDownPostEvent = 8;
  expectUntouched(spread, plan, /*begin=*/0, /*end=*/newLiftOffPostEvent);
  expectSpread(spread, plan, /*begin=*/newLiftOffPostEvent, /*end=*/kLiftOffPostEvent, /*source=*/kLiftOffPostEvent);
  expectUntouched(spread, plan, /*begin=*/kLiftOffPostEvent, /*end=*/newTouchDownPostEvent);
  expectSpread(spread, plan, /*begin=*/newTouchDownPostEvent, /*end=*/kTouchDownPreEvent, /*source=*/kTouchDownPostEvent);
  EXPECT_EQ(spread.postEventIndices_, (size_array_t{newLiftOffPostEvent, newTouchDownPostEvent}));
  expectPostEventSamplesAfterTheirEvents(spread);

  EXPECT_EQ(spreadLoggedSamples(plan.modeSchedule_, earlierGait, plan), (std::vector<size_t>{0, 1, 2, 5, 5, 5, 6, 7, 10}));
}

TEST(TrajectorySpreading, AnEventDelayedOntoANodeBeforeAnotherModeAssignsOnlyTheKeptSamples) {
  const PrimalSolution plan = gaitPlan();
  // The touchdown delayed onto the plan's node at 0.82, with mode 3 after it: mode 1 lasts until 0.82, and the plan is
  // erased from that node on. Mode 1's pre-event values are spread forward over the kept samples only.
  const size_t delayedTouchDownNode = 11;
  const ModeSchedule delayedTouchDown(/*eventTimesInput=*/{kLiftOffTime, plan.timeTrajectory_[delayedTouchDownNode]},
                                      /*modeSequenceInput=*/{0, 1, 3});
  PrimalSolution spread = plan;
  trajectorySpread(plan.modeSchedule_, delayedTouchDown, spread);

  ASSERT_EQ(spread.timeTrajectory_.size(), delayedTouchDownNode);
  ASSERT_EQ(spread.stateTrajectory_.size(), delayedTouchDownNode);
  ASSERT_EQ(spread.inputTrajectory_.size(), delayedTouchDownNode);
  expectUntouched(spread, plan, /*begin=*/0, /*end=*/kTouchDownPostEvent);
  expectSpread(spread, plan, /*begin=*/kTouchDownPostEvent, /*end=*/delayedTouchDownNode, /*source=*/kTouchDownPreEvent);
  EXPECT_EQ(spread.postEventIndices_, (size_array_t{kLiftOffPostEvent})) << "the touchdown's samples were erased";
  expectPostEventSamplesAfterTheirEvents(spread);

  EXPECT_EQ(spreadLoggedSamples(plan.modeSchedule_, delayedTouchDown, plan), (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9}));
}

TEST(TrajectorySpreading, ANewScheduleWithNoModeOfThePlansWindowErasesThePlan) {
  // Modes 3 to 5 lie after the plan's horizon. The new schedule has mode 4 over the whole plan, a mode the plan does not
  // contain, so no sample of the plan is kept; mode 4 of the old schedule, after the plan, is no match either.
  const ModeSchedule oldModeSchedule(/*eventTimesInput=*/{kLiftOffTime, kTouchDownTime, 1.5, 1.8, 2.1},
                                     /*modeSequenceInput=*/{0, 1, 2, 3, 4, 5});
  const PrimalSolution plan = planOf(oldModeSchedule);
  ASSERT_EQ(plan.timeTrajectory_.size(), kPlanSamples);
  const ModeSchedule newModeSchedule(/*eventTimesInput=*/{1.5}, /*modeSequenceInput=*/{4, 5});

  EXPECT_EQ(spreadLoggedSamples(oldModeSchedule, newModeSchedule, plan), std::vector<size_t>());
  PrimalSolution spread = plan;
  trajectorySpread(oldModeSchedule, newModeSchedule, spread);
  EXPECT_TRUE(spread.timeTrajectory_.empty());
  EXPECT_TRUE(spread.stateTrajectory_.empty());
  EXPECT_TRUE(spread.inputTrajectory_.empty());
  EXPECT_TRUE(spread.postEventIndices_.empty());
}

TEST(TrajectorySpreading, ANewScheduleWithoutAnyMatchingModeErasesThePlanWithoutReadingPastItsEvents) {
  // Mode 0 again after the plan's horizon. A new schedule whose modes over the plan match none of the plan's matches
  // nothing (w == 0); set() then must not compare the event times after the last matched mode, of which there are none.
  // Before the fix, the schedule without events read its event times at index -1, a null data pointer's: a crash.
  const ModeSchedule oldModeSchedule(/*eventTimesInput=*/{kLiftOffTime, kTouchDownTime, 1.5}, /*modeSequenceInput=*/{0, 1, 2, 0});
  const PrimalSolution plan = planOf(oldModeSchedule);
  ASSERT_EQ(plan.timeTrajectory_.size(), kPlanSamples);
  const std::vector<ModeSchedule> newModeSchedules = {
      ModeSchedule(/*eventTimesInput=*/{}, /*modeSequenceInput=*/{5}),
      ModeSchedule(/*eventTimesInput=*/{1.5}, /*modeSequenceInput=*/{5, 0}),
  };
  for (const ModeSchedule& newModeSchedule : newModeSchedules) {
    SCOPED_TRACE(absl::StrCat("new schedule with ", newModeSchedule.eventTimes.size(), " events"));
    EXPECT_EQ(spreadLoggedSamples(oldModeSchedule, newModeSchedule, plan), std::vector<size_t>());
    PrimalSolution spread = plan;
    trajectorySpread(oldModeSchedule, newModeSchedule, spread);
    EXPECT_TRUE(spread.timeTrajectory_.empty());
    EXPECT_TRUE(spread.stateTrajectory_.empty());
    EXPECT_TRUE(spread.inputTrajectory_.empty());
    EXPECT_TRUE(spread.postEventIndices_.empty());
  }
}

TEST(StampPostEventSamplesAtIntervalStart, MovesEachPostEventSampleToTheStartOfItsIntervalOnce) {
  const PrimalSolution plan = gaitPlan();
  PrimalSolution stamped = plan;
  stampPostEventSamplesAtIntervalStart(stamped);

  expectUntouched(stamped, plan, /*begin=*/0, /*end=*/kPlanSamples);
  EXPECT_EQ(stamped.postEventIndices_, plan.postEventIndices_);
  for (size_t sample = 0; sample < kPlanSamples; ++sample) {
    SCOPED_TRACE(absl::StrCat("sample ", sample));
    const bool postEvent = sample == kLiftOffPostEvent || sample == kTouchDownPostEvent;
    EXPECT_EQ(stamped.timeTrajectory_[sample], postEvent ? postEventTime(plan.timeTrajectory_[sample]) : plan.timeTrajectory_[sample]);
  }
  EXPECT_EQ(stamped.timeTrajectory_[kLiftOffPostEvent], kLiftOffTime + numeric_traits::weakEpsilon<scalar_t>());

  // Idempotent: a stamped sample lies after the sample before it.
  PrimalSolution twice = stamped;
  stampPostEventSamplesAtIntervalStart(twice);
  EXPECT_EQ(twice.timeTrajectory_, stamped.timeTrajectory_);
}

TEST(StampPostEventSamplesAtIntervalStart, LeavesALoneFrontPostEventSampleAndSamplesAfterTheirEventsAlone) {
  // An event at the initial time: the front node is a lone post-event node, which toPostEventIndices() does not name.
  const ModeSchedule modeSchedule(/*eventTimesInput=*/{0.0, kLiftOffTime}, /*modeSequenceInput=*/{0, 1, 2});
  const std::vector<AnnotatedTime> grid = gridOf(/*initTime=*/0.0, modeSchedule);
  ASSERT_EQ(grid.front().event, AnnotatedTime::Event::PostEvent);
  const PrimalSolution plan = planOn(grid, modeSchedule);
  ASSERT_EQ(plan.postEventIndices_, (size_array_t{kLiftOffPostEvent}));
  PrimalSolution stamped = plan;
  stampPostEventSamplesAtIntervalStart(stamped);
  EXPECT_EQ(stamped.timeTrajectory_.front(), 0.0);
  EXPECT_EQ(stamped.timeTrajectory_[kLiftOffPostEvent], postEventTime(kLiftOffTime));

  // Named, the front sample still has no sample before it.
  PrimalSolution front = plan;
  front.postEventIndices_ = {0};
  stampPostEventSamplesAtIntervalStart(front);
  EXPECT_EQ(front.timeTrajectory_, plan.timeTrajectory_);

  // A rollout's post-event samples, and any sample later than the one before it, keep their times.
  PrimalSolution rollout;
  rollout.timeTrajectory_ = {0.0, kLiftOffTime, postEventTime(kLiftOffTime), 0.4, 0.5, 0.6};
  rollout.postEventIndices_ = {2, 4};
  const scalar_array_t rolloutTimes = rollout.timeTrajectory_;
  stampPostEventSamplesAtIntervalStart(rollout);
  EXPECT_EQ(rollout.timeTrajectory_, rolloutTimes);
}

TEST(StampPostEventSamplesAtIntervalStart, StopsAtTheNextSample) {
  // A sample closer than weakEpsilon after the event, which timeDiscretizationWithEvents() never makes (it merges nodes
  // closer than dt_min): the stamp keeps the times sorted.
  const scalar_t nextSampleTime = kLiftOffTime + 0.5 * numeric_traits::weakEpsilon<scalar_t>();
  PrimalSolution solution;
  solution.timeTrajectory_ = {0.0, kLiftOffTime, kLiftOffTime, nextSampleTime, 0.45};
  solution.postEventIndices_ = {2};
  stampPostEventSamplesAtIntervalStart(solution);
  EXPECT_EQ(solution.timeTrajectory_, (scalar_array_t{0.0, kLiftOffTime, nextSampleTime, nextSampleTime, 0.45}));
}

}  // namespace
}  // namespace ocs2
