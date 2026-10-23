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

#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerFactory.h"

namespace ocs2::humanoid {

absl::StatusOr<std::shared_ptr<ContactPlannerModule>> ContactPlannerModule::Create(
    std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr,
    ContactPlanningConfig config,
    std::optional<ContactPlanningModelParameters> modelParameters) {
  if (referenceManagerPtr == nullptr) {
    return absl::InvalidArgumentError("[ContactPlannerModule] the reference manager must not be null");
  }
  if (modelParameters.has_value()) modelParameters->applyTo(config);
  // makeContactPlanner validates first and returns the rejection that names the key.
  ASSIGN_OR_RETURN(std::unique_ptr<ContactPlannerInterface> planner, makeContactPlanner(config));
  RETURN_IF_ERROR(referenceManagerPtr->setConfigStatus(config));
  // absl::WrapUnique: the constructor is private.
  return std::shared_ptr<ContactPlannerModule>(absl::WrapUnique(
      new ContactPlannerModule(std::move(referenceManagerPtr), std::move(config), std::move(modelParameters), std::move(planner))));
}

ContactPlannerModule::ContactPlannerModule(std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr,
                                           ContactPlanningConfig config,
                                           std::optional<ContactPlanningModelParameters> modelParameters,
                                           std::unique_ptr<ContactPlannerInterface> planner)
    : referenceManagerPtr_(std::move(referenceManagerPtr)),
      config_(std::move(config)),
      modelParameters_(std::move(modelParameters)),
      planner_(std::move(planner)),
      plannerType_(config_.planner.type) {
  logPlans_.store(config_.planner.logPlans);
  LOG(INFO) << "[ContactPlannerModule] contact planner formulation:\n" << planner_->getFormulationSummary();
  if (config_.planner.runInBackgroundThread) {
    startWorker();
  }
}

ContactPlannerModule::~ContactPlannerModule() {
  stopWorker();
}

void ContactPlannerModule::startWorker() {
  if (worker_.joinable()) stopWorker();  // never assign over a joinable thread (std::terminate)
  {
    // A snapshot posted before the thread was stopped, or the throttle timestamp of the previous incarnation, must not
    // leak into the new worker: it would plan from a stale time, or the first post after re-enabling would be throttled.
    std::lock_guard<std::mutex> lock(inputMutex_);
    pendingInput_.reset();
    pendingInputUrgent_ = false;
    throttle_ = SnapshotThrottle{};
  }
  running_.store(true);
  worker_ = std::thread([this]() { workerLoop(); });
}

void ContactPlannerModule::stopWorker() {
  {
    // The stop flag is part of the condition the worker waits on, so it has to be written under the same mutex the
    // worker evaluates the predicate under. Otherwise a store + notify that lands between the worker's predicate check
    // and its wait is a lost wake-up: the worker blocks forever and the join below hangs the calling thread (the solver
    // thread on a runtime toggle, or shutdown).
    std::lock_guard<std::mutex> lock(inputMutex_);
    running_.store(false);
    pendingInput_.reset();
    pendingInputUrgent_ = false;
  }
  inputCondition_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

absl::Status ContactPlannerModule::setConfig(const ContactPlanningConfig& configIn) {
  ContactPlanningConfig config = configIn;
  if (modelParameters_.has_value()) modelParameters_->applyTo(config);
  // The reference manager validates it and builds its rules before replacing anything; a rejection leaves both as they
  // were.
  RETURN_IF_ERROR(referenceManagerPtr_->setConfigStatus(config));

  bool startWorkerThread = false;
  bool stopWorkerThread = false;
  ContactPlanningConfig previous;

  {
    std::lock_guard<std::mutex> lock(configMutex_);
    if (config_.planner.runInBackgroundThread != config.planner.runInBackgroundThread) {
      if (config.planner.runInBackgroundThread) {
        startWorkerThread = true;
      } else {
        stopWorkerThread = true;
      }
    }
    previous = config_;
    config_ = config;
    configChanged_ = true;
    logPlans_.store(config.planner.logPlans);
  }
  const std::optional<std::string> summary = reloadSummary(previous, config);
  if (summary.has_value()) {
    LOG(INFO) << "[ContactPlannerModule] contact planner formulation reloaded:\n" << *summary;
  }

  if (startWorkerThread) {
    startWorker();
  } else if (stopWorkerThread) {
    stopWorker();
  }
  return absl::OkStatus();
}

std::optional<std::string> ContactPlannerModule::reloadSummary(const ContactPlanningConfig& previous, const ContactPlanningConfig& next) {
  const absl::StatusOr<std::string> before = contactPlannerSummary(previous);
  const absl::StatusOr<std::string> after = contactPlannerSummary(next);
  std::string afterText = after.ok() ? *after : std::string(after.status().message());
  // A change the summary does not print can still change what the planner is assembled from (the term lists of the
  // mixed-integer formulation), so those always re-print as well.
  const bool structuralChange = previous.formulation != next.formulation || previous.planner.numNodes != next.planner.numNodes ||
                                previous.planner.dt != next.planner.dt || previous.planner.type != next.planner.type;
  const bool summaryChanged = before.ok() != after.ok() || (before.ok() ? *before : std::string(before.status().message())) != afterText;
  if (!structuralChange && !summaryChanged) return std::nullopt;
  return afterText;
}

ContactPlanningConfig ContactPlannerModule::getConfig() const {
  std::lock_guard<std::mutex> lock(configMutex_);
  return config_;
}

ContactPlannerModule::Statistics ContactPlannerModule::getStatistics() const {
  Statistics statistics;
  {
    std::lock_guard<std::mutex> lock(statisticsMutex_);
    statistics = statistics_;
  }
  statistics.numStalePlansDropped = referenceManagerPtr_->numStalePlansDropped();
  statistics.numInconsistentPlansDropped = referenceManagerPtr_->numInconsistentPlansDropped();
  return statistics;
}

void ContactPlannerModule::reset() {
  {
    std::lock_guard<std::mutex> lock(inputMutex_);
    pendingInput_.reset();
    pendingInputUrgent_ = false;
    throttle_ = SnapshotThrottle{};
  }
  // The planner is reset by the thread that plans, at the first snapshot of the new plan epoch (runPlanner()).
}

void ContactPlannerModule::runPlanner(const ContactPlannerInput& input, uint64_t planEpoch) {
  // A snapshot taken before an MPC reset describes the robot before it: the reference manager would refuse its plan,
  // and planning it would leave the planner's warm start and previous plan to the robot before the reset.
  if (planEpoch != referenceManagerPtr_->planEpoch()) return;
  // Whatever the planner carried over belongs to the epoch it planned in, and is dropped at the first snapshot of a new
  // one. Keyed to the epoch rather than to a request that reset() raises: a plan of the old epoch that was already under
  // way on the worker when the reset came would otherwise consume the request, and leave its own state behind for the
  // first plan after the reset.
  if (planEpoch != plannerEpoch_) {
    planner_->reset();
    plannerEpoch_ = planEpoch;
  }
  {
    std::lock_guard<std::mutex> lock(configMutex_);
    if (configChanged_) {
      if (config_.planner.type != plannerType_) {
        // The implementation itself changed: build the new one and drop whatever the old one carried over.
        absl::StatusOr<std::unique_ptr<ContactPlannerInterface>> rebuilt = makeContactPlanner(config_);
        if (rebuilt.ok()) {
          planner_ = *std::move(rebuilt);
          plannerType_ = config_.planner.type;
        } else {
          LOG(ERROR) << "[ContactPlannerModule] keeping the '" << plannerType_ << "' planner: " << rebuilt.status().message();
        }
      }
      // The configuration was validated by setConfig(); a planner that still cannot run it keeps its previous one, and
      // this thread - the worker's, when planning in the background - must not throw.
      const absl::Status applied = planner_->setConfig(config_);
      if (!applied.ok()) {
        LOG(ERROR) << "[ContactPlannerModule] the '" << plannerType_ << "' planner keeps its previous configuration: " << applied.message();
      }
      configChanged_ = false;
    }
  }
  // ContactPlannerInterface::plan() reports a failure to plan as an invalid plan, never by throwing.
  const ContactPlan plan = planner_->plan(input);
  if (logPlans_.load()) LOG(INFO) << "[ContactPlannerModule] " << plan.describe();
  // A plan made from a snapshot taken before an MPC reset describes the robot before it: the reference manager refuses
  // it, and the snapshot waiting for the worker (taken after the reset) is the one to plan from next.
  if (plan.valid && referenceManagerPtr_->setContactPlan(plan, planEpoch)) {
    // Drop any snapshot taken before this plan is applied: the next plan must start from the schedule that includes it,
    // otherwise its committed window would disagree with the applied schedule and the merge could cut phases short.
    // A snapshot posted after a contact event is the exception: the schedule it was taken from has already been re-timed
    // by that event, which makes this plan stale, and the reference manager's one-shot re-plan request has already been
    // consumed for it; dropping it would delay the re-plan by a full planning period.
    std::lock_guard<std::mutex> lock(inputMutex_);
    if (!pendingInputUrgent_ && pendingInput_.has_value()) {
      pendingInput_.reset();
      throttle_.dropped();
    }
  }
  std::lock_guard<std::mutex> lock(statisticsMutex_);
  ++statistics_.numPlans;
  if (!plan.valid) ++statistics_.numFailedPlans;
  statistics_.lastPlanValid = plan.valid;
  statistics_.lastPlanStartTime = input.time;
  statistics_.lastSolveTime = plan.solveTime;
  statistics_.lastNumBranchAndBoundNodes = plan.numBranchAndBoundNodes;
  statistics_.lastOptimal = plan.optimal;
  statistics_.lastNodeLimitHit = plan.nodeLimitHit;
  statistics_.lastTimeLimitHit = plan.timeLimitHit;
  statistics_.lastObjective = plan.objective;
}

void ContactPlannerModule::workerLoop() {
  while (running_.load()) {
    ContactPlannerInput input;
    uint64_t planEpoch = 0;
    {
      std::unique_lock<std::mutex> lock(inputMutex_);
      inputCondition_.wait(lock, [this]() { return !running_.load() || pendingInput_.has_value(); });
      if (!running_.load()) return;
      if (!pendingInput_.has_value()) continue;  // the wait above returns with one; this states it for the reader
      input = std::move(*pendingInput_);
      planEpoch = pendingInputPlanEpoch_;
      pendingInput_.reset();
      pendingInputUrgent_ = false;
      throttle_.taken();
    }
    runPlanner(input, planEpoch);
  }
}

void ContactPlannerModule::postSolverRun(const PrimalSolution& primalSolution) {
  const ContactPlanningConfig config = getConfig();
  if (!config.formulation.needsPredictedTrajectory()) return;
  referenceManagerPtr_->setPredictedTrajectory(primalSolution.timeTrajectory_, primalSolution.stateTrajectory_);
}

void ContactPlannerModule::preSolverRun(scalar_t initTime,
                                        scalar_t /*finalTime*/,
                                        const vector_t& initState,
                                        const ReferenceManagerInterface& /*referenceManager*/) {
  // The commanded CoM velocity, as the reference manager read it off the target's momentum channel at this cycle,
  // before its execution rules ran. Taking it from the target here instead would feed the planner its own output
  // whenever planned_com_override is listed, since that rule rewrites exactly that channel with the planned velocity.
  const vector2_t velocityCommand = referenceManagerPtr_->commandedVelocity();
  const ContactPlannerInput input = referenceManagerPtr_->makePlannerInput(initTime, initState, velocityCommand);
  // Taken on the solver thread, like the reference manager's reset(): the snapshot and its epoch belong together.
  const uint64_t planEpoch = referenceManagerPtr_->planEpoch();

  // A contact event (early / late touch-down) invalidates the timing the last plan was built on: plan again right away
  // instead of waiting for the next planning period.
  const bool replanRequested = referenceManagerPtr_->consumeReplanRequest();

  const ContactPlanningConfig config = getConfig();
  if (!config.planner.runInBackgroundThread) {
    runPlanner(input, planEpoch);
    return;
  }

  // A plan handed over after this cycle's activation (the worker finished between the reference manager's and this
  // module's pre-solve hooks) is not in the schedule this snapshot was taken from: the worker, idle again, would plan
  // from a schedule without it, and the drop above never sees that snapshot. Wait for the next cycle, which activates
  // the plan first, unless a contact event asks for an immediate re-plan.
  if (!replanRequested && referenceManagerPtr_->hasPendingPlan()) {
    return;
  }

  const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  const std::chrono::duration<scalar_t> minPeriod(1.0 / config.planner.planningFrequency);
  {
    std::lock_guard<std::mutex> lock(inputMutex_);
    if (!replanRequested && !throttle_.allows(now, minPeriod)) {
      return;
    }
    pendingInput_ = input;  // latest snapshot wins
    pendingInputPlanEpoch_ = planEpoch;
    pendingInputUrgent_ = pendingInputUrgent_ || replanRequested;  // urgency outlives the snapshot that carried it
    throttle_.posted(now);
  }
  inputCondition_.notify_one();
}

}  // namespace ocs2::humanoid
