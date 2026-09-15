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

#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"

#include <iostream>

#include "absl/log/log.h"

namespace ocs2::humanoid {

ContactPlannerModule::ContactPlannerModule(std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr,
                                           ContactPlanningConfig config)
    : referenceManagerPtr_(std::move(referenceManagerPtr)), config_(std::move(config)), planner_(config_) {
  if (referenceManagerPtr_ == nullptr) {
    throw std::invalid_argument("[ContactPlannerModule] reference manager must not be null");
  }
  referenceManagerPtr_->setConfig(config_);
  if (config_.runInBackgroundThread) {
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

void ContactPlannerModule::setModelParameters(const ContactPlanningModelParameters& parameters) {
  modelParameters_ = parameters;
  setConfig(getConfig());
}

void ContactPlannerModule::setConfig(const ContactPlanningConfig& configIn) {
  ContactPlanningConfig config = configIn;
  if (modelParameters_.has_value()) modelParameters_->applyTo(config);
  config.validate();
  referenceManagerPtr_->setConfig(config);

  bool startWorkerThread = false;
  bool stopWorkerThread = false;

  {
    std::lock_guard<std::mutex> lock(configMutex_);
    if (config_.runInBackgroundThread != config.runInBackgroundThread) {
      if (config.runInBackgroundThread) {
        startWorkerThread = true;
      } else {
        stopWorkerThread = true;
      }
    }
    config_ = config;
    configChanged_ = true;
  }

  if (startWorkerThread) {
    startWorker();
  } else if (stopWorkerThread) {
    stopWorker();
  }
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

void ContactPlannerModule::runPlanner(const ContactPlannerInput& input) {
  {
    std::lock_guard<std::mutex> lock(configMutex_);
    if (configChanged_) {
      planner_.setConfig(config_);
      configChanged_ = false;
    }
  }
  ContactPlan plan;
  try {
    plan = planner_.plan(input);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[ContactPlannerModule] planning failed: " << e.what();
    plan.valid = false;
  }
  if (plan.valid) {
    referenceManagerPtr_->setContactPlan(plan);
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
    {
      std::unique_lock<std::mutex> lock(inputMutex_);
      inputCondition_.wait(lock, [this]() { return !running_.load() || pendingInput_.has_value(); });
      if (!running_.load()) return;
      input = std::move(*pendingInput_);
      pendingInput_.reset();
      pendingInputUrgent_ = false;
      throttle_.taken();
    }
    runPlanner(input);
  }
}

void ContactPlannerModule::postSolverRun(const PrimalSolution& primalSolution) {
  const ContactPlanningConfig config = getConfig();
  if (!config.enableDcmStepAdjustment && !config.enableEnergyCadenceModulation) return;
  referenceManagerPtr_->setPredictedTrajectory(primalSolution.timeTrajectory_, primalSolution.stateTrajectory_);
}

void ContactPlannerModule::preSolverRun(scalar_t initTime,
                                        scalar_t /*finalTime*/,
                                        const vector_t& initState,
                                        const ReferenceManagerInterface& referenceManager) {
  // The commanded CoM velocity is the normalized linear momentum target at the start of the horizon.
  vector2_t velocityCommand = vector2_t::Zero();
  const TargetTrajectories& targetTrajectories = referenceManager.getTargetTrajectories();
  if (!targetTrajectories.empty()) {
    const vector_t desiredState = targetTrajectories.getDesiredState(initTime);
    if (desiredState.size() >= 2) {
      velocityCommand = desiredState.head<2>();
    }
  }
  const ContactPlannerInput input = referenceManagerPtr_->makePlannerInput(initTime, initState, velocityCommand);

  // A contact event (early / late touch-down) invalidates the timing the last plan was built on: plan again right away
  // instead of waiting for the next planning period.
  const bool replanRequested = referenceManagerPtr_->consumeReplanRequest();

  const ContactPlanningConfig config = getConfig();
  if (!config.runInBackgroundThread) {
    runPlanner(input);
    return;
  }

  // A plan handed over after this cycle's activation (the worker finished between the reference manager's and this
  // module's pre-solve hooks) is not in the schedule this snapshot was taken from: the worker, idle again, would plan
  // from a schedule without it, and the drop above never sees that snapshot. Wait for the next cycle, which activates
  // the plan first, unless a contact event asks for an immediate re-plan.
  if (!replanRequested && referenceManagerPtr_->hasPendingPlan()) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  const std::chrono::duration<scalar_t> minPeriod(1.0 / config.planningFrequency);
  {
    std::lock_guard<std::mutex> lock(inputMutex_);
    if (!replanRequested && !throttle_.allows(now, minPeriod)) {
      return;
    }
    pendingInput_ = input;                                         // latest snapshot wins
    pendingInputUrgent_ = pendingInputUrgent_ || replanRequested;  // urgency outlives the snapshot that carried it
    throttle_.posted(now);
  }
  inputCondition_.notify_one();
}

}  // namespace ocs2::humanoid
