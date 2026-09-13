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
  running_.store(true);
  worker_ = std::thread([this]() { workerLoop(); });
}

void ContactPlannerModule::stopWorker() {
  running_.store(false);
  inputCondition_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void ContactPlannerModule::setConfig(const ContactPlanningConfig& config) {
  config.validate();
  referenceManagerPtr_->setConfig(config);
  std::lock_guard<std::mutex> lock(configMutex_);
  config_ = config;
  configChanged_ = true;
}

ContactPlanningConfig ContactPlannerModule::getConfig() const {
  std::lock_guard<std::mutex> lock(configMutex_);
  return config_;
}

ContactPlannerModule::Statistics ContactPlannerModule::getStatistics() const {
  std::lock_guard<std::mutex> lock(statisticsMutex_);
  return statistics_;
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
    std::lock_guard<std::mutex> lock(inputMutex_);
    pendingInput_.reset();
  }
  std::lock_guard<std::mutex> lock(statisticsMutex_);
  ++statistics_.numPlans;
  if (!plan.valid) ++statistics_.numFailedPlans;
  statistics_.lastPlanValid = plan.valid;
  statistics_.lastPlanStartTime = input.time;
  statistics_.lastSolveTime = plan.solveTime;
  statistics_.lastNumBranchAndBoundNodes = plan.numBranchAndBoundNodes;
  statistics_.lastOptimal = plan.optimal;
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
    }
    runPlanner(input);
  }
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

  const ContactPlanningConfig config = getConfig();
  if (!config.runInBackgroundThread) {
    runPlanner(input);
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  const std::chrono::duration<scalar_t> minPeriod(1.0 / config.planningFrequency);
  if (hasPosted_ && (now - lastPostTime_) < minPeriod) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(inputMutex_);
    pendingInput_ = input;  // latest snapshot wins
  }
  lastPostTime_ = now;
  hasPosted_ = true;
  inputCondition_.notify_one();
}

}  // namespace ocs2::humanoid
