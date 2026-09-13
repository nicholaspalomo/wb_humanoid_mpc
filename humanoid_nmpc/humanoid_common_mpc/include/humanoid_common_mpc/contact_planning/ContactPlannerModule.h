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

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"

namespace ocs2::humanoid {

/**
 * Solver synchronized module that runs the mixed-integer contact planner and feeds its plans to the
 * ContactPlanningReferenceManager.
 *
 * Before every MPC solve the module snapshots the planner input (CoM state, feet, applied contact phases and the commanded
 * velocity from the target trajectories). With `runInBackgroundThread` the snapshot is posted to a worker thread that plans
 * at most at `planningFrequency`; otherwise the plan is computed synchronously inside the pre-solve hook. A finished plan is
 * handed to the reference manager and becomes active at the next solve.
 */
class ContactPlannerModule final : public SolverSynchronizedModule {
 public:
  struct Statistics {
    size_t numPlans = 0;
    size_t numFailedPlans = 0;
    bool lastPlanValid = false;
    scalar_t lastPlanStartTime = 0.0;
    scalar_t lastSolveTime = 0.0;
    int lastNumBranchAndBoundNodes = 0;
    bool lastOptimal = false;
    scalar_t lastObjective = 0.0;
  };

  ContactPlannerModule(std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr, ContactPlanningConfig config);
  ~ContactPlannerModule() override;

  ContactPlannerModule(const ContactPlannerModule&) = delete;
  ContactPlannerModule& operator=(const ContactPlannerModule&) = delete;

  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& initState,
                    const ReferenceManagerInterface& referenceManager) override;
  void postSolverRun(const PrimalSolution& /*primalSolution*/) override {}

  /** Updates the planner and reference manager configuration (thread-safe, applied before the next plan). */
  void setConfig(const ContactPlanningConfig& config);
  ContactPlanningConfig getConfig() const;

  Statistics getStatistics() const;

 private:
  void workerLoop();
  void runPlanner(const ContactPlannerInput& input);
  void startWorker();
  void stopWorker();

  std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr_;

  mutable std::mutex configMutex_;
  ContactPlanningConfig config_;
  bool configChanged_ = false;

  LipContactPlanner planner_;  // used by the worker thread, or by the solver thread in synchronous mode

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::mutex inputMutex_;
  std::condition_variable inputCondition_;
  std::optional<ContactPlannerInput> pendingInput_;
  std::chrono::steady_clock::time_point lastPostTime_;
  bool hasPosted_ = false;

  mutable std::mutex statisticsMutex_;
  Statistics statistics_;
};

}  // namespace ocs2::humanoid
