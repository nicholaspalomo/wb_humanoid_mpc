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
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
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
    bool lastOptimal = false;  // false for a plan that is not valid, or whose search hit a limit or a solver failure
    bool lastNodeLimitHit = false;
    bool lastTimeLimitHit = false;
    scalar_t lastObjective = 0.0;
    // Plans the reference manager dropped at activation instead of applying (see ContactPlanningReferenceManager). A
    // valid plan that is dropped counts as a success here and as a drop there; while plans keep being dropped the
    // executed schedule runs out and the robot stops walking.
    size_t numStalePlansDropped = 0;
    size_t numInconsistentPlansDropped = 0;
  };

  /**
   * Rate limiter of the snapshots posted to the worker. The period is counted from the last snapshot the worker took (or
   * from the one still waiting for it), not from the last post: a snapshot posted while the worker was busy and dropped
   * at the hand-over would otherwise hold the next post for a whole period although the worker is idle, which halved the
   * planning rate whenever a plan took longer than the period. Guarded by the module's input mutex; public for the test.
   */
  struct SnapshotThrottle {
    using Clock = std::chrono::steady_clock;
    bool hasPosted = false;
    bool pending = false;  // a posted snapshot is waiting for the worker
    Clock::time_point pendingPostTime{};
    Clock::time_point lastTakenPostTime{};

    /** Whether a snapshot may be posted at `now`. */
    bool allows(Clock::time_point now, std::chrono::duration<scalar_t> minPeriod) const {
      if (!hasPosted) return true;
      return now - (pending ? pendingPostTime : lastTakenPostTime) >= minPeriod;
    }
    void posted(Clock::time_point now) {
      hasPosted = true;
      pending = true;
      pendingPostTime = now;
    }
    void taken() {
      pending = false;
      lastTakenPostTime = pendingPostTime;
    }
    void dropped() { pending = false; }
  };

  ContactPlannerModule(std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr, ContactPlanningConfig config);
  ~ContactPlannerModule() override;

  ContactPlannerModule(const ContactPlannerModule&) = delete;
  ContactPlannerModule& operator=(const ContactPlannerModule&) = delete;

  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& initState,
                    const ReferenceManagerInterface& referenceManager) override;
  /** Hands the predicted trajectory to the reference manager for the closed-form corrections (when they are enabled). */
  void postSolverRun(const PrimalSolution& primalSolution) override;

  /** Updates the planner and reference manager configuration (thread-safe, applied before the next plan). */
  void setConfig(const ContactPlanningConfig& config);
  /**
   * Parameters derived from the robot model (torque limits, foot yaw bounds, and comHeight / ZMP box where the task
   * file leaves them at 0). Applied to the current configuration and to every configuration set later, including the
   * hot reloads from the task file, which do not carry them.
   */
  void setModelParameters(const ContactPlanningModelParameters& parameters);
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
  std::optional<ContactPlanningModelParameters> modelParameters_;
  bool configChanged_ = false;

  LipContactPlanner planner_;  // used by the worker thread, or by the solver thread in synchronous mode

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::mutex inputMutex_;
  std::condition_variable inputCondition_;
  std::optional<ContactPlannerInput> pendingInput_;
  bool pendingInputUrgent_ = false;  // the pending snapshot follows a contact event and must not be dropped
  SnapshotThrottle throttle_;

  mutable std::mutex statisticsMutex_;
  Statistics statistics_;
  std::atomic<bool> logPlans_{false};  // planner.logPlans, mirrored here so the worker needs no config lock per plan
};

}  // namespace ocs2::humanoid
