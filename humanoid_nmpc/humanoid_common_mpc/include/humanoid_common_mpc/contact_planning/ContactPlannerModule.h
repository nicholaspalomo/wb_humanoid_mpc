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

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_oc/synchronized_module/SolverSynchronizedModule.h"

#include "humanoid_common_mpc/contact_planning/ContactPlannerInterface.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Solver synchronized module that runs the contact planner `planner.type` names (ContactPlannerFactory: the closed-form
 * H-LIP planner or the mixed-integer one) and feeds its plans to the ContactPlanningReferenceManager.
 *
 * Before every MPC solve the module snapshots the planner input (CoM state, feet, applied contact phases, and the
 * operator's command as the reference manager recorded it - ContactPlanningReferenceManager::commandedVelocity(), not
 * the live target, which planned_com_override rewrites). With `runInBackgroundThread` the snapshot is posted to a
 * worker thread that plans at most at `planningFrequency`; otherwise the plan is computed synchronously inside the
 * pre-solve hook. Either way a finished plan is handed to the reference manager as PENDING and becomes active at the
 * next solve: the solver runs the reference manager's pre-solve hook (which activates pending plans) before the
 * modules', so even a synchronous plan made from this cycle's state acts one MPC cycle later. What the synchronous
 * path removes is the planning period and the worker's queueing, not that cycle.
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

  /**
   * Builds the module and the planner `config.planner.type` names, and hands the configuration to the reference
   * manager. `modelParameters` - the planner parameters derived from the robot model (torque limits, foot yaw bounds,
   * comHeight where the file leaves it out, the ZMP box where it leaves it at 0) - are applied to `config` first and to every configuration
   * set later, including the hot reloads, which do not carry them. A null reference manager, an unknown planner or a
   * configuration that does not validate is an InvalidArgument naming the key to change.
   */
  static absl::StatusOr<std::shared_ptr<ContactPlannerModule>> Create(
      std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr,
      ContactPlanningConfig config,
      std::optional<ContactPlanningModelParameters> modelParameters = std::nullopt);
  ~ContactPlannerModule() override;

  ContactPlannerModule(const ContactPlannerModule&) = delete;
  ContactPlannerModule& operator=(const ContactPlannerModule&) = delete;

  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& initState,
                    const ReferenceManagerInterface& referenceManager) override;
  /** Hands the predicted trajectory to the reference manager for the closed-form corrections (when they are enabled). */
  void postSolverRun(const PrimalSolution& primalSolution) override;

  /**
   * Drops the snapshot waiting for the worker and the throttle's history. The configuration and the statistics are
   * kept. Solver thread, between solves.
   *
   * The planner itself is reset by the thread that plans (runPlanner()), keyed to the reference manager's plan epoch,
   * which the reference manager's reset() - run first by MPC_BASE::reset() - advances: a snapshot taken before the reset
   * is not planned from at all, and the first snapshot of the new epoch is planned by a planner reset
   * (ContactPlannerInterface::reset(): its warm start, its previous plan). A plan still being computed from a snapshot
   * of the old epoch when the reset comes is refused by the reference manager when it is handed over, and the state it
   * leaves in the planner belongs to the old epoch, so it is dropped before the next plan.
   */
  void reset() override;

  /**
   * Updates the planner and reference manager configuration (thread-safe, applied before the next plan), and logs the
   * planner's formulation summary again whenever reloadSummary() says it changed. The model parameters are applied
   * first; a configuration that then does not validate, or that the reference manager refuses, is returned as the
   * InvalidArgument naming the key, and the running configuration is kept.
   */
  absl::Status setConfig(const ContactPlanningConfig& config);

  /**
   * The formulation summary to log after a reload from `previous` to `next`, or empty when the reload changes nothing
   * the summary reports.
   *
   * The summary is not only a record: for the H-LIP planner it carries the start-up check of whether the first step out
   * of a standstill fits the reach (HlipContactPlanner::formulationSummary), and every key that check depends on -
   * hlip.sspDuration, hlip.dspDuration, hlip.stepWidth, hlip.maxStepWidth - is a hot-reloadable slider. Re-printing only
   * when the formulation, the grid or the planner type changed meant a cadence tuned from the GUI was never checked at
   * all. Comparing the summaries themselves re-runs every check the start-up print runs, for exactly the keys it
   * reports, without this module knowing which keys those are.
   */
  static std::optional<std::string> reloadSummary(const ContactPlanningConfig& previous, const ContactPlanningConfig& next);
  ContactPlanningConfig getConfig() const;

  Statistics getStatistics() const;

 private:
  /** Private: Create() validates the configuration and builds the planner first. */
  ContactPlannerModule(std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr,
                       ContactPlanningConfig config,
                       std::optional<ContactPlanningModelParameters> modelParameters,
                       std::unique_ptr<ContactPlannerInterface> planner);

  void workerLoop();
  /**
   * Plans from `input`, a snapshot taken in the reference manager's plan epoch `planEpoch`, and hands the plan over. A
   * snapshot of an epoch that is no longer the reference manager's is dropped unplanned; the first one of a new epoch
   * is planned by a reset planner (see reset()).
   */
  void runPlanner(const ContactPlannerInput& input, uint64_t planEpoch);
  void startWorker();
  void stopWorker();

  std::shared_ptr<ContactPlanningReferenceManager> referenceManagerPtr_;

  mutable std::mutex configMutex_;
  ContactPlanningConfig config_;
  std::optional<ContactPlanningModelParameters> modelParameters_;
  bool configChanged_ = false;

  // Used by the worker thread, or by the solver thread in synchronous mode. The implementation is chosen by name
  // (`planner.type`, ContactPlannerFactory) and is rebuilt when that name changes.
  std::unique_ptr<ContactPlannerInterface> planner_;
  std::string plannerType_;  // the name `planner_` was built from; guarded by the same access rules as `planner_`
  // The plan epoch of the snapshot `planner_` last planned from: what it carries over (warm start, previous plan)
  // belongs to that epoch. Same access rules as `planner_`.
  uint64_t plannerEpoch_ = 0;

  std::thread worker_;
  std::atomic<bool> running_{false};
  std::mutex inputMutex_;
  std::condition_variable inputCondition_;
  std::optional<ContactPlannerInput> pendingInput_;
  uint64_t pendingInputPlanEpoch_ = 0;  // the reference manager's plan epoch when pendingInput_ was taken
  bool pendingInputUrgent_ = false;     // the pending snapshot follows a contact event and must not be dropped
  SnapshotThrottle throttle_;

  mutable std::mutex statisticsMutex_;
  Statistics statistics_;
  std::atomic<bool> logPlans_{false};  // planner.logPlans, mirrored here so the worker needs no config lock per plan
};

}  // namespace ocs2::humanoid
