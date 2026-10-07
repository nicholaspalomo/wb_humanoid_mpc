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
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_oc/synchronized_module/SolverSynchronizedModule.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/**
 * SolverSynchronizedModule that applies a reloaded configuration to the running MPC of either formulation: it updates
 * the cost and constraint parameters in place on the solver's existing OCP objects, through their setGains/setWeights/
 * setConfig methods, rather than rebuilding the MPC interface (which would leave the solver with dangling references).
 * What a reload writes is decided by the hot-field appliers the formulation hands it (Options::appliers;
 * centroidalHotFieldAppliers(), wholeBodyHotFieldAppliers()); this module is the machinery both share.
 *
 * A reload is a whole typed file, and it is applied as that file (humanoid_nmpc/humanoid_mpc_config/README.md, "Live
 * updates"): a block it leaves out is the block's defaults, exactly as at start-up. The fields that are applied are the
 * ones the schema marks RELOAD_HOT (humanoid_nmpc/humanoid_mpc_config/tuning_options.proto), listed by appliedFields().
 * Every RELOAD_START_UP field this MPC reads (no consumer, or "mpc", and its formulation's, by layout.mpc) that differs
 * from the file the MPC started with is logged once per reload, by its path, as taking effect at the MPC's next start;
 * the robot process's RELOAD_START_UP fields are logged apart, as read at its own next start, and the GUI's fields, the
 * other formulation's and the ones no one applies are not reported. A block whose conversion refuses it is logged by
 * its field path and keeps its running values; the rest of the file still applies, and an applier that throws is logged
 * while the next one runs.
 *
 * Two pathways deliver a reload:
 *   1. File watching: the task file and the contact planner's file, polled about once a second (every 100th
 *      preSolverRun()), and the reference file, whose command limits go to the registered reloaders
 *      (addReferenceFileReloader()).
 *   2. enqueueParameterUpdate(): the tuning GUI's whole task file and contact planner's file, applied by the next
 *      preSolverRun() without touching any file. The transport is the caller's (the MPC node forwards
 *      operator/mpc_parameters).
 *
 * preSolverRun() runs on the solver thread, before any worker of the solve, after the reference manager's own
 * preSolverRun(); enqueueParameterUpdate() is thread-safe, and so are the const methods. Nothing here runs on the
 * robot's realtime thread: the robot is another process. A reset of the MPC keeps the applied tuning and a pending
 * update (SolverSynchronizedModule::reset() is not overridden).
 */
class MpcParameterUpdaterModule final : public SolverSynchronizedModule {
 public:
  /**
   * A consumer of the reference file (addReferenceFileReloader()): applies its command limits, the ones the updater
   * converted the reloaded file to (referenceSettingsFromConfig()), or returns why it refused them.
   */
  using ReferenceFileReloader = std::function<absl::Status(const ReferenceSettings&)>;

  /** What the updater reaches, watches and applies with, for Create(). */
  struct Options {
    // The task file watched for changes, its textproto; empty: none is watched.
    std::string taskFile;
    // The reference file watched for changes, for the reloaders; empty: none is watched.
    std::string referenceFile;
    // The contact planner's file watched for changes; empty: none is watched (a robot without one).
    std::string contactPlanningFile;
    // The task file the running problem was built from, which a reload's start-up fields are compared with.
    mpc_config::TaskFile runningTask;
    // The coordinates the task file's weights are addressed on: stateInputLayout() of the MPC's model settings.
    StateInputLayout layout;
    // The dimension of the OCP input, i.e. the input layout the solver actually optimizes over. With basis-vector contact
    // inputs this is the basis-space dimension, not the wrench-space one.
    size_t inputDim = 0;
    // Optional; must outlive the updater. Its swing trajectory planner is reconfigured, and it owns the ground: a
    // reloaded terrain_height is handed to it (setTerrainHeight), and the contact-implicit terms follow the ground it
    // applied (getAppliedTerrainHeight) from the next solve on, so that they never disagree with the swing trajectories
    // or the landing targets about where it is. Without one, the terms take a reloaded height at once.
    SwitchedModelReferenceManager* absl_nullable referenceManager = nullptr;
    // What a reload writes, in the order it is applied; none may be null.
    std::vector<std::unique_ptr<HotFieldApplier>> appliers;
  };

  /**
   * The updater of `mpcPtr` (may be nullptr; nothing but the start-up report and the reloaders then runs, and the solver
   * updates are skipped), with what `options` names, or InvalidArgument when an applier is null. The appliers that can
   * be refused were refused by their own Create() already.
   */
  static absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> Create(MPC_BASE* absl_nullable mpcPtr, Options options);

  ~MpcParameterUpdaterModule() override = default;
  MpcParameterUpdaterModule(const MpcParameterUpdaterModule&) = delete;
  MpcParameterUpdaterModule& operator=(const MpcParameterUpdaterModule&) = delete;

  /**
   * The fields of the task file that a reload applies, by their path from the file message (a block's path stands for
   * every field below it): the sorted union of the appliers' fields(), which the formulation's static list
   * (centroidalHotFieldNames(), wholeBodyHotFieldNames()) equals and the reload-class coverage test compares with the
   * schema. The contact planner's file is applied whole.
   */
  std::vector<std::string> appliedFields() const;

  /**
   * Hands the updater a whole task file and, for a robot with a contact planner, its file (as the tuning GUI publishes
   * them on operator/mpc_parameters), to be applied by the next preSolverRun() exactly as edits of the files on disk
   * would be, without writing any file. An update enqueued before the previous one was applied replaces it. Thread-safe,
   * for any thread but the solver's; copies the update.
   */
  void enqueueParameterUpdate(const mpc_config::MpcParameterUpdate& update) ABSL_LOCKS_EXCLUDED(pendingMutex_);

  /**
   * Registers the contact planner module (may be nullptr; an OCS2 synchronized module, hence shared) so that its
   * configuration is hot-reloadable as well: the contact planner's file (watched like the task file), and the contact
   * planner's file of an enqueued update.
   */
  void setContactPlannerModule(std::shared_ptr<ContactPlannerModule> contactPlannerModule);

  /**
   * Registers something whose parameters come from the reference file, to be reloaded when that file changes on disk.
   *
   * The command limits and the command ramps are read from that file at construction by the target trajectories
   * calculator and the procedural motion manager (makeCommandLimitsReloaders()). Whenever the file changes, on the solver
   * thread, the updater reads and converts it once, and calls every reloader with its command limits; a file that does
   * not parse or convert is logged naming it and reaches no reloader, so the running limits are kept, and a reloader
   * that refuses the limits leaves the others to run. Each consumer is responsible for the thread safety of what it
   * writes (both of the above hold their limits in atomics).
   */
  void addReferenceFileReloader(ReferenceFileReloader reloader);

  /**
   * Runs every applier's beforeEverySolve(), then applies the update enqueued last, if any, and every 100th call the
   * watched files that changed. On the solver thread.
   */
  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& currentState,
                    const ReferenceManagerInterface& referenceManager) override;

  void postSolverRun(const PrimalSolution& /*primalSolution*/) override {}

 private:
  /** Use Create(), which checks the options this only stores; `sqpSolverPtr` is the SqpSolver of `mpcPtr`, if any. */
  MpcParameterUpdaterModule(MPC_BASE* absl_nullable mpcPtr, SqpSolver* absl_nullable sqpSolverPtr, Options options);

  /**
   * Applies the hot fields of `task` with every applier in turn, and logs the other fields that differ from the running
   * file. `source` names where it came from, for the log.
   */
  void applyTaskFile(const mpc_config::TaskFile& task, absl::string_view source);

  /** Applies the contact planner's file `file` to the contact planner, if one is registered. */
  void applyContactPlanningFile(const mpc_config::ContactPlanningFile& file, absl::string_view source);

  /**
   * Logs the RELOAD_START_UP fields of `task` that differ from the running file's: this MPC's (by its formulation) as
   * taking effect at its next start, the robot process's apart.
   */
  void reportStartUpChanges(const mpc_config::TaskFile& task, absl::string_view source) const;

  /** Applies the update enqueued last, if any. */
  void applyPendingUpdate() ABSL_LOCKS_EXCLUDED(pendingMutex_);

  /** Polls the watched files' modification times and applies the ones that changed. */
  void applyChangedFiles();

  /** Reads and converts the reference file and hands it to the reloaders; logs what refused it. */
  void reloadReferenceFile();

  MPC_BASE* absl_nullable mpcPtr_;
  // The solver of mpcPtr_ when it is an SqpSolver, whose per-worker problems and settings a reload writes; null
  // otherwise, and then nothing is applied.
  SqpSolver* absl_nullable sqpSolverPtr_;
  const std::string taskFile_;
  const std::string referenceFile_;
  const std::string contactPlanningFile_;
  // The task file the running problem was built from (Options::runningTask).
  const mpc_config::TaskFile runningTask_;
  const StateInputLayout layout_;
  const size_t inputDim_;
  SwitchedModelReferenceManager* absl_nullable referenceManagerPtr_;
  // Never null (Create()); applied in this order.
  std::vector<std::unique_ptr<HotFieldApplier>> appliers_;
  /// Optional contact planner whose configuration is hot-reloaded from the contact planner's file.
  std::shared_ptr<ContactPlannerModule> contactPlannerModulePtr_;

  // File-watching state
  std::filesystem::file_time_type taskFileLastWriteTime_;
  std::filesystem::file_time_type contactPlanningFileLastWriteTime_;
  std::filesystem::file_time_type referenceFileLastWriteTime_;
  std::vector<ReferenceFileReloader> referenceFileReloaders_;
  size_t checkCounter_ = 0;

  // The update enqueued by enqueueParameterUpdate() and not yet applied; hasPendingUpdate_ says, without the lock, that
  // there is one.
  absl::Mutex pendingMutex_;
  std::optional<mpc_config::MpcParameterUpdate> pendingUpdate_ ABSL_GUARDED_BY(pendingMutex_);
  std::atomic<bool> hasPendingUpdate_ = false;
};

}  // namespace ocs2::humanoid
