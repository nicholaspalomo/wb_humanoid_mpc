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

#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/ConfigReload.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplierList.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"
#include "humanoid_common_mpc/parameter_update/OcpTermUpdates.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"

namespace ocs2::humanoid {

namespace {

// Where an enqueued update comes from, for the log.
constexpr absl::string_view kEnqueuedUpdateSource = "the enqueued parameter update (operator/mpc_parameters)";

/**
 * The SqpSolver of `mpc`, or nullptr when it runs another solver: the hot reload reaches the settings and the per-worker
 * problems of an SqpSolver only.
 */
SqpSolver* absl_nullable sqpSolverOf(MPC_BASE& mpc) {
  // NOLINTNEXTLINE(rtti): MPC_BASE hands its solver out as a SolverBase, and only an SqpSolver has what is reloaded.
  return dynamic_cast<SqpSolver*>(mpc.getSolverPtr());
}

/** Logs the reload of `source` as stopped part-way when `status`, what exceptionsToStatus() caught out of it, is an error. */
void reportStoppedReload(absl::string_view source, const absl::Status& status) {
  if (!status.ok()) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] the reload of " << source << " stopped part-way: " << status.message();
  }
}

/** The name the tuning options give the formulation of `mpc` (ConfigReload.h). */
absl::string_view formulationName(StateInputLayout::Mpc mpc) {
  switch (mpc) {
    case StateInputLayout::Mpc::kCentroidal:
      return kCentroidalFormulation;
    case StateInputLayout::Mpc::kWholeBody:
      return kWholeBodyFormulation;
  }
  // Unreachable for the enumerators above; a value cast from outside them reports only what every formulation reads.
  return absl::string_view();
}

/** The modification time of `path`, or nullopt when it cannot be read (the file does not exist, or `path` is empty). */
std::optional<std::filesystem::file_time_type> lastWriteTime(const std::string& path) {
  if (path.empty()) return std::nullopt;
  std::error_code error;
  const std::filesystem::file_time_type time = std::filesystem::last_write_time(path, error);
  if (error) return std::nullopt;
  return time;
}

}  // namespace

absl::StatusOr<std::unique_ptr<MpcParameterUpdaterModule>> MpcParameterUpdaterModule::Create(MPC_BASE* absl_nullable mpcPtr,
                                                                                             Options options) {
  for (size_t i = 0; i < options.appliers.size(); ++i) {
    if (options.appliers[i] == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("[MpcParameterUpdaterModule] applier ", i, " of ", options.appliers.size(), " is null."));
    }
  }
  SqpSolver* absl_nullable sqpSolverPtr = mpcPtr != nullptr ? sqpSolverOf(*mpcPtr) : nullptr;
  return absl::WrapUnique(new MpcParameterUpdaterModule(mpcPtr, sqpSolverPtr, std::move(options)));
}

MpcParameterUpdaterModule::MpcParameterUpdaterModule(MPC_BASE* absl_nullable mpcPtr, SqpSolver* absl_nullable sqpSolverPtr, Options options)
    : mpcPtr_(mpcPtr),
      sqpSolverPtr_(sqpSolverPtr),
      taskFile_(std::move(options.taskFile)),
      referenceFile_(std::move(options.referenceFile)),
      contactPlanningFile_(std::move(options.contactPlanningFile)),
      runningTask_(std::move(options.runningTask)),
      layout_(std::move(options.layout)),
      inputDim_(options.inputDim),
      referenceManagerPtr_(options.referenceManager),
      appliers_(std::move(options.appliers)) {
  taskFileLastWriteTime_ = lastWriteTime(taskFile_).value_or(std::filesystem::file_time_type());
  referenceFileLastWriteTime_ = lastWriteTime(referenceFile_).value_or(std::filesystem::file_time_type());
  contactPlanningFileLastWriteTime_ = lastWriteTime(contactPlanningFile_).value_or(std::filesystem::file_time_type());
}

std::vector<std::string> MpcParameterUpdaterModule::appliedFields() const {
  std::vector<absl::Span<const absl::string_view>> fieldLists;
  fieldLists.reserve(appliers_.size());
  for (const std::unique_ptr<HotFieldApplier>& applier : appliers_) fieldLists.push_back(applier->fields());
  return sortedFieldUnion(fieldLists);
}

void MpcParameterUpdaterModule::setContactPlannerModule(std::shared_ptr<ContactPlannerModule> contactPlannerModule) {
  contactPlannerModulePtr_ = std::move(contactPlannerModule);
}

void MpcParameterUpdaterModule::addReferenceFileReloader(ReferenceFileReloader reloader) {
  referenceFileReloaders_.push_back(std::move(reloader));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MpcParameterUpdaterModule::preSolverRun(scalar_t /*initTime*/,
                                             scalar_t /*finalTime*/,
                                             const vector_t& /*currentState*/,
                                             const ReferenceManagerInterface& /*referenceManager*/) {
  // First, before any reload: the reference manager has just rebuilt this solve's references, so this is the moment
  // what follows them (the contact-implicit terms' ground) can take them.
  if (sqpSolverPtr_ != nullptr) {
    for (std::unique_ptr<HotFieldApplier>& applier : appliers_) {
      reportStoppedReload(applier->name(), exceptionsToStatus([&]() { applier->beforeEverySolve(*sqpSolverPtr_, referenceManagerPtr_); }));
    }
  }

  // Pathway 1: an update enqueued by enqueueParameterUpdate(), which takes priority.
  if (hasPendingUpdate_.load(std::memory_order_acquire)) {
    applyPendingUpdate();
  }

  // Pathway 2: the modification times of the watched files at roughly 1 Hz (the solver runs around 100 Hz).
  if (checkCounter_++ % 100 == 0) {
    applyChangedFiles();
  }
}

void MpcParameterUpdaterModule::applyPendingUpdate() {
  std::optional<mpc_config::MpcParameterUpdate> update;
  {
    absl::MutexLock lock(&pendingMutex_);
    update = std::move(pendingUpdate_);
    pendingUpdate_.reset();
    hasPendingUpdate_.store(false, std::memory_order_release);
  }
  if (!update.has_value()) return;
  // Every conversion reports a block it cannot use by its field and keeps the running one, and every applier runs
  // inside exceptionsToStatus(); this one is the last line of defense, so that nothing an update contains can throw out
  // of the solver's pre-solve hook.
  reportStoppedReload(kEnqueuedUpdateSource, exceptionsToStatus([&]() {
                        applyTaskFile(update->task, kEnqueuedUpdateSource);
                        if (update->contact_planning.has_value()) {
                          applyContactPlanningFile(*update->contact_planning, kEnqueuedUpdateSource);
                        }
                      }));
}

void MpcParameterUpdaterModule::applyChangedFiles() {
  if (const std::optional<std::filesystem::file_time_type> time = lastWriteTime(taskFile_);
      time.has_value() && *time != taskFileLastWriteTime_) {
    taskFileLastWriteTime_ = *time;
    if (absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile_); task.ok()) {
      reportStoppedReload(taskFile_, exceptionsToStatus([&]() { applyTaskFile(*task, taskFile_); }));
    } else {
      LOG(ERROR) << "[MpcParameterUpdaterModule] " << taskFile_ << " was not reloaded, nothing was applied: " << task.status().message();
    }
  }
  if (const std::optional<std::filesystem::file_time_type> time = lastWriteTime(contactPlanningFile_);
      time.has_value() && *time != contactPlanningFileLastWriteTime_) {
    contactPlanningFileLastWriteTime_ = *time;
    if (absl::StatusOr<mpc_config::ContactPlanningFile> file = loadContactPlanningFile(contactPlanningFile_); file.ok()) {
      reportStoppedReload(contactPlanningFile_, exceptionsToStatus([&]() { applyContactPlanningFile(*file, contactPlanningFile_); }));
    } else {
      LOG(ERROR) << "[MpcParameterUpdaterModule] " << contactPlanningFile_
                 << " was not reloaded, the running planner is kept: " << file.status().message();
    }
  }
  // The command limits and ramps live in the reference file, which nothing else watches: without this a change to it
  // needed a restart of the controller (the Command Limits tab of the remote control writes exactly this file).
  if (referenceFileReloaders_.empty()) return;
  if (const std::optional<std::filesystem::file_time_type> time = lastWriteTime(referenceFile_);
      time.has_value() && *time != referenceFileLastWriteTime_) {
    referenceFileLastWriteTime_ = *time;
    reloadReferenceFile();
  }
}

void MpcParameterUpdaterModule::reloadReferenceFile() {
  // Converted once for every consumer, so that they all apply the same limits, and a file they cannot use is reported
  // once, naming it, and reaches none of them.
  const absl::StatusOr<mpc_config::ReferenceFile> file = loadReferenceFile(referenceFile_);
  const absl::StatusOr<ReferenceSettings> settings = file.ok() ? referenceSettingsFromConfig(*file) : file.status();
  if (!settings.ok()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] " << referenceFile_
                 << " was not reloaded, the running command limits are kept: " << settings.status().message();
    return;
  }
  bool refused = false;
  for (const ReferenceFileReloader& reload : referenceFileReloaders_) {
    // A reloader that refuses the limits leaves the others to run.
    if (const absl::Status status = reload(*settings); !status.ok()) {
      LOG(WARNING) << "[MpcParameterUpdaterModule] Failed to reload " << referenceFile_ << ": " << status.message();
      refused = true;
    }
  }
  if (!refused) LOG(INFO) << "[MpcParameterUpdaterModule] Reloaded command limits from " << referenceFile_;
}

void MpcParameterUpdaterModule::enqueueParameterUpdate(const mpc_config::MpcParameterUpdate& update) {
  absl::MutexLock lock(&pendingMutex_);
  pendingUpdate_ = update;
  hasPendingUpdate_.store(true, std::memory_order_release);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MpcParameterUpdaterModule::reportStartUpChanges(const mpc_config::TaskFile& task, absl::string_view source) const {
  humanoid_mpc_config::TaskFile running;
  mpc_config::ToProto(runningTask_, &running);
  humanoid_mpc_config::TaskFile reloaded;
  mpc_config::ToProto(task, &reloaded);
  // Only what this MPC reads at its start: a field of the other formulation, the GUI's own and one no one applies
  // (kUnspecified) are not this MPC's to restart for, and the robot process reads its own copy of the file.
  const absl::string_view formulation = formulationName(layout_.mpc);
  std::vector<std::string> mpcFields;
  std::vector<std::string> robotFields;
  for (const ConfigChange& change : changedStartUpFields(running, reloaded)) {
    if (change.reload != ConfigReload::kStartUp) continue;
    if (isReadByTheMpc(change.consumer)) {
      if (isReadByFormulation(change.formulations, formulation)) mpcFields.push_back(change.path);
    } else if (change.consumer == kRobotConsumer) {
      robotFields.push_back(change.path);
    }
  }
  if (!mpcFields.empty()) {
    LOG(WARNING) << "[MpcParameterUpdaterModule] " << source << " changes " << mpcFields.size()
                 << " field(s) that a reload does not apply; they take effect at the MPC's next start: " << absl::StrJoin(mpcFields, ", ");
  }
  if (!robotFields.empty()) {
    LOG(INFO) << "[MpcParameterUpdaterModule] " << source << " changes " << robotFields.size()
              << " field(s) the robot process reads, at its own next start, from its copy of the file: "
              << absl::StrJoin(robotFields, ", ");
  }
}

void MpcParameterUpdaterModule::applyTaskFile(const mpc_config::TaskFile& task, absl::string_view source) {
  LOG(INFO) << "[MpcParameterUpdaterModule] Applying in-place parameter updates from " << source << "...";
  reportStartUpChanges(task, source);

  if (mpcPtr_ == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] mpcPtr_ is null.";
    return;
  }
  if (sqpSolverPtr_ == nullptr) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] Underlying solver is not SqpSolver. Cannot update parameters.";
    return;
  }
  if (sqpSolverPtr_->getOcpDefinitions().empty()) {
    LOG(ERROR) << "[MpcParameterUpdaterModule] the SqpSolver has no problem to update.";
    return;
  }

  // Every applier writes its own objects, so one that throws leaves the others' writes intact and the next one runs.
  HotUpdateTarget target(&task, sqpSolverPtr_, referenceManagerPtr_, &layout_, inputDim_, source);
  bool stopped = false;
  for (std::unique_ptr<HotFieldApplier>& applier : appliers_) {
    if (const absl::Status status = exceptionsToStatus([&]() { applier->apply(target); }); !status.ok()) {
      LOG(ERROR) << "[MpcParameterUpdaterModule] " << applier->name() << " of the reload of " << source
                 << " stopped part-way: " << status.message();
      stopped = true;
    }
  }
  if (!stopped) LOG(INFO) << "[MpcParameterUpdaterModule] Successfully applied in-place parameter updates to SqpSolver.";
}

void MpcParameterUpdaterModule::applyContactPlanningFile(const mpc_config::ContactPlanningFile& file, absl::string_view source) {
  if (contactPlannerModulePtr_ == nullptr) return;
  // Converted WITHOUT the validation of ContactPlanningConfig, exactly as the MPC interface does at start-up: the
  // parameters a robot may leave out of its file, or at 0, to mean "derive this one from the model" - shared.com_height
  // and the two zmp_support_region half widths - are only filled in by ContactPlanningModelParameters::applyTo(), which
  // ContactPlannerModule::setConfig() runs before it validates. Validating first would refuse every reload of a robot
  // that takes the documented option; setConfig() still refuses, and the log reports, a genuinely inconsistent edit.
  absl::StatusOr<ContactPlanningConfig> config =
      contactPlanningConfigFromConfig(file, ContactPlanningValidation::kDeferUntilModelParametersApplied);
  absl::Status applied = config.status();
  if (applied.ok()) applied = contactPlannerModulePtr_->setConfig(*config);
  if (applied.ok()) {
    LOG(INFO) << "[MpcParameterUpdaterModule] Applied the contact planner's configuration from " << source << ".";
  } else {
    LOG(WARNING) << "[MpcParameterUpdaterModule] the contact planner's configuration of " << source
                 << " was not applied, the running one is kept: " << applied.message();
  }
}

}  // namespace ocs2::humanoid
