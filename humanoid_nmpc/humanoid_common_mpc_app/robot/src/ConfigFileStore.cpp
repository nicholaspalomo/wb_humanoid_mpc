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

#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_common_mpc_app/robot/AtomicFileWrite.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/StoredConfigFileKinds.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/reference_file.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.pb.h"
#include "humanoid_mpc_msgs/config_file_save.pb.h"
#include "humanoid_mpc_msgs/config_file_save_status.pb.h"
#include "nproto/Schema.h"
#include "robot_ipc/Delivery.h"

namespace ocs2::humanoid {
namespace {

using Result = humanoid_mpc_msgs::ConfigFileSaveStatus::Result;

constexpr double kLogPeriodSeconds = 5.0;
constexpr absl::string_view kBackupSuffix = ".bak";

/** What the messages call the file of `kind`. */
absl::string_view kindName(msgs::ConfigFileKind kind) {
  return configFileKindName(kind);
}

/** The answer to the save (`kind`, `sequence`, `configPath`). */
humanoid_mpc_msgs::ConfigFileSaveStatus answerOf(humanoid_mpc_msgs::ConfigFileKind kind,
                                                 uint64_t sequence,
                                                 const std::string& configPath,
                                                 Result result,
                                                 std::string message,
                                                 std::string storedPath) {
  humanoid_mpc_msgs::ConfigFileSaveStatus status;
  status.set_sequence(sequence);
  status.set_kind(kind);
  status.set_config_path(configPath);
  status.set_result(result);
  status.set_message(std::move(message));
  status.set_stored_path(std::move(storedPath));
  return status;
}

/** The protobuf enum of `kind`. */
humanoid_mpc_msgs::ConfigFileKind protoKind(msgs::ConfigFileKind kind) {
  humanoid_mpc_msgs::ConfigFileKind proto = humanoid_mpc_msgs::CONFIG_FILE_KIND_UNSPECIFIED;
  msgs::ToProto(kind, &proto);
  return proto;
}

}  // namespace

std::string configFileSchemaFingerprint(msgs::ConfigFileKind kind) {
  switch (kind) {
    case msgs::ConfigFileKind::kTask:
      return nproto::SchemaFingerprint(*humanoid_mpc_config::TaskFile::descriptor());
    case msgs::ConfigFileKind::kReference:
      return nproto::SchemaFingerprint(*humanoid_mpc_config::ReferenceFile::descriptor());
    case msgs::ConfigFileKind::kJointPdGains:
      return nproto::SchemaFingerprint(*humanoid_mpc_config::JointPdGainsFile::descriptor());
    case msgs::ConfigFileKind::kUnspecified:
      return std::string();
  }
  return std::string();
}

ConfigFileStore::Config configFileStoreConfig(const RobotConfigDirectory& directory, std::string robotName) {
  ConfigFileStore::Config config;
  for (const msgs::ConfigFileKind kind : kStoredConfigFileKinds) {
    if (ConfigFileStore::StoredFile* absl_nullable file = ConfigFileStore::fileIn(config, kind); file != nullptr) {
      *file = {.path = directory.storedPath(kind), .identity = directory.identity(kind), .inPlacePath = directory.readPath(kind)};
    }
  }
  config.robotName = std::move(robotName);
  config.unavailableReason = directory.storeError();
  return config;
}

absl::StatusOr<std::unique_ptr<ConfigFileStore>> ConfigFileStore::Create(Config config, Hooks hooks) {
  if (config.queueCapacity == 0 || config.maxTextBytes == 0) {
    return absl::InvalidArgumentError("ConfigFileStore: the queue capacity and the text limit must be positive");
  }
  return absl::WrapUnique(new ConfigFileStore(std::move(config), std::move(hooks)));
}

ConfigFileStore::ConfigFileStore(Config config, Hooks hooks) : config_(std::move(config)), hooks_(std::move(hooks)) {
  for (size_t index = 0; index < kNumStoredConfigFileKinds; ++index) {
    fingerprints_[index] = configFileSchemaFingerprint(kStoredConfigFileKinds[index]);
  }
}

ConfigFileStore::~ConfigFileStore() {
  stop();
}

absl::Status ConfigFileStore::registerOnBus(robot::ipc::Bus* absl_nonnull bus) {
  RETURN_IF_ERROR(bus->subscribe<humanoid_mpc_msgs::ConfigFileSave>(
      ipc::topics::kOperatorConfigSave, robot::ipc::Delivery::kAll,
      [this](const humanoid_mpc_msgs::ConfigFileSave& message) { onConfigSave(message); }));
  bus_ = bus;
  return absl::OkStatus();
}

void ConfigFileStore::start() {
  if (started_) return;
  started_ = true;
  writer_ = std::thread([this]() { writerLoop(); });
}

void ConfigFileStore::stop() {
  std::deque<PendingSave> unanswered;
  {
    absl::MutexLock lock(mutex_);
    if (stopping_) return;
    stopping_ = true;
    unanswered.swap(queue_);
  }
  if (writer_.joinable()) writer_.join();
  for (const PendingSave& save : unanswered) {
    answer(answerOf(protoKind(save.kind), save.sequence, save.configPath, humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED,
                    "the robot process stopped before it wrote the file; save again once it runs", /*storedPath=*/""));
  }
}

ConfigFileStore::StoredFile* absl_nullable ConfigFileStore::fileIn(Config& config, msgs::ConfigFileKind kind) {
  switch (kind) {
    case msgs::ConfigFileKind::kTask:
      return &config.task;
    case msgs::ConfigFileKind::kReference:
      return &config.reference;
    case msgs::ConfigFileKind::kJointPdGains:
      return &config.pdGains;
    case msgs::ConfigFileKind::kUnspecified:
      return nullptr;
  }
  return nullptr;
}

const ConfigFileStore::StoredFile* absl_nullable ConfigFileStore::fileOf(msgs::ConfigFileKind kind) const {
  switch (kind) {
    case msgs::ConfigFileKind::kTask:
      return &config_.task;
    case msgs::ConfigFileKind::kReference:
      return &config_.reference;
    case msgs::ConfigFileKind::kJointPdGains:
      return &config_.pdGains;
    case msgs::ConfigFileKind::kUnspecified:
      return nullptr;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------------------------------------------------
// The IO thread
// ---------------------------------------------------------------------------------------------------------------------

absl::StatusOr<msgs::ConfigFileKind> ConfigFileStore::checkOnIoThread(const humanoid_mpc_msgs::ConfigFileSave& message) const {
  RETURN_IF_ERROR(checkPayloadSchema(message));
  msgs::ConfigFileKind kind = msgs::ConfigFileKind::kUnspecified;
  const std::optional<size_t> index =
      msgs::FromProto(message.kind(), &kind).ok() ? storedConfigFileKindIndex(kind) : std::optional<size_t>();
  const StoredFile* absl_nullable stored = index.has_value() ? fileOf(kind) : nullptr;
  if (!index.has_value() || stored == nullptr) {
    std::vector<absl::string_view> stores;
    for (const msgs::ConfigFileKind storedKind : kStoredConfigFileKinds) stores.push_back(configFileKindName(storedKind));
    return absl::InvalidArgumentError(absl::StrCat("the save names no file kind this robot stores (kind ", static_cast<int>(message.kind()),
                                                   "): it stores the ", absl::StrJoin(stores, ", the "), ""));
  }
  const StoredFile& file = *stored;
  if (!file.identity.empty() && message.config_path() != file.identity) {
    return absl::FailedPreconditionError(
        absl::StrCat("the save is the ", kindName(kind), " ", message.config_path().empty() ? "<no config_path>" : message.config_path(),
                     ", and this robot runs ", file.identity, ": a file of another configuration is not stored"));
  }
  if (!config_.robotName.empty() && message.robot_name() != config_.robotName) {
    return absl::FailedPreconditionError(
        absl::StrCat("the save is of the robot '", message.robot_name(), "', and this is '", config_.robotName, "'"));
  }
  if (message.schema_fingerprint() != fingerprints_[*index]) {
    return absl::FailedPreconditionError(absl::StrCat("the ", kindName(kind), " has the schema fingerprint '", message.schema_fingerprint(),
                                                      "', this build's is '", fingerprints_[*index],
                                                      "': it was sent by a build with another schema version"));
  }
  if (message.text().size() > config_.maxTextBytes) {
    return absl::InvalidArgumentError(absl::StrCat("the ", kindName(kind), " is ", message.text().size(), " bytes, more than the ",
                                                   config_.maxTextBytes, " a configuration file may have"));
  }
  return kind;
}

void ConfigFileStore::onConfigSave(const humanoid_mpc_msgs::ConfigFileSave& message) {
  received_.fetch_add(1);
  const absl::StatusOr<msgs::ConfigFileKind> kind = checkOnIoThread(message);
  if (!kind.ok()) {
    LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
        << "[ConfigFileStore] Refusing a save of " << message.config_path() << ": " << kind.status().message();
    answer(answerOf(message.kind(), message.sequence(), message.config_path(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED,
                    std::string(kind.status().message()), /*storedPath=*/""));
    return;
  }
  // checkOnIoThread() accepts the stored kinds only.
  const size_t index = storedConfigFileKindIndex(*kind).value_or(0);
  const StoredFile& file = *ABSL_DIE_IF_NULL(fileOf(*kind));
  if (file.path.empty()) {
    if (!config_.unavailableReason.empty()) {
      answer(answerOf(message.kind(), message.sequence(), message.config_path(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED,
                      config_.unavailableReason, /*storedPath=*/""));
    } else {
      // The file it reads in place, which the GUI compares with the laptop's: the save changed it only when the two are one.
      answer(answerOf(message.kind(), message.sequence(), message.config_path(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_NOT_STORED,
                      absl::StrCat("this robot process has no configuration store (--config_store_dir): it reads ", file.inPlacePath,
                                   " in place, which this save did not change"),
                      file.inPlacePath));
    }
    return;
  }
  std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> immediate;
  {
    absl::MutexLock lock(mutex_);
    if (std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> recorded = recordedAnswer(index, message.sequence()); recorded.has_value()) {
      repeated_.fetch_add(1);
      immediate = *std::move(recorded);
    } else if (isPending(*kind, message.sequence())) {
      repeated_.fetch_add(1);
      return;  // its answer comes when the writer is done with it
    } else if (stopping_) {
      immediate =
          answerOf(message.kind(), message.sequence(), message.config_path(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED,
                   "the robot process is stopping; save again once it runs", /*storedPath=*/"");
    } else if (queue_.size() >= config_.queueCapacity) {
      immediate =
          answerOf(message.kind(), message.sequence(), message.config_path(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED,
                   absl::StrCat("busy: ", queue_.size(), " saves are waiting to be written; save again"), /*storedPath=*/"");
    } else {
      queue_.push_back(
          PendingSave{.kind = *kind, .sequence = message.sequence(), .configPath = message.config_path(), .text = message.text()});
    }
  }
  if (immediate.has_value()) answer(*immediate);
}

std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> ConfigFileStore::recordedAnswer(size_t kind, uint64_t sequence) const {
  for (const humanoid_mpc_msgs::ConfigFileSaveStatus& status : answered_[kind]) {
    if (status.sequence() == sequence) return status;
  }
  return std::nullopt;
}

bool ConfigFileStore::isPending(msgs::ConfigFileKind kind, uint64_t sequence) const {
  if (inProgress_.has_value() && inProgress_->kind == kind && inProgress_->sequence == sequence) return true;
  for (const PendingSave& save : queue_) {
    if (save.kind == kind && save.sequence == sequence) return true;
  }
  return false;
}

void ConfigFileStore::answer(const humanoid_mpc_msgs::ConfigFileSaveStatus& status) {
  const Result result = status.result();
  if (result == humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED) {
    saved_.fetch_add(1);
  } else if (result == humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_UNCHANGED) {
    unchanged_.fetch_add(1);
  } else if (result == humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_NOT_STORED) {
    notStored_.fetch_add(1);
  } else if (result == humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED) {
    refused_.fetch_add(1);
  } else {
    failed_.fetch_add(1);
  }
  if (bus_ != nullptr) {
    const absl::Status published = bus_->publish(ipc::topics::kRobotConfigSaveStatus, status);
    if (!published.ok()) {
      LOG_EVERY_N_SEC(WARNING, kLogPeriodSeconds)
          << "[ConfigFileStore] Publishing robot/config_save_status failed: " << published.message();
    }
  }
  if (hooks_.statusObserver) hooks_.statusObserver(status);
}

// ---------------------------------------------------------------------------------------------------------------------
// The writer thread
// ---------------------------------------------------------------------------------------------------------------------

bool ConfigFileStore::writerHasWork() const {
  return stopping_ || !queue_.empty();
}

void ConfigFileStore::writerLoop() {
  while (true) {
    PendingSave save;
    {
      absl::MutexLock lock(mutex_);
      mutex_.Await(absl::Condition(this, &ConfigFileStore::writerHasWork));
      if (stopping_) return;
      save = std::move(queue_.front());
      queue_.pop_front();
      inProgress_ = save;
    }
    const humanoid_mpc_msgs::ConfigFileSaveStatus status = write(save);
    {
      absl::MutexLock lock(mutex_);
      std::deque<humanoid_mpc_msgs::ConfigFileSaveStatus>& answered = answered_[storedConfigFileKindIndex(save.kind).value_or(0)];
      answered.push_back(status);
      if (answered.size() > kRecordedAnswers) answered.pop_front();
      inProgress_.reset();
    }
    answer(status);
  }
}

absl::StatusOr<ConfigFileCandidate> ConfigFileStore::parse(const PendingSave& save) const {
  ConfigFileCandidate candidate;
  candidate.kind = save.kind;
  candidate.source = absl::StrCat(save.configPath.empty() ? kindName(save.kind) : absl::string_view(save.configPath), " (",
                                  ipc::topics::kOperatorConfigSave, ")");
  switch (save.kind) {
    case msgs::ConfigFileKind::kTask: {
      ASSIGN_OR_RETURN(candidate.task, parseTaskFile(save.text, candidate.source));
      return candidate;
    }
    case msgs::ConfigFileKind::kReference: {
      ASSIGN_OR_RETURN(candidate.reference, parseReferenceFile(save.text, candidate.source));
      return candidate;
    }
    case msgs::ConfigFileKind::kJointPdGains: {
      ASSIGN_OR_RETURN(candidate.pdGains, parseJointPdGainsFile(save.text, candidate.source));
      return candidate;
    }
    case msgs::ConfigFileKind::kUnspecified:
      break;
  }
  return absl::InvalidArgumentError("the save names no file kind");
}

humanoid_mpc_msgs::ConfigFileSaveStatus ConfigFileStore::write(const PendingSave& save) const {
  const humanoid_mpc_msgs::ConfigFileKind kind = protoKind(save.kind);
  // The IO thread queues the stored kinds only.
  const std::string& path = ABSL_DIE_IF_NULL(fileOf(save.kind))->path;
  const absl::StatusOr<ConfigFileCandidate> candidate = parse(save);
  absl::Status valid = candidate.status();
  if (candidate.ok() && hooks_.validateConfigFile) valid = hooks_.validateConfigFile(*candidate);
  if (!valid.ok()) {
    LOG(WARNING) << "[ConfigFileStore] Refusing the saved " << kindName(save.kind) << " " << save.configPath << ": " << valid.message();
    return answerOf(kind, save.sequence, save.configPath, humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED,
                    absl::StrCat("the robot would not start with it: ", valid.message()), /*storedPath=*/"");
  }
  const absl::StatusOr<std::string> current = readFileBytes(path);
  if (!current.ok() && !absl::IsNotFound(current.status())) {
    return answerOf(kind, save.sequence, save.configPath, humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED,
                    std::string(current.status().message()), path);
  }
  if (current.ok() && *current == save.text) {
    return answerOf(kind, save.sequence, save.configPath, humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_UNCHANGED,
                    absl::StrCat(path, " holds this text already"), path);
  }
  absl::Status backedUp = current.ok() ? writeFileAtomically(absl::StrCat(path, kBackupSuffix), *current) : absl::OkStatus();
  if (absl::IsDataLoss(backedUp)) {
    // The backup is there, only perhaps not after a power loss: the save goes on.
    LOG(WARNING) << "[ConfigFileStore] " << backedUp.message();
    backedUp = absl::OkStatus();
  }
  const absl::Status written = backedUp.ok() ? writeFileAtomically(path, save.text) : backedUp;
  if (absl::IsDataLoss(written)) {
    // Replaced: the watchers apply it, and the answer says so, with the doubt.
    LOG(WARNING) << "[ConfigFileStore] Stored the saved " << kindName(save.kind) << ", but not durably: " << written.message();
    return answerOf(kind, save.sequence, save.configPath, humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED,
                    absl::StrCat("stored in ", path, ", but not yet durable (a power loss may undo it): ", written.message()), path);
  }
  if (!written.ok()) {
    LOG(ERROR) << "[ConfigFileStore] Storing the saved " << kindName(save.kind) << " failed: " << written.message();
    return answerOf(kind, save.sequence, save.configPath, humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED,
                    absl::StrCat("writing failed, the robot's copy is the one it had: ", written.message()), path);
  }
  LOG(INFO) << "[ConfigFileStore] Stored the saved " << kindName(save.kind) << " in " << path << " (the copy it replaced: " << path
            << kBackupSuffix << ").";
  return answerOf(kind, save.sequence, save.configPath, humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED,
                  absl::StrCat("stored in ", path), path);
}

ConfigFileStore::Statistics ConfigFileStore::statistics() const {
  Statistics statistics;
  statistics.received = received_.load();
  statistics.saved = saved_.load();
  statistics.unchanged = unchanged_.load();
  statistics.notStored = notStored_.load();
  statistics.refused = refused_.load();
  statistics.failed = failed_.load();
  statistics.repeated = repeated_.load();
  return statistics;
}

}  // namespace ocs2::humanoid
