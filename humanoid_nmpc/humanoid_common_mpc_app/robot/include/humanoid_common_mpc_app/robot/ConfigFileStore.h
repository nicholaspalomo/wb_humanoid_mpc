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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/StoredConfigFileKinds.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"
#include "humanoid_mpc_msgs/config_file_save.pb.h"
#include "humanoid_mpc_msgs/config_file_save_status.pb.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid {

/** A configuration file a save delivered, parsed strictly: the file of `kind` is set, the other two are not. */
struct ConfigFileCandidate {
  msgs::ConfigFileKind kind = msgs::ConfigFileKind::kUnspecified;
  /** Names the text in errors: "<config_path> (operator/config_save)". */
  std::string source;
  std::optional<mpc_config::TaskFile> task;
  std::optional<mpc_config::ReferenceFile> reference;
  std::optional<mpc_config::JointPdGainsFile> pdGains;
};

/**
 * The robot's half of the tuning GUI's two-copy Save (humanoid_nmpc/docs/distributed_runtime/README.md, "Saving the
 * configuration"): takes a ConfigFileSave of operator/config_save, checks it as the robot process would check the file at
 * start-up, writes it into the robot's persistent copy (RobotConfigDirectory) and answers on robot/config_save_status.
 *
 * THE IO THREAD (onConfigSave(), the bus's handler) does only what is cheap and touches no file: a payload of another
 * schema version (unknown fields), an unknown kind, another configuration (config_path is compared with the kind's
 * identity, never used as a path), another robot (robot_name), another schema fingerprint of the kind's file message, or
 * a text over Config::maxTextBytes is refused at once; a repeat of a save already answered gets that answer again, and
 * one still queued is dropped (its answer comes). Without a store every other save is answered NOT_STORED, naming the
 * file the robot reads in place (which the save did not change), and with a store that cannot be written FAILED. The rest is queued for the
 * writer thread; a full queue answers REFUSED ("busy").
 *
 * THE WRITER THREAD (start() to stop()) parses the text strictly (parseTaskFile() and its siblings), runs
 * Hooks::validateConfigFile - the checks start-up runs (checkConfigFileCandidate()) - and then answers UNCHANGED when
 * the stored copy holds the text already, or keeps the stored copy as `<file>.bak` and writes the text atomically
 * (writeFileAtomically()), answering SAVED with the stored path - saying so when the file was replaced but its
 * directory could not be synced (a power loss may undo it) - or FAILED with the copy untouched. The robot process's
 * watchers then see the new file, as they see an edit. The realtime thread never sees the store.
 *
 * Thread-safe as described; statistics() from any thread.
 */
class ConfigFileStore {
 public:
  /** One of the files a save may write. */
  struct StoredFile {
    /** The robot's persistent copy (RobotConfigDirectory::storedPath()); empty: none. */
    std::string path;
    /** configFileIdentity() of the bundled file: a save that names another config_path is refused. Empty: not checked. */
    std::string identity;
    /**
     * The file the robot reads of this kind (RobotConfigDirectory::readPath()): without a store, the bundled file it
     * reads in place, which a NOT_STORED answer names so that the GUI can tell whether the save changed it.
     */
    std::string inPlacePath;
  };

  struct Config {
    StoredFile task;
    StoredFile reference;
    StoredFile pdGains;
    /** The running robot (ModelSettings::robotName): a save of another robot_name is refused. Empty: not checked. */
    std::string robotName;
    /**
     * Why the store cannot be written (RobotConfigDirectory::storeError()); not empty: every save that passes the cheap
     * checks is answered FAILED with it.
     */
    std::string unavailableReason;
    /** The saves waiting for the writer thread; one more is refused as busy. */
    size_t queueCapacity = 4;
    /** [bytes] The largest text accepted. */
    size_t maxTextBytes = 1024 * 1024;
  };

  struct Hooks {
    /**
     * On the writer thread: OK when the robot would start with `candidate` in place of its kind's file; the refusal
     * otherwise. Empty: a text that parses is accepted.
     */
    std::function<absl::Status(const ConfigFileCandidate& candidate)> validateConfigFile;
    /** Every answer, after it was published; on the IO or the writer thread. For tests and tools; may be empty. */
    std::function<void(const humanoid_mpc_msgs::ConfigFileSaveStatus& status)> statusObserver;
  };

  /** What the store answered so far; every counter only grows. */
  struct Statistics {
    uint64_t received = 0;
    uint64_t saved = 0;
    uint64_t unchanged = 0;
    uint64_t notStored = 0;
    uint64_t refused = 0;
    uint64_t failed = 0;
    /** Repeats of a save already answered or still queued: answered again, or dropped. */
    uint64_t repeated = 0;
  };

  /** InvalidArgument for a zero queue capacity or text limit. */
  static absl::StatusOr<std::unique_ptr<ConfigFileStore>> Create(Config config, Hooks hooks);

  /** stop(). */
  ~ConfigFileStore();
  ConfigFileStore(const ConfigFileStore&) = delete;
  ConfigFileStore& operator=(const ConfigFileStore&) = delete;

  /**
   * Subscribes operator/config_save on `bus` (not running yet) and publishes the answers on robot/config_save_status
   * from then on; keeps `bus`, which must outlive the store, or stop() must have run before it goes.
   */
  absl::Status registerOnBus(robot::ipc::Bus* absl_nonnull bus);

  /** The entry of `config` for the file of `kind`; nullptr for a kind the robot does not store. */
  static StoredFile* absl_nullable fileIn(Config& config, msgs::ConfigFileKind kind);

  /** Starts the writer thread. Once. */
  void start();

  /**
   * Finishes the save the writer is on, answers the queued ones FAILED and joins the writer thread. Idempotent; call it
   * before the bus stops, so that the answers go out.
   */
  void stop();

  /** The handler of operator/config_save: the bus's IO thread (and tests). */
  void onConfigSave(const humanoid_mpc_msgs::ConfigFileSave& message);

  Statistics statistics() const;

 private:
  /** Answers kept per kind for repeats. */
  static constexpr size_t kRecordedAnswers = 8;

  /** A save the IO thread accepted for the writer. */
  struct PendingSave {
    msgs::ConfigFileKind kind = msgs::ConfigFileKind::kUnspecified;
    uint64_t sequence = 0;
    std::string configPath;
    std::string text;
  };

  ConfigFileStore(Config config, Hooks hooks);

  /** The file of `kind`; nullptr for a kind the robot does not store. */
  const StoredFile* absl_nullable fileOf(msgs::ConfigFileKind kind) const;
  /** The cheap checks of the IO thread: the message's kind, or the refusal. */
  absl::StatusOr<msgs::ConfigFileKind> checkOnIoThread(const humanoid_mpc_msgs::ConfigFileSave& message) const;
  /** The answer the writer gave the save (kind, sequence), if it has written it. */
  std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> recordedAnswer(size_t kind, uint64_t sequence) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  /** Whether the save (kind, sequence) is queued or being written. */
  bool isPending(msgs::ConfigFileKind kind, uint64_t sequence) const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void writerLoop();
  bool writerHasWork() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  /** The writer's answer to `save`. */
  humanoid_mpc_msgs::ConfigFileSaveStatus write(const PendingSave& save) const;
  absl::StatusOr<ConfigFileCandidate> parse(const PendingSave& save) const;
  /** Counts `status`, publishes it and hands it to the observer. */
  void answer(const humanoid_mpc_msgs::ConfigFileSaveStatus& status);

  const Config config_;
  const Hooks hooks_;
  /** The schema fingerprints of the stored kinds' file messages in this build, by index in kStoredConfigFileKinds. */
  std::array<std::string, kNumStoredConfigFileKinds> fingerprints_;
  robot::ipc::Bus* absl_nullable bus_ = nullptr;

  mutable absl::Mutex mutex_;
  std::deque<PendingSave> queue_ ABSL_GUARDED_BY(mutex_);
  /** The save the writer is on; nullopt when it waits. Its kind and sequence. */
  std::optional<PendingSave> inProgress_ ABSL_GUARDED_BY(mutex_);
  /** The answers of the writer, by index in kStoredConfigFileKinds. */
  std::array<std::deque<humanoid_mpc_msgs::ConfigFileSaveStatus>, kNumStoredConfigFileKinds> answered_ ABSL_GUARDED_BY(mutex_);
  bool stopping_ ABSL_GUARDED_BY(mutex_) = false;
  std::thread writer_;
  bool started_ = false;

  std::atomic<uint64_t> received_{0};
  std::atomic<uint64_t> saved_{0};
  std::atomic<uint64_t> unchanged_{0};
  std::atomic<uint64_t> notStored_{0};
  std::atomic<uint64_t> refused_{0};
  std::atomic<uint64_t> failed_{0};
  std::atomic<uint64_t> repeated_{0};
};

/** The store's configuration of `directory`, for the robot `robotName` (ModelSettings::robotName). */
ConfigFileStore::Config configFileStoreConfig(const RobotConfigDirectory& directory, std::string robotName);

/** The schema fingerprint of the file message of `kind` (TaskFile, ReferenceFile, JointPdGainsFile); "" for kUnspecified. */
std::string configFileSchemaFingerprint(msgs::ConfigFileKind kind);

}  // namespace ocs2::humanoid
