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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/robot/AtomicFileWrite.h"
#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/RobotConfigurationCheck.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"
#include "humanoid_mpc_msgs/config_file_kind.pb.h"
#include "humanoid_mpc_msgs/config_file_save.pb.h"
#include "humanoid_mpc_msgs/config_file_save_status.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotState.h"

/*
 * The configuration store (ConfigFileStore.h), the robot's half of the GUI's two-copy Save, on files in the test's
 * scratch space: a save is checked on the IO thread and on the writer thread as start-up would check the file, then
 * written atomically with the replaced copy kept, or answered UNCHANGED, REFUSED, FAILED or NOT_STORED without touching
 * the stored copy; a full queue is busy, and a repeated save is answered without being written again.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::HasSubstr;
using Result = humanoid_mpc_msgs::ConfigFileSaveStatus::Result;

constexpr char kTaskIdentity[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kPdGainsIdentity[] = "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.textproto";
constexpr char kReferenceIdentity[] = "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";
constexpr char kRobotName[] = "g1";

constexpr char kStoredTask[] = "model_settings { robot_name: \"g1\" }\ncontact_estimator: \"robot_state\"\n";
constexpr char kSavedTask[] = "model_settings { robot_name: \"g1\" }\ncontact_estimator: \"always_in_contact\"\n";
constexpr char kStoredGains[] = "default_gains { kp: 100.0 kd: 2.0 }\n";
constexpr char kStoredReference[] = "target_displacement_velocity: 0.5\n";

/** The test's scratch directory `name`, empty. */
std::string scratchDirectory(absl::string_view name) {
  const std::filesystem::path directory = std::filesystem::path(std::getenv("TEST_TMPDIR")) / std::string(name);
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  std::filesystem::create_directories(directory, error);
  return directory.string();
}

/** Writes `contents` at `path`, creating its directory. */
void writeFile(const std::string& path, absl::string_view contents) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream(path, std::ios::trunc) << contents;
}

std::string readFile(const std::string& path) {
  const absl::StatusOr<std::string> bytes = readFileBytes(path);
  return bytes.ok() ? *bytes : std::string("<unreadable>");
}

std::filesystem::file_time_type writeTimeOf(const std::string& path) {
  std::error_code error;
  return std::filesystem::last_write_time(path, error);
}

/** A store's files, at their layout below `directory`. */
RobotConfigDirectory::Files writeStore(const std::string& directory) {
  RobotConfigDirectory::Files files{.taskFile = directory + "/mpc/task.textproto",
                                    .referenceFile = directory + "/command/reference.textproto",
                                    .pdGainsFile = directory + "/controller/joint_pd_gains.textproto"};
  writeFile(files.taskFile, kStoredTask);
  writeFile(files.referenceFile, kStoredReference);
  writeFile(files.pdGainsFile, kStoredGains);
  return files;
}

/** The store's configuration over `files`, for the whole-body G1. */
ConfigFileStore::Config storeConfig(const RobotConfigDirectory::Files& files) {
  ConfigFileStore::Config config;
  config.task = {.path = files.taskFile, .identity = kTaskIdentity};
  config.reference = {.path = files.referenceFile, .identity = kReferenceIdentity};
  config.pdGains = {.path = files.pdGainsFile, .identity = kPdGainsIdentity};
  config.robotName = kRobotName;
  config.maxTextBytes = 4096;
  return config;
}

/** A save of the task file as the GUI sends it. */
humanoid_mpc_msgs::ConfigFileSave taskSave(absl::string_view text, uint64_t sequence) {
  humanoid_mpc_msgs::ConfigFileSave save;
  save.set_kind(humanoid_mpc_msgs::CONFIG_FILE_KIND_TASK);
  save.set_robot_name(kRobotName);
  save.set_config_path(kTaskIdentity);
  save.set_schema_fingerprint(configFileSchemaFingerprint(msgs::ConfigFileKind::kTask));
  save.set_text(std::string(text));
  save.set_sequence(sequence);
  return save;
}

/** What the robot checks a save against: no contact estimator of a simulator, the mujoco backend's options. */
RobotConfigurationCheckContext checkContext(const robot::model::ContactEstimatorRegistry& registry) {
  RobotConfigurationCheckContext context;
  context.robotName = kRobotName;
  context.contactEstimators = &registry;
  context.backendName = "mujoco";
  context.backendOptions.mjcfFile = robot_test::kAtlasScene;
  context.backendOptions.initialState.emplace(robot_test::atlasDescription());
  return context;
}

/** The store's answers, as an observer collects them, with the thread each was given on. Thread-safe. */
class Answers {
 public:
  void add(const humanoid_mpc_msgs::ConfigFileSaveStatus& status) {
    absl::MutexLock lock(mutex_);
    answers_.push_back(status);
    threads_.push_back(std::this_thread::get_id());
  }

  /** The first answer to `sequence`, once it came within 10 s. */
  std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> waitFor(uint64_t sequence, size_t occurrence = 1) const {
    std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> found;
    robot_test::waitFor([&]() {
      absl::MutexLock lock(mutex_);
      size_t seen = 0;
      for (const humanoid_mpc_msgs::ConfigFileSaveStatus& status : answers_) {
        if (status.sequence() == sequence && ++seen == occurrence) found = status;
      }
      return found.has_value();
    });
    return found;
  }

  size_t count() const {
    absl::MutexLock lock(mutex_);
    return answers_.size();
  }

  std::thread::id threadOf(uint64_t sequence) const {
    absl::MutexLock lock(mutex_);
    for (size_t i = 0; i < answers_.size(); ++i) {
      if (answers_[i].sequence() == sequence) return threads_[i];
    }
    return std::thread::id();
  }

 private:
  mutable absl::Mutex mutex_;
  std::vector<humanoid_mpc_msgs::ConfigFileSaveStatus> answers_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::thread::id> threads_ ABSL_GUARDED_BY(mutex_);
};

/** A started store with its answers collected, and the validator `validate` (empty: parse only). */
std::unique_ptr<ConfigFileStore> startedStore(ConfigFileStore::Config config,
                                              Answers& answers,
                                              std::function<absl::Status(const ConfigFileCandidate&)> validate) {
  ConfigFileStore::Hooks hooks;
  hooks.validateConfigFile = std::move(validate);
  hooks.statusObserver = [&answers](const humanoid_mpc_msgs::ConfigFileSaveStatus& status) { answers.add(status); };
  absl::StatusOr<std::unique_ptr<ConfigFileStore>> store = ConfigFileStore::Create(std::move(config), std::move(hooks));
  EXPECT_TRUE(store.ok()) << store.status();
  if (!store.ok()) return nullptr;
  (*store)->start();
  return *std::move(store);
}

/** The result of `status`; RESULT_UNSPECIFIED when there is none. */
Result resultOf(std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> status) {
  return status.has_value() ? status->result() : humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_UNSPECIFIED;
}

TEST(ConfigFileStore, ASaveIsWrittenOnTheWriterThreadWithTheCopyItReplacedKept) {
  const RobotConfigDirectory::Files files = writeStore(scratchDirectory("saved"));
  Answers answers;
  robot::model::ContactEstimatorRegistry registry;
  const RobotConfigurationCheckContext context = checkContext(registry);
  std::atomic<std::thread::id> validatedOn;
  const std::unique_ptr<ConfigFileStore> store = startedStore(storeConfig(files), answers, [&](const ConfigFileCandidate& candidate) {
    validatedOn.store(std::this_thread::get_id());
    return checkConfigFileCandidate(files, candidate, context);
  });
  ASSERT_NE(store, nullptr);
  const std::filesystem::file_time_type before = writeTimeOf(files.taskFile);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/1));
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer = answers.waitFor(/*sequence=*/1);
  if (!answer.has_value()) FAIL() << "no answer";
  ASSERT_EQ(answer->result(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED) << answer->message();
  EXPECT_EQ(answer->stored_path(), files.taskFile);
  EXPECT_EQ(answer->config_path(), kTaskIdentity);
  EXPECT_EQ(answer->kind(), humanoid_mpc_msgs::CONFIG_FILE_KIND_TASK);
  EXPECT_EQ(readFile(files.taskFile), kSavedTask) << "the stored copy is the GUI's bytes";
  EXPECT_EQ(readFile(files.taskFile + ".bak"), kStoredTask) << "the copy it replaced is kept";
  EXPECT_NE(writeTimeOf(files.taskFile), before);
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(std::filesystem::path(files.taskFile).parent_path())) {
    EXPECT_FALSE(absl::StrContains(entry.path().filename().string(), ".tmp.")) << entry.path() << " was left behind";
  }
  // The IO thread (the test, here) only checked and queued: parsing, checking and writing ran on the writer thread.
  EXPECT_NE(validatedOn.load(), std::this_thread::get_id());
  EXPECT_EQ(answers.threadOf(/*sequence=*/1), validatedOn.load());
  EXPECT_EQ(store->statistics().saved, 1u);
}

TEST(ConfigFileStore, TheTextTheStoredCopyHoldsIsUnchangedAndNothingIsWritten) {
  const RobotConfigDirectory::Files files = writeStore(scratchDirectory("unchanged"));
  Answers answers;
  const std::unique_ptr<ConfigFileStore> store = startedStore(storeConfig(files), answers, /*validate=*/nullptr);
  ASSERT_NE(store, nullptr);
  const std::filesystem::file_time_type before = writeTimeOf(files.taskFile);
  store->onConfigSave(taskSave(kStoredTask, /*sequence=*/2));
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer = answers.waitFor(/*sequence=*/2);
  EXPECT_EQ(resultOf(answer), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_UNCHANGED);
  EXPECT_EQ(writeTimeOf(files.taskFile), before);
  EXPECT_FALSE(std::filesystem::exists(files.taskFile + ".bak"));
}

/** A save the robot refuses, and what the refusal names. */
struct Refusal {
  std::string name;
  humanoid_mpc_msgs::ConfigFileSave save;
  std::string expected;
};

std::vector<Refusal> refusals() {
  std::vector<Refusal> cases;
  cases.push_back({"malformed text", taskSave("model_settings {\n", /*sequence=*/10), "operator/config_save"});
  cases.push_back({"an unknown field in the text", taskSave("no_such_field: 1\n", /*sequence=*/11), "no_such_field"});
  humanoid_mpc_msgs::ConfigFileSave unknownField;
  // A varint field 103: a GUI built from a newer schema.
  EXPECT_TRUE(unknownField.ParseFromString(taskSave(kSavedTask, /*sequence=*/12).SerializeAsString() + std::string("\xb8\x06\x01", 3)));
  cases.push_back({"an unknown field in the payload", unknownField, "field 103"});
  humanoid_mpc_msgs::ConfigFileSave otherRobot = taskSave(kSavedTask, /*sequence=*/13);
  otherRobot.set_robot_name("atlas");
  cases.push_back({"another robot_name in the message", otherRobot, "atlas"});
  cases.push_back({"another robot_name in the text",
                   taskSave("model_settings { robot_name: \"atlas\" }\ncontact_estimator: \"robot_state\"\n", /*sequence=*/14),
                   "model_settings.robot_name"});
  humanoid_mpc_msgs::ConfigFileSave otherConfiguration = taskSave(kSavedTask, /*sequence=*/15);
  otherConfiguration.set_config_path("robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto");
  cases.push_back({"another configuration of the robot", otherConfiguration, "g1_centroidal_mpc"});
  humanoid_mpc_msgs::ConfigFileSave otherFingerprint = taskSave(kSavedTask, /*sequence=*/16);
  otherFingerprint.set_schema_fingerprint("0123456789abcdef");
  cases.push_back({"another schema fingerprint", otherFingerprint, "0123456789abcdef"});
  cases.push_back({"an oversize text", taskSave(std::string(5000, '#') + "\n" + kSavedTask, /*sequence=*/17), "4096"});
  cases.push_back({"a contact estimator the robot's backend did not register",
                   taskSave("model_settings { robot_name: \"g1\" }\ncontact_estimator: \"cheater_sim\"\n", /*sequence=*/18),
                   "contact_estimator"});
  cases.push_back(
      {"an unknown telemetry sink",
       taskSave("model_settings { robot_name: \"g1\" }\ncontact_estimator: \"robot_state\"\ntelemetry_sinks: \"carrier_pigeon\"\n",
                /*sequence=*/19),
       "telemetry_sinks"});
  cases.push_back(
      {"a simulator option the backend refuses",
       taskSave("model_settings { robot_name: \"g1\" }\ncontact_estimator: \"robot_state\"\ngantry_hold: \"rope\"\n", /*sequence=*/20),
       "gantry_hold"});
  humanoid_mpc_msgs::ConfigFileSave noKind = taskSave(kSavedTask, /*sequence=*/21);
  noKind.set_kind(humanoid_mpc_msgs::CONFIG_FILE_KIND_UNSPECIFIED);
  cases.push_back({"a save of no file kind", noKind, "no file kind"});
  return cases;
}

TEST(ConfigFileStore, ARefusedSaveLeavesTheStoredCopyAndItsTimeAlone) {
  const RobotConfigDirectory::Files files = writeStore(scratchDirectory("refused"));
  Answers answers;
  robot::model::ContactEstimatorRegistry registry;
  const RobotConfigurationCheckContext context = checkContext(registry);
  const std::unique_ptr<ConfigFileStore> store = startedStore(storeConfig(files), answers, [&](const ConfigFileCandidate& candidate) {
    return checkConfigFileCandidate(files, candidate, context);
  });
  ASSERT_NE(store, nullptr);
  const std::filesystem::file_time_type before = writeTimeOf(files.taskFile);
  for (const Refusal& refusal : refusals()) {
    store->onConfigSave(refusal.save);
    const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer = answers.waitFor(refusal.save.sequence());
    EXPECT_EQ(resultOf(answer), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED) << refusal.name;
    if (answer.has_value()) {
      EXPECT_THAT(answer->message(), HasSubstr(refusal.expected)) << refusal.name;
    }
    EXPECT_EQ(readFile(files.taskFile), kStoredTask) << refusal.name;
    EXPECT_EQ(writeTimeOf(files.taskFile), before) << refusal.name;
  }
  EXPECT_FALSE(std::filesystem::exists(files.taskFile + ".bak"));
  EXPECT_EQ(store->statistics().refused, refusals().size());
}

TEST(ConfigFileStore, AStoreThatCannotBeWrittenFailsWithTheCopyAsItWas) {
  const std::string directory = scratchDirectory("unwritable");
  // A regular file where the store's mpc/ directory belongs: no write below it succeeds, whoever runs the test.
  writeFile(directory + "/mpc", "not a directory\n");
  RobotConfigDirectory::Files files;
  files.taskFile = directory + "/mpc/task.textproto";
  Answers answers;
  const std::unique_ptr<ConfigFileStore> store = startedStore(storeConfig(files), answers, /*validate=*/nullptr);
  ASSERT_NE(store, nullptr);
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/30));
  EXPECT_EQ(resultOf(answers.waitFor(/*sequence=*/30)), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED);

  // A store the directory could not prepare answers FAILED at once, with its reason.
  ConfigFileStore::Config unavailable = storeConfig(files);
  unavailable.task.path.clear();
  unavailable.unavailableReason = "the configuration store /x cannot be used: read-only file system";
  Answers unavailableAnswers;
  const std::unique_ptr<ConfigFileStore> unavailableStore = startedStore(unavailable, unavailableAnswers, /*validate=*/nullptr);
  ASSERT_NE(unavailableStore, nullptr);
  unavailableStore->onConfigSave(taskSave(kSavedTask, /*sequence=*/31));
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer = unavailableAnswers.waitFor(/*sequence=*/31);
  EXPECT_EQ(resultOf(answer), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED);
  if (answer.has_value()) {
    EXPECT_THAT(answer->message(), HasSubstr("read-only file system"));
  }
}

TEST(ConfigFileStore, WithoutAStoreASaveIsNotStoredAndTheAnswerNamesTheFileReadInPlace) {
  ConfigFileStore::Config config;
  config.task.identity = kTaskIdentity;
  config.task.inPlacePath = "/bundle/robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
  config.robotName = kRobotName;
  Answers answers;
  const std::unique_ptr<ConfigFileStore> store = startedStore(config, answers, /*validate=*/nullptr);
  ASSERT_NE(store, nullptr);
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/40));
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer = answers.waitFor(/*sequence=*/40);
  EXPECT_EQ(resultOf(answer), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_NOT_STORED);
  if (answer.has_value()) {
    // The GUI compares it with the laptop's file: the save changed what the robot reads only when the two are one.
    EXPECT_EQ(answer->stored_path(), config.task.inPlacePath);
    EXPECT_THAT(answer->message(), HasSubstr("did not change"));
  }
  // The cheap checks still come first: another configuration is refused, not "not stored".
  humanoid_mpc_msgs::ConfigFileSave other = taskSave(kSavedTask, /*sequence=*/41);
  other.set_config_path("robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto");
  store->onConfigSave(other);
  EXPECT_EQ(resultOf(answers.waitFor(/*sequence=*/41)), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED);
}

/** A validator that holds the writer until it is released (or 30 s passed, so a failed test ends), counting its calls. */
class HeldValidator {
 public:
  absl::Status validate() {
    calls_.fetch_add(1);
    entered_.store(true);
    release_.WaitForNotificationWithTimeout(absl::Seconds(30));
    return absl::OkStatus();
  }
  void release() { release_.Notify(); }
  bool entered() const { return entered_.load(); }
  int calls() const { return calls_.load(); }

 private:
  absl::Notification release_;
  std::atomic<bool> entered_{false};
  std::atomic<int> calls_{0};
};

TEST(ConfigFileStore, AFullQueueIsBusyAndARepeatOfAQueuedSaveIsDropped) {
  const RobotConfigDirectory::Files files = writeStore(scratchDirectory("busy"));
  Answers answers;
  HeldValidator validator;
  const std::unique_ptr<ConfigFileStore> store =
      startedStore(storeConfig(files), answers, [&validator](const ConfigFileCandidate& /*candidate*/) { return validator.validate(); });
  ASSERT_NE(store, nullptr);
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/50));
  ASSERT_TRUE(robot_test::waitFor([&]() { return validator.entered(); }));
  // The writer holds 50; 51-54 fill the queue of four, and 55 finds it full.
  for (uint64_t sequence = 51; sequence <= 54; ++sequence) store->onConfigSave(taskSave(kSavedTask, sequence));
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/55));
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> busy = answers.waitFor(/*sequence=*/55);
  EXPECT_EQ(resultOf(busy), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED);
  if (busy.has_value()) {
    EXPECT_THAT(busy->message(), HasSubstr("busy"));
  }
  // A repeat of a save still waiting gets no second answer: the first one is coming.
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/52));
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/50));
  EXPECT_EQ(store->statistics().repeated, 2u);
  validator.release();
  for (uint64_t sequence = 50; sequence <= 54; ++sequence) EXPECT_TRUE(answers.waitFor(sequence).has_value()) << sequence;
  EXPECT_EQ(validator.calls(), 5);
  EXPECT_EQ(answers.count(), 6u);
}

TEST(ConfigFileStore, ARepeatOfAnAnsweredSaveGetsItsAnswerAgainAndIsNotWrittenAgain) {
  const RobotConfigDirectory::Files files = writeStore(scratchDirectory("repeated"));
  Answers answers;
  std::atomic<int> validations{0};
  const std::unique_ptr<ConfigFileStore> store =
      startedStore(storeConfig(files), answers, [&validations](const ConfigFileCandidate& /*candidate*/) {
        validations.fetch_add(1);
        return absl::OkStatus();
      });
  ASSERT_NE(store, nullptr);
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/60));
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> first = answers.waitFor(/*sequence=*/60);
  if (!first.has_value()) FAIL() << "no answer";
  ASSERT_EQ(first->result(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED);
  // Another save in between, then the repeat (its answer was lost): SAVED again, as it was, and nothing is written.
  const std::string later = std::string(kSavedTask) + "telemetry_frequency: 20\n";
  store->onConfigSave(taskSave(later, /*sequence=*/61));
  ASSERT_EQ(resultOf(answers.waitFor(/*sequence=*/61)), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED);
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/60));
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> repeated = answers.waitFor(/*sequence=*/60, /*occurrence=*/2);
  if (!repeated.has_value()) FAIL() << "the repeat was not answered";
  EXPECT_EQ(repeated->result(), first->result());
  EXPECT_EQ(repeated->message(), first->message());
  EXPECT_EQ(repeated->stored_path(), first->stored_path());
  EXPECT_EQ(validations.load(), 2);
  EXPECT_EQ(readFile(files.taskFile), later) << "the repeat did not write the older text back";
  EXPECT_EQ(store->statistics().repeated, 1u);
}

TEST(ConfigFileStore, StopFinishesTheSaveInProgressAndAnswersTheQueuedOnesFailed) {
  const RobotConfigDirectory::Files files = writeStore(scratchDirectory("stopped"));
  Answers answers;
  HeldValidator validator;
  const std::unique_ptr<ConfigFileStore> store =
      startedStore(storeConfig(files), answers, [&validator](const ConfigFileCandidate& /*candidate*/) { return validator.validate(); });
  ASSERT_NE(store, nullptr);
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/70));
  ASSERT_TRUE(robot_test::waitFor([&]() { return validator.entered(); }));
  store->onConfigSave(taskSave(kStoredGains, /*sequence=*/71));
  std::thread stopper([&store]() { store->stop(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  validator.release();
  stopper.join();
  EXPECT_EQ(resultOf(answers.waitFor(/*sequence=*/70)), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED);
  EXPECT_EQ(resultOf(answers.waitFor(/*sequence=*/71)), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED);
  // After stop() a save is answered at once.
  store->onConfigSave(taskSave(kSavedTask, /*sequence=*/72));
  EXPECT_EQ(resultOf(answers.waitFor(/*sequence=*/72)), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_FAILED);
}

TEST(ConfigFileStore, TheFingerprintOfEachKindIsItsFileMessages) {
  EXPECT_EQ(configFileSchemaFingerprint(msgs::ConfigFileKind::kTask).size(), 16u);
  EXPECT_NE(configFileSchemaFingerprint(msgs::ConfigFileKind::kTask), configFileSchemaFingerprint(msgs::ConfigFileKind::kReference));
  EXPECT_NE(configFileSchemaFingerprint(msgs::ConfigFileKind::kReference),
            configFileSchemaFingerprint(msgs::ConfigFileKind::kJointPdGains));
  EXPECT_TRUE(configFileSchemaFingerprint(msgs::ConfigFileKind::kUnspecified).empty());
}

}  // namespace
}  // namespace ocs2::humanoid
