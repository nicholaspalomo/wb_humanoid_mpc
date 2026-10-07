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

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc_app/robot/ConfigFileStore.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.pb.h"
#include "humanoid_mpc_msgs/config_file_kind.pb.h"
#include "humanoid_mpc_msgs/config_file_save.pb.h"
#include "humanoid_mpc_msgs/config_file_save_status.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/ChildProcess.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/LoopbackNetwork.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/ScriptedOperator.h"
#include "nproto/Textproto.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NetworkConfig.h"

/*
 * The GUI's two-copy Save end to end, on the robot's side: humanoid_centroidal_mpc_robot is started as its own process
 * with a configuration store in the test's scratch space (the DRC Atlas files are its seeds), and the test sends saves
 * as push_robot_config does, from the bus node config_push. A task file and a PD gains file are stored byte for byte;
 * a save of another robot, of another configuration or one that does not parse is refused with the stored copy
 * untouched; after SIGTERM a new start runs the saved task file (its telemetry rate, a fifth of the bundle's, shows on
 * robot/state); a stored task file the robot would not start with, and a start that did not confirm its boot, make the
 * next start fall back to the bundled files, leaving the stored copy as .rejected. The edits are made on what the
 * shipped files hold, so that retuning them changes nothing here.
 */

namespace ocs2::humanoid {
namespace {

constexpr char kRobotBinary[] = "humanoid_nmpc/humanoid_centroidal_mpc_app/humanoid_centroidal_mpc_robot";
constexpr char kAtlasTask[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kAtlasReference[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto";
constexpr char kAtlasGains[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.textproto";
constexpr char kAtlasUrdf[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kAtlasScene[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
/** model_settings.robot_name of the Atlas task file. */
constexpr char kRobotName[] = "atlas";

using test_support::ChildProcess;
using test_support::waitFor;

std::string readFile(const std::string& path) {
  std::ifstream file(path);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, absl::string_view contents) {
  std::ofstream(path, std::ios::trunc) << contents;
}

bool exists(const std::string& path) {
  std::error_code error;
  return std::filesystem::exists(path, error);
}

/**
 * The test's side of the bus as push_robot_config's node: it sends saves and keeps the robot's answers, and counts the
 * robot/state samples it receives. Thread-safe.
 */
class SaveClient {
 public:
  explicit SaveClient(const robot::ipc::NetworkConfig& network) : bus_(test_support::createBus(network, "config_push")) {
    CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::ConfigFileSaveStatus>(ipc::topics::kRobotConfigSaveStatus, robot::ipc::Delivery::kAll,
                                                                      [this](const humanoid_mpc_msgs::ConfigFileSaveStatus& status) {
                                                                        absl::MutexLock lock(mutex_);
                                                                        answers_[status.sequence()] = status;
                                                                      }));
    CHECK_OK(bus_->subscribe<humanoid_mpc_msgs::RobotStateSample>(ipc::topics::kRobotState, robot::ipc::Delivery::kAll,
                                                                  [this](const humanoid_mpc_msgs::RobotStateSample& sample) {
                                                                    absl::MutexLock lock(mutex_);
                                                                    sampleTimes_.push_back(sample.time());
                                                                  }));
    CHECK_OK(bus_->start());
  }
  ~SaveClient() { bus_->stop(); }
  SaveClient(const SaveClient&) = delete;
  SaveClient& operator=(const SaveClient&) = delete;

  /** Sends `save` every 0.5 s, as push_robot_config does, until the robot answers it; nullopt after 20 s. */
  std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> send(const humanoid_mpc_msgs::ConfigFileSave& save) {
    std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer;
    const absl::Time end = absl::Now() + absl::Seconds(20);
    while (!answer.has_value() && absl::Now() < end) {
      bus_->publish(ipc::topics::kOperatorConfigSave, save).IgnoreError();  // a lost message is sent again below
      waitFor(
          [&]() {
            absl::MutexLock lock(mutex_);
            const std::map<uint64_t, humanoid_mpc_msgs::ConfigFileSaveStatus>::const_iterator found = answers_.find(save.sequence());
            if (found != answers_.end()) answer = found->second;
            return answer.has_value();
          },
          absl::Milliseconds(500));
    }
    return answer;
  }

  /** [Hz] The rate of the robot/state samples received during `window`, by their robot time; 0 with fewer than two. */
  double telemetryRate(absl::Duration window) {
    {
      absl::MutexLock lock(mutex_);
      sampleTimes_.clear();
    }
    absl::SleepFor(window);
    absl::MutexLock lock(mutex_);
    if (sampleTimes_.size() < 2 || !(sampleTimes_.back() > sampleTimes_.front())) return 0.0;
    return static_cast<double>(sampleTimes_.size() - 1) / (sampleTimes_.back() - sampleTimes_.front());
  }

  /** Whether robot/state samples arrive: the robot runs its loop. */
  bool receivesTelemetry() {
    {
      absl::MutexLock lock(mutex_);
      sampleTimes_.clear();
    }
    return waitFor(
        [&]() {
          absl::MutexLock lock(mutex_);
          return sampleTimes_.size() > 10;
        },
        absl::Seconds(60));
  }

 private:
  std::unique_ptr<robot::ipc::Bus> bus_;
  absl::Mutex mutex_;
  std::map<uint64_t, humanoid_mpc_msgs::ConfigFileSaveStatus> answers_ ABSL_GUARDED_BY(mutex_);
  std::vector<double> sampleTimes_ ABSL_GUARDED_BY(mutex_);
};

/** The robot binary on the Atlas files, with the store `storeDirectory`, on the network file `networkFile`. */
std::unique_ptr<ChildProcess> startRobot(const std::string& networkFile, const std::string& storeDirectory) {
  return std::make_unique<ChildProcess>(std::vector<std::string>{
      kRobotBinary, "--robot_name=drc_atlas", absl::StrCat("--task_file=", kAtlasTask), absl::StrCat("--reference_file=", kAtlasReference),
      absl::StrCat("--urdf_file=", kAtlasUrdf), absl::StrCat("--mjcf_file=", kAtlasScene), absl::StrCat("--network_config=", networkFile),
      "--headless", "--realtime_cores=none", "--backend_cores=none", absl::StrCat("--config_store_dir=", storeDirectory),
      "--config_seed=when_bundle_changes"});
}

/** A save of `text` as the file of `kind` of the Atlas configuration, stamped as the GUI stamps it. */
humanoid_mpc_msgs::ConfigFileSave saveOf(msgs::ConfigFileKind kind,
                                         const std::string& bundledFile,
                                         const std::string& text,
                                         uint64_t sequence) {
  humanoid_mpc_msgs::ConfigFileSave save;
  humanoid_mpc_msgs::ConfigFileKind protoKind = humanoid_mpc_msgs::CONFIG_FILE_KIND_UNSPECIFIED;
  msgs::ToProto(kind, &protoKind);
  save.set_kind(protoKind);
  save.set_robot_name(kRobotName);
  save.set_config_path(configFileIdentity(bundledFile));
  save.set_schema_fingerprint(configFileSchemaFingerprint(kind));
  save.set_text(text);
  save.set_sequence(sequence);
  return save;
}

/**
 * `text` with the value of the first line that sets `field` ("<field>: <value>", indented or not) replaced by `value`,
 * as an editor writes it; empty when no line sets it. The tests edit what the shipped files hold, whatever it is tuned to.
 */
std::string withFieldValue(absl::string_view text, absl::string_view field, absl::string_view value) {
  std::vector<std::string> lines = absl::StrSplit(text, '\n');
  for (std::string& line : lines) {
    const absl::string_view stripped = absl::StripLeadingAsciiWhitespace(line);
    if (!absl::StartsWith(stripped, absl::StrCat(field, ":"))) continue;
    line = absl::StrCat(line.substr(0, line.size() - stripped.size()), field, ": ", value);
    return absl::StrJoin(lines, "\n");
  }
  return std::string();
}

/** The textproto `text` parsed as `Message`; a test failure and the default message when it does not parse. */
template <typename Message>
Message parsed(const std::string& text) {
  absl::StatusOr<Message> message = nproto::ParseTextproto<Message>(text, "the test's file");
  EXPECT_TRUE(message.ok()) << message.status();
  return message.ok() ? *std::move(message) : Message();
}

/** The result of `answer`; RESULT_UNSPECIFIED when there is none. */
humanoid_mpc_msgs::ConfigFileSaveStatus::Result resultOf(std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> answer) {
  return answer.has_value() ? answer->result() : humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_UNSPECIFIED;
}

TEST(ConfigSaveEndToEnd, SavesReachTheRobotsStoreAndARestartRunsThemOrFallsBackToTheBundle) {
  const std::filesystem::path scratch = std::getenv("TEST_TMPDIR");
  const std::string networkFile = (scratch / "network.textproto").string();
  const absl::StatusOr<robot::ipc::NetworkConfig> network =
      test_support::writeLoopbackNetworkFile(networkFile, {"robot", "operator", "config_push"});
  ASSERT_TRUE(network.ok()) << network.status();
  const std::string store = (scratch / "robot_config" / "drc_atlas").string();
  const std::string storedTask = store + "/mpc/task.textproto";
  const std::string storedGains = store + "/controller/joint_pd_gains.textproto";
  const std::string bundledTask = readFile(kAtlasTask);
  // The telemetry rate, a start-up field the robot/state samples show: the bundle's, and a fifth of it in the save.
  const humanoid_mpc_config::TaskFile bundledTaskMessage = parsed<humanoid_mpc_config::TaskFile>(bundledTask);
  ASSERT_TRUE(bundledTaskMessage.has_telemetry_frequency()) << "the Atlas task file sets no telemetry rate to change";
  const double bundledRate = bundledTaskMessage.telemetry_frequency();
  const double savedRate = bundledRate / 5.0;
  const std::string savedTask = withFieldValue(bundledTask, "telemetry_frequency", absl::StrCat(savedRate));
  ASSERT_FALSE(savedTask.empty());
  ASSERT_NE(savedTask, bundledTask);
  ASSERT_EQ(parsed<humanoid_mpc_config::TaskFile>(savedTask).telemetry_frequency(), savedRate);
  SaveClient client(*network);

  // The first start seeds the store from the bundle, and runs it: robot/state at the bundle's rate.
  std::unique_ptr<ChildProcess> robot = startRobot(networkFile, store);
  {
    test_support::ScriptedOperator remoteControl(*network);
    remoteControl.start();
    ASSERT_TRUE(waitFor([&]() { return remoteControl.fsmState().has_value() || !robot->running(); }, absl::Seconds(60)));
    ASSERT_TRUE(robot->running()) << "the robot process exited on start-up";
    EXPECT_EQ(remoteControl.fsmState().value_or(humanoid_mpc_msgs::FsmState()).mode(), "ZERO_TORQUE");
  }
  EXPECT_EQ(readFile(storedTask), bundledTask);
  EXPECT_NEAR(client.telemetryRate(absl::Seconds(2)), bundledRate, 0.2 * bundledRate);

  // Saves of the task file and of the PD gains are stored byte for byte; the copies they replaced are kept.
  const std::optional<humanoid_mpc_msgs::ConfigFileSaveStatus> task =
      client.send(saveOf(msgs::ConfigFileKind::kTask, kAtlasTask, savedTask, /*sequence=*/1));
  if (!task.has_value()) FAIL() << "the robot did not answer the save";
  ASSERT_EQ(task->result(), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED) << task->message();
  EXPECT_EQ(task->stored_path(), storedTask);
  EXPECT_EQ(readFile(storedTask), savedTask);
  EXPECT_EQ(readFile(storedTask + ".bak"), bundledTask);
  const std::string bundledGains = readFile(kAtlasGains);
  const double bundledKp = parsed<humanoid_mpc_config::JointPdGainsFile>(bundledGains).default_gains().kp();
  const std::string savedGains = withFieldValue(bundledGains, "kp", absl::StrCat(0.8 * bundledKp));
  ASSERT_FALSE(savedGains.empty());
  ASSERT_NE(savedGains, bundledGains);
  ASSERT_EQ(parsed<humanoid_mpc_config::JointPdGainsFile>(savedGains).default_gains().kp(), 0.8 * bundledKp)
      << "the edit is not the default gains' kp";
  EXPECT_EQ(resultOf(client.send(saveOf(msgs::ConfigFileKind::kJointPdGains, kAtlasGains, savedGains, /*sequence=*/2))),
            humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_SAVED);
  EXPECT_EQ(readFile(storedGains), savedGains);
  // The telemetry rate is a start-up field: the running robot keeps its rate until it starts again.
  EXPECT_NEAR(client.telemetryRate(absl::Seconds(1)), bundledRate, 0.2 * bundledRate);

  // Refused: another robot, another configuration, a file that does not parse; the stored copy is the one saved above.
  humanoid_mpc_msgs::ConfigFileSave otherRobot = saveOf(msgs::ConfigFileKind::kTask, kAtlasTask, bundledTask, /*sequence=*/3);
  otherRobot.set_robot_name("r1");
  humanoid_mpc_msgs::ConfigFileSave otherConfiguration = saveOf(msgs::ConfigFileKind::kTask, kAtlasTask, bundledTask, /*sequence=*/4);
  otherConfiguration.set_config_path("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const humanoid_mpc_msgs::ConfigFileSave malformed =
      saveOf(msgs::ConfigFileKind::kTask, kAtlasTask, "telemetry_frequency: {\n", /*sequence=*/5);
  for (const humanoid_mpc_msgs::ConfigFileSave& refused : {otherRobot, otherConfiguration, malformed}) {
    EXPECT_EQ(resultOf(client.send(refused)), humanoid_mpc_msgs::ConfigFileSaveStatus::RESULT_REFUSED) << refused.sequence();
  }
  EXPECT_EQ(readFile(storedTask), savedTask);

  // A clean stop confirms the boot.
  EXPECT_EQ(robot->terminate(absl::Seconds(20)), 0) << "the robot process did not end cleanly on SIGTERM";
  EXPECT_FALSE(exists(store + "/.booting"));

  // The next start runs the saved task file: robot/state at its rate.
  robot = startRobot(networkFile, store);
  ASSERT_TRUE(client.receivesTelemetry()) << "the robot did not start on its stored files";
  EXPECT_NEAR(client.telemetryRate(absl::Seconds(2)), savedRate, 0.25 * savedRate);
  EXPECT_EQ(readFile(storedTask), savedTask) << "a restart without a deploy keeps the robot's save";
  EXPECT_EQ(robot->terminate(absl::Seconds(20)), 0);

  // A stored task file the robot would not start with, planted bypassing the store's checks: the start falls back to
  // the bundled files, and the planted copy is kept as .rejected.
  const std::string planted = withFieldValue(savedTask, "contact_estimator", "\"no_such_estimator\"");
  ASSERT_FALSE(planted.empty()) << "the Atlas task file names no contact_estimator to change";
  ASSERT_NE(planted, savedTask);
  writeFile(storedTask, planted);
  robot = startRobot(networkFile, store);
  ASSERT_TRUE(client.receivesTelemetry()) << "the robot did not start on the bundled files";
  EXPECT_NEAR(client.telemetryRate(absl::Seconds(2)), bundledRate, 0.2 * bundledRate);
  EXPECT_EQ(readFile(storedTask), bundledTask);
  EXPECT_EQ(readFile(storedTask + ".rejected"), planted);
  EXPECT_EQ(robot->terminate(absl::Seconds(20)), 0);

  // A start that died before it confirmed its boot (its .booting is still there): the next one runs the bundle too.
  writeFile(storedTask, savedTask);
  writeFile(store + "/.booting", "when_bundle_changes\n");
  robot = startRobot(networkFile, store);
  ASSERT_TRUE(client.receivesTelemetry()) << "the robot did not start after a boot that was not confirmed";
  EXPECT_NEAR(client.telemetryRate(absl::Seconds(2)), bundledRate, 0.2 * bundledRate);
  EXPECT_EQ(readFile(storedTask + ".rejected"), savedTask);
  EXPECT_EQ(robot->terminate(absl::Seconds(20)), 0);
}

}  // namespace
}  // namespace ocs2::humanoid
