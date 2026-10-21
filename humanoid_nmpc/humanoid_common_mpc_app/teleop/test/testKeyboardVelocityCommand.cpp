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

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/teleop/LineReader.h"
#include "humanoid_common_mpc_app/teleop/VelocityCommandRepeater.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "robot_core/ResourcePaths.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_runtime/robot_ipc/test/TestNetwork.h"

/*
 * The keyboard velocity teleoperation: the command limits it reads from reference.textproto (the ones the MPC scales
 * the command back with), a typed line, the normalized message, the line reader that does not block a shutdown, and the
 * republication of the latest command at the topic's rate.
 */

namespace ocs2::humanoid::teleop {
namespace {

// The header of a reference file, which names its schema.
constexpr char kReferenceHeader[] = "# proto-message: humanoid_mpc_config.ReferenceFile\n";

/** The reference file `content`, on the line below kReferenceHeader, written to `name`; its path. */
std::string writeTemporaryFile(const std::string& name, const std::string& content) {
  const char* absl_nullable directory = std::getenv("TEST_TMPDIR");
  const std::string path = absl::StrCat(directory != nullptr ? directory : "/tmp", "/", name);
  std::ofstream(path) << kReferenceHeader << content;
  return path;
}

TEST(KeyboardCommandLimits, EveryShippedReferenceFileHasPositiveLimits) {
  for (const char* absl_nonnull file : {"robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto",
                                        "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/command/reference.textproto",
                                        "robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.textproto",
                                        "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto",
                                        "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/command/reference.textproto"}) {
    const absl::StatusOr<std::string> path = robot::resolveResourcePath(file);
    ASSERT_TRUE(path.ok()) << path.status();
    const absl::StatusOr<KeyboardCommandLimits> limits = loadKeyboardCommandLimits(*path);
    ASSERT_TRUE(limits.ok()) << limits.status();
    EXPECT_TRUE((limits->limits.array() > 0.0).all()) << file;
    EXPECT_GT(limits->defaultBaseHeight, 0.0) << file;
  }
}

TEST(KeyboardCommandLimits, RefusesAMissingFieldALimitThatIsNotPositiveAnUnknownFieldAndAMissingFile) {
  const std::string complete =
      "max_displacement_velocity_x: 0.6\nmax_displacement_velocity_y: 0.3\nmax_delta_pelvis_height: 0.2\n"
      "max_rotation_velocity: 0.8\ndefault_base_height: 0.75\n";
  const absl::StatusOr<KeyboardCommandLimits> loaded = loadKeyboardCommandLimits(writeTemporaryFile("complete.textproto", complete));
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ(loaded->limits(0), 0.6);
  EXPECT_EQ(loaded->limits(3), 0.8);
  EXPECT_EQ(loaded->defaultBaseHeight, 0.75);

  const std::string missingFile = writeTemporaryFile("missing.textproto", "max_displacement_velocity_x: 0.6\n");
  const absl::StatusOr<KeyboardCommandLimits> missing = loadKeyboardCommandLimits(missingFile);
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StartsWith(missing.status().message(), missingFile)) << missing.status();
  EXPECT_TRUE(absl::StrContains(missing.status().message(), "max_displacement_velocity_y")) << missing.status();

  std::string zero = complete;
  zero.replace(zero.find("0.3"), 3, "0.0");
  const absl::StatusOr<KeyboardCommandLimits> notPositive = loadKeyboardCommandLimits(writeTemporaryFile("zero.textproto", zero));
  EXPECT_EQ(notPositive.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(notPositive.status().message(), "max_displacement_velocity_y")) << notPositive.status();

  // The file is read strictly: a field the schema does not have is refused where it is written.
  const std::string unknownFile = writeTemporaryFile("unknown.textproto", absl::StrCat(complete, "max_speed: 2\n"));
  const absl::StatusOr<KeyboardCommandLimits> unknown = loadKeyboardCommandLimits(unknownFile);
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), absl::StrCat(unknownFile, ":7:"))) << unknown.status();

  EXPECT_EQ(loadKeyboardCommandLimits("/nonexistent/reference.textproto").status().code(), absl::StatusCode::kNotFound);
}

TEST(ParseKeyboardCommandLine, ReadsUpToFourNumbersAndZerosTheRest) {
  absl::StatusOr<vector4_t> command = parseKeyboardCommandLine("0.3 -0.1");
  ASSERT_TRUE(command.ok()) << command.status();
  EXPECT_EQ(*command, vector4_t(0.3, -0.1, 0.0, 0.0));

  command = parseKeyboardCommandLine("  0.1\t0.2 0.05 0.4 9 9\n");
  ASSERT_TRUE(command.ok()) << command.status();
  EXPECT_EQ(*command, vector4_t(0.1, 0.2, 0.05, 0.4));

  command = parseKeyboardCommandLine("");
  ASSERT_TRUE(command.ok()) << command.status();
  EXPECT_TRUE(command->isZero(0.0));
}

TEST(ParseKeyboardCommandLine, RefusesAWordThatIsNotAFiniteNumber) {
  const absl::StatusOr<vector4_t> word = parseKeyboardCommandLine("0.3 fast");
  EXPECT_EQ(word.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(word.status().message(), "fast")) << word.status();
  EXPECT_FALSE(parseKeyboardCommandLine("nan").ok());
  EXPECT_FALSE(parseKeyboardCommandLine("0 inf").ok());
}

TEST(KeyboardCommandToMessage, NormalizesByTheLimitsAndClampsToThem) {
  KeyboardCommandLimits limits;
  limits.limits = vector4_t(0.5, 0.25, 0.2, 1.0);
  limits.defaultBaseHeight = 0.7;

  const humanoid_mpc_msgs::WalkingVelocityCommand inside = keyboardCommandToMessage(vector4_t(0.25, -0.125, -0.1, 0.5), limits);
  EXPECT_DOUBLE_EQ(inside.linear_velocity_x(), 0.5);
  EXPECT_DOUBLE_EQ(inside.linear_velocity_y(), -0.5);
  EXPECT_DOUBLE_EQ(inside.desired_pelvis_height(), 0.6);
  EXPECT_DOUBLE_EQ(inside.angular_velocity_z(), 0.5);

  const humanoid_mpc_msgs::WalkingVelocityCommand beyond = keyboardCommandToMessage(vector4_t(3.0, -3.0, 1.0, -9.0), limits);
  EXPECT_DOUBLE_EQ(beyond.linear_velocity_x(), 1.0);
  EXPECT_DOUBLE_EQ(beyond.linear_velocity_y(), -1.0);
  EXPECT_DOUBLE_EQ(beyond.desired_pelvis_height(), 0.9);
  EXPECT_DOUBLE_EQ(beyond.angular_velocity_z(), -1.0);
}

TEST(LineReader, ReadsLinesAndTheLastOneWithoutABreak) {
  int pipeEnds[2];
  ASSERT_EQ(::pipe(pipeEnds), 0);
  const std::string input = "0.1 0\n\n0.2 0.1";
  ASSERT_EQ(::write(pipeEnds[1], input.data(), input.size()), static_cast<ssize_t>(input.size()));
  ::close(pipeEnds[1]);
  LineReader reader(pipeEnds[0], absl::Milliseconds(10));
  const std::function<bool()> never = []() { return false; };
  EXPECT_EQ(reader.readLine(never), std::optional<std::string>("0.1 0"));
  EXPECT_EQ(reader.readLine(never), std::optional<std::string>(""));
  EXPECT_EQ(reader.readLine(never), std::optional<std::string>("0.2 0.1"));
  EXPECT_EQ(reader.readLine(never), std::nullopt);
  ::close(pipeEnds[0]);
}

TEST(LineReader, GivesUpWaitingOnceTheStopPredicateHolds) {
  int pipeEnds[2];
  ASSERT_EQ(::pipe(pipeEnds), 0);
  LineReader reader(pipeEnds[0], absl::Milliseconds(10));
  const absl::Time stopAt = absl::Now() + absl::Milliseconds(100);
  const absl::Time start = absl::Now();
  EXPECT_EQ(reader.readLine([stopAt]() { return absl::Now() > stopAt; }), std::nullopt);
  EXPECT_LT(absl::Now() - start, absl::Seconds(2));
  ::close(pipeEnds[0]);
  ::close(pipeEnds[1]);
}

// A complete reference file whose max_displacement_velocity_x is `maxVelocityX` (default_base_height 0.75 m).
std::string referenceWithMaxVelocityX(const std::string& maxVelocityX) {
  return absl::StrCat("max_displacement_velocity_x: ", maxVelocityX,
                      "\nmax_displacement_velocity_y: 0.3\nmax_delta_pelvis_height: 0.2\nmax_rotation_velocity: 0.8\n"
                      "default_base_height: 0.75\n");
}

// Rewrites `path` with the reference file `content` below kReferenceHeader and moves its modification time `secondsLater` seconds past the
// previous one, so that a file system of coarse timestamps still shows the change.
void rewriteLater(const std::string& path, const std::string& content, int secondsLater) {
  std::error_code error;
  const std::filesystem::file_time_type before = std::filesystem::last_write_time(path, error);
  ASSERT_FALSE(error) << error.message();
  std::ofstream(path, std::ios::trunc) << kReferenceHeader << content;
  std::filesystem::last_write_time(path, before + std::chrono::seconds(secondsLater), error);
  ASSERT_FALSE(error) << error.message();
}

// The commands a subscriber on its own loopback bus receives on operator/walking_velocity_command.
class CommandRecorder {
 public:
  void record(const humanoid_mpc_msgs::WalkingVelocityCommand& command) {
    absl::MutexLock lock(mutex_);
    last_ = command;
    ++count_;
  }
  std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> last() const {
    absl::MutexLock lock(mutex_);
    return last_;
  }
  int count() const {
    absl::MutexLock lock(mutex_);
    return count_;
  }
  // Waits up to `timeout` until the last command satisfies `done`.
  bool waitFor(const std::function<bool(const humanoid_mpc_msgs::WalkingVelocityCommand&)>& done, absl::Duration timeout) const {
    const absl::Time deadline = absl::Now() + timeout;
    while (absl::Now() < deadline) {
      const std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> command = last();
      if (command.has_value() && done(*command)) return true;
      absl::SleepFor(absl::Milliseconds(5));
    }
    return false;
  }

 private:
  mutable absl::Mutex mutex_;
  std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> last_ ABSL_GUARDED_BY(mutex_);
  int count_ ABSL_GUARDED_BY(mutex_) = 0;
};

std::unique_ptr<robot::ipc::Bus> loopbackBus(const std::string& name) {
  robot::ipc::BusOptions options;
  options.nodeName = name;
  options.network = robot::ipc::isolatedTestNetwork({name});
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  EXPECT_TRUE(bus.ok()) << bus.status();
  return *std::move(bus);
}

TEST(VelocityCommandRepeater, PublishesNothingBeforeTheFirstCommandAndThenTheLatestOneAtTheTopicsRate) {
  std::unique_ptr<robot::ipc::Bus> teleop = loopbackBus("teleop");
  std::unique_ptr<robot::ipc::Bus> mpc = loopbackBus("mpc");
  absl::Mutex mutex;
  std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> received;
  std::atomic<int> receivedCount{0};
  ASSERT_TRUE(mpc->subscribe<humanoid_mpc_msgs::WalkingVelocityCommand>(ipc::topics::kOperatorWalkingVelocityCommand,
                                                                        robot::ipc::Delivery::kAll,
                                                                        [&](const humanoid_mpc_msgs::WalkingVelocityCommand& command) {
                                                                          absl::MutexLock lock(mutex);
                                                                          received = command;
                                                                          receivedCount.fetch_add(1);
                                                                        })
                  .ok());
  const std::string referenceFile = writeTemporaryFile("repeater_rate.textproto", referenceWithMaxVelocityX("1.0"));
  const absl::StatusOr<KeyboardCommandLimits> limits = loadKeyboardCommandLimits(referenceFile);
  ASSERT_TRUE(limits.ok()) << limits.status();
  absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>> repeater = VelocityCommandRepeater::Create(*teleop, referenceFile, *limits);
  ASSERT_TRUE(repeater.ok()) << repeater.status();
  ASSERT_TRUE(teleop->start().ok());
  ASSERT_TRUE(mpc->start().ok());

  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_EQ((*repeater)->published(), 0);
  EXPECT_EQ(receivedCount.load(), 0);

  (*repeater)->setCommand(vector4_t(0.5, 0.0, 0.0, 0.0), *limits);
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (receivedCount.load() < 3 && absl::Now() < deadline) absl::SleepFor(absl::Milliseconds(5));
  ASSERT_GE(receivedCount.load(), 3);
  {
    absl::MutexLock lock(mutex);
    // An if, not ASSERT_TRUE: clang-tidy sees the check (bugprone-unchecked-optional-access) only through an if.
    if (!received.has_value()) FAIL() << "no command";
    EXPECT_EQ(received->linear_velocity_x(), 0.5);
  }
  // Repeated at about 25 Hz, not as fast as the IO thread can.
  const uint64_t before = (*repeater)->published();
  absl::SleepFor(absl::Milliseconds(400));
  const uint64_t inPeriod = (*repeater)->published() - before;
  EXPECT_GE(inPeriod, 2);
  EXPECT_LE(inPeriod, 15);
  teleop->stop();
  mpc->stop();
}

// A pair of loopback buses, the teleop one with a repeater on `referenceFile`, the MPC one recording what arrives.
struct RepeaterRig {
  std::unique_ptr<robot::ipc::Bus> teleop;
  std::unique_ptr<robot::ipc::Bus> mpc;
  std::unique_ptr<CommandRecorder> recorder;
  std::unique_ptr<VelocityCommandRepeater> repeater;
};

RepeaterRig startRepeaterRig(const std::string& referenceFile, const KeyboardCommandLimits& limits, absl::Duration limitsCheckPeriod) {
  RepeaterRig rig{
      .teleop = loopbackBus("teleop"), .mpc = loopbackBus("mpc"), .recorder = std::make_unique<CommandRecorder>(), .repeater = nullptr};
  CommandRecorder* absl_nonnull recorder = rig.recorder.get();
  EXPECT_TRUE(rig.mpc
                  ->subscribe<humanoid_mpc_msgs::WalkingVelocityCommand>(
                      ipc::topics::kOperatorWalkingVelocityCommand, robot::ipc::Delivery::kAll,
                      [recorder](const humanoid_mpc_msgs::WalkingVelocityCommand& command) { recorder->record(command); })
                  .ok());
  absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>> repeater =
      VelocityCommandRepeater::Create(*rig.teleop, referenceFile, limits, kVelocityCommandPeriod, limitsCheckPeriod);
  EXPECT_TRUE(repeater.ok()) << repeater.status();
  if (repeater.ok()) rig.repeater = *std::move(repeater);
  EXPECT_TRUE(rig.teleop->start().ok());
  EXPECT_TRUE(rig.mpc->start().ok());
  return rig;
}

TEST(VelocityCommandRepeater, ASavedChangeOfTheLimitsKeepsTheTypedSpeed) {
  const std::string referenceFile = writeTemporaryFile("repeater_limits.textproto", referenceWithMaxVelocityX("1.0"));
  const absl::StatusOr<KeyboardCommandLimits> limits = loadKeyboardCommandLimits(referenceFile);
  ASSERT_TRUE(limits.ok()) << limits.status();
  RepeaterRig rig = startRepeaterRig(referenceFile, *limits, /*limitsCheckPeriod=*/absl::Milliseconds(80));
  ASSERT_NE(rig.repeater, nullptr);

  // 0.5 m/s at a limit of 1.0 m/s goes out normalized as 0.5.
  rig.repeater->setCommand(vector4_t(0.5, 0.0, 0.0, 0.0), *limits);
  ASSERT_TRUE(rig.recorder->waitFor([](const humanoid_mpc_msgs::WalkingVelocityCommand& c) { return c.linear_velocity_x() == 0.5; },
                                    absl::Seconds(10)));

  // The MPC now scales by 2.0 m/s: the repeater sends 0.25, so the robot still walks at the 0.5 m/s the terminal showed.
  rewriteLater(referenceFile, referenceWithMaxVelocityX("2.0"), /*secondsLater=*/2);
  EXPECT_TRUE(rig.recorder->waitFor([](const humanoid_mpc_msgs::WalkingVelocityCommand& c) { return c.linear_velocity_x() == 0.25; },
                                    absl::Seconds(10)));
  EXPECT_GE(rig.repeater->limitsReloads(), 1u);

  // A lowered limit clamps the held command: 0.5 m/s at a limit of 0.4 m/s is the full 1.0.
  rewriteLater(referenceFile, referenceWithMaxVelocityX("0.4"), /*secondsLater=*/4);
  EXPECT_TRUE(rig.recorder->waitFor([](const humanoid_mpc_msgs::WalkingVelocityCommand& c) { return c.linear_velocity_x() == 1.0; },
                                    absl::Seconds(10)));

  // A file that no longer reads keeps the previous limits.
  rewriteLater(referenceFile, "max_displacement_velocity_x: \n", /*secondsLater=*/6);
  absl::SleepFor(absl::Milliseconds(400));
  const std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> kept = rig.recorder->last();
  if (!kept.has_value()) FAIL() << "no command";
  EXPECT_EQ(kept->linear_velocity_x(), 1.0);
  rig.teleop->stop();
  rig.mpc->stop();
}

TEST(VelocityCommandRepeater, EndsWithTheStopCommandKeepingTheHeight) {
  const std::string referenceFile = writeTemporaryFile("repeater_stop.textproto", referenceWithMaxVelocityX("1.0"));
  const absl::StatusOr<KeyboardCommandLimits> limits = loadKeyboardCommandLimits(referenceFile);
  ASSERT_TRUE(limits.ok()) << limits.status();
  RepeaterRig rig = startRepeaterRig(referenceFile, *limits, kCommandLimitsCheckPeriod);
  ASSERT_NE(rig.repeater, nullptr);

  // Nothing was ever commanded: nothing to stop, and nothing is published.
  EXPECT_FALSE(rig.repeater->publishStopCommand(/*timeout=*/absl::Milliseconds(100), /*gracePeriod=*/absl::ZeroDuration()));
  EXPECT_EQ(rig.repeater->published(), 0u);

  rig.repeater->setCommand(vector4_t(0.5, 0.1, 0.05, 0.2), *limits);
  ASSERT_TRUE(rig.recorder->waitFor([](const humanoid_mpc_msgs::WalkingVelocityCommand& c) { return c.linear_velocity_x() == 0.5; },
                                    absl::Seconds(10)));
  EXPECT_TRUE(rig.repeater->publishStopCommand());
  EXPECT_TRUE(rig.recorder->waitFor(
      [](const humanoid_mpc_msgs::WalkingVelocityCommand& c) {
        return c.linear_velocity_x() == 0.0 && c.linear_velocity_y() == 0.0 && c.angular_velocity_z() == 0.0;
      },
      absl::Seconds(10)));
  const std::optional<humanoid_mpc_msgs::WalkingVelocityCommand> stop = rig.recorder->last();
  if (!stop.has_value()) FAIL() << "no command";
  EXPECT_DOUBLE_EQ(stop->desired_pelvis_height(), 0.75 + 0.05);
  rig.teleop->stop();
  rig.mpc->stop();
}

TEST(VelocityCommandRepeater, ABusThatIsNotRunningNeverSendsTheStopCommand) {
  const std::string referenceFile = writeTemporaryFile("repeater_stopped.textproto", referenceWithMaxVelocityX("1.0"));
  const absl::StatusOr<KeyboardCommandLimits> limits = loadKeyboardCommandLimits(referenceFile);
  ASSERT_TRUE(limits.ok()) << limits.status();
  std::unique_ptr<robot::ipc::Bus> teleop = loopbackBus("teleop");
  absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>> repeater = VelocityCommandRepeater::Create(*teleop, referenceFile, *limits);
  ASSERT_TRUE(repeater.ok()) << repeater.status();
  (*repeater)->setCommand(vector4_t(0.5, 0.0, 0.0, 0.0), *limits);
  EXPECT_FALSE((*repeater)->publishStopCommand(/*timeout=*/absl::Milliseconds(200), /*gracePeriod=*/absl::ZeroDuration()));
}

TEST(VelocityCommandRepeater, RefusesAPeriodThatIsNotPositive) {
  std::unique_ptr<robot::ipc::Bus> teleop = loopbackBus("teleop");
  EXPECT_EQ(VelocityCommandRepeater::Create(*teleop, "unused", KeyboardCommandLimits{}, absl::ZeroDuration()).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::teleop
