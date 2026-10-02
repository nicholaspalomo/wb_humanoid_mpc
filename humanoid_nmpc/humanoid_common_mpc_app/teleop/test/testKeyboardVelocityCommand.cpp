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

#include <gtest/gtest.h>

#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/teleop/LineReader.h"
#include "humanoid_common_mpc_app/teleop/VelocityCommandRepeater.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "robot_core/ResourcePaths.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NodeEndpoint.h"

/*
 * The keyboard velocity teleoperation: the command limits it reads from reference.yaml (the ones the MPC scales the
 * command back with), a typed line, the normalized message, the line reader that does not block a shutdown, and the
 * republication of the latest command at the topic's rate.
 */

namespace ocs2::humanoid::teleop {
namespace {

std::string writeTemporaryFile(const std::string& name, const std::string& content) {
  const char* directory = std::getenv("TEST_TMPDIR");
  const std::string path = absl::StrCat(directory != nullptr ? directory : "/tmp", "/", name);
  std::ofstream(path) << content;
  return path;
}

TEST(KeyboardCommandLimits, EveryShippedReferenceFileHasPositiveLimits) {
  for (const char* file : {"robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml",
                           "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/command/reference.yaml",
                           "robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.yaml",
                           "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml",
                           "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/command/reference.yaml"}) {
    const absl::StatusOr<std::string> path = robot::resolveResourcePath(file);
    ASSERT_TRUE(path.ok()) << path.status();
    const absl::StatusOr<KeyboardCommandLimits> limits = loadKeyboardCommandLimits(*path);
    ASSERT_TRUE(limits.ok()) << limits.status();
    EXPECT_TRUE((limits->limits.array() > 0.0).all()) << file;
    EXPECT_GT(limits->defaultBaseHeight, 0.0) << file;
  }
}

TEST(KeyboardCommandLimits, RefusesAMissingKeyALimitThatIsNotPositiveAndAMissingFile) {
  const std::string complete =
      "maxDisplacementVelocityX: 0.6\nmaxDisplacementVelocityY: 0.3\nmaxDeltaPelvisHeight: 0.2\nmaxRotationVelocity: 0.8\n"
      "defaultBaseHeight: 0.75\n";
  const absl::StatusOr<KeyboardCommandLimits> loaded = loadKeyboardCommandLimits(writeTemporaryFile("complete.yaml", complete));
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ(loaded->limits(0), 0.6);
  EXPECT_EQ(loaded->limits(3), 0.8);
  EXPECT_EQ(loaded->defaultBaseHeight, 0.75);

  const absl::StatusOr<KeyboardCommandLimits> missing =
      loadKeyboardCommandLimits(writeTemporaryFile("missing.yaml", "maxDisplacementVelocityX: 0.6\n"));
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(missing.status().message(), "maxDisplacementVelocityY")) << missing.status();

  std::string zero = complete;
  zero.replace(zero.find("0.3"), 3, "0.0");
  const absl::StatusOr<KeyboardCommandLimits> notPositive = loadKeyboardCommandLimits(writeTemporaryFile("zero.yaml", zero));
  EXPECT_EQ(notPositive.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(notPositive.status().message(), "maxDisplacementVelocityY")) << notPositive.status();

  EXPECT_EQ(loadKeyboardCommandLimits("/nonexistent/reference.yaml").status().code(), absl::StatusCode::kNotFound);
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

std::unique_ptr<robot::ipc::Bus> loopbackBus(const std::string& name) {
  robot::ipc::BusOptions options;
  options.nodeName = name;
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = name, .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  EXPECT_TRUE(bus.ok()) << bus.status();
  return *std::move(bus);
}

TEST(VelocityCommandRepeater, PublishesNothingBeforeTheFirstCommandAndThenTheLatestOneAtTheTopicsRate) {
  std::unique_ptr<robot::ipc::Bus> teleop = loopbackBus("teleop");
  std::unique_ptr<robot::ipc::Bus> mpc = loopbackBus("mpc");
  ASSERT_TRUE(mpc->connect(teleop->boundEndpoint()).ok());
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
  absl::StatusOr<std::unique_ptr<VelocityCommandRepeater>> repeater = VelocityCommandRepeater::Create(*teleop);
  ASSERT_TRUE(repeater.ok()) << repeater.status();
  ASSERT_TRUE(teleop->start().ok());
  ASSERT_TRUE(mpc->start().ok());

  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_EQ((*repeater)->published(), 0);
  EXPECT_EQ(receivedCount.load(), 0);

  humanoid_mpc_msgs::WalkingVelocityCommand command;
  command.set_linear_velocity_x(0.5);
  command.set_desired_pelvis_height(0.8);
  (*repeater)->setCommand(command);
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (receivedCount.load() < 3 && absl::Now() < deadline) absl::SleepFor(absl::Milliseconds(5));
  ASSERT_GE(receivedCount.load(), 3);
  {
    absl::MutexLock lock(mutex);
    ASSERT_TRUE(received.has_value());
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

}  // namespace
}  // namespace ocs2::humanoid::teleop
