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
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "VisualizationTestRobot.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_app/visualization/SceneContract.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/telemetry_series.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NodeEndpoint.h"

namespace ocs2::humanoid::visualization {
namespace {

constexpr absl::Duration kTimeout = absl::Seconds(20);

bool waitFor(const std::function<bool()>& condition, absl::Duration timeout = kTimeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (absl::Now() < deadline) {
    if (condition()) {
      return true;
    }
    absl::SleepFor(absl::Milliseconds(1));
  }
  return condition();
}

/** Records what the publisher publishes; can hold the telemetry until released, like a slow network. */
class RecordingSink {
 public:
  VisualizationPublisher::PublishFunction function() {
    return [this](absl::string_view topic, const google::protobuf::Message& message) { return publish(topic, message); };
  }

  /** Every telemetry publish waits for release() from now on. */
  void holdTelemetry() { holding_ = true; }
  void release() { released_.Notify(); }
  /** True while a telemetry publish is held. */
  bool holdingOne() const { return heldNow_.load(); }

  std::vector<double> telemetryTimes() const {
    absl::MutexLock lock(&mutex_);
    return telemetryTimes_;
  }
  std::vector<humanoid_mpc_msgs::VisualizationScene> scenes() const {
    absl::MutexLock lock(&mutex_);
    return scenes_;
  }
  size_t sceneCount() const {
    absl::MutexLock lock(&mutex_);
    return scenes_.size();
  }
  size_t telemetryCount() const {
    absl::MutexLock lock(&mutex_);
    return telemetryTimes_.size();
  }

 private:
  absl::Status publish(absl::string_view topic, const google::protobuf::Message& message) {
    if (topic == ipc::topics::kVizTelemetry) {
      if (holding_.load()) {
        heldNow_ = true;
        released_.WaitForNotification();
        heldNow_ = false;
      }
      absl::MutexLock lock(&mutex_);
      telemetryTimes_.push_back(static_cast<const humanoid_mpc_msgs::TelemetrySeries&>(message).time());
    } else if (topic == ipc::topics::kVizScene) {
      absl::MutexLock lock(&mutex_);
      scenes_.push_back(static_cast<const humanoid_mpc_msgs::VisualizationScene&>(message));
    } else {
      ADD_FAILURE() << "published on " << topic;
    }
    return absl::OkStatus();
  }

  std::atomic<bool> holding_{false};
  std::atomic<bool> heldNow_{false};
  absl::Notification released_;
  mutable absl::Mutex mutex_;
  std::vector<double> telemetryTimes_ ABSL_GUARDED_BY(mutex_);
  std::vector<humanoid_mpc_msgs::VisualizationScene> scenes_ ABSL_GUARDED_BY(mutex_);
};

class VisualizationPublisherTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { robot_ = test::TestRobot::load(test::g1CentroidalFiles(), test::Formulation::kCentroidal).release(); }
  static void TearDownTestSuite() {
    delete robot_;
    robot_ = nullptr;
  }

  std::unique_ptr<VisualizationPublisher> create(VisualizationPublisher::Options options = VisualizationPublisher::Options(),
                                                 const std::string& taskFile = "") {
    VisualizationModel model = robot_->model();
    if (!taskFile.empty()) {
      model.taskFile = taskFile;
    }
    absl::StatusOr<std::unique_ptr<VisualizationPublisher>> publisher = VisualizationPublisher::Create(model, sink_.function(), options);
    EXPECT_TRUE(publisher.ok()) << publisher.status();
    return publisher.ok() ? std::move(*publisher) : nullptr;
  }

  /** A copy of the G1's task file with rerunSceneFrequency set to `frequency`. */
  static std::string taskFileWithSceneFrequency(double frequency) {
    std::ifstream stream(robot_->taskFile());
    std::stringstream text;
    text << stream.rdbuf();
    const std::string original = text.str();
    const std::string replaced =
        absl::StrReplaceAll(original, {{"rerunSceneFrequency: 30", absl::StrCat("rerunSceneFrequency: ", frequency)}});
    EXPECT_NE(replaced, original) << "the G1's task file does not set rerunSceneFrequency: 30";
    const char* directory = std::getenv("TEST_TMPDIR");
    const std::string path = absl::StrCat(directory != nullptr ? directory : "/tmp", "/task_scene_frequency.yaml");
    std::ofstream(path) << replaced;
    return path;
  }

  static humanoid_mpc_msgs::RobotStateSample sampleAt(double time) {
    return robot_->robotState(time, vector3_t(0.0, 0.0, 0.75), vector3_t::Zero());
  }

  static test::TestRobot* robot_;
  RecordingSink sink_;
};

test::TestRobot* VisualizationPublisherTest::robot_ = nullptr;

TEST_F(VisualizationPublisherTest, EverySampleBecomesOneSeriesInOrder) {
  std::unique_ptr<VisualizationPublisher> publisher = create();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  std::vector<double> times;
  for (int sample = 0; sample < 20; ++sample) {
    times.push_back(0.01 * sample);
    ASSERT_TRUE(publisher->pushRobotState(sampleAt(times.back())));
  }
  ASSERT_TRUE(waitFor([&] { return sink_.telemetryCount() >= times.size(); }));
  publisher->stop();
  EXPECT_EQ(sink_.telemetryTimes(), times);
  const VisualizationPublisher::Statistics statistics = publisher->statistics();
  EXPECT_EQ(statistics.robotStatesQueued, times.size());
  EXPECT_EQ(statistics.robotStatesDropped, 0u);
  EXPECT_EQ(statistics.telemetryPublished, times.size());
  EXPECT_EQ(statistics.buildFailures, 0u);
}

TEST_F(VisualizationPublisherTest, ASlowPublisherMakesTheQueueDropSamplesAndNeverBlocksTheFeeders) {
  VisualizationPublisher::Options options;
  options.robotStateQueueCapacity = 8;
  std::unique_ptr<VisualizationPublisher> publisher = create(options);
  ASSERT_NE(publisher, nullptr);
  sink_.holdTelemetry();
  ASSERT_TRUE(publisher->start().ok());
  ASSERT_TRUE(publisher->pushRobotState(sampleAt(0.0)));
  ASSERT_TRUE(waitFor([&] { return sink_.holdingOne(); }));

  // The visualization thread is stuck publishing: every feeding call still returns at once, and the queue keeps at
  // most its capacity (one slot is the sample being published).
  CommandData command;
  PrimalSolution solution;
  robot_->makePolicy(/*startTime=*/0.0, /*nodes=*/11, /*normalForce=*/300.0, vector2_t::Zero(), &command, &solution);
  const SystemObservation observation = robot_->observation(/*time=*/0.0, ModeNumber::STANCE);
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  size_t accepted = 0;
  const size_t pushes = 100;
  for (size_t sample = 1; sample <= pushes; ++sample) {
    accepted += publisher->pushRobotState(sampleAt(0.01 * static_cast<double>(sample))) ? 1 : 0;
    publisher->setObservation(observation);
    publisher->setPolicy(command, solution);
  }
  const std::chrono::steady_clock::duration feeding = std::chrono::steady_clock::now() - start;
  EXPECT_LT(feeding, std::chrono::seconds(2));
  EXPECT_EQ(accepted, options.robotStateQueueCapacity - 1);
  EXPECT_EQ(publisher->statistics().robotStatesDropped, pushes - accepted);
  EXPECT_EQ(publisher->statistics().observationsSet, pushes);
  EXPECT_EQ(publisher->statistics().policiesSet, pushes);

  sink_.release();
  ASSERT_TRUE(waitFor([&] { return sink_.telemetryCount() >= 1 + accepted; }));
  absl::SleepFor(absl::Milliseconds(50));
  publisher->stop();
  EXPECT_EQ(sink_.telemetryCount(), 1 + accepted);
  EXPECT_EQ(publisher->statistics().telemetryPublished, 1 + accepted);
}

TEST_F(VisualizationPublisherTest, TheSceneIsPublishedAtMostAtItsFrequency) {
  const double frequency = 20.0;
  std::unique_ptr<VisualizationPublisher> publisher = create(VisualizationPublisher::Options(), taskFileWithSceneFrequency(frequency));
  ASSERT_NE(publisher, nullptr);
  EXPECT_EQ(publisher->config().sceneFrequency, frequency);
  ASSERT_TRUE(publisher->start().ok());
  SystemObservation observation = robot_->observation(/*time=*/0.0, ModeNumber::STANCE);
  publisher->setObservation(observation);
  ASSERT_TRUE(waitFor([&] { return sink_.sceneCount() >= 1; }));
  const size_t first = sink_.sceneCount();
  const absl::Duration window = absl::Seconds(1);
  const absl::Time end = absl::Now() + window;
  while (absl::Now() < end) {
    observation.time += 0.002;
    publisher->setObservation(observation);
    absl::SleepFor(absl::Milliseconds(2));
  }
  const size_t published = sink_.sceneCount() - first;
  publisher->stop();
  EXPECT_LE(published, static_cast<size_t>(frequency * absl::ToDoubleSeconds(window)) + 2);
  EXPECT_GE(published, 5u);
  for (const humanoid_mpc_msgs::VisualizationScene& scene : sink_.scenes()) {
    EXPECT_GE(scene.robots_size(), 1);
  }
}

TEST_F(VisualizationPublisherTest, NothingNewPublishesNoScene) {
  std::unique_ptr<VisualizationPublisher> publisher = create();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_EQ(sink_.sceneCount(), 0u) << "a scene without anything to draw";
  publisher->setObservation(robot_->observation(/*time=*/0.0, ModeNumber::STANCE));
  ASSERT_TRUE(waitFor([&] { return sink_.sceneCount() >= 1; }));
  absl::SleepFor(absl::Milliseconds(300));
  EXPECT_EQ(sink_.sceneCount(), 1u);
  publisher->setObservation(robot_->observation(/*time=*/0.1, ModeNumber::STANCE));
  ASSERT_TRUE(waitFor([&] { return sink_.sceneCount() >= 2; }));
  publisher->stop();
  EXPECT_EQ(publisher->statistics().scenesPublished, 2u);
}

TEST_F(VisualizationPublisherTest, TheSceneDrawsTheLatestPolicyAndSample) {
  std::unique_ptr<VisualizationPublisher> publisher = create();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  CommandData command;
  PrimalSolution solution;
  robot_->makePolicy(/*startTime=*/0.0, /*nodes=*/11, /*normalForce=*/300.0, vector2_t::Zero(), &command, &solution);
  // As MpcServer calls it after publishing a policy.
  publisher->postSolveObserver()(command, solution, PerformanceIndex());
  EXPECT_EQ(publisher->statistics().policiesSet, 1u);
  EXPECT_EQ(publisher->statistics().observationsSet, 1u);
  ASSERT_TRUE(publisher->pushRobotState(sampleAt(0.05)));
  ASSERT_TRUE(waitFor([&] {
    for (const humanoid_mpc_msgs::VisualizationScene& scene : sink_.scenes()) {
      if (scene.robots_size() == 3 && scene.time() == 0.05) {
        return true;
      }
    }
    return false;
  }));
  publisher->stop();
}

TEST_F(VisualizationPublisherTest, ASampleThatDoesNotDecodeIsCountedAndNotPlotted) {
  std::unique_ptr<VisualizationPublisher> publisher = create();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  humanoid_mpc_msgs::RobotStateSample sample = sampleAt(0.0);
  sample.mutable_joint_positions()->RemoveLast();
  ASSERT_TRUE(publisher->pushRobotState(sample));
  ASSERT_TRUE(waitFor([&] { return publisher->statistics().robotStatesRejected == 1; }));
  ASSERT_TRUE(publisher->pushRobotState(sampleAt(0.01)));
  ASSERT_TRUE(waitFor([&] { return sink_.telemetryCount() == 1; }));
  publisher->stop();
  EXPECT_EQ(sink_.telemetryTimes(), std::vector<double>{0.01});
}

TEST_F(VisualizationPublisherTest, AClockThatGoesBackIsPlottedToo) {
  std::unique_ptr<VisualizationPublisher> publisher = create();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  for (const double time : {5.0, 5.01, 0.0, 0.01}) {
    ASSERT_TRUE(publisher->pushRobotState(sampleAt(time)));
  }
  ASSERT_TRUE(waitFor([&] { return sink_.telemetryCount() == 4; }));
  publisher->setObservation(robot_->observation(/*time=*/0.02, ModeNumber::STANCE));
  ASSERT_TRUE(waitFor([&] { return sink_.sceneCount() >= 1; }));
  publisher->stop();
  EXPECT_EQ(sink_.telemetryTimes(), (std::vector<double>{5.0, 5.01, 0.0, 0.01}));
}

TEST_F(VisualizationPublisherTest, StartAndStopFollowTheirContract) {
  std::unique_ptr<VisualizationPublisher> neverStarted = create();
  ASSERT_NE(neverStarted, nullptr);
  neverStarted.reset();

  std::unique_ptr<VisualizationPublisher> publisher = create();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(publisher->start().ok());
  EXPECT_EQ(publisher->start().code(), absl::StatusCode::kFailedPrecondition);
  const absl::Time before = absl::Now();
  publisher->stop();
  EXPECT_LT(absl::Now() - before, absl::Seconds(1));
  publisher->stop();
}

TEST_F(VisualizationPublisherTest, InvalidOptionsAreRefused) {
  VisualizationPublisher::Options options;
  options.robotStateQueueCapacity = 0;
  EXPECT_EQ(VisualizationPublisher::Create(robot_->model(), sink_.function(), options).status().code(), absl::StatusCode::kInvalidArgument);
  options = VisualizationPublisher::Options();
  options.pollPeriod = absl::ZeroDuration();
  EXPECT_EQ(VisualizationPublisher::Create(robot_->model(), sink_.function(), options).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(VisualizationPublisher::Create(robot_->model(), VisualizationPublisher::PublishFunction()).status().code(),
            absl::StatusCode::kInvalidArgument);
}

std::unique_ptr<robot::ipc::Bus> createBus(const std::string& name) {
  robot::ipc::BusOptions options;
  options.nodeName = name;
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = name, .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  EXPECT_TRUE(bus.ok()) << bus.status();
  return bus.ok() ? std::move(*bus) : nullptr;
}

// robot/state from a robot process reaches the plots over the bus: the subscription, the queue, the thread and the
// publish on the MPC node's bus.
TEST_F(VisualizationPublisherTest, RobotStateOnTheBusBecomesTelemetryOnTheBus) {
  std::unique_ptr<robot::ipc::Bus> mpcBus = createBus("mpc");
  std::unique_ptr<robot::ipc::Bus> robotBus = createBus("robot");
  std::unique_ptr<robot::ipc::Bus> bridgeBus = createBus("bridge");
  ASSERT_TRUE(mpcBus != nullptr && robotBus != nullptr && bridgeBus != nullptr);
  ASSERT_TRUE(mpcBus->connect(robotBus->boundEndpoint()).ok());
  ASSERT_TRUE(bridgeBus->connect(mpcBus->boundEndpoint()).ok());

  absl::StatusOr<std::unique_ptr<VisualizationPublisher>> publisher = VisualizationPublisher::Create(robot_->model(), *mpcBus);
  ASSERT_TRUE(publisher.ok()) << publisher.status();
  ASSERT_TRUE((*publisher)->subscribeRobotState(*mpcBus).ok());
  absl::Mutex mutex;
  std::vector<double> received;
  ASSERT_TRUE(bridgeBus
                  ->subscribe<humanoid_mpc_msgs::TelemetrySeries>(ipc::topics::kVizTelemetry, robot::ipc::Delivery::kAll,
                                                                  [&](const humanoid_mpc_msgs::TelemetrySeries& series) {
                                                                    absl::MutexLock lock(&mutex);
                                                                    received.push_back(series.time());
                                                                  })
                  .ok());
  ASSERT_TRUE(mpcBus->start().ok());
  ASSERT_TRUE(robotBus->start().ok());
  ASSERT_TRUE(bridgeBus->start().ok());
  ASSERT_TRUE((*publisher)->start().ok());

  double time = 0.0;
  ASSERT_TRUE(waitFor([&] {
    time += 0.01;
    EXPECT_TRUE(robotBus->publish(ipc::topics::kRobotState, sampleAt(time)).ok());
    absl::SleepFor(absl::Milliseconds(5));
    absl::MutexLock lock(&mutex);
    return received.size() >= 5;
  }));
  (*publisher)->stop();
  bridgeBus->stop();
  robotBus->stop();
  mpcBus->stop();
  absl::MutexLock lock(&mutex);
  for (size_t index = 1; index < received.size(); ++index) {
    EXPECT_LT(received[index - 1], received[index]);
  }
}

// The MPC node's attacher (node::MpcNodeRuntime::VisualizationAttacher): the publisher it creates on the node's bus takes
// robot/state from the robot and the policies from the observer it returns, and publishes viz/telemetry and viz/scene
// on that bus once the node starts it.
TEST_F(VisualizationPublisherTest, TheBusAttacherPublishesOnTheNodesBusFromTheRobotAndTheObserver) {
  std::unique_ptr<robot::ipc::Bus> mpcBus = createBus("mpc");
  std::unique_ptr<robot::ipc::Bus> robotBus = createBus("robot");
  std::unique_ptr<robot::ipc::Bus> bridgeBus = createBus("bridge");
  ASSERT_TRUE(mpcBus != nullptr && robotBus != nullptr && bridgeBus != nullptr);
  ASSERT_TRUE(mpcBus->connect(robotBus->boundEndpoint()).ok());
  ASSERT_TRUE(bridgeBus->connect(mpcBus->boundEndpoint()).ok());

  std::unique_ptr<VisualizationPublisher> publisher;
  const VisualizationPublisher::BusAttacher attacher =
      VisualizationPublisher::MakeBusAttacher(robot_->model(), VisualizationPublisher::Options(), &publisher);
  absl::StatusOr<VisualizationPublisher::PostSolveObserver> observer = attacher(*mpcBus);
  ASSERT_TRUE(observer.ok()) << observer.status();
  ASSERT_NE(publisher, nullptr);
  ASSERT_TRUE(*observer);

  absl::Mutex mutex;
  size_t telemetryReceived = 0;
  size_t scenesReceived = 0;
  ASSERT_TRUE(bridgeBus
                  ->subscribe<humanoid_mpc_msgs::TelemetrySeries>(ipc::topics::kVizTelemetry, robot::ipc::Delivery::kAll,
                                                                  [&](const humanoid_mpc_msgs::TelemetrySeries& /*series*/) {
                                                                    absl::MutexLock lock(&mutex);
                                                                    ++telemetryReceived;
                                                                  })
                  .ok());
  ASSERT_TRUE(bridgeBus
                  ->subscribe<humanoid_mpc_msgs::VisualizationScene>(ipc::topics::kVizScene, robot::ipc::Delivery::kAll,
                                                                     [&](const humanoid_mpc_msgs::VisualizationScene& /*scene*/) {
                                                                       absl::MutexLock lock(&mutex);
                                                                       ++scenesReceived;
                                                                     })
                  .ok());
  ASSERT_TRUE(mpcBus->start().ok());
  ASSERT_TRUE(robotBus->start().ok());
  ASSERT_TRUE(bridgeBus->start().ok());
  // As the node does: the publisher starts with it.
  ASSERT_TRUE(publisher->start().ok());

  CommandData command;
  PrimalSolution solution;
  robot_->makePolicy(/*startTime=*/0.0, /*nodes=*/11, /*normalForce=*/300.0, vector2_t::Zero(), &command, &solution);
  double time = 0.0;
  ASSERT_TRUE(waitFor([&] {
    time += 0.01;
    // As the MpcServer calls it after every policy it publishes, and as the robot publishes its samples.
    (*observer)(command, solution, PerformanceIndex());
    EXPECT_TRUE(robotBus->publish(ipc::topics::kRobotState, sampleAt(time)).ok());
    absl::SleepFor(absl::Milliseconds(5));
    absl::MutexLock lock(&mutex);
    return telemetryReceived >= 3 && scenesReceived >= 1;
  }));
  EXPECT_GT(publisher->statistics().policiesSet, 0u);
  EXPECT_GT(publisher->statistics().robotStatesQueued, 0u);
  publisher->stop();
  bridgeBus->stop();
  robotBus->stop();
  mpcBus->stop();
}

TEST_F(VisualizationPublisherTest, TheBusAttacherNeedsAPlaceForItsPublisher) {
  std::unique_ptr<robot::ipc::Bus> mpcBus = createBus("mpc");
  ASSERT_NE(mpcBus, nullptr);
  const VisualizationPublisher::BusAttacher attacher =
      VisualizationPublisher::MakeBusAttacher(robot_->model(), VisualizationPublisher::Options(), /*publisher=*/nullptr);
  EXPECT_EQ(attacher(*mpcBus).status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::visualization
