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

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_common_mpc_app/robot/FsmStateMailbox.h"
#include "humanoid_common_mpc_app/robot/JointNamesByIndex.h"
#include "humanoid_common_mpc_app/robot/OperatorCommandMailbox.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/RealtimeLoopRunner.h"
#include "humanoid_common_mpc_app/robot/TelemetrySampler.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_msgs/dodgeball_throw.pb.h"
#include "humanoid_mpc_msgs/fsm_command.pb.h"
#include "humanoid_mpc_msgs/joint_targets.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/RobotTestSupport.h"
#include "mujoco_sim_interface/MujocoSimInterface.h"
#include "nproto/Textproto.h"
#include "robot_model/ContactEstimator.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * What the realtime thread of the robot process calls on its mailboxes, its event log and its telemetry sampler
 * allocates nothing once they are built, and does not wait on the locks of the communication thread. The counter
 * counts the calling thread's allocations only, so the handlers run on another thread here, as on the bus.
 */

namespace ocs2::humanoid {
namespace {

constexpr char kAtlasUrdf[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** An estimator whose construction waits until it is released: the communication thread holding the mailbox's lock. */
class GatedEstimator final : public robot::model::ContactEstimator {
 public:
  void estimateContactFlags(const robot::model::RobotState& /*state*/, std::vector<bool>& flags) override { flags.assign(2, true); }
  std::string getName() const override { return "gated"; }
};

OperatorCommandMailbox::Config mailboxConfig(const robot::model::RobotDescription& description) {
  OperatorCommandMailbox::Config config;
  config.jointNames = jointNamesByIndex(description);
  config.initialNominalPositions.assign(description.getNumJoints(), 0.0);
  return config;
}

/** Runs `handlers` on a thread of their own, as the bus's IO thread runs them, and waits for it. */
template <typename Handlers>
void onAnotherThread(Handlers handlers) {
  std::thread thread(handlers);
  thread.join();
}

TEST(RobotRealtimeAllocations, TheOperatorMailboxHandsEverythingOverWithoutAllocating) {
  const robot::model::RobotDescription description = robot_test::atlasDescription();
  robot::model::ContactEstimatorRegistry registry;
  absl::StatusOr<std::unique_ptr<OperatorCommandMailbox>> created = OperatorCommandMailbox::Create(mailboxConfig(description), registry);
  ASSERT_TRUE(created.ok()) << created.status();
  OperatorCommandMailbox& mailbox = **created;

  // The realtime thread's storage, sized once.
  std::vector<double> nominal(description.getNumJoints(), 0.0);
  FsmCommandEvent command;
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow throwCommand;
  std::string estimatorName;
  estimatorName.reserve(64);

  for (int round = 0; round < 3; ++round) {
    onAnotherThread([&]() {
      humanoid_mpc_msgs::FsmCommand fsm;
      fsm.set_command("JOINT_PD");
      fsm.set_sequence(static_cast<uint64_t>(round + 1));
      mailbox.onFsmCommand(fsm);
      humanoid_mpc_msgs::JointTargets targets;
      (*targets.mutable_positions())[description.getJointName(0)] = 0.1 * round;
      mailbox.onJointTargets(targets);
      humanoid_mpc_msgs::WalkingVelocityCommand walk;
      walk.set_desired_pelvis_height(0.9);
      mailbox.onWalkingVelocityCommand(walk);
      humanoid_mpc_config::MpcParameterUpdate parameters;
      parameters.mutable_task()->set_contact_estimator(round % 2 == 0 ? "always_in_contact" : "robot_state");
      if (round % 2 == 0) parameters.mutable_task()->mutable_contact_wrench_gate()->set_ramp_time(0.04);
      parameters.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
      mailbox.onMpcParameters(parameters);
      const absl::StatusOr<humanoid_mpc_msgs::DodgeballThrow> dodgeball = nproto::ParseTextprotoFile<humanoid_mpc_msgs::DodgeballThrow>(
          "humanoid_nmpc/humanoid_common_mpc_app/robot/test/data/dodgeball_throw.textproto");
      ASSERT_TRUE(dodgeball.ok()) << dodgeball.status();
      mailbox.onDodgeballThrow(*dodgeball);
    });
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    const bool tookCommand = mailbox.takeFsmCommand(command);
    const bool tookPosture = mailbox.takeNominalPosture(nominal);
    const bool tookThrow = mailbox.takeDodgeballThrow(throwCommand);
    const std::optional<double> height = mailbox.desiredGantryHeight();
    bool tookSettings = false;
    mailbox.takeControllerSettings([&](const ControllerSettingsUpdate& update) {
      tookSettings = true;
      if (update.hasContactEstimator && update.contactEstimatorName != estimatorName) estimatorName.assign(update.contactEstimatorName);
    });
    const size_t allocations = robot::realtime::heapAllocationCountOnThisThread() - before;
    EXPECT_TRUE(tookCommand && tookPosture && tookThrow && height.has_value() && tookSettings) << "round " << round;
    EXPECT_EQ(allocations, 0u) << "round " << round;
  }
  EXPECT_NEAR(nominal.front(), 0.2, 1.0e-12);
  EXPECT_EQ(estimatorName, "always_in_contact");
}

TEST(RobotRealtimeAllocations, TheRealtimeSideNeverWaitsForTheCommunicationThreadsLock) {
  const robot::model::RobotDescription description = robot_test::atlasDescription();
  robot::model::ContactEstimatorRegistry registry;
  absl::Notification building;
  absl::Notification release;
  registry.add("gated", "waits to be built", [&]() {
    building.Notify();
    release.WaitForNotification();
    return std::make_shared<GatedEstimator>();
  });
  absl::StatusOr<std::unique_ptr<OperatorCommandMailbox>> created = OperatorCommandMailbox::Create(mailboxConfig(description), registry);
  ASSERT_TRUE(created.ok()) << created.status();
  OperatorCommandMailbox& mailbox = **created;
  humanoid_mpc_msgs::FsmCommand fsm;
  fsm.set_command("WB_MPC");
  mailbox.onFsmCommand(fsm);

  // The communication thread builds an estimator under the mailbox's lock, and is held there.
  std::thread io([&]() { mailbox.contactEstimator("gated").IgnoreError(); });
  building.WaitForNotification();
  // Meanwhile every call of the realtime thread returns at once.
  FsmCommandEvent command;
  std::vector<double> nominal(description.getNumJoints(), 0.0);
  EXPECT_TRUE(mailbox.takeFsmCommand(command));
  EXPECT_EQ(command.kind, FsmCommandKind::kWbMpc);
  EXPECT_FALSE(mailbox.takeNominalPosture(nominal));
  EXPECT_FALSE(mailbox.takeControllerSettings([](const ControllerSettingsUpdate&) {}));
  release.Notify();
  io.join();
}

TEST(RobotRealtimeAllocations, TheEventLogAndTheFsmStatePostWithoutAllocating) {
  RealtimeEventLog log(/*capacity=*/16);
  FsmStateMailbox states;
  msgs::FsmState taken;
  states.write("GRAVITY_COMP", /*gantryLocked=*/true, /*controllerResets=*/0, /*mpcHealthy=*/true);
  states.take(taken);
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  for (int i = 0; i < 10; ++i) {
    log.post(RealtimeEventCode::kTorquesEnabled, /*detail=*/0, "ENABLE_TORQUES");
    states.write(i % 2 == 0 ? "WB_MPC" : "GRAVITY_COMP", /*gantryLocked=*/i % 3 == 0, /*controllerResets=*/static_cast<uint64_t>(i),
                 /*mpcHealthy=*/true);
  }
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
  EXPECT_EQ(log.drain([](const RealtimeEvent&) {}), 10u);
}

TEST(RobotRealtimeAllocations, TheTelemetrySamplerTakesSamplesWithoutAllocating) {
  const robot::model::RobotDescription description = robot_test::atlasDescription();
  TelemetrySampler::Config config;
  config.jointNames = jointNamesByIndex(description);
  config.decimation = 1;
  config.capacity = 8;
  TelemetrySampler sampler(config);
  robot::model::RobotState state(description);
  robot::model::RobotJointAction action(description);
  const contact_flag_t flags{true, false};
  const std::array<vector3_t, kNumContacts> forces{vector3_t(1.0, 2.0, 3.0), vector3_t::Zero()};

  // The first round warms nothing up: the slots are built with their shape at construction.
  for (int round = 0; round < 3; ++round) {
    const size_t before = robot::realtime::heapAllocationCountOnThisThread();
    for (int i = 0; i < 4; ++i) sampler.sample(state, action, "WB_MPC", flags, forces);
    EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u) << "round " << round;
    onAnotherThread([&]() { sampler.drain([](const msgs::RobotStateSample&) {}); });
  }
  // A full ring drops and counts, still without allocating.
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  for (int i = 0; i < 20; ++i) sampler.sample(state, action, "JOINT_PD", flags, forces);
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
  EXPECT_EQ(sampler.dropped(), 12u);
}

TEST(RobotRealtimeAllocations, TheLoopRunnerAllocatesNothingBetweenCycles) {
  RealtimeLoopConfig config;
  config.period = std::chrono::milliseconds(1);
  config.reportingWindow = std::chrono::milliseconds(20);
  RealtimeLoopRunner runner(config);
  std::atomic<int> cycles{0};
  std::atomic<size_t> atCycle10{0};
  std::atomic<size_t> atCycle100{0};
  ASSERT_TRUE(runner
                  .start([&]() {
                    const int cycle = cycles.fetch_add(1);
                    if (cycle == 10) atCycle10.store(robot::realtime::heapAllocationCountOnThisThread());
                    if (cycle == 100) atCycle100.store(robot::realtime::heapAllocationCountOnThisThread());
                  })
                  .ok());
  while (cycles.load() <= 100) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  runner.stop();
  // Ninety cycles and several closed reporting windows in between.
  EXPECT_EQ(atCycle100.load() - atCycle10.load(), 0u);
}

TEST(RobotRealtimeAllocations, TheBackendHandOverAllocatesNothingAndTakesNoLock) {
  // What the realtime thread calls on the MuJoCo backend every cycle, against its physics thread running: the state and
  // the action through RobotHWInterfaceBase's triple buffers, the measured foot forces, the torque switch and a throw.
  robot::mujoco_sim_interface::MujocoSimConfig config;
  config.scenePath = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
  config.headless = true;
  config.gantryHold = "weld_constraint";
  absl::StatusOr<std::unique_ptr<robot::mujoco_sim_interface::MujocoSimInterface>> created =
      robot::mujoco_sim_interface::MujocoSimInterface::Create(config, kAtlasUrdf);
  ASSERT_TRUE(created.ok()) << created.status();
  robot::mujoco_sim_interface::MujocoSimInterface& sim = **created;
  sim.initSim();
  sim.startSim();
  vector3_t left = vector3_t::Zero();
  vector3_t right = vector3_t::Zero();
  robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow throwCommand;
  throwCommand.flightTime = 0.5;
  throwCommand.mass = 0.4;
  const std::function<void(int)> cycle = [&](int index) {
    sim.updateInterfaceStateFromRobot();
    sim.getRobotJointAction().at(0)->feed_forward_effort = 0.0;
    sim.applyJointAction();
    sim.takeMeasuredFootForces(left, right);
    if (index % 10 == 0) sim.enableTorques();
    if (index % 10 == 5) sim.disableTorques();
    if (index == 20) sim.throwDodgeball(throwCommand);
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  };
  for (int index = 0; index < 10; ++index) cycle(index);
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  for (int index = 0; index < 50; ++index) cycle(index);
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
  EXPECT_GT(sim.getRobotState().getTime(), 0.0) << "the physics thread published its states";
}

}  // namespace
}  // namespace ocs2::humanoid
