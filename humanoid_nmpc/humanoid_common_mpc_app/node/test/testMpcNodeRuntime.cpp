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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_oc/synchronized_module/SolverSynchronizedModule.h>

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/node/test_support/ScriptedRobot.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_mpc_msgs/yaml_document.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/Delivery.h"
#include "support/AtlasReferenceStack.h"

/*
 * MpcNodeRuntime, the MPC node without its formulation: the DRC Atlas references and procedural motion manager around
 * OCS2's scripted MPC (AtlasReferenceStack, no CppAD), served on a loopback bus to a robot and an operator that the test
 * scripts. Every topic is the one the robot process and the GUI use.
 */

namespace ocs2::humanoid::node {
namespace {

namespace topics = ::ocs2::humanoid::ipc::topics;

/**
 * Stands in for the MPC parameter updater: documents are handed to it on the bus's IO thread and taken into use in the
 * preSolverRun() of the next solve, as MpcParameterUpdaterModule does; it records the solve each one was applied at.
 */
class RecordingParameterModule final : public SolverSynchronizedModule {
 public:
  struct Applied {
    std::string yamlText;
    scalar_t solveTime = 0.0;
  };

  void enqueue(std::string yamlText) {
    absl::MutexLock lock(mutex_);
    pending_ = std::move(yamlText);
  }

  void preSolverRun(scalar_t initTime,
                    scalar_t /*finalTime*/,
                    const vector_t& /*initState*/,
                    const ReferenceManagerInterface& /*referenceManager*/) override {
    absl::MutexLock lock(mutex_);
    if (pending_.has_value()) {
      applied_.push_back({*std::move(pending_), initTime});
      pending_.reset();
    }
  }
  void postSolverRun(const PrimalSolution& /*primalSolution*/) override {}

  std::vector<Applied> applied() const {
    absl::MutexLock lock(mutex_);
    return applied_;
  }

 private:
  mutable absl::Mutex mutex_;
  std::optional<std::string> pending_ ABSL_GUARDED_BY(mutex_);
  std::vector<Applied> applied_ ABSL_GUARDED_BY(mutex_);
};

/** A non-owning shared_ptr, for an object of the reference stack, which outlives the runtime. */
template <typename T>
std::shared_ptr<T> borrowed(T* object) {
  return std::shared_ptr<T>(std::shared_ptr<void>(), object);
}

/** The node's runtime around the Atlas reference stack, a scripted robot and a scripted operator. */
class Harness {
 public:
  explicit Harness(AtlasReferenceStack::ScheduleSource scheduleSource = AtlasReferenceStack::ScheduleSource::kGaitSchedule)
      : stack_(scheduleSource), parameterModule_(std::make_shared<RecordingParameterModule>()) {
    stack_.mpc().getSolverPtr()->addSynchronizedModule(parameterModule_);
  }

  ~Harness() {
    if (runtime_ != nullptr) runtime_->stop();
  }

  MpcNodeRuntime::Components components() {
    MpcNodeRuntime::Components components;
    components.mpc = &stack_.mpc();
    components.resetTargetTrajectories = [this](const SystemObservation& observation) {
      return stack_.resetTarget(observation.time, observation.state);
    };
    components.motionManager = borrowed(&stack_.motionManager());
    const std::shared_ptr<RecordingParameterModule> module = parameterModule_;
    components.parameterUpdateSink = [module](std::string yamlText) { module->enqueue(std::move(yamlText)); };
    if (stack_.planningReferenceManager() != nullptr) {
      components.contactPlanningReferenceManager = borrowed<const ContactPlanningReferenceManager>(stack_.planningReferenceManager());
    }
    return components;
  }

  MpcNodeRuntime::Config config() const {
    MpcNodeRuntime::Config config;
    config.dimensions = {.stateDim = stack_.model().getStateDim(), .inputDim = stack_.model().getInputDim(), .numModes = kNumHumanoidModes};
    config.solverThread = defaultSolverThreadConfig(/*realtimePriority=*/0);
    return config;
  }

  /** Creates the runtime on a loopback bus connected to the scripted robot and operator, and starts everything. */
  absl::Status create(MpcNodeRuntime::Components components) {
    std::unique_ptr<robot::ipc::Bus> mpcBus = ipc::test_support::createNodeBus("mpc");
    robot_.connect(*mpcBus);
    absl::StatusOr<std::unique_ptr<MpcNodeRuntime>> runtime = MpcNodeRuntime::Create(std::move(mpcBus), std::move(components), config());
    if (!runtime.ok()) return runtime.status();
    runtime_ = *std::move(runtime);
    const absl::Status started = robot_.start();
    if (!started.ok()) return started;
    return runtime_->start();
  }
  absl::Status create() { return create(components()); }

  /** The robot's observation at `time`, standing, with the reset counters (requested, fullRequested). */
  humanoid_mpc_msgs::MpcObservation observation(double time, uint64_t sequence, uint64_t requested = 0, uint64_t fullRequested = 0) const {
    SystemObservation observation;
    observation.time = time;
    observation.state = stack_.initialState();
    observation.input = vector_t::Zero(stack_.model().getInputDim());
    observation.mode = ModeNumber::STANCE;
    return test_support::ScriptedRobot::observationMessage(observation, sequence, requested, fullRequested);
  }

  std::optional<humanoid_mpc_msgs::MpcPolicy> solve(const humanoid_mpc_msgs::MpcObservation& observation) {
    return robot_.solve(observation, /*timeoutSeconds=*/10.0);
  }

  bool sendAsOperator(absl::string_view topic, const google::protobuf::Message& message, const std::function<bool()>& received) {
    return robot_.sendAsOperator(topic, message, received);
  }

  AtlasReferenceStack& stack() { return stack_; }
  MpcNodeRuntime& runtime() { return *runtime_; }
  RecordingParameterModule& parameterModule() { return *parameterModule_; }

 private:
  AtlasReferenceStack stack_;
  std::shared_ptr<RecordingParameterModule> parameterModule_;
  test_support::ScriptedRobot robot_;
  std::unique_ptr<MpcNodeRuntime> runtime_;
};

humanoid_mpc_msgs::WalkingVelocityCommand velocityCommand(double vx, double vy, double height, double yawRate) {
  humanoid_mpc_msgs::WalkingVelocityCommand message;
  message.set_linear_velocity_x(vx);
  message.set_linear_velocity_y(vy);
  message.set_desired_pelvis_height(height);
  message.set_angular_velocity_z(yawRate);
  return message;
}

TEST(MpcNodeRuntime, AnObservationGetsAPolicySolvedFromIt) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  const std::optional<humanoid_mpc_msgs::MpcPolicy> policy = harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1));
  ASSERT_TRUE(policy.has_value());
  EXPECT_EQ(policy->init_observation().time(), 1.0);
  EXPECT_GT(policy->time_trajectory_size(), 1);
  EXPECT_EQ(policy->time_trajectory(0), 1.0);
  ASSERT_GT(policy->state_trajectory_size(), 0);
  EXPECT_EQ(policy->state_trajectory(0).data_size(), static_cast<int>(harness.stack().model().getStateDim()));
  EXPECT_TRUE(policy->solver_status().healthy());
  // The start-up reset is a full one, before the first solve.
  const MpcNodeRuntime::Statistics statistics = harness.runtime().statistics();
  EXPECT_EQ(statistics.server.fullResets, 1);
  EXPECT_EQ(statistics.server.solverResets, 0);
}

TEST(MpcNodeRuntime, ServesTheResetsTheObservationCountersRequest) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  std::optional<humanoid_mpc_msgs::MpcPolicy> policy = harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1));
  ASSERT_TRUE(policy.has_value());
  EXPECT_EQ(policy->resets_served(), 0);
  EXPECT_EQ(policy->full_resets_served(), 0);

  // A solver reset: requested grows, full_requested does not.
  policy = harness.solve(harness.observation(/*time=*/1.1, /*sequence=*/2, /*requested=*/1, /*fullRequested=*/0));
  ASSERT_TRUE(policy.has_value());
  EXPECT_EQ(policy->resets_served(), 1);
  EXPECT_EQ(policy->full_resets_served(), 0);
  EXPECT_EQ(harness.runtime().statistics().server.solverResets, 1);
  EXPECT_EQ(harness.runtime().statistics().server.fullResets, 1);

  // A full reset.
  policy = harness.solve(harness.observation(/*time=*/1.2, /*sequence=*/3, /*requested=*/2, /*fullRequested=*/1));
  ASSERT_TRUE(policy.has_value());
  EXPECT_EQ(policy->resets_served(), 2);
  EXPECT_EQ(policy->full_resets_served(), 1);
  EXPECT_EQ(harness.runtime().statistics().server.fullResets, 2);

  // No new request: no reset.
  policy = harness.solve(harness.observation(/*time=*/1.3, /*sequence=*/4, /*requested=*/2, /*fullRequested=*/1));
  ASSERT_TRUE(policy.has_value());
  EXPECT_EQ(harness.runtime().statistics().server.fullResets, 2);
  EXPECT_EQ(harness.runtime().statistics().server.solverResets, 1);
}

TEST(MpcNodeRuntime, TheVelocityCommandReachesTheMotionManagerAndGoesOutWithThePolicy) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  scalar_t maxVelocityX = std::numeric_limits<scalar_t>::quiet_NaN();
  scalar_t maxRotationVelocity = std::numeric_limits<scalar_t>::quiet_NaN();
  loadData::loadCppDataType(harness.stack().referenceFile(), "maxDisplacementVelocityX", maxVelocityX);
  loadData::loadCppDataType(harness.stack().referenceFile(), "maxRotationVelocity", maxRotationVelocity);

  ProceduralMpcMotionManager& motionManager = harness.stack().motionManager();
  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorWalkingVelocityCommand,
                                     velocityCommand(/*vx=*/0.5, /*vy=*/0.0, /*height=*/0.8, /*yawRate=*/-0.25),
                                     [&]() { return motionManager.getScaledWalkingVelocityCommand().linear_velocity_x != 0.0; }));
  const WalkingVelocityCommand scaled = motionManager.getScaledWalkingVelocityCommand();
  EXPECT_DOUBLE_EQ(scaled.linear_velocity_x, 0.5 * maxVelocityX);
  EXPECT_EQ(scaled.linear_velocity_y, 0.0);
  EXPECT_DOUBLE_EQ(scaled.angular_velocity_z, -0.25 * maxRotationVelocity);

  // The annotations of the next policy carry the scaled command the solve started from.
  const std::optional<humanoid_mpc_msgs::MpcPolicy> policy = harness.solve(harness.observation(/*time=*/2.0, /*sequence=*/1));
  ASSERT_TRUE(policy.has_value());
  EXPECT_EQ(policy->annotations().scaled_velocity_x(), scaled.linear_velocity_x);
  EXPECT_EQ(policy->annotations().scaled_yaw_rate(), scaled.angular_velocity_z);
  // Under the gait schedule there is no contact planner, so no target contact patch.
  EXPECT_EQ(policy->annotations().target_contact_patches_size(), 0);
}

TEST(MpcNodeRuntime, AVelocityCommandThatIsNotFiniteIsRefusedAndChangesNothing) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  ProceduralMpcMotionManager& motionManager = harness.stack().motionManager();
  const WalkingVelocityCommand before = motionManager.getScaledWalkingVelocityCommand();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorWalkingVelocityCommand,
                                     velocityCommand(/*vx=*/nan, /*vy=*/0.0, /*height=*/0.8, /*yawRate=*/0.0),
                                     [&]() { return harness.runtime().statistics().velocityCommandsRejected > 0; }));
  const WalkingVelocityCommand after = motionManager.getScaledWalkingVelocityCommand();
  EXPECT_EQ(after.linear_velocity_x, before.linear_velocity_x);
  EXPECT_EQ(after.angular_velocity_z, before.angular_velocity_z);
}

TEST(ApplyWalkingVelocityCommand, ScalesAConvertedCommandIntoTheMotionManagerAndRefusesOneThatIsNotFinite) {
  // The function the node's subscription and the lockstep closed loop apply a message with, without a bus.
  AtlasReferenceStack stack;
  ProceduralMpcMotionManager& motionManager = stack.motionManager();
  scalar_t maxVelocityX = std::numeric_limits<scalar_t>::quiet_NaN();
  loadData::loadCppDataType(stack.referenceFile(), "maxDisplacementVelocityX", maxVelocityX);

  ASSERT_TRUE(applyWalkingVelocityCommand(velocityCommand(/*vx=*/3.0, /*vy=*/0.0, /*height=*/0.8, /*yawRate=*/0.0), motionManager).ok());
  EXPECT_DOUBLE_EQ(motionManager.getScaledWalkingVelocityCommand().linear_velocity_x, maxVelocityX) << "clamped to the stick's stop";

  const WalkingVelocityCommand before = motionManager.getScaledWalkingVelocityCommand();
  const absl::Status refused = applyWalkingVelocityCommand(
      velocityCommand(/*vx=*/0.1, /*vy=*/0.0, /*height=*/std::numeric_limits<double>::infinity(), /*yawRate=*/0.0), motionManager);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(motionManager.getScaledWalkingVelocityCommand().linear_velocity_x, before.linear_velocity_x);
  EXPECT_EQ(motionManager.getScaledWalkingVelocityCommand().desired_pelvis_height, before.desired_pelvis_height);
}

TEST(MpcNodeRuntime, AParameterDocumentIsAppliedBeforeTheNextSolve) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());
  EXPECT_TRUE(harness.parameterModule().applied().empty());

  humanoid_mpc_msgs::YamlDocument document;
  document.set_yaml("contactEstimator: always_in_contact\n");
  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorMpcParameters, document,
                                     [&]() { return harness.runtime().statistics().parameterUpdatesReceived > 0; }));
  // No observation has come since, so nothing has been solved, and the document waits for the next solve.
  EXPECT_TRUE(harness.parameterModule().applied().empty());

  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.5, /*sequence=*/2)).has_value());
  const std::vector<RecordingParameterModule::Applied> applied = harness.parameterModule().applied();
  ASSERT_FALSE(applied.empty());
  EXPECT_EQ(applied.front().yamlText, document.yaml());
  EXPECT_EQ(applied.front().solveTime, 1.5);
}

TEST(MpcNodeRuntime, WithoutAParameterSinkTheParameterDocumentsAreNotTaken) {
  Harness harness;
  MpcNodeRuntime::Components components = harness.components();
  components.parameterUpdateSink = nullptr;
  ASSERT_TRUE(harness.create(std::move(components)).ok());
  // The operator reaches the node: its velocity commands arrive.
  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorWalkingVelocityCommand,
                                     velocityCommand(/*vx=*/0.1, /*vy=*/0.0, /*height=*/0.8, /*yawRate=*/0.0),
                                     [&]() { return harness.runtime().statistics().velocityCommandsReceived > 0; }));
  humanoid_mpc_msgs::YamlDocument document;
  document.set_yaml("contactEstimator: always_in_contact\n");
  const absl::Time end = absl::Now() + absl::Milliseconds(200);
  while (absl::Now() < end) {
    harness.sendAsOperator(topics::kOperatorMpcParameters, document, []() { return true; });
    absl::SleepFor(absl::Milliseconds(10));
  }
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());
  EXPECT_EQ(harness.runtime().statistics().parameterUpdatesReceived, 0);
  EXPECT_TRUE(harness.parameterModule().applied().empty());
}

TEST(MpcNodeRuntime, UnderTheContactPlannerThePolicyCarriesATargetPatchPerFoot) {
  Harness harness(AtlasReferenceStack::ScheduleSource::kContactPlanner);
  ASSERT_NE(harness.stack().planningReferenceManager(), nullptr);
  ASSERT_TRUE(harness.create().ok());
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());
  const std::optional<humanoid_mpc_msgs::MpcPolicy> policy = harness.solve(harness.observation(/*time=*/1.1, /*sequence=*/2));
  ASSERT_TRUE(policy.has_value());
  EXPECT_EQ(policy->annotations().target_contact_patches_size(), static_cast<int>(N_CONTACTS));
}

TEST(MpcNodeRuntime, TheVisualizationSeamGetsTheBusBeforeItStartsAndObservesEveryPolicy) {
  Harness harness;
  std::atomic<robot::ipc::Bus*> attachedBus{nullptr};
  std::atomic<bool> busWasRunning{true};
  const std::shared_ptr<std::atomic<int>> observed = std::make_shared<std::atomic<int>>(0);
  MpcNodeRuntime::Components components = harness.components();
  components.attachVisualization = [&](robot::ipc::Bus& bus) -> absl::StatusOr<ipc::MpcServer::PostSolveObserver> {
    attachedBus.store(&bus);
    busWasRunning.store(bus.isRunning());
    return ipc::MpcServer::PostSolveObserver([observed](const CommandData& /*command*/, const PrimalSolution& /*solution*/,
                                                        const PerformanceIndex& /*performance*/) { observed->fetch_add(1); });
  };
  ASSERT_TRUE(harness.create(std::move(components)).ok());
  EXPECT_EQ(attachedBus.load(), &harness.runtime().bus());
  EXPECT_FALSE(busWasRunning.load());
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());
  EXPECT_TRUE(test_support::waitFor([&]() { return observed->load() >= 1; }, /*timeoutSeconds=*/10.0));
}

TEST(MpcNodeRuntime, CreateRefusesWhatIsMissingAndPassesOnTheVisualizationError) {
  Harness harness;
  MpcNodeRuntime::Components noMpc = harness.components();
  noMpc.mpc = nullptr;
  EXPECT_EQ(MpcNodeRuntime::Create(ipc::test_support::createNodeBus("mpc"), std::move(noMpc), harness.config()).status().code(),
            absl::StatusCode::kInvalidArgument);

  MpcNodeRuntime::Components noMotionManager = harness.components();
  noMotionManager.motionManager = nullptr;
  EXPECT_EQ(MpcNodeRuntime::Create(ipc::test_support::createNodeBus("mpc"), std::move(noMotionManager), harness.config()).status().code(),
            absl::StatusCode::kInvalidArgument);

  MpcNodeRuntime::Components noResetTarget = harness.components();
  noResetTarget.resetTargetTrajectories = nullptr;
  EXPECT_EQ(MpcNodeRuntime::Create(ipc::test_support::createNodeBus("mpc"), std::move(noResetTarget), harness.config()).status().code(),
            absl::StatusCode::kInvalidArgument);

  MpcNodeRuntime::Components failingVisualization = harness.components();
  failingVisualization.attachVisualization = [](robot::ipc::Bus& /*bus*/) -> absl::StatusOr<ipc::MpcServer::PostSolveObserver> {
    return absl::FailedPreconditionError("no robot model");
  };
  EXPECT_EQ(
      MpcNodeRuntime::Create(ipc::test_support::createNodeBus("mpc"), std::move(failingVisualization), harness.config()).status().code(),
      absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace ocs2::humanoid::node
