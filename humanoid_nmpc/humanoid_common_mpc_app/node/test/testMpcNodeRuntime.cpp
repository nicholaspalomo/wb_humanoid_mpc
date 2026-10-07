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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/synchronized_module/SolverSynchronizedModule.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_mpc_config/mpc_parameter_update.nproto.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/node/test/ScriptedRobot.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/Delivery.h"
#include "support/AtlasReferenceStack.h"

/*
 * MpcNodeRuntime, the MPC node without its formulation: the DRC Atlas references and procedural motion manager around
 * OCS2's scripted MPC (AtlasReferenceStack, no CppAD), served on a loopback bus to a robot and an operator that the test
 * scripts. Every topic is the one the robot process and the GUI use.
 *
 * A policy is checked with `if (!policy.has_value()) FAIL()` before it is read, not with ASSERT_TRUE: clang-tidy's
 * bugprone-unchecked-optional-access sees through an `if` that returns, not through googletest's assertion macros.
 */

namespace ocs2::humanoid::node {
namespace {

namespace topics = ::ocs2::humanoid::ipc::topics;

/**
 * Stands in for the MPC parameter updater: updates are handed to it on the bus's IO thread and taken into use in the
 * preSolverRun() of the next solve, as MpcParameterUpdaterModule does; it records the contact estimator of each and the
 * solve it was applied at.
 */
class RecordingParameterModule final : public SolverSynchronizedModule {
 public:
  struct Applied {
    std::string contactEstimator;
    scalar_t solveTime = 0.0;
  };

  void enqueue(const mpc_config::MpcParameterUpdate& update) {
    absl::MutexLock lock(mutex_);
    pending_ = update.task.contact_estimator;
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
std::shared_ptr<T> borrowed(T* absl_nonnull object) {
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
  Harness(const Harness&) = delete;
  Harness& operator=(const Harness&) = delete;

  MpcNodeRuntime::Components components() {
    MpcNodeRuntime::Components components;
    components.mpc = &stack_.mpc();
    components.resetTargetTrajectories = [this](const SystemObservation& observation) {
      return stack_.resetTarget(observation.time, observation.state);
    };
    components.motionManager = borrowed(&stack_.motionManager());
    const std::shared_ptr<RecordingParameterModule> module = parameterModule_;
    components.parameterUpdateSink = [module](const mpc_config::MpcParameterUpdate& update) { module->enqueue(update); };
    if (stack_.planningReferenceManager() != nullptr) {
      components.contactPlanningReferenceManager = borrowed<const ContactPlanningReferenceManager>(stack_.planningReferenceManager());
    }
    return components;
  }

  MpcNodeRuntime::Config config() const {
    MpcNodeRuntime::Config config;
    config.dimensions = {.stateDim = stack_.model().getStateDim(), .inputDim = stack_.model().getInputDim(), .numModes = kNumHumanoidModes};
    config.solverThread = defaultSolverThreadConfig(/*realtimePriority=*/0);
    config.robotName = robotName_;
    config.taskFileIdentity = taskFileIdentity_;
    return config;
  }

  /** The robot the runtime is created for (MpcNodeRuntime::Config::robotName); empty by default: not checked. */
  void setRobotName(std::string robotName) { robotName_ = std::move(robotName); }
  /** The configuration the runtime runs (MpcNodeRuntime::Config::taskFileIdentity); empty by default: not checked. */
  void setTaskFileIdentity(std::string taskFileIdentity) { taskFileIdentity_ = std::move(taskFileIdentity); }

  /** Creates the runtime on a loopback bus connected to the scripted robot and operator, and starts everything. */
  absl::Status create(MpcNodeRuntime::Components components) {
    std::unique_ptr<robot::ipc::Bus> mpcBus = ipc::test_support::createNodeBus("mpc");
    robot_.connect(*mpcBus);
    absl::StatusOr<std::unique_ptr<MpcNodeRuntime>> runtime = MpcNodeRuntime::Create(std::move(mpcBus), std::move(components), config());
    if (!runtime.ok()) return runtime.status();
    runtime_ = *std::move(runtime);
    absl::Status started = robot_.start();
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
    observation.mode = ModeNumber::kStance;
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
  std::string robotName_;
  std::string taskFileIdentity_;
};

humanoid_mpc_msgs::WalkingVelocityCommand velocityCommand(double vx, double vy, double height, double yawRate) {
  humanoid_mpc_msgs::WalkingVelocityCommand message;
  message.set_linear_velocity_x(vx);
  message.set_linear_velocity_y(vy);
  message.set_desired_pelvis_height(height);
  message.set_angular_velocity_z(yawRate);
  return message;
}

/** The command limits of the reference file `referenceFile`, which the motion manager scales the commands by. */
ReferenceSettings commandLimits(const std::string& referenceFile) {
  const absl::StatusOr<mpc_config::ReferenceFile> file = loadReferenceFile(referenceFile);
  if (!file.ok()) {
    ADD_FAILURE() << file.status();
    return {};
  }
  absl::StatusOr<ReferenceSettings> settings = referenceSettingsFromConfig(*file);
  if (!settings.ok()) {
    ADD_FAILURE() << settings.status();
    return {};
  }
  return *std::move(settings);
}

TEST(MpcNodeRuntime, AnObservationGetsAPolicySolvedFromIt) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  const std::optional<humanoid_mpc_msgs::MpcPolicy> policy = harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1));
  if (!policy.has_value()) FAIL() << "no policy";
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
  if (!policy.has_value()) FAIL() << "no policy";
  EXPECT_EQ(policy->resets_served(), 0);
  EXPECT_EQ(policy->full_resets_served(), 0);

  // A solver reset: requested grows, full_requested does not.
  policy = harness.solve(harness.observation(/*time=*/1.1, /*sequence=*/2, /*requested=*/1, /*fullRequested=*/0));
  if (!policy.has_value()) FAIL() << "no policy";
  EXPECT_EQ(policy->resets_served(), 1);
  EXPECT_EQ(policy->full_resets_served(), 0);
  EXPECT_EQ(harness.runtime().statistics().server.solverResets, 1);
  EXPECT_EQ(harness.runtime().statistics().server.fullResets, 1);

  // A full reset.
  policy = harness.solve(harness.observation(/*time=*/1.2, /*sequence=*/3, /*requested=*/2, /*fullRequested=*/1));
  if (!policy.has_value()) FAIL() << "no policy";
  EXPECT_EQ(policy->resets_served(), 2);
  EXPECT_EQ(policy->full_resets_served(), 1);
  EXPECT_EQ(harness.runtime().statistics().server.fullResets, 2);

  // No new request: no reset.
  policy = harness.solve(harness.observation(/*time=*/1.3, /*sequence=*/4, /*requested=*/2, /*fullRequested=*/1));
  if (!policy.has_value()) FAIL() << "no policy";
  EXPECT_EQ(harness.runtime().statistics().server.fullResets, 2);
  EXPECT_EQ(harness.runtime().statistics().server.solverResets, 1);
}

TEST(MpcNodeRuntime, TheVelocityCommandReachesTheMotionManagerAndGoesOutWithThePolicy) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  const ReferenceSettings limits = commandLimits(harness.stack().referenceFile());
  const scalar_t maxVelocityX = limits.maxDisplacementVelocityX;
  const scalar_t maxRotationVelocity = limits.maxRotationVelocity;

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
  if (!policy.has_value()) FAIL() << "no policy";
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
  const scalar_t maxVelocityX = commandLimits(stack.referenceFile()).maxDisplacementVelocityX;

  ASSERT_TRUE(applyWalkingVelocityCommand(velocityCommand(/*vx=*/3.0, /*vy=*/0.0, /*height=*/0.8, /*yawRate=*/0.0), motionManager).ok());
  EXPECT_DOUBLE_EQ(motionManager.getScaledWalkingVelocityCommand().linear_velocity_x, maxVelocityX) << "clamped to the stick's stop";

  const WalkingVelocityCommand before = motionManager.getScaledWalkingVelocityCommand();
  const absl::Status refused = applyWalkingVelocityCommand(
      velocityCommand(/*vx=*/0.1, /*vy=*/0.0, /*height=*/std::numeric_limits<double>::infinity(), /*yawRate=*/0.0), motionManager);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(motionManager.getScaledWalkingVelocityCommand().linear_velocity_x, before.linear_velocity_x);
  EXPECT_EQ(motionManager.getScaledWalkingVelocityCommand().desired_pelvis_height, before.desired_pelvis_height);
}

TEST(MpcNodeRuntime, AParameterUpdateIsAppliedBeforeTheNextSolve) {
  Harness harness;
  ASSERT_TRUE(harness.create().ok());
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());
  EXPECT_TRUE(harness.parameterModule().applied().empty());

  // The tuning GUI's whole task file, typed.
  humanoid_mpc_config::MpcParameterUpdate update;
  update.mutable_task()->set_contact_estimator("always_in_contact");
  update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorMpcParameters, update,
                                     [&]() { return harness.runtime().statistics().parameterUpdatesReceived > 0; }));
  // No observation has come since, so nothing has been solved, and the update waits for the next solve.
  EXPECT_TRUE(harness.parameterModule().applied().empty());

  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.5, /*sequence=*/2)).has_value());
  const std::vector<RecordingParameterModule::Applied> applied = harness.parameterModule().applied();
  ASSERT_FALSE(applied.empty());
  EXPECT_EQ(applied.front().contactEstimator, "always_in_contact");
  EXPECT_EQ(applied.front().solveTime, 1.5);
  EXPECT_EQ(harness.runtime().statistics().parameterUpdatesRejected, 0u);
}

TEST(MpcNodeRuntime, AnUpdateOfAnotherSchemaVersionOrAnotherRobotIsNotApplied) {
  Harness harness;
  harness.setRobotName("drc_atlas");
  ASSERT_TRUE(harness.create().ok());
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());

  humanoid_mpc_config::MpcParameterUpdate update;
  update.mutable_task()->mutable_model_settings()->set_robot_name("drc_atlas");
  update.mutable_task()->set_contact_estimator("always_in_contact");
  update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  humanoid_mpc_config::MpcParameterUpdate otherRobot = update;
  otherRobot.mutable_task()->mutable_model_settings()->set_robot_name("unitree_r1");
  humanoid_mpc_config::MpcParameterUpdate otherFingerprint = update;
  otherFingerprint.set_schema_fingerprint("0123456789abcdef");
  // A field 103 more, as a GUI built from a newer schema sends it.
  humanoid_mpc_config::MpcParameterUpdate unknownField;
  ASSERT_TRUE(unknownField.ParseFromString(update.SerializeAsString() + std::string("\xb8\x06\x01", 3)));
  uint64_t rejected = 0;
  for (const humanoid_mpc_config::MpcParameterUpdate& refused : {otherRobot, otherFingerprint, unknownField}) {
    ++rejected;
    ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorMpcParameters, refused,
                                       [&]() { return harness.runtime().statistics().parameterUpdatesRejected >= rejected; }));
  }
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.5, /*sequence=*/2)).has_value());
  EXPECT_TRUE(harness.parameterModule().applied().empty()) << "a refused update reached the parameter updater";

  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorMpcParameters, update,
                                     [&]() { return harness.runtime().statistics().parameterUpdatesReceived > rejected; }));
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/2.0, /*sequence=*/3)).has_value());
  ASSERT_FALSE(harness.parameterModule().applied().empty());
  EXPECT_EQ(harness.runtime().statistics().parameterUpdatesRejected, 3u);
}

// Two configurations of one robot share its robot_name: only the task file's path tells them apart.
constexpr char kWholeBodyG1TaskFile[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kCentroidalG1TaskFile[] = "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";

/** A G1 update of `configPath` (empty: none), stamped with this build's fingerprint. */
humanoid_mpc_config::MpcParameterUpdate g1Update(const std::string& configPath) {
  humanoid_mpc_config::MpcParameterUpdate update;
  update.mutable_task()->mutable_model_settings()->set_robot_name("g1");
  update.mutable_task()->set_contact_estimator("always_in_contact");
  update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  update.set_config_path(configPath);
  return update;
}

TEST(MpcNodeRuntime, AnUpdateOfAnotherConfigurationOrWithoutAPathIsNotApplied) {
  Harness harness;
  harness.setRobotName("g1");
  harness.setTaskFileIdentity(kWholeBodyG1TaskFile);
  ASSERT_TRUE(harness.create().ok());
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());

  uint64_t rejected = 0;
  for (const humanoid_mpc_config::MpcParameterUpdate& refused : {g1Update(kCentroidalG1TaskFile), g1Update(/*configPath=*/"")}) {
    ++rejected;
    ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorMpcParameters, refused,
                                       [&]() { return harness.runtime().statistics().parameterUpdatesRejected >= rejected; }));
  }
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.5, /*sequence=*/2)).has_value());
  EXPECT_TRUE(harness.parameterModule().applied().empty()) << "an update of another configuration reached the parameter updater";

  // The running configuration's own update is applied.
  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorMpcParameters, g1Update(kWholeBodyG1TaskFile),
                                     [&]() { return harness.runtime().statistics().parameterUpdatesReceived > rejected; }));
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/2.0, /*sequence=*/3)).has_value());
  EXPECT_EQ(harness.parameterModule().applied().size(), 1u);
  EXPECT_EQ(harness.runtime().statistics().parameterUpdatesRejected, 2u);
}

TEST(MpcNodeRuntime, WithoutATaskFileIdentityAnUpdateOfAnyConfigurationIsApplied) {
  Harness harness;
  harness.setRobotName("g1");
  ASSERT_TRUE(harness.create().ok());
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.0, /*sequence=*/1)).has_value());
  ASSERT_TRUE(harness.sendAsOperator(topics::kOperatorMpcParameters, g1Update(kCentroidalG1TaskFile),
                                     [&]() { return harness.runtime().statistics().parameterUpdatesReceived > 0; }));
  ASSERT_TRUE(harness.solve(harness.observation(/*time=*/1.5, /*sequence=*/2)).has_value());
  EXPECT_EQ(harness.parameterModule().applied().size(), 1u);
  EXPECT_EQ(harness.runtime().statistics().parameterUpdatesRejected, 0u);
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
  humanoid_mpc_config::MpcParameterUpdate update;
  update.mutable_task()->set_contact_estimator("always_in_contact");
  update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  const absl::Time end = absl::Now() + absl::Milliseconds(200);
  while (absl::Now() < end) {
    harness.sendAsOperator(topics::kOperatorMpcParameters, update, []() { return true; });
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
  if (!policy.has_value()) FAIL() << "no policy";
  EXPECT_EQ(policy->annotations().target_contact_patches_size(), static_cast<int>(kNumContacts));
}

TEST(MpcNodeRuntime, TheVisualizationSeamGetsTheBusBeforeItStartsAndObservesEveryPolicy) {
  Harness harness;
  std::atomic<robot::ipc::Bus* absl_nullable> attachedBus{nullptr};
  std::atomic<bool> busWasRunning{true};
  const std::shared_ptr<std::atomic<int>> observed = std::make_shared<std::atomic<int>>(0);
  MpcNodeRuntime::Components components = harness.components();
  components.attachVisualization = [&](robot::ipc::Bus& bus) -> absl::StatusOr<ipc::MpcServer::PostSolveObserver> {
    attachedBus.store(&bus);
    busWasRunning.store(bus.isRunning());
    return ipc::MpcServer::PostSolveObserver(
        [observed](const CommandData& /*command*/, const PrimalSolution& /*solution*/, const PerformanceIndex& /*performance*/) {
          observed->fetch_add(1);
          return absl::OkStatus();
        });
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
