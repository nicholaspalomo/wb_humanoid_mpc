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
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"
#include "humanoid_centroidal_mpc_app/CentroidalMpcNode.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc_app/node/DummySimLoop.h"
#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/teleop/KeyboardVelocityCommand.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_mpc_msgs/loop_timing.pb.h"
#include "humanoid_mpc_msgs/mpc_status.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/node/test/ScriptedRobot.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/ChildProcess.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/LoopbackNetwork.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/robot/test/ScriptedOperator.h"
#include "nproto/Textproto.h"
#include "robot_core/ResourcePaths.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/NodeEndpoint.h"
#include "robot_model/RobotDescription.h"

/*
 * The centroidal MPC node on the DRC Atlas task file, with the real SQP solver, served over loopback buses: to a robot
 * and an operator the test scripts, and to the dummy simulator for a few seconds. Builds the CppAD libraries of the
 * Atlas MPC on a cold cache (minutes); every node after the first loads them.
 */

namespace ocs2::humanoid {
namespace {

namespace topics = ::ocs2::humanoid::ipc::topics;
using node::test_support::ScriptedRobot;

node::MpcFiles atlasFiles() {
  node::MpcFiles files;
  // A file missing from the runfiles throws here, naming the data dependency to add (robot::resolveResourcePath).
  files.taskFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto").value();
  files.referenceFile =
      robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.textproto").value();
  files.urdfFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();
  files.gaitFile = robot::resolveResourcePath("humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto").value();
  return files;
}

std::unique_ptr<robot::ipc::Bus> loopbackBus(const std::string& name) {
  robot::ipc::BusOptions options;
  options.nodeName = name;
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = name, .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort, .bindHost = ""}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  EXPECT_TRUE(bus.ok()) << bus.status();
  return *std::move(bus);
}

/** The Atlas MPC node, serving a scripted robot and operator. */
class NodeHarness {
 public:
  NodeHarness() : files_(atlasFiles()) {
    std::unique_ptr<robot::ipc::Bus> mpcBus = loopbackBus("mpc");
    robot_.connect(*mpcBus);
    absl::StatusOr<std::unique_ptr<CentroidalMpcNode>> node =
        CentroidalMpcNode::Create(files_, std::move(mpcBus), CentroidalMpcNode::Options());
    EXPECT_TRUE(node.ok()) << node.status();
    node_ = *std::move(node);
    EXPECT_TRUE(robot_.start().ok());
    EXPECT_TRUE(node_->start().ok());
  }

  NodeHarness(const NodeHarness&) = delete;
  NodeHarness& operator=(const NodeHarness&) = delete;
  ~NodeHarness() { node_->stop(); }

  /** The task file's standing state at `time`. */
  SystemObservation standing(scalar_t time) {
    SystemObservation observation;
    observation.time = time;
    observation.state = node_->interface().getInitialState();
    observation.input = vector_t::Zero(node_->interface().getEffectiveMpcRobotModel().getInputDim());
    observation.mode = ModeNumber::kStance;
    return observation;
  }

  std::optional<humanoid_mpc_msgs::MpcPolicy> solve(scalar_t time, uint64_t sequence, uint64_t requested = 0, uint64_t fullRequested = 0) {
    return robot_.solve(ScriptedRobot::observationMessage(standing(time), sequence, requested, fullRequested), /*timeoutSeconds=*/60.0);
  }

  const node::MpcFiles& files() const { return files_; }
  CentroidalMpcNode& node() { return *node_; }
  ScriptedRobot& robot() { return robot_; }

 private:
  node::MpcFiles files_;
  ScriptedRobot robot_;
  std::unique_ptr<CentroidalMpcNode> node_;
};

TEST(CentroidalMpcNode, SolvesTheRobotsObservationsServesTheResetsTheyRequestAndDrawsThePolicies) {
  NodeHarness harness;
  const ipc::ModelDimensions dimensions = harness.node().dimensions();

  std::optional<humanoid_mpc_msgs::MpcPolicy> policy = harness.solve(/*time=*/0.0, /*sequence=*/1);
  if (!policy.has_value()) GTEST_FAIL() << "the node served no policy";
  EXPECT_TRUE(policy->solver_status().healthy());
  ASSERT_GT(policy->time_trajectory_size(), 1);
  EXPECT_EQ(policy->time_trajectory(0), 0.0);
  EXPECT_EQ(policy->state_trajectory(0).data_size(), static_cast<int>(dimensions.stateDim));
  EXPECT_EQ(policy->input_trajectory(0).data_size(), static_cast<int>(dimensions.inputDim));
  EXPECT_EQ(policy->resets_served(), 0);

  // A solver reset, then a full one, requested through the observation's counters.
  policy = harness.solve(/*time=*/0.02, /*sequence=*/2, /*requested=*/1, /*fullRequested=*/0);
  if (!policy.has_value()) GTEST_FAIL() << "the node served no policy";
  EXPECT_EQ(policy->resets_served(), 1);
  EXPECT_EQ(policy->full_resets_served(), 0);
  policy = harness.solve(/*time=*/0.04, /*sequence=*/3, /*requested=*/2, /*fullRequested=*/1);
  if (!policy.has_value()) GTEST_FAIL() << "the node served no policy";
  EXPECT_EQ(policy->resets_served(), 2);
  EXPECT_EQ(policy->full_resets_served(), 1);

  const ipc::MpcServer::Statistics statistics = harness.node().runtime().statistics().server;
  EXPECT_EQ(statistics.fullResets, 2);  // the start-up reset and the requested one
  EXPECT_EQ(statistics.solverResets, 1);
  EXPECT_EQ(statistics.failedAttempts, 0);

  // Every published policy also feeds the node's visualization publisher, which draws a scene of it on the bus.
  EXPECT_TRUE(node::test_support::waitFor([&]() {
    const visualization::VisualizationPublisherStatistics drawn = harness.node().visualization().statistics();
    return drawn.policiesSet == harness.node().runtime().statistics().server.policiesPublished && drawn.scenesPublished > 0;
  }));
  const visualization::VisualizationPublisherStatistics drawn = harness.node().visualization().statistics();
  EXPECT_GE(drawn.policiesSet, 3u);
  EXPECT_EQ(drawn.buildFailures, 0u);
  EXPECT_EQ(drawn.publishFailures, 0u);
}

TEST(CentroidalMpcNode, TheVelocityCommandReachesTheMotionManager) {
  NodeHarness harness;
  ASSERT_TRUE(harness.solve(/*time=*/0.0, /*sequence=*/1).has_value());
  humanoid_mpc_msgs::WalkingVelocityCommand command;
  command.set_linear_velocity_x(0.5);
  command.set_desired_pelvis_height(0.8);
  command.set_angular_velocity_z(-0.25);
  ProceduralMpcMotionManager& motionManager = harness.node().motionManager();
  ASSERT_TRUE(harness.robot().sendAsOperator(topics::kOperatorWalkingVelocityCommand, command,
                                             [&]() { return motionManager.getScaledWalkingVelocityCommand().linear_velocity_x > 0.0; }));
  const WalkingVelocityCommand scaled = motionManager.getScaledWalkingVelocityCommand();
  EXPECT_LT(scaled.angular_velocity_z, 0.0);
  // The next policy carries it.
  const std::optional<humanoid_mpc_msgs::MpcPolicy> policy = harness.solve(/*time=*/0.02, /*sequence=*/2);
  if (!policy.has_value()) GTEST_FAIL() << "the node served no policy";
  EXPECT_EQ(policy->annotations().scaled_velocity_x(), scaled.linear_velocity_x);
  EXPECT_EQ(policy->annotations().scaled_yaw_rate(), scaled.angular_velocity_z);
}

std::string readFile(const std::string& path) {
  std::ifstream file(path);
  std::stringstream content;
  content << file.rdbuf();
  return content.str();
}

/** The SQP solver of the node's MPC. */
SqpSolver& solverOf(CentroidalMpcNode& node) {
  return dynamic_cast<SqpSolver&>(*node.mpc().getSolverPtr());
}

/** Q of the running problem's quadratic state cost (the Atlas lists stateQuadraticCost). */
matrix_t runningStateWeights(CentroidalMpcNode& node) {
  matrix_t Q;
  matrix_t R;
  matrix_t P;
  solverOf(node).getOcpDefinitions().front().costPtr->get<QuadraticStateInputCost>(kStateQuadraticCostTerm).getGains(Q, R, P);
  return Q;
}

/**
 * The tuning GUI's update of the node's task file with `edit` applied, as it publishes it: the whole file, typed, with
 * the schema fingerprint and the file's configuration path `configPath`.
 */
humanoid_mpc_config::MpcParameterUpdate parameterUpdateOf(const std::string& taskFile,
                                                          const std::function<void(humanoid_mpc_config::TaskFile&)>& edit,
                                                          const std::string& configPath) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> task =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(readFile(taskFile), "task");
  EXPECT_TRUE(task.ok()) << task.status();
  humanoid_mpc_config::MpcParameterUpdate update;
  if (!task.ok()) return update;
  *update.mutable_task() = *task;
  edit(*update.mutable_task());
  update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  update.set_config_path(configPath);
  return update;
}

TEST(CentroidalMpcNode, AParameterUpdateIsAppliedBeforeTheNextSolve) {
  NodeHarness harness;
  ASSERT_TRUE(harness.solve(/*time=*/0.0, /*sequence=*/1).has_value());
  const matrix_t shippedQ = runningStateWeights(harness.node());
  const size_t shippedIterations = solverOf(harness.node()).getSettings().sqpIteration;

  // The tuning GUI sends the whole task file, typed; this one doubles the state weights and adds an SQP iteration.
  const humanoid_mpc_config::MpcParameterUpdate update = parameterUpdateOf(
      harness.files().taskFile,
      [](humanoid_mpc_config::TaskFile& task) {
        task.mutable_state_weights()->set_scaling(2.0 * task.state_weights().scaling());
        task.mutable_multiple_shooting()->set_sqp_iteration(task.multiple_shooting().sqp_iteration() + 1);
      },
      configFileIdentity(harness.files().taskFile));
  ASSERT_TRUE(harness.robot().sendAsOperator(topics::kOperatorMpcParameters, update,
                                             [&]() { return harness.node().runtime().statistics().parameterUpdatesReceived > 0; }));
  // Received, not yet applied: no observation has come since, so nothing has been solved.
  EXPECT_TRUE(runningStateWeights(harness.node()) == shippedQ) << "the update was applied before a solve";

  ASSERT_TRUE(harness.solve(/*time=*/0.02, /*sequence=*/2).has_value());
  EXPECT_EQ(harness.node().runtime().statistics().parameterUpdatesRejected, 0U);
  EXPECT_TRUE(runningStateWeights(harness.node()).isApprox(2.0 * shippedQ, 1.0e-12)) << "the edited state weights were not applied";
  EXPECT_EQ(solverOf(harness.node()).getSettings().sqpIteration, shippedIterations + 1);
}

TEST(CentroidalMpcNode, AnUpdateOfAnotherConfigurationIsRefused) {
  // The whole-body G1 and the centroidal G1 share robot_name "g1": the configuration path tells them apart. An update of
  // another task file than the node's is counted as rejected and reaches no solver.
  NodeHarness harness;
  ASSERT_TRUE(harness.solve(/*time=*/0.0, /*sequence=*/1).has_value());
  const matrix_t shippedQ = runningStateWeights(harness.node());
  const humanoid_mpc_config::MpcParameterUpdate update = parameterUpdateOf(
      harness.files().taskFile,
      [](humanoid_mpc_config::TaskFile& task) { task.mutable_state_weights()->set_scaling(3.0 * task.state_weights().scaling()); },
      "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  ASSERT_TRUE(harness.robot().sendAsOperator(topics::kOperatorMpcParameters, update,
                                             [&]() { return harness.node().runtime().statistics().parameterUpdatesRejected > 0; }));
  ASSERT_TRUE(harness.solve(/*time=*/0.02, /*sequence=*/2).has_value());
  EXPECT_EQ(harness.node().runtime().statistics().parameterUpdatesRejected, 1U);
  EXPECT_TRUE(runningStateWeights(harness.node()) == shippedQ) << "an update of another configuration was applied";
}

TEST(CentroidalMpcNode, TheUpdaterAppliesTheCentroidalFormulationsHotFields) {
  // The instantiated updater applies exactly the static list of the formulation, which the reload-class coverage test
  // compares with the schema without building an MPC.
  NodeHarness harness;
  EXPECT_EQ(harness.node().parameterUpdater().appliedFields(), centroidalHotFieldNames());
}

void expectSameTarget(const TargetTrajectories& expected, const TargetTrajectories& actual) {
  ASSERT_EQ(expected.timeTrajectory, actual.timeTrajectory);
  ASSERT_EQ(expected.stateTrajectory.size(), actual.stateTrajectory.size());
  ASSERT_EQ(expected.inputTrajectory.size(), actual.inputTrajectory.size());
  for (size_t i = 0; i < expected.stateTrajectory.size(); ++i) {
    EXPECT_EQ(expected.stateTrajectory[i], actual.stateTrajectory[i]) << "node " << i;
    EXPECT_EQ(expected.inputTrajectory[i], actual.inputTrajectory[i]) << "node " << i;
  }
}

TEST(CentroidalMpcNode, ResetsToTheTargetTheControllerHandsItsInProcessLink) {
  NodeHarness harness;
  CentroidalMpcInterface& interface = harness.node().interface();
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(harness.files().urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  // The controller as the robot process builds it, with a link that keeps the reset target the controller hands it.
  mpc_test::ScriptedMpc scriptedMpc(interface.mpcSettings(), interface.getEffectiveMpcRobotModel().getInputDim());
  MpcLink::ResetTargetFunction controllerResetTarget;
  const MpcLinkFactory capturing = [&](MpcLink::ResetTargetFunction resetTarget) -> std::unique_ptr<MpcLink> {
    controllerResetTarget = resetTarget;
    return std::make_unique<InProcessMpcLink>(scriptedMpc, std::move(resetTarget), InProcessMpcLink::Config());
  };
  absl::StatusOr<std::unique_ptr<CentroidalMpcMrtJointController>> controllerOrStatus = CentroidalMpcMrtJointController::Create(
      description, interface.modelSettings(), interface.getMpcRobotModel(), capturing, interface.getPinocchioInterface(),
      /*pdGainsFile=*/"", &interface.getEffectiveMpcRobotModel());
  // The controller, which owns the link the factory captured, lives in controllerOrStatus for the whole test.
  ASSERT_TRUE(controllerOrStatus.ok()) << controllerOrStatus.status();
  ASSERT_TRUE(controllerResetTarget);
  for (const scalar_t time : {0.0, 1.5, 12.25}) {
    SystemObservation observation = harness.standing(time);
    observation.state(6) += 0.2 * time;          // base x
    observation.state(10) = 0.03;                // pitch
    observation.state.head(6).setConstant(0.1);  // momentum
    expectSameTarget(controllerResetTarget(observation), harness.node().resetTargetTrajectories(observation));
  }
}

/** The dummy simulator against the node, on a bus of its own, playing the robot; run() on a thread of its own. */
class DummySimRunner {
 public:
  DummySimRunner(CentroidalMpcNode& node, robot::ipc::Bus& mpcBus) {
    std::unique_ptr<robot::ipc::Bus> robotBus = loopbackBus("robot");
    EXPECT_TRUE(robotBus->connect(mpcBus.boundEndpoint()).ok());
    EXPECT_TRUE(mpcBus.connect(robotBus->boundEndpoint()).ok());
    node::DummySimLoop::Config config;
    config.mpcDesiredFrequency = node.interface().mpcSettings().mpcDesiredFrequency_;
    config.link.dimensions = node.dimensions();
    absl::StatusOr<std::unique_ptr<node::DummySimLoop>> loop =
        node::DummySimLoop::Create(std::move(robotBus), node.interface().getRollout(), config);
    EXPECT_TRUE(loop.ok()) << loop.status();
    loop_ = *std::move(loop);
    initialObservation_.time = 0.0;
    initialObservation_.state = node.interface().getInitialState();
    initialObservation_.input = vector_t::Zero(node.dimensions().inputDim);
    initialObservation_.mode = ModeNumber::kStance;
  }

  /** Runs the loop for `duration` of wall time once its first policy has arrived, then stops it. */
  absl::Status runFor(absl::Duration duration) {
    absl::Status status;
    std::thread runner([&]() { status = loop_->run(initialObservation_, [this]() { return stop_.load(); }); });
    const bool started = node::test_support::waitFor([&]() { return loop_->statistics().steps > 0; }, /*timeoutSeconds=*/120.0);
    if (started) absl::SleepFor(duration);
    stop_.store(true);
    runner.join();
    if (!started) return absl::DeadlineExceededError("the dummy simulator received no policy");
    return status;
  }

  node::DummySimLoop& loop() { return *loop_; }
  const SystemObservation& initialObservation() const { return initialObservation_; }

 private:
  std::unique_ptr<node::DummySimLoop> loop_;
  SystemObservation initialObservation_;
  std::atomic<bool> stop_{false};
};

TEST(CentroidalMpcNode, TheDummySimulatorStandsOnTheNodesPoliciesWithoutAReset) {
  const node::MpcFiles files = atlasFiles();
  std::unique_ptr<robot::ipc::Bus> mpcBus = loopbackBus("mpc");
  robot::ipc::Bus* absl_nonnull mpcBusPointer = mpcBus.get();
  absl::StatusOr<std::unique_ptr<CentroidalMpcNode>> node =
      CentroidalMpcNode::Create(files, std::move(mpcBus), CentroidalMpcNode::Options());
  ASSERT_TRUE(node.ok()) << node.status();
  DummySimRunner dummySim(**node, *mpcBusPointer);
  ASSERT_TRUE((*node)->start().ok());
  ASSERT_TRUE(dummySim.runFor(absl::Seconds(4)).ok());
  (*node)->stop();

  const node::DummySimLoop::Statistics statistics = dummySim.loop().statistics();
  const ipc::MpcServer::Statistics server = (*node)->runtime().statistics().server;
  // Policies flowed: synchronized with the 50 Hz MPC, one per update of the 100 Hz plant.
  EXPECT_GT(statistics.steps, 20);
  EXPECT_EQ(statistics.synchronizedPolicies, (statistics.steps + 1) / 2);
  EXPECT_GE(server.policiesPublished, statistics.synchronizedPolicies);
  // No reset after the start-up one, and nothing dropped.
  EXPECT_EQ(server.fullResets, 1);
  EXPECT_EQ(server.solverResets, 0);
  EXPECT_EQ(server.failedAttempts, 0);
  EXPECT_EQ(server.robotSessions, 1);
  EXPECT_EQ(statistics.link.linkLosses, 0);
  EXPECT_EQ(statistics.link.stalePoliciesDropped, 0);
  EXPECT_EQ(statistics.link.foreignPoliciesDropped, 0);
  EXPECT_TRUE(statistics.link.solverHealthy);

  // It stands: the base stays where it started, upright, within a few centimeters.
  const MpcRobotModelBase<scalar_t>& model = (*node)->interface().getMpcRobotModel();
  const SystemObservation latest = dummySim.loop().latestObservation();
  const vector3_t startPosition = model.getBasePosition(dummySim.initialObservation().state);
  const vector3_t position = model.getBasePosition(latest.state);
  EXPECT_NEAR(position.z(), startPosition.z(), 0.05);
  EXPECT_LT((position - startPosition).head<2>().norm(), 0.05);
  const vector3_t orientation = model.getBaseOrientationEulerZYX(latest.state);
  EXPECT_LT(std::abs(orientation(1)), 0.1);  // pitch
  EXPECT_LT(std::abs(orientation(2)), 0.1);  // roll
  EXPECT_TRUE(latest.state.allFinite());
}

constexpr char kNodeBinary[] = "humanoid_nmpc/humanoid_centroidal_mpc_app/humanoid_centroidal_mpc_node";
constexpr char kRobotBinary[] = "humanoid_nmpc/humanoid_centroidal_mpc_app/humanoid_centroidal_mpc_robot";
/** [m] How far the base may sink below, or rise above, its standing height while the robot is up. */
constexpr double kBaseHeightTolerance = 0.1;
/** How long the operator lets a mode settle before the next command. */
constexpr absl::Duration kSettleTime = absl::Seconds(1);

/** The latest FSM state the operator received: a test failure and an empty state when none has come. */
humanoid_mpc_msgs::FsmState fsmStateOf(const test_support::ScriptedOperator& remoteControl) {
  std::optional<humanoid_mpc_msgs::FsmState> state = remoteControl.fsmState();
  if (!state.has_value()) {
    ADD_FAILURE() << "the operator has received no FSM state";
    return humanoid_mpc_msgs::FsmState();
  }
  return *std::move(state);
}

/** The latest loop timing the operator received: a test failure and an empty one when none has come. */
humanoid_mpc_msgs::LoopTiming timingOf(const test_support::ScriptedOperator& remoteControl) {
  std::optional<humanoid_mpc_msgs::LoopTiming> timing = remoteControl.timing();
  if (!timing.has_value()) {
    ADD_FAILURE() << "the operator has received no loop timing";
    return humanoid_mpc_msgs::LoopTiming();
  }
  return *std::move(timing);
}

/** The base position of the newest robot/state sample. */
vector3_t basePosition(const test_support::ScriptedOperator& remoteControl) {
  const std::optional<humanoid_mpc_msgs::RobotStateSample> sample = remoteControl.sample();
  if (!sample.has_value()) return vector3_t::Constant(std::numeric_limits<scalar_t>::quiet_NaN());
  return vector3_t(sample->base_position_world().x(), sample->base_position_world().y(), sample->base_position_world().z());
}

// The hardware topology end to end, as every simulation launch runs it: the MPC node binary and the robot binary (the
// MuJoCo backend, headless) as two processes on a network file of the test, over the remote MPC link, and an operator.
// The node loads the CppAD libraries the tests above generated in the working directory (it generates them on a cold
// cache, when this test runs alone). The robot comes up on the gantry, enters WB_MPC through JOINT_PD on the node's
// policies, is released, stands, walks forward and stops without falling; the loop holds its period, the policy in use
// stays fresh, the node's visualization reaches the bus, and SIGTERM ends both processes cleanly.
TEST(CentroidalMpcNodeEndToEnd, TheRobotProcessStandsAndWalksOnTheNodeBinarysPolicies) {
  const node::MpcFiles files = atlasFiles();
  const std::string sceneFile = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml").value();
  const absl::StatusOr<teleop::KeyboardCommandLimits> limits = teleop::loadKeyboardCommandLimits(files.referenceFile);
  ASSERT_TRUE(limits.ok()) << limits.status();
  const std::string networkFile = absl::StrCat(std::getenv("TEST_TMPDIR"), "/robot_and_node.textproto");
  const absl::StatusOr<robot::ipc::NetworkConfig> network =
      test_support::writeLoopbackNetworkFile(networkFile, {"robot", "mpc", "operator"});
  ASSERT_TRUE(network.ok()) << network.status();
  test_support::ScriptedOperator remoteControl(*network);
  remoteControl.start();

  const std::vector<std::string> sharedFlags = {
      "--robot_name=drc_atlas",
      absl::StrCat("--task_file=", files.taskFile),
      absl::StrCat("--reference_file=", files.referenceFile),
      absl::StrCat("--urdf_file=", files.urdfFile),
      absl::StrCat("--network_config=", networkFile),
  };
  std::vector<std::string> nodeCommand = {kNodeBinary, absl::StrCat("--gait_file=", files.gaitFile)};
  nodeCommand.insert(nodeCommand.end(), sharedFlags.begin(), sharedFlags.end());
  test_support::ChildProcess mpcNode(nodeCommand);
  std::vector<std::string> robotCommand = {kRobotBinary, absl::StrCat("--mjcf_file=", sceneFile), "--headless", "--realtime_cores=none",
                                           "--backend_cores=none"};
  robotCommand.insert(robotCommand.end(), sharedFlags.begin(), sharedFlags.end());
  test_support::ChildProcess robotProcess(robotCommand);

  // The robot on the gantry, and the node's first healthy solve of its observations (minutes on a cold CppAD cache).
  ASSERT_TRUE(test_support::waitFor([&]() { return remoteControl.fsmState().has_value() || !robotProcess.running(); }, absl::Seconds(60)));
  ASSERT_TRUE(robotProcess.running()) << "the robot process exited on start-up";
  EXPECT_EQ(fsmStateOf(remoteControl).mode(), "ZERO_TORQUE");
  ASSERT_TRUE(test_support::waitFor(
      [&]() {
        const std::optional<humanoid_mpc_msgs::MpcStatus> status = remoteControl.mpcStatus();
        return (status.has_value() && status->solver_status().healthy()) || !mpcNode.running();
      },
      absl::Seconds(600)));
  ASSERT_TRUE(mpcNode.running()) << "the MPC node exited on start-up";

  // Into WB_MPC on the gantry, then released, as an operator does it: JOINT_PD until the posture has settled from the
  // slump of ZERO_TORQUE, WB_MPC until its entry ramp from the JOINT_PD action (the task file's mpc_entry_blend_time) is
  // over.
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(files.taskFile);
  ASSERT_TRUE(task.ok()) << task.status();
  const scalar_t entryBlendTime = task->mpc_entry_blend_time.value_or(0.0);
  ASSERT_TRUE(remoteControl.enterMode("JOINT_PD"));
  absl::SleepFor(kSettleTime);
  ASSERT_TRUE(remoteControl.enterMode("WB_MPC"));
  absl::SleepFor(kSettleTime + absl::Seconds(entryBlendTime));
  ASSERT_TRUE(fsmStateOf(remoteControl).mpc_healthy());
  ASSERT_TRUE(
      remoteControl.sendFsmCommand("UNLOCK_GANTRY", [](const humanoid_mpc_msgs::FsmState& state) { return !state.gantry_locked(); }));
  const uint64_t resetsAtRelease = fsmStateOf(remoteControl).controller_resets();
  const vector3_t released = basePosition(remoteControl);
  ASSERT_TRUE(released.allFinite());

  // Stands, walks forward at 0.3 m/s as the keyboard teleoperation commands it, and stops: the base stays at its
  // standing height throughout and travels forward while the command does.
  remoteControl.resetBaseHeightRange();
  remoteControl.sendVelocityCommand(teleop::keyboardCommandToMessage(vector4_t::Zero(), *limits), absl::Seconds(3));
  const vector3_t walkStart = basePosition(remoteControl);
  remoteControl.sendVelocityCommand(teleop::keyboardCommandToMessage(vector4_t(0.3, 0.0, 0.0, 0.0), *limits), absl::Seconds(5));
  const vector3_t walkEnd = basePosition(remoteControl);
  remoteControl.sendVelocityCommand(teleop::keyboardCommandToMessage(vector4_t::Zero(), *limits), absl::Seconds(3));
  const test_support::BaseHeightRange heights = remoteControl.baseHeightRange();
  EXPECT_GT(heights.samples, 500u) << "robot/state at the telemetry rate";
  EXPECT_GT(heights.min, released.z() - kBaseHeightTolerance) << "the robot fell";
  EXPECT_LT(heights.max, released.z() + kBaseHeightTolerance);
  EXPECT_GT(walkEnd.x() - walkStart.x(), 0.3) << "a fifth of the commanded 1.5 m at least";
  const humanoid_mpc_msgs::FsmState state = fsmStateOf(remoteControl);
  EXPECT_EQ(state.mode(), "WB_MPC");
  EXPECT_FALSE(state.gantry_locked());
  EXPECT_EQ(state.controller_resets(), resetsAtRelease) << "the fall recovery caught the robot";
  EXPECT_TRUE(state.mpc_healthy());

  // The loop held its period and used fresh policies of the node, whose visualization reached the bus.
  ASSERT_TRUE(remoteControl.timing().has_value());
  const humanoid_mpc_msgs::LoopTiming timing = timingOf(remoteControl);
  EXPECT_LE(timing.overruns(), timing.cycles() / 100) << timing.overruns() << " overruns in " << timing.cycles() << " cycles";
  // A period lost to a late wake-up overruns nothing: the skipped periods are checked on their own.
  EXPECT_LE(timing.missed_periods(), timing.cycles() / 100) << timing.missed_periods() << " skipped periods in " << timing.cycles()
                                                            << " cycles (latest wake-up " << timing.max_lateness_s() << " s late)";
  EXPECT_NEAR(timing.mean_period_s(), timing.target_period_s(), 0.2 * timing.target_period_s());
  EXPECT_GE(timing.policy_age_s(), 0.0) << "a policy of the node is in use";
  EXPECT_LT(timing.policy_age_s(), 0.2) << "older than the link-loss timeout";
  EXPECT_GT(remoteControl.scenesReceived(), 0u) << "no viz/scene from the node";

  EXPECT_EQ(robotProcess.terminate(absl::Seconds(20)), 0) << "the robot process did not end cleanly on SIGTERM";
  EXPECT_EQ(mpcNode.terminate(absl::Seconds(20)), 0) << "the MPC node did not end cleanly on SIGTERM";
  remoteControl.stop();
}

}  // namespace
}  // namespace ocs2::humanoid
