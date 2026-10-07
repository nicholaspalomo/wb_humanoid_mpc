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
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/mrt/MpcLink.h"
#include "humanoid_common_mpc_app/node/DummySimLoop.h"
#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_ipc/Topics.h"
#include "humanoid_mpc_msgs/walking_velocity_command.pb.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/node/test/ScriptedRobot.h"
#include "humanoid_wb_mpc/WBMpcPreComputation.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "humanoid_wb_mpc/parameter_update/WholeBodyHotFieldAppliers.h"
#include "humanoid_wb_mpc_app/WBMpcNode.h"
#include "nproto/Textproto.h"
#include "robot_core/ResourcePaths.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/NodeEndpoint.h"
#include "robot_model/RobotDescription.h"

/*
 * The whole-body MPC node on the Unitree G1 task file, with the real SQP solver, served over loopback buses: to a robot
 * and an operator the test scripts, and to the dummy simulator for a few seconds. Builds the CppAD libraries of the G1
 * whole-body MPC on a cold cache (minutes); every node after the first loads them.
 */

namespace ocs2::humanoid {
namespace {

namespace topics = ::ocs2::humanoid::ipc::topics;
using node::test_support::ScriptedRobot;

node::MpcFiles g1Files() {
  node::MpcFiles files;
  // A file missing from the runfiles throws here, naming the data dependency to add (robot::resolveResourcePath).
  files.taskFile = robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto").value();
  files.referenceFile = robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto").value();
  files.urdfFile = robot::resolveResourcePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf").value();
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

/** The G1 whole-body MPC node, serving a scripted robot and operator. */
class NodeHarness {
 public:
  NodeHarness() : files_(g1Files()) {
    std::unique_ptr<robot::ipc::Bus> mpcBus = loopbackBus("mpc");
    robot_.connect(*mpcBus);
    absl::StatusOr<std::unique_ptr<WBMpcNode>> node = WBMpcNode::Create(files_, std::move(mpcBus), WBMpcNode::Options());
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
    observation.input = vector_t::Zero(node_->interface().getMpcRobotModel().getInputDim());
    observation.mode = ModeNumber::kStance;
    return observation;
  }

  std::optional<humanoid_mpc_msgs::MpcPolicy> solve(scalar_t time, uint64_t sequence, uint64_t requested = 0, uint64_t fullRequested = 0) {
    return robot_.solve(ScriptedRobot::observationMessage(standing(time), sequence, requested, fullRequested), /*timeoutSeconds=*/60.0);
  }

  const node::MpcFiles& files() const { return files_; }
  WBMpcNode& node() { return *node_; }
  ScriptedRobot& robot() { return robot_; }

 private:
  node::MpcFiles files_;
  ScriptedRobot robot_;
  std::unique_ptr<WBMpcNode> node_;
};

TEST(WBMpcNode, SolvesTheRobotsObservationsServesTheResetsTheyRequestAndDrawsThePolicies) {
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

TEST(WBMpcNode, TheVelocityCommandReachesTheMotionManager) {
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

/** The tuning GUI's update of the task file at `taskFile` with its state weights scaled by `stateWeightScale`. */
humanoid_mpc_config::MpcParameterUpdate stateWeightUpdate(const std::string& taskFile, double stateWeightScale) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> task = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(taskFile);
  EXPECT_TRUE(task.ok()) << task.status();
  humanoid_mpc_config::MpcParameterUpdate update;
  if (!task.ok()) return update;
  *update.mutable_task() = *task;
  update.mutable_task()->mutable_state_weights()->set_scaling(task->state_weights().scaling() * stateWeightScale);
  update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  update.set_config_path(configFileIdentity(taskFile));
  return update;
}

/** The weights of the quadratic state cost of `task`'s state_weights, on the whole-body coordinates of `interface`. */
matrix_t stateWeightsOf(const humanoid_mpc_config::TaskFile& task, const WBMpcInterface& interface) {
  mpc_config::TaskFile typed;
  EXPECT_TRUE(mpc_config::FromProto(task, &typed).ok());
  const absl::StatusOr<matrix_t> weights = stateWeightsFromConfig(
      typed.state_weights, stateInputLayout(interface.modelSettings(), StateInputLayout::Mpc::kWholeBody), "state_weights");
  EXPECT_TRUE(weights.ok()) << weights.status();
  return weights.ok() ? *weights : matrix_t();
}

/**
 * Expects every worker's running quadratic state cost to weigh the state by `expected` (its Hessian, at the task file's
 * initial state), and the swing foot's gains of every worker's pre-computation to be the model settings'.
 */
void expectRunningStateWeightsAndShippedFootGains(WBMpcNode& node, const matrix_t& expected) {
  const WBMpcInterface& interface = node.interface();
  const ModelSettings::FootConstraintConfig& shipped = interface.modelSettings().footConstraintConfig;
  const vector_t state = interface.getInitialState();
  const vector_t input = vector_t::Zero(static_cast<Eigen::Index>(interface.getMpcRobotModel().getInputDim()));
  // The solver thread waits for the next observation: the problems are not being solved.
  for (OptimalControlProblem& problem : node.mpc().getSolverPtr()->getOcpDefinitions()) {
    const StateInputCost& cost = problem.costPtr->get(kStateQuadraticCostTerm);
    const matrix_t hessian =
        cost.getQuadraticApproximation(/*time=*/0.0, state, input, interface.getReferenceManagerPtr()->getTargetTrajectories(),
                                       *problem.preComputationPtr)
            .dfdxx;
    EXPECT_TRUE(hessian == expected) << "the running state weights are not the expected ones";
    const WBMpcPreComputation& preComputation = cast<WBMpcPreComputation>(*problem.preComputationPtr);
    EXPECT_EQ(preComputation.getSwingFootGains().linearVelocityErrorGainZ, shipped.linearVelocityErrorGain_z);
    EXPECT_EQ(preComputation.getSwingFootGains().linearAccelerationErrorGainZ, shipped.linearAccelerationErrorGain_z);
    EXPECT_EQ(preComputation.getNormalVelocityPositionErrorGain(), shipped.positionErrorGain_z);
  }
}

TEST(WBMpcNode, AnMpcParameterUpdateIsAppliedBeforeTheNextSolve) {
  NodeHarness harness;
  ASSERT_TRUE(harness.solve(/*time=*/0.0, /*sequence=*/1).has_value());
  // The tuning GUI's whole task file of this robot and configuration, its state weights doubled.
  const humanoid_mpc_config::MpcParameterUpdate update = stateWeightUpdate(harness.files().taskFile, /*stateWeightScale=*/2.0);
  ASSERT_TRUE(harness.robot().sendAsOperator(topics::kOperatorMpcParameters, update,
                                             [&]() { return harness.node().runtime().statistics().parameterUpdatesReceived > 0; }));
  EXPECT_EQ(harness.node().runtime().statistics().parameterUpdatesRejected, 0u);
  // The update is applied by the solve after it arrived, before any worker runs.
  ASSERT_TRUE(harness.solve(/*time=*/0.02, /*sequence=*/2).has_value());
  expectRunningStateWeightsAndShippedFootGains(harness.node(), stateWeightsOf(update.task(), harness.node().interface()));
}

TEST(WBMpcNode, AnUpdateOfAnotherConfigurationIsRefused) {
  NodeHarness harness;
  ASSERT_TRUE(harness.solve(/*time=*/0.0, /*sequence=*/1).has_value());
  // The centroidal G1's file: the same robot_name, another configuration. The GUI stamps its config_path.
  const std::string centroidalTaskFile =
      robot::resolveResourcePath("robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto").value();
  const humanoid_mpc_config::MpcParameterUpdate update = stateWeightUpdate(centroidalTaskFile, /*stateWeightScale=*/2.0);
  ASSERT_EQ(update.task().model_settings().robot_name(), harness.node().interface().modelSettings().robotName);
  const node::MpcNodeRuntime::Statistics before = harness.node().runtime().statistics();
  ASSERT_TRUE(harness.robot().sendAsOperator(topics::kOperatorMpcParameters, update, [&]() {
    return harness.node().runtime().statistics().parameterUpdatesRejected > before.parameterUpdatesRejected;
  }));
  EXPECT_EQ(harness.node().runtime().statistics().parameterUpdatesRejected, before.parameterUpdatesRejected + 1);
  ASSERT_TRUE(harness.solve(/*time=*/0.02, /*sequence=*/2).has_value());
  // Nothing of it reached the problem: the weights and the gains are this file's.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> shipped =
      nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(harness.files().taskFile);
  ASSERT_TRUE(shipped.ok()) << shipped.status();
  expectRunningStateWeightsAndShippedFootGains(harness.node(), stateWeightsOf(*shipped, harness.node().interface()));
}

TEST(WBMpcNode, TheUpdaterAppliesTheWholeBodyHotFields) {
  NodeHarness harness;
  EXPECT_EQ(harness.node().parameterUpdater().appliedFields(), wholeBodyHotFieldNames());
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

TEST(WBMpcNode, ResetsToTheTargetTheControllerHandsItsInProcessLink) {
  NodeHarness harness;
  WBMpcInterface& interface = harness.node().interface();
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(harness.files().urdfFile);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  // The controller as the robot process builds it, with a link that keeps the reset target the controller hands it.
  mpc_test::ScriptedMpc scriptedMpc(interface.mpcSettings(), interface.getMpcRobotModel().getInputDim());
  MpcLink::ResetTargetFunction controllerResetTarget;
  const MpcLinkFactory capturing = [&](MpcLink::ResetTargetFunction resetTarget) -> std::unique_ptr<MpcLink> {
    controllerResetTarget = resetTarget;
    return std::make_unique<InProcessMpcLink>(scriptedMpc, std::move(resetTarget), InProcessMpcLink::Config());
  };
  const absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> controller =
      WBMpcMrtJointController::Create(description, interface.modelSettings(), capturing, interface.getPinocchioInterface());
  ASSERT_TRUE(controller.ok()) << controller.status();
  ASSERT_TRUE(controllerResetTarget);
  const size_t numCoordinates = interface.getMpcRobotModel().getGenCoordinatesDim();
  for (const scalar_t time : {0.0, 1.5, 12.25}) {
    SystemObservation observation = harness.standing(time);
    observation.state(0) += 0.2 * time;                       // base x
    observation.state(4) = 0.03;                              // pitch
    observation.state.tail(numCoordinates).setConstant(0.1);  // velocities
    expectSameTarget(controllerResetTarget(observation), harness.node().resetTargetTrajectories(observation));
  }
}

/** The dummy simulator against the node, on a bus of its own, playing the robot; run() on a thread of its own. */
class DummySimRunner {
 public:
  DummySimRunner(WBMpcNode& node, robot::ipc::Bus& mpcBus) {
    std::unique_ptr<robot::ipc::Bus> robotBus = loopbackBus("robot");
    EXPECT_TRUE(robotBus->connect(mpcBus.boundEndpoint()).ok());
    EXPECT_TRUE(mpcBus.connect(robotBus->boundEndpoint()).ok());
    node::DummySimLoop::Config config;
    config.simulationFrequency = 80.0;  // as humanoid_wb_mpc_dummy_sim
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

TEST(WBMpcNode, TheDummySimulatorStandsOnTheNodesPoliciesWithoutAReset) {
  const node::MpcFiles files = g1Files();
  std::unique_ptr<robot::ipc::Bus> mpcBus = loopbackBus("mpc");
  robot::ipc::Bus* absl_nonnull mpcBusPointer = mpcBus.get();
  absl::StatusOr<std::unique_ptr<WBMpcNode>> node = WBMpcNode::Create(files, std::move(mpcBus), WBMpcNode::Options());
  ASSERT_TRUE(node.ok()) << node.status();
  DummySimRunner dummySim(**node, *mpcBusPointer);
  ASSERT_TRUE((*node)->start().ok());
  ASSERT_TRUE(dummySim.runFor(absl::Seconds(4)).ok());
  (*node)->stop();

  const node::DummySimLoop::Statistics statistics = dummySim.loop().statistics();
  const ipc::MpcServer::Statistics server = (*node)->runtime().statistics().server;
  // Policies flowed: synchronized with the MPC, one per update of the plant.
  EXPECT_GT(statistics.steps, 20);
  EXPECT_GT(statistics.synchronizedPolicies, 10);
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

}  // namespace
}  // namespace ocs2::humanoid
