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

// The shipped whole-body G1 MPC on its quaternion state, with the SQP's
// feedback policy on (multiple_shooting. use_feedback_policy, which the shipped
// file leaves off), over the network MPC link as the deployment runs it: an
// MpcServer serves the SQP's ManifoldLinearController over a loopback bus to a
// RemoteMpcLink on the state manifold the robot process builds for itself
// (stateManifoldOf(), WBMpcRobotMain.cpp): the whole body's [E(3), unit
// quaternion at ambient offset 3, E(2 n_j + 6)], where the centroidal test
// covers the centroidal state's (offset 9). The same MPC solved in process
// (MPC_MRT_Interface with the interface's manifold) is the reference: the
// policy the robot decodes is CONTROLLER_TYPE_MANIFOLD_LINEAR, node for node
// the in-process one, and the robot's realtime evaluation of it
// (RealtimePolicyEvaluator) computes what MRT_BASE::evaluatePolicy() computes
// in process, on both sides of a contact event, bit for bit. Bitwise is what
// holds: both stacks solve the same problem from the same observation on one
// SQP thread (multiple_shooting.n_threads: 1, so that even the line search's
// per-thread sums are fixed), and the link carries the doubles of the
// controller's nodes unchanged (policyToProto(), policyFromProto()).
//
// Builds the CppAD libraries of the whole-body G1 MPC on a cold cache
// (minutes); the second stack loads them.

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/MPC_MRT_Interface.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_sqp/SqpMpc.h"
#include "ocs2_sqp/SqpSettings.h"

#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/MpcMessageConversions.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/mrt/WBMpcResetTarget.h"
#include "robot_core/ResourcePaths.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid {
namespace {

/** [s] Where the inserted steps start: the left foot alone for 0.35 s, then
 * both for 0.35 s, inside the 1 s horizon. */
constexpr scalar_t kStepStartTime = 0.3;
/** How long a wait for the whole-body solves may take on a loaded machine: one
 * solve is tens of milliseconds. */
constexpr absl::Duration kSolveTimeout = absl::Seconds(120);

/** The shipped whole-body G1 MPC and the SQP MPC the node builds on it, on one
 * solver thread, the feedback policy on. */
struct WholeBodyMpc {
  std::unique_ptr<WBMpcInterface> interface;
  std::unique_ptr<SqpMpc> mpc;
};

WholeBodyMpc makeWholeBodyMpc() {
  WholeBodyMpc wholeBody;
  // A file missing from the runfiles ends the test here, naming the data
  // dependency to add (robot::resolveResourcePath).
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> interface =
      WBMpcInterface::Create(
          robot::resolveResourcePath(
              "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto")
              .value(),
          robot::resolveResourcePath(
              "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf")
              .value(),
          robot::resolveResourcePath("robot_models/unitree_g1/g1_wb_mpc/config/"
                                     "command/reference.textproto")
              .value());
  EXPECT_TRUE(interface.ok()) << interface.status();
  if (!interface.ok()) return wholeBody;
  wholeBody.interface = *std::move(interface);
  sqp::Settings settings = wholeBody.interface->sqpSettings();
  settings.nThreads = 1;
  settings.useFeedbackPolicy = true;
  wholeBody.mpc =
      std::make_unique<SqpMpc>(wholeBody.interface->mpcSettings(), settings,
                               wholeBody.interface->getOptimalControlProblem(),
                               wholeBody.interface->getInitializer());
  wholeBody.mpc->getSolverPtr()->setReferenceManager(
      wholeBody.interface->getReferenceManagerPtr());
  return wholeBody;
}

/**
 * Inserts steps into the gait schedule of `interface`, so that the policy has a
 * contact event inside its horizon: from kStepStartTime the left foot alone,
 * then both feet. Before every solve, after the reset that restores the shipped
 * gait (SwitchedModelReferenceManager::reset()), on both stacks alike: the same
 * schedule each time.
 */
void insertSteps(WBMpcInterface& interface) {
  interface.getSwitchedModelReferenceManagerPtr()
      ->getGaitSchedule()
      ->insertModeSequenceTemplate(
          ModeSequenceTemplate({0.0, 0.35, 0.7},
                               {ModeNumber::kLf, ModeNumber::kStance}),
          /*startTime=*/kStepStartTime, /*finalTime=*/2.0);
}

/** The task file's standing state at t = 0, as the robot observes it before its
 * first policy. */
SystemObservation standingObservation(const WBMpcInterface& interface) {
  SystemObservation observation;
  observation.time = 0.0;
  observation.state = interface.getInitialState();
  observation.input = vector_t::Zero(
      static_cast<Eigen::Index>(interface.getMpcRobotModel().getInputDim()));
  observation.mode = ModeNumber::kStance;
  return observation;
}

/** The reset target the node resets the MPC to (wbMpcResetTargetTrajectories(),
 * as WBMpcNode's server does). */
TargetTrajectories resetTargetsOf(WBMpcInterface& interface,
                                  const SystemObservation& observation) {
  return wbMpcResetTargetTrajectories(observation, interface.getMpcRobotModel(),
                                      interface.getPinocchioInterface());
}

/** The dimensions of the link and the server: the model's state and inputs, the
 * humanoid's four modes. */
ipc::ModelDimensions dimensionsOf(const WBMpcInterface& interface) {
  const MpcRobotModelBase<scalar_t>& model = interface.getMpcRobotModel();
  return {.stateDim = model.getStateDim(),
          .inputDim = model.getInputDim(),
          .numModes = ipc::test_support::kNumModes};
}

/** The time of the first event of `schedule` strictly inside (startTime,
 * finalTime) that changes the contacts; empty without one. */
std::optional<scalar_t> firstContactEventIn(const ModeSchedule& schedule,
                                            scalar_t startTime,
                                            scalar_t finalTime) {
  for (size_t event = 0; event < schedule.eventTimes.size(); ++event) {
    const scalar_t time = schedule.eventTimes[event];
    if (time > startTime && time < finalTime &&
        schedule.modeSequence[event] != schedule.modeSequence[event + 1])
      return time;
  }
  return std::nullopt;
}

/** What a policy evaluates to at one time and state. */
struct Evaluation {
  vector_t state;
  vector_t input;
  size_t mode = 0;
};

/** MRT_BASE::evaluatePolicy() of `mrt` at `time` and `state`. */
Evaluation evaluateInProcess(MRT_BASE& mrt,
                             scalar_t time,
                             const vector_t& state) {
  Evaluation evaluation;
  mrt.evaluatePolicy(time, state, evaluation.state, evaluation.input,
                     evaluation.mode);
  return evaluation;
}

/** The robot's realtime evaluation of `policy` at `time` and `state`: a test
 * failure when it refuses. */
Evaluation evaluateOnTheRobot(ipc::RealtimePolicyEvaluator& evaluator,
                              const PrimalSolution& policy,
                              scalar_t time,
                              const vector_t& state) {
  Evaluation evaluation;
  EXPECT_EQ(evaluator.evaluate(policy, time, state, evaluation.state,
                               evaluation.input, evaluation.mode),
            ipc::RealtimePolicyEvaluator::Outcome::kEvaluated)
      << "t = " << time;
  return evaluation;
}

void expectSameEvaluation(const Evaluation& robot,
                          const Evaluation& inProcess,
                          scalar_t time) {
  SCOPED_TRACE(::testing::Message() << "t = " << time);
  EXPECT_EQ(robot.state, inProcess.state);
  EXPECT_EQ(robot.input, inProcess.input);
  EXPECT_EQ(robot.mode, inProcess.mode);
}

TEST(WBMpcManifoldPolicyOverBus,
     TheRobotEvaluatesTheDecodedManifoldPolicyAsTheInProcessMrtDoes) {
  WholeBodyMpc inProcess = makeWholeBodyMpc();
  WholeBodyMpc remote = makeWholeBodyMpc();
  ASSERT_NE(inProcess.mpc, nullptr);
  ASSERT_NE(remote.mpc, nullptr);
  const SystemObservation observation =
      standingObservation(*inProcess.interface);
  const ipc::ModelDimensions dimensions = dimensionsOf(*inProcess.interface);

  // The robot process's side, as WBMpcRobotMain.cpp builds it: the link on a
  // manifold of the robot's own model.
  const absl::StatusOr<std::shared_ptr<const StateManifold>> robotManifold =
      stateManifoldOf(remote.interface->getMpcRobotModel());
  ASSERT_TRUE(robotManifold.ok()) << robotManifold.status();
  ASSERT_EQ((*robotManifold)->getAmbientDim(),
            (*robotManifold)->getTangentDim() + 1)
      << "one unit quaternion in the state";
  ASSERT_EQ(remote.interface->getMpcRobotModel().getBaseOrientationStartIndex(),
            3U)
      << "the whole body's quaternion follows p_W";
  std::unique_ptr<robot::ipc::Bus> robotBus =
      ipc::test_support::createNodeBus("robot");
  std::unique_ptr<robot::ipc::Bus> mpcBus =
      ipc::test_support::createNodeBus("mpc");
  MpcResetSupervisor supervisor;
  ipc::RemoteMpcLink::Config linkConfig;
  linkConfig.dimensions = dimensions;
  linkConfig.stateManifold = *robotManifold;
  absl::StatusOr<std::unique_ptr<ipc::RemoteMpcLink>> link =
      ipc::RemoteMpcLink::Create(robotBus.get(), &supervisor, linkConfig);
  ASSERT_TRUE(link.ok()) << link.status();
  ipc::MpcServer::Config serverConfig;
  serverConfig.dimensions = dimensions;
  WBMpcInterface& remoteInterface = *remote.interface;
  ipc::MpcServer::Hooks hooks;
  hooks.preSolveObserver =
      [&remoteInterface](const SystemObservation& /*observation*/) {
        insertSteps(remoteInterface);
      };
  absl::StatusOr<std::unique_ptr<ipc::MpcServer>> server =
      ipc::MpcServer::Create(
          mpcBus.get(), remote.mpc.get(),
          [&remoteInterface](const SystemObservation& resetObservation) {
            return resetTargetsOf(remoteInterface, resetObservation);
          },
          serverConfig, std::move(hooks));
  ASSERT_TRUE(server.ok()) << server.status();
  ASSERT_TRUE(robotBus->start().ok());
  ASSERT_TRUE(mpcBus->start().ok());
  ASSERT_TRUE((*server)->start().ok());

  // The first observations may go out before the subscriptions have reached the
  // publishers: the same observation, until a policy solved from it arrives,
  // then until the server has published every solve and the link has received
  // it.
  ASSERT_TRUE(ipc::test_support::waitFor(
      [&]() {
        (*link)->setCurrentObservation(observation);
        absl::SleepFor(absl::Milliseconds(5));
        return (*link)->statistics().policiesAccepted > 0;
      },
      kSolveTimeout));
  ASSERT_TRUE(ipc::test_support::waitFor(
      [&]() {
        const ipc::MpcServer::Statistics served = (*server)->statistics();
        return served.statusesPublished == served.solveAttempts &&
               (*link)->statistics().policiesReceived ==
                   served.policiesPublished;
      },
      kSolveTimeout));
  (*server)->stop();
  const ipc::MpcServer::Statistics served = (*server)->statistics();
  ASSERT_EQ(served.failedAttempts, 0u);
  ASSERT_EQ((*link)->statistics().invalidPoliciesRejected, 0u)
      << "the robot refused a policy of the node";
  ASSERT_EQ((*link)->statistics().policiesAccepted, served.policiesPublished);
  ASSERT_TRUE((*link)->updatePolicy());
  const PrimalSolution& decoded = (*link)->getPolicy();

  // The node's solves, in process: as many solves as the server ran, all from
  // the observation, after the start-up reset.
  MPC_MRT_Interface mrt(*inProcess.mpc);
  mrt.setStateManifold(inProcess.interface->getStateManifold());
  WBMpcInterface& inProcessInterface = *inProcess.interface;
  mrt.setSolverObservationObserver(
      [&inProcessInterface](const SystemObservation& /*observation*/) {
        insertSteps(inProcessInterface);
      });
  mrt.setCurrentObservation(observation);
  mrt.resetMpcNode(resetTargetsOf(*inProcess.interface, observation));
  for (uint64_t solve = 0; solve < served.solveAttempts; ++solve)
    ASSERT_TRUE(mrt.advanceMpc().ok()) << "solve " << solve;
  ASSERT_TRUE(mrt.updatePolicy());
  const PrimalSolution& solved = mrt.getPolicy();

  // The robot holds the node's ManifoldLinearController, node for node.
  ASSERT_NE(decoded.controllerPtr_, nullptr);
  ASSERT_NE(solved.controllerPtr_, nullptr);
  EXPECT_EQ(solved.controllerPtr_->getType(), ControllerType::MANIFOLD_LINEAR);
  EXPECT_EQ(decoded.controllerPtr_->getType(), ControllerType::MANIFOLD_LINEAR);
  EXPECT_EQ(decoded.timeTrajectory_, solved.timeTrajectory_);
  EXPECT_EQ(decoded.stateTrajectory_, solved.stateTrajectory_);
  EXPECT_EQ(decoded.inputTrajectory_, solved.inputTrajectory_);
  EXPECT_EQ(decoded.postEventIndices_, solved.postEventIndices_);
  ASSERT_FALSE(decoded.postEventIndices_.empty())
      << "the policy has no contact event";
  const scalar_t finalTime = decoded.timeTrajectory_.back();
  const std::optional<scalar_t> firstEvent =
      firstContactEventIn(decoded.modeSchedule_, observation.time, finalTime);
  if (!firstEvent.has_value()) {
    FAIL() << "no contact event in the plan's horizon: "
           << decoded.modeSchedule_;
  }
  const scalar_t eventTime = *firstEvent;
  LOG(INFO) << "The plan's mode schedule: " << decoded.modeSchedule_
            << "; evaluated around the event at t = " << eventTime << " s.";

  // Off the plan, so that the feedback acts: the observed state moved along the
  // tangent.
  const StateManifold& manifold = **robotManifold;
  vector_t offPlan;
  manifold.retract(
      observation.state,
      vector_t::LinSpaced(static_cast<Eigen::Index>(manifold.getTangentDim()),
                          -0.01, 0.012),
      /*alpha=*/1.0, offPlan);
  ipc::RealtimePolicyEvaluator evaluator(dimensions, &manifold);
  for (const scalar_t time : {0.001, eventTime - 0.001, eventTime,
                              eventTime + 0.001, 0.5 * finalTime}) {
    expectSameEvaluation(evaluateOnTheRobot(evaluator, decoded, time, offPlan),
                         evaluateInProcess(mrt, time, offPlan), time);
    // The link evaluates as the robot does too (MRT_BASE::evaluatePolicy() with
    // the link's manifold).
    expectSameEvaluation(evaluateOnTheRobot(evaluator, decoded, time, offPlan),
                         evaluateInProcess(**link, time, offPlan), time);
  }
  // Positive controls: the feedback changes the input off the plan, and the
  // input changes across the event.
  EXPECT_NE(
      evaluateOnTheRobot(evaluator, decoded, /*time=*/0.001, offPlan).input,
      evaluateOnTheRobot(evaluator, decoded, /*time=*/0.001, observation.state)
          .input);
  EXPECT_NE(
      evaluateOnTheRobot(evaluator, decoded, eventTime - 0.001, offPlan).input,
      evaluateOnTheRobot(evaluator, decoded, eventTime + 0.001, offPlan).input);
}

}  // namespace
}  // namespace ocs2::humanoid
