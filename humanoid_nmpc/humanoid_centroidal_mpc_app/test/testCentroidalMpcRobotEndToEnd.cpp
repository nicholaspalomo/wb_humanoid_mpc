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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocs2_mpc_test/ScriptedMpc.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <robot_model/RobotDescription.h>

#include "humanoid_common_mpc_app/robot/test_support/ChildProcess.h"
#include "humanoid_common_mpc_app/robot/test_support/LoopbackNetwork.h"
#include "humanoid_common_mpc_app/robot/test_support/ScriptedOperator.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_msgs/fsm_state.pb.h"
#include "humanoid_mpc_msgs/loop_timing.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/NetworkConfig.h"

/*
 * The robot binary end to end, in the topology the hardware runs: humanoid_centroidal_mpc_robot is started as its own
 * process (the MuJoCo backend, headless, on the DRC Atlas files, the remote MPC link), and the test plays the laptop: an
 * MpcServer around a scripted MPC, and an operator, on a loopback network file of its own. The robot comes up in
 * ZERO_TORQUE on the gantry and streams observations; the operator takes it through JOINT_PD into WB_MPC; the policy of
 * the MPC - the robot held where it is, with one joint offset so that its action can be told from JOINT_PD's - reaches
 * the joint targets; the loop holds its period; and when the MPC side goes away the link is lost and the controller
 * holds the robot with the JOINT_PD action, while the mode stays WB_MPC. SIGTERM ends the process cleanly.
 */

namespace ocs2::humanoid {
namespace {

constexpr const char* kRobotBinary = "humanoid_nmpc/humanoid_centroidal_mpc_app/humanoid_centroidal_mpc_robot";
constexpr const char* kAtlasTask = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml";
constexpr const char* kAtlasReference = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml";
constexpr const char* kAtlasUrdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr const char* kAtlasScene = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
/** [rad] What the scripted MPC adds to the last joint of the model, so that its action differs from JOINT_PD's. */
constexpr double kJointOffset = 0.1;
/** The modes of a humanoid MPC: the contact combinations of its two feet. */
constexpr size_t kNumModes = 4;

using test_support::ChildProcess;
using test_support::createBus;
using test_support::ScriptedOperator;
using test_support::waitFor;

TEST(CentroidalMpcRobotEndToEnd, ReachesWbMpcAppliesThePolicyHoldsItsPeriodAndHoldsJointPdOnLinkLoss) {
  // The laptop's view of the model: the controller's dimensions, with no optimal control problem built.
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::CreateControllerModels(kAtlasTask, kAtlasUrdf, kAtlasReference);
  ASSERT_TRUE(interface.ok()) << interface.status();
  const MpcRobotModelBase<scalar_t>& model = (*interface)->getEffectiveMpcRobotModel();
  const ipc::ModelDimensions dimensions{.stateDim = model.getStateDim(), .inputDim = model.getInputDim(), .numModes = kNumModes};
  const std::string offsetJoint = (*interface)->modelSettings().mpcModelJointNames.back();
  const robot::model::RobotDescription description(kAtlasUrdf);
  const size_t offsetJointIndex = description.getJointIndex(offsetJoint);

  // A network file of the test's own: the robot, the MPC and the operator on free loopback ports.
  const std::string networkFile = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "network.textproto").string();
  const absl::StatusOr<robot::ipc::NetworkConfig> network =
      test_support::writeLoopbackNetworkFile(networkFile, {"robot", "mpc", "operator"});
  ASSERT_TRUE(network.ok()) << network.status();

  // The MPC side: a scripted MPC that holds the robot where it solved from, with the last joint offset.
  mpc::Settings settings;
  settings.timeHorizon_ = 1.0;
  settings.solutionTimeWindow_ = -1.0;
  std::unique_ptr<mpc_test::ScriptedMpc> mpc = std::make_unique<mpc_test::ScriptedMpc>(settings, dimensions.inputDim);
  // The JOINT_PD posture of the offset joint: the task file's initial state, which the robot starts in.
  const vector_t initialJointAngles = model.getJointAngles((*interface)->getInitialState());
  const double nominal = initialJointAngles(initialJointAngles.size() - 1);
  const Eigen::Index offsetIndex = static_cast<Eigen::Index>(dimensions.stateDim) - 1;
  mpc->solver().setPlan([offsetIndex, nominal](scalar_t /*time*/, scalar_t /*initTime*/, const vector_t& initState) {
    vector_t state = initState;
    state(offsetIndex) = nominal + kJointOffset;
    return state;
  });
  std::unique_ptr<robot::ipc::Bus> mpcBus = createBus(*network, "mpc");
  ipc::MpcServer::Config serverConfig;
  serverConfig.dimensions = dimensions;
  serverConfig.mpcDesiredFrequency = 50.0;
  absl::StatusOr<std::unique_ptr<ipc::MpcServer>> server = ipc::MpcServer::Create(
      *mpcBus, *mpc,
      [](const SystemObservation& observation) { return TargetTrajectories({observation.time}, {observation.state}, {observation.input}); },
      serverConfig);
  ASSERT_TRUE(server.ok()) << server.status();

  // The operator.
  ScriptedOperator remoteControl(*network);
  ASSERT_TRUE(mpcBus->start().ok());
  ASSERT_TRUE((*server)->start().ok());
  remoteControl.start();

  // The robot.
  ChildProcess robot({kRobotBinary, "--robot_name=drc_atlas", absl::StrCat("--task_file=", kAtlasTask),
                      absl::StrCat("--reference_file=", kAtlasReference), absl::StrCat("--urdf_file=", kAtlasUrdf),
                      absl::StrCat("--mjcf_file=", kAtlasScene), absl::StrCat("--network_config=", networkFile), "--headless",
                      "--realtime_cores=none", "--backend_cores=none"});
  ASSERT_TRUE(waitFor([&]() { return remoteControl.fsmState().has_value() || !robot.running(); }, absl::Seconds(60)));
  ASSERT_TRUE(robot.running()) << "the robot process exited on start-up";
  EXPECT_EQ(remoteControl.fsmState()->mode(), "ZERO_TORQUE");
  EXPECT_TRUE(remoteControl.fsmState()->gantry_locked());

  // The robot streams observations from ZERO_TORQUE on, and the MPC solves them.
  ASSERT_TRUE(waitFor([&]() { return (*server)->statistics().policiesPublished > 5; }, absl::Seconds(20)));

  ASSERT_TRUE(remoteControl.enterMode("JOINT_PD"));
  const uint64_t resetsBeforeEntry = (*server)->statistics().robotResetsServed;
  ASSERT_TRUE(remoteControl.enterMode("WB_MPC"));

  // The entry into WB_MPC resets the MPC from the robot as it is; the policy solved after it is ramped in, and the
  // offset joint's target leaves the JOINT_PD posture for the MPC's.
  EXPECT_TRUE(waitFor([&]() { return (*server)->statistics().robotResetsServed > resetsBeforeEntry; }, absl::Seconds(10)));
  // The offset joint's target: its JOINT_PD posture plus the scripted offset under the MPC's action (once the entry
  // ramp, mpcEntryBlendTime, is over), the posture itself under the JOINT_PD action.
  const std::function<std::optional<double>(absl::string_view)> offsetTarget = [&](absl::string_view mode) -> std::optional<double> {
    const std::optional<humanoid_mpc_msgs::RobotStateSample> sample = remoteControl.sample();
    if (!sample.has_value() || sample->control_mode() != mode) return std::nullopt;
    return sample->joint_position_targets(static_cast<int>(offsetJointIndex)) - nominal;
  };
  ASSERT_TRUE(waitFor(
      [&]() {
        const std::optional<double> offset = offsetTarget("WB_MPC");
        return offset.has_value() && std::abs(*offset - kJointOffset) < 0.01;
      },
      absl::Seconds(20)))
      << "the MPC's policy never reached the joint targets";
  ASSERT_TRUE(remoteControl.fsmState()->mpc_healthy());
  ASSERT_TRUE(remoteControl.timing().has_value());
  EXPECT_GE(remoteControl.timing()->policy_age_s(), 0.0) << "a policy of the MPC is in use";

  // The loop holds its period: a few overruns and skipped periods at most, over the whole run. A period lost to a late
  // wake-up overruns nothing, so the skipped periods are checked on their own.
  ASSERT_TRUE(waitFor([&]() { return remoteControl.timing().has_value() && remoteControl.timing()->cycles() > 1000; }, absl::Seconds(20)));
  const humanoid_mpc_msgs::LoopTiming timing = *remoteControl.timing();
  EXPECT_DOUBLE_EQ(timing.target_period_s(), 0.01) << "the shipped rate of the Atlas (mpc.mrtDesiredFrequency 100 Hz)";
  EXPECT_LE(timing.overruns(), timing.cycles() / 100) << timing.overruns() << " overruns in " << timing.cycles() << " cycles";
  EXPECT_LE(timing.missed_periods(), timing.cycles() / 100) << timing.missed_periods() << " skipped periods in " << timing.cycles()
                                                            << " cycles (latest wake-up " << timing.max_lateness_s() << " s late)";
  EXPECT_NEAR(timing.mean_period_s(), 0.01, 0.002);

  // The MPC side goes away: the link is lost after mpcLink.policyTimeout, the controller holds the robot with the
  // JOINT_PD action (the offset is gone from the targets), and the mode stays WB_MPC.
  (*server)->stop();
  mpcBus->stop();
  ASSERT_TRUE(waitFor([&]() { return !remoteControl.fsmState()->mpc_healthy(); }, absl::Seconds(10))) << "no link loss reported";
  EXPECT_EQ(remoteControl.fsmState()->mode(), "WB_MPC");
  EXPECT_TRUE(waitFor(
      [&]() {
        const std::optional<double> offset = offsetTarget("WB_MPC");
        return offset.has_value() && std::abs(*offset) < 1e-9;
      },
      absl::Seconds(10)))
      << "the robot was not held with the JOINT_PD action";

  EXPECT_EQ(robot.terminate(absl::Seconds(20)), 0) << "the robot process did not end cleanly on SIGTERM";
  remoteControl.stop();
}

}  // namespace
}  // namespace ocs2::humanoid
