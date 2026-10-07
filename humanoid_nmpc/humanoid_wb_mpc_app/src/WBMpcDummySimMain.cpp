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

#include <memory>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_app/node/DummySimLoop.h"
#include "humanoid_common_mpc_app/node/MpcAppFlags.h"
#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/node/NodeBus.h"
#include "humanoid_common_mpc_app/node/ShutdownSignal.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "robot_ipc/Bus.h"

// LINT.IfChange(dummy_sim_flags)
ABSL_FLAG(std::string, ipc_node, "robot", "The bus node this process publishes as: it plays the robot.");
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc_app/README.md:dummy_sim_flags, //humanoid_nmpc/humanoid_centroidal_mpc_app/src/CentroidalMpcDummySimMain.cpp:dummy_sim_flags)
// clang-format on

/*
 * humanoid_wb_mpc_dummy_sim: the robot played by the whole-body MPC's own model, rolled out under the policies
 * of the MPC node (humanoid_wb_mpc_node) over the bus, until SIGINT or SIGTERM. See
 * humanoid_nmpc/humanoid_wb_mpc_app/README.md.
 */
int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::SetProgramUsageMessage(
      "The whole-body MPC's dummy simulator: plays the robot on the bus (robot/mpc_observation) with the MPC model as the "
      "plant.\n  humanoid_wb_mpc_dummy_sim --robot_name=g1 --task_file=... --reference_file=... --urdf_file=... "
      "[--network_config=...] [--ipc_node=robot]");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  ocs2::humanoid::node::installShutdownSignalHandlers();

  // The gait file is the MPC node's; the plant does not need it.
  const ocs2::humanoid::node::MpcFiles files = ocs2::humanoid::node::mpcFilesFromFlags();
  for (const absl::Status& valid : {ocs2::humanoid::node::validateFileFlag("--task_file", files.taskFile),
                                    ocs2::humanoid::node::validateFileFlag("--reference_file", files.referenceFile),
                                    ocs2::humanoid::node::validateFileFlag("--urdf_file", files.urdfFile)}) {
    if (!valid.ok()) {
      LOG(ERROR) << valid.message();
      return 2;
    }
  }

  // The plant: the rollout of the MPC's model.
  absl::StatusOr<std::unique_ptr<ocs2::humanoid::WBMpcInterface>> interface =
      ocs2::humanoid::WBMpcInterface::Create(files.taskFile, files.urdfFile, files.referenceFile);
  if (!interface.ok()) {
    LOG(ERROR) << "The whole-body MPC model: " << interface.status();
    return 1;
  }
  const ocs2::humanoid::MpcRobotModelBase<ocs2::scalar_t>& model = (*interface)->getMpcRobotModel();

  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus =
      ocs2::humanoid::node::createNodeBus(absl::GetFlag(FLAGS_network_config), absl::GetFlag(FLAGS_ipc_node));
  if (!bus.ok()) {
    LOG(ERROR) << "The bus: " << bus.status();
    return 2;
  }
  ocs2::humanoid::node::DummySimLoop::Config config;
  // The whole-body dummy of the ROS era stepped at 80 Hz (the centroidal one at 100 Hz).
  config.simulationFrequency = 80.0;
  config.mpcDesiredFrequency = (*interface)->mpcSettings().mpcDesiredFrequency_;
  config.link.dimensions = {
      .stateDim = model.getStateDim(), .inputDim = model.getInputDim(), .numModes = ocs2::humanoid::node::kNumHumanoidModes};
  absl::StatusOr<std::unique_ptr<ocs2::humanoid::node::DummySimLoop>> loop =
      ocs2::humanoid::node::DummySimLoop::Create(*std::move(bus), (*interface)->getRollout(), config);
  if (!loop.ok()) {
    LOG(ERROR) << "The dummy simulator: " << loop.status();
    return 1;
  }

  ocs2::SystemObservation initialObservation;
  initialObservation.time = 0.0;
  initialObservation.state = (*interface)->getInitialState();
  initialObservation.input = ocs2::vector_t::Zero(model.getInputDim());
  initialObservation.mode = ocs2::humanoid::ModeNumber::kStance;
  const absl::Status ran = (*loop)->run(initialObservation, []() { return ocs2::humanoid::node::shutdownRequested(); });
  if (!ran.ok()) {
    LOG(ERROR) << ran;
    return 1;
  }
  return 0;
}
