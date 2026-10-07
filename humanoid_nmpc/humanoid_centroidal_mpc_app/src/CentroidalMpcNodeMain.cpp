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

#include <pthread.h>

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

#include "humanoid_centroidal_mpc_app/CentroidalMpcNode.h"
#include "humanoid_common_mpc/common/ThreadAffinity.h"
#include "humanoid_common_mpc_app/node/MpcAppFlags.h"
#include "humanoid_common_mpc_app/node/MpcFiles.h"
#include "humanoid_common_mpc_app/node/MpcNodeRuntime.h"
#include "humanoid_common_mpc_app/node/NodeBus.h"
#include "humanoid_common_mpc_app/node/ShutdownSignal.h"
#include "robot_ipc/Bus.h"

// The default of --realtime_priority: the solver thread stays on the time-sharing scheduler.
constexpr int kTimeSharingScheduler = 0;

// LINT.IfChange(node_flags)
ABSL_FLAG(std::string, ipc_node, "mpc", "The bus node this process publishes as (a node of the network file).");
ABSL_FLAG(int,
          realtime_priority,
          kTimeSharingScheduler,
          "SCHED_FIFO priority of the MPC solver thread, 1-99; 0 keeps it on the time-sharing scheduler.");
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc_app/README.md:node_flags, //humanoid_nmpc/humanoid_wb_mpc_app/src/WBMpcNodeMain.cpp:node_flags)
// clang-format on

/*
 * humanoid_centroidal_mpc_node: the centroidal MPC of a robot, served on the bus to the robot process (or to the dummy
 * simulator) until SIGINT or SIGTERM. See humanoid_nmpc/humanoid_centroidal_mpc_app/README.md.
 */
int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::SetProgramUsageMessage(
      "The centroidal MPC node: solves the robot's observations (robot/mpc_observation) and publishes the policies "
      "(mpc/policy, mpc/status) on the bus.\n  humanoid_centroidal_mpc_node --robot_name=drc_atlas --task_file=... "
      "--reference_file=... --urdf_file=... --gait_file=... [--network_config=...] [--ipc_node=mpc] [--realtime_priority=0]");
  absl::ParseCommandLine(argc, argv);
  // Without InitializeLog() Abseil warns once and writes everything to stderr anyway; with it the default stderr
  // threshold is ERROR, so the INFO records have to be asked for.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  // Before any thread starts, so that every thread inherits the signal handling.
  ocs2::humanoid::node::installShutdownSignalHandlers();

  const ocs2::humanoid::node::MpcFiles files = ocs2::humanoid::node::mpcFilesFromFlags();
  if (const absl::Status valid = ocs2::humanoid::node::validateMpcFiles(files); !valid.ok()) {
    LOG(ERROR) << valid.message();
    return 2;
  }
  const std::string robotName = absl::GetFlag(FLAGS_robot_name);
  LOG(INFO) << "[humanoid_centroidal_mpc_node] " << (robotName.empty() ? std::string("robot") : robotName) << ": task file "
            << files.taskFile;

  // Every thread this process starts from here on (the SQP solver's worker threads, the bus's IO thread) inherits the MPC
  // cores, so that none of them competes with the realtime loop of a robot process on the same machine.
  const ocs2::humanoid::SystemCoreAllocation cores = ocs2::humanoid::getDefaultCoreAllocation();
  ocs2::humanoid::setThreadCpuAffinity(cores.mpcCores, pthread_self(), "MPC node");

  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus =
      ocs2::humanoid::node::createNodeBus(absl::GetFlag(FLAGS_network_config), absl::GetFlag(FLAGS_ipc_node));
  if (!bus.ok()) {
    LOG(ERROR) << "The bus: " << bus.status();
    return 2;
  }

  ocs2::humanoid::CentroidalMpcNode::Options options;
  options.solverThread = ocs2::humanoid::node::defaultSolverThreadConfig(absl::GetFlag(FLAGS_realtime_priority));
  // The node also publishes viz/scene and viz/telemetry for the Rerun bridge (humanoid_common_mpc_app/visualization), from
  // a thread of its own at a raised nice value, on the MPC cores it inherits from this thread.
  absl::StatusOr<std::unique_ptr<ocs2::humanoid::CentroidalMpcNode>> node =
      ocs2::humanoid::CentroidalMpcNode::Create(files, *std::move(bus), std::move(options));
  if (!node.ok()) {
    LOG(ERROR) << "The centroidal MPC: " << node.status();
    return 1;
  }
  if (const absl::Status started = (*node)->start(); !started.ok()) {
    LOG(ERROR) << "Starting the MPC node: " << started;
    return 1;
  }
  LOG(INFO) << "[humanoid_centroidal_mpc_node] Serving the MPC as bus node \"" << absl::GetFlag(FLAGS_ipc_node)
            << "\"; waiting for the robot's observations.";
  ocs2::humanoid::node::waitForShutdown();
  LOG(INFO) << "[humanoid_centroidal_mpc_node] Shutting down.";
  (*node)->stop();
  return 0;
}
