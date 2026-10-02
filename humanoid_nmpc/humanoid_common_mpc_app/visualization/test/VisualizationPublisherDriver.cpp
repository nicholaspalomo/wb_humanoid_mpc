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

// Publishes the G1's viz/scene and viz/telemetry from synthetic data, through the visualization publisher, on a bus
// bound to an ephemeral loopback port, for test_end_to_end.py. Prints "PORT <n>" on stdout once the bus is up, then
// feeds robot/state samples at --sample_rate and a new policy every --policy_period until --duration has passed.

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "VisualizationTestRobot.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/NodeEndpoint.h"

ABSL_FLAG(absl::Duration, duration, absl::Seconds(60), "How long to publish; the test usually stops the driver earlier.");
ABSL_FLAG(double, sample_rate, /*default_value=*/100.0, "[Hz] robot/state samples fed to the publisher.");
ABSL_FLAG(absl::Duration, policy_period, absl::Milliseconds(50), "How often a new policy and observation are fed.");

namespace ocs2::humanoid::visualization {
namespace {

int run() {
  const std::unique_ptr<test::TestRobot> robot = test::TestRobot::load(test::g1CentroidalFiles(), test::Formulation::kCentroidal);
  robot::ipc::BusOptions options;
  options.nodeName = "visualization";
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = "visualization", .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  absl::StatusOr<std::unique_ptr<VisualizationPublisher>> publisher = VisualizationPublisher::Create(robot->model(), **bus);
  CHECK_OK(publisher.status());
  CHECK_OK((*bus)->start());
  CHECK_OK((*publisher)->start());
  std::cout << "PORT " << (*bus)->boundPort() << std::endl;

  const absl::Duration samplePeriod = absl::Seconds(1.0 / absl::GetFlag(FLAGS_sample_rate));
  const absl::Time start = absl::Now();
  const absl::Time end = start + absl::GetFlag(FLAGS_duration);
  absl::Time nextPolicy = start;
  CommandData command;
  PrimalSolution solution;
  for (absl::Time now = start; now < end; now = absl::Now()) {
    const scalar_t time = absl::ToDoubleSeconds(now - start);
    if (now >= nextPolicy) {
      robot->makePolicy(time, /*nodes=*/21, /*normalForce=*/300.0, vector2_t(0.02, 0.0), &command, &solution);
      (*publisher)->setPolicy(command, solution);
      (*publisher)->setObservation(command.mpcInitObservation_);
      nextPolicy += absl::GetFlag(FLAGS_policy_period);
    }
    (*publisher)->pushRobotState(robot->robotState(time, vector3_t(0.3 * time, 0.0, 0.75), vector3_t(0.0, 0.0, 0.1 * time)));
    absl::SleepFor(samplePeriod);
  }
  (*publisher)->stop();
  const VisualizationPublisher::Statistics statistics = (*publisher)->statistics();
  LOG(INFO) << "scenes " << statistics.scenesPublished << ", series " << statistics.telemetryPublished << ", samples dropped "
            << statistics.robotStatesDropped;
  (*bus)->stop();
  return 0;
}

}  // namespace
}  // namespace ocs2::humanoid::visualization

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  return ocs2::humanoid::visualization::run();
}
