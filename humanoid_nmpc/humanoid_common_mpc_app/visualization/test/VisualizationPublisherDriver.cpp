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

// Publishes the G1's viz/scene and viz/telemetry from synthetic data, through the visualization publisher, as node
// "visualization" of the network file --network_config (the test's own DDS domain), for test_end_to_end.py. Prints
// "READY" on stdout once the bus is up, then feeds robot/state samples at --sample_rate and a new policy every
// --policy_period until --duration has passed.

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ocs2_core/constraint/LinearStateInputConstraint.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/cost/QuadraticStateInputCost.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc_app/visualization/VisualizationPublisher.h"
#include "humanoid_nmpc/humanoid_common_mpc_app/visualization/test/VisualizationTestRobot.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/NetworkConfig.h"
#include "web_viewer/ViewerThreadPriority.h"

namespace {

/** [Hz] The default of --sample_rate. */
constexpr double kDefaultSampleRate = 100.0;

}  // namespace

ABSL_FLAG(std::string, network_config, "", "The network file the test wrote, with the node \"visualization\".");
ABSL_FLAG(absl::Duration, duration, absl::Seconds(60), "How long to publish; the test usually stops the driver earlier.");
ABSL_FLAG(double, sample_rate, kDefaultSampleRate, "[Hz] robot/state samples fed to the publisher.");
ABSL_FLAG(absl::Duration, policy_period, absl::Milliseconds(50), "How often a new policy and observation are fed.");

namespace ocs2::humanoid::visualization {
namespace {

/**
 * Adds a stand-in for the solver's problem to `problem`: a term in a cost, a final cost and a constraint collection, so
 * that the mpc_solution groups carry series (the node hands over its solver's own problem).
 */
absl::Status addStandInTerms(Eigen::Index stateDim, Eigen::Index inputDim, OptimalControlProblem& problem) {
  try {  // OCS2's Collection::add() reports a duplicate term name by throwing.
    problem.costPtr->add("tracking", std::make_unique<QuadraticStateInputCost>(matrix_t::Identity(stateDim, stateDim),
                                                                               matrix_t::Identity(inputDim, inputDim)));
    problem.finalCostPtr->add("terminal", std::make_unique<QuadraticStateCost>(matrix_t::Identity(stateDim, stateDim)));
    problem.equalityConstraintPtr->add(
        "eq", std::make_unique<LinearStateInputConstraint>(vector_t::Zero(1), matrix_t::Ones(1, stateDim), matrix_t::Zero(1, inputDim)));
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat("adding the stand-in terms threw: ", e.what()));
  }
  return absl::OkStatus();
}

int run() {
  const std::unique_ptr<test::TestRobot> robot = test::TestRobot::load(test::g1CentroidalFiles(), test::Formulation::kCentroidal);
  robot::ipc::BusOptions options;
  options.nodeName = "visualization";
  absl::StatusOr<robot::ipc::NetworkConfig> network = robot::ipc::loadNetworkConfig(absl::GetFlag(FLAGS_network_config));
  CHECK_OK(network.status());
  options.network = *std::move(network);
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  // The driver shares the machine with other tests: its publisher keeps the driver's scheduling (the node's is SCHED_IDLE).
  VisualizationPublisher::Options publisherOptions;
  publisherOptions.threadPriority = robot::web_viewer::ViewerThreadPriority::kInherited;
  OptimalControlProblem problem;
  CHECK_OK(addStandInTerms(static_cast<Eigen::Index>(robot->robotModel().getStateDim()),
                           static_cast<Eigen::Index>(robot->robotModel().getInputDim()), problem));
  VisualizationModel model = robot->model();
  model.solverProblem = &problem;
  absl::StatusOr<std::unique_ptr<VisualizationPublisher>> publisher = VisualizationPublisher::Create(model, **bus, publisherOptions);
  CHECK_OK(publisher.status());
  CHECK_OK((*bus)->start());
  CHECK_OK((*publisher)->start());
  // Flushed at once: test_end_to_end.py reads the line before the driver writes anything else.
  std::cout << "READY" << '\n' << std::flush;

  const absl::Duration samplePeriod = absl::Seconds(1.0 / absl::GetFlag(FLAGS_sample_rate));
  const absl::Time start = absl::Now();
  const absl::Time end = start + absl::GetFlag(FLAGS_duration);
  absl::Time nextPolicy = start;
  CommandData command;
  PrimalSolution solution;
  const VisualizationPublisher::PostSolveObserver observer = (*publisher)->postSolveObserver();
  PerformanceIndex performance;
  for (absl::Time now = start; now < end; now = absl::Now()) {
    const scalar_t time = absl::ToDoubleSeconds(now - start);
    if (now >= nextPolicy) {
      robot->makePolicy(time, /*nodes=*/21, /*normalForce=*/300.0, vector2_t(0.02, 0.0), &command, &solution);
      performance.merit = time;
      CHECK_OK(observer(command, solution, performance));
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

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  return ocs2::humanoid::visualization::run();
}
