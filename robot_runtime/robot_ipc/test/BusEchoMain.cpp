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

// The C++ half of //robot_runtime/robot_ipc:cross_language_test. It binds an ephemeral loopback port, connects its
// subscriber to the --peer endpoint (the Python bus), and publishes every TestSample of --input_topic back on
// --output_topic with origin set to kOrigin. It prints "READY <its endpoint>" once it runs, and when its standard input
// closes it stops and prints "STATS received=<n> delivered=<n> rejected=<n>" for --input_topic.

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NodeEndpoint.h"
#include "robot_ipc/TopicStatistics.h"
#include "robot_ipc_test/test_sample.pb.h"

ABSL_FLAG(std::string, peer, "", "the endpoint of the bus to answer, e.g. tcp://127.0.0.1:41234");
ABSL_FLAG(std::string, input_topic, "test/to_cpp", "the topic of the samples to echo");
ABSL_FLAG(std::string, output_topic, "test/from_cpp", "the topic the echoes go out on");

namespace {

// LINT.IfChange(echo_origin)
constexpr char kOrigin[] = "robot_ipc_bus_echo";
// LINT.ThenChange(//robot_runtime/robot_ipc/test/test_cross_language.py:echo_origin)

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string peer = absl::GetFlag(FLAGS_peer);
  const std::string inputTopic = absl::GetFlag(FLAGS_input_topic);
  const std::string outputTopic = absl::GetFlag(FLAGS_output_topic);
  if (peer.empty()) {
    LOG(ERROR) << "--peer is required";
    return 2;
  }

  robot::ipc::BusOptions options;
  options.nodeName = "echo";
  options.network.nodes = {robot::ipc::NodeEndpoint{.name = "echo", .host = "127.0.0.1", .port = robot::ipc::kEphemeralPort}};
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> created = robot::ipc::Bus::Create(std::move(options));
  if (!created.ok()) {
    LOG(ERROR) << created.status();
    return 1;
  }
  robot::ipc::Bus& bus = **created;
  absl::Status status = bus.connect(peer);
  if (status.ok()) {
    status = bus.subscribe<robot_ipc_test::TestSample>(inputTopic, robot::ipc::Delivery::kAll,
                                                       [&bus, &outputTopic](const robot_ipc_test::TestSample& sample) {
                                                         robot_ipc_test::TestSample echo = sample;
                                                         echo.set_origin(kOrigin);
                                                         const absl::Status published = bus.publishFromIoThread(outputTopic, echo);
                                                         if (!published.ok()) {
                                                           LOG(ERROR) << published;
                                                         }
                                                       });
  }
  if (status.ok()) {
    status = bus.start();
  }
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }

  std::cout << "READY " << bus.boundEndpoint() << std::endl;
  std::string line;
  while (std::getline(std::cin, line)) {
    // Runs until the test closes standard input.
  }
  bus.stop();

  const robot::ipc::TopicStatistics statistics = bus.topicStatistics(inputTopic);
  std::cout << "STATS received=" << statistics.received << " delivered=" << statistics.delivered << " rejected=" << statistics.rejected
            << std::endl;
  return 0;
}
