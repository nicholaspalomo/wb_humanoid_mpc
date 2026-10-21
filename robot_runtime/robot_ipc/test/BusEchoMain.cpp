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

// The C++ half of //robot_runtime/robot_ipc:cross_language_test. It joins the network of --network_config as node
// --node and, for every "<input>=<output>" pair of --echo, publishes every TestSample of the input topic back on the
// output topic with origin set to kOrigin. It prints "CONFIG <its Cyclone DDS configuration>" and "READY" once it runs,
// and when its standard input closes it stops and prints, for each input topic,
// "STATS <topic> received=<n> delivered=<n> rejected=<n> missed=<n>".

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"

#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/TopicStatistics.h"
#include "robot_ipc_test/test_sample.pb.h"

ABSL_FLAG(std::string, network_config, "", "the network file the test wrote");
ABSL_FLAG(std::string, node, "echo", "the node to publish as");
ABSL_FLAG(std::string, echo, "test/to_cpp=test/from_cpp", "comma-separated <input topic>=<output topic> pairs");

namespace {

// LINT.IfChange(echo_origin)
constexpr char kOrigin[] = "robot_ipc_bus_echo";
// LINT.ThenChange(//robot_runtime/robot_ipc/test/test_cross_language.py:echo_origin)
constexpr int kUsageError = 2;

struct EchoPair {
  std::string input;
  std::string output;
};

absl::Status subscribeEcho(robot::ipc::Bus& bus, const EchoPair& pair) {
  return bus.subscribe<robot_ipc_test::TestSample>(pair.input, robot::ipc::Delivery::kAll,
                                                   [&bus, output = pair.output](const robot_ipc_test::TestSample& sample) {
                                                     robot_ipc_test::TestSample echo = sample;
                                                     echo.set_origin(kOrigin);
                                                     const absl::Status published = bus.publishFromIoThread(output, echo);
                                                     if (!published.ok()) {
                                                       LOG(ERROR) << published;
                                                     }
                                                   });
}

}  // namespace

int main(int argc, char* absl_nonnull* absl_nonnull argv) {
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_network_config).empty()) {
    LOG(ERROR) << "--network_config is required";
    return kUsageError;
  }
  std::vector<EchoPair> pairs;
  for (const absl::string_view pair : absl::StrSplit(absl::GetFlag(FLAGS_echo), ',', absl::SkipEmpty())) {
    const std::vector<std::string> topics = absl::StrSplit(pair, '=');
    if (topics.size() != 2) {
      LOG(ERROR) << "--echo: '" << pair << "' is not <input>=<output>";
      return kUsageError;
    }
    pairs.push_back(EchoPair{.input = topics[0], .output = topics[1]});
  }

  absl::StatusOr<robot::ipc::NetworkConfig> network = robot::ipc::loadNetworkConfig(absl::GetFlag(FLAGS_network_config));
  if (!network.ok()) {
    LOG(ERROR) << network.status();
    return 1;
  }
  robot::ipc::BusOptions options;
  options.nodeName = absl::GetFlag(FLAGS_node);
  options.network = *std::move(network);
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> created = robot::ipc::Bus::Create(std::move(options));
  if (!created.ok()) {
    LOG(ERROR) << created.status();
    return 1;
  }
  robot::ipc::Bus& bus = **created;
  absl::Status status = absl::OkStatus();
  for (const EchoPair& pair : pairs) {
    if (status.ok()) status = subscribeEcho(bus, pair);
  }
  if (status.ok()) {
    status = bus.start();
  }
  if (!status.ok()) {
    LOG(ERROR) << status;
    return 1;
  }

  // The test reads these lines before it writes.
  std::cout << "CONFIG " << bus.domainConfig() << '\n' << "READY" << '\n' << std::flush;
  std::string line;
  while (std::getline(std::cin, line)) {
    // Runs until the test closes standard input.
  }
  bus.stop();

  for (const EchoPair& pair : pairs) {
    const robot::ipc::TopicStatistics statistics = bus.topicStatistics(pair.input);
    std::cout << "STATS " << pair.input << " received=" << statistics.received << " delivered=" << statistics.delivered
              << " rejected=" << statistics.rejected << " missed=" << statistics.missed << '\n';
  }
  std::cout << std::flush;
  return 0;
}
