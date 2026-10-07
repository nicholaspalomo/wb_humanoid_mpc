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

// The network file: what it accepts, that a malformed textproto is an error naming its line and column, that every
// invalid node is an error naming the node, that formatNetworkConfig() and parseNetworkConfig() round-trip, and that the
// shipped file is the network localhostNetworkConfig() builds.

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "gtest/gtest.h"

#include "robot_ipc/Delivery.h"
#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/NodeEndpoint.h"

namespace robot::ipc {
namespace {

// Relative to the test's runfiles directory (data dependency //config/ipc:network.textproto).
constexpr char kShippedNetworkFile[] = "config/ipc/network.textproto";
constexpr char kSource[] = "test.textproto";

absl::StatusOr<NetworkConfig> parse(const std::string& text) {
  return parseNetworkConfig(text, kSource);
}

TEST(NetworkConfigTest, ParsesTheNodesInFileOrder) {
  const absl::StatusOr<NetworkConfig> config = parse(R"(
# A comment.
nodes { name: "robot" host: "127.0.0.1" port: 5600 }
nodes {
  name: "mpc"
  host: "192.168.1.20"
  port: 5610
  bind_host: "0.0.0.0"  # every interface
}
)");
  ASSERT_TRUE(config.ok()) << config.status();
  ASSERT_EQ(config->nodes.size(), 2u);
  EXPECT_EQ(config->nodeNames(), (std::vector<std::string>{"robot", "mpc"}));

  const NodeEndpoint* absl_nullable robot = config->find("robot");
  ASSERT_NE(robot, nullptr);
  EXPECT_EQ(*robot, (NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = 5600}));
  EXPECT_EQ(robot->bindEndpoint(), "tcp://127.0.0.1:5600");
  EXPECT_EQ(robot->connectEndpoint(), "tcp://127.0.0.1:5600");

  const NodeEndpoint* absl_nullable mpc = config->find("mpc");
  ASSERT_NE(mpc, nullptr);
  // bind_host decides where the node binds; host stays where the others connect.
  EXPECT_EQ(mpc->bindEndpoint(), "tcp://*:5610");
  EXPECT_EQ(mpc->connectEndpoint(), "tcp://192.168.1.20:5610");

  EXPECT_EQ(config->find("operator"), nullptr);
}

TEST(NetworkConfigTest, ShippedFileIsTheLocalhostNetwork) {
  const absl::StatusOr<NetworkConfig> shipped = loadNetworkConfig(kShippedNetworkFile);
  ASSERT_TRUE(shipped.ok()) << shipped.status();
  EXPECT_EQ(*shipped, localhostNetworkConfig());
  EXPECT_TRUE(validateNetworkConfig(localhostNetworkConfig()).ok());
  // push_robot_config publishes as a node of its own, so that it runs next to the GUI, which binds "operator".
  const NodeEndpoint* absl_nullable configPush = shipped->find("config_push");
  ASSERT_NE(configPush, nullptr);
  EXPECT_EQ(configPush->connectEndpoint(), "tcp://127.0.0.1:5622");
}

TEST(NetworkConfigTest, WildcardHostBindsEveryInterfaceAndIsReachedOverLoopback) {
  const absl::StatusOr<NetworkConfig> config = parse(R"(
nodes { name: "robot" host: "0.0.0.0" port: 5600 }
nodes { name: "mpc" host: "*" port: 5610 }
)");
  ASSERT_TRUE(config.ok()) << config.status();
  for (const NodeEndpoint& node : config->nodes) {
    EXPECT_TRUE(isWildcardHost(node.host));
    EXPECT_TRUE(absl::StartsWith(node.bindEndpoint(), "tcp://*:")) << node.bindEndpoint();
    EXPECT_TRUE(absl::StartsWith(node.connectEndpoint(), "tcp://127.0.0.1:")) << node.connectEndpoint();
  }
}

TEST(NetworkConfigTest, EphemeralPortIsForNetworksBuiltInCode) {
  NetworkConfig config;
  config.nodes.push_back(NodeEndpoint{.name = "test", .host = "127.0.0.1", .port = kEphemeralPort});
  EXPECT_TRUE(validateNetworkConfig(config).ok());
  EXPECT_EQ(config.nodes[0].bindEndpoint(), "tcp://127.0.0.1:*");

  // In a file, port 0 is a port left out: the other processes could not connect to it.
  const absl::StatusOr<NetworkConfig> parsed = parse(R"(nodes { name: "test" host: "127.0.0.1" port: 0 })");
  EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(parsed.status().message(), "nodes[0] (test).port")) << parsed.status();
}

TEST(NetworkConfigTest, MissingFileIsNotFoundAndNamesThePath) {
  const absl::StatusOr<NetworkConfig> config = loadNetworkConfig("does/not/exist.textproto");
  EXPECT_EQ(config.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(config.status().message(), "does/not/exist.textproto")) << config.status();
}

TEST(NetworkConfigTest, FormattedNetworksParseBackUnchanged) {
  NetworkConfig withBindHost;
  withBindHost.nodes = {
      NodeEndpoint{.name = "robot", .host = "192.168.1.10", .port = 5600, .bindHost = "0.0.0.0"},
      NodeEndpoint{.name = "mpc-laptop_2", .host = "laptop.local", .port = 65535},
      NodeEndpoint{.name = "operator", .host = "*", .port = 1},
  };
  for (const NetworkConfig& config : {localhostNetworkConfig(), withBindHost}) {
    const std::string text = formatNetworkConfig(config);
    // One line per node.
    EXPECT_EQ(std::vector<std::string>(absl::StrSplit(text, '\n', absl::SkipEmpty())).size(), config.nodes.size()) << text;
    const absl::StatusOr<NetworkConfig> parsed = parse(text);
    ASSERT_TRUE(parsed.ok()) << parsed.status() << "\n" << text;
    EXPECT_EQ(*parsed, config) << text;
  }
}

// A file that is not a valid textproto of the schema, and the line its error must name.
struct MalformedFile {
  std::string description;
  std::string text;
  int line = 0;
  // A word the error message must contain (the protobuf parsers of C++ and Python word their errors differently, so
  // this is only set where both name the same thing).
  std::string fragment;
};

class MalformedNetworkFileTest : public ::testing::TestWithParam<MalformedFile> {};

TEST_P(MalformedNetworkFileTest, IsRejectedWithItsLineAndColumn) {
  const MalformedFile& file = GetParam();
  const absl::StatusOr<NetworkConfig> config = parse(file.text);
  ASSERT_FALSE(config.ok()) << file.description << " was accepted";
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << config.status();
  // "<source>:<line>:<column>: <problem>"
  const std::vector<std::string> parts = absl::StrSplit(config.status().message(), absl::MaxSplits(':', /*limit=*/3));
  ASSERT_EQ(parts.size(), 4u) << config.status();
  EXPECT_EQ(parts[0], kSource) << config.status();
  EXPECT_EQ(parts[1], absl::StrCat(file.line)) << file.description << ": " << config.status();
  int column = 0;
  EXPECT_TRUE(absl::SimpleAtoi(parts[2], &column) && column >= 1) << config.status();
  EXPECT_TRUE(absl::StrContains(parts[3], file.fragment)) << file.description << ": " << config.status();
}

// The Python test checks the same table.
// LINT.IfChange(malformed_files)
INSTANTIATE_TEST_SUITE_P(
    Rejections,
    MalformedNetworkFileTest,
    ::testing::Values(MalformedFile{"an unknown field of a node",
                                    "nodes { name: \"robot\" host: \"127.0.0.1\" port: 5600 }\nnodes { name: \"mpc\" prot: 5610 }", 2,
                                    "prot"},
                      MalformedFile{"an unknown top-level field", "# The nodes.\nhosts { name: \"robot\" }", 2, "hosts"},
                      MalformedFile{"a quoted port", "nodes {\n  name: \"robot\"\n  port: \"5600\"\n}", 3, ""},
                      MalformedFile{"a fractional port", "nodes {\n  name: \"robot\"\n  port: 5600.5\n}", 3, ""},
                      MalformedFile{"a negative port", "nodes {\n  name: \"robot\"\n  port: -1\n}", 3, ""},
                      MalformedFile{"a port beyond 32 bits", "nodes {\n  name: \"robot\"\n  port: 4294967296\n}", 3, ""},
                      MalformedFile{"a boolean port", "nodes {\n  name: \"robot\"\n  port: true\n}", 3, ""},
                      MalformedFile{"a port that is a word", "nodes {\n  name: \"robot\"\n  port: fifty\n}", 3, ""},
                      MalformedFile{"a host given twice", "nodes {\n  name: \"robot\"\n  host: \"a\"\n  host: \"b\"\n}", 4, "host"},
                      MalformedFile{"an unquoted host", "nodes {\n  name: \"robot\"\n  host: 127.0.0.1\n}", 3, ""},
                      MalformedFile{"a missing colon", "nodes { name: \"robot\" }\nnodes { name \"mpc\" }", 2, ""},
                      MalformedFile{"an unclosed node", "nodes { name: \"robot\" host: \"127.0.0.1\" port: 5600", 1, ""},
                      MalformedFile{"the old YAML network file", "nodes:\n  robot: {host: 127.0.0.1, port: 5600}", 2, ""}));
// LINT.ThenChange(//robot_runtime/robot_ipc/test/test_network_config.py:malformed_files)

// A valid textproto whose nodes the bus cannot use, and the key the error must name.
struct InvalidFile {
  std::string description;
  std::string text;
  std::string key;
};

class InvalidNetworkFileTest : public ::testing::TestWithParam<InvalidFile> {};

TEST_P(InvalidNetworkFileTest, IsRejectedWithTheOffendingNode) {
  const InvalidFile& file = GetParam();
  const absl::StatusOr<NetworkConfig> config = parse(file.text);
  ASSERT_FALSE(config.ok()) << file.description << " was accepted";
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << config.status();
  EXPECT_TRUE(absl::StartsWith(config.status().message(), absl::StrCat(kSource, ": ", file.key)))
      << file.description << ": " << config.status();
}

// The Python test checks the same table.
// LINT.IfChange(invalid_files)
INSTANTIATE_TEST_SUITE_P(
    Rejections,
    InvalidNetworkFileTest,
    ::testing::Values(
        InvalidFile{"an empty file", "", "nodes"},
        InvalidFile{"comments only", "# no node\n", "nodes"},
        InvalidFile{"a name missing", "nodes { host: \"127.0.0.1\" port: 5600 }", "nodes[0].name"},
        InvalidFile{"a host missing", "nodes { name: \"robot\" port: 5600 }", "nodes[0] (robot).host"},
        InvalidFile{"a port missing", "nodes { name: \"robot\" host: \"127.0.0.1\" }", "nodes[0] (robot).port"},
        InvalidFile{"port 0", "nodes { name: \"robot\" host: \"127.0.0.1\" port: 0 }", "nodes[0] (robot).port"},
        InvalidFile{"a port out of range", "nodes { name: \"robot\" host: \"127.0.0.1\" port: 70000 }", "nodes[0] (robot).port"},
        InvalidFile{"a host that is an endpoint", "nodes { name: \"robot\" host: \"tcp://127.0.0.1\" port: 5600 }",
                    "nodes[0] (robot).host"},
        InvalidFile{"a host with a port", "nodes { name: \"robot\" host: \"127.0.0.1:5600\" port: 5600 }", "nodes[0] (robot).host"},
        InvalidFile{"a host with a space", "nodes { name: \"robot\" host: \"127.0.0.1 \" port: 5600 }", "nodes[0] (robot).host"},
        InvalidFile{"a bind_host that is an endpoint", "nodes { name: \"robot\" host: \"127.0.0.1\" port: 5600 bind_host: \"tcp://*\" }",
                    "nodes[0] (robot).bind_host"},
        InvalidFile{"an invalid node name", "nodes { name: \"ro bot\" host: \"127.0.0.1\" port: 5600 }", "nodes[0] (ro bot).name"},
        InvalidFile{"a name used twice",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" port: 5600 }\nnodes { name: \"robot\" host: \"127.0.0.1\" port: 5601 }",
                    "nodes[1] (robot).name"},
        InvalidFile{"two nodes on one endpoint",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" port: 5600 }\nnodes { name: \"mpc\" host: \"127.0.0.1\" port: 5600 }",
                    "nodes[1] (mpc)"},
        InvalidFile{"the second node invalid",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" port: 5600 }\nnodes { name: \"mpc\" host: \"127.0.0.1\" port: 65536 }",
                    "nodes[1] (mpc).port"}));
// LINT.ThenChange(//robot_runtime/robot_ipc/test/test_network_config.py:invalid_files)

TEST(NetworkConfigTest, ValidationOfACodeBuiltNetworkNamesTheNode) {
  NetworkConfig twice;
  twice.nodes.push_back(NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = 5600});
  twice.nodes.push_back(NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = 5601});
  const absl::Status named = validateNetworkConfig(twice);
  EXPECT_TRUE(absl::StartsWith(named.message(), "nodes[1] (robot).name")) << named;

  EXPECT_FALSE(validateNetworkConfig(NetworkConfig()).ok());

  NetworkConfig badPort;
  badPort.nodes.push_back(NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = kMaxPort + 1});
  EXPECT_TRUE(absl::StartsWith(validateNetworkConfig(badPort).message(), "nodes[0] (robot).port"));

  NetworkConfig negativePort;
  negativePort.nodes.push_back(NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = -1});
  EXPECT_TRUE(absl::StartsWith(validateNetworkConfig(negativePort).message(), "nodes[0] (robot).port"));

  NetworkConfig badBindHost;
  badBindHost.nodes.push_back(NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = 5600, .bindHost = "tcp://*"});
  EXPECT_TRUE(absl::StartsWith(validateNetworkConfig(badBindHost).message(), "nodes[0] (robot).bind_host"));
}

TEST(DeliveryTest, NamesRoundTripAndUnknownNamesListTheValidOnes) {
  for (const Delivery delivery : {Delivery::kLatest, Delivery::kAll}) {
    const absl::StatusOr<Delivery> parsed = parseDelivery(deliveryName(delivery));
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, delivery);
  }
  const absl::StatusOr<Delivery> unknown = parseDelivery("newest");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "latest")) << unknown.status();
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "all")) << unknown.status();
}

}  // namespace
}  // namespace robot::ipc
