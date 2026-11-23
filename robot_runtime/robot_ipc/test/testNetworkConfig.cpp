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

// The network file: what it accepts, that a malformed textproto is an error
// naming its line and column, that every invalid node, topic or field is an
// error naming it, that formatNetworkConfig() and parseNetworkConfig()
// round-trip, that the shipped file is the network localhostNetworkConfig()
// builds, the Zenoh configuration and ports SessionConfig.h derives from a
// network, the key expressions of KeyExprs.h and the envelope of
// WireProtocol.h.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "robot_ipc/Delivery.h"
#include "robot_ipc/KeyExprs.h"
#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/NodeEndpoint.h"
#include "robot_ipc/QosProfile.h"
#include "robot_ipc/SessionConfig.h"
#include "robot_ipc/WireProtocol.h"

namespace robot::ipc {
namespace {

// Relative to the test's runfiles directory (data dependency
// //config/ipc:network.textproto).
constexpr char kShippedNetworkFile[] = "config/ipc/network.textproto";
constexpr char kTwoMachineNetworkFile[] =
    "config/ipc/two_machine.example.textproto";
constexpr char kSource[] = "test.textproto";
// Where Linux's ephemeral ports start.
constexpr int kFirstEphemeralPort = 32768;

absl::StatusOr<NetworkConfig> parse(absl::string_view text) {
  return parseNetworkConfig(text, kSource);
}

TEST(NetworkConfigTest, ParsesTheNodesTopicsAndTransport) {
  const absl::StatusOr<NetworkConfig> config = parse(R"(
# A comment.
domain_id: 7
domain_tag: "lab.robot-2"
nodes { name: "robot" host: "192.168.1.10" }
nodes {
  name: "mpc"
  host: "laptop.local"  # a host name
}
topics { name: "mpc/policy" qos_profile: "reliable_latest" publishers: "mpc" }
topics { name: "robot/state" qos_profile: "latest" publishers: ["robot"] }
default_qos_profile: "latest"
transport { socket_receive_buffer_bytes: 4096 lease_duration_ms: 2500 }
)");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->nodeNames(), (std::vector<std::string>{"robot", "mpc"}));
  EXPECT_EQ(config->domainId, 7);
  EXPECT_EQ(config->domainTag, "lab.robot-2");
  const NodeEndpoint* absl_nullable mpc = config->find("mpc");
  ASSERT_NE(mpc, nullptr);
  EXPECT_EQ(*mpc, (NodeEndpoint{.name = "mpc", .host = "laptop.local"}));
  EXPECT_EQ(config->find("operator"), nullptr);
  EXPECT_EQ(config->qosProfileOf("mpc/policy"), QosProfile::kReliableLatest);
  EXPECT_EQ(config->qosProfileOf("robot/state"), QosProfile::kLatest);
  EXPECT_EQ(config->qosProfileOf("unlisted/topic"), QosProfile::kLatest);
  EXPECT_EQ(config->topicsPublishedBy("mpc"),
            std::vector<std::string>{"mpc/policy"});
  EXPECT_EQ(config->peerHosts(),
            (std::vector<std::string>{"192.168.1.10", "laptop.local"}));
  EXPECT_EQ(config->transport.socketReceiveBufferBytes, 4096);
  EXPECT_EQ(config->transport.leaseDuration, absl::Milliseconds(2500));
}

TEST(NetworkConfigTest,
     ShippedFileIsTheLocalhostNetworkAndTheExampleIsItOnTwoMachines) {
  const absl::StatusOr<NetworkConfig> shipped =
      loadNetworkConfig(kShippedNetworkFile);
  ASSERT_TRUE(shipped.ok()) << shipped.status();
  EXPECT_EQ(*shipped, localhostNetworkConfig());
  for (const std::string& host : shipped->peerHosts()) {
    EXPECT_TRUE(isLoopbackHost(host)) << host;
  }
  const absl::StatusOr<NetworkConfig> example =
      loadNetworkConfig(kTwoMachineNetworkFile);
  ASSERT_TRUE(example.ok()) << example.status();
  EXPECT_EQ(example->nodeNames(), shipped->nodeNames());
  EXPECT_EQ(example->topics, shipped->topics);
  EXPECT_EQ(example->domainId, shipped->domainId);
  EXPECT_EQ(example->peerHosts().size(), 2u)
      << "the robot's computer and the laptop";
  // A peer that vanished is disconnected within seconds.
  EXPECT_GT(example->transport.leaseDuration, absl::ZeroDuration());
  EXPECT_LE(example->transport.leaseDuration, absl::Seconds(3));
}

TEST(NetworkConfigTest, TheShippedTopicsHaveTheProfilesOfTheirTraffic) {
  const NetworkConfig network = localhostNetworkConfig();
  // A policy is 0.5-0.75 MB: it may wait briefly for room in a busy link.
  EXPECT_EQ(network.qosProfileOf("mpc/policy"), QosProfile::kReliableLatest);
  // The observation stream comes again every control cycle: a lost sample is
  // replaced, not repaired.
  EXPECT_EQ(network.qosProfileOf("robot/mpc_observation"), QosProfile::kLatest);
  // A one-off command must arrive, and the publisher exists before it is put.
  EXPECT_EQ(network.qosProfileOf("operator/fsm_command"),
            QosProfile::kReliableEvents);
  const TopicQos* absl_nullable command =
      network.findTopic("operator/fsm_command");
  ASSERT_NE(command, nullptr);
  EXPECT_EQ(command->publishers, std::vector<std::string>{"operator"});
}

TEST(NetworkConfigTest, MissingFileIsNotFoundAndNamesThePath) {
  const absl::StatusOr<NetworkConfig> config =
      loadNetworkConfig("does/not/exist.textproto");
  EXPECT_EQ(config.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(
      absl::StrContains(config.status().message(), "does/not/exist.textproto"))
      << config.status();
}

TEST(NetworkConfigTest, FormattedNetworksParseBackUnchanged) {
  NetworkConfig custom;
  custom.nodes = {
      NodeEndpoint{.name = "robot", .host = "192.168.1.10"},
      NodeEndpoint{.name = "mpc-laptop_2", .host = "laptop.local"},
  };
  custom.domainId = NetworkConfig::kMaxDomainId;
  custom.domainTag = "tag";
  custom.topics = {TopicQos{.topic = "a/b",
                            .profile = QosProfile::kLatest,
                            .publishers = {"robot", "mpc-laptop_2"}}};
  custom.defaultQosProfile = QosProfile::kReliableLatest;
  custom.transport.socketSendBufferBytes = 1 << 20;
  custom.transport.maxProcessesPerHost = 4;
  for (const NetworkConfig& config : {localhostNetworkConfig(), custom}) {
    const std::string text = formatNetworkConfig(config);
    const absl::StatusOr<NetworkConfig> parsed = parse(text);
    ASSERT_TRUE(parsed.ok()) << parsed.status() << "\n" << text;
    EXPECT_EQ(*parsed, config) << text;
  }
  // One line for the domain, one per node and per topic, one for the transport.
  const std::string text = formatNetworkConfig(custom);
  EXPECT_EQ(
      std::vector<std::string>(absl::StrSplit(text, '\n', absl::SkipEmpty()))
          .size(),
      1 + 2 + 1 + 1)
      << text;
}

// A file that is not a valid textproto of the schema, and the line its error
// must name.
struct MalformedFile {
  std::string description;
  std::string text;
  int line = 0;
  // A word the error message must contain (the protobuf parsers of C++ and
  // Python word their errors differently, so this is only set where both name
  // the same thing).
  std::string fragment;
};

class MalformedNetworkFileTest
    : public ::testing::TestWithParam<MalformedFile> {};

TEST_P(MalformedNetworkFileTest, IsRejectedWithItsLineAndColumn) {
  const MalformedFile& file = GetParam();
  const absl::StatusOr<NetworkConfig> config = parse(file.text);
  ASSERT_FALSE(config.ok()) << file.description << " was accepted";
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument)
      << config.status();
  // "<source>:<line>:<column>: <problem>"
  const std::vector<std::string> parts = absl::StrSplit(
      config.status().message(), absl::MaxSplits(':', /*limit=*/3));
  ASSERT_EQ(parts.size(), 4u) << config.status();
  EXPECT_EQ(parts[0], kSource) << config.status();
  EXPECT_EQ(parts[1], absl::StrCat(file.line))
      << file.description << ": " << config.status();
  int column = 0;
  EXPECT_TRUE(absl::SimpleAtoi(parts[2], &column) && column >= 1)
      << config.status();
  EXPECT_TRUE(absl::StrContains(parts[3], file.fragment))
      << file.description << ": " << config.status();
}

// The Python test checks the same table.
// LINT.IfChange(malformed_files)
INSTANTIATE_TEST_SUITE_P(
    Rejections,
    MalformedNetworkFileTest,
    ::testing::Values(
        MalformedFile{"an unknown field of a node",
                      "nodes { name: \"robot\" host: \"127.0.0.1\" }\nnodes { "
                      "name: \"mpc\" hots: \"a\" }",
                      2, "hots"},
        MalformedFile{"the retired port of a node",
                      "nodes {\n  name: \"robot\"\n  port: 5600\n}", 3, "port"},
        MalformedFile{"the retired DDS message size",
                      "transport {\n  max_message_size_bytes: 1400\n}", 2,
                      "max_message_size_bytes"},
        MalformedFile{"an unknown top-level field",
                      "# The nodes.\nhosts { name: \"robot\" }", 2, "hosts"},
        MalformedFile{"a quoted domain id", "domain_id: \"7\"", 1, ""},
        MalformedFile{"a fractional domain id", "domain_id: 7.5", 1, ""},
        MalformedFile{
            "a negative domain id",
            "nodes { name: \"robot\" host: \"127.0.0.1\" }\ndomain_id: -1", 2,
            ""},
        MalformedFile{"a domain id beyond 32 bits", "domain_id: 4294967296", 1,
                      ""},
        MalformedFile{"a boolean domain id", "domain_id: true", 1, ""},
        MalformedFile{
            "a host given twice",
            "nodes {\n  name: \"robot\"\n  host: \"a\"\n  host: \"b\"\n}", 4,
            "host"},
        MalformedFile{"an unquoted host",
                      "nodes {\n  name: \"robot\"\n  host: 127.0.0.1\n}", 3,
                      ""},
        MalformedFile{"a missing colon",
                      "nodes { name: \"robot\" }\nnodes { name \"mpc\" }", 2,
                      ""},
        MalformedFile{"an unclosed node",
                      "nodes { name: \"robot\" host: \"127.0.0.1\"", 1, ""}));
// LINT.ThenChange(//robot_runtime/robot_ipc/test/test_network_config.py:malformed_files)

// A valid textproto that the bus cannot use, and the key the error must name.
struct InvalidFile {
  std::string description;
  std::string text;
  std::string key;
};

class InvalidNetworkFileTest : public ::testing::TestWithParam<InvalidFile> {};

TEST_P(InvalidNetworkFileTest, IsRejectedWithTheOffendingEntry) {
  const InvalidFile& file = GetParam();
  const absl::StatusOr<NetworkConfig> config = parse(file.text);
  ASSERT_FALSE(config.ok()) << file.description << " was accepted";
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument)
      << config.status();
  EXPECT_TRUE(absl::StartsWith(config.status().message(),
                               absl::StrCat(kSource, ": ", file.key)))
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
        InvalidFile{"a name missing", "nodes { host: \"127.0.0.1\" }",
                    "nodes[0].name"},
        InvalidFile{"a host missing", "nodes { name: \"robot\" }",
                    "nodes[0] (robot).host"},
        InvalidFile{"a wildcard host",
                    "nodes { name: \"robot\" host: \"0.0.0.0\" }",
                    "nodes[0] (robot).host"},
        InvalidFile{"a host that is an endpoint",
                    "nodes { name: \"robot\" host: \"udp://127.0.0.1\" }",
                    "nodes[0] (robot).host"},
        InvalidFile{"a host with a port",
                    "nodes { name: \"robot\" host: \"127.0.0.1:7410\" }",
                    "nodes[0] (robot).host"},
        InvalidFile{"a host with a space",
                    "nodes { name: \"robot\" host: \"127.0.0.1 \" }",
                    "nodes[0] (robot).host"},
        InvalidFile{"a host with a control character",
                    R"(nodes { name: "robot" host: "127.0.0.1\034" })",
                    "nodes[0] (robot).host"},
        InvalidFile{"a host with a non-ASCII space",
                    R"(nodes { name: "robot" host: "127.0.0.1\302\240" })",
                    "nodes[0] (robot).host"},
        InvalidFile{"a node name ending in a newline",
                    R"(nodes { name: "robot\n" host: "127.0.0.1" })",
                    "nodes[0]"},
        InvalidFile{"an invalid node name",
                    "nodes { name: \"ro bot\" host: \"127.0.0.1\" }",
                    "nodes[0] (ro bot).name"},
        InvalidFile{"a name used twice",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\nnodes { "
                    "name: \"robot\" host: \"127.0.0.1\" }",
                    "nodes[1] (robot).name"},
        InvalidFile{
            "a domain id out of range",
            "nodes { name: \"robot\" host: \"127.0.0.1\" }\ndomain_id: 233",
            "domain_id"},
        InvalidFile{"an invalid domain tag",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" "
                    "}\ndomain_tag: \"a b\"",
                    "domain_tag"},
        InvalidFile{"a topic that is no key expression chunk",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntopics { "
                    "name: \"robot/a-b\" qos_profile: \"latest\" }",
                    "topics[0] (robot/a-b).name"},
        InvalidFile{"a topic with an empty chunk",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntopics { "
                    "name: \"robot//state\" qos_profile: \"latest\" }",
                    "topics[0] (robot//state).name"},
        InvalidFile{"a topic ending in a slash",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntopics { "
                    "name: \"robot/\" qos_profile: \"latest\" }",
                    "topics[0] (robot/).name"},
        InvalidFile{"a topic listed twice",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntopics { "
                    "name: \"a\" qos_profile: \"latest\" }\n"
                    "topics { name: \"a\" qos_profile: \"latest\" }",
                    "topics[1] (a).name"},
        InvalidFile{"a topic without a profile",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntopics { "
                    "name: \"a\" }",
                    "topics[0] (a).qos_profile"},
        InvalidFile{"an unknown profile",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntopics { "
                    "name: \"a\" qos_profile: \"fast\" }",
                    "topics[0] (a).qos_profile"},
        InvalidFile{"a publisher that is no node",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntopics { "
                    "name: \"a\" qos_profile: \"latest\" publishers: \"mpc\" }",
                    "topics[0] (a).publishers"},
        InvalidFile{"an unknown default profile",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" "
                    "}\ndefault_qos_profile: \"fast\"",
                    "default_qos_profile"},
        InvalidFile{"too many processes per host",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\ntransport "
                    "{ max_processes_per_host: 101 }",
                    "transport.max_processes_per_host"},
        InvalidFile{"the second node invalid",
                    "nodes { name: \"robot\" host: \"127.0.0.1\" }\nnodes { "
                    "name: \"mpc\" host: \"*\" }",
                    "nodes[1] (mpc).host"}));
// LINT.ThenChange(//robot_runtime/robot_ipc/test/test_network_config.py:invalid_files)

TEST(NetworkConfigTest, ValidationOfACodeBuiltNetworkNamesTheEntry) {
  NetworkConfig twice;
  twice.nodes.push_back(NodeEndpoint{.name = "robot", .host = "127.0.0.1"});
  twice.nodes.push_back(NodeEndpoint{.name = "robot", .host = "127.0.0.1"});
  EXPECT_TRUE(absl::StartsWith(validateNetworkConfig(twice).message(),
                               "nodes[1] (robot).name"))
      << validateNetworkConfig(twice);

  EXPECT_FALSE(validateNetworkConfig(NetworkConfig()).ok());

  NetworkConfig negativeDomain;
  negativeDomain.nodes.push_back(
      NodeEndpoint{.name = "robot", .host = "127.0.0.1"});
  negativeDomain.domainId = -1;
  EXPECT_TRUE(absl::StartsWith(validateNetworkConfig(negativeDomain).message(),
                               "domain_id"));

  NetworkConfig negativeBuffer;
  negativeBuffer.nodes.push_back(
      NodeEndpoint{.name = "robot", .host = "127.0.0.1"});
  negativeBuffer.transport.socketReceiveBufferBytes = -1;
  EXPECT_TRUE(absl::StartsWith(validateNetworkConfig(negativeBuffer).message(),
                               "transport"));
}

TEST(NetworkConfigTest, HostKinds) {
  EXPECT_TRUE(isIpv4Address("192.168.1.10"));
  EXPECT_TRUE(isIpv4Address("0.0.0.0"));
  EXPECT_FALSE(isIpv4Address("256.1.1.1"));
  EXPECT_FALSE(isIpv4Address("1.2.3"));
  EXPECT_FALSE(isIpv4Address("1.2.3.4.5"));
  EXPECT_FALSE(isIpv4Address("1.2.3.x"));
  EXPECT_FALSE(isIpv4Address("laptop.local"));
  EXPECT_TRUE(isLoopbackHost("127.0.0.1"));
  EXPECT_TRUE(isLoopbackHost("127.1.2.3"));
  EXPECT_TRUE(isLoopbackHost("localhost"));
  EXPECT_FALSE(isLoopbackHost("192.168.1.10"));
  EXPECT_FALSE(isLoopbackHost("127.example.com"));
}

TEST(DeliveryTest, NamesRoundTripAndUnknownNamesListTheValidOnes) {
  for (const Delivery delivery : {Delivery::kLatest, Delivery::kAll}) {
    const absl::StatusOr<Delivery> parsed =
        parseDelivery(deliveryName(delivery));
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, delivery);
  }
  const absl::StatusOr<Delivery> unknown = parseDelivery("newest");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "latest"))
      << unknown.status();
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "all"))
      << unknown.status();
}

TEST(QosProfileTest, NamesRoundTripAndEachProfileHasItsCongestionControl) {
  for (const QosProfile profile : kAllQosProfiles) {
    const absl::StatusOr<QosProfile> parsed =
        parseQosProfile(qosProfileName(profile));
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, profile);
  }
  const absl::StatusOr<QosProfile> unknown = parseQosProfile("fast");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "reliable_latest"))
      << unknown.status();
  // The latest values drop under congestion rather than build a backlog; the
  // events wait for room, ahead of the data.
  EXPECT_EQ(qosPolicy(QosProfile::kLatest).congestionControl,
            CongestionControl::kDrop);
  EXPECT_TRUE(qosPolicy(QosProfile::kLatest).express);
  EXPECT_EQ(qosPolicy(QosProfile::kReliableLatest).congestionControl,
            CongestionControl::kDrop);
  EXPECT_EQ(qosPolicy(QosProfile::kReliableEvents).congestionControl,
            CongestionControl::kBlock);
  EXPECT_EQ(qosPolicy(QosProfile::kReliableEvents).priority,
            Priority::kInteractiveHigh);
}

// ---------------------------------------------------------------------------------------------------------------------
// The Zenoh configuration, the key expressions and the envelope
// ---------------------------------------------------------------------------------------------------------------------

TEST(SessionConfigTest, TheTcpPortsStartAtZenohsPortAndStayBelowTheEphemeral) {
  NetworkConfig network = localhostNetworkConfig();
  EXPECT_EQ(tcpPortRange(network), (PortRange{.first = 7447, .last = 7466}));
  network.domainId = 1;
  EXPECT_EQ(tcpPortRange(network), (PortRange{.first = 7547, .last = 7566}));
  // The largest domain with the most processes per host stays below the
  // ephemeral ports, and the domains never overlap.
  network.domainId = NetworkConfig::kMaxDomainId;
  network.transport.maxProcessesPerHost = kZenohDomainGain;
  EXPECT_TRUE(validateNetworkConfig(network).ok());
  EXPECT_LT(tcpPortRange(network).last, kFirstEphemeralPort);
  const int last = tcpPortRange(network).last;
  network.domainId = NetworkConfig::kMaxDomainId - 1;
  EXPECT_EQ(tcpPortRange(network).last + kZenohDomainGain, last);
}

TEST(SessionConfigTest, EveryProcessConnectsToEveryPortOfEveryNodeHost) {
  NetworkConfig network;
  network.nodes = {NodeEndpoint{.name = "robot", .host = "192.0.2.1"},
                   NodeEndpoint{.name = "mpc", .host = "laptop.example"},
                   NodeEndpoint{.name = "operator", .host = "laptop.example"}};
  network.transport.maxProcessesPerHost = 2;
  EXPECT_EQ(connectEndpoints(network),
            (std::vector<std::string>{
                "tcp/192.0.2.1:7447", "tcp/192.0.2.1:7448",
                "tcp/laptop.example:7447", "tcp/laptop.example:7448"}));
}

TEST(SessionConfigTest, ALoopbackNetworkListensOnLoopbackWithoutMulticast) {
  const absl::StatusOr<std::string> config = zenohSessionConfig(
      localhostNetworkConfig(), "robot", /*listenPort=*/7450);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_TRUE(absl::StartsWith(*config, R"({"mode":"peer",)")) << *config;
  EXPECT_TRUE(absl::StrContains(
      *config, R"("listen":{"endpoints":["tcp/127.0.0.1:7450"],)"))
      << *config;
  EXPECT_TRUE(absl::StrContains(*config, R"("tcp/127.0.0.1:7447",)"))
      << *config;
  EXPECT_TRUE(absl::StrContains(*config, R"("tcp/127.0.0.1:7466"])"))
      << *config;
  EXPECT_TRUE(absl::StrContains(*config, R"("multicast":{"enabled":false})"))
      << *config;
  EXPECT_TRUE(absl::StrContains(*config, R"("lease":3000)")) << *config;
  EXPECT_TRUE(absl::StrContains(*config, R"("so_rcvbuf":8388608)")) << *config;
  EXPECT_FALSE(absl::StrContains(*config, "so_sndbuf")) << *config;
  // A subscriber of the same network gets the same configuration, so that both
  // may share a process's session.
  const absl::StatusOr<std::string> subscriber = zenohSessionConfig(
      localhostNetworkConfig(), /*nodeName=*/"", /*listenPort=*/7450);
  ASSERT_TRUE(subscriber.ok()) << subscriber.status();
  EXPECT_EQ(*subscriber, *config);
}

TEST(SessionConfigTest,
     ANodeListensOnItsHostsAddressOrRefusesAnotherMachinesAddress) {
  NetworkConfig network;
  network.nodes = {NodeEndpoint{.name = "robot", .host = "192.0.2.1"},
                   NodeEndpoint{.name = "mpc", .host = "laptop.example"},
                   NodeEndpoint{.name = "local", .host = "127.0.0.1"}};
  EXPECT_EQ(interfaceAddressFor(network, "robot").status().code(),
            absl::StatusCode::kUnavailable);
  EXPECT_EQ(
      zenohSessionConfig(network, "robot", /*listenPort=*/7447).status().code(),
      absl::StatusCode::kUnavailable);
  const absl::StatusOr<std::string> named = interfaceAddressFor(network, "mpc");
  ASSERT_TRUE(named.ok()) << named.status();
  EXPECT_EQ(*named, "") << "a host name: every interface";
  const absl::StatusOr<std::string> local =
      interfaceAddressFor(network, "local");
  ASSERT_TRUE(local.ok()) << local.status();
  EXPECT_EQ(*local, "127.0.0.1");
  // A subscriber on a machine that is no node's listens on every interface.
  const absl::StatusOr<std::string> subscriber =
      interfaceAddressFor(network, /*nodeName=*/"");
  ASSERT_TRUE(subscriber.ok()) << subscriber.status();
  EXPECT_EQ(*subscriber, "");
  const absl::StatusOr<std::string> config =
      zenohSessionConfig(network, "mpc", /*listenPort=*/7449);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_TRUE(absl::StrContains(*config, R"(["tcp/0.0.0.0:7449"])")) << *config;
}

TEST(SessionConfigTest, TheTransportReachesTheConfiguration) {
  NetworkConfig network;
  network.nodes = {NodeEndpoint{.name = "robot", .host = "127.0.0.1"}};
  network.transport.maxProcessesPerHost = 4;
  network.transport.leaseDuration = absl::Milliseconds(1500);
  network.transport.socketSendBufferBytes = 1 << 20;
  const absl::StatusOr<std::string> config =
      zenohSessionConfig(network, "robot", /*listenPort=*/7447);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_TRUE(absl::StrContains(
      *config,
      R"("connect":{"endpoints":["tcp/127.0.0.1:7448","tcp/127.0.0.1:7449","tcp/127.0.0.1:7450"],)"))
      << *config << "every port but its own listener's";
  EXPECT_TRUE(absl::StrContains(
      *config, R"("link":{"tx":{"lease":1500},"tcp":{"so_sndbuf":1048576}})"))
      << *config;
}

TEST(KeyExprsTest, TheDomainTagAndProfileKeepTopicsApart) {
  NetworkConfig network;
  network.nodes = {NodeEndpoint{.name = "robot", .host = "127.0.0.1"}};
  network.domainId = 3;
  network.domainTag = "lab.robot-2";
  network.topics = {TopicQos{.topic = "mpc/policy",
                             .profile = QosProfile::kReliableLatest,
                             .publishers = {}}};
  EXPECT_EQ(dataKeyExpr(network, "mpc/policy"),
            "robot_ipc/3/tag_lab.robot-2/reliable_latest/mpc/policy");
  EXPECT_EQ(dataKeyExpr(network, "unlisted"),
            "robot_ipc/3/tag_lab.robot-2/reliable_events/unlisted");
  EXPECT_EQ(allTopicsKeyExpr(network), "robot_ipc/3/tag_lab.robot-2/*/**");
  EXPECT_EQ(
      parseDataKeyExpr(network, dataKeyExpr(network, "mpc/policy")),
      (DataKey{.profile = QosProfile::kReliableLatest, .topic = "mpc/policy"}));
  EXPECT_EQ(parseDataKeyExpr(network, "robot_ipc/3/tag_/latest/a"),
            std::nullopt)
      << "another tag";
  EXPECT_EQ(parseDataKeyExpr(network, "robot_ipc/3/tag_lab.robot-2/fast/a"),
            std::nullopt)
      << "no profile";
  network.domainTag.clear();
  EXPECT_EQ(dataKeyExpr(network, "mpc/policy"),
            "robot_ipc/3/tag_/reliable_latest/mpc/policy");
}

TEST(KeyExprsTest, EndpointTokensRoundTrip) {
  const NetworkConfig network = localhostNetworkConfig();
  const EndpointToken token{.role = EndpointRole::kSubscriber,
                            .profile = QosProfile::kLatest,
                            .busId = "00000000deadbeef",
                            .topic = "robot/state"};
  const std::string keyExpr = endpointTokenKeyExpr(network, token);
  EXPECT_EQ(keyExpr,
            "robot_ipc_endpoints/0/tag_/subscriber/latest/00000000deadbeef/"
            "robot/state");
  EXPECT_EQ(parseEndpointTokenKeyExpr(network, keyExpr), token);
  EXPECT_EQ(allEndpointTokensKeyExpr(network), "robot_ipc_endpoints/0/tag_/**");
  EXPECT_EQ(parseEndpointTokenKeyExpr(
                network, "robot_ipc_endpoints/0/tag_/owner/latest/1/a"),
            std::nullopt);
  EXPECT_EQ(parseEndpointTokenKeyExpr(network,
                                      "robot_ipc_endpoints/0/tag_/publisher/"
                                      "latest/1"),
            std::nullopt)
      << "no topic";
}

TEST(WireProtocolTest, AnEnvelopeRoundTripsAndATruncatedOneDoesNotDecode) {
  const EnvelopeHeader header{.protocolVersion = kWireProtocolVersion,
                              .writerId = 0x0102030405060708,
                              .sequenceNumber = 42,
                              .sourceTimestampNs = -5};
  const std::string bytes =
      encodeEnvelope(header, "pkg.Message", std::string("\x00\x01payload", 9));
  ASSERT_EQ(bytes.size(), kEnvelopeHeaderBytes + 11 + 9);
  // Little-endian: the version's low byte first.
  EXPECT_EQ(bytes[0], static_cast<char>(kWireProtocolVersion));
  EXPECT_EQ(envelopeProtocolVersion(bytes), kWireProtocolVersion);
  const std::optional<EnvelopeView> envelope = decodeEnvelope(bytes);
  ASSERT_TRUE(envelope.has_value());
  EXPECT_EQ(envelope->header.writerId, header.writerId);
  EXPECT_EQ(envelope->header.sequenceNumber, 42u);
  EXPECT_EQ(envelope->header.sourceTimestampNs, -5);
  EXPECT_EQ(envelope->typeName, "pkg.Message");
  EXPECT_EQ(envelope->payload, std::string("\x00\x01payload", 9));

  EXPECT_EQ(envelopeProtocolVersion("abc"), std::nullopt);
  EXPECT_FALSE(
      decodeEnvelope(bytes.substr(0, kEnvelopeHeaderBytes - 1)).has_value());
  EXPECT_FALSE(
      decodeEnvelope(bytes.substr(0, kEnvelopeHeaderBytes + 10)).has_value())
      << "the type name runs past the end";
  const std::optional<EnvelopeView> empty =
      decodeEnvelope(bytes.substr(0, kEnvelopeHeaderBytes + 11));
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->payload.empty());
}

}  // namespace
}  // namespace robot::ipc
