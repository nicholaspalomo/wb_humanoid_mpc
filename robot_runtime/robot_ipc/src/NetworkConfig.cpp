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

#include "robot_ipc/NetworkConfig.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"

#include "robot_ipc/DomainConfig.h"
#include "robot_ipc/NodeEndpoint.h"
#include "robot_ipc/QosProfile.h"
#include "robot_ipc_proto/network_config.pb.h"
#include "robot_runtime/robot_ipc/LocalhostNetworkText.h"
#include "robot_runtime/robot_ipc/src/StatusMacros.h"
#include "robot_runtime/robot_ipc/src/Textproto.h"

namespace robot::ipc {
namespace {

using NodeProto = robot_ipc_proto::NetworkConfig::Node;
using TopicProto = robot_ipc_proto::NetworkConfig::Topic;
using TransportProto = robot_ipc_proto::NetworkConfig::Transport;

// The rules of the network file and the words of their error messages. The
// Python loader applies the same ones.
// LINT.IfChange(network_rules)
constexpr absl::string_view kNodesField = "nodes";
constexpr absl::string_view kTopicsField = "topics";
constexpr absl::string_view kTransportField = "transport";
// A node name is a non-empty run of letters, digits and these.
constexpr absl::string_view kNodeNamePunctuation = "_-";
// A domain tag is a run of letters, digits and these.
constexpr absl::string_view kDomainTagPunctuation = "_-.";
// A DDS topic name is letters, digits and these, not starting with a digit
// (Cyclone DDS's own rule).
constexpr absl::string_view kTopicNamePunctuation = "_/";
// The hosts that name every interface rather than a machine.
constexpr absl::string_view kWildcardHostAny = "0.0.0.0";
constexpr absl::string_view kWildcardHostStar = "*";
constexpr absl::string_view kLocalhostName = "localhost";
constexpr absl::string_view kLoopbackPrefix = "127.";
// A host is printable ASCII without a space (host names and IPv4 addresses are
// ASCII; an IDN name is written in punycode), and without these, which belong
// to an endpoint or a port.
constexpr unsigned char kFirstHostCharacter = '!';
constexpr unsigned char kLastHostCharacter = '~';
constexpr absl::string_view kHostRefusedPunctuation = "/:";
// The UDP payload sizes Cyclone DDS can send: from one that holds a DDSI header
// and a fragment, to the largest IPv4 UDP payload.
constexpr int64_t kMinMessageSizeBytes = 1024;
constexpr int64_t kMaxMessageSizeBytes = 65500;
// The largest process count per host whose unicast ports (two per process) a
// domain's 250 ports leave room for.
constexpr int kMaxProcessesPerHost = 115;
// Packets per thousand: 1000 drops every one.
constexpr int kPerMille = 1000;
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/network_config.py:network_rules)
// The largest UDP port.
constexpr int kMaxPort = 65535;
// An IPv4 address: four decimal octets.
constexpr size_t kIpv4Octets = 4;
constexpr int kMaxOctet = 255;
constexpr size_t kMaxOctetDigits = 3;

absl::Status invalid(absl::string_view key, absl::string_view problem) {
  return absl::InvalidArgumentError(absl::StrCat(key, ": ", problem));
}

absl::Status withSource(absl::string_view source, const absl::Status& status) {
  return absl::Status(status.code(),
                      absl::StrCat(source, ": ", status.message()));
}

// "nodes[1] (mpc)": the position of an entry in the file, and its name when it
// has one.
std::string entryKey(absl::string_view field,
                     size_t index,
                     absl::string_view name) {
  const std::string position = absl::StrCat(field, "[", index, "]");
  return name.empty() ? position : absl::StrCat(position, " (", name, ")");
}

std::string fieldKey(absl::string_view field,
                     size_t index,
                     absl::string_view name,
                     absl::string_view member) {
  return absl::StrCat(entryKey(field, index, name), ".", member);
}

bool isRunOf(absl::string_view text, absl::string_view punctuation) {
  for (const char character : text) {
    if (!absl::ascii_isalnum(static_cast<unsigned char>(character)) &&
        !absl::StrContains(punctuation, character)) {
      return false;
    }
  }
  return true;
}

// A host is an IPv4 address or a host name; not a wildcard, an endpoint
// ("udp://..."), "host:port" or IPv6.
absl::Status validateHost(absl::string_view key, absl::string_view host) {
  if (host.empty()) {
    return invalid(key, "is missing");
  }
  if (host == kWildcardHostAny || host == kWildcardHostStar) {
    return invalid(key,
                   absl::StrCat("'", host,
                                "' names every interface, not a machine: write "
                                "the address of the node's machine"));
  }
  for (const char character : host) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (byte < kFirstHostCharacter || byte > kLastHostCharacter ||
        absl::StrContains(kHostRefusedPunctuation, character)) {
      return invalid(key,
                     absl::StrCat("'", host,
                                  "' is not a host: write an IPv4 address or a "
                                  "host name, without a scheme or a port"));
    }
  }
  return absl::OkStatus();
}

absl::Status validateNodes(const NetworkConfig& config) {
  if (config.nodes.empty()) {
    return invalid(kNodesField, "names no node");
  }
  absl::flat_hash_map<std::string, size_t> names;
  for (size_t index = 0; index < config.nodes.size(); ++index) {
    const NodeEndpoint& node = config.nodes[index];
    const std::string nameKey = fieldKey(kNodesField, index, node.name, "name");
    if (node.name.empty()) {
      return invalid(nameKey, "is missing");
    }
    if (!isRunOf(node.name, kNodeNamePunctuation)) {
      return invalid(
          nameKey,
          "a node name is a non-empty run of letters, digits, '_' and '-'");
    }
    const std::pair<absl::flat_hash_map<std::string, size_t>::iterator, bool>
        named = names.emplace(node.name, index);
    if (!named.second) {
      return invalid(
          nameKey,
          absl::StrCat("is also the name of ",
                       entryKey(kNodesField, named.first->second, node.name)));
    }
    ROBOT_IPC_RETURN_IF_ERROR(validateHost(
        fieldKey(kNodesField, index, node.name, "host"), node.host));
  }
  return absl::OkStatus();
}

absl::Status validateTopics(const NetworkConfig& config) {
  absl::flat_hash_map<std::string, size_t> topics;
  for (size_t index = 0; index < config.topics.size(); ++index) {
    const TopicQos& topic = config.topics[index];
    const absl::Status name = validateTopicName(topic.topic);
    if (!name.ok()) {
      return invalid(fieldKey(kTopicsField, index, topic.topic, "name"),
                     name.message());
    }
    const std::pair<absl::flat_hash_map<std::string, size_t>::iterator, bool>
        listed = topics.emplace(topic.topic, index);
    if (!listed.second) {
      return invalid(fieldKey(kTopicsField, index, topic.topic, "name"),
                     absl::StrCat("is also the name of ",
                                  entryKey(kTopicsField, listed.first->second,
                                           topic.topic)));
    }
    for (const std::string& publisher : topic.publishers) {
      if (config.find(publisher) == nullptr) {
        return invalid(
            fieldKey(kTopicsField, index, topic.topic, "publishers"),
            absl::StrCat("'", publisher,
                         "' is not a node of the network (nodes: ",
                         absl::StrJoin(config.nodeNames(), ", "), ")"));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status validateTransport(const TransportOptions& transport) {
  const std::string key(kTransportField);
  if (transport.maxMessageSizeBytes != 0 &&
      (transport.maxMessageSizeBytes < kMinMessageSizeBytes ||
       transport.maxMessageSizeBytes > kMaxMessageSizeBytes)) {
    return invalid(
        absl::StrCat(key, ".max_message_size_bytes"),
        absl::StrCat(transport.maxMessageSizeBytes,
                     " is not 0 (the default) or in [", kMinMessageSizeBytes,
                     ", ", kMaxMessageSizeBytes, "]"));
  }
  if (transport.socketReceiveBufferBytes < 0 ||
      transport.socketSendBufferBytes < 0 ||
      transport.writerUnackedBytesLimit < 0) {
    return invalid(key, "the buffer sizes must be >= 0 (0: the default)");
  }
  if (transport.maxProcessesPerHost < 0 ||
      transport.maxProcessesPerHost > kMaxProcessesPerHost) {
    return invalid(absl::StrCat(key, ".max_processes_per_host"),
                   absl::StrCat(transport.maxProcessesPerHost,
                                " is not 0 (the default) or in [1, ",
                                kMaxProcessesPerHost, "]"));
  }
  if (transport.leaseDuration < absl::ZeroDuration()) {
    return invalid(absl::StrCat(key, ".lease_duration_ms"),
                   "must be >= 0 (0: the default)");
  }
  if (transport.simulatedSendLossPerMille < 0 ||
      transport.simulatedSendLossPerMille >= kPerMille) {
    return invalid(absl::StrCat(key, ".simulated_send_loss_per_mille"),
                   absl::StrCat(transport.simulatedSendLossPerMille,
                                " is not in [0, ", kPerMille, ")"));
  }
  return absl::OkStatus();
}

absl::StatusOr<NetworkConfig> fromProto(
    const robot_ipc_proto::NetworkConfig& message) {
  NetworkConfig config;
  config.nodes.reserve(message.nodes_size());
  for (const NodeProto& node : message.nodes()) {
    config.nodes.push_back(
        NodeEndpoint{.name = node.name(), .host = node.host()});
  }
  if (message.domain_id() >
      static_cast<uint32_t>(NetworkConfig::kMaxDomainId)) {
    return invalid("domain_id", absl::StrCat(message.domain_id(),
                                             " is not a domain id in [0, ",
                                             NetworkConfig::kMaxDomainId, "]"));
  }
  config.domainId = static_cast<int>(message.domain_id());
  config.domainTag = message.domain_tag();
  for (int index = 0; index < message.topics_size(); ++index) {
    const TopicProto& topic = message.topics(index);
    const std::string profileKey = fieldKey(
        kTopicsField, static_cast<size_t>(index), topic.name(), "qos_profile");
    if (topic.qos_profile().empty()) {
      return invalid(profileKey, "is missing");
    }
    const absl::StatusOr<QosProfile> profile =
        parseQosProfile(topic.qos_profile());
    if (!profile.ok()) {
      return invalid(profileKey, profile.status().message());
    }
    config.topics.push_back(TopicQos{
        .topic = topic.name(),
        .profile = *profile,
        .publishers = {topic.publishers().begin(), topic.publishers().end()}});
  }
  if (!message.default_qos_profile().empty()) {
    const absl::StatusOr<QosProfile> profile =
        parseQosProfile(message.default_qos_profile());
    if (!profile.ok()) {
      return invalid("default_qos_profile", profile.status().message());
    }
    config.defaultQosProfile = *profile;
  }
  const TransportProto& transport = message.transport();
  config.transport = TransportOptions{
      .maxMessageSizeBytes =
          static_cast<int64_t>(transport.max_message_size_bytes()),
      .socketReceiveBufferBytes =
          static_cast<int64_t>(transport.socket_receive_buffer_bytes()),
      .socketSendBufferBytes =
          static_cast<int64_t>(transport.socket_send_buffer_bytes()),
      .writerUnackedBytesLimit =
          static_cast<int64_t>(transport.writer_unacked_bytes_limit()),
      .maxProcessesPerHost = static_cast<int>(
          std::min<uint32_t>(transport.max_processes_per_host(),
                             std::numeric_limits<int32_t>::max())),
      .leaseDuration = absl::Milliseconds(transport.lease_duration_ms()),
      .simulatedSendLossPerMille = static_cast<int>(std::min<uint32_t>(
          transport.simulated_send_loss_per_mille(), kPerMille)),
  };
  ROBOT_IPC_RETURN_IF_ERROR(validateNetworkConfig(config));
  return config;
}

robot_ipc_proto::NetworkConfig toProto(const NetworkConfig& config) {
  robot_ipc_proto::NetworkConfig message;
  for (const NodeEndpoint& node : config.nodes) {
    NodeProto& entry = *message.add_nodes();
    entry.set_name(node.name);
    entry.set_host(node.host);
  }
  message.set_domain_id(static_cast<uint32_t>(std::max(config.domainId, 0)));
  message.set_domain_tag(config.domainTag);
  for (const TopicQos& topic : config.topics) {
    TopicProto& entry = *message.add_topics();
    entry.set_name(topic.topic);
    entry.set_qos_profile(std::string(qosProfileName(topic.profile)));
    for (const std::string& publisher : topic.publishers) {
      entry.add_publishers(publisher);
    }
  }
  if (config.defaultQosProfile != QosProfile::kReliableEvents) {
    message.set_default_qos_profile(
        std::string(qosProfileName(config.defaultQosProfile)));
  }
  const TransportOptions& transport = config.transport;
  if (!(transport == TransportOptions{})) {
    TransportProto& entry = *message.mutable_transport();
    entry.set_max_message_size_bytes(
        static_cast<uint32_t>(transport.maxMessageSizeBytes));
    entry.set_socket_receive_buffer_bytes(
        static_cast<uint32_t>(transport.socketReceiveBufferBytes));
    entry.set_socket_send_buffer_bytes(
        static_cast<uint32_t>(transport.socketSendBufferBytes));
    entry.set_writer_unacked_bytes_limit(
        static_cast<uint32_t>(transport.writerUnackedBytesLimit));
    entry.set_max_processes_per_host(
        static_cast<uint32_t>(transport.maxProcessesPerHost));
    entry.set_lease_duration_ms(static_cast<uint32_t>(
        absl::ToInt64Milliseconds(transport.leaseDuration)));
    entry.set_simulated_send_loss_per_mille(
        static_cast<uint32_t>(transport.simulatedSendLossPerMille));
  }
  return message;
}

// One line of the network file: `message` printed on a single line.
std::string singleLine(const google::protobuf::Message& message) {
  google::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(/*single_line_mode=*/true);
  std::string line;
  // Printing a message without unknown fields into a string cannot fail.
  static_cast<void>(printer.PrintToString(message, &line));
  return std::string(absl::StripTrailingAsciiWhitespace(line));
}

}  // namespace

bool isLoopbackHost(absl::string_view host) {
  return host == kLocalhostName ||
         (isIpv4Address(host) && absl::StartsWith(host, kLoopbackPrefix));
}

bool isIpv4Address(absl::string_view host) {
  const std::vector<absl::string_view> octets = absl::StrSplit(host, '.');
  if (octets.size() != kIpv4Octets) {
    return false;
  }
  for (const absl::string_view octet : octets) {
    int value = 0;
    if (octet.empty() || octet.size() > kMaxOctetDigits) {
      return false;
    }
    for (const char digit : octet) {
      if (!absl::ascii_isdigit(static_cast<unsigned char>(digit))) {
        return false;
      }
    }
    if (!absl::SimpleAtoi(octet, &value) || value > kMaxOctet) {
      return false;
    }
  }
  return true;
}

const NodeEndpoint* absl_nullable NetworkConfig::find(
    absl::string_view name) const {
  for (const NodeEndpoint& node : nodes) {
    if (node.name == name) {
      return &node;
    }
  }
  return nullptr;
}

std::vector<std::string> NetworkConfig::nodeNames() const {
  std::vector<std::string> names;
  names.reserve(nodes.size());
  for (const NodeEndpoint& node : nodes) {
    names.push_back(node.name);
  }
  return names;
}

const TopicQos* absl_nullable NetworkConfig::findTopic(
    absl::string_view topic) const {
  for (const TopicQos& entry : topics) {
    if (entry.topic == topic) {
      return &entry;
    }
  }
  return nullptr;
}

QosProfile NetworkConfig::qosProfileOf(absl::string_view topic) const {
  const TopicQos* absl_nullable entry = findTopic(topic);
  return entry == nullptr ? defaultQosProfile : entry->profile;
}

std::vector<std::string> NetworkConfig::topicsPublishedBy(
    absl::string_view nodeName) const {
  std::vector<std::string> published;
  for (const TopicQos& entry : topics) {
    for (const std::string& publisher : entry.publishers) {
      if (publisher == nodeName) {
        published.push_back(entry.topic);
        break;
      }
    }
  }
  return published;
}

std::vector<std::string> NetworkConfig::peerHosts() const {
  std::vector<std::string> hosts;
  absl::flat_hash_set<absl::string_view> seen;
  for (const NodeEndpoint& node : nodes) {
    if (seen.insert(node.host).second) {
      hosts.push_back(node.host);
    }
  }
  return hosts;
}

absl::Status validateTopicName(absl::string_view topic) {
  if (topic.empty()) {
    return absl::InvalidArgumentError("the topic is empty");
  }
  if (absl::ascii_isdigit(static_cast<unsigned char>(topic.front())) ||
      !isRunOf(topic, kTopicNamePunctuation)) {
    return absl::InvalidArgumentError(
        absl::StrCat("'", topic,
                     "' is not a DDS topic name: letters, digits, '_' and '/', "
                     "not starting with a digit"));
  }
  return absl::OkStatus();
}

absl::Status validateNetworkConfig(const NetworkConfig& config) {
  ROBOT_IPC_RETURN_IF_ERROR(validateNodes(config));
  if (config.domainId < 0 || config.domainId > NetworkConfig::kMaxDomainId) {
    return invalid("domain_id",
                   absl::StrCat(config.domainId, " is not a domain id in [0, ",
                                NetworkConfig::kMaxDomainId, "]"));
  }
  if (!isRunOf(config.domainTag, kDomainTagPunctuation)) {
    return invalid(
        "domain_tag",
        absl::StrCat(
            "'", config.domainTag,
            "': a domain tag is a run of letters, digits, '_', '-' and '.'"));
  }
  ROBOT_IPC_RETURN_IF_ERROR(validateTopics(config));
  ROBOT_IPC_RETURN_IF_ERROR(validateTransport(config.transport));
  if (const int lastPort = unicastPortRange(config).last; lastPort > kMaxPort) {
    return invalid(
        "domain_id",
        absl::StrCat(
            "domain ", config.domainId, " with ",
            lastPort - unicastPortRange(config).first + 1,
            " ports per host reaches UDP port ", lastPort, ", beyond ",
            kMaxPort,
            ": a smaller domain_id or transport.max_processes_per_host"));
  }
  return absl::OkStatus();
}

absl::StatusOr<NetworkConfig> parseNetworkConfig(absl::string_view text,
                                                 absl::string_view source) {
  robot_ipc_proto::NetworkConfig message;
  ROBOT_IPC_RETURN_IF_ERROR(parseTextproto(text, source, message));
  absl::StatusOr<NetworkConfig> config = fromProto(message);
  if (!config.ok()) {
    return withSource(source, config.status());
  }
  return config;
}

absl::StatusOr<NetworkConfig> loadNetworkConfig(const std::string& path) {
  robot_ipc_proto::NetworkConfig message;
  absl::Status loaded = loadTextproto(path, message);
  if (absl::IsNotFound(loaded)) {
    return absl::NotFoundError(
        absl::StrCat("cannot open the network file '", path, "'"));
  }
  if (!loaded.ok()) return loaded;
  absl::StatusOr<NetworkConfig> config = fromProto(message);
  if (!config.ok()) {
    return withSource(path, config.status());
  }
  return config;
}

std::string formatNetworkConfig(const NetworkConfig& config) {
  const robot_ipc_proto::NetworkConfig message = toProto(config);
  std::string text;
  robot_ipc_proto::NetworkConfig scalars = message;
  scalars.clear_nodes();
  scalars.clear_topics();
  scalars.clear_transport();
  if (const std::string line = singleLine(scalars); !line.empty()) {
    absl::StrAppend(&text, line, "\n");
  }
  for (const NodeProto& node : message.nodes()) {
    robot_ipc_proto::NetworkConfig entry;
    *entry.add_nodes() = node;
    absl::StrAppend(&text, singleLine(entry), "\n");
  }
  for (const TopicProto& topic : message.topics()) {
    robot_ipc_proto::NetworkConfig entry;
    *entry.add_topics() = topic;
    absl::StrAppend(&text, singleLine(entry), "\n");
  }
  if (message.has_transport()) {
    robot_ipc_proto::NetworkConfig entry;
    *entry.mutable_transport() = message.transport();
    absl::StrAppend(&text, singleLine(entry), "\n");
  }
  return text;
}

NetworkConfig localhostNetworkConfig() {
  absl::StatusOr<NetworkConfig> config =
      parseNetworkConfig(kLocalhostNetworkText, "config/ipc/network.textproto");
  // The text is the shipped file, built in; network_config_test parses the same
  // file.
  ABSL_CHECK_OK(config.status());
  return *std::move(config);
}

}  // namespace robot::ipc
