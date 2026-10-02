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
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "robot_ipc/NodeEndpoint.h"
#include "robot_ipc_proto/network_config.pb.h"
#include "robot_runtime/robot_ipc/src/StatusMacros.h"
#include "robot_runtime/robot_ipc/src/Textproto.h"

namespace robot::ipc {
namespace {

// The rules of a node and the words of their error messages. The Python loader applies the same ones.
// LINT.IfChange(node_rules)
constexpr int kMinFilePort = 1;
constexpr absl::string_view kNodesField = "nodes";
// A wildcard host binds every interface, and the other processes reach it over loopback.
constexpr std::array<absl::string_view, 2> kWildcardHosts = {"*", "0.0.0.0"};
constexpr absl::string_view kLoopbackHost = "127.0.0.1";
// A node name is a non-empty run of letters, digits and these.
constexpr absl::string_view kNodeNamePunctuation = "_-";
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/network_config.py:node_rules)

absl::Status invalid(absl::string_view key, absl::string_view problem) {
  return absl::InvalidArgumentError(absl::StrCat(key, ": ", problem));
}

absl::Status withSource(absl::string_view source, const absl::Status& status) {
  return absl::Status(status.code(), absl::StrCat(source, ": ", status.message()));
}

// "nodes[1] (mpc)": the position of the node in the file, and its name when it has one.
std::string nodeKey(size_t index, absl::string_view name) {
  const std::string position = absl::StrCat(kNodesField, "[", index, "]");
  return name.empty() ? position : absl::StrCat(position, " (", name, ")");
}

std::string fieldKey(size_t index, absl::string_view name, absl::string_view field) {
  return absl::StrCat(nodeKey(index, name), ".", field);
}

bool isValidNodeName(absl::string_view name) {
  if (name.empty()) {
    return false;
  }
  for (const char character : name) {
    if (!absl::ascii_isalnum(static_cast<unsigned char>(character)) && !absl::StrContains(kNodeNamePunctuation, character)) {
      return false;
    }
  }
  return true;
}

// A host is an IPv4 address, a host name or a wildcard; not an endpoint ("tcp://..."), not "host:port" and not IPv6.
absl::Status validateHost(absl::string_view key, absl::string_view host) {
  if (host.empty()) {
    return invalid(key, "is missing");
  }
  if (isWildcardHost(host)) {
    return absl::OkStatus();
  }
  for (const char character : host) {
    if (absl::ascii_isspace(static_cast<unsigned char>(character)) || character == '/' || character == ':') {
      return invalid(key, absl::StrCat("'", host, "' is not a host: write an IPv4 address or a host name, without a scheme or a port"));
    }
  }
  return absl::OkStatus();
}

std::string portProblem(int64_t port) {
  return absl::StrCat(port, " is not a port in [", kMinFilePort, ", ", kMaxPort, "]");
}

const google::protobuf::FieldDescriptor& nodeField(int number) {
  return *robot_ipc_proto::NetworkConfig::Node::descriptor()->FindFieldByNumber(number);
}

// The checks only a file needs: a network built in code may leave out what a file must name (kEphemeralPort).
absl::StatusOr<NetworkConfig> fromProto(const robot_ipc_proto::NetworkConfig& message) {
  const std::string nameField(nodeField(robot_ipc_proto::NetworkConfig::Node::kNameFieldNumber).name());
  const std::string portField(nodeField(robot_ipc_proto::NetworkConfig::Node::kPortFieldNumber).name());
  NetworkConfig config;
  config.nodes.reserve(message.nodes_size());
  for (int index = 0; index < message.nodes_size(); ++index) {
    const robot_ipc_proto::NetworkConfig::Node& node = message.nodes(index);
    const size_t position = static_cast<size_t>(index);
    if (node.name().empty()) {
      return invalid(fieldKey(position, node.name(), nameField), "is missing");
    }
    if (node.port() == 0) {
      return invalid(fieldKey(position, node.name(), portField),
                     absl::StrCat("is missing (a network file names a port in [", kMinFilePort, ", ", kMaxPort, "])"));
    }
    if (node.port() > static_cast<uint32_t>(kMaxPort)) {
      return invalid(fieldKey(position, node.name(), portField), portProblem(node.port()));
    }
    config.nodes.push_back(
        NodeEndpoint{.name = node.name(), .host = node.host(), .port = static_cast<int>(node.port()), .bindHost = node.bind_host()});
  }
  ROBOT_IPC_RETURN_IF_ERROR(validateNetworkConfig(config));
  return config;
}

}  // namespace

bool isWildcardHost(absl::string_view host) {
  return std::find(kWildcardHosts.begin(), kWildcardHosts.end(), host) != kWildcardHosts.end();
}

std::string NodeEndpoint::bindEndpoint() const {
  const absl::string_view address = bindHost.empty() ? absl::string_view(host) : absl::string_view(bindHost);
  const absl::string_view boundAddress = isWildcardHost(address) ? absl::string_view("*") : address;
  if (port == kEphemeralPort) {
    return absl::StrCat("tcp://", boundAddress, ":*");
  }
  return absl::StrCat("tcp://", boundAddress, ":", port);
}

std::string NodeEndpoint::connectEndpoint() const {
  const absl::string_view address = isWildcardHost(host) ? kLoopbackHost : absl::string_view(host);
  return absl::StrCat("tcp://", address, ":", port);
}

const NodeEndpoint* NetworkConfig::find(absl::string_view name) const {
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

absl::Status validateNetworkConfig(const NetworkConfig& config) {
  const std::string nameField(nodeField(robot_ipc_proto::NetworkConfig::Node::kNameFieldNumber).name());
  const std::string hostField(nodeField(robot_ipc_proto::NetworkConfig::Node::kHostFieldNumber).name());
  const std::string portField(nodeField(robot_ipc_proto::NetworkConfig::Node::kPortFieldNumber).name());
  const std::string bindHostField(nodeField(robot_ipc_proto::NetworkConfig::Node::kBindHostFieldNumber).name());
  if (config.nodes.empty()) {
    return invalid(kNodesField, "names no node");
  }
  // name -> its position, and "host:port" -> the key of the node that has it.
  absl::flat_hash_map<std::string, size_t> names;
  absl::flat_hash_map<std::string, std::string> endpoints;
  for (size_t index = 0; index < config.nodes.size(); ++index) {
    const NodeEndpoint& node = config.nodes[index];
    if (!isValidNodeName(node.name)) {
      return invalid(fieldKey(index, node.name, nameField),
                     node.name.empty() ? "is missing" : "a node name is a non-empty run of letters, digits, '_' and '-'");
    }
    const std::pair<absl::flat_hash_map<std::string, size_t>::iterator, bool> named = names.emplace(node.name, index);
    if (!named.second) {
      return invalid(fieldKey(index, node.name, nameField), absl::StrCat("is also the name of ", nodeKey(named.first->second, node.name)));
    }
    ROBOT_IPC_RETURN_IF_ERROR(validateHost(fieldKey(index, node.name, hostField), node.host));
    if (!node.bindHost.empty()) {
      ROBOT_IPC_RETURN_IF_ERROR(validateHost(fieldKey(index, node.name, bindHostField), node.bindHost));
    }
    if (node.port == kEphemeralPort) {
      continue;
    }
    if (node.port < kMinFilePort || node.port > kMaxPort) {
      return invalid(fieldKey(index, node.name, portField), portProblem(node.port));
    }
    const std::string endpoint = absl::StrCat(node.host, ":", node.port);
    const std::pair<absl::flat_hash_map<std::string, std::string>::iterator, bool> inserted =
        endpoints.emplace(endpoint, nodeKey(index, node.name));
    if (!inserted.second) {
      return invalid(nodeKey(index, node.name), absl::StrCat(endpoint, " is also the endpoint of ", inserted.first->second));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<NetworkConfig> parseNetworkConfig(absl::string_view text, absl::string_view source) {
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
  const absl::Status loaded = loadTextproto(path, message);
  if (absl::IsNotFound(loaded)) {
    return absl::NotFoundError(absl::StrCat("cannot open the network file '", path, "'"));
  }
  ROBOT_IPC_RETURN_IF_ERROR(loaded);
  absl::StatusOr<NetworkConfig> config = fromProto(message);
  if (!config.ok()) {
    return withSource(path, config.status());
  }
  return config;
}

std::string formatNetworkConfig(const NetworkConfig& config) {
  google::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(/*single_line_mode=*/true);
  std::string text;
  for (const NodeEndpoint& node : config.nodes) {
    robot_ipc_proto::NetworkConfig message;
    robot_ipc_proto::NetworkConfig::Node& entry = *message.add_nodes();
    entry.set_name(node.name);
    entry.set_host(node.host);
    entry.set_port(node.port < 0 ? 0 : static_cast<uint32_t>(node.port));
    entry.set_bind_host(node.bindHost);
    std::string line;
    // Printing a message without unknown fields into a string cannot fail.
    static_cast<void>(printer.PrintToString(message, &line));
    absl::StrAppend(&text, absl::StripTrailingAsciiWhitespace(line), "\n");
  }
  return text;
}

// LINT.IfChange(localhost_nodes)
NetworkConfig localhostNetworkConfig() {
  NetworkConfig config;
  config.nodes = {
      NodeEndpoint{.name = "robot", .host = "127.0.0.1", .port = 5600},
      NodeEndpoint{.name = "mpc", .host = "127.0.0.1", .port = 5610},
      NodeEndpoint{.name = "operator", .host = "127.0.0.1", .port = 5620},
      NodeEndpoint{.name = "teleop", .host = "127.0.0.1", .port = 5621},
  };
  return config;
}
// clang-format off
// LINT.ThenChange(//config/ipc/network.textproto:localhost_nodes, //robot_runtime/robot_ipc/python/robot_ipc/network_config.py:localhost_nodes)
// clang-format on

}  // namespace robot::ipc
