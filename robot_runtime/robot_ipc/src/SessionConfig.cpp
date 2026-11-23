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

#include "robot_ipc/SessionConfig.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/NodeEndpoint.h"

namespace robot::ipc {
namespace {

// The address a listener on every interface binds.
constexpr absl::string_view kEveryInterface = "0.0.0.0";

int maxProcessesPerHost(const TransportOptions& transport) {
  return transport.maxProcessesPerHost > 0
             ? transport.maxProcessesPerHost
             : TransportOptions::kDefaultMaxProcessesPerHost;
}

// The JSON of the configuration. The Python bus writes the same text
// (robot_ipc/session_config.py).
// LINT.IfChange(zenoh_session_json)
// A JSON string: `text` quoted, with '"' and '\' escaped (a host is printable
// ASCII, NetworkConfig.h).
std::string jsonString(absl::string_view text) {
  std::string quoted = "\"";
  for (const char character : text) {
    if (character == '"' || character == '\\') {
      quoted += '\\';
    }
    quoted += character;
  }
  quoted += '"';
  return quoted;
}

std::string jsonStrings(absl::Span<const std::string> texts) {
  std::vector<std::string> quoted;
  quoted.reserve(texts.size());
  for (const std::string& text : texts) {
    quoted.push_back(jsonString(text));
  }
  return absl::StrCat("[", absl::StrJoin(quoted, ","), "]");
}

std::string tcpEndpoint(absl::string_view host, int port) {
  return absl::StrCat("tcp/", host, ":", port);
}

std::string linkSection(const TransportOptions& transport) {
  const absl::Duration lease = transport.leaseDuration > absl::ZeroDuration()
                                   ? transport.leaseDuration
                                   : TransportOptions::kDefaultLeaseDuration;
  std::string link = absl::StrCat(R"("link":{"tx":{"lease":)",
                                  absl::ToInt64Milliseconds(lease), "}");
  std::vector<std::string> tcp;
  if (transport.socketReceiveBufferBytes > 0) {
    tcp.push_back(
        absl::StrCat(R"("so_rcvbuf":)", transport.socketReceiveBufferBytes));
  }
  if (transport.socketSendBufferBytes > 0) {
    tcp.push_back(
        absl::StrCat(R"("so_sndbuf":)", transport.socketSendBufferBytes));
  }
  if (!tcp.empty()) {
    absl::StrAppend(&link, R"(,"tcp":{)", absl::StrJoin(tcp, ","), "}");
  }
  absl::StrAppend(&link, "}");
  return link;
}
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/session_config.py:zenoh_session_json)

// Whether `address` (IPv4) belongs to an interface of this machine: a socket
// can bind it.
bool isLocalAddress(absl::string_view address) {
  sockaddr_in socketAddress{};
  socketAddress.sin_family = AF_INET;
  // sin_port 0: any port. inet_pton() needs a NUL-terminated address.
  const std::string addressText(address);
  if (::inet_pton(AF_INET, addressText.c_str(), &socketAddress.sin_addr) != 1) {
    return false;
  }
  const int socketFd =
      ::socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, /*protocol=*/0);
  if (socketFd < 0) {
    return false;
  }
  const bool bound =
      ::bind(socketFd, reinterpret_cast<const sockaddr*>(&socketAddress),
             sizeof(socketAddress)) == 0;
  ::close(socketFd);
  return bound;
}

bool everyNodeIsOnLoopback(const NetworkConfig& network) {
  for (const NodeEndpoint& node : network.nodes) {
    if (!isLoopbackHost(node.host)) {
      return false;
    }
  }
  return true;
}

// connectEndpoints() without the session's own listener, which Zenoh would
// refuse (a connection to self) on every retry.
std::vector<std::string> peerEndpoints(const NetworkConfig& network,
                                       absl::string_view ownEndpoint) {
  std::vector<std::string> endpoints = connectEndpoints(network);
  endpoints.erase(std::remove(endpoints.begin(), endpoints.end(), ownEndpoint),
                  endpoints.end());
  return endpoints;
}

}  // namespace

PortRange tcpPortRange(const NetworkConfig& network) {
  const int first = kZenohPortBase + kZenohDomainGain * network.domainId;
  return PortRange{.first = first,
                   .last = first + maxProcessesPerHost(network.transport) - 1};
}

absl::StatusOr<std::string> interfaceAddressFor(const NetworkConfig& network,
                                                absl::string_view nodeName) {
  if (nodeName.empty()) {
    if (everyNodeIsOnLoopback(network)) {
      return std::string(kLoopbackHost);
    }
    for (const NodeEndpoint& node : network.nodes) {
      if (isIpv4Address(node.host) && !isLoopbackHost(node.host) &&
          isLocalAddress(node.host)) {
        return node.host;
      }
    }
    return std::string();
  }
  const NodeEndpoint* absl_nullable node = network.find(nodeName);
  if (node == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("node '", nodeName, "' is not in the network"));
  }
  if (isLoopbackHost(node->host)) {
    return std::string(kLoopbackHost);
  }
  if (!isIpv4Address(node->host)) {
    return std::string();
  }
  if (!isLocalAddress(node->host)) {
    return absl::UnavailableError(
        absl::StrCat("node '", node->name, "' is on ", node->host,
                     ", which is no address of this machine: run it on that "
                     "machine, or fix its host"));
  }
  return node->host;
}

std::vector<std::string> connectEndpoints(const NetworkConfig& network) {
  const PortRange ports = tcpPortRange(network);
  std::vector<std::string> endpoints;
  for (const std::string& host : network.peerHosts()) {
    for (int port = ports.first; port <= ports.last; ++port) {
      endpoints.push_back(tcpEndpoint(host, port));
    }
  }
  return endpoints;
}

bool isTcpPortFree(absl::string_view address, int port) {
  sockaddr_in socketAddress{};
  socketAddress.sin_family = AF_INET;
  socketAddress.sin_port = htons(static_cast<uint16_t>(port));
  const std::string addressText(address.empty() ? kEveryInterface : address);
  if (::inet_pton(AF_INET, addressText.c_str(), &socketAddress.sin_addr) != 1) {
    return false;
  }
  const int socketFd =
      ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, /*protocol=*/0);
  if (socketFd < 0) {
    return false;
  }
  // As a listener binds it: a port in TIME_WAIT is free.
  const int reuse = 1;
  static_cast<void>(
      ::setsockopt(socketFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)));
  const bool bound =
      ::bind(socketFd, reinterpret_cast<const sockaddr*>(&socketAddress),
             sizeof(socketAddress)) == 0;
  ::close(socketFd);
  return bound;
}

absl::StatusOr<std::string> zenohSessionConfig(const NetworkConfig& network,
                                               absl::string_view nodeName,
                                               int listenPort) {
  const absl::StatusOr<std::string> interfaceAddress =
      interfaceAddressFor(network, nodeName);
  if (!interfaceAddress.ok()) {
    return interfaceAddress.status();
  }
  const absl::string_view listenHost =
      interfaceAddress->empty() ? kEveryInterface : *interfaceAddress;
  // LINT.IfChange(zenoh_session_layout)
  return absl::StrCat(
      R"({"mode":"peer","listen":{"endpoints":)",
      jsonStrings({tcpEndpoint(listenHost, listenPort)}),
      R"(,"timeout_ms":0,"exit_on_failure":true},"connect":{"endpoints":)",
      jsonStrings(peerEndpoints(network, tcpEndpoint(listenHost, listenPort))),
      R"(,"timeout_ms":0,"exit_on_failure":false},)",
      R"("scouting":{"multicast":{"enabled":false},"gossip":{"enabled":true}},)",
      R"("transport":{)", linkSection(network.transport), "}}");
  // LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/session_config.py:zenoh_session_layout)
}

}  // namespace robot::ipc
