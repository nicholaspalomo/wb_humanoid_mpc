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

#include "robot_ipc/DomainConfig.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"

#include "robot_ipc/NodeEndpoint.h"

namespace robot::ipc {
namespace {

// The XML of the configuration. The Python bus writes the same text
// (robot_ipc/domain_config.py).
// LINT.IfChange(domain_config_xml)
// "&<>\"'" escaped, for a host or a tag in an attribute or an element.
std::string xmlEscaped(absl::string_view text) {
  std::string escaped;
  for (const char character : text) {
    switch (character) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&apos;";
        break;
      default:
        escaped += character;
    }
  }
  return escaped;
}

std::string bytes(int64_t value) {
  return absl::StrCat(value, "B");
}

std::string generalSection(absl::string_view interfaceAddress,
                           const TransportOptions& transport) {
  std::string general = "<General>";
  if (interfaceAddress.empty()) {
    absl::StrAppend(
        &general,
        R"(<Interfaces><NetworkInterface autodetermine="true"/></Interfaces>)");
  } else {
    absl::StrAppend(&general, R"(<Interfaces><NetworkInterface address=")",
                    xmlEscaped(interfaceAddress), R"("/></Interfaces>)");
  }
  absl::StrAppend(&general, "<AllowMulticast>false</AllowMulticast>");
  if (transport.maxMessageSizeBytes > 0) {
    absl::StrAppend(&general, "<MaxMessageSize>",
                    bytes(transport.maxMessageSizeBytes), "</MaxMessageSize>");
  }
  absl::StrAppend(&general, "</General>");
  return general;
}

std::string discoverySection(const NetworkConfig& network) {
  const int maxProcesses = network.transport.maxProcessesPerHost > 0
                               ? network.transport.maxProcessesPerHost
                               : TransportOptions::kDefaultMaxProcessesPerHost;
  const absl::Duration lease =
      network.transport.leaseDuration > absl::ZeroDuration()
          ? network.transport.leaseDuration
          : TransportOptions::kDefaultLeaseDuration;
  std::string discovery = "<Discovery>";
  if (!network.domainTag.empty()) {
    absl::StrAppend(&discovery, "<Tag>", xmlEscaped(network.domainTag),
                    "</Tag>");
  }
  absl::StrAppend(
      &discovery,
      "<ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>",
      maxProcesses - 1, "</MaxAutoParticipantIndex>",
      R"(<Peers AddLocalhost="false">)");
  for (const std::string& host : network.peerHosts()) {
    absl::StrAppend(&discovery, R"(<Peer Address=")", xmlEscaped(host),
                    R"("/>)");
  }
  absl::StrAppend(&discovery, "</Peers><LeaseDuration>",
                  absl::ToInt64Milliseconds(lease),
                  "ms</LeaseDuration></Discovery>");
  return discovery;
}

std::string internalSection(const TransportOptions& transport) {
  const int64_t unacked =
      transport.writerUnackedBytesLimit > 0
          ? transport.writerUnackedBytesLimit
          : TransportOptions::kDefaultWriterUnackedBytesLimit;
  std::string internal = "<Internal>";
  if (transport.socketReceiveBufferBytes > 0) {
    absl::StrAppend(&internal,
                    R"(<SocketReceiveBufferSize min="default" max=")",
                    bytes(transport.socketReceiveBufferBytes), R"("/>)");
  }
  if (transport.socketSendBufferBytes > 0) {
    absl::StrAppend(&internal, R"(<SocketSendBufferSize min="default" max=")",
                    bytes(transport.socketSendBufferBytes), R"("/>)");
  }
  absl::StrAppend(&internal, "<Watermarks><WhcHigh>", bytes(unacked),
                  "</WhcHigh><WhcAdaptive>false</WhcAdaptive></Watermarks>");
  if (transport.simulatedSendLossPerMille > 0) {
    absl::StrAppend(&internal, "<Test><XmitLossiness>",
                    transport.simulatedSendLossPerMille,
                    "</XmitLossiness></Test>");
  }
  absl::StrAppend(&internal, "</Internal>");
  return internal;
}
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/domain_config.py:domain_config_xml)

// Whether `address` (IPv4) belongs to an interface of this machine: a UDP
// socket can bind it.
bool isLocalAddress(const std::string& address) {
  sockaddr_in socketAddress{};
  socketAddress.sin_family = AF_INET;
  // sin_port 0: any port.
  if (::inet_pton(AF_INET, address.c_str(), &socketAddress.sin_addr) != 1) {
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

}  // namespace

PortRange unicastPortRange(const NetworkConfig& network) {
  const int maxProcesses = network.transport.maxProcessesPerHost > 0
                               ? network.transport.maxProcessesPerHost
                               : TransportOptions::kDefaultMaxProcessesPerHost;
  const int domainBase = kDdsPortBase + kDdsDomainGain * network.domainId;
  return PortRange{.first = domainBase + kDdsUnicastDiscoveryOffset,
                   .last = domainBase + kDdsUnicastDataOffset +
                           kDdsParticipantGain * (maxProcesses - 1)};
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

absl::StatusOr<std::string> cycloneDomainConfig(const NetworkConfig& network,
                                                absl::string_view nodeName) {
  const absl::StatusOr<std::string> interfaceAddress =
      interfaceAddressFor(network, nodeName);
  if (!interfaceAddress.ok()) {
    return interfaceAddress.status();
  }
  return absl::StrCat(R"(<CycloneDDS><Domain Id="any">)",
                      generalSection(*interfaceAddress, network.transport),
                      discoverySection(network),
                      internalSection(network.transport),
                      "</Domain></CycloneDDS>");
}

}  // namespace robot::ipc
