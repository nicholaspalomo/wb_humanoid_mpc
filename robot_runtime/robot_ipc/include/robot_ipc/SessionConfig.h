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

#pragma once

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "robot_ipc/NetworkConfig.h"

namespace robot::ipc {

// The TCP ports of a domain: process index i of a machine listens on
// kZenohPortBase + kZenohDomainGain * domainId + i, for i below
// TransportOptions::maxProcessesPerHost (at most kZenohDomainGain). 7447 is
// Zenoh's own port; domain 232 ends below 32768, where Linux's ephemeral ports
// start. The Python bus and the docs use the same numbers.
// LINT.IfChange(zenoh_ports)
inline constexpr int kZenohPortBase = 7447;
inline constexpr int kZenohDomainGain = 100;
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/session_config.py:zenoh_ports, //robot_runtime/robot_ipc/README.md:zenoh_ports)

/** A closed range of TCP ports. */
struct PortRange {
  int first = 0;
  int last = 0;

  bool operator==(const PortRange& other) const = default;
};

/**
 * The TCP ports the processes of one machine listen on in `network`'s domain:
 * the process of index i (the lowest port free when it opens its session)
 * listens on kZenohPortBase + kZenohDomainGain * domainId + i. Domain 0 with 20
 * processes: 7447-7466. A firewall between the machines opens this range for
 * TCP; the bus sends no multicast and no UDP.
 */
PortRange tcpPortRange(const NetworkConfig& network);

/**
 * The address the process of `nodeName` listens on: its node's host when that
 * is an IPv4 address (127.0.0.1 for a loopback host); for a bus that only
 * subscribes (`nodeName` empty), 127.0.0.1 when every node is on loopback, else
 * the first node host that is an address of this machine. Empty: every
 * interface (a node whose host is a host name, or a subscriber on a machine
 * that is no node's). A node host that is an IPv4 address of no interface of
 * this machine is Unavailable: the node runs on another machine.
 */
absl::StatusOr<std::string> interfaceAddressFor(const NetworkConfig& network,
                                                absl::string_view nodeName);

/**
 * The endpoints every process of `network` connects to: "tcp/<host>:<port>"
 * for the host of every node (NetworkConfig::peerHosts()) and every port of
 * tcpPortRange(), host by host. Zenoh retries the ones without a listener in
 * the background, so the processes start in any order.
 */
std::vector<std::string> connectEndpoints(const NetworkConfig& network);

/**
 * Whether a TCP listener can bind `port` on `address` (empty: every interface)
 * now: the bus opens its session on the lowest free port of tcpPortRange().
 */
bool isTcpPortFree(absl::string_view address, int port);

/**
 * The Zenoh configuration (JSON, which Zenoh reads as JSON5) of a process of
 * `network` publishing as `nodeName` (empty: only subscribing) that listens on
 * `listenPort`, which the bus passes to z_open(): peer mode, a TCP listener on
 * interfaceAddressFor() and `listenPort`, connectEndpoints() but the listener
 * itself retried in the background, no multicast scouting, gossip to learn of
 * the peers' peers, and the lease and TCP buffers of NetworkConfig::transport.
 * The same arguments give the same text in C++ and in Python
 * (robot_ipc.session_config.zenoh_session_config). Errors are those of
 * interfaceAddressFor().
 */
absl::StatusOr<std::string> zenohSessionConfig(const NetworkConfig& network,
                                               absl::string_view nodeName,
                                               int listenPort);

}  // namespace robot::ipc
