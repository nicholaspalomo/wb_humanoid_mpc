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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "robot_ipc/NetworkConfig.h"

namespace robot::ipc {

// The DDSI port mapping (DDSI-RTPS 2.5, 9.6.1.1), which Cyclone DDS applies
// with its defaults. The Python bus and the docs use the same numbers.
// LINT.IfChange(dds_ports)
inline constexpr int kDdsPortBase = 7400;
inline constexpr int kDdsDomainGain = 250;
inline constexpr int kDdsParticipantGain = 2;
inline constexpr int kDdsUnicastDiscoveryOffset = 10;
inline constexpr int kDdsUnicastDataOffset = 11;
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/network_config.py:dds_ports, //robot_runtime/robot_ipc/README.md:dds_ports)

/** A closed range of UDP ports. */
struct PortRange {
  int first = 0;
  int last = 0;

  bool operator==(const PortRange& other) const = default;
};

/**
 * The unicast UDP ports the processes of one machine use on `network`: process
 * index i (Cyclone picks the lowest free one) listens on kDdsPortBase +
 * kDdsDomainGain * domainId + kDdsUnicastDiscoveryOffset + kDdsParticipantGain
 * * i (discovery) and on the port after it (data), for i below
 * TransportOptions::maxProcessesPerHost. Domain 0 with 20 processes: 7410-7449.
 * A firewall between the machines opens this range for UDP; the bus sends no
 * multicast.
 */
PortRange unicastPortRange(const NetworkConfig& network);

/**
 * The address of the network interface the process of `nodeName` uses: its
 * node's host when that is an IPv4 address (127.0.0.1 for a loopback host); for
 * a bus that only subscribes (`nodeName` empty), 127.0.0.1 when every node is
 * on loopback, else the first node host that is an address of this machine.
 * Empty: Cyclone DDS picks the interface (a node whose host is a host name, or
 * a subscriber on a machine that is no node's). A node host that is an IPv4
 * address of no interface of this machine is Unavailable: the node runs on
 * another machine.
 */
absl::StatusOr<std::string> interfaceAddressFor(const NetworkConfig& network,
                                                absl::string_view nodeName);

/**
 * The Cyclone DDS configuration (XML) of a process of `network` publishing as
 * `nodeName` (empty: only subscribing), which the bus passes to
 * dds_create_domain(): the interface of interfaceAddressFor(), no multicast,
 * unicast discovery to the host of every node with automatic process indices,
 * the domain tag, the lease and the transport sizes of
 * NetworkConfig::transport. The same network and node give the same text in C++
 * and in Python (robot_ipc.domain_config.cyclone_domain_config). Errors are
 * those of interfaceAddressFor().
 */
absl::StatusOr<std::string> cycloneDomainConfig(const NetworkConfig& network,
                                                absl::string_view nodeName);

}  // namespace robot::ipc
