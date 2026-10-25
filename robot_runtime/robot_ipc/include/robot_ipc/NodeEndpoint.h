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

#include "absl/strings/string_view.h"

namespace robot::ipc {

/** The host of a single-machine network: every process on the loopback
 * interface. */
inline constexpr absl::string_view kLoopbackHost = "127.0.0.1";

/** True for an address of the loopback interface ("127.0.0.1", any 127.x.y.z,
 * and "localhost"). */
bool isLoopbackHost(absl::string_view host);

/** True for a dotted-quad IPv4 address, such as "192.168.1.10" (a host name is
 * not). */
bool isIpv4Address(absl::string_view host);

/**
 * One node of the network: a process that publishes under this name, and the
 * machine it runs on. Built from a robot_ipc_proto.NetworkConfig.Node of the
 * network file (robot_runtime/robot_ipc/proto/network_config.proto).
 *
 * DDS needs no address per node: the bus sends its discovery packets to the
 * host of every node (the unicast peer list), and a process publishing as a
 * node uses the network interface of its node's host when that host is an IPv4
 * address. The ports follow from NetworkConfig::domainId (DomainConfig.h).
 */
struct NodeEndpoint {
  /** The node name a process publishes as (BusOptions::nodeName), e.g. "robot".
   */
  std::string name;
  /** The machine of the node: an IPv4 address or a host name, e.g.
   * "192.168.1.10" or "127.0.0.1". */
  std::string host;

  bool operator==(const NodeEndpoint& other) const = default;
};

}  // namespace robot::ipc
