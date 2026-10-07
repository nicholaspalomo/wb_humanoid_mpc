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

/**
 * A port that only a network built in code may name (never a network file, where a port of 0 is a missing port): the
 * node binds an ephemeral port that the kernel picks, which Bus::boundEndpoint() reports, and no other node can
 * connect to it from the network alone (Bus::connect() can). Tests use it so that they never collide on a port.
 */
inline constexpr int kEphemeralPort = 0;

/** The largest TCP port. */
inline constexpr int kMaxPort = 65535;

/** True for "*" and "0.0.0.0", the hosts that bind every interface. */
bool isWildcardHost(absl::string_view host);

/**
 * One node of the network: where its PUB socket binds, and where every SUB socket of the network connects. Built from a
 * robot_ipc_proto.NetworkConfig.Node of the network file (robot_runtime/robot_ipc/proto/network_config.proto).
 */
struct NodeEndpoint {
  /** The node name a process publishes as (BusOptions::nodeName), e.g. "robot". */
  std::string name;
  /**
   * The address the other processes connect to, and the one this node binds unless bindHost is set. A wildcard host
   * ("0.0.0.0" or "*") binds every interface, and the other processes then connect over the loopback interface, which
   * only suits a single machine.
   */
  std::string host;
  /** The TCP port, in [1, kMaxPort], or kEphemeralPort. */
  int port = kEphemeralPort;
  /**
   * The interface to bind instead of host, e.g. "0.0.0.0" on a machine whose address peers know. Empty: host. Defaulted
   * here, so that the designated initializers that leave it out (`{.name = ..., .host = ..., .port = ...}`) set it.
   */
  std::string bindHost = "";

  /** "tcp://<bindHost or host>:<port>", with "*" for a wildcard host and for kEphemeralPort. */
  std::string bindEndpoint() const;
  /** "tcp://<host>:<port>", with 127.0.0.1 for a wildcard host. */
  std::string connectEndpoint() const;

  bool operator==(const NodeEndpoint& other) const = default;
};

}  // namespace robot::ipc
