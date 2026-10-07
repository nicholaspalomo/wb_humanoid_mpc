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

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "robot_ipc/NodeEndpoint.h"

namespace robot::ipc {

/**
 * The nodes of the bus, as the network file names them: a textproto of robot_ipc_proto.NetworkConfig
 * (robot_runtime/robot_ipc/proto/network_config.proto), such as config/ipc/network.textproto:
 *
 *   nodes { name: "robot" host: "127.0.0.1" port: 5600 }
 *   nodes { name: "mpc" host: "192.168.1.20" port: 5610 bind_host: "0.0.0.0" }
 */
struct NetworkConfig {
  /** In the order of the file. */
  std::vector<NodeEndpoint> nodes;

  /** The node of that name, or nullptr. */
  const NodeEndpoint* absl_nullable find(absl::string_view name) const;
  /** The node names, in order, e.g. for an error message that lists the valid ones. */
  std::vector<std::string> nodeNames() const;

  bool operator==(const NetworkConfig& other) const = default;
};

/**
 * Reads and validates a network file. It is parsed strictly (google::protobuf::TextFormat): a syntax error, an unknown
 * field or a value of the wrong type is an InvalidArgument error that names the file, the line and the column, such as
 * "config/ipc/network.textproto:3:42: Message type "robot_ipc_proto.NetworkConfig.Node" has no field named "prot"."
 * The nodes are then validated, and an error names the node and its field, such as
 * "config/ipc/network.textproto: nodes[0] (robot).port: 70000 is not a port in [1, 65535]". A missing name, host or
 * port is an error too. A file that cannot be read is NotFound.
 */
absl::StatusOr<NetworkConfig> loadNetworkConfig(const std::string& path);

/** loadNetworkConfig() on the text of a network file; `source` names it in the error messages. */
absl::StatusOr<NetworkConfig> parseNetworkConfig(absl::string_view text, absl::string_view source);

/**
 * The checks of the network file that also apply to a network built in code (Bus::Create() runs them): at least one
 * node, valid and unique names, valid hosts, ports in [1, kMaxPort] or kEphemeralPort, no two nodes on one endpoint.
 * Errors are InvalidArgument and name the node, as in loadNetworkConfig().
 */
absl::Status validateNetworkConfig(const NetworkConfig& config);

/**
 * The network file of `config`, one line per node; parseNetworkConfig() reads it back as `config` when it is valid and
 * names no kEphemeralPort.
 */
std::string formatNetworkConfig(const NetworkConfig& config);

/** The shipped network file, config/ipc/network.textproto: robot, mpc, operator, teleop and config_push on 127.0.0.1. */
NetworkConfig localhostNetworkConfig();

}  // namespace robot::ipc
