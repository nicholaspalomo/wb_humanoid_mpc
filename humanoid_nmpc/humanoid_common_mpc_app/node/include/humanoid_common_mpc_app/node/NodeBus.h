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

#include <memory>
#include <string>

#include "absl/status/statusor.h"

#include "robot_ipc/Bus.h"
#include "robot_ipc/NetworkConfig.h"

namespace ocs2::humanoid::node {

/**
 * The network of a binary's --network_config: the network file at `networkConfigPath` (robot_ipc::loadNetworkConfig(),
 * strict), or, when the path is empty, the shipped single-machine network (config/ipc/network.textproto,
 * robot::ipc::localhostNetworkConfig()).
 */
absl::StatusOr<robot::ipc::NetworkConfig> loadNodeNetwork(const std::string& networkConfigPath);

/**
 * The bus of a binary that publishes as `nodeName` (--ipc_node) on the network of `networkConfigPath`
 * (loadNodeNetwork()), created and not yet started. The errors of Bus::Create(): InvalidArgument for a node name the
 * network does not have (the message lists the ones it has), Unavailable for an endpoint that cannot be bound.
 */
absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> createNodeBus(const std::string& networkConfigPath, const std::string& nodeName);

}  // namespace ocs2::humanoid::node
