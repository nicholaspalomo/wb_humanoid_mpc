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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"

#include "robot_ipc/Bus.h"
#include "robot_ipc/NetworkConfig.h"

namespace ocs2::humanoid::test_support {

/** A TCP port of the loopback interface that is free now (the kernel's choice). */
int freeLoopbackPort();

/**
 * Writes a network file at `path` that puts each of `nodes` on 127.0.0.1 at a free port, and loads it back, as the
 * processes of a test read it (--network_config).
 */
absl::StatusOr<robot::ipc::NetworkConfig> writeLoopbackNetworkFile(const std::string& path, const std::vector<std::string>& nodes);

/** A bus of `network` that publishes as `node`, with a short IO poll period. CHECK-fails when it cannot be created. */
std::unique_ptr<robot::ipc::Bus> createBus(const robot::ipc::NetworkConfig& network, const std::string& node);

/** Polls `condition` every 5 ms until it holds or `timeout` passes; whether it held. */
bool waitFor(const std::function<bool()>& condition, absl::Duration timeout);

}  // namespace ocs2::humanoid::test_support
