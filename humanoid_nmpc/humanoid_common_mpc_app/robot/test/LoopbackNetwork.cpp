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

#include "humanoid_common_mpc_app/robot/test_support/LoopbackNetwork.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "robot_ipc/BusOptions.h"

namespace ocs2::humanoid::test_support {

int freeLoopbackPort() {
  const int socketFd = ::socket(AF_INET, SOCK_STREAM, /*protocol=*/0);
  CHECK_GE(socketFd, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  CHECK_EQ(::bind(socketFd, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
  socklen_t length = sizeof(address);
  CHECK_EQ(::getsockname(socketFd, reinterpret_cast<sockaddr*>(&address), &length), 0);
  ::close(socketFd);
  return ntohs(address.sin_port);
}

absl::StatusOr<robot::ipc::NetworkConfig> writeLoopbackNetworkFile(const std::string& path, const std::vector<std::string>& nodes) {
  {
    std::ofstream file(path);
    for (const std::string& node : nodes) {
      file << "nodes { name: \"" << node << "\" host: \"127.0.0.1\" port: " << freeLoopbackPort() << " }\n";
    }
  }
  return robot::ipc::loadNetworkConfig(path);
}

std::unique_ptr<robot::ipc::Bus> createBus(const robot::ipc::NetworkConfig& network, const std::string& node) {
  robot::ipc::BusOptions options;
  options.nodeName = node;
  options.network = network;
  options.ioPollPeriod = absl::Milliseconds(5);
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus = robot::ipc::Bus::Create(std::move(options));
  CHECK_OK(bus.status());
  return *std::move(bus);
}

bool waitFor(const std::function<bool()>& condition, absl::Duration timeout) {
  const absl::Time deadline = absl::Now() + timeout;
  while (!condition()) {
    if (absl::Now() > deadline) return false;
    absl::SleepFor(absl::Milliseconds(5));
  }
  return true;
}

}  // namespace ocs2::humanoid::test_support
