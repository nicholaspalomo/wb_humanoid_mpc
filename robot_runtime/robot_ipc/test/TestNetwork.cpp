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

#include "robot_runtime/robot_ipc/test/TestNetwork.h"

#include <unistd.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/strings/str_cat.h"

#include "robot_ipc/NodeEndpoint.h"

namespace robot::ipc {
namespace {

// The test domains: away from domain 0 of the shipped network file, and within the ports of NetworkConfig.
constexpr int kFirstTestDomain = 100;
constexpr int kTestDomains = 100;

struct ProcessDomain {
  int domainId = 0;
  std::string tag;
};

const ProcessDomain& processDomain() {
  static const absl::NoDestructor<ProcessDomain> kDomain([] {
    std::random_device entropy;
    const uint64_t draw = (static_cast<uint64_t>(entropy()) << 32) | entropy();
    const int64_t pid = ::getpid();
    return ProcessDomain{.domainId = kFirstTestDomain + static_cast<int>(draw % kTestDomains),
                         .tag = absl::StrCat("robot_ipc_test_", pid, "_", absl::Hex(draw))};
  }());
  return *kDomain;
}

}  // namespace

NetworkConfig isolatedTestNetwork(const std::vector<std::string>& nodeNames, int domainIndex) {
  NetworkConfig network;
  for (const std::string& name : nodeNames) {
    network.nodes.push_back(NodeEndpoint{.name = name, .host = std::string(kLoopbackHost)});
  }
  network.domainId = kFirstTestDomain + (processDomain().domainId - kFirstTestDomain + domainIndex) % kTestDomains;
  network.domainTag = domainIndex == 0 ? processDomain().tag : absl::StrCat(processDomain().tag, "_", domainIndex);
  return network;
}

}  // namespace robot::ipc
