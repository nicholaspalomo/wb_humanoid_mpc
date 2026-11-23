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

#include "robot_ipc/KeyExprs.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/QosProfile.h"

namespace robot::ipc {
namespace {

// The chunks between the domain prefix and the topic: a data key's profile, and
// a token key's role, profile and bus id.
constexpr size_t kDataKeyChunks = 1;
constexpr size_t kTokenKeyChunks = 3;
// The bus id's index among a token key's chunks.
constexpr size_t kBusIdChunk = 2;

std::string domainPrefix(absl::string_view root, const NetworkConfig& network) {
  return absl::StrCat(root, "/", network.domainId, "/", kDomainTagChunkPrefix,
                      network.domainTag);
}

absl::string_view roleName(EndpointRole role) {
  switch (role) {
    case EndpointRole::kPublisher:
      return kPublisherRoleName;
    case EndpointRole::kSubscriber:
      return kSubscriberRoleName;
  }
  return kSubscriberRoleName;
}

// The rest of `keyExpr` after `prefix` and a '/', split into its first
// `chunks` chunks and the remainder (the topic); nullopt when it has fewer.
std::optional<std::vector<absl::string_view>> splitAfter(
    absl::string_view keyExpr, absl::string_view prefix, size_t chunks) {
  if (!absl::ConsumePrefix(&keyExpr, prefix) ||
      !absl::ConsumePrefix(&keyExpr, "/")) {
    return std::nullopt;
  }
  std::vector<absl::string_view> parts =
      absl::StrSplit(keyExpr, absl::MaxSplits('/', chunks));
  if (parts.size() != chunks + 1 || parts.back().empty()) {
    return std::nullopt;
  }
  return parts;
}

}  // namespace

std::string dataKeyExpr(const NetworkConfig& network, absl::string_view topic) {
  return absl::StrCat(domainPrefix(kDataKeyExprRoot, network), "/",
                      qosProfileName(network.qosProfileOf(topic)), "/", topic);
}

std::string allTopicsKeyExpr(const NetworkConfig& network) {
  return absl::StrCat(domainPrefix(kDataKeyExprRoot, network), "/*/**");
}

std::optional<DataKey> parseDataKeyExpr(const NetworkConfig& network,
                                        absl::string_view keyExpr) {
  const std::optional<std::vector<absl::string_view>> parts = splitAfter(
      keyExpr, domainPrefix(kDataKeyExprRoot, network), kDataKeyChunks);
  if (!parts.has_value()) {
    return std::nullopt;
  }
  const absl::StatusOr<QosProfile> profile = parseQosProfile((*parts)[0]);
  if (!profile.ok()) {
    return std::nullopt;
  }
  return DataKey{.profile = *profile, .topic = std::string((*parts)[1])};
}

std::string endpointTokenKeyExpr(const NetworkConfig& network,
                                 const EndpointToken& token) {
  return absl::StrCat(domainPrefix(kEndpointKeyExprRoot, network), "/",
                      roleName(token.role), "/", qosProfileName(token.profile),
                      "/", token.busId, "/", token.topic);
}

std::string allEndpointTokensKeyExpr(const NetworkConfig& network) {
  return absl::StrCat(domainPrefix(kEndpointKeyExprRoot, network), "/**");
}

std::optional<EndpointToken> parseEndpointTokenKeyExpr(
    const NetworkConfig& network, absl::string_view keyExpr) {
  const std::optional<std::vector<absl::string_view>> parts = splitAfter(
      keyExpr, domainPrefix(kEndpointKeyExprRoot, network), kTokenKeyChunks);
  if (!parts.has_value()) {
    return std::nullopt;
  }
  EndpointToken token;
  if ((*parts)[0] == kPublisherRoleName) {
    token.role = EndpointRole::kPublisher;
  } else if ((*parts)[0] == kSubscriberRoleName) {
    token.role = EndpointRole::kSubscriber;
  } else {
    return std::nullopt;
  }
  const absl::StatusOr<QosProfile> profile = parseQosProfile((*parts)[1]);
  if (!profile.ok() || (*parts)[kBusIdChunk].empty()) {
    return std::nullopt;
  }
  token.profile = *profile;
  token.busId = std::string((*parts)[kBusIdChunk]);
  token.topic = std::string((*parts)[kTokenKeyChunks]);
  return token;
}

}  // namespace robot::ipc
