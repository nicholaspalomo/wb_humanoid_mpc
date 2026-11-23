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

#include <optional>
#include <string>

#include "absl/strings/string_view.h"

#include "robot_ipc/NetworkConfig.h"
#include "robot_ipc/QosProfile.h"

namespace robot::ipc {

// The key expressions of the bus. A topic's samples are put on
//   robot_ipc/<domain_id>/tag_<domain_tag>/<qos profile>/<topic>
// so that the processes of another domain, another tag or another profile of
// the topic never exchange samples, and every bus declares a liveliness token
//   robot_ipc_endpoints/<domain_id>/tag_<domain_tag>/<role>/<qos profile>/<bus
//   id>/<topic>
// for each of its publishers and subscribers, which the other buses count
// (TopicStatistics::matchedReaders, matchedWriters, incompatibleQos). The
// Python bus builds the same keys.
// LINT.IfChange(key_exprs)
inline constexpr absl::string_view kDataKeyExprRoot = "robot_ipc";
inline constexpr absl::string_view kEndpointKeyExprRoot = "robot_ipc_endpoints";
inline constexpr absl::string_view kDomainTagChunkPrefix = "tag_";
inline constexpr absl::string_view kPublisherRoleName = "publisher";
inline constexpr absl::string_view kSubscriberRoleName = "subscriber";
// LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/key_exprs.py:key_exprs)

/** Whether a liveliness token stands for a publisher or a subscriber. */
enum class EndpointRole {
  kPublisher,
  kSubscriber,
};

/** The key expression `topic`'s samples are put on in `network`. */
std::string dataKeyExpr(const NetworkConfig& network, absl::string_view topic);

/** The key expression of every topic of `network`, of every profile. */
std::string allTopicsKeyExpr(const NetworkConfig& network);

/** The profile and topic of a data key expression (dataKeyExpr()). */
struct DataKey {
  QosProfile profile = QosProfile::kReliableEvents;
  std::string topic;

  bool operator==(const DataKey& other) const = default;
};

/** The profile and topic of a data key expression of `network`; nullopt for a
 * key of another domain or tag, or of no profile. */
std::optional<DataKey> parseDataKeyExpr(const NetworkConfig& network,
                                        absl::string_view keyExpr);

/** One publisher or subscriber of a bus, as its liveliness token names it. */
struct EndpointToken {
  EndpointRole role = EndpointRole::kPublisher;
  QosProfile profile = QosProfile::kReliableEvents;
  /** The bus that declared it: a random id in hexadecimal. */
  std::string busId;
  std::string topic;

  bool operator==(const EndpointToken& other) const = default;
};

/** The key expression of the liveliness token `token` in `network`. */
std::string endpointTokenKeyExpr(const NetworkConfig& network,
                                 const EndpointToken& token);

/** The key expression of every liveliness token of `network`. */
std::string allEndpointTokensKeyExpr(const NetworkConfig& network);

/** The token of a key expression from endpointTokenKeyExpr(); nullopt for a key
 * of another domain or tag, or one that names no role or profile. */
std::optional<EndpointToken> parseEndpointTokenKeyExpr(
    const NetworkConfig& network, absl::string_view keyExpr);

}  // namespace robot::ipc
