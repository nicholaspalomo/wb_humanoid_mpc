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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

#include "robot_ipc/NodeEndpoint.h"
#include "robot_ipc/QosProfile.h"

namespace robot::ipc {

/** One topic of the network file and its quality of service. */
struct TopicQos {
  /** The bus topic, which is also the DDS topic name, e.g. "mpc/policy". */
  std::string topic;
  QosProfile profile = QosProfile::kReliableEvents;
  /** The nodes that publish it; their buses create the topic's DDS writer in
   * Bus::Create(). */
  std::vector<std::string> publishers;

  bool operator==(const TopicQos& other) const = default;
};

/** The DDS transport of the network file. Every field is optional: 0 (or a zero
 * duration) keeps its default. */
struct TransportOptions {
  // The Python bus has the same defaults (robot_ipc/network_config.py).
  // LINT.IfChange(transport_defaults)
  /** The defaults of the fields that are not Cyclone DDS's own defaults. */
  static constexpr int64_t kDefaultWriterUnackedBytesLimit = int64_t{8} << 20;
  static constexpr int kDefaultMaxProcessesPerHost = 20;
  static constexpr absl::Duration kDefaultLeaseDuration = absl::Seconds(3);
  // LINT.ThenChange(//robot_runtime/robot_ipc/python/robot_ipc/network_config.py:transport_defaults)

  /** General/MaxMessageSize [bytes]; 0: Cyclone's 14720 B. */
  int64_t maxMessageSizeBytes = 0;
  /** Internal/SocketReceiveBufferSize max [bytes]; 0: Cyclone's 1 MiB request.
   */
  int64_t socketReceiveBufferBytes = 0;
  /** Internal/SocketSendBufferSize max [bytes]; 0: the OS default. */
  int64_t socketSendBufferBytes = 0;
  /** Internal/Watermarks/WhcHigh [bytes]; 0: kDefaultWriterUnackedBytesLimit.
   */
  int64_t writerUnackedBytesLimit = 0;
  /** Discovery/MaxAutoParticipantIndex + 1; 0: kDefaultMaxProcessesPerHost. */
  int maxProcessesPerHost = 0;
  /** Discovery/LeaseDuration; zero: kDefaultLeaseDuration. */
  absl::Duration leaseDuration = absl::ZeroDuration();
  /** Internal/Test/XmitLossiness: UDP packets dropped per 1000 sent, for tests
   * and rehearsals; 0 on a real network. */
  int simulatedSendLossPerMille = 0;

  bool operator==(const TransportOptions& other) const = default;
};

/**
 * The DDS network of the bus, as the network file names it: a textproto of
 * robot_ipc_proto.NetworkConfig
 * (robot_runtime/robot_ipc/proto/network_config.proto), such as
 * config/ipc/network.textproto:
 *
 *   domain_id: 0
 *   nodes { name: "robot" host: "192.168.1.10" }
 *   nodes { name: "mpc" host: "192.168.1.20" }
 *   topics { name: "mpc/policy" qos_profile: "reliable_latest" publishers:
 * "mpc" }
 *
 * DomainConfig.h turns it into the Cyclone DDS configuration of a process.
 */
struct NetworkConfig {
  /** The largest domain id whose ports stay below 65536 (DomainConfig.h). */
  static constexpr int kMaxDomainId = 232;

  /** In the order of the file. */
  std::vector<NodeEndpoint> nodes;
  /** The DDS domain, in [0, kMaxDomainId]. */
  int domainId = 0;
  /** Discovery/Tag: only processes with the same tag discover each other.
   * Empty: none. */
  std::string domainTag;
  /** The topics the file lists, in its order, each at most once. */
  std::vector<TopicQos> topics;
  /** The profile of a topic that `topics` does not list. */
  QosProfile defaultQosProfile = QosProfile::kReliableEvents;
  TransportOptions transport;

  /** The node of that name, or nullptr. */
  const NodeEndpoint* absl_nullable find(absl::string_view name) const;
  /** The node names, in order, e.g. for an error message that lists the valid
   * ones. */
  std::vector<std::string> nodeNames() const;
  /** The entry of `topic` in `topics`, or nullptr. */
  const TopicQos* absl_nullable findTopic(absl::string_view topic) const;
  /** The profile of `topic`: its entry's, or defaultQosProfile. */
  QosProfile qosProfileOf(absl::string_view topic) const;
  /** The topics that `nodeName` publishes, in the order of the file. */
  std::vector<std::string> topicsPublishedBy(absl::string_view nodeName) const;
  /** The distinct hosts of the nodes, in the order of the file: the unicast
   * discovery peers. */
  std::vector<std::string> peerHosts() const;

  bool operator==(const NetworkConfig& other) const = default;
};

/**
 * Reads and validates a network file. It is parsed strictly
 * (google::protobuf::TextFormat): a syntax error, an unknown field or a value
 * of the wrong type is an InvalidArgument error that names the file, the line
 * and the column, such as "config/ipc/network.textproto:3:42: Message type
 * "robot_ipc_proto.NetworkConfig.Node" has no field named "hots"." The content
 * is then validated, and an error names the node, the topic or the field, such
 * as "config/ipc/network.textproto: domain_id: 300 is not a domain id in [0,
 * 232]". A file that cannot be read is NotFound.
 */
absl::StatusOr<NetworkConfig> loadNetworkConfig(const std::string& path);

/** loadNetworkConfig() on the text of a network file; `source` names it in the
 * error messages. */
absl::StatusOr<NetworkConfig> parseNetworkConfig(absl::string_view text,
                                                 absl::string_view source);

/**
 * The checks of the network file that also apply to a network built in code
 * (Bus::Create() runs them): at least one node, valid and unique node names,
 * valid hosts, a domain id in [0, kMaxDomainId], a valid tag, valid and unique
 * topic names whose publishers are nodes of the network, and transport values
 * in range. Errors are InvalidArgument and name the node, the topic or the
 * field, as in loadNetworkConfig().
 */
absl::Status validateNetworkConfig(const NetworkConfig& config);

/**
 * The bus topic as a DDS topic name: letters, digits, '_' and '/', not starting
 * with a digit (Cyclone DDS refuses anything else). An empty or invalid topic
 * is InvalidArgument. The topic is used as it is, so a DDS tool (ddsls) shows
 * the bus's own names.
 */
absl::Status validateTopicName(absl::string_view topic);

/**
 * The network file of `config`, one line per node and per topic;
 * parseNetworkConfig() reads it back as `config` when it is valid.
 */
std::string formatNetworkConfig(const NetworkConfig& config);

/**
 * The shipped network file, config/ipc/network.textproto: robot, mpc, operator,
 * teleop, visualization and config_push on 127.0.0.1 in domain 0, with the QoS
 * of every topic of humanoid_mpc_ipc/Topics.h.
 */
NetworkConfig localhostNetworkConfig();

}  // namespace robot::ipc
