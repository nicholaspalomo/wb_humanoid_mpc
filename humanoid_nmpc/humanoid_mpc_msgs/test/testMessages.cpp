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

// The C++ code generated from the humanoid_mpc_msgs protos compiles,
// round-trips through the protobuf runtime it links, and travels over the DDS
// bus in its envelope (wire protocol version, full type name, serialized
// message; see humanoid_nmpc/docs/distributed_runtime/README.md).

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_msgs/controller_type.pb.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/target_contact_patch.pb.h"
#include "humanoid_mpc_msgs/vector.pb.h"
#include "robot_ipc/Bus.h"
#include "robot_ipc/BusOptions.h"
#include "robot_ipc/Delivery.h"
#include "robot_runtime/robot_ipc/test/TestNetwork.h"

namespace humanoid_mpc_msgs {
namespace {

constexpr size_t kTimeNodes = 4;
constexpr size_t kStateDim = 3;

// A policy that sets a field of every kind MpcPolicy has: scalars, repeated
// scalars, nested and repeated messages, an enum and a nested enum.
MpcPolicy makePolicy() {
  MpcPolicy policy;
  policy.set_resets_served(3);
  policy.set_full_resets_served(1);
  policy.mutable_solver_status()->set_healthy(true);
  policy.mutable_solver_status()->set_solve_time_ms(4.5);
  policy.mutable_init_observation()->set_time(1.25);
  policy.mutable_init_observation()->add_state(0.5);
  policy.mutable_target_trajectories()->add_time(1.25);
  for (size_t node = 0; node < kTimeNodes; ++node) {
    policy.add_time_trajectory(1.25 + 0.01 * static_cast<double>(node));
    Vector* absl_nonnull state = policy.add_state_trajectory();
    for (size_t i = 0; i < kStateDim; ++i) {
      state->add_data(static_cast<double>(node * kStateDim + i));
    }
    policy.add_input_trajectory()->add_data(-static_cast<double>(node));
    policy.add_controller_data()->add_data(static_cast<double>(node) + 0.5);
  }
  policy.add_post_event_indices(2);
  policy.mutable_mode_schedule()->add_event_times(1.27);
  policy.mutable_mode_schedule()->add_mode_sequence(3);
  policy.mutable_mode_schedule()->add_mode_sequence(1);
  policy.set_controller_type(CONTROLLER_TYPE_LINEAR);
  policy.mutable_performance()->set_merit(10.0);
  TargetContactPatch* absl_nonnull patch =
      policy.mutable_annotations()->add_target_contact_patches();
  patch->set_kind(TargetContactPatch::KIND_NEXT_SWING);
  patch->mutable_pose()->set_x(0.25);
  return policy;
}

TEST(MpcPolicyTest, SerializeThenParseReproducesTheMessage) {
  const MpcPolicy policy = makePolicy();
  const std::string bytes = policy.SerializeAsString();
  ASSERT_FALSE(bytes.empty());

  MpcPolicy parsed;
  ASSERT_TRUE(parsed.ParseFromString(bytes));
  // MpcPolicy has no map fields, so equal messages serialize to equal bytes.
  EXPECT_EQ(parsed.SerializeAsString(), bytes);
  EXPECT_EQ(parsed.controller_type(), CONTROLLER_TYPE_LINEAR);
  EXPECT_EQ(parsed.annotations().target_contact_patches(0).kind(),
            TargetContactPatch::KIND_NEXT_SWING);
  ASSERT_EQ(static_cast<size_t>(parsed.state_trajectory_size()), kTimeNodes);
  EXPECT_EQ(static_cast<size_t>(parsed.state_trajectory(0).data_size()),
            kStateDim);
}

TEST(MpcPolicyTest, DefaultMessageIsEmptyOnTheWire) {
  // proto3 omits default values, so an observation that carries nothing costs
  // nothing.
  EXPECT_TRUE(MpcPolicy().SerializeAsString().empty());
  EXPECT_TRUE(MpcObservation().SerializeAsString().empty());
}

TEST(MpcPolicyTest, TypeNameIsTheFullProtobufName) {
  // The second frame of every bus message; tools/ipc decodes topics by it.
  EXPECT_EQ(MpcPolicy::descriptor()->full_name(),
            "humanoid_mpc_msgs.MpcPolicy");
  EXPECT_EQ(MpcPolicy().GetTypeName(), "humanoid_mpc_msgs.MpcPolicy");
}

TEST(BusTest, CarriesAPolicyWithItsTypeName) {
  constexpr absl::Duration kTimeout = absl::Seconds(20);
  const std::string topic = "mpc/policy";
  robot::ipc::BusOptions options;
  options.nodeName = "mpc";
  options.network = robot::ipc::isolatedTestNetwork({"mpc"});
  absl::StatusOr<std::unique_ptr<robot::ipc::Bus>> bus =
      robot::ipc::Bus::Create(std::move(options));
  ASSERT_TRUE(bus.ok()) << bus.status();
  absl::Mutex mutex;
  std::vector<std::pair<std::string, std::string>> received;
  ASSERT_TRUE(
      (*bus)
          ->subscribeRaw(topic, robot::ipc::Delivery::kAll,
                         [&mutex, &received](absl::string_view typeName,
                                             absl::string_view payload) {
                           absl::MutexLock lock(mutex);
                           received.emplace_back(std::string(typeName),
                                                 std::string(payload));
                         })
          .ok());
  ASSERT_TRUE((*bus)->start().ok());

  const MpcPolicy policy = makePolicy();
  const std::string payload = policy.SerializeAsString();
  const absl::Time deadline = absl::Now() + kTimeout;
  bool arrived = false;
  while (!arrived && absl::Now() < deadline) {
    ASSERT_TRUE((*bus)->publish(topic, policy).ok());
    absl::SleepFor(absl::Milliseconds(10));
    absl::MutexLock lock(mutex);
    arrived = !received.empty();
  }
  ASSERT_TRUE(arrived) << "timed out";
  absl::MutexLock lock(mutex);
  EXPECT_EQ(received[0].first, MpcPolicy::descriptor()->full_name());
  MpcPolicy parsed;
  ASSERT_TRUE(parsed.ParseFromString(received[0].second));
  EXPECT_EQ(parsed.SerializeAsString(), payload);
}

}  // namespace
}  // namespace humanoid_mpc_msgs
