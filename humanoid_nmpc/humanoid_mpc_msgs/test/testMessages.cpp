// The C++ code generated from the humanoid_mpc_msgs protos compiles, round-trips through the protobuf runtime it links,
// and travels over ZeroMQ in the bus's three-frame layout (topic, full type name, serialized message; see
// humanoid_nmpc/docs/distributed_runtime/README.md).

#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <zmq.hpp>

#include "humanoid_mpc_msgs/controller_type.pb.h"
#include "humanoid_mpc_msgs/mpc_observation.pb.h"
#include "humanoid_mpc_msgs/mpc_policy.pb.h"
#include "humanoid_mpc_msgs/target_contact_patch.pb.h"
#include "humanoid_mpc_msgs/vector.pb.h"

namespace humanoid_mpc_msgs {
namespace {

constexpr size_t kTimeNodes = 4;
constexpr size_t kStateDim = 3;

// A policy that sets a field of every kind MpcPolicy has: scalars, repeated scalars, nested and repeated messages, an
// enum and a nested enum.
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
    Vector* state = policy.add_state_trajectory();
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
  TargetContactPatch* patch = policy.mutable_annotations()->add_target_contact_patches();
  patch->set_valid(true);
  patch->set_kind(TargetContactPatch::KIND_NEXT_SWING);
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
  EXPECT_EQ(parsed.annotations().target_contact_patches(0).kind(), TargetContactPatch::KIND_NEXT_SWING);
  ASSERT_EQ(static_cast<size_t>(parsed.state_trajectory_size()), kTimeNodes);
  EXPECT_EQ(static_cast<size_t>(parsed.state_trajectory(0).data_size()), kStateDim);
}

TEST(MpcPolicyTest, DefaultMessageIsEmptyOnTheWire) {
  // proto3 omits default values, so an observation that carries nothing costs nothing.
  EXPECT_TRUE(MpcPolicy().SerializeAsString().empty());
  EXPECT_TRUE(MpcObservation().SerializeAsString().empty());
}

TEST(MpcPolicyTest, TypeNameIsTheFullProtobufName) {
  // The second frame of every bus message; tools/ipc decodes topics by it.
  EXPECT_EQ(MpcPolicy::descriptor()->full_name(), "humanoid_mpc_msgs.MpcPolicy");
  EXPECT_EQ(MpcPolicy().GetTypeName(), "humanoid_mpc_msgs.MpcPolicy");
}

TEST(ZeroMqTest, CarriesAPolicyInThreeFrames) {
  constexpr int kLingerMs = 0;
  constexpr int kTimeoutMs = 5000;
  const std::string endpoint = "inproc://humanoid_mpc_msgs_test";
  const std::string topic = "mpc/policy";

  zmq::context_t context;
  zmq::socket_t receiver(context, zmq::socket_type::pair);
  receiver.set(zmq::sockopt::linger, kLingerMs);
  receiver.set(zmq::sockopt::rcvtimeo, kTimeoutMs);
  receiver.bind(endpoint);
  zmq::socket_t sender(context, zmq::socket_type::pair);
  sender.set(zmq::sockopt::linger, kLingerMs);
  sender.connect(endpoint);

  const MpcPolicy policy = makePolicy();
  const std::string typeName(MpcPolicy::descriptor()->full_name());
  const std::string payload = policy.SerializeAsString();
  ASSERT_TRUE(sender.send(zmq::buffer(topic), zmq::send_flags::sndmore).has_value());
  ASSERT_TRUE(sender.send(zmq::buffer(typeName), zmq::send_flags::sndmore).has_value());
  ASSERT_TRUE(sender.send(zmq::buffer(payload), zmq::send_flags::none).has_value());

  std::vector<std::string> frames;
  bool more = true;
  while (more) {
    zmq::message_t frame;
    const zmq::recv_result_t received = receiver.recv(frame, zmq::recv_flags::none);
    ASSERT_TRUE(received.has_value()) << "timed out after " << frames.size() << " frames";
    frames.push_back(frame.to_string());
    more = frame.more();
  }

  ASSERT_EQ(frames.size(), 3u);
  EXPECT_EQ(frames[0], topic);
  EXPECT_EQ(frames[1], typeName);
  MpcPolicy parsed;
  ASSERT_TRUE(parsed.ParseFromString(frames[2]));
  EXPECT_EQ(parsed.SerializeAsString(), payload);
}

}  // namespace
}  // namespace humanoid_mpc_msgs
