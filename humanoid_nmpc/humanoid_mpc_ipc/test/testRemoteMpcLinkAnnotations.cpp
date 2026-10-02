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

// The viewer annotations of the policies the robot accepts (MpcPolicy.annotations: the contact planner's target contact
// patches and the scaled walking command), handed to one consumer of the robot process by RemoteMpcLink::takeAnnotations():
// those of the newest accepted policy, once each, and none of a policy the link drops.

#include <gtest/gtest.h>

#include <atomic>
#include <memory>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include <ocs2_mpc/CommandData.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

#include "humanoid_common_mpc/mrt/MpcResetSupervisor.h"
#include "humanoid_mpc_ipc/MpcServer.h"
#include "humanoid_mpc_ipc/RemoteMpcLink.h"
#include "humanoid_mpc_msgs/viewer_annotations.nproto.h"
#include "humanoid_mpc_msgs/viewer_annotations.pb.h"
#include "humanoid_nmpc/humanoid_mpc_ipc/test/MpcLinkTestSupport.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid::ipc {
namespace {

using test_support::modelDimensions;
using test_support::observationAt;
using test_support::resetTargetsFor;
using test_support::waitFor;

TEST(RemoteMpcLinkAnnotations, TheNewestAcceptedPolicysAnnotationsAreTakenOnce) {
  std::unique_ptr<mpc_test::ScriptedMpc> mpc = test_support::makeScriptedMpc();
  std::unique_ptr<robot::ipc::Bus> robotBus = test_support::createNodeBus("robot");
  std::unique_ptr<robot::ipc::Bus> mpcBus = test_support::createNodeBus("mpc");
  test_support::connectBoth(*robotBus, *mpcBus);
  MpcResetSupervisor supervisor;
  RemoteMpcLink::Config linkConfig;
  linkConfig.dimensions = modelDimensions();
  absl::StatusOr<std::unique_ptr<RemoteMpcLink>> link = RemoteMpcLink::Create(*robotBus, supervisor, linkConfig);
  ASSERT_TRUE(link.ok()) << link.status();

  // Every policy carries two target patches and a velocity that counts the policies.
  std::atomic<int> solves{0};
  MpcServer::Hooks hooks;
  hooks.annotationsProvider = [&](const CommandData& /*command*/, const PrimalSolution& /*solution*/,
                                  humanoid_mpc_msgs::ViewerAnnotations* annotations) {
    const int solve = solves.fetch_add(1) + 1;
    annotations->clear_target_contact_patches();
    for (int contact = 0; contact < 2; ++contact) {
      humanoid_mpc_msgs::TargetContactPatch* patch = annotations->add_target_contact_patches();
      patch->set_valid(true);
      patch->set_kind(contact == 0 ? humanoid_mpc_msgs::TargetContactPatch::KIND_NEXT_SWING
                                   : humanoid_mpc_msgs::TargetContactPatch::KIND_STANCE);
      patch->set_x(0.1 * contact);
      patch->set_yaw(0.2);
    }
    annotations->set_scaled_velocity_x(static_cast<double>(solve));
    annotations->set_scaled_yaw_rate(-0.3);
  };
  MpcServer::Config serverConfig;
  serverConfig.dimensions = modelDimensions();
  absl::StatusOr<std::unique_ptr<MpcServer>> server = MpcServer::Create(*mpcBus, *mpc, resetTargetsFor, serverConfig, hooks);
  ASSERT_TRUE(server.ok()) << server.status();
  ASSERT_TRUE(robotBus->start().ok());
  ASSERT_TRUE(mpcBus->start().ok());
  ASSERT_TRUE((*server)->start().ok());

  msgs::ViewerAnnotations annotations;
  EXPECT_FALSE((*link)->takeAnnotations(annotations)) << "no policy yet";
  scalar_t time = 1.0;
  ASSERT_TRUE(waitFor([&]() {
    time += 0.001;
    (*link)->setCurrentObservation(observationAt(time, /*value=*/1.0));
    absl::SleepFor(absl::Milliseconds(2));
    return (*link)->statistics().policiesAccepted > 0;
  }));
  ASSERT_TRUE((*link)->takeAnnotations(annotations));
  ASSERT_EQ(annotations.target_contact_patches.size(), 2u);
  EXPECT_TRUE(annotations.target_contact_patches[0].valid);
  EXPECT_EQ(annotations.target_contact_patches[0].kind, msgs::TargetContactPatch::Kind::kNextSwing);
  EXPECT_EQ(annotations.target_contact_patches[1].kind, msgs::TargetContactPatch::Kind::kStance);
  EXPECT_DOUBLE_EQ(annotations.target_contact_patches[1].x, 0.1);
  EXPECT_DOUBLE_EQ(annotations.scaled_yaw_rate, -0.3);
  EXPECT_GE(annotations.scaled_velocity_x, 1.0);
  // Taken once: the same policy's annotations are not handed out again; the next accepted policy brings new ones.
  const double first = annotations.scaled_velocity_x;
  if (!(*link)->takeAnnotations(annotations)) {
    const uint64_t accepted = (*link)->statistics().policiesAccepted;
    ASSERT_TRUE(waitFor([&]() {
      time += 0.001;
      (*link)->setCurrentObservation(observationAt(time, /*value=*/1.0));
      absl::SleepFor(absl::Milliseconds(2));
      return (*link)->statistics().policiesAccepted > accepted;
    }));
    ASSERT_TRUE((*link)->takeAnnotations(annotations));
  }
  EXPECT_GT(annotations.scaled_velocity_x, first) << "the annotations of a newer policy";
  (*server)->stop();
  mpcBus->stop();
  robotBus->stop();
}

}  // namespace
}  // namespace ocs2::humanoid::ipc
