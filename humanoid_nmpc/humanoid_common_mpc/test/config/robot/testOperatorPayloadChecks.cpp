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

// The checks of an operator payload before it is converted (OperatorPayloadChecks.h): a field of another schema version
// is refused by its path, an MpcParameterUpdate of another schema fingerprint, of another robot or of another
// configuration is refused, and one of this build, this robot and this configuration passes.

#include <string>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/mpc_parameter_update.pb.h"

namespace ocs2::humanoid {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;

// A varint field 103 of value 1: no message of the schemas has a field 103.
constexpr char kUnknownField103[] = "\xb8\x06\x01";

/** An update of the robot `robotName`, stamped as the GUI stamps it. */
humanoid_mpc_config::MpcParameterUpdate updateOf(const std::string& robotName) {
  humanoid_mpc_config::MpcParameterUpdate update;
  update.mutable_task()->mutable_model_settings()->set_robot_name(robotName);
  update.mutable_task()->mutable_contact_wrench_gate()->set_ramp_time(0.04);
  update.set_schema_fingerprint(mpcParameterUpdateSchemaFingerprint());
  return update;
}

TEST(OperatorPayloadChecks, AnUpdateOfThisBuildAndThisRobotPasses) {
  EXPECT_TRUE(checkMpcParameterUpdate(updateOf("atlas"), "atlas", /*taskFileIdentity=*/"").ok());
  // Without a robot to compare with, the robot is not checked.
  EXPECT_TRUE(checkMpcParameterUpdate(updateOf("r1"), /*robotName=*/"", /*taskFileIdentity=*/"").ok());
  EXPECT_EQ(mpcParameterUpdateSchemaFingerprint().size(), 16U);
}

TEST(OperatorPayloadChecks, AFieldOfAnotherSchemaVersionIsRefusedByItsPath) {
  humanoid_mpc_config::MpcParameterUpdate sent = updateOf("atlas");
  // A gate with a field this build does not have, as a GUI built from a newer schema would send it.
  std::string gate = sent.task().contact_wrench_gate().SerializeAsString() + std::string(kUnknownField103, sizeof(kUnknownField103) - 1);
  humanoid_mpc_config::MpcParameterUpdate received = sent;
  ASSERT_TRUE(received.mutable_task()->mutable_contact_wrench_gate()->ParseFromString(gate));
  const absl::Status refused = checkMpcParameterUpdate(received, "atlas", /*taskFileIdentity=*/"");
  EXPECT_EQ(refused.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(refused.message(), AllOf(HasSubstr("task.contact_wrench_gate: field 103"), HasSubstr("another schema version")));

  humanoid_mpc_config::JointPdGainsFile gains;
  gains.mutable_default_gains()->set_kp(100.0);
  humanoid_mpc_config::JointPdGainsFile newer;
  ASSERT_TRUE(newer.ParseFromString(gains.SerializeAsString() + std::string(kUnknownField103, sizeof(kUnknownField103) - 1)));
  EXPECT_THAT(checkPayloadSchema(newer).message(), HasSubstr(": field 103"));
  EXPECT_TRUE(checkPayloadSchema(gains).ok());
}

TEST(OperatorPayloadChecks, AnUpdateOfAnotherSchemaFingerprintIsRefused) {
  // A sender whose schema lacks fields this build has carries none of them, and no unknown field either: only the
  // fingerprint tells. One without a fingerprint was built before the payload carried one.
  for (const std::string& fingerprint : {std::string(), std::string("0123456789abcdef")}) {
    humanoid_mpc_config::MpcParameterUpdate update = updateOf("atlas");
    update.set_schema_fingerprint(fingerprint);
    const absl::Status refused = checkMpcParameterUpdate(update, "atlas", /*taskFileIdentity=*/"");
    EXPECT_EQ(refused.code(), absl::StatusCode::kFailedPrecondition) << fingerprint;
    EXPECT_THAT(refused.message(), HasSubstr(mpcParameterUpdateSchemaFingerprint())) << fingerprint;
  }
}

TEST(OperatorPayloadChecks, AnUpdateOfAnotherRobotIsRefused) {
  const absl::Status refused = checkMpcParameterUpdate(updateOf("drc_atlas"), "unitree_r1", /*taskFileIdentity=*/"");
  EXPECT_EQ(refused.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(refused.message(), AllOf(HasSubstr("drc_atlas"), HasSubstr("unitree_r1")));
}

// The configuration on the live path: two configurations of one robot share its robot_name.
constexpr char kWholeBodyG1[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kCentroidalG1[] = "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";

TEST(OperatorPayloadChecks, AnUpdateOfTheRunningConfigurationPasses) {
  humanoid_mpc_config::MpcParameterUpdate update = updateOf("g1");
  update.set_config_path(kWholeBodyG1);
  EXPECT_TRUE(checkMpcParameterUpdate(update, "g1", kWholeBodyG1).ok());
}

TEST(OperatorPayloadChecks, AnUpdateOfAnotherConfigurationOfTheSameRobotIsRefusedWithBothPaths) {
  humanoid_mpc_config::MpcParameterUpdate update = updateOf("g1");
  update.set_config_path(kCentroidalG1);
  const absl::Status refused = checkMpcParameterUpdate(update, "g1", kWholeBodyG1);
  EXPECT_EQ(refused.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(refused.message(), AllOf(HasSubstr(kCentroidalG1), HasSubstr(kWholeBodyG1)));
}

TEST(OperatorPayloadChecks, WithoutANodeIdentityTheConfigurationIsNotChecked) {
  humanoid_mpc_config::MpcParameterUpdate update = updateOf("g1");
  update.set_config_path(kCentroidalG1);
  EXPECT_TRUE(checkMpcParameterUpdate(update, "g1", /*taskFileIdentity=*/"").ok());
  // Nor is a message without a path: the receiver has nothing to compare it with.
  EXPECT_TRUE(checkMpcParameterUpdate(updateOf("g1"), "g1", /*taskFileIdentity=*/"").ok());
}

TEST(OperatorPayloadChecks, AnUpdateWithoutAConfigPathIsRefusedByAReceiverWithAnIdentity) {
  // A sender of this build always fills config_path; one that does not cannot be told from another configuration.
  const absl::Status refused = checkMpcParameterUpdate(updateOf("g1"), "g1", kWholeBodyG1);
  EXPECT_EQ(refused.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(refused.message(), AllOf(HasSubstr("<no config_path>"), HasSubstr(kWholeBodyG1)));
}

}  // namespace
}  // namespace ocs2::humanoid
