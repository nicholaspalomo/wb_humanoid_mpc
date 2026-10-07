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

#include "humanoid_common_mpc/config/OperatorPayloadChecks.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"

#include "humanoid_mpc_config/mpc_parameter_update.pb.h"
#include "nproto/Schema.h"

namespace ocs2::humanoid {

absl::Status checkPayloadSchema(const google::protobuf::Message& message) {
  const std::vector<std::string> unknown = nproto::UnknownFieldPaths(message);
  if (unknown.empty()) return absl::OkStatus();
  return absl::FailedPreconditionError(absl::StrCat("the ", message.GetDescriptor()->full_name(),
                                                    " has fields this build's schema does not (", absl::StrJoin(unknown, ", "),
                                                    "): it was sent by a build with another schema version"));
}

std::string mpcParameterUpdateSchemaFingerprint() {
  return nproto::SchemaFingerprint(*humanoid_mpc_config::MpcParameterUpdate::descriptor());
}

absl::Status checkMpcParameterUpdate(const humanoid_mpc_config::MpcParameterUpdate& message,
                                     absl::string_view robotName,
                                     absl::string_view taskFileIdentity) {
  if (absl::Status schema = checkPayloadSchema(message); !schema.ok()) {
    return schema;
  }
  if (const std::string own = mpcParameterUpdateSchemaFingerprint(); message.schema_fingerprint() != own) {
    return absl::FailedPreconditionError(absl::StrCat("the MpcParameterUpdate has the schema fingerprint '", message.schema_fingerprint(),
                                                      "', this build's is '", own,
                                                      "': it was sent by a build with another schema version"));
  }
  if (!robotName.empty() && message.task().model_settings().robot_name() != robotName) {
    return absl::FailedPreconditionError(absl::StrCat("the MpcParameterUpdate is the task file of the robot '",
                                                      message.task().model_settings().robot_name(), "', and this is '", robotName, "'"));
  }
  if (!taskFileIdentity.empty() && message.config_path() != taskFileIdentity) {
    return absl::FailedPreconditionError(
        absl::StrCat("the MpcParameterUpdate is the task file ", message.config_path().empty() ? "<no config_path>" : message.config_path(),
                     ", and this one runs ", taskFileIdentity, ": an update of another configuration is not applied"));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
