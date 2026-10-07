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

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"

#include "humanoid_mpc_config/mpc_parameter_update.pb.h"

/**
 * What a receiver checks of an operator payload before it converts it (humanoid_nmpc/humanoid_mpc_config/README.md,
 * "Version skew"): that the GUI that sent it was built from this build's schema, and, for the MPC parameters, for this
 * robot and this configuration of it. A conversion would take a payload of another schema version without a word - a
 * field the receiver does not know is dropped, a field the sender does not know takes its default - and a payload of
 * another robot or configuration in part. For the IO thread: the checks allocate.
 */
namespace ocs2::humanoid {

/**
 * OK when `message` carries no field this build's schema lacks; FailedPrecondition listing each one by its path and
 * number (nproto::UnknownFieldPaths()) otherwise: the sender was built from another version of the schema.
 */
absl::Status checkPayloadSchema(const google::protobuf::Message& message);

/** The schema fingerprint of MpcParameterUpdate in this build (nproto::SchemaFingerprint()), which a sender stamps. */
std::string mpcParameterUpdateSchemaFingerprint();

/**
 * checkPayloadSchema() of `message`, and that its schema_fingerprint is this build's (a sender without a field this
 * build has passes the first check but not this one), that its task file is the running robot's:
 * task.model_settings.robot_name equals `robotName` (ModelSettings::robotName; empty: not checked), and that it is the
 * running configuration's: config_path equals `taskFileIdentity` (configFileIdentity() of the receiver's task file;
 * empty: not checked). Two configurations of one robot share its robot_name (unitree_g1's centroidal and whole-body
 * MPCs), so only the path tells a centroidal G1 GUI's update from the whole-body one's; a receiver with an identity
 * refuses an update without a config_path, which a sender of this build always fills.
 *
 * @return OK, or FailedPrecondition naming the unknown fields, the two fingerprints, the two robots or the two paths.
 */
absl::Status checkMpcParameterUpdate(const humanoid_mpc_config::MpcParameterUpdate& message,
                                     absl::string_view robotName,
                                     absl::string_view taskFileIdentity);

}  // namespace ocs2::humanoid
