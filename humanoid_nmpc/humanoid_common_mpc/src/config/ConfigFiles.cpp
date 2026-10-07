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

#include "humanoid_common_mpc/config/ConfigFiles.h"

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.pb.h"
#include "humanoid_mpc_config/contact_planning_file.pb.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/gait_file.nproto.pb.h"
#include "humanoid_mpc_config/gait_file.pb.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.pb.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.pb.h"
#include "humanoid_mpc_config/reference_file.pb.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

/** The robot packages' directory, which a configuration file's identity starts at. */
// LINT.IfChange(robot_models_directory)
constexpr absl::string_view kRobotModelsDirectory = "robot_models";
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/config_files.py:robot_models_dir)

/** The textproto `text` as the nproto struct `Struct` of its message `Message`, as nproto::LoadTextprotoFile() reads a file. */
template <typename Struct, typename Message>
absl::StatusOr<Struct> parseTextprotoAs(absl::string_view text, absl::string_view source) {
  absl::StatusOr<Message> message = nproto::ParseTextproto<Message>(text, source);
  if (!message.ok()) {
    return message.status();
  }
  Struct value;
  const absl::Status status = FromProto(*message, &value);
  if (!status.ok()) {
    return absl::Status(status.code(), absl::StrCat(source, ": ", status.message()));
  }
  return value;
}

}  // namespace

absl::StatusOr<mpc_config::TaskFile> loadTaskFile(absl::string_view path) {
  return nproto::LoadTextprotoFile<mpc_config::TaskFile, humanoid_mpc_config::TaskFile>(path);
}

absl::StatusOr<mpc_config::ReferenceFile> loadReferenceFile(absl::string_view path) {
  return nproto::LoadTextprotoFile<mpc_config::ReferenceFile, humanoid_mpc_config::ReferenceFile>(path);
}

absl::StatusOr<mpc_config::GaitFile> loadGaitFile(absl::string_view path) {
  return nproto::LoadTextprotoFile<mpc_config::GaitFile, humanoid_mpc_config::GaitFile>(path);
}

absl::StatusOr<mpc_config::ContactPlanningFile> loadContactPlanningFile(absl::string_view path) {
  return nproto::LoadTextprotoFile<mpc_config::ContactPlanningFile, humanoid_mpc_config::ContactPlanningFile>(path);
}

absl::StatusOr<mpc_config::JointPdGainsFile> loadJointPdGainsFile(absl::string_view path) {
  return nproto::LoadTextprotoFile<mpc_config::JointPdGainsFile, humanoid_mpc_config::JointPdGainsFile>(path);
}

absl::StatusOr<mpc_config::TaskFile> parseTaskFile(absl::string_view text, absl::string_view source) {
  return parseTextprotoAs<mpc_config::TaskFile, humanoid_mpc_config::TaskFile>(text, source);
}

absl::StatusOr<mpc_config::ReferenceFile> parseReferenceFile(absl::string_view text, absl::string_view source) {
  return parseTextprotoAs<mpc_config::ReferenceFile, humanoid_mpc_config::ReferenceFile>(text, source);
}

absl::StatusOr<mpc_config::JointPdGainsFile> parseJointPdGainsFile(absl::string_view text, absl::string_view source) {
  return parseTextprotoAs<mpc_config::JointPdGainsFile, humanoid_mpc_config::JointPdGainsFile>(text, source);
}

// The tuning GUI stamps the same identity on what it sends; a receiver compares the two.
// LINT.IfChange(config_file_identity)
std::string configFileIdentity(absl::string_view path) {
  const std::filesystem::path normal = std::filesystem::path(std::string(path)).lexically_normal();
  std::vector<std::filesystem::path> components(normal.begin(), normal.end());
  // The last robot_models/ component that has something below it.
  size_t start = components.size();
  for (size_t index = 0; index + 1 < components.size(); ++index) {
    if (components[index].native() == kRobotModelsDirectory) start = index;
  }
  if (start == components.size()) {
    return normal.generic_string();
  }
  std::filesystem::path identity;
  for (size_t index = start; index < components.size(); ++index) identity /= components[index];
  return identity.generic_string();
}
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/robot_config_save.py:config_path_of)

std::string contactPlanningFileBeside(absl::string_view taskFile) {
  const std::filesystem::path task = std::string(taskFile);
  return (task.parent_path() / std::string(kContactPlanningFileName)).string();
}

std::string jointPdGainsFileBeside(absl::string_view taskFile) {
  const std::filesystem::path configDirectory = std::filesystem::path(std::string(taskFile)).parent_path().parent_path();
  return (configDirectory / std::string(kJointPdGainsFileInConfigDirectory)).string();
}

absl::StatusOr<std::string> existingJointPdGainsFileBeside(absl::string_view taskFile) {
  std::string path = jointPdGainsFileBeside(taskFile);
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error)) {
    return absl::NotFoundError(absl::StrCat("the joint PD gains file ", path, " of the task file ", taskFile, " is not there (",
                                            error ? error.message() : "no regular file of that name",
                                            "): the controller would run on built-in gains that are no robot's"));
  }
  return path;
}

absl::StatusOr<std::optional<mpc_config::ContactPlanningFile>> loadContactPlanningFileBeside(absl::string_view taskFile) {
  const std::string path = contactPlanningFileBeside(taskFile);
  std::error_code error;
  const std::filesystem::file_status status = std::filesystem::status(path, error);
  // Only a file that is not there is the robot without one; one that cannot be looked at is not taken for absent.
  if (status.type() == std::filesystem::file_type::not_found) {
    return std::nullopt;
  }
  if (error) {
    return absl::UnavailableError(absl::StrCat(path, ": cannot be read: ", error.message()));
  }
  if (status.type() != std::filesystem::file_type::regular) {
    return absl::FailedPreconditionError(absl::StrCat(path, " is not a regular file"));
  }
  absl::StatusOr<mpc_config::ContactPlanningFile> file = loadContactPlanningFile(path);
  if (!file.ok()) {
    return file.status();
  }
  return std::optional<mpc_config::ContactPlanningFile>(*std::move(file));
}

absl::Status withConfigFile(const absl::Status& status, absl::string_view path) {
  if (status.ok()) {
    return status;
  }
  return absl::Status(status.code(), absl::StrCat(path, ": ", status.message()));
}

}  // namespace ocs2::humanoid
