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

#include <array>
#include <cstddef>
#include <optional>

#include "absl/strings/string_view.h"

#include "humanoid_mpc_msgs/config_file_kind.nproto.h"

namespace ocs2::humanoid {

// The kinds of configuration file a robot process stores (RobotConfigDirectory, ConfigFileStore): every kind of
// msgs::ConfigFileKind but kUnspecified, in the order of the stores' arrays. A kind not listed here is refused by both.
// LINT.IfChange(stored_config_file_kinds)
inline constexpr std::array<msgs::ConfigFileKind, 3> kStoredConfigFileKinds = {
    msgs::ConfigFileKind::kTask,
    msgs::ConfigFileKind::kReference,
    msgs::ConfigFileKind::kJointPdGains,
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_msgs/config_file_kind.proto:config_file_kinds)

inline constexpr size_t kNumStoredConfigFileKinds = kStoredConfigFileKinds.size();

/** The suffix of the copy of a stored file's previous contents both stores write before they replace the file. */
// LINT.IfChange(stored_config_backup_suffix)
inline constexpr absl::string_view kStoredConfigFileBackupSuffix = ".bak";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:config_store_layout, //tools/deploy/README.md:config_store_layout)
// clang-format on

/** Returns the index of `kind` in kStoredConfigFileKinds; nullopt for a kind the robot does not store (kUnspecified). */
std::optional<size_t> storedConfigFileKindIndex(msgs::ConfigFileKind kind);

/** Returns what the messages and the log call the file of `kind`: "task file", ...; "file of no kind" for kUnspecified. */
absl::string_view configFileKindName(msgs::ConfigFileKind kind);

}  // namespace ocs2::humanoid
