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

#include "humanoid_common_mpc_app/robot/TaskFileWatcher.h"

#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

TaskFileWatcher::TaskFileWatcher(absl::string_view file,
                                 std::optional<std::filesystem::file_time_type> readAt,
                                 std::function<void(const std::string& file)> onChange)
    : file_(file), onChange_(std::move(onChange)) {
  if (!readAt.has_value()) readAt = writeTimeOf(file_);
  if (readAt.has_value()) {
    lastWriteTime_ = *readAt;
    haveWriteTime_ = true;
  }
}

TaskFileWatcher::TaskFileWatcher(absl::string_view file, std::function<void(const std::string& file)> onChange)
    : TaskFileWatcher(file, /*readAt=*/std::nullopt, std::move(onChange)) {}

std::optional<std::filesystem::file_time_type> TaskFileWatcher::writeTimeOf(absl::string_view file) {
  std::error_code error;
  const std::filesystem::file_time_type writeTime = std::filesystem::last_write_time(std::string(file), error);
  if (error) return std::nullopt;
  return writeTime;
}

bool TaskFileWatcher::poll() {
  std::error_code error;
  const std::filesystem::file_time_type writeTime = std::filesystem::last_write_time(file_, error);
  if (error) {
    return false;  // gone for the moment (an editor that saves by renaming); the next poll sees the new file
  }
  if (haveWriteTime_ && writeTime == lastWriteTime_) {
    return false;
  }
  lastWriteTime_ = writeTime;
  haveWriteTime_ = true;
  if (onChange_) onChange_(file_);
  return true;
}

}  // namespace ocs2::humanoid
