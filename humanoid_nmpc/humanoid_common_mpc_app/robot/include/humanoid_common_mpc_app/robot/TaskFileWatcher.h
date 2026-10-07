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

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * Calls a function when a task file has been written since the last poll (its modification time changed), for the
 * communication thread to poll at the rate the file should be checked: the robot process re-reads the controller-side
 * settings of its task file (contact_estimator, contact_wrench_gate) at about 1 Hz, as the in-process MPC's parameter
 * updater does. The baseline is the file's time when its owner read it (or at construction), so an unchanged file never
 * calls and a file written after it was read always does. Not thread-safe: one polling thread.
 */
class TaskFileWatcher {
 public:
  /**
   * Watches `file`, the robot's task file, which is handed to `onChange`, from `readAt`: the write time the file had
   * when its contents were last read (writeTimeOf(), taken before the read). A write after that time is reported by the
   * first poll, also one made before the watcher existed (a save while the process was starting). nullopt: the file's
   * time now.
   */
  TaskFileWatcher(absl::string_view file,
                  std::optional<std::filesystem::file_time_type> readAt,
                  std::function<void(const std::string& file)> onChange);

  /** Watches `file` from its time now: the constructor above with writeTimeOf(file). */
  TaskFileWatcher(absl::string_view file, std::function<void(const std::string& file)> onChange);

  /** The write time of `file`; nullopt when it has none to read (it does not exist). */
  static std::optional<std::filesystem::file_time_type> writeTimeOf(absl::string_view file);

  /** Calls onChange with file() when the file's modification time differs from the last one seen; true when it did. */
  bool poll();

  /** The file watched. */
  const std::string& file() const { return file_; }

 private:
  const std::string file_;
  const std::function<void(const std::string& file)> onChange_;
  std::filesystem::file_time_type lastWriteTime_{};
  bool haveWriteTime_ = false;
};

}  // namespace ocs2::humanoid
