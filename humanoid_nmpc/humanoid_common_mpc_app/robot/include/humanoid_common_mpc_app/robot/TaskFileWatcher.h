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
#include <string>

namespace ocs2::humanoid {

/**
 * Calls a function when a file has been written since the last poll (its modification time changed), for the
 * communication thread to poll at the rate the file should be checked: the robot process re-reads the controller-side
 * keys of its task file (`contactEstimator`, `contact_wrench_gate`) at about 1 Hz, as the in-process MPC's parameter
 * updater does. The file's time at construction is the baseline, so an unchanged file never calls. Not thread-safe:
 * one polling thread.
 */
class TaskFileWatcher {
 public:
  TaskFileWatcher(std::string file, std::function<void(const std::string& file)> onChange);

  /** Calls onChange when the file's modification time differs from the last one seen; true when it did. */
  bool poll();

  const std::string& file() const { return file_; }

 private:
  const std::string file_;
  const std::function<void(const std::string& file)> onChange_;
  std::filesystem::file_time_type lastWriteTime_{};
  bool haveWriteTime_ = false;
};

}  // namespace ocs2::humanoid
