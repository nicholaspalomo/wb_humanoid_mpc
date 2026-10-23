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

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/robot/TaskFileWatcher.h"

/*
 * The robot process's watcher of its task file (the task.textproto the tuning GUI saves): a write is reported once, an
 * unchanged or vanished file never.
 */

namespace ocs2::humanoid {
namespace {

/** Moves the file's modification time on by a second, so that a rewrite inside the file system's clock tick counts. */
void touchForward(const std::string& file) {
  std::filesystem::last_write_time(file, std::filesystem::last_write_time(file) + std::chrono::seconds(1));
}

TEST(TaskFileWatcher, AWriteIsReportedOnceAndAnUnchangedFileNever) {
  const std::string file = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "watched_task.textproto").string();
  std::ofstream(file, std::ios::trunc) << "contact_estimator: \"robot_state\"\n";
  std::vector<std::string> changes;
  TaskFileWatcher watcher(file, [&changes](const std::string& changed) { changes.push_back(changed); });
  EXPECT_EQ(watcher.file(), file);
  EXPECT_FALSE(watcher.poll()) << "the file's time at construction is the baseline";
  touchForward(file);
  EXPECT_TRUE(watcher.poll());
  EXPECT_FALSE(watcher.poll());
  EXPECT_EQ(changes, std::vector<std::string>{file});
}

TEST(TaskFileWatcher, AWriteBetweenTheReadAndTheWatchersStartIsReportedByTheFirstPoll) {
  const std::string file = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "saved_while_starting.textproto").string();
  std::ofstream(file, std::ios::trunc) << "contact_estimator: \"robot_state\"\n";
  // The process reads the file's settings, taking its time first ...
  const std::optional<std::filesystem::file_time_type> readAt = TaskFileWatcher::writeTimeOf(file);
  ASSERT_TRUE(readAt.has_value());
  // ... the GUI saves while the process is still starting ...
  std::ofstream(file, std::ios::trunc) << "contact_estimator: \"always_in_contact\"\n";
  touchForward(file);
  // ... and the watcher, made only now, starts from the read.
  int changes = 0;
  TaskFileWatcher watcher(file, readAt, [&changes](const std::string& /*changed*/) { ++changes; });
  EXPECT_TRUE(watcher.poll()) << "the save the start-up read missed";
  EXPECT_FALSE(watcher.poll());
  EXPECT_EQ(changes, 1);
  EXPECT_FALSE(TaskFileWatcher::writeTimeOf(absl::StrCat(file, ".missing")).has_value());
}

TEST(TaskFileWatcher, AFileThatIsGoneForAMomentIsNotAChange) {
  const std::string file = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / "vanishing_task.textproto").string();
  std::ofstream(file, std::ios::trunc) << "\n";
  int changes = 0;
  TaskFileWatcher watcher(file, [&changes](const std::string& /*changed*/) { ++changes; });
  std::filesystem::remove(file);
  EXPECT_FALSE(watcher.poll()) << "an editor that saves by renaming removes the file for a moment";
  std::ofstream(file, std::ios::trunc) << "\n";
  touchForward(file);
  EXPECT_TRUE(watcher.poll());
  EXPECT_EQ(changes, 1);
  EXPECT_TRUE(absl::EndsWith(watcher.file(), "vanishing_task.textproto"));
}

}  // namespace
}  // namespace ocs2::humanoid
