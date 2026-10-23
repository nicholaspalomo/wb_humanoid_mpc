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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc_app/robot/AtomicFileWrite.h"

/*
 * The contract of the crash-safe writes (AtomicFileWrite.h): a write replaces the file and leaves no temporary, and an
 * error that is not DataLoss leaves the file as it was. DataLoss - the file replaced, its directory not synced - cannot
 * be provoked on a test's file system, whose directories always sync; ConfigFileStore and RobotConfigDirectory take it
 * as written.
 */
namespace ocs2::humanoid {
namespace {

std::string scratch(absl::string_view name) {
  const std::filesystem::path directory = std::filesystem::path(std::getenv("TEST_TMPDIR")) / std::string(name);
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  std::filesystem::create_directories(directory, error);
  return directory.string();
}

std::string contentsOf(const std::string& path) {
  std::ifstream file(path);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

/** Whether `directory` holds anything but `name`. */
bool holdsOnly(const std::string& directory, absl::string_view name) {
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().filename() != std::string(name)) return false;
  }
  return true;
}

TEST(AtomicFileWrite, AWriteReplacesTheFileAndLeavesNoTemporary) {
  const std::string directory = scratch("replaces");
  const std::string path = directory + "/task.textproto";
  ASSERT_TRUE(writeFileAtomically(path, "first\n").ok());
  ASSERT_TRUE(writeFileAtomically(path, "second\n").ok());
  EXPECT_EQ(contentsOf(path), "second\n");
  EXPECT_TRUE(holdsOnly(directory, "task.textproto"));
  const absl::StatusOr<std::string> read = readFileBytes(path);
  ASSERT_TRUE(read.ok()) << read.status();
  EXPECT_EQ(*read, "second\n");
}

TEST(AtomicFileWrite, AWriteThatFailsBeforeTheRenameLeavesTheFileAsItWasAndIsNoDataLoss) {
  const std::string directory = scratch("fails");
  // The target is a directory that is not empty: the temporary is written, and the rename over it fails.
  const std::string target = directory + "/occupied";
  std::filesystem::create_directories(target + "/inside");
  const absl::Status failed = writeFileAtomically(target, "text\n");
  EXPECT_FALSE(failed.ok());
  EXPECT_FALSE(absl::IsDataLoss(failed)) << failed;
  EXPECT_TRUE(std::filesystem::is_directory(target + "/inside")) << "the target was touched";
  EXPECT_TRUE(holdsOnly(directory, "occupied")) << "the temporary was left behind";

  // No directory to write into: nothing is created.
  const absl::Status missing = writeFileAtomically(directory + "/no/such/directory/file", "text\n");
  EXPECT_FALSE(missing.ok());
  EXPECT_FALSE(absl::IsDataLoss(missing)) << missing;
}

TEST(AtomicFileWrite, AMissingFileIsNotFoundAndRemovingItIsNoError) {
  const std::string directory = scratch("missing");
  EXPECT_TRUE(absl::IsNotFound(readFileBytes(directory + "/none").status()));
  EXPECT_TRUE(removeFile(directory + "/none").ok());
  ASSERT_TRUE(writeFileAtomically(directory + "/a", "a\n").ok());
  ASSERT_TRUE(renameFile(directory + "/a", directory + "/b").ok());
  EXPECT_EQ(contentsOf(directory + "/b"), "a\n");
  EXPECT_TRUE(removeFile(directory + "/b").ok());
  EXPECT_TRUE(std::filesystem::is_empty(directory));
}

}  // namespace
}  // namespace ocs2::humanoid
