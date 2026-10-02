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

#include "robot_core/ResourcePaths.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"

namespace robot {
namespace {

// LINT.IfChange(probe_file)
/** The data file of this test, and its contents. */
constexpr char kProbeFile[] = "robot_runtime/robot_core/test/testdata/resource_paths_probe.txt";
// LINT.ThenChange(//robot_runtime/robot_core/BUILD.bazel:resource_paths_test_data)
constexpr char kProbeContents[] = "resource paths probe\n";

std::string readFile(const std::string& path) {
  std::ifstream stream(path);
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

/** Sets (or, with nullptr, unsets) an environment variable for the lifetime of the object, then restores it. */
class ScopedEnvironmentVariable {
 public:
  ScopedEnvironmentVariable(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) previous_ = std::string(previous);
    set(value);
  }
  ~ScopedEnvironmentVariable() { set(previous_.has_value() ? previous_->c_str() : nullptr); }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

 private:
  void set(const char* value) const {
    if (value == nullptr) {
      unsetenv(name_.c_str());
    } else {
      constexpr int kReplaceExistingValue = 1;
      setenv(name_.c_str(), value, kReplaceExistingValue);
    }
  }

  const std::string name_;
  std::optional<std::string> previous_;
};

/** A fresh directory under the test's temporary directory, removed again at the end of the test. */
class ScopedTemporaryDirectory {
 public:
  explicit ScopedTemporaryDirectory(const std::string& name) : path_(std::filesystem::path(::testing::TempDir()) / name) {
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  ~ScopedTemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  ScopedTemporaryDirectory(const ScopedTemporaryDirectory&) = delete;
  ScopedTemporaryDirectory& operator=(const ScopedTemporaryDirectory&) = delete;

  const std::filesystem::path& path() const { return path_; }

  /** Writes `contents` to `relativePath` under the directory, creating the directories in between. */
  void writeFile(const std::string& relativePath, const std::string& contents) const {
    const std::filesystem::path file = path_ / relativePath;
    std::filesystem::create_directories(file.parent_path());
    std::ofstream(file) << contents;
  }

 private:
  const std::filesystem::path path_;
};

TEST(ResourcePathsTest, ResolvesADataFileOfTheTestToItsContents) {
  const absl::StatusOr<std::string> path = resolveResourcePath(kProbeFile);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_TRUE(std::filesystem::path(*path).is_absolute()) << *path;
  EXPECT_EQ(readFile(*path), kProbeContents);
}

TEST(ResourcePathsTest, ResolvesADirectoryOfTheRunfiles) {
  const absl::StatusOr<std::string> path = resolveResourcePath("robot_runtime/robot_core/test/testdata");
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_TRUE(std::filesystem::is_directory(*path)) << *path;
}

TEST(ResourcePathsTest, AFileMissingFromTheDataIsNotFoundNamingThePathAndTheDataDependency) {
  // A real file of the repository, but not in this test's `data`.
  const std::string urdf = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
  const absl::StatusOr<std::string> path = resolveResourcePath(urdf);
  ASSERT_FALSE(path.ok()) << "resolved to " << *path;
  EXPECT_EQ(path.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(path.status().message(), urdf)) << path.status();
  EXPECT_TRUE(absl::StrContains(path.status().message(), "//robot_models/drc_atlas/drc_atlas_description that holds it to the `data`"))
      << path.status();
}

TEST(ResourcePathsTest, TheMissingDataDependencyIsThePackageInFrontOfTheDataDirectory) {
  const absl::StatusOr<std::string> config = resolveResourcePath("humanoid_nmpc/humanoid_common_mpc/config/command/not_shipped.yaml");
  EXPECT_TRUE(absl::StrContains(config.status().message(), "Add the target of //humanoid_nmpc/humanoid_common_mpc that holds it"))
      << config.status();

  // Without a data directory in the path: the parent of a file, or a directory itself.
  const absl::StatusOr<std::string> file = resolveResourcePath("robot_runtime/robot_core/not_shipped.txt");
  EXPECT_TRUE(absl::StrContains(file.status().message(), "Add the target of //robot_runtime/robot_core that holds it")) << file.status();
  const absl::StatusOr<std::string> directory = resolveResourcePath("robot_models/unitree_r1/unitree_r1_description");
  EXPECT_TRUE(absl::StrContains(directory.status().message(), "Add the target of //robot_models/unitree_r1/unitree_r1_description that"))
      << directory.status();
}

TEST(ResourcePathsTest, RejectsPathsThatAreNotInsideTheRepository) {
  for (const char* path : {"", "/etc/hostname", "../outside.txt", "robot_runtime/../../outside.txt", "robot_runtime//robot_core",
                           "./robot_runtime", "robot_runtime/"}) {
    const absl::StatusOr<std::string> resolved = resolveResourcePath(path);
    EXPECT_EQ(resolved.status().code(), absl::StatusCode::kInvalidArgument) << "'" << path << "': " << resolved.status();
  }
}

TEST(ResourcePathsTest, FindsTheRunfilesNextToTheExecutableWithoutTheEnvironment) {
  // A binary started from .bazel/bin, or through a symlink to it, has no runfiles variables in its environment.
  const ScopedEnvironmentVariable testSrcDir("TEST_SRCDIR", /*value=*/nullptr);
  const ScopedEnvironmentVariable runfilesDir("RUNFILES_DIR", /*value=*/nullptr);
  const ScopedEnvironmentVariable runfilesManifest("RUNFILES_MANIFEST_FILE", /*value=*/nullptr);

  const absl::StatusOr<std::string> path = resolveResourcePath(kProbeFile);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(readFile(*path), kProbeContents);
}

// `bazel run` names the checkout in BUILD_WORKSPACE_DIRECTORY. A file found there would let a binary with an
// incomplete `data` work on the developer's machine and fail on the robot computer, so the checkout is not searched.
TEST(ResourcePathsTest, TheCheckoutOfBazelRunIsNotSearchedForAFileMissingFromTheData) {
  const ScopedTemporaryDirectory checkout("checkout_only");
  checkout.writeFile("only_in_the_checkout/file.txt", "from the checkout\n");
  const ScopedEnvironmentVariable workspace("BUILD_WORKSPACE_DIRECTORY", checkout.path().c_str());

  const absl::StatusOr<std::string> path = resolveResourcePath("only_in_the_checkout/file.txt");
  ASSERT_FALSE(path.ok()) << "resolved to " << *path;
  EXPECT_EQ(path.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(path.status().message(), "Add the target of //only_in_the_checkout that holds it")) << path.status();
}

TEST(ResourcePathsTest, TheRunfilesTakePrecedenceOverTheCheckout) {
  const ScopedTemporaryDirectory checkout("checkout_shadowing_the_runfiles");
  checkout.writeFile(kProbeFile, "a stale copy in the checkout\n");
  const ScopedEnvironmentVariable workspace("BUILD_WORKSPACE_DIRECTORY", checkout.path().c_str());

  const absl::StatusOr<std::string> path = resolveResourcePath(kProbeFile);
  ASSERT_TRUE(path.ok()) << path.status();
  EXPECT_EQ(readFile(*path), kProbeContents);
}

}  // namespace
}  // namespace robot
