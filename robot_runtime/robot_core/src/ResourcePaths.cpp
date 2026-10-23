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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "rules_cc/cc/runfiles/runfiles.h"

namespace robot {
namespace {

using ::rules_cc::cc::runfiles::Runfiles;

/** The runfiles directory of the main repository under Bzlmod. */
constexpr absl::string_view kMainRepository = "_main";

/**
 * The directories in which the Bazel packages of this repository keep their data files. A data file belongs to the
 * package in front of the first of them, so robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf belongs to
 * //robot_models/drc_atlas/drc_atlas_description.
 */
constexpr std::array<absl::string_view, 7> kDataDirectories = {"config", "launch", "materials", "meshes", "test", "testdata", "urdf"};

std::string environmentVariable(const char* absl_nonnull name) {
  const char* absl_nullable value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

/** InvalidArgument unless `path` is a normalized relative path that stays inside the repository. */
absl::Status validateRepositoryRelativePath(absl::string_view path) {
  if (path.empty()) return absl::InvalidArgumentError("resolveResourcePath: the path is empty");
  if (absl::StartsWith(path, "/")) {
    return absl::InvalidArgumentError(
        absl::StrCat("resolveResourcePath: '", path, "' is absolute; pass the path relative to the repository root"));
  }
  for (const absl::string_view component : absl::StrSplit(path, '/')) {
    if (component.empty() || component == "." || component == "..") {
      return absl::InvalidArgumentError(absl::StrCat(
          "resolveResourcePath: '", path, "' is not a normalized path inside the repository (no empty, '.' or '..' components)"));
    }
  }
  return absl::OkStatus();
}

/**
 * The label of the Bazel package that `path` belongs to, as far as the path shows it. The package ends before the
 * first data directory (kDataDirectories). Without one, the package is the path itself when the path names a
 * directory (no extension), or the file's parent directory otherwise.
 */
std::string owningPackageLabel(absl::string_view path) {
  const std::vector<absl::string_view> components = absl::StrSplit(path, '/');
  size_t packageLength = absl::StrContains(components.back(), '.') ? components.size() - 1 : components.size();
  for (size_t i = 0; i + 1 < components.size(); ++i) {
    if (std::find(kDataDirectories.begin(), kDataDirectories.end(), components[i]) != kDataDirectories.end()) {
      packageLength = i;
      break;
    }
  }
  const std::vector<absl::string_view> package(components.begin(), components.begin() + static_cast<ptrdiff_t>(packageLength));
  return absl::StrCat("//", absl::StrJoin(package, "/"));
}

/** Appends `entry` to `searched` unless it is there already: two places can name the same runfiles tree. */
void recordSearched(std::string entry, std::vector<std::string>& searched) {
  if (std::find(searched.begin(), searched.end(), entry) == searched.end()) searched.push_back(std::move(entry));
}

/** `path` as an absolute path, unchanged when the working directory is unknown. */
std::string absolutePath(const std::string& path) {
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  return error ? path : absolute.string();
}

/** The path of `runfilesPath` in `runfiles` when it exists there. Otherwise records where it was searched for. */
std::optional<std::string> findInRunfiles(const Runfiles& runfiles,
                                          const std::string& runfilesPath,
                                          absl::string_view runfilesName,
                                          std::vector<std::string>& searched) {
  const std::string candidate = runfiles.Rlocation(runfilesPath);
  if (candidate.empty()) {
    recordSearched(absl::StrCat(runfilesName, " (not listed in its manifest)"), searched);
    return std::nullopt;
  }
  std::error_code error;
  if (std::filesystem::exists(candidate, error)) return absolutePath(candidate);
  recordSearched(candidate, searched);
  return std::nullopt;
}

/** The runfiles the environment names, or nullptr with the reason in `error`. */
std::unique_ptr<Runfiles> environmentRunfiles(std::string& error) {
  if (!environmentVariable("TEST_SRCDIR").empty()) return std::unique_ptr<Runfiles>(Runfiles::CreateForTest(&error));
  if (environmentVariable("RUNFILES_DIR").empty() && environmentVariable("RUNFILES_MANIFEST_FILE").empty()) {
    error = "none of TEST_SRCDIR, RUNFILES_DIR and RUNFILES_MANIFEST_FILE is set";
    return nullptr;
  }
  return std::unique_ptr<Runfiles>(Runfiles::Create(/*argv0=*/"", &error));
}

/** The path of the running executable with every symlink resolved, so a binary started through a link finds its runfiles. */
std::optional<std::string> executablePath() {
  std::error_code error;
  const std::filesystem::path path = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || path.empty()) return std::nullopt;
  return path.string();
}

}  // namespace

absl::StatusOr<std::string> resolveResourcePath(absl::string_view repositoryRelativePath) {
  absl::Status valid = validateRepositoryRelativePath(repositoryRelativePath);
  if (!valid.ok()) return valid;

  const std::string runfilesPath = absl::StrCat(kMainRepository, "/", repositoryRelativePath);
  std::vector<std::string> searched;

  // 1. The runfiles the environment names: `bazel test`, or a Bazel binary that started this one.
  std::string error;
  const std::unique_ptr<Runfiles> fromEnvironment = environmentRunfiles(error);
  if (fromEnvironment != nullptr) {
    const std::optional<std::string> found = findInRunfiles(*fromEnvironment, runfilesPath, "the runfiles of the environment", searched);
    if (found.has_value()) return *found;
  } else {
    recordSearched(absl::StrCat("no runfiles in the environment: ", error), searched);
  }

  // 2. The runfiles tree next to the executable: `bazel run`, or a binary started from .bazel/bin. The environment is
  // ignored here, because it can name the runfiles of a parent process that lists other files.
  const std::optional<std::string> executable = executablePath();
  if (executable.has_value()) {
    error.clear();
    const std::unique_ptr<Runfiles> nextToExecutable(
        Runfiles::Create(*executable, /*runfiles_manifest_file=*/"", /*runfiles_dir=*/"", &error));
    if (nextToExecutable != nullptr) {
      const std::optional<std::string> found =
          findInRunfiles(*nextToExecutable, runfilesPath, absl::StrCat("the runfiles of ", *executable), searched);
      if (found.has_value()) return *found;
    } else {
      recordSearched(absl::StrCat("no runfiles next to ", *executable), searched);
    }
  } else {
    recordSearched("no runfiles next to the executable: /proc/self/exe cannot be read", searched);
  }

  // The checkout `bazel run` names in BUILD_WORKSPACE_DIRECTORY is not searched (see the header): a file found there
  // would hide the missing `data` dependency that the message below names.

  return absl::NotFoundError(absl::StrCat(repositoryRelativePath, " is not in the runfiles of this binary. Add the target of ",
                                          owningPackageLabel(repositoryRelativePath),
                                          " that holds it to the `data` of the test or binary. Searched: ", absl::StrJoin(searched, "; ")));
}

}  // namespace robot
