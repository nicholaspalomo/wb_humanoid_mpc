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

#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"

#include <sys/stat.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc_app/robot/AtomicFileWrite.h"
#include "humanoid_common_mpc_app/robot/StoredConfigFileKinds.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"

namespace ocs2::humanoid {
namespace {

// The layout of the store: each file at its path below the robot's config/ directory, and the files the store keeps
// beside them.
// LINT.IfChange(config_store_layout)
constexpr absl::string_view kStoredTaskFile = "mpc/task.textproto";
constexpr absl::string_view kStoredReferenceFile = "command/reference.textproto";
// The PD gains: kJointPdGainsFileInConfigDirectory; the contact planner's mirror: kContactPlanningFileName in mpc/.
constexpr absl::string_view kSeedSuffix = ".seed";
constexpr absl::string_view kBackupSuffix = ".bak";
constexpr absl::string_view kRejectedSuffix = ".rejected";
constexpr absl::string_view kBootMarker = ".booting";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:config_store_layout, //tools/deploy/README.md:config_store_layout)
// clang-format on

/** The path of the file of `kind` below the store `directory`; empty for a kind the robot does not store. */
std::string storedPathIn(const std::string& directory, msgs::ConfigFileKind kind) {
  const std::filesystem::path root(directory);
  switch (kind) {
    case msgs::ConfigFileKind::kTask:
      return (root / std::string(kStoredTaskFile)).string();
    case msgs::ConfigFileKind::kReference:
      return (root / std::string(kStoredReferenceFile)).string();
    case msgs::ConfigFileKind::kJointPdGains:
      return (root / std::string(kJointPdGainsFileInConfigDirectory)).string();
    case msgs::ConfigFileKind::kUnspecified:
      return std::string();
  }
  return std::string();
}

// The index of the task file in the arrays of RobotConfigDirectory.
constexpr size_t kTaskIndex = 0;
static_assert(kStoredConfigFileKinds[kTaskIndex] == msgs::ConfigFileKind::kTask, "the task file is the first stored kind");

/**
 * Whether the file at `path` was last written before the machine booted (the boot time of /proc/stat): a start that was
 * ended by the machine going down, not by what it ran. False when either time cannot be read.
 */
bool writtenBeforeTheMachineBooted(const std::string& path) {
  struct stat status = {};
  if (::stat(path.c_str(), &status) != 0) return false;
  const absl::StatusOr<std::string> machine = readFileBytes("/proc/stat");
  if (!machine.ok()) return false;
  for (const absl::string_view line : absl::StrSplit(*machine, '\n')) {
    absl::string_view value = line;
    int64_t bootTime = 0;
    if (absl::ConsumePrefix(&value, "btime ") && absl::SimpleAtoi(value, &bootTime)) {
      return static_cast<int64_t>(status.st_mtime) < bootTime;
    }
  }
  return false;
}

/** A file's contents, when there is a file. */
struct ReadFile {
  bool exists = false;
  /** Empty when there is no file. */
  std::string bytes;
};

/** The file at `path`, which need not exist; an error other than its absence is returned. */
absl::StatusOr<ReadFile> readIfThere(const std::string& path) {
  absl::StatusOr<std::string> bytes = readFileBytes(path);
  if (bytes.ok()) return ReadFile{.exists = true, .bytes = *std::move(bytes)};
  if (absl::IsNotFound(bytes.status())) return ReadFile{};
  return bytes.status();
}

/**
 * `done`, an operation on a file of the store, with a DataLoss (the file was written, renamed or removed, but its
 * directory could not be synced: AtomicFileWrite.h) logged and taken as done: the robot reads what was written.
 */
absl::Status doneOrNotDurable(absl::Status done) {
  if (!absl::IsDataLoss(done)) return done;
  LOG(WARNING) << "[RobotConfigDirectory] " << done.message();
  return absl::OkStatus();
}

/** Creates the directory of `path` and its parents. */
absl::Status createParentOf(const std::string& path) {
  std::error_code error;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), error);
  if (error) return absl::UnavailableError(absl::StrCat("cannot create the directory of ", path, ": ", error.message()));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<ConfigSeedPolicy> configSeedPolicyFromName(absl::string_view name) {
  if (name == kWhenBundleChangesSeedPolicyName) return ConfigSeedPolicy::kWhenBundleChanges;
  if (name == kEveryStartSeedPolicyName) return ConfigSeedPolicy::kEveryStart;
  if (name == kNeverSeedPolicyName) return ConfigSeedPolicy::kNever;
  return absl::InvalidArgumentError(
      absl::StrCat("There is no seed policy '", name, "' (--config_seed). Available: ",
                   absl::StrJoin({kWhenBundleChangesSeedPolicyName, kEveryStartSeedPolicyName, kNeverSeedPolicyName}, ", "), "."));
}

absl::string_view configSeedPolicyName(ConfigSeedPolicy policy) {
  switch (policy) {
    case ConfigSeedPolicy::kWhenBundleChanges:
      return kWhenBundleChangesSeedPolicyName;
    case ConfigSeedPolicy::kEveryStart:
      return kEveryStartSeedPolicyName;
    case ConfigSeedPolicy::kNever:
      return kNeverSeedPolicyName;
  }
  return "unknown";
}

RobotConfigDirectory::RobotConfigDirectory(Options options) : options_(std::move(options)), files_(options_.seeds) {}

absl::StatusOr<RobotConfigDirectory> RobotConfigDirectory::Open(Options options) {
  RobotConfigDirectory directory(std::move(options));
  for (size_t kind = 0; kind < kNumStoredConfigFileKinds; ++kind) {
    absl::StatusOr<std::string> seed = readFileBytes(directory.seedOf(kind));
    if (!seed.ok()) {
      return absl::Status(seed.status().code(), absl::StrCat("the bundled configuration file cannot be read: ", seed.status().message()));
    }
    directory.seedBytes_[kind] = *std::move(seed);
  }
  if (directory.options_.storeDirectory.empty()) {
    LOG(INFO) << "[RobotConfigDirectory] No configuration store (--config_store_dir): reading " << directory.files_.taskFile
              << " and the files beside it in place.";
    return directory;
  }
  if (const absl::Status prepared = directory.prepareStore(); !prepared.ok()) {
    directory.useSeedsInPlace(prepared);
  }
  return directory;
}

const std::string& RobotConfigDirectory::seedOf(size_t kind) const {
  switch (kStoredConfigFileKinds[kind]) {
    case msgs::ConfigFileKind::kTask:
      return options_.seeds.taskFile;
    case msgs::ConfigFileKind::kReference:
      return options_.seeds.referenceFile;
    case msgs::ConfigFileKind::kJointPdGains:
      return options_.seeds.pdGainsFile;
    case msgs::ConfigFileKind::kUnspecified:
      break;
  }
  // Unreachable: kStoredConfigFileKinds lists no kUnspecified.
  return options_.seeds.taskFile;
}

std::string RobotConfigDirectory::bootMarker() const {
  return (std::filesystem::path(options_.storeDirectory) / std::string(kBootMarker)).string();
}

absl::Status RobotConfigDirectory::prepareStore() {
  for (size_t kind = 0; kind < kNumStoredConfigFileKinds; ++kind) {
    storedPaths_[kind] = storedPathIn(options_.storeDirectory, kStoredConfigFileKinds[kind]);
  }
  const std::string marker = bootMarker();
  ASSIGN_OR_RETURN(const ReadFile previousBoot, readIfThere(marker));
  for (size_t kind = 0; kind < kNumStoredConfigFileKinds; ++kind) RETURN_IF_ERROR(seedFile(kind));
  RETURN_IF_ERROR(mirrorContactPlanningFile());
  if (previousBoot.exists && writtenBeforeTheMachineBooted(marker)) {
    // The machine went down during the previous start (a power cut, a reboot): that says nothing about its files.
    LOG(WARNING) << "[RobotConfigDirectory] The previous start did not confirm its boot (" << marker
                 << " is there), but the machine has booted since: the stored copies are kept.";
  } else if (previousBoot.exists) {
    // The previous start died before it confirmed: what it ran on is not run again.
    LOG(ERROR) << "[RobotConfigDirectory] The previous start did not confirm its boot (" << marker
               << " is there): falling back to the bundled files for every stored copy that differs from them.";
    for (size_t kind = 0; kind < kNumStoredConfigFileKinds; ++kind) RETURN_IF_ERROR(rejectStoredCopy(kind));
  }
  RETURN_IF_ERROR(doneOrNotDurable(writeFileAtomically(marker, absl::StrCat(configSeedPolicyName(options_.seedPolicy), "\n"))));
  files_ = Files{.taskFile = storedPathIn(options_.storeDirectory, msgs::ConfigFileKind::kTask),
                 .referenceFile = storedPathIn(options_.storeDirectory, msgs::ConfigFileKind::kReference),
                 .pdGainsFile = storedPathIn(options_.storeDirectory, msgs::ConfigFileKind::kJointPdGains)};
  LOG(INFO) << "[RobotConfigDirectory] Reading the configuration store " << options_.storeDirectory
            << " (--config_seed=" << configSeedPolicyName(options_.seedPolicy) << ")"
            << (usesStoredCopies() ? ", which holds saved files." : ".");
  return absl::OkStatus();
}

absl::Status RobotConfigDirectory::seedFile(size_t kind) {
  const std::string& stored = storedPaths_[kind];
  const std::string& seed = seedBytes_[kind];
  const std::string seedMarker = absl::StrCat(stored, kSeedSuffix);
  ASSIGN_OR_RETURN(const ReadFile current, readIfThere(stored));
  ASSIGN_OR_RETURN(const ReadFile seededFrom, readIfThere(seedMarker));
  bool replace = !current.exists;
  if (current.exists && current.bytes != seed) {
    switch (options_.seedPolicy) {
      case ConfigSeedPolicy::kEveryStart:
        replace = true;
        break;
      case ConfigSeedPolicy::kWhenBundleChanges:
        replace = !seededFrom.exists || seededFrom.bytes != seed;
        break;
      case ConfigSeedPolicy::kNever:
        replace = false;
        break;
    }
  }
  if (replace) {
    RETURN_IF_ERROR(createParentOf(stored));
    if (current.exists) {
      RETURN_IF_ERROR(doneOrNotDurable(writeFileAtomically(absl::StrCat(stored, kBackupSuffix), current.bytes)));
      LOG(INFO) << "[RobotConfigDirectory] " << stored << " is the bundle's " << seedOf(kind)
                << " again (--config_seed=" << configSeedPolicyName(options_.seedPolicy) << "); the copy it replaced is " << stored
                << kBackupSuffix << ".";
    } else {
      LOG(INFO) << "[RobotConfigDirectory] Seeded " << stored << " from " << seedOf(kind) << ".";
    }
    RETURN_IF_ERROR(doneOrNotDurable(writeFileAtomically(stored, seed)));
  }
  // The bundled bytes the copy was last seeded from; under kNever a copy that was kept is left as it was.
  const bool seeded = replace || current.bytes == seed;
  if ((seeded || options_.seedPolicy != ConfigSeedPolicy::kNever) && (!seededFrom.exists || seededFrom.bytes != seed)) {
    RETURN_IF_ERROR(doneOrNotDurable(writeFileAtomically(seedMarker, seed)));
  }
  differsFromSeed_[kind] = !replace && current.bytes != seed;
  return absl::OkStatus();
}

absl::Status RobotConfigDirectory::mirrorContactPlanningFile() const {
  const std::string seed = contactPlanningFileBeside(options_.seeds.taskFile);
  const std::string mirror = contactPlanningFileBeside(storedPaths_[kTaskIndex]);
  ASSIGN_OR_RETURN(const ReadFile bundled, readIfThere(seed));
  if (!bundled.exists) return doneOrNotDurable(removeFile(mirror));
  ASSIGN_OR_RETURN(const ReadFile current, readIfThere(mirror));
  if (current.exists && current.bytes == bundled.bytes) return absl::OkStatus();
  RETURN_IF_ERROR(createParentOf(mirror));
  return doneOrNotDurable(writeFileAtomically(mirror, bundled.bytes));
}

absl::Status RobotConfigDirectory::rejectStoredCopy(size_t kind) {
  if (!differsFromSeed_[kind]) return absl::OkStatus();
  const std::string& stored = storedPaths_[kind];
  const std::string rejected = absl::StrCat(stored, kRejectedSuffix);
  RETURN_IF_ERROR(doneOrNotDurable(renameFile(stored, rejected)));
  RETURN_IF_ERROR(doneOrNotDurable(writeFileAtomically(stored, seedBytes_[kind])));
  RETURN_IF_ERROR(doneOrNotDurable(writeFileAtomically(absl::StrCat(stored, kSeedSuffix), seedBytes_[kind])));
  differsFromSeed_[kind] = false;
  LOG(ERROR) << "[RobotConfigDirectory] Rejected the stored " << stored << ": it is now " << rejected << ", and " << stored
             << " is the bundle's " << seedOf(kind) << " again.";
  return absl::OkStatus();
}

void RobotConfigDirectory::useSeedsInPlace(const absl::Status& error) {
  storeError_ = absl::StrCat("the configuration store ", options_.storeDirectory, " cannot be used: ", error.message());
  LOG(ERROR) << "[RobotConfigDirectory] " << storeError_ << ". Reading the bundled files in place; a save from the GUI fails.";
  files_ = options_.seeds;
  storedPaths_.fill(std::string());
  differsFromSeed_.fill(false);
}

std::string RobotConfigDirectory::identity(msgs::ConfigFileKind kind) const {
  const std::optional<size_t> index = storedConfigFileKindIndex(kind);
  if (!index.has_value()) return std::string();
  return configFileIdentity(seedOf(*index));
}

std::string RobotConfigDirectory::storedPath(msgs::ConfigFileKind kind) const {
  const std::optional<size_t> index = storedConfigFileKindIndex(kind);
  if (!index.has_value()) return std::string();
  return storedPaths_[*index];
}

std::string RobotConfigDirectory::readPath(msgs::ConfigFileKind kind) const {
  const std::optional<size_t> index = storedConfigFileKindIndex(kind);
  if (!index.has_value()) return std::string();
  return storedPaths_[*index].empty() ? seedOf(*index) : storedPaths_[*index];
}

RobotConfigDirectory RobotConfigDirectory::bundleInPlace() const {
  RobotConfigDirectory bundle(options_);
  bundle.options_.storeDirectory.clear();
  bundle.seedBytes_ = seedBytes_;
  return bundle;
}

bool RobotConfigDirectory::usesStoredCopies() const {
  for (const bool differs : differsFromSeed_) {
    if (differs) return true;
  }
  return false;
}

absl::Status RobotConfigDirectory::confirmBoot() {
  if (storedPaths_[kTaskIndex].empty()) return absl::OkStatus();
  return doneOrNotDurable(removeFile(bootMarker()));
}

absl::Status RobotConfigDirectory::withdrawBootMarker() {
  // The same file as confirmBoot(), for another reason: the start failed, but not on the stored copies.
  return confirmBoot();
}

absl::Status RobotConfigDirectory::rejectStoredCopies() {
  if (storedPaths_[kTaskIndex].empty()) return absl::OkStatus();
  for (size_t kind = 0; kind < kNumStoredConfigFileKinds; ++kind) RETURN_IF_ERROR(rejectStoredCopy(kind));
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid
