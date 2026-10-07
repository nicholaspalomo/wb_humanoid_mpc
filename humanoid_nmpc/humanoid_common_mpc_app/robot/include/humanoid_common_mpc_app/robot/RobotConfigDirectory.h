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
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/robot/StoredConfigFileKinds.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"

namespace ocs2::humanoid {

/**
 * When a stored copy is replaced by the bundled file it was seeded from (--config_seed). A deploy that changes a file of
 * the bundle reaches the robot under kWhenBundleChanges; a restart without a deploy keeps the robot's saves.
 */
enum class ConfigSeedPolicy {
  /** Replaced when the bundled file differs from the one it was last seeded from (`.seed`): a deploy changed it. */
  kWhenBundleChanges,
  /** Replaced at every start: the robot runs the bundle's files, and a save lasts until the next start (simulation). */
  kEveryStart,
  /** Never replaced once there: only a save changes it. */
  kNever,
};

// LINT.IfChange(config_seed_policies)
inline constexpr absl::string_view kWhenBundleChangesSeedPolicyName = "when_bundle_changes";
inline constexpr absl::string_view kEveryStartSeedPolicyName = "every_start";
inline constexpr absl::string_view kNeverSeedPolicyName = "never";
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:config_seed_policies, //tools/deploy/README.md:config_seed_policies, //tools/deploy/deploy_robot.sh:config_seed_policies, //humanoid_nmpc/humanoid_common_mpc_app/robot/src/RobotAppFlags.cpp:robot_flags)
// clang-format on

/** The policy `name` names (--config_seed); InvalidArgument listing the names for any other. */
absl::StatusOr<ConfigSeedPolicy> configSeedPolicyFromName(absl::string_view name);

/** The name of `policy`. */
absl::string_view configSeedPolicyName(ConfigSeedPolicy policy);

/**
 * The configuration files a robot process reads, and the persistent copies the tuning GUI's Save writes (ConfigFileStore,
 * humanoid_nmpc/humanoid_common_mpc_app/robot/README.md, "The configuration store").
 *
 * Without a store directory (--config_store_dir empty, a `bazel run` in the dev container) the process reads the bundled
 * files - the seeds - in place. With one, every file the robot reads has a copy there at its path below config/
 * (mpc/task.textproto, command/reference.textproto, controller/joint_pd_gains.textproto, so that the PD gains file is
 * still the task file's jointPdGainsFileBeside()), with `<file>.seed` the bundled bytes it was last seeded from and
 * `<file>.bak` the copy a replacement or a save replaced. Open() seeds a missing copy and applies the seed policy; the
 * contact planner's file beside the task file is mirrored from the bundle at every start (it is the MPC's, read here only
 * for the viewer, and no save writes it), and removed when the bundle has none.
 *
 * A store that cannot be created or written is logged as an ERROR and the seeds are read in place; storeError() says
 * why, and the configuration store answers every save FAILED with it.
 *
 * A stored configuration never keeps the robot from starting: Open() writes the boot marker `.booting`, which
 * confirmBoot() removes once the process has run. A marker left by the previous start (it died before it confirmed)
 * makes Open() reject every stored copy that differs from its seed: renamed `<file>.rejected`, the seed copied in its
 * place, logged as an ERROR - unless the machine has booted since the marker was written (a power cut or a reboot ended
 * that start, not its files), which keeps them. A start that fails on stored copies is tried on the bundle in place
 * (bundleInPlace(), which touches no stored file); only when the bundle starts are the stored copies rejected
 * (rejectStoredCopies()), and when it fails too the failure is not theirs: they are kept, and the marker withdrawn
 * (withdrawBootMarker()). runRobot() does all of it.
 *
 * Not thread-safe: the main thread opens it before anything reads a file; its const methods may then be called from any
 * thread.
 */
class RobotConfigDirectory {
 public:
  /** The robot's configuration files, by path. */
  struct Files {
    std::string taskFile;
    std::string referenceFile;
    /** jointPdGainsFileBeside(taskFile). */
    std::string pdGainsFile;
  };

  struct Options {
    /** The persistent directory of this robot configuration (--config_store_dir); empty: read the seeds in place. */
    std::string storeDirectory;
    ConfigSeedPolicy seedPolicy = ConfigSeedPolicy::kWhenBundleChanges;
    /** The bundled files (--task_file, --reference_file and the PD gains beside the task file). */
    Files seeds;
  };

  /**
   * Prepares the files of `options` (see the class comment). NotFound or the read error naming a seed that cannot be
   * read; a store that cannot be used is no error (storeError()).
   */
  static absl::StatusOr<RobotConfigDirectory> Open(Options options);

  RobotConfigDirectory(const RobotConfigDirectory&) = default;
  RobotConfigDirectory& operator=(const RobotConfigDirectory&) = default;
  RobotConfigDirectory(RobotConfigDirectory&&) noexcept = default;
  RobotConfigDirectory& operator=(RobotConfigDirectory&&) noexcept = default;
  ~RobotConfigDirectory() = default;

  /** The files to read: the stored copies, or the seeds when there is no usable store. */
  const Files& files() const { return files_; }
  /** The bundled files. */
  const Files& seeds() const { return options_.seeds; }
  /** configFileIdentity() of the seed of `kind`: what a save or a live update of this configuration names; "" for kUnspecified. */
  std::string identity(msgs::ConfigFileKind kind) const;
  /** identity(kTask). */
  std::string taskFileIdentity() const { return identity(msgs::ConfigFileKind::kTask); }
  /** The persistent copy of `kind`'s file; empty without a usable store and for kUnspecified. */
  std::string storedPath(msgs::ConfigFileKind kind) const;
  /** The file of `kind` the robot reads (files()): the stored copy, or the seed without a usable store; "" for kUnspecified. */
  std::string readPath(msgs::ConfigFileKind kind) const;
  /** The store directory (--config_store_dir); empty: none. */
  const std::string& storeDirectory() const { return options_.storeDirectory; }
  /** Why the store directory cannot be used; empty when it can or when there is none. */
  const std::string& storeError() const { return storeError_; }
  /** Whether a file in use differs from its seed: what rejectStoredCopies() would replace. */
  bool usesStoredCopies() const;

  /** Removes the boot marker: this start ran. OK without a store. */
  absl::Status confirmBoot();

  /**
   * Removes the boot marker of a start that failed for a cause other than the stored copies (it failed on the bundle
   * too), so that the next start runs them again. OK without a store.
   */
  absl::Status withdrawBootMarker();

  /**
   * A copy of this directory that reads the bundled files in place and has no store: for a trial start on the bundle,
   * which touches no stored file.
   */
  RobotConfigDirectory bundleInPlace() const;

  /**
   * Renames every stored copy that differs from its seed to `<file>.rejected` (replacing an older one) and copies the
   * seed in its place, logging each as an ERROR: for a set-up that failed on them. OK without a store.
   */
  absl::Status rejectStoredCopies();

 private:
  explicit RobotConfigDirectory(Options options);

  /** The seed of the kind of index `kind` in kStoredConfigFileKinds; the arrays below are indexed alike. */
  const std::string& seedOf(size_t kind) const;
  /** The boot marker in the store directory. */
  std::string bootMarker() const;
  /** Prepares the store; an error leaves the seeds in use (storeError_). */
  absl::Status prepareStore();
  absl::Status seedFile(size_t kind);
  absl::Status rejectStoredCopy(size_t kind);
  absl::Status mirrorContactPlanningFile() const;
  void useSeedsInPlace(const absl::Status& error);

  Options options_;
  Files files_;
  /** The stored paths by kind index; empty without a usable store. */
  std::array<std::string, kNumStoredConfigFileKinds> storedPaths_;
  /** The seeds' bytes by kind index, read by Open(). */
  std::array<std::string, kNumStoredConfigFileKinds> seedBytes_;
  /** Whether the stored copy of each kind differs from its seed. */
  std::array<bool, kNumStoredConfigFileKinds> differsFromSeed_ = {};
  std::string storeError_;
};

}  // namespace ocs2::humanoid
