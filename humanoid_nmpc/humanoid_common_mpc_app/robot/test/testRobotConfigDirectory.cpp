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
#include <functional>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc_app/robot/RobotConfigDirectory.h"
#include "humanoid_common_mpc_app/robot/RobotStack.h"
#include "humanoid_common_mpc_app/robot/RobotStartup.h"
#include "humanoid_mpc_msgs/config_file_kind.nproto.h"

/*
 * The robot's configuration directory (RobotConfigDirectory.h) on a bundle and a store in the test's scratch space: an
 * empty store is seeded, a restart keeps a save, each seed policy replaces what it should, the contact planner's file is
 * mirrored, a store that cannot be written leaves the bundled files in use, and a start that did not confirm its boot
 * makes the next one fall back to the bundled files unless the machine booted in between. The start's fallback
 * (bringUpRobot()) rejects the stored copies only when the bundle starts where they did not.
 */

namespace ocs2::humanoid {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr char kTask[] = "model_settings { robot_name: \"g1\" }\n";
constexpr char kReference[] = "target_displacement_velocity: 0.5\n";
constexpr char kGains[] = "default_gains { kp: 100.0 }\n";
constexpr char kPlanner[] = "# the contact planner's file\n";
constexpr char kSaved[] = "model_settings { robot_name: \"g1\" }\ntelemetry_frequency: 20\n";

std::string readFile(const std::string& path) {
  std::ifstream file(path);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, absl::string_view contents) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream(path, std::ios::trunc) << contents;
}

bool exists(const std::string& path) {
  std::error_code error;
  return std::filesystem::exists(path, error);
}

/** A bundle of one robot configuration and an empty store, in the scratch directory `name`. */
struct Fixture {
  std::string root;
  RobotConfigDirectory::Options options;
  std::string storedTask;
};

Fixture bundleAndStore(absl::string_view name) {
  Fixture fixture;
  fixture.root = (std::filesystem::path(std::getenv("TEST_TMPDIR")) / std::string(name)).string();
  std::error_code error;
  std::filesystem::remove_all(fixture.root, error);
  const std::string config = fixture.root + "/bundle/robot_models/unitree_g1/g1_wb_mpc/config";
  fixture.options.seeds = {.taskFile = config + "/mpc/task.textproto",
                           .referenceFile = config + "/command/reference.textproto",
                           .pdGainsFile = config + "/controller/joint_pd_gains.textproto"};
  writeFile(fixture.options.seeds.taskFile, kTask);
  writeFile(fixture.options.seeds.referenceFile, kReference);
  writeFile(fixture.options.seeds.pdGainsFile, kGains);
  writeFile(config + "/mpc/contact_planning.textproto", kPlanner);
  fixture.options.storeDirectory = fixture.root + "/store/g1_wb";
  fixture.storedTask = fixture.options.storeDirectory + "/mpc/task.textproto";
  return fixture;
}

/** Opens the directory of `options` and confirms its boot, as a start that ran would. */
absl::StatusOr<RobotConfigDirectory> openAndBoot(const RobotConfigDirectory::Options& options) {
  absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(options);
  if (!directory.ok()) return directory.status();
  absl::Status confirmed = directory->confirmBoot();
  if (!confirmed.ok()) return confirmed;
  return directory;
}

TEST(RobotConfigDirectory, WithoutAStoreTheBundledFilesAreReadInPlace) {
  Fixture fixture = bundleAndStore("in_place");
  fixture.options.storeDirectory.clear();
  const absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(directory.ok()) << directory.status();
  EXPECT_EQ(directory->files().taskFile, fixture.options.seeds.taskFile);
  EXPECT_EQ(directory->files().pdGainsFile, fixture.options.seeds.pdGainsFile);
  EXPECT_TRUE(directory->storedPath(msgs::ConfigFileKind::kTask).empty());
  EXPECT_TRUE(directory->storeError().empty());
  EXPECT_FALSE(directory->usesStoredCopies());
  EXPECT_EQ(directory->identity(msgs::ConfigFileKind::kTask), "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  EXPECT_EQ(directory->taskFileIdentity(), directory->identity(msgs::ConfigFileKind::kTask));
  EXPECT_EQ(directory->identity(msgs::ConfigFileKind::kJointPdGains),
            "robot_models/unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.textproto");
  EXPECT_TRUE(directory->identity(msgs::ConfigFileKind::kUnspecified).empty());
}

TEST(RobotConfigDirectory, AnEmptyStoreIsSeededWithEveryFileAtItsPathBelowConfig) {
  const Fixture fixture = bundleAndStore("seeded");
  const absl::StatusOr<RobotConfigDirectory> directory = openAndBoot(fixture.options);
  ASSERT_TRUE(directory.ok()) << directory.status();
  const RobotConfigDirectory::Files& files = directory->files();
  EXPECT_EQ(files.taskFile, fixture.storedTask);
  EXPECT_EQ(files.referenceFile, fixture.options.storeDirectory + "/command/reference.textproto");
  EXPECT_EQ(files.pdGainsFile, jointPdGainsFileBeside(files.taskFile)) << "the PD gains are still beside the task file";
  EXPECT_EQ(directory->storedPath(msgs::ConfigFileKind::kTask), files.taskFile);
  EXPECT_EQ(readFile(files.taskFile), kTask);
  EXPECT_EQ(readFile(files.referenceFile), kReference);
  EXPECT_EQ(readFile(files.pdGainsFile), kGains);
  EXPECT_EQ(readFile(files.taskFile + ".seed"), kTask);
  EXPECT_EQ(readFile(contactPlanningFileBeside(files.taskFile)), kPlanner) << "the contact planner's file is mirrored";
  EXPECT_FALSE(directory->usesStoredCopies());
  EXPECT_FALSE(exists(fixture.options.storeDirectory + "/.booting")) << "confirmBoot() removed the marker";
}

TEST(RobotConfigDirectory, ARestartKeepsASaveUnlessTheBundleChanged) {
  Fixture fixture = bundleAndStore("restart");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  writeFile(fixture.storedTask, kSaved);  // as a save writes it
  const absl::StatusOr<RobotConfigDirectory> restarted = openAndBoot(fixture.options);
  ASSERT_TRUE(restarted.ok()) << restarted.status();
  EXPECT_EQ(readFile(restarted->files().taskFile), kSaved);
  EXPECT_TRUE(restarted->usesStoredCopies());

  // A deploy that changes the bundled task file replaces the saved copy, which is kept as .bak.
  const std::string deployed = std::string(kTask) + "# a deploy\n";
  writeFile(fixture.options.seeds.taskFile, deployed);
  const absl::StatusOr<RobotConfigDirectory> redeployed = openAndBoot(fixture.options);
  ASSERT_TRUE(redeployed.ok()) << redeployed.status();
  EXPECT_EQ(readFile(fixture.storedTask), deployed);
  EXPECT_EQ(readFile(fixture.storedTask + ".bak"), kSaved);
  EXPECT_EQ(readFile(fixture.storedTask + ".seed"), deployed);
  EXPECT_FALSE(redeployed->usesStoredCopies());
}

TEST(RobotConfigDirectory, EveryStartReplacesASaveAndNeverKeepsItThroughADeploy) {
  Fixture fixture = bundleAndStore("policies");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  writeFile(fixture.storedTask, kSaved);
  fixture.options.seedPolicy = ConfigSeedPolicy::kEveryStart;
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  EXPECT_EQ(readFile(fixture.storedTask), kTask);
  EXPECT_EQ(readFile(fixture.storedTask + ".bak"), kSaved);

  writeFile(fixture.storedTask, kSaved);
  writeFile(fixture.options.seeds.taskFile, std::string(kTask) + "# a deploy\n");
  fixture.options.seedPolicy = ConfigSeedPolicy::kNever;
  const absl::StatusOr<RobotConfigDirectory> kept = openAndBoot(fixture.options);
  ASSERT_TRUE(kept.ok()) << kept.status();
  EXPECT_EQ(readFile(fixture.storedTask), kSaved);
  EXPECT_TRUE(kept->usesStoredCopies());
  // The contact planner's file is the bundle's whatever the policy.
  writeFile(contactPlanningFileBeside(fixture.options.seeds.taskFile), "# a new planner\n");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  EXPECT_EQ(readFile(contactPlanningFileBeside(fixture.storedTask)), "# a new planner\n");
}

TEST(RobotConfigDirectory, AStoredPlannerFileGoesWhenTheBundleHasNone) {
  Fixture fixture = bundleAndStore("planner");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  ASSERT_TRUE(exists(contactPlanningFileBeside(fixture.storedTask)));
  std::filesystem::remove(contactPlanningFileBeside(fixture.options.seeds.taskFile));
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  EXPECT_FALSE(exists(contactPlanningFileBeside(fixture.storedTask)));
}

TEST(RobotConfigDirectory, AStoreThatCannotBeWrittenLeavesTheBundledFilesInUse) {
  Fixture fixture = bundleAndStore("unwritable");
  // A regular file where the store's parent directory belongs: nothing can be created below it, whoever runs the test.
  writeFile(fixture.root + "/store", "not a directory\n");
  const absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(directory.ok()) << directory.status();
  EXPECT_EQ(directory->files().taskFile, fixture.options.seeds.taskFile);
  EXPECT_TRUE(directory->storedPath(msgs::ConfigFileKind::kTask).empty());
  EXPECT_THAT(directory->storeError(), HasSubstr(fixture.options.storeDirectory));
  EXPECT_FALSE(directory->usesStoredCopies());
}

TEST(RobotConfigDirectory, AStartThatDidNotConfirmItsBootFallsBackToTheBundledFiles) {
  Fixture fixture = bundleAndStore("boot_marker");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  writeFile(fixture.storedTask, kSaved);
  {
    // This start dies before it confirms: the marker stays.
    const absl::StatusOr<RobotConfigDirectory> died = RobotConfigDirectory::Open(fixture.options);
    ASSERT_TRUE(died.ok()) << died.status();
    EXPECT_TRUE(died->usesStoredCopies());
    EXPECT_TRUE(exists(fixture.options.storeDirectory + "/.booting"));
  }
  const absl::StatusOr<RobotConfigDirectory> next = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_FALSE(next->usesStoredCopies());
  EXPECT_EQ(readFile(fixture.storedTask), kTask);
  EXPECT_EQ(readFile(fixture.storedTask + ".rejected"), kSaved);
  // A marker with the stored copies equal to the bundle's changes nothing.
  const absl::StatusOr<RobotConfigDirectory> again = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(again.ok()) << again.status();
  EXPECT_EQ(readFile(fixture.storedTask), kTask);
  EXPECT_EQ(readFile(fixture.storedTask + ".rejected"), kSaved);
}

TEST(RobotConfigDirectory, RejectingTheStoredCopiesRestoresTheBundledFiles) {
  Fixture fixture = bundleAndStore("rejected");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  writeFile(fixture.options.storeDirectory + "/controller/joint_pd_gains.textproto", "default_gains { kp: 1.0 }\n");
  absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(directory.ok()) << directory.status();
  ASSERT_TRUE(directory->usesStoredCopies());
  ASSERT_TRUE(directory->rejectStoredCopies().ok());
  EXPECT_FALSE(directory->usesStoredCopies());
  EXPECT_EQ(readFile(directory->files().pdGainsFile), kGains);
  EXPECT_EQ(readFile(directory->files().pdGainsFile + ".rejected"), "default_gains { kp: 1.0 }\n");
  EXPECT_EQ(readFile(directory->files().taskFile), kTask) << "a copy equal to its seed is not touched";
  EXPECT_FALSE(exists(directory->files().taskFile + ".rejected"));
}

TEST(RobotConfigDirectory, AMarkerFromBeforeTheMachineBootedKeepsTheStoredCopies) {
  Fixture fixture = bundleAndStore("power_cut");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  writeFile(fixture.storedTask, kSaved);
  {
    // This start dies before it confirms, and the machine goes down with it: the marker predates the boot.
    const absl::StatusOr<RobotConfigDirectory> died = RobotConfigDirectory::Open(fixture.options);
    ASSERT_TRUE(died.ok()) << died.status();
  }
  std::error_code error;
  // Thirty years ago: before any boot of the machine the test runs on.
  std::filesystem::last_write_time(fixture.options.storeDirectory + "/.booting",
                                   std::filesystem::file_time_type::clock::now() - std::chrono::hours(24 * 365 * 30), error);
  ASSERT_FALSE(error) << error.message();
  const absl::StatusOr<RobotConfigDirectory> next = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_TRUE(next->usesStoredCopies());
  EXPECT_EQ(readFile(fixture.storedTask), kSaved);
  EXPECT_FALSE(exists(fixture.storedTask + ".rejected"));
}

/** A bring-up that records the task file of every attempt and fails on the files `fails` refuses. */
struct FakeBringUp {
  std::function<bool(const RobotConfigDirectory&)> fails;
  std::vector<std::string> attempts;

  absl::StatusOr<RobotStack> operator()(const RobotConfigDirectory& directory) {
    attempts.push_back(directory.files().taskFile);
    if (fails(directory)) return absl::UnavailableError("the bus port 5600 is in use");
    return RobotStack{};
  }
};

TEST(RobotStartup, AFailureTheBundleSharesKeepsTheStoredCopiesAndWithdrawsTheMarker) {
  Fixture fixture = bundleAndStore("shared_failure");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  writeFile(fixture.storedTask, kSaved);
  absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(directory.ok()) << directory.status();
  ASSERT_TRUE(directory->usesStoredCopies());
  FakeBringUp bringUp{.fails = [](const RobotConfigDirectory& /*files*/) { return true; }};
  const absl::StatusOr<RobotStack> stack =
      bringUpRobot(*directory, [&bringUp](const RobotConfigDirectory& files) { return bringUp(files); });
  EXPECT_EQ(stack.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(bringUp.attempts, ElementsAre(fixture.storedTask, fixture.options.seeds.taskFile)) << "the store, then the bundle in place";
  EXPECT_EQ(readFile(fixture.storedTask), kSaved) << "a failure that is not the stored copies' changed them";
  EXPECT_FALSE(exists(fixture.storedTask + ".rejected"));
  EXPECT_FALSE(exists(fixture.options.storeDirectory + "/.booting")) << "the next start would reject them";
  const absl::StatusOr<RobotConfigDirectory> next = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_TRUE(next->usesStoredCopies()) << "the next start runs the stored copies again";
}

TEST(RobotStartup, AFailureOnTheStoredCopiesRejectsThemAndStartsOnTheBundle) {
  Fixture fixture = bundleAndStore("stored_failure");
  ASSERT_TRUE(openAndBoot(fixture.options).ok());
  writeFile(fixture.storedTask, kSaved);
  absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(directory.ok()) << directory.status();
  FakeBringUp bringUp{.fails = [](const RobotConfigDirectory& files) { return readFile(files.files().taskFile) == kSaved; }};
  const absl::StatusOr<RobotStack> stack =
      bringUpRobot(*directory, [&bringUp](const RobotConfigDirectory& files) { return bringUp(files); });
  EXPECT_TRUE(stack.ok()) << stack.status();
  EXPECT_THAT(bringUp.attempts, ElementsAre(fixture.storedTask, fixture.options.seeds.taskFile, fixture.storedTask))
      << "the store, the bundle in place, the store again";
  EXPECT_EQ(readFile(fixture.storedTask), kTask);
  EXPECT_EQ(readFile(fixture.storedTask + ".rejected"), kSaved);
  EXPECT_FALSE(directory->usesStoredCopies());
}

TEST(RobotStartup, AFailureWithoutStoredCopiesIsReturnedAsItIs) {
  Fixture fixture = bundleAndStore("bundle_failure");
  absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(fixture.options);
  ASSERT_TRUE(directory.ok()) << directory.status();
  FakeBringUp bringUp{.fails = [](const RobotConfigDirectory& /*files*/) { return true; }};
  const absl::StatusOr<RobotStack> stack =
      bringUpRobot(*directory, [&bringUp](const RobotConfigDirectory& files) { return bringUp(files); });
  EXPECT_EQ(stack.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(bringUp.attempts, ElementsAre(fixture.storedTask)) << "nothing to fall back from";
}

TEST(RobotConfigDirectory, ASeedThatCannotBeReadIsAnError) {
  Fixture fixture = bundleAndStore("missing_seed");
  std::filesystem::remove(fixture.options.seeds.referenceFile);
  const absl::StatusOr<RobotConfigDirectory> directory = RobotConfigDirectory::Open(fixture.options);
  EXPECT_EQ(directory.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(directory.status().message(), HasSubstr("reference.textproto"));
}

TEST(ConfigSeedPolicy, IsSelectedByName) {
  EXPECT_EQ(configSeedPolicyFromName("when_bundle_changes").value_or(ConfigSeedPolicy::kNever), ConfigSeedPolicy::kWhenBundleChanges);
  EXPECT_EQ(configSeedPolicyFromName("every_start").value_or(ConfigSeedPolicy::kNever), ConfigSeedPolicy::kEveryStart);
  EXPECT_EQ(configSeedPolicyFromName("never").value_or(ConfigSeedPolicy::kEveryStart), ConfigSeedPolicy::kNever);
  for (const ConfigSeedPolicy policy : {ConfigSeedPolicy::kWhenBundleChanges, ConfigSeedPolicy::kEveryStart, ConfigSeedPolicy::kNever}) {
    EXPECT_EQ(configSeedPolicyFromName(configSeedPolicyName(policy)).value_or(ConfigSeedPolicy::kNever), policy);
  }
  const absl::StatusOr<ConfigSeedPolicy> unknown = configSeedPolicyFromName("sometimes");
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(unknown.status().message(), HasSubstr("when_bundle_changes, every_start, never"));
}

}  // namespace
}  // namespace ocs2::humanoid
