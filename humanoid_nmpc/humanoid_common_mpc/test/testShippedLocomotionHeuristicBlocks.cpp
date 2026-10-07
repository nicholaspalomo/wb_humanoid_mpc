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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ostream>
#include <regex>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {

namespace {

/**
 * The `locomotion_heuristics` block of every robot's SHIPPED centroidal task file, read from the files on disk rather
 * than from a string written for the test.
 *
 * Only DRC Atlas's file is ever loaded by the integration suite, because that is the robot every interface-building
 * test uses; the other three blocks carry different values and comments and are passed through the conversion by no
 * other Bazel test. So a value validate() rejects, a name uncommented by accident, or a parameter left out of the SA01,
 * G1 or R1 file would surface only when that robot is launched, with CI green. This suite is where those become test
 * failures.
 *
 * The scan for the commented candidates is textual, because a comment is exactly what the parser throws away; what the
 * file says is compared with what the MPC makes of it (loadTaskFile(), locomotionHeuristicConfigFromConfig()).
 */
struct ShippedTaskFile {
  const char* absl_nonnull robot;
  const char* absl_nonnull relativePath;  // under the runfiles root, i.e. the repository root
};

void PrintTo(const ShippedTaskFile& file, std::ostream* absl_nonnull os) {
  *os << file.robot << " (" << file.relativePath << ")";
}

// Every robot that ships a centroidal task file. The BUILD target's `data` has to list the same packages: a robot here
// but not there fails in SetUp, and one there but not here fails TheCentroidalTaskFilesInTheRunfilesAreExactlyTheList.
// Every robot's block carries every parameter of the schema, and the parameter deriver knows every robot it can derive
// for.
// LINT.IfChange(shipped_centroidal_task_files)
constexpr ShippedTaskFile kShippedTaskFiles[] = {
    {.robot = "drc_atlas", .relativePath = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto"},
    {.robot = "engineai_sa01", .relativePath = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto"},
    {.robot = "unitree_g1", .relativePath = "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto"},
    {.robot = "unitree_r1", .relativePath = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto"},
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:shipped_centroidal_task_files_data, //humanoid_nmpc/humanoid_mpc_config/locomotion_heuristics_config.proto:locomotion_heuristic_keys, //tools/locomotion_heuristics/derive_parameters.py:derive_parameters_robots)
// clang-format on

/** The package-name suffixes of the two MPC formulations' configuration packages under robot_models/<robot>/. */
constexpr char kCentroidalPackageSuffix[] = "_centroidal_mpc";
constexpr char kWholeBodyPackageSuffix[] = "_wb_mpc";

/** The block's field, spelled out here rather than taken from the schema so that the file and the schema are compared. */
constexpr char kBlockKey[] = "locomotion_heuristics";

/** The three name lists of the block and the kind each one holds, as the task-file format spells them. */
struct ListKey {
  const char* absl_nonnull key;
  HeuristicKind kind;
};
constexpr ListKey kListKeys[] = {
    {.key = "base_pose", .kind = HeuristicKind::kBasePose},
    {.key = "foothold", .kind = HeuristicKind::kFoothold},
    {.key = "wrench", .kind = HeuristicKind::kWrench},
};

const ListKey* absl_nullable findListKey(absl::string_view key) {
  for (const ListKey& list : kListKeys) {
    if (key == list.key) return &list;
  }
  return nullptr;
}

/** The directories the main repository's runfiles may be rooted at, most specific first. */
std::vector<std::filesystem::path> runfilesRoots() {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  // `bazel test` also runs the binary from the main repository's runfiles directory.
  roots.emplace_back(std::filesystem::current_path());
  return roots;
}

/** The absolute path of a data file of this test, or empty when the runfiles do not contain it. */
std::string runfilePath(absl::string_view relativePath) {
  for (const std::filesystem::path& root : runfilesRoots()) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/**
 * Every `robot_models/<robot>/<package>/config/mpc/task.textproto` in the runfiles whose package name ends in
 * `packageSuffix`, as a path relative to the runfiles root, sorted. Found by walking the directories rather than by
 * looking up known names, so that a package this test's `data` pulls in but its lists do not name is still seen. The
 * walk stops at the first root that has a `robot_models` directory.
 */
std::vector<std::string> taskFilesInRunfiles(absl::string_view packageSuffix) {
  std::vector<std::string> found;
  for (const std::filesystem::path& root : runfilesRoots()) {
    const std::filesystem::path robotModels = root / "robot_models";
    if (!std::filesystem::is_directory(robotModels)) continue;
    for (const std::filesystem::directory_entry& robot : std::filesystem::directory_iterator(robotModels)) {
      if (!robot.is_directory()) continue;
      const std::string robotName = robot.path().filename().string();
      for (const std::filesystem::directory_entry& package : std::filesystem::directory_iterator(robot.path())) {
        const std::string packageName = package.path().filename().string();
        if (!package.is_directory() || !absl::EndsWith(packageName, packageSuffix)) continue;
        // Built by hand rather than with std::filesystem::relative, which resolves the runfiles' symlinks into the
        // source tree and would then no longer be relative to the runfiles root.
        const std::string relativePath = absl::StrCat("robot_models/", robotName, "/", packageName, "/config/mpc/task.textproto");
        if (std::filesystem::exists(root / relativePath)) found.push_back(relativePath);
      }
    }
    break;
  }
  std::sort(found.begin(), found.end());
  return found;
}

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string writeTempFile(const std::string& name, const std::string& content) {
  const std::string file = (std::filesystem::path(testing::TempDir()) / name).string();
  std::ofstream out(file);
  out << content;
  return file;
}

std::vector<std::string> splitLines(const std::string& text) {
  return absl::StrSplit(text, '\n');
}

std::vector<std::string> sorted(std::vector<std::string> names) {
  std::sort(names.begin(), names.end());
  return names;
}

std::string describe(const std::vector<std::string>& names) {
  return names.empty() ? "(none)" : absl::StrJoin(names, ", ");
}

/** A `# <list>: "<name>"` line of the block: what the operator is told to uncomment. */
struct CommentedCandidate {
  const ListKey* absl_nonnull list;
  std::string name;
  size_t lineIndex;
};

/** The half-open range [begin, end) of line indices. */
struct LineRange {
  size_t begin;
  size_t end;
};

/**
 * The lines INSIDE the block, found textually: from the line after the top-level `locomotion_heuristics {` up to, not
 * including, the line that closes it, the next `}` at the start of a line. Empty when the file has no block. Every
 * textual scan of the block goes through this, so a same-named field of another top-level block
 * (`model_settings`, `contacts`) is never mistaken for one of the block's own.
 */
LineRange blockLineRange(const std::vector<std::string>& lines) {
  const std::regex blockStart(absl::StrCat("^", kBlockKey, R"(\s*\{\s*(#.*)?$)"));
  const std::regex blockEnd("^\\}");
  size_t begin = 0;
  while (begin < lines.size() && !std::regex_search(lines[begin], blockStart)) ++begin;
  if (begin == lines.size()) return LineRange{.begin = lines.size(), .end = lines.size()};
  ++begin;
  size_t end = begin;
  while (end < lines.size() && !std::regex_search(lines[end], blockEnd)) ++end;
  return LineRange{.begin = begin, .end = end};
}

/** Every commented list entry of the block, found TEXTUALLY, because a comment is exactly what the parser throws away. */
std::vector<CommentedCandidate> commentedListCandidates(const std::vector<std::string>& lines) {
  const std::regex commentedEntry(R"re(^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"([A-Za-z_][A-Za-z0-9_]*)")re");
  std::vector<CommentedCandidate> candidates;
  const LineRange block = blockLineRange(lines);
  for (size_t i = block.begin; i < block.end; ++i) {
    std::smatch match;
    if (!std::regex_search(lines[i], match, commentedEntry)) continue;
    const ListKey* absl_nullable list = findListKey(match[1].str());
    if (list != nullptr) candidates.push_back(CommentedCandidate{.list = list, .name = match[2].str(), .lineIndex = i});
  }
  return candidates;
}

/** The file with one line uncommented the way an operator does it: the `#` and the space after it removed, in place. */
std::string withLineUncommented(std::vector<std::string> lines, size_t lineIndex) {
  lines[lineIndex] = std::regex_replace(lines[lineIndex], std::regex("^(\\s*)# ?"), "$1", std::regex_constants::format_first_only);
  return absl::StrJoin(lines, "\n");
}

/** Whether any line of the text opens a live `locomotion_heuristics` block, at any indentation; comments do not count. */
bool hasLiveBlockKey(const std::string& text) {
  const std::regex liveKey(absl::StrCat("^[ \\t]*", kBlockKey, "[ \\t]*[:{]"));
  for (const std::string& line : splitLines(text)) {
    if (std::regex_search(line, liveKey)) return true;
  }
  return false;
}

/** The heuristics of the task file at `path`, as the MPC reads them: loadTaskFile(), then their conversion. */
absl::StatusOr<LocomotionHeuristicConfig> heuristicsOfFile(const std::string& path) {
  absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(path);
  if (!task.ok()) return task.status();
  return locomotionHeuristicConfigFromConfig(task->locomotion_heuristics);
}

class ShippedLocomotionHeuristicBlockTest : public ::testing::TestWithParam<ShippedTaskFile> {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath(GetParam().relativePath);
    ASSERT_FALSE(taskFile_.empty()) << GetParam().relativePath
                                    << " is not in the runfiles; add its *_centroidal_mpc filegroup to this test's `data`.";
  }

  /** A file name for a derived copy of this robot's task file, unique per robot so the parameterized cases do not collide. */
  std::string tempName(absl::string_view suffix) const { return absl::StrCat("shipped_", GetParam().robot, "_", suffix, ".textproto"); }

  std::string taskFile_;
};

std::string robotName(const ::testing::TestParamInfo<ShippedTaskFile>& info) {
  return info.param.robot;
}

// The cases stay in the anonymous namespace with their parameter type, which GCC otherwise warns about
// (-Wsubobject-linkage): gtest's generated factory would hold an internal-linkage type in an external-linkage class.

TEST_P(ShippedLocomotionHeuristicBlockTest, LoadsValidatesAndListsNoHeuristic) {
  // Every heuristic changes the closed loop and none has been validated in simulation on these robots, so every robot
  // ships with all three lists empty.
  //
  // Positive control first: the block is actually in the file. Without it the conversion returns the default
  // configuration, whose lists are also empty, so every assertion below would pass on a file that had lost its block.
  // (UncommentingAnyOneCandidateLoadsWithExactlyThatName is the other half of the control: in this same file the
  // conversion does see a list entry when there is one.)
  const absl::StatusOr<humanoid_mpc_config::TaskFile> parsed = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(taskFile_);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  ASSERT_TRUE(parsed->has_locomotion_heuristics()) << taskFile_ << " has no `" << kBlockKey << "` block.";

  const absl::StatusOr<LocomotionHeuristicConfig> config = heuristicsOfFile(taskFile_);
  ASSERT_TRUE(config.ok()) << taskFile_ << ": " << config.status().message();
  const absl::Status validation = config->validate();
  EXPECT_TRUE(validation.ok()) << taskFile_ << ": " << validation.message();
  for (const ListKey& list : kListKeys) {
    EXPECT_TRUE(config->formulation.list(list.kind).empty())
        << kBlockKey << "." << list.key << " in " << taskFile_ << " lists " << describe(config->formulation.list(list.kind))
        << ". Every shipped list stays empty until the heuristic has been validated in simulation on this robot.";
  }
  EXPECT_TRUE(config->formulation.empty());
}

TEST_P(ShippedLocomotionHeuristicBlockTest, UncommentingAnyOneCandidateLoadsWithExactlyThatName) {
  // The opt-in step the README and every task file give is "uncomment one name". This test makes the documented edit,
  // line by line, for every candidate in every robot's file, and reads the result as the MPC does.
  const std::string text = readFile(taskFile_);
  ASSERT_FALSE(text.empty()) << taskFile_;
  const std::vector<std::string> lines = splitLines(text);
  const std::vector<CommentedCandidate> candidates = commentedListCandidates(lines);

  // Every registered heuristic is offered, under its own list. This is also what keeps the loop below from passing
  // vacuously: a file whose candidates the scan could not find fails here rather than uncommenting nothing.
  for (const ListKey& list : kListKeys) {
    std::vector<std::string> offered;
    for (const CommentedCandidate& candidate : candidates) {
      if (candidate.list == &list) offered.push_back(candidate.name);
    }
    EXPECT_EQ(sorted(offered), sorted(knownHeuristicNames(list.kind)))
        << "the names commented under " << kBlockKey << "." << list.key << " in " << taskFile_ << " are not the registered " << list.key
        << " heuristics; the operator can only turn on what the file offers.";
  }

  for (const CommentedCandidate& candidate : candidates) {
    const std::string edited = withLineUncommented(lines, candidate.lineIndex);
    ASSERT_NE(edited, text) << "uncommenting line " << candidate.lineIndex + 1 << " changed nothing";
    const std::string file = writeTempFile(tempName(absl::StrCat("uncommented_", candidate.name)), edited);
    const absl::StatusOr<LocomotionHeuristicConfig> config = heuristicsOfFile(file);
    if (!config.ok()) {
      ADD_FAILURE() << "uncommenting `" << lines[candidate.lineIndex] << "` (line " << candidate.lineIndex + 1 << " of " << taskFile_
                    << ") gives a file that does not load: " << config.status().message();
      continue;
    }
    for (const ListKey& list : kListKeys) {
      const std::vector<std::string> expected =
          (candidate.list == &list) ? std::vector<std::string>{candidate.name} : std::vector<std::string>{};
      EXPECT_EQ(config->formulation.list(list.kind), expected)
          << "after uncommenting `" << candidate.name << "` under " << kBlockKey << "." << candidate.list->key << " in " << taskFile_
          << ", " << kBlockKey << "." << list.key << " should list " << describe(expected);
    }
  }
}

TEST_P(ShippedLocomotionHeuristicBlockTest, EveryParameterOfTheSchemaIsWrittenInTheFile) {
  // A parameter the file leaves out runs on the schema's default, which nobody chose for this robot, and the file no
  // longer documents it. The schema's LINT directive asks for every new parameter to be written in every robot's
  // file; this is what enforces it. The parser has already refused anything the schema does not know.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> parsed = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(taskFile_);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  const google::protobuf::Message& block = parsed->locomotion_heuristics();
  const google::protobuf::Reflection* absl_nonnull reflection = block.GetReflection();
  const google::protobuf::Descriptor* absl_nonnull descriptor = block.GetDescriptor();
  size_t parameters = 0;
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* absl_nonnull heuristic = descriptor->field(i);
    if (heuristic->message_type() == nullptr) continue;  // a name list, which ships empty
    if (!reflection->HasField(block, heuristic)) {
      ADD_FAILURE() << kBlockKey << "." << heuristic->name() << " is not written in " << taskFile_;
      continue;
    }
    const google::protobuf::Message& parameterBlock = reflection->GetMessage(block, heuristic);
    for (int j = 0; j < parameterBlock.GetDescriptor()->field_count(); ++j) {
      const google::protobuf::FieldDescriptor* absl_nonnull parameter = parameterBlock.GetDescriptor()->field(j);
      EXPECT_TRUE(parameterBlock.GetReflection()->HasField(parameterBlock, parameter))
          << kBlockKey << "." << heuristic->name() << "." << parameter->name() << " is not written in " << taskFile_;
      ++parameters;
    }
  }
  // Positive control: the walk found the parameters, so a schema without blocks cannot pass by having none.
  EXPECT_GT(parameters, 0u);
}

INSTANTIATE_TEST_SUITE_P(ShippedRobots, ShippedLocomotionHeuristicBlockTest, ::testing::ValuesIn(kShippedTaskFiles), robotName);

TEST(ShippedTaskFileScanTest, TheCandidateScanFindsTheBlocksOwnEntriesAndNothingElse) {
  // The scan behind UncommentingAnyOneCandidateLoadsWithExactlyThatName: a commented entry with a trailing comment is
  // still a candidate, a commented field of a parameter block is not, and a list-named field of another top-level
  // block is never mistaken for one of the block's own.
  const std::vector<std::string> lines = {
      "model_settings {",
      "  # foothold: \"capture_point\"",
      "}",
      "locomotion_heuristics {",
      "  # base_pose: \"orientation_compensation\"   # the base-pose family",
      "  # foothold: \"capture_point\"",
      "  # wrench: \"impulse_scaling\"",
      "  capture_point {",
      "    # gain: 1.0",
      "  }",
      "}",
      "other_block {",
      "  # wrench: \"impulse_scaling\"",
      "}",
  };
  const std::vector<CommentedCandidate> candidates = commentedListCandidates(lines);
  ASSERT_EQ(candidates.size(), 3u);
  EXPECT_EQ(candidates[0].name, "orientation_compensation");
  EXPECT_EQ(candidates[0].list->kind, HeuristicKind::kBasePose);
  EXPECT_EQ(candidates[1].name, "capture_point");
  EXPECT_EQ(candidates[1].lineIndex, 5u);
  EXPECT_EQ(candidates[2].name, "impulse_scaling");
  // Uncommented, the line is the entry with its trailing comment, at its indentation.
  const std::vector<std::string> uncommented = splitLines(withLineUncommented(lines, candidates[0].lineIndex));
  ASSERT_EQ(uncommented.size(), lines.size());
  EXPECT_EQ(uncommented[4], "  base_pose: \"orientation_compensation\"   # the base-pose family");
}

TEST(ShippedTaskFileScanTest, TheCentroidalTaskFilesInTheRunfilesAreExactlyTheList) {
  // kShippedTaskFiles and the BUILD target's `data` name the same robots, and SetUp enforces only one direction: a robot
  // in the list but not in `data` fails there. A robot in `data` but not in the list - the likely half-finished
  // onboarding, since `data` is what makes the file reachable - would never be tested and nothing would say so. The
  // runfiles hold exactly what `data` pulls in, so walking them for centroidal task files closes that direction.
  const std::vector<std::string> found = taskFilesInRunfiles(kCentroidalPackageSuffix);
  // Positive control: the walk finds at least the files the list names, so it is looking in the right place and an
  // empty result cannot pass for "nothing extra".
  ASSERT_GE(found.size(), std::size(kShippedTaskFiles)) << "the runfiles walk found only: " << describe(found);

  std::vector<std::string> listed;
  for (const ShippedTaskFile& file : kShippedTaskFiles) listed.emplace_back(file.relativePath);
  EXPECT_EQ(found, sorted(listed)) << "the centroidal task files in this test's runfiles (its BUILD `data`) are not the ones "
                                      "kShippedTaskFiles names; add the robot to kShippedTaskFiles so its block is tested.";
}

TEST(ShippedTaskFileScanTest, TheWholeBodyTaskFilesHaveNoLocomotionHeuristicsBlock) {
  // WBMpcInterface never builds the heuristic layer, so a `locomotion_heuristics` block in a whole-body task file is
  // read by nothing: an operator who uncomments a name there sees no change and no error.
  // Positive control: both checks find the block in a centroidal file of a robot that also has a whole-body file.
  const std::string centroidal = runfilePath("robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto");
  ASSERT_FALSE(centroidal.empty());
  EXPECT_TRUE(hasLiveBlockKey(readFile(centroidal))) << centroidal;
  const absl::StatusOr<humanoid_mpc_config::TaskFile> centroidalTask =
      nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(centroidal);
  ASSERT_TRUE(centroidalTask.ok()) << centroidalTask.status();
  EXPECT_TRUE(centroidalTask->has_locomotion_heuristics()) << centroidal;

  const std::vector<std::string> wholeBody = taskFilesInRunfiles(kWholeBodyPackageSuffix);
  // A failure rather than a skip: Bazel reports a skip as a pass, so a dropped `data` entry would retire this check.
  ASSERT_FALSE(wholeBody.empty()) << "no *" << kWholeBodyPackageSuffix << "/config/mpc/task.textproto in the runfiles; add "
                                  << "//robot_models/unitree_g1/g1_wb_mpc to this test's `data` in "
                                     "humanoid_nmpc/humanoid_common_mpc/BUILD.bazel.";
  for (const std::string& relativePath : wholeBody) {
    const std::string taskFile = runfilePath(relativePath);
    ASSERT_FALSE(taskFile.empty()) << relativePath;
    EXPECT_FALSE(hasLiveBlockKey(readFile(taskFile)))
        << relativePath << " has a `" << kBlockKey << "` block, which the whole-body MPC never reads; remove it.";
    const absl::StatusOr<humanoid_mpc_config::TaskFile> task = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(taskFile);
    ASSERT_TRUE(task.ok()) << task.status();
    EXPECT_FALSE(task->has_locomotion_heuristics()) << relativePath << " sets the `" << kBlockKey << "` block.";
  }
}

}  // namespace
}  // namespace ocs2::humanoid
