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

#include <gtest/gtest.h>

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

#include <yaml-cpp/yaml.h>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"

namespace ocs2::humanoid {

namespace {

/**
 * The `locomotion_heuristics` block of every robot's SHIPPED centroidal task file, read from the files on disk rather
 * than from a string written for the test.
 *
 * Only DRC Atlas's file is ever loaded by the integration suite, because that is the robot every interface-building
 * test uses; the other three blocks carry different values and comments and were never passed through the loader by
 * any Bazel test. So a value validate() rejects, a name uncommented by accident, or a misspelled key in the SA01, G1 or
 * R1 file would surface only when that robot is launched, with CI green. This suite is where those become test
 * failures, and it needs nothing heavier than the loader and yaml-cpp.
 *
 * The walk over the YAML tree is done with yaml-cpp directly, independently of the loader under test: the point is to
 * compare what the FILE says with what the LOADER makes of it.
 */
struct ShippedTaskFile {
  const char* robot;
  const char* relativePath;  // under the runfiles root, i.e. the repository root
};

void PrintTo(const ShippedTaskFile& file, std::ostream* os) {
  *os << file.robot << " (" << file.relativePath << ")";
}

// Every robot that ships a centroidal task file. The BUILD target's `data` has to list the same packages: a robot here
// but not there fails in SetUp, and one there but not here fails TheCentroidalTaskFilesInTheRunfilesAreExactlyTheList.
// The loader's key table names every robot's block, and the parameter deriver knows every robot it can derive for.
// LINT.IfChange(shipped_centroidal_task_files)
constexpr ShippedTaskFile kShippedTaskFiles[] = {
    {"drc_atlas", "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml"},
    {"engineai_sa01", "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml"},
    {"unitree_g1", "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml"},
    {"unitree_r1", "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml"},
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:shipped_centroidal_task_files_data, //humanoid_nmpc/humanoid_common_mpc/src/locomotion_heuristics/LocomotionHeuristicConfig.cpp:locomotion_heuristic_keys, //tools/locomotion_heuristics/derive_parameters.py:derive_parameters_robots)
// clang-format on

/** The package-name suffixes of the two MPC formulations' configuration packages under robot_models/<robot>/. */
constexpr const char* kCentroidalPackageSuffix = "_centroidal_mpc";
constexpr const char* kWholeBodyPackageSuffix = "_wb_mpc";

/** The block's key, spelled out here rather than taken from the header so that the file and the loader are compared. */
constexpr const char* kBlockKey = "locomotion_heuristics";

/** The three list keys of the block and the kind each one holds, as the task-file format spells them. */
struct ListKey {
  const char* key;
  HeuristicKind kind;
};
constexpr ListKey kListKeys[] = {
    {"base_pose", HeuristicKind::BASE_POSE},
    {"foothold", HeuristicKind::FOOTHOLD},
    {"wrench", HeuristicKind::WRENCH},
};

const ListKey* findListKey(absl::string_view key) {
  for (const ListKey& list : kListKeys) {
    if (key == list.key) return &list;
  }
  return nullptr;
}

/**
 * Each coefficient key the loader reads and the struct member it must land in, written out independently of the
 * loader's own table so that a key wired to the wrong member is a failure rather than a second copy of the same
 * mistake. EachShippedCoefficientMovesItsOwnMemberAndNoOther fails, naming the key, when this table and
 * locomotionHeuristicCoefficientKeys() disagree.
 */
using MemberReader = scalar_t (*)(const LocomotionHeuristicConfig&);
struct CoefficientMember {
  const char* key;
  MemberReader read;
};

#define COEFFICIENT_MEMBER(key, field)                                                    \
  CoefficientMember {                                                                     \
    key, [](const LocomotionHeuristicConfig& config) -> scalar_t { return config.field; } \
  }

constexpr CoefficientMember kCoefficientMembers[] = {
    COEFFICIENT_MEMBER("orientation_compensation.rollPerLateralVelocity", orientationCompensation.rollPerLateralVelocity),
    COEFFICIENT_MEMBER("orientation_compensation.rollOffset", orientationCompensation.rollOffset),
    COEFFICIENT_MEMBER("orientation_compensation.pitchPerForwardVelocity", orientationCompensation.pitchPerForwardVelocity),
    COEFFICIENT_MEMBER("orientation_compensation.pitchOffset", orientationCompensation.pitchOffset),
    COEFFICIENT_MEMBER("orientation_compensation.maximumTilt", orientationCompensation.maximumTilt),
    COEFFICIENT_MEMBER("periodic_orientation.rollAmplitude", periodicOrientation.rollAmplitude),
    COEFFICIENT_MEMBER("periodic_orientation.rollPhaseRate", periodicOrientation.rollPhaseRate),
    COEFFICIENT_MEMBER("periodic_orientation.rollPhaseOffset", periodicOrientation.rollPhaseOffset),
    COEFFICIENT_MEMBER("periodic_orientation.pitchAmplitude", periodicOrientation.pitchAmplitude),
    COEFFICIENT_MEMBER("periodic_orientation.pitchPhaseRate", periodicOrientation.pitchPhaseRate),
    COEFFICIENT_MEMBER("periodic_orientation.pitchPhaseOffset", periodicOrientation.pitchPhaseOffset),
    COEFFICIENT_MEMBER("height_compensation.heightPerSpeedSquared", heightCompensation.heightPerSpeedSquared),
    COEFFICIENT_MEMBER("height_compensation.heightPerSpeed", heightCompensation.heightPerSpeed),
    COEFFICIENT_MEMBER("height_compensation.heightOffset", heightCompensation.heightOffset),
    COEFFICIENT_MEMBER("height_compensation.maximumHeightOffset", heightCompensation.maximumHeightOffset),
    COEFFICIENT_MEMBER("hip_centered_stepping.lateralScale", hipCenteredStepping.lateralScale),
    COEFFICIENT_MEMBER("hip_centered_stepping.longitudinalScale", hipCenteredStepping.longitudinalScale),
    COEFFICIENT_MEMBER("capture_point.gain", capturePoint.gain),
    COEFFICIENT_MEMBER("capture_point.comHeightOverride", capturePoint.comHeightOverride),
    COEFFICIENT_MEMBER("capture_point.gravity", capturePoint.gravity),
    COEFFICIENT_MEMBER("capture_point.maximumOffset", capturePoint.maximumOffset),
    COEFFICIENT_MEMBER("translational_stepping.forwardPerForwardVelocity", translationalStepping.forwardPerForwardVelocity),
    COEFFICIENT_MEMBER("translational_stepping.forwardStanceFraction", translationalStepping.forwardStanceFraction),
    COEFFICIENT_MEMBER("translational_stepping.forwardOffset", translationalStepping.forwardOffset),
    COEFFICIENT_MEMBER("translational_stepping.lateralPerLateralVelocity", translationalStepping.lateralPerLateralVelocity),
    COEFFICIENT_MEMBER("translational_stepping.lateralStanceFraction", translationalStepping.lateralStanceFraction),
    COEFFICIENT_MEMBER("translational_stepping.lateralOffset", translationalStepping.lateralOffset),
    COEFFICIENT_MEMBER("in_place_turning.forwardPerYawRate", inPlaceTurning.forwardPerYawRate),
    COEFFICIENT_MEMBER("in_place_turning.forwardStanceLever", inPlaceTurning.forwardStanceLever),
    COEFFICIENT_MEMBER("in_place_turning.forwardOffset", inPlaceTurning.forwardOffset),
    COEFFICIENT_MEMBER("in_place_turning.lateralPerYawRate", inPlaceTurning.lateralPerYawRate),
    COEFFICIENT_MEMBER("in_place_turning.lateralOffset", inPlaceTurning.lateralOffset),
    COEFFICIENT_MEMBER("high_speed_turning.forwardPerCrossTerm", highSpeedTurning.forwardPerCrossTerm),
    COEFFICIENT_MEMBER("high_speed_turning.forwardOffset", highSpeedTurning.forwardOffset),
    COEFFICIENT_MEMBER("high_speed_turning.lateralPerCrossTerm", highSpeedTurning.lateralPerCrossTerm),
    COEFFICIENT_MEMBER("high_speed_turning.lateralOffset", highSpeedTurning.lateralOffset),
    COEFFICIENT_MEMBER("impulse_scaling.scale", impulseScaling.scale),
    COEFFICIENT_MEMBER("impulse_scaling.minimumDutyFactor", impulseScaling.minimumDutyFactor),
    COEFFICIENT_MEMBER("impulse_scaling.maximumForceRatio", impulseScaling.maximumForceRatio),
    COEFFICIENT_MEMBER("centripetal_acceleration.scale", centripetalAcceleration.scale),
    COEFFICIENT_MEMBER("centripetal_acceleration.maximumForce", centripetalAcceleration.maximumForce),
    COEFFICIENT_MEMBER("centripetal_acceleration.maximumForceRatioOfWeight", centripetalAcceleration.maximumForceRatioOfWeight),
};

#undef COEFFICIENT_MEMBER

const CoefficientMember* findCoefficientMember(absl::string_view key) {
  for (const CoefficientMember& member : kCoefficientMembers) {
    if (key == member.key) return &member;
  }
  return nullptr;
}

/** The directories the main repository's runfiles may be rooted at, most specific first. */
std::vector<std::filesystem::path> runfilesRoots() {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
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
 * Every `robot_models/<robot>/<package>/config/mpc/task.yaml` in the runfiles whose package name ends in
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
        const std::string relativePath = absl::StrCat("robot_models/", robotName, "/", packageName, "/config/mpc/task.yaml");
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

/** A `# - <name>` line under one of the three list keys: what the operator is told to uncomment. */
struct CommentedCandidate {
  const ListKey* list;
  std::string name;
  size_t lineIndex;
};

/** The half-open range [begin, end) of line indices. */
struct LineRange {
  size_t begin;
  size_t end;
};

/**
 * The lines INSIDE the block, found textually: from the line after the top-level `locomotion_heuristics:` key up to,
 * not including, the next top-level key. Empty when the file has no block. Every textual scan of the block goes
 * through this, so a same-named key in another top-level block (`model_settings.nominal_foothold` has children too)
 * is never mistaken for one of the block's own.
 */
LineRange blockLineRange(const std::vector<std::string>& lines) {
  const std::regex blockStart(absl::StrCat("^", kBlockKey, ":\\s*(#.*)?$"));
  const std::regex topLevelKey("^[A-Za-z_]");
  size_t begin = 0;
  while (begin < lines.size() && !std::regex_search(lines[begin], blockStart)) ++begin;
  if (begin == lines.size()) return LineRange{lines.size(), lines.size()};
  ++begin;
  size_t end = begin;
  while (end < lines.size() && !std::regex_search(lines[end], topLevelKey)) ++end;
  return LineRange{begin, end};
}

/**
 * Every commented list entry of the block, found TEXTUALLY, because a comment is exactly what a YAML parser throws away.
 *
 * Inside the block, a two-space indented `<key>:` line opens a child - a list when it is one of the three list keys, a
 * parameter block otherwise - and only the commented entries under a list key are candidates. The
 * `# ---- one parameter block ...` banner and the prose in the parameter blocks are therefore not.
 */
std::vector<CommentedCandidate> commentedListCandidates(const std::vector<std::string>& lines) {
  const std::regex blockChild("^  ([A-Za-z_][A-Za-z0-9_]*):");
  const std::regex commentedEntry("^\\s*#\\s*-\\s+([A-Za-z_][A-Za-z0-9_]*)");
  std::vector<CommentedCandidate> candidates;
  const LineRange block = blockLineRange(lines);
  const ListKey* currentList = nullptr;
  for (size_t i = block.begin; i < block.end; ++i) {
    const std::string& line = lines[i];
    std::smatch match;
    if (std::regex_search(line, match, blockChild)) {
      currentList = findListKey(match[1].str());
      continue;
    }
    if (currentList != nullptr && std::regex_search(line, match, commentedEntry)) {
      candidates.push_back(CommentedCandidate{currentList, match[1].str(), i});
    }
  }
  return candidates;
}

/**
 * Rewrites the block's list keys from the layout they ship in - a bare `  base_pose:`, with or without a trailing
 * comment - to the one they used to ship in, `  base_pose: []`, keeping the comment. Only lines inside the block are
 * touched and only the names in kListKeys are matched. Returns how many keys were rewritten.
 */
size_t rewriteListKeysAsFlowEmpty(std::vector<std::string>* lines) {
  const std::string names = absl::StrJoin(kListKeys, "|", [](std::string* out, const ListKey& list) { absl::StrAppend(out, list.key); });
  const std::regex bareListKey(absl::StrCat("^  (", names, "):\\s*(#.*)?$"));
  const LineRange block = blockLineRange(*lines);
  size_t rewritten = 0;
  for (size_t i = block.begin; i < block.end; ++i) {
    std::string& line = (*lines)[i];
    std::smatch match;
    if (!std::regex_match(line, match, bareListKey)) continue;
    // YAML needs whitespace before a `#` for it to open a comment.
    const std::string comment = match[2].matched ? absl::StrCat(" ", match[2].str()) : std::string();
    line = absl::StrCat("  ", match[1].str(), ": []", comment);
    ++rewritten;
  }
  return rewritten;
}

/** The file with one line uncommented the way an operator does it: the `#` and the space after it removed, in place. */
std::string withLineUncommented(std::vector<std::string> lines, size_t lineIndex) {
  lines[lineIndex] = std::regex_replace(lines[lineIndex], std::regex("^(\\s*)# ?"), "$1", std::regex_constants::format_first_only);
  return absl::StrJoin(lines, "\n");
}

std::string describe(const std::vector<std::string>& names) {
  return names.empty() ? "(none)" : absl::StrJoin(names, ", ");
}

/** The block as yaml-cpp reads it; an undefined node when the file has none. */
YAML::Node heuristicBlock(const std::string& taskFile) {
  const YAML::Node root = YAML::LoadFile(taskFile);
  return root[kBlockKey];
}

/** Whether any line of the text is a live `locomotion_heuristics:` key, at any indentation; comments do not count. */
bool hasLiveBlockKey(const std::string& text) {
  const std::regex liveKey(absl::StrCat("^[ \\t]*", kBlockKey, "[ \\t]*:"));
  for (const std::string& line : splitLines(text)) {
    if (std::regex_search(line, liveKey)) return true;
  }
  return false;
}

/** "<heuristic>.<key>" for every child of every parameter block of the block, i.e. of every child that is not a list. */
struct BlockLeaf {
  std::string key;
  YAML::Node value;
};

std::vector<BlockLeaf> parameterLeaves(const YAML::Node& block) {
  std::vector<BlockLeaf> leaves;
  for (YAML::const_iterator child = block.begin(); child != block.end(); ++child) {
    const std::string name = child->first.as<std::string>();
    if (findListKey(name) != nullptr || !child->second.IsMap()) continue;
    for (YAML::const_iterator leaf = child->second.begin(); leaf != child->second.end(); ++leaf) {
      leaves.push_back(BlockLeaf{absl::StrCat(name, ".", leaf->first.as<std::string>()), leaf->second});
    }
  }
  return leaves;
}

class ShippedLocomotionHeuristicBlockTest : public ::testing::TestWithParam<ShippedTaskFile> {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath(GetParam().relativePath);
    ASSERT_FALSE(taskFile_.empty()) << GetParam().relativePath
                                    << " is not in the runfiles; add its *_centroidal_mpc filegroup to this test's `data`.";
  }

  /** A file name for a derived copy of this robot's task file, unique per robot so the parameterized cases do not collide. */
  std::string tempName(absl::string_view suffix) const { return absl::StrCat("shipped_", GetParam().robot, "_", suffix, ".yaml"); }

  std::string taskFile_;
};

std::string robotName(const ::testing::TestParamInfo<ShippedTaskFile>& info) {
  return info.param.robot;
}

// The cases stay in the anonymous namespace with their parameter type, which GCC otherwise warns about
// (-Wsubobject-linkage): gtest's generated factory would hold an internal-linkage type in an external-linkage class.

TEST_P(ShippedLocomotionHeuristicBlockTest, LoadsValidatesAndListsNoHeuristic) {
  // Every heuristic changes the closed loop and none has been validated in simulation on these robots, so every robot
  // ships with all three lists empty. Only Atlas's file was ever checked by a Bazel test; this is the check for all four.
  //
  // Positive control first: the block is actually in the file. Without it the loader returns the default
  // configuration, whose lists are also empty, so every assertion below would pass on a file that had lost its block.
  // (UncommentingAnyOneCandidateLoadsWithExactlyThatName is the other half of the control: in this same file the loader
  // does see a list entry when there is one.)
  const YAML::Node block = heuristicBlock(taskFile_);
  ASSERT_TRUE(block.IsDefined() && block.IsMap()) << taskFile_ << " has no `" << kBlockKey << "` map at its root.";

  // Each list ships as a BARE key - a YAML null - with its names commented beneath it. A flow `[]` is also empty but
  // turns into invalid YAML the moment a name under it is uncommented, which is what the files tell the operator to do.
  for (const ListKey& list : kListKeys) {
    const YAML::Node value = block[list.key];
    if (!value.IsDefined()) {
      ADD_FAILURE() << kBlockKey << "." << list.key << " is missing from " << taskFile_
                    << "; ship it as a bare key with its candidate names commented beneath it.";
      continue;
    }
    EXPECT_TRUE(value.IsNull()) << kBlockKey << "." << list.key << " in " << taskFile_
                                << " must ship as a bare key (a YAML null), so that uncommenting one name beneath it yields a valid "
                                   "list; it is not null (a flow `[]`, or a name left uncommented).";
  }

  const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(taskFile_);
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
  // The opt-in step the README and every task file give is "uncomment one name". When the lists shipped as `[]` that
  // edit produced a file yaml-cpp rejects, so following the documented step stopped the controller from starting; the
  // integration test never noticed because it swapped the whole list region out with a regex instead of uncommenting.
  // This test makes the documented edit, line by line, for every candidate in every robot's file.
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
    const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
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

TEST_P(ShippedLocomotionHeuristicBlockTest, TheUncommentCheckRejectsTheFlowEmptyListLayout) {
  // Negative control for the test above: with the three list keys written the way they used to ship, `base_pose: []`,
  // the same uncomment edit must produce a file that does NOT load. Otherwise the check above could not tell the
  // layout that works from the one that silently broke the documented opt-in step.
  const std::string text = readFile(taskFile_);
  std::vector<std::string> lines = splitLines(text);
  // A bare key the file annotates with a trailing comment is still a bare key, and is rewritten with its comment kept.
  // Whether each key IS bare is LoadsValidatesAndListsNoHeuristic's to report; this only needs to find all of them.
  ASSERT_EQ(rewriteListKeysAsFlowEmpty(&lines), std::size(kListKeys))
      << "expected each list key of the `" << kBlockKey << "` block of " << taskFile_ << " as a bare key";

  // The flow layout on its own is valid and empty; it is only the uncommented entry beneath it that is not YAML.
  const std::string flowFile = writeTempFile(tempName("flow_lists"), absl::StrJoin(lines, "\n"));
  const absl::StatusOr<LocomotionHeuristicConfig> flowConfig = loadLocomotionHeuristicConfig(flowFile);
  ASSERT_TRUE(flowConfig.ok()) << flowConfig.status().message();
  EXPECT_TRUE(flowConfig->formulation.empty());

  // One candidate under each list: the defect is the layout of every list key, not of one of them.
  const std::vector<CommentedCandidate> candidates = commentedListCandidates(lines);
  for (const ListKey& list : kListKeys) {
    const std::vector<CommentedCandidate>::const_iterator first = std::find_if(
        candidates.begin(), candidates.end(), [&list](const CommentedCandidate& candidate) { return candidate.list == &list; });
    if (first == candidates.end()) {
      ADD_FAILURE() << "no commented candidate under " << kBlockKey << "." << list.key << " in " << taskFile_;
      continue;
    }
    const std::string broken =
        writeTempFile(tempName(absl::StrCat("flow_lists_uncommented_", first->name)), withLineUncommented(lines, first->lineIndex));
    EXPECT_FALSE(loadLocomotionHeuristicConfig(broken).ok()) << "uncommenting `" << first->name << "` beneath a flow `" << list.key
                                                             << ": []` loaded, so the uncomment check cannot see the defect";
  }
}

TEST_P(ShippedLocomotionHeuristicBlockTest, TheParameterBlocksAreExactlyTheRegisteredHeuristics) {
  // One parameter block per registered heuristic, keyed by its registry name - the loader reads `<name>.<key>` and the
  // tuning GUI renders the tree it finds. A block with a misspelled name is read by nothing, and a registered heuristic
  // with no block cannot be tuned from the file. Any child of the block that is neither a list key nor a parameter
  // block is, likewise, something the loader never reads.
  const YAML::Node block = heuristicBlock(taskFile_);
  ASSERT_TRUE(block.IsDefined() && block.IsMap()) << taskFile_;

  std::vector<std::string> parameterBlocks;
  for (YAML::const_iterator child = block.begin(); child != block.end(); ++child) {
    const std::string name = child->first.as<std::string>();
    if (findListKey(name) != nullptr) continue;
    EXPECT_TRUE(child->second.IsMap()) << kBlockKey << "." << name << " in " << taskFile_
                                       << " is neither a list key nor a parameter block, so the loader reads nothing from it.";
    parameterBlocks.push_back(name);
  }

  std::vector<std::string> registered;
  for (HeuristicKind kind : allHeuristicKinds()) {
    const std::vector<std::string>& names = knownHeuristicNames(kind);
    registered.insert(registered.end(), names.begin(), names.end());
  }
  ASSERT_FALSE(registered.empty());
  EXPECT_EQ(sorted(parameterBlocks), sorted(registered)) << "the parameter blocks of " << taskFile_ << " are not the registered heuristics";
}

TEST_P(ShippedLocomotionHeuristicBlockTest, EveryParameterLeafIsANumberTheLoaderReads) {
  // The loader, like every loader here, ignores a key it does not know: `pitchPerForwardVelocty: 0.04` in a task file
  // leaves the coefficient at its default of 0 and nothing says so. So every leaf of every parameter block has to be
  // one of the keys the loader reads - and a number, because the loader reads only numbers.
  const YAML::Node block = heuristicBlock(taskFile_);
  ASSERT_TRUE(block.IsDefined() && block.IsMap()) << taskFile_;
  const std::vector<std::string> loaderKeys = locomotionHeuristicCoefficientKeys();
  const absl::flat_hash_set<std::string> readByLoader(loaderKeys.begin(), loaderKeys.end());

  const std::vector<BlockLeaf> leaves = parameterLeaves(block);
  // Positive control: the walk found the leaves, so an empty or restructured block cannot pass by having none.
  ASSERT_FALSE(leaves.empty()) << "no parameter leaves found in " << taskFile_;
  for (const BlockLeaf& leaf : leaves) {
    EXPECT_TRUE(readByLoader.contains(leaf.key))
        << kBlockKey << "." << leaf.key << " in " << taskFile_ << " is not a key the loader reads, so its value is silently ignored.";
    double value = 0.0;
    EXPECT_TRUE(leaf.value.IsScalar() && YAML::convert<double>::decode(leaf.value, value))
        << kBlockKey << "." << leaf.key << " in " << taskFile_ << " is not a number.";
  }
}

TEST_P(ShippedLocomotionHeuristicBlockTest, EveryCoefficientTheLoaderReadsIsWrittenInTheFile) {
  // The converse: a key the loader reads but the file does not carry keeps its struct default silently, and the
  // tuning GUI, which renders what it finds in the file, has no slider for it. The loader's LINT directive asks for
  // every new key to be documented in every robot's file; this is what enforces it.
  const YAML::Node block = heuristicBlock(taskFile_);
  ASSERT_TRUE(block.IsDefined() && block.IsMap()) << taskFile_;
  const std::vector<std::string> loaderKeys = locomotionHeuristicCoefficientKeys();
  ASSERT_FALSE(loaderKeys.empty());

  absl::flat_hash_set<std::string> written;
  for (const BlockLeaf& leaf : parameterLeaves(block)) written.insert(leaf.key);
  for (const std::string& key : loaderKeys) {
    EXPECT_TRUE(written.contains(key)) << kBlockKey << "." << key << " is read by the loader but not written in " << taskFile_;
  }
}

TEST_P(ShippedLocomotionHeuristicBlockTest, EachShippedCoefficientMovesItsOwnMemberAndNoOther) {
  // What the file says is what the controller runs with. Two ways that can fail without any key being unknown: the
  // loader wires a key to the wrong member, or two keys to the same one. Comparing each loaded member with its leaf
  // catches the first only where the two members happen to ship different values - most coefficients ship at 0.0 -
  // so each leaf is also nudged ON ITS OWN and the reload must move that member, by exactly the nudge, and nothing else.

  // This test's own table has to cover the loader's keys, or the checks below would skip the ones it lacks.
  const std::vector<std::string> loaderKeys = locomotionHeuristicCoefficientKeys();
  ASSERT_FALSE(loaderKeys.empty());
  for (const std::string& key : loaderKeys) {
    EXPECT_NE(findCoefficientMember(key), nullptr) << "the loader reads " << key << " but kCoefficientMembers has no entry for it";
  }
  const absl::flat_hash_set<std::string> readByLoader(loaderKeys.begin(), loaderKeys.end());
  for (const CoefficientMember& member : kCoefficientMembers) {
    EXPECT_TRUE(readByLoader.contains(member.key)) << member.key << " is in kCoefficientMembers but the loader does not read it";
  }

  const YAML::Node block = heuristicBlock(taskFile_);
  ASSERT_TRUE(block.IsDefined() && block.IsMap()) << taskFile_;
  const absl::StatusOr<LocomotionHeuristicConfig> shipped = loadLocomotionHeuristicConfig(taskFile_);
  ASSERT_TRUE(shipped.ok()) << shipped.status().message();

  const std::vector<BlockLeaf> leaves = parameterLeaves(block);
  ASSERT_FALSE(leaves.empty()) << taskFile_;
  size_t nudged = 0;
  for (const BlockLeaf& leaf : leaves) {
    const CoefficientMember* member = findCoefficientMember(leaf.key);
    double fileValue = 0.0;
    // An unknown or non-numeric leaf is EveryParameterLeafIsANumberTheLoaderReads's failure to report.
    if (member == nullptr || !YAML::convert<double>::decode(leaf.value, fileValue)) continue;
    EXPECT_DOUBLE_EQ(member->read(*shipped), fileValue) << kBlockKey << "." << leaf.key << " in " << taskFile_;

    // The nudge goes up unless that leaves the key's admissible range (minimumDutyFactor lies in (0, 1]), then down.
    const std::vector<std::string> path = absl::StrSplit(leaf.key, absl::MaxSplits('.', /*limit=*/1));
    ASSERT_EQ(path.size(), 2u) << leaf.key;
    bool loaded = false;
    std::string failures;
    for (const scalar_t nudge : {1e-3, -1e-3}) {
      const scalar_t nudgedValue = fileValue + nudge;
      YAML::Node document;
      document[kBlockKey] = YAML::Clone(block);
      document[kBlockKey][path[0]][path[1]] = nudgedValue;
      YAML::Emitter emitter;
      emitter << document;
      const std::string file = writeTempFile(tempName(absl::StrCat("nudged_", leaf.key)), emitter.c_str());
      const absl::StatusOr<LocomotionHeuristicConfig> config = loadLocomotionHeuristicConfig(file);
      if (!config.ok()) {
        absl::StrAppend(&failures, "\n  ", nudgedValue, ": ", config.status().message());
        continue;
      }
      loaded = true;
      ++nudged;
      EXPECT_DOUBLE_EQ(member->read(*config), nudgedValue) << "nudging " << kBlockKey << "." << leaf.key << " did not reach its member";
      for (const CoefficientMember& other : kCoefficientMembers) {
        if (&other == member) continue;
        EXPECT_DOUBLE_EQ(other.read(*config), other.read(*shipped))
            << "nudging " << kBlockKey << "." << leaf.key << " also moved " << other.key << ", so the two share a member";
      }
      break;
    }
    EXPECT_TRUE(loaded) << "neither nudge of " << kBlockKey << "." << leaf.key << " in " << taskFile_ << " loaded:" << failures;
  }
  EXPECT_EQ(nudged, loaderKeys.size()) << "not every coefficient the loader reads was nudged in " << taskFile_;
}

INSTANTIATE_TEST_SUITE_P(ShippedRobots, ShippedLocomotionHeuristicBlockTest, ::testing::ValuesIn(kShippedTaskFiles), robotName);

TEST(ShippedTaskFileScanTest, TheFlowLayoutRewriteFindsCommentedBareKeysAndStaysInsideTheBlock) {
  // The rewrite behind TheUncommentCheckRejectsTheFlowEmptyListLayout used to match `^  (base_pose|foothold|wrench):\s*$`
  // over the whole file. So a bare list key with a trailing comment - still a YAML null, still a valid layout - was not
  // found and the test aborted on a correct file, and a two-space `foothold:` in another top-level block was counted
  // as one of the block's own. This text has both, and the old scan happens to count three here as well, so only the
  // exact lines tell the two apart.
  std::vector<std::string> lines = {
      "model_settings:",
      "  foothold:",
      "    stepWidth: 0.1",
      "locomotion_heuristics:",
      "  base_pose:   # the base-pose family",
      "    # - orientation_compensation",
      "  foothold:",
      "    # - capture_point",
      "  wrench: # the wrench family",
      "    # - impulse_scaling",
      "  capture_point:",
      "    gain: 1.0",
      "other_block:",
      "  wrench:",
  };
  const std::vector<std::string> original = lines;

  // Positive control for the scan the rewrite shares: the block's own candidates, and only those, are found.
  const std::vector<CommentedCandidate> candidates = commentedListCandidates(lines);
  ASSERT_EQ(candidates.size(), 3u);
  EXPECT_EQ(candidates[0].name, "orientation_compensation");
  EXPECT_EQ(candidates[1].name, "capture_point");
  EXPECT_EQ(candidates[2].name, "impulse_scaling");

  ASSERT_EQ(rewriteListKeysAsFlowEmpty(&lines), 3u);
  const std::vector<std::string> expected = {
      "model_settings:",
      "  foothold:",
      "    stepWidth: 0.1",
      "locomotion_heuristics:",
      "  base_pose: [] # the base-pose family",
      "    # - orientation_compensation",
      "  foothold: []",
      "    # - capture_point",
      "  wrench: [] # the wrench family",
      "    # - impulse_scaling",
      "  capture_point:",
      "    gain: 1.0",
      "other_block:",
      "  wrench:",
  };
  EXPECT_EQ(lines, expected);

  // The kept comment leaves each list a valid, empty flow sequence, and the keys outside the block untouched.
  const YAML::Node rewritten = YAML::Load(absl::StrJoin(lines, "\n"));
  for (const ListKey& list : kListKeys) {
    const YAML::Node value = rewritten[kBlockKey][list.key];
    EXPECT_TRUE(value.IsSequence() && value.size() == 0) << list.key;
  }
  EXPECT_TRUE(rewritten["model_settings"]["foothold"].IsMap());
  EXPECT_TRUE(rewritten["other_block"]["wrench"].IsNull());
  // And the untouched text parses the way the shipped files do: every list key a null.
  const YAML::Node unrewritten = YAML::Load(absl::StrJoin(original, "\n"));
  for (const ListKey& list : kListKeys) {
    EXPECT_TRUE(unrewritten[kBlockKey][list.key].IsNull()) << list.key;
  }
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
  // read by nothing: an operator who uncomments a name there sees no change and no error. The Python suite checks
  // this, but `make test-all` runs only Bazel. The textual search also catches the key nested under another block,
  // where it would be just as inert.
  // Positive control: both searches find the block in a centroidal file of a robot that also has a whole-body file.
  const std::string centroidal = runfilePath("robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml");
  ASSERT_FALSE(centroidal.empty());
  EXPECT_TRUE(hasLiveBlockKey(readFile(centroidal))) << centroidal;
  EXPECT_TRUE(heuristicBlock(centroidal).IsDefined()) << centroidal;

  const std::vector<std::string> wholeBody = taskFilesInRunfiles(kWholeBodyPackageSuffix);
  // A failure rather than a skip: Bazel reports a skip as a pass, so a dropped `data` entry would retire this check.
  ASSERT_FALSE(wholeBody.empty()) << "no *" << kWholeBodyPackageSuffix << "/config/mpc/task.yaml in the runfiles; add "
                                  << "//robot_models/unitree_g1/g1_wb_mpc to this test's `data` in "
                                     "humanoid_nmpc/humanoid_common_mpc/BUILD.bazel.";
  for (const std::string& relativePath : wholeBody) {
    const std::string taskFile = runfilePath(relativePath);
    ASSERT_FALSE(taskFile.empty()) << relativePath;
    EXPECT_FALSE(hasLiveBlockKey(readFile(taskFile)))
        << relativePath << " has a `" << kBlockKey << ":` key, which the whole-body MPC never reads; remove it.";
    EXPECT_FALSE(heuristicBlock(taskFile).IsDefined())
        << relativePath << " has a `" << kBlockKey << "` block, which the whole-body MPC never reads.";
  }
}

}  // namespace
}  // namespace ocs2::humanoid
