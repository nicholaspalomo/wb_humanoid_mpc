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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "robot_core/ResourcePaths.h"
#include "support/ProblemFingerprint.h"
#include "support/ShippedCentroidalRobot.h"
#include "support/TypedConfigFiles.h"

/**
 * The parity of the runtime contact set (milestone 2 of
 * humanoid_nmpc/docs/multi_contact_planning/README.md): the whole optimal
 * control problem the centroidal MPC assembles for every shipped robot - every
 * term's name, and the value and the first and second derivatives of the cost,
 * the soft constraints, the dynamics and the equality constraints at the probe
 * points of ProblemFingerprint - is bit for bit the problem it assembled
 * before the contact count became a runtime value.
 *
 * The record of each variant is in test/golden/centroidal_contact_set_parity/,
 * one line per fingerprint entry: its label, its size and the FNV-1a hash of
 * the bit patterns of its entries, so that a single flipped bit anywhere fails
 * the test and names the entry. It was recorded from the problem as it was
 * assembled before the change, by this test, whose computed record goes to
 * $TEST_UNDECLARED_OUTPUTS_DIR/<variant>.txt on every run.
 *
 * Beside the shipped task files, three variants reach the terms the shipped
 * files leave out: the DRC Atlas on wrench inputs (the contact wrench cone,
 * which basis-vector inputs replace), the contact-implicit formulation on
 * wrench inputs (the un-gated cones and the complementarity terms), and the
 * Unitree G1 with its stance constraint soft and holding the yaw rate.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kGoldenDirectory[] =
    "humanoid_nmpc/humanoid_centroidal_mpc/test/golden/"
    "centroidal_contact_set_parity/";

/** How a variant's task file differs from the shipped one. */
enum class TaskEdit {
  kShipped,
  kWrenchInputs,
  kContactImplicit,
  kSoftYawHoldingStance,
};

/** A configuration whose problem is recorded. */
struct ParityVariant {
  const char* absl_nonnull name;
  ShippedCentroidalRobot robot;
  TaskEdit edit;
};

void PrintTo(const ParityVariant& variant, std::ostream* absl_nonnull os) {
  *os << variant.name;
}

constexpr ParityVariant kVariants[] = {
    {.name = "drc_atlas",
     .robot = ShippedCentroidalRobot::kDrcAtlas,
     .edit = TaskEdit::kShipped},
    {.name = "engineai_sa01",
     .robot = ShippedCentroidalRobot::kEngineAiSa01,
     .edit = TaskEdit::kShipped},
    {.name = "unitree_g1",
     .robot = ShippedCentroidalRobot::kUnitreeG1,
     .edit = TaskEdit::kShipped},
    {.name = "unitree_r1",
     .robot = ShippedCentroidalRobot::kUnitreeR1,
     .edit = TaskEdit::kShipped},
    {.name = "drc_atlas_wrench_inputs",
     .robot = ShippedCentroidalRobot::kDrcAtlas,
     .edit = TaskEdit::kWrenchInputs},
    {.name = "drc_atlas_contact_implicit",
     .robot = ShippedCentroidalRobot::kDrcAtlas,
     .edit = TaskEdit::kContactImplicit},
    {.name = "unitree_g1_soft_yaw_holding_stance",
     .robot = ShippedCentroidalRobot::kUnitreeG1,
     .edit = TaskEdit::kSoftYawHoldingStance},
};

/** `task` edited as `edit` says. */
void applyEdit(TaskEdit edit, mpc_config::TaskFile& task) {
  switch (edit) {
    case TaskEdit::kShipped:
      return;
    case TaskEdit::kWrenchInputs:
      task.contact_input_parameterization = "wrench";
      return;
    case TaskEdit::kContactImplicit:
      task.contact_input_parameterization = "wrench";
      task.hard_constraints = {};
      task.soft_constraints = {"joint_limits",        "foot_collision",
                               "contact_wrench_cone", "friction_force_cone",
                               "normal_velocity",     "contact_complementarity",
                               "force_weighted_slip", "ground_penetration"};
      task.terrain_height = 0.037;
      return;
    case TaskEdit::kSoftYawHoldingStance:
      task.hard_constraints = {"zero_wrench", "normal_velocity"};
      task.soft_constraints.push_back("zero_velocity");
      task.model_settings.foot_constraint.stance_constraint =
          "position_and_orientation";
      return;
  }
}

/** The 64-bit FNV-1a hash of the bit patterns of `value`'s entries, column
 * by column. */
uint64_t bitPatternHash(const matrix_t& value) {
  uint64_t hash = 14695981039346656037ULL;
  for (Eigen::Index column = 0; column < value.cols(); ++column) {
    for (Eigen::Index row = 0; row < value.rows(); ++row) {
      const double entry = value(row, column);
      unsigned char bytes[sizeof(double)];
      std::memcpy(bytes, &entry, sizeof(double));
      for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
      }
    }
  }
  return hash;
}

/** The record of a fingerprint: one line per entry. */
std::string recordOf(const test::Fingerprint& fingerprint) {
  std::string record;
  for (const std::pair<std::string, matrix_t>& entry : fingerprint) {
    absl::StrAppendFormat(&record, "%s\t%dx%d\t%016x\n", entry.first,
                          entry.second.rows(), entry.second.cols(),
                          bitPatternHash(entry.second));
  }
  return record;
}

/** The contents of `path`, or nullopt when it cannot be read. */
std::optional<std::string> readFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) return std::nullopt;
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

/** Writes `record` to the test's undeclared outputs as `<name>.txt`, and
 * returns where (empty when it cannot). */
std::string writeOutput(absl::string_view name, const std::string& record) {
  const char* absl_nullable outputDir =
      std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (outputDir == nullptr) return std::string();
  const std::string path =
      (std::filesystem::path(outputDir) / absl::StrCat(name, ".txt")).string();
  std::ofstream file(path, std::ios::trunc);
  file << record;
  return file.good() ? path : std::string();
}

class CentroidalContactSetParityTest
    : public ::testing::TestWithParam<ParityVariant> {};

TEST_P(CentroidalContactSetParityTest, theProblemIsBitForBitTheRecordedOne) {
  const ParityVariant& variant = GetParam();
  const CentroidalRobotFiles files = shippedCentroidalRobotFiles(variant.robot);
  absl::StatusOr<CentroidalMpcConfig> config = loadConfigOf(files);
  ASSERT_TRUE(config.ok()) << config.status();
  applyEdit(variant.edit, config->task);
  absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(*config, files.urdfFile);
  ASSERT_TRUE(interface.ok()) << interface.status();

  const std::string record = recordOf(test::fingerprintOf(**interface));
  const std::string output = writeOutput(variant.name, record);

  const absl::StatusOr<std::string> goldenPath = robot::resolveResourcePath(
      absl::StrCat(kGoldenDirectory, variant.name, ".txt"));
  ASSERT_TRUE(goldenPath.ok())
      << goldenPath.status() << "; the computed record is in " << output;
  const std::optional<std::string> golden = readFile(*goldenPath);
  ASSERT_TRUE(golden.has_value()) << "cannot read " << *goldenPath;
  if (*golden == record) return;

  const std::vector<absl::string_view> expected = absl::StrSplit(*golden, '\n');
  const std::vector<absl::string_view> actual = absl::StrSplit(record, '\n');
  size_t line = 0;
  while (line < expected.size() && line < actual.size() &&
         expected[line] == actual[line])
    ++line;
  ADD_FAILURE() << variant.name << " differs from the record from line "
                << line + 1 << " on: expected '"
                << (line < expected.size() ? expected[line] : "(end)")
                << "', assembled '"
                << (line < actual.size() ? actual[line] : "(end)")
                << "'. The computed record is in " << output;
}

INSTANTIATE_TEST_SUITE_P(
    ShippedAndVariants,
    CentroidalContactSetParityTest,
    ::testing::ValuesIn(kVariants),
    [](const ::testing::TestParamInfo<ParityVariant>& info) {
      return std::string(info.param.name);
    });

}  // namespace
}  // namespace ocs2::humanoid
