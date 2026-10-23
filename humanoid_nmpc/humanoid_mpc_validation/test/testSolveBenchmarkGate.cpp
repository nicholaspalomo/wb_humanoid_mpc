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

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/benchmark/SolveBenchmarkGate.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "humanoid_mpc_validation/io/JsonValue.h"
#include "humanoid_mpc_validation/io/RunProvenance.h"

/*
 * The real-time gate of section 4.6 (compareSolveBenchmarks), on synthetic documents of runSolveBenchmark()'s shape: a
 * document passes against itself (one that names no toolchain compares its tape counts only), each criterion fails on
 * its own when breached, only the libraries both documents count are compared one by one, and a solve time of another
 * machine or toolchain is refused while the tape counts are still compared. Also: the solve benchmark's default
 * recorded states exist for every robot.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr char kStatesDirectory[] = "humanoid_nmpc/humanoid_mpc_validation/data/benchmark/states";

// Three of the Unitree G1's CppAD libraries, as tape_operation_counts names them; the first is a small share of the total.
constexpr char kSmallLibrary[] = "cppad_centroidal_mpc_g1/wrench_inputs/foot_l_contact_TaskSpaceKinematicsCost_yawRefHalfAngle";
constexpr char kFlowMapLibrary[] = "cppad_centroidal_mpc_g1/wrench_inputs/dynamics_systemFlowMap";
constexpr char kTwistLibrary[] = "cppad_centroidal_mpc_g1/wrench_inputs/foot_l_contact_twist";

/** An object with `members`, in their order. */
JsonValue objectWith(const std::vector<std::pair<std::string, JsonValue>>& members) {
  JsonValue object = JsonValue::object();
  for (const std::pair<std::string, JsonValue>& member : members) object.set(member.first, member.second);
  return object;
}

/** The provenance's toolchain object, as makeRunProvenanceJson() writes it. */
JsonValue toolchainObject(const RunToolchain& toolchain) {
  return objectWith({{"compiler", JsonValue::string(toolchain.compiler)},
                     {"hpipm_target", JsonValue::string(toolchain.hpipmTarget)},
                     {"container_image", JsonValue::string(toolchain.containerImage)}});
}

/** The toolchain of the synthetic documents: this build's compiler. */
RunToolchain testToolchain() {
  return RunToolchain{.compiler = compilerVersion(), .hpipmTarget = "GENERIC", .containerImage = "sha256:0123"};
}

/**
 * A solve benchmark document of a centroidal Unitree G1 at 80 Hz, as runSolveBenchmark() writes the members the gate
 * reads: the provenance's machine and toolchain, the solve times, the real-time budget and the tape operation counts.
 */
JsonValue benchmarkDocument() {
  const double p99 = 8.3;
  const double periodMs = 12.5;
  const std::vector<std::pair<std::string, double>> counts = {{kSmallLibrary, 611.0}, {kFlowMapLibrary, 4136.0}, {kTwistLibrary, 4324.0}};
  JsonValue libraries = JsonValue::object();
  double total = 0.0;
  for (const std::pair<std::string, double>& count : counts) {
    libraries.set(count.first, JsonValue::number(count.second));
    total += count.second;
  }
  return objectWith({
      {"schema", JsonValue::string("humanoid_mpc_validation.solve_benchmark.v1")},
      {"label", JsonValue::string("test")},
      {"robot", JsonValue::string("unitree_g1")},
      {"provenance", objectWith({{"machine", JsonValue::string("a test machine, 8 logical cores, 16.0 GiB")},
                                 {"toolchain", toolchainObject(testToolchain())}})},
      {"solve_time_ms", objectWith({{"total", objectWith({{"mean", JsonValue::number(7.5)}, {"p99", JsonValue::number(p99)}})},
                                    {"lq_approximation", objectWith({{"mean", JsonValue::number(4.6)}})}})},
      {"real_time",
       objectWith({{"mpc_period_ms", JsonValue::number(periodMs)}, {"p99_fraction_of_period", JsonValue::number(p99 / periodMs)}})},
      {"tape_operation_counts", libraries},
      {"total_tape_operations", JsonValue::number(total)},
  });
}

/** `document` with the number at the key path `path` (from `depth` on) scaled by `factor`; members keep their order. */
JsonValue scaled(const JsonValue& document, const std::vector<std::string>& path, double factor, size_t depth = 0) {
  JsonValue copy = document;
  const JsonValue* absl_nullable member = document.find(path[depth]);
  CHECK(member != nullptr) << path[depth];
  if (depth + 1 == path.size()) {
    CHECK(member->isNumber()) << path[depth];
    copy.set(path[depth], JsonValue::number(factor * member->asNumber()));
  } else {
    copy.set(path[depth], scaled(*member, path, factor, depth + 1));
  }
  return copy;
}

/** `document` with `toolchain` as its provenance's toolchain. */
JsonValue withToolchain(const JsonValue& document, const RunToolchain& toolchain) {
  const JsonValue* absl_nullable recorded = document.find("provenance");
  CHECK(recorded != nullptr) << "a benchmark document has a provenance";
  JsonValue provenance = *recorded;
  provenance.set("toolchain", toolchainObject(toolchain));
  JsonValue copy = document;
  copy.set("provenance", provenance);
  return copy;
}

/** `document` whose provenance names no toolchain. */
JsonValue withoutToolchain(const JsonValue& document) {
  const JsonValue* absl_nullable recorded = document.find("provenance");
  CHECK(recorded != nullptr) << "a benchmark document has a provenance";
  JsonValue provenance = JsonValue::object();
  for (size_t i = 0; i < recorded->size(); ++i) {
    if (recorded->keyAt(i) != "toolchain") provenance.set(recorded->keyAt(i), recorded->valueAt(i));
  }
  JsonValue copy = document;
  copy.set("provenance", provenance);
  return copy;
}

TEST(SolveBenchmarkGate, ADocumentPassesAgainstItself) {
  const JsonValue document = benchmarkDocument();
  const std::vector<std::string> violations = compareSolveBenchmarks(document, document, SolveBenchmarkGate());
  EXPECT_TRUE(violations.empty()) << (violations.empty() ? "" : violations.front());

  // A document that names no toolchain: its solve times are refused even against itself, its tape counts pass.
  const JsonValue unnamed = withoutToolchain(document);
  const std::vector<std::string> refused = compareSolveBenchmarks(unnamed, unnamed, SolveBenchmarkGate());
  ASSERT_EQ(refused.size(), 1u);
  EXPECT_EQ(refused[0].rfind("solve_time_ms: not compared", 0), 0u) << refused[0];
}

TEST(SolveBenchmarkGate, EachCriterionFailsOnItsOwn) {
  const JsonValue baseline = benchmarkDocument();
  EXPECT_TRUE(compareSolveBenchmarks(scaled(baseline, {"solve_time_ms", "total", "mean"}, /*factor=*/1.049), baseline, SolveBenchmarkGate())
                  .empty());
  EXPECT_EQ(
      compareSolveBenchmarks(scaled(baseline, {"solve_time_ms", "total", "mean"}, /*factor=*/1.051), baseline, SolveBenchmarkGate()).size(),
      1u);
  EXPECT_EQ(
      compareSolveBenchmarks(scaled(baseline, {"solve_time_ms", "total", "p99"}, /*factor=*/1.11), baseline, SolveBenchmarkGate()).size(),
      1u);
  EXPECT_EQ(compareSolveBenchmarks(scaled(baseline, {"solve_time_ms", "lq_approximation", "mean"}, /*factor=*/1.09), baseline,
                                   SolveBenchmarkGate())
                .size(),
            1u);
  JsonValue overBudget = baseline;
  overBudget.set("real_time", JsonValue::object()).set("p99_fraction_of_period", JsonValue::number(0.81));
  EXPECT_EQ(compareSolveBenchmarks(overBudget, baseline, SolveBenchmarkGate()).size(), 1u);
  // One library 6 % larger: that library, and not the total, which its 611 of 9071 operations grow by 0.4 %.
  const JsonValue smallGrown = scaled(baseline, {"tape_operation_counts", kSmallLibrary}, /*factor=*/1.06);
  const std::vector<std::string> grown = compareSolveBenchmarks(
      scaled(smallGrown, {"total_tape_operations"}, /*factor=*/(9071.0 + 0.06 * 611.0) / 9071.0), baseline, SolveBenchmarkGate());
  ASSERT_EQ(grown.size(), 1u);
  EXPECT_EQ(grown[0].rfind(absl::StrCat("tape_operation_counts.", kSmallLibrary), 0), 0u) << grown[0];
  // Every library 6 % larger: each of them, and the total.
  JsonValue allGrown = baseline;
  for (const char* absl_nonnull library : {kSmallLibrary, kFlowMapLibrary, kTwistLibrary}) {
    allGrown = scaled(allGrown, {"tape_operation_counts", library}, /*factor=*/1.06);
  }
  allGrown = scaled(allGrown, {"total_tape_operations"}, /*factor=*/1.06);
  const std::vector<std::string> everyLibrary = compareSolveBenchmarks(allGrown, baseline, SolveBenchmarkGate());
  ASSERT_EQ(everyLibrary.size(), 4u);
  EXPECT_EQ(everyLibrary[3].rfind("total_tape_operations", 0), 0u) << everyLibrary[3];
}

TEST(SolveBenchmarkGate, OnlyTheLibrariesBothDocumentsCountAreComparedOneByOne) {
  // The candidate's small library under another name, twice as large: no library of the other is its own, so it is
  // compared through the total alone, which it grows by 611 of 9071 operations (6.7 %).
  const JsonValue baseline = benchmarkDocument();
  const JsonValue* absl_nullable libraries = baseline.find("tape_operation_counts");
  ASSERT_NE(libraries, nullptr);
  JsonValue renamed = JsonValue::object();
  for (size_t i = 0; i < libraries->size(); ++i) {
    const bool small = libraries->keyAt(i) == kSmallLibrary;
    renamed.set(small ? absl::StrCat(kSmallLibrary, "_renamed") : libraries->keyAt(i),
                JsonValue::number(small ? 2.0 * libraries->valueAt(i).asNumber() : libraries->valueAt(i).asNumber()));
  }
  JsonValue candidate = baseline;
  candidate.set("tape_operation_counts", renamed);
  candidate = scaled(candidate, {"total_tape_operations"}, /*factor=*/(9071.0 + 611.0) / 9071.0);
  const std::vector<std::string> violations = compareSolveBenchmarks(candidate, baseline, SolveBenchmarkGate());
  ASSERT_EQ(violations.size(), 1u) << "the total only";
  EXPECT_EQ(violations[0].rfind("total_tape_operations", 0), 0u) << violations[0];
}

TEST(SolveBenchmarkGate, ASolveTimeOfAnotherMachineOrToolchainIsRefusedAndTheTapeCountsStillCompared) {
  const JsonValue baseline = benchmarkDocument();
  // Twice as slow and a library 6 % larger: both found on one toolchain.
  const JsonValue slower = scaled(scaled(baseline, {"solve_time_ms", "total", "mean"}, /*factor=*/2.0),
                                  {"tape_operation_counts", kSmallLibrary}, /*factor=*/1.06);
  ASSERT_EQ(compareSolveBenchmarks(slower, baseline, SolveBenchmarkGate()).size(), 2u);

  RunToolchain otherCompiler = testToolchain();
  otherCompiler.compiler = "14.2.0";
  RunToolchain otherHpipm = testToolchain();
  otherHpipm.hpipmTarget = "AVX";
  RunToolchain otherImage = testToolchain();
  otherImage.containerImage = "sha256:4567";
  for (const RunToolchain& other : {otherCompiler, otherHpipm, otherImage}) {
    const std::vector<std::string> violations = compareSolveBenchmarks(withToolchain(slower, other), baseline, SolveBenchmarkGate());
    ASSERT_EQ(violations.size(), 2u) << "the refusal instead of the mean, and the library";
    EXPECT_EQ(violations[0].rfind("solve_time_ms: not compared, the toolchains differ", 0), 0u) << violations[0];
    EXPECT_NE(violations[1].find(kSmallLibrary), std::string::npos) << violations[1];
  }
  // Another machine, or a baseline that names no toolchain, is refused the same way.
  JsonValue otherMachine = slower;
  JsonValue provenance = *otherMachine.find("provenance");
  provenance.set("machine", JsonValue::string("another machine"));
  otherMachine.set("provenance", provenance);
  const std::vector<std::string> machine = compareSolveBenchmarks(otherMachine, baseline, SolveBenchmarkGate());
  ASSERT_EQ(machine.size(), 2u);
  EXPECT_EQ(machine[0].rfind("solve_time_ms: not compared, the machines differ", 0), 0u) << machine[0];
  const std::vector<std::string> unnamed = compareSolveBenchmarks(slower, withoutToolchain(baseline), SolveBenchmarkGate());
  ASSERT_EQ(unnamed.size(), 2u);
  EXPECT_EQ(unnamed[0].rfind("solve_time_ms: not compared, the baseline names no toolchain", 0), 0u) << unnamed[0];
}

TEST(SolveBenchmarkGate, EveryRobotHasItsDefaultRecordedStates) {
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    const std::filesystem::path states =
        std::filesystem::path(kStatesDirectory) / absl::StrCat(configuration.name, "_", configuration.walkingScenario, "_states.txt");
    EXPECT_TRUE(std::filesystem::exists(states)) << configuration.name << ": " << states;
  }
}

}  // namespace
}  // namespace ocs2::humanoid::validation
