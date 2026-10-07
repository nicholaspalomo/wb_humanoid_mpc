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
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/StateLayout.h"
#include "humanoid_mpc_validation/benchmark/SolveBenchmarkGate.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "humanoid_mpc_validation/io/JsonValue.h"

/*
 * The real-time gate of section 4.6 (compareSolveBenchmarks): every recorded baseline passes against itself, each
 * criterion fails on its own when breached, and a library compares with itself across the move into the layout-tagged
 * CppAD folder of Step 8. Also: the solve benchmark's default recorded states exist for every robot.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr char kBenchmarkDirectory[] = "humanoid_nmpc/humanoid_mpc_validation/data/benchmark";

std::vector<JsonValue> recordedBaselines() {
  std::vector<JsonValue> documents;
  for (const std::filesystem::directory_entry& entry : std::filesystem::recursive_directory_iterator(kBenchmarkDirectory)) {
    if (entry.path().extension() != ".json") continue;
    absl::StatusOr<JsonValue> document = readJsonFile(entry.path().string());
    CHECK(document.ok()) << document.status();
    documents.push_back(*std::move(document));
  }
  return documents;
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

TEST(SolveBenchmarkGate, EveryRecordedBaselinePassesAgainstItself) {
  const std::vector<JsonValue> baselines = recordedBaselines();
  ASSERT_FALSE(baselines.empty());
  for (const JsonValue& baseline : baselines) {
    const std::vector<std::string> violations = compareSolveBenchmarks(baseline, baseline, SolveBenchmarkGate());
    EXPECT_TRUE(violations.empty()) << baseline.find("robot")->asString() << ": " << (violations.empty() ? "" : violations.front());
  }
}

TEST(SolveBenchmarkGate, EachCriterionFailsOnItsOwn) {
  const JsonValue baseline = recordedBaselines().front();
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
  // One library 6 % larger: that library, and not the total (its share of the total is small).
  const JsonValue* absl_nullable libraries = baseline.find("tape_operation_counts");
  ASSERT_TRUE(libraries != nullptr && libraries->size() > 1);
  const std::string library = libraries->keyAt(0);
  const std::vector<std::string> grown =
      compareSolveBenchmarks(scaled(baseline, {"tape_operation_counts", library}, /*factor=*/1.06), baseline, SolveBenchmarkGate());
  ASSERT_EQ(grown.size(), 1u);
  EXPECT_NE(grown[0].find(library), std::string::npos) << grown[0];
}

TEST(SolveBenchmarkGate, ALibraryComparesWithItselfAcrossTheLayoutTaggedFolder) {
  EXPECT_EQ(normalizedLibraryKey(absl::StrCat("cppad_centroidal_mpc_atlas/", kStateLayoutTag, "/basis11_05bee4540d9a43eb/icp_Cost")),
            "cppad_centroidal_mpc_atlas/basis11_05bee4540d9a43eb/icp_Cost");
  EXPECT_EQ(normalizedLibraryKey("cppad_wb_mpc_g1/FlowMap"), "cppad_wb_mpc_g1/FlowMap");

  // A candidate whose libraries all moved into the tagged folder still compares, library by library.
  const JsonValue baseline = recordedBaselines().front();
  JsonValue moved = baseline;
  JsonValue tagged = JsonValue::object();
  const JsonValue* absl_nullable libraries = baseline.find("tape_operation_counts");
  ASSERT_NE(libraries, nullptr);
  for (size_t i = 0; i < libraries->size(); ++i) {
    const std::string& key = libraries->keyAt(i);
    const size_t slash = key.find('/');
    const std::string movedKey = absl::StrCat(key.substr(0, slash), "/", kStateLayoutTag, key.substr(slash));
    tagged.set(movedKey, JsonValue::number(i == 0 ? 2.0 * libraries->valueAt(i).asNumber() : libraries->valueAt(i).asNumber()));
  }
  moved.set("tape_operation_counts", tagged);
  const std::vector<std::string> violations = compareSolveBenchmarks(moved, baseline, SolveBenchmarkGate());
  ASSERT_EQ(violations.size(), 1u) << "only the doubled library, found under its moved key";
  EXPECT_NE(violations[0].find(normalizedLibraryKey(libraries->keyAt(0))), std::string::npos) << violations[0];
}

TEST(SolveBenchmarkGate, EveryRobotHasItsDefaultRecordedStates) {
  for (const RobotConfiguration& configuration : robotConfigurations()) {
    const std::filesystem::path states = std::filesystem::path(kBenchmarkDirectory) / "states" /
                                         absl::StrCat(configuration.name, "_", configuration.walkingScenario, "_states.txt");
    EXPECT_TRUE(std::filesystem::exists(states)) << configuration.name << ": " << states;
  }
}

}  // namespace
}  // namespace ocs2::humanoid::validation
