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

// lazyWeightedAStar() on a toy problem: a 4-connected grid with walls whose
// edges are validated only when popped. A* finds the shortest path, the
// weighted search one within its bound, and the budgets and an unreachable
// goal stop it with their reasons.

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"

#include "humanoid_multi_contact/planner/LazyWeightedAStar.h"

namespace ocs2::humanoid::multi_contact {
namespace {

constexpr int kSize = 10;

struct Cell {
  int x = 0;
  int y = 0;
};

/** A grid from (0, 0) to `goal`; a wall at x = 5 for y < 8 makes the path
 * detour. Each candidate is a new node: its edge is unknown until
 * validated. */
class GridProblem : public LazySearchProblem {
 public:
  explicit GridProblem(Cell goal) : goal_(goal) { cells_.push_back(Cell{}); }

  std::vector<SearchCandidate> candidates(int node) override {
    std::vector<SearchCandidate> result;
    const Cell cell = cells_[node];
    const std::vector<Cell> neighbors = {{cell.x + 1, cell.y},
                                         {cell.x - 1, cell.y},
                                         {cell.x, cell.y + 1},
                                         {cell.x, cell.y - 1}};
    for (const Cell& neighbor : neighbors) {
      if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= kSize ||
          neighbor.y >= kSize) {
        continue;
      }
      cells_.push_back(neighbor);
      result.push_back(SearchCandidate{
          .node = static_cast<int>(cells_.size()) - 1, .estimatedCost = 1.0});
    }
    return result;
  }

  absl::StatusOr<std::optional<double>> validate(int node) override {
    validated_.insert(node);
    if (isWall(cells_[node])) return std::optional<double>();
    return std::optional<double>(1.0);
  }

  double heuristic(int node) override {
    return std::abs(cells_[node].x - goal_.x) +
           std::abs(cells_[node].y - goal_.y);
  }

  std::string key(int node) override {
    return absl::StrCat(cells_[node].x, ",", cells_[node].y);
  }

  absl::StatusOr<bool> isGoal(int node) override {
    return cells_[node].x == goal_.x && cells_[node].y == goal_.y;
  }

  bool isWall(const Cell& cell) const { return cell.x == 5 && cell.y < 8; }
  const Cell& cell(int node) const { return cells_[node]; }
  int validatedCount() const { return static_cast<int>(validated_.size()); }

 private:
  Cell goal_;
  std::vector<Cell> cells_;
  absl::flat_hash_set<int> validated_;
};

TEST(LazyWeightedAStar, FindsTheShortestPathAroundTheWall) {
  GridProblem problem(Cell{.x = 9, .y = 0});
  const absl::StatusOr<LazySearchResult> result =
      lazyWeightedAStar(problem, /*start=*/0, LazySearchOptions());
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_TRUE(result->found);
  // Up to y = 8, across, and down again: 8 + 9 + 8.
  EXPECT_EQ(result->cost, 25.0);
  EXPECT_EQ(result->path.size(), 26U);
  for (size_t i = 1; i < result->path.size(); ++i) {
    const Cell& from = problem.cell(result->path[i - 1]);
    const Cell& to = problem.cell(result->path[i]);
    EXPECT_EQ(std::abs(from.x - to.x) + std::abs(from.y - to.y), 1);
    EXPECT_FALSE(problem.isWall(to));
  }
  EXPECT_EQ(result->validations, problem.validatedCount());
}

TEST(LazyWeightedAStar, WeightedSearchStaysWithinItsBound) {
  GridProblem problem(Cell{.x = 9, .y = 0});
  LazySearchOptions options;
  options.heuristicWeight = 3.0;
  const absl::StatusOr<LazySearchResult> weighted =
      lazyWeightedAStar(problem, /*start=*/0, options);
  ASSERT_TRUE(weighted.ok()) << weighted.status();
  ASSERT_TRUE(weighted->found);
  EXPECT_LE(weighted->cost, 3.0 * 25.0);
  EXPECT_GE(weighted->cost, 25.0);
}

TEST(LazyWeightedAStar, StopsAtItsBudgets) {
  GridProblem problem(Cell{.x = 9, .y = 0});
  LazySearchOptions options;
  options.maxValidations = 5;
  const absl::StatusOr<LazySearchResult> few =
      lazyWeightedAStar(problem, /*start=*/0, options);
  ASSERT_TRUE(few.ok()) << few.status();
  EXPECT_FALSE(few->found);
  EXPECT_EQ(few->stopReason, "validations");
  EXPECT_EQ(few->validations, 5);

  GridProblem timed(Cell{.x = 9, .y = 0});
  options = LazySearchOptions();
  options.timeBudget = absl::ZeroDuration();
  const absl::StatusOr<LazySearchResult> late =
      lazyWeightedAStar(timed, /*start=*/0, options);
  ASSERT_TRUE(late.ok()) << late.status();
  EXPECT_FALSE(late->found);
  EXPECT_EQ(late->stopReason, "time");
}

TEST(LazyWeightedAStar, ReportsAnUnreachableGoal) {
  // The goal is inside the wall.
  GridProblem problem(Cell{.x = 5, .y = 0});
  const absl::StatusOr<LazySearchResult> result =
      lazyWeightedAStar(problem, /*start=*/0, LazySearchOptions());
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_FALSE(result->found);
  EXPECT_EQ(result->stopReason, "exhausted");
}

TEST(LazyWeightedAStar, RefusesAWeightBelowOne) {
  GridProblem problem(Cell{.x = 9, .y = 0});
  LazySearchOptions options;
  options.heuristicWeight = 0.5;
  EXPECT_FALSE(lazyWeightedAStar(problem, /*start=*/0, options).ok());
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
