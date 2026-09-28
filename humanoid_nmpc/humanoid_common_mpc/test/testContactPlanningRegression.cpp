/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

/**
 * Regression tests of the term-assembled planner against recorded fixtures (test/data/contact_planning/). The fixtures
 * were recorded from the planner at the point where it had been shown, term for term, to build the same problem, to
 * propagate every assignment the same way and to plan the same walks as the previous monolithic planner (the
 * equivalence tests that accompanied the refactor); they pin that formulation down for every later change:
 *  1. problems: for a corpus of inputs, with and without the heading model, the dimensions and a hash of every stage
 *     matrix, of the input bounds and of the general rows (as a set), so that a change of any term shows up with the
 *     stage and the matrix it touched;
 *  2. logic: on a six-node horizon, the set of feasible complete assignments under a grid of configurations and a hash
 *     of their assignment costs;
 *  3. plans: eight-step receding-horizon walks, contacts exactly and trajectories to 1e-6.
 *
 * To re-record after an intended change of the formulation, run the test binary with
 *   REGENERATE_CONTACT_PLANNING_FIXTURES=<absolute path of test/data/contact_planning>
 * and review the diff of the fixture files.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ocs2::humanoid {

namespace {

constexpr const char* kFixtureDir = "humanoid_nmpc/humanoid_common_mpc/test/data/contact_planning";

ContactPlanningConfig makeConfig(bool heading) {
  ContactPlanningConfig config;
  config.planner.dt = 0.1;
  config.planner.numNodes = 12;
  config.planner.commitTime = 0.0;
  config.planner.maxBranchAndBoundNodes = 3000;
  config.planner.maxSolveTime = 10.0;
  config.shared.comHeight = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.5;
  config.shared.gaitLimits.minContactDuration = 0.15;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.1;
  config.eventShiftLocalSearch.maxTime = 2.0;
  if (heading) {
    config.setHeadingModel(true);
    config.yawTorqueBudget.torsionalFrictionTorque = 20.0;
    config.yawTorqueBudget.doubleSupportYawCouple = 40.0;
    config.hipYawRange.lower = {-0.4, -0.6};
    config.hipYawRange.upper = {0.6, 0.4};
  }
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  return config;
}

ContactPlannerInput standing() {
  ContactPlannerInput input;
  input.time = 3.0;
  input.footPositions[0] = vector2_t(0.0, 0.125);
  input.footPositions[1] = vector2_t(0.0, -0.125);
  input.contacts = {true, true};
  input.phaseElapsedTime = {5.0, 5.0};
  input.yawInertia = 10.0;
  return input;
}

ContactPlannerInput walking() {
  ContactPlannerInput input = standing();
  input.velocityCommand = vector2_t(0.4, 0.1);
  input.comVelocity = vector2_t(0.2, 0.0);
  input.yaw = 0.3;
  input.heading = 0.3;
  input.headingRateCommand = 0.4;
  input.footYaws = {0.25, 0.35};
  return input;
}

ContactPlannerInput pushed() {
  ContactPlannerInput input = standing();
  input.comVelocity = vector2_t(0.6, -0.2);
  input.comPosition = vector2_t(0.05, 0.02);
  return input;
}

ContactPlannerInput midSwing() {
  ContactPlannerInput input = walking();
  input.contacts = {false, true};
  input.phaseElapsedTime = {0.16, 0.7};
  input.lastSwungFoot = 0;
  input.footPositions[0] = vector2_t(0.1, 0.13);
  input.committedUntil = input.time + 0.25;
  input.committedContacts = {{false, true}, {false, true}, {true, true}};
  input.committedPhaseStartTimes = {
      makeFeetArray(input.time - 0.16), makeFeetArray(input.time - 0.16), {input.time + 0.27, input.time - 0.7}};
  return input;
}

std::vector<std::pair<std::string, ContactPlannerInput>> corpus() {
  return {{"standing", standing()}, {"walking", walking()}, {"pushed", pushed()}, {"mid_swing", midSwing()}};
}

/*---------------------------------------------- hashing ----------------------------------------------*/

constexpr scalar_t kQuantum = 1e-9;

std::int64_t quantize(scalar_t v) {
  return static_cast<std::int64_t>(std::llround(v / kQuantum));
}

struct Fnv {
  std::uint64_t h = 1469598103934665603ull;
  void add(std::int64_t v) {
    const std::uint64_t u = static_cast<std::uint64_t>(v);
    for (int i = 0; i < 8; ++i) {
      h ^= (u >> (8 * i)) & 0xffu;
      h *= 1099511628211ull;
    }
  }
  std::string hex() const {
    std::ostringstream out;
    out << std::hex << h;
    return out.str();
  }
};

/** nnz and hash of the nonzero entries of a matrix, in row-major order. */
std::pair<int, std::string> hashMatrix(const matrix_t& m) {
  Fnv fnv;
  int nnz = 0;
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    for (Eigen::Index j = 0; j < m.cols(); ++j) {
      const std::int64_t q = quantize(m(i, j));
      if (q == 0) continue;
      ++nnz;
      fnv.add(i);
      fnv.add(j);
      fnv.add(q);
    }
  }
  return {nnz, fnv.hex()};
}

std::pair<int, std::string> hashVector(const vector_t& v) {
  return hashMatrix(matrix_t(v));
}

std::string hashBounds(const OcpQpStage& s) {
  Fnv fnv;
  for (size_t i = 0; i < s.idxbu.size(); ++i) {
    fnv.add(s.idxbu[i]);
    fnv.add(quantize(s.lbu(static_cast<Eigen::Index>(i))));
    fnv.add(quantize(s.ubu(static_cast<Eigen::Index>(i))));
  }
  return fnv.hex();
}

/** The general rows as a set: sorted records of coefficients, bounds, softness and penalty. */
std::string hashRows(const OcpQpStage& s) {
  std::vector<std::vector<std::int64_t>> records;
  for (int i = 0; i < s.numGeneralConstraints(); ++i) {
    std::vector<std::int64_t> record;
    for (int j = 0; j < s.C.cols(); ++j) record.push_back(quantize(s.C(i, j)));
    for (int j = 0; j < s.D.cols(); ++j) record.push_back(quantize(s.D(i, j)));
    record.push_back(quantize(std::max(s.lg(i), -1e7)));
    record.push_back(quantize(std::min(s.ug(i), 1e7)));
    const std::vector<int>::const_iterator soft = std::find(s.softGeneralIndices.begin(), s.softGeneralIndices.end(), i);
    if (soft == s.softGeneralIndices.end()) {
      record.insert(record.end(), {0, 0, 0});
    } else {
      const long k = soft - s.softGeneralIndices.begin();
      record.push_back(1);
      record.push_back(quantize(s.Zl(k)));
      record.push_back(quantize(s.zl(k)));
    }
    records.push_back(std::move(record));
  }
  std::sort(records.begin(), records.end());
  Fnv fnv;
  for (const std::vector<std::int64_t>& record : records) {
    for (const std::int64_t v : record) fnv.add(v);
  }
  return fnv.hex();
}

/*---------------------------------------------- fixture i/o ----------------------------------------------*/

/** The fixture as an ordered list of "key value" lines; the key names the problem, stage and quantity. */
using Fixture = std::vector<std::pair<std::string, std::string>>;

std::string fixtureText(const Fixture& fixture) {
  std::ostringstream out;
  for (const std::pair<std::string, std::string>& entry : fixture) out << entry.first << " " << entry.second << "\n";
  return out.str();
}

Fixture readFixture(const std::string& file) {
  Fixture fixture;
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t space = line.find(' ');
    fixture.emplace_back(line.substr(0, space), space == std::string::npos ? "" : line.substr(space + 1));
  }
  return fixture;
}

const char* regenerateDir() {
  return std::getenv("REGENERATE_CONTACT_PLANNING_FIXTURES");
}

std::string fixturePath(const char* name) {
  const char* dir = regenerateDir();
  return std::string(dir != nullptr ? dir : kFixtureDir) + "/" + name;
}

/** Writes the fixture when regenerating; otherwise compares it line by line with the recorded one. */
void checkOrRecord(const char* name, const Fixture& fixture, const char* header) {
  const std::string path = fixturePath(name);
  if (regenerateDir() != nullptr) {
    std::ofstream out(path);
    out << "# " << header << "\n# recorded by testContactPlanningRegression; regenerate with REGENERATE_CONTACT_PLANNING_FIXTURES\n";
    out << fixtureText(fixture);
    ASSERT_TRUE(out.good()) << "cannot write " << path;
    LOG(INFO) << "[regression] recorded " << path;
    return;
  }
  const Fixture recorded = readFixture(path);
  ASSERT_FALSE(recorded.empty()) << "missing or empty fixture " << path;
  std::map<std::string, std::string> recordedByKey(recorded.begin(), recorded.end());
  ASSERT_EQ(recorded.size(), fixture.size()) << "the fixture " << name << " has a different number of entries";
  for (const std::pair<std::string, std::string>& entry : fixture) {
    const std::string& key = entry.first;
    const std::string& value = entry.second;
    const std::map<std::string, std::string>::const_iterator it = recordedByKey.find(key);
    ASSERT_NE(it, recordedByKey.end()) << "no recorded entry for " << key;
    EXPECT_EQ(it->second, value) << name << ": " << key << " changed (recorded '" << it->second << "', now '" << value << "')";
  }
}

std::string formatNumbers(const std::vector<scalar_t>& values, int precision) {
  std::ostringstream out;
  out.precision(precision);
  for (size_t i = 0; i < values.size(); ++i) out << (i > 0 ? " " : "") << values[i];
  return out.str();
}

std::vector<scalar_t> parseNumbers(const std::string& text) {
  std::vector<scalar_t> values;
  std::istringstream in(text);
  scalar_t v;
  while (in >> v) values.push_back(v);
  return values;
}

}  // namespace

/*============================================ 1. the problems =============================================*/

TEST(ContactPlanningRegression, AssembledProblemsMatchTheRecordedFormulation) {
  Fixture fixture;
  for (const bool heading : {false, true}) {
    const std::unique_ptr<LipContactPlanner> planner = LipContactPlanner::Create(makeConfig(heading)).value();
    for (const std::pair<std::string, ContactPlannerInput>& named : corpus()) {
      const std::string& name = named.first;
      const ContactPlannerInput& input = named.second;
      const std::string prefix = name + (heading ? ".heading" : ".lip");
      const OcpQpProblem problem = planner->buildProblem(input);
      fixture.emplace_back(
          prefix + ".x0", formatNumbers(std::vector<scalar_t>(problem.x0.data(), problem.x0.data() + problem.x0.size()), /*precision=*/12));
      for (size_t k = 0; k < problem.stages.size(); ++k) {
        const OcpQpStage& s = problem.stages[k];
        const std::string stage = prefix + ".stage" + std::to_string(k);
        std::ostringstream dims;
        dims << s.numStates() << " " << s.numInputs() << " " << s.numGeneralConstraints() << " " << s.softGeneralIndices.size();
        fixture.emplace_back(stage + ".dims", dims.str());
        const std::function<void(const char*, const std::pair<int, std::string>&)> entry = [&](const char* what,
                                                                                               const std::pair<int, std::string>& h) {
          fixture.emplace_back(stage + "." + what, std::to_string(h.first) + " " + h.second);
        };
        entry("Q", hashMatrix(s.Q));
        entry("R", hashMatrix(s.R));
        entry("S", hashMatrix(s.S));
        entry("q", hashVector(s.q));
        entry("r", hashVector(s.r));
        entry("A", hashMatrix(s.A));
        entry("B", hashMatrix(s.B));
        entry("b", hashVector(s.b));
        fixture.emplace_back(stage + ".bounds", std::to_string(s.idxbu.size()) + " " + hashBounds(s));
        fixture.emplace_back(stage + ".rows", hashRows(s));
      }
    }
  }
  checkOrRecord("problems.txt", fixture,
                "assembled OCP-QPs: per problem x0 and per stage the dimensions and hashes (nnz hash) of the matrices");
}

/*============================================ 2. the logic ================================================*/

namespace {

struct LogicCase {
  scalar_t maxContactDuration;
  scalar_t minDoubleSupportDuration;
  bool alternation;
};

std::vector<std::pair<std::string, ContactPlannerInput>> logicInputs() {
  ContactPlannerInput stand = standing();
  ContactPlannerInput swing = standing();
  swing.contacts = {false, true};
  swing.phaseElapsedTime = {0.16, 0.7};
  swing.lastSwungFoot = 0;
  ContactPlannerInput landed = standing();
  landed.phaseElapsedTime = {0.03, 0.8};
  landed.lastSwungFoot = 0;
  ContactPlannerInput committed = standing();
  committed.contacts = {true, false};
  committed.phaseElapsedTime = {0.9, 0.21};
  committed.lastSwungFoot = 1;
  committed.committedUntil = committed.time + 0.2;
  committed.committedContacts = {{true, false}, {true, true}};
  committed.committedPhaseStartTimes = {{committed.time - 0.9, committed.time - 0.21}, {committed.time - 0.9, committed.time + 0.17}};
  ContactPlannerInput unknownLast = standing();
  unknownLast.phaseElapsedTime = {0.5, 0.5};
  unknownLast.lastSwungFoot = -1;
  return {{"stand", stand}, {"swing", swing}, {"landed", landed}, {"committed", committed}, {"unknown_last", unknownLast}};
}

}  // namespace

TEST(ContactPlanningRegression, FeasibleAssignmentsMatchTheRecordedRules) {
  const std::vector<LogicCase> cases{{0.0, 0.0, true}, {0.0, 0.1, true}, {0.4, 0.1, true}, {0.4, 0.15, false}, {0.0, 0.0, false}};
  Fixture fixture;
  for (size_t c = 0; c < cases.size(); ++c) {
    ContactPlanningConfig config = makeConfig(false);
    config.planner.numNodes = 6;
    config.shared.gaitLimits.maxContactDuration = cases[c].maxContactDuration;
    config.shared.gaitLimits.minDoubleSupportDuration = cases[c].minDoubleSupportDuration;
    config.formulation.setLogicRule(term::kAlternatingFeet, cases[c].alternation);
    EXPECT_EQ(config.validateStatus(), absl::OkStatus());
    const std::unique_ptr<LipContactPlanner> planner = LipContactPlanner::Create(config).value();
    const int numBinaries = 2 * config.planner.numNodes;
    for (const std::pair<std::string, ContactPlannerInput>& named : logicInputs()) {
      const std::string& inputName = named.first;
      const ContactPlannerInput& input = named.second;
      // The feasible complete assignments as a bitmap, in hex, four codes per digit; the costs of the feasible ones hashed.
      std::string bitmap;
      Fnv costs;
      int numFeasible = 0;
      unsigned digit = 0;
      for (int code = 0; code < (1 << numBinaries); ++code) {
        MiqpAssignment a(static_cast<size_t>(numBinaries));
        for (int i = 0; i < numBinaries; ++i) a[static_cast<size_t>(i)] = static_cast<std::int8_t>((code >> i) & 1);
        const bool ok = planner->propagate(input, a);
        if (ok) {
          ++numFeasible;
          costs.add(quantize(planner->assignmentCost(input, a)));
        }
        digit |= (ok ? 1u : 0u) << (code % 4);
        if (code % 4 == 3) {
          bitmap.push_back("0123456789abcdef"[digit]);
          digit = 0;
        }
      }
      EXPECT_GT(numFeasible, 0) << "case " << c << " input " << inputName;
      const std::string key = "case" + std::to_string(c) + "." + inputName;
      fixture.emplace_back(key + ".feasible", std::to_string(numFeasible) + " " + bitmap);
      fixture.emplace_back(key + ".costs", costs.hex());
    }
  }
  checkOrRecord("logic.txt", fixture,
                "six-node horizon: feasible complete assignments (count, bitmap) and a hash of their assignment costs");
}

/*================================ 2b. propagation on the search tree =================================*/

/**
 * Propagation on a PARTIAL assignment may only prune what has no feasible completion, and may only fix what every
 * feasible completion agrees on (MiqpPropagateFn, ContactLogicRule).
 *
 * This is the property the branch-and-bound relies on, and the fixture above cannot see it: it enumerates COMPLETE
 * assignments, where every rule sees a fully fixed prefix. MixedIntegerOcpQp calls propagate() on partial assignments
 * at every node, and a rule that is merely too eager there loses feasible plans silently - the subtree is dropped with
 * no diagnostic, the search still reports `optimal`, and no recorded fixture changes.
 *
 * The oracle is the complete enumeration: the feasible complete assignments that agree with the committed prefix
 * (LipContactPlanner::initialAssignment, which is where every search starts). The partial assignments checked are
 * exactly the ones the branch-and-bound can hand to propagate(): the propagated root, and then every propagated parent
 * with ONE more binary fixed either way, closed under that step. That set matters on both sides:
 *  - it starts from the committed prefix, never from an all-free assignment. The rules deliberately do not constrain
 *    the committed nodes (the executed schedule is what it is), so a partial assignment that contradicts the prefix
 *    says nothing about soundness - an earlier version of this test enumerated such assignments and failed on them
 *    for good;
 *  - it is NOT the set of prefixes in the binaries' declared order. MixedIntegerOcpQp branches on the first FRACTIONAL
 *    binary and skips free ones whose relaxation happens to be integral, and diving fixes many at once, so any binary
 *    can be the next one fixed. The unsound PhaseDurationsRule fixing that pruned the other foot's recovery step was
 *    reachable only that way.
 */
namespace {

struct SoundnessCase {
  std::string name;
  scalar_t minSwingDuration;
  scalar_t maxSwingDuration;
  scalar_t minContactDuration;
  scalar_t maxContactDuration;
  scalar_t minDoubleSupportDuration;
  bool alternation;
  bool noFlight = true;
};

std::vector<SoundnessCase> soundnessCases() {
  return {
      {"maxContact 0.4, minDS 0.1, alternation", 0.3, 0.5, 0.15, 0.4, 0.1, true},
      {"maxContact 0.4, minDS 0.1", 0.3, 0.5, 0.15, 0.4, 0.1, false},
      {"maxContact 0.4, minDS 0.15", 0.3, 0.5, 0.15, 0.4, 0.15, false},
      {"minDS 0.1, alternation", 0.3, 0.5, 0.15, 0.0, 0.1, true},
      // A double-support hold of two nodes: a touch-down fixed at one node holds the other foot at the next one too,
      // which a hold read only at the touch-down node itself (or from a snapshot taken before the pass) misses.
      {"swing 0.3, maxContact 0.4, minDS 0.2", 0.3, 0.3, 0.15, 0.4, 0.2, false},
      {"swing 0.3, maxContact 0.4, minDS 0.2, alternation", 0.3, 0.3, 0.15, 0.4, 0.2, true},
      {"swing 0.2-0.3, contact 0.1-0.3, minDS 0.2", 0.2, 0.3, 0.1, 0.3, 0.2, false},
      // Without the no-flight rule the two feet of one node are no longer tied together, and a rule that reads them in
      // index order must not treat the first foot's still-open lift-off as decided when it reads the second.
      {"alternation without the no-flight rule", 0.3, 0.5, 0.15, 0.0, 0.1, true, false},
  };
}

std::vector<std::pair<std::string, ContactPlannerInput>> soundnessInputs() {
  std::vector<std::pair<std::string, ContactPlannerInput>> inputs = logicInputs();
  // One foot overdue with the other one free to step first: the maximum contact duration must not pick the order.
  ContactPlannerInput leftOverdue = standing();
  leftOverdue.phaseElapsedTime = {0.7, 0.2};
  leftOverdue.lastSwungFoot = 1;
  inputs.emplace_back("left_overdue", leftOverdue);
  // Both feet overdue, no known last swing.
  ContactPlannerInput bothOverdue = standing();
  bothOverdue.phaseElapsedTime = {0.51, 0.26};
  bothOverdue.lastSwungFoot = -1;
  inputs.emplace_back("both_overdue", bothOverdue);
  // A foot that has just become overdue while the other one has stood for long.
  ContactPlannerInput longStand = standing();
  longStand.phaseElapsedTime = {0.3, 2.0};
  longStand.lastSwungFoot = -1;
  inputs.emplace_back("long_stand", longStand);
  // Early in a left swing, the right foot not yet overdue: its lift-off after the landing waits out a hold.
  ContactPlannerInput earlySwing = standing();
  earlySwing.contacts = {false, true};
  earlySwing.phaseElapsedTime = {0.1, 0.3};
  earlySwing.lastSwungFoot = 0;
  inputs.emplace_back("early_swing", earlySwing);
  // Both feet down, the right foot swung last.
  ContactPlannerInput rightSwungLast = standing();
  rightSwungLast.phaseElapsedTime = {0.3, 0.1};
  rightSwungLast.lastSwungFoot = 1;
  inputs.emplace_back("right_swung_last", rightSwungLast);
  return inputs;
}

/** What a walk over the search tree found. */
struct SearchTreeAudit {
  int numFeasibleCompletions = 0;              // of the committed prefix
  int numPartialAssignments = 0;               // distinct propagated partial assignments reached
  int numPrunedChildren = 0;                   // children propagate() declared infeasible
  int numFixingsChecked = 0;                   // binaries propagate() fixed on a reachable child, each checked against the oracle
  std::vector<std::string> pruningViolations;  // a pruned child that had a feasible completion
  std::vector<std::string> fixingViolations;   // a fixed binary some feasible completion disagrees with
};

/** Node by node, left then right: 1 contact, 0 air, - free. */
std::string describeAssignment(const MiqpAssignment& a) {
  std::vector<std::string> nodes;
  for (size_t i = 0; i + 1 < a.size(); i += 2) {
    const std::string left = a[i] == kMiqpFree ? "-" : absl::StrCat(static_cast<int>(a[i]));
    const std::string right = a[i + 1] == kMiqpFree ? "-" : absl::StrCat(static_cast<int>(a[i + 1]));
    nodes.push_back(absl::StrCat(left, right));
  }
  return absl::StrJoin(nodes, " ");
}

bool agrees(const MiqpAssignment& partial, int code) {
  for (size_t i = 0; i < partial.size(); ++i) {
    if (partial[i] != kMiqpFree && partial[i] != static_cast<std::int8_t>((code >> i) & 1)) return false;
  }
  return true;
}

/** Records into `audit` every way `propagated` (the result of propagating `before`) contradicts the oracle. */
void checkPropagation(const std::string& where,
                      const MiqpAssignment& before,
                      bool declaredFeasible,
                      const MiqpAssignment& propagated,
                      const std::vector<int>& feasible,
                      SearchTreeAudit& audit) {
  std::vector<int> completions;
  for (const int code : feasible) {
    if (agrees(before, code)) completions.push_back(code);
  }
  if (!declaredFeasible) {
    ++audit.numPrunedChildren;
    if (!completions.empty()) {
      audit.pruningViolations.push_back(absl::StrCat(where, ": propagate() declared [", describeAssignment(before),
                                                     "] infeasible, but the complete assignment ", completions.front(),
                                                     " completes it and is feasible"));
    }
    return;
  }
  for (size_t i = 0; i < propagated.size(); ++i) {
    if (propagated[i] == kMiqpFree || before[i] != kMiqpFree) continue;
    ++audit.numFixingsChecked;
    for (const int code : completions) {
      if (static_cast<int>(propagated[i]) != ((code >> i) & 1)) {
        audit.fixingViolations.push_back(absl::StrCat(where, ": propagate() fixed binary ", i, " of [", describeAssignment(before), "] to ",
                                                      static_cast<int>(propagated[i]), ", against the feasible completion ", code));
        break;
      }
    }
  }
}

/**
 * Walks every partial assignment the branch-and-bound can reach from `input` - the propagated root, then a propagated
 * parent with one more binary fixed, closed under that step - and checks each propagation against the oracle.
 */
SearchTreeAudit auditSearchTree(const LipContactPlanner& planner, const ContactPlannerInput& input, int numBinaries) {
  SearchTreeAudit audit;
  const MiqpAssignment initial = planner.initialAssignment(input);
  std::vector<int> feasible;
  for (int code = 0; code < (1 << numBinaries); ++code) {
    if (!agrees(initial, code)) continue;
    MiqpAssignment complete(static_cast<size_t>(numBinaries));
    for (int i = 0; i < numBinaries; ++i) complete[static_cast<size_t>(i)] = static_cast<std::int8_t>((code >> i) & 1);
    if (planner.propagate(input, complete)) feasible.push_back(code);
  }
  audit.numFeasibleCompletions = static_cast<int>(feasible.size());

  MiqpAssignment root = initial;
  const bool rootFeasible = planner.propagate(input, root);
  checkPropagation("root", initial, rootFeasible, root, feasible, audit);
  if (!rootFeasible) return audit;

  std::set<MiqpAssignment> seen{root};
  std::vector<MiqpAssignment> open{root};
  while (!open.empty()) {
    const MiqpAssignment parent = open.back();
    open.pop_back();
    ++audit.numPartialAssignments;
    for (size_t i = 0; i < parent.size(); ++i) {
      if (parent[i] != kMiqpFree) continue;
      for (const std::int8_t value : {std::int8_t(0), std::int8_t(1)}) {
        MiqpAssignment child = parent;
        child[i] = value;
        MiqpAssignment propagated = child;
        const bool childFeasible = planner.propagate(input, propagated);
        checkPropagation(absl::StrCat("from [", describeAssignment(parent), "] fixing binary ", i, " to ", static_cast<int>(value)), child,
                         childFeasible, propagated, feasible, audit);
        if (childFeasible && seen.insert(propagated).second) open.push_back(propagated);
      }
    }
  }
  return audit;
}

/** The audit of every case and input, computed once for both tests below. */
const std::vector<std::pair<std::string, SearchTreeAudit>>& searchTreeAudits() {
  static const std::vector<std::pair<std::string, SearchTreeAudit>> audits = [] {
    std::vector<std::pair<std::string, SearchTreeAudit>> result;
    for (const SoundnessCase& soundnessCase : soundnessCases()) {
      ContactPlanningConfig config = makeConfig(false);
      config.planner.numNodes = 5;
      config.shared.gaitLimits.minSwingDuration = soundnessCase.minSwingDuration;
      config.shared.gaitLimits.maxSwingDuration = soundnessCase.maxSwingDuration;
      config.shared.gaitLimits.minContactDuration = soundnessCase.minContactDuration;
      config.shared.gaitLimits.maxContactDuration = soundnessCase.maxContactDuration;
      config.shared.gaitLimits.minDoubleSupportDuration = soundnessCase.minDoubleSupportDuration;
      config.formulation.setLogicRule(term::kAlternatingFeet, soundnessCase.alternation);
      config.formulation.setLogicRule(term::kNoFlight, soundnessCase.noFlight);
      EXPECT_EQ(config.validateStatus(), absl::OkStatus());
      const std::unique_ptr<const LipContactPlanner> planner = LipContactPlanner::Create(config).value();
      const int numBinaries = LipContactPlanner::kBinariesPerNode * config.planner.numNodes;
      for (const std::pair<std::string, ContactPlannerInput>& named : soundnessInputs()) {
        result.emplace_back(absl::StrCat(soundnessCase.name, " / ", named.first), auditSearchTree(*planner, named.second, numBinaries));
      }
    }
    return result;
  }();
  return audits;
}

}  // namespace

TEST(ContactPlanningRegression, PartialPropagationNeverPrunesAFeasibleSubtree) {
  int numPruned = 0;
  int numPartials = 0;
  for (const std::pair<std::string, SearchTreeAudit>& named : searchTreeAudits()) {
    const SearchTreeAudit& audit = named.second;
    EXPECT_GT(audit.numFeasibleCompletions, 0) << named.first << ": every case has to have a plan, or it proves nothing";
    numPruned += audit.numPrunedChildren;
    numPartials += audit.numPartialAssignments;
    for (const std::string& violation : audit.pruningViolations) ADD_FAILURE() << named.first << ": " << violation;
  }
  // Positive controls: the walk reaches a real search tree, and propagation does prune in it - a propagation that
  // never returned false would pass the check above for free.
  EXPECT_GT(numPartials, 1000);
  EXPECT_GT(numPruned, 0);
}

TEST(ContactPlanningRegression, PartialPropagationOnlyFixesWhatEveryFeasibleCompletionAgreesOn) {
  int numFixings = 0;
  for (const std::pair<std::string, SearchTreeAudit>& named : searchTreeAudits()) {
    numFixings += named.second.numFixingsChecked;
    for (const std::string& violation : named.second.fixingViolations) ADD_FAILURE() << named.first << ": " << violation;
  }
  // Positive control: propagation fixes binaries along the way (the rules are not idle), so the check above has
  // something to check.
  EXPECT_GT(numFixings, 0);
}

/*============================================ 3. the plans ================================================*/

TEST(ContactPlanningRegression, RecedingHorizonPlansMatchTheRecordedWalks) {
  Fixture fixture;
  for (const bool heading : {false, true}) {
    const ContactPlanningConfig config = makeConfig(heading);
    const std::unique_ptr<LipContactPlanner> planner = LipContactPlanner::Create(config).value();
    ContactPlannerInput input = walking();
    for (int step = 0; step < 8; ++step) {
      const std::string prefix = std::string(heading ? "heading" : "lip") + ".step" + std::to_string(step);
      const ContactPlan plan = planner->plan(input);
      ASSERT_TRUE(plan.valid) << prefix;
      std::string contacts;
      for (const contact_flag_t& c : plan.contacts) {
        contacts += (contacts.empty() ? "" : " ") + std::string(c[0] ? "1" : "0") + (c[1] ? "1" : "0");
      }
      fixture.emplace_back(prefix + ".contacts", contacts);
      std::vector<scalar_t> trajectory;
      for (int k = 0; k <= plan.numIntervals(); ++k) {
        for (int axis = 0; axis < 2; ++axis) trajectory.push_back(plan.comPosition[k](axis));
        for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
          for (int axis = 0; axis < 2; ++axis) trajectory.push_back(plan.footholds[k][foot](axis));
        }
        if (heading) trajectory.push_back(plan.heading[k]);
      }
      fixture.emplace_back(prefix + ".trajectory", formatNumbers(trajectory, /*precision=*/9));

      // Advance one node along the plan (no commit window in this configuration).
      const scalar_t nextTime = input.time + config.planner.dt;
      const contact_flag_t contactsNext = plan.contactsAtTime(nextTime);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        input.phaseElapsedTime[foot] =
            (contactsNext[foot] == input.contacts[foot]) ? input.phaseElapsedTime[foot] + config.planner.dt : 0.0;
        if (input.contacts[foot] && !contactsNext[foot]) input.lastSwungFoot = static_cast<int>(foot);
        input.footPositions[foot] = plan.footholds[1][foot];
        if (heading) input.footYaws[foot] = plan.footYaws[1][foot];
      }
      input.contacts = contactsNext;
      input.comPosition = plan.comPosition[1];
      input.comVelocity = plan.comVelocity[1];
      if (heading) {
        input.heading = plan.heading[1];
        input.headingRate = plan.headingRate[1];
        input.yaw = input.heading;
      }
      input.time = nextTime;
    }
  }

  // The contacts must match exactly, the trajectories to the solver's tolerance.
  const std::string path = fixturePath("plans.txt");
  if (regenerateDir() != nullptr) {
    checkOrRecord("plans.txt", fixture, "receding-horizon walks: contacts per step and the CoM / foothold (/ heading) trajectory per node");
    return;
  }
  const Fixture recorded = readFixture(path);
  ASSERT_FALSE(recorded.empty()) << "missing or empty fixture " << path;
  std::map<std::string, std::string> recordedByKey(recorded.begin(), recorded.end());
  ASSERT_EQ(recorded.size(), fixture.size());
  for (const std::pair<std::string, std::string>& entry : fixture) {
    const std::string& key = entry.first;
    const std::string& value = entry.second;
    const std::map<std::string, std::string>::const_iterator it = recordedByKey.find(key);
    ASSERT_NE(it, recordedByKey.end()) << "no recorded entry for " << key;
    if (key.find(".contacts") != std::string::npos) {
      EXPECT_EQ(it->second, value) << key << " changed";
    } else {
      const std::vector<scalar_t> expected = parseNumbers(it->second);
      const std::vector<scalar_t> actual = parseNumbers(value);
      ASSERT_EQ(expected.size(), actual.size()) << key;
      for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-6) << key << " entry " << i;
      }
    }
  }
}

}  // namespace ocs2::humanoid
