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
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"

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
  config.validate();
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
    const auto u = static_cast<std::uint64_t>(v);
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
    const auto soft = std::find(s.softGeneralIndices.begin(), s.softGeneralIndices.end(), i);
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
  for (const auto& record : records) {
    for (const std::int64_t v : record) fnv.add(v);
  }
  return fnv.hex();
}

/*---------------------------------------------- fixture i/o ----------------------------------------------*/

/** The fixture as an ordered list of "key value" lines; the key names the problem, stage and quantity. */
using Fixture = std::vector<std::pair<std::string, std::string>>;

std::string fixtureText(const Fixture& fixture) {
  std::ostringstream out;
  for (const auto& [key, value] : fixture) out << key << " " << value << "\n";
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
    std::cout << "[regression] recorded " << path << std::endl;
    return;
  }
  const Fixture recorded = readFixture(path);
  ASSERT_FALSE(recorded.empty()) << "missing or empty fixture " << path;
  std::map<std::string, std::string> recordedByKey(recorded.begin(), recorded.end());
  ASSERT_EQ(recorded.size(), fixture.size()) << "the fixture " << name << " has a different number of entries";
  for (const auto& [key, value] : fixture) {
    const auto it = recordedByKey.find(key);
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
    LipContactPlanner planner(makeConfig(heading));
    for (const auto& [name, input] : corpus()) {
      const std::string prefix = name + (heading ? ".heading" : ".lip");
      const OcpQpProblem problem = planner.buildProblem(input);
      fixture.emplace_back(prefix + ".x0",
                           formatNumbers(std::vector<scalar_t>(problem.x0.data(), problem.x0.data() + problem.x0.size()), 12));
      for (size_t k = 0; k < problem.stages.size(); ++k) {
        const OcpQpStage& s = problem.stages[k];
        const std::string stage = prefix + ".stage" + std::to_string(k);
        std::ostringstream dims;
        dims << s.numStates() << " " << s.numInputs() << " " << s.numGeneralConstraints() << " " << s.softGeneralIndices.size();
        fixture.emplace_back(stage + ".dims", dims.str());
        const auto entry = [&](const char* what, const std::pair<int, std::string>& h) {
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
    config.validate();
    LipContactPlanner planner(config);
    const int numBinaries = 2 * config.planner.numNodes;
    for (const auto& [inputName, input] : logicInputs()) {
      // The feasible complete assignments as a bitmap, in hex, four codes per digit; the costs of the feasible ones hashed.
      std::string bitmap;
      Fnv costs;
      int numFeasible = 0;
      unsigned digit = 0;
      for (int code = 0; code < (1 << numBinaries); ++code) {
        MiqpAssignment a(static_cast<size_t>(numBinaries));
        for (int i = 0; i < numBinaries; ++i) a[static_cast<size_t>(i)] = static_cast<std::int8_t>((code >> i) & 1);
        const bool ok = planner.propagate(input, a);
        if (ok) {
          ++numFeasible;
          costs.add(quantize(planner.assignmentCost(input, a)));
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

/*============================================ 3. the plans ================================================*/

TEST(ContactPlanningRegression, RecedingHorizonPlansMatchTheRecordedWalks) {
  Fixture fixture;
  for (const bool heading : {false, true}) {
    const ContactPlanningConfig config = makeConfig(heading);
    LipContactPlanner planner(config);
    ContactPlannerInput input = walking();
    for (int step = 0; step < 8; ++step) {
      const std::string prefix = std::string(heading ? "heading" : "lip") + ".step" + std::to_string(step);
      const ContactPlan plan = planner.plan(input);
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
      fixture.emplace_back(prefix + ".trajectory", formatNumbers(trajectory, 9));

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
  for (const auto& [key, value] : fixture) {
    const auto it = recordedByKey.find(key);
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
