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

#include "humanoid_common_mpc/contact_planning/LipContactPlanner.h"

#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "humanoid_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "humanoid_common_mpc/contact_planning/model/LipBlockIndices.h"

namespace ocs2::humanoid {

static_assert(static_cast<int>(LipContactPlanner::CX) == static_cast<int>(LIP_CX) &&
                  static_cast<int>(LipContactPlanner::PRY) == static_cast<int>(LIP_PRY) &&
                  static_cast<int>(LipContactPlanner::STATE_DIM) == static_cast<int>(LIP_STATE_DIM),
              "the planner's state enum is the layout of the LIP and foothold blocks");
static_assert(static_cast<int>(LipContactPlanner::ZX) == static_cast<int>(LIP_ZX) &&
                  static_cast<int>(LipContactPlanner::CR) == static_cast<int>(LIP_CR) &&
                  static_cast<int>(LipContactPlanner::INPUT_DIM) == static_cast<int>(LIP_INPUT_DIM),
              "the planner's input enum is the layout of the LIP and foothold blocks");

namespace {

using Clock = std::chrono::steady_clock;

scalar_t elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<scalar_t>(Clock::now() - start).count();
}

/** HPIPM settings for branch-and-bound relaxations: moderate accuracy is enough for bounding and integrality decisions. */
OcpQpHpipmSolver::Settings relaxationQpSettings(const ContactPlanningConfig& config) {
  OcpQpHpipmSolver::Settings qpSettings;
  qpSettings.iterMax = config.planner.maxQpIterations;
  qpSettings.hpipmMode = 2;  // BALANCE
  qpSettings.mu0 = 1e1;
  qpSettings.tolStat = 1e-5;
  qpSettings.tolEq = 1e-6;
  qpSettings.tolIneq = 1e-6;
  qpSettings.tolComp = 1e-6;
  return qpSettings;
}

}  // namespace

Layout LipContactPlanner::makeLayout(const ContactPlanningConfig& config) {
  return ContactPlanningTermFactory::buildProblem(config).layout();
}

scalar_t LipContactPlanner::yawInertia(const ContactPlannerInput& input) const {
  if (input.yawInertia <= 0.0) {
    throw std::invalid_argument("[LipContactPlanner] the heading model needs a positive yaw inertia in the input (from the robot model)");
  }
  return input.yawInertia;
}

LipContactPlanner::LipContactPlanner(ContactPlanningConfig config) : config_(std::move(config)) {
  config_.validate();
  problem_ = ContactPlanningTermFactory::buildProblem(config_);
  searchStages_ = ContactPlanningTermFactory::buildSearchStages(config_);
  rebuildSolver();
}

void LipContactPlanner::rebuildSolver() {
  MiqpSettings miqpSettings;
  miqpSettings.maxNodes = config_.planner.maxBranchAndBoundNodes;
  miqpSettings.maxSolveTime = config_.planner.maxSolveTime;
  miqpSettings.verbose = config_.planner.verbose;
  miqpSettings.useDivingHeuristic = false;  // the diving stage turns it on before every search
  miqp_ = std::make_unique<MixedIntegerOcpQp>(relaxationQpSettings(config_), miqpSettings);
}

void LipContactPlanner::setConfig(const ContactPlanningConfig& config) {
  config.validate();
  ContactPlanningProblem problem = ContactPlanningTermFactory::buildProblem(config);
  TermCollection<SearchStage> stages = ContactPlanningTermFactory::buildSearchStages(config);
  // The warm start is only invalid when the grid or the decision variables change; it survives a change of weights,
  // limits or term lists that keep the layout.
  const Layout& layout = problem.layout();
  const bool sameProblem = config.planner.numNodes == config_.planner.numNodes &&
                           std::abs(config.planner.dt - config_.planner.dt) <= 1e-12 && layout.nx == problem_.layout().nx &&
                           layout.nu == problem_.layout().nu && layout.hasHeading == problem_.layout().hasHeading;
  config_ = config;
  problem_ = std::move(problem);
  searchStages_ = std::move(stages);
  rebuildSolver();
  if (!sameProblem) reset();
}

void LipContactPlanner::reset() {
  previousPlan_.reset();
  previousAssignment_.clear();
}

std::string LipContactPlanner::getFormulationSummary() const {
  std::ostringstream out;
  const PlannerSettings& p = config_.planner;
  out << "planner: " << p.numNodes << " nodes x " << p.dt << " s, commit " << p.commitTime << " s, at most " << p.maxBranchAndBoundNodes
      << " relaxations / " << p.maxSolveTime << " s, " << (p.runInBackgroundThread ? "background thread" : "synchronous") << " at "
      << p.planningFrequency << " Hz\n";
  out << problem_.summary();
  out << "search (" << searchStages_.size() << "):\n";
  for (size_t i = 0; i < searchStages_.size(); ++i) {
    out << "  - " << searchStages_.nameAt(i) << ": " << searchStages_.at(i).describe() << "\n";
  }
  return out.str();
}

std::string LipContactPlanner::formulationSummary(const ContactPlanningConfig& config) {
  return LipContactPlanner(config).getFormulationSummary();
}

int LipContactPlanner::previousPlanShift(const ContactPlannerInput& input) const {
  if (!previousPlan_ || !previousPlan_->valid || previousAssignment_.empty()) return -1;
  const ContactPlan& prev = *previousPlan_;
  if (prev.numIntervals() != config_.planner.numNodes || std::abs(prev.dt - config_.planner.dt) > 1e-9) return -1;
  const int shift = static_cast<int>(std::lround((input.time - prev.startTime) / config_.planner.dt));
  if (shift < 0 || shift >= config_.planner.numNodes) return -1;
  return shift;
}

std::vector<MiqpBinaryVariable> LipContactPlanner::binaryVariables() const {
  return problem_.binaryVariables(config_.planner.numNodes);
}

MiqpAssignment LipContactPlanner::initialAssignment(const ContactPlannerInput& input) const {
  const int N = config_.planner.numNodes;
  MiqpAssignment assignment(static_cast<size_t>(kBinariesPerNode * N), kMiqpFree);
  const int numCommitted = std::min(static_cast<int>(input.committedContacts.size()), N);
  for (int k = 0; k < numCommitted; ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      assignment[static_cast<size_t>(contactBinaryIndex(k, foot))] = input.committedContacts[static_cast<size_t>(k)][foot] ? 1 : 0;
    }
  }
  return assignment;
}

ContactLogicState LipContactPlanner::makeLogicState(const ContactPlannerInput& input) const {
  const int shift = previousPlanShift(input);
  return ContactLogicState::make(input, config_, shift, shift >= 0 ? &previousAssignment_ : nullptr);
}

bool LipContactPlanner::propagate(const ContactPlannerInput& input, MiqpAssignment& assignment) const {
  return problem_.propagate(makeLogicState(input), assignment);
}

scalar_t LipContactPlanner::assignmentCost(const ContactPlannerInput& input, const MiqpAssignment& assignment) const {
  return problem_.assignmentCost(makeLogicState(input), assignment);
}

HeadingNominal LipContactPlanner::defaultNominal(const ContactPlannerInput& input) const {
  const int N = config_.planner.numNodes;
  HeadingNominal nominal;
  nominal.heading.resize(static_cast<size_t>(N) + 1);
  nominal.com.resize(static_cast<size_t>(N) + 1);
  nominal.feet.resize(static_cast<size_t>(N) + 1);
  const int shift = previousPlanShift(input);
  const bool fromPrevious =
      shift >= 0 && previousPlan_ && previousPlan_->hasHeading() && previousPlan_->heading.size() == static_cast<size_t>(N + 1) &&
      previousPlan_->comPosition.size() == static_cast<size_t>(N + 1) && previousPlan_->footholds.size() == static_cast<size_t>(N + 1);
  // The measured heading arrives wrapped to [-pi, pi] (an atan2 of the base quaternion), the previous plan's heading is
  // whatever branch that plan started on. Across a crossing of +-pi the two differ by 2 pi, and a nominal on the old
  // branch made every first-order frame term g (theta - theta_n) worth g 2 pi (over a metre on the step width) and
  // aimed the foot yaw tracking a full turn away. One offset moves the whole previous heading trajectory onto the branch
  // of the measurement, which keeps it continuous whatever the plan turns through.
  scalar_t branchOffset = 0.0;
  if (fromPrevious) {
    const scalar_t previousNow = previousPlan_->heading[static_cast<size_t>(std::min(shift, N))];
    branchOffset = moduloAngleWithReference(previousNow, input.heading) - previousNow;
  }
  for (int k = 0; k <= N; ++k) {
    const size_t i = static_cast<size_t>(k);
    if (fromPrevious) {
      const size_t source = static_cast<size_t>(std::min(k + shift, N));
      nominal.heading[i] = previousPlan_->heading[source] + branchOffset;
      nominal.com[i] = previousPlan_->comPosition[source];
      nominal.feet[i] = previousPlan_->footholds[source];
    } else {
      const scalar_t t = static_cast<scalar_t>(k) * config_.planner.dt;
      nominal.heading[i] = input.heading + input.headingRateCommand * t;
      nominal.com[i] = input.comPosition + input.comVelocity * t;
      nominal.feet[i] = input.footPositions;
    }
  }
  return nominal;
}

HeadingNominal LipContactPlanner::nominalFromSolution(const Layout& layout, const OcpQpSolution& solution) {
  return ocs2::humanoid::nominalFromSolution(layout, solution.x);
}

ContactPlanningContext LipContactPlanner::makeContext(const ContactPlannerInput& input,
                                                      const HeadingNominal& nominal,
                                                      scalar_t dtOverride) const {
  const int N = config_.planner.numNodes;
  const Layout& layout = problem_.layout();
  if (layout.hasHeading && (nominal.heading.size() < static_cast<size_t>(N + 1) || nominal.feet.size() < static_cast<size_t>(N + 1) ||
                            nominal.com.size() < static_cast<size_t>(N + 1))) {
    throw std::invalid_argument("[LipContactPlanner] the nominal heading trajectory must cover every node");
  }
  if (layout.hasHeading && !config_.hasModelParameters()) {
    throw std::invalid_argument(
        "[LipContactPlanner] the heading model needs the model-derived parameters (torque limits, foot yaw bounds); apply "
        "ContactPlanningModelParameters to the configuration first");
  }
  ContactPlanningContext ctx;
  ctx.input = &input;
  ctx.layout = &layout;
  ctx.nominal = &nominal;
  ctx.config = &config_;
  ctx.previousPlanShift = previousPlanShift(input);
  ctx.previousPlan = ctx.previousPlanShift >= 0 ? &*previousPlan_ : nullptr;
  ctx.yawInertia = layout.hasHeading ? yawInertia(input) : 1.0;
  ctx.dt = dtOverride > 0.0 ? dtOverride : config_.planner.dt;
  ctx.numNodes = N;
  ctx.omega = config_.omega();
  ctx.bigM = config_.shared.bigM;
  ctx.computeAxes();
  return ctx;
}

OcpQpProblem LipContactPlanner::buildProblem(const ContactPlannerInput& input) const {
  const HeadingNominal nominal = defaultNominal(input);
  return buildProblem(input, nominal);
}

OcpQpProblem LipContactPlanner::buildProblem(const ContactPlannerInput& input, const HeadingNominal& nominal) const {
  return problem_.assemble(makeContext(input, nominal));
}

ContactPlan LipContactPlanner::decode(const ContactPlannerInput& input, const ContactPlanningContext& ctx, const MiqpResult& result) const {
  ContactPlan plan;
  plan.startTime = input.time;
  plan.dt = config_.planner.dt;
  plan.committedUntil = input.committedUntil;
  plan.yaw = input.yaw;
  plan.numBranchAndBoundNodes = statistics_.numBranchAndBoundRelaxations + statistics_.numLocalSearchQps;
  plan.solveTime = statistics_.branchAndBoundTime + statistics_.localSearchTime;
  // A search that ended without an incumbent (an infeasible root) is exhausted, but there is no optimal plan to report.
  plan.optimal = result.optimal && result.hasIncumbent;
  plan.nodeLimitHit = result.nodeLimitHit;
  plan.timeLimitHit = result.timeLimitHit;
  if (!result.hasIncumbent) {
    return plan;
  }
  plan.valid = true;
  plan.objective = result.incumbentObjective;
  problem_.decode(ctx, result, plan);
  return plan;
}

ContactPlan LipContactPlanner::plan(const ContactPlannerInput& input) {
  const auto start = Clock::now();
  statistics_ = Statistics();
  const HeadingNominal nominal = defaultNominal(input);
  ContactPlanningContext ctx;
  try {
    ctx = makeContext(input, nominal);
    lastProblem_ = problem_.assemble(ctx);
  } catch (const std::exception& e) {
    std::cerr << "[LipContactPlanner] cannot build the problem: " << e.what() << std::endl;
    lastResult_ = MiqpResult();
    ctx.numNodes = config_.planner.numNodes;
    return decode(input, ctx, lastResult_);
  }
  const std::vector<MiqpBinaryVariable> binaries = binaryVariables();
  const MiqpAssignment initial = initialAssignment(input);
  const ContactLogicState logicState = makeLogicState(input);
  const MiqpPropagateFn propagateFn = [this, &logicState](MiqpAssignment& assignment) {
    return problem_.propagate(logicState, assignment);
  };
  const MiqpAssignmentCostFn costFn = [this, &logicState](const MiqpAssignment& assignment) {
    return problem_.assignmentCost(logicState, assignment);
  };

  // Stages before the search: the warm start and the solver's own heuristics.
  SearchSetup setup;
  setup.previousPlan = ctx.previousPlan;
  setup.previousAssignment = ctx.previousPlanShift >= 0 ? &previousAssignment_ : nullptr;
  setup.previousPlanShift = ctx.previousPlanShift;
  setup.numNodes = config_.planner.numNodes;
  setup.miqpSettings = miqp_->getSettings();
  setup.miqpSettings.useDivingHeuristic = false;
  for (const auto& stage : searchStages_) stage->beforeSearch(setup);
  miqp_->setSettings(setup.miqpSettings);

  try {
    lastResult_ = miqp_->solve(lastProblem_, binaries, initial, propagateFn, setup.warmStart ? &*setup.warmStart : nullptr, costFn);
  } catch (const std::exception& e) {
    std::cerr << "[LipContactPlanner] solver failure: " << e.what() << std::endl;
    lastResult_ = MiqpResult();
  }
  statistics_.numBranchAndBoundRelaxations = lastResult_.numNodes;
  statistics_.totalQpIterations = lastResult_.totalQpIterations;
  statistics_.branchAndBoundTime = lastResult_.solveTime;

  // Stages after the search: refinements of the incumbent.
  SearchRun run;
  run.input = &input;
  run.config = &config_;
  run.layout = &problem_.layout();
  run.binaries = &binaries;
  run.initialAssignment = &initial;
  run.propagate = &propagateFn;
  run.assignmentCost = &costFn;
  run.miqp = miqp_.get();
  run.problem = &lastProblem_;
  run.result = &lastResult_;
  run.statistics = &statistics_;
  run.assembleWithNominal = [this, &input](const HeadingNominal& relinearised) {
    return problem_.assemble(makeContext(input, relinearised));
  };
  run.assembleWithGrid = [this, &input, &nominal](scalar_t nodeDuration) {
    return problem_.assemble(makeContext(input, nominal, nodeDuration));
  };
  run.start = start;
  run.verbose = config_.planner.verbose;
  for (size_t i = 0; i < searchStages_.size(); ++i) {
    try {
      searchStages_.at(i).afterSearch(run);
    } catch (const std::exception& e) {
      std::cerr << "[LipContactPlanner] search stage '" << searchStages_.nameAt(i) << "' failed: " << e.what() << std::endl;
    }
  }

  ContactPlan plan = decode(input, ctx, lastResult_);
  // A stage that re-timed the grid (cadence_stretch) reports the node duration the plan is to carry. ContactPlan holds
  // its own dt, so the whole plan - event times, horizon, foothold lookup - follows from this one field.
  if (run.chosenDt > 0.0) plan.dt = run.chosenDt;
  if (config_.planner.verbose) {
    std::cout << "[LipContactPlanner] valid=" << plan.valid << " objective=" << plan.objective
              << " relaxations=" << statistics_.numBranchAndBoundRelaxations << " localSearchQps=" << statistics_.numLocalSearchQps
              << " time=" << elapsedSeconds(start) << "s optimal=" << plan.optimal << std::endl;
  }
  if (plan.valid) {
    previousPlan_ = plan;
    previousAssignment_ = lastResult_.assignment;
  }
  return plan;
}

}  // namespace ocs2::humanoid
