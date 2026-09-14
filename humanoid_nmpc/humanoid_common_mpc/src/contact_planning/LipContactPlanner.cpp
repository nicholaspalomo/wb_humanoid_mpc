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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kLooseBound = kOcpQpInfiniteBound;
constexpr scalar_t kInputRegularization = 1.0e-6;
constexpr scalar_t kStateRegularization = 1.0e-8;

using Coefficients = std::vector<std::pair<int, scalar_t>>;
using Clock = std::chrono::steady_clock;

scalar_t elapsedSeconds(const Clock::time_point& start) {
  return std::chrono::duration<scalar_t>(Clock::now() - start).count();
}

/** Collects general constraint rows of one stage and writes them into the OcpQpStage. */
class RowBuilder {
 public:
  RowBuilder(int nx, int nu) : nx_(nx), nu_(nu) {}

  void add(const Coefficients& xCoefficients, const Coefficients& uCoefficients, scalar_t lower, scalar_t upper, bool soft) {
    vector_t cRow = vector_t::Zero(nx_);
    vector_t dRow = vector_t::Zero(nu_);
    for (const auto& [index, value] : xCoefficients) cRow(index) += value;
    for (const auto& [index, value] : uCoefficients) dRow(index) += value;
    cRows_.push_back(std::move(cRow));
    dRows_.push_back(std::move(dRow));
    lower_.push_back(lower);
    upper_.push_back(upper);
    if (soft) softRows_.push_back(static_cast<int>(cRows_.size()) - 1);
  }

  void writeTo(OcpQpStage& stage, scalar_t slackQuadraticWeight, scalar_t slackLinearWeight) const {
    const int ng = static_cast<int>(cRows_.size());
    stage.C.resize(ng, nx_);
    stage.D.resize(ng, nu_);
    stage.lg.resize(ng);
    stage.ug.resize(ng);
    for (int i = 0; i < ng; ++i) {
      stage.C.row(i) = cRows_[i].transpose();
      stage.D.row(i) = dRows_[i].transpose();
      stage.lg(i) = lower_[i];
      stage.ug(i) = upper_[i];
    }
    stage.softGeneralIndices = softRows_;
    const int ns = static_cast<int>(softRows_.size());
    stage.Zl = vector_t::Constant(ns, slackQuadraticWeight);
    stage.Zu = vector_t::Constant(ns, slackQuadraticWeight);
    stage.zl = vector_t::Constant(ns, slackLinearWeight);
    stage.zu = vector_t::Constant(ns, slackLinearWeight);
  }

 private:
  int nx_, nu_;
  std::vector<vector_t> cRows_, dRows_;
  std::vector<scalar_t> lower_, upper_;
  std::vector<int> softRows_;
};

/** Adds weight * (sum_i xc_i x_i + sum_j uc_j u_j + offset)^2 to the stage cost (0.5 x'Qx convention). */
void addQuadraticResidual(
    OcpQpStage& stage, const Coefficients& xCoefficients, const Coefficients& uCoefficients, scalar_t offset, scalar_t weight) {
  if (weight <= 0.0) return;
  vector_t lx = vector_t::Zero(stage.numStates());
  vector_t lu = vector_t::Zero(stage.numInputs());
  for (const auto& [index, value] : xCoefficients) lx(index) += value;
  for (const auto& [index, value] : uCoefficients) lu(index) += value;
  stage.Q.noalias() += 2.0 * weight * lx * lx.transpose();
  stage.q.noalias() += 2.0 * weight * offset * lx;
  if (stage.numInputs() > 0) {
    stage.R.noalias() += 2.0 * weight * lu * lu.transpose();
    stage.S.noalias() += 2.0 * weight * lu * lx.transpose();
    stage.r.noalias() += 2.0 * weight * offset * lu;
  }
}

/** HPIPM settings for branch-and-bound relaxations: moderate accuracy is enough for bounding and integrality decisions. */
OcpQpHpipmSolver::Settings relaxationQpSettings(const ContactPlanningConfig& config) {
  OcpQpHpipmSolver::Settings qpSettings;
  qpSettings.iterMax = config.maxQpIterations;
  qpSettings.hpipmMode = 2;  // BALANCE
  qpSettings.mu0 = 1e1;
  qpSettings.tolStat = 1e-5;
  qpSettings.tolEq = 1e-6;
  qpSettings.tolIneq = 1e-6;
  qpSettings.tolComp = 1e-6;
  return qpSettings;
}

}  // namespace

LipContactPlanner::Layout LipContactPlanner::makeLayout(const ContactPlanningConfig& config) {
  Layout layout;
  if (!config.useAcomDynamics) return layout;
  layout.hasHeading = true;
  layout.heading = STATE_DIM;
  layout.headingRate = STATE_DIM + 1;
  layout.footYaw0 = STATE_DIM + 2;
  layout.nx = STATE_DIM + 2 + static_cast<int>(N_CONTACTS);
  layout.yawTorque0 = INPUT_DIM;
  layout.footYawDelta0 = INPUT_DIM + static_cast<int>(N_CONTACTS);
  layout.nu = INPUT_DIM + 2 * static_cast<int>(N_CONTACTS);
  return layout;
}

scalar_t LipContactPlanner::yawInertia(const ContactPlannerInput& input) const {
  if (input.yawInertia <= 0.0) {
    throw std::invalid_argument("[LipContactPlanner] the heading model needs a positive yaw inertia in the input (from the robot model)");
  }
  return input.yawInertia;
}

LipContactPlanner::LipContactPlanner(ContactPlanningConfig config) : config_(std::move(config)) {
  config_.validate();
  layout_ = makeLayout(config_);
  rebuildSolver();
}

void LipContactPlanner::rebuildSolver() {
  MiqpSettings miqpSettings;
  miqpSettings.maxNodes = config_.maxBranchAndBoundNodes;
  miqpSettings.maxSolveTime = config_.maxSolveTime;
  miqpSettings.verbose = config_.verbose;
  miqp_ = std::make_unique<MixedIntegerOcpQp>(relaxationQpSettings(config_), miqpSettings);
}

void LipContactPlanner::setConfig(const ContactPlanningConfig& config) {
  config.validate();
  config_ = config;
  layout_ = makeLayout(config_);
  rebuildSolver();
  reset();
}

void LipContactPlanner::reset() {
  previousPlan_.reset();
  previousAssignment_.clear();
}

int LipContactPlanner::initialPhaseNodes(const ContactPlannerInput& input, size_t foot, bool roundUp) const {
  const int cap = 2 * config_.numNodes;
  const scalar_t nodes = std::max(0.0, input.phaseElapsedTime[foot]) / config_.dt;
  // Rounding to the nearest node let both limits be violated by up to half a node: a swing 0.16 s old counted as two
  // nodes and could end one node later at 0.26 s against a 0.3 s minimum.
  const int elapsed = static_cast<int>(roundUp ? std::ceil(nodes - 1e-9) : std::floor(nodes + 1e-9));
  return std::clamp(elapsed, 0, cap);
}

std::vector<MiqpBinaryVariable> LipContactPlanner::binaryVariables() const {
  std::vector<MiqpBinaryVariable> binaries;
  binaries.reserve(kBinariesPerNode * config_.numNodes);
  for (int k = 0; k < config_.numNodes; ++k) {
    binaries.push_back({k, CL});
    binaries.push_back({k, CR});
  }
  return binaries;
}

MiqpAssignment LipContactPlanner::initialAssignment(const ContactPlannerInput& input) const {
  MiqpAssignment assignment(kBinariesPerNode * config_.numNodes, kMiqpFree);
  const int numCommitted = std::min(static_cast<int>(input.committedContacts.size()), config_.numNodes);
  for (int k = 0; k < numCommitted; ++k) {
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      assignment[contactBinaryIndex(k, foot)] = input.committedContacts[k][foot] ? 1 : 0;
    }
  }
  return assignment;
}

int LipContactPlanner::previousPlanShift(const ContactPlannerInput& input) const {
  if (!previousPlan_ || !previousPlan_->valid || previousAssignment_.empty()) return -1;
  const ContactPlan& prev = *previousPlan_;
  if (prev.numIntervals() != config_.numNodes || std::abs(prev.dt - config_.dt) > 1e-9) return -1;
  const int shift = static_cast<int>(std::lround((input.time - prev.startTime) / config_.dt));
  if (shift < 0 || shift >= config_.numNodes) return -1;
  return shift;
}

scalar_t LipContactPlanner::assignmentCost(const ContactPlannerInput& input, const MiqpAssignment& assignment) const {
  int numInconsistent = 0;
  const int shift = previousPlanShift(input);
  if (shift >= 0 && config_.planConsistencyCost > 0.0) {
    for (int k = 0; k < config_.numNodes; ++k) {
      const int source = std::min(k + shift, config_.numNodes - 1);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        const std::int8_t value = assignment[contactBinaryIndex(k, foot)];
        if (value != kMiqpFree && value != previousAssignment_[contactBinaryIndex(source, foot)]) ++numInconsistent;
      }
    }
  }
  int numSwitches = 0;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    int previous = input.contacts[foot] ? 1 : 0;
    for (int k = 0; k < config_.numNodes; ++k) {
      const std::int8_t value = assignment[contactBinaryIndex(k, foot)];
      if (value == kMiqpFree) {
        continue;
      }
      if (previous >= 0 && value != previous) ++numSwitches;
      previous = value;
    }
  }
  return config_.contactSwitchCost * static_cast<scalar_t>(numSwitches) +
         config_.planConsistencyCost * static_cast<scalar_t>(numInconsistent);
}

bool LipContactPlanner::propagate(const ContactPlannerInput& input, MiqpAssignment& a) const {
  const int N = config_.numNodes;
  const int nSwingMin = config_.minSwingNodes();
  const int nSwingMax = config_.maxSwingNodes();
  const int nContactMin = config_.minContactNodes();
  const int nContactMax = config_.maxContactNodes();
  const int numCommitted = std::min(static_cast<int>(input.committedContacts.size()), N);

  bool ok = true;
  const auto fix = [&](int index, std::int8_t value) {
    if (a[index] == kMiqpFree) {
      a[index] = value;
      return true;  // changed
    }
    if (a[index] != value) ok = false;
    return false;
  };

  // Nodes at which a foot is held in contact by the minimum double support after the other foot's touch-down, along the
  // fixed prefix. The maximum-contact rule yields to it (and to the no-flight rule): a foot that is overdue to lift but
  // cannot, because the other foot is in the air or has just landed, lifts at the first node where it can instead of
  // making the horizon infeasible. Without this, a standing start with maxContactDuration set (both feet overdue) had no
  // feasible assignment at all, since both feet were forced to lift at the same node.
  const int nDoubleSupportHold = config_.minDoubleSupportNodes();
  std::array<std::vector<bool>, N_CONTACTS> heldByDoubleSupport;
  for (auto& hold : heldByDoubleSupport) hold.assign(N, false);
  const auto computeDoubleSupportHold = [&]() {
    for (auto& hold : heldByDoubleSupport) std::fill(hold.begin(), hold.end(), false);
    if (nDoubleSupportHold <= 0) return;
    std::array<std::int8_t, N_CONTACTS> kappa;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) kappa[foot] = input.contacts[foot] ? 1 : 0;
    int lastTouchDown = -1000;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (input.contacts[foot] && input.contacts[1 - foot]) lastTouchDown = std::max(lastTouchDown, -initialPhaseNodes(input, foot));
    }
    for (int k = 0; k < N; ++k) {
      int latestTouchDown = lastTouchDown;
      bool allFixed = true;
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        const std::int8_t value = a[contactBinaryIndex(k, foot)];
        allFixed = allFixed && value != kMiqpFree;
        if (kappa[foot] == 0 && value == 1) latestTouchDown = k;
      }
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        heldByDoubleSupport[foot][k] = kappa[foot] == 1 && k < latestTouchDown + nDoubleSupportHold;
      }
      if (!allFixed) break;
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        const std::int8_t value = a[contactBinaryIndex(k, foot)];
        if (kappa[foot] == 0 && value == 1) lastTouchDown = k;
        kappa[foot] = value;
      }
    }
  };
  // Contact state of a foot just before `node` along the fixed prefix: 1 in contact, 0 in the air, -1 unknown.
  const auto stateBefore = [&](size_t foot, int node) {
    std::int8_t kappa = input.contacts[foot] ? 1 : 0;
    for (int k = 0; k < node; ++k) {
      const std::int8_t value = a[contactBinaryIndex(k, foot)];
      if (value == kMiqpFree) return -1;
      kappa = value;
    }
    return static_cast<int>(kappa);
  };
  // Elapsed contact nodes (rounded up) of every foot along the fixed prefix, to decide which of two overdue feet lifts.
  const auto contactAge = [&](size_t foot, int node) {
    std::int8_t kappa = input.contacts[foot] ? 1 : 0;
    int tau = initialPhaseNodes(input, foot, true);
    for (int k = 0; k < node; ++k) {
      const std::int8_t value = a[contactBinaryIndex(k, foot)];
      if (value == kMiqpFree) return -1;
      if (value == kappa) {
        ++tau;
      } else {
        tau = 1;
        kappa = value;
      }
    }
    return kappa == 1 ? tau : 0;
  };

  for (int iteration = 0; iteration < 4 * N && ok; ++iteration) {
    bool changed = false;
    computeDoubleSupportHold();

    // Per-foot forward pass along the fixed prefix: minimum and maximum phase durations.
    for (size_t foot = 0; foot < N_CONTACTS && ok; ++foot) {
      std::int8_t kappa = input.contacts[foot] ? 1 : 0;
      // The phase active at planning time has lasted a non-integer number of nodes: count it rounded down against the
      // minimum duration and rounded up against the maximum, so that neither limit is violated by the grid. Once the
      // phase switched inside the horizon both counts are exact and coincide.
      int tauMin = initialPhaseNodes(input, foot, false);
      int tauMax = initialPhaseNodes(input, foot, true);
      for (int k = 0; k < N; ++k) {
        const int index = contactBinaryIndex(k, foot);
        if (k >= numCommitted) {
          bool mustStay = (kappa == 1 && tauMin < nContactMin) || (kappa == 0 && tauMin < nSwingMin);
          bool mustSwitch = (kappa == 0 && tauMax >= nSwingMax) || (kappa == 1 && nContactMax > 0 && tauMax >= nContactMax);
          // A swing that starts this late cannot reach its minimum duration before the horizon ends. Letting it start
          // leaves the plan ending mid-swing, and ContactPlan::toModeSchedule() then closes the schedule with a
          // touch-down at endTime(), handing the controller a swing far shorter than minSwingDuration (measured: 0.1 s
          // against a 0.3 s minimum), whose swing height is scaled down in turn. Keep the foot down instead and let the
          // next plan start the step; the maximum contact duration yields to this, as it does to the no-flight rule.
          if (kappa == 1 && k + nSwingMin > N) {
            mustStay = true;
            mustSwitch = false;
          }
          if (mustStay && mustSwitch) {
            // The grid cannot honour both limits for this phase (they are less than a node apart); decide by the nearest
            // node, as before, instead of declaring the whole horizon infeasible.
            const int tauNearest = static_cast<int>(std::lround(std::max(0.0, input.phaseElapsedTime[foot]) / config_.dt)) +
                                   (tauMin - initialPhaseNodes(input, foot, false));
            mustStay = (kappa == 1 && tauNearest < nContactMin) || (kappa == 0 && tauNearest < nSwingMin);
            mustSwitch = (kappa == 0 && tauNearest >= nSwingMax) || (kappa == 1 && nContactMax > 0 && tauNearest >= nContactMax);
            if (mustStay && mustSwitch) return false;
          }
          if (mustSwitch && kappa == 1) {
            // Overdue to lift. Yield to the rules that can make lifting impossible right now, and when both feet are
            // overdue at once let the one that has stood longer (or that did not swing last) go first.
            const size_t other = 1 - foot;
            const std::int8_t otherValue = a[contactBinaryIndex(k, other)];
            // The other foot has to be known to support: in contact at this node, or in contact before it with its
            // value here still free (it can then only stay, since this foot lifting forbids it to lift too). If it is
            // in the air and free it might land right here, and lifting now would violate the double support.
            const bool otherSupports = otherValue == 1 || (otherValue == kMiqpFree && stateBefore(other, k) == 1);
            const bool held = heldByDoubleSupport[foot][k];
            const int otherAge = contactAge(other, k);
            const bool otherOverdue = otherValue != 0 && nContactMax > 0 && otherAge >= nContactMax;
            bool thisGoesFirst = true;
            if (otherOverdue) {
              if (input.lastSwungFoot >= 0) {
                thisGoesFirst = static_cast<size_t>(input.lastSwungFoot) != foot;
              } else {
                const int thisAge = contactAge(foot, k);
                thisGoesFirst = thisAge > otherAge || (thisAge == otherAge && foot < other);
              }
            }
            if (!otherSupports || held || !thisGoesFirst) mustSwitch = false;
          }
          if (mustStay) changed |= fix(index, kappa);
          if (mustSwitch) changed |= fix(index, static_cast<std::int8_t>(1 - kappa));
          if (!ok) return false;
        }
        if (a[index] == kMiqpFree) break;
        if (a[index] == kappa) {
          ++tauMin;
          ++tauMax;
        } else {
          tauMin = tauMax = 1;
          kappa = a[index];
        }
      }
    }

    // Cross constraints per node: no flight phase.
    for (int k = 0; k < N && ok; ++k) {
      const int cL = contactBinaryIndex(k, 0), cR = contactBinaryIndex(k, 1);
      if (a[cL] == 0 && a[cR] == 0) return false;
      if (a[cL] == 0) changed |= fix(cR, 1);
      if (a[cR] == 0) changed |= fix(cL, 1);
    }

    // Minimum double support: after a touch-down at node k the other foot may not lift off before node k + n_ds.
    const int nDoubleSupport = config_.minDoubleSupportNodes();
    if (nDoubleSupport > 0 && ok) {
      std::array<std::int8_t, N_CONTACTS> kappa{input.contacts[0] ? std::int8_t(1) : std::int8_t(0),
                                                input.contacts[1] ? std::int8_t(1) : std::int8_t(0)};
      int lastTouchDown = -1000;
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        // A double support already in progress counts from the elapsed time of the shorter contact phase.
        if (input.contacts[foot] && input.contacts[1 - foot]) {
          lastTouchDown = std::max(lastTouchDown, -initialPhaseNodes(input, foot));
        }
      }
      for (int k = 0; k < N && ok; ++k) {
        // A touch-down decided at this very node already binds the other foot at this node: lifting it here would turn
        // single support on one foot into single support on the other with no double support at all, which is the
        // weight transfer the rule forbids. Counting it only after the node let that instantaneous switch through.
        int latestTouchDown = lastTouchDown;
        for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
          if (kappa[foot] == 0 && a[contactBinaryIndex(k, foot)] == 1) latestTouchDown = k;
        }
        bool allFixed = true;
        for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
          const int index = contactBinaryIndex(k, foot);
          if (kappa[foot] == 1 && k < latestTouchDown + nDoubleSupport && k >= numCommitted) {
            if (a[index] == 0) return false;
            changed |= fix(index, 1);
          }
          allFixed = allFixed && a[index] != kMiqpFree;
        }
        if (!allFixed || !ok) break;
        for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
          const std::int8_t value = a[contactBinaryIndex(k, foot)];
          if (kappa[foot] == 0 && value == 1) lastTouchDown = k;
          kappa[foot] = value;
        }
      }
    }

    // Foot alternation along the fixed prefix: a foot may not lift off twice without the other foot swinging in between.
    if (config_.enforceAlternatingFeet && ok) {
      int lastSwung = input.lastSwungFoot;
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        if (!input.contacts[foot]) lastSwung = static_cast<int>(foot);
      }
      std::array<std::int8_t, N_CONTACTS> kappa{input.contacts[0] ? std::int8_t(1) : std::int8_t(0),
                                                input.contacts[1] ? std::int8_t(1) : std::int8_t(0)};
      for (int k = 0; k < N && ok; ++k) {
        bool allFixed = true;
        for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
          const int index = contactBinaryIndex(k, foot);
          if (kappa[foot] == 1) {
            if (a[index] == 0) {
              if (lastSwung == static_cast<int>(foot)) return false;
              lastSwung = static_cast<int>(foot);
            } else if (a[index] == kMiqpFree && lastSwung == static_cast<int>(foot)) {
              changed |= fix(index, 1);
            }
          }
          allFixed = allFixed && a[index] != kMiqpFree;
        }
        if (!allFixed || !ok) break;
        for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
          kappa[foot] = a[contactBinaryIndex(k, foot)];
        }
      }
    }
    if (!changed) break;
  }
  return ok;
}

LipContactPlanner::HeadingNominal LipContactPlanner::defaultNominal(const ContactPlannerInput& input) const {
  const int N = config_.numNodes;
  HeadingNominal nominal;
  nominal.heading.resize(N + 1);
  nominal.com.resize(N + 1);
  nominal.feet.resize(N + 1);
  const int shift = previousPlanShift(input);
  const bool fromPrevious =
      shift >= 0 && previousPlan_ && previousPlan_->hasHeading() && previousPlan_->heading.size() == static_cast<size_t>(N + 1) &&
      previousPlan_->comPosition.size() == static_cast<size_t>(N + 1) && previousPlan_->footholds.size() == static_cast<size_t>(N + 1);
  for (int k = 0; k <= N; ++k) {
    if (fromPrevious) {
      const int source = std::min(k + shift, N);
      nominal.heading[k] = previousPlan_->heading[source];
      nominal.com[k] = previousPlan_->comPosition[source];
      nominal.feet[k] = previousPlan_->footholds[source];
    } else {
      const scalar_t t = static_cast<scalar_t>(k) * config_.dt;
      nominal.heading[k] = input.heading + input.headingRateCommand * t;
      nominal.com[k] = input.comPosition + input.comVelocity * t;
      nominal.feet[k] = input.footPositions;
    }
  }
  return nominal;
}

LipContactPlanner::HeadingNominal LipContactPlanner::nominalFromSolution(const Layout& layout, const OcpQpSolution& solution) {
  HeadingNominal nominal;
  const size_t n = solution.x.size();
  nominal.heading.resize(n);
  nominal.com.resize(n);
  nominal.feet.resize(n);
  for (size_t k = 0; k < n; ++k) {
    const vector_t& x = solution.x[k];
    nominal.heading[k] = layout.hasHeading ? x(layout.heading) : 0.0;
    nominal.com[k] = x.segment<2>(CX);
    nominal.feet[k][0] = x.segment<2>(PLX);
    nominal.feet[k][1] = x.segment<2>(PRX);
  }
  return nominal;
}

OcpQpProblem LipContactPlanner::buildProblem(const ContactPlannerInput& input) const {
  return buildProblem(input, defaultNominal(input));
}

OcpQpProblem LipContactPlanner::buildProblem(const ContactPlannerInput& input, const HeadingNominal& nominal) const {
  const ContactPlanningConfig& cfg = config_;
  const Layout& L = layout_;
  const bool hasHeading = L.hasHeading;
  if (hasHeading &&
      (nominal.heading.size() < static_cast<size_t>(cfg.numNodes + 1) || nominal.feet.size() < static_cast<size_t>(cfg.numNodes + 1) ||
       nominal.com.size() < static_cast<size_t>(cfg.numNodes + 1))) {
    throw std::invalid_argument("[LipContactPlanner] the nominal heading trajectory must cover every node");
  }
  if (hasHeading && !cfg.hasModelParameters()) {
    throw std::invalid_argument(
        "[LipContactPlanner] the heading model needs the model-derived parameters (torque limits, foot yaw bounds); apply "
        "ContactPlanningModelParameters to the configuration first");
  }
  const scalar_t inertia = hasHeading ? yawInertia(input) : 1.0;
  const scalar_t torsion = cfg.torsionalFrictionTorque;
  const scalar_t couple = cfg.doubleSupportYawCouple;
  constexpr scalar_t kYawBigM = 2.0 * M_PI;
  const int N = cfg.numNodes;
  const scalar_t dt = cfg.dt;
  const scalar_t omega = cfg.omega();
  const scalar_t ch = std::cosh(omega * dt);
  const scalar_t sh = std::sinh(omega * dt);
  const scalar_t M = cfg.bigM;

  // Yaw-aligned constraint frame: components of the world-frame unit vectors of the local x and y axes. With the heading
  // model the frame of node k is the nominal heading of node k, and every projection e_a(theta) . d gets its first-order
  // heading term d e_a / d theta (theta_n) . d_n (theta - theta_n), so that the heading and the footholds are decided
  // together; d e_x / d theta = e_y and d e_y / d theta = -e_x.
  const auto axesAt = [&](int k) -> std::array<vector2_t, 2> {
    const scalar_t theta = hasHeading ? nominal.heading[k] : input.yaw;
    return {vector2_t(std::cos(theta), std::sin(theta)), vector2_t(-std::sin(theta), std::cos(theta))};
  };
  // Coefficient on the heading state and the constant offset of the first-order term of e_axis(theta) . d around
  // (theta_n, d_n): g (theta - theta_n) = g theta - g theta_n.
  const auto frameTerm = [&](int axis, int k, const vector2_t& dn) -> std::pair<scalar_t, scalar_t> {
    if (!hasHeading) return {0.0, 0.0};
    const std::array<vector2_t, 2> ax = axesAt(k);
    const vector2_t de = (axis == 0) ? ax[1] : vector2_t(-ax[0]);
    const scalar_t g = de.dot(dn);
    return {g, -g * nominal.heading[k]};
  };
  const std::array<scalar_t, 2> zmpHalfWidth{cfg.zmpHalfWidthX, cfg.zmpHalfWidthY};
  const std::array<int, 2> pIndex{PLX, PRX};  // x index of each foot's position; y follows
  const std::array<int, 2> dIndex{DLX, DRX};
  const std::array<int, 2> cIndex{CL, CR};
  const int previousShift = previousPlanShift(input);

  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(L.nx);
  problem.x0.segment<2>(CX) = input.comPosition;
  problem.x0.segment<2>(VX) = input.comVelocity;
  problem.x0.segment<2>(PLX) = input.footPositions[0];
  problem.x0.segment<2>(PRX) = input.footPositions[1];
  if (hasHeading) {
    problem.x0(L.heading) = input.heading;
    problem.x0(L.headingRate) = input.headingRate;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) problem.x0(L.footYaw(foot)) = input.footYaws[foot];
  }

  problem.stages.resize(N + 1);
  for (int k = 0; k <= N; ++k) {
    const bool terminal = (k == N);
    OcpQpStage& s = problem.stages[k];
    s = OcpQpStage::Zero(L.nx, terminal ? 0 : L.nu, !terminal);
    s.Q.diagonal().array() += kStateRegularization;
    const std::array<vector2_t, 2> axes = axesAt(k);
    const vector2_t& ey = axes[1];

    // Foothold consistency: pull the feet towards the previous plan, shifted to the current time (only matters while a
    // foot moves, the contact constraints pin it otherwise).
    if (previousShift >= 0 && cfg.previousFootholdWeight > 0.0) {
      const int source = std::min(k + previousShift, static_cast<int>(previousPlan_->footholds.size()) - 1);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        for (int axis = 0; axis < 2; ++axis) {
          addQuadraticResidual(s, {{pIndex[foot] + axis, 1.0}}, {}, -previousPlan_->footholds[source][foot](axis),
                               cfg.previousFootholdWeight);
        }
      }
    }
    // Velocity tracking and nominal step width at every node.
    for (int axis = 0; axis < 2; ++axis) {
      addQuadraticResidual(s, {{VX + axis, 1.0}}, {}, -input.velocityCommand(axis), cfg.velocityTrackingWeight);
    }
    {
      Coefficients xc;
      for (int axis = 0; axis < 2; ++axis) {
        xc.push_back({PLX + axis, ey(axis)});
        xc.push_back({PRX + axis, -ey(axis)});
      }
      const auto [g, offset] = frameTerm(1, k, hasHeading ? vector2_t(nominal.feet[k][0] - nominal.feet[k][1]) : vector2_t::Zero());
      if (hasHeading) xc.push_back({L.heading, g});
      addQuadraticResidual(s, xc, {}, -cfg.nominalStepWidth + offset, cfg.stepWidthWeight);
    }
    if (hasHeading) {
      // Heading rate towards the commanded yaw rate, heading towards the commanded heading, foot yaws towards the heading.
      addQuadraticResidual(s, {{L.headingRate, 1.0}}, {}, -input.headingRateCommand, cfg.headingRateTrackingWeight);
      const scalar_t commandedHeading = input.heading + input.headingRateCommand * static_cast<scalar_t>(k) * dt;
      addQuadraticResidual(s, {{L.heading, 1.0}}, {}, -commandedHeading, cfg.headingTrackingWeight);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        addQuadraticResidual(s, {{L.footYaw(foot), 1.0}, {L.heading, -1.0}}, {}, 0.0, cfg.footYawTrackingWeight);
      }
    }
    if (terminal) {
      continue;
    }
    s.R.diagonal().array() += kInputRegularization;
    if (hasHeading) {
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        addQuadraticResidual(s, {}, {{L.yawTorque(foot), 1.0}}, 0.0, cfg.yawTorqueWeight);
        addQuadraticResidual(s, {}, {{L.footYawDelta(foot), 1.0}}, 0.0, cfg.footYawRegularizationWeight);
      }
    }

    // ---------------- dynamics ----------------
    s.A.setIdentity();
    for (int axis = 0; axis < 2; ++axis) {
      s.A(CX + axis, CX + axis) = ch;
      s.A(CX + axis, VX + axis) = sh / omega;
      s.A(VX + axis, CX + axis) = omega * sh;
      s.A(VX + axis, VX + axis) = ch;
      s.B(CX + axis, ZX + axis) = 1.0 - ch;
      s.B(VX + axis, ZX + axis) = -omega * sh;
      s.B(PLX + axis, DLX + axis) = 1.0;
      s.B(PRX + axis, DRX + axis) = 1.0;
    }
    if (hasHeading) {
      s.A(L.heading, L.headingRate) = dt;
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        s.B(L.heading, L.yawTorque(foot)) = 0.5 * dt * dt / inertia;
        s.B(L.headingRate, L.yawTorque(foot)) = dt / inertia;
        s.B(L.footYaw(foot), L.footYawDelta(foot)) = 1.0;
      }
    }

    // ---------------- input boxes ----------------
    s.idxbu = {DLX, DLY, DRX, DRY, CL, CR};
    s.lbu = (vector_t(6) << -M, -M, -M, -M, 0.0, 0.0).finished();
    s.ubu = (vector_t(6) << M, M, M, M, 1.0, 1.0).finished();
    if (hasHeading) {
      const scalar_t torqueBound = torsion + couple;
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        s.idxbu.push_back(L.yawTorque(foot));
        s.idxbu.push_back(L.footYawDelta(foot));
      }
      vector_t lbu(s.idxbu.size()), ubu(s.idxbu.size());
      lbu.head(6) = s.lbu;
      ubu.head(6) = s.ubu;
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        lbu(6 + 2 * foot) = -torqueBound;
        ubu(6 + 2 * foot) = torqueBound;
        lbu(7 + 2 * foot) = -kYawBigM;
        ubu(7 + 2 * foot) = kYawBigM;
      }
      s.lbu = lbu;
      s.ubu = ubu;
    }

    // ---------------- running cost ----------------
    for (int axis = 0; axis < 2; ++axis) {
      addQuadraticResidual(s, {{CX + axis, -1.0}}, {{ZX + axis, 1.0}}, 0.0, cfg.zmpRegularizationWeight);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        addQuadraticResidual(s, {}, {{dIndex[foot] + axis, 1.0}}, 0.0, cfg.footholdRegularizationWeight);
      }
    }
    if (k == N - 1) {
      // Terminal capturability: xi_N - zmp_{N-1} = e^{omega dt} (xi_{N-1} - zmp_{N-1}).
      const scalar_t gain = std::exp(2.0 * omega * dt);
      for (int axis = 0; axis < 2; ++axis) {
        addQuadraticResidual(s, {{CX + axis, 1.0}, {VX + axis, 1.0 / omega}}, {{ZX + axis, -1.0}}, 0.0, cfg.terminalDcmWeight * gain);
      }
    }

    // ---------------- general constraints ----------------
    RowBuilder rows(L.nx, L.nu);
    // No flight phase.
    rows.add({}, {{CL, 1.0}, {CR, 1.0}}, 1.0, 2.0, false);
    if (hasHeading) {
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        // Yaw torque of a foot: torsional friction while it stands, half the friction couple on top in double support:
        // +-tau_i - T_t c_i - T_c (c_L + c_R) / 2 <= -T_c / 2.
        for (const scalar_t sign : {1.0, -1.0}) {
          rows.add({}, {{L.yawTorque(foot), sign}, {cIndex[foot], -torsion}, {CL, -0.5 * couple}, {CR, -0.5 * couple}}, -kLooseBound,
                   -0.5 * couple, false);
        }
        // A foot's yaw only changes while the foot is in the air: +-dpsi_i + 2 pi c_i <= 2 pi.
        for (const scalar_t sign : {1.0, -1.0}) {
          rows.add({}, {{L.footYawDelta(foot), sign}, {cIndex[foot], kYawBigM}}, -kLooseBound, kYawBigM, false);
        }
        // Hip range: the foot yaw stays within the hip yaw limits of the heading (soft).
        const auto [yawLower, yawUpper] = cfg.footYawOffsetBounds(foot);
        rows.add({{L.footYaw(foot), 1.0}, {L.heading, -1.0}}, {}, yawLower, yawUpper, true);
      }
    }

    // ZMP support region (soft). Single support: the box of the supporting foot. Double support: laterally the exact hull
    // of both boxes (the feet never cross laterally), along the heading a box around the midpoint of the feet, which is a
    // conservative inner approximation of the hull that needs no extra binary for the foot order.
    const auto zmpMinusFoot = [&](size_t foot, int axis, scalar_t sign, Coefficients& xc, Coefficients& uc) {
      for (int w = 0; w < 2; ++w) {
        xc.push_back({pIndex[foot] + w, -sign * axes[axis](w)});
        uc.push_back({ZX + w, sign * axes[axis](w)});
      }
    };
    // Single-support boxes: +-e_j'(zmp - p_i) <= r_j + M (1 - c_i) + M c_other.
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const size_t other = 1 - foot;
      for (int axis = 0; axis < 2; ++axis) {
        for (const scalar_t sign : {1.0, -1.0}) {
          Coefficients xc, uc;
          zmpMinusFoot(foot, axis, sign, xc, uc);
          uc.push_back({cIndex[foot], M});
          uc.push_back({cIndex[other], -M});
          rows.add(xc, uc, -kLooseBound, zmpHalfWidth[axis] + M, true);
        }
      }
    }
    // Double support, heading axis: +-e_x'(zmp - (p_L + p_R) / 2) <= r_x + M (1 - c_L) + M (1 - c_R).
    for (const scalar_t sign : {1.0, -1.0}) {
      Coefficients xc, uc;
      for (int w = 0; w < 2; ++w) {
        xc.push_back({PLX + w, -0.5 * sign * axes[0](w)});
        xc.push_back({PRX + w, -0.5 * sign * axes[0](w)});
        uc.push_back({ZX + w, sign * axes[0](w)});
      }
      uc.push_back({CL, M});
      uc.push_back({CR, M});
      rows.add(xc, uc, -kLooseBound, zmpHalfWidth[0] + 2.0 * M, true);
    }
    // Double support, lateral axis: upper bound from the left foot, lower bound from the right foot.
    {
      Coefficients xc, uc;  // e_y'(zmp - p_L) <= r + M (1 - c_L)
      zmpMinusFoot(0, 1, 1.0, xc, uc);
      uc.push_back({CL, M});
      rows.add(xc, uc, -kLooseBound, zmpHalfWidth[1] + M, true);
    }
    {
      Coefficients xc, uc;  // -e_y'(zmp - p_R) <= r + M (1 - c_R)
      zmpMinusFoot(1, 1, -1.0, xc, uc);
      uc.push_back({CR, M});
      rows.add(xc, uc, -kLooseBound, zmpHalfWidth[1] + M, true);
    }
    // A foot only moves while it is not in contact: +-dp_ij + M c_i <= M.
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      for (int axis = 0; axis < 2; ++axis) {
        for (const scalar_t sign : {1.0, -1.0}) {
          rows.add({}, {{dIndex[foot] + axis, sign}, {cIndex[foot], M}}, -kLooseBound, M, false);
        }
      }
    }
    // Reachability of the feet with respect to the CoM (soft), yaw frame (with its first-order heading term).
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      for (int axis = 0; axis < 2; ++axis) {
        Coefficients xc;
        for (int w = 0; w < 2; ++w) {
          xc.push_back({pIndex[foot] + w, axes[axis](w)});
          xc.push_back({CX + w, -axes[axis](w)});
        }
        scalar_t lower, upper;
        if (axis == 0) {
          lower = -cfg.reachX;
          upper = cfg.reachX;
        } else if (foot == 0) {
          lower = cfg.reachYInner;
          upper = cfg.reachYOuter;
        } else {
          lower = -cfg.reachYOuter;
          upper = -cfg.reachYInner;
        }
        const auto [g, offset] = frameTerm(axis, k, hasHeading ? vector2_t(nominal.feet[k][foot] - nominal.com[k]) : vector2_t::Zero());
        if (hasHeading) xc.push_back({L.heading, g});
        rows.add(xc, {}, lower - offset, upper - offset, true);
      }
    }
    // Foot separation (soft): step length and width bounds.
    for (int axis = 0; axis < 2; ++axis) {
      Coefficients xc;
      for (int w = 0; w < 2; ++w) {
        xc.push_back({PLX + w, axes[axis](w)});
        xc.push_back({PRX + w, -axes[axis](w)});
      }
      const auto [g, offset] = frameTerm(axis, k, hasHeading ? vector2_t(nominal.feet[k][0] - nominal.feet[k][1]) : vector2_t::Zero());
      if (hasHeading) xc.push_back({L.heading, g});
      if (axis == 0) {
        rows.add(xc, {}, -cfg.maxStepLength - offset, cfg.maxStepLength - offset, true);
      } else {
        rows.add(xc, {}, cfg.minStepWidth - offset, cfg.maxStepWidth - offset, true);
      }
    }
    rows.writeTo(s, cfg.constraintSlackWeight, cfg.constraintSlackLinearWeight);
  }
  return problem;
}

std::optional<MiqpAssignment> LipContactPlanner::warmStartAssignment(const ContactPlannerInput& input) const {
  const int shift = previousPlanShift(input);
  if (shift < 0) return std::nullopt;
  MiqpAssignment warm(kBinariesPerNode * config_.numNodes, kMiqpFree);
  for (int k = 0; k < config_.numNodes; ++k) {
    const int source = std::min(k + shift, config_.numNodes - 1);
    for (int j = 0; j < kBinariesPerNode; ++j) {
      warm[kBinariesPerNode * k + j] = previousAssignment_[kBinariesPerNode * source + j];
    }
  }
  return warm;
}

void LipContactPlanner::localSearch(const ContactPlannerInput& input, const MiqpAssignment& initial, scalar_t timeBudget) {
  const auto start = Clock::now();
  if (!lastResult_.hasIncumbent || config_.localSearchIterations <= 0 || timeBudget <= 0.0) return;
  const std::vector<MiqpBinaryVariable> binaries = binaryVariables();
  const MiqpPropagateFn propagateFn = [this, &input](MiqpAssignment& assignment) { return propagate(input, assignment); };
  const MiqpAssignmentCostFn costFn = [this, &input](const MiqpAssignment& assignment) { return assignmentCost(input, assignment); };
  const int N = config_.numNodes;

  std::set<MiqpAssignment> evaluated;
  evaluated.insert(lastResult_.assignment);
  for (int round = 0; round < config_.localSearchIterations; ++round) {
    if (elapsedSeconds(start) > timeBudget) break;
    const MiqpAssignment base = lastResult_.assignment;
    bool improved = false;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      for (int k = 1; k < N; ++k) {
        const int index = contactBinaryIndex(k, foot);
        const int previousIndex = contactBinaryIndex(k - 1, foot);
        if (base[index] == base[previousIndex]) continue;  // no transition at k
        // Move the transition one node earlier (the previous node takes the new value) or later (this node keeps the old).
        for (const int shift : {-1, +1}) {
          MiqpAssignment candidate = base;
          if (shift < 0) {
            if (initial[previousIndex] != kMiqpFree) continue;
            candidate[previousIndex] = base[index];
          } else {
            if (initial[index] != kMiqpFree) continue;
            candidate[index] = base[previousIndex];
          }
          if (!propagateFn(candidate)) continue;
          if (!evaluated.insert(candidate).second) continue;
          if (elapsedSeconds(start) > timeBudget) break;
          OcpQpSolution solution;
          scalar_t objective = 0.0;
          ++statistics_.numLocalSearchQps;
          if (!miqp_->solveFixed(problem_, binaries, candidate, propagateFn, costFn, solution, objective)) continue;
          statistics_.totalQpIterations += solution.iterations;
          if (objective < lastResult_.incumbentObjective - 1e-6) {
            lastResult_.incumbentObjective = objective;
            lastResult_.solution = solution;
            lastResult_.assignment = candidate;
            improved = true;
            statistics_.localSearchImproved = true;
            if (config_.verbose) {
              std::cout << "[LipContactPlanner] local search improved the objective to " << objective << std::endl;
            }
          }
        }
      }
    }
    if (!improved) break;
  }
  statistics_.localSearchTime = elapsedSeconds(start);
}

ContactPlan LipContactPlanner::decode(const ContactPlannerInput& input, const MiqpResult& result) const {
  ContactPlan plan;
  plan.startTime = input.time;
  plan.dt = config_.dt;
  plan.committedUntil = input.committedUntil;
  plan.yaw = input.yaw;
  plan.numBranchAndBoundNodes = statistics_.numBranchAndBoundRelaxations + statistics_.numLocalSearchQps;
  plan.solveTime = statistics_.branchAndBoundTime + statistics_.localSearchTime;
  plan.optimal = result.optimal;
  plan.nodeLimitHit = result.nodeLimitHit;
  plan.timeLimitHit = result.timeLimitHit;
  if (!result.hasIncumbent) {
    return plan;
  }
  const int N = config_.numNodes;
  plan.valid = true;
  plan.objective = result.incumbentObjective;
  plan.contacts.resize(N);
  plan.zmp.resize(N);
  plan.footholds.resize(N + 1);
  plan.comPosition.resize(N + 1);
  plan.comVelocity.resize(N + 1);
  if (layout_.hasHeading) {
    plan.heading.resize(N + 1);
    plan.headingRate.resize(N + 1);
    plan.footYaws.resize(N + 1);
  }
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[k];
    plan.comPosition[k] = x.segment<2>(CX);
    plan.comVelocity[k] = x.segment<2>(VX);
    plan.footholds[k][0] = x.segment<2>(PLX);
    plan.footholds[k][1] = x.segment<2>(PRX);
    if (layout_.hasHeading) {
      plan.heading[k] = x(layout_.heading);
      plan.headingRate[k] = x(layout_.headingRate);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) plan.footYaws[k][foot] = x(layout_.footYaw(foot));
    }
    if (k < N) {
      const vector_t& u = result.solution.u[k];
      plan.zmp[k] = u.segment<2>(ZX);
      plan.contacts[k] = contact_flag_t{result.assignment[contactBinaryIndex(k, 0)] == 1, result.assignment[contactBinaryIndex(k, 1)] == 1};
    }
  }
  return plan;
}

ContactPlan LipContactPlanner::plan(const ContactPlannerInput& input) {
  const auto start = Clock::now();
  statistics_ = Statistics();
  try {
    problem_ = buildProblem(input);
  } catch (const std::exception& e) {
    std::cerr << "[LipContactPlanner] cannot build the problem: " << e.what() << std::endl;
    lastResult_ = MiqpResult();
    return decode(input, lastResult_);
  }
  const std::vector<MiqpBinaryVariable> binaries = binaryVariables();
  const MiqpAssignment initial = initialAssignment(input);
  const std::optional<MiqpAssignment> warm = warmStartAssignment(input);
  const MiqpPropagateFn propagateFn = [this, &input](MiqpAssignment& assignment) { return propagate(input, assignment); };
  const MiqpAssignmentCostFn costFn = [this, &input](const MiqpAssignment& assignment) { return assignmentCost(input, assignment); };

  try {
    lastResult_ = miqp_->solve(problem_, binaries, initial, propagateFn, warm ? &*warm : nullptr, costFn);
  } catch (const std::exception& e) {
    std::cerr << "[LipContactPlanner] solver failure: " << e.what() << std::endl;
    lastResult_ = MiqpResult();
  }
  statistics_.numBranchAndBoundRelaxations = lastResult_.numNodes;
  statistics_.totalQpIterations = lastResult_.totalQpIterations;
  statistics_.branchAndBoundTime = lastResult_.solveTime;

  try {
    localSearch(input, initial, config_.localSearchMaxTime);
  } catch (const std::exception& e) {
    std::cerr << "[LipContactPlanner] local search failure: " << e.what() << std::endl;
  }

  // Heading model: the foothold frame was linearised around a nominal heading trajectory. Re-linearise around the
  // incumbent's own heading and footholds and re-solve the QP with the contacts fixed (successive linearisation), so
  // that the frame the constraints were written in is the frame the plan actually turns through.
  if (layout_.hasHeading && lastResult_.hasIncumbent && config_.headingLinearizationPasses > 0) {
    const scalar_t timeBudget = config_.maxSolveTime + config_.localSearchMaxTime;
    for (int pass = 0; pass < config_.headingLinearizationPasses; ++pass) {
      if (elapsedSeconds(start) > timeBudget) break;
      try {
        OcpQpProblem relinearised = buildProblem(input, nominalFromSolution(layout_, lastResult_.solution));
        OcpQpSolution solution;
        scalar_t objective = 0.0;
        ++statistics_.numHeadingRelinearizations;
        if (!miqp_->solveFixed(relinearised, binaries, lastResult_.assignment, propagateFn, costFn, solution, objective)) break;
        statistics_.totalQpIterations += solution.iterations;
        problem_ = std::move(relinearised);
        lastResult_.solution = std::move(solution);
        lastResult_.incumbentObjective = objective;
      } catch (const std::exception& e) {
        std::cerr << "[LipContactPlanner] heading re-linearisation failure: " << e.what() << std::endl;
        break;
      }
    }
  }

  ContactPlan plan = decode(input, lastResult_);
  if (config_.verbose) {
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
