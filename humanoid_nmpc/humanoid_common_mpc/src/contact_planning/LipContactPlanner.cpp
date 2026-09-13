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
#include <cmath>
#include <iostream>
#include <utility>

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kLooseBound = kOcpQpInfiniteBound;
constexpr scalar_t kInputRegularization = 1.0e-6;
constexpr scalar_t kStateRegularization = 1.0e-8;

using Coefficients = std::vector<std::pair<int, scalar_t>>;

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

}  // namespace

namespace {
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

LipContactPlanner::LipContactPlanner(ContactPlanningConfig config) : config_(std::move(config)) {
  config_.validate();
  const OcpQpHpipmSolver::Settings qpSettings = relaxationQpSettings(config_);
  MiqpSettings miqpSettings;
  miqpSettings.maxNodes = config_.maxBranchAndBoundNodes;
  miqpSettings.maxSolveTime = config_.maxSolveTime;
  miqpSettings.verbose = config_.verbose;
  miqp_ = std::make_unique<MixedIntegerOcpQp>(qpSettings, miqpSettings);
}

void LipContactPlanner::setConfig(const ContactPlanningConfig& config) {
  config.validate();
  config_ = config;
  const OcpQpHpipmSolver::Settings qpSettings = relaxationQpSettings(config_);
  MiqpSettings miqpSettings;
  miqpSettings.maxNodes = config_.maxBranchAndBoundNodes;
  miqpSettings.maxSolveTime = config_.maxSolveTime;
  miqpSettings.verbose = config_.verbose;
  miqp_ = std::make_unique<MixedIntegerOcpQp>(qpSettings, miqpSettings);
  reset();
}

void LipContactPlanner::reset() {
  previousPlan_.reset();
  previousAssignment_.clear();
}

int LipContactPlanner::initialPhaseNodes(const ContactPlannerInput& input, size_t foot) const {
  const int cap = 2 * config_.numNodes;
  const int elapsed = static_cast<int>(std::lround(std::max(0.0, input.phaseElapsedTime[foot]) / config_.dt));
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

  for (int iteration = 0; iteration < 4 * N && ok; ++iteration) {
    bool changed = false;

    // Per-foot forward pass along the fixed prefix.
    for (size_t foot = 0; foot < N_CONTACTS && ok; ++foot) {
      std::int8_t kappa = input.contacts[foot] ? 1 : 0;
      int tau = initialPhaseNodes(input, foot);
      for (int k = 0; k < N; ++k) {
        const int index = contactBinaryIndex(k, foot);
        if (k >= numCommitted) {
          const bool mustStay = (kappa == 1 && tau < nContactMin) || (kappa == 0 && tau < nSwingMin);
          const bool mustSwitch = (kappa == 0 && tau >= nSwingMax) || (kappa == 1 && nContactMax > 0 && tau >= nContactMax);
          if (mustStay && mustSwitch) return false;
          if (mustStay) changed |= fix(index, kappa);
          if (mustSwitch) changed |= fix(index, static_cast<std::int8_t>(1 - kappa));
          if (!ok) return false;
        }
        if (a[index] == kMiqpFree) break;
        if (a[index] == kappa) {
          ++tau;
        } else {
          tau = 1;
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

OcpQpProblem LipContactPlanner::buildProblem(const ContactPlannerInput& input) const {
  const ContactPlanningConfig& cfg = config_;
  const int N = cfg.numNodes;
  const scalar_t dt = cfg.dt;
  const scalar_t omega = cfg.omega();
  const scalar_t ch = std::cosh(omega * dt);
  const scalar_t sh = std::sinh(omega * dt);
  const scalar_t bigMTau = 4.0 * N + 4.0;
  const scalar_t tauUpper = 4.0 * N;
  const int nSwingMin = cfg.minSwingNodes();
  const int nSwingMax = cfg.maxSwingNodes();
  const int nContactMin = cfg.minContactNodes();
  const int nContactMax = cfg.maxContactNodes();
  const int numCommitted = std::min(static_cast<int>(input.committedContacts.size()), N);

  // Yaw-aligned constraint frame: components of the world-frame unit vectors of the local x and y axes.
  const vector2_t ex(std::cos(input.yaw), std::sin(input.yaw));
  const vector2_t ey(-std::sin(input.yaw), std::cos(input.yaw));
  const std::array<vector2_t, 2> axes{ex, ey};
  const std::array<scalar_t, 2> zmpHalfWidth{cfg.zmpHalfWidthX, cfg.zmpHalfWidthY};
  const std::array<int, 2> pIndex{PLX, PRX};  // x index of each foot's position; y follows
  const std::array<int, 2> dIndex{DLX, DRX};
  const std::array<int, 2> cIndex{CL, CR};
  const std::array<int, 2> sIndex{SL, SR};
  const std::array<int, 2> kIndex{KL, KR};
  const std::array<int, 2> tIndex{TL, TR};
  const std::array<int, 2> tnIndex{TNL, TNR};

  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(STATE_DIM);
  problem.x0.segment<2>(CX) = input.comPosition;
  problem.x0.segment<2>(VX) = input.comVelocity;
  problem.x0.segment<2>(PLX) = input.footPositions[0];
  problem.x0.segment<2>(PRX) = input.footPositions[1];
  problem.x0(KL) = input.contacts[0] ? 1.0 : 0.0;
  problem.x0(KR) = input.contacts[1] ? 1.0 : 0.0;
  problem.x0(TL) = static_cast<scalar_t>(initialPhaseNodes(input, 0));
  problem.x0(TR) = static_cast<scalar_t>(initialPhaseNodes(input, 1));

  problem.stages.resize(N + 1);
  for (int k = 0; k <= N; ++k) {
    const bool terminal = (k == N);
    OcpQpStage& s = problem.stages[k];
    s = OcpQpStage::Zero(STATE_DIM, terminal ? 0 : INPUT_DIM, !terminal);
    s.Q.diagonal().array() += kStateRegularization;

    // Velocity tracking at every node.
    for (int axis = 0; axis < 2; ++axis) {
      addQuadraticResidual(s, {{VX + axis, 1.0}}, {}, -input.velocityCommand(axis), cfg.velocityTrackingWeight);
    }
    // Nominal step width.
    {
      Coefficients xc;
      for (int axis = 0; axis < 2; ++axis) {
        xc.push_back({PLX + axis, ey(axis)});
        xc.push_back({PRX + axis, -ey(axis)});
      }
      addQuadraticResidual(s, xc, {}, -cfg.nominalStepWidth, cfg.stepWidthWeight);
    }
    if (terminal) {
      continue;
    }
    s.R.diagonal().array() += kInputRegularization;

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
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      s.A(kIndex[foot], kIndex[foot]) = 0.0;
      s.B(kIndex[foot], cIndex[foot]) = 1.0;
      s.A(tIndex[foot], tIndex[foot]) = 0.0;
      s.B(tIndex[foot], tnIndex[foot]) = 1.0;
    }

    // ---------------- input boxes ----------------
    s.idxbu = {DLX, DLY, DRX, DRY, CL, CR, SL, SR, TNL, TNR};
    s.lbu = vector_t::Zero(s.idxbu.size());
    s.ubu = vector_t::Zero(s.idxbu.size());
    for (std::size_t i = 0; i < s.idxbu.size(); ++i) {
      const int idx = s.idxbu[i];
      if (idx == DLX || idx == DLY || idx == DRX || idx == DRY) {
        s.lbu(i) = -cfg.bigM;
        s.ubu(i) = cfg.bigM;
      } else if (idx == TNL || idx == TNR) {
        s.lbu(i) = 1.0;
        s.ubu(i) = tauUpper;
      } else {
        s.lbu(i) = 0.0;
        s.ubu(i) = 1.0;
      }
    }

    // ---------------- running cost ----------------
    for (int axis = 0; axis < 2; ++axis) {
      addQuadraticResidual(s, {{CX + axis, -1.0}}, {{ZX + axis, 1.0}}, 0.0, cfg.zmpRegularizationWeight);
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        addQuadraticResidual(s, {}, {{dIndex[foot] + axis, 1.0}}, 0.0, cfg.footholdRegularizationWeight);
      }
    }
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      s.r(sIndex[foot]) += cfg.contactSwitchCost;
    }
    if (k == N - 1) {
      // Terminal capturability: xi_N - zmp_{N-1} = e^{omega dt} (xi_{N-1} - zmp_{N-1}).
      const scalar_t gain = std::exp(2.0 * omega * dt);
      for (int axis = 0; axis < 2; ++axis) {
        addQuadraticResidual(s, {{CX + axis, 1.0}, {VX + axis, 1.0 / omega}}, {{ZX + axis, -1.0}}, 0.0, cfg.terminalDcmWeight * gain);
      }
    }

    // ---------------- general constraints ----------------
    RowBuilder rows(STATE_DIM, INPUT_DIM);
    // No flight phase.
    rows.add({}, {{CL, 1.0}, {CR, 1.0}}, 1.0, 2.0, false);
    // ZMP support region (soft). Single support: the box of the supporting foot. Double support: laterally the exact hull
    // of both boxes (the feet never cross laterally), along the heading a box around the midpoint of the feet, which is a
    // conservative inner approximation of the hull that needs no extra binary for the foot order.
    {
      const scalar_t M = cfg.bigM;
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
    }
    // A foot only moves while it is not in contact: +-dp_ij + M c_i <= M.
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      for (int axis = 0; axis < 2; ++axis) {
        for (const scalar_t sign : {1.0, -1.0}) {
          rows.add({}, {{dIndex[foot] + axis, sign}, {cIndex[foot], cfg.bigM}}, -kLooseBound, cfg.bigM, false);
        }
      }
    }
    // Reachability of the feet with respect to the CoM (soft), yaw frame.
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
        rows.add(xc, {}, lower, upper, true);
      }
    }
    // Foot separation (soft): step length and width bounds.
    for (int axis = 0; axis < 2; ++axis) {
      Coefficients xc;
      for (int w = 0; w < 2; ++w) {
        xc.push_back({PLX + w, axes[axis](w)});
        xc.push_back({PRX + w, -axes[axis](w)});
      }
      if (axis == 0) {
        rows.add(xc, {}, -cfg.maxStepLength, cfg.maxStepLength, true);
      } else {
        rows.add(xc, {}, cfg.minStepWidth, cfg.maxStepWidth, true);
      }
    }
    // Switch indicator s = |c - kappa| (exact for binary c, kappa) and run-length counter update.
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const int c = cIndex[foot], a = sIndex[foot], kap = kIndex[foot], tau = tIndex[foot], tauNext = tnIndex[foot];
      rows.add({{kap, 1.0}}, {{a, 1.0}, {c, -1.0}}, 0.0, 2.0, false);                    // s >= c - kappa
      rows.add({{kap, -1.0}}, {{a, 1.0}, {c, 1.0}}, 0.0, 2.0, false);                    // s >= kappa - c
      rows.add({{kap, -1.0}}, {{a, 1.0}, {c, -1.0}}, -2.0, 0.0, false);                  // s <= c + kappa
      rows.add({{kap, 1.0}}, {{a, 1.0}, {c, 1.0}}, 0.0, 2.0, false);                     // s <= 2 - c - kappa
      rows.add({{tau, -1.0}}, {{tauNext, 1.0}}, -kLooseBound, 1.0, false);               // tau+ <= tau + 1
      rows.add({{tau, -1.0}}, {{tauNext, 1.0}, {a, bigMTau}}, 1.0, kLooseBound, false);  // tau+ >= tau + 1 - M s
      rows.add({}, {{tauNext, 1.0}, {a, bigMTau}}, -kLooseBound, 1.0 + bigMTau, false);  // tau+ <= 1 + M (1 - s)
      if (k >= numCommitted) {
        // Minimum durations: a switch is only allowed once the current phase lasted long enough.
        rows.add({{tau, 1.0}, {kap, static_cast<scalar_t>(nSwingMin)}}, {{c, -static_cast<scalar_t>(nSwingMin)}}, 0.0, kLooseBound, false);
        rows.add({{tau, 1.0}, {kap, -static_cast<scalar_t>(nContactMin)}}, {{c, static_cast<scalar_t>(nContactMin)}}, 0.0, kLooseBound,
                 false);
        // Maximum swing duration: staying in swing (c = kappa = 0) requires tau <= nSwingMax - 1.
        rows.add({{tau, 1.0}, {kap, -bigMTau}}, {{c, -bigMTau}}, -kLooseBound, static_cast<scalar_t>(nSwingMax - 1), false);
        if (nContactMax > 0) {
          rows.add({{tau, 1.0}, {kap, bigMTau}}, {{c, bigMTau}}, -kLooseBound, static_cast<scalar_t>(nContactMax - 1) + 2.0 * bigMTau,
                   false);
        }
      }
    }
    rows.writeTo(s, cfg.constraintSlackWeight, cfg.constraintSlackLinearWeight);
  }
  return problem;
}

std::optional<MiqpAssignment> LipContactPlanner::warmStartAssignment(const ContactPlannerInput& input) const {
  if (!previousPlan_ || !previousPlan_->valid || previousAssignment_.empty()) return std::nullopt;
  const ContactPlan& prev = *previousPlan_;
  if (prev.numIntervals() != config_.numNodes || std::abs(prev.dt - config_.dt) > 1e-9) return std::nullopt;
  const int shift = static_cast<int>(std::lround((input.time - prev.startTime) / config_.dt));
  if (shift < 0 || shift >= config_.numNodes) return std::nullopt;
  MiqpAssignment warm(kBinariesPerNode * config_.numNodes, kMiqpFree);
  for (int k = 0; k < config_.numNodes; ++k) {
    const int source = std::min(k + shift, config_.numNodes - 1);
    for (int j = 0; j < kBinariesPerNode; ++j) {
      warm[kBinariesPerNode * k + j] = previousAssignment_[kBinariesPerNode * source + j];
    }
  }
  return warm;
}

ContactPlan LipContactPlanner::decode(const ContactPlannerInput& input, const MiqpResult& result) const {
  ContactPlan plan;
  plan.startTime = input.time;
  plan.dt = config_.dt;
  plan.numBranchAndBoundNodes = result.numNodes;
  plan.solveTime = result.solveTime;
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
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[k];
    plan.comPosition[k] = x.segment<2>(CX);
    plan.comVelocity[k] = x.segment<2>(VX);
    plan.footholds[k][0] = x.segment<2>(PLX);
    plan.footholds[k][1] = x.segment<2>(PRX);
    if (k < N) {
      const vector_t& u = result.solution.u[k];
      plan.zmp[k] = u.segment<2>(ZX);
      plan.contacts[k] = contact_flag_t{result.assignment[contactBinaryIndex(k, 0)] == 1, result.assignment[contactBinaryIndex(k, 1)] == 1};
    }
  }
  return plan;
}

ContactPlan LipContactPlanner::plan(const ContactPlannerInput& input) {
  problem_ = buildProblem(input);
  const std::vector<MiqpBinaryVariable> binaries = binaryVariables();
  const MiqpAssignment initial = initialAssignment(input);
  const std::optional<MiqpAssignment> warm = warmStartAssignment(input);
  const MiqpPropagateFn propagateFn = [this, &input](MiqpAssignment& assignment) { return propagate(input, assignment); };

  try {
    lastResult_ = miqp_->solve(problem_, binaries, initial, propagateFn, warm ? &*warm : nullptr);
  } catch (const std::exception& e) {
    std::cerr << "[LipContactPlanner] solver failure: " << e.what() << std::endl;
    lastResult_ = MiqpResult();
  }
  ContactPlan plan = decode(input, lastResult_);
  if (config_.verbose) {
    std::cout << "[LipContactPlanner] valid=" << plan.valid << " objective=" << plan.objective << " nodes=" << plan.numBranchAndBoundNodes
              << " time=" << plan.solveTime << "s optimal=" << plan.optimal << std::endl;
  }
  if (plan.valid) {
    previousPlan_ = plan;
    previousAssignment_ = lastResult_.assignment;
  }
  return plan;
}

}  // namespace ocs2::humanoid
