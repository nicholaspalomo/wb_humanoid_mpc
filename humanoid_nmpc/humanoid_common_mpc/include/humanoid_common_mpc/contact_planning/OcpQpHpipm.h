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

#pragma once

#include <memory>
#include <vector>

#include <ocs2_core/Types.h>

namespace ocs2::humanoid {

/**
 * Bound magnitude from which a general constraint bound is treated as absent (masked out in HPIPM) rather than as a
 * finite, very loose bound. Use +-kOcpQpInfiniteBound for one-sided constraints.
 */
constexpr scalar_t kOcpQpInfiniteBound = 1.0e6;

/**
 * One node of a discrete-time linear-quadratic optimal control problem with box and general inequality constraints.
 *
 * Dynamics (absent at the terminal node):     x_{k+1} = A x_k + B u_k + b
 * Cost:                                       0.5 x'Qx + 0.5 u'Ru + u'S x + q'x + r'u
 * Input box constraints:                      lbu <= u[idxbu] <= ubu
 * State box constraints:                      lbx <= x[idxbx] <= ubx
 * General constraints:                        lg <= C x + D u <= ug
 * Soft general constraints:                   rows listed in softGeneralIndices get slacks s_l, s_u >= 0 that relax the
 *                                             lower/upper bound and are penalised by 0.5 Z s^2 + z s.
 *
 * All matrices are Eigen column-major, which is the layout HPIPM expects.
 */
struct OcpQpStage {
  matrix_t A, B;
  vector_t b;

  matrix_t Q, R, S;  // S is (nu x nx), i.e. the dfdux block of the OCS2 convention.
  vector_t q, r;

  std::vector<int> idxbu;
  vector_t lbu, ubu;

  std::vector<int> idxbx;
  vector_t lbx, ubx;

  matrix_t C, D;
  vector_t lg, ug;

  std::vector<int> softGeneralIndices;  // indices into the general constraint rows
  vector_t Zl, Zu, zl, zu;              // one entry per soft constraint

  int numStates() const { return static_cast<int>(Q.rows()); }
  int numInputs() const { return static_cast<int>(R.rows()); }
  int numGeneralConstraints() const { return static_cast<int>(C.rows()); }

  /** Allocates zero-initialised data for the given sizes (no constraints). */
  static OcpQpStage Zero(int nx, int nu, bool hasDynamics);
};

/** A full OCP-QP: N stages with dynamics plus a terminal node without inputs. The initial state is fixed to x0. */
struct OcpQpProblem {
  vector_t x0;
  std::vector<OcpQpStage> stages;  // size N + 1, the last one has no inputs and no dynamics

  int numStages() const { return static_cast<int>(stages.size()) - 1; }
};

struct OcpQpSolution {
  enum class Status { SUCCESS, MAX_ITER, MIN_STEP, NAN_SOL, INCONS_EQ, UNKNOWN };

  Status status = Status::UNKNOWN;
  int iterations = 0;
  scalar_t objective = 0.0;  // includes the soft-constraint slack penalties
  std::vector<vector_t> x;   // size N + 1
  std::vector<vector_t> u;   // size N

  bool success() const { return status == Status::SUCCESS; }
};

/** Evaluates the objective of the problem (including soft constraint penalties) for the given trajectories. */
scalar_t evaluateOcpQpObjective(const OcpQpProblem& problem, const std::vector<vector_t>& x, const std::vector<vector_t>& u);

/** Maximum violation of the hard box / general constraints and of the dynamics for the given trajectories. */
scalar_t evaluateOcpQpMaxHardViolation(const OcpQpProblem& problem, const std::vector<vector_t>& x, const std::vector<vector_t>& u);

/**
 * Thin wrapper around the HPIPM interior point solver for OCP-QPs with box, general and soft inequality constraints.
 *
 * The OCS2 HpipmInterface only exposes equality constraints, so the contact planner needs its own wrapper. Memory is
 * (re)allocated whenever the problem dimensions change; repeated solves with identical dimensions are allocation free.
 */
class OcpQpHpipmSolver {
 public:
  struct Settings {
    int iterMax = 60;
    scalar_t alphaMin = 1e-12;
    scalar_t mu0 = 1e2;
    scalar_t tolStat = 1e-6;
    scalar_t tolEq = 1e-8;
    scalar_t tolIneq = 1e-8;
    scalar_t tolComp = 1e-8;
    scalar_t regPrim = 1e-10;
    int warmStart = 0;
    int predCorr = 1;
    int hpipmMode = 2;  // hpipm_mode: 0 SPEED_ABS, 1 SPEED, 2 BALANCE, 3 ROBUST
  };

  OcpQpHpipmSolver();
  explicit OcpQpHpipmSolver(const Settings& settings);
  ~OcpQpHpipmSolver();

  OcpQpHpipmSolver(const OcpQpHpipmSolver&) = delete;
  OcpQpHpipmSolver& operator=(const OcpQpHpipmSolver&) = delete;

  /** Solves the problem. Throws std::invalid_argument for inconsistent problem data. */
  OcpQpSolution solve(const OcpQpProblem& problem);

  const Settings& getSettings() const { return settings_; }

 private:
  class Impl;
  std::unique_ptr<Impl> pImpl_;
  Settings settings_;
};

}  // namespace ocs2::humanoid
