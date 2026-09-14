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

#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

extern "C" {
#include <hpipm_common.h>
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_ocp_qp_sol.h>
}

namespace ocs2::humanoid {

namespace {

/** Owns a raw block of memory for HPIPM and only reallocates when it has to grow. */
class MemoryBlock {
 public:
  MemoryBlock() = default;
  ~MemoryBlock() { std::free(ptr_); }
  MemoryBlock(const MemoryBlock&) = delete;
  MemoryBlock& operator=(const MemoryBlock&) = delete;

  void reserve(std::size_t size) {
    if (size > size_) {
      std::free(ptr_);
      ptr_ = std::malloc(size);
      if (ptr_ == nullptr) {
        throw std::bad_alloc();
      }
      size_ = size;
    }
  }
  void* get() { return ptr_; }

 private:
  void* ptr_ = nullptr;
  std::size_t size_ = 0;
};

struct Dimensions {
  int N = 0;
  std::vector<int> nx, nu, nbx, nbu, ng, nsbx, nsbu, nsg;

  bool operator==(const Dimensions& rhs) const {
    return N == rhs.N && nx == rhs.nx && nu == rhs.nu && nbx == rhs.nbx && nbu == rhs.nbu && ng == rhs.ng && nsbx == rhs.nsbx &&
           nsbu == rhs.nsbu && nsg == rhs.nsg;
  }
};

void checkStage(const OcpQpStage& stage, int k, int N) {
  const int nx = stage.numStates();
  const int nu = stage.numInputs();
  const auto fail = [&](const std::string& what) {
    throw std::invalid_argument("[OcpQpHpipmSolver] stage " + std::to_string(k) + ": " + what);
  };
  if (k == N && nu != 0) fail("terminal node must not have inputs");
  if (stage.Q.rows() != nx || stage.Q.cols() != nx) fail("Q must be nx x nx");
  if (stage.R.rows() != nu || stage.R.cols() != nu) fail("R must be nu x nu");
  if (nu > 0 && (stage.S.rows() != nu || stage.S.cols() != nx)) fail("S must be nu x nx");
  if (stage.q.size() != nx) fail("q must have nx entries");
  if (stage.r.size() != nu) fail("r must have nu entries");
  if (k < N) {
    if (stage.A.rows() == 0 || stage.A.cols() != nx) fail("A must be nx_next x nx");
    if (stage.B.rows() != stage.A.rows() || stage.B.cols() != nu) fail("B must be nx_next x nu");
    if (stage.b.size() != stage.A.rows()) fail("b must have nx_next entries");
  }
  if (static_cast<int>(stage.idxbu.size()) != stage.lbu.size() || stage.lbu.size() != stage.ubu.size()) fail("inconsistent input box");
  if (static_cast<int>(stage.idxbx.size()) != stage.lbx.size() || stage.lbx.size() != stage.ubx.size()) fail("inconsistent state box");
  for (int idx : stage.idxbu) {
    if (idx < 0 || idx >= nu) fail("input box index out of range");
  }
  for (int idx : stage.idxbx) {
    if (idx < 0 || idx >= nx) fail("state box index out of range");
  }
  const int ng = stage.numGeneralConstraints();
  if (ng > 0) {
    if (stage.C.cols() != nx) fail("C must be ng x nx");
    if (stage.D.rows() != ng || stage.D.cols() != nu) fail("D must be ng x nu");
    if (stage.lg.size() != ng || stage.ug.size() != ng) fail("lg/ug must have ng entries");
  }
  const int ns = static_cast<int>(stage.softGeneralIndices.size());
  if (ns > 0) {
    if (stage.Zl.size() != ns || stage.Zu.size() != ns || stage.zl.size() != ns || stage.zu.size() != ns) {
      fail("Zl/Zu/zl/zu must have one entry per soft constraint");
    }
    for (int idx : stage.softGeneralIndices) {
      if (idx < 0 || idx >= ng) fail("soft constraint index out of range");
    }
    if (!std::is_sorted(stage.softGeneralIndices.begin(), stage.softGeneralIndices.end())) fail("soft constraint indices must be sorted");
  }
}

}  // namespace

OcpQpStage OcpQpStage::Zero(int nx, int nu, bool hasDynamics) {
  OcpQpStage stage;
  if (hasDynamics) {
    stage.A = matrix_t::Zero(nx, nx);
    stage.B = matrix_t::Zero(nx, nu);
    stage.b = vector_t::Zero(nx);
  }
  stage.Q = matrix_t::Zero(nx, nx);
  stage.R = matrix_t::Zero(nu, nu);
  stage.S = matrix_t::Zero(nu, nx);
  stage.q = vector_t::Zero(nx);
  stage.r = vector_t::Zero(nu);
  stage.C = matrix_t::Zero(0, nx);
  stage.D = matrix_t::Zero(0, nu);
  stage.lg = vector_t::Zero(0);
  stage.ug = vector_t::Zero(0);
  stage.lbu = vector_t::Zero(0);
  stage.ubu = vector_t::Zero(0);
  stage.lbx = vector_t::Zero(0);
  stage.ubx = vector_t::Zero(0);
  return stage;
}

scalar_t evaluateOcpQpObjective(const OcpQpProblem& problem, const std::vector<vector_t>& x, const std::vector<vector_t>& u) {
  const int N = problem.numStages();
  scalar_t objective = 0.0;
  for (int k = 0; k <= N; ++k) {
    const OcpQpStage& s = problem.stages[k];
    const vector_t& xk = x[k];
    objective += 0.5 * xk.dot(s.Q * xk) + s.q.dot(xk);
    if (k < N) {
      const vector_t& uk = u[k];
      objective += 0.5 * uk.dot(s.R * uk) + uk.dot(s.S * xk) + s.r.dot(uk);
    }
    if (!s.softGeneralIndices.empty()) {
      vector_t g = s.C * xk;
      if (k < N) {
        g.noalias() += s.D * u[k];
      }
      for (std::size_t i = 0; i < s.softGeneralIndices.size(); ++i) {
        const int row = s.softGeneralIndices[i];
        const scalar_t lowerViolation = std::max(0.0, s.lg(row) - g(row));
        const scalar_t upperViolation = std::max(0.0, g(row) - s.ug(row));
        objective += 0.5 * s.Zl(i) * lowerViolation * lowerViolation + s.zl(i) * lowerViolation;
        objective += 0.5 * s.Zu(i) * upperViolation * upperViolation + s.zu(i) * upperViolation;
      }
    }
  }
  return objective;
}

scalar_t evaluateOcpQpMaxHardViolation(const OcpQpProblem& problem, const std::vector<vector_t>& x, const std::vector<vector_t>& u) {
  const int N = problem.numStages();
  scalar_t maxViolation = (x[0] - problem.x0).cwiseAbs().maxCoeff();
  for (int k = 0; k <= N; ++k) {
    const OcpQpStage& s = problem.stages[k];
    const vector_t& xk = x[k];
    for (std::size_t i = 0; i < s.idxbx.size(); ++i) {
      maxViolation = std::max({maxViolation, s.lbx(i) - xk(s.idxbx[i]), xk(s.idxbx[i]) - s.ubx(i)});
    }
    if (k < N) {
      const vector_t& uk = u[k];
      for (std::size_t i = 0; i < s.idxbu.size(); ++i) {
        maxViolation = std::max({maxViolation, s.lbu(i) - uk(s.idxbu[i]), uk(s.idxbu[i]) - s.ubu(i)});
      }
      const vector_t residual = s.A * xk + s.B * uk + s.b - x[k + 1];
      maxViolation = std::max(maxViolation, residual.cwiseAbs().maxCoeff());
    }
    if (s.numGeneralConstraints() > 0) {
      vector_t g = s.C * xk;
      if (k < N) {
        g.noalias() += s.D * u[k];
      }
      for (int row = 0; row < s.numGeneralConstraints(); ++row) {
        const bool isSoft = std::binary_search(s.softGeneralIndices.begin(), s.softGeneralIndices.end(), row);
        if (isSoft) continue;
        maxViolation = std::max({maxViolation, s.lg(row) - g(row), g(row) - s.ug(row)});
      }
    }
  }
  return maxViolation;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

class OcpQpHpipmSolver::Impl {
 public:
  explicit Impl(const Settings& settings) : settings_(settings) {}

  void allocate(const Dimensions& dims) {
    if (allocated_ && dims == dims_) {
      return;
    }
    dims_ = dims;
    dimMem_.reserve(d_ocp_qp_dim_memsize(dims_.N));
    d_ocp_qp_dim_create(dims_.N, &dim_, dimMem_.get());
    d_ocp_qp_dim_set_all(dims_.nx.data(), dims_.nu.data(), dims_.nbx.data(), dims_.nbu.data(), dims_.ng.data(), dims_.nsbx.data(),
                         dims_.nsbu.data(), dims_.nsg.data(), &dim_);

    qpMem_.reserve(d_ocp_qp_memsize(&dim_));
    d_ocp_qp_create(&dim_, &qp_, qpMem_.get());

    solMem_.reserve(d_ocp_qp_sol_memsize(&dim_));
    d_ocp_qp_sol_create(&dim_, &sol_, solMem_.get());

    argMem_.reserve(d_ocp_qp_ipm_arg_memsize(&dim_));
    d_ocp_qp_ipm_arg_create(&dim_, &arg_, argMem_.get());
    d_ocp_qp_ipm_arg_set_default(static_cast<hpipm_mode>(settings_.hpipmMode), &arg_);
    d_ocp_qp_ipm_arg_set_iter_max(&settings_.iterMax, &arg_);
    d_ocp_qp_ipm_arg_set_alpha_min(&settings_.alphaMin, &arg_);
    d_ocp_qp_ipm_arg_set_mu0(&settings_.mu0, &arg_);
    d_ocp_qp_ipm_arg_set_tol_stat(&settings_.tolStat, &arg_);
    d_ocp_qp_ipm_arg_set_tol_eq(&settings_.tolEq, &arg_);
    d_ocp_qp_ipm_arg_set_tol_ineq(&settings_.tolIneq, &arg_);
    d_ocp_qp_ipm_arg_set_tol_comp(&settings_.tolComp, &arg_);
    d_ocp_qp_ipm_arg_set_reg_prim(&settings_.regPrim, &arg_);
    d_ocp_qp_ipm_arg_set_warm_start(&settings_.warmStart, &arg_);
    d_ocp_qp_ipm_arg_set_pred_corr(&settings_.predCorr, &arg_);

    wsMem_.reserve(d_ocp_qp_ipm_ws_memsize(&dim_, &arg_));
    d_ocp_qp_ipm_ws_create(&dim_, &arg_, &ws_, wsMem_.get());
    allocated_ = true;
  }

  OcpQpSolution solve(const OcpQpProblem& problem) {
    const int N = problem.numStages();
    if (N < 1) {
      throw std::invalid_argument("[OcpQpHpipmSolver] the problem needs at least one stage");
    }
    for (int k = 0; k <= N; ++k) {
      checkStage(problem.stages[k], k, N);
      if (k < N && problem.stages[k].A.rows() != problem.stages[k + 1].numStates()) {
        throw std::invalid_argument("[OcpQpHpipmSolver] dynamics of stage " + std::to_string(k) + " do not match the next state size");
      }
    }
    if (problem.x0.size() != problem.stages[0].numStates()) {
      throw std::invalid_argument("[OcpQpHpipmSolver] x0 does not match the state dimension of stage 0");
    }

    // The initial state is imposed as an equality box constraint on the stage-0 state (it replaces any user box there).
    const int nx0 = problem.stages[0].numStates();
    std::vector<int> idxbx0(nx0);
    for (int i = 0; i < nx0; ++i) idxbx0[i] = i;
    vector_t lbx0 = problem.x0;
    vector_t ubx0 = problem.x0;

    Dimensions dims;
    dims.N = N;
    const auto resizeAll = [&](std::vector<int>& v) { v.assign(N + 1, 0); };
    resizeAll(dims.nx);
    resizeAll(dims.nu);
    resizeAll(dims.nbx);
    resizeAll(dims.nbu);
    resizeAll(dims.ng);
    resizeAll(dims.nsbx);
    resizeAll(dims.nsbu);
    resizeAll(dims.nsg);
    for (int k = 0; k <= N; ++k) {
      const OcpQpStage& s = problem.stages[k];
      dims.nx[k] = s.numStates();
      dims.nu[k] = s.numInputs();
      dims.nbx[k] = (k == 0) ? nx0 : static_cast<int>(s.idxbx.size());
      dims.nbu[k] = static_cast<int>(s.idxbu.size());
      dims.ng[k] = s.numGeneralConstraints();
      dims.nsg[k] = static_cast<int>(s.softGeneralIndices.size());
    }
    allocate(dims);

    // Pointer tables. HPIPM copies the data inside d_ocp_qp_set_all, so stack-lifetime buffers are fine.
    std::vector<double*> A(N + 1, nullptr), B(N + 1, nullptr), b(N + 1, nullptr);
    std::vector<double*> Q(N + 1, nullptr), S(N + 1, nullptr), R(N + 1, nullptr), q(N + 1, nullptr), r(N + 1, nullptr);
    std::vector<int*> idxbx(N + 1, nullptr), idxbu(N + 1, nullptr), idxs(N + 1, nullptr);
    std::vector<double*> lbx(N + 1, nullptr), ubx(N + 1, nullptr), lbu(N + 1, nullptr), ubu(N + 1, nullptr);
    std::vector<double*> C(N + 1, nullptr), D(N + 1, nullptr), lg(N + 1, nullptr), ug(N + 1, nullptr);
    std::vector<double*> Zl(N + 1, nullptr), Zu(N + 1, nullptr), zl(N + 1, nullptr), zu(N + 1, nullptr), lls(N + 1, nullptr),
        lus(N + 1, nullptr);

    // Buffers that must outlive the set_all call.
    std::vector<std::vector<int>> idxsBuffers(N + 1);
    std::vector<vector_t> zeroSlackBounds(N + 1);
    std::vector<matrix_t> C0Buffer(1);

    for (int k = 0; k <= N; ++k) {
      OcpQpStage& s = const_cast<OcpQpStage&>(problem.stages[k]);  // HPIPM takes non-const pointers but does not write.
      if (k < N) {
        A[k] = s.A.data();
        B[k] = s.B.data();
        b[k] = s.b.data();
      }
      Q[k] = s.Q.data();
      q[k] = s.q.data();
      if (s.numInputs() > 0) {
        R[k] = s.R.data();
        S[k] = s.S.data();
        r[k] = s.r.data();
      }
      if (k == 0) {
        idxbx[k] = idxbx0.data();
        lbx[k] = lbx0.data();
        ubx[k] = ubx0.data();
      } else if (!s.idxbx.empty()) {
        idxbx[k] = s.idxbx.data();
        lbx[k] = s.lbx.data();
        ubx[k] = s.ubx.data();
      }
      if (!s.idxbu.empty()) {
        idxbu[k] = s.idxbu.data();
        lbu[k] = s.lbu.data();
        ubu[k] = s.ubu.data();
      }
      if (s.numGeneralConstraints() > 0) {
        C[k] = s.C.data();
        if (s.numInputs() > 0) {
          D[k] = s.D.data();
        }
        lg[k] = s.lg.data();
        ug[k] = s.ug.data();
      }
      if (!s.softGeneralIndices.empty()) {
        // Soft constraint indices count the box constraints first (inputs, then states), then the general rows.
        const int numBox = dims.nbu[k] + dims.nbx[k];
        idxsBuffers[k].resize(s.softGeneralIndices.size());
        for (std::size_t i = 0; i < s.softGeneralIndices.size(); ++i) {
          idxsBuffers[k][i] = numBox + s.softGeneralIndices[i];
        }
        idxs[k] = idxsBuffers[k].data();
        Zl[k] = s.Zl.data();
        Zu[k] = s.Zu.data();
        zl[k] = s.zl.data();
        zu[k] = s.zu.data();
        zeroSlackBounds[k] = vector_t::Zero(static_cast<Eigen::Index>(s.softGeneralIndices.size()));
        lls[k] = zeroSlackBounds[k].data();
        lus[k] = zeroSlackBounds[k].data();
      }
    }

    d_ocp_qp_set_all(A.data(), B.data(), b.data(), Q.data(), S.data(), R.data(), q.data(), r.data(), idxbx.data(), lbx.data(), ubx.data(),
                     idxbu.data(), lbu.data(), ubu.data(), C.data(), D.data(), lg.data(), ug.data(), Zl.data(), Zu.data(), zl.data(),
                     zu.data(), idxs.data(), lls.data(), lus.data(), &qp_);

    // One-sided general constraints are declared with +-kOcpQpInfiniteBound; mask those bounds out so that HPIPM does not
    // carry a slack and a multiplier for a bound that can never become active.
    for (int k = 0; k <= N; ++k) {
      const OcpQpStage& s = problem.stages[k];
      const int ng = s.numGeneralConstraints();
      if (ng == 0) continue;
      vector_t lgMask = vector_t::Ones(ng);
      vector_t ugMask = vector_t::Ones(ng);
      for (int i = 0; i < ng; ++i) {
        if (s.lg(i) <= -0.5 * kOcpQpInfiniteBound) lgMask(i) = 0.0;
        if (s.ug(i) >= 0.5 * kOcpQpInfiniteBound) ugMask(i) = 0.0;
      }
      d_ocp_qp_set_lg_mask(k, lgMask.data(), &qp_);
      d_ocp_qp_set_ug_mask(k, ugMask.data(), &qp_);
    }

    d_ocp_qp_ipm_solve(&qp_, &sol_, &arg_, &ws_);

    OcpQpSolution solution;
    int status = -1;
    d_ocp_qp_ipm_get_status(&ws_, &status);
    d_ocp_qp_ipm_get_iter(&ws_, &solution.iterations);
    switch (status) {
      case hpipm_status::SUCCESS:
        solution.status = OcpQpSolution::Status::SUCCESS;
        break;
      case hpipm_status::MAX_ITER:
        solution.status = OcpQpSolution::Status::MAX_ITER;
        break;
      case hpipm_status::MIN_STEP:
        solution.status = OcpQpSolution::Status::MIN_STEP;
        break;
      case hpipm_status::NAN_SOL:
        solution.status = OcpQpSolution::Status::NAN_SOL;
        break;
      case hpipm_status::INCONS_EQ:
        solution.status = OcpQpSolution::Status::INCONS_EQ;
        break;
      default:
        solution.status = OcpQpSolution::Status::UNKNOWN;
    }

    solution.x.resize(N + 1);
    solution.u.resize(N);
    bool finite = true;
    for (int k = 0; k <= N; ++k) {
      solution.x[k].resize(dims.nx[k]);
      d_ocp_qp_sol_get_x(k, &sol_, solution.x[k].data());
      finite = finite && solution.x[k].allFinite();
      if (k < N) {
        solution.u[k].resize(dims.nu[k]);
        d_ocp_qp_sol_get_u(k, &sol_, solution.u[k].data());
        finite = finite && solution.u[k].allFinite();
      }
    }
    if (!finite) {
      solution.status = OcpQpSolution::Status::NAN_SOL;
      solution.objective = std::numeric_limits<scalar_t>::infinity();
      return solution;
    }
    solution.objective = evaluateOcpQpObjective(problem, solution.x, solution.u);
    return solution;
  }

 private:
  Settings settings_;
  bool allocated_ = false;
  Dimensions dims_;

  MemoryBlock dimMem_, qpMem_, solMem_, argMem_, wsMem_;
  d_ocp_qp_dim dim_{};
  d_ocp_qp qp_{};
  d_ocp_qp_sol sol_{};
  d_ocp_qp_ipm_arg arg_{};
  d_ocp_qp_ipm_ws ws_{};
};

OcpQpHpipmSolver::OcpQpHpipmSolver() : OcpQpHpipmSolver(Settings()) {}

OcpQpHpipmSolver::OcpQpHpipmSolver(const Settings& settings) : pImpl_(new Impl(settings)), settings_(settings) {}

OcpQpHpipmSolver::~OcpQpHpipmSolver() = default;

OcpQpSolution OcpQpHpipmSolver::solve(const OcpQpProblem& problem) {
  return pImpl_->solve(problem);
}

}  // namespace ocs2::humanoid
