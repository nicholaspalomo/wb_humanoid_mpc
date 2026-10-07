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

#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"

extern "C" {
#include "hpipm_common.h"        // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_ocp_qp.h"      // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_ocp_qp_dim.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_ocp_qp_ipm.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "hpipm_d_ocp_qp_sol.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
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

  /** Grows the block to at least `size` bytes; ResourceExhausted, with the block left empty, when malloc fails. */
  absl::Status reserve(size_t size) {
    if (size > size_) {
      std::free(ptr_);
      ptr_ = std::malloc(size);
      if (ptr_ == nullptr) {
        size_ = 0;
        return absl::ResourceExhaustedError(absl::StrCat("[OcpQpHpipmSolver] cannot allocate ", size, " bytes for HPIPM"));
      }
      size_ = size;
    }
    return absl::OkStatus();
  }
  void* absl_nullable get() { return ptr_; }

 private:
  void* absl_nullable ptr_ = nullptr;
  size_t size_ = 0;
};

struct Dimensions {
  int N = 0;
  std::vector<int> nx, nu, nbx, nbu, ng, nsbx, nsbu, nsg;

  bool operator==(const Dimensions& rhs) const {
    return N == rhs.N && nx == rhs.nx && nu == rhs.nu && nbx == rhs.nbx && nbu == rhs.nbu && ng == rhs.ng && nsbx == rhs.nsbx &&
           nsbu == rhs.nsbu && nsg == rhs.nsg;
  }
};

absl::Status stageError(int k, absl::string_view what) {
  return absl::InvalidArgumentError(absl::StrCat("[OcpQpHpipmSolver] stage ", k, ": ", what));
}

/** Checks the sizes and indices of one stage; the InvalidArgument names the stage and what is inconsistent. */
absl::Status checkStage(const OcpQpStage& stage, int k, int N) {
  const int nx = stage.numStates();
  const int nu = stage.numInputs();
  if (k == N && nu != 0) return stageError(k, "terminal node must not have inputs");
  if (stage.Q.rows() != nx || stage.Q.cols() != nx) return stageError(k, "Q must be nx x nx");
  if (stage.R.rows() != nu || stage.R.cols() != nu) return stageError(k, "R must be nu x nu");
  if (nu > 0 && (stage.S.rows() != nu || stage.S.cols() != nx)) return stageError(k, "S must be nu x nx");
  if (stage.q.size() != nx) return stageError(k, "q must have nx entries");
  if (stage.r.size() != nu) return stageError(k, "r must have nu entries");
  if (k < N) {
    if (stage.A.rows() == 0 || stage.A.cols() != nx) return stageError(k, "A must be nx_next x nx");
    if (stage.B.rows() != stage.A.rows() || stage.B.cols() != nu) return stageError(k, "B must be nx_next x nu");
    if (stage.b.size() != stage.A.rows()) return stageError(k, "b must have nx_next entries");
  }
  if (static_cast<int>(stage.idxbu.size()) != stage.lbu.size() || stage.lbu.size() != stage.ubu.size()) {
    return stageError(k, "inconsistent input box");
  }
  if (static_cast<int>(stage.idxbx.size()) != stage.lbx.size() || stage.lbx.size() != stage.ubx.size()) {
    return stageError(k, "inconsistent state box");
  }
  for (int idx : stage.idxbu) {
    if (idx < 0 || idx >= nu) return stageError(k, "input box index out of range");
  }
  for (int idx : stage.idxbx) {
    if (idx < 0 || idx >= nx) return stageError(k, "state box index out of range");
  }
  const int ng = stage.numGeneralConstraints();
  if (ng > 0) {
    if (stage.C.cols() != nx) return stageError(k, "C must be ng x nx");
    if (stage.D.rows() != ng || stage.D.cols() != nu) return stageError(k, "D must be ng x nu");
    if (stage.lg.size() != ng || stage.ug.size() != ng) return stageError(k, "lg/ug must have ng entries");
  }
  const int ns = static_cast<int>(stage.softGeneralIndices.size());
  if (ns > 0) {
    if (stage.Zl.size() != ns || stage.Zu.size() != ns || stage.zl.size() != ns || stage.zu.size() != ns) {
      return stageError(k, "Zl/Zu/zl/zu must have one entry per soft constraint");
    }
    for (int idx : stage.softGeneralIndices) {
      if (idx < 0 || idx >= ng) return stageError(k, "soft constraint index out of range");
    }
    if (!std::is_sorted(stage.softGeneralIndices.begin(), stage.softGeneralIndices.end())) {
      return stageError(k, "soft constraint indices must be sorted");
    }
  }
  return absl::OkStatus();
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
  stage.constant = 0.0;
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
    // The stage's variable-independent part is summed with everything else. It used to be left out, which made this
    // function report the assembled functional minus a per-problem constant: harmless while only solutions of one
    // assembled problem are compared, wrong as soon as two problems assembled on different node grids are, because
    // their constants differ. See the comment on OcpQpStage::constant.
    objective += s.constant;
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
      for (size_t i = 0; i < s.softGeneralIndices.size(); ++i) {
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
    for (size_t i = 0; i < s.idxbx.size(); ++i) {
      maxViolation = std::max({maxViolation, s.lbx(i) - xk(s.idxbx[i]), xk(s.idxbx[i]) - s.ubx(i)});
    }
    if (k < N) {
      const vector_t& uk = u[k];
      for (size_t i = 0; i < s.idxbu.size(); ++i) {
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

  /**
   * Creates the HPIPM structures for `dims`, unless they already exist for these dimensions. A failed allocation leaves
   * nothing marked as allocated, so that the next solve starts over.
   */
  absl::Status allocate(const Dimensions& dims) {
    if (allocated_ && dims == dims_) {
      return absl::OkStatus();
    }
    allocated_ = false;
    dims_ = dims;
    RETURN_IF_ERROR(dimMem_.reserve(d_ocp_qp_dim_memsize(dims_.N)));
    d_ocp_qp_dim_create(dims_.N, &dim_, dimMem_.get());
    d_ocp_qp_dim_set_all(dims_.nx.data(), dims_.nu.data(), dims_.nbx.data(), dims_.nbu.data(), dims_.ng.data(), dims_.nsbx.data(),
                         dims_.nsbu.data(), dims_.nsg.data(), &dim_);

    RETURN_IF_ERROR(qpMem_.reserve(d_ocp_qp_memsize(&dim_)));
    d_ocp_qp_create(&dim_, &qp_, qpMem_.get());

    RETURN_IF_ERROR(solMem_.reserve(d_ocp_qp_sol_memsize(&dim_)));
    d_ocp_qp_sol_create(&dim_, &sol_, solMem_.get());

    RETURN_IF_ERROR(argMem_.reserve(d_ocp_qp_ipm_arg_memsize(&dim_)));
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

    RETURN_IF_ERROR(wsMem_.reserve(d_ocp_qp_ipm_ws_memsize(&dim_, &arg_)));
    d_ocp_qp_ipm_ws_create(&dim_, &arg_, &ws_, wsMem_.get());
    allocated_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<OcpQpSolution> solve(const OcpQpProblem& problem) {
    const int N = problem.numStages();
    if (N < 1) {
      return absl::InvalidArgumentError("[OcpQpHpipmSolver] the problem needs at least one stage");
    }
    for (int k = 0; k <= N; ++k) {
      RETURN_IF_ERROR(checkStage(problem.stages[k], k, N));
      if (k < N && problem.stages[k].A.rows() != problem.stages[k + 1].numStates()) {
        return absl::InvalidArgumentError(absl::StrCat("[OcpQpHpipmSolver] dynamics of stage ", k, " do not match the next state size"));
      }
    }
    if (problem.x0.size() != problem.stages[0].numStates()) {
      return absl::InvalidArgumentError("[OcpQpHpipmSolver] x0 does not match the state dimension of stage 0");
    }

    // The initial state is imposed as an equality box constraint on the stage-0 state (it replaces any user box there).
    const int nx0 = problem.stages[0].numStates();
    std::vector<int> idxbx0(nx0);
    for (int i = 0; i < nx0; ++i) idxbx0[i] = i;
    vector_t lbx0 = problem.x0;
    vector_t ubx0 = problem.x0;

    Dimensions dims;
    dims.N = N;
    const std::function<void(std::vector<int>&)> resizeAll = [N](std::vector<int>& v) { v.assign(N + 1, 0); };
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
    RETURN_IF_ERROR(allocate(dims));

    // Pointer tables. HPIPM copies the data inside d_ocp_qp_set_all, so stack-lifetime buffers are fine.
    std::vector<double* absl_nullable> A(N + 1, nullptr);
    std::vector<double* absl_nullable> B(N + 1, nullptr);
    std::vector<double* absl_nullable> b(N + 1, nullptr);
    std::vector<double* absl_nullable> Q(N + 1, nullptr);
    std::vector<double* absl_nullable> S(N + 1, nullptr);
    std::vector<double* absl_nullable> R(N + 1, nullptr);
    std::vector<double* absl_nullable> q(N + 1, nullptr);
    std::vector<double* absl_nullable> r(N + 1, nullptr);
    std::vector<int* absl_nullable> idxbx(N + 1, nullptr);
    std::vector<int* absl_nullable> idxbu(N + 1, nullptr);
    std::vector<int* absl_nullable> idxs(N + 1, nullptr);
    std::vector<double* absl_nullable> lbx(N + 1, nullptr);
    std::vector<double* absl_nullable> ubx(N + 1, nullptr);
    std::vector<double* absl_nullable> lbu(N + 1, nullptr);
    std::vector<double* absl_nullable> ubu(N + 1, nullptr);
    std::vector<double* absl_nullable> C(N + 1, nullptr);
    std::vector<double* absl_nullable> D(N + 1, nullptr);
    std::vector<double* absl_nullable> lg(N + 1, nullptr);
    std::vector<double* absl_nullable> ug(N + 1, nullptr);
    std::vector<double* absl_nullable> Zl(N + 1, nullptr);
    std::vector<double* absl_nullable> Zu(N + 1, nullptr);
    std::vector<double* absl_nullable> zl(N + 1, nullptr);
    std::vector<double* absl_nullable> zu(N + 1, nullptr);
    std::vector<double* absl_nullable> lls(N + 1, nullptr);
    std::vector<double* absl_nullable> lus(N + 1, nullptr);

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
        for (size_t i = 0; i < s.softGeneralIndices.size(); ++i) {
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

    // Rewrite the soft-constraint mapping in full, rather than trusting what d_ocp_qp_set_all left behind.
    //
    // HPIPM keeps the mapping in qp_.idxs_rev: one entry per constrained row of the stage, in the order (input box,
    // state box, general rows), holding the index of the slack pair that relaxes that row, or -1 when the row is hard.
    // That buffer is filled with -1 exactly once, inside d_ocp_qp_create, and d_ocp_qp_set_all then only touches the
    // rows that happen to be soft in the problem it is handed (it executes `idxs_rev[k][idxs[k][j]] = j` for the ns[k]
    // soft rows and clears nothing). Every other buffer of the QP - the dynamics, the Hessian, the constraint matrix,
    // the bounds, the slack weights - is rewritten in its entirety by that same call, which is what makes the
    // allocation-free reuse in allocate() safe; idxs_rev is the single exception.
    //
    // This mattered because allocate() keeps the previously created qp_ whenever the Dimensions compare equal, and
    // Dimensions records only the counts (N, nx, nu, nbx, nbu, ng, nsbx, nsbu, nsg), never which general rows the
    // caller declared soft. Previously, a second solve on the same solver instance with the same counts but a
    // different softGeneralIndices therefore inherited the first solve's entries: a row that the caller had declared
    // hard stayed mapped onto a slack pair, and shared that slack pair with the row that had just become soft. HPIPM
    // then relaxed a constraint the caller expected to be enforced, so the wrapper returned either a point violating a
    // hard row - which evaluateOcpQpMaxHardViolation, skipping exactly the rows listed in softGeneralIndices, reports
    // as a violation - or a spurious non-SUCCESS status on data that is perfectly feasible. Inside MixedIntegerOcpQp,
    // which holds one OcpQpHpipmSolver across hundreds of same-dimension relaxations, such a spurious failure silently
    // prunes a branch-and-bound node and the planner loses the solution without any diagnostic. Nothing in the
    // documented contract of this wrapper forbids the caller from changing which rows are soft between solves:
    // OcpQpStage accepts any sorted, in-range subset and checkStage validates exactly that. The formulation happens to
    // emit all hard rows before all soft ones today, which pins the soft set through (ng, nsg) alone and hides the
    // problem, but that ordering is an incidental choice of the assembler and must not be load bearing here.
    //
    // d_ocp_qp_set_idxs_rev writes all nb + ng entries of the stage, so stale entries are overwritten instead of being
    // left in place. The buffer is reused across the stages of the solve so that this costs at most one allocation.
    std::vector<int> idxsRevBuffer;
    for (int k = 0; k <= N; ++k) {
      const int numBox = dims.nbu[k] + dims.nbx[k];
      idxsRevBuffer.assign(static_cast<size_t>(numBox + dims.ng[k]), -1);
      const std::vector<int>& softGeneralIndices = problem.stages[k].softGeneralIndices;
      for (size_t i = 0; i < softGeneralIndices.size(); ++i) {
        idxsRevBuffer[static_cast<size_t>(numBox + softGeneralIndices[i])] = static_cast<int>(i);
      }
      d_ocp_qp_set_idxs_rev(k, idxsRevBuffer.data(), &qp_);
    }

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
        solution.status = OcpQpSolution::Status::kSuccess;
        break;
      case hpipm_status::MAX_ITER:
        solution.status = OcpQpSolution::Status::kMaxIter;
        break;
      case hpipm_status::MIN_STEP:
        solution.status = OcpQpSolution::Status::kMinStep;
        break;
      case hpipm_status::NAN_SOL:
        solution.status = OcpQpSolution::Status::kNanSol;
        break;
      case hpipm_status::INCONS_EQ:
        solution.status = OcpQpSolution::Status::kInconsEq;
        break;
      default:
        solution.status = OcpQpSolution::Status::kUnknown;
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
      solution.status = OcpQpSolution::Status::kNanSol;
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

OcpQpHpipmSolver::OcpQpHpipmSolver(const Settings& settings) : pImpl_(std::make_unique<Impl>(settings)), settings_(settings) {}

OcpQpHpipmSolver::~OcpQpHpipmSolver() = default;

absl::StatusOr<OcpQpSolution> OcpQpHpipmSolver::solve(const OcpQpProblem& problem) {
  return pImpl_->solve(problem);
}

}  // namespace ocs2::humanoid
