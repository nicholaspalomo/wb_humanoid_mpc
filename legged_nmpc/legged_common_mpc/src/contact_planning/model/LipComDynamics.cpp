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

#include "legged_common_mpc/contact_planning/model/LipComDynamics.h"

#include <cmath>
#include <cstddef>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/contact_planning/ContactPlan.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "legged_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "legged_common_mpc/contact_planning/OcpQpHpipm.h"
#include "legged_common_mpc/contact_planning/model/LipBlockIndices.h"
#include "legged_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "legged_common_mpc/contact_planning/problem/Layout.h"
#include "legged_common_mpc/contact_planning/problem/LayoutBuilder.h"

namespace ocs2::legged {

std::string LipComDynamics::describe() const {
  return absl::StrCat(
      "c_{k+1} = LIP(c_k, v_k, zmp_k), zero-order hold, omega = ", omega_,
      " 1/s (nomPosComRtFeetmeanEwrtWorldZ ", nomPosComRtFeetmeanEwrtWorldZ_,
      " m)");
}

void LipComDynamics::configure(const ContactPlanningConfig& config) {
  nomPosComRtFeetmeanEwrtWorldZ_ = config.getNomPosComRtFeetmeanEwrtWorldZ();
  omega_ = config.omega();
}

void LipComDynamics::bind(const Layout& layout) {
  idx_.bind(layout);
  // ContactPlanningProblem::finalize() places the LIP block first before it
  // binds any term.
  CHECK(idx_.posComRtWorldEwrtWorld[0] == kLipCx &&
        idx_.linvelComRtWorldEwrtWorld[0] == kLipVx &&
        idx_.posZmpRtWorldEwrtWorld[0] == kLipZx)
      << "[lip_com] the LIP block must be the first block of the layout";
}

void LipComDynamics::declareVariables(LayoutBuilder& layout) const {
  layout.addState(var::kPosComRtWorldEwrtWorldX);
  layout.addState(var::kPosComRtWorldEwrtWorldY);
  layout.addState(var::kLinvelComRtWorldEwrtWorldX);
  layout.addState(var::kLinvelComRtWorldEwrtWorldY);
  layout.addInput(var::kPosZmpRtWorldEwrtWorldX);
  layout.addInput(var::kPosZmpRtWorldEwrtWorldY);
}

void LipComDynamics::addDynamics(const ContactPlanningContext& ctx,
                                 int /*node*/,
                                 OcpQpStage& s) const {
  const scalar_t omega = ctx.omega;
  const scalar_t ch = std::cosh(omega * ctx.dt);
  const scalar_t sh = std::sinh(omega * ctx.dt);
  for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
    s.A(idx_.posComRtWorldEwrtWorld[axis], idx_.posComRtWorldEwrtWorld[axis]) =
        ch;
    s.A(idx_.posComRtWorldEwrtWorld[axis],
        idx_.linvelComRtWorldEwrtWorld[axis]) = sh / omega;
    s.A(idx_.linvelComRtWorldEwrtWorld[axis],
        idx_.posComRtWorldEwrtWorld[axis]) = omega * sh;
    s.A(idx_.linvelComRtWorldEwrtWorld[axis],
        idx_.linvelComRtWorldEwrtWorld[axis]) = ch;
    s.B(idx_.posComRtWorldEwrtWorld[axis], idx_.posZmpRtWorldEwrtWorld[axis]) =
        1.0 - ch;
    s.B(idx_.linvelComRtWorldEwrtWorld[axis],
        idx_.posZmpRtWorldEwrtWorld[axis]) = -omega * sh;
  }
}

void LipComDynamics::setInitialState(const ContactPlanningContext& ctx,
                                     vector_t& x0) const {
  for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
    x0(idx_.posComRtWorldEwrtWorld[axis]) =
        ctx.input->posComRtWorldEwrtWorldXy(axis);
    x0(idx_.linvelComRtWorldEwrtWorld[axis]) =
        ctx.input->proxyLinvelComRtWorldEwrtWorldXy(axis);
  }
}

void LipComDynamics::decode(const ContactPlanningContext& ctx,
                            const MiqpResult& result,
                            ContactPlan& plan) const {
  const int N = ctx.numNodes;
  plan.posComRtWorldEwrtWorldXyTraj.resize(static_cast<size_t>(N) + 1);
  plan.linvelComRtWorldEwrtWorldXyTraj.resize(static_cast<size_t>(N) + 1);
  plan.posZmpRtWorldEwrtWorldXyTraj.resize(static_cast<size_t>(N));
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[static_cast<size_t>(k)];
    plan.posComRtWorldEwrtWorldXyTraj[static_cast<size_t>(k)] = vector2_t(
        x(idx_.posComRtWorldEwrtWorld[0]), x(idx_.posComRtWorldEwrtWorld[1]));
    plan.linvelComRtWorldEwrtWorldXyTraj[static_cast<size_t>(k)] =
        vector2_t(x(idx_.linvelComRtWorldEwrtWorld[0]),
                  x(idx_.linvelComRtWorldEwrtWorld[1]));
    if (k < N) {
      const vector_t& u = result.solution.u[static_cast<size_t>(k)];
      plan.posZmpRtWorldEwrtWorldXyTraj[static_cast<size_t>(k)] = vector2_t(
          u(idx_.posZmpRtWorldEwrtWorld[0]), u(idx_.posZmpRtWorldEwrtWorld[1]));
    }
  }
}

}  // namespace ocs2::legged
