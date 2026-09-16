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

#include "humanoid_common_mpc/contact_planning/model/LipComDynamics.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/model/LipBlockIndices.h"

namespace ocs2::humanoid {

std::string LipComDynamics::describe() const {
  std::ostringstream out;
  out << "c_{k+1} = LIP(c_k, v_k, zmp_k), zero-order hold, omega = " << omega_ << " 1/s (comHeight " << comHeight_ << " m)";
  return out.str();
}

void LipComDynamics::configure(const ContactPlanningConfig& config) {
  if (config.shared.comHeight <= 0.0 || config.shared.gravity <= 0.0) {
    throw std::invalid_argument("[lip_com] comHeight and gravity must be positive");
  }
  comHeight_ = config.shared.comHeight;
  omega_ = config.omega();
}

void LipComDynamics::bind(const Layout& layout) {
  idx_.bind(layout);
  if (idx_.com[0] != LIP_CX || idx_.vel[0] != LIP_VX || idx_.zmp[0] != LIP_ZX) {
    throw std::logic_error("[lip_com] the LIP block must be the first block of the layout");
  }
}

void LipComDynamics::declareVariables(LayoutBuilder& layout) const {
  layout.addState(var::kComX);
  layout.addState(var::kComY);
  layout.addState(var::kVelX);
  layout.addState(var::kVelY);
  layout.addInput(var::kZmpX);
  layout.addInput(var::kZmpY);
}

void LipComDynamics::addDynamics(const ContactPlanningContext& ctx, int /*node*/, OcpQpStage& s) const {
  const scalar_t omega = ctx.omega;
  const scalar_t ch = std::cosh(omega * ctx.dt);
  const scalar_t sh = std::sinh(omega * ctx.dt);
  for (int axis = 0; axis < 2; ++axis) {
    s.A(idx_.com[axis], idx_.com[axis]) = ch;
    s.A(idx_.com[axis], idx_.vel[axis]) = sh / omega;
    s.A(idx_.vel[axis], idx_.com[axis]) = omega * sh;
    s.A(idx_.vel[axis], idx_.vel[axis]) = ch;
    s.B(idx_.com[axis], idx_.zmp[axis]) = 1.0 - ch;
    s.B(idx_.vel[axis], idx_.zmp[axis]) = -omega * sh;
  }
}

void LipComDynamics::setInitialState(const ContactPlanningContext& ctx, vector_t& x0) const {
  for (int axis = 0; axis < 2; ++axis) {
    x0(idx_.com[axis]) = ctx.input->comPosition(axis);
    x0(idx_.vel[axis]) = ctx.input->comVelocity(axis);
  }
}

void LipComDynamics::decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const {
  const int N = ctx.numNodes;
  plan.comPosition.resize(static_cast<size_t>(N) + 1);
  plan.comVelocity.resize(static_cast<size_t>(N) + 1);
  plan.zmp.resize(static_cast<size_t>(N));
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[static_cast<size_t>(k)];
    plan.comPosition[static_cast<size_t>(k)] = vector2_t(x(idx_.com[0]), x(idx_.com[1]));
    plan.comVelocity[static_cast<size_t>(k)] = vector2_t(x(idx_.vel[0]), x(idx_.vel[1]));
    if (k < N) {
      const vector_t& u = result.solution.u[static_cast<size_t>(k)];
      plan.zmp[static_cast<size_t>(k)] = vector2_t(u(idx_.zmp[0]), u(idx_.zmp[1]));
    }
  }
}

}  // namespace ocs2::humanoid
