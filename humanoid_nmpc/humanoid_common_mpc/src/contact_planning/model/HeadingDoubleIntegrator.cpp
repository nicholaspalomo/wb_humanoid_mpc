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

#include "humanoid_common_mpc/contact_planning/model/HeadingDoubleIntegrator.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

std::string HeadingDoubleIntegrator::describe() const {
  std::ostringstream out;
  out << "theta'' = (tau_L + tau_R) / I_zz (I_zz from the model at every plan), psi_{i,k+1} = psi_{i,k} + dpsi_{i,k}, |tau_i| <= "
      << torqueBound_ << " N m, |dpsi_i| <= 2 pi";
  return out.str();
}

void HeadingDoubleIntegrator::configure(const ContactPlanningConfig& config) {
  const scalar_t torsion = config.yawTorqueBudget.torsionalFrictionTorque;
  const scalar_t couple = config.yawTorqueBudget.doubleSupportYawCouple;
  if (torsion < 0.0 || couple < 0.0) throw std::invalid_argument("[heading_double_integrator] yaw torque limits must be >= 0");
  // No foot ever carries more than the whole weight's torsion alone or half of the double-support budget.
  torqueBound_ = std::max(torsion, 0.5 * (torsion + couple));
}

void HeadingDoubleIntegrator::bind(const Layout& layout) {
  idx_.bind(layout);
}

void HeadingDoubleIntegrator::declareVariables(LayoutBuilder& layout) const {
  layout.addState(var::kHeading);
  layout.addState(var::kHeadingRate);
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) layout.addState(var::footYaw(foot));
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) layout.addInput(var::yawTorque(foot));
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) layout.addInput(var::footYawDelta(foot));
}

void HeadingDoubleIntegrator::addDynamics(const ContactPlanningContext& ctx, int /*node*/, OcpQpStage& s) const {
  const scalar_t dt = ctx.dt;
  const scalar_t inertia = ctx.yawInertia;
  s.A(idx_.heading, idx_.heading) = 1.0;
  s.A(idx_.heading, idx_.headingRate) = dt;
  s.A(idx_.headingRate, idx_.headingRate) = 1.0;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    s.A(idx_.footYaw[foot], idx_.footYaw[foot]) = 1.0;
    s.B(idx_.heading, idx_.yawTorque[foot]) = 0.5 * dt * dt / inertia;
    s.B(idx_.headingRate, idx_.yawTorque[foot]) = dt / inertia;
    s.B(idx_.footYaw[foot], idx_.footYawDelta[foot]) = 1.0;
  }
}

void HeadingDoubleIntegrator::addInputBounds(const ContactPlanningContext& /*ctx*/, int /*node*/, InputBoundsBuilder& bounds) const {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    bounds.add(idx_.yawTorque[foot], -torqueBound_, torqueBound_);
    bounds.add(idx_.footYawDelta[foot], -kYawBigM, kYawBigM);
  }
}

void HeadingDoubleIntegrator::setInitialState(const ContactPlanningContext& ctx, vector_t& x0) const {
  x0(idx_.heading) = ctx.input->heading;
  x0(idx_.headingRate) = ctx.input->headingRate;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) x0(idx_.footYaw[foot]) = ctx.input->footYaws[foot];
}

void HeadingDoubleIntegrator::decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const {
  const int N = ctx.numNodes;
  plan.heading.resize(static_cast<size_t>(N) + 1);
  plan.headingRate.resize(static_cast<size_t>(N) + 1);
  plan.footYaws.resize(static_cast<size_t>(N) + 1);
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[static_cast<size_t>(k)];
    plan.heading[static_cast<size_t>(k)] = x(idx_.heading);
    plan.headingRate[static_cast<size_t>(k)] = x(idx_.headingRate);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) plan.footYaws[static_cast<size_t>(k)][foot] = x(idx_.footYaw[foot]);
  }
}

}  // namespace ocs2::humanoid
