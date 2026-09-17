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

#include "humanoid_common_mpc/contact_planning/model/VerticalDoubleIntegrator.h"

#include <sstream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"

namespace ocs2::humanoid {

std::string VerticalDoubleIntegrator::describe() const {
  std::ostringstream out;
  out << "z'' = az, -g <= az <= 2 a_max - g with a_max = " << maxContactAcceleration_ << " m/s^2 per stance foot (F_max / m)";
  return out.str();
}

void VerticalDoubleIntegrator::configure(const ContactPlanningConfig& config) {
  const scalar_t aMax = config.verticalDoubleIntegrator.maxContactAcceleration;
  if (!(aMax > config.shared.gravity)) {
    throw std::invalid_argument(
        "[vertical_double_integrator] maxContactAcceleration must exceed gravity, or one foot cannot carry the robot");
  }
  maxContactAcceleration_ = aMax;
  gravity_ = config.shared.gravity;
}

void VerticalDoubleIntegrator::bind(const Layout& layout) {
  idx_.bind(layout);
}

void VerticalDoubleIntegrator::declareVariables(LayoutBuilder& layout) const {
  layout.addState(var::kHeight);
  layout.addState(var::kHeightRate);
  layout.addInput(var::kHeightAccel);
}

void VerticalDoubleIntegrator::addDynamics(const ContactPlanningContext& ctx, int /*node*/, OcpQpStage& s) const {
  const scalar_t dt = ctx.dt;
  s.A(idx_.height, idx_.height) = 1.0;
  s.A(idx_.height, idx_.heightRate) = dt;
  s.A(idx_.heightRate, idx_.heightRate) = 1.0;
  s.B(idx_.height, idx_.heightAccel) = 0.5 * dt * dt;
  s.B(idx_.heightRate, idx_.heightAccel) = dt;
}

void VerticalDoubleIntegrator::addInputBounds(const ContactPlanningContext& /*ctx*/, int /*node*/, InputBoundsBuilder& bounds) const {
  bounds.add(idx_.heightAccel, -gravity_, 2.0 * maxContactAcceleration_ - gravity_);
}

void VerticalDoubleIntegrator::setInitialState(const ContactPlanningContext& ctx, vector_t& x0) const {
  x0(idx_.height) = ctx.input->comHeight;
  x0(idx_.heightRate) = ctx.input->comHeightRate;
}

void VerticalDoubleIntegrator::decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const {
  const int N = ctx.numNodes;
  plan.comHeight.resize(static_cast<size_t>(N) + 1);
  plan.comHeightRate.resize(static_cast<size_t>(N) + 1);
  plan.comHeightAccel.resize(static_cast<size_t>(N));
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[static_cast<size_t>(k)];
    plan.comHeight[static_cast<size_t>(k)] = x(idx_.height);
    plan.comHeightRate[static_cast<size_t>(k)] = x(idx_.heightRate);
    if (k < N) plan.comHeightAccel[static_cast<size_t>(k)] = result.solution.u[static_cast<size_t>(k)](idx_.heightAccel);
  }
}

}  // namespace ocs2::humanoid
