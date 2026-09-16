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

#include <cmath>

#include "humanoid_common_mpc/contact_planning/problem/LipIndices.h"
#include "humanoid_common_mpc/contact_planning/problem/LipModelBlock.h"

namespace ocs2::humanoid {

/**
 * `heading_double_integrator`: the whole-body heading theta, its rate omega (angular momentum about the vertical over
 * the yaw inertia, taken from the input at every plan), and one yaw psi_i per foot; inputs are the yaw torque tau_i
 * each foot carries and the foot yaw displacements dpsi_i:
 *   omega_{k+1} = omega_k + dt (tau_L + tau_R) / I_zz,
 *   theta_{k+1} = theta_k + dt omega_k + dt^2 (tau_L + tau_R) / (2 I_zz)   (exact zero-order hold of the double integrator),
 *   psi_{i,k+1} = psi_{i,k} + dpsi_{i,k}.
 * The torque bound is the ground's: no foot ever carries more than the whole weight's torsion alone or half of the
 * double-support budget (the yaw_torque_budget rows do the rest).
 */
class HeadingDoubleIntegrator final : public LipModelBlock {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  void bind(const Layout& layout) override;
  void declareVariables(LayoutBuilder& layout) const override;
  void addDynamics(const ContactPlanningContext& ctx, int node, OcpQpStage& stage) const override;
  void addInputBounds(const ContactPlanningContext& ctx, int node, InputBoundsBuilder& bounds) const override;
  void setInitialState(const ContactPlanningContext& ctx, vector_t& x0) const override;
  void decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const override;

  static constexpr scalar_t kYawBigM = 2.0 * M_PI;  // [rad] bound on a foot yaw displacement while the foot is in the air

 private:
  scalar_t torqueBound_ = 0.0;
  LipIndices idx_;
};

}  // namespace ocs2::humanoid
