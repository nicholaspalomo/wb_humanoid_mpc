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

#pragma once

#include "humanoid_common_mpc/contact_planning/problem/LipIndices.h"
#include "humanoid_common_mpc/contact_planning/problem/LipModelBlock.h"

namespace ocs2::humanoid {

/**
 * `vertical_double_integrator`: the vertical motion of the centre of mass, so that a flight phase has a beginning and an
 * end the physics can produce. States z (CoM height) and vz, input az (vertical CoM acceleration):
 *   z_{k+1}  = z_k + dt vz_k + dt^2 az_k / 2,
 *   vz_{k+1} = vz_k + dt az_k                 (exact zero-order hold of the double integrator).
 * The input is bounded by gravity below (the ground cannot pull) and by the thrust of two feet above; how much of it a
 * node may use follows from the contact binaries in the vertical_thrust_limit rows (ballistic fall in flight). The
 * horizontal LIP keeps its fixed frequency omega = sqrt(g / z_nom).
 */
class VerticalDoubleIntegrator final : public LipModelBlock {
 public:
  std::string describe() const override;
  void configure(const ContactPlanningConfig& config) override;
  void bind(const Layout& layout) override;
  void declareVariables(LayoutBuilder& layout) const override;
  void addDynamics(const ContactPlanningContext& ctx, int node, OcpQpStage& stage) const override;
  void addInputBounds(const ContactPlanningContext& ctx, int node, InputBoundsBuilder& bounds) const override;
  void setInitialState(const ContactPlanningContext& ctx, vector_t& x0) const override;
  void decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const override;

  scalar_t maxContactAcceleration() const { return maxContactAcceleration_; }
  scalar_t gravity() const { return gravity_; }

 private:
  scalar_t maxContactAcceleration_ = 0.0;  // [m/s^2] per stance foot, upward, gravity included (F_max / m)
  scalar_t gravity_ = 9.81;
  LipIndices idx_;
};

}  // namespace ocs2::humanoid
