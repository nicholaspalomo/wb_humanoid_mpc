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

#include "humanoid_common_mpc/contact_planning/model/HeadingDoubleIntegrator.h"

#include <algorithm>
#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/InputBoundsBuilder.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"
#include "humanoid_common_mpc/contact_planning/problem/LayoutBuilder.h"

namespace ocs2::humanoid {

std::string HeadingDoubleIntegrator::describe() const {
  return absl::StrCat(
      "theta'' = (tau_L + tau_R) / I_zz (I_zz from the model at every plan), "
      "psi_{i,k+1} = psi_{i,k} + dpsi_{i,k}, |tau_i| <= ",
      maxTorqueFootEwrtWorldZ_, " N m, |dpsi_i| <= 2 pi");
}

void HeadingDoubleIntegrator::configure(const ContactPlanningConfig& config) {
  const scalar_t torsion =
      config.yawTorqueBudgetParameters.maxTorqueFootEwrtFootZ;
  const scalar_t couple =
      config.yawTorqueBudgetParameters.doubleSupportMaxTorqueFeetmeanEwrtWorldZ;
  // No foot ever carries more than the whole weight's torsion alone or half of
  // the double-support budget.
  maxTorqueFootEwrtWorldZ_ = std::max(torsion, 0.5 * (torsion + couple));
}

void HeadingDoubleIntegrator::bind(const Layout& layout) {
  idx_.bind(layout);
}

void HeadingDoubleIntegrator::declareVariables(LayoutBuilder& layout) const {
  layout.addState(var::kHeadingModelYaw);
  layout.addState(var::kHeadingModelYawRate);
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    layout.addState(var::eulerzyxFootRtWorldYaw(foot));
  }
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    layout.addInput(var::torqueFootEwrtWorldZ(foot));
  }
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    layout.addInput(var::errEulerzyxFootRtWorldYaw(foot));
  }
}

void HeadingDoubleIntegrator::addDynamics(const ContactPlanningContext& ctx,
                                          int /*node*/,
                                          OcpQpStage& s) const {
  const scalar_t dt = ctx.dt;
  const scalar_t inertiaComEwrtWorldZz = ctx.inertiaComEwrtWorldZz;
  s.A(idx_.headingModelYaw, idx_.headingModelYaw) = 1.0;
  s.A(idx_.headingModelYaw, idx_.headingModelYawRate) = dt;
  s.A(idx_.headingModelYawRate, idx_.headingModelYawRate) = 1.0;
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    s.A(idx_.eulerzyxFeetRtWorldYaw[foot], idx_.eulerzyxFeetRtWorldYaw[foot]) =
        1.0;
    s.B(idx_.headingModelYaw, idx_.torqueFeetEwrtWorldZ[foot]) =
        0.5 * dt * dt / inertiaComEwrtWorldZz;
    s.B(idx_.headingModelYawRate, idx_.torqueFeetEwrtWorldZ[foot]) =
        dt / inertiaComEwrtWorldZz;
    s.B(idx_.eulerzyxFeetRtWorldYaw[foot],
        idx_.errEulerzyxFeetRtWorldYaw[foot]) = 1.0;
  }
}

void HeadingDoubleIntegrator::addInputBounds(
    const ContactPlanningContext& /*ctx*/,
    int /*node*/,
    InputBoundsBuilder& bounds) const {
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    bounds.add(idx_.torqueFeetEwrtWorldZ[foot], -maxTorqueFootEwrtWorldZ_,
               maxTorqueFootEwrtWorldZ_);
    bounds.add(idx_.errEulerzyxFeetRtWorldYaw[foot], -kYawBigM, kYawBigM);
  }
}

void HeadingDoubleIntegrator::setInitialState(const ContactPlanningContext& ctx,
                                              vector_t& x0) const {
  x0(idx_.headingModelYaw) = ctx.input->initYawHeadingRtWorld;
  x0(idx_.headingModelYawRate) = ctx.input->initRateYawHeadingRtWorld;
  for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
    x0(idx_.eulerzyxFeetRtWorldYaw[foot]) =
        ctx.input->eulerzyxFeetRtWorldYaw[foot];
  }
}

void HeadingDoubleIntegrator::decode(const ContactPlanningContext& ctx,
                                     const MiqpResult& result,
                                     ContactPlan& plan) const {
  const int N = ctx.numNodes;
  plan.headingModelYawTraj.resize(static_cast<size_t>(N) + 1);
  plan.headingModelYawRateTraj.resize(static_cast<size_t>(N) + 1);
  plan.desEulerzyxFeetRtWorldYawTraj.resize(static_cast<size_t>(N) + 1);
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[static_cast<size_t>(k)];
    plan.headingModelYawTraj[static_cast<size_t>(k)] = x(idx_.headingModelYaw);
    plan.headingModelYawRateTraj[static_cast<size_t>(k)] =
        x(idx_.headingModelYawRate);
    for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
      plan.desEulerzyxFeetRtWorldYawTraj[static_cast<size_t>(k)][foot] =
          x(idx_.eulerzyxFeetRtWorldYaw[foot]);
    }
  }
}

}  // namespace ocs2::humanoid
