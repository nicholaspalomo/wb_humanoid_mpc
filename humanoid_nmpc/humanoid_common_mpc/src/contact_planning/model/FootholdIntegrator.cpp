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

#include "humanoid_common_mpc/contact_planning/model/FootholdIntegrator.h"

#include <sstream>
#include <stdexcept>

#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "humanoid_common_mpc/contact_planning/model/LipBlockIndices.h"

namespace ocs2::humanoid {

static_assert(N_CONTACTS == 2, "the foothold block lays out two feet (LipStateIndex / LipInputIndex)");

std::string FootholdIntegrator::describe() const {
  std::ostringstream out;
  out << "p_{i,k+1} = p_{i,k} + dp_{i,k}, |dp| <= bigM = " << bigM_ << " m, contact binaries c_i in {0, 1}";
  return out.str();
}

void FootholdIntegrator::configure(const ContactPlanningConfig& config) {
  if (config.shared.bigM <= 0.0) throw std::invalid_argument("[foothold_integrator] bigM must be positive");
  bigM_ = config.shared.bigM;
}

void FootholdIntegrator::bind(const Layout& layout) {
  idx_.bind(layout);
  if (idx_.foot[0][0] != LIP_PLX || idx_.footDelta[0][0] != LIP_DLX || idx_.contact[0] != LIP_CL) {
    throw std::logic_error("[foothold_integrator] the foothold block must follow the LIP block in the layout");
  }
}

void FootholdIntegrator::declareVariables(LayoutBuilder& layout) const {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    layout.addState(var::footX(foot));
    layout.addState(var::footY(foot));
  }
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    layout.addInput(var::footDeltaX(foot));
    layout.addInput(var::footDeltaY(foot));
  }
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) layout.addInput(var::contact(foot));
}

void FootholdIntegrator::addDynamics(const ContactPlanningContext& /*ctx*/, int /*node*/, OcpQpStage& s) const {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int axis = 0; axis < 2; ++axis) {
      s.A(idx_.foot[foot][axis], idx_.foot[foot][axis]) = 1.0;
      s.B(idx_.foot[foot][axis], idx_.footDelta[foot][axis]) = 1.0;
    }
  }
}

void FootholdIntegrator::addInputBounds(const ContactPlanningContext& ctx, int /*node*/, InputBoundsBuilder& bounds) const {
  const scalar_t M = ctx.bigM;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int axis = 0; axis < 2; ++axis) bounds.add(idx_.footDelta[foot][axis], -M, M);
  }
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) bounds.add(idx_.contact[foot], 0.0, 1.0);
}

void FootholdIntegrator::setInitialState(const ContactPlanningContext& ctx, vector_t& x0) const {
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    for (int axis = 0; axis < 2; ++axis) x0(idx_.foot[foot][axis]) = ctx.input->footPositions[foot](axis);
  }
}

void FootholdIntegrator::decode(const ContactPlanningContext& ctx, const MiqpResult& result, ContactPlan& plan) const {
  const int N = ctx.numNodes;
  plan.footholds.resize(static_cast<size_t>(N) + 1);
  plan.contacts.resize(static_cast<size_t>(N));
  for (int k = 0; k <= N; ++k) {
    const vector_t& x = result.solution.x[static_cast<size_t>(k)];
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      plan.footholds[static_cast<size_t>(k)][foot] = vector2_t(x(idx_.foot[foot][0]), x(idx_.foot[foot][1]));
    }
    if (k < N) {
      for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
        plan.contacts[static_cast<size_t>(k)][foot] =
            result.assignment[static_cast<size_t>(ContactLogicState::contactBinaryIndex(k, foot))] == 1;
      }
    }
  }
}

std::vector<int> FootholdIntegrator::binaryInputs() const {
  std::vector<int> binaries;
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) binaries.push_back(idx_.contact[foot]);
  return binaries;
}

}  // namespace ocs2::humanoid
