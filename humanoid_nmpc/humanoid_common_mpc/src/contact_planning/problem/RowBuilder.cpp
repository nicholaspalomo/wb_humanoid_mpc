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

#include "humanoid_common_mpc/contact_planning/problem/RowBuilder.h"

namespace ocs2::humanoid {

void RowBuilder::add(const Coefficients& xCoefficients,
                     const Coefficients& uCoefficients,
                     scalar_t lower,
                     scalar_t upper,
                     bool soft,
                     const SlackPenalty& penalty) {
  vector_t cRow = vector_t::Zero(nx_);
  vector_t dRow = vector_t::Zero(nu_);
  for (const auto& [index, value] : xCoefficients) cRow(index) += value;
  for (const auto& [index, value] : uCoefficients) dRow(index) += value;
  cRows_.push_back(std::move(cRow));
  dRows_.push_back(std::move(dRow));
  lower_.push_back(lower);
  upper_.push_back(upper);
  if (soft) {
    softRows_.push_back(static_cast<int>(cRows_.size()) - 1);
    softPenalties_.push_back(penalty);
  }
}

void RowBuilder::writeTo(OcpQpStage& stage) const {
  const int ng = static_cast<int>(cRows_.size());
  stage.C.resize(ng, nx_);
  stage.D.resize(ng, nu_);
  stage.lg.resize(ng);
  stage.ug.resize(ng);
  for (int i = 0; i < ng; ++i) {
    stage.C.row(i) = cRows_[i].transpose();
    stage.D.row(i) = dRows_[i].transpose();
    stage.lg(i) = lower_[i];
    stage.ug(i) = upper_[i];
  }
  stage.softGeneralIndices = softRows_;
  const int ns = static_cast<int>(softRows_.size());
  stage.Zl.resize(ns);
  stage.Zu.resize(ns);
  stage.zl.resize(ns);
  stage.zu.resize(ns);
  for (int i = 0; i < ns; ++i) {
    stage.Zl(i) = softPenalties_[i].quadratic;
    stage.Zu(i) = softPenalties_[i].quadratic;
    stage.zl(i) = softPenalties_[i].linear;
    stage.zu(i) = softPenalties_[i].linear;
  }
}

}  // namespace ocs2::humanoid
