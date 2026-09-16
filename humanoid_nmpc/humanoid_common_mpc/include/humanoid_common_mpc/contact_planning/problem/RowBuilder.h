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

#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/OcpQpHpipm.h"
#include "humanoid_common_mpc/contact_planning/problem/LipCoefficients.h"

namespace ocs2::humanoid {

/** Collects general constraint rows of one stage and writes them into the OcpQpStage, hard or soft with a penalty. */
class RowBuilder {
 public:
  RowBuilder(int nx, int nu) : nx_(nx), nu_(nu) {}

  /** A hard row lower <= C x + D u <= upper. */
  void addHard(const Coefficients& xCoefficients, const Coefficients& uCoefficients, scalar_t lower, scalar_t upper) {
    add(xCoefficients, uCoefficients, lower, upper, false, SlackPenalty{});
  }
  /** A soft row: the same, relaxed by slacks penalised with `penalty`. */
  void addSoft(
      const Coefficients& xCoefficients, const Coefficients& uCoefficients, scalar_t lower, scalar_t upper, const SlackPenalty& penalty) {
    add(xCoefficients, uCoefficients, lower, upper, true, penalty);
  }

  int numRows() const { return static_cast<int>(cRows_.size()); }
  int numSoftRows() const { return static_cast<int>(softRows_.size()); }

  void writeTo(OcpQpStage& stage) const;

 private:
  void add(const Coefficients& xCoefficients,
           const Coefficients& uCoefficients,
           scalar_t lower,
           scalar_t upper,
           bool soft,
           const SlackPenalty& penalty);

  int nx_, nu_;
  std::vector<vector_t> cRows_, dRows_;
  std::vector<scalar_t> lower_, upper_;
  std::vector<int> softRows_;
  std::vector<SlackPenalty> softPenalties_;
};

}  // namespace ocs2::humanoid
