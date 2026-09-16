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

#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_common_mpc/contact_planning/search/SearchStage.h"

namespace ocs2::humanoid {

/**
 * `heading_relinearisation`: the foothold frame was linearised around a nominal heading trajectory; re-linearise it
 * around the incumbent's own heading and footholds and re-solve the QP with the contacts fixed (successive
 * linearisation), within the plan's time budget, so that the frame the constraints were written in is the frame the
 * plan actually turns through.
 */
class HeadingRelinearisationStage final : public SearchStage {
 public:
  std::string describe() const override;
  std::vector<std::string> requiredBlocks() const override { return {term::kHeadingDoubleIntegrator}; }
  void configure(const ContactPlanningConfig& config) override;
  void afterSearch(SearchRun& run) const override;

 private:
  int passes_ = 1;
  scalar_t timeBudget_ = 0.0;
};

}  // namespace ocs2::humanoid
