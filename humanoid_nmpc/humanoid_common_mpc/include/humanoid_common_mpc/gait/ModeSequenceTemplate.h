/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

/**
 * ModeSequenceTemplate describes a periodic sequence of modes. It is defined by
 * - switching times (size N+1), where the first time is 0, and the last time denotes the period of the cycle
 * - modeSequence (size N), indicating the mode between the switching times.
 */
struct ModeSequenceTemplate {
  /**
   * Constructor for a ModeSequenceTemplate. A valid template has N > 0 modes and N + 1 strictly increasing switching
   * times; the constructor takes any, and modeSequenceTemplateFromConfig() refuses an invalid one.
   * @param [in] switchingTimesInput : switching times of size N + 1
   * @param [in] modeSequenceInput : mode sequence of size N
   */
  ModeSequenceTemplate(std::vector<scalar_t> switchingTimesInput, std::vector<size_t> modeSequenceInput)
      : switchingTimes(std::move(switchingTimesInput)), modeSequence(std::move(modeSequenceInput)) {}

  /**
   * Defined as [t_0=0, t_1, .., t_n, t_(n+1)=T], where T is the overall duration
   * of the template logic. t_1 to t_n are the event moments.
   */
  std::vector<scalar_t> switchingTimes;

  /**
   * Defined as [sys_0, sys_n], are the switching systems IDs. Here sys_i is
   * active in period [t_i, t_(i+1)]
   */
  std::vector<size_t> modeSequence;
};

/** Swap two modesequence templates */
inline void swap(ModeSequenceTemplate& lh, ModeSequenceTemplate& rh) noexcept {
  lh.switchingTimes.swap(rh.switchingTimes);
  lh.modeSequence.swap(rh.modeSequence);
}

/** Formats the mode sequence template for absl::StrCat, absl::StrFormat("%v") and Abseil logging: two lines. */
template <typename Sink>
void AbslStringify(Sink& sink, const ModeSequenceTemplate& modeSequenceTemplate) {
  absl::Format(&sink, "Template switching times: {%s}\nTemplate mode sequence:   {%s}\n",
               absl::StrJoin(modeSequenceTemplate.switchingTimes, ", "), absl::StrJoin(modeSequenceTemplate.modeSequence, ", "));
}

}  // namespace ocs2::humanoid
