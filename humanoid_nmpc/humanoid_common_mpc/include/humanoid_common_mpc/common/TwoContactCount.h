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

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/BipedFeet.h"

namespace ocs2::humanoid {

/**
 * Returns OK when the contact set has exactly two contacts, and otherwise an
 * InvalidArgument that names `component` and the count:
 * "FootSeparationConstraint is written for two contacts; the contact set has 4
 * contacts". It checks the count only: the contact planner and its biped-only
 * terms read the set as a left and a right contact (kContactLeftIndex,
 * kContactRightIndex) and accept two point contacts as well as two 6-DoF feet.
 * A component calls it where it is built or validated (a Create() factory, a
 * configuration conversion), never per solve.
 */
inline absl::Status checkTwoContactCount(absl::string_view component,
                                         size_t numContacts) {
  if (numContacts == static_cast<size_t>(kBipedFootCount)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      component, " is written for two contacts; the contact set has ",
      numContacts, " contacts"));
}

}  // namespace ocs2::humanoid
