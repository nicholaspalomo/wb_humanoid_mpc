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

#include "legged_common_mpc/contact_planning/problem/Layout.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "legged_common_mpc/common/BipedFeet.h"

namespace ocs2::legged {

namespace var {
std::string footLabel(size_t foot) {
  return foot == kContactLeftIndex ? "L" : "R";
}
std::string posFootRtWorldEwrtWorldX(size_t foot) {
  return "p_" + footLabel(foot) + "x";
}
std::string posFootRtWorldEwrtWorldY(size_t foot) {
  return "p_" + footLabel(foot) + "y";
}
std::string errPosFootRtWorldEwrtWorldX(size_t foot) {
  return "dp_" + footLabel(foot) + "x";
}
std::string errPosFootRtWorldEwrtWorldY(size_t foot) {
  return "dp_" + footLabel(foot) + "y";
}
std::string contact(size_t foot) {
  return "c_" + footLabel(foot);
}
std::string eulerzyxFootRtWorldYaw(size_t foot) {
  return "psi_" + footLabel(foot);
}
std::string torqueFootEwrtWorldZ(size_t foot) {
  return "tau_" + footLabel(foot);
}
std::string errEulerzyxFootRtWorldYaw(size_t foot) {
  return "dpsi_" + footLabel(foot);
}
}  // namespace var

namespace {
int indexOf(absl::Span<const std::string> names,
            absl::string_view name,
            const char* absl_nonnull what) {
  const absl::Span<const std::string>::const_iterator it =
      std::find(names.begin(), names.end(), name);
  CHECK(it != names.end()) << "[Layout] no " << what << " named '" << name
                           << "'";
  return static_cast<int>(it - names.begin());
}
}  // namespace

int Layout::state(absl::string_view name) const {
  return indexOf(stateNames, name, "state");
}

int Layout::input(absl::string_view name) const {
  return indexOf(inputNames, name, "input");
}

bool Layout::hasState(absl::string_view name) const {
  return std::find(stateNames.begin(), stateNames.end(), name) !=
         stateNames.end();
}

bool Layout::hasInput(absl::string_view name) const {
  return std::find(inputNames.begin(), inputNames.end(), name) !=
         inputNames.end();
}

std::string Layout::describe() const {
  return absl::StrCat("x = [", absl::StrJoin(stateNames, " "), "] (", nx,
                      "), u = [", absl::StrJoin(inputNames, " "), "] (", nu,
                      ")");
}

}  // namespace ocs2::legged
