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

#include "legged_common_mpc/contact_planning/problem/LipIndices.h"

#include <cstddef>

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/contact_planning/problem/Layout.h"

namespace ocs2::legged {

void LipIndices::bind(const Layout& layout) {
  posComRtWorldEwrtWorld = {layout.state(var::kPosComRtWorldEwrtWorldX),
                            layout.state(var::kPosComRtWorldEwrtWorldY)};
  linvelComRtWorldEwrtWorld = {layout.state(var::kLinvelComRtWorldEwrtWorldX),
                               layout.state(var::kLinvelComRtWorldEwrtWorldY)};
  posZmpRtWorldEwrtWorld = {layout.input(var::kPosZmpRtWorldEwrtWorldX),
                            layout.input(var::kPosZmpRtWorldEwrtWorldY)};
  for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
    posFeetRtWorldEwrtWorld[i] = {
        layout.state(var::posFootRtWorldEwrtWorldX(i)),
        layout.state(var::posFootRtWorldEwrtWorldY(i))};
    errPosFeetRtWorldEwrtWorld[i] = {
        layout.input(var::errPosFootRtWorldEwrtWorldX(i)),
        layout.input(var::errPosFootRtWorldEwrtWorldY(i))};
    contact[i] = layout.input(var::contact(i));
  }
  hasHeading = layout.hasHeading;
  headingModelYaw = layout.headingModelYaw;
  headingModelYawRate = layout.headingModelYawRate;
  for (size_t i = 0; i < static_cast<size_t>(kBipedFootCount); ++i) {
    eulerzyxFeetRtWorldYaw[i] =
        hasHeading ? layout.eulerzyxFootRtWorldYaw(i) : -1;
    torqueFeetEwrtWorldZ[i] = hasHeading ? layout.torqueFootEwrtWorldZ(i) : -1;
    errEulerzyxFeetRtWorldYaw[i] =
        hasHeading ? layout.errEulerzyxFootRtWorldYaw(i) : -1;
  }
}

}  // namespace ocs2::legged
