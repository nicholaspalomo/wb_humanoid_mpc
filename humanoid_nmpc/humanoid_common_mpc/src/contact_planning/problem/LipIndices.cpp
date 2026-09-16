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

#include "humanoid_common_mpc/contact_planning/problem/LipIndices.h"

namespace ocs2::humanoid {

void LipIndices::bind(const Layout& layout) {
  com = {layout.state(var::kComX), layout.state(var::kComY)};
  vel = {layout.state(var::kVelX), layout.state(var::kVelY)};
  zmp = {layout.input(var::kZmpX), layout.input(var::kZmpY)};
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    foot[i] = {layout.state(var::footX(i)), layout.state(var::footY(i))};
    footDelta[i] = {layout.input(var::footDeltaX(i)), layout.input(var::footDeltaY(i))};
    contact[i] = layout.input(var::contact(i));
  }
  hasHeading = layout.hasHeading;
  heading = layout.heading;
  headingRate = layout.headingRate;
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    footYaw[i] = hasHeading ? layout.footYaw(i) : -1;
    yawTorque[i] = hasHeading ? layout.yawTorque(i) : -1;
    footYawDelta[i] = hasHeading ? layout.footYawDelta(i) : -1;
  }
}

}  // namespace ocs2::humanoid
