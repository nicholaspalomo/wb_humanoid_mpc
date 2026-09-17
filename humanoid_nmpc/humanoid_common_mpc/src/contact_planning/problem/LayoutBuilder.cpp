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

#include "humanoid_common_mpc/contact_planning/problem/LayoutBuilder.h"

#include <stdexcept>

namespace ocs2::humanoid {

int LayoutBuilder::addState(const std::string& name) {
  if (layout_.hasState(name)) throw std::invalid_argument("[Layout] state '" + name + "' declared twice");
  layout_.stateNames.push_back(name);
  layout_.nx = static_cast<int>(layout_.stateNames.size());
  return layout_.nx - 1;
}

int LayoutBuilder::addInput(const std::string& name) {
  if (layout_.hasInput(name)) throw std::invalid_argument("[Layout] input '" + name + "' declared twice");
  layout_.inputNames.push_back(name);
  layout_.nu = static_cast<int>(layout_.inputNames.size());
  return layout_.nu - 1;
}

Layout LayoutBuilder::build() const {
  Layout layout = layout_;
  layout.hasHeading = layout.hasState(var::kHeading);
  if (layout.hasHeading) {
    layout.heading = layout.state(var::kHeading);
    layout.headingRate = layout.state(var::kHeadingRate);
    layout.footYaw0 = layout.state(var::footYaw(0));
    layout.yawTorque0 = layout.input(var::yawTorque(0));
    layout.footYawDelta0 = layout.input(var::footYawDelta(0));
  }
  layout.hasHeight = layout.hasState(var::kHeight);
  if (layout.hasHeight) {
    layout.height = layout.state(var::kHeight);
    layout.heightRate = layout.state(var::kHeightRate);
    layout.heightAccel = layout.input(var::kHeightAccel);
  }
  return layout;
}

}  // namespace ocs2::humanoid
