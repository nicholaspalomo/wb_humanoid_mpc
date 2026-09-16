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

#include <array>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

namespace ocs2::humanoid {

/**
 * The indices of the LIP and heading variables a term uses, resolved once from the layout in ContactPlanningTerm::bind().
 * Every entry is -1 when its block is absent (the heading ones); the LIP and foothold entries are always present.
 */
struct LipIndices {
  std::array<int, 2> com{-1, -1};                                                          // c_x, c_y
  std::array<int, 2> vel{-1, -1};                                                          // v_x, v_y
  std::array<int, 2> zmp{-1, -1};                                                          // zmp_x, zmp_y (input)
  feet_array_t<std::array<int, 2>> foot = makeFeetArray(std::array<int, 2>{-1, -1});       // p_i (x, y)
  feet_array_t<std::array<int, 2>> footDelta = makeFeetArray(std::array<int, 2>{-1, -1});  // dp_i (input)
  feet_array_t<int> contact = makeFeetArray(-1);                                           // c_i (binary input)
  bool hasHeading = false;
  int heading = -1;
  int headingRate = -1;
  feet_array_t<int> footYaw = makeFeetArray(-1);
  feet_array_t<int> yawTorque = makeFeetArray(-1);     // input
  feet_array_t<int> footYawDelta = makeFeetArray(-1);  // input

  void bind(const Layout& layout);
};

}  // namespace ocs2::humanoid
