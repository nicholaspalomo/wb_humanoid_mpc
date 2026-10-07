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

namespace ocs2::humanoid {

/**
 * Fixed indices of the LIP and foothold blocks: they are always the first two blocks, so these are the layout's 0..7.
 * Plain constants rather than an enum: they index Eigen vectors and matrices and are compared with Layout indices.
 */
// States: CoM position and velocity, then the left and right foot positions.
inline constexpr int kLipCx = 0;
inline constexpr int kLipCy = 1;
inline constexpr int kLipVx = 2;
inline constexpr int kLipVy = 3;
inline constexpr int kLipPlx = 4;
inline constexpr int kLipPly = 5;
inline constexpr int kLipPrx = 6;
inline constexpr int kLipPry = 7;
inline constexpr int kLipStateDim = 8;
// Inputs: ZMP, the left and right foot displacements, then the contact binaries.
inline constexpr int kLipZx = 0;
inline constexpr int kLipZy = 1;
inline constexpr int kLipDlx = 2;
inline constexpr int kLipDly = 3;
inline constexpr int kLipDrx = 4;
inline constexpr int kLipDry = 5;
inline constexpr int kLipCl = 6;
inline constexpr int kLipCr = 7;
inline constexpr int kLipInputDim = 8;

}  // namespace ocs2::humanoid
