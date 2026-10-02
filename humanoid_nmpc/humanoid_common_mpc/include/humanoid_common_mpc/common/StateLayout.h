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

#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * The name of the state layout the MPCs' CppAD libraries are generated for (design decision D14 of
 * humanoid_nmpc/docs/quaternion_base_orientation/README.md).
 *
 * Every shipped task file sets `recompileLibrariesCppAd: false`, and a library is reused whenever its .so file exists,
 * so a library taped for one state layout would silently be loaded for another. The library folder therefore carries
 * this tag (cppad_code_gen/cppad_<mpc><robot>/<tag>), and the tag changes whenever the layout of the state, the input or
 * the Pinocchio configuration a library reads changes: a new layout then generates new libraries in a new folder
 * instead of loading stale ones.
 *
 * "translation_spherical_quaternion_v1" is the layout of the quaternion switch: the root joint
 * Composite(Translation, Spherical), the base orientation stored as a unit quaternion (x, y, z, w) and the base angular
 * velocity in the body frame. The CppAD library folder adopts the tag in the same commit as that layout (Step 8 of the
 * design); until then nothing reads it.
 */
// LINT.IfChange(cppad_layout_tag)
inline constexpr absl::string_view kStateLayoutTag = "translation_spherical_quaternion_v1";
// LINT.ThenChange(//humanoid_nmpc/docs/quaternion_base_orientation/README.md)

}  // namespace ocs2::humanoid
