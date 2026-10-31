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

#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The surface patch a contact touches, as the contact terms written in its
 * frame read it (humanoid_nmpc/docs/multi_contact_planning/README.md, section
 * 7.2): the rotation that takes a world vector into the patch frame, whose z
 * axis is the patch's outward normal, and the patch's offset along that normal.
 *
 * A parameter of the contact phase, read by the terms at every evaluation from
 * the reference manager
 * (SwitchedModelReferenceManager::getContactPatchFrame()): it is never recorded
 * on a CppAD tape. The default is the world-aligned patch, the flat ground
 * every shipped robot stands on; a term given a world-aligned patch evaluates
 * exactly the arithmetic it evaluated before patches existed.
 */
struct ContactPatchFrame {
  /** p_R_w: the patch-frame coordinates of a world vector v are p_R_w v. */
  matrix3_t rotmatWorldRtSurface = matrix3_t::Identity();
  /** [m] n . o for a point o of the patch and its normal n: the height of a
   * world-aligned patch. */
  scalar_t normalOffset = 0.0;
};

/** Whether `patch` is world aligned: its rotation is exactly the identity, so
 * its normal is the world z axis. */
inline bool isWorldAligned(const ContactPatchFrame& patch) {
  return patch.rotmatWorldRtSurface == matrix3_t::Identity();
}

/** The outward normal of `patch` in the world frame: the last row of p_R_w. */
inline vector3_t patchNormal(const ContactPatchFrame& patch) {
  return patch.rotmatWorldRtSurface.row(kZAxisIndex).transpose();
}

}  // namespace ocs2::humanoid
