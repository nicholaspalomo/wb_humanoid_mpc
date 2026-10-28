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

#include <string>

#include "absl/status/statusor.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

/** The repository paths of the DRC Atlas's multi-contact files. */
inline constexpr char kAtlasMultiContactTaskFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "task.textproto";
inline constexpr char kAtlasMultiContactReferenceFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "reference.textproto";
inline constexpr char kAtlasContactSetFile[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "contact_set.textproto";
inline constexpr char kAtlasUrdfFile[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

/** The DRC Atlas's multi-contact task file, its reference file and the URDF
 * in the test's runfiles; a missing one fails the calling test and is empty.
 */
CentroidalRobotFiles atlasMultiContactFiles();

/** The DRC Atlas's contact set (contact_set.textproto). */
absl::StatusOr<multi_contact::ContactSet> loadAtlasContactSet();

}  // namespace ocs2::humanoid
