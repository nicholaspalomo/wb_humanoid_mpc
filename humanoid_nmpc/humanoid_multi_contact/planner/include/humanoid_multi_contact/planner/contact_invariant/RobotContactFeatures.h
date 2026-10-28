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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/statusor.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/planner/contact_invariant/ContactFeatureFunction.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/** One candidate of a robot's contact features: its contact of the
 * centroidal MPC and its pad. Passive data. */
struct RobotFeatureCandidate {
  // The MPC's contact frame of the candidate (its name) and the contact's
  // index in the MPC's contact order (3-DoF contacts first).
  std::string frameName;
  size_t mpcContactIndex = 0;
  ContactKind kind = ContactKind::kPoint;
  // parent_R_pad of the contact set (ContactCandidate::padRotation): its z
  // axis is the terrain normal when the pad rests on a surface.
  Eigen::Matrix3d padRotation = Eigen::Matrix3d::Identity();
  // [m] The radius of the pad's collision geometry; 0 for a flat pad.
  double geomRadius = 0.0;
};

/** The constants recorded on a robot's contact-feature tape. Passive data. */
struct RobotFeatureSettings {
  // The friction coefficient of the point candidates' cones: at most the
  // terrain's, inscribed in the certifier's pyramids.
  double friction = 0.0;
  // eps of the cone's smooth norm, in units of the robot's weight.
  double coneSmoothing = 0.0;
  // [rad] The tilt a point pad's normal may have from the surface's for free
  // (design section 3.1, max_pad_tilt).
  double maxPointTilt = 0.0;
};

/**
 * The contact features (ContactFeatureRows.h) of `candidates` of a robot on
 * the centroidal MPC's model, its Pinocchio model `pinocchio` and centroidal
 * model `info`, against `terrain` (design section 4.2): for each candidate,
 * its pad's signed distance to the terrain at the pad's frame, the pad's
 * velocity tangent to the blended normal n(p) (TerrainModel::normal(), the
 * softmin-weighted normal of the nearest faces), the tilt of the pad's normal
 * from n(p), the friction cone of a point pad's force about n(p) and the
 * clearance of its collision sphere; and the normalized momentum rate of the
 * centroidal dynamics (getNormalizedCentroidalMomentumRate()). The terrain
 * and the constants are recorded on the tape: the library's name must carry
 * the scene's hash and the candidates.
 *
 * NotFound for a frame that is not one of the model's; the tape's errors
 * (ContactFeatureFunction::Create()).
 */
absl::StatusOr<std::unique_ptr<ContactFeatureFunction>> robotContactFeatures(
    const PinocchioInterface& pinocchio,
    const CentroidalModelInfo& info,
    const std::vector<RobotFeatureCandidate>& candidates,
    const TerrainModel& terrain,
    const RobotFeatureSettings& settings,
    const FeatureLibrarySettings& library);

}  // namespace ocs2::humanoid::multi_contact
