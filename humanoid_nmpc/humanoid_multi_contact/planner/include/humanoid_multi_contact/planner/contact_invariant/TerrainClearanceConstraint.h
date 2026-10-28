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
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/constraint/StateConstraintCppAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_multi_contact/collision/CollisionModel.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid::multi_contact {

/** One collision sphere on a joint of the MPC's model. Passive data. */
struct ClearanceSphere {
  pinocchio::JointIndex joint = 0;
  // [m] Its center in the joint's frame, and its radius.
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double radius = 0.0;
};

/**
 * The spheres of `collision` (built for `stanceModel`) on the joints of the
 * MPC's Pinocchio model `pinocchio`, by joint name. NotFound for a joint the
 * MPC's model does not have (a joint it fixes).
 */
absl::StatusOr<std::vector<ClearanceSphere>> clearanceSpheresOf(
    const CollisionModel& collision,
    const RobotStanceModel& stanceModel,
    const PinocchioInterface& pinocchio);

/**
 * Keeps the robot's collision spheres (design section 5.4) out of the
 * terrain:
 *
 *   h_k(x) = sphereDistance(c_k(q), r_k) - margin >= 0,
 *
 * one row per sphere, with c_k the sphere's center at the configuration of
 * the centroidal state x. With a margin of 0 a sphere may touch, as the
 * spheres of a resting pad's link do, so no sphere is exempt. The terrain is
 * recorded on the tape as constants: the library's name must carry the
 * scene's hash. Wrapped in a relaxed barrier (StateSoftConstraint) it is a
 * soft constraint of planner A's solves; the certifier fails a knot whose
 * spheres outside the active pads' neighborhood go more than 5 mm in.
 *
 * Cloned per solver worker; a single instance is not thread-safe.
 */
class TerrainClearanceConstraint final : public StateConstraintCppAd {
 public:
  /** The rows of `spheres` on the MPC's model `pinocchio` (copied) with
   * centroidal model `info`, taped as `libraryName` in `folder`. */
  TerrainClearanceConstraint(const PinocchioInterface& pinocchio,
                             const CentroidalModelInfo& info,
                             std::vector<ClearanceSphere> spheres,
                             const TerrainModel& terrain,
                             scalar_t margin,
                             const std::string& libraryName,
                             const std::string& folder,
                             bool recompile,
                             bool verbose);
  ~TerrainClearanceConstraint() override = default;
  TerrainClearanceConstraint* absl_nonnull clone() const override {
    return new TerrainClearanceConstraint(*this);
  }
  TerrainClearanceConstraint& operator=(const TerrainClearanceConstraint&) =
      delete;
  TerrainClearanceConstraint(TerrainClearanceConstraint&&) = delete;
  TerrainClearanceConstraint& operator=(TerrainClearanceConstraint&&) = delete;

  size_t getNumConstraints(scalar_t /*time*/) const override {
    return spheres_.size();
  }

 private:
  TerrainClearanceConstraint(const TerrainClearanceConstraint& other) = default;

  ad_vector_t constraintFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& parameters) const override;

  PinocchioInterfaceCppAd pinocchioCppAd_;
  CentroidalModelInfoCppAd infoCppAd_;
  std::vector<ClearanceSphere> spheres_;
  TerrainModel terrain_;
  scalar_t margin_ = 0.0;
};

}  // namespace ocs2::humanoid::multi_contact
