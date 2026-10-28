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

#include "absl/base/nullability.h"
#include "ocs2_core/constraint/StateConstraintCppAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"

namespace ocs2::humanoid {

/**
 * Keeps one contact's collision sphere out of the terrain while the contact is
 * not in contact (the soft constraint terrain_non_penetration; design sections
 * 5.4 and 7.2, humanoid_nmpc/docs/multi_contact_planning/README.md):
 *
 *   h(x) = d(p_c(q)) - r - margin >= 0,
 *
 * with d the terrain's signed distance (TerrainModel, its smooth minimum over
 * the primitives), p_c the center of the sphere of the contact's collision
 * geometry - the contact frame's origin moved by `centerOffset`, a vector in
 * the contact frame (minus the geom radius along the pad normal) - r its
 * radius and the margin [m]. The radius, the margin and the offset are
 * parameters of the tape; the terrain is recorded on it as constants, so the
 * library's name carries the scene's hash. Wrapped in a relaxed barrier it is
 * a soft constraint, inactive while the mode schedule has the contact in
 * contact, where its pad rests on the terrain.
 *
 * Like every OCS2 term it is cloned for each solver thread, and a single
 * instance is not thread-safe. The reference manager, the robot model and the
 * terrain must outlive it.
 */
class TerrainNonPenetrationConstraint final : public StateConstraintCppAd {
 public:
  /** The geometry of one contact's collision sphere. Passive data. */
  struct Sphere {
    // [m] The sphere's center in the contact frame.
    vector3_t centerOffset = vector3_t::Zero();
    // [m] Not negative.
    scalar_t radius = 0.0;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * The constraint of contact `contactIndex` (its frame
   * ModelSettings::contactNames[contactIndex]) and `sphere`, `margin` [m] off
   * `terrain`, taped as the CppAD library `libraryName` in the model settings'
   * folder: a name that must carry the terrain's identity (its hash).
   * `referenceManager`, `mpcRobotModel` and `terrain` are retained.
   */
  TerrainNonPenetrationConstraint(
      const SwitchedModelReferenceManager& referenceManager,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
      const multi_contact::TerrainModel& terrain,
      size_t contactIndex,
      const Sphere& sphere,
      scalar_t margin,
      const std::string& libraryName,
      const ModelSettings& modelSettings);
  ~TerrainNonPenetrationConstraint() override = default;
  TerrainNonPenetrationConstraint& operator=(
      const TerrainNonPenetrationConstraint&) = delete;
  TerrainNonPenetrationConstraint(TerrainNonPenetrationConstraint&&) = delete;
  TerrainNonPenetrationConstraint& operator=(
      TerrainNonPenetrationConstraint&&) = delete;
  TerrainNonPenetrationConstraint* absl_nonnull clone() const override {
    return new TerrainNonPenetrationConstraint(*this);
  }

  /** Not while the contact is in contact. */
  bool isActive(scalar_t time) const override;
  size_t getNumConstraints(scalar_t /*time*/) const override { return 1; }
  vector_t getParameters(scalar_t time,
                         const PreComputation& preComputation) const override;

  /** The parameters of the tape: the sphere's center offset, its radius and
   * the margin. */
  static constexpr int kNumParameters = kPositionDim + 2;

 private:
  TerrainNonPenetrationConstraint(const TerrainNonPenetrationConstraint& other);

  ad_vector_t constraintFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& parameters) const override;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  const MpcRobotModelBase<ad_scalar_t>* absl_nonnull mpcRobotModelPtr_;
  const multi_contact::TerrainModel* absl_nonnull terrainPtr_;
  const size_t contactIndex_;
  const std::string frameName_;
  const Sphere sphere_;
  const scalar_t margin_;
};

}  // namespace ocs2::humanoid
