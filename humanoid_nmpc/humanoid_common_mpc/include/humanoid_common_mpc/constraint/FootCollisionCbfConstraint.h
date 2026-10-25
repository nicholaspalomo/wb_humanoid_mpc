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

#include "pinocchio/fwd.hpp"

#include <string>

#include "absl/base/nullability.h"
#include "ocs2_core/constraint/StateInputConstraintCppAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/FootCollisionSpheres.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Keeps the collision spheres of the feet and the knees apart with a zeroing
 * control barrier function (Ames et al., "Control Barrier Functions: Theory and
 * Applications", ECC 2019), one row per pair of kFootCollisionPairs:
 *
 *   h(q) = |p_a(q) - p_b(q)|^2 - (2 r)^2,   c(x, u) = dh/dt + gamma h >= 0,
 * dh/dt = 2 (p_a - p_b)^T (v_a - v_b).
 *
 * FootCollisionConstraint only bounds h, so the solver may drive a swing foot
 * at full speed onto the bound; this bounds the speed of approach as well, to
 * gamma times the remaining margin, so h decays at most exponentially towards
 * zero.
 *
 * v_a - v_b is taken with the base at rest. The base's linear velocity moves
 * both points alike, and its angular velocity w adds w x (p_a - p_b), which is
 * orthogonal to p_a - p_b: neither changes dh/dt, so the row is exact in both
 * formulations although the centroidal state holds no base twist. The joint
 * velocities are the model's getJointVelocities(), the input of the centroidal
 * MPC and the state of the whole-body MPC.
 *
 * Added once to the MPC's state-input soft constraints; the reference manager
 * and the robot model must outlive it. Like FootCollisionConstraint it is off
 * while both feet are in contact. Like every OCS2 term it is cloned for each
 * solver thread, and a single instance is not thread-safe.
 */
class FootCollisionCbfConstraint final : public StateInputConstraintCppAd {
 public:
  /** The name of its generated library, of its own so that a change of the
   * collection name never renames it. */
  static constexpr char kLibraryName[] = "FootCollisionCbfConstraint";

  /** The spheres and the barrier's rate. */
  struct Config {
    FootCollisionSpheres spheres;
    // [1/s] The rate gamma at which h may decay; positive.
    scalar_t gamma = 0.0;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * Tapes (or loads) the library kLibraryName in
   * modelSettings.modelFolderCppAd. `referenceManager` and `mpcRobotModel` are
   * retained and must outlive the constraint and its clones; the frames of
   * `config` must be frames of `pinocchioInterface`'s model
   * (checkFootCollisionFramesExist()).
   */
  FootCollisionCbfConstraint(
      const SwitchedModelReferenceManager& referenceManager,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
      const Config& config,
      const ModelSettings& modelSettings);

  ~FootCollisionCbfConstraint() override = default;
  FootCollisionCbfConstraint& operator=(const FootCollisionCbfConstraint&) =
      delete;
  FootCollisionCbfConstraint(FootCollisionCbfConstraint&&) = delete;
  FootCollisionCbfConstraint& operator=(FootCollisionCbfConstraint&&) = delete;
  FootCollisionCbfConstraint* absl_nonnull clone() const override {
    return new FootCollisionCbfConstraint(*this);
  }

  bool isActive(scalar_t time) const override;

  size_t getNumConstraints(scalar_t /*time*/) const override {
    return kNumFootCollisionPairs;
  }

  vector_t getParameters(scalar_t time,
                         const PreComputation& preComputation) const override;

  /** The barrier's rate, a parameter of the generated library: a change takes
   * effect at the next evaluation. */
  scalar_t getGamma() const { return config_.gamma; }
  void setGamma(scalar_t gamma) { config_.gamma = gamma; }

 private:
  /** The parameters of the taped function: the foot's and the knee's collision
   * sphere radius, and gamma. */
  static constexpr int kNumParameters = 3;

  FootCollisionCbfConstraint(const FootCollisionCbfConstraint& other);

  ad_vector_t constraintFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) const override;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  const MpcRobotModelBase<ad_scalar_t>* absl_nonnull mpcRobotModelPtr_;
  Config config_;
};

}  // namespace ocs2::humanoid
