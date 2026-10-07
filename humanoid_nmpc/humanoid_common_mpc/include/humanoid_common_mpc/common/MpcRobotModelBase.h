/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The state and input layout of one MPC formulation: where the base pose, the joints and the contact wrenches sit in
 * the state and input vectors, and how to read and write them, for scalar_t and for the CppAD scalar.
 *
 * Each formulation derives one. It keeps a reference to the ModelSettings it is built with, which must outlive it, and
 * is cloned into every term that needs it. The base class holds only the layout, and its const functions are safe to
 * call concurrently.
 */
template <typename SCALAR_T>
class MpcRobotModelBase {
 public:
  MpcRobotModelBase(const ModelSettings& modelSettings, scalar_t state_dim, scalar_t input_dim)
      : modelSettings(modelSettings),
        state_dim(state_dim),
        input_dim(input_dim),
        gen_coordinates_dim(base_dim + modelSettings.mpc_joint_dim) {}

  virtual ~MpcRobotModelBase() = default;
  MpcRobotModelBase& operator=(const MpcRobotModelBase&) = delete;
  MpcRobotModelBase(MpcRobotModelBase&&) = delete;
  MpcRobotModelBase& operator=(MpcRobotModelBase&&) = delete;
  virtual MpcRobotModelBase* absl_nonnull clone() const = 0;

  /******************************************************************************************************/
  /*                                           Dimensions                                               */
  /******************************************************************************************************/

  size_t getStateDim() const { return state_dim; }
  size_t getInputDim() const { return input_dim; }
  size_t getJointDim() const { return modelSettings.mpc_joint_dim; }
  size_t getGenCoordinatesDim() const { return gen_coordinates_dim; }

  /******************************************************************************************************/
  /*                                          Start indices                                             */
  /******************************************************************************************************/

  virtual size_t getBaseStartindex() const = 0;
  virtual size_t getJointStartindex() const = 0;
  virtual size_t getJointVelocitiesStartindex() const = 0;

  // Assumes contact wrench [f_x, f_y, f_z, M_x, M_y, M_z]^T
  virtual size_t getContactWrenchStartIndices(size_t contactIndex) const = 0;
  virtual size_t getContactForceStartIndices(size_t contactIndex) const { return getContactWrenchStartIndices(contactIndex); }
  virtual size_t getContactMomentStartIndices(size_t contactIndex) const { return 6 * contactIndex + 3; }

  /**
   * Number of input variables that parameterize the contact wrench of one contact, occupying the input block that
   * starts at getContactWrenchStartIndices(contactIndex). Six for a direct wrench parameterization; the basis-vector
   * parameterization (BasisInputsModelDecorator) overrides this with its number of generators per foot. Use it
   * instead of inferring the block size from the surrounding start indices, which differ between models.
   */
  virtual size_t getContactInputDim(size_t /*contactIndex*/) const { return kContactWrenchDim; }

  /******************************************************************************************************/
  /*                                     Generalized coordinates                                        */
  /******************************************************************************************************/

  virtual VECTOR_T<SCALAR_T> getGeneralizedCoordinates(const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR6_T<SCALAR_T> getBasePose(const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR3_T<SCALAR_T> getBasePosition(const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR3_T<SCALAR_T> getBaseOrientationEulerZYX(const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR3_T<SCALAR_T> getBaseComLinearVelocity(const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR6_T<SCALAR_T> getBaseComVelocity(const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR_T<SCALAR_T> getJointAngles(const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR_T<SCALAR_T> getJointVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& input) const = 0;

  virtual VECTOR_T<SCALAR_T> getGeneralizedVelocities(const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& input) = 0;

  /******************************************************************************************************/

  virtual void setGeneralizedCoordinates(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& generalizedCorrdinates) const = 0;

  virtual void setBasePose(VECTOR_T<SCALAR_T>& state, const VECTOR6_T<SCALAR_T>& basePose) const = 0;

  virtual void setBasePosition(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& position) const = 0;

  virtual void setBaseOrientationEulerZYX(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& eulerAnglesZYX) const = 0;

  /** The counterpart of getBaseComLinearVelocity(), with the same meaning in each model's state layout. */
  virtual void setBaseComLinearVelocity(VECTOR_T<SCALAR_T>& state, const VECTOR3_T<SCALAR_T>& velocity) const = 0;

  virtual void setJointAngles(VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& jointAngles) const = 0;

  virtual void setJointVelocities(VECTOR_T<SCALAR_T>& state,
                                  VECTOR_T<SCALAR_T>& input,
                                  const VECTOR_T<SCALAR_T>& jointVelocities) const = 0;

  virtual void adaptBasePoseHeight(VECTOR_T<SCALAR_T>& state, scalar_t heightChange) const = 0;

  /******************************************************************************************************/
  /*                                          Contacts                                                  */
  /******************************************************************************************************/

  /**
   * Input-only contact wrench accessors.
   *
   * These read/write the contact wrench *as it is parameterized in the input vector*. For wrench-space
   * models (CentroidalMpcRobotModel, WBAccelMpcRobotModel) the input stores the wrench in the world
   * frame, so these accessors return world-frame quantities. For the basis-vector decorator
   * (BasisInputsModelDecorator) the input stores non-negative scalings of a *local contact-frame* basis,
   * so these accessors return the wrench in the local contact frame.
   *
   * Whenever a world-frame wrench is required, prefer the state-aware
   * `...InWorldFrame` accessors below, which are frame-correct for every model.
   */
  virtual VECTOR6_T<SCALAR_T> getContactWrench(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const = 0;

  virtual VECTOR3_T<SCALAR_T> getContactForce(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const = 0;

  virtual VECTOR3_T<SCALAR_T> getContactMoment(const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const = 0;

  /******************************************************************************************************/

  virtual void setContactWrench(VECTOR_T<SCALAR_T>& input, const VECTOR6_T<SCALAR_T>& wrench, size_t contactIndex) const = 0;

  virtual void setContactForce(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& force, size_t contactIndex) const = 0;

  virtual void setContactMoment(VECTOR_T<SCALAR_T>& input, const VECTOR3_T<SCALAR_T>& moment, size_t contactIndex) const = 0;

  /******************************************************************************************************/
  /*                              State-aware world-frame contact accessors                             */
  /******************************************************************************************************/

  /**
   * Returns the contact wrench [f, tau] expressed in the world (inertial) frame.
   *
   * The state is required because some input parameterizations (basis-vector scalings) are defined in
   * the local contact frame, whose orientation depends on the robot configuration. The default
   * implementation assumes the input already stores a world-frame wrench and ignores the state.
   */
  virtual VECTOR6_T<SCALAR_T> getContactWrenchInWorldFrame(const VECTOR_T<SCALAR_T>& /*state*/,
                                                           const VECTOR_T<SCALAR_T>& input,
                                                           size_t contactIndex) const {
    return getContactWrench(input, contactIndex);
  }

  virtual VECTOR3_T<SCALAR_T> getContactForceInWorldFrame(const VECTOR_T<SCALAR_T>& /*state*/,
                                                          const VECTOR_T<SCALAR_T>& input,
                                                          size_t contactIndex) const {
    return getContactForce(input, contactIndex);
  }

  virtual VECTOR3_T<SCALAR_T> getContactMomentInWorldFrame(const VECTOR_T<SCALAR_T>& /*state*/,
                                                           const VECTOR_T<SCALAR_T>& input,
                                                           size_t contactIndex) const {
    return getContactMoment(input, contactIndex);
  }

  /**
   * Writes a world-frame contact wrench into the input vector, converting into the input
   * parameterization (and local contact frame) where necessary.
   */
  virtual void setContactWrenchInWorldFrame(const VECTOR_T<SCALAR_T>& /*state*/,
                                            VECTOR_T<SCALAR_T>& input,
                                            const VECTOR6_T<SCALAR_T>& wrenchInWorld,
                                            size_t contactIndex) const {
    setContactWrench(input, wrenchInWorld, contactIndex);
  }

  virtual void setContactForceInWorldFrame(const VECTOR_T<SCALAR_T>& /*state*/,
                                           VECTOR_T<SCALAR_T>& input,
                                           const VECTOR3_T<SCALAR_T>& forceInWorld,
                                           size_t contactIndex) const {
    setContactForce(input, forceInWorld, contactIndex);
  }

  /******************************************************************************************************/
  /*                                         Joint angle                                                */
  /******************************************************************************************************/

  /** The index of `jointName` among the MPC joints, or NotFound when it is not an active joint of the MPC model. */
  absl::StatusOr<size_t> findJointIndex(const std::string& jointName) const {
    const absl::flat_hash_map<std::string, size_t>::const_iterator found = modelSettings.jointIndexMap.find(jointName);
    if (found == modelSettings.jointIndexMap.end()) {
      return absl::NotFoundError(absl::StrCat("Joint name ", jointName, " is not contained in MPC model!"));
    }
    return found->second;
  }

 protected:
  MpcRobotModelBase(const MpcRobotModelBase& rhs)
      : modelSettings(rhs.modelSettings),
        state_dim(rhs.state_dim),
        input_dim(rhs.input_dim),
        gen_coordinates_dim(rhs.gen_coordinates_dim) {}

 public:
  const ModelSettings& modelSettings;

  const size_t state_dim;
  const size_t input_dim;
  const size_t base_dim = 6;
  const size_t gen_coordinates_dim;
};

}  // namespace ocs2::humanoid
