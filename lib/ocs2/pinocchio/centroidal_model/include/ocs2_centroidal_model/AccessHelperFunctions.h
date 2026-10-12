/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <Eigen/Core>

#include "ocs2_centroidal_model/CentroidalModelInfo.h"

namespace ocs2 {
namespace centroidal_model {

/**
 * Provides read/write access to the contact forces.
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, 3, 1> getContactForces(Eigen::MatrixBase<Derived>& input, size_t contactIndex,
                                             const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the contact forces.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, 3, 1> getContactForces(const Eigen::MatrixBase<Derived>& input, size_t contactIndex,
                                                         const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the contact torques.
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, 3, 1> getContactTorques(Eigen::MatrixBase<Derived>& input, size_t contactIndex,
                                              const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the contact torques.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, 3, 1> getContactTorques(const Eigen::MatrixBase<Derived>& input, size_t contactIndex,
                                                          const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the joint velocities.
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, -1, 1> getJointVelocities(Eigen::MatrixBase<Derived>& input, const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the joint velocities.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, -1, 1> getJointVelocities(const Eigen::MatrixBase<Derived>& input,
                                                            const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the mass-normalized momentum.
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, 6, 1> getNormalizedMomentum(Eigen::MatrixBase<Derived>& state, const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the mass-normalized momentum.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, 6, 1> getNormalizedMomentum(const Eigen::MatrixBase<Derived>& state,
                                                              const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the base position p_W (3 entries at 6), in the world frame.
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, 3, 1> getBasePosition(Eigen::MatrixBase<Derived>& state, const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the base position p_W.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, 3, 1> getBasePosition(const Eigen::MatrixBase<Derived>& state, const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the base orientation xi (4 entries at 9): the coefficients (x, y, z, w) of the quaternion
 * from the base frame to the world, as the state stores them, not normalized.
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, 4, 1> getBaseQuaternion(Eigen::MatrixBase<Derived>& state, const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the base orientation xi, as the state stores it.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, 4, 1> getBaseQuaternion(const Eigen::MatrixBase<Derived>& state,
                                                          const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the joint angles.
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, -1, 1> getJointAngles(Eigen::MatrixBase<Derived>& state, const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the joint angles.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, -1, 1> getJointAngles(const Eigen::MatrixBase<Derived>& state,
                                                        const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the generalized coordinates [p_W, xi, q_j] (nq = info.configurationDim entries at 6), as
 * the state stores them: the quaternion is not normalized (CentroidalModelPinocchioMapping::getPinocchioJointPosition()
 * normalizes it).
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, -1, 1> getGeneralizedCoordinates(Eigen::MatrixBase<Derived>& state, const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read access to the generalized coordinates, as the state stores them.
 */
template <typename Derived, typename SCALAR>
const Eigen::Block<const Derived, -1, 1> getGeneralizedCoordinates(const Eigen::MatrixBase<Derived>& state,
                                                                   const CentroidalModelInfoTpl<SCALAR>& info);

/**
 * Provides read/write access to the rows of a state derivative that hold the time derivative of the generalized
 * coordinates: d/dt [p_W, xi, q_j] (nq entries at 6), which computeConfigurationDerivative() computes from the pinocchio
 * velocity v (nv entries).
 */
template <typename Derived, typename SCALAR>
Eigen::Block<Derived, -1, 1> configurationDerivative(Eigen::MatrixBase<Derived>& stateDerivative, const CentroidalModelInfoTpl<SCALAR>& info);

}  // namespace centroidal_model
}  // namespace ocs2

#include "implementation/AccessHelperFunctionsImpl.h"
