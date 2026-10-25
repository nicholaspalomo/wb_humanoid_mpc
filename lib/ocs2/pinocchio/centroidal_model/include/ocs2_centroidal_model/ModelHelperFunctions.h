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

#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "ocs2_core/Dimensions.h"

#include "ocs2_centroidal_model/CentroidalModelInfo.h"

namespace ocs2 {

/**
 * Get the inverse of the sub-block of the centroidal momentum matrix which corresponds to the floating base variables.
 *  Ab_inv = [  1/m I_{3,3},    -1/m*Ab_12*Ab_22^(-1),
 *              O_{3,3},         Ab_22^(-1)          ]
 *
 * @param [in] A(q): centroidal momentum matrix
 * @return Ab_inv(q): inverse of the 6x6 left-block of A(q)
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, centroidal_model::kBaseVelocityDim, centroidal_model::kCentroidalMomentumDim>
computeFloatingBaseCentroidalMomentumMatrixInverse(
    const Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, centroidal_model::kBaseVelocityDim>& Ab);

/**
 * Returns the base block A_b (6 x 6) of the SingleRigidBodyDynamics model's centroidal momentum matrix at q, whose
 * joint block is zero: h = A_b v_b with the base velocity v_b = [dp_W, w_B] of the Composite(Translation, Spherical)
 * root, the CoM c = p_W - R r_nom moving at dc = dp_W + [R r_nom]x R w_B and the nominal inertia about it turning with
 * the base, L = R I_nom R' R w_B. Fixed-size: it allocates nothing, so a realtime thread can call it.
 * @param [in] info: centroidal model information (robotMass, comToBasePositionNominal, centroidalInertiaNominal)
 * @param [in] q: pinocchio joint positions (generalized coordinates) [p_W, xi, q_j] with a unit quaternion xi
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, centroidal_model::kBaseVelocityDim> getSingleRigidBodyBaseMomentumMatrix(
    const CentroidalModelInfoTpl<SCALAR_T>& info,
    const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& q);

/**
 * Updates the centroidal momentum matrix in data.Ag (6 x nv) and the CoM position in data.com[0] for the
 * FullCentroidalDynamics model and the SingleRigidBodyDynamics Model
 * @param [in] interface: pinocchio robot interface containing model + data
 * @param [in] info: centroidal model information
 * @param [in] q: pinocchio joint positions (generalized coordinates) [p_W, xi, q_j] with a unit quaternion xi
 *
 * @remark: This function also internally calls:
 *       pinocchio::forwardKinematics(model, data, q)
 *       pinocchio::computeJointJacobians(model, data, q) (only for the FullCentroidalDynamics case)
 *       pinocchio::updateFramePlacements(model, data)
 */
template <typename SCALAR_T>
void updateCentroidalDynamics(PinocchioInterfaceTpl<SCALAR_T>& interface, const CentroidalModelInfoTpl<SCALAR_T>& info,
                              const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& q);

/**
 * The time derivative of the generalized coordinates [p_W, xi, q_j] of `state` for the pinocchio velocity
 * `vPinocchio` = [dp_W, w_B, dq_j] (nv = info.velocityDim entries): [dp_W, 1/2 G(xi) w_B, dq_j], nq = info.configurationDim
 * entries, with G the quaternion rate matrix (ocs2_core/manifold/UnitQuaternionMath.h). The quaternion row reads the raw
 * xi of the state, not its normalization, so that the flow map's derivative keeps |xi| constant to first order
 * (d|xi|^2/dt = xi' G(xi) w_B = 0). Polynomial, safe on a CppAD tape.
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1> computeConfigurationDerivative(const CentroidalModelInfoTpl<SCALAR_T>& info,
                                                                          const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& state,
                                                                          const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& vPinocchio);

/**
 * Returns the centroidal momentum matrix from the pinocchioInterface data
 * @param [in] interface: pinocchio robot interface containing model + data
 * @return centroidal momentum matrix from data.Ag
 *
 * @note requires pinocchioInterface to be updated with:
 *       ocs2::updateCentroidalDynamics(interface, info, q)
 */
template <typename SCALAR_T>
const Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, Eigen::Dynamic>& getCentroidalMomentumMatrix(
    const PinocchioInterfaceTpl<SCALAR_T>& interface);

/**
 * Computes the CoM to contact point position in world frame
 *
 * @param [in] interface: pinocchio robot interface containing model + data
 * @param [in] info: centroidal model information
 * @param [in] contactIndex: index of the contact point
 * @return: position of the contact point w.r.t CoM expressed in world frame
 *
 * @note requires pinocchioInterface to be updated with:
 *       ocs2::updateCentroidalDynamics(interface, info, q)
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, kPositionDim, 1> getPositionComToContactPointInWorldFrame(const PinocchioInterfaceTpl<SCALAR_T>& interface,
                                                                                  const CentroidalModelInfoTpl<SCALAR_T>& info,
                                                                                  size_t contactIndex);

/**
 * Computes the derivative of the normalized centroidal momentum (linear + angular) expressed in the centroidal frame
 *
 * @param [in] interface: pinocchio robot interface containing model + data
 * @param [in] info: centroidal model information
 * @param [in] input: system input vector
 * @return: time derivative of normalized centroidal momentum
 *
 * @note requires pinocchioInterface to be updated with:
 *       ocs2::updateCentroidalDynamics(interface, info, q)
 */
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, centroidal_model::kCentroidalMomentumDim, 1> getNormalizedCentroidalMomentumRate(
    const PinocchioInterfaceTpl<SCALAR_T>& interface,
    const CentroidalModelInfoTpl<SCALAR_T>& info,
    const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& input);

}  // namespace ocs2

#include "implementation/ModelHelperFunctionsImpl.h"
