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

#include <string>

#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

template <typename SCALAR_T>
using VECTOR18_T = Eigen::Matrix<SCALAR_T, 18, 1>;

/** The weights of EndEffectorDynamicsFootCost, one per axis of each foot error, as the task file names them. */
struct EndEffectorDynamicsWeights {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // Cost weights [w_x,w_y,w_z]
  vector3_t contactPositionErrorWeight{30.0, 30.0, 30.0};
  vector3_t contactOrientationErrorWeight{30.0, 30.0, 30.0};
  vector3_t contactLinearVelocityErrorWeight{0.1, 0.1, 0.1};
  vector3_t contactAngularVelocityErrorWeight{0.1, 0.1, 0.1};
  vector3_t contactLinearAccelerationErrorWeight{0.01, 0.01, 0.01};
  vector3_t contactAngularAccelerationErrorWeight{0.01, 0.01, 0.01};

  VECTOR18_T<scalar_t> toVector() const;
};

/**
 * The reference of EndEffectorDynamicsFootCost packed in its 18 parameters: the foot position, the normal of the ground
 * plane, and the linear and angular velocities and accelerations, three entries each.
 */
template <typename SCALAR_T>
struct PlanarEndEffectorDynamicsReference {
  explicit PlanarEndEffectorDynamicsReference(const VECTOR18_T<SCALAR_T>& vector) : costElementVector(vector) {}
  PlanarEndEffectorDynamicsReference() : costElementVector(VECTOR18_T<SCALAR_T>::Zero()) {}
  VECTOR18_T<SCALAR_T> costElementVector;

  VECTOR18_T<SCALAR_T> getValues() { return costElementVector; }

  VECTOR3_T<SCALAR_T> getPosition() const { return costElementVector.head(3); }
  VECTOR3_T<SCALAR_T> getPlaneNormal() const { return costElementVector.segment(3, 3); }
  VECTOR3_T<SCALAR_T> getLinearVelocity() const { return costElementVector.segment(6, 3); }
  VECTOR3_T<SCALAR_T> getAngularVelocity() const { return costElementVector.segment(9, 3); }
  VECTOR3_T<SCALAR_T> getLinearAcceleration() const { return costElementVector.segment(12, 3); }
  VECTOR3_T<SCALAR_T> getAngularAcceleration() const { return costElementVector.segment(15, 3); }

  void setPosition(const VECTOR3_T<SCALAR_T>& position) { costElementVector.head(3) = position; }
  void setPlaneNormal(const VECTOR3_T<SCALAR_T>& planeNormal) { costElementVector.segment(3, 3) = planeNormal; }
  void setLinearVelocity(const VECTOR3_T<SCALAR_T>& linearVelocity) { costElementVector.segment(6, 3) = linearVelocity; }
  void setAngularVelocity(const VECTOR3_T<SCALAR_T>& angularVelocity) { costElementVector.segment(9, 3) = angularVelocity; }
};

}  // namespace ocs2::humanoid
