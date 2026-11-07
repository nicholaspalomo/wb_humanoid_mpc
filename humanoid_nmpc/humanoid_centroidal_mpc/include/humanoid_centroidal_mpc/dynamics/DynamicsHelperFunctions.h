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

#include "pinocchio/fwd.hpp"

#include <cstddef>

#include "cppad/cg.hpp"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

/** Computes an input with zero joint velocity and forces which equally
 * distribute the robot weight between contact feet. */

inline vector_t weightCompensatingInput(
    const CentroidalModelInfoTpl<scalar_t>& info,
    const BipedFeetArray<bool>& contactFlags,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  // The mass of THIS model, read per call: a function-static cache kept the
  // first model's weight for every model in the process.
  const scalar_t robotWeight = info.robotMass * kGravityMagnitude;
  const size_t numStanceLegs = numberOfLegsInContacts(contactFlags);
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceLegs > 0) {
    // Vertical in the parameterization frame, which is the world only for the
    // wrench-space models; the state-aware overload below writes a world force.
    const vector3_t forceContactEwrtParam(0.0, 0.0,
                                          robotWeight / numStanceLegs);
    for (size_t i = 0; i < contactFlags.size(); ++i) {
      if (contactFlags[i]) {
        mpcRobotModel.setForceContactEwrtParam(input, forceContactEwrtParam, i);
      }
    }  // end of i loop
  }
  return input;
}

/** State-aware variant: the vertical world-frame force is written through the
 * model's world-frame setter so that it is expressed correctly in the input
 * parameterization (e.g. rotated into the local contact frame for basis-vector
 * inputs). */

inline vector_t weightCompensatingInput(
    const CentroidalModelInfoTpl<scalar_t>& info,
    const BipedFeetArray<bool>& contactFlags,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& state) {
  // The mass of THIS model, read per call: a function-static cache kept the
  // first model's weight for every model in the process.
  const scalar_t robotWeight = info.robotMass * kGravityMagnitude;
  const size_t numStanceLegs = numberOfLegsInContacts(contactFlags);
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceLegs > 0) {
    const vector3_t forceContactEwrtWorld(0.0, 0.0,
                                          robotWeight / numStanceLegs);
    for (size_t i = 0; i < contactFlags.size(); ++i) {
      if (contactFlags[i]) {
        mpcRobotModel.setForceContactEwrtWorld(state, input,
                                               forceContactEwrtWorld, i);
      }
    }  // end of i loop
  }
  return input;
}

/** The state-aware weightCompensatingInput() of the contacts `contactFlags`
 * says are active, for a contact set of any size: the weight of `info`'s robot
 * shared evenly by them as vertical world-frame forces through the world-frame
 * setter at `state`. Zero when no contact is active. Contact i of the mask is
 * contact i of `mpcRobotModel`'s set. */
inline vector_t weightCompensatingInput(
    const CentroidalModelInfoTpl<scalar_t>& info,
    ContactMask contactFlags,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& state) {
  const scalar_t robotWeight = info.robotMass * kGravityMagnitude;
  const size_t numStanceContacts = contactFlags.numActive();
  vector_t input = vector_t::Zero(mpcRobotModel.getInputDim());
  if (numStanceContacts == 0) return input;
  // The same arithmetic as the two-foot overload above, so that a biped's
  // initial guess is bit-identical whichever of the two it is built with.
  const vector3_t forceContactEwrtWorld(
      0.0, 0.0, robotWeight / static_cast<scalar_t>(numStanceContacts));
  for (size_t i = 0; i < contactFlags.size(); ++i) {
    if (contactFlags[i]) {
      mpcRobotModel.setForceContactEwrtWorld(state, input,
                                             forceContactEwrtWorld, i);
    }
  }
  return input;
}

}  // namespace ocs2::humanoid
