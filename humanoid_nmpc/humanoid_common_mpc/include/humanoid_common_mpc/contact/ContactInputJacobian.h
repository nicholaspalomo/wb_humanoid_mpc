/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * How one contact's wrench depends on the input vector, read off the model rather than assumed.
 *
 * Three terms need this and none of them may hard-code the layout, because the layout is exactly what the input
 * parameterization changes. In the wrench-space CentroidalMpcRobotModel the contact force IS a three-element slice of
 * the input, so `d(force)/d(input)` is an identity block and a term can get away with writing a 3x3 into the right
 * place. Under BasisInputsModelDecorator the same contact occupies a wider block of non-negative scalings of a
 * local-frame wrench-cone basis, and the force is `B_local * lambda`: the identity block is then wrong in every
 * entry, and a term that writes one hands the solver a Jacobian for a function it is not evaluating.
 *
 * The contact's input block is `[getContactWrenchStartIndices(i), + getContactInputDim(i))` - that pairing is the
 * contract MpcRobotModelBase declares, and it is what these functions probe.
 */

/**
 * The 3 x getInputDim() Jacobian of `contactPointIndex`'s contact force with respect to the input.
 *
 * Probed from the model's own linear accessor with unit inputs, so it is correct for every parameterization the model
 * offers and costs nothing at evaluation time - callers store the result at construction. CHECKs that the force really
 * is linear in the input, which is the assumption the probe rests on.
 *
 * WHICH FRAME the force is in is the model's business, not this function's: `getContactForce()` returns the world-frame
 * force for the wrench-space models and the local contact-frame force under BasisInputsModelDecorator. A caller that
 * needs one particular frame has to rotate, and it must rotate the value and this Jacobian by the same rotation.
 */
matrix_t contactForceInputJacobian(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex);

/**
 * The row with `f_n = normalContactForceRow . input`: the third row of contactForceInputJacobian().
 *
 * The contact-implicit terms use `f_n` as a LOAD INDICATOR - a number that is zero exactly when the foot carries no
 * contact wrench, and grows with the load - rather than as a physical force, which is why both frames above serve.
 * This function additionally CHECKs the two properties that make the indicator sound, because they are what a new
 * input parameterization could quietly break:
 *
 *  - the row is non-negative, so no input can reduce the indicator while adding load. Together with the sign
 *    constraint each parameterization already carries (the friction cone for the wrench models; the non-negativity of
 *    the scalings under BasisInputsModelDecorator, every one of whose generators is built with a local normal force of
 *    exactly 1) this makes the indicator vanish if and only if the whole wrench does;
 *  - the row is not identically zero, i.e. some input of this contact does produce a normal force.
 */
vector_t normalContactForceRow(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex);

}  // namespace ocs2::humanoid
