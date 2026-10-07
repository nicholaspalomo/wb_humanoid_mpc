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

#include <ocs2_core/Types.h>
#include <ocs2_core/manifold/StateManifold.h>

#include "ocs2_oc/multiple_shooting/Transcription.h"

namespace ocs2 {
namespace multiple_shooting {

/*
 * The pull-backs that turn the ambient transcription of a node into its transcription on the state manifold
 * (humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.7). Costs, constraints and dynamics return
 * derivatives with respect to the stored (ambient) state x; the QP is posed in the tangent dx with x (+) dx. Every map
 * works in place on the non-Euclidean rows and columns only. The retraction-curvature term of the cost Hessian,
 * -1/4 (grad_xi l . xi) I on a quaternion block, is not added: it vanishes for terms that see the quaternion only through
 * xi / |xi|, which is the contract of StateManifold.
 */

/** cost <- its tangent approximation: dfdx <- E' dfdx, dfdxx <- E' dfdxx E, dfdux <- dfdux E. */
void pullBackCost(const StateManifold& stateManifold, const vector_t& x, ScalarFunctionQuadraticApproximation& cost);

/** constraint.dfdx <- constraint.dfdx E. A Jacobian without ambient columns (an empty, unset term) is left as it is. */
void pullBackConstraint(const StateManifold& stateManifold, const vector_t& x, VectorFunctionLinearApproximation& constraint);

/**
 * An intermediate node. On input transcription.dynamics holds the discrete flow Phi(x, u) and its ambient sensitivities
 * (before the `- x_next` of the flat transcription); on output the gap difference(x_next, Phi) and the exact Newton
 * linearization of x_next (+) dx_next = Pi(Phi(x (+) dx, u + du)) (StateManifold::pushForwardDynamics). The cost and all
 * constraint Jacobians are pulled back.
 */
void projectIntermediateNodeOnManifold(const StateManifold& stateManifold,
                                       const vector_t& x,
                                       const vector_t& x_next,
                                       Transcription& transcription);

/**
 * An event node. On input transcription.dynamics holds the jump map Phi(x) and its ambient Jacobian; on output the gap
 * difference(x_next, Phi) and its tangent Jacobian, with an empty (tangent x 0) input Jacobian.
 */
void projectEventNodeOnManifold(const StateManifold& stateManifold,
                                const vector_t& x,
                                const vector_t& x_next,
                                EventTranscription& transcription);

/** The terminal node: the cost and the constraint Jacobians are pulled back. */
void projectTerminalNodeOnManifold(const StateManifold& stateManifold, const vector_t& x, TerminalTranscription& transcription);

}  // namespace multiple_shooting
}  // namespace ocs2
