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

#include "ocs2_oc/multiple_shooting/ManifoldProjection.h"

namespace ocs2 {
namespace multiple_shooting {

namespace {

bool hasAmbientColumns(const StateManifold& stateManifold, const matrix_t& matrix) {
  return static_cast<size_t>(matrix.cols()) == stateManifold.getAmbientDim();
}

}  // namespace

void pullBackCost(const StateManifold& stateManifold, const vector_t& x, ScalarFunctionQuadraticApproximation& cost) {
  if (static_cast<size_t>(cost.dfdx.size()) == stateManifold.getAmbientDim()) {
    stateManifold.pullBackGradient(x, cost.dfdx);
  }
  if (hasAmbientColumns(stateManifold, cost.dfdxx)) {
    stateManifold.pullBackHessian(x, cost.dfdxx);
  }
  if (hasAmbientColumns(stateManifold, cost.dfdux)) {
    stateManifold.pullBackStateColumns(x, cost.dfdux);
  }
}

void pullBackConstraint(const StateManifold& stateManifold, const vector_t& x, VectorFunctionLinearApproximation& constraint) {
  if (hasAmbientColumns(stateManifold, constraint.dfdx)) {
    stateManifold.pullBackStateColumns(x, constraint.dfdx);
  }
}

void projectIntermediateNodeOnManifold(const StateManifold& stateManifold,
                                       const vector_t& x,
                                       const vector_t& x_next,
                                       Transcription& transcription) {
  stateManifold.pushForwardDynamics(x, x_next, transcription.dynamics);
  pullBackCost(stateManifold, x, transcription.cost);
  pullBackConstraint(stateManifold, x, transcription.stateEqConstraints);
  pullBackConstraint(stateManifold, x, transcription.stateInputEqConstraints);
  pullBackConstraint(stateManifold, x, transcription.stateIneqConstraints);
  pullBackConstraint(stateManifold, x, transcription.stateInputIneqConstraints);
}

void projectEventNodeOnManifold(const StateManifold& stateManifold,
                                const vector_t& x,
                                const vector_t& x_next,
                                EventTranscription& transcription) {
  stateManifold.pushForwardJump(x, x_next, transcription.dynamics);
  pullBackCost(stateManifold, x, transcription.cost);
  pullBackConstraint(stateManifold, x, transcription.eqConstraints);
  pullBackConstraint(stateManifold, x, transcription.ineqConstraints);
}

void projectTerminalNodeOnManifold(const StateManifold& stateManifold, const vector_t& x, TerminalTranscription& transcription) {
  pullBackCost(stateManifold, x, transcription.cost);
  pullBackConstraint(stateManifold, x, transcription.eqConstraints);
  pullBackConstraint(stateManifold, x, transcription.ineqConstraints);
}

}  // namespace multiple_shooting
}  // namespace ocs2
