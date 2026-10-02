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

#include "ocs2_core/manifold/EuclideanStateManifold.h"

#include "absl/log/check.h"

namespace ocs2 {

void EuclideanStateManifold::retract(const vector_t& x, const vector_t& dx, scalar_t alpha, vector_t& xNew) const {
  CHECK_EQ(static_cast<size_t>(x.size()), dimension_);
  CHECK_EQ(static_cast<size_t>(dx.size()), dimension_);
  xNew = x + alpha * dx;
}

vector_t EuclideanStateManifold::difference(const vector_t& x0, const vector_t& x1) const {
  CHECK_EQ(static_cast<size_t>(x0.size()), dimension_);
  CHECK_EQ(static_cast<size_t>(x1.size()), dimension_);
  return x1 - x0;
}

vector_t EuclideanStateManifold::interpolate(const vector_t& x0, const vector_t& x1, scalar_t alpha) const {
  CHECK_EQ(static_cast<size_t>(x0.size()), dimension_);
  CHECK_EQ(static_cast<size_t>(x1.size()), dimension_);
  return (1.0 - alpha) * x0 + alpha * x1;
}

void EuclideanStateManifold::project(vector_t& x) const {
  CHECK_EQ(static_cast<size_t>(x.size()), dimension_);
}

scalar_t EuclideanStateManifold::getMaximumRotationAngle(const vector_t& dx) const {
  CHECK_EQ(static_cast<size_t>(dx.size()), dimension_);
  return 0.0;
}

void EuclideanStateManifold::pullBackStateColumns(const vector_t& x, matrix_t& J) const {
  CHECK_EQ(static_cast<size_t>(J.cols()), dimension_);
}

void EuclideanStateManifold::pullBackStateRows(const vector_t& x, matrix_t& M) const {
  CHECK_EQ(static_cast<size_t>(M.rows()), dimension_);
}

void EuclideanStateManifold::pullBackGradient(const vector_t& x, vector_t& g) const {
  CHECK_EQ(static_cast<size_t>(g.size()), dimension_);
}

void EuclideanStateManifold::liftStateColumns(const vector_t& x, matrix_t& J) const {
  CHECK_EQ(static_cast<size_t>(J.cols()), dimension_);
}

void EuclideanStateManifold::pushForwardDynamics(const vector_t& x,
                                                 const vector_t& xNext,
                                                 VectorFunctionLinearApproximation& dynamics) const {
  CHECK_EQ(static_cast<size_t>(xNext.size()), dimension_);
  CHECK_EQ(static_cast<size_t>(dynamics.f.size()), dimension_);
  CHECK_EQ(static_cast<size_t>(dynamics.dfdx.rows()), dimension_);
  CHECK_EQ(static_cast<size_t>(dynamics.dfdx.cols()), dimension_);
  dynamics.f -= xNext;
}

}  // namespace ocs2
