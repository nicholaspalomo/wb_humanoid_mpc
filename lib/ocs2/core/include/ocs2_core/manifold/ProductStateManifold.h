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

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/manifold/StateManifold.h>
#include <ocs2_core/manifold/StateManifoldSegment.h>

namespace ocs2 {

/**
 * A product of Euclidean and unit-quaternion factors, laid out in the state vector in the order given, e.g. the
 * centroidal humanoid state [E(9), Q, E(nj)] or the whole-body one [E(3), Q, E(2 nj + 6)].
 *
 * On a unit-quaternion block (README section 2.4):
 *   retract(xi, d, alpha)  = normalize(xi (x) Exp(alpha d)),
 *   difference(xi0, xi1)   = Log(xi0^-1 (x) xi1), shortest path,
 *   interpolate            = shortest-path slerp,
 *   E(xi) = 0.5 G(xi) and E+(xi) = 2 G(xi)' / |xi|^2.
 * Euclidean blocks use addition, subtraction, linear interpolation and the identity. The derivative maps act on the
 * quaternion rows / columns of a matrix in place (O(rows * 12) per quaternion block) and only shift the Euclidean ones.
 */
class ProductStateManifold final : public StateManifold {
 public:
  /**
   * The manifold of `segments`, in state order. Adjacent Euclidean segments are merged. Fails on an empty list, a
   * Euclidean segment of size zero or a unit-quaternion segment whose size is not 4.
   */
  static absl::StatusOr<std::shared_ptr<const ProductStateManifold>> create(std::vector<StateManifoldSegment> segments);

  ~ProductStateManifold() override = default;

  size_t getAmbientDim() const override { return ambientDim_; }
  size_t getTangentDim() const override { return tangentDim_; }

  /** The segments, after adjacent Euclidean segments were merged. */
  const std::vector<StateManifoldSegment>& getSegments() const { return segments_; }

  /** The first ambient index of every unit-quaternion block, in state order. */
  std::vector<size_t> getQuaternionAmbientOffsets() const;

  /** The first tangent index of every unit-quaternion block, in state order. */
  std::vector<size_t> getQuaternionTangentOffsets() const;

  void retract(const vector_t& x, const vector_t& dx, scalar_t alpha, vector_t& xNew) const override;
  vector_t difference(const vector_t& x0, const vector_t& x1) const override;
  vector_t interpolate(const vector_t& x0, const vector_t& x1, scalar_t alpha) const override;
  void project(vector_t& x) const override;
  scalar_t getMaximumRotationAngle(const vector_t& dx) const override;
  void pullBackStateColumns(const vector_t& x, matrix_t& J) const override;
  void pullBackStateRows(const vector_t& x, matrix_t& M) const override;
  void pullBackGradient(const vector_t& x, vector_t& g) const override;
  void liftStateColumns(const vector_t& x, matrix_t& J) const override;
  void pushForwardDynamics(const vector_t& x, const vector_t& xNext, VectorFunctionLinearApproximation& dynamics) const override;

 private:
  /** A segment with its offsets in the ambient and in the tangent vector. */
  struct Block {
    StateManifoldSegment::Type type;
    size_t ambientOffset;
    size_t tangentOffset;
    size_t ambientDim;
    size_t tangentDim;
  };

  explicit ProductStateManifold(std::vector<StateManifoldSegment> segments);

  /** M <- M with the columns of each quaternion block multiplied by columnMaps[k] (4 x 3), the others shifted. */
  void mapQuaternionColumns(const std::vector<Eigen::Matrix<scalar_t, 4, 3>>& columnMaps, matrix_t& M) const;

  /** M <- M with the rows of each quaternion block replaced by rowMaps[k] (3 x 4) times them, the others shifted. */
  template <typename Derived>
  void mapQuaternionRows(const std::vector<Eigen::Matrix<scalar_t, 3, 4>>& rowMaps, Eigen::PlainObjectBase<Derived>& M) const;

  /** E(xi) of every quaternion block of x. */
  std::vector<Eigen::Matrix<scalar_t, 4, 3>> tangentMaps(const vector_t& x) const;

  std::vector<StateManifoldSegment> segments_;
  std::vector<Block> blocks_;
  size_t ambientDim_ = 0;
  size_t tangentDim_ = 0;
};

}  // namespace ocs2
