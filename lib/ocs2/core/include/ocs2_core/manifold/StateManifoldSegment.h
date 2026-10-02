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

namespace ocs2 {

/**
 * One factor of a ProductStateManifold: a block of consecutive state entries that is either Euclidean (ambient and
 * tangent size `dimension`) or a unit quaternion (ambient size 4, the coefficients (x, y, z, w); tangent size 3, a body
 * rotation vector).
 */
struct StateManifoldSegment {
  enum class Type { kEuclidean, kUnitQuaternion };

  /** A Euclidean block of `dimension` entries. */
  static StateManifoldSegment euclidean(size_t dimension) { return StateManifoldSegment{Type::kEuclidean, dimension}; }

  /** A unit-quaternion block (four coefficients x, y, z, w). */
  static StateManifoldSegment unitQuaternion() { return StateManifoldSegment{Type::kUnitQuaternion, 4}; }

  size_t getAmbientDim() const { return dimension; }
  size_t getTangentDim() const { return type == Type::kUnitQuaternion ? 3 : dimension; }

  Type type;
  size_t dimension;  // the ambient size
};

}  // namespace ocs2
