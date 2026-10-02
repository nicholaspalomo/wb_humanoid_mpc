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

#include "ocs2_core/manifold/ProductStateManifold.h"

#include <algorithm>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "ocs2_core/manifold/UnitQuaternionMath.h"

namespace ocs2 {

namespace {

using quaternion_t = quaternion_coeffs_t<scalar_t>;
using rotation_t = rotation_vector_t<scalar_t>;

size_t sizeOf(Eigen::Index n) {
  return static_cast<size_t>(n);
}

}  // namespace

absl::StatusOr<std::shared_ptr<const ProductStateManifold>> ProductStateManifold::create(std::vector<StateManifoldSegment> segments) {
  if (segments.empty()) {
    return absl::InvalidArgumentError("[ProductStateManifold] A product manifold needs at least one segment.");
  }
  std::vector<StateManifoldSegment> merged;
  merged.reserve(segments.size());
  for (size_t i = 0; i < segments.size(); ++i) {
    const StateManifoldSegment& segment = segments[i];
    if (segment.type == StateManifoldSegment::Type::kEuclidean && segment.dimension == 0) {
      return absl::InvalidArgumentError(absl::StrCat("[ProductStateManifold] Segment ", i, " is a Euclidean segment of size 0."));
    }
    if (segment.type == StateManifoldSegment::Type::kUnitQuaternion && segment.dimension != 4) {
      return absl::InvalidArgumentError(absl::StrCat("[ProductStateManifold] Segment ", i, " is a unit quaternion of size ",
                                                     segment.dimension, "; a unit quaternion has the 4 coefficients (x, y, z, w)."));
    }
    if (segment.type == StateManifoldSegment::Type::kEuclidean && !merged.empty() &&
        merged.back().type == StateManifoldSegment::Type::kEuclidean) {
      merged.back().dimension += segment.dimension;
    } else {
      merged.push_back(segment);
    }
  }
  return std::shared_ptr<const ProductStateManifold>(new ProductStateManifold(std::move(merged)));
}

ProductStateManifold::ProductStateManifold(std::vector<StateManifoldSegment> segments) : segments_(std::move(segments)) {
  blocks_.reserve(segments_.size());
  for (const StateManifoldSegment& segment : segments_) {
    blocks_.push_back(Block{segment.type, ambientDim_, tangentDim_, segment.getAmbientDim(), segment.getTangentDim()});
    ambientDim_ += segment.getAmbientDim();
    tangentDim_ += segment.getTangentDim();
  }
}

std::vector<size_t> ProductStateManifold::getQuaternionAmbientOffsets() const {
  std::vector<size_t> offsets;
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kUnitQuaternion) {
      offsets.push_back(block.ambientOffset);
    }
  }
  return offsets;
}

std::vector<size_t> ProductStateManifold::getQuaternionTangentOffsets() const {
  std::vector<size_t> offsets;
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kUnitQuaternion) {
      offsets.push_back(block.tangentOffset);
    }
  }
  return offsets;
}

void ProductStateManifold::retract(const vector_t& x, const vector_t& dx, scalar_t alpha, vector_t& xNew) const {
  CHECK_EQ(sizeOf(x.size()), ambientDim_);
  CHECK_EQ(sizeOf(dx.size()), tangentDim_);
  xNew.resize(ambientDim_);
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kEuclidean) {
      xNew.segment(block.ambientOffset, block.ambientDim) =
          x.segment(block.ambientOffset, block.ambientDim) + alpha * dx.segment(block.tangentOffset, block.tangentDim);
    } else {
      const quaternion_t xi = x.segment<4>(block.ambientOffset);
      const rotation_t step = alpha * dx.segment<3>(block.tangentOffset);
      xNew.segment<4>(block.ambientOffset) = quaternionSafeNormalize(quaternionProduct(xi, quaternionExp(step)));
    }
  }
}

vector_t ProductStateManifold::difference(const vector_t& x0, const vector_t& x1) const {
  CHECK_EQ(sizeOf(x0.size()), ambientDim_);
  CHECK_EQ(sizeOf(x1.size()), ambientDim_);
  vector_t d(tangentDim_);
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kEuclidean) {
      d.segment(block.tangentOffset, block.tangentDim) =
          x1.segment(block.ambientOffset, block.ambientDim) - x0.segment(block.ambientOffset, block.ambientDim);
    } else {
      const quaternion_t xi0 = x0.segment<4>(block.ambientOffset);
      const quaternion_t xi1 = x1.segment<4>(block.ambientOffset);
      d.segment<3>(block.tangentOffset) = quaternionLog(quaternionProduct(quaternionConjugate(xi0), xi1));
    }
  }
  return d;
}

vector_t ProductStateManifold::interpolate(const vector_t& x0, const vector_t& x1, scalar_t alpha) const {
  CHECK_EQ(sizeOf(x0.size()), ambientDim_);
  CHECK_EQ(sizeOf(x1.size()), ambientDim_);
  vector_t x(ambientDim_);
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kEuclidean) {
      x.segment(block.ambientOffset, block.ambientDim) =
          (1.0 - alpha) * x0.segment(block.ambientOffset, block.ambientDim) + alpha * x1.segment(block.ambientOffset, block.ambientDim);
    } else {
      const quaternion_t xi0 = x0.segment<4>(block.ambientOffset);
      const quaternion_t xi1 = x1.segment<4>(block.ambientOffset);
      x.segment<4>(block.ambientOffset) = quaternionSlerp(xi0, xi1, alpha);
    }
  }
  return x;
}

void ProductStateManifold::project(vector_t& x) const {
  CHECK_EQ(sizeOf(x.size()), ambientDim_);
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kUnitQuaternion) {
      const quaternion_t xi = x.segment<4>(block.ambientOffset);
      x.segment<4>(block.ambientOffset) = quaternionSafeNormalize(xi);
    }
  }
}

scalar_t ProductStateManifold::getMaximumRotationAngle(const vector_t& dx) const {
  CHECK_EQ(sizeOf(dx.size()), tangentDim_);
  scalar_t maximumAngle = 0.0;
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kUnitQuaternion) {
      maximumAngle = std::max(maximumAngle, dx.segment<3>(block.tangentOffset).norm());
    }
  }
  return maximumAngle;
}

std::vector<Eigen::Matrix<scalar_t, 4, 3>> ProductStateManifold::tangentMaps(const vector_t& x) const {
  CHECK_EQ(sizeOf(x.size()), ambientDim_);
  std::vector<Eigen::Matrix<scalar_t, 4, 3>> maps;
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kUnitQuaternion) {
      const quaternion_t xi = x.segment<4>(block.ambientOffset);
      maps.push_back(quaternionTangentMap(xi));
    }
  }
  return maps;
}

void ProductStateManifold::mapQuaternionColumns(const std::vector<Eigen::Matrix<scalar_t, 4, 3>>& columnMaps, matrix_t& M) const {
  CHECK_EQ(sizeOf(M.cols()), ambientDim_);
  // Left to right: a block's tangent columns start at or before its ambient ones, so a block is written only over
  // columns already read.
  size_t quaternionIndex = 0;
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kEuclidean) {
      if (block.tangentOffset != block.ambientOffset) {
        for (size_t j = 0; j < block.ambientDim; ++j) {
          M.col(block.tangentOffset + j) = M.col(block.ambientOffset + j);
        }
      }
    } else {
      const matrix_t mapped = M.middleCols<4>(block.ambientOffset) * columnMaps[quaternionIndex++];
      M.middleCols<3>(block.tangentOffset) = mapped;
    }
  }
  M.conservativeResize(Eigen::NoChange, tangentDim_);
}

template <typename Derived>
void ProductStateManifold::mapQuaternionRows(const std::vector<Eigen::Matrix<scalar_t, 3, 4>>& rowMaps,
                                             Eigen::PlainObjectBase<Derived>& M) const {
  CHECK_EQ(sizeOf(M.rows()), ambientDim_);
  size_t quaternionIndex = 0;
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kEuclidean) {
      if (block.tangentOffset != block.ambientOffset) {
        for (size_t i = 0; i < block.ambientDim; ++i) {
          M.row(block.tangentOffset + i) = M.row(block.ambientOffset + i);
        }
      }
    } else {
      const matrix_t mapped = rowMaps[quaternionIndex++] * M.template middleRows<4>(block.ambientOffset);
      M.template middleRows<3>(block.tangentOffset) = mapped;
    }
  }
  M.conservativeResize(tangentDim_, M.cols());
}

void ProductStateManifold::pullBackStateColumns(const vector_t& x, matrix_t& J) const {
  mapQuaternionColumns(tangentMaps(x), J);
}

void ProductStateManifold::pullBackStateRows(const vector_t& x, matrix_t& M) const {
  std::vector<Eigen::Matrix<scalar_t, 3, 4>> rowMaps;
  for (const Eigen::Matrix<scalar_t, 4, 3>& E : tangentMaps(x)) {
    rowMaps.push_back(E.transpose());
  }
  mapQuaternionRows(rowMaps, M);
}

void ProductStateManifold::pullBackGradient(const vector_t& x, vector_t& g) const {
  std::vector<Eigen::Matrix<scalar_t, 3, 4>> rowMaps;
  for (const Eigen::Matrix<scalar_t, 4, 3>& E : tangentMaps(x)) {
    rowMaps.push_back(E.transpose());
  }
  mapQuaternionRows(rowMaps, g);
}

void ProductStateManifold::liftStateColumns(const vector_t& x, matrix_t& J) const {
  CHECK_EQ(sizeOf(x.size()), ambientDim_);
  CHECK_EQ(sizeOf(J.cols()), tangentDim_);
  J.conservativeResize(Eigen::NoChange, ambientDim_);
  // Right to left: a block's ambient columns start at or after its tangent ones, so a block is written only over
  // columns already read (or over the new columns).
  for (std::vector<Block>::const_reverse_iterator block = blocks_.rbegin(); block != blocks_.rend(); ++block) {
    if (block->type == StateManifoldSegment::Type::kEuclidean) {
      if (block->tangentOffset != block->ambientOffset) {
        for (size_t j = block->ambientDim; j > 0; --j) {
          J.col(block->ambientOffset + j - 1) = J.col(block->tangentOffset + j - 1);
        }
      }
    } else {
      const quaternion_t xi = x.segment<4>(block->ambientOffset);
      const matrix_t lifted = J.middleCols<3>(block->tangentOffset) * quaternionTangentMapPseudoInverse(xi);
      J.middleCols<4>(block->ambientOffset) = lifted;
    }
  }
}

void ProductStateManifold::pushForwardDynamics(const vector_t& x,
                                               const vector_t& xNext,
                                               VectorFunctionLinearApproximation& dynamics) const {
  CHECK_EQ(sizeOf(x.size()), ambientDim_);
  CHECK_EQ(sizeOf(xNext.size()), ambientDim_);
  CHECK_EQ(sizeOf(dynamics.f.size()), ambientDim_);
  CHECK_EQ(sizeOf(dynamics.dfdx.rows()), ambientDim_);
  CHECK_EQ(sizeOf(dynamics.dfdx.cols()), ambientDim_);
  const bool hasInputSensitivity = dynamics.dfdu.rows() > 0;
  if (hasInputSensitivity) {
    CHECK_EQ(sizeOf(dynamics.dfdu.rows()), ambientDim_);
  }

  // The gap f = difference(xNext, Phi), and the row map of each quaternion block: Jr^-1(f_q) E+(Pi(Phi)) Pi'(Phi), which is
  // Jr^-1(f_q) 2 G(xi_Phi)' / |xi_Phi|^2 because G(xi)' xi = 0.
  const vector_t& phi = dynamics.f;
  vector_t gap(tangentDim_);
  std::vector<Eigen::Matrix<scalar_t, 3, 4>> rowMaps;
  for (const Block& block : blocks_) {
    if (block.type == StateManifoldSegment::Type::kEuclidean) {
      gap.segment(block.tangentOffset, block.tangentDim) =
          phi.segment(block.ambientOffset, block.ambientDim) - xNext.segment(block.ambientOffset, block.ambientDim);
    } else {
      const quaternion_t xiPhi = phi.segment<4>(block.ambientOffset);
      const quaternion_t xiNext = xNext.segment<4>(block.ambientOffset);
      const rotation_t blockGap = quaternionLog(quaternionProduct(quaternionConjugate(xiNext), xiPhi));
      gap.segment<3>(block.tangentOffset) = blockGap;
      rowMaps.push_back(so3RightJacobianInverse(blockGap) * quaternionTangentMapPseudoInverse(xiPhi));
    }
  }

  mapQuaternionRows(rowMaps, dynamics.dfdx);
  mapQuaternionColumns(tangentMaps(x), dynamics.dfdx);
  if (hasInputSensitivity) {
    mapQuaternionRows(rowMaps, dynamics.dfdu);
  }
  dynamics.f = std::move(gap);
}

}  // namespace ocs2
