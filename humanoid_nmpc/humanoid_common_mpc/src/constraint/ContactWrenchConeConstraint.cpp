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

#include "pinocchio/fwd.hpp"

#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/LU"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"

namespace ocs2::humanoid {

namespace {

constexpr scalar_t kPatchOffsetZeroThreshold = 1.0e-9;

/** The generators a facet of a pointed cone in the six-dimensional wrench space
 * has to contain, at least. */
constexpr int kFacetSubsetSize = 5;
/** Slack of the facet test on unit-norm generators and normals. */
constexpr scalar_t kFacetTolerance = 1.0e-9;
/** Two unit facet normals closer than this are the same facet, found from
 * another subset of its generators. */
constexpr scalar_t kFacetDuplicateTolerance = 1.0e-7;
/** A unit facet normal whose yaw-moment component is below this is a friction
 * or CoP facet, not a torsion facet. */
constexpr scalar_t kYawComponentThreshold = 1.0e-9;
/** Slack of computeLimitsPerNewtonTorqueContactEwrtContactZ() on the rows that
 * do not involve the yaw moment. */
constexpr scalar_t kTorsionLimitTolerance = 1.0e-12;
/** The pyramid's edges sit half a sector away from its facets. */
constexpr scalar_t kPyramidEdgeOffsetInSectors = 0.5;

/** The cross-product matrix [v]x, with [v]x w = v x w. */
matrix3_t skewSymmetric(const vector3_t& v) {
  matrix3_t skew;
  skew << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return skew;
}

}  // namespace

absl::Status ContactWrenchConeConstraint::checkWrenchInputLayout(
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    size_t contactPointIndex) {
  const size_t wrenchStart =
      mpcRobotModel.getContactWrenchStartIndices(contactPointIndex);
  const size_t forceStart =
      mpcRobotModel.getContactForceStartIndices(contactPointIndex);
  const size_t momentStart =
      mpcRobotModel.getContactMomentStartIndices(contactPointIndex);
  const size_t inputDim = mpcRobotModel.getContactInputDim(contactPointIndex);
  if (inputDim != kContactWrenchDim || forceStart != wrenchStart ||
      momentStart != wrenchStart + kForceDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactWrenchConeConstraint] contact ", contactPointIndex,
        " is parameterized by ", inputDim, " inputs starting at ", wrenchStart,
        " (force at ", forceStart, ", moment at ", momentStart,
        "), not by its six-entry wrench. This term reads the wrench in the "
        "world frame and writes its Jacobian at the force and moment "
        "start indices, which is only right for a wrench-space model; with ",
        R"(contact_input_parameterization: "basis_vectors")",
        " the cone is enforced structurally by the non-negativity of the basis "
        "scalings instead "
        "(humanoid_nmpc/docs/contact_basis_vectors/README.md)."));
  }
  return absl::OkStatus();
}

absl::Status ContactWrenchConeConstraint::checkPinocchioVelocityDim(
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel) {
  const int pinocchioVelocityDim = pinocchioInterface.getModel().nv;
  if (static_cast<int64_t>(mpcRobotModel.getVelocityDim()) !=
      static_cast<int64_t>(pinocchioVelocityDim)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ContactWrenchConeConstraint] the robot model's Pinocchio velocity "
        "has ",
        mpcRobotModel.getVelocityDim(),
        " entries, but the Pinocchio model's has ", pinocchioVelocityDim,
        ": the frame Jacobian cannot be taken to the state."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>>
ContactWrenchConeConstraint::Create(
    const SwitchedModelReferenceManager& referenceManager,
    const ContactRectangle& contactRectangle,
    size_t contactPointIndex,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    Config config,
    bool scheduleGated) {
  RETURN_IF_ERROR(checkWrenchInputLayout(mpcRobotModel, contactPointIndex));
  RETURN_IF_ERROR(checkPinocchioVelocityDim(pinocchioInterface, mpcRobotModel));
  RETURN_IF_ERROR(validateConfig(config));
  RETURN_IF_ERROR(validateFootprint(contactRectangle));
  return std::make_unique<ContactWrenchConeConstraint>(
      referenceManager, contactRectangle, contactPointIndex, pinocchioInterface,
      mpcRobotModel, std::move(config), scheduleGated);
}

ContactWrenchConeConstraint::ContactWrenchConeConstraint(
    const SwitchedModelReferenceManager& referenceManager,
    const ContactRectangle& contactRectangle,
    size_t contactPointIndex,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    Config config,
    bool scheduleGated)
    : StateInputConstraint(ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      pinocchioInterfacePtr_(&pinocchioInterface),
      mpcRobotModelPtr_(&mpcRobotModel),
      contactRectangle_(contactRectangle),
      contactPointIndex_(contactPointIndex),
      config_(scheduleGated ? std::move(config)
                            : withoutLoadedFootOffsets(std::move(config))),
      scheduleGated_(scheduleGated) {
  // The documented guards for the callers that construct directly; Create()
  // returns the same refusals as a Status.
  const absl::Status wrenchLayout =
      checkWrenchInputLayout(mpcRobotModel, contactPointIndex);
  CHECK(wrenchLayout.ok()) << wrenchLayout.message();
  const absl::Status velocityDim =
      checkPinocchioVelocityDim(pinocchioInterface, mpcRobotModel);
  CHECK(velocityDim.ok()) << velocityDim.message();
  const absl::Status validConfig = validateConfig(config_);
  CHECK(validConfig.ok()) << validConfig.message();
  const absl::Status validFootprint = validateFootprint(contactRectangle_);
  CHECK(validFootprint.ok()) << validFootprint.message();
  initializeLocalConstraintMatrix();
}

namespace {

/** InvalidArgument naming `<kConfigField>.<field>` unless `inRange`. */
absl::Status checkConeValue(bool inRange,
                            absl::string_view field,
                            scalar_t value,
                            absl::string_view requirement) {
  if (inRange) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("[ContactWrenchConeConstraint] ",
                   ContactWrenchConeConstraint::kConfigField, ".", field,
                   " is ", value, " but must be ", requirement, "."));
}

}  // namespace

absl::Status ContactWrenchConeConstraint::validateConfig(const Config& config) {
  RETURN_IF_ERROR(checkConeValue(
      config.numBasisVectors >= Config::kMinNumBasisVectors &&
          config.numBasisVectors <= Config::kMaxNumBasisVectors,
      "num_basis_vectors", static_cast<scalar_t>(config.numBasisVectors),
      absl::StrCat(
          "from ", Config::kMinNumBasisVectors,
          " (the facets of the friction pyramid) to ",
          Config::kMaxNumBasisVectors,
          " (the torsion rows are enumerated from C(4N, 5) subsets)")));
  RETURN_IF_ERROR(checkConeValue(std::isfinite(config.frictionCoefficient) &&
                                     config.frictionCoefficient > 0.0,
                                 "friction_coefficient",
                                 config.frictionCoefficient,
                                 "finite and positive"));
  RETURN_IF_ERROR(checkConeValue(
      std::isfinite(config.torsionalFrictionCoefficient) &&
          config.torsionalFrictionCoefficient >= 0.0,
      "torsional_friction_coefficient", config.torsionalFrictionCoefficient,
      "finite and non-negative"));
  RETURN_IF_ERROR(checkConeValue(
      std::isfinite(config.minForceContactEwrtContactZ) &&
          config.minForceContactEwrtContactZ >= 0.0,
      "min_force_contact_ewrt_contact_z", config.minForceContactEwrtContactZ,
      "finite and non-negative"));
  RETURN_IF_ERROR(checkConeValue(
      std::isfinite(config.gripForceContactEwrtContactZ) &&
          config.gripForceContactEwrtContactZ >= 0.0,
      "grip_force_contact_ewrt_contact_z", config.gripForceContactEwrtContactZ,
      "finite and non-negative"));
  if (!config.patchOffset.allFinite()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ContactWrenchConeConstraint] the torsional reference "
                     "point (Config::patchOffset) is [",
                     config.patchOffset.x(), ", ", config.patchOffset.y(), ", ",
                     config.patchOffset.z(), "] but must be finite."));
  }
  return absl::OkStatus();
}

absl::Status ContactWrenchConeConstraint::validateFootprint(
    const ContactRectangle& contactRectangle) {
  const PolygonBounds& bounds = contactRectangle.getBounds();
  const bool finite =
      std::isfinite(bounds.x_min) && std::isfinite(bounds.x_max) &&
      std::isfinite(bounds.y_min) && std::isfinite(bounds.y_max);
  if (finite && bounds.x_min < bounds.x_max && bounds.y_min < bounds.y_max) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "[ContactWrenchConeConstraint] ", kFootprintField,
      ".{x_min, x_max, y_min, y_max} is [", bounds.x_min, ", ", bounds.x_max,
      "] x [", bounds.y_min, ", ", bounds.y_max,
      "] but must be finite with x_min < x_max and y_min < y_max: the wrench "
      "cone of a "
      "footprint of zero length or width is not full-dimensional."));
}

ContactWrenchConeConstraint::Config
ContactWrenchConeConstraint::withoutLoadedFootOffsets(Config config) {
  // `minForceContactEwrtContactZ` and `gripForceContactEwrtContactZ` are the
  // only two entries of `b` (the others are homogeneous rows). Both are
  // statements about a foot the mode schedule has declared loaded: the first
  // demands a normal force the foot cannot produce in flight, the second offers
  // an adhesion it does not have. An always-active cone evaluated on a foot at
  // zero wrench would report
  // `-minForceContactEwrtContactZ`, i.e. a permanent violation, and the penalty
  // would buy it off by inventing a normal force. Dropping them leaves the
  // homogeneous cone, which the zero wrench satisfies exactly.
  if (config.minForceContactEwrtContactZ > 0.0 ||
      config.gripForceContactEwrtContactZ > 0.0) {
    LOG(WARNING) << "[ContactWrenchConeConstraint] the contact-implicit "
                    "formulation drops "
                 << kConfigField << ".min_force_contact_ewrt_contact_z ("
                 << config.minForceContactEwrtContactZ
                 << " N) and .grip_force_contact_ewrt_contact_z ("
                 << config.gripForceContactEwrtContactZ
                 << " N): an always-active cone cannot demand either of a foot "
                    "in flight. "
                 << "The homogeneous friction, center-of-pressure and "
                    "torsional limits are unchanged.";
  }
  config.minForceContactEwrtContactZ = 0.0;
  config.gripForceContactEwrtContactZ = 0.0;
  return config;
}

ContactWrenchConeConstraint::ContactWrenchConeConstraint(
    const ContactWrenchConeConstraint& other) = default;

bool ContactWrenchConeConstraint::isActive(scalar_t time) const {
  if (!isActive_) {
    return false;
  }
  // Under the contact-implicit formulation the mode schedule no longer decides
  // which foot carries load, so it cannot be allowed to decide which foot's
  // wrench is bounded either; see contactConstraintsAreScheduleGated().
  if (!scheduleGated_) {
    return true;
  }
  return referenceManagerPtr_->isInContact(time, contactPointIndex_);
}

vector3_t contactPatchReferencePoint(
    const ContactWrenchConeConstraint::Config& config,
    const ContactRectangle& contactRectangle) {
  if (!config.patchOffset.isZero(kPatchOffsetZeroThreshold)) {
    return config.patchOffset;
  }
  const PolygonBounds& bounds = contactRectangle.getBounds();
  return vector3_t(0.5 * (bounds.x_min + bounds.x_max),
                   0.5 * (bounds.y_min + bounds.y_max), 0.0);
}

scalar_t inscribedFrictionCoefficient(
    const ContactWrenchConeConstraint::Config& config) {
  return config.frictionCoefficient *
         std::cos(M_PI / static_cast<scalar_t>(config.numBasisVectors));
}

std::vector<vector3_t> frictionPyramidEdges(
    const ContactWrenchConeConstraint::Config& config) {
  const size_t numEdges = config.numBasisVectors;
  const scalar_t angleStep = 2.0 * M_PI / static_cast<scalar_t>(numEdges);
  std::vector<vector3_t> edges;
  edges.reserve(numEdges);
  for (size_t k = 0; k < numEdges; ++k) {
    const scalar_t angle =
        (static_cast<scalar_t>(k) + kPyramidEdgeOffsetInSectors) * angleStep;
    edges.emplace_back(config.frictionCoefficient * std::cos(angle),
                       config.frictionCoefficient * std::sin(angle), 1.0);
  }
  return edges;
}

std::array<vector2_t, ContactRectangle::kNumCorners> footprintCorners(
    const ContactRectangle& contactRectangle) {
  const PolygonBounds& bounds = contactRectangle.getBounds();
  return {vector2_t(bounds.x_max, bounds.y_max),
          vector2_t(bounds.x_max, bounds.y_min),
          vector2_t(bounds.x_min, bounds.y_max),
          vector2_t(bounds.x_min, bounds.y_min)};
}

namespace {

/** The wrench, in the contact frame and with the moment about its origin, that
 * the point force `generatorForce` (contact-frame axes) generates at the planar
 * point `point`. */
vector6_t generatorWrenchAt(const vector2_t& point,
                            const vector3_t& generatorForce) {
  vector6_t generatorWrench;
  generatorWrench.head<kForceDim>() = generatorForce;
  generatorWrench.tail<kTorqueDim>() =
      vector3_t(point.x(), point.y(), 0.0).cross(generatorForce);
  return generatorWrench;
}

/** Advances `indices`, a strictly increasing k-subset of {0, ..., numElements -
 * 1}, to the next in lexicographic order. */
bool nextCombination(std::array<size_t, kFacetSubsetSize>& indices,
                     size_t numElements) {
  constexpr size_t kSubsetSize = static_cast<size_t>(kFacetSubsetSize);
  for (int64_t index = kFacetSubsetSize - 1; index >= 0; --index) {
    const size_t slot = static_cast<size_t>(index);
    if (indices[slot] + kSubsetSize < numElements + slot) {
      ++indices[slot];
      for (size_t later = slot + 1; later < kSubsetSize; ++later) {
        indices[later] = indices[later - 1] + 1;
      }
      return true;
    }
  }
  return false;
}

/**
 * The unit outward-free normals u (u . g >= 0 for every column g) of the facets
 * of the cone generated by the columns of `generators`, which must be
 * full-dimensional. Brute force over every five-column subset of full rank: the
 * normal of its span is a facet normal when every column lies on one side of
 * it. Every facet of a six-dimensional cone contains five linearly independent
 * generators, so none is missed, and the side test makes every returned row
 * valid.
 */
std::vector<vector6_t> enumerateConeFacets(const matrix_t& generators) {
  const size_t numGenerators = static_cast<size_t>(generators.cols());
  std::vector<vector6_t> facets;
  std::array<size_t, kFacetSubsetSize> subset{};
  std::iota(subset.begin(), subset.end(), size_t{0});
  Eigen::Matrix<scalar_t, kFacetSubsetSize, kWrenchDim> spanRows;
  do {
    for (int i = 0; i < kFacetSubsetSize; ++i) {
      spanRows.row(i) =
          generators
              .col(static_cast<Eigen::Index>(subset[static_cast<size_t>(i)]))
              .transpose();
    }
    Eigen::FullPivLU<Eigen::Matrix<scalar_t, kFacetSubsetSize, kWrenchDim>> lu(
        spanRows);
    lu.setThreshold(kFacetTolerance);
    if (lu.rank() < kFacetSubsetSize) continue;
    vector6_t normal = lu.kernel().col(0);
    normal.normalize();
    const vector_t sides = generators.transpose() * normal;
    if (sides.maxCoeff() <= kFacetTolerance) {
      normal = -normal;
    } else if (sides.minCoeff() < -kFacetTolerance) {
      continue;
    }
    const bool known = std::any_of(
        facets.begin(), facets.end(), [&normal](const vector6_t& facet) {
          return (facet - normal).norm() < kFacetDuplicateTolerance;
        });
    if (!known) facets.push_back(normal);
  } while (nextCombination(subset, numGenerators));
  return facets;
}

/**
 * The torsion rows of the contact wrench cone, (u_F, u_tau) with u_tau.z = +-1,
 * moments about the contact frame origin.
 *
 * The facets are enumerated in a well-conditioned frame: moments about the
 * footprint center c, divided by the larger half extent L, and every generator
 * normalized, none of which changes the cone. A row (n_F, n_tau) there reads
 * n_F . F + n_tau . (tau - c x F) / L, i.e. u_tau = n_tau / L and u_F = n_F -
 * u_tau x c in the contact frame.
 */
std::vector<vector6_t> torsionFacets(
    const ContactWrenchConeConstraint::Config& config,
    const ContactRectangle& contactRectangle) {
  const PolygonBounds& bounds = contactRectangle.getBounds();
  const vector2_t planarCenter(0.5 * (bounds.x_min + bounds.x_max),
                               0.5 * (bounds.y_min + bounds.y_max));
  const vector3_t center(planarCenter.x(), planarCenter.y(), 0.0);
  const scalar_t lengthScale =
      0.5 * std::max(bounds.x_max - bounds.x_min, bounds.y_max - bounds.y_min);

  const std::vector<vector3_t> edges = frictionPyramidEdges(config);
  const std::array<vector2_t, ContactRectangle::kNumCorners> corners =
      footprintCorners(contactRectangle);
  matrix_t scaledGenerators(
      kWrenchDim, static_cast<Eigen::Index>(edges.size() * corners.size()));
  Eigen::Index column = 0;
  for (const vector3_t& edge : edges) {
    for (const vector2_t& corner : corners) {
      vector6_t generator = generatorWrenchAt(corner - planarCenter, edge);
      generator.tail<kTorqueDim>() /= lengthScale;
      scaledGenerators.col(column++) = generator.normalized();
    }
  }

  std::vector<vector6_t> rows;
  for (const vector6_t& normal : enumerateConeFacets(scaledGenerators)) {
    if (std::abs(normal(kWrenchDim - 1)) <= kYawComponentThreshold)
      continue;  // a friction or CoP facet
    const vector3_t momentCoefficients =
        normal.tail<kTorqueDim>() / lengthScale;
    vector6_t row;
    row.head<kForceDim>() =
        normal.head<kForceDim>() - momentCoefficients.cross(center);
    row.tail<kTorqueDim>() = momentCoefficients;
    rows.push_back(row / std::abs(momentCoefficients.z()));
  }
  return rows;
}

}  // namespace

matrix_t cornerContactWrenchGenerators(
    const ContactWrenchConeConstraint::Config& config,
    const ContactRectangle& contactRectangle) {
  const std::vector<vector3_t> edges = frictionPyramidEdges(config);
  const std::array<vector2_t, ContactRectangle::kNumCorners> corners =
      footprintCorners(contactRectangle);
  matrix_t generators(kWrenchDim,
                      static_cast<Eigen::Index>(edges.size() * corners.size()));
  Eigen::Index column = 0;
  for (const vector3_t& edge : edges) {
    for (const vector2_t& corner : corners) {
      generators.col(column++) = generatorWrenchAt(corner, edge);
    }
  }
  return generators;
}

ContactWrenchConeRows buildFrictionPyramidRows(
    const ContactWrenchConeConstraint::Config& config) {
  const absl::Status validConfig =
      ContactWrenchConeConstraint::validateConfig(config);
  CHECK(validConfig.ok()) << validConfig.message();
  const size_t numFrictionRows = config.numBasisVectors;
  const size_t numRows = numFrictionRows + 1;
  ContactWrenchConeRows rows;
  rows.numFrictionRows = numFrictionRows;
  rows.A_f = matrix_t::Zero(numRows, kForceDim);
  rows.A_tau = matrix_t::Zero(numRows, kTorqueDim);
  rows.b = vector_t::Zero(numRows);

  size_t constraintIdx = 0;
  // 1. The facets of the inscribed friction pyramid: mu cos(pi/N) (Fz + F_grip)
  // - (cos(theta_k) Fx + sin(theta_k) Fy) >= 0. Its edges, half a sector away,
  // lie on the Coulomb cone; the circumscribed pyramid this replaced admitted
  // mu / cos(pi/N).
  const scalar_t facetFriction = inscribedFrictionCoefficient(config);
  const scalar_t angleStep =
      2.0 * M_PI / static_cast<scalar_t>(numFrictionRows);
  for (size_t k = 0; k < numFrictionRows; ++k) {
    const scalar_t theta_k = static_cast<scalar_t>(k) * angleStep;
    rows.A_f(constraintIdx, kXAxisIndex) = -std::cos(theta_k);
    rows.A_f(constraintIdx, kYAxisIndex) = -std::sin(theta_k);
    rows.A_f(constraintIdx, kZAxisIndex) = facetFriction;
    rows.b(constraintIdx) = facetFriction * config.gripForceContactEwrtContactZ;
    ++constraintIdx;
  }

  // 2. Normal force limit: Fz - minForceContactEwrtContactZ >= 0
  rows.A_f(constraintIdx, kZAxisIndex) = 1.0;
  rows.b(constraintIdx) = -config.minForceContactEwrtContactZ;
  return rows;
}

ContactWrenchConeRows buildLocalWrenchConeRows(
    const ContactWrenchConeConstraint::Config& config,
    const ContactRectangle& contactRectangle) {
  // Every factory reports these as a Status naming the task-file key; only a
  // caller that bypassed all of them gets here.
  const absl::Status validConfig =
      ContactWrenchConeConstraint::validateConfig(config);
  CHECK(validConfig.ok()) << validConfig.message();
  const absl::Status validFootprint =
      ContactWrenchConeConstraint::validateFootprint(contactRectangle);
  CHECK(validFootprint.ok()) << validFootprint.message();

  const std::vector<vector6_t> torsionRows =
      torsionFacets(config, contactRectangle);
  const size_t numFrictionRows = config.numBasisVectors;
  const size_t numRows = numFrictionRows +
                         ContactWrenchConeRows::kNumNormalForceAndCopRows +
                         torsionRows.size();

  ContactWrenchConeRows rows;
  rows.numFrictionRows = numFrictionRows;
  rows.A_f = matrix_t::Zero(numRows, kForceDim);
  rows.A_tau = matrix_t::Zero(numRows, kTorqueDim);
  rows.b = vector_t::Zero(numRows);

  // 1. and 2. The friction pyramid and the normal force bound, the rows a
  // point contact shares (buildFrictionPyramidRows()).
  const ContactWrenchConeRows pyramid = buildFrictionPyramidRows(config);
  const Eigen::Index numPyramidRows =
      static_cast<Eigen::Index>(pyramid.numRows());
  rows.A_f.topRows(numPyramidRows) = pyramid.A_f;
  rows.b.head(numPyramidRows) = pyramid.b;
  size_t constraintIdx = pyramid.numRows();

  // 3. Center of Pressure (CoP) / Moment constraints (Bounds relative to local
  // foot contact frame)
  const PolygonBounds& bounds = contactRectangle.getBounds();
  // tau_x - y_min * Fz >= 0
  rows.A_f(constraintIdx, kZAxisIndex) = -bounds.y_min;
  rows.A_tau(constraintIdx, kXAxisIndex) = 1.0;
  ++constraintIdx;

  // -tau_x + y_max * Fz >= 0
  rows.A_f(constraintIdx, kZAxisIndex) = bounds.y_max;
  rows.A_tau(constraintIdx, kXAxisIndex) = -1.0;
  ++constraintIdx;

  // -tau_y - x_min * Fz >= 0
  rows.A_f(constraintIdx, kZAxisIndex) = -bounds.x_min;
  rows.A_tau(constraintIdx, kYAxisIndex) = -1.0;
  ++constraintIdx;

  // tau_y + x_max * Fz >= 0
  rows.A_f(constraintIdx, kZAxisIndex) = bounds.x_max;
  rows.A_tau(constraintIdx, kYAxisIndex) = 1.0;
  ++constraintIdx;

  // 4. The torsion facets of the corner cone. The gripper force acts on them as
  // a normal force at the footprint center.
  const vector2_t center(0.5 * (bounds.x_min + bounds.x_max),
                         0.5 * (bounds.y_min + bounds.y_max));
  const vector6_t gripWrenchContactEwrtContact = generatorWrenchAt(
      center, vector3_t(0.0, 0.0, config.gripForceContactEwrtContactZ));
  for (const vector6_t& torsionRow : torsionRows) {
    rows.A_f.row(constraintIdx) = torsionRow.head<kForceDim>().transpose();
    rows.A_tau.row(constraintIdx) = torsionRow.tail<kTorqueDim>().transpose();
    rows.b(constraintIdx) = torsionRow.dot(gripWrenchContactEwrtContact);
    ++constraintIdx;
  }

  ABSL_CHECK_EQ(constraintIdx, numRows)
      << "buildLocalWrenchConeRows(): filled another number of rows than it "
         "sized";
  return rows;
}

std::optional<TorsionInterval> computeLimitsPerNewtonTorqueContactEwrtContactZ(
    const ContactWrenchConeRows& rows, const vector2_t& point) {
  const vector_t values =
      rows.evaluateCone(generatorWrenchAt(point, vector3_t::UnitZ()));
  TorsionInterval limits{.lower = -std::numeric_limits<scalar_t>::infinity(),
                         .upper = std::numeric_limits<scalar_t>::infinity()};
  for (Eigen::Index row = 0; row < values.size(); ++row) {
    // Row value at a yaw moment t added to the wrench: values(row) +
    // yawCoefficient * t >= 0.
    const scalar_t yawCoefficient = rows.A_tau(row, kZAxisIndex);
    if (yawCoefficient > 0.0) {
      limits.lower = std::max(limits.lower, -values(row) / yawCoefficient);
    } else if (yawCoefficient < 0.0) {
      limits.upper = std::min(limits.upper, values(row) / -yawCoefficient);
    } else if (values(row) < -kTorsionLimitTolerance) {
      return std::nullopt;  // a friction or CoP row the unit normal force at
                            // `point` violates whatever its yaw moment
    }
  }
  if (limits.lower > limits.upper + kTorsionLimitTolerance) return std::nullopt;
  limits.lower = std::min(limits.lower, limits.upper);
  return limits;
}

void ContactWrenchConeConstraint::initializeLocalConstraintMatrix() {
  const ContactWrenchConeRows rows =
      buildLocalWrenchConeRows(config_, contactRectangle_);
  numConstraints_ = rows.numRows();
  A_f_local_ = rows.A_f;
  A_tau_local_ = rows.A_tau;
  b_local_ = rows.b;
}

vector_t ContactWrenchConeConstraint::getValue(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  pinocchio::Data data = pinocchioInterfacePtr_->getData();
  updateFramePlacements(mpcRobotModelPtr_->getGeneralizedCoordinates(state),
                        model, data);
  const pinocchio::FrameIndex frameID = getContactFrameIndex(
      *pinocchioInterfacePtr_, *mpcRobotModelPtr_, contactPointIndex_);

  const matrix3_t w_R_l = getRotmatLinkRtWorld(data, frameID);
  const matrix3_t l_R_w = w_R_l.transpose();

  const vector3_t forceContactEwrtWorld =
      mpcRobotModelPtr_->getForceContactEwrtParam(input, contactPointIndex_);
  const vector3_t torqueContactEwrtWorld =
      mpcRobotModelPtr_->getTorqueContactEwrtParam(input, contactPointIndex_);

  const vector3_t forceContactEwrtContact = l_R_w * forceContactEwrtWorld;
  const vector3_t torqueContactEwrtContact = l_R_w * torqueContactEwrtWorld;

  return A_f_local_ * forceContactEwrtContact +
         A_tau_local_ * torqueContactEwrtContact + b_local_;
}

VectorFunctionLinearApproximation
ContactWrenchConeConstraint::getLinearApproximation(
    scalar_t /*time*/,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& /*preComp*/) const {
  const pinocchio::Model& model = pinocchioInterfacePtr_->getModel();
  pinocchio::Data data = pinocchioInterfacePtr_->getData();
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(state);
  updateFramePlacements(q, model, data);
  const pinocchio::FrameIndex frameID = getContactFrameIndex(
      *pinocchioInterfacePtr_, *mpcRobotModelPtr_, contactPointIndex_);

  const matrix3_t w_R_l = getRotmatLinkRtWorld(data, frameID);
  const matrix3_t l_R_w = w_R_l.transpose();

  const matrix_t A_f_world = A_f_local_ * l_R_w;
  const matrix_t A_tau_world = A_tau_local_ * l_R_w;

  const vector3_t forceContactEwrtWorld =
      mpcRobotModelPtr_->getForceContactEwrtParam(input, contactPointIndex_);
  const vector3_t torqueContactEwrtWorld =
      mpcRobotModelPtr_->getTorqueContactEwrtParam(input, contactPointIndex_);
  const vector3_t forceContactEwrtContact = l_R_w * forceContactEwrtWorld;
  const vector3_t torqueContactEwrtContact = l_R_w * torqueContactEwrtWorld;

  // THE STATE DERIVATIVE, through the rotation. With w_R_l' = w_R_l
  // [omega_local]x, the local image of a fixed world vector moves as d(l_R_w
  // v)/dt = -[omega_local]x (l_R_w v) = [v_local]x omega_local, and omega_local
  // is the angular half of the contact frame's LOCAL Jacobian times the
  // Pinocchio velocity v. That gives the derivative in a step of q in v's
  // coordinates (Pinocchio's tangent), which the robot model's
  // getPinocchioTangentJacobian() takes to the state: the identity on the
  // position and joint slots, E^+(xi) on the quaternion's.
  matrix_t frameJacobian = matrix_t::Zero(kTwistDim, model.nv);
  pinocchio::computeFrameJacobian(model, data, q, frameID, pinocchio::LOCAL,
                                  frameJacobian);
  const matrix_t angularJacobian =
      frameJacobian.bottomRows<kRotationTangentDim>();
  const matrix_t dfdq =
      A_f_local_ * skewSymmetric(forceContactEwrtContact) * angularJacobian +
      A_tau_local_ * skewSymmetric(torqueContactEwrtContact) * angularJacobian;

  VectorFunctionLinearApproximation linearApproximation;
  linearApproximation.f = A_f_local_ * forceContactEwrtContact +
                          A_tau_local_ * torqueContactEwrtContact + b_local_;
  linearApproximation.dfdx =
      dfdq * mpcRobotModelPtr_->getPinocchioTangentJacobian(state);
  linearApproximation.dfdu =
      matrix_t::Zero(numConstraints_, mpcRobotModelPtr_->getInputDim());

  const size_t forceStartIdx =
      mpcRobotModelPtr_->getContactForceStartIndices(contactPointIndex_);
  const size_t momentStartIdx =
      mpcRobotModelPtr_->getContactMomentStartIndices(contactPointIndex_);

  linearApproximation.dfdu.block(0, forceStartIdx, numConstraints_, kForceDim) =
      A_f_world;
  linearApproximation.dfdu.block(0, momentStartIdx, numConstraints_,
                                 kTorqueDim) = A_tau_world;

  return linearApproximation;
}

VectorFunctionQuadraticApproximation
ContactWrenchConeConstraint::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const vector_t& input,
    const PreComputation& preComp) const {
  const VectorFunctionLinearApproximation linearApprox =
      getLinearApproximation(time, state, input, preComp);
  VectorFunctionQuadraticApproximation quadraticApproximation;
  quadraticApproximation.f = linearApprox.f;
  quadraticApproximation.dfdx = linearApprox.dfdx;
  quadraticApproximation.dfdu = linearApprox.dfdu;
  quadraticApproximation.dfdxx.resize(
      numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getStateDim(),
                                      mpcRobotModelPtr_->getStateDim()));
  quadraticApproximation.dfduu.resize(
      numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getInputDim(),
                                      mpcRobotModelPtr_->getInputDim()));
  quadraticApproximation.dfdux.resize(
      numConstraints_, matrix_t::Zero(mpcRobotModelPtr_->getInputDim(),
                                      mpcRobotModelPtr_->getStateDim()));
  return quadraticApproximation;
}

}  // namespace ocs2::humanoid
