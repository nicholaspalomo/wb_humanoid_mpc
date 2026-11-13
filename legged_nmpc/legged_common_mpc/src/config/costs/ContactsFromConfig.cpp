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

#include "legged_common_mpc/config/costs/ContactsFromConfig.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"

#include "legged_common_mpc/common/BasisInputsCostTransform.h"
#include "legged_common_mpc/common/ContactInputParameterization.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "legged_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "legged_common_mpc/contact/ContactCenterPoint.h"
#include "legged_common_mpc/contact/ContactPolygon.h"
#include "legged_common_mpc/contact/ContactRectangle.h"
#include "legged_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "legged_mpc_config/contacts_config.nproto.h"
#include "legged_mpc_config/xyz.nproto.h"

namespace ocs2::legged {
namespace {

constexpr absl::string_view kWrenchCone =
    "contacts.contact_wrench_cone_soft_constraint";
/** The fewest basis vectors of a wrench cone, as a signed count, to check a
 * count before it becomes a size_t. */
constexpr int32_t kMinConeBasisVectors = static_cast<int32_t>(
    ContactWrenchConeConstraint::Config::kMinNumBasisVectors);

/** InvalidArgument naming `<block>.<field>` unless `value` is finite. */
absl::Status checkFinite(absl::string_view block,
                         absl::string_view field,
                         double value) {
  if (std::isfinite(value)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      block, ".", field, " is ", value, ", but it must be finite."));
}

/** InvalidArgument naming `<block>.<field>` unless `value` is finite and
 * positive. */
absl::Status checkPositive(absl::string_view block,
                           absl::string_view field,
                           double value) {
  if (std::isfinite(value) && value > 0.0) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat(block, ".", field, " is ", value,
                   ", but it must be finite and positive: a negative barrier "
                   "weight rewards violating the "
                   "constraint, and the relaxed barrier is undefined at a "
                   "non-positive delta."));
}

/** The required field `field` of the wrench cone block, or InvalidArgument
 * saying that every consumer needs it. */
template <typename T>
absl::StatusOr<T> requiredConeValue(std::optional<T> value,
                                    absl::string_view field) {
  if (value.has_value()) {
    return *value;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      kWrenchCone, ".", field,
      " is missing. The block is the ground of the whole-body constraints: the "
      "wrench cone, the generators of the basis-vector "
      "contact inputs and the online contact planner all read it, and none of "
      "them falls back to a library default. Give it "
      "friction_coefficient, torsional_friction_coefficient, "
      "min_force_contact_ewrt_contact_z, "
      "grip_force_contact_ewrt_contact_z and num_basis_vectors."));
}

/** How far from 1 a surface pad axis's length, and from 0 the dot product of
 * its two axes, may be before they are refused; within it they are normalized.
 */
constexpr scalar_t kPadAxisTolerance = 1.0e-3;

/** The entry of contacts.surface_pads for `contactName`, or null. */
const mpc_config::ContactsConfig::SurfacePad* absl_nullable findSurfacePad(
    const mpc_config::ContactsConfig& contacts, absl::string_view contactName) {
  for (const mpc_config::ContactsConfig::SurfacePad& pad :
       contacts.surface_pads) {
    if (pad.contact == contactName) return &pad;
  }
  return nullptr;
}

/** `xyz` as a vector, InvalidArgument naming `<block>.<field>` unless finite.
 */
absl::StatusOr<vector3_t> finiteVector(absl::string_view block,
                                       absl::string_view field,
                                       const mpc_config::Xyz& xyz) {
  const std::string name = absl::StrCat(block, ".", field);
  RETURN_IF_ERROR(checkFinite(name, "x", xyz.x));
  RETURN_IF_ERROR(checkFinite(name, "y", xyz.y));
  RETURN_IF_ERROR(checkFinite(name, "z", xyz.z));
  return vector3_t(xyz.x, xyz.y, xyz.z);
}

/** A unit vector of `block`.`field`, normalized; InvalidArgument when it is
 * not unit within kPadAxisTolerance. */
absl::StatusOr<vector3_t> unitVector(absl::string_view block,
                                     absl::string_view field,
                                     const mpc_config::Xyz& xyz) {
  ASSIGN_OR_RETURN(const vector3_t vector, finiteVector(block, field, xyz));
  if (std::abs(vector.norm() - 1.0) > kPadAxisTolerance) {
    return absl::InvalidArgumentError(absl::StrCat(
        block, ".", field, " has length ", vector.norm(),
        ", but it must be a unit vector (within ", kPadAxisTolerance, ")."));
  }
  return vector3_t(vector.normalized());
}

/**
 * parent_R_contact of a surface pad: z = -normal_pad_ewrt_link, x =
 * x_axis_pad_ewrt_link made
 * orthogonal to z, y = z x x. InvalidArgument naming the field of an axis that
 * is not unit, or of an x axis not orthogonal to the normal.
 */
absl::StatusOr<matrix3_t> parseRotmatPadRtLink(
    absl::string_view block,
    const mpc_config::ContactsConfig::SurfacePad& pad) {
  ASSIGN_OR_RETURN(
      const vector3_t normalPadEwrtLink,
      unitVector(block, "normal_pad_ewrt_link", pad.normal_pad_ewrt_link));
  ASSIGN_OR_RETURN(
      const vector3_t xAxisPadEwrtLink,
      unitVector(block, "x_axis_pad_ewrt_link", pad.x_axis_pad_ewrt_link));
  if (std::abs(normalPadEwrtLink.dot(xAxisPadEwrtLink)) > kPadAxisTolerance) {
    return absl::InvalidArgumentError(absl::StrCat(
        block,
        ".x_axis_pad_ewrt_link is not orthogonal to normal_pad_ewrt_link: ",
        "their dot product is ", normalPadEwrtLink.dot(xAxisPadEwrtLink), "."));
  }
  const vector3_t z = -normalPadEwrtLink;
  const vector3_t x =
      (xAxisPadEwrtLink - xAxisPadEwrtLink.dot(z) * z).normalized();
  matrix3_t rotmatPadRtLink;
  rotmatPadRtLink.col(0) = x;
  rotmatPadRtLink.col(1) = z.cross(x);
  rotmatPadRtLink.col(kZAxisIndex) = z;
  return rotmatPadRtLink;
}

/** The name of contacts.surface_pads[`index`], for the messages. */
std::string surfacePadBlock(size_t index) {
  return absl::StrCat("contacts.surface_pads[", index, "]");
}

/** The index of `pad` in contacts.surface_pads. */
size_t surfacePadIndex(const mpc_config::ContactsConfig& contacts,
                       const mpc_config::ContactsConfig::SurfacePad& pad) {
  return static_cast<size_t>(&pad - contacts.surface_pads.data());
}

/** The rectangle `rectangle` of `block`, checked as contactRectangle's. */
absl::StatusOr<ContactRectangle> rectangleFromConfig(
    absl::string_view block,
    const mpc_config::ContactsConfig::Rectangle& rectangle,
    const ContactCenterPoint& centerPoint) {
  RETURN_IF_ERROR(
      checkFinite(block, "min_pos_corners_rt_contact_ewrt_contact_x",
                  rectangle.min_pos_corners_rt_contact_ewrt_contact_x));
  RETURN_IF_ERROR(
      checkFinite(block, "max_pos_corners_rt_contact_ewrt_contact_x",
                  rectangle.max_pos_corners_rt_contact_ewrt_contact_x));
  RETURN_IF_ERROR(
      checkFinite(block, "min_pos_corners_rt_contact_ewrt_contact_y",
                  rectangle.min_pos_corners_rt_contact_ewrt_contact_y));
  RETURN_IF_ERROR(
      checkFinite(block, "max_pos_corners_rt_contact_ewrt_contact_y",
                  rectangle.max_pos_corners_rt_contact_ewrt_contact_y));
  RETURN_IF_ERROR(checkFinite(block, "scale_factor", rectangle.scale_factor));
  // An empty or inverted rectangle has no interior for the wrench cone and the
  // center of pressure, and a scale factor of 0 or less shrinks it to a point
  // or flips it.
  if (!(rectangle.min_pos_corners_rt_contact_ewrt_contact_x <
        rectangle.max_pos_corners_rt_contact_ewrt_contact_x) ||
      !(rectangle.min_pos_corners_rt_contact_ewrt_contact_y <
        rectangle.max_pos_corners_rt_contact_ewrt_contact_y)) {
    return absl::InvalidArgumentError(absl::StrCat(
        block, " is x in [",
        rectangle.min_pos_corners_rt_contact_ewrt_contact_x, ", ",
        rectangle.max_pos_corners_rt_contact_ewrt_contact_x, "], y in [",
        rectangle.min_pos_corners_rt_contact_ewrt_contact_y, ", ",
        rectangle.max_pos_corners_rt_contact_ewrt_contact_y,
        "], but it must have min_pos_corners_rt_contact_ewrt_contact_x < "
        "max_pos_corners_rt_contact_ewrt_contact_x and "
        "min_pos_corners_rt_contact_ewrt_contact_y < "
        "max_pos_corners_rt_contact_ewrt_contact_y."));
  }
  if (!(rectangle.scale_factor > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat(block, ".scale_factor is ", rectangle.scale_factor,
                     ", but it must be positive."));
  }
  return ContactRectangle(
      PolygonBounds(rectangle.min_pos_corners_rt_contact_ewrt_contact_x,
                    rectangle.max_pos_corners_rt_contact_ewrt_contact_x,
                    rectangle.min_pos_corners_rt_contact_ewrt_contact_y,
                    rectangle.max_pos_corners_rt_contact_ewrt_contact_y),
      centerPoint, rectangle.scale_factor);
}

/** The relaxed barrier `block` {mu delta}, both finite and positive. */
absl::StatusOr<RelaxedBarrierPenalty::Config> relaxedBarrier(
    absl::string_view block, double mu, double delta) {
  RETURN_IF_ERROR(checkPositive(block, "mu", mu));
  RETURN_IF_ERROR(checkPositive(block, "delta", delta));
  return RelaxedBarrierPenalty::Config(mu, delta);
}

}  // namespace

absl::StatusOr<ContactCenterPoint> contactCenterPointFromConfig(
    const mpc_config::ContactsConfig& contacts,
    const ModelSettings& modelSettings,
    int contactIndex) {
  // The 6-DoF contacts follow the 3-DoF ones in the contact set, and their
  // parents are listed in their own order.
  const size_t numPointContacts = modelSettings.contactNames3DoF.size();
  if (contactIndex < 0 ||
      static_cast<size_t>(contactIndex) >= modelSettings.contactNames.size() ||
      (static_cast<size_t>(contactIndex) >= numPointContacts &&
       static_cast<size_t>(contactIndex) - numPointContacts >=
           modelSettings.contactParentJointNames.size())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "contact ", contactIndex, " is not one of the ",
        modelSettings.contactNames.size(), " contacts of the model settings"));
  }
  if (modelSettings.isPointContact(static_cast<size_t>(contactIndex))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "contact ", contactIndex, " (",
        modelSettings.contactNames[static_cast<size_t>(contactIndex)],
        ") is a 3-DoF contact, which has no footprint: the contact rectangle "
        "is a 6-DoF contact's."));
  }
  const size_t index = static_cast<size_t>(contactIndex);
  const std::string& name = modelSettings.contactNames[index];
  const std::string& parent =
      modelSettings.contactParentJointNames[index - numPointContacts];
  // A pad of its own (a seat or a torso pad), or the shared sole.
  if (const mpc_config::ContactsConfig::SurfacePad* absl_nullable pad =
          findSurfacePad(contacts, name);
      pad != nullptr) {
    const std::string block = surfacePadBlock(surfacePadIndex(contacts, *pad));
    ASSIGN_OR_RETURN(const vector3_t posPadRtLinkEwrtLink,
                     finiteVector(block, "pos_pad_rt_link_ewrt_link",
                                  pad->pos_pad_rt_link_ewrt_link));
    ASSIGN_OR_RETURN(const matrix3_t rotmatPadRtLink,
                     parseRotmatPadRtLink(block, *pad));
    return ContactCenterPoint(name, parent, posPadRtLinkEwrtLink,
                              rotmatPadRtLink);
  }
  ASSIGN_OR_RETURN(const vector3_t posContactRtLinkEwrtLink,
                   finiteVector("contacts", "pos_contact_rt_link_ewrt_link",
                                contacts.pos_contact_rt_link_ewrt_link));
  return ContactCenterPoint(name, parent, posContactRtLinkEwrtLink);
}

absl::StatusOr<ContactRectangle> contactRectangleFromConfig(
    const mpc_config::ContactsConfig& contacts,
    const ContactCenterPoint& centerPoint) {
  return rectangleFromConfig("contacts.contact_rectangle",
                             contacts.contact_rectangle, centerPoint);
}

absl::StatusOr<ContactRectangle> contactRectangleFromConfig(
    const mpc_config::ContactsConfig& contacts,
    const ModelSettings& modelSettings,
    int contactIndex) {
  ASSIGN_OR_RETURN(
      const ContactCenterPoint centerPoint,
      contactCenterPointFromConfig(contacts, modelSettings, contactIndex));
  if (const mpc_config::ContactsConfig::SurfacePad* absl_nullable pad =
          findSurfacePad(contacts, centerPoint.frameName);
      pad != nullptr) {
    return rectangleFromConfig(
        absl::StrCat(surfacePadBlock(surfacePadIndex(contacts, *pad)),
                     ".rectangle"),
        pad->rectangle, centerPoint);
  }
  return contactRectangleFromConfig(contacts, centerPoint);
}

absl::Status validateSurfacePads(const mpc_config::ContactsConfig& contacts,
                                 const ModelSettings& modelSettings) {
  absl::flat_hash_set<std::string> listed;
  for (size_t i = 0; i < contacts.surface_pads.size(); ++i) {
    const std::string& name = contacts.surface_pads[i].contact;
    const std::vector<std::string>& surfaces = modelSettings.contactNames6DoF;
    if (std::find(surfaces.begin(), surfaces.end(), name) == surfaces.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          surfacePadBlock(i), ".contact '", name,
          "' is not a 6-DoF contact of model_settings.contact_names_6dof."));
    }
    if (!listed.insert(name).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          surfacePadBlock(i), ".contact '", name, "' is listed twice."));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ContactWrenchConeConstraint::Config>
contactWrenchConeConfigFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::WrenchCone& cone =
      contacts.contact_wrench_cone_soft_constraint;
  ASSIGN_OR_RETURN(
      const scalar_t frictionCoefficient,
      requiredConeValue(cone.friction_coefficient, "friction_coefficient"));
  ASSIGN_OR_RETURN(const scalar_t torsionalFrictionCoefficient,
                   requiredConeValue(cone.torsional_friction_coefficient,
                                     "torsional_friction_coefficient"));
  ASSIGN_OR_RETURN(const scalar_t minForceContactEwrtContactZ,
                   requiredConeValue(cone.min_force_contact_ewrt_contact_z,
                                     "min_force_contact_ewrt_contact_z"));
  ASSIGN_OR_RETURN(const scalar_t gripForceContactEwrtContactZ,
                   requiredConeValue(cone.grip_force_contact_ewrt_contact_z,
                                     "grip_force_contact_ewrt_contact_z"));
  ASSIGN_OR_RETURN(
      const int32_t numBasisVectors,
      requiredConeValue(cone.num_basis_vectors, "num_basis_vectors"));
  // Checked before the conversion to size_t, which would turn a negative count
  // into a huge one.
  if (numBasisVectors < kMinConeBasisVectors) {
    return absl::InvalidArgumentError(
        absl::StrCat(kWrenchCone, ".num_basis_vectors is ", numBasisVectors,
                     " but must be at least ", kMinConeBasisVectors,
                     " (the facets of the friction pyramid)."));
  }
  ContactWrenchConeConstraint::Config config(
      static_cast<size_t>(numBasisVectors), frictionCoefficient,
      torsionalFrictionCoefficient, minForceContactEwrtContactZ,
      gripForceContactEwrtContactZ);
  RETURN_IF_ERROR(ContactWrenchConeConstraint::validateConfig(config));
  return config;
}

absl::StatusOr<RelaxedBarrierPenalty::Config>
contactWrenchConeBarrierFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::WrenchCone& cone =
      contacts.contact_wrench_cone_soft_constraint;
  RelaxedBarrierPenalty::Config barrier;
  if (cone.mu.has_value()) {
    barrier.mu = *cone.mu;
  }
  if (cone.delta.has_value()) {
    barrier.delta = *cone.delta;
  }
  RETURN_IF_ERROR(checkPositive(kWrenchCone, "mu", barrier.mu));
  RETURN_IF_ERROR(checkPositive(kWrenchCone, "delta", barrier.delta));
  return barrier;
}

absl::StatusOr<ContactWrenchConeBasisMatrix> contactWrenchConeBasisFromConfig(
    const mpc_config::ContactsConfig& contacts,
    const ContactRectangle& footprint) {
  ASSIGN_OR_RETURN(const ContactWrenchConeConstraint::Config coneConfig,
                   contactWrenchConeConfigFromConfig(contacts));
  return ContactWrenchConeBasisMatrix::Create(coneConfig, footprint,
                                              contacts.basis_generator_set);
}

absl::StatusOr<std::vector<ContactWrenchConeBasisMatrix>>
contactWrenchConeBasesFromConfig(const mpc_config::ContactsConfig& contacts,
                                 const ModelSettings& modelSettings) {
  std::vector<ContactWrenchConeBasisMatrix> bases;
  bases.reserve(modelSettings.contactNames.size());
  for (size_t i = 0; i < modelSettings.contactNames.size(); ++i) {
    // The generator sets span a 6-DoF contact's wrench cone over its
    // footprint. A point contact has a force cone only, and no generator set
    // for it exists yet.
    if (modelSettings.isPointContact(i)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "contact ", i, " (", modelSettings.contactNames[i],
          ") is a 3-DoF contact, and ", kContactInputParameterizationKey, " '",
          kBasisVectorsContactInputParameterization,
          "' has generator sets for 6-DoF contacts only; select '",
          kWrenchContactInputParameterizationName,
          "', or list the contact in model_settings.contact_names_6dof."));
    }
    ASSIGN_OR_RETURN(const ContactRectangle footprint,
                     contactRectangleFromConfig(contacts, modelSettings,
                                                static_cast<int>(i)));
    ASSIGN_OR_RETURN(ContactWrenchConeBasisMatrix basis,
                     contactWrenchConeBasisFromConfig(contacts, footprint));
    bases.push_back(std::move(basis));
  }
  return bases;
}

absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config>
basisNonNegativityBarrierFromConfig(
    const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::BasisBarrier& barrier =
      contacts.basis_non_negativity_barrier;
  constexpr absl::string_view kBlock = "contacts.basis_non_negativity_barrier";
  // A non-positive mu rewards a negative basis coefficient, and the polynomial
  // barrier is discontinuous for a non-positive delta.
  RETURN_IF_ERROR(checkPositive(kBlock, "mu", barrier.mu));
  RETURN_IF_ERROR(checkPositive(kBlock, "delta", barrier.delta));
  return PieceWisePolynomialBarrierPenalty::Config(barrier.mu, barrier.delta);
}

absl::StatusOr<BasisRegularizationSettings> basisRegularizationFromConfig(
    const mpc_config::ContactsConfig& contacts) {
  RETURN_IF_ERROR(checkFinite("contacts", "basis_scaling_regularization",
                              contacts.basis_scaling_regularization));
  RETURN_IF_ERROR(
      getBasisRegularizationBuilder(contacts.basis_regularization).status());
  BasisRegularizationSettings settings;
  settings.basisScalingRegularization = contacts.basis_scaling_regularization;
  settings.basisRegularization = contacts.basis_regularization;
  return settings;
}

absl::StatusOr<FrictionForceConeConstraint::Config>
frictionForceConeConfigFromConfig(const mpc_config::ContactsConfig& contacts) {
  const double frictionCoefficient =
      contacts.friction_force_cone_soft_constraint.friction_coefficient;
  if (!std::isfinite(frictionCoefficient) || frictionCoefficient <= 0.0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "contacts.friction_force_cone_soft_constraint.friction_coefficient is ",
        frictionCoefficient, ", but it must be finite and positive."));
  }
  return FrictionForceConeConstraint::Config(frictionCoefficient);
}

absl::StatusOr<RelaxedBarrierPenalty::Config>
frictionForceConeBarrierFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::FrictionCone& cone =
      contacts.friction_force_cone_soft_constraint;
  return relaxedBarrier("contacts.friction_force_cone_soft_constraint", cone.mu,
                        cone.delta);
}

absl::StatusOr<RelaxedBarrierPenalty::Config> contactMomentXyBarrierFromConfig(
    const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::ContactMoment& barrierConfig =
      contacts.contact_moment_xy_soft_constraint;
  return relaxedBarrier("contacts.contact_moment_xy_soft_constraint",
                        barrierConfig.mu, barrierConfig.delta);
}

}  // namespace ocs2::legged
