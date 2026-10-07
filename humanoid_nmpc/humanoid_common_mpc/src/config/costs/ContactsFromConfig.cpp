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

#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_mpc_config/contacts_config.nproto.h"

namespace ocs2::humanoid {
namespace {

constexpr absl::string_view kWrenchCone = "contacts.contact_wrench_cone_soft_constraint";

/** InvalidArgument naming `<block>.<field>` unless `value` is finite. */
absl::Status checkFinite(absl::string_view block, absl::string_view field, double value) {
  if (std::isfinite(value)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(block, ".", field, " is ", value, ", but it must be finite."));
}

/** InvalidArgument naming `<block>.<field>` unless `value` is finite and positive. */
absl::Status checkPositive(absl::string_view block, absl::string_view field, double value) {
  if (std::isfinite(value) && value > 0.0) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(block, ".", field, " is ", value,
                                                 ", but it must be finite and positive: a negative barrier weight rewards violating the "
                                                 "constraint, and the relaxed barrier is undefined at a non-positive delta."));
}

/** The required field `field` of the wrench cone block, or InvalidArgument saying that every consumer needs it. */
template <typename T>
absl::StatusOr<T> requiredConeValue(std::optional<T> value, absl::string_view field) {
  if (value.has_value()) {
    return *value;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      kWrenchCone, ".", field,
      " is missing. The block is the ground of the whole-body constraints: the wrench cone, the generators of the basis-vector "
      "contact inputs and the online contact planner all read it, and none of them falls back to a library default. Give it "
      "friction_coefficient, torsional_friction_coefficient, min_normal_force, gripper_force and num_basis_vectors."));
}

/** The relaxed barrier `block` {mu delta}, both finite. */
absl::StatusOr<RelaxedBarrierPenalty::Config> relaxedBarrier(absl::string_view block, double mu, double delta) {
  RETURN_IF_ERROR(checkFinite(block, "mu", mu));
  RETURN_IF_ERROR(checkFinite(block, "delta", delta));
  return RelaxedBarrierPenalty::Config(mu, delta);
}

}  // namespace

absl::StatusOr<ContactCenterPoint> contactCenterPointFromConfig(const mpc_config::ContactsConfig& contacts,
                                                                const ModelSettings& modelSettings,
                                                                int contactIndex) {
  if (contactIndex < 0 || static_cast<size_t>(contactIndex) >= modelSettings.contactNames.size() ||
      static_cast<size_t>(contactIndex) >= modelSettings.contactParentJointNames.size()) {
    return absl::InvalidArgumentError(absl::StrCat("contact ", contactIndex, " is not one of the ", modelSettings.contactNames.size(),
                                                   " contacts of the model settings"));
  }
  const mpc_config::Xyz& translation = contacts.contact_frame_translation;
  constexpr absl::string_view kBlock = "contacts.contact_frame_translation";
  RETURN_IF_ERROR(checkFinite(kBlock, "x", translation.x));
  RETURN_IF_ERROR(checkFinite(kBlock, "y", translation.y));
  RETURN_IF_ERROR(checkFinite(kBlock, "z", translation.z));
  const size_t index = static_cast<size_t>(contactIndex);
  return ContactCenterPoint(modelSettings.contactNames[index], modelSettings.contactParentJointNames[index],
                            vector3_t(translation.x, translation.y, translation.z));
}

absl::StatusOr<ContactRectangle> contactRectangleFromConfig(const mpc_config::ContactsConfig& contacts,
                                                            const ContactCenterPoint& centerPoint) {
  const mpc_config::ContactsConfig::Rectangle& rectangle = contacts.contact_rectangle;
  constexpr absl::string_view kBlock = "contacts.contact_rectangle";
  RETURN_IF_ERROR(checkFinite(kBlock, "x_min", rectangle.x_min));
  RETURN_IF_ERROR(checkFinite(kBlock, "x_max", rectangle.x_max));
  RETURN_IF_ERROR(checkFinite(kBlock, "y_min", rectangle.y_min));
  RETURN_IF_ERROR(checkFinite(kBlock, "y_max", rectangle.y_max));
  RETURN_IF_ERROR(checkFinite(kBlock, "scale_factor", rectangle.scale_factor));
  return ContactRectangle(PolygonBounds(rectangle.x_min, rectangle.x_max, rectangle.y_min, rectangle.y_max), centerPoint,
                          rectangle.scale_factor);
}

absl::StatusOr<ContactRectangle> contactRectangleFromConfig(const mpc_config::ContactsConfig& contacts,
                                                            const ModelSettings& modelSettings,
                                                            int contactIndex) {
  ASSIGN_OR_RETURN(const ContactCenterPoint centerPoint, contactCenterPointFromConfig(contacts, modelSettings, contactIndex));
  return contactRectangleFromConfig(contacts, centerPoint);
}

absl::StatusOr<ContactWrenchConeConstraint::Config> contactWrenchConeConfigFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::WrenchCone& cone = contacts.contact_wrench_cone_soft_constraint;
  ASSIGN_OR_RETURN(const scalar_t frictionCoefficient, requiredConeValue(cone.friction_coefficient, "friction_coefficient"));
  ASSIGN_OR_RETURN(const scalar_t torsionalFrictionCoefficient,
                   requiredConeValue(cone.torsional_friction_coefficient, "torsional_friction_coefficient"));
  ASSIGN_OR_RETURN(const scalar_t minNormalForce, requiredConeValue(cone.min_normal_force, "min_normal_force"));
  ASSIGN_OR_RETURN(const scalar_t gripperForce, requiredConeValue(cone.gripper_force, "gripper_force"));
  ASSIGN_OR_RETURN(const int32_t numBasisVectors, requiredConeValue(cone.num_basis_vectors, "num_basis_vectors"));
  // Checked before the conversion to size_t, which would turn a negative count into a huge one.
  if (numBasisVectors < 3) {
    return absl::InvalidArgumentError(absl::StrCat(kWrenchCone, ".num_basis_vectors is ", numBasisVectors,
                                                   " but must be at least 3 (the facets of the friction pyramid)."));
  }
  ContactWrenchConeConstraint::Config config(static_cast<size_t>(numBasisVectors), frictionCoefficient, torsionalFrictionCoefficient,
                                             minNormalForce, gripperForce);
  RETURN_IF_ERROR(ContactWrenchConeConstraint::validateConfig(config));
  return config;
}

absl::StatusOr<RelaxedBarrierPenalty::Config> contactWrenchConeBarrierFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::WrenchCone& cone = contacts.contact_wrench_cone_soft_constraint;
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

absl::StatusOr<ContactWrenchConeBasisMatrix> contactWrenchConeBasisFromConfig(const mpc_config::ContactsConfig& contacts,
                                                                              const ContactRectangle& footprint) {
  ASSIGN_OR_RETURN(const ContactWrenchConeConstraint::Config coneConfig, contactWrenchConeConfigFromConfig(contacts));
  return ContactWrenchConeBasisMatrix::Create(coneConfig, footprint, contacts.basis_generator_set);
}

absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> contactWrenchConeBasesFromConfig(const mpc_config::ContactsConfig& contacts,
                                                                                            const ModelSettings& modelSettings) {
  static_assert(kNumContacts == 2, "contactWrenchConeBasesFromConfig builds one basis per foot of a biped");
  ASSIGN_OR_RETURN(const ContactRectangle leftFootprint, contactRectangleFromConfig(contacts, modelSettings, /*contactIndex=*/0));
  ASSIGN_OR_RETURN(const ContactRectangle rightFootprint, contactRectangleFromConfig(contacts, modelSettings, /*contactIndex=*/1));
  ASSIGN_OR_RETURN(ContactWrenchConeBasisMatrix leftBasis, contactWrenchConeBasisFromConfig(contacts, leftFootprint));
  ASSIGN_OR_RETURN(ContactWrenchConeBasisMatrix rightBasis, contactWrenchConeBasisFromConfig(contacts, rightFootprint));
  return feet_array_t<ContactWrenchConeBasisMatrix>{std::move(leftBasis), std::move(rightBasis)};
}

absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> basisNonNegativityBarrierFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::BasisBarrier& barrier = contacts.basis_non_negativity_barrier;
  constexpr absl::string_view kBlock = "contacts.basis_non_negativity_barrier";
  RETURN_IF_ERROR(checkFinite(kBlock, "mu", barrier.mu));
  RETURN_IF_ERROR(checkFinite(kBlock, "delta", barrier.delta));
  return PieceWisePolynomialBarrierPenalty::Config(barrier.mu, barrier.delta);
}

absl::StatusOr<BasisRegularizationSettings> basisRegularizationFromConfig(const mpc_config::ContactsConfig& contacts) {
  RETURN_IF_ERROR(checkFinite("contacts", "basis_scaling_regularization", contacts.basis_scaling_regularization));
  RETURN_IF_ERROR(getBasisRegularizationBuilder(contacts.basis_regularization).status());
  BasisRegularizationSettings settings;
  settings.basisScalingRegularization = contacts.basis_scaling_regularization;
  settings.basisRegularization = contacts.basis_regularization;
  return settings;
}

absl::StatusOr<FrictionForceConeConstraint::Config> frictionForceConeConfigFromConfig(const mpc_config::ContactsConfig& contacts) {
  const double frictionCoefficient = contacts.friction_force_cone_soft_constraint.friction_coefficient;
  if (!std::isfinite(frictionCoefficient) || frictionCoefficient <= 0.0) {
    return absl::InvalidArgumentError(absl::StrCat("contacts.friction_force_cone_soft_constraint.friction_coefficient is ",
                                                   frictionCoefficient, ", but it must be finite and positive."));
  }
  return FrictionForceConeConstraint::Config(frictionCoefficient);
}

absl::StatusOr<RelaxedBarrierPenalty::Config> frictionForceConeBarrierFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::FrictionCone& cone = contacts.friction_force_cone_soft_constraint;
  return relaxedBarrier("contacts.friction_force_cone_soft_constraint", cone.mu, cone.delta);
}

absl::StatusOr<RelaxedBarrierPenalty::Config> contactMomentXyBarrierFromConfig(const mpc_config::ContactsConfig& contacts) {
  const mpc_config::ContactsConfig::ContactMoment& moment = contacts.contact_moment_xy_soft_constraint;
  return relaxedBarrier("contacts.contact_moment_xy_soft_constraint", moment.mu, moment.delta);
}

}  // namespace ocs2::humanoid
