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

// The conversions of the contacts block (ContactsFromConfig.h): the schema's defaults are the domain's and name
// registered components, the required ground of the wrench cone is refused by the field it lacks, and the geometry and
// the bases come out of the values given.

#include "pinocchio/fwd.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"

#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_mpc_config/contacts_config.nproto.h"

namespace ocs2::humanoid {
namespace {

using Cone = mpc_config::ContactsConfig::WrenchCone;

// The ground of the wrench cone is required, so its fields keep their presence; so do the cone's mu and delta, whose
// absence the contact-implicit check of the basis barrier tells apart from a value.
static_assert(std::is_same_v<decltype(Cone::friction_coefficient), std::optional<double>>);
static_assert(std::is_same_v<decltype(Cone::num_basis_vectors), std::optional<int32_t>>);
static_assert(std::is_same_v<decltype(Cone::mu), std::optional<double>>);
static_assert(std::is_same_v<decltype(mpc_config::ContactsConfig::basis_generator_set), std::string>);

/** A wrench cone block with every required field. */
mpc_config::ContactsConfig completeCone() {
  mpc_config::ContactsConfig contacts;
  contacts.contact_wrench_cone_soft_constraint.friction_coefficient = 0.5;
  contacts.contact_wrench_cone_soft_constraint.torsional_friction_coefficient = 0.05;
  contacts.contact_wrench_cone_soft_constraint.min_normal_force = 5.0;
  contacts.contact_wrench_cone_soft_constraint.gripper_force = 0.0;
  contacts.contact_wrench_cone_soft_constraint.num_basis_vectors = 4;
  contacts.contact_rectangle.x_min = -0.12;
  contacts.contact_rectangle.x_max = 0.12;
  contacts.contact_rectangle.y_min = -0.055;
  contacts.contact_rectangle.y_max = 0.055;
  return contacts;
}

/** The center point of a left foot. */
ContactCenterPoint centerPoint() {
  return ContactCenterPoint("foot_l_contact", "l_leg_akx", vector3_t(0.04, 0.0, -0.08));
}

TEST(ContactsFromConfigTest, TheSchemaDefaultsAreTheDomainDefaults) {
  const mpc_config::ContactsConfig contacts;
  EXPECT_EQ(contacts.basis_generator_set, kDefaultBasisGeneratorSet);
  EXPECT_EQ(contacts.basis_regularization, kDefaultBasisRegularization);
  // The names the defaults give are registered ones.
  EXPECT_TRUE(getBasisGeneratorSetBuilder(contacts.basis_generator_set).ok());
  EXPECT_TRUE(basisRegularizationFromConfig(contacts).ok());
  // An absent barrier is RelaxedBarrierPenalty's and PieceWisePolynomialBarrierPenalty's own.
  const RelaxedBarrierPenalty::Config relaxed;
  const absl::StatusOr<RelaxedBarrierPenalty::Config> wrenchCone = contactWrenchConeBarrierFromConfig(contacts);
  ASSERT_TRUE(wrenchCone.ok()) << wrenchCone.status();
  EXPECT_EQ(wrenchCone->mu, relaxed.mu);
  EXPECT_EQ(wrenchCone->delta, relaxed.delta);
  EXPECT_EQ(contactMomentXyBarrierFromConfig(contacts)->mu, relaxed.mu);
  EXPECT_EQ(contactMomentXyBarrierFromConfig(contacts)->delta, relaxed.delta);
  EXPECT_EQ(frictionForceConeBarrierFromConfig(contacts)->mu, relaxed.mu);
  EXPECT_EQ(frictionForceConeBarrierFromConfig(contacts)->delta, relaxed.delta);
  EXPECT_EQ(contacts.contact_rectangle.scale_factor, 1.0);
}

TEST(ContactsFromConfigTest, TheRectangleIsScaledAroundTheCenterPoint) {
  mpc_config::ContactsConfig contacts = completeCone();
  contacts.contact_rectangle.scale_factor = 0.5;
  const absl::StatusOr<ContactRectangle> rectangle = contactRectangleFromConfig(contacts, centerPoint());
  ASSERT_TRUE(rectangle.ok()) << rectangle.status();
  EXPECT_EQ(rectangle->getBounds().x_max, 0.06);
  EXPECT_EQ(rectangle->getBounds().y_min, -0.0275);
  EXPECT_EQ(rectangle->getContactCenterPoint().parentJointName, "l_leg_akx");
  EXPECT_EQ(rectangle->getNumberOfContactPoints(), 4U);

  contacts.contact_rectangle.y_max = std::numeric_limits<double>::infinity();
  const absl::StatusOr<ContactRectangle> refused = contactRectangleFromConfig(contacts, centerPoint());
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "contacts.contact_rectangle.y_max")) << refused.status();
}

TEST(ContactsFromConfigTest, AnEmptyOrInvertedRectangleOrANonPositiveScaleIsRefused) {
  using Rectangle = mpc_config::ContactsConfig::Rectangle;
  const std::vector<std::function<void(Rectangle&)>> refusedRectangles = {
      [](Rectangle& r) { r.x_max = r.x_min; },
      [](Rectangle& r) { std::swap(r.x_min, r.x_max); },
      [](Rectangle& r) { r.y_min = r.y_max; },
      [](Rectangle& r) { std::swap(r.y_min, r.y_max); },
      [](Rectangle& r) { r.scale_factor = 0.0; },
      [](Rectangle& r) { r.scale_factor = -1.0; },
      [](Rectangle& r) { r.scale_factor = std::numeric_limits<double>::quiet_NaN(); },
  };
  for (const std::function<void(Rectangle&)>& corrupt : refusedRectangles) {
    mpc_config::ContactsConfig contacts = completeCone();
    corrupt(contacts.contact_rectangle);
    const absl::Status status = contactRectangleFromConfig(contacts, centerPoint()).status();
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(absl::StrContains(status.message(), "contacts.contact_rectangle")) << status;
  }
}

TEST(ContactsFromConfigTest, TheBasisBarrierRefusesANonPositiveOrNanMuOrDelta) {
  for (const double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    mpc_config::ContactsConfig mu;
    mu.basis_non_negativity_barrier.mu = bad;
    EXPECT_TRUE(
        absl::StrContains(basisNonNegativityBarrierFromConfig(mu).status().message(), "contacts.basis_non_negativity_barrier.mu is"))
        << bad;
    mpc_config::ContactsConfig delta;
    delta.basis_non_negativity_barrier.delta = bad;
    EXPECT_TRUE(
        absl::StrContains(basisNonNegativityBarrierFromConfig(delta).status().message(), "contacts.basis_non_negativity_barrier.delta is"))
        << bad;
  }
}

TEST(ContactsFromConfigTest, TheWrenchConeNeedsEveryGroundField) {
  const absl::StatusOr<ContactWrenchConeConstraint::Config> config = contactWrenchConeConfigFromConfig(completeCone());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->numBasisVectors, 4U);
  EXPECT_EQ(config->frictionCoefficient, 0.5);
  EXPECT_TRUE(config->patchOffset.isZero());

  for (const absl::string_view field :
       {"friction_coefficient", "torsional_friction_coefficient", "min_normal_force", "gripper_force", "num_basis_vectors"}) {
    mpc_config::ContactsConfig contacts = completeCone();
    Cone& cone = contacts.contact_wrench_cone_soft_constraint;
    if (field == "friction_coefficient") cone.friction_coefficient.reset();
    if (field == "torsional_friction_coefficient") cone.torsional_friction_coefficient.reset();
    if (field == "min_normal_force") cone.min_normal_force.reset();
    if (field == "gripper_force") cone.gripper_force.reset();
    if (field == "num_basis_vectors") cone.num_basis_vectors.reset();
    const absl::StatusOr<ContactWrenchConeConstraint::Config> refused = contactWrenchConeConfigFromConfig(contacts);
    EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument) << field;
    EXPECT_TRUE(
        absl::StrContains(refused.status().message(), absl::StrCat("contacts.contact_wrench_cone_soft_constraint.", field, " is missing")))
        << refused.status();
  }
}

TEST(ContactsFromConfigTest, TheWrenchConeRefusesWhatItsConstraintRefuses) {
  mpc_config::ContactsConfig contacts = completeCone();
  contacts.contact_wrench_cone_soft_constraint.num_basis_vectors = -4;
  EXPECT_TRUE(absl::StrContains(contactWrenchConeConfigFromConfig(contacts).status().message(), "num_basis_vectors is -4"));
  contacts = completeCone();
  contacts.contact_wrench_cone_soft_constraint.friction_coefficient = 0.0;
  EXPECT_EQ(contactWrenchConeConfigFromConfig(contacts).status().code(), absl::StatusCode::kInvalidArgument);
  contacts = completeCone();
  contacts.contact_wrench_cone_soft_constraint.mu = 0.0;
  EXPECT_TRUE(absl::StrContains(contactWrenchConeBarrierFromConfig(contacts).status().message(),
                                "contacts.contact_wrench_cone_soft_constraint.mu is 0"));
  contacts.contact_wrench_cone_soft_constraint.mu = 0.2;
  contacts.contact_wrench_cone_soft_constraint.delta = 5.0;
  const absl::StatusOr<RelaxedBarrierPenalty::Config> barrier = contactWrenchConeBarrierFromConfig(contacts);
  ASSERT_TRUE(barrier.ok()) << barrier.status();
  EXPECT_EQ(barrier->mu, 0.2);
  EXPECT_EQ(barrier->delta, 5.0);
}

TEST(ContactsFromConfigTest, TheBasisIsTheNamedGeneratorSet) {
  mpc_config::ContactsConfig contacts = completeCone();
  const absl::StatusOr<ContactRectangle> footprint = contactRectangleFromConfig(contacts, centerPoint());
  ASSERT_TRUE(footprint.ok()) << footprint.status();
  const absl::StatusOr<ContactWrenchConeBasisMatrix> inner = contactWrenchConeBasisFromConfig(contacts, *footprint);
  ASSERT_TRUE(inner.ok()) << inner.status();
  EXPECT_EQ(inner->generatorSet(), kDefaultBasisGeneratorSet);
  EXPECT_EQ(inner->numBasis(), 4U + 7U);
  contacts.basis_generator_set = std::string(kExactWrenchConeGeneratorSet);
  const absl::StatusOr<ContactWrenchConeBasisMatrix> exact = contactWrenchConeBasisFromConfig(contacts, *footprint);
  ASSERT_TRUE(exact.ok()) << exact.status();
  // The exact wrench cone (Caron et al.): the N edges of the friction pyramid at each of the 4 corners of the sole.
  const std::optional<int32_t>& numBasisVectors = contacts.contact_wrench_cone_soft_constraint.num_basis_vectors;
  if (!numBasisVectors.has_value()) {
    FAIL() << "completeCone() sets num_basis_vectors";
  }
  EXPECT_EQ(exact->numBasis(), 4U * static_cast<size_t>(*numBasisVectors));
  contacts.basis_generator_set = "no_such_set";
  EXPECT_EQ(contactWrenchConeBasisFromConfig(contacts, *footprint).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ContactsFromConfigTest, TheBasisBarrierAndRegularizationCarryTheirValues) {
  mpc_config::ContactsConfig contacts;
  contacts.basis_non_negativity_barrier.mu = 0.02;
  contacts.basis_scaling_regularization = 1.0e-3;
  contacts.basis_regularization = std::string(kNullSpaceBasisRegularization);
  const absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> barrier = basisNonNegativityBarrierFromConfig(contacts);
  ASSERT_TRUE(barrier.ok()) << barrier.status();
  EXPECT_EQ(barrier->mu, 0.02);
  const absl::StatusOr<BasisRegularizationSettings> regularization = basisRegularizationFromConfig(contacts);
  ASSERT_TRUE(regularization.ok()) << regularization.status();
  EXPECT_EQ(regularization->basisScalingRegularization, 1.0e-3);
  EXPECT_EQ(regularization->basisRegularization, kNullSpaceBasisRegularization);
  contacts.basis_regularization = "no_such_shape";
  EXPECT_EQ(basisRegularizationFromConfig(contacts).status().code(), absl::StatusCode::kInvalidArgument);
  contacts.basis_regularization = std::string(kDefaultBasisRegularization);
  contacts.basis_scaling_regularization = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(absl::StrContains(basisRegularizationFromConfig(contacts).status().message(), "contacts.basis_scaling_regularization"));
}

TEST(ContactsFromConfigTest, TheFrictionConeRefusesANonPositiveCoefficientInsteadOfAbortingOnIt) {
  mpc_config::ContactsConfig contacts;
  contacts.friction_force_cone_soft_constraint.friction_coefficient = 0.7;
  const absl::StatusOr<FrictionForceConeConstraint::Config> config = frictionForceConeConfigFromConfig(contacts);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->frictionCoefficient, 0.7);
  // The constraint's own regularization and offsets, which the file does not set.
  const FrictionForceConeConstraint::Config defaults;
  EXPECT_EQ(config->regularization, defaults.regularization);
  EXPECT_EQ(config->gripperForce, defaults.gripperForce);
  contacts.friction_force_cone_soft_constraint.friction_coefficient = -0.5;
  EXPECT_TRUE(absl::StrContains(frictionForceConeConfigFromConfig(contacts).status().message(),
                                "contacts.friction_force_cone_soft_constraint.friction_coefficient is -0.5"));
}

TEST(ContactsFromConfigTest, TheFrictionConeAndTheContactMomentRefuseANonPositiveBarrier) {
  // The relaxed barrier divides by delta and takes its log below it: at delta = 0 a stance node at zero force is NaN or
  // inf, and a negative mu rewards violating the constraint. Both conversions feed start-up and a reload alike.
  struct Barrier {
    double mu = 0.0;
    double delta = 0.0;
    std::string refusal;
  };
  const std::vector<Barrier> refused = {{.mu = 0.0, .delta = 5.0, .refusal = "mu is 0"},
                                        {.mu = -1.0, .delta = 5.0, .refusal = "mu is -1"},
                                        {.mu = 0.2, .delta = 0.0, .refusal = "delta is 0"},
                                        {.mu = 0.2, .delta = -0.1, .refusal = "delta is -0.1"},
                                        {.mu = 0.2, .delta = std::numeric_limits<double>::infinity(), .refusal = "delta is inf"}};
  for (const Barrier& barrier : refused) {
    mpc_config::ContactsConfig contacts;
    contacts.friction_force_cone_soft_constraint.mu = barrier.mu;
    contacts.friction_force_cone_soft_constraint.delta = barrier.delta;
    contacts.contact_moment_xy_soft_constraint.mu = barrier.mu;
    contacts.contact_moment_xy_soft_constraint.delta = barrier.delta;
    const absl::Status friction = frictionForceConeBarrierFromConfig(contacts).status();
    EXPECT_EQ(friction.code(), absl::StatusCode::kInvalidArgument) << barrier.refusal;
    EXPECT_TRUE(absl::StrContains(friction.message(), absl::StrCat("contacts.friction_force_cone_soft_constraint.", barrier.refusal)))
        << friction;
    const absl::Status moment = contactMomentXyBarrierFromConfig(contacts).status();
    EXPECT_EQ(moment.code(), absl::StatusCode::kInvalidArgument) << barrier.refusal;
    EXPECT_TRUE(absl::StrContains(moment.message(), absl::StrCat("contacts.contact_moment_xy_soft_constraint.", barrier.refusal)))
        << moment;
  }
  mpc_config::ContactsConfig contacts;
  contacts.friction_force_cone_soft_constraint.mu = 0.2;
  contacts.friction_force_cone_soft_constraint.delta = 5.0;
  const absl::StatusOr<RelaxedBarrierPenalty::Config> accepted = frictionForceConeBarrierFromConfig(contacts);
  ASSERT_TRUE(accepted.ok()) << accepted.status();
  EXPECT_EQ(accepted->mu, 0.2);
  EXPECT_EQ(accepted->delta, 5.0);
}

}  // namespace
}  // namespace ocs2::humanoid
