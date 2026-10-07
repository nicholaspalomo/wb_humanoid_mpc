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

#include <string>

#include "absl/status/statusor.h"
#include "ocs2_core/penalties/penalties/PieceWisePolynomialBarrierPenalty.h"
#include "ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_mpc_config/contacts_config.nproto.h"

// The contact geometry and the contact constraints from the task file's contacts block. Every value must be finite; each
// error is an InvalidArgument naming the field of the block.

namespace ocs2::humanoid {

/** The regularization of the basis scalings in the input cost of the basis-vector contact inputs. */
struct BasisRegularizationSettings {
  // The weight `reg` of the regularization (contacts.basis_scaling_regularization).
  scalar_t basisScalingRegularization = 0.0;
  // Its shape, a name of the regularization registry (BasisInputsCostTransform.h).
  std::string basisRegularization;
};

/**
 * The center point of contact `contactIndex` of `modelSettings`, offset by contacts.contact_frame_translation.
 *
 * @return InvalidArgument when `contactIndex` is not a contact of `modelSettings`, or naming a translation that is not
 *         finite.
 */
absl::StatusOr<ContactCenterPoint> contactCenterPointFromConfig(const mpc_config::ContactsConfig& contacts,
                                                                const ModelSettings& modelSettings,
                                                                int contactIndex);

/**
 * The sole of a contact, contacts.contact_rectangle around `centerPoint` (0 for a bound the block leaves out, scale
 * factor 1).
 */
absl::StatusOr<ContactRectangle> contactRectangleFromConfig(const mpc_config::ContactsConfig& contacts,
                                                            const ContactCenterPoint& centerPoint);

/**
 * The sole of contact `contactIndex` of `modelSettings`, around its center point (contactCenterPointFromConfig()).
 */
absl::StatusOr<ContactRectangle> contactRectangleFromConfig(const mpc_config::ContactsConfig& contacts,
                                                            const ModelSettings& modelSettings,
                                                            int contactIndex);

/**
 * The ground of the whole-body contact constraints, contacts.contact_wrench_cone_soft_constraint. That one block is the
 * ground of the wrench cone, the generators of the basis-vector contact inputs and the online contact planner, and none
 * of them falls back to a library default: its five geometry fields are required.
 *
 * @return InvalidArgument naming a field that is missing, a num_basis_vectors below 3, or what
 *         ContactWrenchConeConstraint::validateConfig() refuses.
 */
absl::StatusOr<ContactWrenchConeConstraint::Config> contactWrenchConeConfigFromConfig(const mpc_config::ContactsConfig& contacts);

/**
 * The relaxed barrier of the soft constraint contact_wrench_cone: the mu and delta of
 * contacts.contact_wrench_cone_soft_constraint, RelaxedBarrierPenalty::Config's own for the ones the block leaves out.
 *
 * @return InvalidArgument naming a mu or delta that is not finite and positive.
 */
absl::StatusOr<RelaxedBarrierPenalty::Config> contactWrenchConeBarrierFromConfig(const mpc_config::ContactsConfig& contacts);

/**
 * The wrench-cone basis of one contact of the basis-vector contact inputs: the generator set contacts.basis_generator_set
 * built from contactWrenchConeConfigFromConfig() and `footprint` (ContactWrenchConeBasisMatrix::Create()).
 */
absl::StatusOr<ContactWrenchConeBasisMatrix> contactWrenchConeBasisFromConfig(const mpc_config::ContactsConfig& contacts,
                                                                              const ContactRectangle& footprint);

/**
 * The bases of both contacts of `modelSettings`, each around its own sole (contactRectangleFromConfig()).
 */
absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> contactWrenchConeBasesFromConfig(const mpc_config::ContactsConfig& contacts,
                                                                                            const ModelSettings& modelSettings);

/** The barrier that keeps the basis scalings non-negative, contacts.basis_non_negativity_barrier. */
absl::StatusOr<PieceWisePolynomialBarrierPenalty::Config> basisNonNegativityBarrierFromConfig(const mpc_config::ContactsConfig& contacts);

/**
 * The regularization of the basis scalings, contacts.basis_scaling_regularization and contacts.basis_regularization.
 *
 * @return InvalidArgument for a weight that is not finite, and the registry's for a shape it does not know.
 */
absl::StatusOr<BasisRegularizationSettings> basisRegularizationFromConfig(const mpc_config::ContactsConfig& contacts);

/**
 * The friction cone of the soft constraint friction_force_cone, from contacts.friction_force_cone_soft_constraint: its
 * friction coefficient, with FrictionForceConeConstraint::Config's own regularization and offsets.
 *
 * @return InvalidArgument for a friction coefficient that is not finite and positive, which the constraint's Config
 *         would refuse with a CHECK failure.
 */
absl::StatusOr<FrictionForceConeConstraint::Config> frictionForceConeConfigFromConfig(const mpc_config::ContactsConfig& contacts);

/** The relaxed barrier of friction_force_cone, the mu and delta of contacts.friction_force_cone_soft_constraint. */
absl::StatusOr<RelaxedBarrierPenalty::Config> frictionForceConeBarrierFromConfig(const mpc_config::ContactsConfig& contacts);

/** The relaxed barrier of contact_moment_xy, the mu and delta of contacts.contact_moment_xy_soft_constraint. */
absl::StatusOr<RelaxedBarrierPenalty::Config> contactMomentXyBarrierFromConfig(const mpc_config::ContactsConfig& contacts);

}  // namespace ocs2::humanoid
