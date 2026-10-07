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

#include <string>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"

/**
 * The collection names the MPC interfaces register their terms under and the parameter updater's appliers find them by
 * (CostTermNames.h, ContactTermNames.h), pinned with the strings the shipped problems use. A term's name is also what
 * the solver's metrics and the visualization label it with, so a rename is a change a reviewer must see: it fails here.
 * The CppAD library names that used to be derived from these are defined apart, beside their classes, and pinned by the
 * formulations' own tests (humanoid_centroidal_mpc:testCostTermAndLibraryNames, humanoid_wb_mpc's
 * testJointTorqueCostLibraryName).
 */
namespace ocs2::humanoid {
namespace {

// LINT.IfChange(pinned_cost_term_names)
TEST(CostTermNames, TheQuadraticAndStateSoftConstraintTermsKeepTheirNames) {
  EXPECT_EQ(absl::string_view(kStateInputQuadraticCostTerm), "stateInputQuadraticCost");
  EXPECT_EQ(absl::string_view(kStateQuadraticCostTerm), "stateQuadraticCost");
  EXPECT_EQ(absl::string_view(kInputQuadraticCostTerm), "inputQuadraticCost");
  EXPECT_EQ(absl::string_view(kTerminalCostTerm), "terminalCost");
  EXPECT_EQ(absl::string_view(kJointLimitsTerm), "jointLimits");
  EXPECT_EQ(absl::string_view(kFootCollisionTerm), "FootCollisionSoftConstraint");
  EXPECT_EQ(absl::string_view(kIcpCostTerm), "icp_Cost");
}

TEST(CostTermNames, ThePerLinkAndPerFootTermsAreTheNameAndTheirSuffix) {
  EXPECT_EQ(taskSpaceKinematicsCostName("foot_l_contact"), "foot_l_contact_TaskSpaceKinematicsCost");
  EXPECT_EQ(taskSpaceKinematicsCostName("torso"), "torso_TaskSpaceKinematicsCost");
  EXPECT_EQ(externalTorqueCostName("foot_r_contact"), "foot_r_contact_ExternalTorqueQuadraticCost");
  EXPECT_EQ(absl::string_view(kZeroVelocityTermSuffix), "_zeroVelocity");
  EXPECT_EQ(zeroVelocityTermName("foot_l_contact"), "foot_l_contact_zeroVelocity");
  EXPECT_EQ(basisNonNegativityTermName("foot_l_contact"), "foot_l_contact_basisNonNegativity");
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/CostTermNames.h:cost_term_names)

TEST(CostTermNames, TheContactTermsKeepTheirSuffixes) {
  EXPECT_EQ(contact_term::name("foot_l_contact", contact_term::kContactWrenchCone), "foot_l_contact_contactWrenchCone");
  EXPECT_EQ(contact_term::name("foot_l_contact", contact_term::kFrictionForceCone), "foot_l_contact_frictionForceCone");
  EXPECT_EQ(contact_term::name("foot_l_contact", contact_term::kContactMomentXY), "foot_l_contact_contactMomentXY");
  EXPECT_EQ(contact_term::name("foot_l_contact", contact_term::kContactComplementarity), "foot_l_contact_contactComplementarity");
  EXPECT_EQ(contact_term::name("foot_l_contact", contact_term::kForceWeightedSlip), "foot_l_contact_forceWeightedSlip");
  EXPECT_EQ(contact_term::name("foot_l_contact", contact_term::kGroundPenetration), "foot_l_contact_groundPenetration");
  EXPECT_EQ(contact_term::name("foot_l_contact", contact_term::kNormalVelocitySoft), "foot_l_contact_normalVelocitySoft");
}

}  // namespace
}  // namespace ocs2::humanoid
