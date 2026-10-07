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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/cost/CentroidalMpcEndEffectorFootCost.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/cost/ICPCost.h"
#include "humanoid_common_mpc/common/CostTermNames.h"

/**
 * The names CentroidalMpcInterface registers its CppAD terms under and the names it builds their CppAD libraries with,
 * pinned with today's strings. Each pair is the same string, defined twice on purpose: the term name is what the
 * parameter updater finds the term by, the library name is the folder of generated code under cppad_code_gen/ (the
 * library suffixes are the classes' own). A rename of a term must never rename a library, which would regenerate it and
 * could change the problem's numbers, so either change fails here. Builds no CppAD model.
 */
namespace ocs2::humanoid {
namespace {

TEST(CostTermAndLibraryNames, TheDcmTerminalCostKeepsItsTermAndLibraryNames) {
  EXPECT_EQ(absl::string_view(DcmTerminalCost::kTermName), "dcmTerminalCost");
  // The library is "dcmTerminalCost_p11" (DcmTerminalCost.cpp appends its parameter count).
  EXPECT_EQ(absl::string_view(DcmTerminalCost::kLibraryName), "dcmTerminalCost");
}

TEST(CostTermAndLibraryNames, TheIcpCostKeepsItsTermAndLibraryNames) {
  EXPECT_EQ(absl::string_view(kIcpCostTerm), "icp_Cost");
  EXPECT_EQ(absl::string_view(ICPCost::kLibraryName), "icp_Cost");
}

TEST(CostTermAndLibraryNames, TheFootCostKeepsItsTermAndLibraryNames) {
  for (const std::string footName : {"foot_l_contact", "foot_r_contact", "left_ankle_roll_link"}) {
    SCOPED_TRACE(footName);
    EXPECT_EQ(taskSpaceKinematicsCostName(footName), absl::StrCat(footName, "_TaskSpaceKinematicsCost"));
    // The library is "<foot>_TaskSpaceKinematicsCost_yawRefHalfAngle" (CentroidalMpcEndEffectorFootCost.cpp appends it).
    EXPECT_EQ(CentroidalMpcEndEffectorFootCost::libraryName(footName), absl::StrCat(footName, "_TaskSpaceKinematicsCost"));
  }
}

TEST(CostTermAndLibraryNames, TheOtherTermsKeepTheNamesTheInterfaceRegistersThemUnder) {
  // Terms whose libraries are named after their frames or the factory's own literals, not after these.
  EXPECT_EQ(absl::string_view(kStateInputQuadraticCostTerm), "stateInputQuadraticCost");
  EXPECT_EQ(absl::string_view(kStateQuadraticCostTerm), "stateQuadraticCost");
  EXPECT_EQ(absl::string_view(kInputQuadraticCostTerm), "inputQuadraticCost");
  EXPECT_EQ(absl::string_view(kTerminalCostTerm), "terminalCost");
  EXPECT_EQ(absl::string_view(kJointLimitsTerm), "jointLimits");
  EXPECT_EQ(absl::string_view(kFootCollisionTerm), "FootCollisionSoftConstraint");
  EXPECT_EQ(externalTorqueCostName("foot_l_contact"), "foot_l_contact_ExternalTorqueQuadraticCost");
  EXPECT_EQ(zeroVelocityTermName("foot_l_contact"), "foot_l_contact_zeroVelocity");
  EXPECT_EQ(basisNonNegativityTermName("foot_l_contact"), "foot_l_contact_basisNonNegativity");
}

}  // namespace
}  // namespace ocs2::humanoid
