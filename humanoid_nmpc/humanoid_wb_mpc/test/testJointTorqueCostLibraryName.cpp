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

#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"
#include "humanoid_wb_mpc/cost/JointTorqueCostCppAd.h"

/*
 * The names of the whole-body MPC's CppAD-taped costs: the name each is added under in the problem's cost collection,
 * which the parameter updater finds it by, and the name its CppAD library is compiled and cached under. The two are
 * defined apart, and pinned here with today's spellings, so that renaming a term fails this test instead of renaming -
 * and so regenerating - a library. JointTorqueCostCppAd tapes computeJointTorques, which now carries the joint-base
 * coupling of the mass matrix that it used to leave out. The robots ship model_settings.recompile_libraries_cpp_ad:
 * false, so a library compiled from the old function would be loaded in place of the new one if it were cached under the
 * same name: the name must differ from the one it had.
 */

namespace ocs2::humanoid {
namespace {

TEST(JointTorqueCostLibraryName, aLibraryTapedFromTheFormerInverseDynamicsIsNeverLoaded) {
  // WBMpcInterface names the cost "jointTorqueCost", which was also the name of its library.
  const std::string name = JointTorqueCostCppAd::libraryName(JointTorqueCostCppAd::kLibraryCostName);
  EXPECT_NE(name, "jointTorqueCost");
  EXPECT_TRUE(absl::StartsWith(name, "jointTorqueCost")) << "the library stays recognizable in the model folder: " << name;
  EXPECT_NE(JointTorqueCostCppAd::libraryName("otherCost"), name) << "two costs never share a library";
}

// LINT.IfChange(term_and_library_names)
TEST(WholeBodyCostNames, TheTermAndLibraryNamesKeepTheirSpellings) {
  EXPECT_EQ(std::string(JointTorqueCostCppAd::kTermName), "jointTorqueCost");
  EXPECT_EQ(JointTorqueCostCppAd::libraryName(JointTorqueCostCppAd::kLibraryCostName), "jointTorqueCost_fullMassMatrix");
  EXPECT_EQ(EndEffectorDynamicsFootCost::termName("foot_l_contact"), "foot_l_contact_TaskSpaceTrackingCost");
  EXPECT_EQ(EndEffectorDynamicsFootCost::termName("foot_r_contact"), "foot_r_contact_TaskSpaceTrackingCost");
  EXPECT_EQ(EndEffectorDynamicsFootCost::libraryModelName("foot_l_contact"), "foot_l_contact_TaskSpaceTrackingCost");
  EXPECT_EQ(EndEffectorDynamicsFootCost::libraryModelName("foot_r_contact"), "foot_r_contact_TaskSpaceTrackingCost");
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/cost/JointTorqueCostCppAd.h:joint_torque_cost_names, //humanoid_nmpc/humanoid_wb_mpc/src/cost/EndEffectorDynamicsFootCost.cpp:foot_cost_names)
// clang-format on

}  // namespace
}  // namespace ocs2::humanoid
