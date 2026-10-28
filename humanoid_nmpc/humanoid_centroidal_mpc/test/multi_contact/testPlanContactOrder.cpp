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
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/multi_contact/PlanContactOrder.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/MultiContactTestFiles.h"

/*
 * The map between a plan's contact modes, numbered in the contact set's order
 * (bit i for candidate i), and the MPC's mode numbers, numbered in the MPC's
 * contact order (3-DoF contacts first, the first contact the most significant
 * bit): built by name, never by bit position.
 */
namespace ocs2::humanoid {
namespace {

/** The multi-contact task file's contacts, in the MPC's order. */
std::vector<std::string> mpcContactNames() {
  const CentroidalRobotFiles files = atlasMultiContactFiles();
  const mpc_config::TaskFile task = checkedValue(loadTaskFile(files.taskFile));
  return checkedValue(ModelSettings::Create(task, files.urdfFile,
                                            "plan_contact_order_",
                                            /*verbose=*/false))
      .contactNames;
}

TEST(PlanContactOrder, mapsEveryCandidateByNameNotByBit) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactSet contactSet,
                                loadAtlasContactSet());
  const std::vector<std::string> names = mpcContactNames();
  HUMANOID_ASSERT_OK_AND_ASSIGN(const PlanContactOrder order,
                                PlanContactOrder::Create(contactSet, names));
  ASSERT_EQ(order.numMpcContacts(), names.size());
  for (int k = 0; k < contactSet.size(); ++k) {
    const std::optional<size_t> mpc = order.mpcIndexOf(static_cast<size_t>(k));
    ASSERT_TRUE(mpc.has_value()) << contactSet.candidate(k).name;
    EXPECT_EQ(names[*mpc], contactSet.candidate(k).name);
    EXPECT_EQ(order.planIndexOf(*mpc), static_cast<size_t>(k));
  }
  // The candidates' order is not the MPC's: the feet are the contact set's
  // first two and the MPC's last two.
  EXPECT_EQ(contactSet.candidate(0).name, "foot_l_contact");
  EXPECT_NE(names.front(), "foot_l_contact");
}

TEST(PlanContactOrder, theFeetOfAPlanAreTheBipedsStance) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactSet contactSet,
                                loadAtlasContactSet());
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const PlanContactOrder order,
      PlanContactOrder::Create(contactSet, mpcContactNames()));
  const std::vector<std::string> feet = {"foot_l_contact", "foot_r_contact"};
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactMask bothFeet,
                                contactSet.maskOf(feet));
  HUMANOID_ASSERT_OK_AND_ASSIGN(const size_t stance, order.modeOf(bothFeet));
  EXPECT_EQ(stance, static_cast<size_t>(ModeNumber::kStance));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::ContactMask leftFoot,
      contactSet.maskOf(std::vector<std::string>{"foot_l_contact"}));
  HUMANOID_ASSERT_OK_AND_ASSIGN(const size_t left, order.modeOf(leftFoot));
  EXPECT_EQ(left, static_cast<size_t>(ModeNumber::kLf));
  // A knee, the plan's candidate 2, is the MPC's first contact: the most
  // significant bit of a mode.
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::ContactMask knee,
      contactSet.maskOf(std::vector<std::string>{"knee_l_contact"}));
  HUMANOID_ASSERT_OK_AND_ASSIGN(const size_t kneeMode, order.modeOf(knee));
  EXPECT_EQ(kneeMode,
            contactModeBit(/*contactIndex=*/0, order.numMpcContacts()));
}

TEST(PlanContactOrder, everyModeRoundTrips) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactSet contactSet,
                                loadAtlasContactSet());
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const PlanContactOrder order,
      PlanContactOrder::Create(contactSet, mpcContactNames()));
  const multi_contact::ContactMask numMasks = multi_contact::ContactMask{1}
                                              << contactSet.size();
  for (multi_contact::ContactMask mask = 0; mask < numMasks; ++mask) {
    const absl::StatusOr<size_t> mode = order.modeOf(mask);
    ASSERT_TRUE(mode.ok()) << mode.status();
    const absl::StatusOr<multi_contact::ContactMask> back = order.maskOf(*mode);
    ASSERT_TRUE(back.ok()) << back.status();
    ASSERT_EQ(*back, mask);
    // Contact by contact, by name.
    for (int k = 0; k < contactSet.size(); ++k) {
      const size_t mpc = *order.mpcIndexOf(static_cast<size_t>(k));
      ASSERT_EQ(contactIsActiveInMode(*mode, mpc, order.numMpcContacts()),
                multi_contact::isContactActive(mask, k));
    }
  }
}

TEST(PlanContactOrder, refusesAContactTheOtherSideDoesNotHave) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactSet contactSet,
                                loadAtlasContactSet());
  // The MPC without the seat: a plan that sits is refused, one that stands is
  // not.
  std::vector<std::string> names = mpcContactNames();
  names.erase(std::find(names.begin(), names.end(), "seat_contact"));
  HUMANOID_ASSERT_OK_AND_ASSIGN(const PlanContactOrder order,
                                PlanContactOrder::Create(contactSet, names));
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      const multi_contact::ContactMask seated,
      contactSet.maskOf(std::vector<std::string>{"seat_contact"}));
  const absl::StatusOr<size_t> refused = order.modeOf(seated);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "seat_contact"))
      << refused.status();
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactMask standing,
                                contactSet.maskOf(std::vector<std::string>{
                                    "foot_l_contact", "foot_r_contact"}));
  EXPECT_TRUE(order.modeOf(standing).ok());
  // An MPC contact the plan's set does not have is never in a plan's mode.
  std::vector<std::string> extended = mpcContactNames();
  extended.insert(extended.begin(), "elbow_l_contact");
  HUMANOID_ASSERT_OK_AND_ASSIGN(const PlanContactOrder wider,
                                PlanContactOrder::Create(contactSet, extended));
  EXPECT_FALSE(wider.planIndexOf(0).has_value());
  const absl::StatusOr<multi_contact::ContactMask> unknown =
      wider.maskOf(contactModeBit(/*contactIndex=*/0, wider.numMpcContacts()));
  EXPECT_FALSE(unknown.ok());
  EXPECT_TRUE(absl::StrContains(unknown.status().message(), "elbow_l_contact"))
      << unknown.status();
}

TEST(PlanContactOrder, refusesAnMpcContactListedTwice) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(const multi_contact::ContactSet contactSet,
                                loadAtlasContactSet());
  const std::vector<std::string> twice = {"foot_l_contact", "foot_l_contact"};
  EXPECT_FALSE(PlanContactOrder::Create(contactSet, twice).ok());
}

}  // namespace
}  // namespace ocs2::humanoid
