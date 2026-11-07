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

#include <cstddef>
#include <type_traits>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

/**
 * ContactMask: the active contacts of a set as their mode number, the first
 * contact the most significant bit. On a biped's two feet it is the reading of
 * FLY, RF, LF and STANCE; on any set it agrees with contactIsActiveInMode().
 */
namespace ocs2::humanoid {
namespace {

static_assert(std::is_trivially_copyable_v<ContactMask>);

constexpr size_t kBipedContacts = 2;
constexpr size_t kQuadrupedContacts = 4;
// Four point contacts, then the two feet.
constexpr size_t kPointsAndFeetContacts = 6;

TEST(ContactMask, TheBipedModesRoundTrip) {
  constexpr size_t kNumNumbersTried = 16;
  for (size_t mode = 0; mode < kNumNumbersTried; ++mode) {
    const ContactMask mask = ContactMask::fromModeNumber(mode, kBipedContacts);
    const BipedFeetArray<bool> feet = bipedFootContactsOfMode(mode);
    EXPECT_EQ(mask[kContactLeftIndex], feet[kContactLeftIndex]) << mode;
    EXPECT_EQ(mask[kContactRightIndex], feet[kContactRightIndex]) << mode;
    if (mode < kNumBipedModes) {
      EXPECT_EQ(mask.modeNumber(), mode);
      EXPECT_EQ(bipedModeOfFootContacts(feet), mode);
    } else {
      EXPECT_TRUE(mask.none()) << mode;
    }
  }
  EXPECT_TRUE(
      ContactMask::fromModeNumber(kBipedModeFly, kBipedContacts).none());
  EXPECT_TRUE(
      ContactMask::fromModeNumber(kBipedModeStance, kBipedContacts).all());
  ContactMask leftOnly(kBipedContacts);
  leftOnly.set(kContactLeftIndex, /*active=*/true);
  EXPECT_EQ(leftOnly.modeNumber(), kBipedModeLf);
  ContactMask rightOnly(kBipedContacts);
  rightOnly.set(kContactRightIndex, /*active=*/true);
  EXPECT_EQ(rightOnly.modeNumber(), kBipedModeRf);
}

TEST(ContactMask, AQuadrupedLikeSetOfFourPointContacts) {
  const ContactMask all = ContactMask::allActive(kQuadrupedContacts);
  EXPECT_EQ(all.modeNumber(), 0b1111U);
  EXPECT_EQ(all.numActive(), 4U);
  EXPECT_TRUE(all.all());
  EXPECT_FALSE(all.none());

  ContactMask mask = all;
  mask.set(/*contactIndex=*/1, /*active=*/false);
  mask.set(/*contactIndex=*/2, /*active=*/false);
  EXPECT_EQ(mask.modeNumber(), 0b1001U);
  EXPECT_EQ(mask.numActive(), 2U);
  EXPECT_FALSE(mask.all());
  mask.set(/*contactIndex=*/2, /*active=*/true);
  EXPECT_EQ(mask.modeNumber(), 0b1011U);
  // Setting an active contact again changes nothing.
  mask.set(/*contactIndex=*/2, /*active=*/true);
  EXPECT_EQ(mask.modeNumber(), 0b1011U);

  const ContactMask diagonal =
      ContactMask::fromModeNumber(/*modeNumber=*/0b1001, kQuadrupedContacts);
  EXPECT_TRUE(diagonal[0]);
  EXPECT_FALSE(diagonal[1]);
  EXPECT_FALSE(diagonal[2]);
  EXPECT_TRUE(diagonal[3]);
  EXPECT_EQ(diagonal.size(), kQuadrupedContacts);

  // 16 and above are no mode of four contacts.
  EXPECT_TRUE(
      ContactMask::fromModeNumber(/*modeNumber=*/0b10000, kQuadrupedContacts)
          .none());
  EXPECT_TRUE(
      ContactMask::fromModeNumber(/*modeNumber=*/0b11001, kQuadrupedContacts)
          .none());
  EXPECT_EQ(
      ContactMask::fromModeNumber(/*modeNumber=*/0b11001, kQuadrupedContacts)
          .size(),
      kQuadrupedContacts);
}

TEST(ContactMask, TheBipedStanceIsTheLastTwoContactsOfALargerSet) {
  const ContactMask mask =
      ContactMask::fromModeNumber(kBipedModeStance, kPointsAndFeetContacts);
  EXPECT_EQ(mask.numActive(), 2U);
  for (size_t i = 0; i < mask.size(); ++i) {
    EXPECT_EQ(mask[i], i >= kPointsAndFeetContacts - kBipedContacts) << i;
  }
}

TEST(ContactMask, AgreesWithTheModeNumberHelpersOnEverySetOfUpToSixContacts) {
  constexpr size_t kLargestSetTried = 6;
  for (size_t numContacts = 1; numContacts <= kLargestSetTried; ++numContacts) {
    // Every mode of the set and the first numbers past it.
    for (size_t mode = 0; mode < 2 * numModeNumbersOf(numContacts); ++mode) {
      const ContactMask mask = ContactMask::fromModeNumber(mode, numContacts);
      ContactMask rebuilt(numContacts);
      for (size_t i = 0; i < numContacts; ++i) {
        EXPECT_EQ(mask[i], contactIsActiveInMode(mode, i, numContacts))
            << numContacts << " contacts, mode " << mode << ", contact " << i;
        rebuilt.set(i, contactIsActiveInMode(mode, i, numContacts));
      }
      EXPECT_EQ(rebuilt, mask);
      if (mode < numModeNumbersOf(numContacts)) {
        EXPECT_EQ(mask.modeNumber(), mode);
      }
    }
    for (size_t i = 0; i < numContacts; ++i) {
      ContactMask single(numContacts);
      single.set(i, /*active=*/true);
      EXPECT_EQ(single.modeNumber(), contactModeBit(i, numContacts));
      EXPECT_EQ(single.numActive(), 1U);
    }
    EXPECT_EQ(ContactMask::allActive(numContacts).modeNumber() + 1,
              numModeNumbersOf(numContacts));
  }
}

TEST(ContactMask, TheLargestSetUsesEveryBit) {
  const ContactMask all = ContactMask::allActive(kMaxNumContacts);
  EXPECT_EQ(all.numActive(), kMaxNumContacts);
  EXPECT_TRUE(all.all());
  ContactMask first(kMaxNumContacts);
  first.set(/*contactIndex=*/0, /*active=*/true);
  EXPECT_EQ(first.modeNumber(), size_t{1} << (kMaxNumContacts - 1));
  EXPECT_TRUE(first[0]);
  EXPECT_FALSE(first[kMaxNumContacts - 1]);
}

TEST(ContactMask, EqualityComparesTheSetSizeAndTheActiveContacts) {
  EXPECT_EQ(ContactMask(kQuadrupedContacts), ContactMask(kQuadrupedContacts));
  EXPECT_NE(ContactMask(kQuadrupedContacts), ContactMask(kBipedContacts));
  EXPECT_NE(ContactMask::allActive(kQuadrupedContacts),
            ContactMask(kQuadrupedContacts));
  const ContactMask empty;
  EXPECT_EQ(empty.size(), 0U);
  EXPECT_TRUE(empty.none());
  EXPECT_TRUE(empty.all());
}

TEST(ContactMask, PrintsTheFlagsInContactOrder) {
  EXPECT_EQ(absl::StrCat(ContactMask::fromModeNumber(/*modeNumber=*/0b1011,
                                                     kQuadrupedContacts)),
            "{1,0,1,1}");
  EXPECT_EQ(
      absl::StrCat(ContactMask::fromModeNumber(kBipedModeRf, kBipedContacts)),
      "{0,1}");
  EXPECT_EQ(absl::StrCat(ContactMask()), "{}");
}

TEST(ContactMaskDeathTest, MoreContactsThanTheCapacityAreRefused) {
  EXPECT_DEATH(ContactMask(kMaxNumContacts + 1), "kMaxNumContacts");
  EXPECT_DEATH(ContactMask::allActive(kMaxNumContacts + 1), "kMaxNumContacts");
}

}  // namespace
}  // namespace ocs2::humanoid
