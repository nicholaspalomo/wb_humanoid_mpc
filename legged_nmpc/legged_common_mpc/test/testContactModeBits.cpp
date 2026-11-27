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

#include "gtest/gtest.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_common_mpc/gait/MotionPhaseDefinition.h"

/**
 * A mode number is the bitmask of the active contacts in contact-set order,
 * the first contact the most significant bit
 * (legged_nmpc/docs/multi_contact_planning/README.md, section 3.1): on the
 * biped's two feet it is the reading of FLY, RF, LF and STANCE every gait file
 * has, and every number that is not a mode has no active contact.
 */
namespace ocs2::legged {
namespace {

TEST(ContactModeBits, theBipedsModesKeepTheirMeaning) {
  constexpr size_t kNumNumbersTried = 16;
  for (size_t mode = 0; mode < kNumNumbersTried; ++mode) {
    const BipedFeetArray<bool> biped = bipedFootContactsOfMode(mode);
    for (size_t foot = 0; foot < biped.size(); ++foot) {
      EXPECT_EQ(contactIsActiveInMode(mode, foot, biped.size()), biped[foot])
          << "mode " << mode << ", foot " << foot;
    }
  }
}

TEST(ContactModeBits, theFirstContactIsTheMostSignificantBit) {
  constexpr size_t kNumContactsOfTheSet = 5;
  constexpr size_t kFirstAndLast = 0b10001;
  EXPECT_TRUE(contactIsActiveInMode(kFirstAndLast, /*contactIndex=*/0,
                                    kNumContactsOfTheSet));
  EXPECT_FALSE(contactIsActiveInMode(kFirstAndLast, /*contactIndex=*/1,
                                     kNumContactsOfTheSet));
  EXPECT_FALSE(contactIsActiveInMode(kFirstAndLast, /*contactIndex=*/3,
                                     kNumContactsOfTheSet));
  EXPECT_TRUE(contactIsActiveInMode(kFirstAndLast, /*contactIndex=*/4,
                                    kNumContactsOfTheSet));
  // Past the contacts: no contact, and no mode.
  EXPECT_FALSE(contactIsActiveInMode(kFirstAndLast, /*contactIndex=*/5,
                                     kNumContactsOfTheSet));
  EXPECT_FALSE(contactIsActiveInMode(size_t{1} << kNumContactsOfTheSet,
                                     /*contactIndex=*/0, kNumContactsOfTheSet));
  // The biped's numbers mean the feet when they are the set's last two.
  EXPECT_TRUE(contactIsActiveInMode(kBipedModeStance, /*contactIndex=*/3,
                                    kNumContactsOfTheSet));
  EXPECT_TRUE(contactIsActiveInMode(kBipedModeStance, /*contactIndex=*/4,
                                    kNumContactsOfTheSet));
  EXPECT_FALSE(contactIsActiveInMode(kBipedModeStance, /*contactIndex=*/2,
                                     kNumContactsOfTheSet));
}

// The biped's mode numbers and its per-foot flags are each other's inverse on
// the four modes, and the two-foot ContactMask reads the same flags.
TEST(ContactModeBits, theBipedModesAndTheirFootContactsRoundTrip) {
  for (size_t mode = 0; mode < kNumBipedModes; ++mode) {
    EXPECT_EQ(bipedModeOfFootContacts(bipedFootContactsOfMode(mode)), mode)
        << "mode " << mode;
    const ContactMask mask =
        ContactMask::fromModeNumber(mode, static_cast<size_t>(kBipedFootCount));
    const BipedFeetArray<bool> feet = bipedFootContactsOfMode(mode);
    EXPECT_EQ(mask[kContactLeftIndex], feet[kContactLeftIndex])
        << "mode " << mode;
    EXPECT_EQ(mask[kContactRightIndex], feet[kContactRightIndex])
        << "mode " << mode;
  }
  EXPECT_EQ(bipedFootContactsOfMode(kNumBipedModes),
            makeBipedFeetArray(/*value=*/false));
}

}  // namespace
}  // namespace ocs2::legged
