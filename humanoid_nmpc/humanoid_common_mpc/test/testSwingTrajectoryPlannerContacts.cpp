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
#include <vector>

#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"

/**
 * The swing trajectory planner on four contacts, ordered like a quadruped's
 * feet LF, RF, LH, RH (mode bits 8, 4, 2, 1): each contact's height reference
 * follows its own swings, read from the mode schedule per contact, on a trot
 * (the diagonal pairs swing together) and on a walk (one contact at a time).
 */
namespace ocs2::humanoid {
namespace {

constexpr size_t kNumContacts = 4;
constexpr scalar_t kTolerance = 1.0e-12;
constexpr scalar_t kPosSurfaceRtWorldEwrtWorldZ = 0.02;

// The modes of the four contacts.
constexpr size_t kAllFour = 0b1111;
constexpr size_t kTrotLfRh = 0b1001;
constexpr size_t kTrotRfLh = 0b0110;

/** A configuration whose swings are longer than its time scale, so that every
 * swing reaches the full apex. */
SwingTrajectoryPlanner::Config config() {
  SwingTrajectoryPlanner::Config planner;
  planner.apexPosFootRtSwingendEwrtWorldZ = 0.08;
  planner.swingTimeScale = 0.1;
  planner.liftoffLinvelFootRtWorldEwrtWorldZ = 0.05;
  planner.touchdownLinvelFootRtWorldEwrtWorldZ = -0.05;
  return planner;
}

/** One swing of one contact: its lift-off and touch-down. */
struct Swing {
  size_t contact = 0;
  scalar_t liftOff = 0.0;
  scalar_t touchDown = 0.0;
};

/** Where `time` lies for `contact`: inside one of its `swings` by more than
 * `margin`, within `margin` of a lift-off or touch-down, or in stance. */
enum class SwingPart { kSwing, kEvent, kStance };
SwingPart swingPartAt(const std::vector<Swing>& swings,
                      size_t contact,
                      scalar_t time,
                      scalar_t margin) {
  for (const Swing& swing : swings) {
    if (swing.contact != contact) continue;
    if (time > swing.liftOff + margin && time < swing.touchDown - margin) {
      return SwingPart::kSwing;
    }
    if (time > swing.liftOff - margin && time < swing.touchDown + margin) {
      return SwingPart::kEvent;
    }
  }
  return SwingPart::kStance;
}

/**
 * Expects every contact's height reference at `planner` to be the ground
 * outside its swings and above it inside, the apex at mid-swing, and the
 * lift-off and touch-down velocities of the configuration at the ends of each
 * swing; sampled every `kStep` over [0, `finalTime`].
 */
void expectSwings(const SwingTrajectoryPlanner& planner,
                  const std::vector<Swing>& swings,
                  scalar_t finalTime) {
  constexpr scalar_t kStep = 0.01;
  constexpr scalar_t kMargin = 1.0e-3;
  const int numSamples = static_cast<int>(finalTime / kStep);
  for (int sample = 0; sample <= numSamples; ++sample) {
    const scalar_t time = kStep * static_cast<scalar_t>(sample);
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      const scalar_t z = planner.getDesPosFootRtWorldEwrtWorldZ(contact, time);
      switch (swingPartAt(swings, contact, time, kMargin)) {
        case SwingPart::kSwing:
          EXPECT_GT(z, kPosSurfaceRtWorldEwrtWorldZ)
              << "contact " << contact << " at " << time;
          break;
        case SwingPart::kStance:
          EXPECT_NEAR(z, kPosSurfaceRtWorldEwrtWorldZ, kTolerance)
              << "contact " << contact << " at " << time;
          break;
        case SwingPart::kEvent:
          break;
      }
    }
  }
  const SwingTrajectoryPlanner::Config& planned = planner.getConfig();
  for (const Swing& swing : swings) {
    SCOPED_TRACE(::testing::Message() << "contact " << swing.contact
                                      << " swinging from " << swing.liftOff);
    const scalar_t midSwing = 0.5 * (swing.liftOff + swing.touchDown);
    EXPECT_NEAR(
        planner.getDesPosFootRtWorldEwrtWorldZ(swing.contact, midSwing),
        kPosSurfaceRtWorldEwrtWorldZ + planned.apexPosFootRtSwingendEwrtWorldZ,
        kTolerance);
    // At the lift-off instant the contact is still in stance (the event is
    // ahead, ModeSchedule::modeAtTime()); just after it, it moves at the
    // configured lift-off rate.
    constexpr scalar_t kAfterLiftOff = 1.0e-9;
    constexpr scalar_t kRateTolerance = 1.0e-6;
    EXPECT_NEAR(planner.getDesLinvelFootRtWorldEwrtWorldZ(
                    swing.contact, swing.liftOff + kAfterLiftOff),
                planned.liftoffLinvelFootRtWorldEwrtWorldZ, kRateTolerance);
    EXPECT_NEAR(planner.getDesLinvelFootRtWorldEwrtWorldZ(swing.contact,
                                                          swing.touchDown),
                planned.touchdownLinvelFootRtWorldEwrtWorldZ, kTolerance);
  }
}

TEST(SwingTrajectoryPlannerContacts, eachDiagonalPairSwingsOnATrot) {
  const ModeSchedule trot({0.2, 0.5, 0.8, 1.1}, {kAllFour, kTrotLfRh, kTrotRfLh,
                                                 kTrotLfRh, kAllFour});
  SwingTrajectoryPlanner planner(config(), kNumContacts);
  planner.update(trot, kPosSurfaceRtWorldEwrtWorldZ);
  ASSERT_EQ(planner.numFeet(), kNumContacts);
  // LF_RH stands first, so RF and LH swing first; LF and RH swing once, in
  // the middle phase.
  ASSERT_TRUE(ContactMask::fromModeNumber(kTrotLfRh, kNumContacts)[0]);
  const std::vector<Swing> swings = {
      {.contact = 1, .liftOff = 0.2, .touchDown = 0.5},
      {.contact = 2, .liftOff = 0.2, .touchDown = 0.5},
      {.contact = 0, .liftOff = 0.5, .touchDown = 0.8},
      {.contact = 3, .liftOff = 0.5, .touchDown = 0.8},
      {.contact = 1, .liftOff = 0.8, .touchDown = 1.1},
      {.contact = 2, .liftOff = 0.8, .touchDown = 1.1},
  };
  expectSwings(planner, swings, /*finalTime=*/1.3);
}

TEST(SwingTrajectoryPlannerContacts, oneContactSwingsAtATimeOnAWalk) {
  // LF, RH, RF, LH in turn, each a 0.3 s swing between all-four stances.
  constexpr size_t kLfSwings = 0b0111;
  constexpr size_t kRhSwings = 0b1110;
  constexpr size_t kRfSwings = 0b1011;
  constexpr size_t kLhSwings = 0b1101;
  const ModeSchedule walk({0.1, 0.4, 0.5, 0.8, 0.9, 1.2, 1.3, 1.6},
                          {kAllFour, kLfSwings, kAllFour, kRhSwings, kAllFour,
                           kRfSwings, kAllFour, kLhSwings, kAllFour});
  SwingTrajectoryPlanner planner(config(), kNumContacts);
  planner.update(walk, kPosSurfaceRtWorldEwrtWorldZ);
  const std::vector<Swing> swings = {
      {.contact = 0, .liftOff = 0.1, .touchDown = 0.4},
      {.contact = 3, .liftOff = 0.5, .touchDown = 0.8},
      {.contact = 1, .liftOff = 0.9, .touchDown = 1.2},
      {.contact = 2, .liftOff = 1.3, .touchDown = 1.6},
  };
  expectSwings(planner, swings, /*finalTime=*/1.8);
  // The impact proximity factor is 1 in stance and below 1 mid-swing, per
  // contact.
  EXPECT_EQ(planner.getImpactProximityFactor(/*leg=*/0, /*time=*/0.7), 1.0);
  EXPECT_LT(planner.getImpactProximityFactor(/*leg=*/0, /*time=*/0.25), 1.0);
  EXPECT_EQ(planner.getImpactProximityFactor(/*leg=*/1, /*time=*/0.25), 1.0);
}

}  // namespace
}  // namespace ocs2::humanoid
