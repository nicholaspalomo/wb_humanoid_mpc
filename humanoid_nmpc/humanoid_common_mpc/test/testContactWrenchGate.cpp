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

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactWrenchGate.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"

namespace ocs2::humanoid {

namespace {
const vector6_t kPlannedWrenchLfootEwrtWorld =
    (vector6_t() << 1, 2, 300, 4, 5, 6).finished();
const vector6_t kPlannedWrenchRfootEwrtWorld =
    (vector6_t() << -1, -2, 400, -4, -5, -6).finished();

/** The planned wrenches of a biped's left and right foot. */
ContactArray<vector6_t> plannedWrenchFeetEwrtWorld() {
  ContactArray<vector6_t> wrenchFeetEwrtWorld(/*numContacts=*/2);
  wrenchFeetEwrtWorld[0] = kPlannedWrenchLfootEwrtWorld;
  wrenchFeetEwrtWorld[1] = kPlannedWrenchRfootEwrtWorld;
  return wrenchFeetEwrtWorld;
}

/** The scales `values`, one per contact. */
ContactArray<scalar_t> scalesOf(std::initializer_list<scalar_t> values) {
  ContactArray<scalar_t> scales(values.size());
  size_t i = 0;
  for (const scalar_t value : values) scales[i++] = value;
  return scales;
}

/** The measured contact of a biped's left and right foot. */
ContactMask feet(bool left, bool right) {
  ContactMask mask(/*numContacts=*/2);
  mask.set(/*contactIndex=*/0, left);
  mask.set(/*contactIndex=*/1, right);
  return mask;
}
}  // namespace

// Default configuration: the instantaneous gate. The planned wrench passes in
// full from the first cycle a foot is measured in contact and is dropped the
// cycle it is not.
TEST(ContactWrenchGate, defaultConfigurationIsTheInstantaneousGate) {
  ContactWrenchGate gate(/*numContacts=*/2);
  EXPECT_EQ(gate.getScales(), scalesOf({1.0, 1.0}));  // before the first update
  EXPECT_EQ(gate.update(/*time=*/0.0, feet(/*left=*/true, /*right=*/true)),
            scalesOf({1.0, 1.0}));
  EXPECT_EQ(gate.update(/*time=*/0.002, feet(/*left=*/true, /*right=*/false)),
            scalesOf({1.0, 0.0}));
  ContactArray<vector6_t> gatedWrenchFeetEwrtWorld =
      plannedWrenchFeetEwrtWorld();
  gate.apply(gatedWrenchFeetEwrtWorld);
  EXPECT_TRUE(
      gatedWrenchFeetEwrtWorld[0].isApprox(kPlannedWrenchLfootEwrtWorld));
  EXPECT_TRUE(gatedWrenchFeetEwrtWorld[1].isZero());
  EXPECT_EQ(gate.update(/*time=*/0.004, feet(/*left=*/false, /*right=*/true)),
            scalesOf({0.0, 1.0}));  // re-contact is immediate
}

TEST(ContactWrenchGate, debounceWithholdsTheWrenchUntilContactPersisted) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      ContactWrenchGate gate,
      ContactWrenchGate::Create({.debounceTime = 0.02, .rampTime = 0.0},
                                /*numContacts=*/2));
  EXPECT_NEAR(
      gate.update(/*time=*/1.000, feet(/*left=*/true, /*right=*/true))[0], 0.0,
      1.0e-12);  // onset
  EXPECT_NEAR(
      gate.update(/*time=*/1.010, feet(/*left=*/true, /*right=*/true))[0], 0.0,
      1.0e-12);
  EXPECT_NEAR(
      gate.update(/*time=*/1.020, feet(/*left=*/true, /*right=*/true))[0], 1.0,
      1.0e-12);  // persisted for debounceTime: full wrench, no ramp
  // A bounce below the detection threshold restarts the debounce.
  EXPECT_NEAR(
      gate.update(/*time=*/1.022, feet(/*left=*/false, /*right=*/true))[0], 0.0,
      1.0e-12);
  EXPECT_NEAR(
      gate.update(/*time=*/1.024, feet(/*left=*/true, /*right=*/true))[0], 0.0,
      1.0e-12);
  EXPECT_NEAR(
      gate.update(/*time=*/1.044, feet(/*left=*/true, /*right=*/true))[0], 1.0,
      1.0e-12);
  // The other foot, in contact throughout, is unaffected.
  EXPECT_NEAR(gate.getScales()[1], 1.0, 1.0e-12);
}

TEST(ContactWrenchGate, rampRisesLinearlyAfterTheDebounce) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      ContactWrenchGate gate,
      ContactWrenchGate::Create({.debounceTime = 0.01, .rampTime = 0.04},
                                /*numContacts=*/2));
  gate.update(/*time=*/2.00, feet(/*left=*/true, /*right=*/false));
  EXPECT_NEAR(gate.getScales()[0], 0.0, 1.0e-12);
  EXPECT_NEAR(
      gate.update(/*time=*/2.01, feet(/*left=*/true, /*right=*/false))[0], 0.0,
      1.0e-12);  // debounce over, ramp starts
  EXPECT_NEAR(
      gate.update(/*time=*/2.02, feet(/*left=*/true, /*right=*/false))[0], 0.25,
      1.0e-12);
  EXPECT_NEAR(
      gate.update(/*time=*/2.03, feet(/*left=*/true, /*right=*/false))[0], 0.5,
      1.0e-12);
  ContactArray<vector6_t> gatedWrenchFeetEwrtWorld =
      plannedWrenchFeetEwrtWorld();
  gate.apply(gatedWrenchFeetEwrtWorld);
  EXPECT_TRUE(
      gatedWrenchFeetEwrtWorld[0].isApprox(0.5 * kPlannedWrenchLfootEwrtWorld));
  EXPECT_TRUE(gatedWrenchFeetEwrtWorld[1].isZero());
  EXPECT_NEAR(
      gate.update(/*time=*/2.05, feet(/*left=*/true, /*right=*/false))[0], 1.0,
      1.0e-12);
  EXPECT_NEAR(
      gate.update(/*time=*/3.00, feet(/*left=*/true, /*right=*/false))[0], 1.0,
      1.0e-12);  // saturates
  // Lift-off is never shaped.
  EXPECT_NEAR(
      gate.update(/*time=*/3.01, feet(/*left=*/false, /*right=*/false))[0], 0.0,
      1.0e-12);
}

TEST(ContactWrenchGate, resetAndTimeRewindRestartTheOnset) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      ContactWrenchGate gate,
      ContactWrenchGate::Create({.debounceTime = 0.0, .rampTime = 0.1},
                                /*numContacts=*/2));
  gate.update(/*time=*/5.0, feet(/*left=*/true, /*right=*/true));
  gate.update(/*time=*/5.1, feet(/*left=*/true, /*right=*/true));
  EXPECT_NEAR(gate.getScales()[0], 1.0, 1.0e-12);
  gate.reset();
  EXPECT_EQ(gate.getScales(), scalesOf({1.0, 1.0}));
  EXPECT_NEAR(gate.update(/*time=*/5.2, feet(/*left=*/true, /*right=*/true))[0],
              0.0, 1.0e-12);
  EXPECT_NEAR(
      gate.update(/*time=*/5.25, feet(/*left=*/true, /*right=*/true))[0], 0.5,
      1.0e-12);
  // A simulator reset moves time backwards: the onset restarts instead of
  // producing a negative elapsed time.
  EXPECT_NEAR(gate.update(/*time=*/0.0, feet(/*left=*/true, /*right=*/true))[0],
              0.0, 1.0e-12);
  EXPECT_NEAR(gate.update(/*time=*/0.1, feet(/*left=*/true, /*right=*/true))[0],
              1.0, 1.0e-12);
}

TEST(ContactWrenchGate, configurationIsValidated) {
  EXPECT_EQ(ContactWrenchGate::validateConfig(
                {.debounceTime = -0.01, .rampTime = 0.0})
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ContactWrenchGate::validateConfig({.debounceTime = 0.0, .rampTime = -1.0})
          .code(),
      absl::StatusCode::kInvalidArgument);
  const scalar_t nan = std::numeric_limits<scalar_t>::quiet_NaN();
  EXPECT_EQ(
      ContactWrenchGate::validateConfig({.debounceTime = nan, .rampTime = 0.0})
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      ContactWrenchGate::validateConfig({.debounceTime = 0.0, .rampTime = nan})
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(ContactWrenchGate::validateConfig(
                  {.debounceTime = -0.01, .rampTime = 0.0})
                  .message(),
              ::testing::HasSubstr("must be non-negative"));
  EXPECT_TRUE(
      ContactWrenchGate::validateConfig({.debounceTime = 0.0, .rampTime = 0.0})
          .ok());

  ContactWrenchGate gate(/*numContacts=*/2);
  EXPECT_TRUE(gate.setConfig({.debounceTime = 0.02, .rampTime = 0.05}));
  EXPECT_DOUBLE_EQ(gate.getConfig().debounceTime, 0.02);
  EXPECT_DOUBLE_EQ(gate.getConfig().rampTime, 0.05);
  EXPECT_FALSE(
      ContactWrenchGate::isValidConfig({.debounceTime = 0.0, .rampTime = nan}));
  EXPECT_TRUE(
      ContactWrenchGate::isValidConfig({.debounceTime = 0.0, .rampTime = 0.0}));
}

// The realtime thread sets the configuration, where nothing may end the
// process: an invalid one is refused and the one in use kept. Create() runs at
// setup and reports an invalid configuration as a Status.
TEST(ContactWrenchGate,
     setConfigRefusesAnInvalidConfigurationAndKeepsTheOneInUse) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      ContactWrenchGate gate,
      ContactWrenchGate::Create({.debounceTime = 0.02, .rampTime = 0.05},
                                /*numContacts=*/2));
  EXPECT_FALSE(gate.setConfig({.debounceTime = 0.0, .rampTime = -1.0}));
  EXPECT_FALSE(gate.setConfig(
      {.debounceTime = std::numeric_limits<scalar_t>::quiet_NaN(),
       .rampTime = 0.0}));
  EXPECT_DOUBLE_EQ(gate.getConfig().debounceTime, 0.02);
  EXPECT_DOUBLE_EQ(gate.getConfig().rampTime, 0.05);
}

TEST(ContactWrenchGate, createRefusesAnInvalidConfiguration) {
  const scalar_t nan = std::numeric_limits<scalar_t>::quiet_NaN();
  for (const ContactWrenchGate::Config& config :
       {ContactWrenchGate::Config{.debounceTime = -0.01, .rampTime = 0.0},
        ContactWrenchGate::Config{.debounceTime = 0.0, .rampTime = -1.0},
        ContactWrenchGate::Config{.debounceTime = nan, .rampTime = 0.0},
        ContactWrenchGate::Config{.debounceTime = 0.0, .rampTime = nan}}) {
    const absl::StatusOr<ContactWrenchGate> gate =
        ContactWrenchGate::Create(config, /*numContacts=*/2);
    EXPECT_EQ(gate.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(gate.status().message(),
                ::testing::HasSubstr("must be non-negative"));
  }
  const absl::StatusOr<ContactWrenchGate> instantaneous =
      ContactWrenchGate::Create(ContactWrenchGate::Config{},
                                /*numContacts=*/2);
  ASSERT_TRUE(instantaneous.ok()) << instantaneous.status();
  EXPECT_EQ(instantaneous->getScales(), scalesOf({1.0, 1.0}));
}

// A gate created from a configuration behaves as the default gate after
// setConfig() of the same configuration.
TEST(ContactWrenchGate,
     createdGateMatchesADefaultGateGivenTheSameConfiguration) {
  const ContactWrenchGate::Config config{.debounceTime = 0.01,
                                         .rampTime = 0.03};
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      ContactWrenchGate created,
      ContactWrenchGate::Create(config, /*numContacts=*/2));
  ContactWrenchGate reconfigured(/*numContacts=*/2);
  ASSERT_TRUE(reconfigured.setConfig(config));
  const std::array<ContactMask, 4> flags = {
      feet(/*left=*/true, /*right=*/false), feet(/*left=*/true, /*right=*/true),
      feet(/*left=*/false, /*right=*/true),
      feet(/*left=*/true, /*right=*/true)};
  scalar_t time = 0.0;
  for (int cycle = 0; cycle < 40; ++cycle) {
    const ContactMask& measured = flags[static_cast<size_t>(cycle / 10)];
    EXPECT_EQ(created.update(time, measured),
              reconfigured.update(time, measured))
        << "cycle " << cycle;
    time += 0.002;
  }
}

// The gate reads contact i from entry i of the mask; a contact past the mask's
// size is not in contact.
TEST(ContactWrenchGate, footPastTheMaskIsNotInContact) {
  ContactWrenchGate gate(/*numContacts=*/2);
  ContactMask leftOnly(/*numContacts=*/1);
  leftOnly.set(/*contactIndex=*/0, /*active=*/true);
  const ContactArray<scalar_t>& scales = gate.update(/*time=*/0.0, leftOnly);
  EXPECT_EQ(scales[0], 1.0);
  EXPECT_EQ(scales[1], 0.0);
}

// A gate of four point contacts (a quadruped's feet) gates each contact on its
// own, and apply() scales every contact's wrench.
TEST(ContactWrenchGate, gatesEveryContactOfAFourContactSet) {
  HUMANOID_ASSERT_OK_AND_ASSIGN(
      ContactWrenchGate gate,
      ContactWrenchGate::Create({.debounceTime = 0.0, .rampTime = 0.1},
                                /*numContacts=*/4));
  EXPECT_EQ(gate.numContacts(), 4);
  EXPECT_EQ(gate.getScales(), scalesOf({1.0, 1.0, 1.0, 1.0}));
  // Contacts 0 and 3 in contact, 1 and 2 in the air (mode 0b1001).
  const ContactMask diagonal =
      ContactMask::fromModeNumber(/*modeNumber=*/0b1001, /*numContacts=*/4);
  gate.update(/*time=*/0.0, diagonal);
  EXPECT_EQ(gate.update(/*time=*/0.05, diagonal),
            scalesOf({0.5, 0.0, 0.0, 0.5}));
  ContactArray<vector6_t> wrenchContactsEwrtWorld(/*numContacts=*/4,
                                                  kPlannedWrenchLfootEwrtWorld);
  gate.apply(wrenchContactsEwrtWorld);
  EXPECT_TRUE(
      wrenchContactsEwrtWorld[0].isApprox(0.5 * kPlannedWrenchLfootEwrtWorld));
  EXPECT_TRUE(wrenchContactsEwrtWorld[1].isZero());
  EXPECT_TRUE(wrenchContactsEwrtWorld[2].isZero());
  EXPECT_TRUE(
      wrenchContactsEwrtWorld[3].isApprox(0.5 * kPlannedWrenchLfootEwrtWorld));
}

TEST(ContactWrenchGate, createRefusesMoreContactsThanTheCapacity) {
  const absl::StatusOr<ContactWrenchGate> gate = ContactWrenchGate::Create(
      ContactWrenchGate::Config{}, kMaxNumContacts + 1);
  EXPECT_EQ(gate.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(gate.status().message(), ::testing::HasSubstr("33 contacts"));
}

}  // namespace ocs2::humanoid
