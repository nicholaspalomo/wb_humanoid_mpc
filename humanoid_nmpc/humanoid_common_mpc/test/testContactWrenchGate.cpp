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

#include <gtest/gtest.h>

#include <stdexcept>

#include "humanoid_common_mpc/contact/ContactWrenchGate.h"

using namespace ocs2;
using namespace ocs2::humanoid;

namespace {
const vector6_t kLeft = (vector6_t() << 1, 2, 300, 4, 5, 6).finished();
const vector6_t kRight = (vector6_t() << -1, -2, 400, -4, -5, -6).finished();
const std::array<vector6_t, 2> kPlanned{kLeft, kRight};
}  // namespace

// Default configuration: the instantaneous gate. The planned wrench passes in full from the first cycle a foot is
// measured in contact and is dropped the cycle it is not.
TEST(ContactWrenchGate, defaultConfigurationIsTheInstantaneousGate) {
  ContactWrenchGate gate;
  EXPECT_EQ(gate.getScales(), (feet_array_t<scalar_t>{1.0, 1.0}));  // before the first update
  EXPECT_EQ(gate.update(0.0, {true, true}), (feet_array_t<scalar_t>{1.0, 1.0}));
  EXPECT_EQ(gate.update(0.002, {true, false}), (feet_array_t<scalar_t>{1.0, 0.0}));
  const auto gated = gate.apply(kPlanned);
  EXPECT_TRUE(gated[0].isApprox(kLeft));
  EXPECT_TRUE(gated[1].isZero());
  EXPECT_EQ(gate.update(0.004, {false, true}), (feet_array_t<scalar_t>{0.0, 1.0}));  // re-contact is immediate
}

TEST(ContactWrenchGate, debounceWithholdsTheWrenchUntilContactPersisted) {
  ContactWrenchGate gate({0.02, 0.0});
  EXPECT_NEAR(gate.update(1.000, {true, true})[0], 0.0, 1e-12);  // onset
  EXPECT_NEAR(gate.update(1.010, {true, true})[0], 0.0, 1e-12);
  EXPECT_NEAR(gate.update(1.020, {true, true})[0], 1.0, 1e-12);  // persisted for debounceTime: full wrench, no ramp
  // A bounce below the detection threshold restarts the debounce.
  EXPECT_NEAR(gate.update(1.022, {false, true})[0], 0.0, 1e-12);
  EXPECT_NEAR(gate.update(1.024, {true, true})[0], 0.0, 1e-12);
  EXPECT_NEAR(gate.update(1.044, {true, true})[0], 1.0, 1e-12);
  // The other foot, in contact throughout, is unaffected.
  EXPECT_NEAR(gate.getScales()[1], 1.0, 1e-12);
}

TEST(ContactWrenchGate, rampRisesLinearlyAfterTheDebounce) {
  ContactWrenchGate gate({0.01, 0.04});
  gate.update(2.00, {true, false});
  EXPECT_NEAR(gate.getScales()[0], 0.0, 1e-12);
  EXPECT_NEAR(gate.update(2.01, {true, false})[0], 0.0, 1e-12);  // debounce over, ramp starts
  EXPECT_NEAR(gate.update(2.02, {true, false})[0], 0.25, 1e-12);
  EXPECT_NEAR(gate.update(2.03, {true, false})[0], 0.5, 1e-12);
  const auto half = gate.apply(kPlanned);
  EXPECT_TRUE(half[0].isApprox(0.5 * kLeft));
  EXPECT_TRUE(half[1].isZero());
  EXPECT_NEAR(gate.update(2.05, {true, false})[0], 1.0, 1e-12);
  EXPECT_NEAR(gate.update(3.00, {true, false})[0], 1.0, 1e-12);  // saturates
  // Lift-off is never shaped.
  EXPECT_NEAR(gate.update(3.01, {false, false})[0], 0.0, 1e-12);
}

TEST(ContactWrenchGate, resetAndTimeRewindRestartTheOnset) {
  ContactWrenchGate gate({0.0, 0.1});
  gate.update(5.0, {true, true});
  gate.update(5.1, {true, true});
  EXPECT_NEAR(gate.getScales()[0], 1.0, 1e-12);
  gate.reset();
  EXPECT_EQ(gate.getScales(), (feet_array_t<scalar_t>{1.0, 1.0}));
  EXPECT_NEAR(gate.update(5.2, {true, true})[0], 0.0, 1e-12);
  EXPECT_NEAR(gate.update(5.25, {true, true})[0], 0.5, 1e-12);
  // A simulator reset moves time backwards: the onset restarts instead of producing a negative elapsed time.
  EXPECT_NEAR(gate.update(0.0, {true, true})[0], 0.0, 1e-12);
  EXPECT_NEAR(gate.update(0.1, {true, true})[0], 1.0, 1e-12);
}

TEST(ContactWrenchGate, configurationIsValidated) {
  EXPECT_THROW(ContactWrenchGate({-0.01, 0.0}), std::invalid_argument);
  EXPECT_THROW(ContactWrenchGate({0.0, -1.0}), std::invalid_argument);
  ContactWrenchGate gate;
  EXPECT_NO_THROW(gate.setConfig({0.02, 0.05}));
  EXPECT_DOUBLE_EQ(gate.getConfig().debounceTime, 0.02);
  EXPECT_DOUBLE_EQ(gate.getConfig().rampTime, 0.05);
}
