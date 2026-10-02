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

#include <fstream>
#include <sstream>
#include <string>

#include "humanoid_common_mpc_app/robot/DodgeballThrowParser.h"

namespace ocs2::humanoid {
namespace {

using DodgeballThrow = robot::mujoco_sim_interface::MujocoSimInterface::DodgeballThrow;

/** The payload the GUI publishes, captured from its own throw_payload; see the file's header. */
std::string goldenPayload() {
  std::ifstream file("humanoid_nmpc/humanoid_common_mpc_app/robot/test/data/dodgeball_payload.yaml");
  std::stringstream content;
  content << file.rdbuf();
  return content.str();
}

/**
 * The golden payload with one key's value replaced, or the key removed when `replacement` is empty. A key's block-style
 * list items ("  - value" lines under it) go with it, so what is left is still YAML and the parser is tested on the
 * missing key rather than on a syntax error.
 */
std::string withLine(const std::string& payload, const std::string& key, const std::string& replacement) {
  std::istringstream in(payload);
  std::ostringstream out;
  std::string line;
  bool skippingItems = false;
  while (std::getline(in, line)) {
    if (skippingItems && line.rfind("  - ", 0) == 0) continue;
    skippingItems = false;
    if (line.rfind("  " + key + ":", 0) == 0) {
      if (!replacement.empty()) out << "  " << key << ": " << replacement << "\n";
      skippingItems = true;
      continue;
    }
    out << line << "\n";
  }
  return out.str();
}

}  // namespace

TEST(ParseDodgeballThrow, ReadsTheGuisOwnPayload) {
  // The contract across the wire: a key renamed in dodgeball.py's throw_payload (the golden file is generated from it
  // and its Python test checks it still matches) fails here, and one renamed here fails here too.
  const std::string payload = goldenPayload();
  ASSERT_FALSE(payload.empty()) << "the golden payload is missing from the runfiles";
  const absl::StatusOr<DodgeballThrow> parsed = parseDodgeballThrow(payload);
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_DOUBLE_EQ(parsed->spawnOffset[0], 2.509549);
  EXPECT_DOUBLE_EQ(parsed->spawnOffset[1], 1.448889);
  EXPECT_DOUBLE_EQ(parsed->spawnOffset[2], 0.776457);
  EXPECT_DOUBLE_EQ(parsed->launchVelocity[0], -7.762142);
  EXPECT_DOUBLE_EQ(parsed->launchVelocity[1], -4.481475);
  EXPECT_DOUBLE_EQ(parsed->launchVelocity[2], -0.815798);
  EXPECT_DOUBLE_EQ(parsed->flightTime, 0.323306);
  EXPECT_DOUBLE_EQ(parsed->mass, 1.2) << "the mass slider's value has to reach the simulator";
}

TEST(ParseDodgeballThrow, EveryValueTheSimulatorUsesIsRequired) {
  // Including the mass: a missing one used to be filled in with 0.45 kg, a default nobody chose.
  for (const std::string key : {"spawnOffset", "launchVelocity", "flightTime", "mass"}) {
    const absl::StatusOr<DodgeballThrow> parsed = parseDodgeballThrow(withLine(goldenPayload(), key, ""));
    ASSERT_FALSE(parsed.ok()) << "accepted a payload without '" << key << "'";
    EXPECT_NE(std::string(parsed.status().message()).find(key), std::string::npos) << "the message should name '" << key << "'";
  }
}

TEST(ParseDodgeballThrow, TheDocumentationOnlyKeysAreNotRequired) {
  // The operator's slider values travel for the benefit of a recorded bag; a payload without them still throws.
  std::string payload = goldenPayload();
  for (const std::string key : {"azimuthDeg", "elevationDeg", "distance", "speed", "launchSpeed", "impactMomentum"}) {
    payload = withLine(payload, key, "");
  }
  EXPECT_TRUE(parseDodgeballThrow(payload).ok());
}

TEST(ParseDodgeballThrow, RejectsNonFiniteValues) {
  // A NaN typed into the GUI becomes `.nan` in YAML, and a NaN launch velocity makes MuJoCo reset the whole
  // simulation, robot included. Infinity is no better: an infinite mass times a velocity is an infinite force.
  for (const std::string bad : {".nan", ".inf", "-.inf"}) {
    EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "mass", bad)).ok()) << "mass " << bad;
    EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "flightTime", bad)).ok()) << "flightTime " << bad;
  }
  const std::string nanVector = withLine(goldenPayload(), "launchVelocity", "[-7.0, .nan, 1.0]");
  EXPECT_FALSE(parseDodgeballThrow(nanVector).ok());
}

TEST(ParseDodgeballThrow, RejectsANegativeFlightANonPositiveMassAndMalformedVectors) {
  EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "flightTime", "-0.1")).ok());
  EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "mass", "0")).ok());
  EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "mass", "-1")).ok());
  EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "spawnOffset", "[1.0, 2.0]")).ok()) << "two components";
  EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "spawnOffset", "[1.0, 2.0, x]")).ok()) << "not a number";
  EXPECT_FALSE(parseDodgeballThrow(withLine(goldenPayload(), "mass", "heavy")).ok());
}

TEST(ParseDodgeballThrow, AMassAboveTheSliderRangeIsLeftForTheSimulatorToClamp) {
  // The simulator clamps with clampProjectileMass on both of its paths; rejecting here would make the two disagree.
  const absl::StatusOr<DodgeballThrow> parsed = parseDodgeballThrow(withLine(goldenPayload(), "mass", "50"));
  ASSERT_TRUE(parsed.ok()) << parsed.status().message();
  EXPECT_DOUBLE_EQ(parsed->mass, 50.0);
}

TEST(ParseDodgeballThrow, RejectsWhatIsNotAThrowAtAll) {
  EXPECT_FALSE(parseDodgeballThrow("").ok());
  EXPECT_FALSE(parseDodgeballThrow("{").ok()) << "not YAML";
  EXPECT_FALSE(parseDodgeballThrow("- a list").ok());
  EXPECT_FALSE(parseDodgeballThrow("fsm_command: JOINT_PD").ok()) << "no 'dodgeball' block";
  EXPECT_FALSE(parseDodgeballThrow("dodgeball: 3").ok()) << "a 'dodgeball' that is not a block";
}

}  // namespace ocs2::humanoid
