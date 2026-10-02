/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

/**
 * The contact-implicit formulation, assembled end to end by CentroidalMpcInterface.
 *
 * It ships DISABLED (the task file lists zero_wrench / zero_velocity / normal_velocity and comments the three
 * contact-implicit terms out), which is correct - it changes the closed loop and has not been validated in simulation.
 * But it also meant the entire assembly path had NO coverage whatsoever: the one test that reached for these terms, a
 * hot-reload test of MpcParameterUpdaterModule, called GTEST_SKIP() because they were not in the shipped configuration.
 * That is how every one of the defects this file now pins went unnoticed.
 *
 * So these tests write a task file with the formulation switched ON and build the real interface from it, as the
 * hot-reload tests of testMpcParameterUpdaterModule now do too.
 */
class ContactImplicitFormulationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shippedTaskFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml").value();
    referenceFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml").value();
    urdfFile_ = robot::resolveResourcePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf").value();
  }

  void TearDown() override {
    for (const std::string& path : writtenFiles_) {
      std::remove(path.c_str());
    }
  }

  /**
   * The shipped task file with its two constraint lists REPLACED, written to a temporary path.
   *
   * Whole blocks are rewritten rather than individual lines substituted, deliberately. These tests have to work while
   * an engineer is mid-experiment with the formulation toggled on in the working tree - matching on the exact text of
   * a commented-out line would make the suite go red the moment somebody tried the thing the suite exists to support.
   */
  std::string writeTaskFile(const std::string& name,
                            const std::vector<std::string>& hardConstraints,
                            const std::vector<std::string>& softConstraints) {
    std::ifstream in(shippedTaskFile_);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string content = buffer.str();
    content = replaceListBlock(content, "hard_constraints:", hardConstraints);
    content = replaceListBlock(content, "soft_constraints:", softConstraints);

    const std::string path = absl::StrCat(testing::TempDir(), "/testContactImplicit_", name, ".yaml");
    std::ofstream out(path);
    out << content;
    out.close();
    writtenFiles_.push_back(path);
    return path;
  }

  /** Replaces everything from `key` up to the next blank line with `key` followed by the given entries. */
  static std::string replaceListBlock(const std::string& content, const std::string& key, const std::vector<std::string>& entries) {
    const size_t keyPos = content.find(absl::StrCat("\n", key));
    EXPECT_NE(keyPos, std::string::npos) << "the task file has no " << key << " block";
    if (keyPos == std::string::npos) return content;
    const size_t blockStart = keyPos + 1;
    size_t blockEnd = content.find("\n\n", blockStart);
    if (blockEnd == std::string::npos) blockEnd = content.size();

    std::string block = absl::StrCat(key, "\n");
    for (const std::string& entry : entries) {
      absl::StrAppend(&block, "  - ", entry, "\n");
    }
    return absl::StrCat(content.substr(0, blockStart), block, content.substr(blockEnd + 1));
  }

  /** The soft constraints the shipped robot always needs, whatever the contact formulation. */
  static std::vector<std::string> baseSoftConstraints() { return {"joint_limits", "foot_collision", "contact_wrench_cone"}; }

  static std::vector<std::string> contactImplicitSoftConstraints() {
    std::vector<std::string> soft = baseSoftConstraints();
    soft.push_back("normal_velocity");
    soft.push_back("contact_complementarity");
    soft.push_back("force_weighted_slip");
    soft.push_back("ground_penetration");
    return soft;
  }

  /** A ground height away from the shipped default of zero, so a comparison of two terrain heights compares something. */
  static constexpr scalar_t kTestTerrainHeight = 0.037;

  /**
   * A soft normal_velocity weight that is neither the shipped value nor ModelSettings' default, so that a test reading
   * it back from the built problem can tell the task file's value from a default or a hard-coded one.
   */
  static constexpr scalar_t kTestNormalVelocityWeight = 437.25;

  /**
   * The whole formulation switched on: no schedule-gated hard constraints; the three contact-implicit terms and the soft
   * normal_velocity in, the latter at kTestNormalVelocityWeight; the ground at kTestTerrainHeight.
   */
  std::string writeContactImplicitTaskFile(const std::string& name) {
    const std::string path = writeTaskFile(name, {}, contactImplicitSoftConstraints());
    rewriteScalarKey(path, "terrainHeight", kTestTerrainHeight);
    rewriteNestedScalarKey(path, "normalVelocitySoftConstraintWeight", kTestNormalVelocityWeight);
    return path;
  }

  /** Rewrites the value of the one line, at any indentation, that sets `key`; fails the test unless exactly one does. */
  static void rewriteNestedScalarKey(const std::string& path, const std::string& key, scalar_t value) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    std::string rewritten;
    std::string line;
    size_t matches = 0;
    while (std::getline(buffer, line)) {
      const size_t firstNonBlank = line.find_first_not_of(' ');
      if (firstNonBlank != std::string::npos && line.compare(firstNonBlank, key.size() + 1, absl::StrCat(key, ":")) == 0) {
        line = absl::StrCat(line.substr(0, firstNonBlank), key, ": ", value);
        ++matches;
      }
      absl::StrAppend(&rewritten, line, "\n");
    }
    ASSERT_EQ(matches, 1U) << "expected exactly one '" << key << ":' line in " << path;
    std::ofstream out(path);
    out << rewritten;
  }

  /** Renames the one key, at any indentation, spelled `from:` to `to:`; fails the test unless exactly one line has it. */
  static void renameNestedKey(const std::string& path, const std::string& from, const std::string& to) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    std::string rewritten;
    std::string line;
    size_t matches = 0;
    while (std::getline(buffer, line)) {
      const size_t firstNonBlank = line.find_first_not_of(' ');
      if (firstNonBlank != std::string::npos && line.compare(firstNonBlank, from.size() + 1, absl::StrCat(from, ":")) == 0) {
        line = absl::StrCat(line.substr(0, firstNonBlank), to, line.substr(firstNonBlank + from.size()));
        ++matches;
      }
      absl::StrAppend(&rewritten, line, "\n");
    }
    ASSERT_EQ(matches, 1U) << "expected exactly one '" << from << ":' line in " << path;
    std::ofstream out(path);
    out << rewritten;
  }

  /** Rewrites a top-level scalar key of an already written task file. */
  static void rewriteScalarKey(const std::string& path, const std::string& key, scalar_t value) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    std::string content = buffer.str();
    const size_t keyPos = content.find(absl::StrCat("\n", key, ":"));
    ASSERT_NE(keyPos, std::string::npos) << "the task file has no top-level " << key;
    const size_t valueStart = keyPos + 1 + key.size() + 1;
    const size_t lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, absl::StrCat(" ", value));
    std::ofstream out(path);
    out << content;
  }

  std::string shippedTaskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::vector<std::string> writtenFiles_;
};

// ---------------------------------------------------------------------------------------------------------------
// The formulation loader refuses every half-way combination.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactImplicitFormulationTest, theHardNormalVelocityConstraintIsRefusedAlongsideTheContactImplicitTerms) {
  // normal_velocity is a HARD equality on the swing foot's vertical velocity, so it fixes the entire height profile of
  // the swing and the solver cannot move a touch-down it is being asked to choose. It was the one schedule-gated hard
  // constraint the loader did not refuse, and the README explicitly left it listed.
  // zero_wrench and zero_velocity ARE dropped, and the soft normal_velocity is NOT listed, so every earlier check
  // passes and this one is reached: with the soft one listed too the hard-and-soft duplicate check would refuse the
  // file first, and the test would prove nothing about this rule.
  std::vector<std::string> soft = contactImplicitSoftConstraints();
  soft.erase(std::remove(soft.begin(), soft.end(), std::string("normal_velocity")), soft.end());
  const std::string taskFile = writeTaskFile("normalVelocityKept", {"normal_velocity"}, soft);
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_FALSE(tasks.ok()) << "normal_velocity fixes the whole height profile of the swing and must be refused";
  EXPECT_TRUE(absl::StrContains(tasks.status().message(), "the hard 'normal_velocity' constraint are mutually exclusive"))
      << tasks.status().message();
}

TEST_F(ContactImplicitFormulationTest, complementarityWithoutTheSlipTermIsRefused) {
  // The three relaxed complementarity conditions are a set. The complementarity product and the penetration hinge are
  // both POSITIONAL - one forbids load above the ground, the other the foot below it - so with force_weighted_slip
  // missing nothing holds a loaded foot still and a foot carrying full body weight may slide frictionlessly. The
  // loader already refused complementarity without ground_penetration; this is the same argument for the third term.
  std::vector<std::string> soft = contactImplicitSoftConstraints();
  soft.erase(std::remove(soft.begin(), soft.end(), std::string("force_weighted_slip")), soft.end());
  const std::string taskFile = writeTaskFile("noSlip", {}, soft);
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_FALSE(tasks.ok()) << "a loaded foot would be free to slide";
  EXPECT_NE(std::string(tasks.status().message()).find("force_weighted_slip"), std::string::npos) << tasks.status().message();
}

TEST_F(ContactImplicitFormulationTest, complementarityCannotBePairedWithAScheduleGatedStanceConstraint) {
  // Reached transitively: requiring force_weighted_slip above makes the existing force_weighted_slip / zero_velocity
  // exclusion fire, so complementarity alongside a schedule-gated zero_velocity is now refused as well. Before that
  // rule it was accepted, which left the solver told to choose contact by one term and told it by the schedule by
  // another.
  const std::string taskFile = writeTaskFile("complementarityWithZeroVelocity", {"zero_velocity"}, contactImplicitSoftConstraints());
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_FALSE(tasks.ok());
  EXPECT_NE(std::string(tasks.status().message()).find("zero_velocity"), std::string::npos) << tasks.status().message();
}

TEST_F(ContactImplicitFormulationTest, groundPenetrationOnItsOwnIsRefusedAsAPartialFormulation) {
  // The three terms are listed together or not at all, which is what usesContactImplicitFormulation() and every caller
  // of it assume. A lone ground_penetration used to load - this test once recorded that as deliberate - and beside the
  // shipped hard normal_velocity it was then refused by a check whose message, about a touch-down the solver is asked
  // to choose, had nothing to do with it. It is now refused up front, with a message that names what is missing.
  const std::string taskFile = writeTaskFile("penetrationOnly", {"zero_wrench", "normal_velocity", "zero_velocity"},
                                             {"joint_limits", "contact_wrench_cone", "ground_penetration"});
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_FALSE(tasks.ok());
  EXPECT_TRUE(absl::StrContains(tasks.status().message(), "together or not at all")) << tasks.status().message();
  EXPECT_FALSE(absl::StrContains(tasks.status().message(), "touch-down")) << tasks.status().message();
}

TEST_F(ContactImplicitFormulationTest, theTaskFileInTheWorkingTreeIsSelfConsistent) {
  // Whatever the file is currently set to - the formulation ships off, but an engineer testing it has it on - the
  // loader must accept it and the contact-constraint gate must agree with it. That the COMMITTED file keeps the
  // formulation off is a separate rule, pinned by humanoid_common_mpc:testMpcFormulationConfig over every robot's
  // task file, so that an experiment in the working tree turns that one target red rather than this whole suite.
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(shippedTaskFile_, /*verbose=*/false);
  ASSERT_TRUE(tasks.ok()) << tasks.status().message();

  // The gate follows `zero_wrench`, and nothing else.
  EXPECT_EQ(contactConstraintsAreScheduleGated(*tasks), tasks->hasHardConstraint(MpcHardConstraintType::ZeroWrench));

  // And the combinations the loader refuses really are absent, since it accepted the file.
  if (usesContactImplicitFormulation(*tasks)) {
    EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::ZeroWrench));
    EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::ZeroVelocity));
    EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::NormalVelocity));
    EXPECT_TRUE(tasks->hasSoftConstraint(MpcSoftConstraintType::NormalVelocity));
  }
}

TEST_F(ContactImplicitFormulationTest, theFormulationWithoutTheSoftNormalVelocityIsRefused) {
  // README section 3 and the task file call the soft servo "not optional": with the hard one refused it is the only
  // term that can lift a swing foot, and without it the robot shuffled. The loader used to accept the omission.
  std::vector<std::string> soft = contactImplicitSoftConstraints();
  soft.erase(std::remove(soft.begin(), soft.end(), std::string("normal_velocity")), soft.end());
  const std::string taskFile = writeTaskFile("noSoftNormalVelocity", {}, soft);
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_FALSE(tasks.ok());
  EXPECT_TRUE(absl::StrContains(tasks.status().message(), "needs 'normal_velocity' in soft_constraints")) << tasks.status().message();

  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  EXPECT_FALSE(interface.ok());
}

TEST_F(ContactImplicitFormulationTest, anOutOfRangeContactImplicitValueIsAStatusNamingItsKey) {
  // The contact_implicit values are divisors and penalty weights. A zero gapSmoothing used to reach the complementarity
  // term's constructor, whose CHECK aborted the process from inside the Status-returning Create(); it is refused before
  // any term is built, with a message naming the key to change.
  const std::string taskFile = writeContactImplicitTaskFile("zeroGapSmoothing");
  rewriteNestedScalarKey(taskFile, "gapSmoothing", /*value=*/0.0);
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  ASSERT_FALSE(interface.ok());
  EXPECT_EQ(interface.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(interface.status().message(), "contact_implicit.gapSmoothing")) << interface.status();
}

TEST_F(ContactImplicitFormulationTest, aNegativePenetrationWeightIsAStatusNamingItsKey) {
  // A negative weight turns the hinge into a reward for a foot that goes through the floor; nothing aborts on it, so
  // without the check it would simply have run.
  const std::string taskFile = writeContactImplicitTaskFile("negativePenetrationWeight");
  rewriteNestedScalarKey(taskFile, "penetrationWeight", /*value=*/-5.0e4);
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  ASSERT_FALSE(interface.ok());
  EXPECT_EQ(interface.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(interface.status().message(), "contact_implicit.penetrationWeight")) << interface.status();
}

TEST_F(ContactImplicitFormulationTest, aContactImplicitKeyNothingReadsIsRefusedWhetherOrNotTheFormulationIsListed) {
  // Audit findings A11/A22. A key renamed in the task file but not in the code was skipped by ModelSettings, by the
  // validator and by the parameter updater alike: its term ran on the default and its tuning slider reached nothing.
  // Refused here with the shipped, formulation-off lists, before any CppAD model is built.
  const std::string taskFile =
      writeTaskFile("renamedContactImplicitKey", {"zero_wrench", "normal_velocity", "zero_velocity"}, baseSoftConstraints());
  renameNestedKey(taskFile, "gapSmoothing", "gap_smoothing");
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  ASSERT_FALSE(interface.ok());
  EXPECT_EQ(interface.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(interface.status().message(), "contact_implicit.gap_smoothing")) << interface.status();
  EXPECT_TRUE(absl::StrContains(interface.status().message(), "gapSmoothing")) << "the message lists the keys the block may carry";
}

TEST_F(ContactImplicitFormulationTest, theContactImplicitTaskFileLoadsCleanly) {
  const std::string taskFile = writeContactImplicitTaskFile("loads");
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_TRUE(tasks.ok()) << tasks.status().message();
  EXPECT_TRUE(usesContactImplicitFormulation(*tasks));
  EXPECT_FALSE(contactConstraintsAreScheduleGated(*tasks));
  EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::NormalVelocity));
}

TEST_F(ContactImplicitFormulationTest, noConeAtAllIsRefusedWhateverTheInputParameterization) {
  // f_n >= 0 is the FIRST of the three conditions of rigid contact, and the formulation supplies only the other two:
  // `ground_penetration` gives h >= 0 and `contact_complementarity` gives f_n h = 0, which at h = 0 is satisfied by
  // any f_n at all, a negative one included. So a foot resting on the floor could pull on it for free. The bound is
  // the cone's, and the cones stop bounding anything the moment `zero_wrench` goes, because that is what they gate
  // themselves on. This is refused at the LOADER, before any parameterization-specific reasoning.
  std::vector<std::string> soft = contactImplicitSoftConstraints();
  soft.erase(std::remove(soft.begin(), soft.end(), std::string("contact_wrench_cone")), soft.end());
  const std::string taskFile = writeTaskFile("noCone", {}, soft);

  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_FALSE(tasks.ok()) << "nothing would bound any foot's wrench";
  EXPECT_NE(std::string(tasks.status().message()).find("friction_force_cone"), std::string::npos) << tasks.status().message();

  // And the interface refuses it too, since it loads through the same function.
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  EXPECT_FALSE(interface.ok());
}

TEST_F(ContactImplicitFormulationTest, basisVectorInputsBoundTheScalingsWhicheverConeIsListed) {
  // Findings A76/A84. The loader accepts EITHER cone, because either bounds the normal force below. Under basis-vector
  // inputs neither is what bounds the individual scalings: `friction_force_cone` bounds the assembled wrench only, and
  // lambda >= 0 is the whole of the cone there. The interface used to build that barrier only when
  // `contact_wrench_cone` was listed, and so refused this combination rather than leave the scalings free. The barrier
  // now belongs to the parameterization, so the combination is sound and is built - with the barrier, un-gated here
  // because zero_wrench is gone, beside the friction cone the file asked for.
  std::vector<std::string> soft = contactImplicitSoftConstraints();
  std::replace(soft.begin(), soft.end(), std::string("contact_wrench_cone"), std::string("friction_force_cone"));
  const std::string taskFile = writeTaskFile("frictionConeOnly", {}, soft);

  // The loader is satisfied: a friction cone does bound f_n below.
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_TRUE(tasks.ok()) << tasks.status().message();
  ASSERT_FALSE(tasks->hasSoftConstraint(MpcSoftConstraintType::ContactWrenchCone));

  // This robot runs contactInputParameterization: basis_vectors.
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  ASSERT_TRUE(interface.ok()) << interface.status().message();
  ASSERT_TRUE((*interface)->usesContactBasisVectorInputs());
  OptimalControlProblem& problem = (*interface)->getOptimalControlProblemRef();
  for (const std::string& footName : (*interface)->modelSettings().contactNames) {
    SCOPED_TRACE(footName);
    size_t index = 0;
    ASSERT_TRUE(problem.costPtr->getTermIndex(absl::StrCat(footName, "_basisNonNegativity"), index))
        << "basis-vector contact inputs without their lambda >= 0 barrier";
    EXPECT_FALSE(problem.costPtr->get<BasisScalingNonNegativityConstraint>(absl::StrCat(footName, "_basisNonNegativity")).isScheduleGated())
        << "zero_wrench is gone, so the barrier must act on a foot the schedule calls a swing foot too";
    ASSERT_TRUE(problem.softConstraintPtr->getTermIndex(absl::StrCat(footName, "_frictionForceCone"), index));
    // Audit finding A15, at the interface: the friction cone it asked the factory for is the UN-gated one, wrapped in a
    // squared hinge whose zero sits on the cone - silent at zero force, where a foot in flight lies - and not in the
    // relaxed barrier of a schedule-gated cone, which would pay the solver to load that foot.
    StateInputSoftConstraint& frictionCone =
        problem.softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_frictionForceCone"));
    EXPECT_FALSE(frictionCone.get<FrictionForceConeConstraint>().isScheduleGated());
    for (const std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : frictionCone.getPenalty().getPenaltyPtrArray()) {
      EXPECT_EQ(penalty->name(), "SquaredHingePenalty");
      vector_t parameters;
      penalty->getParameters(parameters);
      ASSERT_EQ(parameters.size(), 2);
      EXPECT_DOUBLE_EQ(parameters(1), 0.0) << "the hinge's zero must sit on the cone";
      EXPECT_DOUBLE_EQ(penalty->getValue(0.0, 0.0, 0.0), 0.0);
      EXPECT_DOUBLE_EQ(penalty->getDerivative(0.0, 0.0, 0.0), 0.0);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The assembled problem.
// ---------------------------------------------------------------------------------------------------------------

class ContactImplicitProblemTest : public ContactImplicitFormulationTest {
 protected:
  void SetUp() override {
    ContactImplicitFormulationTest::SetUp();
    const std::string taskFile = writeContactImplicitTaskFile("problem");
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
    ASSERT_TRUE(created.ok()) << created.status().message();
    interface_ = *std::move(created);
  }

  OptimalControlProblem& problem() const { return interface_->getOptimalControlProblemRef(); }
  const std::vector<std::string>& contactNames() const { return interface_->modelSettings().contactNames; }

  /** State index of a foot's ankle pitch joint, looked up by name so the test cannot tilt the wrong joint. */
  long anklePitchStateIndex(size_t contactIndex) const {
    const std::string jointName = contactIndex == CONTACT_LEFT_INDEX ? "l_leg_aky" : "r_leg_aky";
    const std::vector<std::string>& jointNames = interface_->modelSettings().mpcModelJointNames;
    const std::vector<std::string>::const_iterator found = std::find(jointNames.begin(), jointNames.end(), jointName);
    EXPECT_NE(found, jointNames.end()) << "no joint named " << jointName;
    return static_cast<long>(interface_->getMpcRobotModel().getJointStartindex() +
                             static_cast<size_t>(std::distance(jointNames.begin(), found)));
  }

  /** [m] the world height of a frame at a state, from the numeric Pinocchio model the interface was built on. */
  scalar_t frameHeight(const std::string& frameName, const vector_t& state) const {
    const pinocchio::Model& model = interface_->getPinocchioInterface().getModel();
    pinocchio::Data data(model);
    updateFramePlacements(interface_->getMpcRobotModel().getGeneralizedCoordinates(state), model, data);
    return data.oMf[model.getFrameId(frameName)].translation().z();
  }

  /**
   * Replaces the mode schedule with one in which `swingFoot` swings over [kSwingStart, kSwingEnd] between two stance
   * phases, and runs the reference manager over it so the swing trajectories are planned. The brackets far outside the
   * window keep the gait schedule from tiling its template over it.
   */
  void scheduleOneSwing(size_t swingFoot) const {
    contact_flag_t swingFlags = makeFeetArray(true);
    swingFlags[swingFoot] = false;
    const ModeSchedule schedule(
        {-10.0, kSwingStart, kSwingEnd, 10.0},
        {ModeNumber::STANCE, ModeNumber::STANCE, stanceLeg2ModeNumber(swingFlags), ModeNumber::STANCE, ModeNumber::STANCE});
    SwitchedModelReferenceManager& referenceManager = *interface_->getSwitchedModelReferenceManagerPtr();
    referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
    referenceManager.preSolverRun(kSwingStart - 0.2, kSwingEnd + 0.5, interface_->getInitialState(), ModeNumber::STANCE);
  }

  static constexpr scalar_t kSwingStart = 0.0;
  static constexpr scalar_t kSwingEnd = 0.6;

  std::unique_ptr<CentroidalMpcInterface> interface_;
};

TEST_F(ContactImplicitProblemTest, allThreeContactImplicitTermsAreBuiltForEveryFoot) {
  for (const std::string& footName : contactNames()) {
    EXPECT_NO_THROW(problem().softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_contactComplementarity")))
        << footName;
    EXPECT_NO_THROW(problem().softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_forceWeightedSlip"))) << footName;
    EXPECT_NO_THROW(problem().stateSoftConstraintPtr->get<StateSoftConstraint>(absl::StrCat(footName, "_groundPenetration"))) << footName;
  }
}

TEST_F(ContactImplicitProblemTest, theScheduleGatedContactConstraintsAreGone) {
  // Nothing may force the swing foot's wrench or velocity from the mode schedule any more.
  for (const std::string& footName : contactNames()) {
    EXPECT_THROW(problem().equalityConstraintPtr->get<StateInputConstraint>(absl::StrCat(footName, "_zeroWrench")), std::out_of_range)
        << footName;
    EXPECT_THROW(problem().equalityConstraintPtr->get<StateInputConstraint>(absl::StrCat(footName, "_zeroVelocity")), std::out_of_range)
        << footName;
    EXPECT_THROW(problem().equalityConstraintPtr->get<StateInputConstraint>(absl::StrCat(footName, "_normalVelocity")), std::out_of_range)
        << footName;
  }
}

TEST_F(ContactImplicitProblemTest, theBasisScalingBarrierIsNotGatedOnTheModeSchedule) {
  // On the shipped DRC Atlas (contactInputParameterization: basis_vectors) this term is the ONLY lower bound on the contact
  // wrench, because the explicit wrench cone is skipped in favor of the structural guarantee of lambda >= 0. Gated, a
  // foot the schedule calls a swing foot had no bound at all.
  ASSERT_TRUE(interface_->getBasisDecoratorPtr() != nullptr) << "this robot is expected to run basis-vector inputs";
  for (const std::string& footName : contactNames()) {
    const BasisScalingNonNegativityConstraint& barrier =
        problem().costPtr->get<BasisScalingNonNegativityConstraint>(absl::StrCat(footName, "_basisNonNegativity"));
    EXPECT_FALSE(barrier.isScheduleGated()) << footName;
  }
}

TEST_F(ContactImplicitProblemTest, groundPenetrationIsCheckedAtEveryCornerOfTheFootprint) {
  // The sole CENTER alone is not enough: this formulation deliberately leaves the foot's rocking rates free, so a foot
  // pitched about a center held at ground level buries its toe for nothing. The DRC Atlas footprint has four corners
  // 0.12 m fore and aft of the center.
  for (const std::string& footName : contactNames()) {
    const GroundPenetrationConstraint& penetration =
        problem()
            .stateSoftConstraintPtr->get<StateSoftConstraint>(absl::StrCat(footName, "_groundPenetration"))
            .get<GroundPenetrationConstraint>();
    EXPECT_EQ(penetration.getNumPoints(), 4U) << footName << ": one row per corner of the contact polygon";
    EXPECT_EQ(penetration.getNumConstraints(0.0), 4U) << footName;
  }
}

TEST_F(ContactImplicitProblemTest, theComplementarityGapAndThePenetrationRowsAreMeasuredAtTheSamePoints) {
  // The two terms are the two halves of one condition, so the assembled problem must not contain two different ideas
  // of where the foot is. It did: the hinge was moved to the footprint corners while the product was left reading the
  // sole center, and a foot rocked onto its heel was then "airborne" to one and "touching" to the other.
  //
  // Checked on PITCHED feet. At the initial state the soles are flat, every corner sits at the sole center's height,
  // and a gap measured at the center satisfies the bracket below exactly - so a flat foot cannot tell the wiring this
  // test is named after from the regression it guards against.
  const ModelSettings::ContactImplicitConfig& config = interface_->modelSettings().contactImplicitConfig;
  vector_t state = interface_->getInitialState();
  for (size_t foot = 0; foot < contactNames().size(); ++foot) {
    state(anklePitchStateIndex(foot)) += 0.25;
  }
  const PreComputation preComp;
  for (const std::string& footName : contactNames()) {
    const ContactComplementarityConstraint& complementarity =
        problem()
            .softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_contactComplementarity"))
            .get<ContactComplementarityConstraint>();
    const GroundPenetrationConstraint& penetration =
        problem()
            .stateSoftConstraintPtr->get<StateSoftConstraint>(absl::StrCat(footName, "_groundPenetration"))
            .get<GroundPenetrationConstraint>();
    EXPECT_NEAR(complementarity.getGapSmoothing(), config.gapSmoothing, 1e-12) << footName;

    // The penetration rows are the corner clearances; the gap is their smoothed minimum, so it is bracketed by the
    // smallest of them and that value plus log(N) * gapSmoothing. Measured at the sole center it would not be.
    const vector_t clearances = penetration.getValue(/*time=*/0.0, state, preComp);
    ASSERT_EQ(clearances.size(), static_cast<long>(penetration.getNumPoints())) << footName;
    const scalar_t bound = std::log(static_cast<scalar_t>(penetration.getNumPoints())) * config.gapSmoothing;
    EXPECT_GE(complementarity.getGap(state), clearances.minCoeff() - 1e-12) << footName;
    EXPECT_LE(complementarity.getGap(state), clearances.minCoeff() + bound + 1e-12) << footName;

    // Positive control: the pitched sole's center is well above its lowest corner, so a gap taken at the contact frame
    // would sit outside the bracket and fail it.
    const scalar_t soleCenterClearance = frameHeight(footName, state) - complementarity.getTerrainHeight();
    ASSERT_GT(soleCenterClearance - clearances.minCoeff(), 0.01) << footName << ": the ankle pitch must actually tilt the sole";
    EXPECT_LT(complementarity.getGap(state), soleCenterClearance - bound - 0.005) << footName;
  }
}

TEST_F(ContactImplicitProblemTest, groundPenetrationIsAHingeAndNotALogBarrier) {
  // A relaxed log barrier never reaches zero: at h = 0 its derivative is -2*mu/delta, constant and upward, so it pushes
  // every foot off the ground and the complementarity term has to hold it down. Balancing those two put a foot at half
  // body weight - i.e. both feet, throughout double support - a centimeter above the floor.
  for (const std::string& footName : contactNames()) {
    StateSoftConstraint& softConstraint =
        problem().stateSoftConstraintPtr->get<StateSoftConstraint>(absl::StrCat(footName, "_groundPenetration"));
    for (const std::unique_ptr<augmented::AugmentedPenaltyBase>& penalty : softConstraint.getPenalty().getPenaltyPtrArray()) {
      ASSERT_NE(penalty, nullptr);
      // Zero cost AND zero gradient for a foot resting exactly on the ground, and for one above it.
      EXPECT_DOUBLE_EQ(penalty->getValue(0.0, 0.0, 0.0), 0.0) << footName;
      EXPECT_DOUBLE_EQ(penalty->getDerivative(0.0, 0.0, 0.0), 0.0) << footName;
      EXPECT_DOUBLE_EQ(penalty->getValue(0.0, 0.0, 0.05), 0.0) << footName;
      // And a real cost below it.
      EXPECT_GT(penalty->getValue(0.0, 0.0, -0.01), 0.0) << footName;
      EXPECT_LT(penalty->getDerivative(0.0, 0.0, -0.01), 0.0) << footName;
    }
  }
}

TEST_F(ContactImplicitProblemTest, theTerrainHeightIsTheSameEverywhereItIsUsed) {
  const scalar_t terrainHeight = interface_->modelSettings().terrainHeight;
  ASSERT_NEAR(terrainHeight, kTestTerrainHeight, 1e-12)
      << "the test is vacuous unless the configured ground is away from the default of zero";
  for (const std::string& footName : contactNames()) {
    const ContactComplementarityConstraint& complementarity =
        problem()
            .softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_contactComplementarity"))
            .get<ContactComplementarityConstraint>();
    const GroundPenetrationConstraint& penetration =
        problem()
            .stateSoftConstraintPtr->get<StateSoftConstraint>(absl::StrCat(footName, "_groundPenetration"))
            .get<GroundPenetrationConstraint>();
    // One says a foot may not carry load above the ground and the other that it may not go below it. Two definitions
    // of where the ground is would make the pair incoherent, and there used to be two: this and the reference
    // manager's, which computed an estimate and discarded it.
    EXPECT_DOUBLE_EQ(complementarity.getTerrainHeight(), terrainHeight) << footName;
    EXPECT_DOUBLE_EQ(penetration.getTerrainHeight(), terrainHeight) << footName;
  }
  // And the reference manager, which owns the ground from start-up on and builds the swing trajectories on it, starts
  // from the same one. (That the two keep agreeing after a hot reload is testMpcParameterUpdaterModule's to show.)
  EXPECT_DOUBLE_EQ(interface_->getSwitchedModelReferenceManagerPtr()->getTerrainHeight(), terrainHeight);
  EXPECT_DOUBLE_EQ(interface_->getSwitchedModelReferenceManagerPtr()->getAppliedTerrainHeight(), terrainHeight);
}

TEST_F(ContactImplicitProblemTest, theSlipTermUsesTheConfiguredReferencesInTheirOwnUnits) {
  const ModelSettings::ContactImplicitConfig& config = interface_->modelSettings().contactImplicitConfig;
  for (const std::string& footName : contactNames()) {
    const ForceWeightedSlipConstraint& slip =
        problem()
            .softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_forceWeightedSlip"))
            .get<ForceWeightedSlipConstraint>();
    const vector3_t inverseReferences = slip.getInverseTwistReference();
    EXPECT_NEAR(inverseReferences(0), 1.0 / config.velocityReference, 1e-9) << footName;
    EXPECT_NEAR(inverseReferences(1), 1.0 / config.velocityReference, 1e-9) << footName;
    // The yaw row is a rate, and must be normalized by the ANGULAR reference: sharing the linear one declared one
    // rad/s to be exactly as bad as one m/s, which is a statement about SI units rather than about the robot.
    EXPECT_NEAR(inverseReferences(2), 1.0 / config.angularVelocityReference, 1e-9) << footName;
    EXPECT_NE(config.velocityReference, config.angularVelocityReference) << "the test is vacuous if the two agree";
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The soft `normal_velocity` term: the swing-foot vertical servo, priced instead of imposed.
//
// Removing the HARD constraint is necessary - it fixes the whole height profile of a scheduled swing, so the solver
// can neither land early nor late - but removing it with nothing in its place leaves no term with the authority to
// lift a foot at all. The only remaining vertical term is task_space_foot_cost_weights.pos_z at 150, against the
// leg-joint entries of Q whose reference posture is the foot ON THE FLOOR. The robot shuffles.
// ---------------------------------------------------------------------------------------------------------------

TEST_F(ContactImplicitFormulationTest, normalVelocityMayBeHardOrSoftButNotBoth) {
  const std::string taskFile = writeTaskFile("normalVelocityBoth", {"normal_velocity"}, {"joint_limits", "normal_velocity"});
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_FALSE(tasks.ok());
  EXPECT_NE(std::string(tasks.status().message()).find("normal_velocity"), std::string::npos) << tasks.status().message();
}

TEST_F(ContactImplicitFormulationTest, theSoftNormalVelocityIsAcceptedAlongsideTheContactImplicitTerms) {
  // The hard form is refused with the contact-implicit terms; the soft form is exactly what should replace it, so the
  // loader must NOT refuse it. Getting this backwards would leave the formulation with no way to lift a foot.
  const std::string taskFile = writeTaskFile("softNormalVelocity", {}, contactImplicitSoftConstraints());
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, /*verbose=*/false);
  ASSERT_TRUE(tasks.ok()) << tasks.status().message();
  EXPECT_TRUE(tasks->hasSoftConstraint(MpcSoftConstraintType::NormalVelocity));
  EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::NormalVelocity));
  EXPECT_TRUE(usesContactImplicitFormulation(*tasks));
}

TEST_F(ContactImplicitProblemTest, theSoftNormalVelocityTermIsBuiltAndTheHardOneIsNot) {
  for (const std::string& footName : contactNames()) {
    EXPECT_NO_THROW(problem().softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_normalVelocitySoft"))) << footName;
    // The same row must NOT also be an equality, or the swing height is pinned again.
    EXPECT_THROW(problem().equalityConstraintPtr->get<StateInputConstraint>(absl::StrCat(footName, "_normalVelocity")), std::out_of_range)
        << footName;
  }
}

TEST_F(ContactImplicitProblemTest, theSoftNormalVelocityPricesAFootThatFailsToLeaveTheGroundAndOnlyDuringItsSwing) {
  // The residual is v_z - zdot_ref - positionErrorGain_z * (z_ref - z). A foot sitting still on the ground during a
  // scheduled swing therefore carries a non-zero residual, which is precisely the pressure to lift that removing the
  // hard constraint took away. The term must also be swing-only, so it never fights a planted stance foot, and its
  // weight must be the one the task file configures - here a value that is neither the shipped one nor the default.
  scheduleOneSwing(CONTACT_LEFT_INDEX);
  const scalar_t swingTime = 0.5 * (kSwingStart + kSwingEnd);
  const scalar_t stanceTime = kSwingEnd + 0.3;
  const vector_t& state = interface_->getInitialState();
  // No joint motion and, from the initial state, no base motion: both feet are planted.
  const vector_t input = vector_t::Zero(interface_->getEffectiveMpcRobotModel().getInputDim());
  const TargetTrajectories noTarget;

  ASSERT_DOUBLE_EQ(interface_->modelSettings().footConstraintConfig.normalVelocitySoftConstraintWeight, kTestNormalVelocityWeight)
      << "the test task file's weight did not reach ModelSettings";

  StateInputSoftConstraint& swingFootTerm =
      problem().softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(contactNames()[CONTACT_LEFT_INDEX], "_normalVelocitySoft"));
  StateInputSoftConstraint& stanceFootTerm =
      problem().softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(contactNames()[CONTACT_RIGHT_INDEX], "_normalVelocitySoft"));
  EXPECT_EQ(swingFootTerm.get<NormalVelocityConstraintCppAd>().getNumConstraints(swingTime), 1U) << "one row, the vertical servo";

  // Gated on the schedule: active for the foot the plan wants in the air, and only while it does.
  EXPECT_TRUE(swingFootTerm.isActive(swingTime));
  EXPECT_FALSE(swingFootTerm.isActive(stanceTime));
  EXPECT_FALSE(stanceFootTerm.isActive(swingTime));

  // Priced: the planted swing foot has a residual, and the cost is the CONFIGURED weight's quadratic in it.
  problem().preComputationPtr->request(Request::Cost + Request::SoftConstraint, swingTime, state, input);
  const vector_t residual =
      swingFootTerm.get<NormalVelocityConstraintCppAd>().getValue(swingTime, state, input, *problem().preComputationPtr);
  ASSERT_EQ(residual.size(), 1);
  EXPECT_GT(std::abs(residual(0)), 1.0e-2) << "a foot that stays down mid-swing must be told to lift";
  EXPECT_NEAR(swingFootTerm.getValue(swingTime, state, input, noTarget, *problem().preComputationPtr),
              0.5 * kTestNormalVelocityWeight * residual(0) * residual(0), 1e-9 * (1.0 + kTestNormalVelocityWeight));
}

}  // namespace
}  // namespace ocs2::humanoid
