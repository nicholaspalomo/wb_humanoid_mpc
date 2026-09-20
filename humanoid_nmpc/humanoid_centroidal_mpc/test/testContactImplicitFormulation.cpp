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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/penalties/penalties/SquaredHingePenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "absl/strings/str_cat.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/constraint/NormalVelocityConstraintCppAd.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/constraint/BasisScalingNonNegativityConstraint.h"
#include "humanoid_common_mpc/constraint/ContactComplementarityConstraint.h"
#include "humanoid_common_mpc/constraint/ForceWeightedSlipConstraint.h"
#include "humanoid_common_mpc/constraint/GroundPenetrationConstraint.h"

namespace ocs2::humanoid {
namespace {

/**
 * The contact-implicit formulation, assembled end to end by CentroidalMpcInterface.
 *
 * It ships DISABLED (the task file lists zero_wrench / zero_velocity / normal_velocity and comments the three
 * contact-implicit terms out), which is correct - it changes the closed loop and has not been validated in simulation.
 * But it also meant the entire assembly path had NO coverage whatsoever: the one existing test that reaches for these
 * terms, MpcParameterUpdaterModuleTest.ContactImplicitTuningReachesEveryTerm, calls GTEST_SKIP() because they are not
 * in the shipped configuration. That is how every one of the defects this file now pins went unnoticed.
 *
 * So these tests write a task file with the formulation switched ON and build the real interface from it.
 */
class ContactImplicitFormulationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    shippedTaskFile_ = configDir + "/config/mpc/task.yaml";
    referenceFile_ = configDir + "/config/command/reference.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";
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
    const size_t keyPos = content.find("\n" + key);
    EXPECT_NE(keyPos, std::string::npos) << "the task file has no " << key << " block";
    if (keyPos == std::string::npos) return content;
    const size_t blockStart = keyPos + 1;
    size_t blockEnd = content.find("\n\n", blockStart);
    if (blockEnd == std::string::npos) blockEnd = content.size();

    std::string block = key + "\n";
    for (const std::string& entry : entries) {
      block += "  - " + entry + "\n";
    }
    return content.substr(0, blockStart) + block + content.substr(blockEnd + 1);
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

  /** The whole formulation switched on: no schedule-gated hard constraints, the three soft ones in. */
  std::string writeContactImplicitTaskFile(const std::string& name) {
    const std::string path = writeTaskFile(name, {}, contactImplicitSoftConstraints());
    rewriteScalarKey(path, "terrainHeight", kTestTerrainHeight);
    return path;
  }

  /** Rewrites a top-level scalar key of an already written task file. */
  static void rewriteScalarKey(const std::string& path, const std::string& key, scalar_t value) {
    std::ifstream in(path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    std::string content = buffer.str();
    const size_t keyPos = content.find("\n" + key + ":");
    ASSERT_NE(keyPos, std::string::npos) << "the task file has no top-level " << key;
    const size_t valueStart = keyPos + 1 + key.size() + 1;
    const size_t lineEnd = content.find('\n', valueStart);
    content.replace(valueStart, lineEnd - valueStart, " " + std::to_string(value));
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
  // zero_wrench and zero_velocity ARE dropped, so the two checks that already existed pass and this one is reached:
  // otherwise the loader would refuse on zero_wrench first and the test would prove nothing about normal_velocity.
  const std::string taskFile = writeTaskFile("normalVelocityKept", {"normal_velocity"}, contactImplicitSoftConstraints());
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, false);
  ASSERT_FALSE(tasks.ok()) << "normal_velocity fixes the whole height profile of the swing and must be refused";
  EXPECT_NE(std::string(tasks.status().message()).find("normal_velocity"), std::string::npos) << tasks.status().message();
}

TEST_F(ContactImplicitFormulationTest, theTaskFileInTheWorkingTreeIsSelfConsistent) {
  // Whatever the file is currently set to - the formulation ships off, but an engineer testing it has it on - the
  // loader must accept it and the contact-constraint gate must agree with it. Asserting the formulation is OFF here
  // would turn the whole suite red the moment somebody tried the thing it exists to support; that rule is about what
  // gets committed, not about what is in the working tree, and it is stated in
  // humanoid_nmpc/docs/contact_implicit_mpc/README.md instead.
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(shippedTaskFile_, false);
  ASSERT_TRUE(tasks.ok()) << tasks.status().message();

  // The gate follows `zero_wrench`, and nothing else.
  EXPECT_EQ(contactConstraintsAreScheduleGated(*tasks), tasks->hasHardConstraint(MpcHardConstraintType::ZeroWrench));

  // And the combinations the loader refuses really are absent, since it accepted the file.
  if (usesContactImplicitFormulation(*tasks)) {
    EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::ZeroWrench));
    EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::ZeroVelocity));
    EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::NormalVelocity));
  }
}

TEST_F(ContactImplicitFormulationTest, theContactImplicitTaskFileLoadsCleanly) {
  const std::string taskFile = writeContactImplicitTaskFile("loads");
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, false);
  ASSERT_TRUE(tasks.ok()) << tasks.status().message();
  EXPECT_TRUE(usesContactImplicitFormulation(*tasks));
  EXPECT_FALSE(contactConstraintsAreScheduleGated(*tasks));
  EXPECT_FALSE(tasks->hasHardConstraint(MpcHardConstraintType::NormalVelocity));
}

TEST_F(ContactImplicitFormulationTest, basisVectorInputsWithoutTheWrenchConeKeyAreRefused) {
  // In basis-vector mode CentroidalMpcInterface builds the non-negativity barrier on the basis scalings only inside
  // the `contact_wrench_cone` branch. Without zero_wrench to pin the swing foot's scalings to zero, a task file that
  // omits that key would leave them with no lower bound at all - and a negative scaling is an adhesive,
  // outside-the-cone wrench that the sign-blind complementarity product would not object to.
  std::vector<std::string> soft = contactImplicitSoftConstraints();
  soft.erase(std::remove(soft.begin(), soft.end(), std::string("contact_wrench_cone")), soft.end());
  const std::string taskFile = writeTaskFile("noWrenchCone", {}, soft);
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile_, referenceFile_);
  ASSERT_FALSE(interface.ok());
  EXPECT_NE(std::string(interface.status().message()).find("contact_wrench_cone"), std::string::npos) << interface.status().message();
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
  // On the shipped DRC Atlas (useContactBasisVectorInputs: true) this term is the ONLY lower bound on the contact
  // wrench, because the explicit wrench cone is skipped in favour of the structural guarantee of lambda >= 0. Gated, a
  // foot the schedule calls a swing foot had no bound at all.
  ASSERT_TRUE(interface_->getBasisDecoratorPtr() != nullptr) << "this robot is expected to run basis-vector inputs";
  for (const std::string& footName : contactNames()) {
    const BasisScalingNonNegativityConstraint& barrier =
        problem().costPtr->get<BasisScalingNonNegativityConstraint>(absl::StrCat(footName, "_basisNonNegativity"));
    EXPECT_FALSE(barrier.isScheduleGated()) << footName;
  }
}

TEST_F(ContactImplicitProblemTest, groundPenetrationIsCheckedAtEveryCornerOfTheFootprint) {
  // The sole CENTRE alone is not enough: this formulation deliberately leaves the foot's rocking rates free, so a foot
  // pitched about a centre held at ground level buries its toe for nothing. The DRC Atlas footprint has four corners
  // 0.12 m fore and aft of the centre.
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
  // sole centre, and a foot rocked onto its heel was then "airborne" to one and "touching" to the other.
  const ModelSettings::ContactImplicitConfig& config = interface_->modelSettings().contactImplicitConfig;
  const vector_t& state = interface_->getInitialState();
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
    // smallest of them and that value plus log(N) * gapSmoothing. Measured at the sole centre it would not be.
    const vector_t clearances = penetration.getValue(0.0, state, preComp);
    ASSERT_EQ(clearances.size(), static_cast<long>(penetration.getNumPoints())) << footName;
    const scalar_t bound = std::log(static_cast<scalar_t>(penetration.getNumPoints())) * config.gapSmoothing;
    EXPECT_GE(complementarity.getGap(state), clearances.minCoeff() - 1e-12) << footName;
    EXPECT_LE(complementarity.getGap(state), clearances.minCoeff() + bound + 1e-12) << footName;
  }
}

TEST_F(ContactImplicitProblemTest, groundPenetrationIsAHingeAndNotALogBarrier) {
  // A relaxed log barrier never reaches zero: at h = 0 its derivative is -2*mu/delta, constant and upward, so it pushes
  // every foot off the ground and the complementarity term has to hold it down. Balancing those two put a foot at half
  // body weight - i.e. both feet, throughout double support - a centimetre above the floor.
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
    // The yaw row is a rate, and must be normalised by the ANGULAR reference: sharing the linear one declared one
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
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, false);
  ASSERT_FALSE(tasks.ok());
  EXPECT_NE(std::string(tasks.status().message()).find("normal_velocity"), std::string::npos) << tasks.status().message();
}

TEST_F(ContactImplicitFormulationTest, theSoftNormalVelocityIsAcceptedAlongsideTheContactImplicitTerms) {
  // The hard form is refused with the contact-implicit terms; the soft form is exactly what should replace it, so the
  // loader must NOT refuse it. Getting this backwards would leave the formulation with no way to lift a foot.
  const std::string taskFile = writeTaskFile("softNormalVelocity", {}, contactImplicitSoftConstraints());
  const absl::StatusOr<MpcFormulationTasks> tasks = loadMpcFormulationTasks(taskFile, false);
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

TEST_F(ContactImplicitProblemTest, theSoftNormalVelocityPricesAFootThatFailsToLeaveTheGround) {
  // The residual is v_z - zdot_ref - positionErrorGain_z * (z_ref - z). A foot sitting still on the ground during a
  // scheduled swing therefore carries a non-zero residual, which is precisely the pressure to lift that removing the
  // hard constraint took away. The term must also be swing-only, so it never fights a planted stance foot.
  for (const std::string& footName : contactNames()) {
    StateInputSoftConstraint& softConstraint =
        problem().softConstraintPtr->get<StateInputSoftConstraint>(absl::StrCat(footName, "_normalVelocitySoft"));
    const NormalVelocityConstraintCppAd& row = softConstraint.get<NormalVelocityConstraintCppAd>();
    // Gated on the schedule: this is a REFERENCE for the foot the plan wants in the air, not a bound that holds
    // everywhere. A stance foot must be left alone.
    EXPECT_TRUE(row.getActive()) << footName;
    EXPECT_EQ(row.getNumConstraints(0.0), 1U) << footName << ": one row, the vertical velocity servo";
  }

  // And the weight the interface gave it is the configured one, not a default that silently ignores the task file.
  EXPECT_DOUBLE_EQ(interface_->modelSettings().footConstraintConfig.normalVelocitySoftConstraintWeight, 500.0);
}

}  // namespace
}  // namespace ocs2::humanoid
