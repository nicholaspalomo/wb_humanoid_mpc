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

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/model_data/ModelData.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/approximate_model/LinearQuadraticApproximator.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"

#include "humanoid_centroidal_mpc/CentroidalContactSet.h"
#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_centroidal_mpc/initialization/CentroidalWeightCompInitializer.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/ArmSwing.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "support/FourPointContactSet.h"
#include "support/TypedConfigFiles.h"

/**
 * The centroidal MPC on a contact set of four 3-DoF point contacts and no 6-DoF
 * contact, the contact set of a point-footed quadruped: the toe and the heel of
 * each DRC Atlas sole, listed front-left, front-right, hind-left, hind-right.
 *  - The task file selects only terms that run on any contact set, so
 *    checkCentroidalContactSet() passes and the interface builds.
 *  - The input is the four contact forces, then the joint velocities.
 *  - A mode number is the bitmask of the active contacts, contact 0 the most
 *    significant bit: 0b1111 is all four, 0b1001 and 0b0110 are the two
 *    diagonal pairs of a trot.
 *  - The LQ approximation of the problem is finite, its dynamics take each
 *    contact's force, and its equality constraints hold each inactive
 *    contact's force at zero and leave the active ones free.
 *  - The weight-compensating initializer shares the weight among the active
 *    contacts.
 *  - The gait schedule's stance is all four contacts.
 *  - Every term written for two feet is refused by name on this set.
 */
namespace ocs2::humanoid {
namespace {

using four_point_set::fourPointConfig;
using four_point_set::kAllFour;
using four_point_set::kFrontLeft;
using four_point_set::kFrontLeftHindRight;
using four_point_set::kFrontRight;
using four_point_set::kFrontRightHindLeft;
using four_point_set::kHindLeft;
using four_point_set::kHindRight;

constexpr size_t kNumPoints = four_point_set::kNumContacts;
constexpr scalar_t kTolerance = 1.0e-9;

// The trot of the LQ tests: all four, one diagonal, the other, all four.
constexpr scalar_t kFirstLiftOffTime = 0.2;
constexpr scalar_t kDiagonalSwitchTime = 0.5;
constexpr scalar_t kLastTouchDownTime = 0.8;
constexpr scalar_t kFirstDiagonalTime = 0.35;
constexpr scalar_t kSecondDiagonalTime = 0.65;

/** The shipped DRC Atlas configuration. */
CentroidalMpcConfig loadAtlas() {
  return checkedValue(loadConfigOf(atlasFiles()));
}

/** The four-point task described by its legs LF, RF, LH and RH. */
mpc_config::TaskFile leggedFourPointTask() {
  return four_point_set::leggedFourPointConfig(loadAtlas()).task;
}

/** The model settings of `task` on the Atlas URDF. */
ModelSettings modelSettingsOf(const mpc_config::TaskFile& task) {
  return checkedValue(ModelSettings::Create(task, atlasFiles().urdfFile,
                                            "point_contact_set_",
                                            /*verbose=*/false));
}

/**
 * The four-point interface, built once for the whole suite: it generates the
 * CppAD libraries of the dynamics and of the four contact frames.
 */
class PointContactSetTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ =
        checkedValue(CentroidalMpcInterface::Create(
                         fourPointConfig(loadAtlas()), atlasFiles().urdfFile))
            .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  static CentroidalMpcInterface& interface() { return *interface_; }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable PointContactSetTest::interface_ = nullptr;

/**
 * Makes `schedule` the schedule of the interface's reference manager, sets a
 * standing target and runs preSolverRun(), the way the MPC loop prepares a
 * solve, and points the problem at that target.
 */
void prepareSolve(CentroidalMpcInterface& mpc, const ModeSchedule& schedule) {
  SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  const vector_t& state = mpc.getInitialState();
  const vector_t input = vector_t::Zero(
      static_cast<Eigen::Index>(mpc.getMpcRobotModel().getInputDim()));
  referenceManager.setTargetTrajectories(
      TargetTrajectories({0.0, 2.0}, {state, state}, {input, input}));
  referenceManager.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0, state,
                                schedule.modeAtTime(0.0));
  mpc.getOptimalControlProblemRef().targetTrajectoriesPtr =
      &referenceManager.getTargetTrajectories();
}

/** The trot of the LQ tests, every swing ending in a contact. */
ModeSchedule trot() {
  return ModeSchedule(
      {kFirstLiftOffTime, kDiagonalSwitchTime, kLastTouchDownTime},
      {kAllFour, kFrontLeftHindRight, kFrontRightHindLeft, kAllFour});
}

/** `mode` around time 0, bracketed by all four contacts. */
ModeSchedule constantMode(size_t mode) {
  constexpr scalar_t kBracket = 10.0;
  return ModeSchedule({-kBracket, kBracket}, {kAllFour, mode, kAllFour});
}

/** The weight-compensating input of the interface's initializer at `time`. */
vector_t weightCompensatingInputAt(CentroidalMpcInterface& mpc, scalar_t time) {
  const std::unique_ptr<CentroidalWeightCompInitializer> initializer(
      mpc.getInitializer().clone());
  vector_t input;
  vector_t nextState;
  initializer->compute(time, mpc.getInitialState(), time + 0.1, input,
                       nextState);
  return input;
}

TEST_F(PointContactSetTest, theContactSetIsTheFourPointsInOrder) {
  const ModelSettings& settings = interface().modelSettings();
  ASSERT_EQ(settings.numContacts(), kNumPoints);
  EXPECT_EQ(settings.contactNames[0], kFrontLeft);
  EXPECT_EQ(settings.contactNames[1], kFrontRight);
  EXPECT_EQ(settings.contactNames[2], kHindLeft);
  EXPECT_EQ(settings.contactNames[3], kHindRight);
  for (size_t i = 0; i < kNumPoints; ++i) {
    EXPECT_TRUE(settings.isPointContact(i)) << settings.contactNames[i];
  }
  EXPECT_TRUE(settings.contactNames6DoF.empty());
  EXPECT_FALSE(settings.isBipedContactSet());
  const CentroidalModelInfo& info = interface().getCentroidalModelInfo();
  EXPECT_EQ(info.numThreeDofContacts, kNumPoints);
  EXPECT_EQ(info.numSixDofContacts, 0U);
  HUMANOID_ASSERT_OK(
      checkCentroidalContactSet(settings, interface().config().task));
}

TEST_F(PointContactSetTest, theInputIsTheFourForcesThenTheJoints) {
  const CentroidalMpcRobotModel<scalar_t>& model =
      interface().getMpcRobotModel();
  EXPECT_EQ(model.getInputDim(), kForceDim * kNumPoints + model.getJointDim());
  EXPECT_EQ(model.getInputDim(), interface().getCentroidalModelInfo().inputDim);
  for (size_t i = 0; i < kNumPoints; ++i) {
    EXPECT_EQ(model.getContactWrenchStartIndices(i), kForceDim * i);
    EXPECT_EQ(model.getContactInputDim(i), kForceDim);
  }
  EXPECT_EQ(model.getJointVelocitiesStartindex(), kForceDim * kNumPoints);
}

// The gait schedule stands on all four contacts: the reference file's STANCE
// reads as 0b1111, and the stance it appends after a given schedule runs out
// and starts the schedule with is 0b1111, not the biped's 0b0011 (the hind
// pair alone).
TEST_F(PointContactSetTest, theGaitScheduleStandsOnAllFourContacts) {
  CentroidalMpcInterface& mpc = interface();
  GaitSchedule& gaitSchedule =
      *mpc.getSwitchedModelReferenceManagerPtr()->getGaitSchedule();
  EXPECT_EQ(gaitSchedule.modeLayout().numContacts, kNumPoints);
  EXPECT_EQ(gaitSchedule.modeLayout().stanceMode, kAllFour);
  // The dummy simulator starts from this mode outside a tracked plan
  // (CentroidalMpcDummySimMain.cpp).
  EXPECT_EQ(gaitModeLayoutOf(mpc.modelSettings()).stanceMode, kAllFour);

  gaitSchedule.reset();
  for (const size_t mode : gaitSchedule.getCurrentModeSchedule().modeSequence) {
    EXPECT_EQ(mode, kAllFour) << "the reference file's STANCE";
  }

  gaitSchedule.updateModeSchedule(
      ModeSchedule({kFirstLiftOffTime, kDiagonalSwitchTime},
                   {kAllFour, kFrontLeftHindRight, kFrontRightHindLeft}));
  constexpr scalar_t kPastTheSchedule = 2.0;
  const ModeSchedule extended = gaitSchedule.getModeSchedule(
      /*lowerBoundTime=*/kFirstDiagonalTime, kPastTheSchedule);
  ASSERT_FALSE(extended.modeSequence.empty());
  EXPECT_EQ(extended.modeSequence.front(), kAllFour)
      << "the schedule starts in stance";
  EXPECT_EQ(extended.modeSequence.back(), kAllFour) << "the appended stance";
  EXPECT_EQ(extended.modeAtTime(kPastTheSchedule), kAllFour);
  gaitSchedule.reset();
}

TEST_F(PointContactSetTest, aTrotModeIsTheMaskOfItsDiagonalPair) {
  CentroidalMpcInterface& mpc = interface();
  prepareSolve(mpc, trot());
  const SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  ContactMask frontLeftHindRight(kNumPoints);
  frontLeftHindRight.set(/*contactIndex=*/0, /*active=*/true);
  frontLeftHindRight.set(/*contactIndex=*/3, /*active=*/true);
  EXPECT_EQ(frontLeftHindRight.modeNumber(), kFrontLeftHindRight);
  EXPECT_EQ(referenceManager.getContactFlags(kFirstDiagonalTime),
            frontLeftHindRight);
  EXPECT_EQ(referenceManager.getContactFlags(kSecondDiagonalTime),
            ContactMask::fromModeNumber(kFrontRightHindLeft, kNumPoints));
  EXPECT_TRUE(referenceManager.getContactFlags(/*time=*/0.0).all());
  EXPECT_TRUE(referenceManager.isInStancePhase(/*time=*/0.0));
  EXPECT_FALSE(referenceManager.isInStancePhase(kFirstDiagonalTime));
  EXPECT_TRUE(referenceManager.isInContact(kSecondDiagonalTime,
                                           /*contactIndex=*/1));
  EXPECT_FALSE(referenceManager.isInContact(kSecondDiagonalTime,
                                            /*contactIndex=*/0));
}

TEST_F(PointContactSetTest, theWeightIsSharedAmongTheActiveContacts) {
  CentroidalMpcInterface& mpc = interface();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  const scalar_t weight =
      mpc.getCentroidalModelInfo().robotMass * kGravityMagnitude;

  prepareSolve(mpc, constantMode(kAllFour));
  const vector_t allFour = weightCompensatingInputAt(mpc, /*time=*/0.0);
  for (size_t i = 0; i < kNumPoints; ++i) {
    EXPECT_NEAR(model.getForceContactEwrtParam(allFour, i).z(), 0.25 * weight,
                kTolerance)
        << "contact " << i;
  }
  EXPECT_TRUE(allFour.tail(model.getJointDim()).isZero(0.0));

  prepareSolve(mpc, constantMode(kFrontRightHindLeft));
  const vector_t diagonal = weightCompensatingInputAt(mpc, /*time=*/0.0);
  const ContactMask active =
      ContactMask::fromModeNumber(kFrontRightHindLeft, kNumPoints);
  for (size_t i = 0; i < kNumPoints; ++i) {
    if (active[i]) {
      EXPECT_NEAR(model.getForceContactEwrtParam(diagonal, i).z(), 0.5 * weight,
                  kTolerance)
          << "contact " << i;
    } else {
      EXPECT_TRUE(model.getForceContactEwrtParam(diagonal, i).isZero(0.0))
          << "contact " << i;
    }
  }
  // The reference of the input cost shares it the same way.
  const vector_t desired =
      mpc.getSwitchedModelReferenceManagerPtr()->getDesiredInput(
          TargetTrajectories({0.0}, {mpc.getInitialState()}, {diagonal}),
          mpc.getInitialState(), /*time=*/0.0);
  EXPECT_TRUE(desired.isApprox(diagonal, kTolerance));
}

// The input reference shares the weight among the active contacts of any
// mode of the set: three of four here.
TEST_F(PointContactSetTest, theDesiredInputSharesTheWeightAmongThreeContacts) {
  CentroidalMpcInterface& mpc = interface();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  const scalar_t weight =
      mpc.getCentroidalModelInfo().robotMass * kGravityMagnitude;
  constexpr size_t kAllButFrontRight = 0b1011;
  prepareSolve(mpc, constantMode(kAllButFrontRight));
  const vector_t desired =
      mpc.getSwitchedModelReferenceManagerPtr()->getDesiredInput(
          mpc.getSwitchedModelReferenceManagerPtr()->getTargetTrajectories(),
          mpc.getInitialState(), /*time=*/0.0);
  const ContactMask active =
      ContactMask::fromModeNumber(kAllButFrontRight, kNumPoints);
  ASSERT_EQ(active.numActive(), 3U);
  for (size_t i = 0; i < kNumPoints; ++i) {
    // On wrench inputs the input parameterization is the world frame.
    const vector3_t refForceContactEwrtWorld =
        model.getForceContactEwrtParam(desired, i);
    if (active[i]) {
      EXPECT_NEAR(refForceContactEwrtWorld.z(), weight / 3.0, kTolerance)
          << "contact " << i;
      EXPECT_TRUE(refForceContactEwrtWorld.head<2>().isZero(0.0))
          << "contact " << i;
    } else {
      EXPECT_TRUE(refForceContactEwrtWorld.isZero(0.0)) << "contact " << i;
    }
  }
  EXPECT_TRUE(desired.tail(model.getJointDim()).isZero(0.0));
}

// The procedural arm swing follows the biped's gait phase: a reload that
// selects it on this set is refused with the generator and the count, and the
// running settings stay.
TEST_F(PointContactSetTest, aReloadedProceduralArmSwingIsRefused) {
  SwitchedModelReferenceManager& referenceManager =
      *interface().getSwitchedModelReferenceManagerPtr();
  ASSERT_TRUE(referenceManager.isArmSwingReferenceActive())
      << "the Atlas names its arm joints";
  ASSERT_EQ(referenceManager.getArmSwingSettings().generator,
            ArmSwingGenerator::kNone);
  ArmSwingSettings procedural;
  procedural.generator = ArmSwingGenerator::kProcedural;
  const absl::Status refused = referenceManager.setArmSwingSettings(procedural);
  EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << refused;
  EXPECT_TRUE(
      absl::StrContains(refused.message(), "arm_swing.generator 'procedural'"))
      << refused;
  EXPECT_TRUE(absl::StrContains(refused.message(), "has 4 contacts"))
      << refused;
  EXPECT_EQ(referenceManager.getArmSwingSettings().generator,
            ArmSwingGenerator::kNone);
  ArmSwingSettings none;
  none.generator = ArmSwingGenerator::kNone;
  HUMANOID_ASSERT_OK(referenceManager.setArmSwingSettings(none));
}

// The DCM terminal cost blends a left and a right foot: built on this set by
// hand, past checkCentroidalContactSet(), it is refused with its name and the
// count before anything is generated.
TEST_F(PointContactSetTest, theDcmTerminalCostRefusesTheSet) {
  CentroidalMpcInterface& mpc = interface();
  constexpr scalar_t kNomPosComRtFeetmeanEwrtWorldZ = 0.9;
  const absl::StatusOr<std::unique_ptr<DcmTerminalCost>> cost =
      DcmTerminalCost::Create(
          *mpc.getSwitchedModelReferenceManagerPtr(), DcmTerminalCost::Config{},
          kNomPosComRtFeetmeanEwrtWorldZ, mpc.getPinocchioInterface(),
          mpc.getEffectiveMpcRobotModelAD(), DcmTerminalCost::kLibraryName,
          mpc.modelSettings());
  ASSERT_FALSE(cost.ok());
  EXPECT_EQ(cost.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "dcm_terminal_cost"))
      << cost.status();
  EXPECT_TRUE(absl::StrContains(cost.status().message(), "has 4 contacts"))
      << cost.status();
}

/** The largest magnitude of the force columns of contact `contactIndex`. */
scalar_t contactColumnsMagnitude(const matrix_t& jacobianWrtInput,
                                 const CentroidalMpcRobotModel<scalar_t>& model,
                                 size_t contactIndex) {
  if (jacobianWrtInput.rows() == 0) return 0.0;
  return jacobianWrtInput
      .middleCols<kForceDim>(static_cast<Eigen::Index>(
          model.getContactWrenchStartIndices(contactIndex)))
      .cwiseAbs()
      .maxCoeff();
}

TEST_F(PointContactSetTest, theLqApproximationGatesEachContactByItsMode) {
  CentroidalMpcInterface& mpc = interface();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  const scalar_t mass = mpc.getCentroidalModelInfo().robotMass;
  prepareSolve(mpc, trot());
  OptimalControlProblem& problem = mpc.getOptimalControlProblemRef();

  for (const scalar_t time : {kFirstDiagonalTime, kSecondDiagonalTime}) {
    SCOPED_TRACE(absl::StrCat("time ", time));
    const vector_t input = weightCompensatingInputAt(mpc, time);
    const ModelData lq =
        approximateIntermediateLQ(problem, time, mpc.getInitialState(), input);
    EXPECT_TRUE(std::isfinite(lq.cost.f));
    EXPECT_TRUE(lq.cost.dfdx.allFinite());
    EXPECT_TRUE(lq.cost.dfdu.allFinite());
    EXPECT_TRUE(lq.cost.dfdxx.allFinite());
    EXPECT_TRUE(lq.cost.dfdux.allFinite());
    EXPECT_TRUE(lq.cost.dfduu.allFinite());
    EXPECT_TRUE(lq.dynamics.f.allFinite());
    EXPECT_TRUE(lq.dynamics.dfdx.allFinite());
    EXPECT_TRUE(lq.dynamics.dfdu.allFinite());
    EXPECT_TRUE(lq.stateInputEqConstraint.f.allFinite());
    EXPECT_TRUE(lq.stateInputEqConstraint.dfdx.allFinite());
    EXPECT_TRUE(lq.stateInputEqConstraint.dfdu.allFinite());

    const ContactMask active =
        mpc.getSwitchedModelReferenceManagerPtr()->getContactFlags(time);
    ASSERT_EQ(active.numActive(), 2U);
    for (size_t i = 0; i < kNumPoints; ++i) {
      SCOPED_TRACE(mpc.modelSettings().contactNames[i]);
      // The dynamics take every contact's force: m hdot_lin = sum f_i + m g.
      const Eigen::Index start =
          static_cast<Eigen::Index>(model.getContactWrenchStartIndices(i));
      const matrix3_t contactBlockOfDfdu =
          lq.dynamics.dfdu.block<kPositionDim, kForceDim>(0, start);
      EXPECT_TRUE(contactBlockOfDfdu.isApprox(matrix3_t::Identity() / mass,
                                              kTolerance));
      // The zero-force rows act on an inactive contact's force alone.
      const scalar_t equality =
          contactColumnsMagnitude(lq.stateInputEqConstraint.dfdu, model, i);
      if (active[i]) {
        EXPECT_EQ(equality, 0.0);
      } else {
        EXPECT_GT(equality, 0.0);
      }
    }
  }
}

TEST_F(PointContactSetTest, aStanceOnAllFourHasNoZeroForceRows) {
  CentroidalMpcInterface& mpc = interface();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  prepareSolve(mpc, constantMode(kAllFour));
  const vector_t input = weightCompensatingInputAt(mpc, /*time=*/0.0);
  const ModelData lq =
      approximateIntermediateLQ(mpc.getOptimalControlProblemRef(), /*time=*/0.0,
                                mpc.getInitialState(), input);
  EXPECT_TRUE(std::isfinite(lq.cost.f));
  for (size_t i = 0; i < kNumPoints; ++i) {
    EXPECT_EQ(contactColumnsMagnitude(lq.stateInputEqConstraint.dfdu, model, i),
              0.0)
        << "contact " << i;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The four points with the two feet: the biped's mode numbers are the feet's.
// ---------------------------------------------------------------------------------------------------------------

TEST(PointContactSetWithFeet, theBipedStanceIsTheTwoFeetListedLast) {
  mpc_config::TaskFile task = fourPointConfig(loadAtlas()).task;
  const mpc_config::TaskFile atlas = loadAtlas().task;
  task.model_settings.contact_names_6dof =
      atlas.model_settings.contact_names_6dof;
  task.model_settings.contact_parent_joint_names =
      atlas.model_settings.contact_parent_joint_names;
  const ModelSettings settings = modelSettingsOf(task);
  constexpr size_t kNumContacts = kNumPoints + 2;
  ASSERT_EQ(settings.numContacts(), kNumContacts);
  EXPECT_EQ(settings.contactNames[4], "foot_l_contact");
  EXPECT_EQ(settings.contactNames[5], "foot_r_contact");
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, task));

  ContactMask feet(kNumContacts);
  feet.set(/*contactIndex=*/4, /*active=*/true);
  feet.set(/*contactIndex=*/5, /*active=*/true);
  EXPECT_EQ(ContactMask::fromModeNumber(kBipedModeStance, kNumContacts), feet);
  for (size_t i = 0; i < kNumContacts; ++i) {
    EXPECT_EQ(contactIsActiveInMode(kBipedModeStance, i, kNumContacts),
              i >= kNumPoints)
        << "contact " << i;
  }
  EXPECT_TRUE(ContactMask::fromModeNumber(kBipedModeLf, kNumContacts)[4]);
  EXPECT_FALSE(ContactMask::fromModeNumber(kBipedModeLf, kNumContacts)[5]);
  // The four points standing and the feet in the air.
  EXPECT_EQ(
      ContactMask::fromModeNumber(kAllFour << 2, kNumContacts).numActive(),
      kNumPoints);
  EXPECT_FALSE(ContactMask::fromModeNumber(kAllFour << 2, kNumContacts)[4]);
}

// ---------------------------------------------------------------------------------------------------------------
// What is written for two feet is refused on the four points.
// ---------------------------------------------------------------------------------------------------------------

/** A term written for two 6-DoF feet, by its list. */
struct ListedTerm {
  std::vector<std::string> mpc_config::TaskFile::*absl_nonnull list;
  const char* absl_nonnull listName;
  const char* absl_nonnull term;
};

// The biped-only terms that run beside the zero-velocity stance constraint;
// the contact-implicit ones replace it and are checked below.
// LINT.IfChange(biped_only_terms)
// The terms of the three lists, refused on any non-biped set, and the swing
// terms, which run on a set its legs describe and are refused on a set
// without legs. The planner, the basis vectors, the heuristics (base-pose and
// wrench on any non-biped set, all of them on a set without legs) and the
// step_width foothold anchor are tested by their own cases below.
constexpr ListedTerm kBipedOnlyTerms[] = {
    {.list = &mpc_config::TaskFile::costs,
     .listName = "costs",
     .term = "external_torque_cost"},
    {.list = &mpc_config::TaskFile::costs,
     .listName = "costs",
     .term = "dcm_terminal_cost"},
    {.list = &mpc_config::TaskFile::soft_constraints,
     .listName = "soft_constraints",
     .term = "foot_collision"},
    {.list = &mpc_config::TaskFile::soft_constraints,
     .listName = "soft_constraints",
     .term = "foot_collision_cbf"},
    {.list = &mpc_config::TaskFile::hard_constraints,
     .listName = "hard_constraints",
     .term = "knee_joint_mimic"},
};
constexpr ListedTerm kSwingTerms[] = {
    {.list = &mpc_config::TaskFile::costs,
     .listName = "costs",
     .term = "task_space_foot_cost"},
    {.list = &mpc_config::TaskFile::soft_constraints,
     .listName = "soft_constraints",
     .term = "normal_velocity"},
    {.list = &mpc_config::TaskFile::hard_constraints,
     .listName = "hard_constraints",
     .term = "normal_velocity"},
};
constexpr std::array<absl::string_view, 3> kContactImplicitTerms = {
    "contact_complementarity", "force_weighted_slip", "ground_penetration"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/CentroidalContactSet.cpp:biped_only_terms)

/** `status` is InvalidArgument and its message says `selection`. */
void expectRefused(const absl::Status& status, absl::string_view selection) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), selection))
      << "expected the refusal to name " << selection << ": " << status;
  EXPECT_TRUE(absl::StrContains(status.message(), "not the biped's two 6-DoF"))
      << status;
}

TEST(PointContactSetRefusal, everyTermWrittenForTwoFeetIsRefusedByName) {
  const mpc_config::TaskFile points = fourPointConfig(loadAtlas()).task;
  const ModelSettings settings = modelSettingsOf(points);
  ASSERT_FALSE(settings.isLeggedContactSet());
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, points));
  for (const ListedTerm& listed : kBipedOnlyTerms) {
    mpc_config::TaskFile task = points;
    (task.*listed.list).push_back(listed.term);
    expectRefused(checkCentroidalContactSet(settings, task),
                  absl::StrCat(listed.listName, " '", listed.term, "'"));
  }
  // Without legs the swing terms have no leg to swing either.
  for (const ListedTerm& listed : kSwingTerms) {
    mpc_config::TaskFile task = points;
    (task.*listed.list).push_back(listed.term);
    const absl::Status refused = checkCentroidalContactSet(settings, task);
    expectRefused(refused,
                  absl::StrCat(listed.listName, " '", listed.term, "'"));
    EXPECT_TRUE(absl::StrContains(refused.message(), "model_settings.legs"))
        << refused;
  }
}

// On the set its legs describe, the swing terms run: the swing-foot cost and
// the normal-velocity servo of every contact, each alone and the cost with
// either servo (normal_velocity is listed as hard or as soft, not both). The
// biped-only terms are still refused.
TEST(PointContactSetRefusal, aLeggedSetRunsTheSwingTerms) {
  const mpc_config::TaskFile legged = leggedFourPointTask();
  const ModelSettings settings = modelSettingsOf(legged);
  ASSERT_TRUE(settings.isLeggedContactSet());
  for (const ListedTerm& listed : kSwingTerms) {
    mpc_config::TaskFile task = legged;
    (task.*listed.list).push_back(listed.term);
    HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, task));
  }
  for (const ListedTerm& servo : {kSwingTerms[1], kSwingTerms[2]}) {
    mpc_config::TaskFile task = legged;
    (task.*kSwingTerms[0].list).push_back(kSwingTerms[0].term);
    (task.*servo.list).push_back(servo.term);
    HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, task));
    for (const ListedTerm& listed : kBipedOnlyTerms) {
      mpc_config::TaskFile refusedTask = task;
      (refusedTask.*listed.list).push_back(listed.term);
      expectRefused(checkCentroidalContactSet(settings, refusedTask),
                    absl::StrCat(listed.listName, " '", listed.term, "'"));
    }
  }
}

TEST(PointContactSetRefusal, theContactImplicitTermsAreRefusedByName) {
  const mpc_config::TaskFile points = fourPointConfig(loadAtlas()).task;
  const ModelSettings settings = modelSettingsOf(points);
  mpc_config::TaskFile task = points;
  task.hard_constraints = {};
  // The contact-implicit formulation needs the soft normal velocity, itself
  // refused on this set.
  task.soft_constraints = {"joint_limits", "contact_wrench_cone",
                           "normal_velocity"};
  for (const absl::string_view term : kContactImplicitTerms) {
    task.soft_constraints.emplace_back(term);
  }
  const absl::Status refused = checkCentroidalContactSet(settings, task);
  for (const absl::string_view term : kContactImplicitTerms) {
    expectRefused(refused, absl::StrCat("soft_constraints '", term, "'"));
  }
}

TEST(PointContactSetRefusal, thePlannerTheBasisAndTheHeuristicsAreRefused) {
  const mpc_config::TaskFile points = fourPointConfig(loadAtlas()).task;
  const ModelSettings settings = modelSettingsOf(points);
  mpc_config::TaskFile planner = points;
  planner.contact_schedule_source = "contact_planner";
  expectRefused(checkCentroidalContactSet(settings, planner),
                "contact_schedule_source 'contact_planner'");
  mpc_config::TaskFile basis = points;
  basis.contact_input_parameterization = "basis_vectors";
  expectRefused(checkCentroidalContactSet(settings, basis),
                "contact_input_parameterization 'basis_vectors'");
  mpc_config::TaskFile heuristics = points;
  heuristics.locomotion_heuristics.foothold_heuristics = {
      "hip_centered_stepping"};
  expectRefused(checkCentroidalContactSet(settings, heuristics),
                "locomotion_heuristics");
}

TEST(PointContactSetRefusal, aLeggedSetRunsTheFootholdHeuristicsAlone) {
  mpc_config::TaskFile task = leggedFourPointTask();
  task.nominal_foothold.anchor = "nominal_stance";
  task.locomotion_heuristics.foothold_heuristics = {"hip_centered_stepping",
                                                    "translational_stepping"};
  const ModelSettings settings = modelSettingsOf(task);
  ASSERT_TRUE(settings.isLeggedContactSet());
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, task));

  mpc_config::TaskFile shaped = task;
  shaped.locomotion_heuristics.base_pose_heuristics = {"periodic_orientation"};
  expectRefused(
      checkCentroidalContactSet(settings, shaped),
      "locomotion_heuristics.base_pose_heuristics 'periodic_orientation'");
  mpc_config::TaskFile scaled = task;
  scaled.locomotion_heuristics.wrench_heuristics = {"impulse_scaling"};
  expectRefused(checkCentroidalContactSet(settings, scaled),
                "locomotion_heuristics.wrench_heuristics 'impulse_scaling'");
}

TEST(PointContactSetRefusal, theStepWidthAnchorIsRefusedWithAStepWidth) {
  mpc_config::TaskFile task = leggedFourPointTask();
  task.nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.3;
  const ModelSettings settings = modelSettingsOf(task);
  const absl::Status refused = checkCentroidalContactSet(settings, task);
  expectRefused(refused, "nominal_foothold.anchor 'step_width'");
  EXPECT_TRUE(absl::StrContains(refused.message(), "a contact set of 4"))
      << refused;
  // Without a width the anchor does nothing on this set, and nothing is
  // refused; nominal_stance takes a width without reading it.
  task.nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.0;
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(modelSettingsOf(task), task));
  task.nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.3;
  task.nominal_foothold.anchor = "nominal_stance";
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(modelSettingsOf(task), task));
}

TEST(PointContactSetRefusal, theInterfaceRefusesBeforeItBuildsAModel) {
  CentroidalMpcConfig config = fourPointConfig(loadAtlas());
  config.task.soft_constraints.push_back("foot_collision");
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(config, atlasFiles().urdfFile);
  ASSERT_FALSE(interface.ok());
  expectRefused(interface.status(), "soft_constraints 'foot_collision'");
}

// The procedural arm swing on this set is refused when the interface is built,
// before any model is generated.
TEST(PointContactSetRefusal, theProceduralArmSwingIsRefusedByName) {
  CentroidalMpcConfig config = fourPointConfig(loadAtlas());
  config.task.arm_swing.generator = "procedural";
  ASSERT_TRUE(modelSettingsOf(config.task).hasArmSwingJoints);
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(config, atlasFiles().urdfFile);
  ASSERT_FALSE(interface.ok());
  EXPECT_EQ(interface.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(interface.status().message(),
                                "[ArmSwing] arm_swing.generator 'procedural'"))
      << interface.status();
  EXPECT_TRUE(absl::StrContains(interface.status().message(), "has 4 contacts"))
      << interface.status();
}

}  // namespace
}  // namespace ocs2::humanoid
