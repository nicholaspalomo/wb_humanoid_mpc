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

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_centroidal_mpc/CentroidalContactSet.h"
#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_centroidal_mpc/constraint/ZeroVelocityConstraintCppAd.h"
#include "humanoid_centroidal_mpc/initialization/CentroidalWeightCompInitializer.h"
#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/ContactTermNames.h"
#include "humanoid_common_mpc/common/CostTermNames.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "humanoid_common_mpc/constraint/FrictionForceConeConstraint.h"
#include "humanoid_common_mpc/constraint/PointFrictionConeConstraint.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/RealtimeRigidBodyDynamics.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/input_weights.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "support/TypedConfigFiles.h"

/**
 * The centroidal MPC on a contact set that is not the biped's
 * (humanoid_nmpc/docs/multi_contact_planning/README.md, milestone 2): the DRC
 * Atlas's two 6-DoF feet with a 3-DoF contact on each hand, from a task file
 * written as a textproto and read back.
 *  - The input is the force of each 3-DoF contact, the wrench of each 6-DoF
 *    contact, then the joint velocities: 3 n3 + 6 n6 + nj entries.
 *  - The dynamics are the centroidal momentum rate of those contacts:
 *    m hdot_lin = sum f_i + m g and m hdot_ang = sum ((p_i - c) x f_i + tau_i),
 *    derived here from Pinocchio's frame placements and center of mass.
 *  - A mode number is the bitmask of the active contacts, the first contact the
 *    most significant bit, so the feet listed last keep the biped's bits.
 *  - Every selection written for exactly two feet is refused by name.
 */
namespace ocs2::humanoid {
namespace {

constexpr char kLeftHand[] = "hand_l_contact";
constexpr char kRightHand[] = "hand_r_contact";
constexpr size_t kNumPointContacts = 2;
constexpr size_t kNumSurfaceContacts = 2;
constexpr scalar_t kTolerance = 1.0e-9;

/** A 3-DoF contact of the task file, 0.1 m to the side of a locked wrist. */
mpc_config::ModelSettingsConfig::PointContact handContact(
    absl::string_view name,
    absl::string_view wrist,
    scalar_t posContactRtLinkEwrtLinkY) {
  mpc_config::ModelSettingsConfig::PointContact contact;
  contact.name = std::string(name);
  contact.parent_joint_name = std::string(wrist);
  contact.pos_contact_rt_link_ewrt_link.y = posContactRtLinkEwrtLinkY;
  return contact;
}

/** The input weight of a 3-DoF contact: its force alone. */
mpc_config::InputWeights::ContactWrench handWeight(absl::string_view name) {
  mpc_config::InputWeights::ContactWrench weight;
  weight.contact = std::string(name);
  mpc_config::Xyz forceWeights;
  forceWeights.x = 1.0e-5;
  forceWeights.y = 1.0e-5;
  forceWeights.z = 5.0e-5;
  // NOLINTNEXTLINE(frame-naming): a per-component weight field
  weight.force = forceWeights;
  return weight;
}

/**
 * The shipped DRC Atlas configuration with a 3-DoF contact on each hand, on
 * wrench inputs and the gait schedule, and with only the terms that run on any
 * contact set.
 */
CentroidalMpcConfig mixedContactSetConfig(const CentroidalMpcConfig& atlas) {
  CentroidalMpcConfig config = atlas;
  mpc_config::TaskFile& task = config.task;
  task.model_settings.contacts_3dof = {
      handContact(kLeftHand, "l_arm_wrx", /*posContactRtLinkEwrtLinkY=*/0.1),
      handContact(kRightHand, "r_arm_wrx", /*posContactRtLinkEwrtLinkY=*/-0.1)};
  // NOLINTBEGIN(frame-naming): a per-component weight field
  task.input_weights.contact_wrenches.push_back(handWeight(kLeftHand));
  task.input_weights.contact_wrenches.push_back(handWeight(kRightHand));
  // NOLINTEND(frame-naming)
  task.contact_input_parameterization = "wrench";
  task.contact_schedule_source = "gait_schedule";
  // The Atlas's step width is the biped's step_width foothold anchor, which
  // this set does not run.
  task.nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.0;
  // The procedural arm swing follows the biped's gait phase, and is refused on
  // this set.
  task.arm_swing.generator = "none";
  task.costs = {"state_quadratic_cost", "input_quadratic_cost"};
  task.hard_constraints = {"zero_wrench", "zero_velocity"};
  task.soft_constraints = {"joint_limits", "contact_wrench_cone",
                           "friction_force_cone"};
  return config;
}

/** The shipped DRC Atlas configuration, loaded once per test. */
CentroidalMpcConfig loadAtlas() {
  return checkedValue(loadConfigOf(atlasFiles()));
}

/**
 * The mixed contact set's interface, built once for the whole suite from the
 * task file written to and read back from the test's temporary directory: it
 * generates the CppAD libraries of the dynamics and of four contact frames.
 */
class MixedContactSetTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const CentroidalRobotFiles atlasFilesInRunfiles = atlasFiles();
    const CentroidalMpcConfig config = mixedContactSetConfig(loadAtlas());
    const std::string directory =
        (std::filesystem::path(::testing::TempDir()) / "mixed_contact_set")
            .string();
    const CentroidalRobotFiles files = checkedValue(
        writeConfig(directory, config, atlasFilesInRunfiles.urdfFile));
    interface_ =
        checkedValue(CentroidalMpcInterface::Create(
                         files.taskFile, files.urdfFile, files.referenceFile))
            .release();
  }

  static void TearDownTestSuite() {
    delete interface_;
    interface_ = nullptr;
  }

  static CentroidalMpcInterface& interface() { return *interface_; }

  /** A state near the initial one and an input with every contact loaded and
   * every joint moving: the probe point. */
  static vector_t probeState() {
    vector_t state = interface().getInitialState();
    state(0) = 0.3;
    state(kRotationTangentDim + 1) = 0.02;
    return state;
  }
  static vector_t probeInput() {
    const MpcRobotModelBase<scalar_t>& model = interface().getMpcRobotModel();
    vector_t input(model.getInputDim());
    for (Eigen::Index i = 0; i < input.size(); ++i) {
      input(i) = 20.0 * std::sin(0.7 * static_cast<scalar_t>(i) + 0.3);
    }
    return input;
  }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
};

CentroidalMpcInterface* absl_nullable MixedContactSetTest::interface_ = nullptr;

TEST_F(MixedContactSetTest, theContactSetIsThePointContactsThenTheFeet) {
  const ModelSettings& settings = interface().modelSettings();
  ASSERT_EQ(settings.contactNames.size(),
            kNumPointContacts + kNumSurfaceContacts);
  EXPECT_EQ(settings.contactNames[0], kLeftHand);
  EXPECT_EQ(settings.contactNames[1], kRightHand);
  EXPECT_EQ(settings.contactNames[2], "foot_l_contact");
  EXPECT_EQ(settings.contactNames[3], "foot_r_contact");
  EXPECT_TRUE(settings.isPointContact(0));
  EXPECT_TRUE(settings.isPointContact(1));
  EXPECT_FALSE(settings.isPointContact(2));
  EXPECT_FALSE(settings.isBipedContactSet());
  const CentroidalModelInfo& info = interface().getCentroidalModelInfo();
  EXPECT_EQ(info.numThreeDofContacts, kNumPointContacts);
  EXPECT_EQ(info.numSixDofContacts, kNumSurfaceContacts);
}

TEST_F(MixedContactSetTest, theInputIsTheForcesThenTheWrenchesThenTheJoints) {
  const CentroidalMpcRobotModel<scalar_t>& model =
      interface().getMpcRobotModel();
  const size_t joints = model.getJointDim();
  EXPECT_EQ(model.getInputDim(), kForceDim * kNumPointContacts +
                                     kWrenchDim * kNumSurfaceContacts + joints);
  EXPECT_EQ(model.getInputDim(), interface().getCentroidalModelInfo().inputDim);
  EXPECT_EQ(model.getContactWrenchStartIndices(0), 0U);
  EXPECT_EQ(model.getContactWrenchStartIndices(1), kForceDim);
  EXPECT_EQ(model.getContactWrenchStartIndices(2), 2 * kForceDim);
  EXPECT_EQ(model.getContactWrenchStartIndices(3), 2 * kForceDim + kWrenchDim);
  EXPECT_EQ(model.getContactMomentStartIndices(2), 2 * kForceDim + kForceDim);
  EXPECT_EQ(model.getJointVelocitiesStartindex(),
            2 * kForceDim + 2 * kWrenchDim);
  EXPECT_EQ(model.getContactInputDim(0), kForceDim);
  EXPECT_EQ(model.getContactInputDim(3), kWrenchDim);

  // A point contact's wrench is its force with no moment; its force is the
  // three entries of its block.
  const vector_t input = probeInput();
  const vector6_t wrenchHandEwrtParam =
      model.getWrenchContactEwrtParam(input, /*contactIndex=*/1);
  EXPECT_EQ(wrenchHandEwrtParam.head<kForceDim>(),
            input.segment<kForceDim>(kForceDim));
  EXPECT_TRUE(wrenchHandEwrtParam.tail<kTorqueDim>().isZero(0.0));
  vector_t written = vector_t::Zero(model.getInputDim());
  model.setWrenchContactEwrtParam(written, wrenchHandEwrtParam,
                                  /*contactIndex=*/1);
  EXPECT_EQ(written.segment<kForceDim>(kForceDim),
            input.segment<kForceDim>(kForceDim));
  EXPECT_TRUE(written.segment<kForceDim>(2 * kForceDim).isZero(0.0))
      << "a point contact's moment must not spill into the next contact";
}

TEST_F(MixedContactSetTest, theDynamicsAreTheHandDerivedMomentumRate) {
  CentroidalMpcInterface& mpc = interface();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  const ModelSettings& settings = mpc.modelSettings();
  const vector_t state = probeState();
  const vector_t input = probeInput();

  // The contact points and the center of mass, from Pinocchio alone.
  PinocchioInterface pinocchio = mpc.getPinocchioInterface();
  const pinocchio::Model& pinocchioModel = pinocchio.getModel();
  pinocchio::Data& data = pinocchio.getData();
  const vector_t q = model.getGeneralizedCoordinates(state);
  pinocchio::forwardKinematics(pinocchioModel, data, q);
  pinocchio::updateFramePlacements(pinocchioModel, data);
  const vector3_t posComRtWorldEwrtWorld =
      pinocchio::centerOfMass(pinocchioModel, data, q);
  const scalar_t mass = mpc.getCentroidalModelInfo().robotMass;

  vector3_t forceComEwrtWorld = vector3_t::Zero();
  vector3_t torqueComEwrtWorld = vector3_t::Zero();
  for (size_t i = 0; i < settings.contactNames.size(); ++i) {
    const vector3_t posContactRtWorldEwrtWorld =
        data.oMf[pinocchioModel.getFrameId(settings.contactNames[i])]
            .translation();
    const Eigen::Index start =
        static_cast<Eigen::Index>(model.getContactWrenchStartIndices(i));
    const vector3_t forceContactEwrtWorld = input.segment<kForceDim>(start);
    forceComEwrtWorld += forceContactEwrtWorld;
    torqueComEwrtWorld += (posContactRtWorldEwrtWorld - posComRtWorldEwrtWorld)
                              .cross(forceContactEwrtWorld);
    if (!settings.isPointContact(i)) {
      torqueComEwrtWorld += input.segment<kTorqueDim>(start + kForceDim);
    }
  }
  const vector3_t gravityEwrtWorld(0.0, 0.0, -kGravityMagnitude);
  const vector3_t linaccComRtWorldEwrtWorld =
      forceComEwrtWorld / mass + gravityEwrtWorld;
  const vector3_t mnormRateAngmomComRtWorldEwrtWorld =
      torqueComEwrtWorld / mass;

  const PreComputation preComputation;
  const vector_t flowMap =
      mpc.getOptimalControlProblem().dynamicsPtr->computeFlowMap(
          /*time=*/0.0, state, input, preComputation);
  EXPECT_TRUE(flowMap.head<kPositionDim>().isApprox(linaccComRtWorldEwrtWorld,
                                                    kTolerance))
      << flowMap.head<kPositionDim>().transpose() << " against "
      << linaccComRtWorldEwrtWorld.transpose();
  EXPECT_TRUE(flowMap.segment<kRotationTangentDim>(kPositionDim)
                  .isApprox(mnormRateAngmomComRtWorldEwrtWorld, kTolerance))
      << flowMap.segment<kRotationTangentDim>(kPositionDim).transpose()
      << " against " << mnormRateAngmomComRtWorldEwrtWorld.transpose();
  // The joints move with their velocity inputs.
  EXPECT_TRUE(flowMap.tail(model.getJointDim())
                  .isApprox(input.tail(model.getJointDim()), kTolerance));

  // A hand's force enters the rate as I / m and [p - c]x / m.
  const VectorFunctionLinearApproximation linearization =
      mpc.getOptimalControlProblem().dynamicsPtr->linearApproximation(
          /*time=*/0.0, state, input, preComputation);
  const vector3_t posHandRtComEwrtWorld =
      data.oMf[pinocchioModel.getFrameId(kRightHand)].translation() -
      posComRtWorldEwrtWorld;
  matrix3_t skewPosHandRtComEwrtWorld;
  skewPosHandRtComEwrtWorld << 0.0, -posHandRtComEwrtWorld.z(),
      posHandRtComEwrtWorld.y(), posHandRtComEwrtWorld.z(), 0.0,
      -posHandRtComEwrtWorld.x(), -posHandRtComEwrtWorld.y(),
      posHandRtComEwrtWorld.x(), 0.0;
  const Eigen::Index handStart =
      static_cast<Eigen::Index>(model.getContactWrenchStartIndices(1));
  const matrix3_t linearRowsOfHandBlock =
      linearization.dfdu.block<kPositionDim, kForceDim>(0, handStart);
  const matrix3_t angularRowsOfHandBlock =
      linearization.dfdu.block<kRotationTangentDim, kForceDim>(kPositionDim,
                                                               handStart);
  EXPECT_TRUE(
      linearRowsOfHandBlock.isApprox(matrix3_t::Identity() / mass, kTolerance));
  EXPECT_TRUE(angularRowsOfHandBlock.isApprox(skewPosHandRtComEwrtWorld / mass,
                                              kTolerance));
}

TEST_F(MixedContactSetTest, aModeNumberIsTheBitmaskOfTheContactsInOrder) {
  const SwitchedModelReferenceManager& referenceManager =
      *interface().getSwitchedModelReferenceManagerPtr();
  constexpr size_t kNumMixedContacts = kNumPointContacts + kNumSurfaceContacts;
  // The first contact is the most significant bit: the feet, listed last,
  // keep the biped's bits, so STANCE is still both feet.
  EXPECT_FALSE(contactIsActiveInMode(kBipedModeStance, /*contactIndex=*/0,
                                     kNumMixedContacts));
  EXPECT_FALSE(contactIsActiveInMode(kBipedModeStance, /*contactIndex=*/1,
                                     kNumMixedContacts));
  EXPECT_TRUE(contactIsActiveInMode(kBipedModeStance, /*contactIndex=*/2,
                                    kNumMixedContacts));
  EXPECT_TRUE(contactIsActiveInMode(kBipedModeStance, /*contactIndex=*/3,
                                    kNumMixedContacts));
  EXPECT_TRUE(contactIsActiveInMode(kBipedModeLf, /*contactIndex=*/2,
                                    kNumMixedContacts));
  EXPECT_FALSE(contactIsActiveInMode(kBipedModeLf, /*contactIndex=*/3,
                                     kNumMixedContacts));
  constexpr size_t kLeftHandAndRightFoot = 0b1001;
  EXPECT_TRUE(contactIsActiveInMode(kLeftHandAndRightFoot, /*contactIndex=*/0,
                                    kNumMixedContacts));
  EXPECT_FALSE(contactIsActiveInMode(kLeftHandAndRightFoot, /*contactIndex=*/1,
                                     kNumMixedContacts));
  EXPECT_FALSE(contactIsActiveInMode(kLeftHandAndRightFoot, /*contactIndex=*/2,
                                     kNumMixedContacts));
  EXPECT_TRUE(contactIsActiveInMode(kLeftHandAndRightFoot, /*contactIndex=*/3,
                                    kNumMixedContacts));
  // A number with a bit past the contacts is no mode.
  EXPECT_FALSE(contactIsActiveInMode(size_t{1} << kNumMixedContacts,
                                     /*contactIndex=*/3, kNumMixedContacts));
  // The reference manager reads the schedule with the same bits.
  EXPECT_EQ(
      referenceManager.isInContact(/*time=*/0.0, /*contactIndex=*/0),
      contactIsActiveInMode(referenceManager.getModeSchedule().modeAtTime(0.0),
                            /*contactIndex=*/0, kNumMixedContacts));
}

/** A reference manager whose every contact touches the one patch `patch`. */
class OnePatchReferenceManager final : public SwitchedModelReferenceManager {
 public:
  OnePatchReferenceManager(CentroidalMpcInterface& mpc, ContactPatchFrame patch)
      : SwitchedModelReferenceManager(
            checkedValue(GaitSchedule::Create(mpc.config().reference,
                                              mpc.modelSettings())),
            std::make_unique<SwingTrajectoryPlanner>(
                SwingTrajectoryPlanner::Config(),
                mpc.modelSettings().numContacts()),
            mpc.getPinocchioInterface(),
            mpc.getMpcRobotModel()),
        patch_(std::move(patch)) {}

  ContactPatchFrame getContactPatchFrame(size_t /*contactIndex*/,
                                         scalar_t /*time*/) const override {
    return patch_;
  }

 private:
  const ContactPatchFrame patch_;
};

TEST_F(MixedContactSetTest, aPointContactIsHeldByItsTranslationAlone) {
  OptimalControlProblem& problem = interface().getOptimalControlProblemRef();
  const ZeroVelocityConstraintCppAd& hand =
      problem.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(
          zeroVelocityTermName(kLeftHand));
  EXPECT_EQ(hand.getContactKind(),
            ZeroVelocityConstraintCppAd::ContactKind::kPoint);
  EXPECT_EQ(hand.getNumConstraints(/*time=*/0.0), kPositionDim);
  const ZeroVelocityConstraintCppAd& foot =
      problem.equalityConstraintPtr->get<ZeroVelocityConstraintCppAd>(
          zeroVelocityTermName("foot_l_contact"));
  EXPECT_EQ(foot.getNumConstraints(/*time=*/0.0), kPoseTangentDim);
}

TEST_F(MixedContactSetTest, theListedConesAreThoseOfEachContactsKind) {
  const std::vector<std::string> soft = interface().getSoftConstraintNames();
  const absl::flat_hash_set<std::string> lists(soft.begin(), soft.end());
  EXPECT_TRUE(lists.contains(
      contact_term::name(kLeftHand, contact_term::kPointFrictionCone)));
  EXPECT_FALSE(lists.contains(
      contact_term::name(kLeftHand, contact_term::kContactWrenchCone)));
  EXPECT_TRUE(lists.contains(
      contact_term::name("foot_l_contact", contact_term::kContactWrenchCone)));
  EXPECT_FALSE(lists.contains(
      contact_term::name("foot_l_contact", contact_term::kPointFrictionCone)));
  EXPECT_TRUE(lists.contains(
      contact_term::name(kRightHand, contact_term::kFrictionForceCone)));
}

/** The point cone of the right hand on `referenceManager`, gated. */
std::unique_ptr<PointFrictionConeConstraint> handCone(
    const SwitchedModelReferenceManager& referenceManager,
    const CentroidalMpcInterface& mpc) {
  const ContactWrenchConeConstraint::Config config = checkedValue(
      contactWrenchConeConfigFromConfig(mpc.config().task.contacts));
  return checkedValue(PointFrictionConeConstraint::Create(
      referenceManager, /*contactIndex=*/1, mpc.getMpcRobotModel(), config,
      /*scheduleGated=*/true));
}

TEST_F(MixedContactSetTest, thePointConeIsTheFrictionRowsOfTheWrenchCone) {
  CentroidalMpcInterface& mpc = interface();
  const OnePatchReferenceManager flat(mpc, ContactPatchFrame());
  const std::unique_ptr<PointFrictionConeConstraint> cone = handCone(flat, mpc);
  const ContactWrenchConeConstraint::Config config = checkedValue(
      contactWrenchConeConfigFromConfig(mpc.config().task.contacts));
  const ContactWrenchConeRows rows = buildFrictionPyramidRows(config);
  ASSERT_EQ(cone->getNumConstraints(/*time=*/0.0), config.numBasisVectors + 1);

  const vector_t state = probeState();
  const vector_t input = probeInput();
  const vector3_t forceHandEwrtParam = input.segment<kForceDim>(kForceDim);
  const PreComputation preComputation;
  const vector_t value =
      cone->getValue(/*time=*/0.0, state, input, preComputation);
  EXPECT_TRUE(
      value.isApprox(rows.A_f * forceHandEwrtParam + rows.b, kTolerance));
  // The pyramid's rows are the wrench cone's first rows, on any footprint.
  const ContactRectangle footprint = checkedValue(contactRectangleFromConfig(
      mpc.config().task.contacts, mpc.modelSettings(), /*contactIndex=*/2));
  const ContactWrenchConeRows wrenchCone =
      buildLocalWrenchConeRows(config, footprint);
  EXPECT_EQ(wrenchCone.A_f.topRows(rows.A_f.rows()), rows.A_f);
  EXPECT_EQ(wrenchCone.b.head(rows.b.size()), rows.b);

  // The input Jacobian is the rows' on the hand's three force entries alone.
  const VectorFunctionLinearApproximation linearization =
      cone->getLinearApproximation(/*time=*/0.0, state, input, preComputation);
  EXPECT_TRUE(linearization.f.isApprox(value, kTolerance));
  EXPECT_TRUE(linearization.dfdx.isZero(0.0));
  matrix_t expected = matrix_t::Zero(value.size(), input.size());
  expected.middleCols<kForceDim>(kForceDim) = rows.A_f;
  EXPECT_TRUE(linearization.dfdu.isApprox(expected, kTolerance));
}

TEST_F(MixedContactSetTest, aPatchFrameConeIsTheWorldConeOfTheRotatedForce) {
  CentroidalMpcInterface& mpc = interface();
  // A patch tilted about an oblique axis, like a ramp or a box edge.
  const matrix3_t rotmatWorldRtSurface =
      Eigen::AngleAxis<scalar_t>(0.4, vector3_t(1.0, -2.0, 0.5).normalized())
          .toRotationMatrix();
  const OnePatchReferenceManager flat(mpc, ContactPatchFrame());
  const OnePatchReferenceManager tilted(
      mpc, ContactPatchFrame{.rotmatWorldRtSurface = rotmatWorldRtSurface,
                             .posSurfaceRtWorldEwrtSurfaceZ = 0.0});
  const vector_t state = probeState();
  const vector_t input = probeInput();
  vector_t rotatedInput = input;
  rotatedInput.segment<kForceDim>(kForceDim) =
      rotmatWorldRtSurface * input.segment<kForceDim>(kForceDim);
  const PreComputation preComputation;

  const std::unique_ptr<PointFrictionConeConstraint> tiltedCone =
      handCone(tilted, mpc);
  const std::unique_ptr<PointFrictionConeConstraint> flatCone =
      handCone(flat, mpc);
  EXPECT_TRUE(tiltedCone->getValue(/*time=*/0.0, state, input, preComputation)
                  .isApprox(flatCone->getValue(/*time=*/0.0, state,
                                               rotatedInput, preComputation),
                            kTolerance));
  // d/du of the rotated cone is the world cone's times p_R_w.
  const matrix_t tiltedJacobian =
      tiltedCone
          ->getLinearApproximation(/*time=*/0.0, state, input, preComputation)
          .dfdu.middleCols<kForceDim>(kForceDim);
  const matrix_t flatJacobian =
      flatCone
          ->getLinearApproximation(/*time=*/0.0, state, rotatedInput,
                                   preComputation)
          .dfdu.middleCols<kForceDim>(kForceDim);
  EXPECT_TRUE(
      tiltedJacobian.isApprox(flatJacobian * rotmatWorldRtSurface, kTolerance));

  // The friction cone of the same hand, which the patch rotates the same way.
  const FrictionForceConeConstraint::Config frictionConfig;
  const FrictionForceConeConstraint tiltedFriction(
      tilted, frictionConfig, /*contactPointIndex=*/1, mpc.getMpcRobotModel(),
      mpc.getPinocchioInterface());
  const FrictionForceConeConstraint flatFriction(
      flat, frictionConfig, /*contactPointIndex=*/1, mpc.getMpcRobotModel(),
      mpc.getPinocchioInterface());
  EXPECT_TRUE(
      tiltedFriction.getValue(/*time=*/0.0, state, input, preComputation)
          .isApprox(flatFriction.getValue(/*time=*/0.0, state, rotatedInput,
                                          preComputation),
                    kTolerance));
  const VectorFunctionQuadraticApproximation tiltedQuadratic =
      tiltedFriction.getQuadraticApproximation(/*time=*/0.0, state, input,
                                               preComputation);
  const VectorFunctionQuadraticApproximation flatQuadratic =
      flatFriction.getQuadraticApproximation(/*time=*/0.0, state, rotatedInput,
                                             preComputation);
  const Eigen::Index handStart = kForceDim;
  EXPECT_TRUE(tiltedQuadratic.dfdu.middleCols<kForceDim>(handStart).isApprox(
      flatQuadratic.dfdu.middleCols<kForceDim>(handStart) *
          rotmatWorldRtSurface,
      kTolerance));
  const matrix3_t tiltedHessian =
      tiltedQuadratic.dfduu.front().block<kForceDim, kForceDim>(handStart,
                                                                handStart);
  const matrix3_t flatHessian =
      flatQuadratic.dfduu.front().block<kForceDim, kForceDim>(handStart,
                                                              handStart);
  // The Hessian shift on the diagonal is not rotated.
  const scalar_t shift = frictionConfig.hessianDiagonalShift;
  EXPECT_TRUE((tiltedHessian + shift * matrix3_t::Identity())
                  .isApprox(rotmatWorldRtSurface.transpose() *
                                (flatHessian + shift * matrix3_t::Identity()) *
                                rotmatWorldRtSurface,
                            kTolerance));
}

/**
 * Makes `mode` the mode of the interface's reference manager around time 0,
 * bracketed by STANCE (the swing planner needs every swing to end in a
 * contact), and runs its preSolverRun().
 */
void setMode(CentroidalMpcInterface& mpc, size_t mode) {
  constexpr scalar_t kBracket = 10.0;
  SwitchedModelReferenceManager& referenceManager =
      *mpc.getSwitchedModelReferenceManagerPtr();
  referenceManager.getGaitSchedule()->updateModeSchedule(ModeSchedule(
      {-kBracket, kBracket}, {kBipedModeStance, mode, kBipedModeStance}));
  referenceManager.preSolverRun(/*initTime=*/0.0, /*finalTime=*/1.0,
                                mpc.getInitialState(), mode);
}

TEST_F(MixedContactSetTest, theWeightIsSharedAmongTheActiveContacts) {
  CentroidalMpcInterface& mpc = interface();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  const scalar_t weight =
      mpc.getCentroidalModelInfo().robotMass * kGravityMagnitude;
  const std::unique_ptr<CentroidalWeightCompInitializer> initializer(
      mpc.getInitializer().clone());
  const vector_t state = mpc.getInitialState();
  vector_t input;
  vector_t nextState;

  constexpr size_t kAllFour = 0b1111;
  setMode(mpc, kAllFour);
  initializer->compute(/*time=*/0.0, state, /*nextTime=*/0.1, input, nextState);
  for (size_t i = 0; i < kNumPointContacts + kNumSurfaceContacts; ++i) {
    EXPECT_NEAR(model.getForceContactEwrtParam(input, i).z(), 0.25 * weight,
                kTolerance)
        << "contact " << i;
  }
  // The reference of the input costs shares it the same way.
  const vector_t desired =
      mpc.getSwitchedModelReferenceManagerPtr()->getDesiredInput(
          TargetTrajectories({0.0}, {state}, {input}), state, /*time=*/0.0);
  EXPECT_TRUE(desired.isApprox(input, kTolerance));

  setMode(mpc, kBipedModeStance);
  initializer->compute(/*time=*/0.0, state, /*nextTime=*/0.1, input, nextState);
  EXPECT_TRUE(
      model.getForceContactEwrtParam(input, /*contactIndex=*/0).isZero(0.0));
  EXPECT_TRUE(
      model.getForceContactEwrtParam(input, /*contactIndex=*/1).isZero(0.0));
  EXPECT_NEAR(model.getForceContactEwrtParam(input, /*contactIndex=*/2).z(),
              0.5 * weight, kTolerance);
  EXPECT_NEAR(model.getForceContactEwrtParam(input, /*contactIndex=*/3).z(),
              0.5 * weight, kTolerance);
}

// The contact frames of the inverse dynamics are every contact's, in contact
// order: getContactFrameIndexArray() once returned the first two contacts,
// which here are the hands, so the realtime inverse dynamics applied the
// wrenches of the feet's slots at the hands.
TEST_F(MixedContactSetTest, theContactFramesAreEveryContactsInContactOrder) {
  CentroidalMpcInterface& mpc = interface();
  const ModelSettings& settings = mpc.modelSettings();
  const PinocchioInterface& pinocchioInterface = mpc.getPinocchioInterface();
  const pinocchio::Model& pinocchioModel = pinocchioInterface.getModel();
  const ContactArray<pinocchio::FrameIndex> frames =
      getContactFrameIndexArray<scalar_t>(pinocchioInterface,
                                          mpc.getMpcRobotModel());
  ASSERT_EQ(frames.size(), kNumPointContacts + kNumSurfaceContacts);
  for (size_t i = 0; i < frames.size(); ++i) {
    EXPECT_EQ(frames[i], pinocchioModel.getFrameId(settings.contactNames[i]))
        << settings.contactNames[i];
  }
  EXPECT_TRUE(
      checkContactFramesExist(pinocchioModel, settings.contactNames).ok());

  // A planned wrench on the left foot (contact 2) alone: the realtime inverse
  // dynamics on those frames is the inverse dynamics at the foot's own frame.
  absl::StatusOr<std::unique_ptr<RealtimeRigidBodyDynamics>> dynamics =
      RealtimeRigidBodyDynamics::Create(pinocchioModel, frames);
  ASSERT_TRUE(dynamics.ok()) << dynamics.status();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  const vector_t state = probeState();
  const vector_t q = model.getGeneralizedCoordinates(state);
  const vector_t v = vector_t::Zero(pinocchioModel.nv);
  const vector_t qddJoints = vector_t::Zero(model.getJointDim());
  ContactArray<vector6_t> wrenchContactsEwrtWorld(frames.size(),
                                                  vector6_t::Zero());
  constexpr size_t kLeftFoot = kNumPointContacts + kContactLeftIndex;
  wrenchContactsEwrtWorld[kLeftFoot] << 10.0, -5.0, 400.0, 2.0, -3.0, 1.0;
  const vector_t realtime = (*dynamics)->freeBaseJointTorques(
      q, v, qddJoints, wrenchContactsEwrtWorld);
  ContactArray<pinocchio::FrameIndex> footFrame(
      /*numContacts=*/1, pinocchioModel.getFrameId("foot_l_contact"));
  ContactArray<vector6_t> wrenchFootEwrtWorld(
      /*numContacts=*/1, wrenchContactsEwrtWorld[kLeftFoot]);
  PinocchioInterface working = pinocchioInterface;
  const vector_t expected = computeJointTorques<scalar_t>(
      q, v, qddJoints, wrenchFootEwrtWorld, footFrame, working);
  EXPECT_LT((realtime - expected).cwiseAbs().maxCoeff(),
            1.0e-9 * (1.0 + expected.cwiseAbs().maxCoeff()));
}

// The MRT controller's feedforward covers every contact of the set: with all
// four measured in contact and no policy yet, the weight is shared by the four
// and the feedforward is the base-held inverse dynamics of those wrenches at
// the hands' and the feet's own frames.
TEST_F(MixedContactSetTest, theMrtFeedforwardAppliesEveryContactAtItsFrame) {
  CentroidalMpcInterface& mpc = interface();
  const ModelSettings& settings = mpc.modelSettings();
  const CentroidalMpcRobotModel<scalar_t>& model = mpc.getMpcRobotModel();
  const size_t numContacts = settings.contactNames.size();
  const ::robot::model::RobotDescription description = checkedValue(
      ::robot::model::RobotDescription::Create(atlasFiles().urdfFile));
  // A generous torque limit keeps the controller's clamp from masking the
  // comparison.
  const std::filesystem::path gainsFile =
      std::filesystem::path(::testing::TempDir()) /
      "mixed_contact_set_pd_gains.textproto";
  {
    std::ofstream gains(gainsFile);
    gains << "# proto-message: humanoid_mpc_config.JointPdGainsFile\n"
             "default_gains { kp: 100.0 kd: 10.0 max_joint_torque: 1000000.0 "
             "}\n";
  }
  mpc_test::ScriptedMpc scriptedMpc(mpc::Settings{}, model.getInputDim());
  std::unique_ptr<CentroidalMpcMrtJointController> controller =
      checkedValue(CentroidalMpcMrtJointController::Create(
          description, settings, model, scriptedMpc,
          /*solverObservationObserver=*/nullptr, mpc.getPinocchioInterface(),
          /*mpcDesiredFrequency=*/400.0, gainsFile.string()));

  // Every contact measured in contact (the default estimator reads the
  // state's flags); no policy arrives, so the controller compensates the
  // weight.
  ::robot::model::RobotState robotState(description, numContacts);
  for (size_t i = 0; i < numContacts; ++i) {
    robotState.setContactFlag(i, /*contactFlag=*/true);
  }
  ::robot::model::RobotJointAction action(description);
  controller->computeJointControlAction(/*time=*/0.01, robotState, action);
  ASSERT_EQ(controller->getMeasuredContactFlags(),
            ContactMask::allActive(numContacts));

  PinocchioInterface pinocchioInterface = mpc.getPinocchioInterface();
  const pinocchio::Model& pinocchioModel = pinocchioInterface.getModel();
  ContactArray<pinocchio::FrameIndex> frames(numContacts, /*value=*/0);
  ContactArray<vector6_t> wrenchContactsEwrtWorld(numContacts,
                                                  vector6_t::Zero());
  const scalar_t weight = computeRobotWeight(pinocchioInterface);
  for (size_t i = 0; i < numContacts; ++i) {
    frames[i] = pinocchioModel.getFrameId(settings.contactNames[i]);
    wrenchContactsEwrtWorld[i](2) = weight / static_cast<scalar_t>(numContacts);
  }
  const SystemObservation& observation = controller->getCurrentObservation();
  // getGeneralizedVelocities() updates the model's own Pinocchio data: a copy.
  const std::unique_ptr<MpcRobotModelBase<scalar_t>> modelCopy(model.clone());
  const vector_t q = modelCopy->getGeneralizedCoordinates(observation.state);
  const vector_t v =
      modelCopy->getGeneralizedVelocities(observation.state, observation.input);
  const vector_t expected = computeBaseHeldJointTorques<scalar_t>(
      q, v, vector_t::Zero(model.getJointDim()), wrenchContactsEwrtWorld,
      frames, pinocchioInterface);
  // Positive control: the feet alone carrying the weight are other torques.
  ContactArray<vector6_t> feetOnly = wrenchContactsEwrtWorld;
  for (size_t i = 0; i < numContacts; ++i) {
    feetOnly[i](2) = settings.isPointContact(i) ? 0.0 : 0.5 * weight;
  }
  const vector_t positiveControlJointTorques =
      computeBaseHeldJointTorques<scalar_t>(
          q, v, vector_t::Zero(model.getJointDim()), feetOnly, frames,
          pinocchioInterface);
  ASSERT_GT((expected - positiveControlJointTorques).cwiseAbs().maxCoeff(),
            1.0);

  const std::vector<size_t> mpcJointIndices =
      description.getJointIndices(settings.mpcModelJointNames);
  ASSERT_EQ(static_cast<Eigen::Index>(mpcJointIndices.size()), expected.size());
  for (size_t i = 0; i < mpcJointIndices.size(); ++i) {
    const std::optional<::robot::model::JointAction>& jointAction =
        action[mpcJointIndices[i]];
    if (!jointAction.has_value()) {
      GTEST_FAIL() << "no action for " << settings.mpcModelJointNames[i];
    }
    EXPECT_NEAR(jointAction->feed_forward_effort,
                expected[static_cast<Eigen::Index>(i)], 1.0e-6)
        << settings.mpcModelJointNames[i];
  }
  std::filesystem::remove(gainsFile);
}

// ---------------------------------------------------------------------------------------------------------------
// What is written for two feet is refused on any other contact set.
// ---------------------------------------------------------------------------------------------------------------

/** The terms written for two 6-DoF feet, each by its list. */
struct ListedTerm {
  std::vector<std::string> mpc_config::TaskFile::*absl_nonnull list;
  const char* absl_nonnull listName;
  const char* absl_nonnull term;
};

// LINT.IfChange(biped_only_terms)
// The terms of the three lists: the biped-only ones, and the swing terms
// (task_space_foot_cost, normal_velocity) that a set without legs, as this
// one is, does not run either. The planner, the basis vectors, the heuristics
// (base-pose and wrench on any non-biped set, all of them on a set without
// legs) and the step_width foothold anchor are tested by their own cases
// below.
constexpr ListedTerm kBipedOnlyTerms[] = {
    {.list = &mpc_config::TaskFile::costs,
     .listName = "costs",
     .term = "task_space_foot_cost"},
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
    {.list = &mpc_config::TaskFile::soft_constraints,
     .listName = "soft_constraints",
     .term = "normal_velocity"},
    {.list = &mpc_config::TaskFile::hard_constraints,
     .listName = "hard_constraints",
     .term = "normal_velocity"},
    {.list = &mpc_config::TaskFile::hard_constraints,
     .listName = "hard_constraints",
     .term = "knee_joint_mimic"},
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/src/CentroidalContactSet.cpp:biped_only_terms)

/** The model settings of the mixed contact set's `task`. */
ModelSettings mixedModelSettings(const mpc_config::TaskFile& task) {
  return checkedValue(ModelSettings::Create(task, atlasFiles().urdfFile,
                                            "mixed_contact_set_refusal_",
                                            /*verbose=*/false));
}

/** `status` is InvalidArgument and its message says `selection`. */
void expectRefused(const absl::Status& status, absl::string_view selection) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_TRUE(absl::StrContains(status.message(), selection))
      << "expected the refusal to name " << selection << ": " << status;
  EXPECT_TRUE(absl::StrContains(status.message(), "not the biped's two 6-DoF"))
      << status;
}

TEST(MixedContactSetRefusal, everyTermWrittenForTwoFeetIsRefusedByName) {
  const CentroidalMpcConfig mixed = mixedContactSetConfig(loadAtlas());
  const ModelSettings settings = mixedModelSettings(mixed.task);
  ASSERT_FALSE(settings.isLeggedContactSet());
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, mixed.task));
  for (const ListedTerm& listed : kBipedOnlyTerms) {
    mpc_config::TaskFile task = mixed.task;
    (task.*listed.list).push_back(listed.term);
    expectRefused(checkCentroidalContactSet(settings, task),
                  absl::StrCat(listed.listName, " '", listed.term, "'"));
  }
}

TEST(MixedContactSetRefusal, theContactImplicitTermsAreRefused) {
  const CentroidalMpcConfig mixed = mixedContactSetConfig(loadAtlas());
  const ModelSettings settings = mixedModelSettings(mixed.task);
  mpc_config::TaskFile task = mixed.task;
  task.hard_constraints = {};
  task.soft_constraints = {"joint_limits",        "contact_wrench_cone",
                           "normal_velocity",     "contact_complementarity",
                           "force_weighted_slip", "ground_penetration"};
  const absl::Status refused = checkCentroidalContactSet(settings, task);
  expectRefused(refused, "soft_constraints 'contact_complementarity'");
  expectRefused(refused, "soft_constraints 'force_weighted_slip'");
  expectRefused(refused, "soft_constraints 'ground_penetration'");
}

TEST(MixedContactSetRefusal, thePlannersTheBasisAndTheHeuristicsAreRefused) {
  const CentroidalMpcConfig mixed = mixedContactSetConfig(loadAtlas());
  const ModelSettings settings = mixedModelSettings(mixed.task);
  mpc_config::TaskFile planner = mixed.task;
  planner.contact_schedule_source = "contact_planner";
  expectRefused(checkCentroidalContactSet(settings, planner),
                "contact_schedule_source 'contact_planner'");
  mpc_config::TaskFile basis = mixed.task;
  basis.contact_input_parameterization = "basis_vectors";
  expectRefused(checkCentroidalContactSet(settings, basis),
                "contact_input_parameterization 'basis_vectors'");
  mpc_config::TaskFile heuristics = mixed.task;
  heuristics.locomotion_heuristics.foothold_heuristics = {
      "hip_centered_stepping"};
  expectRefused(checkCentroidalContactSet(settings, heuristics),
                "locomotion_heuristics");
}

TEST(MixedContactSetRefusal, theBasisRunsOnASetOfSixDofContactsOfAnyCount) {
  // The two feet and the seat pad, all 6-DoF: one wrench cone basis each.
  mpc_config::TaskFile task = mixedContactSetConfig(loadAtlas()).task;
  task.model_settings.contacts_3dof = {};
  task.model_settings.contact_names_6dof.push_back("seat_contact");
  task.model_settings.contact_parent_joint_names.push_back("root_joint");
  task.contact_input_parameterization = "basis_vectors";
  const ModelSettings settings = mixedModelSettings(task);
  ASSERT_FALSE(settings.isBipedContactSet());
  ASSERT_EQ(settings.contactNames.size(), 3U);
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, task));
}

TEST(MixedContactSetRefusal, theInterfaceRefusesBeforeItBuildsAModel) {
  CentroidalMpcConfig config = mixedContactSetConfig(loadAtlas());
  config.task.costs.push_back("dcm_terminal_cost");
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(config, atlasFiles().urdfFile);
  ASSERT_FALSE(interface.ok());
  expectRefused(interface.status(), "costs 'dcm_terminal_cost'");
}

TEST(MixedContactSetRefusal, everyShippedBipedPassesTheCheck) {
  const CentroidalMpcConfig atlas = loadAtlas();
  const ModelSettings settings = checkedValue(ModelSettings::Create(
      atlas.task, atlasFiles().urdfFile, "biped_check_", /*verbose=*/false));
  EXPECT_TRUE(settings.isBipedContactSet());
  HUMANOID_ASSERT_OK(checkCentroidalContactSet(settings, atlas.task));
}

TEST(MixedContactSetRefusal, aMisnamedPointContactIsRefused) {
  CentroidalMpcConfig config = mixedContactSetConfig(loadAtlas());
  config.task.model_settings.contacts_3dof[1].name = "foot_l_contact";
  const absl::StatusOr<ModelSettings> sameName =
      ModelSettings::Create(config.task, atlasFiles().urdfFile, "misnamed_",
                            /*verbose=*/false);
  ASSERT_FALSE(sameName.ok());
  EXPECT_TRUE(absl::StrContains(sameName.status().message(),
                                "model_settings.contacts_3dof[1].name"))
      << sameName.status();

  config = mixedContactSetConfig(loadAtlas());
  config.task.model_settings.contacts_3dof[0].parent_joint_name = "no_joint";
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> noParent =
      CentroidalMpcInterface::Create(config, atlasFiles().urdfFile);
  ASSERT_FALSE(noParent.ok());
  EXPECT_TRUE(absl::StrContains(noParent.status().message(),
                                "contacts_3dof[0].parent_joint_name"))
      << noParent.status();
}

}  // namespace
}  // namespace ocs2::humanoid
