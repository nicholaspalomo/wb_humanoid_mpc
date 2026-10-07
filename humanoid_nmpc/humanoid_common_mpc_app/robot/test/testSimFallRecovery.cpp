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

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"
#include "pinocchio/algorithm/rnea.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc_app/robot/RealtimeEventLog.h"
#include "humanoid_common_mpc_app/robot/SimFallRecovery.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.h"

/*
 * The fall recovery of the robot process with the MuJoCo backend, against the real simulator, headless, on the shipped
 * DRC Atlas scene.
 *
 * The loops used to notice a catch only as a change of the gantry lock between two reads inside one control cycle. The
 * simulator's own resets lock the gantry on the simulation thread, at any moment of the cycle, and so went unnoticed:
 * the controller stayed in WB_MPC with its pre-fall plan, and the remote control was never told. A caught robot was
 * also re-entered into WB_MPC from where it hung, its feet loaded wherever they landed, and fell again after the unlock.
 */

namespace ocs2::humanoid {
namespace {

using robot::mujoco_sim_interface::MujocoSimConfig;
using robot::mujoco_sim_interface::MujocoSimInterface;

constexpr char kAtlasScene[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml";
constexpr char kAtlasUrdf[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kAtlasTask[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr char kAtlasGains[] = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.textproto";
constexpr scalar_t kControlPeriod = 0.01;  // [s]

std::unique_ptr<MujocoSimInterface> makeSim(bool gantryLocked) {
  MujocoSimConfig config;
  config.scenePath = kAtlasScene;
  config.headless = true;
  config.isGantryLocked = gantryLocked;
  config.gantryHold = "weld_constraint";
  absl::StatusOr<std::unique_ptr<MujocoSimInterface>> sim = MujocoSimInterface::Create(config, kAtlasUrdf);
  ABSL_CHECK_OK(sim.status());
  return *std::move(sim);
}

/** The joints of the Atlas MPC model, which the sim loops judge rest on. */
std::vector<size_t> mpcJoints(const MujocoSimInterface& sim) {
  const ModelSettings modelSettings = ModelSettings::Create(kAtlasTask, kAtlasUrdf, "centroidal_mpc_", /*verbose=*/false).value();
  return sim.getRobotDescription().getJointIndices(modelSettings.mpcModelJointNames);
}

SimFallRecovery::Config recoveryConfig(scalar_t catchLift) {
  SimFallRecovery::Config config;
  config.maxBaseTiltAngle = 1.0;
  config.catchLift = catchLift;
  return config;
}

/** A base pose at (x, y, z), rolled by `roll` about x, as setBaseStateForTesting takes it. */
std::array<double, 7> basePose(double x, double y, double z, double roll) {
  return {x, y, z, std::cos(0.5 * roll), std::sin(0.5 * roll), 0.0, 0.0};
}

constexpr std::array<double, 6> kAtRest{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/** Steps the simulator through one control period and reads its state, as the sim loops do. */
const robot::model::RobotState& advance(MujocoSimInterface& sim) {
  const int steps = static_cast<int>(std::round(kControlPeriod / sim.getModel()->opt.timestep));
  for (int i = 0; i < steps; ++i) sim.simulationStep();
  sim.updateInterfaceStateFromRobot();
  return sim.getRobotState();
}

/** The joint positions of the robot's current state, indexed like its joints: the nominal posture of JOINT_PD. */
std::vector<scalar_t> currentPosture(MujocoSimInterface& sim) {
  sim.updateInterfaceStateFromRobot();
  std::vector<scalar_t> posture(sim.getRobotDescription().getNumJoints(), 0.0);
  for (size_t joint = 0; joint < posture.size(); ++joint) posture[joint] = sim.getRobotState().getJointPosition(joint);
  return posture;
}

/**
 * JOINT_PD as CentroidalMpcMrtJointController computes it: a PD to `nominal` with the robot's own gains
 * (config/controller/joint_pd_gains.textproto), and on the joints of the MPC model the gravity torques of the base-held
 * robot as feedforward.
 */
class AtlasJointPd {
 public:
  explicit AtlasJointPd(const MujocoSimInterface& sim)
      : modelSettings_(ModelSettings::Create(kAtlasTask, kAtlasUrdf, "centroidal_mpc_", /*verbose=*/false).value()),
        pinocchioInterface_(loadCustomPinocchioInterface(kAtlasTask, kAtlasUrdf, modelSettings_).value()),
        mpcJoints_(sim.getRobotDescription().getJointIndices(modelSettings_.mpcModelJointNames)) {
    const absl::StatusOr<mpc_config::JointPdGainsFile> gains = loadJointPdGainsFile(kAtlasGains);
    ABSL_CHECK_OK(gains.status());
    const mpc_config::JointPdGainsFile::Gains& defaults = gains->default_gains;
    ABSL_CHECK(defaults.kp.has_value() && defaults.kd.has_value()) << kAtlasGains << " sets no default_gains";
    const robot::model::RobotDescription& description = sim.getRobotDescription();
    kp_.assign(description.getNumJoints(), defaults.kp.value_or(0.0));
    kd_.assign(description.getNumJoints(), defaults.kd.value_or(0.0));
    for (const mpc_config::JointPdGainsFile::JointGains& entry : gains->joint_gains) {
      if (!description.containsJoint(entry.joint)) continue;
      const size_t index = description.getJointIndex(entry.joint);
      if (entry.kp.has_value()) kp_[index] = *entry.kp;
      if (entry.kd.has_value()) kd_[index] = *entry.kd;
    }
  }

  const std::vector<size_t>& mpcJoints() const { return mpcJoints_; }

  void command(MujocoSimInterface& sim, const std::vector<scalar_t>& nominal) {
    const robot::model::RobotState& state = sim.getRobotState();
    vector_t q(6 + mpcJoints_.size());
    q.head<3>() = state.getRootPositionInWorldFrame();
    q.segment<3>(3) = quaternionToEulerZYX(state.getRootRotationLocalToWorldFrame());
    for (size_t i = 0; i < mpcJoints_.size(); ++i) q(6 + i) = state.getJointPosition(mpcJoints_[i]);
    const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
    PinocchioInterface::Data& data = pinocchioInterface_.getData();
    pinocchio::nonLinearEffects(model, data, q, vector_t::Zero(model.nv));
    const vector_t gravity = data.nle.tail(mpcJoints_.size());

    robot::model::RobotJointAction& action = sim.getRobotJointAction();
    for (size_t joint = 0; joint < nominal.size(); ++joint) {
      std::optional<robot::model::JointAction>& slot = action.at(joint);
      if (!slot.has_value()) continue;
      slot->q_des = nominal[joint];
      slot->qd_des = 0.0;
      slot->kp = kp_[joint];
      slot->kd = kd_[joint];
      slot->feed_forward_effort = 0.0;
    }
    for (size_t i = 0; i < mpcJoints_.size(); ++i) {
      std::optional<robot::model::JointAction>& slot = action.at(mpcJoints_[i]);
      if (slot.has_value()) slot->feed_forward_effort = gravity[i];
    }
    sim.applyJointAction();
  }

 private:
  ModelSettings modelSettings_;
  PinocchioInterface pinocchioInterface_;
  std::vector<size_t> mpcJoints_;
  std::vector<scalar_t> kp_;
  std::vector<scalar_t> kd_;
};

/** Why a state is or is not at rest, for the failure messages. */
std::string describeRest(const robot::model::RobotState& state, const std::vector<scalar_t>& nominal, const std::vector<size_t>& joints) {
  scalar_t maxJointError = 0.0;
  scalar_t maxJointSpeed = 0.0;
  size_t worstJoint = 0;
  size_t fastestJoint = 0;
  for (size_t joint : joints) {
    const scalar_t error = std::abs(state.getJointPosition(joint) - nominal[joint]);
    if (error > maxJointError) {
      maxJointError = error;
      worstJoint = joint;
    }
    if (std::abs(state.getJointVelocity(joint)) > maxJointSpeed) {
      maxJointSpeed = std::abs(state.getJointVelocity(joint));
      fastestJoint = joint;
    }
  }
  return absl::StrCat("tilt ", SimFallRecovery::baseTiltAngle(state.getRootRotationLocalToWorldFrame()), " rad, base speed ",
                      state.getRootLinearVelocityInLocalFrame().norm(), " m/s and ", state.getRootAngularVelocityInLocalFrame().norm(),
                      " rad/s, joint error ", maxJointError, " rad (joint ", worstJoint, "), joint speed ", maxJointSpeed, " rad/s (joint ",
                      fastestJoint, ")");
}

TEST(SimFallRecovery, EverySimulatorResetIsOneDiscontinuityWhereverTheGantryWas) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(/*gantryLocked=*/false);
  SimFallRecovery recovery(recoveryConfig(0.0), *sim, mpcJoints(*sim));
  const std::vector<scalar_t> nominal = currentPosture(*sim);
  std::string mode = "WB_MPC";
  EXPECT_FALSE(recovery.update(advance(*sim), nominal, *sim, mode).discontinuity);

  // The base drops below the floor limit: the simulator resets the robot and locks the gantry itself, inside its step,
  // i.e. between two control cycles.
  sim->setBaseStateForTesting(basePose(2.0, 0.0, 0.1, 0.0), kAtRest);
  const robot::model::RobotState& state = advance(*sim);
  ASSERT_EQ(sim->resetEpoch(), 1u);
  // The simulator locked the gantry itself, inside its step: the loops' old bracket, which read the lock before and
  // after the cycle's own commands, read it locked both times and saw no transition at all.
  ASSERT_TRUE(sim->isGantryLocked());

  const SimFallRecovery::Cycle caught = recovery.update(state, nominal, *sim, mode);
  EXPECT_TRUE(caught.discontinuity);
  EXPECT_EQ(caught.cause, DiscontinuityCause::kSimulatorReset);
  EXPECT_NE(recovery.describeDiscontinuity(caught).find("reset epoch 1"), std::string::npos) << recovery.describeDiscontinuity(caught);
  EXPECT_TRUE(caught.modeChanged);
  EXPECT_EQ(mode, "JOINT_PD");
  EXPECT_FALSE(sim->isZeroTorqueMode()) << "JOINT_PD needs the torques";
  EXPECT_FALSE(recovery.update(advance(*sim), nominal, *sim, mode).discontinuity) << "one event, one discontinuity";

  // A reset while the gantry is locked already changes nothing a loop could see but the epoch: it is still one.
  ASSERT_TRUE(sim->isGantryLocked());
  mode = "WB_MPC";
  sim->reset();
  const SimFallRecovery::Cycle again = recovery.update(advance(*sim), nominal, *sim, mode);
  EXPECT_TRUE(again.discontinuity);
  EXPECT_EQ(mode, "JOINT_PD");
  EXPECT_FALSE(recovery.update(advance(*sim), nominal, *sim, mode).discontinuity);
}

TEST(SimFallRecovery, ATiltIsCaughtOnceAndAnOperatorLockAndUnlockAreSeen) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(/*gantryLocked=*/false);
  SimFallRecovery recovery(recoveryConfig(0.0), *sim, mpcJoints(*sim));
  const std::vector<scalar_t> nominal = currentPosture(*sim);
  const double height = sim->getGantryHeight();
  std::string mode = "WB_MPC";
  advance(*sim);

  // Tipped past the limit: caught on the gantry, once.
  sim->setBaseStateForTesting(basePose(/*x=*/1.0, /*y=*/0.5, height, /*roll=*/1.2), kAtRest);
  const SimFallRecovery::Cycle caught = recovery.update(advance(*sim), nominal, *sim, mode);
  EXPECT_TRUE(caught.discontinuity);
  EXPECT_TRUE(sim->isGantryLocked());
  EXPECT_EQ(mode, "JOINT_PD");
  EXPECT_FALSE(recovery.update(advance(*sim), nominal, *sim, mode).discontinuity);

  // Held upright where it was caught, then released by the operator: reported, and not a discontinuity.
  for (int cycle = 0; cycle < 200; ++cycle) recovery.update(advance(*sim), nominal, *sim, mode);
  ASSERT_LT(SimFallRecovery::baseTiltAngle(sim->getRobotState().getRootRotationLocalToWorldFrame()), 0.2);
  sim->unlockGantry();
  const SimFallRecovery::Cycle released = recovery.update(advance(*sim), nominal, *sim, mode);
  EXPECT_TRUE(released.gantryUnlocked);
  EXPECT_FALSE(released.discontinuity);

  // The operator's LOCK_GANTRY: one discontinuity, and no settle sequence for a robot that did not fall.
  sim->setBaseStateForTesting(basePose(/*x=*/0.0, /*y=*/0.0, height, /*roll=*/0.0), kAtRest);
  advance(*sim);
  recovery.update(sim->getRobotState(), nominal, *sim, mode);
  sim->lockGantry();
  mode = "WB_MPC";
  const SimFallRecovery::Cycle locked = recovery.update(advance(*sim), nominal, *sim, mode);
  EXPECT_TRUE(locked.discontinuity);
  EXPECT_FALSE(recovery.isSettling());
  EXPECT_FALSE(recovery.update(advance(*sim), nominal, *sim, mode).discontinuity);
}

TEST(SimFallRecovery, ACaughtRobotIsLiftedSettledAndLoweredBeforeWbMpcIsAccepted) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(/*gantryLocked=*/false);
  const SimFallRecovery::Config config = recoveryConfig(0.15);
  SimFallRecovery recovery(config, *sim, mpcJoints(*sim));
  const std::vector<scalar_t> nominal = currentPosture(*sim);
  const double standingHeight = sim->getGantryHeight();
  AtlasJointPd jointPd(*sim);
  sim->enableTorques();
  std::string mode = "WB_MPC";

  // Knocked over with the gantry unlocked.
  jointPd.command(*sim, nominal);
  advance(*sim);
  sim->setBaseStateForTesting(basePose(/*x=*/0.5, /*y=*/0.0, standingHeight - 0.1, /*roll=*/1.2), {0.0, 1.0, -0.5, 2.0, 0.0, 0.0});

  const std::vector<size_t> joints = mpcJoints(*sim);
  std::vector<SimFallRecovery::Phase> phases;
  std::vector<scalar_t> phaseStartTimes;
  std::vector<std::string> phaseEndStates;  // the robot when each phase began, for the failure messages
  double maxGantryHeight = 0.0;
  bool wbMpcLetThroughWhileSettling = false;
  bool caught = false;
  for (int cycle = 0; cycle < 3000; ++cycle) {
    const robot::model::RobotState& state = advance(*sim);
    mode = "WB_MPC";  // the operator asks for WB_MPC at every cycle
    const SimFallRecovery::Cycle result = recovery.update(state, nominal, *sim, mode);
    caught = caught || result.discontinuity;
    if (phases.empty() || phases.back() != recovery.phase()) {
      phases.push_back(recovery.phase());
      phaseStartTimes.push_back(state.getTime());
      phaseEndStates.push_back(describeRest(state, nominal, joints));
    }
    maxGantryHeight = std::max(maxGantryHeight, sim->getGantryHeight());
    if (recovery.isSettling() && mode == "WB_MPC") wbMpcLetThroughWhileSettling = true;
    jointPd.command(*sim, nominal);
    if (caught && !recovery.isSettling()) break;
  }

  ASSERT_TRUE(caught);
  EXPECT_FALSE(wbMpcLetThroughWhileSettling) << "WB_MPC was accepted before the caught robot had settled";
  const std::vector<SimFallRecovery::Phase> expected{SimFallRecovery::Phase::kLifting, SimFallRecovery::Phase::kSettlingLifted,
                                                     SimFallRecovery::Phase::kLowering, SimFallRecovery::Phase::kSettlingOnFeet,
                                                     SimFallRecovery::Phase::kIdle};
  EXPECT_EQ(phases, expected);
  // Each wait ended because the robot came to rest, not because it timed out.
  for (size_t phase = 1; phase < phaseStartTimes.size(); ++phase) {
    EXPECT_LT(phaseStartTimes[phase] - phaseStartTimes[phase - 1], config.settleTimeout)
        << "phase " << phase - 1 << " timed out; when it ended: " << phaseEndStates[phase];
  }
  EXPECT_NEAR(maxGantryHeight, standingHeight + config.catchLift, 1.0e-6) << "the gantry did not lift the robot clear of the ground";
  EXPECT_NEAR(sim->getGantryHeight(), standingHeight, 1.0e-6) << "the gantry was not lowered back to the height of the catch";
  EXPECT_EQ(mode, "WB_MPC") << "WB_MPC is accepted once the robot has settled";
  EXPECT_TRUE(SimFallRecovery::isAtRest(sim->getRobotState(), nominal, joints, config))
      << "the robot was accepted without being at rest: " << describeRest(sim->getRobotState(), nominal, joints);
}

TEST(SimFallRecovery, AtRestMeansUprightStillAndAtTheNominalPosture) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(/*gantryLocked=*/true);
  const std::vector<scalar_t> nominal = currentPosture(*sim);
  const SimFallRecovery::Config config = recoveryConfig(0.15);
  const std::vector<size_t> joints = mpcJoints(*sim);
  robot::model::RobotState state = sim->getRobotState();
  state.setRootRotationLocalToWorldFrame(quaternion_t::Identity());
  state.setRootLinearVelocityInLocalFrame(vector3_t::Zero());
  state.setRootAngularVelocityInLocalFrame(vector3_t::Zero());
  for (size_t joint = 0; joint < nominal.size(); ++joint) state.setJointVelocity(joint, /*jointVelocity=*/0.0);
  EXPECT_TRUE(SimFallRecovery::isAtRest(state, nominal, joints, config));

  // What the robot looked like when it was caught at its standing height, feet loaded where they landed: pitched and
  // off its posture. Each on its own is not at rest.
  robot::model::RobotState pitched = state;
  pitched.setRootRotationLocalToWorldFrame(quaternion_t(Eigen::AngleAxis<scalar_t>(0.14, vector3_t::UnitY())));
  EXPECT_FALSE(SimFallRecovery::isAtRest(pitched, nominal, joints, config));
  robot::model::RobotState bent = state;
  bent.setJointPosition(joints.front(), nominal[joints.front()] + 0.3);
  EXPECT_FALSE(SimFallRecovery::isAtRest(bent, nominal, joints, config));
  robot::model::RobotState moving = state;
  moving.setRootLinearVelocityInLocalFrame(vector3_t(0.2, 0.0, 0.0));
  EXPECT_FALSE(SimFallRecovery::isAtRest(moving, nominal, joints, config));

  // A joint the state does not report has no say, whatever its nominal position.
  std::vector<scalar_t> nominalWithAnExtraJoint = nominal;
  nominalWithAnExtraJoint.push_back(5.0);
  std::vector<size_t> jointsWithAnExtraJoint = joints;
  jointsWithAnExtraJoint.push_back(nominal.size());
  ASSERT_FALSE(state.hasJoint(nominal.size()));
  EXPECT_TRUE(SimFallRecovery::isAtRest(state, nominalWithAnExtraJoint, jointsWithAnExtraJoint, config));
}

TEST(SimFallRecovery, ReportsTheCatchToTheEventLogInsteadOfLogging) {
  const std::unique_ptr<MujocoSimInterface> sim = makeSim(/*gantryLocked=*/false);
  RealtimeEventLog log;
  SimFallRecovery recovery(recoveryConfig(0.15), *sim, mpcJoints(*sim), &log);
  const std::vector<scalar_t> nominal = currentPosture(*sim);
  const double height = sim->getGantryHeight();
  std::string mode = "WB_MPC";
  advance(*sim);
  sim->setBaseStateForTesting(basePose(/*x=*/0.0, /*y=*/0.0, height, /*roll=*/1.2), kAtRest);
  const SimFallRecovery::Cycle caught = recovery.update(advance(*sim), nominal, *sim, mode);
  ASSERT_TRUE(caught.discontinuity);
  EXPECT_EQ(caught.cause, DiscontinuityCause::kTiltCaught);
  EXPECT_GT(caught.tilt, 1.0);
  mode = "WB_MPC";
  recovery.update(advance(*sim), nominal, *sim, mode);
  EXPECT_EQ(mode, "JOINT_PD") << "refused while settling";
  std::vector<RealtimeEventCode> codes;
  std::string caughtLine;
  log.drain([&](const RealtimeEvent& event) {
    codes.push_back(event.code);
    if (event.code == RealtimeEventCode::kCaughtAndSettling) caughtLine = formatRealtimeEvent(event);
  });
  EXPECT_EQ(codes,
            (std::vector<RealtimeEventCode>{RealtimeEventCode::kCaughtAndSettling, RealtimeEventCode::kMpcModeRefusedWhileSettling}));
  EXPECT_NE(caughtLine.find("past sim_max_base_tilt_angle 1 rad"), std::string::npos) << caughtLine;
  EXPECT_NE(caughtLine.find("0.15 m (sim_gantry_catch_lift)"), std::string::npos) << caughtLine;
}

}  // namespace
}  // namespace ocs2::humanoid
