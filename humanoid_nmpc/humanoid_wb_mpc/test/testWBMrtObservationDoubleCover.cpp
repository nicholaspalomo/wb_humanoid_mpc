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
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_mpc_test/ScriptedMpc.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/solver/SolverSettingsFromConfig.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/mrt/WBMpcMrtJointController.h"
#include "robot_core/ResourcePaths.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"

/*
 * The whole-body MRT joint controller under the double cover of the unit
 * quaternion (design decision D12: no sign alignment is needed), on the G1
 * around a scripted solver whose plan turns the base and moves the joints. A
 * robot measured at -xi is the robot measured at xi: the observation handed to
 * the solver is the same but for the sign of its quaternion, and the policy
 * input and every joint command are the same bit for bit, in the passive modes,
 * in WB_MPC before the first policy and in WB_MPC with one. And a measured
 * stream that an estimator canonicalizes to w >= 0 - its quaternion flips sign
 * where the base turns through a heading of +-pi - commands, cycle by cycle,
 * what the continuous stream commands, its heading continuous. The solver
 * iterations run on the test's thread (InProcessMpcLink:: Execution::kCaller),
 * so that two controllers see the same sequence.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kControlPeriod = 0.01;  // [s]
// [rad/s] the turn of the scripted plan and of the measured stream.
constexpr scalar_t kTurnRate = 1.0;

/** The action of joint `index`, which every joint of the description holds: a
 * test failure and a zero action if not. */
const robot::model::JointAction& actionOf(
    const robot::model::RobotJointAction& action, size_t index) {
  const std::optional<robot::model::JointAction>& entry = action.at(index);
  if (!entry.has_value()) {
    ADD_FAILURE() << "no action for joint " << index;
    static const robot::model::JointAction kNoAction;
    return kNoAction;
  }
  return *entry;
}

/** Expects the commands of every joint of `description` to be the same in
 * `expected` and `actual`, bit for bit. */
void expectSameCommands(const robot::model::RobotDescription& description,
                        const robot::model::RobotJointAction& expected,
                        const robot::model::RobotJointAction& actual) {
  for (size_t index : description.getJointIndices()) {
    const robot::model::JointAction& a = actionOf(expected, index);
    const robot::model::JointAction& b = actionOf(actual, index);
    EXPECT_EQ(a.q_des, b.q_des) << "joint " << index;
    EXPECT_EQ(a.qd_des, b.qd_des) << "joint " << index;
    EXPECT_EQ(a.kp, b.kp) << "joint " << index;
    EXPECT_EQ(a.kd, b.kd) << "joint " << index;
    EXPECT_EQ(a.feed_forward_effort, b.feed_forward_effort)
        << "joint " << index;
  }
}

/** The unit quaternion of a turn by `angle` about the world z axis. */
vector4_t headingRotation(scalar_t angle) {
  return vector4_t(0.0, 0.0, std::sin(0.5 * angle), std::cos(0.5 * angle));
}

/**
 * A scripted plan that turns the base about the world z axis at kTurnRate from
 * the observed attitude, xi(t) = q_z(kTurnRate (t - t0)) (x) xi_0, the product
 * negating with xi_0 bit for bit, and moves every joint at 0.05 rad/s.
 */
mpc_test::ScriptedSolver::PlanFunction turningPlan(
    const WBAccelMpcRobotModel<scalar_t>& model) {
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  const Eigen::Index jointStart =
      static_cast<Eigen::Index>(model.getJointStartindex());
  const Eigen::Index jointDim = static_cast<Eigen::Index>(model.getJointDim());
  return [orientationStart, jointStart, jointDim](
             scalar_t time, scalar_t initTime, const vector_t& initState) {
    vector_t planned = initState;
    const vector4_t initialOrientation = initState.segment<4>(orientationStart);
    planned.segment<4>(orientationStart) = quaternionProduct<scalar_t>(
        headingRotation(kTurnRate * (time - initTime)), initialOrientation);
    planned.segment(jointStart, jointDim).array() += 0.05 * (time - initTime);
    return planned;
  };
}

/** The G1 whole-body controller around a scripted solver whose iterations the
 * test runs, and the robot it measures. */
class WholeBodyController {
 public:
  WholeBodyController()
      // LINT.IfChange(robot_files)
      : task_(checkedValue(loadTaskFile(checkedValue(robot::resolveResourcePath(
            "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto"))))),
        urdfFile_(checkedValue(robot::resolveResourcePath(
            "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"))),
        // LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:double_cover_test_data)
        modelSettings_(checkedValue(ModelSettings::Create(
            task_, urdfFile_, "wb_mpc_", /*verbose=*/false))),
        pinocchioInterface_(checkedValue(
            loadCustomPinocchioInterface(task_, urdfFile_, modelSettings_))),
        model_(modelSettings_),
        mpc_(*toMpcSettings(task_.mpc), model_.getInputDim()),
        description_(
            checkedValue(robot::model::RobotDescription::Create(urdfFile_))),
        robotState_(description_),
        action_(description_) {
    mpcJointIndices_ =
        description_.getJointIndices(modelSettings_.mpcModelJointNames);
    mpc_.solver().setPlan(turningPlan(model_));
    InProcessMpcLink::Config config;
    config.execution = InProcessMpcLink::Execution::kCaller;
    absl::StatusOr<std::unique_ptr<WBMpcMrtJointController>> created =
        WBMpcMrtJointController::Create(
            description_, modelSettings_,
            InProcessMpcLink::factory(mpc_, std::move(config), &link_),
            pinocchioInterface_,
            checkedValue(robot::resolveResourcePath(
                "robot_models/unitree_g1/g1_wb_mpc/config/controller/"
                "joint_pd_gains.textproto")));
    EXPECT_TRUE(created.ok()) << created.status();
    if (created.ok()) controller_ = *std::move(created);
  }
  WholeBodyController(const WholeBodyController&) = delete;
  WholeBodyController& operator=(const WholeBodyController&) = delete;
  ~WholeBodyController() = default;

  bool ok() const { return controller_ != nullptr; }
  WBMpcMrtJointController& controller() { return *controller_; }
  const WBAccelMpcRobotModel<scalar_t>& model() const { return model_; }
  const robot::model::RobotDescription& description() const {
    return description_;
  }
  mpc_test::ScriptedMpc& mpc() { return mpc_; }

  /**
   * Measures the robot at the base orientation `orientation` (x, y, z, w, as
   * the estimator hands it), standing at a fixed position, turning at kTurnRate
   * about the world z axis and moving every joint, both feet on the ground.
   */
  void measure(const vector4_t& orientation) {
    robotState_.setConfigurationToZero();
    robotState_.setRootPositionInWorldFrame(vector3_t(0.1, -0.05, 0.76));
    const quaternion_t rotation(orientation);
    robotState_.setRootRotationLocalToWorldFrame(rotation);
    robotState_.setRootLinearVelocityInLocalFrame(rotation.conjugate() *
                                                  vector3_t(0.2, 0.05, 0.0));
    robotState_.setRootAngularVelocityInLocalFrame(
        rotation.conjugate() * vector3_t(0.0, 0.0, kTurnRate));
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      robotState_.setJointPosition(
          mpcJointIndices_[i], 0.1 * std::sin(0.7 * static_cast<scalar_t>(i)));
      robotState_.setJointVelocity(
          mpcJointIndices_[i], 0.2 * std::cos(1.3 * static_cast<scalar_t>(i)));
    }
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  }

  /** startMpcThread() at the current time on the caller's link: serves the
   * start-up reset and starts no thread. */
  void start() {
    robotState_.setTime(time_);
    controller_->startMpcThread(robotState_);
  }

  /** One solver iteration on this thread, from the observation of the last
   * cycle. */
  void solve() {
    const InProcessMpcLink::SolverIterationResult iteration =
        link_->runSolverIteration();
    EXPECT_TRUE(iteration.status.ok()) << iteration.status;
  }

  /** One control cycle at the current time; returns the action. */
  const robot::model::RobotJointAction& cycle() {
    robotState_.setTime(time_);
    controller_->computeJointControlAction(time_, robotState_, action_);
    time_ += kControlPeriod;
    return action_;
  }

 private:
  mpc_config::TaskFile task_;
  std::string urdfFile_;
  ModelSettings modelSettings_;
  PinocchioInterface pinocchioInterface_;
  WBAccelMpcRobotModel<scalar_t> model_;
  mpc_test::ScriptedMpc mpc_;
  robot::model::RobotDescription description_;
  robot::model::RobotState robotState_;
  robot::model::RobotJointAction action_;
  std::vector<size_t> mpcJointIndices_;
  scalar_t time_ = 1.0;
  InProcessMpcLink* absl_nullable link_ = nullptr;  // the controller's
  // Last: destroyed first, as its link reads the scripted MPC.
  std::unique_ptr<WBMpcMrtJointController> controller_;
};

/**
 * The measured orientation of the stream at control cycle `cycle`: q_z(psi) (x)
 * `tilt`, the heading psi turning at kTurnRate through pi from pi - 0.25.
 */
vector4_t streamOrientation(const vector4_t& tilt, int cycle) {
  return quaternionProduct<scalar_t>(
      headingRotation(M_PI - 0.25 +
                      kTurnRate * kControlPeriod *
                          static_cast<scalar_t>(cycle)),
      tilt);
}

/**
 * Expects `actual` to be `expected` with its quaternion multiplied by `sign`,
 * bit for bit: the same state of the robot with the other sign of the
 * quaternion for -1.
 */
void expectSameStateUpToTheQuaternionSign(
    const WBAccelMpcRobotModel<scalar_t>& model,
    const vector_t& expected,
    const vector_t& actual,
    scalar_t sign) {
  ASSERT_EQ(expected.size(), actual.size());
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  vector_t signedExpected = expected;
  signedExpected.segment<4>(orientationStart) *= sign;
  EXPECT_EQ(actual, signedExpected);
}

/** Expects the observations of the two controllers to be the same up to the
 * sign `sign` of the second's quaternion. */
void expectSameObservation(WholeBodyController& expected,
                           WholeBodyController& actual,
                           scalar_t sign) {
  const SystemObservation& a = expected.controller().getCurrentObservation();
  const SystemObservation& b = actual.controller().getCurrentObservation();
  EXPECT_EQ(a.time, b.time);
  EXPECT_EQ(a.mode, b.mode);
  EXPECT_EQ(a.input, b.input);
  expectSameStateUpToTheQuaternionSign(expected.model(), a.state, b.state,
                                       sign);
}

TEST(WBMrtObservationDoubleCover,
     ANegatedMeasuredQuaternionIsTheSameRobotToTheSolverAndToEveryJoint) {
  // A level attitude, one turned near a heading of pi and one at a pitch of 90
  // degrees.
  for (const vector3_t& eulerZyx :
       {vector3_t(0.3, 0.1, -0.05), vector3_t(3.1, -0.2, 0.15),
        vector3_t(-2.0, M_PI / 2.0, 0.3)}) {
    SCOPED_TRACE(eulerZyx.transpose());
    const vector4_t xi = quaternionFromEulerZyx(eulerZyx);
    WholeBodyController positive;
    WholeBodyController negative;
    ASSERT_TRUE(positive.ok() && negative.ok());
    const robot::model::RobotDescription& description = positive.description();

    // The passive modes.
    positive.measure(xi);
    negative.measure(-xi);
    for (const char* absl_nonnull const mode : {"GRAVITY_COMP", "JOINT_PD"}) {
      SCOPED_TRACE(mode);
      positive.controller().setControlMode(mode);
      negative.controller().setControlMode(mode);
      expectSameCommands(description, positive.cycle(), negative.cycle());
      expectSameObservation(positive, negative, /*sign=*/-1.0);
    }

    // WB_MPC: entered, it holds until a policy solved after the entry; then the
    // policy's commands, over several solves of the turning plan.
    positive.start();
    negative.start();
    positive.controller().setControlMode("WB_MPC");
    negative.controller().setControlMode("WB_MPC");
    expectSameCommands(description, positive.cycle(), negative.cycle());
    for (int solve = 0; solve < 5; ++solve) {
      SCOPED_TRACE(solve);
      positive.solve();
      negative.solve();
      ASSERT_EQ(positive.mpc().solver().numSolves(),
                negative.mpc().solver().numSolves());
      expectSameStateUpToTheQuaternionSign(
          positive.model(), positive.mpc().solver().lastInitState(),
          negative.mpc().solver().lastInitState(), /*sign=*/-1.0);
      for (int cycle = 0; cycle < 3; ++cycle) {
        expectSameCommands(description, positive.cycle(), negative.cycle());
        expectSameObservation(positive, negative, /*sign=*/-1.0);
        EXPECT_EQ(positive.controller().getLatestPolicyInput(),
                  negative.controller().getLatestPolicyInput());
      }
    }
    EXPECT_TRUE(positive.controller().ready());
    EXPECT_FALSE(positive.controller().isHolding())
        << "the policy never came into use";
  }
}

TEST(WBMrtObservationDoubleCover,
     AStreamCanonicalizedToANonNegativeWCommandsAsTheContinuousStreamAcrossPi) {
  WholeBodyController continuous;
  WholeBodyController canonical;
  ASSERT_TRUE(continuous.ok() && canonical.ok());
  const robot::model::RobotDescription& description = continuous.description();
  const WBAccelMpcRobotModel<scalar_t>& model = continuous.model();
  // Tilted, turning through a heading of pi at kTurnRate: xi(psi) = q_z(psi)
  // (x) q_tilt is continuous, and its w changes sign there; the canonical
  // stream is -xi wherever w < 0.
  const vector4_t tilt = quaternionFromEulerZyx(vector3_t(0.0, 0.05, 0.1));
  constexpr int kCycles = 50;

  continuous.measure(streamOrientation(tilt, /*cycle=*/0));
  canonical.measure(streamOrientation(tilt, /*cycle=*/0));
  continuous.start();
  canonical.start();
  continuous.solve();
  canonical.solve();
  int flips = 0;
  scalar_t previousSign = 1.0;
  scalar_t previousHeading = 0.0;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    SCOPED_TRACE(cycle);
    const vector4_t xi = streamOrientation(tilt, cycle);
    const scalar_t sign = xi(3) < 0.0 ? -1.0 : 1.0;
    if (cycle > 0 && sign != previousSign) ++flips;
    previousSign = sign;
    continuous.measure(xi);
    canonical.measure(sign * xi);
    expectSameCommands(description, continuous.cycle(), canonical.cycle());
    expectSameObservation(continuous, canonical, sign);
    EXPECT_EQ(continuous.controller().getLatestPolicyInput(),
              canonical.controller().getLatestPolicyInput());
    // The heading the canonical stream reads is continuous across the flip: a
    // step of the turn, never a jump.
    const scalar_t heading = model.getBaseHeading(
        canonical.controller().getCurrentObservation().state);
    if (cycle > 0) {
      EXPECT_NEAR(std::remainder(heading - previousHeading, 2.0 * M_PI),
                  kTurnRate * kControlPeriod, 1.0e-3);
    }
    previousHeading = heading;
    continuous.solve();
    canonical.solve();
    expectSameStateUpToTheQuaternionSign(
        model, continuous.mpc().solver().lastInitState(),
        canonical.mpc().solver().lastInitState(), sign);
  }
  EXPECT_EQ(flips, 1) << "positive control: the canonical stream flips its "
                         "sign once, at the heading of pi";
  EXPECT_FALSE(canonical.controller().isHolding());
}

}  // namespace
}  // namespace ocs2::humanoid
