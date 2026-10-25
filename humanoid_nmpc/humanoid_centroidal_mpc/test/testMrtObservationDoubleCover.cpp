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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcMrtJointController.h"
#include "humanoid_common_mpc/mrt/InProcessMpcLink.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotJointAction.h"
#include "robot_model/RobotState.h"
#include "support/AtlasReferenceStack.h"

/*
 * The double cover of the measured base orientation through
 * CentroidalMpcMrtJointController (design README section 5(c)): the quaternions
 * xi and -xi are the same attitude, so an observation measured with either
 * gives the solver the same state up to the sign of its quaternion - every
 * other entry bit for bit, the momentum included - and the robot the same joint
 * commands, bit for bit; and a measured stream canonicalized to w >= 0, which
 * flips the sign of the quaternion where a turn crosses a heading of +-pi,
 * commands exactly what the continuous stream does. On the lockstep link
 * (InProcessMpcLink::Execution::kCaller) around the DRC Atlas references'
 * scripted solver; builds no CppAD model.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kControlPeriod =
    0.01;  // [s] of observation time per control cycle

/** The measured base of one control cycle. Passive data. */
struct MeasuredBase {
  vector3_t position = vector3_t::Zero();
  quaternion_t orientation = quaternion_t::Identity();
  vector3_t linearVelocityLocal = vector3_t::Zero();
  vector3_t angularVelocityLocal = vector3_t::Zero();
};

/** `orientation` with its coefficients negated: the same rotation, the other
 * quaternion of the double cover. */
quaternion_t negated(const quaternion_t& orientation) {
  return quaternion_t(vector4_t(-orientation.coeffs()));
}

/** The controller on the lockstep link, its robot, and the record of the joint
 * actions it commanded. */
class Harness {
 public:
  Harness()
      : description_(checkedValue(
            robot::model::RobotDescription::Create(stack_.urdfFile()))),
        robotState_(description_),
        action_(description_) {
    InProcessMpcLink::Config config;
    config.execution = InProcessMpcLink::Execution::kCaller;
    controller_ = checkedValue(CentroidalMpcMrtJointController::Create(
        description_, stack_.modelSettings(), stack_.model(),
        InProcessMpcLink::factory(stack_.mpc(), std::move(config), &link_),
        stack_.pinocchioInterface(), stack_.pdGainsFile()));
    mpcJointIndices_ =
        description_.getJointIndices(stack_.modelSettings().mpcModelJointNames);
    // A plan whose joints lead the observed ones by 0.1 rad, so that the policy
    // commands something the measurement moves.
    const size_t jointStart = stack_.model().getJointStartindex();
    const size_t jointDim = stack_.model().getJointDim();
    stack_.mpc().solver().setPlan(
        [jointStart, jointDim](scalar_t /*time*/, scalar_t /*initTime*/,
                               const vector_t& initState) {
          vector_t planned = initState;
          planned.segment(jointStart, jointDim).array() += 0.1;
          return planned;
        });
  }

  AtlasReferenceStack& stack() { return stack_; }
  InProcessMpcLink& link() { return *link_; }
  CentroidalMpcMrtJointController& controller() { return *controller_; }

  /** The robot measured at `base`, its joints swaying about the task file's
   * initial state with `phase`. */
  void measure(const MeasuredBase& base, scalar_t phase) {
    const vector_t joints =
        stack_.model().getJointAngles(stack_.initialState());
    robotState_.setConfigurationToZero();
    robotState_.setTime(time_);
    robotState_.setRootPositionInWorldFrame(base.position);
    robotState_.setRootRotationLocalToWorldFrame(base.orientation);
    robotState_.setRootLinearVelocityInLocalFrame(base.linearVelocityLocal);
    robotState_.setRootAngularVelocityInLocalFrame(base.angularVelocityLocal);
    for (size_t i = 0; i < mpcJointIndices_.size(); ++i) {
      const scalar_t jointPhase = phase + static_cast<scalar_t>(i);
      robotState_.setJointPosition(mpcJointIndices_[i],
                                   joints[i] + 0.02 * std::sin(jointPhase));
      robotState_.setJointVelocity(mpcJointIndices_[i],
                                   0.1 * std::cos(jointPhase));
    }
    robotState_.setContactFlag(/*index=*/0, /*contactFlag=*/true);
    robotState_.setContactFlag(/*index=*/1, /*contactFlag=*/true);
  }

  /** startMpcThread() at the current measurement: on the lockstep link it
   * serves the start-up reset and starts no thread. */
  void start() { controller_->startMpcThread(robotState_); }

  /** One control cycle on the current measurement; its joint actions, as exact
   * text; the time then moves on. */
  std::string cycle() {
    controller_->computeJointControlAction(time_, robotState_, action_);
    std::string record;
    for (size_t joint = 0; joint < description_.getNumJoints(); ++joint) {
      const std::optional<robot::model::JointAction>& slot = action_.at(joint);
      if (!slot.has_value()) {
        absl::StrAppendFormat(&record, "%d none\n", joint);
        continue;
      }
      absl::StrAppendFormat(&record, "%d %a %a %a %a %a\n", joint, slot->q_des,
                            slot->qd_des, slot->kp, slot->kd,
                            slot->feed_forward_effort);
    }
    time_ += kControlPeriod;
    return record;
  }

  /** One solve of the scripted solver from the last observation; the state it
   * was handed. */
  vector_t solve() {
    EXPECT_TRUE(link_->runSolverIteration().status.ok());
    return stack_.mpc().solver().lastInitState();
  }

 private:
  AtlasReferenceStack stack_;
  robot::model::RobotDescription description_;
  robot::model::RobotState robotState_;
  robot::model::RobotJointAction action_;
  std::vector<size_t> mpcJointIndices_;
  scalar_t time_ = 1.0;
  InProcessMpcLink* absl_nullable link_ = nullptr;
  std::unique_ptr<CentroidalMpcMrtJointController> controller_;
};

/** The swaying, turned and tilted base of cycle `cycle`. */
MeasuredBase swayingBase(int cycle) {
  const scalar_t phase = 0.3 * static_cast<scalar_t>(cycle);
  MeasuredBase base;
  base.position = vector3_t(0.2 + 0.01 * std::sin(phase),
                            -0.1 + 0.005 * std::cos(phase), 0.9);
  base.orientation = quaternion_t(quaternionFromEulerZyx(
      vector3_t(2.5 + 0.02 * std::sin(phase), 0.08, -0.05 * std::cos(phase))));
  base.linearVelocityLocal = vector3_t(0.1 * std::cos(phase), -0.05, 0.02);
  base.angularVelocityLocal = vector3_t(0.03, -0.02 * std::sin(phase), 0.4);
  return base;
}

/**
 * Expects the solver states `a` and `b` of the same measurement to be one
 * state: every entry bit for bit but the base quaternion's, which are the same
 * or exactly negated, as `negatedQuaternion` says.
 */
void expectTheSameSolverState(const MpcRobotModelBase<scalar_t>& model,
                              const vector_t& a,
                              const vector_t& b,
                              bool negatedQuaternion) {
  ASSERT_EQ(a.size(), b.size());
  const Eigen::Index orientationStart =
      static_cast<Eigen::Index>(model.getBaseOrientationStartIndex());
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    const bool isQuaternion = i >= orientationStart && i < orientationStart + 4;
    EXPECT_EQ(a(i), isQuaternion && negatedQuaternion ? -b(i) : b(i))
        << "state entry " << i;
  }
}

/** Measures cycle `cycle`'s swaying base on `positive` with its quaternion, and
 * on `negative` with its negation. */
void measureWithBothQuaternions(Harness& positive,
                                Harness& negative,
                                int cycle) {
  const MeasuredBase base = swayingBase(cycle);
  MeasuredBase other = base;
  other.orientation = negated(base.orientation);
  positive.measure(base, /*phase=*/0.3 * static_cast<scalar_t>(cycle));
  negative.measure(other, /*phase=*/0.3 * static_cast<scalar_t>(cycle));
}

TEST(MrtObservationDoubleCover,
     BothQuaternionsOfAnAttitudeCommandTheSameJoints) {
  Harness positive;
  Harness negative;
  const MpcRobotModelBase<scalar_t>& model = positive.stack().model();
  int cycle = 0;
  measureWithBothQuaternions(positive, negative, cycle);
  positive.start();
  negative.start();
  for (int solve = 0; solve < 4; ++solve) {
    SCOPED_TRACE(solve);
    for (int k = 0; k < 3; ++k, ++cycle) {
      measureWithBothQuaternions(positive, negative, cycle);
      EXPECT_EQ(positive.cycle(), negative.cycle())
          << "cycle " << cycle << ": the joint commands differ";
    }
    const vector_t positiveState = positive.solve();
    const vector_t negativeState = negative.solve();
    expectTheSameSolverState(model, positiveState, negativeState,
                             /*negatedQuaternion=*/true);
    // Positive control: the two observations do carry opposite quaternions.
    EXPECT_GT(model.getBaseOrientation(positiveState)
                  .dot(model.getBaseOrientation(negativeState)),
              -1.0 - 1.0e-15);
    EXPECT_LT(model.getBaseOrientation(positiveState)
                  .dot(model.getBaseOrientation(negativeState)),
              -1.0 + 1.0e-15);
  }
}

/** [rad/s] The turn in place of the stream across pi. */
constexpr scalar_t kYawRate = 2.0;

/**
 * Measures cycle `cycle` of the turn in place through a heading of pi on
 * `continuous` with the continuous quaternion (0, 0, sin(psi / 2), cos(psi /
 * 2)), and on `canonicalized` with it canonicalized to w >= 0; whether that
 * flipped it.
 */
bool measureAcrossPi(Harness& continuous, Harness& canonicalized, int cycle) {
  MeasuredBase base;
  base.position = vector3_t(0.0, 0.0, 0.9);
  const scalar_t heading =
      M_PI - 0.1 + kYawRate * kControlPeriod * static_cast<scalar_t>(cycle);
  base.orientation =
      quaternion_t(Eigen::AngleAxis<scalar_t>(heading, vector3_t::UnitZ()));
  base.angularVelocityLocal = vector3_t(0.0, 0.0, kYawRate);
  MeasuredBase canonical = base;
  const bool flipped = canonical.orientation.w() < 0.0;
  if (flipped) canonical.orientation = negated(canonical.orientation);
  continuous.measure(base, /*phase=*/0.3 * static_cast<scalar_t>(cycle));
  canonicalized.measure(canonical,
                        /*phase=*/0.3 * static_cast<scalar_t>(cycle));
  return flipped;
}

TEST(
    MrtObservationDoubleCover,
    AStreamCanonicalizedToNonNegativeWAcrossPiCommandsWhatTheContinuousStreamDoes) {
  // Turning in place at 2 rad/s through a heading of pi: the continuous
  // quaternion has w < 0 past pi, where an estimator that keeps w >= 0 flips it
  // to its negation, a jump of the stored coefficients by 2.
  Harness continuous;
  Harness canonicalized;
  const MpcRobotModelBase<scalar_t>& model = continuous.stack().model();
  int flips = 0;
  int cycle = 0;
  measureAcrossPi(continuous, canonicalized, cycle);
  continuous.start();
  canonicalized.start();
  scalar_t previousHeading = std::numeric_limits<scalar_t>::quiet_NaN();
  for (int solve = 0; solve < 6; ++solve) {
    SCOPED_TRACE(solve);
    bool flipped = false;
    for (int k = 0; k < 2; ++k, ++cycle) {
      flipped = measureAcrossPi(continuous, canonicalized, cycle);
      if (flipped) ++flips;
      EXPECT_EQ(continuous.cycle(), canonicalized.cycle())
          << "cycle " << cycle << ": the joint commands differ";
    }
    const vector_t continuousState = continuous.solve();
    const vector_t canonicalizedState = canonicalized.solve();
    expectTheSameSolverState(model, continuousState, canonicalizedState,
                             flipped);
    // The heading the solver reads moves on continuously across pi, by the turn
    // of two cycles.
    const scalar_t heading = model.getBaseHeading(canonicalizedState);
    if (!std::isnan(previousHeading)) {
      EXPECT_NEAR(std::remainder(heading - previousHeading, 2.0 * M_PI),
                  2.0 * kYawRate * kControlPeriod, 1.0e-12);
    }
    previousHeading = heading;
  }
  EXPECT_GT(flips, 0)
      << "the stream must cross pi, or the canonicalization changes nothing";
}

}  // namespace
}  // namespace ocs2::humanoid
