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
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_sqp/SqpSolver.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/GaitFromConfig.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/ModeSequenceTemplate.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/gait_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_nmpc/humanoid_common_mpc/test/support/StatusTestMacros.h"
#include "robot_core/ResourcePaths.h"
#include "support/FourPointContactSet.h"
#include "support/TypedConfigFiles.h"

/**
 * The centroidal MPC trots the four point contacts of FourPointContactSet.h,
 * described by their legs LF, RF, LH and RH, on the trot of the shared
 * quadruped gait file: the gait schedule reads the file's leg names against
 * the contact set, the swing-foot cost tracks every contact to its
 * nominal_stance foothold, and several SQP iterations from the standing state
 * give a plan in which
 *  - every iterate and the plan are finite, the iterates become feasible and
 *    the merit settles within a bound of the first;
 *  - a contact in stance holds still;
 *  - a contact in swing rises off the ground at mid-swing;
 *  - it lands near the foothold the reference manager planned for it.
 *
 * The set is the DRC Atlas's sole corners: the toe and the heel of one foot
 * sit on the same ankle link, so a trot (left toe with right heel, then right
 * toe with left heel) pivots each foot about its stance corner. The swing
 * apex is therefore small, and the gait file's 0.35 s phases are used as
 * shipped: the swing planner's swing_time_scale (0.25 s on the Atlas) is
 * shorter, so the swing reaches its full apex without scaling.
 */
namespace ocs2::humanoid {
namespace {

using four_point_set::kAllFour;
using four_point_set::kFrontLeftHindRight;
using four_point_set::kFrontRightHindLeft;

constexpr size_t kNumPoints = four_point_set::kNumContacts;

constexpr char kQuadrupedGaitFile[] =
    "humanoid_nmpc/humanoid_common_mpc/config/command/quadruped_gait.textproto";
constexpr char kTrotGait[] = "trot";
// [s] The trot phase of the gait file (LF_RH, then RF_LH).
constexpr scalar_t kTrotPhaseDuration = 0.35;
// [s] Where the trot begins after standing on all four.
constexpr scalar_t kTrotStartTime = 0.2;

// The robot name the CppAD libraries of this test are built under
// (cppad_code_gen/cppad_centroidal_mpc_<robot name>), apart from those of the
// other four-point tests, which tape other cost and constraint sets.
constexpr char kRobotName[] = "atlas_quadruped_trot";
// [m] The swing apex above the touch-down height: a toe or heel lift of the
// pivoting foot, about 0.1 rad of ankle pitch on the 0.24 m Atlas sole.
constexpr scalar_t kSwingApexHeight = 0.025;
// The weight of a swing contact's x and y position error to its foothold; the
// shipped Atlas weighs neither (it ships no foothold). A swing pivots its foot
// about the foot's stance corner, and the plan yaws and rolls the foot outward
// at mid-swing, where the state cost's joint regularization is cheaper than
// keeping the swing corner on its line: at 30 the first run landed 9.8 mm from
// the foothold, at 300 still 9.6 mm, at 3000 at most 6.4 mm.
constexpr scalar_t kFootholdPositionWeight = 3000.0;
// The SQP iterations of the one solve the tests read.
constexpr int kSqpIterations = 8;

// [m/s] A stance contact's speed in the plan, twice the 5 mm/s of the first
// run. The hard zero_velocity constraint holds at the nodes; between them the
// contact drifts with the discretization.
constexpr scalar_t kMaxStanceSpeed = 1.0e-2;
// [m] A stance contact's distance from the surface in the plan; the first run
// drifted 0.1 mm.
constexpr scalar_t kMaxStanceHeight = 1.0e-3;
// [m] A swing contact's height above the surface at mid-swing: 80 % of the
// apex, which the first run reached (0.025 to 0.026 m).
constexpr scalar_t kMinMidSwingHeight = 0.8 * kSwingApexHeight;
// [m] A swing contact's horizontal distance to its foothold at touch-down;
// the first run's worst landing was 6.4 mm off (kFootholdPositionWeight).
constexpr scalar_t kMaxFootholdDistance = 1.0e-2;
// [m] A swing contact's height above the surface at touch-down; the first run
// landed within 0.5 mm.
constexpr scalar_t kMaxTouchDownHeight = 2.0e-3;
// [s] An interval of the time discretization shorter than this is the empty
// one between the two nodes of an event.
constexpr scalar_t kMinIntervalDuration = 1.0e-6;
// The merit of the first iterate is that of an infeasible guess (constraint
// violation 5e-5), so the merit may rise while the iterates become feasible;
// the first run's rose by 0.1 %. Bounded: within 1 % of the first iterate's.
constexpr scalar_t kMaxRelativeMeritChange = 1.0e-2;
// The relative change of the merit over the last iteration of a converged
// solve.
constexpr scalar_t kMeritConvergenceTolerance = 1.0e-4;

/** The legged four-point configuration with the swing-foot cost, the hard
 * normal-velocity servo and the nominal_stance foothold anchor. */
CentroidalMpcConfig trotConfig() {
  CentroidalMpcConfig config = four_point_set::leggedFourPointConfig(
      checkedValue(loadConfigOf(atlasFiles())));
  mpc_config::TaskFile& task = config.task;
  task.model_settings.robot_name = kRobotName;
  task.costs.emplace_back("task_space_foot_cost");
  task.hard_constraints.emplace_back("normal_velocity");
  task.nominal_foothold.anchor = "nominal_stance";
  task.swing_trajectory_config.apex_pos_foot_rt_swingend_ewrt_world_z =
      kSwingApexHeight;
  // The Atlas plans its sole center's touch-down 0.04 m above the ground
  // (task.textproto); a sole corner lands on it.
  task.swing_trajectory_config.touchdown_pos_foot_rt_surface_ewrt_world_z = 0.0;
  // NOLINTBEGIN(frame-naming): a per-component weight field
  task.task_space_foot_cost.weights.pos_x = kFootholdPositionWeight;
  task.task_space_foot_cost.weights.pos_y = kFootholdPositionWeight;
  // NOLINTEND(frame-naming)
  task.multiple_shooting.sqp_iteration = kSqpIterations;
  return config;
}

/** Every gait of a gait file by name. */
using GaitMap = std::map<std::string, ModeSequenceTemplate>;

/** The trot of the quadruped gait file, read against `layout`. */
absl::StatusOr<ModeSequenceTemplate> trotOf(const GaitModeLayout& layout) {
  ASSIGN_OR_RETURN(const std::string path,
                   robot::resolveResourcePath(kQuadrupedGaitFile));
  ASSIGN_OR_RETURN(const mpc_config::GaitFile file, loadGaitFile(path));
  ASSIGN_OR_RETURN(const GaitMap gaits, gaitMapFromConfig(file, layout));
  const GaitMap::const_iterator trot = gaits.find(kTrotGait);
  if (trot == gaits.end()) {
    return absl::NotFoundError(
        absl::StrCat(kQuadrupedGaitFile, " has no gait '", kTrotGait, "'"));
  }
  return trot->second;
}

/** Where a contact is and how fast it moves at one node of the plan. Passive
 * data. */
struct ContactMotion {
  vector3_t posContactRtWorldEwrtWorld = vector3_t::Zero();
  vector3_t linvelContactRtWorldEwrtWorld = vector3_t::Zero();
};

/** One swing of a contact inside the horizon. Passive data. */
struct SwingWindow {
  size_t contactIndex = 0;
  scalar_t liftOffTime = 0.0;    // [s]
  scalar_t touchDownTime = 0.0;  // [s]
};

/** Every swing of `schedule` that lifts off and touches down inside
 * (startTime, finalTime). */
std::vector<SwingWindow> swingsOf(const ModeSchedule& schedule,
                                  scalar_t startTime,
                                  scalar_t finalTime) {
  std::vector<SwingWindow> swings;
  for (size_t contact = 0; contact < kNumPoints; ++contact) {
    std::optional<scalar_t> liftOffTime;
    for (size_t event = 0; event < schedule.eventTimes.size(); ++event) {
      const bool before = contactIsActiveInMode(schedule.modeSequence[event],
                                                contact, kNumPoints);
      const bool after = contactIsActiveInMode(schedule.modeSequence[event + 1],
                                               contact, kNumPoints);
      const scalar_t time = schedule.eventTimes[event];
      if (before && !after) liftOffTime = time;
      if (!before && after && liftOffTime.has_value()) {
        if (*liftOffTime > startTime && time < finalTime) {
          swings.push_back({.contactIndex = contact,
                            .liftOffTime = *liftOffTime,
                            .touchDownTime = time});
        }
        liftOffTime.reset();
      }
    }
  }
  return swings;
}

/**
 * The plan of one SQP solve of the trot from the standing state, and the
 * interface it was solved on, built once for the whole suite: the interface
 * generates the CppAD libraries of the dynamics, the contact frames, the four
 * foot costs and the four normal-velocity servos.
 */
class PointContactQuadrupedTrotTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    interface_ = checkedValue(CentroidalMpcInterface::Create(
                                  trotConfig(), atlasFiles().urdfFile))
                     .release();
    SwitchedModelReferenceManager& referenceManager =
        *interface_->getSwitchedModelReferenceManagerPtr();
    GaitSchedule& gaitSchedule = *referenceManager.getGaitSchedule();
    const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
    gaitSchedule.insertModeSequenceTemplate(
        checkedValue(trotOf(gaitSchedule.modeLayout())), kTrotStartTime,
        /*finalTime=*/2.0 * horizon);
    const vector_t& state = interface_->getInitialState();
    const vector_t input = vector_t::Zero(static_cast<Eigen::Index>(
        interface_->getMpcRobotModel().getInputDim()));
    referenceManager.setTargetTrajectories(TargetTrajectories(
        {0.0, 2.0 * horizon}, {state, state}, {input, input}));

    solver_ = new SqpSolver(interface_->sqpSettings(),
                            interface_->getOptimalControlProblem(),
                            interface_->getInitializer());
    solver_->setReferenceManager(interface_->getReferenceManagerPtr());
    solver_->run(/*initTime=*/0.0, state, /*initMode=*/kAllFour,
                 /*finalTime=*/horizon);
    solution_ = new PrimalSolution();
    solver_->getPrimalSolution(horizon, solution_);
  }

  static void TearDownTestSuite() {
    delete solution_;
    solution_ = nullptr;
    delete solver_;
    solver_ = nullptr;
    delete interface_;
    interface_ = nullptr;
  }

  static CentroidalMpcInterface& interface() { return *interface_; }
  static const SqpSolver& solver() { return *solver_; }
  static const PrimalSolution& solution() { return *solution_; }
  static const SwitchedModelReferenceManager& referenceManager() {
    return *interface_->getSwitchedModelReferenceManagerPtr();
  }
  static scalar_t horizon() { return interface_->mpcSettings().timeHorizon_; }

  /** Every contact's motion at node `node` of the plan, in contact order. */
  static std::vector<ContactMotion> contactMotionAt(size_t node) {
    // getGeneralizedVelocities() updates the model's own Pinocchio data: a
    // copy.
    const std::unique_ptr<MpcRobotModelBase<scalar_t>> model(
        interface_->getMpcRobotModel().clone());
    const vector_t& state = solution_->stateTrajectory_[node];
    const vector_t& input = solution_->inputTrajectory_[node];
    const vector_t q = model->getGeneralizedCoordinates(state);
    const vector_t v = model->getGeneralizedVelocities(state, input);
    PinocchioInterface pinocchio = interface_->getPinocchioInterface();
    const pinocchio::Model& pinocchioModel = pinocchio.getModel();
    pinocchio::Data& data = pinocchio.getData();
    pinocchio::forwardKinematics(pinocchioModel, data, q, v);
    pinocchio::updateFramePlacements(pinocchioModel, data);
    std::vector<ContactMotion> motions;
    for (const std::string& name : interface_->modelSettings().contactNames) {
      const pinocchio::FrameIndex frame = pinocchioModel.getFrameId(name);
      motions.push_back(
          {.posContactRtWorldEwrtWorld = data.oMf[frame].translation(),
           .linvelContactRtWorldEwrtWorld =
               pinocchio::getFrameVelocity(pinocchioModel, data, frame,
                                           pinocchio::LOCAL_WORLD_ALIGNED)
                   .linear()});
    }
    return motions;
  }

  /** The node of the plan closest to `time`. */
  static size_t nodeAt(scalar_t time) {
    const scalar_array_t& times = solution_->timeTrajectory_;
    size_t closest = 0;
    for (size_t node = 1; node < times.size(); ++node) {
      if (std::abs(times[node] - time) < std::abs(times[closest] - time)) {
        closest = node;
      }
    }
    return closest;
  }

 private:
  static CentroidalMpcInterface* absl_nullable interface_;
  static SqpSolver* absl_nullable solver_;
  static PrimalSolution* absl_nullable solution_;
};

CentroidalMpcInterface* absl_nullable
PointContactQuadrupedTrotTest::interface_ = nullptr;
SqpSolver* absl_nullable PointContactQuadrupedTrotTest::solver_ = nullptr;
PrimalSolution* absl_nullable PointContactQuadrupedTrotTest::solution_ =
    nullptr;

TEST_F(PointContactQuadrupedTrotTest, theScheduleIsTheGaitFilesTrot) {
  const ModeSchedule& schedule = referenceManager().getModeSchedule();
  EXPECT_EQ(schedule.modeAtTime(0.5 * kTrotStartTime), kAllFour);
  for (size_t phase = 0; phase < 2; ++phase) {
    const scalar_t midPhase =
        kTrotStartTime +
        (static_cast<scalar_t>(phase) + 0.5) * kTrotPhaseDuration;
    EXPECT_EQ(schedule.modeAtTime(midPhase),
              phase % 2 == 0 ? kFrontLeftHindRight : kFrontRightHindLeft)
        << "phase " << phase;
  }
  EXPECT_EQ(
      modeNumber2String(kFrontLeftHindRight,
                        referenceManager().getGaitSchedule()->modeLayout()),
      "LF_RH");
}

TEST_F(PointContactQuadrupedTrotTest,
       everyIterateIsFiniteAndTheSolveConverges) {
  const std::vector<PerformanceIndex>& iterations = solver().getIterationsLog();
  ASSERT_GE(iterations.size(), 2U) << "the solve stopped after one iteration";
  for (size_t i = 0; i < iterations.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("iteration ", i));
    EXPECT_TRUE(std::isfinite(iterations[i].merit));
    EXPECT_TRUE(std::isfinite(iterations[i].cost));
    EXPECT_TRUE(std::isfinite(iterations[i].dynamicsViolationSSE));
    EXPECT_TRUE(std::isfinite(iterations[i].equalityConstraintsSSE));
  }
  const PerformanceIndex& first = iterations.front();
  const PerformanceIndex& last = iterations.back();
  const PerformanceIndex& secondToLast = iterations[iterations.size() - 2];
  EXPECT_LE(std::abs(last.merit - first.merit),
            kMaxRelativeMeritChange * std::abs(first.merit));
  EXPECT_LE(std::abs(last.merit - secondToLast.merit),
            kMeritConvergenceTolerance * std::abs(first.merit));
  EXPECT_LT(last.equalityConstraintsSSE + last.dynamicsViolationSSE,
            first.equalityConstraintsSSE + first.dynamicsViolationSSE);
  ASSERT_FALSE(solution().stateTrajectory_.empty());
  for (size_t node = 0; node < solution().stateTrajectory_.size(); ++node) {
    EXPECT_TRUE(solution().stateTrajectory_[node].allFinite()) << node;
    EXPECT_TRUE(solution().inputTrajectory_[node].allFinite()) << node;
  }
}

TEST_F(PointContactQuadrupedTrotTest, aStanceContactHoldsStill) {
  const ModeSchedule& schedule = referenceManager().getModeSchedule();
  const scalar_array_t& times = solution().timeTrajectory_;
  const scalar_t posSurfaceRtWorldEwrtWorldZ =
      referenceManager().getAppliedPosSurfaceRtWorldEwrtWorldZ();
  size_t checked = 0;
  for (size_t node = 0; node + 1 < times.size(); ++node) {
    // A node's input acts over the interval it starts. An event has two nodes
    // at its time: the one ending the interval before (an empty interval
    // here, with no path constraint) and the one starting the next mode's.
    if (times[node + 1] - times[node] < kMinIntervalDuration) continue;
    const size_t mode =
        schedule.modeAtTime(std::midpoint(times[node], times[node + 1]));
    // The first node of a mode starts with the momentum the swing ended with:
    // the centroidal model has no impact map, so a contact that touches down
    // there still moves at about the swing's touch-down speed.
    if (node > 0 && times[node] - times[node - 1] < kMinIntervalDuration) {
      continue;
    }
    const std::vector<ContactMotion> motions = contactMotionAt(node);
    for (size_t i = 0; i < kNumPoints; ++i) {
      if (!contactIsActiveInMode(mode, i, kNumPoints)) continue;
      SCOPED_TRACE(absl::StrCat(interface().modelSettings().contactNames[i],
                                " at ", times[node]));
      EXPECT_LT(motions[i].linvelContactRtWorldEwrtWorld.norm(),
                kMaxStanceSpeed);
      EXPECT_LT(std::abs(motions[i].posContactRtWorldEwrtWorld.z() -
                         posSurfaceRtWorldEwrtWorldZ),
                kMaxStanceHeight);
      ++checked;
    }
  }
  EXPECT_GT(checked, 0U);
}

TEST_F(PointContactQuadrupedTrotTest, aSwingContactLiftsAndLandsOnItsFoothold) {
  const std::vector<SwingWindow> swings = swingsOf(
      referenceManager().getModeSchedule(), /*startTime=*/0.0, horizon());
  // Each contact swings once inside the horizon: the first trot phase swings
  // RF and LH, the second LF and RH.
  ASSERT_EQ(swings.size(), kNumPoints);
  const scalar_t posSurfaceRtWorldEwrtWorldZ =
      referenceManager().getAppliedPosSurfaceRtWorldEwrtWorldZ();
  for (const SwingWindow& swing : swings) {
    const size_t i = swing.contactIndex;
    SCOPED_TRACE(absl::StrCat(interface().modelSettings().contactNames[i],
                              " swinging from ", swing.liftOffTime, " to ",
                              swing.touchDownTime));
    const std::vector<ContactMotion> midSwing = contactMotionAt(
        nodeAt(0.5 * (swing.liftOffTime + swing.touchDownTime)));
    EXPECT_GT(midSwing[i].posContactRtWorldEwrtWorld.z(),
              posSurfaceRtWorldEwrtWorldZ + kMinMidSwingHeight);

    const std::optional<vector2_t> plannedFoothold =
        referenceManager().getNomPosFootRtWorldEwrtWorldXy(i,
                                                           swing.touchDownTime);
    if (!plannedFoothold.has_value()) {
      ADD_FAILURE() << "the nominal_stance anchor plans no foothold";
      continue;
    }
    const vector2_t& nomPosFootRtWorldEwrtWorldXy = *plannedFoothold;
    const vector3_t posContactRtWorldEwrtWorld =
        contactMotionAt(nodeAt(swing.touchDownTime))[i]
            .posContactRtWorldEwrtWorld;
    EXPECT_LT(
        (posContactRtWorldEwrtWorld.head<2>() - nomPosFootRtWorldEwrtWorldXy)
            .norm(),
        kMaxFootholdDistance)
        << posContactRtWorldEwrtWorld.transpose() << " vs "
        << nomPosFootRtWorldEwrtWorldXy.transpose();
    EXPECT_LT(posContactRtWorldEwrtWorld.z() - posSurfaceRtWorldEwrtWorldZ,
              kMaxTouchDownHeight);
  }
}

}  // namespace
}  // namespace ocs2::humanoid
