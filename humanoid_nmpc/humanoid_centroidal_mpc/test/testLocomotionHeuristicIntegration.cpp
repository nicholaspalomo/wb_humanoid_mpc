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
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/cost/QuadraticStateCost.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/common/MpcFormulationConfig.h"
#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/cost/BasePoseShapedQuadraticStateCost.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/locomotion_heuristics_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "support/TypedConfigFiles.h"

namespace ocs2::humanoid {

namespace {

// Every seam test solves on the same periodic walk, one stride per second from every integer k:
//
//   [k, k + 0.1)        double support
//   [k + 0.1, k + 0.5)  LF: left stance, the RIGHT foot swings and lands at k + 0.5
//   [k + 0.5, k + 0.6)  double support
//   [k + 0.6, k + 1)    RF: right stance, the LEFT foot swings and lands at k + 1
//
// So each foot is down 0.6 s of its own 1 s stride (beta = 0.6), and every stance lasts 0.6 s: the right foot that
// lands at 1.5 lifts off again at 2.1. The numbers the tests expect are worked out by hand from this table, never by
// calling the code under test.
constexpr scalar_t kWalkStanceDuration = 0.6;

ModeSchedule walkSchedule() {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence{static_cast<size_t>(ModeNumber::kRf)};
  for (int k = -3; k <= 8; ++k) {
    const scalar_t start = static_cast<scalar_t>(k);
    for (const scalar_t offset : {0.0, 0.1, 0.5, 0.6}) eventTimes.push_back(start + offset);
    for (const ModeNumber mode : {ModeNumber::kStance, ModeNumber::kLf, ModeNumber::kStance, ModeNumber::kRf}) {
      modeSequence.push_back(static_cast<size_t>(mode));
    }
  }
  return ModeSchedule(eventTimes, modeSequence);
}

// A slower walk, one stride per 1.4 s from every 1.4 k:
//
//   [s, s + 0.1)        double support
//   [s + 0.1, s + 0.7)  LF: left stance, the RIGHT foot swings
//   [s + 0.7, s + 0.8)  double support
//   [s + 0.8, s + 1.4)  RF: right stance, the LEFT foot swings
//
// Each foot is down 0.8 s of its own 1.4 s stride (beta = 4/7). What matters is that 1.4 s does not divide the 1 s MPC
// horizon of the DRC Atlas: the contact fraction of a foot over the next horizon - the window-based beta this
// subsystem used to compute - then varies from node to node (0.4 to 0.75 here), whereas on the 1 s walk above it is 0.6
// at every node, the same number as the stride's own duty factor, and no test run on that walk can tell the two apart.
constexpr scalar_t kLongStride = 1.4;
constexpr scalar_t kLongStrideDutyFactor = 0.8 / 1.4;

ModeSchedule longStrideWalkSchedule() {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence{static_cast<size_t>(ModeNumber::kRf)};
  for (int k = -3; k <= 6; ++k) {
    const scalar_t start = kLongStride * static_cast<scalar_t>(k);
    for (const scalar_t offset : {0.0, 0.1, 0.7, 0.8}) eventTimes.push_back(start + offset);
    for (const ModeNumber mode : {ModeNumber::kStance, ModeNumber::kLf, ModeNumber::kStance, ModeNumber::kRf}) {
      modeSequence.push_back(static_cast<size_t>(mode));
    }
  }
  return ModeSchedule(eventTimes, modeSequence);
}

// A run, one stride per 0.8 s from every 0.8 k: LF [0.8k, 0.8k + 0.3), flight, RF [0.8k + 0.4, 0.8k + 0.7), flight.
// Each foot is down 0.3 s of its 0.8 s stride (beta = 0.375), and a flight phase has no foot on the ground at all.
constexpr scalar_t kRunDutyFactor = 0.375;

ModeSchedule runSchedule() {
  std::vector<scalar_t> eventTimes;
  std::vector<size_t> modeSequence{static_cast<size_t>(ModeNumber::kFly)};
  for (int k = -4; k <= 12; ++k) {
    const scalar_t start = 0.8 * static_cast<scalar_t>(k);
    for (const scalar_t offset : {0.0, 0.3, 0.4, 0.7}) eventTimes.push_back(start + offset);
    for (const ModeNumber mode : {ModeNumber::kLf, ModeNumber::kFly, ModeNumber::kRf, ModeNumber::kFly}) {
      modeSequence.push_back(static_cast<size_t>(mode));
    }
  }
  return ModeSchedule(eventTimes, modeSequence);
}

/**
 * Re-reads a layer's coefficients from its task file when it goes out of scope, however the test leaves it -
 * including through a failed ASSERT. The suite shares its interfaces across cases, so a case that retunes a layer and
 * does not put the file's numbers back hands the next case a different controller. (It used to "restore" with a
 * default-constructed configuration, whose zeros are not what the DRC Atlas ships, and passed only because gtest ran
 * the cases in declaration order.)
 */
class RestoreCoefficientsFromFileOnExit {
 public:
  RestoreCoefficientsFromFileOnExit(std::shared_ptr<LocomotionHeuristicLayer> layer, const mpc_config::TaskFile& task)
      : layer_(std::move(layer)), heuristics_(task.locomotion_heuristics) {}
  RestoreCoefficientsFromFileOnExit(const RestoreCoefficientsFromFileOnExit&) = delete;
  RestoreCoefficientsFromFileOnExit& operator=(const RestoreCoefficientsFromFileOnExit&) = delete;

  ~RestoreCoefficientsFromFileOnExit() {
    const absl::StatusOr<LocomotionHeuristicConfig> config = locomotionHeuristicConfigFromConfig(heuristics_);
    if (!config.ok()) {
      ADD_FAILURE() << "could not restore the coefficients of the task file: " << config.status().message();
      return;
    }
    const absl::Status status = layer_->reconfigure(*config);
    EXPECT_TRUE(status.ok()) << status.message();
  }

 private:
  std::shared_ptr<LocomotionHeuristicLayer> layer_;
  // The task file's locomotion_heuristics, which the layer is restored to.
  mpc_config::LocomotionHeuristicsConfig heuristics_;
};

/**
 * End-to-end test of the locomotion-heuristic layer on the DRC Atlas model: CentroidalMpcInterface::Create() reads the
 * `locomotion_heuristics` block of a task file, derives the model constants, builds the layer and installs it on the
 * reference manager, and each seam is then read back through the same accessors the costs use, after a real
 * preSolverRun() on a real walking schedule.
 *
 * Five interfaces, built once for the whole suite because each build compiles or loads the CppAD libraries:
 *
 *  - `shipped_`: the task file exactly as it ships, all three lists empty. The PARITY interface: with nothing listed
 *    every reference must be bit for bit what it was before the layer existed.
 *  - `enabled_`: base_pose [orientation_compensation, height_compensation], foothold [hip_centered_stepping,
 *    translational_stepping], wrench [impulse_scaling]. The base-pose, hip-anchored foothold, vertical wrench,
 *    terminal-cost and torso-cost seams. Runs without com_and_acom_tracking_cost, with terminal_cost in place of
 *    dcm_terminal_cost and a base-pose block in Q_final, so that the quadratic terminal cost exists and weighs the
 *    channels the heuristics shape.
 *  - `stanceAnchored_`: base_pose [periodic_orientation], foothold [translational_stepping], wrench
 *    [centripetal_acceleration]. The gait-phase seam, the stance-foot-anchored foothold path, and the world-frame
 *    wrench path.
 *  - `zeroStepWidth_`: foothold [hip_centered_stepping] with nominal_foothold.step_width 0, the configuration of the
 *    SA01, G1 and R1, where the listed heuristic is the only reason anything is measured at all.
 *  - `basePoseOnly_`: the shipped file with base_pose [height_compensation] and nothing else. No foothold heuristic,
 *    so the landing target is the nominal step exactly as on `shipped_`; what differs is that a heuristic is listed.
 *
 * Only the first build compiles the CppAD libraries; the rest load them from the model folder, so an interface after
 * the first costs about a second.
 *
 * Tests that retune coefficients go through reconfigure() and restore the file's numbers on exit, so every case sees
 * the configuration its task file describes whatever order the cases run in. The error paths are covered without an
 * interface in humanoid_nmpc/humanoid_common_mpc/test/testLocomotionHeuristics.cpp; only the ones that have to travel
 * through the task file are repeated here.
 */
class LocomotionHeuristicIntegrationTest : public ::testing::Test {
 protected:
  /** The interface of `config` and the DRC Atlas URDF, owned by the suite; null (and a failure) when it is refused. */
  static CentroidalMpcInterface* absl_nullable build(const CentroidalMpcConfig& config) {
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface = CentroidalMpcInterface::Create(config, files().urdfFile);
    EXPECT_TRUE(interface.ok()) << interface.status().message();
    return interface.ok() ? interface->release() : nullptr;
  }

  /** The shipped DRC Atlas configuration with its three locomotion_heuristics lists replaced. */
  static CentroidalMpcConfig withLists(const std::vector<std::string>& basePose,
                                       const std::vector<std::string>& foothold,
                                       const std::vector<std::string>& wrench) {
    CentroidalMpcConfig config = *shippedConfig_;
    EXPECT_TRUE(config.task.locomotion_heuristics.base_pose.empty() && config.task.locomotion_heuristics.foothold.empty() &&
                config.task.locomotion_heuristics.wrench.empty())
        << "the shipped locomotion_heuristics lists are no longer empty";
    config.task.locomotion_heuristics.base_pose = basePose;
    config.task.locomotion_heuristics.foothold = foothold;
    config.task.locomotion_heuristics.wrench = wrench;
    config.task.contact_schedule_source = std::string(kGaitScheduleContactScheduleSource);
    return config;
  }

  /**
   * `config` with the foot cost's xy weights raised: at the shipped 0 the landing target is computed and multiplied by
   * zero, which is the configuration start-up warns about.
   */
  static void raiseFootPositionWeights(CentroidalMpcConfig& config) {
    config.task.task_space_foot_cost.weights.pos_x = 50.0;
    config.task.task_space_foot_cost.weights.pos_y = 50.0;
  }

  static void SetUpTestSuite() {
    absl::StatusOr<CentroidalMpcConfig> shipped = loadConfigOf(files());
    ASSERT_TRUE(shipped.ok()) << shipped.status();
    *shippedConfig_ = *std::move(shipped);
    shipped_ = build(*shippedConfig_);

    CentroidalMpcConfig enabled = withLists({"orientation_compensation", "height_compensation"},
                                            {"hip_centered_stepping", "translational_stepping"}, {"impulse_scaling"});
    // The quadratic terminal cost only exists with the DCM terminal cost off, and weighs the base pose only without
    // com_and_acom_tracking_cost (the factory zeroes final_state_weights' base-pose block otherwise). The shipped
    // final_state_weights has that block at zero too, so it is given one: a terminal cost that weighs nothing cannot
    // show it pulls towards anything.
    std::vector<std::string>& costs = enabled.task.costs;
    costs.erase(std::remove(costs.begin(), costs.end(), "com_and_acom_tracking_cost"), costs.end());
    std::replace(costs.begin(), costs.end(), std::string("dcm_terminal_cost"), std::string("terminal_cost"));
    enabled.task.final_state_weights.base_position = mpc_config::Xyz{.x = 0.0, .y = 0.0, .z = 20.0};
    enabled.task.final_state_weights.base_orientation = mpc_config::YawPitchRoll{.yaw = 0.0, .pitch = 5.0, .roll = 5.0};
    raiseFootPositionWeights(enabled);
    *enabledConfig_ = enabled;
    enabled_ = build(enabled);

    CentroidalMpcConfig stanceAnchored = withLists({"periodic_orientation"}, {"translational_stepping"}, {"centripetal_acceleration"});
    raiseFootPositionWeights(stanceAnchored);
    *stanceAnchoredConfig_ = stanceAnchored;
    stanceAnchored_ = build(stanceAnchored);

    CentroidalMpcConfig zeroStepWidth = withLists(/*basePose=*/{}, {"hip_centered_stepping"}, /*wrench=*/{});
    zeroStepWidth.task.nominal_foothold.step_width = 0.0;
    raiseFootPositionWeights(zeroStepWidth);
    *zeroStepWidthConfig_ = zeroStepWidth;
    zeroStepWidth_ = build(zeroStepWidth);

    basePoseOnly_ = build(withLists({"height_compensation"}, /*foothold=*/{}, /*wrench=*/{}));

    // Read here, straight after the build and before any case runs a pre-solve, because it is a property of the
    // FIRST solve and the interfaces are shared.
    if (stanceAnchored_ != nullptr) {
      footholdExistedBeforeFirstSolve_ =
          stanceAnchored_->getSwitchedModelReferenceManagerPtr()->nominalFoothold(kContactRightIndex, /*time=*/1.5).has_value();
    }
  }

  /** The DRC Atlas files, from the test's runfiles. */
  static const CentroidalRobotFiles& files() {
    static const absl::NoDestructor<CentroidalRobotFiles> kFiles(atlasFiles());
    return *kFiles;
  }

  static void TearDownTestSuite() {
    delete shipped_;
    delete enabled_;
    delete stanceAnchored_;
    delete zeroStepWidth_;
    delete basePoseOnly_;
    shipped_ = nullptr;
    enabled_ = nullptr;
    stanceAnchored_ = nullptr;
    zeroStepWidth_ = nullptr;
    basePoseOnly_ = nullptr;
  }

  void SetUp() override {
    ASSERT_NE(shipped_, nullptr) << "the shipped interface was not built; see the SetUpTestSuite failure";
    ASSERT_NE(enabled_, nullptr) << "the enabled interface was not built; see the SetUpTestSuite failure";
    ASSERT_NE(stanceAnchored_, nullptr) << "the stance-anchored interface was not built; see the SetUpTestSuite failure";
    ASSERT_NE(zeroStepWidth_, nullptr) << "the zero-step-width interface was not built; see the SetUpTestSuite failure";
    ASSERT_NE(basePoseOnly_, nullptr) << "the base-pose-only interface was not built; see the SetUpTestSuite failure";
  }

  /**
   * The initial state carrying a command: world CoM velocity `worldVelocity`, heading `yaw`, and `normalizedYawMomentum`
   * in the angular-momentum channel. The momentum part of the state is NORMALIZED, [p/m, L/m], so index 5 is L_z/m,
   * which the target calculator writes as I_zz * psidot / m.
   */
  static vector_t commandedState(const CentroidalMpcInterface& interface,
                                 const vector2_t& worldVelocity,
                                 scalar_t yaw,
                                 scalar_t normalizedYawMomentum = 0.0) {
    const CentroidalMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
    vector_t state = interface.getInitialState();
    model.setBaseComLinearVelocity(state, vector3_t(worldVelocity.x(), worldVelocity.y(), 0.0));
    vector6_t basePose = model.getBasePose(state);
    basePose(3) = yaw;
    model.setBasePose(state, basePose);
    state(5) = normalizedYawMomentum;
    return state;
  }

  /** The initial state with the base moved to `basePosition` in the ground plane, joints and heading untouched. */
  static vector_t stateWithBaseAt(const CentroidalMpcInterface& interface, const vector2_t& basePosition) {
    const CentroidalMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
    vector_t state = interface.getInitialState();
    vector3_t position = model.getBasePosition(state);
    position.head<2>() = basePosition;
    model.setBasePosition(state, position);
    return state;
  }

  /** A target holding `state` over all time, with an all-zero input of the dimension the OCP's costs are handed. */
  static TargetTrajectories constantTarget(const CentroidalMpcInterface& interface, const vector_t& state) {
    const vector_t zeroInput = vector_t::Zero(interface.getEffectiveMpcRobotModel().getInputDim());
    return TargetTrajectories({-100.0, 100.0}, {state, state}, {zeroInput, zeroInput});
  }

  /**
   * One pre-solve at `time`, as the MPC loop runs before every solve: the schedule, the target and the measured state
   * all reach the reference manager the way they do on the robot.
   */
  static void solveAt(const CentroidalMpcInterface& interface,
                      const ModeSchedule& schedule,
                      const TargetTrajectories& target,
                      const vector_t& state,
                      scalar_t time) {
    const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = interface.getSwitchedModelReferenceManagerPtr();
    // Through the GAIT schedule: modifyReferences() rebuilds the mode schedule from it on every pre-solve, so a schedule
    // set on the manager's own buffer would be thrown away.
    referenceManager->getGaitSchedule()->updateModeSchedule(schedule);
    referenceManager->setTargetTrajectories(target);
    referenceManager->preSolverRun(time, time + interface.mpcSettings().timeHorizon_, state, schedule.modeAtTime(time));
  }

  /** The file's configuration, for a case to retune a few coefficients of while keeping the rest. */
  static LocomotionHeuristicConfig fileConfig(const mpc_config::TaskFile& task) {
    const absl::StatusOr<LocomotionHeuristicConfig> config = locomotionHeuristicConfigFromConfig(task.locomotion_heuristics);
    EXPECT_TRUE(config.ok()) << config.status().message();
    return config.ok() ? *config : LocomotionHeuristicConfig();
  }

  /** [N] m g, from the model, with the same 9.81 the reference manager and weightCompensatingInput() use. */
  static scalar_t totalWeight(CentroidalMpcInterface& interface) {
    return pinocchio::computeTotalMass(interface.getPinocchioInterface().getModel()) * 9.81;
  }

  static scalar_t totalMass(CentroidalMpcInterface& interface) {
    return pinocchio::computeTotalMass(interface.getPinocchioInterface().getModel());
  }

  /** [kg m^2] The composite yaw inertia about the center of mass at `state`: the I_zz of h_z = I_zz psidot. */
  static scalar_t compositeYawInertia(CentroidalMpcInterface& interface, const vector_t& state) {
    const PinocchioInterface::Model& model = interface.getPinocchioInterface().getModel();
    PinocchioInterface::Data data(model);
    const vector_t q = interface.getMpcRobotModel().getGeneralizedCoordinates(state);
    pinocchio::ccrba(model, data, q, vector_t::Zero(model.nv));
    return data.Ig.inertia().matrix()(2, 2);
  }

  /** The world placement of frame `frameName` at `state`, by forward kinematics independent of any MPC code. */
  static pinocchio::SE3 framePlacement(CentroidalMpcInterface& interface, const vector_t& state, const std::string& frameName) {
    const PinocchioInterface::Model& model = interface.getPinocchioInterface().getModel();
    PinocchioInterface::Data data(model);
    const vector_t q = interface.getMpcRobotModel().getGeneralizedCoordinates(state);
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    return data.oMf[model.getFrameId(frameName)];
  }

  static vector2_t footPosition(CentroidalMpcInterface& interface, const vector_t& state, size_t contactIndex) {
    return framePlacement(interface, state, interface.modelSettings().contactNames[contactIndex]).translation().head<2>();
  }

  static CentroidalMpcInterface* absl_nullable shipped_;
  static CentroidalMpcInterface* absl_nullable enabled_;
  static CentroidalMpcInterface* absl_nullable stanceAnchored_;
  static CentroidalMpcInterface* absl_nullable zeroStepWidth_;
  static CentroidalMpcInterface* absl_nullable basePoseOnly_;
  // The configurations `shipped_`, `enabled_`, `stanceAnchored_` and `zeroStepWidth_` were built from.
  static absl::NoDestructor<CentroidalMpcConfig> shippedConfig_;
  static absl::NoDestructor<CentroidalMpcConfig> enabledConfig_;
  static absl::NoDestructor<CentroidalMpcConfig> stanceAnchoredConfig_;
  static absl::NoDestructor<CentroidalMpcConfig> zeroStepWidthConfig_;
  static bool footholdExistedBeforeFirstSolve_;
};

CentroidalMpcInterface* absl_nullable LocomotionHeuristicIntegrationTest::shipped_ = nullptr;
CentroidalMpcInterface* absl_nullable LocomotionHeuristicIntegrationTest::enabled_ = nullptr;
CentroidalMpcInterface* absl_nullable LocomotionHeuristicIntegrationTest::stanceAnchored_ = nullptr;
CentroidalMpcInterface* absl_nullable LocomotionHeuristicIntegrationTest::zeroStepWidth_ = nullptr;
CentroidalMpcInterface* absl_nullable LocomotionHeuristicIntegrationTest::basePoseOnly_ = nullptr;
absl::NoDestructor<CentroidalMpcConfig> LocomotionHeuristicIntegrationTest::shippedConfig_;
absl::NoDestructor<CentroidalMpcConfig> LocomotionHeuristicIntegrationTest::enabledConfig_;
absl::NoDestructor<CentroidalMpcConfig> LocomotionHeuristicIntegrationTest::stanceAnchoredConfig_;
absl::NoDestructor<CentroidalMpcConfig> LocomotionHeuristicIntegrationTest::zeroStepWidthConfig_;
bool LocomotionHeuristicIntegrationTest::footholdExistedBeforeFirstSolve_ = true;

size_t numberOfStanceFeet(const contact_flag_t& flags) {
  // NOLINTNEXTLINE(argument-comment): `true` is the value counted, which libstdc++ names __value.
  return static_cast<size_t>(std::count(flags.begin(), flags.end(), true));
}

}  // namespace

/******************************************************************************************************/
/*                                   PARITY: the shipped, empty lists                                  */
/******************************************************************************************************/

TEST_F(LocomotionHeuristicIntegrationTest, ShippedTaskFileBuildsAnEmptyLayerAndInstallsIt) {
  // Every robot in this repository ships the block with all three lists empty, and this is the test that says so: the
  // file on disk, unmodified, must produce a layer that changes nothing.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = shipped_->getLocomotionHeuristicLayerPtr();
  ASSERT_NE(layer, nullptr);
  EXPECT_TRUE(layer->empty()) << "the shipped drc_atlas task file lists a heuristic:\n" << layer->summary();
  EXPECT_FALSE(layer->wrenchNeedsWorldFrame());
  EXPECT_EQ(shipped_->getSwitchedModelReferenceManagerPtr()->getLocomotionHeuristicLayer(), layer)
      << "the interface must install the layer it built on the reference manager";
}

TEST_F(LocomotionHeuristicIntegrationTest, EmptyListsLeaveTheBasePoseReferenceBitForBitUnchanged) {
  // Parity has to be shown at a point where a leaking heuristic WOULD show: a command, a heading off the world axes, and
  // a gait phase mid-stance. At the initial state all of those are zero and every shipped coefficient multiplies a
  // zero, so the old version of this test passed whatever the seam did with an empty list. The target also carries yaw
  // momentum, which the reference manager reads back as a commanded yaw rate on every robot with a nominal step width
  // (see NominalStepIsAcrossTheHeadingPredictedForTheTouchDown), so a leaking yaw-rate term would show here too.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = shipped_->getSwitchedModelReferenceManagerPtr();
  const vector_t& state = shipped_->getInitialState();
  const vector_t target = commandedState(*shipped_, vector2_t(-0.3, 1.0), M_PI_2, /*normalizedYawMomentum=*/0.05);
  const TargetTrajectories targetTrajectories = constantTarget(*shipped_, target);
  solveAt(*shipped_, walkSchedule(), targetTrajectories, state, /*time=*/1.02);
  const scalar_t time = 1.2;
  ASSERT_NEAR(referenceManager->getPhaseVariable(time), 0.125, 1.0e-12) << "the gait phase would be degenerate here";

  const vector_t shaped = referenceManager->shapeBasePose(time, target);
  ASSERT_EQ(shaped.size(), target.size());
  EXPECT_TRUE(shaped == target) << "shapeBasePose must return its input bit for bit with no base-pose heuristic listed:\n"
                                << (shaped - target).transpose();
  const vector_t desiredState = referenceManager->getDesiredState(targetTrajectories, state, time);
  const vector6_t desiredBasePose = shipped_->getMpcRobotModel().getBasePose(desiredState);
  const vector6_t targetBasePose = shipped_->getMpcRobotModel().getBasePose(target);
  EXPECT_TRUE(desiredBasePose == targetBasePose) << "the running state costs' reference base pose moved:\n"
                                                 << (desiredBasePose - targetBasePose).transpose();

  // Positive control: the same point on an interface that lists orientation_compensation is NOT a no-op, so the
  // equality above is a statement about the empty list rather than about the point.
  RestoreCoefficientsFromFileOnExit restore(enabled_->getLocomotionHeuristicLayerPtr(), enabledConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  ASSERT_TRUE(enabled_->getLocomotionHeuristicLayerPtr()->reconfigure(config).ok());
  const vector_t enabledTarget = commandedState(*enabled_, vector2_t(-0.3, 1.0), M_PI_2, /*normalizedYawMomentum=*/0.05);
  solveAt(*enabled_, walkSchedule(), constantTarget(*enabled_, enabledTarget), enabled_->getInitialState(), /*time=*/1.02);
  EXPECT_FALSE(enabled_->getSwitchedModelReferenceManagerPtr()->shapeBasePose(time, enabledTarget) == enabledTarget);
}

TEST_F(LocomotionHeuristicIntegrationTest, EmptyListsLeaveTheContactForceReferenceAtWeightCompensation) {
  // The old version of this test read getDesiredInput() before any pre-solve, when the schedule is OCS2's default
  // single FLY mode: both vectors were zero and it could not fail. It also compared against the WRENCH model while the
  // reference manager runs on the basis-vector model, whose input has a different length. Here the walk is installed
  // first, every node checked has a foot on the ground, and the comparison is against the model the OCP uses.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = shipped_->getSwitchedModelReferenceManagerPtr();
  const MpcRobotModelBase<scalar_t>& effectiveModel = shipped_->getEffectiveMpcRobotModel();
  const vector_t& state = shipped_->getInitialState();
  const TargetTrajectories target = constantTarget(*shipped_, state);
  solveAt(*shipped_, walkSchedule(), target, state, /*time=*/1.02);
  const scalar_t weight = totalWeight(*shipped_);

  struct Node {
    scalar_t time;
    contact_flag_t contacts;
  };
  const std::vector<Node> nodes = {{1.05, {true, true}}, {1.3, {true, false}}, {1.8, {false, true}}};
  for (const Node& node : nodes) {
    const contact_flag_t flags = referenceManager->getContactFlags(node.time);
    ASSERT_EQ(flags, node.contacts) << "the walk did not reach the reference manager at t = " << node.time;
    const size_t numStance = numberOfStanceFeet(flags);
    ASSERT_GT(numStance, 0u) << "a node with no stance foot compares two zero vectors";

    const vector_t desired = referenceManager->getDesiredInput(target, state, node.time);
    const vector_t weightCompensation = weightCompensatingInput(shipped_->getPinocchioInterface(), flags, effectiveModel);
    ASSERT_EQ(desired.size(), static_cast<Eigen::Index>(effectiveModel.getInputDim()));
    ASSERT_EQ(desired.size(), weightCompensation.size());
    EXPECT_TRUE(desired == weightCompensation) << "t = " << node.time;

    // And independently of that call: W/n straight up on every stance foot and nothing at all on a swing foot.
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      if (flags[foot]) {
        const vector3_t force = effectiveModel.getContactForce(desired, foot);
        EXPECT_NEAR(force.z(), weight / static_cast<scalar_t>(numStance), 1.0e-9 * weight) << "t = " << node.time << ", foot " << foot;
        EXPECT_NEAR(force.head<2>().norm(), 0.0, 1.0e-9 * weight) << "t = " << node.time << ", foot " << foot;
      } else {
        EXPECT_TRUE(effectiveModel.getContactWrench(desired, foot).isZero(0.0)) << "t = " << node.time << ", foot " << foot;
      }
    }
  }
}

TEST_F(LocomotionHeuristicIntegrationTest, EmptyFootholdListKeepsTheNominalStepStillThroughTheSwing) {
  // With no foothold heuristic the landing target is the nominal step: a step width to this foot's side of the STANCE
  // foot, carried forward at the command over one step, from the stance foot's own touch-down to this one. Both ends
  // are fixed instants, so the target must not move while the solver re-evaluates it through the swing. It used to be
  // carried forward from the LAST SOLVE instead, which shrinks to nothing as the swing ends: the target slid back onto
  // the stance foot, a zero-length step at any speed.
  //
  // The robot is held still: this path reads the stance foot and not the measured base, and a synthetic state that
  // moved the base would drag the stance foot along with it through the kinematics.
  //
  // At heading 0 and again turned to 0.7 rad, with no yaw rate commanded: the step width is across the heading, so at
  // 0.7 the right foot's side is w * (sin 0.7, -cos 0.7) and not -w y. A lateral left unrotated, or rotated the wrong
  // way, is 0.31 m or 0.58 m off there, where at heading 0 all three agree.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = shipped_->getSwitchedModelReferenceManagerPtr();
  const ModeSchedule schedule = walkSchedule();
  const vector2_t command(0.5, 0.1);
  const scalar_t stepWidth = shipped_->modelSettings().nominalFootholdConfig.stepWidth;
  ASSERT_GT(stepWidth, 0.0) << "the DRC Atlas is the robot that ships a nominal step width";

  for (const scalar_t heading : {0.0, 0.7}) {
    const vector_t state = commandedState(*shipped_, vector2_t::Zero(), heading);
    const TargetTrajectories target = constantTarget(*shipped_, commandedState(*shipped_, command, heading));
    solveAt(*shipped_, schedule, target, state, /*time=*/1.05);  // double support: both feet are measured
    // The right foot swings [1.1, 1.5); its stance foot, the left, landed at 1.0.
    const vector2_t expected = footPosition(*shipped_, state, kContactLeftIndex) + command * (1.5 - 1.0) +
                               stepWidth * vector2_t(std::sin(heading), -std::cos(heading));
    for (int k = 0; k < 10; ++k) {
      const scalar_t solveTime = 1.12 + 0.04 * static_cast<scalar_t>(k);
      solveAt(*shipped_, schedule, target, state, solveTime);
      const std::optional<vector2_t> foothold = referenceManager->nominalFoothold(kContactRightIndex, /*time=*/1.5);
      if (!foothold.has_value()) {
        GTEST_FAIL() << "heading " << heading << ", solve at " << solveTime;
      }
      EXPECT_NEAR(foothold->x(), expected.x(), 1.0e-9) << "heading " << heading << ", solve at " << solveTime;
      EXPECT_NEAR(foothold->y(), expected.y(), 1.0e-9) << "heading " << heading << ", solve at " << solveTime;
    }
  }
}

TEST_F(LocomotionHeuristicIntegrationTest, NominalStepIsAcrossTheHeadingPredictedForTheTouchDown) {
  // The nominal step, like every foothold, is laid out across the heading PREDICTED for the touch-down: measured at the
  // solve and carried forward by the commanded yaw rate. Here the robot is measured at heading 0 in double support at
  // 1.05 and told to turn at 0.5 rad/s on the spot while walking at (0.5, 0.1): the right foot lands at 1.5 on heading
  // 0.225, a step width across it from the left foot plus half a second of travel; the left lands at 2.0 on heading
  // 0.475, from the right foot. The measured heading of 0 instead puts them 0.10 m and 0.21 m away.
  //
  // Run on the shipped, empty layer and on `basePoseOnly_`, which lists a base-pose heuristic but no foothold one, so
  // both take the nominal-step path. They must give the same targets, since the base_pose list has nothing to do with
  // footholds. The shipped half used to fail: captureMeasuredState() latched the yaw inertia it inverts the momentum
  // target with only when some list was non-empty, so with every list empty getCommandedYawRate() read 0, the step was
  // laid across the MEASURED heading, and listing any heuristic at all moved the feet on a turn.
  struct Case {
    const char* absl_nonnull name;
    CentroidalMpcInterface* absl_nonnull interface;
  };
  const std::vector<Case> cases = {{"shipped", shipped_}, {"basePoseOnly", basePoseOnly_}};
  for (const Case& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    CentroidalMpcInterface& interface = *testCase.interface;
    const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = interface.getSwitchedModelReferenceManagerPtr();
    ASSERT_TRUE(interface.getLocomotionHeuristicLayerPtr()->footholdEmpty()) << "this must be the nominal-step path";
    const scalar_t stepWidth = interface.modelSettings().nominalFootholdConfig.stepWidth;
    ASSERT_GT(stepWidth, 0.0);
    const scalar_t yawRate = 0.5;
    const vector2_t command(0.5, 0.1);
    const scalar_t solveTime = 1.05;
    const vector_t state = interface.getInitialState();
    const vector_t target =
        commandedState(interface, command, /*yaw=*/0.0, compositeYawInertia(interface, state) * yawRate / totalMass(interface));
    solveAt(interface, walkSchedule(), constantTarget(interface, target), state, solveTime);
    ASSERT_NEAR(referenceManager->getCommandedYawRate(1.5), yawRate, 1.0e-9) << "the yaw rate did not reach the reference manager";

    struct Landing {
      size_t foot;
      size_t stanceFoot;
      scalar_t side;
      scalar_t touchDown;
      scalar_t stanceLanded;
    };
    const std::vector<Landing> landings = {{kContactRightIndex, kContactLeftIndex, -1.0, 1.5, 1.0},
                                           {kContactLeftIndex, kContactRightIndex, 1.0, 2.0, 1.5}};
    for (const Landing& landing : landings) {
      const scalar_t predictedYaw = yawRate * (landing.touchDown - solveTime);
      const vector2_t travel = command * (landing.touchDown - landing.stanceLanded);
      const vector2_t stanceFoot = footPosition(interface, state, landing.stanceFoot);
      const vector2_t expected =
          stanceFoot + travel + landing.side * stepWidth * vector2_t(-std::sin(predictedYaw), std::cos(predictedYaw));
      const std::optional<vector2_t> foothold = referenceManager->nominalFoothold(landing.foot, landing.touchDown);
      if (!foothold.has_value()) {
        GTEST_FAIL();
      }
      EXPECT_LT((*foothold - expected).cwiseAbs().maxCoeff(), 1.0e-9) << "foot " << landing.foot << " landing at " << landing.touchDown
                                                                      << ": " << foothold->transpose() << " vs " << expected.transpose();
      // Control: the step across the MEASURED heading is far from it, so the check above can see a stale heading.
      const vector2_t acrossMeasuredHeading = stanceFoot + travel + vector2_t(0.0, landing.side * stepWidth);
      EXPECT_GT((expected - acrossMeasuredHeading).norm(), 0.09) << "foot " << landing.foot;
    }
  }
}

TEST_F(LocomotionHeuristicIntegrationTest, PhaseVariableMapsLeftStanceToTheFirstHalfAndFreezesThroughDoubleSupport) {
  // periodic_orientation and the derived roll phase of every robot's task file are written against this mapping:
  // [0, 0.5) is LF - left STANCE, the right foot in the air - and [0.5, 1) is RF, frozen at 0.5 through the double
  // support after LF and at 0 through the one after RF. An LF/RF swap would flip the sign of the periodic roll, so the
  // reference would drop the stance-side hip instead of the swing-side one.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = shipped_->getSwitchedModelReferenceManagerPtr();
  const vector_t& state = shipped_->getInitialState();
  // The walk, cut short so that its LAST event is the touch-down that ends a left stance: LF [8.1, 8.5), then nothing
  // but the gait schedule's default STANCE from 8.5 on. (Its final RF event is dropped; getModeSchedule() replaces the
  // last mode with STANCE either way.)
  ModeSchedule schedule = walkSchedule();
  ASSERT_NEAR(schedule.eventTimes.back(), 8.6, 1.0e-12);
  schedule.eventTimes.pop_back();
  schedule.modeSequence.pop_back();
  solveAt(*shipped_, schedule, constantTarget(*shipped_, state), state, /*time=*/1.02);

  struct Sample {
    scalar_t time;
    scalar_t phase;
  };
  // LF [1.1, 1.5), DS [1.5, 1.6), RF [1.6, 2.0), DS [2.0, 2.1): 0.5 * elapsed / 0.4 into each single support. And the
  // last left stance of the schedule, LF [8.1, 8.5), read like every other one.
  const std::vector<Sample> samples = {{1.2, 0.125}, {1.4, 0.375}, {1.52, 0.5}, {1.58, 0.5}, {1.7, 0.625},
                                       {1.9, 0.875}, {2.02, 0.0},  {2.08, 0.0}, {8.3, 0.25}};
  for (const Sample& sample : samples) {
    EXPECT_NEAR(referenceManager->getPhaseVariable(sample.time), sample.phase, 1.0e-12) << "t = " << sample.time;
  }
  // Past the last scheduled event there is nothing to interpolate between, and the degenerate case answers the start of
  // a cycle. Without the guard, upper_bound() returns end(), the mode there is STANCE, and the double-support branch
  // looks just before the last event, finds the left stance that ended there, and answers 0.5 - which is why the
  // schedule above ends where it does. (On the uncut walk it ends after a double support and the unguarded code also
  // answered 0.) Before the FIRST event the same guard cannot be observed from here: the swing planner throws on a
  // schedule whose first phase is a swing, so the first mode is always STANCE, and the unguarded code would then decide
  // between 0 and 0.5 on a value read from before the event array, which only a sanitizer can see. That case is not
  // asserted, because no answer it gives could fail against the bug.
  EXPECT_EQ(referenceManager->getPhaseVariable(50.0), 0.0);
}

/******************************************************************************************************/
/*                                     The base-pose seam                                              */
/******************************************************************************************************/

TEST_F(LocomotionHeuristicIntegrationTest, TheShippedCoefficientsReachTheReferenceWhenAHeuristicIsListed) {
  // The DRC Atlas and the EngineAI SA01 ship DERIVED coefficients rather than zeros - regenerate them with
  // `make derive-heuristic-parameters ROBOT=drc_atlas` - so listing a name on those two is not a no-op. The LIST is
  // still the switch: it is empty on every robot. What is pinned here is that the number in the file is the number
  // that reaches the reference, read off the file rather than hard-coded because it is derived and will move.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  ASSERT_FALSE(enabled_->getLocomotionHeuristicLayerPtr()->basePoseEmpty()) << "the test's task file did not take effect";
  const vector_t& state = enabled_->getInitialState();
  const TargetTrajectories target = constantTarget(*enabled_, commandedState(*enabled_, vector2_t(1.0, 0.0), /*yaw=*/0.0));
  solveAt(*enabled_, walkSchedule(), target, state, /*time=*/1.02);
  const vector6_t basePose = enabled_->getMpcRobotModel().getBasePose(referenceManager->getDesiredState(target, state, /*time=*/1.2));

  const mpc_config::LocomotionHeuristicsConfig::OrientationCompensation& shipped =
      shippedConfig_->task.locomotion_heuristics.orientation_compensation;
  const scalar_t pitchPerForwardVelocity = shipped.pitch_per_forward_velocity;
  const scalar_t pitchOffset = shipped.pitch_offset;
  const scalar_t maximumTilt = shipped.maximum_tilt;
  EXPECT_NE(pitchPerForwardVelocity, 0.0) << "the DRC Atlas is expected to ship a derived, non-zero lean";
  // Index 4 is PITCH: the base pose is Euler ZYX with yaw FIRST, and a 1 m/s forward command at yaw 0 is 1 m/s forward.
  EXPECT_NEAR(basePose(4), std::clamp(pitchPerForwardVelocity * 1.0 + pitchOffset, -maximumTilt, maximumTilt), 1.0e-12)
      << "the shipped coefficient must reach the pitch reference";
}

TEST_F(LocomotionHeuristicIntegrationTest, BasePoseSeamRotatesTheWorldCommandIntoTheReferenceHeading) {
  // At reference yaw pi/2 a world command (-0.3, 1.0) is (1.0, 0.3) in the robot's own frame: 1 m/s forward, 0.3 left.
  // So pitch = 0.05 * 1.0, roll = 0.1 * 0.3, and the height polynomial is evaluated at |v| = sqrt(1.09):
  // 0.01 * 1.09 - 0.02 * 1.044031 + 0.003 = -0.006980613017821. A transposed rotation gives (-1.0, -0.3) and leans the
  // reference nose-UP and to the right; an offset written into the wrong index of [x, y, z, yaw, pitch, roll] shows up
  // as a moved entry that should have stayed put. At yaw 0 none of this was visible.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, enabledConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.orientationCompensation = OrientationCompensationParameters{};
  config.orientationCompensation.rollPerLateralVelocity = 0.1;
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  config.orientationCompensation.maximumTilt = 0.35;
  config.heightCompensation = HeightCompensationParameters{};
  config.heightCompensation.heightPerSpeedSquared = 0.01;
  config.heightCompensation.heightPerSpeed = -0.02;
  config.heightCompensation.heightOffset = 0.003;
  config.heightCompensation.maximumHeightOffset = 0.05;
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  const vector_t& state = enabled_->getInitialState();
  const vector_t target = commandedState(*enabled_, vector2_t(-0.3, 1.0), M_PI_2);
  const TargetTrajectories targetTrajectories = constantTarget(*enabled_, target);
  solveAt(*enabled_, walkSchedule(), targetTrajectories, state, /*time=*/1.02);
  const scalar_t time = 1.2;

  vector_t expected = target;
  expected(6 + 2) += -0.006980613017821;  // height, base pose index 2
  expected(6 + 4) += 0.05;                // pitch, base pose index 4
  expected(6 + 5) += 0.03;                // roll, base pose index 5
  const vector_t shaped = referenceManager->shapeBasePose(time, target);
  ASSERT_EQ(shaped.size(), target.size());
  for (Eigen::Index i = 0; i < target.size(); ++i) {
    if (i == 6 + 2 || i == 6 + 4 || i == 6 + 5) {
      EXPECT_NEAR(shaped(i), expected(i), 1.0e-12) << "state entry " << i;
    } else {
      EXPECT_EQ(shaped(i), target(i)) << "state entry " << i << " is not a channel the base-pose heuristics write";
    }
  }
  // The running state costs read the same shaped pose through getDesiredState().
  const vector6_t desiredBasePose =
      enabled_->getMpcRobotModel().getBasePose(referenceManager->getDesiredState(targetTrajectories, state, time));
  EXPECT_LT((desiredBasePose - enabled_->getMpcRobotModel().getBasePose(expected)).cwiseAbs().maxCoeff(), 1.0e-12);
}

TEST_F(LocomotionHeuristicIntegrationTest, BasePoseSeamReadsTheOperatorCommandNotTheMomentumOfTheStateItShapes) {
  // Bledt's H_Theta and H_z are functions of the COMMANDED velocity. Under online contact planning the
  // planned_com_override rule rewrites the target's momentum channel with the plan's own swaying CoM velocity, and only
  // getCommandedVelocity() still carries the operator's command. So the seam must read that accessor, and not the
  // momentum of the state it is shaping: here the two disagree, and the pitch has to follow the command (1 m/s
  // forward), not the state's channel (-1 m/s, which would lean the reference the other way).
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, enabledConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.orientationCompensation = OrientationCompensationParameters{};
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  config.heightCompensation = HeightCompensationParameters{};
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  const vector_t commanded = commandedState(*enabled_, vector2_t(1.0, 0.0), /*yaw=*/0.0);
  solveAt(*enabled_, walkSchedule(), constantTarget(*enabled_, commanded), enabled_->getInitialState(), /*time=*/1.02);
  const vector_t otherMomentum = commandedState(*enabled_, vector2_t(-1.0, 0.0), /*yaw=*/0.0);
  const vector_t shaped = referenceManager->shapeBasePose(/*time=*/1.2, otherMomentum);
  EXPECT_NEAR(shaped(6 + 4) - otherMomentum(6 + 4), 0.05, 1.0e-12);
}

TEST_F(LocomotionHeuristicIntegrationTest, PeriodicOrientationFollowsTheGaitPhaseOfTheSchedule) {
  // roll = 0.03 sin(2 pi phi), pitch = 0.02 sin(4 pi phi + 0.3), at phi = 0.125 (early left stance, t = 1.2) and
  // phi = 0.625 (early right stance, t = 1.7): roll +/-0.021213203435596, pitch 0.019106729782512 at both. The roll
  // must change sign between the two stances; with the phase read at the degenerate initial state it never moved.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = stanceAnchored_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, stanceAnchoredConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(stanceAnchoredConfig_->task);
  config.periodicOrientation.rollAmplitude = 0.03;
  config.periodicOrientation.rollPhaseRate = 2.0 * M_PI;
  config.periodicOrientation.rollPhaseOffset = 0.0;
  config.periodicOrientation.pitchAmplitude = 0.02;
  config.periodicOrientation.pitchPhaseRate = 4.0 * M_PI;
  config.periodicOrientation.pitchPhaseOffset = 0.3;
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = stanceAnchored_->getSwitchedModelReferenceManagerPtr();
  const vector_t& target = stanceAnchored_->getInitialState();
  solveAt(*stanceAnchored_, walkSchedule(), constantTarget(*stanceAnchored_, target), target, /*time=*/1.02);

  struct Sample {
    scalar_t time;
    scalar_t roll;
    scalar_t pitch;
  };
  const std::vector<Sample> samples = {{1.2, 0.021213203435596, 0.019106729782512}, {1.7, -0.021213203435596, 0.019106729782512}};
  for (const Sample& sample : samples) {
    const vector_t shaped = referenceManager->shapeBasePose(sample.time, target);
    EXPECT_NEAR(shaped(6 + 5) - target(6 + 5), sample.roll, 1.0e-12) << "roll at t = " << sample.time;
    EXPECT_NEAR(shaped(6 + 4) - target(6 + 4), sample.pitch, 1.0e-12) << "pitch at t = " << sample.time;
    EXPECT_EQ(shaped(6 + 2), target(6 + 2)) << "periodic_orientation has no height channel";
    EXPECT_EQ(shaped(6 + 3), target(6 + 3)) << "nor a yaw channel";
  }
}

TEST_F(LocomotionHeuristicIntegrationTest, TerminalCostIsZeroAtTheShapedReference) {
  // Q_final used to measure the deviation from the UNSHAPED target, so with a base-pose heuristic listed the last node
  // of every horizon was pulled back to level pitch and roll and the unoffset height - weighted by terminalCostScaling
  // on one node, more than the running horizon's whole base-pose weight. The terminal cost must now vanish at the
  // shaped pose and charge the unshaped one exactly the offset.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, enabledConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.orientationCompensation = OrientationCompensationParameters{};
  config.orientationCompensation.rollPerLateralVelocity = 0.1;
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  config.heightCompensation = HeightCompensationParameters{};
  config.heightCompensation.heightPerSpeed = -0.02;
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const vector2_t command(0.8, 0.2);
  const vector_t target = commandedState(*enabled_, command, /*yaw=*/0.0);
  const TargetTrajectories targetTrajectories = constantTarget(*enabled_, target);
  solveAt(*enabled_, walkSchedule(), targetTrajectories, enabled_->getInitialState(), /*time=*/1.02);
  const scalar_t time = 1.2;

  QuadraticStateCost& terminalCost = enabled_->getOptimalControlProblem().finalCostPtr->get<QuadraticStateCost>("terminalCost");
  EXPECT_NE(dynamic_cast<const BasePoseShapedQuadraticStateCost*>(&terminalCost), nullptr)
      << "the factory must install the shaped terminal cost";

  // Shaped by hand: pitch 0.05 * 0.8, roll 0.1 * 0.2, height -0.02 * |(0.8, 0.2)|.
  vector_t shaped = target;
  shaped(6 + 2) += -0.02 * std::hypot(0.8, 0.2);
  shaped(6 + 4) += 0.05 * 0.8;
  shaped(6 + 5) += 0.1 * 0.2;
  EXPECT_LT(terminalCost.getValue(time, shaped, targetTrajectories, PreComputation()), 1.0e-20);

  // Positive control, and the exact charge for standing on the unshaped target: 0.5 d' Q_final d with d the offsets.
  const absl::StatusOr<matrix_t> terminalWeights =
      stateWeightsFromConfig(enabledConfig_->task.final_state_weights,
                             stateInputLayout(enabled_->modelSettings(), StateInputLayout::Mpc::kCentroidal), "final_state_weights");
  ASSERT_TRUE(terminalWeights.ok()) << terminalWeights.status();
  ASSERT_TRUE(enabledConfig_->task.terminal_cost_scaling.has_value());
  const scalar_t terminalCostScaling = enabledConfig_->task.terminal_cost_scaling.value_or(0.0);
  const vector_t deviation = target - shaped;
  const scalar_t expected = 0.5 * terminalCostScaling * deviation.dot(*terminalWeights * deviation);
  ASSERT_GT(expected, 1.0e-3) << "the enabled configuration's final_state_weights weighs no base-pose channel, so this would test nothing";
  EXPECT_NEAR(terminalCost.getValue(time, target, targetTrajectories, PreComputation()), expected, 1.0e-9 * expected);
}

TEST_F(LocomotionHeuristicIntegrationTest, TorsoTaskSpaceCostTracksTheShapedBaseOrientation) {
  // The torso task-space cost builds its orientation reference by forward kinematics of the reference state. It used to
  // take the UNSHAPED target, so its orientation weights pulled the torso level while Q pulled the base into the lean,
  // and the stiffer of the two canceled the heuristic without a word. With a pure pitch offset at zero yaw and roll
  // the torso reference must be the torso at the shaped state, rotated from the unshaped one by exactly that pitch
  // about the world y axis (R_torso = R_base R_joints, and R_y(d) R_y(0)^T = R_y(d)).
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, enabledConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.orientationCompensation = OrientationCompensationParameters{};
  config.orientationCompensation.pitchPerForwardVelocity = 0.05;
  config.heightCompensation = HeightCompensationParameters{};
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const vector_t target = commandedState(*enabled_, vector2_t(1.0, 0.0), /*yaw=*/0.0);
  const TargetTrajectories targetTrajectories = constantTarget(*enabled_, target);
  solveAt(*enabled_, walkSchedule(), targetTrajectories, enabled_->getInitialState(), /*time=*/1.02);
  const scalar_t time = 1.2;

  // The Atlas task file's `task_space_costs.torso` block, on link `utorso`.
  EndEffectorKinematicsQuadraticCost& torsoCost =
      enabled_->getOptimalControlProblem().costPtr->get<EndEffectorKinematicsQuadraticCost>("torso_TaskSpaceKinematicsCost");
  const vector_t parameters = torsoCost.getParameters(time, targetTrajectories, PreComputation());
  // Parameters: [position(3), orientation quaternion coefficients (x, y, z, w), linear velocity(3), angular velocity(3), weights].
  const matrix3_t referenceRotation = Eigen::Quaternion<scalar_t>(vector4_t(parameters.segment<4>(3))).normalized().toRotationMatrix();

  vector_t shaped = target;
  shaped(6 + 4) += 0.05;
  const matrix3_t shapedRotation = framePlacement(*enabled_, shaped, "utorso").rotation();
  EXPECT_LT((referenceRotation - shapedRotation).cwiseAbs().maxCoeff(), 1.0e-9) << "the torso reference is not the shaped torso";

  const matrix3_t unshapedRotation = framePlacement(*enabled_, target, "utorso").rotation();
  const Eigen::AngleAxis<scalar_t> difference(matrix3_t(referenceRotation * unshapedRotation.transpose()));
  EXPECT_NEAR(difference.angle(), 0.05, 1.0e-9) << "the torso reference must lean by the pitch offset";
  EXPECT_NEAR(difference.axis().y(), 1.0, 1.0e-9) << "about the world y axis, nose down";
}

/******************************************************************************************************/
/*                                       The wrench seam                                               */
/******************************************************************************************************/

TEST_F(LocomotionHeuristicIntegrationTest, ImpulseScalingSplitsTheWeightByDutyFactorAndAveragesToTheWeightOverAStride) {
  // f_z = W / (F beta) on every stance foot, F = 2 feet, beta = 4/7 on the 1.4 s walk: 7 W / 8 = 0.875 W on each foot
  // in double support (1.75 times static weight compensation, since a single support is coming) and 0.875 W on the one
  // foot in single support. Over one stride that is (0.2 s * 2 + 1.2 s * 1) * 0.875 W / 1.4 s = W, the impulse budget
  // the heuristic exists for.
  //
  // Run on the 1.4 s walk rather than the 1 s one because the stride must not divide the 1 s horizon: the old
  // window-based beta, the contact fraction over [t, t + horizon], is 0.4 or 0.75 at the nodes below instead of 4/7,
  // which asks 1.25 W or 0.67 W of a foot and averages 1.21 W over the stride - where on the 1 s walk it was 0.6
  // everywhere and indistinguishable. The old "W / beta at every instant" form puts 1.75 W on the ground at every
  // node; dropping the weight-compensation baseline from the vertical path asks each foot for the offset alone.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, enabledConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.impulseScaling.scale = 1.0;
  config.impulseScaling.minimumDutyFactor = 0.2;
  config.impulseScaling.maximumForceRatio = 3.0;
  ASSERT_TRUE(layer->reconfigure(config).ok());
  ASSERT_FALSE(layer->wrenchNeedsWorldFrame()) << "impulse_scaling is vertical and must take the input-frame path";

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  const MpcRobotModelBase<scalar_t>& effectiveModel = enabled_->getEffectiveMpcRobotModel();
  const vector_t& state = enabled_->getInitialState();
  const TargetTrajectories target = constantTarget(*enabled_, state);
  solveAt(*enabled_, longStrideWalkSchedule(), target, state, /*time=*/1.42);
  const scalar_t weight = totalWeight(*enabled_);
  const scalar_t perStanceFoot = weight / (2.0 * kLongStrideDutyFactor);
  ASSERT_NEAR(perStanceFoot, 0.875 * weight, 1.0e-12 * weight);

  struct Node {
    scalar_t time;
    contact_flag_t contacts;
  };
  // One node in each phase of the stride [1.4, 2.8): DS [1.4, 1.5), LF [1.5, 2.1), DS [2.1, 2.2), RF [2.2, 2.8).
  const std::vector<Node> nodes = {{1.45, {true, true}}, {1.8, {true, false}}, {2.15, {true, true}}, {2.5, {false, true}}};
  for (const Node& node : nodes) {
    ASSERT_EQ(referenceManager->getContactFlags(node.time), node.contacts) << "t = " << node.time;
    const vector_t input = referenceManager->getDesiredInput(target, state, node.time);
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      if (node.contacts[foot]) {
        // Read in the input's own frame, which is what the vertical path writes: a vertical force is the same there.
        const vector3_t force = effectiveModel.getContactForce(input, foot);
        EXPECT_NEAR(force.z(), perStanceFoot, 1.0e-9 * weight) << "t = " << node.time << ", foot " << foot;
        EXPECT_NEAR(force.head<2>().norm(), 0.0, 1.0e-9 * weight) << "t = " << node.time << ", foot " << foot;
      } else {
        EXPECT_TRUE(effectiveModel.getContactWrench(input, foot).isZero(0.0)) << "t = " << node.time << ", foot " << foot;
      }
    }
  }

  // The time mean over one whole stride, [1.4, 2.8), at the midpoints of 0.01 s cells that the phase boundaries fall
  // between.
  scalar_t totalVertical = 0.0;
  const int numCells = 140;
  for (int i = 0; i < numCells; ++i) {
    const scalar_t time = kLongStride + (static_cast<scalar_t>(i) + 0.5) * 0.01;
    const vector_t input = referenceManager->getDesiredInput(target, state, time);
    for (size_t foot = 0; foot < kNumContacts; ++foot) totalVertical += effectiveModel.getContactForce(input, foot).z();
  }
  EXPECT_NEAR(totalVertical / static_cast<scalar_t>(numCells), weight, 1.0e-6 * weight);
}

TEST_F(LocomotionHeuristicIntegrationTest, ImpulseScalingAsksNothingOfAFlightPhaseAndStillAveragesToTheWeight) {
  // On a run each foot is down 0.3 s of its 0.8 s stride, so a single-support foot is asked for W / (2 * 0.375) =
  // 1.333 W, and a flight phase - no foot on the ground - for nothing at all: 1 / beta is unbounded there and a force
  // reference on a foot in the air is one no foot can track. (0.3 + 0.3) s * 1.333 W / 0.8 s = W again.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, enabledConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.impulseScaling.scale = 1.0;
  config.impulseScaling.minimumDutyFactor = 0.2;
  config.impulseScaling.maximumForceRatio = 3.0;
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  const MpcRobotModelBase<scalar_t>& effectiveModel = enabled_->getEffectiveMpcRobotModel();
  const vector_t& state = enabled_->getInitialState();
  const TargetTrajectories target = constantTarget(*enabled_, state);
  solveAt(*enabled_, runSchedule(), target, state, /*time=*/1.62);
  const scalar_t weight = totalWeight(*enabled_);

  // LF [1.6, 1.9): the positive control for the flight check below.
  ASSERT_EQ(referenceManager->getContactFlags(1.75), contact_flag_t({true, false}));
  const vector_t singleSupport = referenceManager->getDesiredInput(target, state, /*time=*/1.75);
  EXPECT_NEAR(effectiveModel.getContactForce(singleSupport, kContactLeftIndex).z(), weight / (2.0 * kRunDutyFactor), 1.0e-9 * weight);

  // Flight [1.9, 2.0).
  ASSERT_EQ(referenceManager->getContactFlags(1.95), contact_flag_t({false, false}));
  const vector_t flight = referenceManager->getDesiredInput(target, state, /*time=*/1.95);
  EXPECT_TRUE(flight.isZero(0.0)) << flight.transpose();

  scalar_t totalVertical = 0.0;
  const int numCells = 80;
  for (int i = 0; i < numCells; ++i) {
    const scalar_t time = 1.6 + (static_cast<scalar_t>(i) + 0.5) * 0.01;
    const vector_t input = referenceManager->getDesiredInput(target, state, time);
    for (size_t foot = 0; foot < kNumContacts; ++foot) totalVertical += effectiveModel.getContactForce(input, foot).z();
  }
  EXPECT_NEAR(totalVertical / static_cast<scalar_t>(numCells), weight, 1.0e-6 * weight);
}

TEST_F(LocomotionHeuristicIntegrationTest, CentripetalForceReachesTheWorldFrameReference) {
  // m omega x v with omega = 0.5 rad/s about z and v = (1.0, 0.4): m * 0.5 * (-0.4, 1.0), shared over the stance
  // feet, on top of W / n straight up - in the WORLD frame, which on the basis-vector model means the whole reference
  // is rotated into each foot's local contact frame. Read back through the world-frame accessor. The yaw rate is not
  // in the target as such: it is recovered from the momentum channel as m h_z / I_zz, so this also pins that inversion.
  //
  // Measured on a robot TURNED to heading 0.7. At heading 0 the Atlas's sole frames are the world frame to 2e-16 in
  // the initial posture, so writing the world force into the local frame unrotated - getDesiredInput() taking the
  // input-frame path despite wrenchNeedsWorldFrame(), setContactForce() in place of setContactForceInWorldFrame(), or
  // a transposed rotation - gave bit for bit the same numbers there. Here each of them reads back as the right force
  // turned by 0.7 or 1.4 rad, tens of newtons off on the horizontal force of single support.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = stanceAnchored_->getLocomotionHeuristicLayerPtr();
  RestoreCoefficientsFromFileOnExit restore(layer, stanceAnchoredConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(stanceAnchoredConfig_->task);
  config.centripetalAcceleration.scale = 1.0;
  config.centripetalAcceleration.maximumForce = 0.0;
  config.centripetalAcceleration.maximumForceRatioOfWeight = 0.3;  // 0.3 W per foot: far above the ~0.03 W asked for here
  ASSERT_TRUE(layer->reconfigure(config).ok());
  ASSERT_TRUE(layer->wrenchNeedsWorldFrame());
  ASSERT_TRUE(stanceAnchored_->usesContactBasisVectorInputs()) << "the world-frame path is only a rotation on the basis-vector model";

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = stanceAnchored_->getSwitchedModelReferenceManagerPtr();
  const MpcRobotModelBase<scalar_t>& effectiveModel = stanceAnchored_->getEffectiveMpcRobotModel();
  const scalar_t heading = 0.7;
  // The measured state, turned: every pose-dependent read of the world-frame path (the contact-frame rotation, and the
  // yaw inertia the momentum channel is inverted with) is evaluated here. I_zz does not depend on the heading.
  const vector_t state = commandedState(*stanceAnchored_, vector2_t::Zero(), heading);
  const scalar_t mass = totalMass(*stanceAnchored_);
  const scalar_t weight = totalWeight(*stanceAnchored_);
  const scalar_t yawRate = 0.5;
  const vector2_t command(1.0, 0.4);  // in the WORLD, which is the frame the target carries it in
  const TargetTrajectories target = constantTarget(
      *stanceAnchored_, commandedState(*stanceAnchored_, command, heading, compositeYawInertia(*stanceAnchored_, state) * yawRate / mass));
  solveAt(*stanceAnchored_, walkSchedule(), target, state, /*time=*/1.02);

  // Precondition, by forward kinematics independent of the model under test: each sole frame is the world frame turned
  // by the heading, so the local and world frames really do differ at this point.
  const matrix3_t turned = Eigen::AngleAxis<scalar_t>(heading, vector3_t::UnitZ()).toRotationMatrix();
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    const matrix3_t soleRotation = framePlacement(*stanceAnchored_, state, stanceAnchored_->modelSettings().contactNames[foot]).rotation();
    ASSERT_LT((soleRotation - turned).cwiseAbs().maxCoeff(), 1.0e-9) << "foot " << foot << " is not flat and turned by the heading";
  }

  struct Node {
    scalar_t time;
    contact_flag_t contacts;
  };
  // Double support, t = 1.05: per foot m * 0.5 * (-0.4, 1.0) / 2 = m * (-0.1, 0.25), and W / 2. Left stance, t = 1.3:
  // the one stance foot carries all of it, m * (-0.2, 0.5) and W; the swing foot nothing.
  const std::vector<Node> nodes = {{1.05, {true, true}}, {1.3, {true, false}}};
  for (const Node& node : nodes) {
    ASSERT_EQ(referenceManager->getContactFlags(node.time), node.contacts) << "t = " << node.time;
    const scalar_t numStance = static_cast<scalar_t>(numberOfStanceFeet(node.contacts));
    const vector3_t expectedWorld(mass * yawRate * -command.y() / numStance, mass * yawRate * command.x() / numStance, weight / numStance);
    const vector_t input = referenceManager->getDesiredInput(target, state, node.time);
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      if (!node.contacts[foot]) {
        EXPECT_TRUE(effectiveModel.getContactWrench(input, foot).isZero(0.0)) << "t = " << node.time << ", foot " << foot;
        continue;
      }
      const vector3_t world = effectiveModel.getContactForceInWorldFrame(state, input, foot);
      EXPECT_LT((world - expectedWorld).cwiseAbs().maxCoeff(), 1.0e-6)
          << "t = " << node.time << ", foot " << foot << ": " << world.transpose() << " vs " << expectedWorld.transpose();
      // Control: what the input itself holds is that force in the sole's frame, R_z(-heading) times it, and it differs
      // from the world force by more than the tolerance above - so the check above could tell the frames apart.
      const vector3_t local = effectiveModel.getContactForce(input, foot);
      const vector3_t expectedLocal = turned.transpose() * expectedWorld;
      EXPECT_LT((local - expectedLocal).cwiseAbs().maxCoeff(), 1.0e-6)
          << "t = " << node.time << ", foot " << foot << ": " << local.transpose() << " vs " << expectedLocal.transpose();
      EXPECT_GT((expectedLocal - expectedWorld).head<2>().norm(), 10.0) << "the local and world forces coincide at this point";
    }
  }
}

/******************************************************************************************************/
/*                                       The foothold seam                                             */
/******************************************************************************************************/

TEST_F(LocomotionHeuristicIntegrationTest, FootholdSeamHasNoOpinionBeforeTheFirstSolve) {
  // Before the first pre-solve nothing has been measured, so there is nowhere to anchor a landing target: the seam must
  // say so rather than place the foot relative to a zero-initialized base at the world origin.
  EXPECT_FALSE(footholdExistedBeforeFirstSolve_);

  // Positive control: one pre-solve later it has one, and the swing reference is built on it.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = stanceAnchored_->getSwitchedModelReferenceManagerPtr();
  const vector_t& state = stanceAnchored_->getInitialState();
  solveAt(*stanceAnchored_, walkSchedule(), constantTarget(*stanceAnchored_, state), state, /*time=*/1.12);
  EXPECT_TRUE(referenceManager->nominalFoothold(kContactRightIndex, /*time=*/1.5).has_value());
  EXPECT_TRUE(referenceManager->getSwingFootReference(kContactRightIndex, /*time=*/1.3).has_value());
}

TEST_F(LocomotionHeuristicIntegrationTest, TranslationalSteppingLandingTargetHoldsStillThroughTheSwing) {
  // translational_stepping alone, so the anchor is the stance-foot one: along the heading, the base PREDICTED at the
  // touch-down; across it, the stance foot plus a step width. The robot walks at the command, v = (0.5, 0), so the base
  // measured at each solve is 0.5 t and the prediction at the touch-down never changes: neither may the target.
  //
  // Right foot, landing at 1.5 on a 0.6 s stance: x = 0.5 * 1.5 + (0.5 * 0.6) * 0.5 = 0.9, y = left foot - 0.45 - 0.05
  // (the lateral offset is signed by the foot's own side). Left foot, landing at 2.0: x = 0.5 * 2.0 + 0.15 = 1.15,
  // y = right foot + 0.45 + 0.05. A flipped side sign narrows the stance by twice the offset instead of widening it.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = stanceAnchored_->getLocomotionHeuristicLayerPtr();
  ASSERT_FALSE(layer->footholdMovesAnchor()) << "this interface must exercise the stance-foot anchor";
  RestoreCoefficientsFromFileOnExit restore(layer, stanceAnchoredConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(stanceAnchoredConfig_->task);
  config.translationalStepping = TranslationalSteppingParameters{};
  config.translationalStepping.forwardStanceFraction = 0.5;
  config.translationalStepping.lateralStanceFraction = 0.5;
  config.translationalStepping.lateralOffset = 0.05;
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = stanceAnchored_->getSwitchedModelReferenceManagerPtr();
  const ModeSchedule schedule = walkSchedule();
  const scalar_t speed = 0.5;
  const TargetTrajectories target = constantTarget(*stanceAnchored_, commandedState(*stanceAnchored_, vector2_t(speed, 0.0), /*yaw=*/0.0));
  const scalar_t stepWidth = stanceAnchored_->modelSettings().nominalFootholdConfig.stepWidth;
  ASSERT_GT(stepWidth, 0.0);
  const scalar_t lateralOffset = 0.05;

  // Double support: both feet measured, and the right foot's lift-off position is this one.
  const vector_t liftOffState = stateWithBaseAt(*stanceAnchored_, vector2_t(speed * 1.05, 0.0));
  solveAt(*stanceAnchored_, schedule, target, liftOffState, /*time=*/1.05);
  const vector2_t rightLiftOff = footPosition(*stanceAnchored_, liftOffState, kContactRightIndex);
  const scalar_t leftFootY = footPosition(*stanceAnchored_, liftOffState, kContactLeftIndex).y();
  const vector2_t expectedRight(speed * 1.5 + 0.5 * kWalkStanceDuration * speed, leftFootY - stepWidth - lateralOffset);
  const vector2_t expectedLeft(speed * 2.0 + 0.5 * kWalkStanceDuration * speed, rightLiftOff.y() + stepWidth + lateralOffset);

  // The swing-reference check below is chosen by INDEX: it used to be guarded by `solveTime > 1.28 && solveTime <
  // 1.32`, and 1.12 + 0.04 k is exactly the double 1.28 at k = 4 and 1.32 at k = 5, so it never ran. It is counted too,
  // so that a guard which never fires fails the test instead of passing it.
  const int swingCheckIteration = 4;  // the solve at 1.28, mid-way through the right foot's swing [1.1, 1.5)
  int numSwingChecks = 0;
  for (int k = 0; k < 10; ++k) {
    const scalar_t solveTime = 1.12 + 0.04 * static_cast<scalar_t>(k);
    solveAt(*stanceAnchored_, schedule, target, stateWithBaseAt(*stanceAnchored_, vector2_t(speed * solveTime, 0.0)), solveTime);
    const std::optional<vector2_t> right = referenceManager->nominalFoothold(kContactRightIndex, /*time=*/1.5);
    const std::optional<vector2_t> left = referenceManager->nominalFoothold(kContactLeftIndex, /*time=*/2.0);
    if (!(right.has_value() && left.has_value())) {
      GTEST_FAIL() << "solve at " << solveTime;
    }
    EXPECT_NEAR(right->x(), expectedRight.x(), 1.0e-9) << "solve at " << solveTime;
    EXPECT_NEAR(right->y(), expectedRight.y(), 1.0e-9) << "solve at " << solveTime;
    EXPECT_NEAR(left->x(), expectedLeft.x(), 1.0e-9) << "solve at " << solveTime;
    EXPECT_NEAR(left->y(), expectedLeft.y(), 1.0e-9) << "solve at " << solveTime;

    if (k != swingCheckIteration) continue;
    ++numSwingChecks;
    // The swing reference is the cubic blend p(tau) = -tau^3 + tau^2 + tau from where the right foot LIFTED OFF - its
    // position at the 1.05 solve, the last one that saw it in contact - to the target at its TOUCH-DOWN, 1.5. At
    // tau = 0.5 (t = 1.3) it is 0.625 of the way, moving at p'(0.5) / T_swing = 1.25 / 0.4 of the displacement per
    // second; at tau = 0.25 (t = 1.2) it is 0.296875 of the way, at 1.3125 / 0.4. A linear blend (0.5, 0.25), a start
    // at the nominal point rather than the measured lift-off, or a target read at the query time instead of the
    // touch-down (0.05 m short at 1.3, 0.075 m at 1.2: less base travel, a longer upcoming stance) each move these by
    // centimeters.
    struct SwingSample {
      scalar_t time;
      scalar_t blend;
      scalar_t blendRate;
    };
    const vector2_t displacement = expectedRight - rightLiftOff;
    ASSERT_GT(displacement.norm(), 0.1) << "a swing that goes nowhere cannot tell one blend from another";
    for (const SwingSample& sample : {SwingSample{.time = 1.3, .blend = 0.625, .blendRate = 1.25 / 0.4},
                                      SwingSample{.time = 1.2, .blend = 0.296875, .blendRate = 1.3125 / 0.4}}) {
      const std::optional<SwingFootReference> swing = referenceManager->getSwingFootReference(kContactRightIndex, sample.time);
      if (!swing.has_value()) {
        GTEST_FAIL() << "t = " << sample.time;
      }
      EXPECT_LT((swing->position.head<2>() - (rightLiftOff + sample.blend * displacement)).cwiseAbs().maxCoeff(), 1.0e-9)
          << "t = " << sample.time << ": " << swing->position.head<2>().transpose();
      EXPECT_LT((swing->linearVelocity.head<2>() - sample.blendRate * displacement).cwiseAbs().maxCoeff(), 1.0e-9)
          << "t = " << sample.time << ": " << swing->linearVelocity.head<2>().transpose();
    }
  }
  EXPECT_EQ(numSwingChecks, 1) << "the swing reference was never checked";
}

TEST_F(LocomotionHeuristicIntegrationTest, HipCenteredAnchorIsThePredictedBaseAndAdvancesWithTheCommand) {
  // With hip_centered_stepping listed the anchor is the base PREDICTED at the touch-down: measured at the solve and
  // carried forward by the command. It used to be the base measured at the solve with no advance, so every swing in the
  // horizon got the same world target and a step landing 1.4 s out was referenced to where the hip had been.
  //
  // Base measured at (0.3, -0.2) at t = 1.12, command (0.4, 0.1): with every offset zeroed the target IS the predicted
  // base, (0.452, -0.162) for the touch-down at 1.5 and (0.852, -0.062) for the one at 2.5. With the offsets on, two
  // touch-downs of the same foot on equal stances still differ by exactly v * 1.0 s.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  ASSERT_TRUE(layer->footholdMovesAnchor());
  RestoreCoefficientsFromFileOnExit restore(layer, enabledConfig_->task);

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  const vector2_t command(0.4, 0.1);
  const scalar_t solveTime = 1.12;
  const vector_t state = stateWithBaseAt(*enabled_, vector2_t(0.3, -0.2));
  solveAt(*enabled_, walkSchedule(), constantTarget(*enabled_, commandedState(*enabled_, command, /*yaw=*/0.0)), state, solveTime);

  LocomotionHeuristicConfig config = fileConfig(enabledConfig_->task);
  config.hipCenteredStepping.lateralScale = 0.0;
  config.hipCenteredStepping.longitudinalScale = 0.0;
  config.translationalStepping = TranslationalSteppingParameters{};
  ASSERT_TRUE(layer->reconfigure(config).ok());
  struct Landing {
    size_t foot;
    scalar_t touchDown;
    vector2_t predictedBase;
  };
  const std::vector<Landing> landings = {{kContactRightIndex, 1.5, vector2_t(0.452, -0.162)},
                                         {kContactRightIndex, 2.5, vector2_t(0.852, -0.062)},
                                         {kContactLeftIndex, 2.0, vector2_t(0.652, -0.112)}};
  for (const Landing& landing : landings) {
    const std::optional<vector2_t> foothold = referenceManager->nominalFoothold(landing.foot, landing.touchDown);
    if (!foothold.has_value()) {
      GTEST_FAIL();
    }
    EXPECT_LT((*foothold - landing.predictedBase).cwiseAbs().maxCoeff(), 1.0e-12)
        << "foot " << landing.foot << " landing at " << landing.touchDown << ": " << foothold->transpose();
  }

  config.hipCenteredStepping.lateralScale = 1.0;
  config.hipCenteredStepping.longitudinalScale = 1.0;
  config.translationalStepping.forwardStanceFraction = 0.5;
  config.translationalStepping.lateralStanceFraction = 0.5;
  ASSERT_TRUE(layer->reconfigure(config).ok());
  const std::optional<vector2_t> firstRight = referenceManager->nominalFoothold(kContactRightIndex, /*time=*/1.5);
  const std::optional<vector2_t> secondRight = referenceManager->nominalFoothold(kContactRightIndex, /*time=*/2.5);
  const std::optional<vector2_t> firstLeft = referenceManager->nominalFoothold(kContactLeftIndex, /*time=*/2.0);
  if (!(firstRight.has_value() && secondRight.has_value() && firstLeft.has_value())) {
    GTEST_FAIL();
  }
  EXPECT_LT((*secondRight - *firstRight - command * 1.0).cwiseAbs().maxCoeff(), 1.0e-12);

  // Positive control that the hip offset is on and each foot is under its own hip: what is left after the predicted
  // base and the stepping lead (0.5 * 0.6 s * v) are taken off is to the foot's own side, and mirror-symmetric.
  const vector2_t lead = 0.5 * kWalkStanceDuration * command;
  const vector2_t rightHip = *firstRight - vector2_t(0.452, -0.162) - lead;
  const vector2_t leftHip = *firstLeft - vector2_t(0.652, -0.112) - lead;
  EXPECT_GT(leftHip.y(), 0.05) << "the left hip must be to the left of the base";
  EXPECT_LT(rightHip.y(), -0.05) << "the right hip must be to the right of the base";
  EXPECT_NEAR(leftHip.y(), -rightHip.y(), 1.0e-6) << "a symmetric robot has symmetric hips";
  EXPECT_NEAR(leftHip.x(), rightHip.x(), 1.0e-6);
}

TEST_F(LocomotionHeuristicIntegrationTest, HipCenteredSteppingKeepsTheFeetApartOnAZeroStepWidthAtAnyHeading) {
  // On a robot with no nominal step width the listed heuristic is the ONLY reason the reference manager measures
  // anything. Gating the measurement on the step width alone, as it once was, left every foothold heuristic returning
  // nothing, forever, on the SA01, G1 and R1. And the hip offsets must be rotated by the heading predicted for the
  // touch-down: at heading 0 the two hips are apart along world y by 2 h, at heading pi/2 they are apart along world
  // -x by the same 2 h (the robot's left is then the world's -x), which a transposed rotation would put along +x.
  //
  // Those two headings are measured with no yaw rate, so the heading predicted for the touch-down IS the measured one,
  // and a context built with the measured yaw - the stale yaw, which on the last step of a horizon trails a brisk turn
  // by most of a radian - or a prediction without its yaw-rate term both pass them. The third case turns: see below.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = zeroStepWidth_->getLocomotionHeuristicLayerPtr();
  ASSERT_EQ(zeroStepWidth_->modelSettings().nominalFootholdConfig.stepWidth, 0.0) << "the edited configuration did not take effect";
  ASSERT_TRUE(layer->footholdMovesAnchor());
  RestoreCoefficientsFromFileOnExit restore(layer, zeroStepWidthConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(zeroStepWidthConfig_->task);
  config.hipCenteredStepping.lateralScale = 1.0;
  config.hipCenteredStepping.longitudinalScale = 1.0;
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = zeroStepWidth_->getSwitchedModelReferenceManagerPtr();
  const CentroidalMpcRobotModel<scalar_t>& model = zeroStepWidth_->getMpcRobotModel();

  // Standing still at heading 0: the target is the base plus the hip, and the base is at the origin.
  const vector_t facingX = zeroStepWidth_->getInitialState();
  solveAt(*zeroStepWidth_, walkSchedule(), constantTarget(*zeroStepWidth_, facingX), facingX, /*time=*/1.12);
  const std::optional<vector2_t> rightFacingX = referenceManager->nominalFoothold(kContactRightIndex, /*time=*/1.5);
  const std::optional<vector2_t> leftFacingX = referenceManager->nominalFoothold(kContactLeftIndex, /*time=*/2.0);
  if (!(rightFacingX.has_value() && leftFacingX.has_value())) {
    GTEST_FAIL() << "a listed foothold heuristic must be measured for on a zero step width";
  }
  const scalar_t separation = leftFacingX->y() - rightFacingX->y();
  EXPECT_GT(separation, 0.1) << "the feet must land apart, each under its own hip";
  EXPECT_NEAR(leftFacingX->x(), rightFacingX->x(), 1.0e-6) << "a symmetric robot's hips are level fore and aft";
  EXPECT_NEAR(leftFacingX->y(), -rightFacingX->y(), 1.0e-6) << "about a base on the world origin, symmetrically";

  // The same robot turned to heading pi/2, measured there.
  vector_t facingY = facingX;
  vector6_t basePose = model.getBasePose(facingY);
  basePose(3) = M_PI_2;
  model.setBasePose(facingY, basePose);
  solveAt(*zeroStepWidth_, walkSchedule(), constantTarget(*zeroStepWidth_, facingY), facingY, /*time=*/1.12);
  const std::optional<vector2_t> rightFacingY = referenceManager->nominalFoothold(kContactRightIndex, /*time=*/1.5);
  const std::optional<vector2_t> leftFacingY = referenceManager->nominalFoothold(kContactLeftIndex, /*time=*/2.0);
  if (!(rightFacingY.has_value() && leftFacingY.has_value())) {
    GTEST_FAIL();
  }
  EXPECT_NEAR(leftFacingY->x() - rightFacingY->x(), -separation, 1.0e-9) << "the robot's left is the world's -x at heading pi/2";
  EXPECT_NEAR(leftFacingY->y(), rightFacingY->y(), 1.0e-6);

  // Turning on the spot at heading 0: psidot = 0.5 rad/s in the target's momentum channel as the target calculator
  // writes it, h_z = I_zz psidot / m, and no linear command, so the predicted base stays where it was measured. The
  // heading predicted for the right foot's touch-down at 1.5 is 0.5 * (1.5 - 1.12) = 0.19 rad and for the left foot's
  // at 2.0 it is 0.5 * 0.88 = 0.44 rad, so each hip offset measured at heading 0 comes back turned by that much. The
  // stale measured yaw, or a prediction that drops its yaw-rate term, leaves them unturned: 2 |r_hip| sin(0.095) and
  // 2 |r_hip| sin(0.22) off, centimeters against the 1e-9 tolerance.
  const scalar_t yawRate = 0.5;
  const scalar_t solveTime = 1.12;
  const vector2_t base = model.getBasePosition(facingX).head<2>();
  const vector_t turning = commandedState(*zeroStepWidth_, vector2_t::Zero(), /*yaw=*/0.0,
                                          compositeYawInertia(*zeroStepWidth_, facingX) * yawRate / totalMass(*zeroStepWidth_));
  solveAt(*zeroStepWidth_, walkSchedule(), constantTarget(*zeroStepWidth_, turning), facingX, solveTime);
  ASSERT_NEAR(referenceManager->getCommandedYawRate(1.5), yawRate, 1.0e-9) << "the yaw rate did not reach the reference manager";
  struct Landing {
    size_t foot;
    scalar_t touchDown;
    vector2_t hipAtHeadingZero;
  };
  const std::vector<Landing> landings = {{kContactRightIndex, 1.5, vector2_t(*rightFacingX - base)},
                                         {kContactLeftIndex, 2.0, vector2_t(*leftFacingX - base)}};
  for (const Landing& landing : landings) {
    const scalar_t predictedYaw = yawRate * (landing.touchDown - solveTime);
    const vector2_t expected = base + Eigen::Rotation2D<scalar_t>(predictedYaw).toRotationMatrix() * landing.hipAtHeadingZero;
    const std::optional<vector2_t> foothold = referenceManager->nominalFoothold(landing.foot, landing.touchDown);
    if (!foothold.has_value()) {
      GTEST_FAIL();
    }
    EXPECT_LT((*foothold - expected).cwiseAbs().maxCoeff(), 1.0e-9) << "foot " << landing.foot << " landing at " << landing.touchDown
                                                                    << ": " << foothold->transpose() << " vs " << expected.transpose();
    // Control: the turn moves the target by more than the tolerance, so the check above can see a missing turn.
    EXPECT_GT((expected - (base + landing.hipAtHeadingZero)).norm(), 0.01) << "foot " << landing.foot;
  }
}

TEST_F(LocomotionHeuristicIntegrationTest, StanceFootAnchorTurnsWithTheHeadingPredictedForTheTouchDown) {
  // The stance-foot anchor, with translational_stepping listed and every coefficient of it zeroed: the target is the
  // anchor itself, forward * (forward . base) + leftward * (leftward . stanceFoot + side * stepWidth), with forward and
  // leftward the axes of the heading PREDICTED for the touch-down. Measured in double support at heading 0, turning at
  // 0.5 rad/s on the spot: the prediction is 0.5 * (1.5 - 1.05) = 0.225 rad for the right foot and 0.475 rad for the
  // left. Axes built from the stale measured yaw of 0 give (base.x, stanceFoot.y +- stepWidth) instead - several
  // centimeters away, the lateral 0.45 m step width being swung through a quarter of a radian.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = stanceAnchored_->getLocomotionHeuristicLayerPtr();
  ASSERT_FALSE(layer->footholdMovesAnchor()) << "this interface must exercise the stance-foot anchor";
  RestoreCoefficientsFromFileOnExit restore(layer, stanceAnchoredConfig_->task);
  LocomotionHeuristicConfig config = fileConfig(stanceAnchoredConfig_->task);
  config.translationalStepping = TranslationalSteppingParameters{};
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = stanceAnchored_->getSwitchedModelReferenceManagerPtr();
  const scalar_t stepWidth = stanceAnchored_->modelSettings().nominalFootholdConfig.stepWidth;
  ASSERT_GT(stepWidth, 0.0);
  const scalar_t yawRate = 0.5;
  const scalar_t solveTime = 1.05;  // double support, so that both feet are measured where they stand
  const vector_t state = stateWithBaseAt(*stanceAnchored_, vector2_t(0.2, -0.1));
  const vector_t turning = commandedState(*stanceAnchored_, vector2_t::Zero(), /*yaw=*/0.0,
                                          compositeYawInertia(*stanceAnchored_, state) * yawRate / totalMass(*stanceAnchored_));
  solveAt(*stanceAnchored_, walkSchedule(), constantTarget(*stanceAnchored_, turning), state, solveTime);
  ASSERT_NEAR(referenceManager->getCommandedYawRate(1.5), yawRate, 1.0e-9) << "the yaw rate did not reach the reference manager";

  const vector2_t base = stanceAnchored_->getMpcRobotModel().getBasePosition(state).head<2>();
  struct Landing {
    size_t foot;
    size_t stanceFoot;
    scalar_t side;
    scalar_t touchDown;
  };
  const std::vector<Landing> landings = {{kContactRightIndex, kContactLeftIndex, -1.0, 1.5},
                                         {kContactLeftIndex, kContactRightIndex, 1.0, 2.0}};
  for (const Landing& landing : landings) {
    const scalar_t predictedYaw = yawRate * (landing.touchDown - solveTime);
    const vector2_t forward(std::cos(predictedYaw), std::sin(predictedYaw));
    const vector2_t leftward(-std::sin(predictedYaw), std::cos(predictedYaw));
    const vector2_t stanceFoot = footPosition(*stanceAnchored_, state, landing.stanceFoot);
    const vector2_t expected = forward * forward.dot(base) + leftward * (leftward.dot(stanceFoot) + landing.side * stepWidth);
    const std::optional<vector2_t> foothold = referenceManager->nominalFoothold(landing.foot, landing.touchDown);
    if (!foothold.has_value()) {
      GTEST_FAIL();
    }
    EXPECT_LT((*foothold - expected).cwiseAbs().maxCoeff(), 1.0e-9) << "foot " << landing.foot << " landing at " << landing.touchDown
                                                                    << ": " << foothold->transpose() << " vs " << expected.transpose();
    // Control: the anchor on the measured heading, which the stale-yaw mutant would return, is far from it.
    const vector2_t unturned(base.x(), stanceFoot.y() + landing.side * stepWidth);
    EXPECT_GT((expected - unturned).norm(), 0.05) << "foot " << landing.foot;
  }
}

TEST_F(LocomotionHeuristicIntegrationTest, ModelParametersComeFromTheRobotAndNotFromTheTaskFile) {
  // r_hip and the nominal CoM height are derived from the URDF, because a number that has to agree with the model must
  // not be maintained by hand beside it. Checked through the one heuristic that consumes r_hip.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  EXPECT_TRUE(layer->footholdMovesAnchor());

  FootholdHeuristicContext left;
  left.contactIndex = kContactLeftIndex;
  left.side = 1.0;
  FootholdHeuristicContext right;
  right.contactIndex = kContactRightIndex;
  right.side = -1.0;
  const vector2_t leftHip = layer->footholdOffset(left);
  const vector2_t rightHip = layer->footholdOffset(right);
  // Atlas's hips are about 0.18 m apart, so each is roughly 0.09 m to its own side; the exact number is the URDF's.
  EXPECT_GT(leftHip.y(), 0.05) << "the left hip must be to the left of the base";
  EXPECT_LT(rightHip.y(), -0.05) << "the right hip must be to the right of the base";
  EXPECT_NEAR(leftHip.y(), -rightHip.y(), 1.0e-6) << "a symmetric robot has symmetric hips";
}

/******************************************************************************************************/
/*                                 Rejected through the task file                                      */
/******************************************************************************************************/

TEST_F(LocomotionHeuristicIntegrationTest, UnknownNameInTheTaskFileIsRejectedWithAStatus) {
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(withLists({"orientation_compenstaion"}, /*foothold=*/{}, /*wrench=*/{}), files().urdfFile);
  ASSERT_FALSE(interface.ok());
  EXPECT_NE(std::string(interface.status().message()).find("orientation_compensation"), std::string::npos) << interface.status().message();
}

TEST_F(LocomotionHeuristicIntegrationTest, FootholdHeuristicUnderContactPlanningIsRejectedAtStartUp) {
  // A silent no-op is the failure mode this rejection exists to prevent, so it must be loud and at start-up.
  CentroidalMpcConfig config = withLists(/*basePose=*/{}, {"capture_point"}, /*wrench=*/{});
  config.task.contact_schedule_source = std::string(kContactPlannerContactScheduleSource);
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface = CentroidalMpcInterface::Create(config, files().urdfFile);
  ASSERT_FALSE(interface.ok());
  const std::string message(interface.status().message());
  EXPECT_NE(message.find(absl::StrCat(kContactScheduleSourceKey, " is ", kContactPlannerContactScheduleSource)), std::string::npos)
      << message;
  EXPECT_NE(message.find(absl::StrCat(kContactScheduleSourceKey, ": ", kGaitScheduleContactScheduleSource)), std::string::npos)
      << "the refusal must name the way out: " << message;
}

TEST_F(LocomotionHeuristicIntegrationTest, FootholdListWithoutAnAnchorIsRejectedOnAZeroStepWidth) {
  // With no step width and no hip_centered_stepping, nothing keeps the feet apart: every other foothold heuristic nudges
  // a target on the stance foot's own lateral line, and the swing foot would be aimed at the stance foot. The error
  // must name both ways out. (The positive controls are built in SetUpTestSuite: the same list on the shipped 0.45 m
  // step width is `stanceAnchored_`, and a zero step width with hip_centered_stepping listed is `zeroStepWidth_`.)
  CentroidalMpcConfig config = withLists(/*basePose=*/{}, {"translational_stepping"}, /*wrench=*/{});
  config.task.nominal_foothold.step_width = 0.0;
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface = CentroidalMpcInterface::Create(config, files().urdfFile);
  ASSERT_FALSE(interface.ok());
  const std::string message(interface.status().message());
  EXPECT_NE(message.find("nominal_foothold.step_width"), std::string::npos) << message;
  EXPECT_NE(message.find("hip_centered_stepping"), std::string::npos) << message;
}

}  // namespace ocs2::humanoid
