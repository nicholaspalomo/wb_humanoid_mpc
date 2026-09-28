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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_sqp/SqpMpc.h>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "support/AtlasReferenceStack.h"

/*
 * The DRC Atlas centroidal MPC - the real SQP solver on the shipped task file - through a reset after a fall, closed
 * on its own prediction and synchronous, as the controller's solver thread drives it.
 *
 * In the user's run the robot fell while trotting, the simulator put it back at its initial pose, and every solve after
 * that failed. The reset reset the solver's warm start only: the gait schedule went on trotting, so a swing was
 * scheduled at the initial pose, where every foot has a yaw of exactly zero and the swing-foot cost's derivative was
 * NaN (testFootYawResidual), and the clock that the simulator had rewound left every new gait waiting behind events of
 * the old one. On the tree before this test, where resetMpcNode() reset the solver alone, the same sequence failed 82 of
 * the 150 solves after the reset and the robot walked 0.32 m in the 6 s after the rewind. testMpcResetState shows,
 * without the solver, which of the stored state the full reset clears.
 */

namespace ocs2::humanoid {
namespace {

constexpr scalar_t kSolvePeriod = 0.02;  // [s] of solver time per solve

/** The Atlas MPC and its references, wired as CentroidalMpcRobotSim wires them, around the SQP solver. */
class SolverStack {
 public:
  /** `numThreads` > 0 replaces the task file's solver thread count; 1 makes every solve bit-for-bit repeatable. */
  explicit SolverStack(size_t numThreads = 0) {
    const std::string taskFile = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml");
    const std::string referenceFile = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml");
    const std::string urdfFile = atlasRunfilePath("robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf");
    const std::string gaitFile = atlasRunfilePath("humanoid_nmpc/humanoid_common_mpc/config/command/gait.yaml");
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> created = CentroidalMpcInterface::Create(taskFile, urdfFile, referenceFile);
    EXPECT_TRUE(created.ok()) << created.status();
    interface_ = *std::move(created);
    sqp::Settings sqpSettings = interface_->sqpSettings();
    if (numThreads > 0) sqpSettings.nThreads = numThreads;
    mpc_ = std::make_unique<SqpMpc>(interface_->mpcSettings(), sqpSettings, interface_->getOptimalControlProblem(),
                                    interface_->getInitializer());
    calculator_ = std::make_unique<CentroidalMpcTargetTrajectoriesCalculator>(
        referenceFile, interface_->getEffectiveMpcRobotModel(), interface_->getPinocchioInterface(), interface_->getCentroidalModelInfo(),
        interface_->mpcSettings().timeHorizon_);
    calculator_->setTerrainHeightSource(
        [referenceManager = interface_->getSwitchedModelReferenceManagerPtr()]() { return referenceManager->getAppliedTerrainHeight(); });
    CentroidalMpcTargetTrajectoriesCalculator* calculator = calculator_.get();
    motionManager_ = std::make_shared<ProceduralMpcMotionManager>(
        gaitFile, referenceFile, interface_->getSwitchedModelReferenceManagerPtr(), interface_->getEffectiveMpcRobotModel(),
        [calculator](const vector4_t& velocity, scalar_t initTime, scalar_t /*finalTime*/, const vector_t& initState) {
          return calculator->commandedVelocityToTargetTrajectories(velocity, initTime, initState);
        });
    motionManager_->setResetHook([calculator]() { calculator->reset(); });
    mpc_->getSolverPtr()->setReferenceManager(interface_->getReferenceManagerPtr());
    mpc_->getSolverPtr()->addSynchronizedModule(motionManager_);
    mrt_ = std::make_unique<MPC_MRT_Interface>(*mpc_);

    observation_.time = 0.0;
    observation_.state = interface_->getInitialState();
    observation_.input = vector_t::Zero(interface_->getEffectiveMpcRobotModel().getInputDim());
    observation_.mode = ModeNumber::STANCE;
    mrt_->setCurrentObservation(observation_);
    mrt_->resetMpcNode(resetTarget());
  }

  CentroidalMpcInterface& interface() { return *interface_; }
  SqpMpc& mpc() { return *mpc_; }
  MPC_MRT_Interface& mrt() { return *mrt_; }
  SystemObservation& observation() { return observation_; }

  /** The operator's command, raw: 1 is the command limit along x. */
  void command(scalar_t forward) {
    motionManager_->setAndScaleVelocityCommand(WalkingVelocityCommand(forward, /*v_y=*/0.0, /*desired_pelvis_h=*/0.0, /*v_yaw=*/0.0));
  }

  /** The target the controller resets the MPC to: the observation held still and upright, weight on both feet. */
  TargetTrajectories resetTarget() const {
    vector_t target = observation_.state;
    centroidal_model::getNormalizedMomentum(target, interface_->getCentroidalModelInfo()).setZero();
    target(10) = 0.0;
    target(11) = 0.0;
    PinocchioInterface pinocchioInterface = interface_->getPinocchioInterface();
    const vector_t input = weightCompensatingInput(pinocchioInterface, {true, true}, interface_->getEffectiveMpcRobotModel(), target);
    return TargetTrajectories({observation_.time, observation_.time + 2.0}, {target, target}, {input, input});
  }

  /**
   * Solves for `duration` of solver time; with `followPlan` the robot moves as the policy predicts (the plant is the
   * MPC's own model), otherwise it is held where it is. Returns the number of failed solves.
   */
  int run(scalar_t duration, bool followPlan) {
    int failures = 0;
    const scalar_t end = observation_.time + duration;
    while (observation_.time < end - 1e-9) {
      mrt_->setCurrentObservation(observation_);
      const absl::Status status = mrt_->advanceMpc();
      if (!status.ok()) {
        ++failures;
        if (failures <= 3) LOG(WARNING) << "failed solve at t = " << observation_.time << ": " << status.message();
      }
      mrt_->updatePolicy();
      if (followPlan && status.ok() && mrt_->initialPolicyReceived()) {
        vector_t state;
        vector_t input;
        size_t mode = 0;
        mrt_->evaluatePolicy(observation_.time + kSolvePeriod, observation_.state, state, input, mode);
        observation_.state = state;
        observation_.mode = mode;
      }
      observation_.time += kSolvePeriod;
    }
    return failures;
  }

  /** Whether the reference manager's mode schedule has a swing in [from, to]. */
  bool schedulesASwingIn(scalar_t from, scalar_t to) const {
    for (scalar_t time = from; time <= to + 1e-9; time += 0.005) {
      if (!interface_->getSwitchedModelReferenceManagerPtr()->isInStancePhase(time)) return true;
    }
    return false;
  }

  scalar_t baseX() const { return interface_->getMpcRobotModel().getBasePosition(observation_.state)(0); }

 private:
  std::unique_ptr<CentroidalMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> calculator_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManager_;
  std::unique_ptr<MPC_MRT_Interface> mrt_;
  SystemObservation observation_;
};

/** Stands 1 s, then trots forward for 11 s. */
void trot(SolverStack& stack) {
  stack.command(0.0);
  ASSERT_EQ(stack.run(/*duration=*/1.0, /*followPlan=*/true), 0);
  stack.command(1.0);
  ASSERT_EQ(stack.run(/*duration=*/11.0, /*followPlan=*/true), 0) << "the MPC fails while walking, before anything is reset";
  const SystemObservation& observation = stack.observation();
  ASSERT_TRUE(stack.schedulesASwingIn(observation.time, observation.time + stack.interface().mpcSettings().timeHorizon_))
      << "positive control: the robot trots before the fall";
  ASSERT_GT(stack.baseX(), 3.0);
}

/** The simulator puts the robot back at its initial pose - every foot at a yaw of exactly zero - with its clock 0.3 s back. */
void putBackAtTheInitialPose(SolverStack& stack) {
  SystemObservation& observation = stack.observation();
  observation.state = stack.interface().getInitialState();
  observation.mode = ModeNumber::STANCE;
  observation.time -= 0.3;
  stack.command(0.0);  // the remote control re-centers its sticks
}

TEST(MpcResetSolverStack, AfterTheFullResetEverySolveSucceedsAndTheFirstPolicyStandsWhereTheRobotIs) {
  SolverStack stack;
  trot(stack);
  putBackAtTheInitialPose(stack);
  const SystemObservation held = stack.observation();
  stack.mrt().setCurrentObservation(held);
  stack.mrt().resetMpcNode(stack.resetTarget());

  // The first solve after the reset: planned from the robot held at the initial pose, standing.
  ASSERT_EQ(stack.run(kSolvePeriod, /*followPlan=*/false), 0);
  const scalar_t horizon = stack.interface().mpcSettings().timeHorizon_;
  EXPECT_FALSE(stack.schedulesASwingIn(held.time, held.time + horizon)) << "the gait schedule still trots after the reset";
  ASSERT_TRUE(stack.mrt().isActivePolicyCurrent());
  const SystemObservation& solvedFrom = stack.mrt().getCommand().mpcInitObservation_;
  EXPECT_DOUBLE_EQ(solvedFrom.time, held.time);
  EXPECT_LT((solvedFrom.state - held.state).cwiseAbs().maxCoeff(), 1e-12);
  for (scalar_t time = held.time; time <= held.time + horizon; time += 0.01) {
    EXPECT_EQ(stack.mrt().getPolicy().modeSchedule_.modeAtTime(time), ModeNumber::STANCE) << "t = " << time;
  }

  // And every solve after it succeeds, the robot held there.
  EXPECT_EQ(stack.run(150 * kSolvePeriod, /*followPlan=*/false), 0) << "solves failed after the reset";
}

/** The largest difference between two policies' trajectories; infinity when their shapes or schedules differ. */
scalar_t policyDifference(const PrimalSolution& a, const PrimalSolution& b) {
  if (a.timeTrajectory_.size() != b.timeTrajectory_.size() || a.modeSchedule_.modeSequence != b.modeSchedule_.modeSequence ||
      a.modeSchedule_.eventTimes.size() != b.modeSchedule_.eventTimes.size()) {
    return std::numeric_limits<scalar_t>::infinity();
  }
  scalar_t difference = 0.0;
  for (size_t k = 0; k < a.timeTrajectory_.size(); ++k) {
    difference = std::max(difference, std::abs(a.timeTrajectory_[k] - b.timeTrajectory_[k]));
    difference = std::max(difference, (a.stateTrajectory_[k] - b.stateTrajectory_[k]).cwiseAbs().maxCoeff());
    difference = std::max(difference, (a.inputTrajectory_[k] - b.inputTrajectory_[k]).cwiseAbs().maxCoeff());
  }
  for (size_t k = 0; k < a.modeSchedule_.eventTimes.size(); ++k) {
    difference = std::max(difference, std::abs(a.modeSchedule_.eventTimes[k] - b.modeSchedule_.eventTimes[k]));
  }
  return difference;
}

TEST(MpcResetSolverStack, AFullResetSolvesAndWalksExactlyAsAFreshStack) {
  // The same observation at the same time, handed to a stack that has never solved and to one that trotted 11 s and
  // was then reset: whatever the reset leaves behind anywhere in the stack - the references, the modules, the solver,
  // the problem it solves - shows as a difference in the first policy, or in the walk that follows it. One solver
  // thread, so that two stacks given the same problem compute the same numbers in the same order; a second fresh stack
  // is the control that shows they do. A third stack that trotted and had its solver reset alone - what a reset was
  // before - is the positive control: the comparison has to see what that reset leaves behind.
  constexpr scalar_t kResetTime = 2.0;
  SolverStack fresh(1);
  SolverStack control(1);
  SolverStack used(1);
  SolverStack solverOnly(1);
  trot(used);
  trot(solverOnly);

  const std::array<SolverStack*, 4> resetStacks{&fresh, &control, &used, &solverOnly};
  for (SolverStack* stack : resetStacks) {
    SystemObservation& observation = stack->observation();
    observation.state = stack->interface().getInitialState();
    observation.mode = ModeNumber::STANCE;
    observation.time = kResetTime;
    stack->command(0.0);
    stack->mrt().setCurrentObservation(observation);
    if (stack == &solverOnly) {
      stack->mrt().resetMpcSolver(stack->resetTarget());
    } else {
      stack->mrt().resetMpcNode(stack->resetTarget());
    }
    ASSERT_EQ(stack->run(kSolvePeriod, /*followPlan=*/false), 0);
  }
  EXPECT_EQ(policyDifference(fresh.mrt().getPolicy(), control.mrt().getPolicy()), 0.0) << "control: two fresh stacks";
  EXPECT_EQ(policyDifference(fresh.mrt().getPolicy(), used.mrt().getPolicy()), 0.0) << "the first policy after the reset";
  EXPECT_GT(policyDifference(fresh.mrt().getPolicy(), solverOnly.mrt().getPolicy()), 1e-3)
      << "positive control: a reset of the solver alone leaves the trot behind, and the comparison has to see it";

  const std::array<SolverStack*, 3> stacks{&fresh, &control, &used};

  // Stand 1 s, then walk 3 s from a standstill, each stack closed on its own prediction, one solve at a time. After
  // every solve the used stack must agree with the fresh one - in the mode schedule and the target its references hand
  // the solver, in the policy the solver returns and in the state that policy leads to - to the precision to which the
  // control agrees with it. The first quantity that does not names where the reset left something behind.
  constexpr scalar_t kTolerance = 1e-9;
  scalar_t controlDrift = 0.0;
  const int numSolves = static_cast<int>(std::round(4.0 / kSolvePeriod));
  for (int solve = 0; solve < numSolves; ++solve) {
    if (solve == static_cast<int>(std::round(1.0 / kSolvePeriod))) {
      for (SolverStack* stack : stacks) stack->command(1.0);
    }
    for (SolverStack* stack : stacks) ASSERT_EQ(stack->run(kSolvePeriod, /*followPlan=*/true), 0);
    controlDrift = std::max(controlDrift, (fresh.observation().state - control.observation().state).cwiseAbs().maxCoeff());

    const SwitchedModelReferenceManager& freshReferences = *fresh.interface().getSwitchedModelReferenceManagerPtr();
    const SwitchedModelReferenceManager& usedReferences = *used.interface().getSwitchedModelReferenceManagerPtr();
    const ModeSchedule& freshSchedule = freshReferences.getModeSchedule();
    const ModeSchedule& usedSchedule = usedReferences.getModeSchedule();
    scalar_t scheduleDifference =
        freshSchedule.modeSequence == usedSchedule.modeSequence && freshSchedule.eventTimes.size() == usedSchedule.eventTimes.size()
            ? 0.0
            : std::numeric_limits<scalar_t>::infinity();
    for (size_t k = 0; std::isfinite(scheduleDifference) && k < freshSchedule.eventTimes.size(); ++k) {
      scheduleDifference = std::max(scheduleDifference, std::abs(freshSchedule.eventTimes[k] - usedSchedule.eventTimes[k]));
    }
    const TargetTrajectories& freshTarget = freshReferences.getTargetTrajectories();
    const TargetTrajectories& usedTarget = usedReferences.getTargetTrajectories();
    scalar_t targetDifference = freshTarget.size() == usedTarget.size() ? 0.0 : std::numeric_limits<scalar_t>::infinity();
    for (size_t k = 0; std::isfinite(targetDifference) && k < freshTarget.size(); ++k) {
      targetDifference = std::max({targetDifference, std::abs(freshTarget.timeTrajectory[k] - usedTarget.timeTrajectory[k]),
                                   (freshTarget.stateTrajectory[k] - usedTarget.stateTrajectory[k]).cwiseAbs().maxCoeff()});
    }
    const scalar_t policy = policyDifference(fresh.mrt().getPolicy(), used.mrt().getPolicy());
    const scalar_t state = (fresh.observation().state - used.observation().state).cwiseAbs().maxCoeff();
    if (scheduleDifference > kTolerance || targetDifference > kTolerance || policy > kTolerance || state > kTolerance) {
      ADD_FAILURE() << "after the reset the stack stopped solving as a fresh one at t = " << fresh.observation().time << " s ("
                    << solve * kSolvePeriod << " s after the reset): mode schedule " << scheduleDifference << ", target "
                    << targetDifference << ", policy " << policy << ", state " << state << "\nfresh schedule: " << freshSchedule
                    << "\nused schedule: " << usedSchedule;
      break;
    }
  }
  EXPECT_LT(controlDrift, kTolerance) << "control: two fresh stacks do not solve alike, so the comparison means nothing";
  EXPECT_GT(fresh.baseX(), 0.5) << "positive control: the fresh stack walks";
}

TEST(MpcResetSolverStack, AfterAResetAndARewoundClockTheRobotWalksWhenCommanded) {
  SolverStack stack;
  trot(stack);
  // The simulator's automatic reset rewound the clock to 2 s and put the robot back at the origin.
  SystemObservation& observation = stack.observation();
  observation.state = stack.interface().getInitialState();
  observation.mode = ModeNumber::STANCE;
  observation.time = 2.0;
  stack.command(0.0);
  stack.mrt().setCurrentObservation(observation);
  stack.mrt().resetMpcNode(stack.resetTarget());
  ASSERT_EQ(stack.run(/*duration=*/1.0, /*followPlan=*/true), 0);

  const scalar_t start = stack.baseX();
  stack.command(1.0);
  EXPECT_EQ(stack.run(/*duration=*/6.0, /*followPlan=*/true), 0);
  EXPECT_GT(stack.baseX() - start, 2.0) << "commanded to walk after the rewind, the robot stayed where it was";
}

}  // namespace
}  // namespace ocs2::humanoid
