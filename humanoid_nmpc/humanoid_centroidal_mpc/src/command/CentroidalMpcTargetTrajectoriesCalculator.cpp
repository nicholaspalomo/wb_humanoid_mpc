/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"

namespace ocs2::humanoid {

namespace {
// [s] A time jump of more than this since the last command (a reset, a restart)
// restarts the joint-state filter from the current joints.
constexpr scalar_t kMaxJointFilterTimeGap = 0.1;
}  // namespace

absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
CentroidalMpcTargetTrajectoriesCalculator::Create(
    const std::string& referenceFile,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    PinocchioInterface pinocchioInterface,
    scalar_t mpcHorizon) {
  ASSIGN_OR_RETURN(const mpc_config::ReferenceFile file,
                   loadReferenceFile(referenceFile));
  absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
      calculator = Create(file, mpcRobotModel, std::move(pinocchioInterface),
                          mpcHorizon);
  if (!calculator.ok()) {
    return withConfigFile(calculator.status(), referenceFile);
  }
  return calculator;
}

absl::StatusOr<std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator>>
CentroidalMpcTargetTrajectoriesCalculator::Create(
    const mpc_config::ReferenceFile& referenceFile,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    PinocchioInterface pinocchioInterface,
    scalar_t mpcHorizon) {
  ASSIGN_OR_RETURN(const ReferenceSettings referenceSettings,
                   referenceSettingsFromConfig(referenceFile));
  const ModelSettings& modelSettings = mpcRobotModel.modelSettings;
  ASSIGN_OR_RETURN(const vector_t defaultJointState,
                   defaultJointStateFromConfig(referenceFile,
                                               modelSettings.mpcModelJointNames,
                                               modelSettings.fixedJointNames));
  // The joint-state filter's time constant, which the centroidal MPC requires
  // (the whole-body MPC does not read it); a time constant that is not positive
  // would make the filter grow rather than decay.
  const std::optional<scalar_t> timeConstant =
      referenceSettings.targetJointStateInterpolationTimeConstant;
  if (!timeConstant.has_value()) {
    return absl::InvalidArgumentError(
        "[CentroidalMpcTargetTrajectoriesCalculator] "
        "target_joint_state_interpolation_time_constant is required by the "
        "centroidal MPC, "
        "but the reference file does not give it.");
  }
  if (!(*timeConstant > 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[CentroidalMpcTargetTrajectoriesCalculator] "
                     "target_joint_state_interpolation_time_constant is ",
                     *timeConstant, ", but it must be positive."));
  }
  return absl::WrapUnique(new CentroidalMpcTargetTrajectoriesCalculator(
      referenceSettings, defaultJointState, *timeConstant, mpcRobotModel,
      std::move(pinocchioInterface), mpcHorizon));
}

CentroidalMpcTargetTrajectoriesCalculator::
    CentroidalMpcTargetTrajectoriesCalculator(
        const ReferenceSettings& referenceSettings,
        const vector_t& defaultJointState,
        scalar_t targetJointStateInterpolationTimeConstant,
        const MpcRobotModelBase<scalar_t>& mpcRobotModel,
        PinocchioInterface pinocchioInterface,
        scalar_t mpcHorizon)
    : TargetTrajectoriesCalculatorBase(
          referenceSettings, defaultJointState, mpcRobotModel, mpcHorizon),
      pinocchioInterface_(std::move(pinocchioInterface)),
      mass_(pinocchio::computeTotalMass(pinocchioInterface_.getModel())),
      targetJointStateInterpolationTimeConstant_(
          targetJointStateInterpolationTimeConstant) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void CentroidalMpcTargetTrajectoriesCalculator::reset() {
  TargetTrajectoriesCalculatorBase::reset();
  filteredJointState_.resize(0);
  lastTime_ = 0.0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

TargetTrajectories CentroidalMpcTargetTrajectoriesCalculator::
    commandedPositionToTargetTrajectories(
        const vector4_t& commandLinePoseTarget,
        scalar_t initTime,
        const vector_t& initState) {
  const vector6_t currentPoseTarget = getCurrentBasePoseTarget(initState);

  const vector6_t targetPose =
      getDeltaBaseTarget(commandLinePoseTarget, currentPoseTarget);

  scalar_t targetReachingTime =
      initTime + estimateTimeToTarget(targetPose - currentPoseTarget);

  // desired time and state trajectories: at rest (zero momentum), the base
  // poses on knots at most a quarter turn apart in yaw, so that the slerp of
  // the orientation between them turns the commanded way (subdivideYawKnots()),
  // and the target joints
  const std::vector<PoseCommandKnot> knots = subdivideYawKnots(
      initTime, currentPoseTarget, targetReachingTime, targetPose);
  scalar_array_t timeTrajectory;
  timeTrajectory.reserve(knots.size());
  vector_array_t stateTrajectory;
  stateTrajectory.reserve(knots.size());
  for (const PoseCommandKnot& knot : knots) {
    timeTrajectory.push_back(knot.time);
    vector_t state = vector_t::Zero(mpcRobotModelPtr_->getStateDim());
    mpcRobotModelPtr_->setBasePoseEulerZyx(state, knot.basePose);
    mpcRobotModelPtr_->setJointAngles(state, targetJointState_);
    stateTrajectory.push_back(std::move(state));
  }

  // desired input trajectory (just right dimensions, they are not used)
  const vector_array_t inputTrajectory(
      knots.size(), vector_t::Zero(mpcRobotModelPtr_->getInputDim()));

  TargetTrajectories targetTrajectories{timeTrajectory, stateTrajectory,
                                        inputTrajectory};

  return targetTrajectories;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

TargetTrajectories CentroidalMpcTargetTrajectoriesCalculator::
    commandedVelocityToTargetTrajectories(
        const vector4_t& commandedVelocities,
        scalar_t initTime,
        const vector_t& initState,
        const BasePlanarMotion& measuredMotion) {
  // The base knots of the command on the arc of its twist, the first segment
  // starting from the measured motion (TargetTrajectoriesCalculatorBase's class
  // comment; design defect D1b). The momentum target of every knot is the
  // command's.
  const std::vector<VelocityCommandKnot> knots = velocityCommandKnots(
      commandedVelocities, initTime, initState, measuredMotion);

  // The state carries the NORMALIZED centroidal momentum h = [p / m, L / m].
  // Its linear part is the CoM velocity, which the target sets to the commanded
  // velocity of the base (the pelvis). That is exact for a command that does
  // not turn: a body turning rigidly at the yaw rate omega moves its CoM at
  // v_base + omega e_z x r, r the CoM's horizontal offset from the base, so on
  // a turn the target leaves out omega |r|, 0.01 to 0.04 m/s at 1 rad/s (the
  // shipped robots' neutral postures put the CoM 0.009 to 0.037 m from the base
  // origin). Its angular part is the momentum of the whole body spinning about
  // the vertical through its CoM at the commanded yaw rate: L = I_G * (0, 0,
  // yaw_rate), i.e. the yaw column of the locked inertia I_G, which is the
  // composite rigid body inertia that ccrba() leaves in data.Ig. That column is
  // dominated by I_zz but also carries the products of inertia I_xz and I_yz,
  // without which the target would not correspond to any rigid rotation of the
  // body. The previous mapping, yaw_rate / m, dropped the inertia altogether,
  // which for the DRC Atlas made the yaw momentum target 8x too small (and
  // dimensionally a rate per kilogram); the MPC then barely turned however hard
  // it was asked.
  const matrix3_t lockedInertia = [&]() {
    const PinocchioInterface::Model& model = pinocchioInterface_.getModel();
    PinocchioInterface::Data& data = pinocchioInterface_.getData();
    pinocchio::ccrba(model, data,
                     mpcRobotModelPtr_->getGeneralizedCoordinates(initState),
                     vector_t::Zero(model.nv));
    return matrix3_t(data.Ig.inertia().matrix());
  }();
  // At a knot whose heading has turned by dpsi, the body is the measured one
  // turned about the world z axis by dpsi: its inertia R_z I_G R_z^T, whose yaw
  // column is R_z I_G e_z, so the knot's angular momentum is the first one's
  // turned by dpsi, the yaw component unchanged
  // (MpcRobotModelBase::getTargetPlanarCommand() reads the same yaw rate back),
  // and its linear momentum is the command in the knot's heading frame (design
  // README section 2.9). Between knots dpsi_k apart the solver interpolates
  // both linearly, which shortens the horizontal angular momentum by up to
  // |L_xy| (1 - cos(dpsi_k / 2)); the knots' subdivision bounds the CoM
  // velocity's share of that (TargetTrajectoriesCalculatorBase's class
  // comment).
  const vector3_t angularMomentum =
      lockedInertia.col(kZAxisIndex) * (knots.front().commandedYawRate / mass_);

  // Extract current joint state from the provided initial state observation
  vector_t currentJointState = initState.tail(targetJointState_.size());

  // Initialize filter if empty, if time jumps (e.g., reset/restart), or if it
  // holds anything that is not a number. The filter then starts from the
  // current joints at this time: decaying it over the time since lastTime_
  // would put it at (or near) the nominal joints at once, after a reset the
  // more so the later the reset comes.
  if (filteredJointState_.size() != targetJointState_.size() ||
      initTime < lastTime_ || (initTime - lastTime_) > kMaxJointFilterTimeGap ||
      !filteredJointState_.allFinite()) {
    filteredJointState_ = currentJointState;
    lastTime_ = initTime;
  }

  // Exponentially decay the filtered state toward the nominal targetJointState_
  scalar_t dt = initTime - lastTime_;
  if (dt > 0.0) {
    scalar_t timeConstant = targetJointStateInterpolationTimeConstant_;
    scalar_t alpha = std::exp(-dt / timeConstant);
    filteredJointState_ =
        alpha * filteredJointState_ + (1.0 - alpha) * targetJointState_;
  }
  lastTime_ = initTime;

  // desired state trajectory: the target momentum, the base poses and the
  // joints: filtered up to the intermediate knot, then interpolated linearly to
  // the target joints at the end. The normalized centroidal momentum h leads
  // the state of every centroidal layout
  // (CentroidalMpcRobotModel::getCentroidalMomentum()).
  scalar_array_t timeTrajectory;
  vector_array_t stateTrajectory;
  for (const VelocityCommandKnot& knot : knots) {
    vector_t state = vector_t::Zero(mpcRobotModelPtr_->getStateDim());
    const scalar_t cosine = std::cos(knot.headingChange);
    const scalar_t sine = std::sin(knot.headingChange);
    state.head<centroidal_model::kCentroidalMomentumDim>()
        << knot.commandedVelocityWorld(0),
        knot.commandedVelocityWorld(1), 0.0,
        cosine * angularMomentum(0) - sine * angularMomentum(1),
        sine * angularMomentum(0) + cosine * angularMomentum(1),
        angularMomentum(kZAxisIndex);
    mpcRobotModelPtr_->setBasePoseEulerZyx(state, knot.basePose);
    if (knot.secondSegmentFraction == 0.0) {
      mpcRobotModelPtr_->setJointAngles(state, filteredJointState_);
    } else if (knot.secondSegmentFraction == 1.0) {
      mpcRobotModelPtr_->setJointAngles(state, targetJointState_);
    } else {
      mpcRobotModelPtr_->setJointAngles(
          state,
          vector_t((1.0 - knot.secondSegmentFraction) * filteredJointState_ +
                   knot.secondSegmentFraction * targetJointState_));
    }
    timeTrajectory.push_back(knot.time);
    stateTrajectory.push_back(std::move(state));
  }

  // desired input trajectory (just right dimensions, they are not used)
  const vector_array_t inputTrajectory(
      stateTrajectory.size(), vector_t::Zero(mpcRobotModelPtr_->getInputDim()));

  TargetTrajectories targetTrajectories{timeTrajectory, stateTrajectory,
                                        inputTrajectory};

  return targetTrajectories;
}

}  // namespace ocs2::humanoid
