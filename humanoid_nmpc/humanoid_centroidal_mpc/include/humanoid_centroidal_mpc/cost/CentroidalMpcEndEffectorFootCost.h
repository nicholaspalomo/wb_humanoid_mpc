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

#pragma once

#include <cmath>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/cost/StateInputGaussNewtonCostAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/PinocchioStateInputMapping.h"
#include "pinocchio/algorithm/frames.hpp"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The swing-foot tracking cost of the centroidal MPC (task_space_foot_cost):
 * the pose and twist of one foot against the swing trajectory and the landing
 * target the reference manager plans for it, weighted by task_space_foot_cost
 * and, near touch-down, by the swing planner's impact proximity. Every
 * reference is read from the reference manager, not from the solver's target
 * (getParameters()):
 *
 *  - the height and the vertical velocity are the swing trajectory planner's
 * (getZpositionConstraint(), getZvelocityConstraint()), the references of the
 * normal-velocity constraint too: the planned swing height in swing, the ground
 * the planner was built on (the reference manager's terrain height) in stance;
 *  - the horizontal position and velocity are the swing foot reference's when
 * it has a landing target (getSwingFootReference()). Without one the xy
 * position weights are 0, and the xy velocity tracks the commanded base
 *    velocity (getSwingFootVelocityReference()) with
 * swing_velocity_reference_weight in place of lin_velocity_x / _y;
 *  - the orientation residual is in radians, so its weights are per rad^2 like
 * those of the end-effector costs (computeTaskSpaceErrors()): the tilt of the
 * foot from the reference plane as a rotation vector in world axes
 *    (tiltAngleToPlane()), whose third entry the planned foot yaw replaces when
 * there is one (footYawError(), the wrapped yaw error). Only the contact
 * planner's heading model plans a foot yaw. Without one the third entry is the
 *    tilt's, 0 on flat ground, and weights.orientation_z weighs next to
 * nothing: CentroidalMpcInterface warns at start-up.
 *
 * Not thread-safe; the solver clones it per worker.
 */
class CentroidalMpcEndEffectorFootCost final
    : public StateInputCostGaussNewtonAd {
 public:
  /**
   * The cost name the interface builds the cost of the contact `footName` with,
   * which names its CppAD library
   * ("<it>" kLibraryVersionSuffix). The same string as its collection name
   * (taskSpaceKinematicsCostName(footName), CostTermNames.h) today, defined
   * apart on purpose: a rename of the term must not rename, and regenerate, the
   * library (testCostTermAndLibraryNames).
   */
  static std::string libraryName(absl::string_view footName) {
    return absl::StrCat(footName, "_TaskSpaceKinematicsCost");
  }

  /**
   * Appended to the cost name to name the CppAD library: the version of the
   * residual, changed with every change of it, so that a library taped from an
   * earlier residual is never loaded in its place. "_yawRefHalfAngle" was the
   * wrapped yaw residual with the half-angle tilt -sin(theta / 2) u;
   * "_tiltAngle" is the tilt angle theta u (tiltAngleToPlane()).
   */
  static constexpr char kLibraryVersionSuffix[] = "_tiltAngle";

  /**
   * The cost of the contact `contactIndex` with the task file's `settings`
   * (taskSpaceFootCostFromConfig()): its weights, active phases and swing
   * velocity reference weight. `referenceManager` is retained and must outlive
   * the cost and every clone of it; the model and the interface are copied.
   * Builds or loads the CppAD library `costName` kLibraryVersionSuffix.
   */
  CentroidalMpcEndEffectorFootCost(
      const SwitchedModelReferenceManager& referenceManager,
      const TaskSpaceFootCostSettings& settings,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
      size_t contactIndex,
      const std::string& costName,
      const ModelSettings& modelSettings);

  ~CentroidalMpcEndEffectorFootCost() override = default;
  CentroidalMpcEndEffectorFootCost* absl_nonnull clone() const override {
    return new CentroidalMpcEndEffectorFootCost(*this);
  }
  // Copied only by clone(), whose copy constructor is private; never assigned
  // or moved.
  CentroidalMpcEndEffectorFootCost& operator=(
      const CentroidalMpcEndEffectorFootCost&) = delete;
  CentroidalMpcEndEffectorFootCost(CentroidalMpcEndEffectorFootCost&&) = delete;
  CentroidalMpcEndEffectorFootCost& operator=(
      CentroidalMpcEndEffectorFootCost&&) = delete;

  vector_t getParameters(scalar_t time,
                         const TargetTrajectories& targetTrajectories,
                         const PreComputation& preComputation) const override;

  bool isActive(scalar_t time) const override {
    if (!isActive_) return false;
    return activeInStance_ ||
           !referenceManagerPtr_->isInContact(time, contactIndex_);
  }

  void setActive(bool active) { isActive_ = active; }
  bool getActive() const { return isActive_; }

  void setActiveInStance(bool activeInStance) {
    activeInStance_ = activeInStance;
  }
  bool getActiveInStance() const { return activeInStance_; }

  void setWeights(const vector12_t& weights) {
    sqrtWeights_ = weights.cwiseSqrt();
  }
  void getWeights(vector12_t& weights) const {
    weights = sqrtWeights_.cwiseProduct(sqrtWeights_);
  }

  /**
   * Sets the weight of the x and y linear velocity errors of a swing without a
   * landing target, against the commanded base velocity
   * (task_space_foot_cost.swing_velocity_reference_weight; non-negative, as the
   * conversion checks).
   */
  void setSwingVelocityReferenceWeight(scalar_t weight) {
    swingVelocityReferenceSqrtWeight_ = std::sqrt(weight);
  }
  scalar_t getSwingVelocityReferenceWeight() const {
    return swingVelocityReferenceSqrtWeight_ *
           swingVelocityReferenceSqrtWeight_;
  }

  /**
   * The yaw error of a foot whose orientation is `orientation` against the
   * planned foot yaw `yawReference`, wrapped to
   * (-pi, pi): the third orientation residual when `hasYawReference` is 1. When
   * it is 0 the reference is replaced by the foot's own heading, and the error
   * is exactly zero.
   *
   * With h the unit heading of the foot, (R00, R10) / |(R00, R10)|, and r the
   * unit reference heading, the error is 2 atan(s / (1 + c)) with s = r x h =
   * rx hy - ry hx and c = h . r: for h = (cos yaw, sin yaw) and r = (cos ref,
   * sin ref) they are sin(yaw - ref) and cos(yaw - ref), so the error is yaw -
   * ref wrapped to (-pi, pi), positive when the foot has turned past its
   * reference. It is the half-angle form of atan2(s, c), the same number for
   * every error short of pi. It replaces atan2(sin(yaw - ref), cos(yaw - ref))
   * with yaw = atan2(R10, R00). CppAD evaluates both branches of an atan2, one
   * of which divides by its first argument, so at a foot yaw of exactly zero -
   * the simulator's reset pose, MuJoCo's own reset pose, any pose built from a
   * zero base yaw - the derivative of the error was 0 * inf = NaN. Multiplying
   * the term by a zero hasYawReference did not help, 0 * NaN being NaN, and
   * every QP of a swing of that foot failed.
   *
   * A template, so that the residual the CppAD model is taped from can be
   * evaluated in double precision by the tests.
   */
  template <typename SCALAR>
  static SCALAR footYawError(
      const Eigen::Matrix<SCALAR, kPositionDim, kPositionDim>& orientation,
      const SCALAR& yawReference,
      const SCALAR& hasYawReference) {
    using std::atan;
    using std::cos;
    using std::sin;
    using std::sqrt;
    const SCALAR one(1.0);
    const SCALAR headingNorm = sqrt(orientation(0, 0) * orientation(0, 0) +
                                    orientation(1, 0) * orientation(1, 0));
    const SCALAR headingX = orientation(0, 0) / headingNorm;
    const SCALAR headingY = orientation(1, 0) / headingNorm;
    const SCALAR referenceX = hasYawReference * cos(yawReference) +
                              (one - hasYawReference) * headingX;
    const SCALAR referenceY = hasYawReference * sin(yawReference) +
                              (one - hasYawReference) * headingY;
    const SCALAR sine = headingY * referenceX - headingX * referenceY;
    const SCALAR cosine = headingX * referenceX + headingY * referenceY;
    return SCALAR(kDoubleAngleFactor) * atan(sine / (one + cosine));
  }

 private:
  /**
   * The factor of the half-angle form atan2(s, c) = 2 atan(s / (1 + c))
   * footYawError() evaluates: atan(s / (1 + c)) is the half angle, which it
   * takes back to the angle.
   */
  static constexpr double kDoubleAngleFactor = 2.0;

  CentroidalMpcEndEffectorFootCost(
      const CentroidalMpcEndEffectorFootCost& other);

  ad_vector_t costVectorFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) override;

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;

  vector12_t sqrtWeights_;
  // The square root of the swing velocity reference weight
  // (setSwingVelocityReferenceWeight()).
  scalar_t swingVelocityReferenceSqrtWeight_;
  bool isActive_ = true;
  bool activeInStance_ = false;

  size_t contactIndex_;
  const pinocchio::FrameIndex frameID_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelAdPtr_;
};

}  // namespace ocs2::humanoid
