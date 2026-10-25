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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

namespace ocs2::humanoid {

/**
 * Tracks the whole-body Center of Mass position and Angular Center of Mass
 * orientation against the reference trajectory, in place of the base pose.
 *
 * The ACoM attitude is xi_a(x) = xi_hat (x) Exp(Delta_theta(q_j))
 * (AngularCenterOfMass::acomAttitude()), the base orientation composed with the
 * learned joint-induced rotation, and its error is the heading-tilt residual
 * e_HT(xi_a(x), xi_a(x_ref)) of orientation/BaseOrientation.h, rows (yaw,
 * pitch, roll) - the rows acom_weights names (design section 2.8.3 of
 * humanoid_nmpc/docs/quaternion_base_orientation/README.md):
 *
 *   L = 1/2 e_com^T Q_com e_com + 1/2 e_acom^T Q_acom e_acom,  e_com = c(q) -
 * c(q_ref),  e_acom = e_HT(xi_a(x), xi_a(x_ref)).
 *
 * Its Gauss-Newton derivatives take the ACoM rows to the state through the body
 * tangent of xi_a: J_HT(xi_a) R(Delta_theta)^T on the base rotation
 * (getBaseRotationTangentJacobian() of the robot model) and J_HT(xi_a)
 * J_r(Delta_theta) J_Delta_theta on the joints; the CoM rows through
 * Pinocchio's CoM Jacobian and getPinocchioTangentJacobian(). Every read of the
 * state is the robot model's, so the cost does not depend on how the state
 * stores the base orientation. Because the offset depends only on the joints,
 * the optimizer can counter-rotate the arms and torso to hold whole-body
 * orientation, which is what produces emergent arm swing during walking.
 *
 * It is selected by listing `com_and_acom_tracking_cost` in the task file's
 * `costs` (MpcCostType), and it REPLACES the base pose rather than adding to
 * it: the base-pose blocks of every quadratic state weight the problem carries
 * - Q in state_quadratic_cost or state_input_quadratic_cost, and Q_final in
 * terminal_cost - are zeroed (zeroBasePoseWeights), so that the same error is
 * not regulated twice in two coordinate systems.
 *
 * The replacement holds at the terminal node too. Zeroing Q_final's block
 * without a substitute would leave the end of the horizon with no CoM, height
 * or orientation regulation at all, and terminalCostScaling makes the terminal
 * node the heaviest of the horizon; so with terminal_cost the problem also
 * carries a second instance in its final cost, weighted by terminalCostScaling
 * * Q_com and terminalCostScaling * Q_acom - exactly how Q_final relates to Q
 * for the rest of the state. Under the DCM terminal cost there is no Q_final
 * and no terminal instance.
 *
 * Not thread-safe: the const evaluations write its mutable Pinocchio data, so
 * the solver gives each worker thread its own clone().
 */
class ComAndAcomTrackingCost : public StateCost {
 public:
  /// The name of the running cost in OptimalControlProblem::stateCostPtr, which
  /// the live parameter updater looks up.
  static constexpr absl::string_view kRunningTermName =
      "comAndAcomTrackingCost";
  /// The name of the terminal instance in OptimalControlProblem::finalCostPtr,
  /// present only beside terminal_cost.
  static constexpr absl::string_view kTerminalTermName =
      "terminalComAndAcomTrackingCost";

  /**
   * Creates the cost, and checks that every piece of it describes the same
   * robot.
   *
   * @param Q_com   3x3 weight on the CoM position error.
   * @param Q_acom  3x3 weight on the ACoM heading-tilt residual, rows and
   * columns (yaw, pitch, roll): the twist heading and the tilt about the
   * heading frame's y and x axes (orientation/BaseOrientation.h).
   * @param pinocchioInterface  Reduced model matching the MPC's joint set.
   * @param info    Centroidal model info for that same model; checked against
   * it, not retained.
   * @param mpcRobotModel  The MPC's robot model, retained: every read of the
   * state (the configuration, the base orientation, the joint angles, the
   * tangent maps) and the reference interpolation (getDesiredState()) are its,
   * so it must outlive the cost.
   * @param robotName  model_settings.robot_name; selects the compiled-in aCOM
   * weights.
   *
   * @return InvalidArgument if a weight matrix is not 3x3, `info` does not
   * describe `pinocchioInterface`, or `mpcRobotModel`'s configuration, velocity
   * or joint dimensions are not the Pinocchio model's; NotFound if no aCOM
   *     network is registered for `robotName`; FailedPrecondition if the
   * network was trained on a joint vector other than the model's, compared name
   * by name (see AngularCenterOfMass::Create).
   */
  static absl::StatusOr<std::unique_ptr<ComAndAcomTrackingCost>> Create(
      matrix_t Q_com,
      matrix_t Q_acom,
      PinocchioInterface pinocchioInterface,
      const CentroidalModelInfo& info,
      const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel,
      absl::string_view robotName);

  ~ComAndAcomTrackingCost() override = default;
  ComAndAcomTrackingCost& operator=(const ComAndAcomTrackingCost&) = delete;
  ComAndAcomTrackingCost(ComAndAcomTrackingCost&&) = delete;
  ComAndAcomTrackingCost& operator=(ComAndAcomTrackingCost&&) = delete;
  ComAndAcomTrackingCost* absl_nonnull clone() const override;

  bool isActive(scalar_t /*time*/) const override { return true; }

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const TargetTrajectories& targetTrajectories,
      const PreComputation& preComp) const override;

  /**
   * Returns InvalidArgument, naming the offending weight block of the task
   * file, unless both matrices are 3x3.
   */
  static absl::Status validateWeights(const matrix_t& Q_com,
                                      const matrix_t& Q_acom);

  /**
   * @brief Update the CoM and ACoM tracking weight matrices in place.
   *
   * Called by the live parameter updater on every per-thread copy of the
   * problem, from the solver thread before the worker pool starts.
   *
   * @return validateWeights()'s InvalidArgument, with the weights left as they
   * were, if either matrix is not 3x3.
   */
  absl::Status setWeights(matrix_t Q_com, matrix_t Q_acom);

  const matrix_t& getQCom() const { return Q_com_; }
  const matrix_t& getQAcom() const { return Q_acom_; }

  /**
   * Zeroes the base-pose block of a centroidal state weight matrix: the 6x6
   * diagonal block Q(6:11, 6:11) of the rows base_position and base_orientation
   * (yaw, pitch, roll) of the centroidal tuning layout (tuningLayoutOf()), i.e.
   * the weights among p_base and the base orientation. Rows of the tuning
   * layout, which Q is written in: they do not move when the state stores the
   * orientation as a quaternion. Entries outside that block, including any that
   * couple the base pose to another state, are left as they are.
   *
   * This cost regulates those quantities in its own coordinates (CoM position
   * and aCOM orientation), so every quadratic state cost that runs alongside it
   * - the running Q and the terminal Q_final, at start-up and on every live
   * reload - has to drop them, or the same physical error is penalized twice in
   * two parameterizations. This is the one definition of that block:
   * trackingStateWeightsFromConfig()
   * (config/weights/StateInputWeightsFromConfig.h) calls it for the start-up
   * factory and the live parameter updater alike, and neither keeps its own
   * copy of the rows.
   *
   * @param Q  A square weight matrix over the centroidal tuning layout (at
   * least 12 x 12). Nothing outside the block changes.
   */
  static void zeroBasePoseWeights(matrix_t& Q);

  /**
   * The joints of `model` that follow its floating base, in the order the
   * centroidal state stores them: the last `actuatedDofNum` joint names. These
   * are what the aCOM network is fed, so they are what its recorded joint names
   * are checked against. InvalidArgument when `info` does not describe the
   * model: its actuatedDofNum is not the number of velocities after the
   * floating base's (the model's root joint's).
   */
  static absl::StatusOr<std::vector<std::string>> actuatedJointNames(
      const PinocchioInterface& pinocchioInterface,
      const CentroidalModelInfo& info);

 private:
  ComAndAcomTrackingCost(
      matrix_t Q_com,
      matrix_t Q_acom,
      PinocchioInterface pinocchioInterface,
      const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModel,
      std::unique_ptr<AngularCenterOfMass> acom);
  ComAndAcomTrackingCost(const ComAndAcomTrackingCost& rhs);

  /** Computes the ACoM attitude xi_a of `state`, from the robot model's base
   * orientation and joint angles. */
  vector4_t acomAttitudeOf(const vector_t& state) const;

  matrix_t Q_com_;
  matrix_t Q_acom_;
  mutable PinocchioInterface pinocchioInterface_;
  // Reads the state and interpolates the reference state of a node; shared,
  // read-only, across the solver's clones.
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;
  std::unique_ptr<AngularCenterOfMass> acom_;
};

}  // namespace ocs2::humanoid
