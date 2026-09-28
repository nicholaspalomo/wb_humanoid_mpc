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

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Analytical linear StateInputConstraint enforcing the contact wrench cone on a contact point.
 *
 * Implements N + 7 linear inequality constraints:
 *  1. Friction cone approximation with numBasisVectors basis vectors:
 *     mu * (Fz_local + F_grip) - (cos(theta_k) * Fx_local + sin(theta_k) * Fy_local) >= 0, theta_k = 2*pi*k/N
 *  2. Normal force lower bound:
 *     Fz_local - minNormalForce >= 0
 *  3. Center of pressure (CoP) / Contact moments within rectangular footprint [x_min, x_max] x [y_min, y_max]:
 *     tau_x_local - y_min * Fz_local >= 0
 *     -tau_x_local + y_max * Fz_local >= 0
 *     -tau_y_local - x_min * Fz_local >= 0
 *     tau_y_local + x_max * Fz_local >= 0
 *  4. Torsional yaw friction moment about the contact patch center:
 *     mu_torsion * (Fz_local + F_grip) +/- (tau_z_local - (x_offset * Fy_local - y_offset * Fx_local)) >= 0
 *
 * The rows are linear in the LOCAL wrench, but the input stores the wrench in the WORLD frame, so every row reads
 * `l_R_w(q) * W_world`: the term depends on the foot's orientation, and its state Jacobian is not zero. It used to be
 * reported as zero. That was never exact - the orientation a foot lands in is a state the swing chooses - but it hurt
 * little while the cone was gated alongside a `zero_velocity` that froze a loaded foot's orientation for the rest of
 * its stance. The contact-implicit formulation un-gates the cone and deliberately leaves the rocking rates of a loaded
 * foot free, and then a friction row of a foot carrying 800 N changes by about 800 per radian of pitch while the
 * solver was told 0. The derivative is now taken through the rotation:
 *
 *   d(l_R_w v) / dq = [l_R_w v]x * J_omega,local(q),
 *
 * with J_omega,local the angular rows of the contact frame's LOCAL Jacobian, for both the force and the moment.
 *
 * WRENCH-SPACE MODELS ONLY. The rows are written into the input Jacobian at getContactForceStartIndices() and
 * getContactMomentStartIndices(), three columns each, which is the layout of a model whose input stores the contact
 * wrench directly. Under BasisInputsModelDecorator both indices are the start of one block of basis scalings, the
 * accessors return a LOCAL wrench that this term would rotate a second time, and the moment block would overwrite the
 * force block - so the term refuses such a model: Create() returns InvalidArgument, and the constructor CHECKs the
 * same condition for the callers that have not moved to Create() yet. On a basis model the cone is enforced
 * structurally, by BasisScalingNonNegativityConstraint.
 */
class ContactWrenchConeConstraint final : public StateInputConstraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    explicit Config(size_t numBasisVectorsParam = 4,
                    scalar_t frictionCoefficientParam = 0.7,
                    scalar_t torsionalFrictionCoefficientParam = 0.05,
                    scalar_t minNormalForceParam = 5.0,
                    scalar_t gripperForceParam = 0.0,
                    vector3_t patchOffsetParam = vector3_t::Zero())
        : numBasisVectors(numBasisVectorsParam),
          frictionCoefficient(frictionCoefficientParam),
          torsionalFrictionCoefficient(torsionalFrictionCoefficientParam),
          minNormalForce(minNormalForceParam),
          gripperForce(gripperForceParam),
          patchOffset(std::move(patchOffsetParam)) {}
    // The ranges are checked by validateConfig(), not here: a constructor cannot report which task-file key is wrong,
    // and the assert()s that stood here were compiled out of every optimized build.

    size_t numBasisVectors;
    scalar_t frictionCoefficient;
    scalar_t torsionalFrictionCoefficient;
    scalar_t minNormalForce;
    scalar_t gripperForce;
    vector3_t patchOffset;
  };

  /**
   * @param scheduleGated  When true (the historical behavior) the term switches itself off while the mode schedule
   *                       calls this foot a swing foot, because the hard `zero_wrench` constraint has already pinned
   *                       its wrench to zero. When false - the contact-implicit formulation, which removes
   *                       `zero_wrench` - the cone is enforced at every node, and the two affine offsets that assume a
   *                       loaded foot (`minNormalForce`, `gripperForce`) are dropped, because a foot in flight carries
   *                       no wrench and must not be asked for a minimum normal force. What is left is the homogeneous
   *                       cone, which the zero wrench satisfies exactly and which is the same set
   *                       ContactWrenchConeBasisMatrix verifies its generators against.
   *                       See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
   */
  ContactWrenchConeConstraint(const SwitchedModelReferenceManager& referenceManager,
                              const ContactRectangle& contactRectangle,
                              size_t contactPointIndex,
                              const PinocchioInterface& pinocchioInterface,
                              const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                              Config config = Config(),
                              bool scheduleGated = true);

  /**
   * The constructor, with the model and the configuration checked first: a basis-vector model, or any model whose input
   * does not store this contact's wrench as six consecutive entries, and a Config that validateConfig() refuses, are
   * reported with InvalidArgument instead of a CHECK. Prefer it.
   *
   * The reference manager, the Pinocchio interface and the robot model are BORROWED, as by every other contact term:
   * they are owned by the MPC interface, which outlives its optimal control problem and every copy of it.
   */
  static absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> Create(const SwitchedModelReferenceManager& referenceManager,
                                                                             const ContactRectangle& contactRectangle,
                                                                             size_t contactPointIndex,
                                                                             const PinocchioInterface& pinocchioInterface,
                                                                             const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                             Config config = Config(),
                                                                             bool scheduleGated = true);

  /**
   * OkStatus when the model's input stores contact `contactPointIndex`'s wrench as [force, moment] in six consecutive
   * entries - the wrench-space layout this term writes its Jacobian in - and InvalidArgument otherwise.
   */
  static absl::Status checkWrenchSpaceInput(const MpcRobotModelBase<scalar_t>& mpcRobotModel, size_t contactPointIndex);

  /** Whether this term is gated on the mode schedule's contact flag; see the constructor. */
  bool isScheduleGated() const { return scheduleGated_; }

  /** The offsets a non-gated cone drops, so that a caller can state what it asked for. */
  static Config withoutLoadedFootOffsets(Config config);

  /** The task-file block a Config is read from; see loadConfig(). */
  static constexpr absl::string_view kConfigBlock = "contacts.contactWrenchConeSoftConstraint";

  /**
   * OkStatus when every value of `config` is in range: numBasisVectors at least 3 (the facets of the friction pyramid),
   * frictionCoefficient finite and positive, torsionalFrictionCoefficient, minNormalForce and gripperForce finite and
   * non-negative, and patchOffset finite. Otherwise InvalidArgument naming the kConfigBlock key to change and the value
   * it holds. loadConfig(), Create() and ContactWrenchConeBasisMatrix::Create() all check through this one function;
   * buildLocalWrenchConeRows() and the constructor CHECK it, for the callers that bypass all three.
   */
  static absl::Status validateConfig(const Config& config);

  /**
   * Reads the ground of the contact wrench cone - frictionCoefficient, torsionalFrictionCoefficient, minNormalForce,
   * gripperForce and numBasisVectors - from the task file's kConfigBlock. That one block is the ground of the whole-body
   * constraints, and every consumer reads it through this function: this term in wrench mode, the generators of the
   * basis-vector parameterization, and the friction and torsion bounds the online contact planner derives. Every key is
   * required. A missing one used to leave the library default of Config (mu 0.7, minNormalForce 5 N) in place without
   * a word, which on a robot that configures its friction elsewhere was a different ground from the one in its file.
   *
   * @return InvalidArgument naming the contacts.contactWrenchConeSoftConstraint.<key> that is missing, not a number or
   *         out of range; NotFound when the file cannot be read.
   */
  static absl::StatusOr<Config> loadConfig(const std::string& taskFile, bool verbose = false);

  ~ContactWrenchConeConstraint() override = default;
  ContactWrenchConeConstraint(const ContactWrenchConeConstraint& other);
  ContactWrenchConeConstraint* clone() const override { return new ContactWrenchConeConstraint(*this); }

  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t time) const override { return numConstraints_; }

  const Config& getConfig() const { return config_; }

  vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override;

  VectorFunctionLinearApproximation getLinearApproximation(scalar_t time,
                                                           const vector_t& state,
                                                           const vector_t& input,
                                                           const PreComputation& preComp) const override;

  /**
   * The value and first-order blocks of getLinearApproximation(), with zero second-order blocks. The term is declared
   * ConstraintOrder::Linear, so the soft-constraint wrapper linearizes it and never calls this; the zeros are a
   * Gauss-Newton approximation, not the exact curvature, which is non-zero in the state (the rotation) and in the
   * state-input cross term.
   */
  VectorFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,
                                                                 const vector_t& state,
                                                                 const vector_t& input,
                                                                 const PreComputation& preComp) const override;

 private:
  void initializeLocalConstraintMatrix();

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  const PinocchioInterface* pinocchioInterfacePtr_;
  // Borrowed, like the two pointers above. It used to hold a clone() in this raw pointer, which nothing deleted, so
  // every construction and every per-thread copy of the problem leaked a whole robot model.
  const MpcRobotModelBase<scalar_t>* mpcRobotModelPtr_;
  const ContactRectangle contactRectangle_;
  const size_t contactPointIndex_;
  const Config config_;

  size_t numConstraints_;
  bool isActive_ = true;
  // Fixed by the formulation at load time rather than tuned, so it is const and the parallel solve reads it without
  // synchronization. It has to survive the copy the SQP solver makes of the whole problem per worker thread.
  const bool scheduleGated_;

  /// Matrix of linear constraint coefficients multiplying the local 3D contact force vector (f_local in R^3).
  /// Enforces friction cone pyramid facets, normal force lower bounds, and CoP/torsional force couplings.
  matrix_t A_f_local_;

  /// Matrix of linear constraint coefficients multiplying the local 3D contact moment vector (tau_local in R^3).
  /// Enforces roll (tau_x) and pitch (tau_y) Center of Pressure (CoP) boundaries as well as yaw (tau_z) torsional friction limits.
  matrix_t A_tau_local_;

  /// Constant offset/bias vector for the linear inequality constraint: A_f_local * f_local + A_tau_local * tau_local + b_local >= 0.
  /// Encodes constant terms such as minimum normal force (-minNormalForce) and gripper adhesion forces (mu * F_grip).
  vector_t b_local_;

  /// d(generalized coordinates) / d(state), getGenCoordinatesDim() x getStateDim(), read off the model once at
  /// construction. getGeneralizedCoordinates() is a selection of the state in every model (the tail of a centroidal
  /// state, the head of a whole-body one), so this is how the frame Jacobian, which is taken in q, reaches the state.
  matrix_t generalizedCoordinatesStateJacobian_;
};

/**
 * The rows of the linearized contact wrench cone in the local contact frame:
 *
 *   A_f * F_local + A_tau * tau_local + b >= 0   (element-wise)
 *
 * `b` holds the non-homogeneous offsets (the gripper force and -minNormalForce). Dropping it leaves the homogeneous
 * cone, which is the set a conic combination of wrench generators has to stay inside.
 */
struct ContactWrenchConeRows {
  matrix_t A_f;    ///< numRows x 3, multiplies the local-frame contact force
  matrix_t A_tau;  ///< numRows x 3, multiplies the local-frame contact moment
  vector_t b;      ///< numRows, non-homogeneous offsets

  size_t numRows() const { return static_cast<size_t>(b.size()); }

  /** Row values of the full constraint. The wrench is admissible when every entry is non-negative. */
  vector_t evaluate(const vector6_t& wrenchLocal) const { return A_f * wrenchLocal.head<3>() + A_tau * wrenchLocal.tail<3>() + b; }

  /** Row values of the homogeneous cone (without `b`), which every wrench-cone generator has to satisfy. */
  vector_t evaluateCone(const vector6_t& wrenchLocal) const { return A_f * wrenchLocal.head<3>() + A_tau * wrenchLocal.tail<3>(); }
};

/**
 * Reference point of the torsional friction rows: `config.patchOffset` when it is set, the center of the footprint
 * otherwise. Only the x and y components are used; the z component is returned unchanged.
 */
vector3_t contactPatchReferencePoint(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle);

/**
 * Builds the local-frame rows of the wrench cone. Shared by ContactWrenchConeConstraint (which enforces them) and by
 * ContactWrenchConeBasisMatrix (whose generators have to satisfy them), so that the two can never drift apart.
 *
 * `config` must pass ContactWrenchConeConstraint::validateConfig(), which every factory of the two checks and reports
 * as a Status naming the task-file key; a config that bypassed them CHECK-fails here with the same message.
 */
ContactWrenchConeRows buildLocalWrenchConeRows(const ContactWrenchConeConstraint::Config& config, const ContactRectangle& contactRectangle);

}  // namespace ocs2::humanoid
