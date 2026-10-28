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

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/constraint/StateInputConstraint.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Analytical linear StateInputConstraint enforcing the contact wrench cone on a
 * contact point.
 *
 * The rows are the H-representation of the contact wrench cone (CWC) of a
 * rectangular footprint: the set of wrenches that point forces at the four
 * footprint corners, each inside its own friction pyramid, can produce (Caron,
 * Pham and Nakamura, "Stability of surface contacts for humanoid robots:
 * Closed-form formulae of the Contact Wrench Cone for rectangular support
 * areas", ICRA 2015). Its V-representation is cornerContactWrenchGenerators(),
 * and humanoid_nmpc/docs/contact_basis_vectors/README.md derives both. The
 * rows, in this order (ContactWrenchConeRows):
 *  1. N friction facets of the pyramid INSCRIBED in the Coulomb cone (its edges
 * have the tangential ratio mu): mu cos(pi/N) (Fz_local + F_grip) -
 * (cos(theta_k) Fx_local + sin(theta_k) Fy_local) >= 0, theta_k = 2 pi k / N
 *  2. Normal force lower bound:
 *     Fz_local - minNormalForce >= 0
 *  3. Center of pressure (CoP) within the footprint [x_min, x_max] x [y_min,
 * y_max]: tau_x_local - y_min * Fz_local >= 0 -tau_x_local + y_max * Fz_local
 * >= 0 -tau_y_local - x_min * Fz_local >= 0 tau_y_local + x_max * Fz_local >= 0
 *  4. The torsion facets of the CWC: every facet of the conic hull of the 4N
 * corner generators whose normal has a yaw moment component, scaled so that
 * component is +-1. For N = 4 they are Caron's eight closed-form rows, tau_min
 * <= tau_z <= tau_max about the footprint center with tau_min = -mu' (X + Y) Fz
 * + |Y Fx - mu' tau_x| + |X Fy - mu' tau_y|, tau_max = +mu' (X + Y) Fz - |Y Fx
 * + mu' tau_x| - |X Fy + mu' tau_y|,   mu' = mu cos(pi/4); for other N there is
 * no closed form, and buildLocalWrenchConeRows() enumerates the facets at
 * construction. They bound the yaw moment the corner friction can actually
 * supply, so a tangential force applied at an off-center CoP - which has a yaw
 * moment about the center but no free torsion - is not rejected. The torsional
 * limit follows from the friction at the corners: torsionalFrictionCoefficient
 * no longer enters the rows (the online contact planner still reads it, capped
 * by torsionLimitsOfNormalForceAt()).
 *
 * THE FRAME. The rows are written in the contact's own frame, not in the
 * world's: the rectangle is the sole, and on a sole resting flat on a surface
 * patch the contact frame is that patch's frame, whatever the patch's
 * orientation. So this cone takes no patch rotation
 * (SwitchedModelReferenceManager::getContactPatchFrame()), unlike the cones
 * written about a world axis (FrictionForceConeConstraint,
 * PointFrictionConeConstraint); see
 * humanoid_nmpc/docs/multi_contact_planning/README.md, section 7.2.
 *
 * The rows are linear in the LOCAL wrench, but the input stores the wrench in
 * the WORLD frame, so every row reads `l_R_w(q) * W_world`: the term depends on
 * the foot's orientation, and its state Jacobian is not zero. It used to be
 * reported as zero. That was never exact - the orientation a foot lands in is a
 * state the swing chooses - but it hurt little while the cone was gated
 * alongside a `zero_velocity` that froze a loaded foot's orientation for the
 * rest of its stance. The contact-implicit formulation un-gates the cone and
 * deliberately leaves the rocking rates of a loaded foot free, and then a
 * friction row of a foot carrying 800 N changes by about 800 per radian of
 * pitch while the solver was told 0. The derivative is now taken through the
 * rotation:
 *
 *   d(l_R_w v) / dq = [l_R_w v]x * J_omega,local(q),
 *
 * with J_omega,local the angular rows of the contact frame's LOCAL Jacobian,
 * for both the force and the moment, and dq a step of q in Pinocchio's tangent,
 * which MpcRobotModelBase::getPinocchioTangentJacobian() takes to the state.
 *
 * WRENCH-SPACE MODELS ONLY. The rows are written into the input Jacobian at
 * getContactForceStartIndices() and getContactMomentStartIndices(), three
 * columns each, which is the layout of a model whose input stores the contact
 * wrench directly. Under BasisInputsModelDecorator both indices are the start
 * of one block of basis scalings, the accessors return a LOCAL wrench that this
 * term would rotate a second time, and the moment block would overwrite the
 * force block - so the term refuses such a model: Create() returns
 * InvalidArgument, and the constructor CHECKs the same condition for the
 * callers that have not moved to Create() yet. On a basis model the cone is
 * enforced structurally, by BasisScalingNonNegativityConstraint.
 *
 * Not thread-safe: it reads the Pinocchio data of one worker's PreComputation,
 * so the solver gives each worker thread its own clone().
 */
class ContactWrenchConeConstraint final : public StateInputConstraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    /** The facets of the default friction pyramid, and the fewest basis vectors
     * a friction pyramid can have: its facets. */
    static constexpr size_t kDefaultNumBasisVectors = 4;
    static constexpr size_t kMinNumBasisVectors = 3;
    /**
     * The most facets a friction pyramid may have. The torsion rows are
     * enumerated from the C(4N, 5) five-generator subsets of the corner cone,
     * about 2 x 10^5 at N = 8; the inscribed octagon already admits 92 % of the
     * friction circle in its worst direction.
     */
    static constexpr size_t kMaxNumBasisVectors = 8;
    static constexpr scalar_t kDefaultFrictionCoefficient = 0.7;
    static constexpr scalar_t kDefaultTorsionalFrictionCoefficient = 0.05;
    /** [N] */
    static constexpr scalar_t kDefaultMinNormalForce = 5.0;

    explicit Config(
        size_t numBasisVectorsParam = kDefaultNumBasisVectors,
        scalar_t frictionCoefficientParam = kDefaultFrictionCoefficient,
        scalar_t torsionalFrictionCoefficientParam =
            kDefaultTorsionalFrictionCoefficient,
        scalar_t minNormalForceParam = kDefaultMinNormalForce,
        scalar_t gripperForceParam = 0.0,
        vector3_t patchOffsetParam = vector3_t::Zero())
        : numBasisVectors(numBasisVectorsParam),
          frictionCoefficient(frictionCoefficientParam),
          torsionalFrictionCoefficient(torsionalFrictionCoefficientParam),
          minNormalForce(minNormalForceParam),
          gripperForce(gripperForceParam),
          patchOffset(std::move(patchOffsetParam)) {}
    // The ranges are checked by validateConfig(), not here: a constructor
    // cannot report which task-file key is wrong, and the assert()s that stood
    // here were compiled out of every optimized build.

    /** The facets of the friction pyramid, and its edges: kMinNumBasisVectors
     * to kMaxNumBasisVectors. */
    size_t numBasisVectors;
    /** The Coulomb coefficient; the pyramid is inscribed in its cone, so its
     * edges have the tangential ratio mu. */
    scalar_t frictionCoefficient;
    /**
     * [m] The yaw torque per unit normal force the online contact planner
     * budgets for one stance foot. Not a parameter of the wrench cone, whose
     * torsion limit follows from the friction at the footprint corners.
     */
    scalar_t torsionalFrictionCoefficient;
    scalar_t minNormalForce;
    /** [N] An adhesion seen by the friction and torsion rows as an extra normal
     * force at the footprint center. */
    scalar_t gripperForce;
    /** The point the conservative basis generators are applied at
     * (contactPatchReferencePoint()); zero: the center. */
    vector3_t patchOffset;
  };

  /**
   * @param scheduleGated  When true (the historical behavior) the term switches
   * itself off while the mode schedule calls this foot a swing foot, because
   * the hard `zero_wrench` constraint has already pinned its wrench to zero.
   * When false - the contact-implicit formulation, which removes `zero_wrench`
   * - the cone is enforced at every node, and the two affine offsets that
   * assume a loaded foot (`minNormalForce`, `gripperForce`) are dropped,
   * because a foot in flight carries no wrench and must not be asked for a
   * minimum normal force. What is left is the homogeneous cone, which the zero
   * wrench satisfies exactly and which is the same set
   *                       ContactWrenchConeBasisMatrix verifies its generators
   * against. See humanoid_nmpc/docs/contact_implicit_mpc/README.md.
   */
  ContactWrenchConeConstraint(
      const SwitchedModelReferenceManager& referenceManager,
      const ContactRectangle& contactRectangle,
      size_t contactPointIndex,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      Config config = Config(),
      bool scheduleGated = true);

  /**
   * The constructor, with the model and the configuration checked first: a
   * basis-vector model, or any model whose input does not store this contact's
   * wrench as six consecutive entries, a robot model whose Pinocchio velocity
   * is not the Pinocchio model's (checkPinocchioVelocityDim()), and a Config
   * that validateConfig() or a footprint that validateFootprint() refuses, are
   * reported with InvalidArgument instead of a CHECK. Prefer it.
   *
   * The reference manager, the Pinocchio interface and the robot model are
   * BORROWED, as by every other contact term: they are owned by the MPC
   * interface, which outlives its optimal control problem and every copy of it.
   */
  static absl::StatusOr<std::unique_ptr<ContactWrenchConeConstraint>> Create(
      const SwitchedModelReferenceManager& referenceManager,
      const ContactRectangle& contactRectangle,
      size_t contactPointIndex,
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      Config config = Config(),
      bool scheduleGated = true);

  /**
   * OkStatus when the model's input stores contact `contactPointIndex`'s wrench
   * as [force, moment] in six consecutive entries - the wrench-space layout
   * this term writes its Jacobian in - and InvalidArgument otherwise.
   */
  static absl::Status checkWrenchSpaceInput(
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      size_t contactPointIndex);

  /**
   * OkStatus when the robot model's Pinocchio velocity
   * (MpcRobotModelBase::getVelocityDim()) has the Pinocchio model's nv entries
   * - the columns of the frame Jacobian this term takes to the state through
   * getPinocchioTangentJacobian() - and InvalidArgument otherwise.
   */
  static absl::Status checkPinocchioVelocityDim(
      const PinocchioInterface& pinocchioInterface,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel);

  /** Whether this term is gated on the mode schedule's contact flag; see the
   * constructor. */
  bool isScheduleGated() const { return scheduleGated_; }

  /** The offsets a non-gated cone drops, so that a caller can state what it
   * asked for. */
  static Config withoutLoadedFootOffsets(Config config);

  /** The task-file block a Config is converted from
   * (contactWrenchConeConfigFromConfig()); the refusals name its fields. */
  static constexpr absl::string_view kConfigField =
      "contacts.contact_wrench_cone_soft_constraint";

  /**
   * OkStatus when every value of `config` is in range: numBasisVectors from 3
   * (the facets of the friction pyramid) to Config::kMaxNumBasisVectors,
   * frictionCoefficient finite and positive, torsionalFrictionCoefficient,
   * minNormalForce and gripperForce finite and non-negative, and patchOffset
   * finite. Otherwise InvalidArgument naming the field of kConfigField to
   * change and the value it holds. contactWrenchConeConfigFromConfig(),
   * Create() and ContactWrenchConeBasisMatrix::Create() all check through this
   * one function; buildLocalWrenchConeRows() and the constructor CHECK it, for
   * the callers that bypass all three.
   */
  static absl::Status validateConfig(const Config& config);

  /** The task-file block a footprint is converted from; validateFootprint()
   * names its fields. */
  static constexpr absl::string_view kFootprintField =
      "contacts.contact_rectangle";

  /**
   * OkStatus when the footprint has finite bounds with x_min < x_max and y_min
   * < y_max, InvalidArgument naming kFootprintField otherwise. The cone of a
   * footprint of zero length or width (a line or a point contact) is not
   * full-dimensional, and its torsion rows would be equalities this
   * construction does not produce.
   */
  static absl::Status validateFootprint(
      const ContactRectangle& contactRectangle);

  ~ContactWrenchConeConstraint() override = default;
  ContactWrenchConeConstraint& operator=(const ContactWrenchConeConstraint&) =
      delete;
  ContactWrenchConeConstraint(ContactWrenchConeConstraint&&) = delete;
  ContactWrenchConeConstraint& operator=(ContactWrenchConeConstraint&&) =
      delete;
  ContactWrenchConeConstraint(const ContactWrenchConeConstraint& other);
  ContactWrenchConeConstraint* absl_nonnull clone() const override {
    return new ContactWrenchConeConstraint(*this);
  }

  bool isActive(scalar_t time) const override;
  void setActive(bool isActive) override { isActive_ = isActive; }
  bool getActive() const override { return isActive_; }
  size_t getNumConstraints(scalar_t /*time*/) const override {
    return numConstraints_;
  }

  const Config& getConfig() const { return config_; }

  vector_t getValue(scalar_t time,
                    const vector_t& state,
                    const vector_t& input,
                    const PreComputation& preComp) const override;

  VectorFunctionLinearApproximation getLinearApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) const override;

  /**
   * The value and first-order blocks of getLinearApproximation(), with zero
   * second-order blocks. The term is declared ConstraintOrder::Linear, so the
   * soft-constraint wrapper linearizes it and never calls this; the zeros are a
   * Gauss-Newton approximation, not the exact curvature, which is non-zero in
   * the state (the rotation) and in the state-input cross term.
   */
  VectorFunctionQuadraticApproximation getQuadraticApproximation(
      scalar_t time,
      const vector_t& state,
      const vector_t& input,
      const PreComputation& preComp) const override;

 private:
  void initializeLocalConstraintMatrix();

  const SwitchedModelReferenceManager* absl_nonnull referenceManagerPtr_;
  const PinocchioInterface* absl_nonnull pinocchioInterfacePtr_;
  // Borrowed, like the two pointers above. It used to hold a clone() in this
  // raw pointer, which nothing deleted, so every construction and every
  // per-thread copy of the problem leaked a whole robot model.
  const MpcRobotModelBase<scalar_t>* absl_nonnull mpcRobotModelPtr_;
  const ContactRectangle contactRectangle_;
  const size_t contactPointIndex_;
  const Config config_;

  size_t numConstraints_ = 0;
  bool isActive_ = true;
  // Fixed by the formulation at load time rather than tuned, so it is const and
  // the parallel solve reads it without synchronization. It has to survive the
  // copy the SQP solver makes of the whole problem per worker thread.
  const bool scheduleGated_;

  /// The rows of buildLocalWrenchConeRows() multiplying the local contact
  /// force: friction facets, normal force, the Fz terms of the CoP rows and the
  /// force terms of the torsion rows.
  matrix_t A_f_local_;

  /// The rows multiplying the local contact moment: the CoP rows (tau_x, tau_y)
  /// and the torsion rows.
  matrix_t A_tau_local_;

  /// Constant offsets of A_f_local * f_local + A_tau_local * tau_local +
  /// b_local >= 0: -minNormalForce and the gripper force terms.
  vector_t b_local_;
};

/**
 * The rows of the linearized contact wrench cone in the local contact frame:
 *
 *   A_f * F_local + A_tau * tau_local + b >= 0   (element-wise)
 *
 * `b` holds the non-homogeneous offsets (the gripper force and
 * -minNormalForce). Dropping it leaves the homogeneous cone, which is the set a
 * conic combination of wrench generators has to stay inside. The rows come in
 * the order of the ContactWrenchConeConstraint class comment: numFrictionRows
 * friction facets, one normal-force row, the four center-of-pressure rows, and
 * the torsion rows from torsionRowStart() to the end.
 */
struct ContactWrenchConeRows {
  /** The rows between the friction facets and the torsion rows: the
   * normal-force row and the four CoP rows. */
  static constexpr size_t kNumNormalForceAndCopRows = 5;

  matrix_t A_f;    ///< numRows x 3, multiplies the local-frame contact force
  matrix_t A_tau;  ///< numRows x 3, multiplies the local-frame contact moment
  vector_t b;      ///< numRows, non-homogeneous offsets
  /** The friction facets, the first rows; Config::numBasisVectors. */
  size_t numFrictionRows = 0;

  size_t numRows() const { return static_cast<size_t>(b.size()); }

  /** Index of the first torsion row. */
  size_t torsionRowStart() const {
    return numFrictionRows + kNumNormalForceAndCopRows;
  }

  /** Number of torsion rows (8 for N = 4). */
  size_t numTorsionRows() const { return numRows() - torsionRowStart(); }

  /** Row values of the full constraint. The wrench is admissible when every
   * entry is non-negative. */
  vector_t evaluate(const vector6_t& wrenchLocal) const {
    return A_f * wrenchLocal.head<kForceDim>() +
           A_tau * wrenchLocal.tail<kTorqueDim>() + b;
  }

  /** Row values of the homogeneous cone (without `b`), which every wrench-cone
   * generator has to satisfy. */
  vector_t evaluateCone(const vector6_t& wrenchLocal) const {
    return A_f * wrenchLocal.head<kForceDim>() +
           A_tau * wrenchLocal.tail<kTorqueDim>();
  }
};

/**
 * The point the conservative basis generators are applied at:
 * `config.patchOffset` when it is set, the center of the footprint otherwise.
 * Only the x and y components are used; the z component is returned unchanged.
 */
vector3_t contactPatchReferencePoint(
    const ContactWrenchConeConstraint::Config& config,
    const ContactRectangle& contactRectangle);

/**
 * The tangential ratio |F_t| / Fz the inscribed friction pyramid admits in
 * every direction, mu cos(pi / N): the radius of the circle inscribed in its Fz
 * = 1 polygon. A consumer that models friction as a circle (the online contact
 * planner) must not assume more than this.
 */
scalar_t inscribedFrictionCoefficient(
    const ContactWrenchConeConstraint::Config& config);

/**
 * The N edges of the friction pyramid at unit normal force, (mu cos(phi_k), mu
 * sin(phi_k), 1) with phi_k = (k + 1/2) 2 pi / N: on the Coulomb cone, half a
 * sector away from the facets of buildLocalWrenchConeRows().
 */
std::vector<vector3_t> frictionPyramidEdges(
    const ContactWrenchConeConstraint::Config& config);

/**
 * The footprint corners in the order every generator set uses: (x_max, y_max),
 * (x_max, y_min), (x_min, y_max), (x_min, y_min).
 */
std::array<vector2_t, ContactRectangle::kNumCorners> footprintCorners(
    const ContactRectangle& contactRectangle);

/**
 * The V-representation of the contact wrench cone: the 6 x 4N local wrenches of
 * a unit-normal point force along each friction-pyramid edge k at each
 * footprint corner c, column k * 4 + c, moments about the contact frame origin.
 * Every wrench the rows of buildLocalWrenchConeRows() admit is a non-negative
 * combination of these columns, and every such combination satisfies the rows.
 */
matrix_t cornerContactWrenchGenerators(
    const ContactWrenchConeConstraint::Config& config,
    const ContactRectangle& contactRectangle);

/**
 * The first numBasisVectors + 1 rows of buildLocalWrenchConeRows(), which
 * bound the force alone: the facets of the friction pyramid and the normal
 * force bound, with A_tau zero. They are the whole cone of a 3-DoF contact
 * (PointFrictionConeConstraint), which carries no moment. `config` must pass
 * ContactWrenchConeConstraint::validateConfig(); one that does not CHECK-fails
 * with its message.
 */
ContactWrenchConeRows buildFrictionPyramidRows(
    const ContactWrenchConeConstraint::Config& config);

/**
 * Builds the local-frame rows of the wrench cone. Shared by
 * ContactWrenchConeConstraint (which enforces them) and by
 * ContactWrenchConeBasisMatrix (whose generators have to satisfy them), so that
 * the two can never drift apart.
 *
 * `config` must pass ContactWrenchConeConstraint::validateConfig() and the
 * footprint validateFootprint(), which every factory of the two checks and
 * reports as a Status naming the task-file key; a config that bypassed them
 * CHECK-fails here with the same message. Cost: the torsion rows are enumerated
 * from the C(4N, 5) five-generator subsets of cornerContactWrenchGenerators(),
 * a few milliseconds at N = 4 and well under a second at N = 8; call it once
 * per foot.
 */
ContactWrenchConeRows buildLocalWrenchConeRows(
    const ContactWrenchConeConstraint::Config& config,
    const ContactRectangle& contactRectangle);

/** The yaw moments a wrench can carry with its other five components fixed:
 * lower <= tau_z <= upper. */
struct TorsionInterval {
  scalar_t lower = 0.0;  ///< [N m] non-positive for a wrench inside the cone
  scalar_t upper = 0.0;  ///< [N m] non-negative for a wrench inside the cone
};

/**
 * The yaw moments the homogeneous cone of `rows` admits for a unit normal force
 * at the planar point `point` of the contact frame with no tangential force:
 * the torsion limits per newton there (for N = 4 at the footprint center,
 * +-mu cos(pi/4) (X + Y) with X, Y the footprint half extents). Both are zero
 * on a footprint corner; nullopt when the point is off the footprint, where no
 * yaw moment makes the wrench admissible.
 */
std::optional<TorsionInterval> torsionLimitsOfNormalForceAt(
    const ContactWrenchConeRows& rows, const vector2_t& point);

}  // namespace ocs2::humanoid
