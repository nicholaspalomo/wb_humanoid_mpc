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

#include <cstddef>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_core/reference/TargetTrajectories.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"  // NOLINT(euler-boundary): Euler boundary accessor (design section 4.2)

namespace ocs2::humanoid {

/** The sizes of an MPC formulation's state, input and Pinocchio model, for
 * MpcRobotModelBase's constructor. */
struct MpcRobotModelDimensions {
  /** Entries of the state vector: its ambient size nx. */
  size_t stateDim = 0;
  /** Entries of the input vector. */
  size_t inputDim = 0;
  /** Entries of the Pinocchio configuration q: nq. */
  size_t configurationDim = 0;
  /** Entries of the Pinocchio velocity v, the tangent of q: nv. */
  size_t velocityDim = 0;
};

/**
 * The Jacobian of the base's body angular velocity w_B with respect to the two
 * blocks of state slots it depends on
 * (MpcRobotModelBase::getJacobianAngvelBaseRtWorldEwrtBase), as small blocks
 * instead of a row block of the state.
 */
struct BaseAngularVelocityJacobian {
  /** d w_B / d(the orientation's state slots), 3 x 4: zero, as the state stores
   * w_B itself. */
  matrix_t orientationSlots;
  /** d w_B / d(the angular-velocity state slots): the identity, for the same
   * reason. */
  matrix3_t angularVelocitySlots = matrix3_t::Zero();
};

/**
 * The point of the robot whose linear velocity a formulation's state stores
 * (MpcRobotModelBase::getStoredLinvelPoint()): the centroidal state stores the
 * center of mass's (the linear part of the normalized centroidal momentum), the
 * whole-body state the base origin's (dp_W). A consumer that reads "the robot's
 * translational velocity" on either model reads the accessor of this point and
 * says which quantity it takes it for.
 */
enum class StoredLinvelPoint { kCenterOfMass, kBaseOrigin };

/**
 * The axes a contact's input-parameterized wrench is expressed in
 * (MpcRobotModelBase::getContactInputFrame()), the frame `param` of
 * getWrenchContactEwrtParam(): world axes for the wrench-space models, whose
 * input stores the world-frame wrench, and the contact frame for
 * BasisInputsModelDecorator, whose scalings weight a basis fixed to the foot.
 * A term that rotates the param-frame wrench itself, rather than calling the
 * state-aware `...ContactEwrtWorld` accessors, reads it to know whether the
 * rotation, and with it the wrench, depends on the state.
 */
enum class ContactInputFrame { kWorld, kContact };

/**
 * The z component of the angular part of a base velocity whose linear part
 * comes first: the yaw rate of the world twist [dp_W, R(xi) w_B] of
 * getTwistBaseRtWorldEwrtWorld(). The same entry of the centroidal state is the
 * vertical angular momentum per mass L_z / m, and of the whole body's
 * tuning-order base rates the body roll rate w_x (design defect D18), so a yaw
 * rate is read through getRefPlanarMotion() instead.
 */
inline constexpr int kTwistYawRateIndex = kPositionDim + kZAxisIndex;

/**
 * The state and input layout of one MPC formulation: where the base pose, the
 * joints and the contact wrenches sit in the state and input vectors, and how
 * to read and write them, for scalar_t and for the CppAD scalar.
 *
 * Each formulation derives one. It keeps a reference to the ModelSettings it is
 * built with, which must outlive it, and is cloned into every term that needs
 * it. The base class holds the layout's sizes and its tuning layout
 * (getTuningLayout()), and its const functions are safe to call concurrently.
 *
 * Layout-agnostic accessors. A consumer reads and writes the state only through
 * the functions below, so that it does not depend on how the base orientation
 * is stored: as the coefficients (x, y, z, w) of a unit quaternion xi from the
 * base frame to the world, four state slots whose tangent is the body rotation
 * vector (three rows), with the Pinocchio root joint Composite(Translation,
 * Spherical) (humanoid_nmpc/docs/quaternion_base_orientation/README.md,
 * sections 2.2 and 2.3). The state therefore has one entry more than its
 * tangent, and Pinocchio's configuration q one more than its velocity v. The
 * accessors cannot fail; those the solver and the control thread call stay
 * unchecked, with the sizes documented as preconditions. The Euler boundary
 * accessors (getEulerzyxBaseRtWorld / setEulerzyxBaseRtWorld,
 * getPoseeulerzyxBaseRtWorld / setPoseeulerzyxBaseRtWorld, and
 * setTurnAngvelBaseRtWorldEwrtBase) are for the human-facing inputs and
 * outputs only - commands and their knots, the locomotion-heuristic seam,
 * telemetry and display - and are in double precision for every SCALAR_T.
 */
template <typename SCALAR_T>
class MpcRobotModelBase {
 public:
  /**
   * The layout of `dimensions`, whose tuning layout `tuningLayout`
   * (tuningLayoutOf()) placeTuningLayout() has placed on a state of
   * dimensions.stateDim entries. Keeps a reference to `modelSettings`, which
   * must outlive the model.
   */
  MpcRobotModelBase(const ModelSettings& modelSettings,
                    const MpcRobotModelDimensions& dimensions,
                    TuningLayout tuningLayout)
      : modelSettings(modelSettings),
        state_dim(dimensions.stateDim),
        input_dim(dimensions.inputDim),
        configurationDim_(dimensions.configurationDim),
        velocityDim_(dimensions.velocityDim),
        stateTangentDim_(getTuningLayoutDim(tuningLayout)),
        tuningLayout_(std::move(tuningLayout)) {
    // Once, here: the tuning layout describes this state (design decision D6).
    ABSL_CHECK_EQ(getTuningLayoutStateDim(tuningLayout_), state_dim)
        << "MpcRobotModelBase: the " << describeTuningLayout(tuningLayout_)
        << " is not placed on the state of " << state_dim << " entries";
  }

  virtual ~MpcRobotModelBase() = default;
  MpcRobotModelBase& operator=(const MpcRobotModelBase&) = delete;
  MpcRobotModelBase(MpcRobotModelBase&&) = delete;
  MpcRobotModelBase& operator=(MpcRobotModelBase&&) = delete;
  virtual MpcRobotModelBase* absl_nonnull clone() const = 0;

  /******************************************************************************************************/
  /*                                           Dimensions */
  /******************************************************************************************************/

  /** Entries of the state vector: its ambient size nx. */
  size_t getStateDim() const { return state_dim; }
  /**
   * Entries of the state's tangent ndx, in which the solver's steps, Q and the
   * feedback gains live: the size of the tuning layout (design decision D6),
   * getStateDim() - 1 (the quaternion's four slots have a three-row tangent).
   */
  size_t getStateTangentDim() const { return stateTangentDim_; }
  size_t getInputDim() const { return input_dim; }
  size_t getJointDim() const { return modelSettings.mpc_joint_dim; }
  /** Entries of the Pinocchio configuration q: nq. */
  size_t getConfigurationDim() const { return configurationDim_; }
  /** Entries of the Pinocchio velocity v, the tangent of q: nv. */
  size_t getVelocityDim() const { return velocityDim_; }

  /******************************************************************************************************/
  /*                                          Start indices */
  /******************************************************************************************************/

  virtual size_t getBaseStartindex() const = 0;
  /**
   * The first state slot of the base orientation: its quaternion coefficients
   * (x, y, z, w), the stateStart of getTuningLayout()'s eulerzyx_base_rt_world
   * block.
   */
  virtual size_t getBaseOrientationStartIndex() const = 0;
  virtual size_t getJointStartindex() const = 0;
  virtual size_t getJointVelocitiesStartindex() const = 0;

  // Assumes contact wrench [f_x, f_y, f_z, M_x, M_y, M_z]^T
  virtual size_t getContactWrenchStartIndices(size_t contactIndex) const = 0;
  virtual size_t getContactForceStartIndices(size_t contactIndex) const {
    return getContactWrenchStartIndices(contactIndex);
  }
  virtual size_t getContactMomentStartIndices(size_t contactIndex) const {
    return kWrenchDim * contactIndex + kForceDim;
  }

  /**
   * Number of input variables that parameterize the contact wrench of one
   * contact, occupying the input block that starts at
   * getContactWrenchStartIndices(contactIndex). Six for a direct wrench
   * parameterization; the basis-vector parameterization
   * (BasisInputsModelDecorator) overrides this with its number of generators
   * per foot. Use it instead of inferring the block size from the surrounding
   * start indices, which differ between models.
   */
  virtual size_t getContactInputDim(size_t /*contactIndex*/) const {
    return kContactWrenchDim;
  }

  /******************************************************************************************************/
  /*                                     Generalized coordinates */
  /******************************************************************************************************/

  virtual VECTOR_T<SCALAR_T> getGeneralizedCoordinates(
      const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR3_T<SCALAR_T> getPosBaseRtWorldEwrtWorld(
      const VECTOR_T<SCALAR_T>& state) const = 0;

  /** Returns the point whose linear velocity the state stores. */
  virtual StoredLinvelPoint getStoredLinvelPoint() const = 0;

  /**
   * Returns the center of mass's linear velocity, which the state stores when
   * getStoredLinvelPoint() is kCenterOfMass (the linear part of the normalized
   * centroidal momentum). A model whose state stores another velocity aborts:
   * calling it there is a programming error.
   */
  virtual VECTOR3_T<SCALAR_T> getLinvelComRtWorldEwrtWorld(
      const VECTOR_T<SCALAR_T>& state) const = 0;

  /**
   * Returns the base origin's linear velocity dp_W, which the state stores when
   * getStoredLinvelPoint() is kBaseOrigin. A model whose state stores another
   * velocity aborts: calling it there is a programming error.
   */
  virtual VECTOR3_T<SCALAR_T> getLinvelBaseRtWorldEwrtWorld(
      const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR_T<SCALAR_T> getJointAngles(
      const VECTOR_T<SCALAR_T>& state) const = 0;

  virtual VECTOR_T<SCALAR_T> getJointVelocities(
      const VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& input) const = 0;

  virtual VECTOR_T<SCALAR_T> getGeneralizedVelocities(
      const VECTOR_T<SCALAR_T>& state, const VECTOR_T<SCALAR_T>& input) = 0;

  /******************************************************************************************************/

  virtual void setGeneralizedCoordinates(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR_T<SCALAR_T>& generalizedCorrdinates) const = 0;

  virtual void setPosBaseRtWorldEwrtWorld(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR3_T<SCALAR_T>& posBaseRtWorldEwrtWorld) const = 0;

  /** Sets what getLinvelComRtWorldEwrtWorld() returns; the same models abort.
   */
  virtual void setLinvelComRtWorldEwrtWorld(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR3_T<SCALAR_T>& linvelComRtWorldEwrtWorld) const = 0;

  /** Sets what getLinvelBaseRtWorldEwrtWorld() returns; the same models abort.
   */
  virtual void setLinvelBaseRtWorldEwrtWorld(
      VECTOR_T<SCALAR_T>& state,
      const VECTOR3_T<SCALAR_T>& linvelBaseRtWorldEwrtWorld) const = 0;

  virtual void setJointAngles(VECTOR_T<SCALAR_T>& state,
                              const VECTOR_T<SCALAR_T>& jointAngles) const = 0;

  virtual void setJointVelocities(
      VECTOR_T<SCALAR_T>& state,
      VECTOR_T<SCALAR_T>& input,
      const VECTOR_T<SCALAR_T>& jointVelocities) const = 0;

  virtual void adaptPosBaseRtWorldEwrtWorldZ(
      VECTOR_T<SCALAR_T>& state,
      scalar_t errPosBaseRtWorldEwrtWorldZ) const = 0;

  /******************************************************************************************************/
  /*                                      Base orientation */
  /******************************************************************************************************/

  /**
   * The base orientation xi of `state`, as the quaternion coefficients (x, y,
   * z, w) of a unit quaternion: the stored coefficients, safely normalized
   * (safelyNormalizedQuaternion(), orientation/BaseOrientation.h; a CppAD
   * conditional on the squared norm, so safe on a tape).
   */
  virtual VECTOR4_T<SCALAR_T> getQuatBaseRtWorldXyzw(
      const VECTOR_T<SCALAR_T>& state) const = 0;

  /**
   * Writes the base orientation `quatBaseRtWorldXyzw` (quaternion coefficients
   * x, y, z, w) into `state`, safely normalized. The sign of the coefficients
   * is kept: xi and -xi are the same rotation.
   */
  virtual void setQuatBaseRtWorldXyzw(
      vector_t& state, const vector4_t& quatBaseRtWorldXyzw) const = 0;

  /**
   * The heading of the base in `state` [rad]: the twist heading
   * yawOf(getQuatBaseRtWorldXyzw(state)) of orientation/BaseOrientation.h
   * (design decision D9), in (-pi, pi]. The angle about the world z axis of the
   * twist-swing split of the attitude: the ZYX yaw, wrapped, at a single-axis
   * tilt, and about -pitch * roll / 2 away from it otherwise; regular at a
   * pitch of 90 degrees, zero upside down. The one heading every reader takes,
   * all of them wrap-safe (cos and sin, wrapped differences,
   * moduloAngleWithReference). Double precision for every SCALAR_T.
   */
  virtual scalar_t getYawBaseRtWorld(const vector_t& state) const = 0;

  /******************************************************************************************************/
  /*                  The Euler boundary (quaternion_base_orientation/README.md,
   * section 4.2)           */
  /******************************************************************************************************/

  // NOLINTBEGIN(euler-boundary): Euler boundary accessor (design section 4.2)
  /**
   * The base pose [x, y, z, yaw, pitch, roll] of `state` (ZYX Euler angles):
   * eulerZyxFromQuaternion of the orientation (orientation/EulerBoundary.h), so
   * the yaw and the roll are in (-pi, pi] and the pitch in [-pi/2, pi/2].
   */
  virtual vector6_t getPoseeulerzyxBaseRtWorld(const vector_t& state) const = 0;

  /**
   * Writes the base pose [x, y, z, yaw, pitch, roll] into `state`:
   * quaternionFromEulerZyx of the angles, of any yaw. A knot at yaw 3.5 reads
   * back as 3.5 - 2 pi; the interpolation between knots
   * (interpolateStateTrajectory()) takes the shorter way, so consecutive knots
   * of a turn are at most pi / 2 apart (the target calculators subdivide them).
   */
  virtual void setPoseeulerzyxBaseRtWorld(
      vector_t& state, const vector6_t& poseeulerzyxBaseRtWorld) const = 0;

  /** The ZYX Euler angles (yaw, pitch, roll) of the base in `state`, as
   * getPoseeulerzyxBaseRtWorld(). */
  virtual vector3_t getEulerzyxBaseRtWorld(const vector_t& state) const = 0;

  /** Writes the ZYX Euler angles (yaw, pitch, roll) of the base into `state`,
   * as setPoseeulerzyxBaseRtWorld(). */
  virtual void setEulerzyxBaseRtWorld(
      vector_t& state, const vector3_t& eulerzyxBaseRtWorld) const = 0;
  // NOLINTEND(euler-boundary)

  /******************************************************************************************************/
  /*                       Base angular velocity (a state that holds the base
   * velocity)                 */
  /******************************************************************************************************/
  /*
   * Only for a model whose state holds the base velocity: one whose
   * getTuningLayout() has a angvel_base_rt_world_ewrt_base block
   * (kZyxOrderedAngularVelocity), the whole-body MPC's. The centroidal state
   * holds the centroidal momentum instead, and its model aborts on these.
   */

  /** The body angular velocity w_B = (x, y, z) of the base in its own frame, as
   * the state stores it. */
  virtual vector3_t getAngvelBaseRtWorldEwrtBase(
      const vector_t& state) const = 0;

  /** Writes the body angular velocity `angvelBaseRtWorldEwrtBase` of the base
   * into `state`. */
  virtual void setAngvelBaseRtWorldEwrtBase(
      vector_t& state, const vector3_t& angvelBaseRtWorldEwrtBase) const = 0;

  /**
   * Writes the angular velocity of a turn at `angvelBaseRtWorldEwrtWorldZ`
   * (rad/s) about the world z axis into `state`, as a velocity command knot
   * holds it: w_B = R(xi_hat)^T (0, 0, angvelBaseRtWorldEwrtWorldZ) at the
   * orientation `state` already holds, (0, 0, angvelBaseRtWorldEwrtWorldZ) for
   * a level knot. Write the knot's orientation first.
   */
  virtual void setTurnAngvelBaseRtWorldEwrtBase(
      vector_t& state, scalar_t angvelBaseRtWorldEwrtWorldZ) const = 0;

  /**
   * d w_B / d(state), as its two nonzero blocks: with respect to the
   * orientation slots (at getBaseOrientationStartIndex()) and to the
   * angular-velocity slots (the stateStart of getTuningLayout()'s
   * angvel_base_rt_world_ewrt_base block).
   */
  virtual BaseAngularVelocityJacobian getJacobianAngvelBaseRtWorldEwrtBase(
      const vector_t& state) const = 0;

  /******************************************************************************************************/
  /*                    The base's motion and a target knot's command (double
   * precision)                */
  /******************************************************************************************************/

  /**
   * The base twist [v_W, w_W] of `state` with the joints moving at
   * `jointVelocities` (getJointDim() entries; the robot models die on another
   * size), in the world frame: the linear velocity of the base origin and the
   * angular velocity of the base, both in world coordinates, so w_W(2) is the
   * turn rate of the base about the world z axis. Exact in every model. The
   * motion manager reads it once per (re)start of its stride history, from the
   * observation the solve starts from, for the motion before the history's
   * first sample, which the targets read and the gait switches do not
   * (ProceduralMpcMotionManager::observe(); design defects D1, D1b, D2).
   *
   * A state that holds the base velocity is read, and `jointVelocities` is not:
   * the whole-body w_W = R(xi) w_B. The centroidal state holds the normalized
   * centroidal momentum h instead, and Pinocchio's centroidal momentum map m h
   * = A_b(q) v_b + A_j(q) dq_j gives the base velocity v_b = A_b^-1 (m h - A_j
   * dq_j) (CentroidalMpcRobotModel): pass the measured joint velocities,
   * getJointVelocities() of the observation's state and input, which the MRT
   * writes. Zero joint velocities do not give the base twist but the twist of
   * the robot with its joints locked, w = I_G^-1 L, off from the base's by
   * I_G^-1 (A_j dq_j)_ang, the joints' angular momentum over the locked
   * inertia: on a walking humanoid more than the base's own yaw rate (design
   * README section 2.9). One Pinocchio pass on the centroidal model, so call it
   * once per solve, not per node. Double precision only.
   */
  virtual vector6_t getTwistBaseRtWorldEwrtWorld(
      const vector_t& state, const vector_t& jointVelocities) const = 0;

  /**
   * The command (v_x, v_y, yaw rate) a target knot `targetState` holds, as its
   * formulation's target calculator wrote it: the velocity in the world's
   * ground plane [m/s] and the turn about the world z axis [rad/s] (design
   * defect D3).
   *
   * A knot that stores the base velocity answers with its base twist, (v_x,
   * v_y, w_z) of getTwistBaseRtWorldEwrtWorld(). A centroidal knot stores the
   * normalized momentum of the whole body: its linear part is the CoM velocity,
   * and the turn is the angular momentum h_z = I_zz psidot / m of the body
   * turning about the vertical through its CoM, which comes back out as m h_z /
   * I_zz with `inertiaComEwrtWorldZz` = I_zz [kg m^2], the composite yaw
   * inertia the knot was built with. The centroidal target calculator builds
   * its knots at the measured configuration, whose inertia
   * SwitchedModelReferenceManager latches every solve; a model whose knots
   * store the base velocity does not read it, and a centroidal yaw rate reads 0
   * while it is not positive (none measured yet). Cheap: the reference code
   * calls it per node. Double precision only.
   */
  virtual vector3_t getRefPlanarMotion(
      const vector_t& targetState, scalar_t inertiaComEwrtWorldZz) const = 0;

  /** The entry of getRefPlanarMotion()'s (v_x, v_y, yaw rate) that holds
   * the yaw rate. */
  static constexpr int kRefPlanarMotionYawRateIndex = 2;

  /******************************************************************************************************/
  /*                                 Tuning layout and state values */
  /******************************************************************************************************/

  /**
   * The tuning layout of the task file's state blocks (initial_state,
   * state_weights, final_state_weights), placed on this model's state:
   * tuningLayoutOf() of the formulation, placed with
   * kQuaternionOrientationStateDim. Its getTuningLayoutDim() is
   * getStateTangentDim(), and its getTuningLayoutStateDim() getStateDim().
   */
  const TuningLayout& getTuningLayout() const { return tuningLayout_; }

  /**
   * The state of a tuning-layout vector, e.g. the task file's initial_state:
   * the one mapping of tuning values onto the state, stateFromTuningLayout() on
   * getTuningLayout() (orientation/EulerBoundary.h): (yaw, pitch, roll) become
   * the quaternion, the whole body's (w_z, w_y, w_x) the body angular velocity
   * (design decision D18), every other row is copied. InvalidArgument, naming
   * the tuning layout, when `tuningVector` does not have getStateTangentDim()
   * entries.
   */
  virtual absl::StatusOr<vector_t> stateFromTuning(
      const vector_t& tuningVector) const = 0;

  /**
   * The tuning-layout vector of `state`, the inverse of stateFromTuning()
   * (tuningLayoutFromState(), the Euler angles of the normalized quaternion).
   * InvalidArgument, naming the tuning layout, when `state` does not have
   * getStateDim() entries.
   */
  virtual absl::StatusOr<vector_t> tuningFromState(
      const vector_t& state) const = 0;

  /******************************************************************************************************/
  /*                                       Interpolation */
  /******************************************************************************************************/

  /** The desired state of `targets` at `time`: interpolateStateTrajectory() of
   * its knots. `targets` is not empty. */
  vector_t getDesiredState(const TargetTrajectories& targets,
                           scalar_t time) const {
    return interpolateStateTrajectory(time, targets.timeTrajectory,
                                      targets.stateTrajectory);
  }

  /**
   * The states `states` at the increasing `times`, interpolated at `time` in
   * this layout and held at the first and the last state outside the times:
   * every row linearly (LinearInterpolation,
   * TargetTrajectories::getDesiredState bit for bit), but the orientation,
   * which is slerped along the shorter way (ocs2::quaternionSlerp, the state
   * manifold's interpolate() bit for bit). `times` and `states` have the same,
   * nonzero size.
   */
  virtual vector_t interpolateStateTrajectory(
      scalar_t time,
      const scalar_array_t& times,
      const vector_array_t& states) const = 0;

  /** The state at rest at the origin: zero positions, momenta, velocities and
   * joint angles, the identity orientation. */
  virtual vector_t getNeutralState() const = 0;

  /******************************************************************************************************/
  /*                                         Tangent maps */
  /******************************************************************************************************/

  /**
   * d(body rotation tangent) / d(the orientation slots): the 3 x 4 matrix J
   * with R(x + dx) = R(x) Exp(J dx_orientation) to first order, E^+(xi) = 2
   * G^T(xi) / |xi|^2 (configurationTangentLift(),
   * orientation/BaseOrientation.h).
   */
  virtual matrix_t getBaseRotationTangentJacobian(
      const vector_t& state) const = 0;

  /**
   * d(Pinocchio configuration tangent) / d(state), getVelocityDim() x
   * getStateDim(): how a state step moves q in Pinocchio's tangent, for the
   * terms that take Pinocchio's Jacobians to the state: the identity on the
   * position and the joint slots, E^+(xi) (getBaseRotationTangentJacobian()) on
   * the quaternion's.
   */
  virtual matrix_t getPinocchioTangentJacobian(const vector_t& state) const = 0;

  /******************************************************************************************************/
  /*                                Realtime measurement helpers */
  /******************************************************************************************************/
  /*
   * For the control thread: they write into the caller's preallocated buffers
   * and allocate, check, log and throw nothing. The sizes are the caller's
   * precondition, unchecked: `jointPositions` and `jointVelocities` have
   * getJointDim() entries, `q` getConfigurationDim() and `v` getVelocityDim().
   * `quatBaseRtWorldXyzw` is the measured base orientation, the quaternion
   * coefficients (x, y, z, w) of the rotation from the base frame to the world,
   * read through its safe normalization (a measured quaternion is unit to
   * round-off; one shorter than the guard is kept as it is).
   */

  /**
   * Writes the Pinocchio configuration q = [posBaseRtWorldEwrtWorld, the
   * normalized orientation, jointPositions] of the measured base position, base
   * orientation and joint positions into `q`.
   */
  virtual void fillGeneralizedCoordinatesFromMeasurement(
      const vector3_t& posBaseRtWorldEwrtWorld,
      const vector4_t& quatBaseRtWorldXyzw,
      const vector_t& jointPositions,
      Eigen::Ref<vector_t> q) const = 0;

  /**
   * Writes the Pinocchio velocity v = [R(xi_hat) linvelBaseRtWorldEwrtBase,
   * angvelBaseRtWorldEwrtBase, jointVelocities] of the measured base
   * velocities, both in the base frame, and joint velocities into `v`: the
   * world-frame linear velocity and the body angular velocity of the
   * Composite(Translation, Spherical) root.
   */
  virtual void fillGeneralizedVelocitiesFromMeasurement(
      const vector4_t& quatBaseRtWorldXyzw,
      const vector3_t& linvelBaseRtWorldEwrtBase,
      const vector3_t& angvelBaseRtWorldEwrtBase,
      const vector_t& jointVelocities,
      Eigen::Ref<vector_t> v) const = 0;

  /******************************************************************************************************/
  /*                                          Contacts */
  /******************************************************************************************************/
  /**
   * Input-only contact wrench accessors.
   *
   * These read/write the contact wrench *as it is parameterized in the input
   * vector*. For wrench-space models (CentroidalMpcRobotModel,
   * WBAccelMpcRobotModel) the input stores the wrench in the world frame, so
   * these accessors return world-frame quantities. For the basis-vector
   * decorator (BasisInputsModelDecorator) the input stores non-negative
   * scalings of a *local contact-frame* basis, so these accessors return the
   * wrench in the local contact frame.
   *
   * Whenever a world-frame wrench is required, prefer the state-aware
   * `...ContactEwrtWorld` accessors below, which are frame-correct for every
   * model.
   */
  virtual VECTOR6_T<SCALAR_T> getWrenchContactEwrtParam(
      const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const = 0;

  /** Returns the frame getWrenchContactEwrtParam() is expressed in: world axes
   * unless a model overrides it. */
  virtual ContactInputFrame getContactInputFrame(
      size_t /*contactIndex*/) const {
    return ContactInputFrame::kWorld;
  }

  virtual VECTOR3_T<SCALAR_T> getForceContactEwrtParam(
      const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const = 0;

  virtual VECTOR3_T<SCALAR_T> getTorqueContactEwrtParam(
      const VECTOR_T<SCALAR_T>& input, size_t contactIndex) const = 0;

  /******************************************************************************************************/

  virtual void setWrenchContactEwrtParam(
      VECTOR_T<SCALAR_T>& input,
      const VECTOR6_T<SCALAR_T>& wrenchContactEwrtParam,
      size_t contactIndex) const = 0;

  virtual void setForceContactEwrtParam(
      VECTOR_T<SCALAR_T>& input,
      const VECTOR3_T<SCALAR_T>& forceContactEwrtParam,
      size_t contactIndex) const = 0;

  virtual void setTorqueContactEwrtParam(
      VECTOR_T<SCALAR_T>& input,
      const VECTOR3_T<SCALAR_T>& torqueContactEwrtParam,
      size_t contactIndex) const = 0;

  /******************************************************************************************************/
  /*                              State-aware world-frame contact accessors */
  /******************************************************************************************************/

  /**
   * Returns the contact wrench [f, tau] expressed in the world (inertial)
   * frame.
   *
   * The state is required because some input parameterizations (basis-vector
   * scalings) are defined in the local contact frame, whose orientation depends
   * on the robot configuration. The default implementation assumes the input
   * already stores a world-frame wrench and ignores the state.
   */
  virtual VECTOR6_T<SCALAR_T> getWrenchContactEwrtWorld(
      const VECTOR_T<SCALAR_T>& /*state*/,
      const VECTOR_T<SCALAR_T>& input,
      size_t contactIndex) const {
    return getWrenchContactEwrtParam(input, contactIndex);
  }

  virtual VECTOR3_T<SCALAR_T> getForceContactEwrtWorld(
      const VECTOR_T<SCALAR_T>& /*state*/,
      const VECTOR_T<SCALAR_T>& input,
      size_t contactIndex) const {
    return getForceContactEwrtParam(input, contactIndex);
  }

  virtual VECTOR3_T<SCALAR_T> getTorqueContactEwrtWorld(
      const VECTOR_T<SCALAR_T>& /*state*/,
      const VECTOR_T<SCALAR_T>& input,
      size_t contactIndex) const {
    return getTorqueContactEwrtParam(input, contactIndex);
  }

  /**
   * Returns getWrenchContactEwrtWorld(state, input, contactIndex) with
   * `rotmatContactRtWorld`, the orientation (contact frame to world) of the
   * contact's frame at the state, given instead of computed from the state: for
   * the realtime controllers, which read it off kinematics of their own, and
   * must not allocate - nor does this, for any model. The default ignores the
   * rotation: the input stores a world-frame wrench.
   */
  virtual VECTOR6_T<SCALAR_T> getWrenchContactEwrtWorld(
      const VECTOR_T<SCALAR_T>& input,
      const MATRIX3_T<SCALAR_T>& /*rotmatContactRtWorld*/,
      size_t contactIndex) const {
    return getWrenchContactEwrtParam(input, contactIndex);
  }

  /**
   * Writes a world-frame contact wrench into the input vector, converting into
   * the input parameterization (and local contact frame) where necessary.
   */
  virtual void setWrenchContactEwrtWorld(
      const VECTOR_T<SCALAR_T>& /*state*/,
      VECTOR_T<SCALAR_T>& input,
      const VECTOR6_T<SCALAR_T>& wrenchContactEwrtWorld,
      size_t contactIndex) const {
    setWrenchContactEwrtParam(input, wrenchContactEwrtWorld, contactIndex);
  }

  virtual void setForceContactEwrtWorld(
      const VECTOR_T<SCALAR_T>& /*state*/,
      VECTOR_T<SCALAR_T>& input,
      const VECTOR3_T<SCALAR_T>& forceContactEwrtWorld,
      size_t contactIndex) const {
    setForceContactEwrtParam(input, forceContactEwrtWorld, contactIndex);
  }

  /******************************************************************************************************/
  /*                                         Joint angle */
  /******************************************************************************************************/

  /** The index of `jointName` among the MPC joints, or NotFound when it is not
   * an active joint of the MPC model. */
  absl::StatusOr<size_t> findJointIndex(absl::string_view jointName) const {
    const absl::flat_hash_map<std::string, size_t>::const_iterator found =
        modelSettings.jointIndexMap.find(jointName);
    if (found == modelSettings.jointIndexMap.end()) {
      return absl::NotFoundError(absl::StrCat(
          "Joint name ", jointName, " is not contained in MPC model!"));
    }
    return found->second;
  }

 protected:
  // For the clone() of a derived model: every member copied, the reference to
  // the ModelSettings included.
  MpcRobotModelBase(const MpcRobotModelBase& rhs) = default;

  /**
   * interpolateStateTrajectory() of a layout whose orientation quaternion
   * starts at getBaseOrientationStartIndex(): LinearInterpolation of every row,
   * then the orientation slerped between the two states of the segment. Outside
   * the times, at a node and between states of different sizes the linear
   * result is kept, which is then one of the states.
   */
  vector_t interpolateWithSlerpedQuatBaseRtWorld(
      scalar_t time,
      const scalar_array_t& times,
      const vector_array_t& states) const {
    vector_t interpolated =
        LinearInterpolation::interpolate(time, times, states);
    if (states.size() <= 1) return interpolated;
    const LinearInterpolation::index_alpha_t indexAlpha =
        LinearInterpolation::timeSegment(time, times);
    const vector_t& left = states[static_cast<size_t>(indexAlpha.first)];
    const vector_t& right = states[static_cast<size_t>(indexAlpha.first) + 1];
    // LinearInterpolation's alpha weighs the left state: 1 at its time, 0 at
    // the right state's.
    if (left.size() != right.size() || indexAlpha.second >= 1.0 ||
        indexAlpha.second <= 0.0) {
      return interpolated;
    }
    const Eigen::Index start =
        static_cast<Eigen::Index>(getBaseOrientationStartIndex());
    const vector4_t leftQuatBaseRtWorldXyzw =
        left.segment<kQuaternionCoefficients>(start);
    const vector4_t rightQuatBaseRtWorldXyzw =
        right.segment<kQuaternionCoefficients>(start);
    interpolated.segment<kQuaternionCoefficients>(start) =
        quaternionSlerp<scalar_t>(leftQuatBaseRtWorldXyzw,
                                  rightQuatBaseRtWorldXyzw,
                                  1.0 - indexAlpha.second);
    return interpolated;
  }

 public:
  const ModelSettings& modelSettings;

  const size_t state_dim;
  const size_t input_dim;

 private:
  size_t configurationDim_ = 0;
  size_t velocityDim_ = 0;
  // getTuningLayoutDim(tuningLayout_).
  size_t stateTangentDim_ = 0;
  // Placed on the state of state_dim entries (checked by the constructor).
  TuningLayout tuningLayout_;
};

/**
 * Returns the center of mass's linear velocity as far as a state of `model`
 * tells it, for the consumers of a reduced model of the center of mass that run
 * on either formulation (the contact planners' LIP, the foothold heuristics,
 * the planned center-of-mass override): the CoM velocity a centroidal state
 * stores, or, on a state that stores the base origin's velocity instead (the
 * whole body), that velocity, which stands in for it (the proxy; the two differ
 * by the joints' share of the linear momentum).
 */
template <typename SCALAR_T>
VECTOR3_T<SCALAR_T> getProxyLinvelComRtWorldEwrtWorld(
    const MpcRobotModelBase<SCALAR_T>& model, const VECTOR_T<SCALAR_T>& state) {
  switch (model.getStoredLinvelPoint()) {
    case StoredLinvelPoint::kCenterOfMass:
      return model.getLinvelComRtWorldEwrtWorld(state);
    case StoredLinvelPoint::kBaseOrigin:
      return model.getLinvelBaseRtWorldEwrtWorld(state);
  }
  ABSL_LOG(FATAL) << "getProxyLinvelComRtWorldEwrtWorld: unknown "
                     "StoredLinvelPoint "
                  << static_cast<int>(model.getStoredLinvelPoint());
  ABSL_UNREACHABLE();
}

/**
 * Writes `proxyLinvelComRtWorldEwrtWorld` where
 * getProxyLinvelComRtWorldEwrtWorld() reads it: into the CoM velocity of a
 * centroidal state, into the base velocity of a state that stores that one
 * instead.
 */
template <typename SCALAR_T>
void setProxyLinvelComRtWorldEwrtWorld(
    const MpcRobotModelBase<SCALAR_T>& model,
    VECTOR_T<SCALAR_T>& state,
    const VECTOR3_T<SCALAR_T>& proxyLinvelComRtWorldEwrtWorld) {
  switch (model.getStoredLinvelPoint()) {
    case StoredLinvelPoint::kCenterOfMass:
      model.setLinvelComRtWorldEwrtWorld(state, proxyLinvelComRtWorldEwrtWorld);
      return;
    case StoredLinvelPoint::kBaseOrigin:
      model.setLinvelBaseRtWorldEwrtWorld(state,
                                          proxyLinvelComRtWorldEwrtWorld);
      return;
  }
  ABSL_LOG(FATAL) << "setProxyLinvelComRtWorldEwrtWorld: unknown "
                     "StoredLinvelPoint "
                  << static_cast<int>(model.getStoredLinvelPoint());
}

}  // namespace ocs2::humanoid
