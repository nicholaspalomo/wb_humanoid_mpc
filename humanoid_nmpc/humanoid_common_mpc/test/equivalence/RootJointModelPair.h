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

// Pinocchio forward declarations must be included first.
#include "pinocchio/fwd.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"

/*
 * The Pinocchio-oracle equivalence suite of the quaternion base orientation
 * (humanoid_nmpc/docs/ quaternion_base_orientation/README.md, sections 2.2
 * and 2.12): every robot's MPC model built twice from the same reduced URDF,
 * once on the quaternion root Composite(Translation, Spherical) of the MPCs and
 * once on the Euler root Composite(Translation, SphericalZYX), so that
 * Pinocchio itself, through two independent joint models, is the oracle for
 * "the same physics in two coordinates". Nothing of the MPCs' own Euler code is
 * duplicated: the chart between the two coordinates is EulerChart's
 * (humanoid_mpc_validation/chart/EulerChart.h) on the whole-body layout, built
 * on orientation/EulerBoundary.h, and the suite checks that chart against
 * Pinocchio as well.
 *
 * Coordinates (Pinocchio 4.1, joint-spherical-ZYX.hxx, joint-spherical.hpp,
 * joint-translation.hpp):
 *
 *   Euler root       q_e = [p_W, Theta, q_j],  Theta = (psi, theta, phi) =
 * (yaw, pitch, roll), R = R_z R_y R_x; v_e = [pd_W, dTheta/dt, qd_j],  a_e =
 * dv_e/dt. Quaternion root  q   = [p_W, xi, q_j],  xi = (x, y, z, w);  v =
 * [pd_W, w_B, qd_j],  a = dv/dt = [pdd_W, dw_B/dt, qdd_j].
 *
 * The translation joint's velocity is the time derivative of p in the parent
 * (world) frame, so pd_W is the same in both. SphericalZYX's motion subspace is
 * S(Theta) = T_B(Theta) (its angular rows; the body angular velocity of the
 * Euler rates, eulerZyxRateToBodyRateMatrix), Spherical's is the
 * identity on w_B.
 *
 * The chart and its derivatives (derived once here; each test cites the line it
 * uses):
 *
 *   (C1) configuration   q = Phi_q(q_e) = [p_W, xi(Theta), q_j],  xi(Theta) =
 * q_z(psi) (x) q_y(theta) (x) q_x(phi) (quaternionFromEulerZyx), so
 * R(xi(Theta)) = R_z(psi) R_y(theta) R_x(phi): the same root placement. (C2)
 * velocity        v = J_v(Theta) v_e,  J_v = blockdiag(I_3, T_B(Theta), I_nj):
 * differentiating R(t) = R(Theta(t)) gives R^T dR/dt = [T_B(Theta) dTheta/dt]x,
 * the body angular velocity; p and q_j are the same coordinates in both. (C3)
 * acceleration    a = J_v a_e + (dJ_v/dt) v_e,  dJ_v/dt = blockdiag(0, dT_B/dt,
 * 0), dT_B/dt dTheta = [d(T_B(Theta) r)/dTheta]_{r = dTheta} dTheta: the time
 * derivative of (C2). Column i of dT_B/dt is [d(T_B e_i)/dTheta] dTheta/dt
 * (bodyRateEulerZyxJacobian). In EulerChart's terms, with the
 * whole-body Euler state x_e = [q_e; v_e] and its physical-tangent Jacobian P_e
 * (EulerChart::physicalTangentJacobian), [v; a] = P_e(x_e) [v_e; a_e]: the top
 * block row of P_e is (C2) and the bottom one (C3). (C4) forces          a
 * generalized force is the dual of a generalized velocity: the power tau^T v of
 * the same force on the same motion is the same in both coordinates, tau_e^T
 * v_e = tau^T J_v v_e for every v_e, so tau_e = J_v^T tau: the base force and
 * the joint torques are equal, the Euler base "torque" is T_B^T times the body
 * moment. (C5) Jacobians       a map whose rate is linear in the velocity, d y
 * / dt = J(q) v (a frame twist, the CoM velocity, the centroidal momentum h =
 * A_G v), gives J_e = J J_v (C2). (C6) mass matrix     the kinetic energy 1/2
 * v^T M v is the same scalar, so M_e = J_v^T M J_v; with the equations of
 *                        motion M a + n = tau, (C3) and (C4): n_e = J_v^T (n +
 * M (dJ_v/dt) v_e) and g_e = J_v^T g (gravity has no velocity term). (C7)
 * momentum rate   d/dt (A_G,e) = d/dt (A_G J_v) = (d A_G / dt) J_v + A_G
 * dJ_v/dt, and h_dot = A_G a + (dA_G/dt) v in both.
 *
 * Tolerances are derived where they are declared, from round-off: both models
 * evaluate the same sequence of rigid-body operations after the root, which
 * differ only in how R is formed (trigonometric products against the
 * quaternion's quadratic form, each entry within a few units of round-off) and
 * how the root velocity is formed (T_B dTheta against w_B); a tree of depth d
 * then carries that difference through d compositions. The suite's labeled
 * bounds, each derived at its declaration:
 *
 *   (T1) round-off of two evaluations of one chain of transforms, 1e-12
 * relative (floor 1): testRootJointKinematics Equivalence.cpp,
 * kKinematicsTolerance; the dynamics, centroidal and root-only tests reuse it;
 *   (T2) the central difference of the configuration map, step 1e-5, 1e-7:
 * testRootJointKinematicsEquivalence.cpp; (T3) Richardson's d A_G / dt, step
 * 1e-3, 1e-8 relative: testRootJointCentroidalEquivalence.cpp; (T4) forward
 * dynamics, (T1) times the mass matrices' condition numbers, and the residual
 * bound of rnea: testRootJointDynamicsEquivalence.cpp,
 * forwardDynamicsTolerance() and inverseDynamicsResidualBound(); (T5) the Euler
 * boundary's inverse on the placements, 1e-11: testQuaternionRootOnlyCases.cpp.
 */

namespace ocs2::humanoid {

/** One shipped MPC configuration, by its files relative to the repository root
 * (string literals). Passive data. */
struct EquivalenceRobot {
  /** The configuration's package, e.g. "g1_wb_mpc": names the test case. */
  const char* absl_nonnull name;
  /** Its task file: which URDF joints the MPC model keeps
   * (ModelSettings::mpcModelJointNames). */
  const char* absl_nonnull taskFile;
  /** Its URDF. */
  const char* absl_nonnull urdfFile;
};

/** Every shipped MPC configuration: the five the suite runs on. */
absl::Span<const EquivalenceRobot> equivalenceRobots();

/** The test-case name of a robot: its package name. */
std::string equivalenceRobotName(
    const ::testing::TestParamInfo<EquivalenceRobot>& info);

/**
 * The MPC's Pinocchio model of one robot built twice from the same reduced URDF
 * (every joint the task file does not keep fixed, as
 * loadCustomPinocchioInterface() reduces it): on the quaternion root and on the
 * Euler root. The two have the same joints, bodies, frames and inertias in the
 * same order; only the root joint differs.
 *
 * Build it with Create(); it is immutable afterwards, so its const methods are
 * safe to call concurrently.
 */
class RootJointModelPair {
 public:
  /**
   * Both models of `robot`, from its files in the test's runfiles.
   * InvalidArgument when a file is missing, the task file or URDF does not
   * load, or the two models do not have the same joints (which would make every
   * comparison meaningless).
   */
  static absl::StatusOr<std::unique_ptr<RootJointModelPair>> Create(
      const EquivalenceRobot& robot);

  RootJointModelPair(const RootJointModelPair&) = delete;
  RootJointModelPair& operator=(const RootJointModelPair&) = delete;
  RootJointModelPair(RootJointModelPair&&) = delete;
  RootJointModelPair& operator=(RootJointModelPair&&) = delete;
  ~RootJointModelPair() = default;

  /** Root Composite(Translation, Spherical): q = [p_W, xi, q_j], v = [pd_W,
   * w_B, qd_j]. */
  const pinocchio::Model& quaternionModel() const { return quaternionModel_; }

  /** Root Composite(Translation, SphericalZYX): q_e = [p_W, (yaw, pitch, roll),
   * q_j], v_e = [pd_W, Euler rates, qd_j]. */
  const pinocchio::Model& eulerModel() const { return eulerModel_; }

  /** The whole-body Euler state layout, which is [q_e; v_e] block for block:
   * the chart (C1)-(C3) of both models. */
  const validation::EulerStateLayout& pinocchioStateLayout() const {
    return pinocchioStateLayout_;
  }

  /** The centroidal Euler state layout [h / m, p_W, (yaw, pitch, roll), q_j] of
   * the same joints. */
  const validation::EulerStateLayout& centroidalStateLayout() const {
    return centroidalStateLayout_;
  }

  /** The number of actuated joints n_j: nv - 6 in both models. */
  Eigen::Index numJoints() const { return eulerModel_.nv - 6; }

 private:
  // pinocchio::Model has no move constructor, so the models are copied in.
  RootJointModelPair(const pinocchio::Model& quaternionModel,
                     const pinocchio::Model& eulerModel,
                     validation::EulerStateLayout pinocchioStateLayout,
                     validation::EulerStateLayout centroidalStateLayout);

  pinocchio::Model quaternionModel_;
  pinocchio::Model eulerModel_;
  validation::EulerStateLayout pinocchioStateLayout_;
  validation::EulerStateLayout centroidalStateLayout_;
};

/** One motion of a robot in one model's coordinates: configuration, velocity
 * and acceleration. Passive data. */
struct PinocchioMotion {
  vector_t q;
  vector_t v;
  vector_t a;
};

/** The chart's velocity maps at one Euler configuration and velocity. Passive
 * data. */
struct ChartVelocityMaps {
  /** J_v(Theta) = blockdiag(I_3, T_B(Theta), I_nj), nv x nv (C2). */
  matrix_t velocityMap;
  /** dJ_v/dt = blockdiag(0, dT_B/dt, 0) along the Euler velocity (C3), nv x nv.
   */
  matrix_t velocityMapRate;
};

/**
 * The motion `eulerMotion` of the Euler model in the quaternion model's
 * coordinates: q and v through EulerChart's Phi on [q_e; v_e] (C1, C2), a = the
 * bottom block row of P_e(x_e) [v_e; a_e] (C3). Fails like
 * validation::physicalStateFromEulerState() when the vectors do not fit
 * `layout`.
 */
absl::StatusOr<PinocchioMotion> quaternionMotionOf(
    const PinocchioMotion& eulerMotion,
    const validation::EulerStateLayout& layout);

/**
 * J_v and dJ_v/dt at `eulerMotion`'s configuration and velocity: J_v is the
 * top-left block of EulerChart's P_e (C2), dJ_v/dt is built column by column
 * from EulerBoundary's bodyRateEulerZyxJacobian (C3). Fails like
 * validation::physicalTangentJacobian().
 */
absl::StatusOr<ChartVelocityMaps> chartVelocityMapsAt(
    const PinocchioMotion& eulerMotion,
    const validation::EulerStateLayout& layout);

/** Where the random motions are drawn. Passive data. */
struct MotionSamplingBounds {
  /** |yaw| <= pi and |roll| <= pi always; |pitch| <= maxAbsPitch, away from the
   * Euler root's singularity at 90 degrees. */
  scalar_t maxAbsPitch = 80.0 / 180.0 * M_PI;
  /** |base position| per axis (the height is drawn in [0.3, 0.3 + 2
   * maxAbsBasePosition]). */
  scalar_t maxAbsBasePosition = 1.5;
  /** |base linear velocity| per axis (m/s). */
  scalar_t maxAbsBaseLinearVelocity = 2.0;
  /** |Euler rate| per axis (rad/s). */
  scalar_t maxAbsEulerRate = 3.0;
  /** |joint velocity| per joint (rad/s). */
  scalar_t maxAbsJointVelocity = 5.0;
  /** |base linear acceleration| per axis (m/s^2). */
  scalar_t maxAbsBaseLinearAcceleration = 10.0;
  /** |Euler acceleration| per axis (rad/s^2). */
  scalar_t maxAbsEulerAcceleration = 20.0;
  /** |joint acceleration| per joint (rad/s^2). */
  scalar_t maxAbsJointAcceleration = 50.0;
};

/**
 * A random motion of the Euler model: yaw and roll uniform in [-pi, pi], pitch
 * uniform in [-maxAbsPitch, maxAbsPitch], joint positions uniform in the
 * model's joint limits (in [-pi, pi] for a joint without finite ones),
 * velocities and accelerations uniform in `bounds`. Deterministic for a given
 * generator state.
 */
PinocchioMotion randomEulerMotion(const pinocchio::Model& eulerModel,
                                  const MotionSamplingBounds& bounds,
                                  std::mt19937& generator);

/** A vector of `size` entries uniform in [-halfWidth, halfWidth]. */
vector_t uniformVector(Eigen::Index size,
                       scalar_t halfWidth,
                       std::mt19937& generator);

/** max |actual - expected| <= relativeTolerance * max(1, max |expected|), with
 * the worst entry in the message. */
::testing::AssertionResult isCloseRelative(const matrix_t& actual,
                                           const matrix_t& expected,
                                           scalar_t relativeTolerance);

/** max |actual - expected| <= absoluteTolerance, with the worst entry in the
 * message. */
::testing::AssertionResult isCloseAbsolute(const matrix_t& actual,
                                           const matrix_t& expected,
                                           scalar_t absoluteTolerance);

/** max |m|, 0 for an empty matrix. */
scalar_t maxAbs(const matrix_t& m);

/** The symmetric matrix whose upper triangle is that of `upper`:
 * pinocchio::crba fills only the upper triangle. */
matrix_t symmetricFromUpper(const matrix_t& upper);

}  // namespace ocs2::humanoid
