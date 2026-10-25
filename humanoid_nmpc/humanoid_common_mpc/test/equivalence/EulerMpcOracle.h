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

#include <array>
#include <memory>
#include <string>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/Types.h"

/*
 * The Euler side of the MPC-level equivalence suite
 * (humanoid_nmpc/docs/quaternion_base_orientation/README.md, sections 2.5
 * and 2.12): the flow maps of both MPC formulations written in ZYX Euler
 * coordinates of the base orientation, on the Euler-root twin of the MPC's own
 * Pinocchio model (every joint, body, inertia and frame of it, the root joint
 * Composite(Translation, SphericalZYX) instead of Composite(Translation,
 * Spherical)), with Pinocchio as the only physics. Nothing of the production
 * flow maps is called: the momentum rate, the base velocity of the centroidal
 * momentum and the floating-base rows of the equations of motion are written
 * here again from their definitions, so that the suite compares two derivations
 * of the same physics, in two coordinates.
 *
 * Euler states (the tuning layout of the task file row for row, EulerChart's
 * EulerStateLayout):
 *
 *   centroidal   x_e = [hbar (6), p_W (3), Theta (3), q_j]       hbar = [m
 * c_dot, L_G] / m in the world frame whole body   x_e = [p_W (3), Theta (3),
 * q_j, pd_W (3), dTheta/dt (3), qd_j]
 *
 * with Theta = (yaw, pitch, roll), R = R_z R_y R_x, the SphericalZYX joint's
 * own coordinates and rates.
 *
 * Flow maps, with q_e = [p_W, Theta, q_j] and the contact wrenches W_i = [f_i;
 * tau_i] in the world frame:
 *
 *   (O1) centroidal momentum rate  hbar_dot = [g + sum_i f_i / m; sum_i ((r_i -
 * c) x f_i + tau_i) / m]: Newton and Euler about the center of mass c of the
 * contact forces at the contact frames' origins r_i and gravity g; (O2)
 * centroidal base velocity  [pd_W; dTheta/dt] = A_b,e^-1 (m hbar - A_j,e qd_j),
 * A_G,e(q_e) = [A_b,e, A_j,e] the centroidal momentum matrix of the Euler model
 * (h = A_G,e v_e): the base velocity whose momentum, with the joint rates of
 * the input, is the state's; (O3) whole-body base acceleration  a_b,e = M_bb^-1
 * (sum_i J_i,b^T W_i - n_b - M_bj qdd_j): the six floating-base rows of M_e a_e
 * + n_e = J_e^T W, with J_i the LOCAL_WORLD_ALIGNED Jacobian of contact frame i
 * (world-frame wrench at the frame origin); (O4) centroidal  f_e = [hbar_dot;
 * pd_W; dTheta/dt; qd_j],  whole body  f_e = [pd_W; dTheta/dt; qd_j; a_b,e;
 * qdd_j].
 *
 * A contact input is a world-frame wrench, or the scalings lambda of a local
 * basis: W_i = blockdiag(R_i, R_i) B_i lambda_i with R_i the contact frame's
 * rotation in the Euler model.
 *
 * The gravity of the oracle is a parameter: the dynamics of a robot pitched by
 * 90 degrees is that of the level robot under a rotated gravity, which is how
 * the suite reaches the Euler root's singularity (testMpcDynamicsEquivalence).
 */

namespace ocs2::humanoid {

/** Which MPC formulation an oracle evaluates. */
enum class OracleFormulation {
  kCentroidal,
  kWholeBody,
};

/** What the contact entries of an MPC input are. */
enum class OracleContactInput {
  /** [f (3), tau (3)] per contact, in the world frame. */
  kWorldWrench,
  /** The scalings lambda of contact i's local basis B_i (6 x n): W_i,local =
     B_i lambda_i. */
  kLocalBasis,
};

/** What an EulerMpcOracle evaluates. Passive data. */
struct EulerMpcOracleOptions {
  OracleFormulation formulation = OracleFormulation::kCentroidal;
  /** The contact frames, left then right: the frames the MPC applies the
   * contact wrenches at. */
  std::array<std::string, kNumContacts> contactFrameNames;
  OracleContactInput contactInput = OracleContactInput::kWorldWrench;
  /** For kLocalBasis: each contact's 6 x n basis; ignored otherwise. */
  std::array<matrix_t, kNumContacts> localBasisMatrices;
  /** [m/s^2] in the world frame. */
  vector3_t gravity = vector3_t(0.0, 0.0, -9.81);
};

/**
 * The model `quaternionRootModel` with its root joint, Composite(Translation,
 * Spherical), replaced by Composite(Translation, SphericalZYX): every other
 * joint, every inertia (as stored, so a mass scaling stays), every frame and
 * the joint position limits copied in their order. InvalidArgument when the
 * root is not a six-DoF joint of seven configuration entries.
 */
absl::StatusOr<pinocchio::Model> eulerRootTwinOf(
    const pinocchio::Model& quaternionRootModel);

/**
 * The Euler-coordinate flow maps (O1)-(O4) of one MPC configuration. Build it
 * with Create(); it is immutable afterwards and every evaluation works on its
 * own Pinocchio data, so its const methods are safe to call concurrently.
 */
class EulerMpcOracle {
 public:
  /**
   * The oracle of the MPC whose Pinocchio model is `quaternionRootModel` (its
   * Euler-root twin, eulerRootTwinOf()). InvalidArgument when the twin cannot
   * be built, a contact frame does not exist, or a basis is not 6 rows by the
   * same number of columns on both contacts; the whole body takes world-frame
   * wrenches only.
   */
  static absl::StatusOr<std::unique_ptr<EulerMpcOracle>> Create(
      const pinocchio::Model& quaternionRootModel,
      EulerMpcOracleOptions options);

  EulerMpcOracle(const EulerMpcOracle&) = delete;
  EulerMpcOracle& operator=(const EulerMpcOracle&) = delete;
  EulerMpcOracle(EulerMpcOracle&&) = delete;
  EulerMpcOracle& operator=(EulerMpcOracle&&) = delete;
  ~EulerMpcOracle() = default;

  /** The Euler-root twin the oracle evaluates. */
  const pinocchio::Model& eulerModel() const { return model_; }
  const EulerMpcOracleOptions& options() const { return options_; }
  /** n_j. */
  Eigen::Index jointDim() const { return model_.nv - 6; }
  /** Rows of an Euler state: 12 + n_j (centroidal), 12 + 2 n_j (whole body). */
  Eigen::Index stateDim() const;
  /** Rows of an input: 12 or 2 n (the contacts) + n_j. */
  Eigen::Index inputDim() const;
  /** [kg] The total mass m of the model. */
  scalar_t totalMass() const { return totalMass_; }

  /** f_e(x_e, u) (O4). The sizes are a precondition, checked with ABSL_CHECK: a
   * test's own defect. */
  vector_t flowMap(const vector_t& eulerState, const vector_t& input) const;

  /** The world-frame contact wrenches [f; tau] of `input` at `eulerState`, left
   * then right. */
  std::array<vector6_t, kNumContacts> worldWrenches(
      const vector_t& eulerState, const vector_t& input) const;

  /**
   * One classical fourth-order Runge-Kutta step of f_e over `dt` with the input
   * held: the node map OCS2's RK4 discretization makes of a flow map (the SQP's
   * integrator), here in Euler coordinates.
   */
  vector_t rk4Step(const vector_t& eulerState,
                   const vector_t& input,
                   scalar_t dt) const;

 private:
  // pinocchio::Model has no move constructor, so the model is copied in.
  EulerMpcOracle(const pinocchio::Model& model,
                 EulerMpcOracleOptions options,
                 std::array<pinocchio::FrameIndex, kNumContacts> contactFrames);

  /** q_e = [p_W, Theta, q_j] of an Euler state. */
  vector_t configurationOf(const vector_t& eulerState) const;

  pinocchio::Model model_;
  EulerMpcOracleOptions options_;
  std::array<pinocchio::FrameIndex, kNumContacts> contactFrames_;
  scalar_t totalMass_;
};

/** A finite-difference Jacobian and a bound on its error, entry by entry.
 * Passive data. */
struct FiniteDifferenceJacobian {
  /** The Richardson extrapolation (4 D(h/2) - D(h)) / 3 of the central
   * differences D. */
  matrix_t value;
  /**
   * |D(h) - D(h/2)| + 3 delta_f / h per entry: the first term is 3/4 of the h^2
   * truncation error of D(h) and so bounds the O(h^4) one of the extrapolation
   * by a wide margin; the second the extrapolation's rounding error, from an
   * evaluation error delta_f of kFunctionEvaluationRelativeError of the row's
   * largest magnitude per evaluation (floor 1; none for a row that is exactly
   * zero at every evaluation, whose differences are exact).
   */
  matrix_t errorBound;
};

/**
 * The relative evaluation error delta_f assumed of a function finite
 * differences are taken of. A calibrated value, not a derived one: the derived
 * ceiling - a Pinocchio pass and a few small dense solves, O(1e4) operations of
 * unit round-off 1.1e-16 at a condition of at most O(1e2), so about 1e-10 of
 * the largest intermediate - would let errors of O(1e-6) through the bound at
 * the steps used; 1e-12 of the row's magnitude is 100 times the evaluation
 * error measured on every shipped robot. The choice can only make a comparison
 * stricter than the derived ceiling would (a too small delta_f shows as a
 * failure, never as a pass), which is the safe side for an equivalence test.
 * Where the function is identically zero (an ambient gradient of 0), its
 * evaluations carry no error at all and the term vanishes: richardsonJacobian()
 * takes it relative to the row's own magnitude, floor 1 only where the row is
 * not zero.
 */
inline constexpr scalar_t kFunctionEvaluationRelativeError = 1.0e-12;

/**
 * d function / d x at `at` by Richardson-extrapolated central differences, with
 * steps h_i = relativeStep * max(1, |x_i|).
 */
FiniteDifferenceJacobian richardsonJacobian(
    absl::FunctionRef<vector_t(const vector_t&)> function,
    const vector_t& at,
    scalar_t relativeStep);

}  // namespace ocs2::humanoid
