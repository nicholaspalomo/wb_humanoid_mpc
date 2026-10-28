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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <cstddef>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_core/constraint/StateInputConstraintCppAd.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/multibody/model.hpp"

namespace ocs2::humanoid::multi_contact {

/** The effort limit of one actuated joint. Passive data. */
struct JointEffortLimit {
  // The joint's index in the Pinocchio velocity (6 + its actuated index).
  size_t velocityIndex = 0;
  // [N m] Positive.
  double effort = 0.0;
};

/**
 * The quasi-static joint torques of design section 2 within a fraction of
 * the effort limits, the torque condition the certifier checks:
 *
 *   tau(q, w) = g(q) - sum_i J_i(q)' w_i,
 *   margin - tau_j / effort_j >= 0,   margin + tau_j / effort_j >= 0,
 *
 * for every limited joint j, with w_i the wrench of each contact of the
 * centroidal MPC at its frame (world axes) and J_i the frame's Jacobian (its
 * linear rows for a 3-DoF contact). The inertial terms are left out: the
 * motion is quasi-static (the momentum-rate costs keep it so), and the
 * certifier judges the static torques. A contact out of contact has zero
 * force (zero_wrench), so the sum runs over all of them. Wrapped in a relaxed
 * barrier (StateInputSoftConstraint), a term of planner A's solve 3.
 *
 * Cloned per solver worker; a single instance is not thread-safe.
 */
class QuasiStaticTorqueConstraint final : public StateInputConstraintCppAd {
 public:
  /** The limits `limits` at `margin` (in (0, 1]) on the MPC's model
   * `pinocchio` (copied), contacts `contactFrames` in the MPC's order, taped
   * as `libraryName` in `folder`. */
  QuasiStaticTorqueConstraint(const PinocchioInterface& pinocchio,
                              const CentroidalModelInfo& info,
                              std::vector<std::string> contactFrames,
                              std::vector<JointEffortLimit> limits,
                              scalar_t margin,
                              const std::string& libraryName,
                              const std::string& folder,
                              bool recompile,
                              bool verbose);
  ~QuasiStaticTorqueConstraint() override = default;
  QuasiStaticTorqueConstraint* absl_nonnull clone() const override {
    return new QuasiStaticTorqueConstraint(*this);
  }
  QuasiStaticTorqueConstraint& operator=(const QuasiStaticTorqueConstraint&) =
      delete;
  QuasiStaticTorqueConstraint(QuasiStaticTorqueConstraint&&) = delete;
  QuasiStaticTorqueConstraint& operator=(QuasiStaticTorqueConstraint&&) =
      delete;

  size_t getNumConstraints(scalar_t time) const override;

  /** [N m] The quasi-static torques of the limited joints at (x, u), in the
   * order of the limits. */
  vector_t torques(const vector_t& state, const vector_t& input) const;

 private:
  QuasiStaticTorqueConstraint(const QuasiStaticTorqueConstraint& other) =
      default;

  ad_vector_t constraintFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) const override;
  /** tau of the limited joints on the tape. */
  ad_vector_t tapedTorques(const ad_vector_t& state,
                           const ad_vector_t& input) const;

  PinocchioInterfaceCppAd pinocchioCppAd_;
  CentroidalModelInfoCppAd infoCppAd_;
  std::vector<pinocchio::FrameIndex> contactFrames_;
  std::vector<JointEffortLimit> limits_;
  scalar_t margin_ = 0.0;
  // The double model, for torques().
  PinocchioInterface pinocchio_;
  CentroidalModelInfo info_;
};

}  // namespace ocs2::humanoid::multi_contact
