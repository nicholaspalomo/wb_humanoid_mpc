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

#include "pinocchio/fwd.hpp"

#include "humanoid_wb_mpc/WBMpcPreComputation.h"

#include <utility>

#include "absl/base/nullability.h"
#include "ocs2_core/misc/Numerics.h"
#include "pinocchio/algorithm/kinematics.hpp"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
WBMpcPreComputation::WBMpcPreComputation(PinocchioInterface pinocchioInterface,
                                         const SwingTrajectoryPlanner& swingTrajectoryPlanner,
                                         const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : HumanoidPreComputation(std::move(pinocchioInterface), swingTrajectoryPlanner, mpcRobotModel),
      swingFootGains_{.linearVelocityErrorGainZ = mpcRobotModel.modelSettings.footConstraintConfig.linearVelocityErrorGain_z,
                      .linearAccelerationErrorGainZ = mpcRobotModel.modelSettings.footConstraintConfig.linearAccelerationErrorGain_z} {
  eeNormalAccConConfigs_.resize(kNumContacts);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

WBMpcPreComputation::WBMpcPreComputation(const WBMpcPreComputation& rhs) = default;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
WBMpcPreComputation* absl_nonnull WBMpcPreComputation::clone() const {
  return new WBMpcPreComputation(*this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
EndEffectorDynamicsLinearAccConstraint::Config WBMpcPreComputation::normalAccelerationConstraintConfig(size_t footIndex, scalar_t t) const {
  // v_z - zdot_ref + k_a (a_z - zddot_ref) + k_p (z - z_ref), the position term left out when its gain is 0.
  EndEffectorDynamicsLinearAccConstraint::Config config;
  config.b = (vector_t(1) << -swingFootGains_.linearVelocityErrorGainZ * swingTrajectoryPlannerPtr_->getZvelocityConstraint(footIndex, t))
                 .finished();
  config.Av = (matrix_t(1, 3) << 0.0, 0.0, swingFootGains_.linearVelocityErrorGainZ).finished();
  config.b(0) -= swingFootGains_.linearAccelerationErrorGainZ * swingTrajectoryPlannerPtr_->getZaccelerationConstraint(footIndex, t);
  config.Aa = (matrix_t(1, 3) << 0.0, 0.0, swingFootGains_.linearAccelerationErrorGainZ).finished();
  if (!numerics::almost_eq(positionErrorGainZ_, /*y=*/0.0)) {
    config.b(0) -= positionErrorGainZ_ * swingTrajectoryPlannerPtr_->getZpositionConstraint(footIndex, t);
    config.Ax = (matrix_t(1, 3) << 0.0, 0.0, positionErrorGainZ_).finished();
  }
  return config;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void WBMpcPreComputation::request(RequestSet request, scalar_t t, const vector_t& x, const vector_t& /*u*/) {
  if (!request.containsAny(Request::Cost + Request::Constraint + Request::SoftConstraint)) {
    return;
  }

  updatePinocchioModelKinematics(mpcRobotModelPtr_->getGeneralizedCoordinates(x));

  if (request.contains(Request::Constraint)) {
    for (size_t i = 0; i < kNumContacts; ++i) {
      eeNormalAccConConfigs_[i] = normalAccelerationConstraintConfig(i, t);
    }
  }
}

}  // namespace ocs2::humanoid
