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

#include "humanoid_common_mpc/HumanoidPreComputation.h"

#include <functional>
#include <optional>
#include <utility>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/misc/Numerics.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
HumanoidPreComputation::HumanoidPreComputation(
    PinocchioInterface pinocchioInterface,
    const SwingTrajectoryPlanner& swingTrajectoryPlanner,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : pinocchioInterface_(std::move(pinocchioInterface)),
      swingTrajectoryPlannerPtr_(&swingTrajectoryPlanner),
      mpcRobotModelPtr_(&mpcRobotModel),
      positionErrorGainZ_(mpcRobotModel.modelSettings.footConstraintConfig
                              .positionErrorGain_z) {
  // The swing planner plans the biped's two feet: a contact set other than
  // theirs has no swing references here, and its terms read the contact's
  // patch instead (ZeroVelocityConstraintCppAd).
  const size_t numPlannedFeet =
      mpcRobotModel.modelSettings.isBipedContactSet() ? kNumContacts : 0;
  eeNormalVelConConfigs_.resize(numPlannedFeet);
  footHeightReferences_.resize(numPlannedFeet);
  for (size_t i = 0; i < numPlannedFeet; ++i) {
    footHeightReferences_[i] = 0.0;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

HumanoidPreComputation::HumanoidPreComputation(
    const HumanoidPreComputation& rhs)
    : pinocchioInterface_(rhs.pinocchioInterface_),
      swingTrajectoryPlannerPtr_(rhs.swingTrajectoryPlannerPtr_),
      mpcRobotModelPtr_(rhs.mpcRobotModelPtr_),
      eeNormalVelConConfigs_(rhs.eeNormalVelConConfigs_),
      footHeightReferences_(rhs.footHeightReferences_),
      contactSwingPlannerPtr_(rhs.contactSwingPlannerPtr_),
      contactSwingReferences_(rhs.contactSwingReferences_),
      // Carried across the clone the SQP solver makes per worker thread;
      // dropping it would reset every worker's gain to the launch value on the
      // next retune, which is the class of bug this member exists to close.
      positionErrorGainZ_(rhs.positionErrorGainZ_) {}

void HumanoidPreComputation::setContactSwingPlanner(
    const ContactSwingPlanner* absl_nullable contactSwingPlanner) {
  contactSwingPlannerPtr_ = contactSwingPlanner;
  contactSwingReferences_.assign(
      contactSwingPlanner != nullptr
          ? mpcRobotModelPtr_->modelSettings.contactNames.size()
          : 0,
      std::nullopt);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
HumanoidPreComputation* absl_nonnull HumanoidPreComputation::clone() const {
  return new HumanoidPreComputation(*this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void HumanoidPreComputation::updatePinocchioModelKinematics(const vector_t& q) {
  const pinocchio::Model& model = pinocchioInterface_.getModel();
  pinocchio::Data& data = pinocchioInterface_.getData();

  // Perform the forward kinematics over the kinematic tree
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void HumanoidPreComputation::request(RequestSet request,
                                     scalar_t t,
                                     const vector_t& x,
                                     const vector_t& /*u*/) {
  if (!request.containsAny(Request::Cost + Request::Constraint +
                           Request::SoftConstraint)) {
    return;
  }

  updatePinocchioModelKinematics(
      mpcRobotModelPtr_->getGeneralizedCoordinates(x));

  // lambda to set config for normal velocity constraints
  const std::function<EndEffectorKinematicsLinearVelConstraint::Config(size_t)>
      eeNormalVelConConfig = [&](size_t footIndex) {
        EndEffectorKinematicsLinearVelConstraint::Config config;
        config.b = (vector_t(1)
                    << -swingTrajectoryPlannerPtr_
                            ->getDesLinvelFootRtWorldEwrtWorldZ(footIndex, t))
                       .finished();
        config.Av = (matrix_t(1, kPositionDim) << 0.0, 0.0, 1.0).finished();
        // The gain comes from this object, not from the shared ModelSettings,
        // so that the parameter updater can retune it per worker thread while
        // the controller runs; see setNormalVelocityPositionErrorGain.
        if (!numerics::almost_eq(positionErrorGainZ_, /*y=*/0.0)) {
          config.b(0) -=
              positionErrorGainZ_ *
              swingTrajectoryPlannerPtr_->getDesPosFootRtWorldEwrtWorldZ(
                  footIndex, t);
          config.Ax =
              (matrix_t(1, kPositionDim) << 0.0, 0.0, positionErrorGainZ_)
                  .finished();
        }
        return config;
      };

  // Widened from `contains(Request::Constraint)` deliberately. Today every OCS2
  // call site that evaluates a state-input term - setupIntermediateNode,
  // LinearQuadraticApproximator, MetricsComputation - requests
  // Request::Constraint unconditionally as a `constexpr`, so the narrower gate
  // was not actually leaving anything stale; the paths that omit it,
  // setupTerminalNode and setupEventNode, go through requestFinal /
  // requestPreJump, which this class does not override and where no state-input
  // term is evaluated.
  //
  // It is widened anyway because the coupling is invisible and the formulation
  // moved underneath it: the soft normal-velocity servo is a SOFT constraint
  // that reads eeNormalVelConConfigs_, so a caller that one day asks for soft
  // constraints alone would silently evaluate it against a stale swing
  // reference rather than fail. One extra flag in the guard removes that trap
  // for the cost of two swing references on a cost-only request.
  if (request.containsAny(Request::Constraint + Request::SoftConstraint)) {
    for (size_t i = 0; i < footHeightReferences_.size(); ++i) {
      eeNormalVelConConfigs_[i] = eeNormalVelConConfig(i);
      footHeightReferences_[i] =
          swingTrajectoryPlannerPtr_->getDesPosFootRtWorldEwrtWorldZ(i, t);
    }
    // The generalized swing planner's references of every contact, read by
    // the contact_swing terms (ContactSwingConstraint).
    if (contactSwingPlannerPtr_ != nullptr) {
      for (size_t i = 0; i < contactSwingReferences_.size(); ++i) {
        contactSwingReferences_[i] = contactSwingPlannerPtr_->reference(i, t);
      }
    }
  }
}

}  // namespace ocs2::humanoid
