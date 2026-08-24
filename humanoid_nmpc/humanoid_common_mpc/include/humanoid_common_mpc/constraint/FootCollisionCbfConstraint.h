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

#include <pinocchio/fwd.hpp>

#include <string>
#include <vector>

#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * Implements a Zeroing Control Barrier Function (ZCBF) constraint:
 *   c(x, u) = \dot{h}(q, v) + \gamma h(q) >= 0
 * to enforce safe foot-to-foot and knee-to-knee separation during swing phases.
 */
class FootCollisionCbfConstraint final : public StateInputConstraintCppAd {
 public:
  struct Config {
    // Ankle joint frames
    std::string leftAnkleFrame;
    std::string rightAnkleFrame;

    // Foot center collision frames (midfoot / contact origin)
    std::string leftFootCenterFrame{"foot_l_contact"};
    std::string rightFootCenterFrame{"foot_r_contact"};

    /**
     * Foot collision sphere 1 (anterior / front offset near toe):
     * Offset along +x of the foot polygon (x_max * 0.6) from the contact center.
     */
    std::string leftFootFrame1{"foot_l_contact_collision_p_1"};
    std::string rightFootFrame1{"foot_r_contact_collision_p_1"};

    /**
     * Foot collision sphere 2 (posterior / rear offset near heel):
     * Offset along -x of the foot polygon (x_min * 0.6) from the contact center.
     */
    std::string leftFootFrame2{"foot_l_contact_collision_p_2"};
    std::string rightFootFrame2{"foot_r_contact_collision_p_2"};

    /// Radius of the collision spheres centered at ankle, foot center, foot1 (front/toe), and foot2 (rear/heel)
    scalar_t footCollisionSphereRadius = 0.065;

    // Knee joint collision frames
    std::string leftKneeFrame;
    std::string rightKneeFrame;
    scalar_t kneeCollisionSphereRadius = 0.07;

    // CBF decay parameter \gamma > 0
    scalar_t gamma = 10.0;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FootCollisionCbfConstraint(const SwitchedModelReferenceManager& referenceManager,
                             const PinocchioInterface& pinocchioInterface,
                             const MpcRobotModelBase<ad_scalar_t>& mpcRobotModel,
                             const Config& config,
                             std::string costName,
                             const ModelSettings& modelSettings);

  ~FootCollisionCbfConstraint() override = default;
  FootCollisionCbfConstraint* clone() const override { return new FootCollisionCbfConstraint(*this); }

  bool isActive(scalar_t time) const override;
  bool getActive() const { return isActive_; }
  void setActive(bool active) { isActive_ = active; }

  size_t getNumConstraints(scalar_t time) const override { return numConstraints_; }

  vector_t getParameters(scalar_t time, const PreComputation& preComputation) const override {
    vector_t parameters(3);
    parameters << cfg_.footCollisionSphereRadius, cfg_.kneeCollisionSphereRadius, cfg_.gamma;
    return parameters;
  }

  void setFootCollisionSphereRadius(scalar_t footCollisionSphereRadius) { cfg_.footCollisionSphereRadius = footCollisionSphereRadius; }
  void setKneeCollisionSphereRadius(scalar_t kneeCollisionSphereRadius) { cfg_.kneeCollisionSphereRadius = kneeCollisionSphereRadius; }

  void setGamma(scalar_t gamma) { cfg_.gamma = gamma; }
  scalar_t getGamma() const { return cfg_.gamma; }

  static Config loadFootCollisionCbfConstraintConfig(absl::string_view taskFile, bool verbose = false);

 private:
  ad_vector_t constraintFunction(ad_scalar_t time,
                                 const ad_vector_t& state,
                                 const ad_vector_t& input,
                                 const ad_vector_t& parameters) const override;

  FootCollisionCbfConstraint(const FootCollisionCbfConstraint& other);

  const SwitchedModelReferenceManager* referenceManagerPtr_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  const MpcRobotModelBase<ad_scalar_t>* const mpcRobotModelPtr_;
  Config cfg_;

  size_t leftAnkleId_ = 0;
  size_t rightAnkleId_ = 0;
  size_t leftFootCenterId_ = 0;
  size_t rightFootCenterId_ = 0;
  size_t leftFoot1Id_ = 0;   // Anterior / toe collision frame (collision_p_1)
  size_t rightFoot1Id_ = 0;  // Anterior / toe collision frame (collision_p_1)
  size_t leftFoot2Id_ = 0;   // Posterior / heel collision frame (collision_p_2)
  size_t rightFoot2Id_ = 0;  // Posterior / heel collision frame (collision_p_2)
  size_t leftKneeId_ = 0;
  size_t rightKneeId_ = 0;

  const size_t numConstraints_ = 16;
  bool isActive_ = true;
};

}  // namespace ocs2::humanoid
