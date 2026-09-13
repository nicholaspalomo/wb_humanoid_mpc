#pragma once

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

class ComAndAcomTrackingCost : public StateCost {
 public:
  ComAndAcomTrackingCost(matrix_t Q_com,
                         matrix_t Q_acom,
                         PinocchioInterface pinocchioInterface,
                         CentroidalModelInfo info,
                         const SwitchedModelReferenceManager& referenceManager);

  ~ComAndAcomTrackingCost() override = default;
  ComAndAcomTrackingCost* clone() const override;

  bool isActive(scalar_t time) const override { return true; }

  scalar_t getValue(scalar_t time,
                    const vector_t& state,
                    const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComp) const override;

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time,
                                                                 const vector_t& state,
                                                                 const TargetTrajectories& targetTrajectories,
                                                                 const PreComputation& preComp) const override;

 private:
  ComAndAcomTrackingCost(const ComAndAcomTrackingCost& rhs);

  matrix_t Q_com_;
  matrix_t Q_acom_;
  PinocchioInterface pinocchioInterface_;
  CentroidalModelInfo info_;
  const SwitchedModelReferenceManager* referenceManagerPtr_;
  std::unique_ptr<AngularCenterOfMass> acom_;
};

}  // namespace ocs2::humanoid
