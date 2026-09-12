// Copyright (c) 2024. All rights reserved.

#pragma once

#include <memory>
#include <utility>

#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

namespace ocs2::humanoid {

/**
 * @brief A PinocchioStateInputMapping decorator that converts basis-vector inputs
 * to wrench-space inputs before delegating to the wrapped mapping.
 *
 * When using basis-vector contact inputs (λ), the OCS2 input vector has the layout:
 *   [λ_0, λ_1, ..., joint_velocities]
 * where λ_i has numBasisPerFoot elements per contact.
 *
 * The wrapped mapping (e.g. CentroidalModelPinocchioMapping) expects:
 *   [W_0, W_1, ..., joint_velocities]
 * where W_i is a 6-DoF wrench (or 3-DoF force).
 *
 * This adapter converts λ → W = B · λ for each contact before delegating.
 * It is used inside CppAD graph compilation so all operations must be
 * compatible with ad_scalar_t.
 *
 * @tparam SCALAR Scalar type (scalar_t or ad_scalar_t)
 */
template <typename SCALAR>
class BasisInputsMappingDecorator final : public PinocchioStateInputMapping<SCALAR> {
 public:
  using typename PinocchioStateInputMapping<SCALAR>::vector_t;

  static constexpr size_t kWrenchDim = 6;

  /**
   * @param wrappedMapping    The original pinocchio mapping to delegate to.
   * @param basisToWrenchMap  The M matrix (double): u_wrench = M * u_basis (wrenchInputDim × basisInputDim).
   * @param wrenchInputDim    Dimension of the wrench-space input vector.
   * @param jointDim          Number of MPC joint velocity entries in the input.
   */
  BasisInputsMappingDecorator(std::unique_ptr<PinocchioStateInputMapping<SCALAR>> wrappedMapping,
                              ocs2::matrix_t basisToWrenchMap,
                              size_t wrenchInputDim,
                              size_t jointDim)
      : wrappedMapping_(std::move(wrappedMapping)), M_(std::move(basisToWrenchMap)), wrenchInputDim_(wrenchInputDim), jointDim_(jointDim) {}

  ~BasisInputsMappingDecorator() override = default;

  BasisInputsMappingDecorator<SCALAR>* clone() const override { return new BasisInputsMappingDecorator(*this); }

  void setPinocchioInterface(const PinocchioInterfaceTpl<SCALAR>& pinocchioInterface) override {
    wrappedMapping_->setPinocchioInterface(pinocchioInterface);
  }

  /** Joint positions come from the state only — no input transformation needed. */
  vector_t getPinocchioJointPosition(const vector_t& state) const override { return wrappedMapping_->getPinocchioJointPosition(state); }

  /**
   * Convert basis-vector input → wrench-space input, then delegate.
   * u_wrench = M * u_basis (contact block: W_i = B_i * λ_i; joint block: pass-through).
   */
  vector_t getPinocchioJointVelocity(const vector_t& state, const vector_t& input) const override {
    const vector_t wrenchInput = M_.template cast<SCALAR>() * input;
    return wrappedMapping_->getPinocchioJointVelocity(state, wrenchInput);
  }

  /**
   * Maps pinocchio Jacobians to OCS2 Jacobians in basis-vector space.
   *
   * The wrapped mapping returns (dfdx, dfdu_wrench).
   * Since u_wrench = M * u_basis, the chain rule gives:
   *   dfdu_basis = dfdu_wrench * M
   */
  std::pair<Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>> getOcs2Jacobian(
      const vector_t& state,
      const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>& Jq,
      const Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>& Jv) const override {
    std::pair<Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Matrix<SCALAR, Eigen::Dynamic, Eigen::Dynamic>>
        wrenchJacobians = wrappedMapping_->getOcs2Jacobian(state, Jq, Jv);
    // Transform input Jacobian: dfdu_basis = dfdu_wrench * M
    wrenchJacobians.second = wrenchJacobians.second * M_.template cast<SCALAR>();
    return wrenchJacobians;
  }

 private:
  BasisInputsMappingDecorator(const BasisInputsMappingDecorator& rhs)
      : PinocchioStateInputMapping<SCALAR>(rhs),
        wrappedMapping_(std::unique_ptr<PinocchioStateInputMapping<SCALAR>>(rhs.wrappedMapping_->clone())),
        M_(rhs.M_),
        wrenchInputDim_(rhs.wrenchInputDim_),
        jointDim_(rhs.jointDim_) {}

  std::unique_ptr<PinocchioStateInputMapping<SCALAR>> wrappedMapping_;
  ocs2::matrix_t M_;  ///< Stored as double; cast to SCALAR at usage.
  size_t wrenchInputDim_;
  size_t jointDim_;
};

}  // namespace ocs2::humanoid
