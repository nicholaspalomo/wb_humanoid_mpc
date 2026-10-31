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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_multi_contact/planner/contact_invariant/QuasiStaticTorqueConstraint.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "ocs2_centroidal_model/AccessHelperFunctions.h"
#include "ocs2_centroidal_model/CentroidalModelPinocchioMapping.h"
#include "ocs2_core/Dimensions.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"

namespace ocs2::humanoid::multi_contact {
namespace {

// Two rows per joint: below the upper limit, above the lower one.
constexpr size_t kRowsPerJoint = 2;

/** tau = g(q) - sum_i J_i' w_i of the limited joints. */
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> quasiStaticTorques(
    PinocchioInterfaceTpl<Scalar> interface,
    const CentroidalModelInfoTpl<Scalar>& info,
    const std::vector<pinocchio::FrameIndex>& frames,
    const std::vector<JointEffortLimit>& limits,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& state,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& input) {
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  const CentroidalModelPinocchioMappingTpl<Scalar> mapping(info);
  const Vector q = mapping.getPinocchioJointPosition(state);
  const typename PinocchioInterfaceTpl<Scalar>::Model& model =
      interface.getModel();
  typename PinocchioInterfaceTpl<Scalar>::Data& data = interface.getData();
  Vector tau = pinocchio::computeGeneralizedGravity(model, data, q);
  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  for (size_t i = 0; i < frames.size(); ++i) {
    Matrix jacobian = Matrix::Zero(kPoseTangentDim, model.nv);
    pinocchio::getFrameJacobian(model, data, frames[i],
                                pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
    Eigen::Matrix<Scalar, kWrenchDim, 1> wrench =
        Eigen::Matrix<Scalar, kWrenchDim, 1>::Zero();
    wrench.template head<kForceDim>() =
        centroidal_model::getForceContactEwrtWorld(input, i, info);
    if (i >= info.numThreeDofContacts) {
      wrench.template tail<kTorqueDim>() =
          centroidal_model::getTorqueContactEwrtWorld(input, i, info);
    }
    tau -= jacobian.transpose() * wrench;
  }
  Vector result(limits.size());
  for (size_t j = 0; j < limits.size(); ++j) {
    result(j) = tau(limits[j].velocityIndex);
  }
  return result;
}

}  // namespace

QuasiStaticTorqueConstraint::QuasiStaticTorqueConstraint(
    const PinocchioInterface& pinocchio,
    const CentroidalModelInfo& info,
    std::vector<std::string> contactFrames,
    std::vector<JointEffortLimit> limits,
    scalar_t margin,
    const std::string& libraryName,
    const std::string& folder,
    bool recompile,
    bool verbose)
    : StateInputConstraintCppAd(ConstraintOrder::Linear),
      pinocchioCppAd_(pinocchio.toCppAd()),
      infoCppAd_(info.toCppAd()),
      limits_(std::move(limits)),
      margin_(margin),
      pinocchio_(pinocchio),
      info_(info) {
  ABSL_CHECK_EQ(contactFrames.size(),
                info.numThreeDofContacts + info.numSixDofContacts);
  for (const std::string& frame : contactFrames) {
    ABSL_CHECK(pinocchio.getModel().existFrame(frame)) << frame;
    contactFrames_.push_back(pinocchio.getModel().getFrameId(frame));
  }
  initialize(info.stateDim, info.inputDim, /*parameterDim=*/0, libraryName,
             folder, recompile, verbose);
}

size_t QuasiStaticTorqueConstraint::getNumConstraints(scalar_t /*time*/) const {
  return kRowsPerJoint * limits_.size();
}

vector_t QuasiStaticTorqueConstraint::torques(const vector_t& state,
                                              const vector_t& input) const {
  return quasiStaticTorques<scalar_t>(pinocchio_, info_, contactFrames_,
                                      limits_, state, input);
}

ad_vector_t QuasiStaticTorqueConstraint::tapedTorques(
    const ad_vector_t& state, const ad_vector_t& input) const {
  return quasiStaticTorques<ad_scalar_t>(pinocchioCppAd_, infoCppAd_,
                                         contactFrames_, limits_, state, input);
}

ad_vector_t QuasiStaticTorqueConstraint::constraintFunction(
    ad_scalar_t /*time*/,
    const ad_vector_t& state,
    const ad_vector_t& input,
    const ad_vector_t& /*parameters*/) const {
  const ad_vector_t tau = tapedTorques(state, input);
  const size_t n = limits_.size();
  ad_vector_t value(kRowsPerJoint * n);
  for (size_t j = 0; j < n; ++j) {
    const ad_scalar_t utilization = tau(j) / ad_scalar_t(limits_[j].effort);
    value(j) = ad_scalar_t(margin_) - utilization;
    value(n + j) = ad_scalar_t(margin_) + utilization;
  }
  return value;
}

}  // namespace ocs2::humanoid::multi_contact
