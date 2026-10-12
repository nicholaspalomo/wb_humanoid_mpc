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

#include "pinocchio/fwd.hpp"

#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/EulerMpcOracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/joint/joint-composite.hpp"
#include "pinocchio/multibody/joint/joint-spherical-ZYX.hpp"
#include "pinocchio/multibody/joint/joint-translation.hpp"
#include "pinocchio/multibody/model.hpp"

namespace ocs2::humanoid {
namespace {

/** The Euler root joint of the oracle: Composite(Translation, SphericalZYX). */
pinocchio::JointModelComposite eulerRoot() {
  pinocchio::JointModelComposite root(2);
  root.addJoint(pinocchio::JointModelTranslation());
  // NOLINTNEXTLINE(spherical-zyx): the Euler root of the oracle the quaternion root is compared with
  root.addJoint(pinocchio::JointModelSphericalZYX());
  return root;
}

/** Rows of a contact's entries of an input. */
Eigen::Index contactInputDim(const EulerMpcOracleOptions& options) {
  return options.contactInput == OracleContactInput::kWorldWrench ? 6 : options.localBasisMatrices[0].cols();
}

}  // namespace

absl::StatusOr<pinocchio::Model> eulerRootTwinOf(const pinocchio::Model& quaternionRootModel) {
  if (quaternionRootModel.njoints < 2 || quaternionRootModel.joints[1].nq() != 7 || quaternionRootModel.joints[1].nv() != 6 ||
      quaternionRootModel.parents[1] != 0) {
    return absl::InvalidArgumentError("[eulerRootTwinOf] the model's first joint is not a six-DoF root of seven configuration entries");
  }
  pinocchio::Model twin;
  twin.name = quaternionRootModel.name;
  twin.gravity = quaternionRootModel.gravity;
  for (pinocchio::JointIndex joint = 1; joint < static_cast<pinocchio::JointIndex>(quaternionRootModel.njoints); ++joint) {
    const pinocchio::JointIndex added = joint == 1
                                            ? twin.addJoint(/*parent=*/0, pinocchio::JointModel(eulerRoot()),
                                                            quaternionRootModel.jointPlacements[joint], quaternionRootModel.names[joint])
                                            : twin.addJoint(quaternionRootModel.parents[joint], quaternionRootModel.joints[joint],
                                                            quaternionRootModel.jointPlacements[joint], quaternionRootModel.names[joint]);
    if (added != joint) return absl::InternalError("[eulerRootTwinOf] the twin's joints are not numbered as the model's");
    twin.inertias[joint] = quaternionRootModel.inertias[joint];
  }
  // Frame 0 is the universe of both; every other frame keeps its index, so its parent frame's index is the model's.
  for (pinocchio::FrameIndex frame = 1; frame < static_cast<pinocchio::FrameIndex>(quaternionRootModel.nframes); ++frame) {
    // The joints' inertias already hold every body's: the frames add none.
    twin.addFrame(quaternionRootModel.frames[frame], /*append_inertia=*/false);
  }
  if (twin.nv != quaternionRootModel.nv || twin.nq != quaternionRootModel.nq - 1 || twin.nframes != quaternionRootModel.nframes) {
    return absl::InternalError(absl::StrCat("[eulerRootTwinOf] the twin has nq ", twin.nq, ", nv ", twin.nv, ", ", twin.nframes,
                                            " frames; the model nq ", quaternionRootModel.nq, ", nv ", quaternionRootModel.nv, ", ",
                                            quaternionRootModel.nframes, " frames"));
  }
  const Eigen::Index joints = twin.nv - 6;
  twin.lowerPositionLimit.tail(joints) = quaternionRootModel.lowerPositionLimit.tail(joints);
  twin.upperPositionLimit.tail(joints) = quaternionRootModel.upperPositionLimit.tail(joints);
  twin.armature = quaternionRootModel.armature;
  return twin;
}

absl::StatusOr<std::unique_ptr<EulerMpcOracle>> EulerMpcOracle::Create(const pinocchio::Model& quaternionRootModel,
                                                                       EulerMpcOracleOptions options) {
  absl::StatusOr<pinocchio::Model> twin = eulerRootTwinOf(quaternionRootModel);
  if (!twin.ok()) return twin.status();
  std::array<pinocchio::FrameIndex, kNumContacts> contactFrames{};
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    if (!twin->existFrame(options.contactFrameNames[contact])) {
      return absl::InvalidArgumentError(absl::StrCat("[EulerMpcOracle] no contact frame ", options.contactFrameNames[contact]));
    }
    contactFrames[contact] = twin->getFrameId(options.contactFrameNames[contact]);
  }
  if (options.contactInput == OracleContactInput::kLocalBasis) {
    if (options.formulation == OracleFormulation::kWholeBody) {
      return absl::InvalidArgumentError("[EulerMpcOracle] the whole-body MPC takes world-frame wrenches");
    }
    for (const matrix_t& basis : options.localBasisMatrices) {
      if (basis.rows() != 6 || basis.cols() != options.localBasisMatrices[0].cols() || basis.cols() == 0) {
        return absl::InvalidArgumentError("[EulerMpcOracle] a local basis is not 6 rows by the same number of columns on both contacts");
      }
    }
  }
  twin->gravity.linear(options.gravity);
  // Not std::make_unique: the constructor is private (ToTW #134).
  return absl::WrapUnique(new EulerMpcOracle(*twin, std::move(options), contactFrames));
}

EulerMpcOracle::EulerMpcOracle(const pinocchio::Model& model,
                               EulerMpcOracleOptions options,
                               std::array<pinocchio::FrameIndex, kNumContacts> contactFrames)
    : model_(model), options_(std::move(options)), contactFrames_(contactFrames), totalMass_(pinocchio::computeTotalMass(model_)) {}

Eigen::Index EulerMpcOracle::stateDim() const {
  return options_.formulation == OracleFormulation::kCentroidal ? 12 + jointDim() : 12 + 2 * jointDim();
}

Eigen::Index EulerMpcOracle::inputDim() const {
  return static_cast<Eigen::Index>(kNumContacts) * contactInputDim(options_) + jointDim();
}

vector_t EulerMpcOracle::configurationOf(const vector_t& eulerState) const {
  return options_.formulation == OracleFormulation::kCentroidal ? vector_t(eulerState.segment(6, model_.nq))
                                                                : vector_t(eulerState.head(model_.nq));
}

std::array<vector6_t, kNumContacts> EulerMpcOracle::worldWrenches(const vector_t& eulerState, const vector_t& input) const {
  ABSL_CHECK_EQ(eulerState.size(), stateDim());
  ABSL_CHECK_EQ(input.size(), inputDim());
  std::array<vector6_t, kNumContacts> wrenches{};
  if (options_.contactInput == OracleContactInput::kWorldWrench) {
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      wrenches[contact] = input.segment<6>(6 * static_cast<Eigen::Index>(contact));
    }
    return wrenches;
  }
  pinocchio::Data data(model_);
  pinocchio::framesForwardKinematics(model_, data, configurationOf(eulerState));
  const Eigen::Index basisDim = contactInputDim(options_);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    const vector6_t local = options_.localBasisMatrices[contact] * input.segment(basisDim * static_cast<Eigen::Index>(contact), basisDim);
    const matrix3_t rotation = data.oMf[contactFrames_[contact]].rotation();
    wrenches[contact] << rotation * local.head<3>(), rotation * local.tail<3>();
  }
  return wrenches;
}

vector_t EulerMpcOracle::flowMap(const vector_t& eulerState, const vector_t& input) const {
  const std::array<vector6_t, kNumContacts> wrenches = worldWrenches(eulerState, input);
  const Eigen::Index nj = jointDim();
  const Eigen::Index nv = model_.nv;
  const vector_t jointRates = input.tail(nj);
  const vector_t q = configurationOf(eulerState);
  pinocchio::Data data(model_);
  vector_t derivative(stateDim());

  if (options_.formulation == OracleFormulation::kCentroidal) {
    // (O1): computeCentroidalMap places the joints and the center of mass; the frames follow from the joints.
    const matrix_t centroidalMap = pinocchio::computeCentroidalMap(model_, data, q);
    pinocchio::updateFramePlacements(model_, data);
    const vector3_t com = data.com[0];
    vector6_t momentumRate;
    momentumRate << totalMass_ * options_.gravity, vector3_t::Zero();
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      const vector3_t force = wrenches[contact].head<3>();
      const vector3_t arm = data.oMf[contactFrames_[contact]].translation() - com;
      momentumRate.head<3>() += force;
      momentumRate.tail<3>() += arm.cross(force) + wrenches[contact].tail<3>();
    }
    // (O2): the base velocity of the momentum m hbar with the input's joint rates.
    const vector6_t baseMomentum = totalMass_ * eulerState.head<6>() - centroidalMap.rightCols(nj) * jointRates;
    const vector6_t baseVelocity = centroidalMap.leftCols<6>().partialPivLu().solve(baseMomentum);
    derivative << momentumRate / totalMass_, baseVelocity, jointRates;
    return derivative;
  }

  // (O3): the floating-base rows of the equations of motion in Euler coordinates.
  const vector_t v = eulerState.tail(nv);
  const matrix_t massMatrixUpper = pinocchio::crba(model_, data, q);
  matrix_t massMatrix = massMatrixUpper;
  massMatrix.triangularView<Eigen::StrictlyLower>() = massMatrixUpper.transpose();
  const vector_t nonlinearEffects = pinocchio::nonLinearEffects(model_, data, q, v);
  vector6_t baseForce = -nonlinearEffects.head<6>() - massMatrix.topRightCorner(6, nj) * jointRates;
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    matrix_t jacobian = matrix_t::Zero(6, nv);
    pinocchio::computeFrameJacobian(model_, data, q, contactFrames_[contact], pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
    baseForce += jacobian.leftCols<6>().transpose() * wrenches[contact];
  }
  const vector6_t baseAcceleration = massMatrix.topLeftCorner<6, 6>().llt().solve(baseForce);
  // (O4): here `jointRates` are the input's joint accelerations.
  derivative << v, baseAcceleration, jointRates;
  return derivative;
}

vector_t EulerMpcOracle::rk4Step(const vector_t& eulerState, const vector_t& input, scalar_t dt) const {
  const vector_t k1 = flowMap(eulerState, input);
  const vector_t k2 = flowMap(eulerState + 0.5 * dt * k1, input);
  const vector_t k3 = flowMap(eulerState + 0.5 * dt * k2, input);
  const vector_t k4 = flowMap(eulerState + dt * k3, input);
  return eulerState + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

FiniteDifferenceJacobian richardsonJacobian(absl::FunctionRef<vector_t(const vector_t&)> function,
                                            const vector_t& at,
                                            scalar_t relativeStep) {
  const vector_t center = function(at);
  const Eigen::Index rows = center.size();
  const Eigen::Index columns = at.size();
  matrix_t coarse(rows, columns);
  matrix_t fine(rows, columns);
  vector_t magnitude = center.cwiseAbs();
  for (Eigen::Index column = 0; column < columns; ++column) {
    const scalar_t step = relativeStep * std::max(1.0, std::abs(at(column)));
    std::array<vector_t, 4> values;
    const std::array<scalar_t, 4> offsets = {step, -step, 0.5 * step, -0.5 * step};
    for (size_t i = 0; i < offsets.size(); ++i) {
      vector_t shifted = at;
      shifted(column) += offsets[i];
      values[i] = function(shifted);
      magnitude = magnitude.cwiseMax(values[i].cwiseAbs());
    }
    coarse.col(column) = (values[0] - values[1]) / (2.0 * step);
    fine.col(column) = (values[2] - values[3]) / step;
  }
  FiniteDifferenceJacobian jacobian;
  jacobian.value = (4.0 * fine - coarse) / 3.0;
  jacobian.errorBound = (coarse - fine).cwiseAbs();
  // A row that is exactly zero at every evaluation (an identically zero function) has no evaluation error: its differences
  // are exact zeros. Every other row's error is relative to its largest magnitude, floor 1.
  const vector_t evaluationScale = (magnitude.array() > 0.0).select(magnitude.cwiseMax(1.0), vector_t::Zero(rows));
  for (Eigen::Index column = 0; column < columns; ++column) {
    const scalar_t step = relativeStep * std::max(1.0, std::abs(at(column)));
    jacobian.errorBound.col(column) += (3.0 * kFunctionEvaluationRelativeError / step) * evaluationScale;
  }
  return jacobian;
}

}  // namespace ocs2::humanoid
