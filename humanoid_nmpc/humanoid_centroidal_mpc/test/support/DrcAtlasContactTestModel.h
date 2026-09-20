/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <memory>
#include <string>
#include <vector>

#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/FootprintCornerHeights.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * The DRC Atlas model, its configuration and a reference manager whose contact flags the test dictates.
 *
 * Every test of a contact term needs the same four things, and building them takes about forty lines of pinocchio and
 * centroidal-model setup that is easy to get subtly wrong: the robot model in BOTH input parameterizations (the
 * wrench-space CentroidalMpcRobotModel and the BasisInputsModelDecorator the shipped Atlas actually runs), the
 * end-effector kinematics of a contact frame, and a SwitchedModelReferenceManager whose mode schedule can be driven
 * from the test.
 *
 * The last one is the point. The terms under test ask the reference manager whether a foot is in contact, and the
 * whole question the contact-implicit formulation raises is what they do when the schedule says "swing" and the
 * solver disagrees. setContactFlags() writes a constant mode schedule, so getContactFlags(t) returns exactly what the
 * test asked for at every time.
 *
 * The model is loaded once per fixture instance. Code generation is disabled (`recompileLibrariesCppAd = false`), so
 * the cached libraries under the repository's cppad_code_gen/ directory are reused; a test that needs a model that has
 * never been generated must still be declared size = "large".
 */
class DrcAtlasContactTestModel {
 public:
  /**
   * @param modelNamePrefix Prefix for the generated CppAD library names, so that two fixtures with different
   *                        end-effector sets do not collide on a cached library.
   */
  explicit DrcAtlasContactTestModel(const std::string& modelNamePrefix);

  ~DrcAtlasContactTestModel();

  /** The time the tests query, and that setContactFlags() makes the schedule report `contacts` at. */
  static constexpr scalar_t kQueryTime = 0.0;

  /** Makes getContactFlags(kQueryTime) return `contacts`, flushing the reference manager's mode-schedule buffer. */
  void setContactFlags(const contact_flag_t& contacts);

  /** Both feet in contact. */
  void setStance() { setContactFlags(makeFeetArray(true)); }

  /** `swingFoot` out of contact, the other foot in contact - the case the schedule gate used to switch terms off in. */
  void setSwing(size_t swingFoot);

  const ModelSettings& modelSettings() const { return *modelSettings_; }
  const PinocchioInterface& pinocchioInterface() const { return *pinocchioInterface_; }
  const CentroidalModelInfo& centroidalModelInfo() const { return info_; }
  SwitchedModelReferenceManager& referenceManager() const { return *referenceManager_; }

  /** The wrench-space model: its input vector stores the contact wrench in the WORLD frame. */
  const CentroidalMpcRobotModel<scalar_t>& wrenchModel() const { return *wrenchModel_; }

  /**
   * The basis-vector model the shipped Atlas runs (`useContactBasisVectorInputs: true`): its input vector stores
   * non-negative scalings of a LOCAL contact-frame wrench-cone basis, so getContactWrench() returns a local-frame
   * wrench where the wrench model returns a world-frame one. That difference is the whole of audit finding A3.
   */
  const BasisInputsModelDecorator<scalar_t>& basisModel() const { return *basisModel_; }

  /** The auto-diff wrench-space model, for the CppAD-generated terms (ContactMomentXYConstraintCppAd). */
  const CentroidalMpcRobotModel<ad_scalar_t>& adWrenchModel() const { return *adWrenchModel_; }

  /** Number of basis scalings per foot, and the start index of a foot's block in the basis input vector. */
  size_t numBasisPerFoot() const { return basisModel_->getNumBasisPerFoot(); }

  /** End-effector kinematics of one contact frame, sized for the given input dimension. */
  std::unique_ptr<PinocchioEndEffectorKinematicsCppAd> makeEndEffectorKinematics(size_t contactIndex, size_t inputDim) const;

  /**
   * The heights of one contact's footprint corners: the geometry that GroundPenetrationConstraint and
   * ContactComplementarityConstraint share. createPinocchioModel() adds one frame per point of each contact polygon,
   * so no new frame is needed.
   *
   * It is built on the wrench-space auto-diff model even in the tests that drive the basis-vector model, and that is
   * not a shortcut: this object only ever asks a STATE for its generalized coordinates, and the input
   * parameterization - the one thing BasisInputsModelDecorator changes - never enters it. Sharing the model also
   * shares the cached CppAD library between the two test binaries.
   */
  std::unique_ptr<FootprintCornerHeights> makeCornerHeights(size_t contactIndex, const std::string& modelNameSuffix) const;

  /** The footprint of a contact, as the wrench cone and the basis matrices read it from the task file. */
  ContactRectangle contactRectangle(size_t contactIndex) const;

  /** The nominal state of the task file, i.e. a plausible standing configuration. */
  const vector_t& nominalState() const { return nominalState_; }

  /**
   * State index of a foot's ankle pitch joint, so a test can tilt the sole.
   *
   * Looked up by joint NAME in ModelSettings::mpcModelJointNames rather than hard-coded, because the index depends on
   * which joints the task file fixes. A test that pitches the wrong joint would still pass its own assertions while
   * proving nothing about the foot.
   */
  long anklePitchStateIndex(size_t contactIndex) const;

  /**
   * A plausible working point for the given model: the named foot carries `normalForce` newtons, the other carries
   * none, and the joints are moving. Written through the model's own setter, so it is correct for both
   * parameterizations.
   */
  vector_t makeInput(const MpcRobotModelBase<scalar_t>& model, size_t loadedFoot, scalar_t normalForce) const;

  const std::string& taskFile() const { return taskFile_; }
  const std::string& referenceFile() const { return referenceFile_; }

 private:
  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string modelNamePrefix_;

  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> wrenchModel_;
  std::unique_ptr<CentroidalMpcRobotModel<ad_scalar_t>> adWrenchModel_;
  std::unique_ptr<BasisInputsModelDecorator<scalar_t>> basisModel_;
  std::unique_ptr<CentroidalModelPinocchioMappingCppAd> mappingCppAd_;
  std::unique_ptr<SwitchedModelReferenceManager> referenceManager_;
  vector_t nominalState_;
};

}  // namespace ocs2::humanoid
