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

#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "ocs2_pinocchio_interface/urdf.h"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "urdf_parser/urdf_parser.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

namespace {

///
/// \brief Create a joint model composite with the appropriate DoF
///

pinocchio::JointModelComposite getBaseJointcomposite() {
  // add 6 DoF for the floating base
  pinocchio::JointModelComposite baseJointComposite(2);
  // baseJointComposite.addJoint(pinocchio::JointModelFreeFlyer());
  baseJointComposite.addJoint(pinocchio::JointModelTranslation());
  baseJointComposite.addJoint(pinocchio::JointModelSphericalZYX());
  return baseJointComposite;
}

///
/// \brief Adds a contact center frame to the location the contact wrench is applied to the system to the pinocchio
/// model.
///
/// \param[in] contactPolygon A contact polygon containinginformation about the contact center frame and corner points
/// \param[in] model The model to which the frames are added.
///

void addContactCenterFrames(const ContactPolygon& contactPolygon, pinocchio::ModelTpl<scalar_t>& model) {
  const ContactCenterPoint& ccp = contactPolygon.getContactCenterPoint();
  pinocchio::SE3 relPoseToParent(matrix3_t::Identity(), ccp.translationFromParent);
  pinocchio::Frame contactCenterFrame(ccp.frameName, model.getJointId(ccp.parentJointName), model.getFrameId(ccp.parentJointName),
                                      relPoseToParent, pinocchio::FIXED_JOINT);
  model.addFrame(contactCenterFrame);
}

///
/// \brief Adds a the collision avoidance frames to the pinocchio model.
///
/// \param[in] contactPolygon A contact polygon containinginformation about the contact center frame and corner points
/// \param[in] model The model to which the frames are added.
///

void addCollisionCenterFrames(const ContactPolygon& contactPolygon, pinocchio::ModelTpl<scalar_t>& model) {
  const ContactCenterPoint& ccp = contactPolygon.getContactCenterPoint();
  pinocchio::SE3 relPoseToParentCP1(matrix3_t::Identity(),
                                    ccp.translationFromParent + vector3_t(contactPolygon.getBounds().x_max * 0.6, 0.0, 0.0));
  pinocchio::Frame contactCenterFrameCP1(ccp.frameName + "_collision_p_1", model.getJointId(ccp.parentJointName),
                                         model.getFrameId(ccp.parentJointName), relPoseToParentCP1, pinocchio::FIXED_JOINT);
  pinocchio::SE3 relPoseToParentCP2(matrix3_t::Identity(),
                                    ccp.translationFromParent + vector3_t(contactPolygon.getBounds().x_min * 0.6, 0.0, 0.0));
  pinocchio::Frame contactCenterFrameCP2(ccp.frameName + "_collision_p_2", model.getJointId(ccp.parentJointName),
                                         model.getFrameId(ccp.parentJointName), relPoseToParentCP2, pinocchio::FIXED_JOINT);
  model.addFrame(contactCenterFrameCP1);
  model.addFrame(contactCenterFrameCP2);
}

///
/// \brief Adds a frame in each corner of the contact polygon to the pinocchio model.
///
/// \param[in] contactPolygon A contact polygon containinginformation about the contact center frame and corner points
/// \param[in] model The model to which the frames are added.
///

void addContactPolygonFrames(const ContactPolygon& contactPolygon, pinocchio::ModelTpl<scalar_t>& model) {
  addContactCenterFrames(contactPolygon, model);
  addCollisionCenterFrames(contactPolygon, model);
  const vector3_t& contactCenterTranslation = contactPolygon.getContactCenterPoint().translationFromParent;  // from parent Joint
  int nPoints = contactPolygon.getNumberOfContactPoints();
  for (int i = 0; i < nPoints; ++i) {
    vector3_t contactPointTranslation = contactCenterTranslation + contactPolygon.getContactPointTranslation(i);
    pinocchio::SE3 relPoseToParentJoint(matrix3_t::Identity(), contactPointTranslation);

    pinocchio::Frame currContactFrame(contactPolygon.getPolygonPointFrameName(i), model.getJointId(contactPolygon.getParentJointName()),
                                      model.getFrameId(contactPolygon.getParentJointName()), relPoseToParentJoint, pinocchio::FIXED_JOINT);
    model.addFrame(currContactFrame);
  }
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

PinocchioInterface createDefaultPinocchioInterface(const std::string& urdfFilePath) {
  PinocchioInterface pinocchioInterface = getPinocchioInterfaceFromUrdfFile(urdfFilePath, getBaseJointcomposite());

  return pinocchioInterface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

namespace {

/**
 * The MPC's Pinocchio model (loadCustomPinocchioInterface()), with the contact frames of the sole that
 * `contactRectangle(i)` gives each contact i.
 */
absl::StatusOr<PinocchioInterface> buildCustomPinocchioInterface(
    const std::string& urdfFilePath,
    const ModelSettings& modelSettings,
    absl::FunctionRef<absl::StatusOr<ContactRectangle>(int contactIndex)> contactRectangle,
    bool scaleTotalMass,
    scalar_t totalMass,
    bool verbose) {
  urdf::ModelInterfaceSharedPtr urdfTree = urdf::parseURDFFile(urdfFilePath);
  if (urdfTree == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("[loadCustomPinocchioInterface] the file '", urdfFilePath, "' does not contain a valid URDF model."));
  }

  using joint_pair_t = std::pair<const std::string, std::shared_ptr<urdf::Joint>>;

  // remove extraneous joints from urdf
  urdf::ModelInterfaceSharedPtr newModel = std::make_shared<urdf::ModelInterface>(*urdfTree);
  const std::vector<std::string>& mpcModelJointNames = modelSettings.mpcModelJointNames;
  for (joint_pair_t& jointPair : newModel->joints_) {
    if (std::find(mpcModelJointNames.begin(), mpcModelJointNames.end(), jointPair.first) == mpcModelJointNames.end()) {
      jointPair.second->type = urdf::Joint::FIXED;
    }
  }

  pinocchio::ModelTpl<scalar_t> model;
  pinocchio::urdf::buildModel(newModel, getBaseJointcomposite(), model);

  for (int i = 0; i < static_cast<int>(kNumContacts); ++i) {
    ASSIGN_OR_RETURN(const ContactRectangle rectangle, contactRectangle(i));
    addContactPolygonFrames(rectangle, model);
  }

  if (scaleTotalMass) {
    scalePinocchioModelInertia(model, totalMass, /*verbose=*/true);
  }

  PinocchioInterface pinocchioInterface(model, urdfTree);
  RETURN_IF_ERROR(checkPinocchioJointNaming(pinocchioInterface, modelSettings, verbose));
  return pinocchioInterface;
}

}  // namespace

absl::StatusOr<PinocchioInterface> loadCustomPinocchioInterface(const mpc_config::TaskFile& taskFile,
                                                                const std::string& urdfFilePath,
                                                                const ModelSettings& modelSettings,
                                                                bool scaleTotalMass,
                                                                scalar_t totalMass,
                                                                bool verbose) {
  return buildCustomPinocchioInterface(
      urdfFilePath, modelSettings,
      [&](int contactIndex) { return contactRectangleFromConfig(taskFile.contacts, modelSettings, contactIndex); }, scaleTotalMass,
      totalMass, verbose);
}

absl::StatusOr<PinocchioInterface> loadCustomPinocchioInterface(const std::string& taskFilePath,
                                                                const std::string& urdfFilePath,
                                                                const ModelSettings& modelSettings,
                                                                bool scaleTotalMass,
                                                                scalar_t totalMass,
                                                                bool verbose) {
  ASSIGN_OR_RETURN(const mpc_config::TaskFile taskFile, loadTaskFile(taskFilePath));
  absl::StatusOr<PinocchioInterface> pinocchioInterface =
      loadCustomPinocchioInterface(taskFile, urdfFilePath, modelSettings, scaleTotalMass, totalMass, verbose);
  if (!pinocchioInterface.ok()) {
    return withConfigFile(pinocchioInterface.status(), taskFilePath);
  }
  return pinocchioInterface;
}

}  // namespace ocs2::humanoid
