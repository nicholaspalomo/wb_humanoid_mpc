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

/** The joints the floating base is composed of: the translation and the
 * spherical joint. */
constexpr int kNumBaseJointComponents = 2;

/**
 * The fraction of the footprint's front (x_max) and back (x_min) extent at
 * which the two collision points of a foot sit, along the sole's x axis.
 */
constexpr scalar_t kCollisionPointFootprintFraction = 0.6;

/**
 * The floating base of every MPC model: Composite(Translation, Spherical), 6
 * degrees of freedom with the configuration q_b = [p_W, xi] (the position in
 * the world and the unit quaternion (x, y, z, w) of the base orientation, 7
 * entries) and the velocity v_b = [dp_W, w_B] (the world-frame linear velocity
 * and the body angular velocity, 6 entries), so that nq = nv + 1
 * (humanoid_nmpc/docs/quaternion_base_orientation/README.md, section 2.2).
 */
pinocchio::JointModelComposite getBaseJointcomposite() {
  pinocchio::JointModelComposite baseJointComposite(kNumBaseJointComponents);
  baseJointComposite.addJoint(pinocchio::JointModelTranslation());
  baseJointComposite.addJoint(pinocchio::JointModelSpherical());
  return baseJointComposite;
}

///
/// \brief Adds a contact center frame to the location the contact wrench is
/// applied to the system to the pinocchio model.
///
/// \param[in] contactPolygon A contact polygon containinginformation about the
/// contact center frame and corner points \param[in] model The model to which
/// the frames are added.
///

void addContactCenterFrames(const ContactPolygon& contactPolygon,
                            pinocchio::ModelTpl<scalar_t>& model) {
  const ContactCenterPoint& ccp = contactPolygon.getContactCenterPoint();
  pinocchio::SE3 contactPlacement(ccp.rotmatContactRtLink,
                                  ccp.posContactRtLinkEwrtLink);
  pinocchio::Frame contactCenterFrame(ccp.frameName,
                                      model.getJointId(ccp.parentJointName),
                                      model.getFrameId(ccp.parentJointName),
                                      contactPlacement, pinocchio::FIXED_JOINT);
  model.addFrame(contactCenterFrame);
}

///
/// \brief Adds a the collision avoidance frames to the pinocchio model.
///
/// \param[in] contactPolygon A contact polygon containinginformation about the
/// contact center frame and corner points \param[in] model The model to which
/// the frames are added.
///

void addCollisionCenterFrames(const ContactPolygon& contactPolygon,
                              pinocchio::ModelTpl<scalar_t>& model) {
  const ContactCenterPoint& ccp = contactPolygon.getContactCenterPoint();
  // Along the contact frame's x axis, which is the parent's for a sole.
  pinocchio::SE3 collisionPointOnePlacement(
      ccp.rotmatContactRtLink,
      ccp.posContactRtLinkEwrtLink +
          ccp.rotmatContactRtLink *
              vector3_t(contactPolygon.getBounds().x_max *
                            kCollisionPointFootprintFraction,
                        0.0, 0.0));
  pinocchio::Frame contactCenterFrameCP1(
      ccp.frameName + "_collision_p_1", model.getJointId(ccp.parentJointName),
      model.getFrameId(ccp.parentJointName), collisionPointOnePlacement,
      pinocchio::FIXED_JOINT);
  pinocchio::SE3 collisionPointTwoPlacement(
      ccp.rotmatContactRtLink,
      ccp.posContactRtLinkEwrtLink +
          ccp.rotmatContactRtLink *
              vector3_t(contactPolygon.getBounds().x_min *
                            kCollisionPointFootprintFraction,
                        0.0, 0.0));
  pinocchio::Frame contactCenterFrameCP2(
      ccp.frameName + "_collision_p_2", model.getJointId(ccp.parentJointName),
      model.getFrameId(ccp.parentJointName), collisionPointTwoPlacement,
      pinocchio::FIXED_JOINT);
  model.addFrame(contactCenterFrameCP1);
  model.addFrame(contactCenterFrameCP2);
}

///
/// \brief Adds a frame in each corner of the contact polygon to the pinocchio
/// model.
///
/// \param[in] contactPolygon A contact polygon containinginformation about the
/// contact center frame and corner points \param[in] model The model to which
/// the frames are added.
///

void addContactPolygonFrames(const ContactPolygon& contactPolygon,
                             pinocchio::ModelTpl<scalar_t>& model) {
  addContactCenterFrames(contactPolygon, model);
  addCollisionCenterFrames(contactPolygon, model);
  const ContactCenterPoint& center = contactPolygon.getContactCenterPoint();
  // The corners are in the contact frame, which the parent's frame rotates by
  // center.rotmatContactRtLink (the identity for a sole).
  int nPoints = contactPolygon.getNumberOfContactPoints();
  for (int i = 0; i < nPoints; ++i) {
    vector3_t contactPointTranslation =
        center.posContactRtLinkEwrtLink +
        center.rotmatContactRtLink *
            contactPolygon.getContactPointTranslation(i);
    pinocchio::SE3 cornerPlacement(center.rotmatContactRtLink,
                                   contactPointTranslation);

    pinocchio::Frame currContactFrame(
        contactPolygon.getPolygonPointFrameName(i),
        model.getJointId(contactPolygon.getParentJointName()),
        model.getFrameId(contactPolygon.getParentJointName()), cornerPlacement,
        pinocchio::FIXED_JOINT);
    model.addFrame(currContactFrame);
  }
}

/**
 * Adds the frame of the 3-DoF contact `contact` to `model`: at its translation
 * from its parent, a joint of the model or the frame of a joint the MPC fixes
 * (whose placement it is then composed with). `index` is the contact's entry of
 * model_settings.contacts_3dof, for the refusals: AlreadyExists when the model
 * has a frame of the contact's name, NotFound when it has neither a joint nor a
 * frame of the parent's.
 */
absl::Status addPointContactFrame(const ModelSettings::PointContact& contact,
                                  size_t index,
                                  pinocchio::ModelTpl<scalar_t>& model) {
  if (model.existFrame(contact.name)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "[loadCustomPinocchioInterface] model_settings.contacts_3dof[", index,
        "].name '", contact.name,
        "' is a frame of the robot model already; a 3-DoF contact names the "
        "frame the MPC adds."));
  }
  const pinocchio::SE3 offset(matrix3_t::Identity(), contact.translation);
  if (model.existJointName(contact.parentJointName)) {
    model.addFrame(pinocchio::Frame(contact.name,
                                    model.getJointId(contact.parentJointName),
                                    model.getFrameId(contact.parentJointName),
                                    offset, pinocchio::FIXED_JOINT));
    return absl::OkStatus();
  }
  if (model.existFrame(contact.parentJointName)) {
    // A joint the MPC fixes is a frame of the reduced model: the contact hangs
    // from that frame's joint, at the frame's placement.
    const pinocchio::FrameIndex parentFrame =
        model.getFrameId(contact.parentJointName);
    const pinocchio::Frame& parent =
        model.frames[static_cast<size_t>(parentFrame)];
    model.addFrame(pinocchio::Frame(contact.name, parent.parentJoint,
                                    parentFrame, parent.placement * offset,
                                    pinocchio::FIXED_JOINT));
    return absl::OkStatus();
  }
  return absl::NotFoundError(absl::StrCat(
      "[loadCustomPinocchioInterface] model_settings.contacts_3dof[", index,
      "].parent_joint_name '", contact.parentJointName,
      "' is neither a joint of the robot model nor a joint the MPC fixes."));
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

PinocchioInterface createDefaultPinocchioInterface(
    const std::string& urdfFilePath) {
  PinocchioInterface pinocchioInterface =
      getPinocchioInterfaceFromUrdfFile(urdfFilePath, getBaseJointcomposite());

  return pinocchioInterface;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

namespace {

/**
 * The MPC's Pinocchio model (loadCustomPinocchioInterface()), with the contact
 * frames of the sole that `contactRectangle(i)` gives each contact i.
 */
absl::StatusOr<PinocchioInterface> buildCustomPinocchioInterface(
    const std::string& urdfFilePath,
    const ModelSettings& modelSettings,
    absl::FunctionRef<absl::StatusOr<ContactRectangle>(int contactIndex)>
        contactRectangle,
    bool scaleTotalMass,
    scalar_t totalMass,
    bool verbose) {
  urdf::ModelInterfaceSharedPtr urdfTree = urdf::parseURDFFile(urdfFilePath);
  if (urdfTree == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("[loadCustomPinocchioInterface] the file '", urdfFilePath,
                     "' does not contain a valid URDF model."));
  }

  using joint_pair_t =
      std::pair<const std::string, std::shared_ptr<urdf::Joint>>;

  // remove extraneous joints from urdf
  urdf::ModelInterfaceSharedPtr newModel =
      std::make_shared<urdf::ModelInterface>(*urdfTree);
  const std::vector<std::string>& mpcModelJointNames =
      modelSettings.mpcModelJointNames;
  for (joint_pair_t& jointPair : newModel->joints_) {
    if (std::find(mpcModelJointNames.begin(), mpcModelJointNames.end(),
                  jointPair.first) == mpcModelJointNames.end()) {
      jointPair.second->type = urdf::Joint::FIXED;
    }
  }

  pinocchio::ModelTpl<scalar_t> model;
  pinocchio::urdf::buildModel(newModel, getBaseJointcomposite(), model);

  // The footprint frames of each 6-DoF contact, which follow the 3-DoF ones in
  // the contact set.
  const int firstSurfaceContact =
      static_cast<int>(modelSettings.contactNames3DoF.size());
  const int numContacts = static_cast<int>(modelSettings.contactNames.size());
  for (int i = firstSurfaceContact; i < numContacts; ++i) {
    ASSIGN_OR_RETURN(const ContactRectangle rectangle, contactRectangle(i));
    // The contact frames hang from model_settings' parent joint of the contact:
    // a name the model does not have would give them an out-of-range parent
    // (Pinocchio throws, without the name).
    if (!model.existJointName(rectangle.getParentJointName())) {
      return absl::NotFoundError(absl::StrCat(
          "[loadCustomPinocchioInterface] the parent joint \"",
          rectangle.getParentJointName(), "\" of contact ", i,
          " (model_settings.contact_parent_joint_names) is not a joint of the ",
          "robot model."));
    }
    addContactPolygonFrames(rectangle, model);
  }
  for (size_t i = 0; i < modelSettings.pointContacts.size(); ++i) {
    RETURN_IF_ERROR(
        addPointContactFrame(modelSettings.pointContacts[i], i, model));
  }

  if (scaleTotalMass) {
    scalePinocchioModelInertia(model, totalMass, Verbosity::kVerbose);
  }

  PinocchioInterface pinocchioInterface(model);
  RETURN_IF_ERROR(checkPinocchioJointNaming(
      pinocchioInterface, modelSettings,
      verbose ? Verbosity::kVerbose : Verbosity::kQuiet));
  return pinocchioInterface;
}

}  // namespace

absl::StatusOr<PinocchioInterface> loadCustomPinocchioInterface(
    const mpc_config::TaskFile& taskFile,
    const std::string& urdfFilePath,
    const ModelSettings& modelSettings,
    bool scaleTotalMass,
    scalar_t totalMass,
    bool verbose) {
  RETURN_IF_ERROR(validateSurfacePads(taskFile.contacts, modelSettings));
  return buildCustomPinocchioInterface(
      urdfFilePath, modelSettings,
      [&](int contactIndex) {
        return contactRectangleFromConfig(taskFile.contacts, modelSettings,
                                          contactIndex);
      },
      scaleTotalMass, totalMass, verbose);
}

absl::StatusOr<PinocchioInterface> loadCustomPinocchioInterface(
    const std::string& taskFilePath,
    const std::string& urdfFilePath,
    const ModelSettings& modelSettings,
    bool scaleTotalMass,
    scalar_t totalMass,
    bool verbose) {
  ASSIGN_OR_RETURN(const mpc_config::TaskFile taskFile,
                   loadTaskFile(taskFilePath));
  absl::StatusOr<PinocchioInterface> pinocchioInterface =
      loadCustomPinocchioInterface(taskFile, urdfFilePath, modelSettings,
                                   scaleTotalMass, totalMass, verbose);
  if (!pinocchioInterface.ok()) {
    return withConfigFile(pinocchioInterface.status(), taskFilePath);
  }
  return pinocchioInterface;
}

}  // namespace ocs2::humanoid
