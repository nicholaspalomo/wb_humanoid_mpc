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
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_pinocchio_interface/urdf.h"
#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/joint/fwd.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/fwd.hpp"
#include "urdf_model/joint.h"
#include "urdf_parser/urdf_parser.h"
#include "urdf_world/types.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "robot_model/UrdfContinuousJoints.h"

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
 * The largest entry-wise difference between the axes of a URDF frame and the
 * axes a 3-DoF contact's configuration gives it for which the contact may name
 * that frame (addPointContactFrame()).
 */
constexpr scalar_t kExistingFrameAxesTolerance = 1.0e-9;

/**
 * [m] The largest per-axis distance between the origin of a URDF frame and the
 * point a 3-DoF contact's configuration puts it at for which the contact may
 * name that frame (addPointContactFrame()).
 */
constexpr scalar_t kExistingFrameOriginToleranceM = 1.0e-9;

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
  pinocchio::SE3 poseContactRtLink(ccp.rotmatContactRtLink,
                                   ccp.posContactRtLinkEwrtLink);
  pinocchio::Frame contactCenterFrame(
      ccp.frameName, model.getJointId(ccp.parentJointName),
      model.getFrameId(ccp.parentJointName), poseContactRtLink,
      pinocchio::FIXED_JOINT);
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
              vector3_t(contactPolygon.getBounds()
                                .max_pos_corners_rt_contact_ewrt_contact_x *
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
              vector3_t(contactPolygon.getBounds()
                                .min_pos_corners_rt_contact_ewrt_contact_x *
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
    vector3_t posCornerRtLinkEwrtLink =
        center.posContactRtLinkEwrtLink +
        center.rotmatContactRtLink *
            contactPolygon.getPosCornerRtContactEwrtContact(i);
    pinocchio::SE3 poseCornerRtLink(center.rotmatContactRtLink,
                                    posCornerRtLinkEwrtLink);

    pinocchio::Frame currContactFrame(
        contactPolygon.getPolygonPointFrameName(i),
        model.getJointId(contactPolygon.getParentJointName()),
        model.getFrameId(contactPolygon.getParentJointName()), poseCornerRtLink,
        pinocchio::FIXED_JOINT);
    model.addFrame(currContactFrame);
  }
}

/** Where the configuration of a 3-DoF contact puts its frame. */
struct PointContactPlacement {
  pinocchio::JointIndex parentJoint = 0;
  pinocchio::FrameIndex parentFrame = 0;
  /** The frame's placement in the frame of `parentJoint`. */
  pinocchio::SE3 placement = pinocchio::SE3::Identity();
};

/**
 * Resolves where `contact` hangs: at its translation from its parent, a joint
 * of the model or the frame of a joint the MPC fixes (whose placement it is
 * then composed with). `entry` is the contact's entry of the task file
 * (ModelSettings::pointContactEntry()), for the NotFound refusal when the model
 * has neither a joint nor a frame of the parent's name.
 */
absl::StatusOr<PointContactPlacement> configuredPointContactPlacement(
    const ModelSettings::PointContact& contact,
    absl::string_view entry,
    const pinocchio::ModelTpl<scalar_t>& model) {
  const pinocchio::SE3 poseContactRtLink(matrix3_t::Identity(),
                                         contact.posContactRtLinkEwrtLink);
  if (model.existJointName(contact.parentJointName)) {
    return PointContactPlacement{
        .parentJoint = model.getJointId(contact.parentJointName),
        .parentFrame = model.getFrameId(contact.parentJointName),
        .placement = poseContactRtLink};
  }
  if (model.existFrame(contact.parentJointName)) {
    // A joint the MPC fixes is a frame of the reduced model: the contact hangs
    // from that frame's joint, at the frame's placement.
    const pinocchio::FrameIndex parentFrame =
        model.getFrameId(contact.parentJointName);
    const pinocchio::Frame& parent =
        model.frames[static_cast<size_t>(parentFrame)];
    return PointContactPlacement{
        .parentJoint = parent.parentJoint,
        .parentFrame = parentFrame,
        .placement = parent.placement * poseContactRtLink};
  }
  return absl::NotFoundError(absl::StrCat(
      "[loadCustomPinocchioInterface] ", entry, ".parent_joint_name '",
      contact.parentJointName,
      "' is neither a joint of the robot model nor a joint the MPC fixes."));
}

/** Formats the translation of `placement` [m] for a refusal. */
std::string translationText(const pinocchio::SE3& placement) {
  return absl::StrFormat("(%.9g, %.9g, %.9g)", placement.translation().x(),
                         placement.translation().y(),
                         placement.translation().z());
}

/**
 * Accepts the frame `existing` of the robot model as the frame of the 3-DoF
 * contact `contact` (the task file's `entry`) when it
 * hangs from the configured joint at the configured placement: the same
 * axes within kExistingFrameAxesTolerance and the same origin within
 * kExistingFrameOriginToleranceM. AlreadyExists otherwise, with both
 * placements.
 */
absl::Status acceptExistingPointContactFrame(
    const ModelSettings::PointContact& contact,
    absl::string_view entry,
    const PointContactPlacement& configured,
    const pinocchio::Frame& existing,
    const pinocchio::ModelTpl<scalar_t>& model) {
  const scalar_t axesDistance =
      (existing.placement.rotation() - configured.placement.rotation())
          .cwiseAbs()
          .maxCoeff();
  const scalar_t originDistance =
      (existing.placement.translation() - configured.placement.translation())
          .cwiseAbs()
          .maxCoeff();
  if (existing.parentJoint == configured.parentJoint &&
      axesDistance <= kExistingFrameAxesTolerance &&
      originDistance <= kExistingFrameOriginToleranceM) {
    return absl::OkStatus();
  }
  return absl::AlreadyExistsError(absl::StrCat(
      "[loadCustomPinocchioInterface] ", entry, ".name '", contact.name,
      "' is a frame of the robot model already, on joint '",
      model.names[existing.parentJoint], "' at ",
      translationText(existing.placement),
      axesDistance <= kExistingFrameAxesTolerance ? "" : " with other axes",
      ", but the configuration puts it on joint '",
      model.names[configured.parentJoint], "' at ",
      translationText(configured.placement),
      "; a 3-DoF contact names either a new frame or an existing frame at "
      "exactly its configured placement."));
}

/**
 * Adds the frame of the 3-DoF contact `contact` to `model` at its configured
 * placement (configuredPointContactPlacement()), or adds nothing when the model
 * has a frame of the contact's name at that placement already, which the MPC
 * then uses as is (a URDF foot link). `entry` is the contact's entry of the
 * task file (ModelSettings::pointContactEntry()), for the refusals: NotFound
 * when the model has neither a joint nor a frame of the parent's, AlreadyExists
 * when the model's frame of the contact's name sits elsewhere.
 */
absl::Status addPointContactFrame(const ModelSettings::PointContact& contact,
                                  absl::string_view entry,
                                  pinocchio::ModelTpl<scalar_t>& model) {
  ASSIGN_OR_RETURN(const PointContactPlacement configured,
                   configuredPointContactPlacement(contact, entry, model));
  if (model.existFrame(contact.name)) {
    const pinocchio::Frame& existing =
        model.frames[static_cast<size_t>(model.getFrameId(contact.name))];
    return acceptExistingPointContactFrame(contact, entry, configured, existing,
                                           model);
  }
  model.addFrame(pinocchio::Frame(contact.name, configured.parentJoint,
                                  configured.parentFrame, configured.placement,
                                  pinocchio::FIXED_JOINT));
  return absl::OkStatus();
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

PinocchioInterface createDefaultPinocchioInterface(
    absl::string_view urdfFilePath) {
  // A continuous joint is built with one coordinate and no position bound
  // (robot::model::parseUrdfFileForPinocchio()).
  const absl::StatusOr<urdf::ModelInterfaceSharedPtr> urdfTree =
      robot::model::parseUrdfFileForPinocchio(urdfFilePath);
  if (urdfTree.ok()) {
    return getPinocchioInterfaceFromUrdfModel(*urdfTree,
                                              getBaseJointcomposite());
  }
  // A file that does not parse is refused as OCS2 refuses it, by its
  // std::invalid_argument, which this function's callers catch.
  return getPinocchioInterfaceFromUrdfFile(std::string(urdfFilePath),
                                           getBaseJointcomposite());
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
    absl::string_view urdfFilePath,
    const ModelSettings& modelSettings,
    absl::FunctionRef<absl::StatusOr<ContactRectangle>(int contactIndex)>
        contactRectangle,
    bool scaleTotalMass,
    scalar_t totalMass,
    bool verbose) {
  urdf::ModelInterfaceSharedPtr urdfTree =
      urdf::parseURDFFile(std::string(urdfFilePath));
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
  // The active continuous joints, a wheel's, get one coordinate and no
  // position bound, as in every Pinocchio model of the URDF.
  robot::model::makeContinuousJointsUnboundedRevolute(*newModel);

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
    RETURN_IF_ERROR(addPointContactFrame(modelSettings.pointContacts[i],
                                         modelSettings.pointContactEntry(i),
                                         model));
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
    absl::string_view urdfFilePath,
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
    absl::string_view taskFilePath,
    absl::string_view urdfFilePath,
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
