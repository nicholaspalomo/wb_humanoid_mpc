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

#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_mpc_config/acom_weights.nproto.h"
#include "humanoid_mpc_config/com_weights.nproto.h"
#include "humanoid_mpc_config/input_weights.nproto.h"
#include "humanoid_mpc_config/joint_value.nproto.h"
#include "humanoid_mpc_config/joint_weights.nproto.h"
#include "humanoid_mpc_config/state_values.nproto.h"
#include "humanoid_mpc_config/state_weights.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/xyz.nproto.h"
#include "humanoid_mpc_config/yaw_pitch_roll.nproto.h"

namespace ocs2::humanoid {
namespace {

constexpr absl::string_view kErrorPrefix = "[StateInputWeightsFromConfig] ";
// The components of a block, in the order the state and the input hold them: a vector's, and an orientation's or its
// rates' (Euler ZYX).
constexpr std::array<absl::string_view, 3> kXyzComponents = {"x", "y", "z"};
constexpr std::array<absl::string_view, 3> kYawPitchRollComponents = {"yaw", "pitch", "roll"};

std::array<double, 3> components(const mpc_config::Xyz& block) {
  return {block.x, block.y, block.z};
}

std::array<double, 3> components(const mpc_config::YawPitchRoll& block) {
  return {block.yaw, block.pitch, block.roll};
}

const std::array<absl::string_view, 3>& componentNames(const mpc_config::Xyz& /*block*/) {
  return kXyzComponents;
}

const std::array<absl::string_view, 3>& componentNames(const mpc_config::YawPitchRoll& /*block*/) {
  return kYawPitchRollComponents;
}

absl::string_view mpcName(StateInputLayout::Mpc mpc) {
  return mpc == StateInputLayout::Mpc::kWholeBody ? "whole-body" : "centroidal";
}

/** `scaling` times the diagonal matrix of `diagonal`, element by element, as loadEigenMatrix scales what it reads. */
matrix_t scaledDiagonal(scalar_t scaling, const vector_t& diagonal) {
  const Eigen::Index size = diagonal.size();
  matrix_t matrix(size, size);
  for (Eigen::Index row = 0; row < size; ++row) {
    for (Eigen::Index col = 0; col < size; ++col) {
      // Off the diagonal too: `scaling` times 0 is -0.0 for a negative scaling.
      matrix(row, col) = scaling * (row == col ? diagonal(row) : 0.0);
    }
  }
  return matrix;
}

/** Whether the MPC a conversion places a block onto has that block in its state or input. */
enum class InThisMpc { kPresent, kAbsent };

InThisMpc inThisMpcIf(bool present) {
  return present ? InThisMpc::kPresent : InThisMpc::kAbsent;
}

/**
 * One conversion of a block of the task file onto a StateInputLayout: builds the vector of the state or the input the
 * block gives, a component at a time, and collects every problem with the block, each naming its field path, for one
 * error. Holds its own copy of the layout. Not thread-safe; used on one thread for one block.
 */
class LayoutConversion {
 public:
  /**
   * @param layout The layout.
   * @param fieldPath The block's path in its file, e.g. "state_weights".
   * @param vectorName "state" or "input": what the block's coordinates are of, for the messages.
   * @param size The size of the vector the conversion builds, zeros until a component is placed; 0 for a conversion that
   *        only checks names.
   */
  LayoutConversion(StateInputLayout layout, absl::string_view fieldPath, absl::string_view vectorName, Eigen::Index size)
      : layout_(std::move(layout)), fieldPath_(fieldPath), vectorName_(vectorName), values_(vector_t::Zero(size)) {
    for (size_t i = 0; i < layout_.jointNames.size(); ++i) {
      jointIndices_.emplace(layout_.jointNames[i], static_cast<Eigen::Index>(i));
    }
    fixedJoints_.insert(layout_.fixedJointNames.begin(), layout_.fixedJointNames.end());
  }

  ~LayoutConversion() = default;
  LayoutConversion(const LayoutConversion&) = delete;
  LayoutConversion& operator=(const LayoutConversion&) = delete;
  LayoutConversion(LayoutConversion&&) = delete;
  LayoutConversion& operator=(LayoutConversion&&) = delete;

  const StateInputLayout& layout() const { return layout_; }

  std::string path(absl::string_view field) const { return absl::StrCat(fieldPath_, ".", field); }

  void addProblem(std::string problem) { problems_.push_back(std::move(problem)); }

  /** Refuses `value` at `path` unless it is finite. */
  void checkFinite(double value, absl::string_view path) {
    if (!std::isfinite(value)) {
      addProblem(absl::StrCat(path, " is ", value, ", not a finite number"));
    }
  }

  /**
   * Places the components of `block` (field `name`) at `start` of the vector when the MPC has the block (`inThisMpc`),
   * which then requires it; refuses it when the MPC does not have it.
   */
  template <typename Block>
  void placeBlock(std::optional<Block> block, InThisMpc inThisMpc, absl::string_view name, Eigen::Index start) {
    const std::string blockPath = path(name);
    if (inThisMpc == InThisMpc::kAbsent) {
      if (block.has_value()) {
        addProblem(absl::StrCat(blockPath, " is not part of the ", mpcName(layout_.mpc), " MPC's ", vectorName_));
      }
      return;
    }
    if (!block.has_value()) {
      addProblem(absl::StrCat(blockPath, " is missing (", mpcName(layout_.mpc), " MPC; {} is a block of zeros)"));
      return;
    }
    const std::array<double, 3> given = components(*block);
    const std::array<absl::string_view, 3>& names = componentNames(*block);
    for (size_t i = 0; i < given.size(); ++i) {
      checkFinite(given[i], absl::StrCat(blockPath, ".", names[i]));
      values_(start + static_cast<Eigen::Index>(i)) = given[i];
    }
  }

  /**
   * Places the value of each joint of `entries` (field `name`) at `start` plus the joint's index in the layout when
   * the MPC has the joint list (`inThisMpc`), which then requires every joint exactly once; refuses a non-empty list
   * when the MPC does not have it.
   */
  void placeJoints(const std::vector<mpc_config::JointValue>& entries, InThisMpc inThisMpc, absl::string_view name, Eigen::Index start) {
    const std::string listPath = path(name);
    if (inThisMpc == InThisMpc::kAbsent) {
      if (!entries.empty()) {
        addProblem(absl::StrCat(listPath, " is not part of the ", mpcName(layout_.mpc), " MPC's ", vectorName_));
      }
      return;
    }
    std::vector<bool> given(layout_.jointNames.size(), false);
    std::vector<std::string> twice;
    for (const mpc_config::JointValue& entry : entries) {
      const std::optional<Eigen::Index> index = jointIndex(entry.joint, listPath);
      if (!index.has_value()) continue;
      if (given[static_cast<size_t>(*index)]) {
        twice.push_back(entry.joint);
        continue;
      }
      given[static_cast<size_t>(*index)] = true;
      checkFinite(entry.value, absl::StrCat(listPath, "[joint=", entry.joint, "].value"));
      values_(start + *index) = entry.value;
    }
    addRepeated(listPath, "joint", twice);
    std::vector<std::string> missing;
    for (size_t i = 0; i < given.size(); ++i) {
      if (!given[i]) missing.push_back(layout_.jointNames[i]);
    }
    if (!missing.empty()) {
      addProblem(absl::StrCat(listPath, ": missing ", absl::StrJoin(missing, ", "), " (every joint of the MPC model exactly once)"));
    }
  }

  /**
   * The index of `joint` in the layout's joints, or nullopt after refusing it as a fixed joint or a name the MPC model
   * does not have (the problem names `listPath`).
   */
  std::optional<Eigen::Index> jointIndex(const std::string& joint, absl::string_view listPath) {
    const absl::flat_hash_map<std::string, Eigen::Index>::const_iterator found = jointIndices_.find(joint);
    if (found != jointIndices_.end()) return found->second;
    if (fixedJoints_.contains(joint)) {
      addProblem(absl::StrCat(listPath, ": ", joint, " is a fixed joint (model_settings.fixed_joint_names), not part of the MPC model"));
    } else {
      addProblem(absl::StrCat(listPath, ": '", joint,
                              "' is not a joint of the MPC model (its joints: ", absl::StrJoin(layout_.jointNames, ", "), ")"));
    }
    return std::nullopt;
  }

  /** Refuses the names `twice` that the list `listPath` gives more than once. */
  void addRepeated(absl::string_view listPath, absl::string_view what, const std::vector<std::string>& twice) {
    if (!twice.empty()) {
      addProblem(absl::StrCat(listPath, ": ", absl::StrJoin(twice, ", "), " given more than once (each ", what, " at most once)"));
    }
  }

  /** OK, or InvalidArgument listing every problem. */
  absl::Status status() const {
    if (problems_.empty()) return absl::OkStatus();
    return absl::InvalidArgumentError(absl::StrCat(kErrorPrefix, absl::StrJoin(problems_, "; ")));
  }

  /** The vector built, or status()'s error when the block has a problem. */
  absl::StatusOr<vector_t> values() const {
    RETURN_IF_ERROR(status());
    return values_;
  }

 private:
  StateInputLayout layout_;
  std::string fieldPath_;
  std::string vectorName_;
  // The vector the placed components are written into, of the size the conversion was made with.
  vector_t values_;
  absl::flat_hash_map<std::string, Eigen::Index> jointIndices_;
  absl::flat_hash_set<std::string> fixedJoints_;
  std::vector<std::string> problems_;
};

// The offsets of the blocks of the state and the input in their vectors, as the MPCs' robot models index them
// (StateInputLayout).
// LINT.IfChange(state_input_offsets)
// The base pose (position, then orientation) and, in the whole-body state, the base velocity.
constexpr Eigen::Index kBaseDimension = 6;
// The wrench of a contact in the input: its force at 0, its moment at 3.
constexpr Eigen::Index kWrenchDimension = 6;
constexpr Eigen::Index kMomentOffset = 3;

/**
 * Places the blocks of a state (a StateWeights or a StateValues, which share their fields) in the vector of
 * `conversion`, on the offsets of StateInputLayout.
 */
template <typename State>
void placeState(const State& state, LayoutConversion& conversion) {
  const bool centroidal = conversion.layout().mpc == StateInputLayout::Mpc::kCentroidal;
  const Eigen::Index joints = static_cast<Eigen::Index>(conversion.layout().jointNames.size());
  // The centroidal state: the momenta, the base pose, the joint positions.
  // The whole-body state: the base pose, the joint positions, the base velocity, the joint velocities.
  const Eigen::Index basePoseStart = centroidal ? kBaseDimension : 0;
  const Eigen::Index jointPositionsStart = basePoseStart + kBaseDimension;
  const Eigen::Index baseVelocityStart = jointPositionsStart + joints;
  const Eigen::Index jointVelocitiesStart = baseVelocityStart + kBaseDimension;
  conversion.placeBlock(state.normalized_linear_momentum, inThisMpcIf(centroidal), "normalized_linear_momentum", /*start=*/0);
  conversion.placeBlock(state.normalized_angular_momentum, inThisMpcIf(centroidal), "normalized_angular_momentum", /*start=*/3);
  conversion.placeBlock(state.base_position, InThisMpc::kPresent, "base_position", basePoseStart);
  conversion.placeBlock(state.base_orientation, InThisMpc::kPresent, "base_orientation", basePoseStart + 3);
  conversion.placeJoints(state.joint_positions, InThisMpc::kPresent, "joint_positions", jointPositionsStart);
  conversion.placeBlock(state.base_linear_velocity, inThisMpcIf(!centroidal), "base_linear_velocity", baseVelocityStart);
  conversion.placeBlock(state.base_angular_velocity, inThisMpcIf(!centroidal), "base_angular_velocity", baseVelocityStart + 3);
  conversion.placeJoints(state.joint_velocities, inThisMpcIf(!centroidal), "joint_velocities", jointVelocitiesStart);
}

/**
 * Places the blocks of an input weight in the vector of `conversion` (the wrench of each contact, then the joints'), on
 * StateInputLayout.
 */
void placeInput(const mpc_config::InputWeights& weights, LayoutConversion& conversion) {
  // Each contact's wrench at six times its index in the layout's contacts.
  absl::flat_hash_map<std::string, Eigen::Index> contactIndices;
  for (size_t i = 0; i < conversion.layout().contactNames.size(); ++i) {
    contactIndices.emplace(conversion.layout().contactNames[i], static_cast<Eigen::Index>(i));
  }
  const std::string contactsPath = conversion.path("contact_wrenches");
  std::vector<bool> given(conversion.layout().contactNames.size(), false);
  std::vector<std::string> twice;
  for (const mpc_config::InputWeights::ContactWrench& wrench : weights.contact_wrenches) {
    const absl::flat_hash_map<std::string, Eigen::Index>::const_iterator found = contactIndices.find(wrench.contact);
    if (found == contactIndices.end()) {
      conversion.addProblem(absl::StrCat(contactsPath, ": '", wrench.contact, "' is not a contact of the MPC model (its contacts: ",
                                         absl::StrJoin(conversion.layout().contactNames, ", "), ")"));
      continue;
    }
    const Eigen::Index index = found->second;
    if (given[static_cast<size_t>(index)]) {
      twice.push_back(wrench.contact);
      continue;
    }
    given[static_cast<size_t>(index)] = true;
    const std::string element = absl::StrCat("contact_wrenches[contact=", wrench.contact, "]");
    conversion.placeBlock(wrench.force, InThisMpc::kPresent, absl::StrCat(element, ".force"), kWrenchDimension * index);
    conversion.placeBlock(wrench.moment, InThisMpc::kPresent, absl::StrCat(element, ".moment"), kWrenchDimension * index + kMomentOffset);
  }
  conversion.addRepeated(contactsPath, "contact", twice);
  std::vector<std::string> missing;
  for (size_t i = 0; i < given.size(); ++i) {
    if (!given[i]) missing.push_back(conversion.layout().contactNames[i]);
  }
  if (!missing.empty()) {
    conversion.addProblem(absl::StrCat(contactsPath, ": missing ", absl::StrJoin(missing, ", "), " (every contact exactly once)"));
  }

  // Then the joints: their velocities (centroidal) or their accelerations (whole body).
  const bool centroidal = conversion.layout().mpc == StateInputLayout::Mpc::kCentroidal;
  const Eigen::Index jointsStart = kWrenchDimension * static_cast<Eigen::Index>(conversion.layout().contactNames.size());
  conversion.placeJoints(weights.joint_velocities, inThisMpcIf(centroidal), "joint_velocities", jointsStart);
  conversion.placeJoints(weights.joint_accelerations, inThisMpcIf(!centroidal), "joint_accelerations", jointsStart);
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h:state_input_indices, //humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/common/WBAccelMpcRobotModel.h:state_input_indices, //humanoid_nmpc/humanoid_common_mpc/src/config/weights/StateInputLayout.cpp:state_input_dimensions, //humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/weights/StateInputLayout.h)
// clang-format on

/**
 * The diagonal 3x3 weight of `scaling` and the three optional `weights` named `names` (of the block `fieldPath`): a
 * component left out weighs 0, a block that names none of them is refused.
 */
absl::StatusOr<matrix_t> threeWeights(scalar_t scaling,
                                      const std::array<std::optional<double>, 3>& weights,
                                      const std::array<absl::string_view, 3>& names,
                                      absl::string_view fieldPath) {
  std::vector<std::string> problems;
  if (!std::isfinite(scaling)) {
    problems.push_back(absl::StrCat(fieldPath, ".scaling is ", scaling, ", not a finite number"));
  }
  vector_t diagonal = vector_t::Zero(3);
  bool named = false;
  for (size_t i = 0; i < weights.size(); ++i) {
    const std::optional<double> weight = weights[i];
    if (!weight.has_value()) continue;
    named = true;
    if (!std::isfinite(*weight)) {
      problems.push_back(absl::StrCat(fieldPath, ".", names[i], " is ", *weight, ", not a finite number"));
    }
    diagonal(static_cast<Eigen::Index>(i)) = *weight;
  }
  if (!named) {
    problems.push_back(absl::StrCat(fieldPath, " names none of ", absl::StrJoin(names, ", "),
                                    ": com_and_acom_tracking_cost would weigh nothing (a component left out weighs 0)"));
  }
  if (!problems.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(kErrorPrefix, absl::StrJoin(problems, "; ")));
  }
  return scaledDiagonal(scaling, diagonal);
}

}  // namespace

absl::StatusOr<matrix_t> stateWeightsFromConfig(const mpc_config::StateWeights& weights,
                                                const StateInputLayout& layout,
                                                absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "state", static_cast<Eigen::Index>(stateDimension(layout)));
  conversion.checkFinite(weights.scaling, conversion.path("scaling"));
  placeState(weights, conversion);
  ASSIGN_OR_RETURN(const vector_t diagonal, conversion.values());
  return scaledDiagonal(weights.scaling, diagonal);
}

absl::StatusOr<vector_t> stateValuesFromConfig(const mpc_config::StateValues& values,
                                               const StateInputLayout& layout,
                                               absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "state", static_cast<Eigen::Index>(stateDimension(layout)));
  placeState(values, conversion);
  return conversion.values();
}

absl::StatusOr<matrix_t> inputWeightsFromConfig(const mpc_config::InputWeights& weights,
                                                const StateInputLayout& layout,
                                                absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "input", static_cast<Eigen::Index>(inputDimension(layout)));
  conversion.checkFinite(weights.scaling, conversion.path("scaling"));
  placeInput(weights, conversion);
  ASSIGN_OR_RETURN(const vector_t diagonal, conversion.values());
  return scaledDiagonal(weights.scaling, diagonal);
}

absl::StatusOr<matrix_t> comWeightsFromConfig(const mpc_config::ComWeights& weights, absl::string_view fieldPath) {
  return threeWeights(weights.scaling, {weights.x, weights.y, weights.z}, kXyzComponents, fieldPath);
}

absl::StatusOr<matrix_t> acomWeightsFromConfig(const mpc_config::AcomWeights& weights, absl::string_view fieldPath) {
  return threeWeights(weights.scaling, {weights.yaw, weights.pitch, weights.roll}, kYawPitchRollComponents, fieldPath);
}

absl::StatusOr<scalar_t> terminalCostScalingFromConfig(const mpc_config::TaskFile& task) {
  if (!task.terminal_cost_scaling.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat(kErrorPrefix,
                     "terminal_cost_scaling is missing: the factor of final_state_weights (terminal_cost) and of the terminal CoM and "
                     "ACoM weights (com_and_acom_tracking_cost) has no default"));
  }
  if (!std::isfinite(*task.terminal_cost_scaling)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kErrorPrefix, "terminal_cost_scaling is ", *task.terminal_cost_scaling, ", not a finite number"));
  }
  return *task.terminal_cost_scaling;
}

absl::StatusOr<ExternalTorqueQuadraticCostAD::Config> legTorqueCostFromConfig(const mpc_config::JointWeights& weights,
                                                                              const StateInputLayout& layout,
                                                                              absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "joints", /*size=*/0);
  conversion.checkFinite(weights.scaling, conversion.path("scaling"));
  const std::string listPath = conversion.path("joints");
  if (weights.joints.empty()) {
    conversion.addProblem(absl::StrCat(listPath, " names no joint (a leg's torque cost weighs at least one)"));
  }
  ExternalTorqueQuadraticCostAD::Config config;
  std::vector<scalar_t> scaled;
  absl::flat_hash_set<std::string> listed;
  std::vector<std::string> twice;
  for (const mpc_config::JointValue& entry : weights.joints) {
    if (!conversion.jointIndex(entry.joint, listPath).has_value()) continue;
    if (!listed.insert(entry.joint).second) {
      twice.push_back(entry.joint);
      continue;
    }
    conversion.checkFinite(entry.value, absl::StrCat(listPath, "[joint=", entry.joint, "].value"));
    config.activeJointNames.push_back(entry.joint);
    scaled.push_back(weights.scaling * entry.value);
  }
  conversion.addRepeated(listPath, "joint", twice);
  RETURN_IF_ERROR(conversion.status());
  config.weights = Eigen::Map<const vector_t>(scaled.data(), static_cast<Eigen::Index>(scaled.size()));
  return config;
}

absl::StatusOr<vector_t> jointTorqueWeightsFromConfig(const mpc_config::JointWeights& weights,
                                                      const StateInputLayout& layout,
                                                      absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "joints", static_cast<Eigen::Index>(layout.jointNames.size()));
  conversion.checkFinite(weights.scaling, conversion.path("scaling"));
  conversion.placeJoints(weights.joints, InThisMpc::kPresent, "joints", /*start=*/0);
  ASSIGN_OR_RETURN(const vector_t values, conversion.values());
  vector_t scaled(values.size());
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    scaled(i) = weights.scaling * values(i);
  }
  return scaled;
}

}  // namespace ocs2::humanoid
