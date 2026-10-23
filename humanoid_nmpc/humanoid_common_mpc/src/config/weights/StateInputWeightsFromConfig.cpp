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
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/cost/ComAndAcomTrackingCost.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
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
// rates' (Euler ZYX), as many as a rotation block of the tuning layout has rows.
constexpr int kBlockComponents = kPositionDim;
static_assert(static_cast<size_t>(kBlockComponents) == kRotationTuningDim, "an orientation block has a vector's components");
constexpr std::array<absl::string_view, kBlockComponents> kXyzComponents = {"x", "y", "z"};
constexpr std::array<absl::string_view, kBlockComponents> kYawPitchRollComponents = {"yaw", "pitch", "roll"};

std::array<double, kBlockComponents> components(const mpc_config::Xyz& block) {
  return {block.x, block.y, block.z};
}

std::array<double, kBlockComponents> components(const mpc_config::YawPitchRoll& block) {
  return {block.yaw, block.pitch, block.roll};
}

const std::array<absl::string_view, kBlockComponents>& componentNames(const mpc_config::Xyz& /*block*/) {
  return kXyzComponents;
}

const std::array<absl::string_view, kBlockComponents>& componentNames(const mpc_config::YawPitchRoll& /*block*/) {
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
      matrix(row, col) = scaling * (row == col ? diagonal(row) : 0.0);
    }
  }
  return matrix;
}

/**
 * What the numbers of a block are: weights of a quadratic cost, which must be at least 0 so that the cost's Hessian
 * (and the Gauss-Newton one the solver builds from it) stays positive semidefinite, or values of a state, which may
 * have any sign.
 */
enum class Numbers { kWeights, kValues };

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
   * @param numbers Whether the block's numbers are weights, which checkNumber() refuses below 0, or state values.
   */
  LayoutConversion(StateInputLayout layout, absl::string_view fieldPath, absl::string_view vectorName, Eigen::Index size, Numbers numbers)
      : layout_(std::move(layout)), fieldPath_(fieldPath), vectorName_(vectorName), numbers_(numbers), values_(vector_t::Zero(size)) {
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

  /** Refuses `value` at `path` unless it is finite and, for weights, at least 0 (negated, so that NaN is refused). */
  void checkNumber(double value, absl::string_view path) {
    if (!std::isfinite(value)) {
      addProblem(absl::StrCat(path, " is ", value, ", not a finite number"));
    } else if (numbers_ == Numbers::kWeights && !(value >= 0.0)) {
      addProblem(absl::StrCat(path, " is ", value, ", but a weight is at least 0 (a negative one makes the cost indefinite)"));
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
    const std::array<double, kBlockComponents> given = components(*block);
    const std::array<absl::string_view, kBlockComponents>& names = componentNames(*block);
    for (size_t i = 0; i < given.size(); ++i) {
      checkNumber(given[i], absl::StrCat(blockPath, ".", names[i]));
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
      checkNumber(entry.value, absl::StrCat(listPath, "[joint=", entry.joint, "].value"));
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
  std::optional<Eigen::Index> jointIndex(absl::string_view joint, absl::string_view listPath) {
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
  Numbers numbers_;
  // The vector the placed components are written into, of the size the conversion was made with.
  vector_t values_;
  absl::flat_hash_map<std::string, Eigen::Index> jointIndices_;
  absl::flat_hash_set<std::string> fixedJoints_;
  std::vector<std::string> problems_;
};

/** The first row of the block `name` in `tuningLayout`, or nullopt when the MPC's state has no such block. */
std::optional<Eigen::Index> tuningRowOf(const TuningLayout& tuningLayout, absl::string_view name) {
  const absl::StatusOr<size_t> row = tuningSegmentStart(tuningLayout, name);
  if (!row.ok()) return std::nullopt;
  return static_cast<Eigen::Index>(*row);
}

/** Places the block `name` of a state at its row of `tuningLayout`, or refuses it when the layout has no such block. */
template <typename Block>
void placeStateBlock(std::optional<Block> block, absl::string_view name, const TuningLayout& tuningLayout, LayoutConversion& conversion) {
  if (const std::optional<Eigen::Index> row = tuningRowOf(tuningLayout, name); row.has_value()) {
    conversion.placeBlock(std::move(block), InThisMpc::kPresent, name, *row);
  } else {
    conversion.placeBlock(std::move(block), InThisMpc::kAbsent, name, /*start=*/0);
  }
}

/** Places the joint list `name` of a state at its row of `tuningLayout`, or refuses it when the layout has no such block. */
void placeStateJoints(const std::vector<mpc_config::JointValue>& entries,
                      absl::string_view name,
                      const TuningLayout& tuningLayout,
                      LayoutConversion& conversion) {
  if (const std::optional<Eigen::Index> row = tuningRowOf(tuningLayout, name); row.has_value()) {
    conversion.placeJoints(entries, InThisMpc::kPresent, name, *row);
  } else {
    conversion.placeJoints(entries, InThisMpc::kAbsent, name, /*start=*/0);
  }
}

/**
 * Places the blocks of a state (a StateWeights or a StateValues, which share their fields) in the vector of
 * `conversion`, each at the first row of its block of the same name in tuningLayoutOf() of the conversion's layout: the
 * centroidal momenta, base pose and joint positions, or the whole-body base pose, joint positions, base velocity and
 * joint velocities. A block the MPC's tuning layout does not have is refused.
 */
template <typename State>
void placeState(const State& state, LayoutConversion& conversion) {
  const TuningLayout tuningLayout = tuningLayoutOf(conversion.layout());
  placeStateBlock(state.normalized_linear_momentum, "normalized_linear_momentum", tuningLayout, conversion);
  placeStateBlock(state.normalized_angular_momentum, "normalized_angular_momentum", tuningLayout, conversion);
  placeStateBlock(state.base_position, "base_position", tuningLayout, conversion);
  placeStateBlock(state.base_orientation, "base_orientation", tuningLayout, conversion);
  placeStateJoints(state.joint_positions, "joint_positions", tuningLayout, conversion);
  placeStateBlock(state.base_linear_velocity, "base_linear_velocity", tuningLayout, conversion);
  placeStateBlock(state.base_angular_velocity, "base_angular_velocity", tuningLayout, conversion);
  placeStateJoints(state.joint_velocities, "joint_velocities", tuningLayout, conversion);
}

// The offsets of the blocks of the input in its vector, as the MPCs' robot models index them (StateInputLayout). The
// state's blocks need none: placeState() puts each at its row of the tuning layout.
// LINT.IfChange(input_offsets)
// The wrench of a contact in the input is kWrenchDim long: its force at 0, its moment at kForceDim.

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
    conversion.placeBlock(wrench.force, InThisMpc::kPresent, absl::StrCat(element, ".force"), kWrenchDim * index);
    conversion.placeBlock(wrench.moment, InThisMpc::kPresent, absl::StrCat(element, ".moment"), kWrenchDim * index + kForceDim);
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
  const Eigen::Index jointsStart = kWrenchDim * static_cast<Eigen::Index>(conversion.layout().contactNames.size());
  conversion.placeJoints(weights.joint_velocities, inThisMpcIf(centroidal), "joint_velocities", jointsStart);
  conversion.placeJoints(weights.joint_accelerations, inThisMpcIf(!centroidal), "joint_accelerations", jointsStart);
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/include/humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h:input_indices, //humanoid_nmpc/humanoid_wb_mpc/include/humanoid_wb_mpc/common/WBAccelMpcRobotModel.h:input_indices, //humanoid_nmpc/humanoid_common_mpc/src/config/weights/StateInputLayout.cpp:state_input_dimensions, //humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/config/weights/StateInputLayout.h)
// clang-format on

/**
 * The diagonal 3x3 weight of `scaling` and the three optional `weights` named `names` (of the block `fieldPath`): a
 * component left out weighs 0, a block that names none of them is refused.
 */
absl::StatusOr<matrix_t> threeWeights(scalar_t scaling,
                                      const std::array<std::optional<double>, kBlockComponents>& weights,
                                      const std::array<absl::string_view, kBlockComponents>& names,
                                      absl::string_view fieldPath) {
  std::vector<std::string> problems;
  // Negated, so that NaN is refused: a negative weight makes the tracking cost indefinite.
  if (!std::isfinite(scaling) || !(scaling >= 0.0)) {
    problems.push_back(absl::StrCat(fieldPath, ".scaling is ", scaling, ", not a finite number of at least 0"));
  }
  vector_t diagonal = vector_t::Zero(kBlockComponents);
  bool named = false;
  for (size_t i = 0; i < weights.size(); ++i) {
    const std::optional<double> weight = weights[i];
    if (!weight.has_value()) continue;
    named = true;
    if (!std::isfinite(*weight) || !(*weight >= 0.0)) {
      problems.push_back(absl::StrCat(fieldPath, ".", names[i], " is ", *weight, ", not a finite number of at least 0"));
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
  LayoutConversion conversion(layout, fieldPath, "state", static_cast<Eigen::Index>(tuningDimension(layout)), Numbers::kWeights);
  conversion.checkNumber(weights.scaling, conversion.path("scaling"));
  placeState(weights, conversion);
  ASSIGN_OR_RETURN(const vector_t diagonal, conversion.values());
  return scaledDiagonal(weights.scaling, diagonal);
}

absl::StatusOr<matrix_t> trackingStateWeightsFromConfig(const mpc_config::StateWeights& weights,
                                                        const StateInputLayout& layout,
                                                        absl::string_view fieldPath,
                                                        BasePoseWeights basePose) {
  ASSIGN_OR_RETURN(matrix_t Q, stateWeightsFromConfig(weights, layout, fieldPath));
  if (basePose == BasePoseWeights::kZeroedForComAndAcomTracking) ComAndAcomTrackingCost::zeroBasePoseWeights(Q);
  return Q;
}

absl::StatusOr<matrix_t> terminalStateWeightsFromConfig(const mpc_config::StateWeights& finalStateWeights,
                                                        scalar_t terminalCostScaling,
                                                        const StateInputLayout& layout,
                                                        absl::string_view fieldPath,
                                                        BasePoseWeights basePose) {
  ASSIGN_OR_RETURN(matrix_t Q_final, trackingStateWeightsFromConfig(finalStateWeights, layout, fieldPath, basePose));
  Q_final *= terminalCostScaling;
  return Q_final;
}

absl::StatusOr<vector_t> stateValuesFromConfig(const mpc_config::StateValues& values,
                                               const StateInputLayout& layout,
                                               absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "state", static_cast<Eigen::Index>(tuningDimension(layout)), Numbers::kValues);
  placeState(values, conversion);
  return conversion.values();
}

absl::StatusOr<matrix_t> inputWeightsFromConfig(const mpc_config::InputWeights& weights,
                                                const StateInputLayout& layout,
                                                absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "input", static_cast<Eigen::Index>(inputDimension(layout)), Numbers::kWeights);
  conversion.checkNumber(weights.scaling, conversion.path("scaling"));
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
  // Negated, so that NaN is refused: the factor of weights, which a negative one makes indefinite.
  if (!std::isfinite(*task.terminal_cost_scaling) || !(*task.terminal_cost_scaling >= 0.0)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kErrorPrefix, "terminal_cost_scaling is ", *task.terminal_cost_scaling, ", not a finite number of at least 0"));
  }
  return *task.terminal_cost_scaling;
}

absl::StatusOr<ExternalTorqueQuadraticCostAD::Config> legTorqueCostFromConfig(const mpc_config::JointWeights& weights,
                                                                              const StateInputLayout& layout,
                                                                              absl::string_view fieldPath) {
  LayoutConversion conversion(layout, fieldPath, "joints", /*size=*/0, Numbers::kWeights);
  conversion.checkNumber(weights.scaling, conversion.path("scaling"));
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
    conversion.checkNumber(entry.value, absl::StrCat(listPath, "[joint=", entry.joint, "].value"));
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
  LayoutConversion conversion(layout, fieldPath, "joints", static_cast<Eigen::Index>(layout.jointNames.size()), Numbers::kWeights);
  conversion.checkNumber(weights.scaling, conversion.path("scaling"));
  conversion.placeJoints(weights.joints, InThisMpc::kPresent, "joints", /*start=*/0);
  // The cost weighs each torque by the square root of its weight, so a negative weight would be a NaN in the cost: the
  // conversion refuses a weight or scaling below 0 (Numbers::kWeights).
  ASSIGN_OR_RETURN(const vector_t values, conversion.values());
  return vector_t(weights.scaling * values);
}

}  // namespace ocs2::humanoid
