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

#include "humanoid_mpc_ipc/MpcMessageConversions.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"

#include "humanoid_mpc_ipc/PolicyControllers.h"
#include "humanoid_mpc_msgs/controller_type.pb.h"
#include "humanoid_mpc_msgs/vector.pb.h"

// absl::Status propagation for this file. humanoid_common_mpc's StatusMacros.h
// is not used because this package builds on OCS2 and the messages alone.
#define IPC_RETURN_IF_ERROR(expr)           \
  do {                                      \
    absl::Status ipc_macro_status = (expr); \
    if (!ipc_macro_status.ok()) {           \
      return ipc_macro_status;              \
    }                                       \
  } while (0)

namespace ocs2::humanoid::ipc {
namespace {

using DoubleField = google::protobuf::RepeatedField<double>;
using IndexField = google::protobuf::RepeatedField<uint64_t>;
using VectorField =
    google::protobuf::RepeatedPtrField<humanoid_mpc_msgs::Vector>;

// Modes and post-event indices are size_t in OCS2 and uint64 on the wire; both
// directions are lossless only if the two have the same range.
static_assert(std::numeric_limits<size_t>::digits ==
                  std::numeric_limits<uint64_t>::digits,
              "size_t and uint64_t must have the same range for modes and "
              "post-event indices to round-trip");

// The paths of the policy's parts in error messages.
constexpr absl::string_view kPolicy = "MpcPolicy";
constexpr absl::string_view kPolicyInitObservation =
    "MpcPolicy.init_observation";
constexpr absl::string_view kPolicyTargetTrajectories =
    "MpcPolicy.target_trajectories";
constexpr absl::string_view kPolicyModeSchedule = "MpcPolicy.mode_schedule";

// =====================================================================================================================
// Writing. Every writer resizes the field it fills, so a field that already has
// the capacity does not allocate.
// =====================================================================================================================

void writeDoubles(const double* absl_nullable values,
                  size_t size,
                  DoubleField* absl_nonnull field) {
  field->resize(static_cast<int>(size));
  std::copy_n(values, size, field->mutable_data());
}

void writeVector(const vector_t& vector,
                 humanoid_mpc_msgs::Vector* absl_nonnull message) {
  writeDoubles(vector.data(), static_cast<size_t>(vector.size()),
               message->mutable_data());
}

void writeScalars(const scalar_array_t& values,
                  DoubleField* absl_nonnull field) {
  writeDoubles(values.data(), values.size(), field);
}

void writeIndices(const std::vector<size_t>& values,
                  IndexField* absl_nonnull field) {
  field->resize(static_cast<int>(values.size()));
  std::copy(values.begin(), values.end(), field->mutable_data());
}

// Gives `field` exactly `size` elements. RepeatedPtrField keeps the elements
// RemoveLast() takes away for Add() to hand out again, and each kept element
// keeps the capacity of its data, so this allocates only when the field grows
// past the number of elements it ever held.
void resizeVectors(size_t size, VectorField* absl_nonnull field) {
  const int target = static_cast<int>(size);
  while (field->size() > target) {
    field->RemoveLast();
  }
  while (field->size() < target) {
    field->Add();
  }
}

void writeVectors(const vector_array_t& vectors,
                  VectorField* absl_nonnull field) {
  resizeVectors(vectors.size(), field);
  for (size_t k = 0; k < vectors.size(); ++k) {
    writeVector(vectors[k], field->Mutable(static_cast<int>(k)));
  }
}

// The nodes of a LinearController in flattenSingle()'s layout: per input i, its
// bias followed by row i of the gain.
void writeLinearControllerNodes(const LinearController& controller,
                                VectorField* absl_nonnull field) {
  resizeVectors(controller.timeStamp_.size(), field);
  for (size_t k = 0; k < controller.timeStamp_.size(); ++k) {
    const vector_t& bias = controller.biasArray_[k];
    const matrix_t& gain = controller.gainArray_[k];
    const Eigen::Index stride = gain.cols() + 1;
    DoubleField* absl_nonnull data =
        field->Mutable(static_cast<int>(k))->mutable_data();
    data->resize(static_cast<int>(bias.size() * stride));
    double* absl_nullable out = data->mutable_data();
    for (Eigen::Index i = 0; i < bias.size(); ++i) {
      out[i * stride] = bias(i);
      Eigen::Map<Eigen::RowVectorXd>(out + i * stride + 1, gain.cols()) =
          gain.row(i);
    }
  }
}

// LINT.IfChange(manifold_node_writer)
// The nodes of a ManifoldLinearController: [u*_k, xbar_k, vec K_k], K_k
// column-major, which its flattenSingle() writes at a node time. The anchors
// are the controller's own, not the state trajectory.
void writeManifoldLinearControllerNodes(
    const ManifoldLinearController& controller,
    VectorField* absl_nonnull field) {
  const size_t nodes = controller.getTimeStamps().size();
  resizeVectors(nodes, field);
  for (size_t k = 0; k < nodes; ++k) {
    const vector_t& nominalInput = controller.getNominalInputs()[k];
    const vector_t& anchorState = controller.getAnchorStates()[k];
    const matrix_t& gain = controller.getFeedbackGains()[k];
    DoubleField* absl_nonnull data =
        field->Mutable(static_cast<int>(k))->mutable_data();
    data->resize(static_cast<int>(nominalInput.size() + anchorState.size() +
                                  gain.size()));
    double* absl_nullable out = data->mutable_data();
    std::copy_n(nominalInput.data(), nominalInput.size(), out);
    std::copy_n(anchorState.data(), anchorState.size(),
                out + nominalInput.size());
    std::copy_n(gain.data(), gain.size(),
                out + nominalInput.size() + anchorState.size());
  }
}
// LINT.ThenChange(:manifold_node_reader, //humanoid_nmpc/humanoid_mpc_msgs/mpc_policy.proto:controller_data)

// The controller sampled at `timeTrajectory` with ControllerBase::flatten(),
// for a controller on other time stamps. The flatten() interface writes
// std::vectors, so this allocates scratch arrays (and Eigen temporaries) on
// every call.
void sampleController(const ControllerBase& controller,
                      const scalar_array_t& timeTrajectory,
                      VectorField* absl_nonnull field) {
  std::vector<std::vector<double>> samples(timeTrajectory.size());
  std::vector<std::vector<double>* absl_nonnull> sampleRefs;
  sampleRefs.reserve(samples.size());
  for (std::vector<double>& sample : samples) {
    sampleRefs.push_back(&sample);
  }
  controller.flatten(timeTrajectory, sampleRefs);
  resizeVectors(samples.size(), field);
  for (size_t k = 0; k < samples.size(); ++k) {
    writeDoubles(samples[k].data(), samples[k].size(),
                 field->Mutable(static_cast<int>(k))->mutable_data());
  }
}

// What policyToProto() needs of a ManifoldLinearController: consistent arrays,
// nodes of its manifold's dimensions, and the time trajectory as its time
// stamps, since it travels only as its own nodes.
absl::Status checkManifoldLinearController(
    const ManifoldLinearController& controller,
    const scalar_array_t& timeTrajectory) {
  const size_t nodes = controller.getTimeStamps().size();
  if (controller.getAnchorStates().size() != nodes ||
      controller.getNominalInputs().size() != nodes ||
      controller.getFeedbackGains().size() != nodes) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the ManifoldLinearController has ", nodes, " time stamps, ",
        controller.getAnchorStates().size(), " anchor states, ",
        controller.getNominalInputs().size(), " nominal inputs and ",
        controller.getFeedbackGains().size(), " gains"));
  }
  if (nodes == 0) {
    return absl::OkStatus();  // checkController() refuses an empty controller
                              // next to a non-empty time trajectory
  }
  if (controller.getTimeStamps() != timeTrajectory) {
    return absl::InvalidArgumentError(
        "the ManifoldLinearController's time stamps are not the time "
        "trajectory; it travels only as its own nodes, which "
        "must be the policy's (the multiple-shooting solvers build it on the "
        "time trajectory)");
  }
  const StateManifold* absl_nullable manifold =
      controller.getStateManifold().get();
  if (manifold == nullptr) {
    return absl::InvalidArgumentError(
        "the ManifoldLinearController has no state manifold");
  }
  const Eigen::Index ambientDim =
      static_cast<Eigen::Index>(manifold->getAmbientDim());
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(manifold->getTangentDim());
  for (size_t k = 0; k < nodes; ++k) {
    const vector_t& anchorState = controller.getAnchorStates()[k];
    const matrix_t& gain = controller.getFeedbackGains()[k];
    if (anchorState.size() != ambientDim ||
        gain.rows() != controller.getNominalInputs()[k].size() ||
        gain.cols() != tangentDim) {
      return absl::InvalidArgumentError(absl::StrCat(
          "node ", k,
          " of the ManifoldLinearController has an anchor state of ",
          anchorState.size(), " entries, a nominal input of ",
          controller.getNominalInputs()[k].size(), " and a gain of ",
          gain.rows(), "x", gain.cols(), "; its state manifold has ",
          ambientDim, " ambient and ", tangentDim, " tangent dimensions"));
    }
  }
  return absl::OkStatus();
}

// What policyToProto() needs of the controller, checked before it writes
// anything.
absl::Status checkController(const PrimalSolution& primalSolution) {
  const ControllerBase* absl_nullable controller =
      primalSolution.controllerPtr_.get();
  if (controller == nullptr) {
    return absl::InvalidArgumentError(
        "PrimalSolution.controllerPtr_ is null; a policy needs a controller");
  }
  if (const FeedforwardController* absl_nullable feedforward =
          asFeedforwardController(*controller);
      feedforward != nullptr) {
    if (feedforward->uffArray_.size() != feedforward->timeStamp_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("the FeedforwardController has ",
                       feedforward->timeStamp_.size(), " time stamps but ",
                       feedforward->uffArray_.size(), " feedforward inputs"));
    }
  } else if (const LinearController* absl_nullable linear =
                 asLinearController(*controller);
             linear != nullptr) {
    if (linear->biasArray_.size() != linear->timeStamp_.size() ||
        linear->gainArray_.size() != linear->timeStamp_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("the LinearController has ", linear->timeStamp_.size(),
                       " time stamps, ", linear->biasArray_.size(),
                       " biases and ", linear->gainArray_.size(), " gains"));
    }
    for (size_t k = 0; k < linear->timeStamp_.size(); ++k) {
      if (linear->gainArray_[k].rows() != linear->biasArray_[k].size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("node ", k, " of the LinearController has a gain of ",
                         linear->gainArray_[k].rows(), " rows but a bias of ",
                         linear->biasArray_[k].size(), " entries"));
      }
    }
  } else if (const ManifoldLinearController* absl_nullable manifoldLinear =
                 asManifoldLinearController(*controller);
             manifoldLinear != nullptr) {
    IPC_RETURN_IF_ERROR(checkManifoldLinearController(
        *manifoldLinear, primalSolution.timeTrajectory_));
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "PrimalSolution.controllerPtr_ is an ocs2::ControllerType ",
        static_cast<int>(controller->getType()),
        " controller; only a FeedforwardController, a LinearController or a "
        "ManifoldLinearController can be sent"));
  }
  if (controller->empty() && !primalSolution.timeTrajectory_.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PrimalSolution.controllerPtr_ is empty but the time trajectory has ",
        primalSolution.timeTrajectory_.size(), " nodes"));
  }
  return absl::OkStatus();
}

// Requires checkController(primalSolution) to have passed.
void writeController(const PrimalSolution& primalSolution,
                     humanoid_mpc_msgs::MpcPolicy* absl_nonnull message) {
  const ControllerBase& controller = *primalSolution.controllerPtr_;
  VectorField* absl_nonnull field = message->mutable_controller_data();
  if (const FeedforwardController* absl_nullable feedforward =
          asFeedforwardController(controller);
      feedforward != nullptr) {
    message->set_controller_type(
        humanoid_mpc_msgs::CONTROLLER_TYPE_FEEDFORWARD);
    if (feedforward->timeStamp_ == primalSolution.timeTrajectory_) {
      writeVectors(feedforward->uffArray_, field);
      return;
    }
  } else if (const LinearController* absl_nullable linear =
                 asLinearController(controller);
             linear != nullptr) {
    message->set_controller_type(humanoid_mpc_msgs::CONTROLLER_TYPE_LINEAR);
    if (linear->timeStamp_ == primalSolution.timeTrajectory_) {
      writeLinearControllerNodes(*linear, field);
      return;
    }
  } else if (const ManifoldLinearController* absl_nullable manifoldLinear =
                 asManifoldLinearController(controller);
             manifoldLinear != nullptr) {
    // On the time trajectory: checkManifoldLinearController() refuses any other
    // time stamps.
    message->set_controller_type(
        humanoid_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR);
    writeManifoldLinearControllerNodes(*manifoldLinear, field);
    return;
  }
  sampleController(controller, primalSolution.timeTrajectory_, field);
}

// =====================================================================================================================
// Validation. The error messages name the field by its path in the message;
// they are built only on failure, so a valid message is checked without
// allocating.
// =====================================================================================================================

// The index of the first entry of `values` that is not finite, or -1 if they
// all are.
int firstNonFinite(const DoubleField& values) {
  for (int i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values.Get(i))) {
      return i;
    }
  }
  return -1;
}

absl::Status checkFinite(const DoubleField& values,
                         absl::string_view path,
                         absl::string_view field) {
  const int index = firstNonFinite(values);
  if (index < 0) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat(path, ".", field, "[", index, "] is ", values.Get(index),
                   "; every value must be finite"));
}

absl::Status checkFiniteVectors(const VectorField& vectors,
                                absl::string_view path,
                                absl::string_view field) {
  for (int k = 0; k < vectors.size(); ++k) {
    const int index = firstNonFinite(vectors.Get(k).data());
    if (index >= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          path, ".", field, "[", k, "].data[", index, "] is ",
          vectors.Get(k).data(index), "; every value must be finite"));
    }
  }
  return absl::OkStatus();
}

absl::Status checkSorted(const DoubleField& times,
                         absl::string_view path,
                         absl::string_view field) {
  for (int i = 1; i < times.size(); ++i) {
    if (times.Get(i) < times.Get(i - 1)) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ".", field, "[", i, "] = ", times.Get(i),
                       " is earlier than ", field, "[", i - 1,
                       "] = ", times.Get(i - 1), "; times must not decrease"));
    }
  }
  return absl::OkStatus();
}

absl::Status checkLength(int size,
                         absl::string_view path,
                         absl::string_view field,
                         int expectedSize,
                         absl::string_view expectedField) {
  if (size == expectedSize) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrCat(path, ".", field, " has ", size, " entries but ", path, ".",
                   expectedField, " has ", expectedSize));
}

absl::Status checkVectorSizes(const VectorField& vectors,
                              absl::string_view path,
                              absl::string_view field,
                              size_t expectedSize,
                              absl::string_view what) {
  for (int k = 0; k < vectors.size(); ++k) {
    if (static_cast<size_t>(vectors.Get(k).data_size()) != expectedSize) {
      return absl::InvalidArgumentError(absl::StrCat(
          path, ".", field, "[", k, "] has ", vectors.Get(k).data_size(),
          " entries; the model's ", what, " has ", expectedSize));
    }
  }
  return absl::OkStatus();
}

absl::Status checkMode(uint64_t mode,
                       absl::string_view path,
                       const ModelDimensions& dimensions) {
  if (mode < dimensions.numModes) {
    return absl::OkStatus();
  }
  if (dimensions.numModes == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        path, " is ", mode,
        "; the model has no modes (ModelDimensions::numModes is 0)"));
  }
  return absl::InvalidArgumentError(
      absl::StrCat(path, " is ", mode, "; the model's modes are 0 to ",
                   dimensions.numModes - 1));
}

absl::Status checkObservationSizes(
    const humanoid_mpc_msgs::SystemObservation& message,
    absl::string_view path,
    const ModelDimensions& dimensions) {
  if (static_cast<size_t>(message.state_size()) != dimensions.stateDim) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, ".state has ", message.state_size(),
                     " entries; the model's state has ", dimensions.stateDim));
  }
  if (static_cast<size_t>(message.input_size()) != dimensions.inputDim) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, ".input has ", message.input_size(),
                     " entries; the model's input has ", dimensions.inputDim));
  }
  return checkMode(message.mode(), absl::StrCat(path, ".mode"), dimensions);
}

absl::Status validateObservation(
    const humanoid_mpc_msgs::SystemObservation& message,
    absl::string_view path) {
  if (!std::isfinite(message.time())) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, ".time is ", message.time(), "; it must be finite"));
  }
  IPC_RETURN_IF_ERROR(checkFinite(message.state(), path, "state"));
  IPC_RETURN_IF_ERROR(checkFinite(message.input(), path, "input"));
  return absl::OkStatus();
}

absl::Status validateModeSchedule(
    const humanoid_mpc_msgs::ModeSchedule& message, absl::string_view path) {
  if (message.mode_sequence_size() == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        path,
        ".mode_sequence is empty; a mode schedule has at least one mode"));
  }
  if (message.mode_sequence_size() != message.event_times_size() + 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        path, ".mode_sequence has ", message.mode_sequence_size(),
        " modes but ", path, ".event_times has ", message.event_times_size(),
        " event times; a mode schedule has one mode more than event times"));
  }
  IPC_RETURN_IF_ERROR(checkFinite(message.event_times(), path, "event_times"));
  IPC_RETURN_IF_ERROR(checkSorted(message.event_times(), path, "event_times"));
  return absl::OkStatus();
}

absl::Status validateTargetTrajectories(
    const humanoid_mpc_msgs::TargetTrajectories& message,
    absl::string_view path) {
  IPC_RETURN_IF_ERROR(checkFinite(message.time(), path, "time"));
  IPC_RETURN_IF_ERROR(checkSorted(message.time(), path, "time"));
  IPC_RETURN_IF_ERROR(checkLength(message.state_size(), path, "state",
                                  message.time_size(), "time"));
  if (message.input_size() != 0 &&
      message.input_size() != message.time_size()) {
    return absl::InvalidArgumentError(
        absl::StrCat(path, ".input has ", message.input_size(), " entries but ",
                     path, ".time has ", message.time_size(),
                     "; target inputs are one per time or none"));
  }
  IPC_RETURN_IF_ERROR(checkFiniteVectors(message.state(), path, "state"));
  IPC_RETURN_IF_ERROR(checkFiniteVectors(message.input(), path, "input"));
  return absl::OkStatus();
}

absl::Status validatePostEventIndices(const IndexField& indices, int nodes) {
  for (int i = 0; i < indices.size(); ++i) {
    const uint64_t index = indices.Get(i);
    // A post-event index names the node after its pre-event node, so it is a
    // node of the policy: at most nodes - 1.
    if (index < 1 || index >= static_cast<uint64_t>(nodes)) {
      return absl::InvalidArgumentError(
          absl::StrCat(kPolicy, ".post_event_indices[", i, "] is ", index,
                       "; a post-event index lies in [1, ", nodes - 1,
                       "], a node after its pre-event node"));
    }
    if (i > 0 && index <= indices.Get(i - 1)) {
      return absl::InvalidArgumentError(absl::StrCat(
          kPolicy, ".post_event_indices[", i, "] = ", index,
          " does not exceed post_event_indices[", i - 1, "] = ",
          indices.Get(i - 1), "; post-event indices increase strictly"));
    }
  }
  return absl::OkStatus();
}

// Whether a policy of `type` can be evaluated on the receiver's state: a flat
// one (no manifold) or one on a manifold.
absl::Status validateControllerType(
    humanoid_mpc_msgs::ControllerType type,
    const StateManifold* absl_nullable stateManifold) {
  if (type != humanoid_mpc_msgs::CONTROLLER_TYPE_FEEDFORWARD &&
      type != humanoid_mpc_msgs::CONTROLLER_TYPE_LINEAR &&
      type != humanoid_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR) {
    return absl::InvalidArgumentError(absl::StrCat(
        kPolicy, ".controller_type is ", static_cast<int>(type),
        "; only CONTROLLER_TYPE_FEEDFORWARD, CONTROLLER_TYPE_LINEAR and "
        "CONTROLLER_TYPE_MANIFOLD_LINEAR can be evaluated"));
  }
  if (type == humanoid_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR &&
      stateManifold == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(kPolicy,
                     ".controller_type is CONTROLLER_TYPE_MANIFOLD_LINEAR, but "
                     "the receiver's state has "
                     "no state manifold to evaluate its feedback on: the MPC "
                     "and the robot disagree "
                     "on the state"));
  }
  if (type == humanoid_mpc_msgs::CONTROLLER_TYPE_LINEAR &&
      stateManifold != nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(kPolicy,
                     ".controller_type is CONTROLLER_TYPE_LINEAR, whose gains "
                     "act on the stored state, "
                     "but the receiver's state lives on a state manifold, "
                     "whose feedback policies are "
                     "CONTROLLER_TYPE_MANIFOLD_LINEAR"));
  }
  return absl::OkStatus();
}

// The states of a policy decoded on a state manifold have its ambient
// dimension.
absl::Status validateManifoldStates(
    const humanoid_mpc_msgs::MpcPolicy& message,
    const StateManifold* absl_nullable stateManifold) {
  if (stateManifold == nullptr) {
    return absl::OkStatus();
  }
  const size_t ambientDim = stateManifold->getAmbientDim();
  for (int k = 0; k < message.state_trajectory_size(); ++k) {
    if (static_cast<size_t>(message.state_trajectory(k).data_size()) !=
        ambientDim) {
      return absl::InvalidArgumentError(
          absl::StrCat(kPolicy, ".state_trajectory[", k, "] has ",
                       message.state_trajectory(k).data_size(),
                       " entries; the receiver's state manifold has ",
                       ambientDim, " ambient dimensions"));
    }
  }
  return absl::OkStatus();
}

// The size of each node's controller data against the dimensions of its state
// and input, and of the state manifold's tangent for a
// ManifoldLinearController. Requires the trajectories to have the same number
// of nodes, a controller type the receiver can evaluate
// (validateControllerType()), and for a ManifoldLinearController a manifold.
absl::Status validateControllerData(
    const humanoid_mpc_msgs::MpcPolicy& message,
    const StateManifold* absl_nullable stateManifold) {
  const humanoid_mpc_msgs::ControllerType type = message.controller_type();
  const size_t tangentDim =
      stateManifold == nullptr ? 0 : stateManifold->getTangentDim();
  for (int k = 0; k < message.controller_data_size(); ++k) {
    const size_t inputDim =
        static_cast<size_t>(message.input_trajectory(k).data_size());
    const size_t stateDim =
        static_cast<size_t>(message.state_trajectory(k).data_size());
    const size_t size =
        static_cast<size_t>(message.controller_data(k).data_size());
    if (type == humanoid_mpc_msgs::CONTROLLER_TYPE_FEEDFORWARD &&
        size != inputDim) {
      return absl::InvalidArgumentError(
          absl::StrCat(kPolicy, ".controller_data[", k, "] has ", size,
                       " entries but a FeedforwardController node with ",
                       inputDim, " inputs has ", inputDim));
    }
    if (type == humanoid_mpc_msgs::CONTROLLER_TYPE_LINEAR &&
        size != inputDim * (stateDim + 1)) {
      return absl::InvalidArgumentError(absl::StrCat(
          kPolicy, ".controller_data[", k, "] has ", size,
          " entries but a LinearController node with ", inputDim,
          " inputs and ", stateDim, " states has ", inputDim * (stateDim + 1)));
    }
    const size_t manifoldNodeSize = inputDim + stateDim + inputDim * tangentDim;
    if (type == humanoid_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR &&
        size != manifoldNodeSize) {
      return absl::InvalidArgumentError(
          absl::StrCat(kPolicy, ".controller_data[", k, "] has ", size,
                       " entries but a ManifoldLinearController node with ",
                       inputDim, " inputs, ", stateDim, " states and ",
                       tangentDim, " tangent dimensions has ", manifoldNodeSize,
                       " = inputs + states + inputs * tangent"));
    }
  }
  return absl::OkStatus();
}

absl::Status validatePrimalSolution(
    const humanoid_mpc_msgs::MpcPolicy& message,
    const StateManifold* absl_nullable stateManifold) {
  const int nodes = message.time_trajectory_size();
  if (nodes == 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        kPolicy, ".time_trajectory is empty; a policy has at least one node"));
  }
  IPC_RETURN_IF_ERROR(
      checkFinite(message.time_trajectory(), kPolicy, "time_trajectory"));
  IPC_RETURN_IF_ERROR(
      checkSorted(message.time_trajectory(), kPolicy, "time_trajectory"));
  IPC_RETURN_IF_ERROR(checkLength(message.state_trajectory_size(), kPolicy,
                                  "state_trajectory", nodes,
                                  "time_trajectory"));
  IPC_RETURN_IF_ERROR(checkLength(message.input_trajectory_size(), kPolicy,
                                  "input_trajectory", nodes,
                                  "time_trajectory"));
  IPC_RETURN_IF_ERROR(checkLength(message.controller_data_size(), kPolicy,
                                  "controller_data", nodes, "time_trajectory"));
  IPC_RETURN_IF_ERROR(checkFiniteVectors(message.state_trajectory(), kPolicy,
                                         "state_trajectory"));
  IPC_RETURN_IF_ERROR(checkFiniteVectors(message.input_trajectory(), kPolicy,
                                         "input_trajectory"));
  IPC_RETURN_IF_ERROR(checkFiniteVectors(message.controller_data(), kPolicy,
                                         "controller_data"));
  IPC_RETURN_IF_ERROR(
      validatePostEventIndices(message.post_event_indices(), nodes));
  IPC_RETURN_IF_ERROR(
      validateModeSchedule(message.mode_schedule(), kPolicyModeSchedule));
  IPC_RETURN_IF_ERROR(
      validateControllerType(message.controller_type(), stateManifold));
  IPC_RETURN_IF_ERROR(validateManifoldStates(message, stateManifold));
  IPC_RETURN_IF_ERROR(validateControllerData(message, stateManifold));
  return absl::OkStatus();
}

// =====================================================================================================================
// Reading. Called only on validated messages.
// =====================================================================================================================

Eigen::Map<const vector_t> asVector(const DoubleField& field) {
  return Eigen::Map<const vector_t>(field.data(), field.size());
}

// Eigen keeps a vector's storage when an assignment does not change its size,
// so reading into vectors of the right sizes allocates nothing.
void readVectors(const VectorField& field,
                 vector_array_t* absl_nonnull vectors) {
  vectors->resize(static_cast<size_t>(field.size()));
  for (int k = 0; k < field.size(); ++k) {
    (*vectors)[static_cast<size_t>(k)] = asVector(field.Get(k).data());
  }
}

void readObservation(const humanoid_mpc_msgs::SystemObservation& message,
                     SystemObservation* absl_nonnull observation) {
  observation->time = message.time();
  observation->state = asVector(message.state());
  observation->input = asVector(message.input());
  observation->mode = static_cast<size_t>(message.mode());
}

void readModeSchedule(const humanoid_mpc_msgs::ModeSchedule& message,
                      ModeSchedule* absl_nonnull modeSchedule) {
  modeSchedule->eventTimes.assign(message.event_times().begin(),
                                  message.event_times().end());
  modeSchedule->modeSequence.assign(message.mode_sequence().begin(),
                                    message.mode_sequence().end());
}

void readTargetTrajectories(
    const humanoid_mpc_msgs::TargetTrajectories& message,
    TargetTrajectories* absl_nonnull targetTrajectories) {
  targetTrajectories->timeTrajectory.assign(message.time().begin(),
                                            message.time().end());
  readVectors(message.state(), &targetTrajectories->stateTrajectory);
  readVectors(message.input(), &targetTrajectories->inputTrajectory);
}

void readPerformanceIndex(const humanoid_mpc_msgs::PerformanceIndex& message,
                          PerformanceIndex* absl_nonnull performanceIndex) {
  performanceIndex->merit = message.merit();
  performanceIndex->cost = message.cost();
  performanceIndex->dualFeasibilitiesSSE = message.dual_feasibilities_sse();
  performanceIndex->dynamicsViolationSSE = message.dynamics_violation_sse();
  performanceIndex->equalityConstraintsSSE = message.equality_constraints_sse();
  performanceIndex->inequalityConstraintsSSE =
      message.inequality_constraints_sse();
  performanceIndex->equalityLagrangian = message.equality_lagrangian();
  performanceIndex->inequalityLagrangian = message.inequality_lagrangian();
}

// LINT.IfChange(manifold_node_reader)
// The ManifoldLinearController of a validated CONTROLLER_TYPE_MANIFOLD_LINEAR
// policy, on `timeTrajectory` and `stateManifold`: the inverse of
// writeManifoldLinearControllerNodes(), as
// ManifoldLinearController::unFlatten() reads the same layout.
std::unique_ptr<ControllerBase> readManifoldLinearController(
    const humanoid_mpc_msgs::MpcPolicy& message,
    const scalar_array_t& timeTrajectory,
    std::shared_ptr<const StateManifold> stateManifold) {
  const size_t nodes = static_cast<size_t>(message.controller_data_size());
  const Eigen::Index tangentDim =
      static_cast<Eigen::Index>(stateManifold->getTangentDim());
  vector_array_t anchorStates(nodes);
  vector_array_t nominalInputs(nodes);
  matrix_array_t gains(nodes);
  for (size_t k = 0; k < nodes; ++k) {
    const int node = static_cast<int>(k);
    const Eigen::Index inputDim = message.input_trajectory(node).data_size();
    const Eigen::Index stateDim = message.state_trajectory(node).data_size();
    const double* absl_nullable data =
        message.controller_data(node).data().data();
    nominalInputs[k] = Eigen::Map<const vector_t>(data, inputDim);
    anchorStates[k] = Eigen::Map<const vector_t>(data + inputDim, stateDim);
    gains[k] = Eigen::Map<const matrix_t>(data + inputDim + stateDim, inputDim,
                                          tangentDim);
  }
  return std::make_unique<ManifoldLinearController>(
      timeTrajectory, std::move(anchorStates), std::move(nominalInputs),
      std::move(gains), std::move(stateManifold));
}
// LINT.ThenChange(:manifold_node_writer)

// The controller of a validated policy, on `timeTrajectory`: the inverse of
// FeedforwardController::flattenSingle() or LinearController::flattenSingle()
// node by node, as their unFlatten() reads it, without the intermediate
// std::vector copies unFlatten() would need; a ManifoldLinearController on
// `stateManifold`.
std::unique_ptr<ControllerBase> readController(
    const humanoid_mpc_msgs::MpcPolicy& message,
    const scalar_array_t& timeTrajectory,
    std::shared_ptr<const StateManifold> stateManifold) {
  const size_t nodes = static_cast<size_t>(message.controller_data_size());
  if (message.controller_type() ==
      humanoid_mpc_msgs::CONTROLLER_TYPE_MANIFOLD_LINEAR) {
    return readManifoldLinearController(message, timeTrajectory,
                                        std::move(stateManifold));
  }
  if (message.controller_type() == humanoid_mpc_msgs::CONTROLLER_TYPE_LINEAR) {
    vector_array_t bias(nodes);
    matrix_array_t gain(nodes);
    for (size_t k = 0; k < nodes; ++k) {
      const int node = static_cast<int>(k);
      const Eigen::Index inputDim = message.input_trajectory(node).data_size();
      const Eigen::Index stateDim = message.state_trajectory(node).data_size();
      const Eigen::Index stride = stateDim + 1;
      const double* absl_nullable data =
          message.controller_data(node).data().data();
      bias[k].resize(inputDim);
      gain[k].resize(inputDim, stateDim);
      for (Eigen::Index i = 0; i < inputDim; ++i) {
        bias[k](i) = data[i * stride];
        gain[k].row(i) = Eigen::Map<const Eigen::RowVectorXd>(
            data + i * stride + 1, stateDim);
      }
    }
    return std::make_unique<LinearController>(timeTrajectory, std::move(bias),
                                              std::move(gain));
  }
  vector_array_t feedforward;
  readVectors(message.controller_data(), &feedforward);
  return std::make_unique<FeedforwardController>(timeTrajectory,
                                                 std::move(feedforward));
}

void readPrimalSolution(const humanoid_mpc_msgs::MpcPolicy& message,
                        std::shared_ptr<const StateManifold> stateManifold,
                        PrimalSolution* absl_nonnull primalSolution) {
  primalSolution->timeTrajectory_.assign(message.time_trajectory().begin(),
                                         message.time_trajectory().end());
  readVectors(message.state_trajectory(), &primalSolution->stateTrajectory_);
  readVectors(message.input_trajectory(), &primalSolution->inputTrajectory_);
  primalSolution->postEventIndices_.assign(message.post_event_indices().begin(),
                                           message.post_event_indices().end());
  readModeSchedule(message.mode_schedule(), &primalSolution->modeSchedule_);
  primalSolution->controllerPtr_ = readController(
      message, primalSolution->timeTrajectory_, std::move(stateManifold));
}

}  // namespace

// =====================================================================================================================
// SystemObservation
// =====================================================================================================================

void toProto(const SystemObservation& observation,
             humanoid_mpc_msgs::SystemObservation* absl_nonnull message) {
  message->set_time(observation.time);
  writeDoubles(observation.state.data(),
               static_cast<size_t>(observation.state.size()),
               message->mutable_state());
  writeDoubles(observation.input.data(),
               static_cast<size_t>(observation.input.size()),
               message->mutable_input());
  message->set_mode(static_cast<uint64_t>(observation.mode));
}

absl::Status fromProto(const humanoid_mpc_msgs::SystemObservation& message,
                       SystemObservation* absl_nonnull observation) {
  IPC_RETURN_IF_ERROR(validateObservation(message, "SystemObservation"));
  readObservation(message, observation);
  return absl::OkStatus();
}

absl::Status checkDimensions(
    const humanoid_mpc_msgs::SystemObservation& message,
    const ModelDimensions& dimensions) {
  return checkObservationSizes(message, "SystemObservation", dimensions);
}

// =====================================================================================================================
// ModeSchedule
// =====================================================================================================================

void toProto(const ModeSchedule& modeSchedule,
             humanoid_mpc_msgs::ModeSchedule* absl_nonnull message) {
  writeScalars(modeSchedule.eventTimes, message->mutable_event_times());
  writeIndices(modeSchedule.modeSequence, message->mutable_mode_sequence());
}

absl::Status fromProto(const humanoid_mpc_msgs::ModeSchedule& message,
                       ModeSchedule* absl_nonnull modeSchedule) {
  IPC_RETURN_IF_ERROR(validateModeSchedule(message, "ModeSchedule"));
  readModeSchedule(message, modeSchedule);
  return absl::OkStatus();
}

// =====================================================================================================================
// TargetTrajectories
// =====================================================================================================================

void toProto(const TargetTrajectories& targetTrajectories,
             humanoid_mpc_msgs::TargetTrajectories* absl_nonnull message) {
  writeScalars(targetTrajectories.timeTrajectory, message->mutable_time());
  writeVectors(targetTrajectories.stateTrajectory, message->mutable_state());
  writeVectors(targetTrajectories.inputTrajectory, message->mutable_input());
}

absl::Status fromProto(const humanoid_mpc_msgs::TargetTrajectories& message,
                       TargetTrajectories* absl_nonnull targetTrajectories) {
  IPC_RETURN_IF_ERROR(
      validateTargetTrajectories(message, "TargetTrajectories"));
  readTargetTrajectories(message, targetTrajectories);
  return absl::OkStatus();
}

// =====================================================================================================================
// PerformanceIndex
// =====================================================================================================================

void toProto(const PerformanceIndex& performanceIndex,
             humanoid_mpc_msgs::PerformanceIndex* absl_nonnull message) {
  message->set_merit(performanceIndex.merit);
  message->set_cost(performanceIndex.cost);
  message->set_dual_feasibilities_sse(performanceIndex.dualFeasibilitiesSSE);
  message->set_dynamics_violation_sse(performanceIndex.dynamicsViolationSSE);
  message->set_equality_constraints_sse(
      performanceIndex.equalityConstraintsSSE);
  message->set_inequality_constraints_sse(
      performanceIndex.inequalityConstraintsSSE);
  message->set_equality_lagrangian(performanceIndex.equalityLagrangian);
  message->set_inequality_lagrangian(performanceIndex.inequalityLagrangian);
}

absl::Status fromProto(const humanoid_mpc_msgs::PerformanceIndex& message,
                       PerformanceIndex* absl_nonnull performanceIndex) {
  readPerformanceIndex(message, performanceIndex);
  return absl::OkStatus();
}

// =====================================================================================================================
// The policy
// =====================================================================================================================

absl::Status policyToProto(const CommandData& commandData,
                           const PrimalSolution& primalSolution,
                           const PerformanceIndex& performanceIndex,
                           humanoid_mpc_msgs::MpcPolicy* absl_nonnull message) {
  IPC_RETURN_IF_ERROR(checkController(primalSolution));
  toProto(commandData.mpcInitObservation_, message->mutable_init_observation());
  toProto(commandData.mpcTargetTrajectories_,
          message->mutable_target_trajectories());
  writeScalars(primalSolution.timeTrajectory_,
               message->mutable_time_trajectory());
  writeVectors(primalSolution.stateTrajectory_,
               message->mutable_state_trajectory());
  writeVectors(primalSolution.inputTrajectory_,
               message->mutable_input_trajectory());
  writeIndices(primalSolution.postEventIndices_,
               message->mutable_post_event_indices());
  toProto(primalSolution.modeSchedule_, message->mutable_mode_schedule());
  writeController(primalSolution, message);
  toProto(performanceIndex, message->mutable_performance());
  return absl::OkStatus();
}

absl::Status policyFromProto(const humanoid_mpc_msgs::MpcPolicy& message,
                             std::shared_ptr<const StateManifold> stateManifold,
                             CommandData* absl_nonnull commandData,
                             PrimalSolution* absl_nonnull primalSolution,
                             PerformanceIndex* absl_nonnull performanceIndex) {
  IPC_RETURN_IF_ERROR(
      validateObservation(message.init_observation(), kPolicyInitObservation));
  IPC_RETURN_IF_ERROR(validateTargetTrajectories(message.target_trajectories(),
                                                 kPolicyTargetTrajectories));
  IPC_RETURN_IF_ERROR(validatePrimalSolution(message, stateManifold.get()));
  readObservation(message.init_observation(),
                  &commandData->mpcInitObservation_);
  readTargetTrajectories(message.target_trajectories(),
                         &commandData->mpcTargetTrajectories_);
  readPrimalSolution(message, std::move(stateManifold), primalSolution);
  readPerformanceIndex(message.performance(), performanceIndex);
  return absl::OkStatus();
}

absl::Status checkDimensions(const humanoid_mpc_msgs::MpcPolicy& message,
                             const ModelDimensions& dimensions) {
  IPC_RETURN_IF_ERROR(checkObservationSizes(
      message.init_observation(), kPolicyInitObservation, dimensions));
  IPC_RETURN_IF_ERROR(checkVectorSizes(message.target_trajectories().state(),
                                       kPolicyTargetTrajectories, "state",
                                       dimensions.stateDim, "state"));
  IPC_RETURN_IF_ERROR(checkVectorSizes(message.target_trajectories().input(),
                                       kPolicyTargetTrajectories, "input",
                                       dimensions.inputDim, "input"));
  IPC_RETURN_IF_ERROR(checkVectorSizes(message.state_trajectory(), kPolicy,
                                       "state_trajectory", dimensions.stateDim,
                                       "state"));
  IPC_RETURN_IF_ERROR(checkVectorSizes(message.input_trajectory(), kPolicy,
                                       "input_trajectory", dimensions.inputDim,
                                       "input"));
  for (int index = 0; index < message.mode_schedule().mode_sequence_size();
       ++index) {
    IPC_RETURN_IF_ERROR(checkMode(
        message.mode_schedule().mode_sequence(index),
        absl::StrCat(kPolicyModeSchedule, ".mode_sequence[", index, "]"),
        dimensions));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::ipc

#undef IPC_RETURN_IF_ERROR
