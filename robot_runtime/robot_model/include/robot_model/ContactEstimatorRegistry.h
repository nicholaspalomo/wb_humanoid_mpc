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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

#include "robot_model/ContactEstimator.h"

namespace robot::model {

/**
 * Contact estimators by name, the way the task file selects one (`contact_estimator: "<name>"`), like the costs and
 * constraints of the MPC formulation or the viewer's visualizations.
 *
 * The registry starts with the estimators that need nothing but the robot state (`robot_state`, `always_in_contact`);
 * an interface that can offer more registers it under its own name (the MuJoCo simulator adds `cheater_sim`, see
 * mujoco_sim_interface/CheaterSimContactEstimator.h). Names are matched case-insensitively with surrounding blanks
 * removed. A name from a file is the caller's to validate with has(), reporting availableNames() for one that is not
 * registered; create() of an unregistered name is a programming error. Not thread-safe: register everything before
 * sharing the registry, which is then read-only.
 */
class ContactEstimatorRegistry {
 public:
  using Factory = std::function<std::shared_ptr<ContactEstimator>()>;

  struct Entry {
    std::string name;
    std::string description;
  };

  static constexpr char kRobotState[] = "robot_state";
  static constexpr char kAlwaysInContact[] = "always_in_contact";

  /** Registers the estimators of robot_model. */
  ContactEstimatorRegistry();

  /**
   * Adds an estimator. An empty name, an empty factory or a name already registered is a programming error, which ends
   * the process.
   */
  void add(absl::string_view name, absl::string_view description, Factory factory);

  /**
   * A new estimator of the one registered under `name`, which must be registered (has()); an unregistered name ends the
   * process with the available ones.
   */
  std::shared_ptr<ContactEstimator> create(absl::string_view name) const;

  bool has(absl::string_view name) const;

  /** The registered estimators, in registration order. */
  std::vector<Entry> available() const;

  /** The registered names, comma separated, for messages. */
  std::string availableNames() const;

  /** `name` as the registry matches it: trimmed and lower case. */
  static std::string canonicalName(absl::string_view name);

 private:
  struct Registered {
    Entry entry;
    Factory factory;
  };
  std::vector<Registered> estimators_;
};

}  // namespace robot::model
