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

#include <robot_model/ContactEstimator.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace robot::model {

/**
 * Contact estimators by name, the way the task file selects one (`contactEstimator: <name>`), like the costs and
 * constraints of the MPC formulation or the viewer's visualizations.
 *
 * The registry starts with the estimators that need nothing but the robot state (`robot_state`, `always_in_contact`);
 * an interface that can offer more registers it under its own name (the MuJoCo simulator adds `cheater_sim`, see
 * mujoco_sim_interface/CheaterSimContactEstimator.h). Names are matched case-insensitively with surrounding blanks
 * removed; an unknown name is an error that lists the available ones.
 */
class ContactEstimatorRegistry {
 public:
  using Factory = std::function<std::shared_ptr<ContactEstimator>()>;

  struct Entry {
    std::string name;
    std::string description;
  };

  static constexpr const char* kRobotState = "robot_state";
  static constexpr const char* kAlwaysInContact = "always_in_contact";

  /** Registers the estimators of robot_model. */
  ContactEstimatorRegistry();

  /** Adds an estimator; a name already registered is a programming error (std::invalid_argument). */
  void add(const std::string& name, const std::string& description, Factory factory);

  /** The estimator registered under `name`; std::invalid_argument for an unknown name, listing the available ones. */
  std::shared_ptr<ContactEstimator> create(const std::string& name) const;

  bool has(const std::string& name) const;

  /** The registered estimators, in registration order. */
  std::vector<Entry> available() const;

  /** The registered names, comma separated, for messages. */
  std::string availableNames() const;

  /** `name` as the registry matches it: trimmed and lower case. */
  static std::string canonicalName(const std::string& name);

 private:
  struct Registered {
    Entry entry;
    Factory factory;
  };
  std::vector<Registered> estimators_;
};

}  // namespace robot::model
