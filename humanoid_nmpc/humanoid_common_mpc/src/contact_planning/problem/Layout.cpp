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

#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ocs2::humanoid {

namespace var {
std::string footLabel(size_t foot) {
  if (N_CONTACTS == 2) return foot == 0 ? "L" : "R";
  return std::to_string(foot);
}
std::string footX(size_t foot) {
  return "p_" + footLabel(foot) + "x";
}
std::string footY(size_t foot) {
  return "p_" + footLabel(foot) + "y";
}
std::string footDeltaX(size_t foot) {
  return "dp_" + footLabel(foot) + "x";
}
std::string footDeltaY(size_t foot) {
  return "dp_" + footLabel(foot) + "y";
}
std::string contact(size_t foot) {
  return "c_" + footLabel(foot);
}
std::string footYaw(size_t foot) {
  return "psi_" + footLabel(foot);
}
std::string yawTorque(size_t foot) {
  return "tau_" + footLabel(foot);
}
std::string footYawDelta(size_t foot) {
  return "dpsi_" + footLabel(foot);
}
}  // namespace var

namespace {
int indexOf(const std::vector<std::string>& names, const std::string& name, const char* what) {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) throw std::out_of_range(std::string("[Layout] no ") + what + " named '" + name + "'");
  return static_cast<int>(it - names.begin());
}
}  // namespace

int Layout::state(const std::string& name) const {
  return indexOf(stateNames, name, "state");
}

int Layout::input(const std::string& name) const {
  return indexOf(inputNames, name, "input");
}

bool Layout::hasState(const std::string& name) const {
  return std::find(stateNames.begin(), stateNames.end(), name) != stateNames.end();
}

bool Layout::hasInput(const std::string& name) const {
  return std::find(inputNames.begin(), inputNames.end(), name) != inputNames.end();
}

std::string Layout::describe() const {
  std::ostringstream out;
  out << "x = [";
  for (size_t i = 0; i < stateNames.size(); ++i) out << (i > 0 ? " " : "") << stateNames[i];
  out << "] (" << nx << "), u = [";
  for (size_t i = 0; i < inputNames.size(); ++i) out << (i > 0 ? " " : "") << inputNames[i];
  out << "] (" << nu << ")";
  return out.str();
}

}  // namespace ocs2::humanoid
