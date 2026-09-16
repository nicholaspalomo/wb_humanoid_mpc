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

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ocs2::humanoid {

/**
 * Ordered collection of named terms, the interface of ocs2::Collection (add / get / erase / getTermNameMap) plus
 * iteration in insertion order, which the assembly of the planner's problem needs and ocs2::Collection keeps protected.
 */
template <typename T>
class TermCollection {
 public:
  using TermPtr = std::unique_ptr<T>;

  bool empty() const { return terms_.empty(); }
  size_t size() const { return terms_.size(); }
  void clear() {
    terms_.clear();
    names_.clear();
    indexByName_.clear();
  }

  /** Adds a term under a unique name and takes ownership of it. Throws std::invalid_argument on a duplicate name. */
  void add(std::string name, TermPtr term) {
    if (term == nullptr) throw std::invalid_argument("[TermCollection] term '" + name + "' is null");
    if (indexByName_.count(name) > 0) throw std::invalid_argument("[TermCollection] term '" + name + "' is already in the collection");
    indexByName_[name] = terms_.size();
    names_.push_back(std::move(name));
    terms_.push_back(std::move(term));
  }

  bool has(const std::string& name) const { return indexByName_.count(name) > 0; }

  /** The term under `name`, cast to `Derived`. Throws std::out_of_range if absent, std::bad_cast on a wrong type. */
  template <typename Derived = T>
  Derived& get(const std::string& name) {
    return dynamic_cast<Derived&>(*terms_.at(index(name)));
  }
  template <typename Derived = T>
  const Derived& get(const std::string& name) const {
    return dynamic_cast<const Derived&>(*terms_.at(index(name)));
  }

  /** Removes the term under `name`; false if there was none. */
  bool erase(const std::string& name) {
    const auto it = indexByName_.find(name);
    if (it == indexByName_.end()) return false;
    const size_t i = it->second;
    terms_.erase(terms_.begin() + static_cast<std::ptrdiff_t>(i));
    names_.erase(names_.begin() + static_cast<std::ptrdiff_t>(i));
    indexByName_.clear();
    for (size_t j = 0; j < names_.size(); ++j) indexByName_[names_[j]] = j;
    return true;
  }

  /** Names in insertion order. */
  const std::vector<std::string>& names() const { return names_; }
  std::unordered_map<std::string, size_t> getTermNameMap() const { return indexByName_; }

  const std::string& nameAt(size_t i) const { return names_.at(i); }
  T& at(size_t i) { return *terms_.at(i); }
  const T& at(size_t i) const { return *terms_.at(i); }

  // Iteration in insertion order over the owned terms.
  typename std::vector<TermPtr>::iterator begin() { return terms_.begin(); }
  typename std::vector<TermPtr>::iterator end() { return terms_.end(); }
  typename std::vector<TermPtr>::const_iterator begin() const { return terms_.begin(); }
  typename std::vector<TermPtr>::const_iterator end() const { return terms_.end(); }

 private:
  size_t index(const std::string& name) const {
    const auto it = indexByName_.find(name);
    if (it == indexByName_.end()) throw std::out_of_range("[TermCollection] no term named '" + name + "'");
    return it->second;
  }

  std::vector<TermPtr> terms_;
  std::vector<std::string> names_;
  std::unordered_map<std::string, size_t> indexByName_;
};

}  // namespace ocs2::humanoid
