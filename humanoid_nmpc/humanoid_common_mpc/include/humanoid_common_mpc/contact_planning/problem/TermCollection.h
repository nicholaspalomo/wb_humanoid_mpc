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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid {

/**
 * Ordered collection of named terms, the interface of ocs2::Collection (add / find / erase / getTermNameMap) plus
 * iteration in insertion order, which the assembly of the planner's problem needs and ocs2::Collection keeps protected.
 * It owns its terms. Not thread-safe; the const methods may be called concurrently.
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

  /**
   * Adds a term under a unique name and takes ownership of it. A null term or a name already in the collection is an
   * InvalidArgument naming the term, and leaves the collection unchanged.
   */
  absl::Status add(std::string name, TermPtr term) {
    if (term == nullptr) return absl::InvalidArgumentError(absl::StrCat("[TermCollection] term '", name, "' is null"));
    if (indexByName_.contains(name)) {
      return absl::InvalidArgumentError(absl::StrCat("[TermCollection] term '", name, "' is already in the collection"));
    }
    indexByName_[name] = terms_.size();
    names_.push_back(std::move(name));
    terms_.push_back(std::move(term));
    return absl::OkStatus();
  }

  bool has(absl::string_view name) const { return indexByName_.contains(name); }

  /** The term under `name`, or null when there is none. The collection keeps ownership. */
  T* absl_nullable find(absl::string_view name) {
    const IndexMap::const_iterator it = indexByName_.find(name);
    return it == indexByName_.end() ? nullptr : terms_[it->second].get();
  }
  const T* absl_nullable find(absl::string_view name) const {
    const IndexMap::const_iterator it = indexByName_.find(name);
    return it == indexByName_.end() ? nullptr : terms_[it->second].get();
  }

  /** Removes the term under `name`; false if there was none. */
  bool erase(absl::string_view name) {
    const IndexMap::const_iterator it = indexByName_.find(name);
    if (it == indexByName_.end()) return false;
    const size_t i = it->second;
    terms_.erase(terms_.begin() + static_cast<ptrdiff_t>(i));
    names_.erase(names_.begin() + static_cast<ptrdiff_t>(i));
    indexByName_.clear();
    for (size_t j = 0; j < names_.size(); ++j) indexByName_[names_[j]] = j;
    return true;
  }

  /** Names in insertion order. */
  const std::vector<std::string>& names() const { return names_; }
  absl::flat_hash_map<std::string, size_t> getTermNameMap() const { return indexByName_; }

  /** The name and the term at position `i` of the insertion order; `i` must be below size(). */
  const std::string& nameAt(size_t i) const { return names_[i]; }
  T& termAt(size_t i) { return *terms_[i]; }
  const T& termAt(size_t i) const { return *terms_[i]; }

  // Iteration in insertion order over the owned terms.
  typename std::vector<TermPtr>::iterator begin() { return terms_.begin(); }
  typename std::vector<TermPtr>::iterator end() { return terms_.end(); }
  typename std::vector<TermPtr>::const_iterator begin() const { return terms_.begin(); }
  typename std::vector<TermPtr>::const_iterator end() const { return terms_.end(); }

 private:
  using IndexMap = absl::flat_hash_map<std::string, size_t>;

  std::vector<TermPtr> terms_;
  std::vector<std::string> names_;
  IndexMap indexByName_;
};

}  // namespace ocs2::humanoid
