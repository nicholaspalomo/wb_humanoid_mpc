/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

#include "Eigen/Dense"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"

namespace robot::model {

// The extractor of IDMapBase::toEigenVector(): a callable that reads one ScalarType out of an element.
template <typename E, typename T, typename ScalarType>
concept IDMapExtractor = requires(E e, const T& val) {
  { e(val) } -> std::same_as<ScalarType>;
};

/**
 * A map from ids to elements over the contiguous range 0 to capacity() - 1: one std::optional<T> slot per id, so that a
 * map whose ids have gaps costs one empty slot per gap, and a lookup is an index. Iteration visits the engaged slots in
 * id order. The capacity is fixed at construction and copying keeps it, so that copying one map into another of the same
 * capacity allocates nothing (the realtime loop copies RobotState and RobotJointAction every cycle). Not thread-safe.
 */
template <typename T>
class IDMapBase {
 public:
  /**
   * The slot of `element_id`. Unchecked, as std::vector's operator[]: `element_id` must be below capacity(), an id of the
   * robot description; code on the realtime path validates its ids once, when it is built.
   */
  std::optional<T>& operator[](size_t element_id) { return map_elements_[element_id]; }
  const std::optional<T>& operator[](size_t element_id) const { return map_elements_[element_id]; }

  /** operator[], checked: an id out of range ends the process. For tests and setup code, not for every cycle. */
  std::optional<T>& at(size_t element_id) {
    ABSL_CHECK_LT(element_id, map_elements_.size()) << "IDMapBase: no id " << element_id;
    return map_elements_[element_id];
  }
  const std::optional<T>& at(size_t element_id) const {
    ABSL_CHECK_LT(element_id, map_elements_.size()) << "IDMapBase: no id " << element_id;
    return map_elements_[element_id];
  }

  template <typename It, typename ScalarType, IDMapExtractor<T, ScalarType> Extractor>
  Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> toEigenVector(It begin, It end, Extractor extractor, ScalarType defaultValue) const {
    Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> vector;
    writeEigenVector(begin, end, extractor, defaultValue, vector);
    return vector;
  }

  /**
   * toEigenVector() into `vector`, resized to the number of ids: no allocation once it has that size. Every id in
   * [begin, end) must be below capacity() (operator[]); an empty slot reads as `defaultValue`.
   */
  template <typename It, typename ScalarType, IDMapExtractor<T, ScalarType> Extractor>
  void writeEigenVector(
      It begin, It end, Extractor extractor, ScalarType defaultValue, Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& vector) const {
    vector.resize(std::distance(begin, end));
    int index = 0;
    for (const std::iter_value_t<It>& id : std::ranges::subrange(begin, end)) {
      const std::optional<T>& val = (*this)[id];
      if (val.has_value()) {
        vector(index) = extractor(*val);
      } else {
        vector(index) = defaultValue;
      }
      ++index;
    }
  }

  bool inRange(size_t id) const noexcept {
    // id == capacity() is out of range: the valid ids are 0 .. capacity() - 1.
    return id < map_elements_.size();
  }

  /** The number of engaged slots. */
  size_t size() const {
    size_t count = 0;
    for (const std::optional<T>& element : map_elements_) {
      if (element.has_value()) ++count;
    }
    return count;
  }

  /** The number of ids, engaged or not. */
  size_t capacity() const { return map_elements_.size(); }

  // iterator class to skip uninitialized elements

  class const_iterator;  // NOLINT(readability-identifier-naming): the STL container interface (G: Exceptions to Naming Rules)
  class iterator;        // NOLINT(readability-identifier-naming): the STL container interface (G: Exceptions to Naming Rules)

  /** What iterator and const_iterator share: the position, which skips the empty slots. */
  template <typename MapType, typename RefType>
  class IteratorBase {
   public:
    IteratorBase(MapType map, size_t it) : map_(map), it_(it) {
      // Skip nullopt elements
      skipNullopt();
    }

    // Increase the iterator
    IteratorBase& operator++() {
      ++it_;
      skipNullopt();
      return *this;
    }

    bool operator==(const IteratorBase& other) const { return it_ == other.it_; }

    bool operator!=(const IteratorBase& other) const { return it_ != other.it_; }

   protected:
    MapType map_;
    size_t it_ = 0;

    // Helper function to skip over elements that are not null optionals
    void skipNullopt() {
      while (it_ != map_->map_elements_.size() && !map_->map_elements_[it_].has_value()) {
        ++it_;  // skip while current element is std::nullopt
      }
    }

    // Allow the derived classes to access each other's private members
    friend class iterator;
    friend class const_iterator;
  };

  // NOLINTNEXTLINE(readability-identifier-naming): the STL container interface (G: Exceptions to Naming Rules)
  class iterator : public IteratorBase<IDMapBase<T>* absl_nonnull, T&> {
   public:
    iterator(IDMapBase<T>* absl_nonnull map, size_t it) : IteratorBase<IDMapBase<T>* absl_nonnull, T&>(map, it) {}

    // Dereference to get underlying data. The iterator stops at engaged slots only (skipNullopt()).
    T& operator*() { return *this->map_->map_elements_[this->it_]; }  // NOLINT(bugprone-unchecked-optional-access): see above
  };

  // NOLINTNEXTLINE(readability-identifier-naming): the STL container interface (G: Exceptions to Naming Rules)
  class const_iterator : public IteratorBase<const IDMapBase<T>* absl_nonnull, const T&> {
   public:
    const_iterator(const IDMapBase<T>* absl_nonnull map, size_t it) : IteratorBase<const IDMapBase<T>* absl_nonnull, const T&>(map, it) {}

    // NOLINTNEXTLINE(google-explicit-constructor, runtime/explicit): an iterator converts to a const_iterator, as in std
    const_iterator(const iterator& other) : IteratorBase<const IDMapBase<T>* absl_nonnull, const T&>(other.map_, other.it_) {}

    // Dereference to get underlying data. The iterator stops at engaged slots only (skipNullopt()).
    const T& operator*() const {
      return *this->map_->map_elements_[this->it_];  // NOLINT(bugprone-unchecked-optional-access): see above
    }
  };

  iterator begin() { return iterator(this, /*it=*/0); }
  iterator end() { return iterator(this, map_elements_.size()); }
  const_iterator begin() const { return const_iterator(this, /*it=*/0); }
  const_iterator end() const { return const_iterator(this, map_elements_.size()); }
  const_iterator cbegin() const { return const_iterator(this, /*it=*/0); }
  const_iterator cend() const { return const_iterator(this, map_elements_.size()); }

 protected:
  /** The fewest and the most ids a map may have (RobotDescription::kMinJoints and kMaxJoints check them for a robot). */
  static constexpr size_t kMinSize = 2;
  static constexpr size_t kMaxSize = 255;

  /** A map of `size` empty slots; `size` is kMinSize to kMaxSize, or the process ends. */
  explicit IDMapBase(size_t size) : map_elements_(size) {
    ABSL_CHECK(size >= kMinSize && size <= kMaxSize) << "IDMapBase: " << size << " ids; a map has " << kMinSize << " to " << kMaxSize;
  }

  IDMapBase(const IDMapBase<T>& other) = default;
  IDMapBase(IDMapBase<T>&& other) noexcept = default;

  /**
   * Copies the slots of `other` into this map's, keeping its capacity: nothing is allocated. Unchecked, as operator[]:
   * `other` must have at least capacity() slots, which two maps of one robot description have (the realtime loop copies
   * RobotState and RobotJointAction every cycle; RobotHWInterfaceBase builds every copy of them from one description).
   */
  IDMapBase& operator=(const IDMapBase<T>& other) {
    std::copy_n(other.map_elements_.begin(), map_elements_.size(), map_elements_.begin());
    return *this;
  }
  /** As the copy: the capacity is kept. */
  IDMapBase& operator=(IDMapBase<T>&& other) noexcept { return *this = static_cast<const IDMapBase<T>&>(other); }

  ~IDMapBase() = default;

 private:
  std::vector<std::optional<T>> map_elements_;
};

}  // namespace robot::model
