/******************************************************************************
Copyright (c) 2026. All rights reserved.

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
#include <istream>
#include <limits>
#include <list>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "absl/strings/string_view.h"

namespace ocs2 {

/**
 * Base of the exceptions a PropertyTree throws. Replaces boost::property_tree::ptree_error: a handler that caught that
 * catches this, and, as before, std::runtime_error and std::exception catch it too.
 */
class PropertyTreeError : public std::runtime_error {
 public:
  explicit PropertyTreeError(const std::string& message) : std::runtime_error(message) {}
};

/**
 * A path that names no node. Replaces boost::property_tree::ptree_bad_path; what() reads "No such node (<path>)", as
 * the Boost message did.
 */
class PropertyTreeBadPath : public PropertyTreeError {
 public:
  PropertyTreeBadPath(const std::string& message, std::string path) : PropertyTreeError(message), path_(std::move(path)) {}

  /** The path that was looked up, in full. */
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

/**
 * A node whose data does not convert to the requested type. Replaces boost::property_tree::ptree_bad_data; unlike the
 * Boost message, what() names the data and, when the value was looked up by path, the path.
 */
class PropertyTreeBadData : public PropertyTreeError {
 public:
  PropertyTreeBadData(const std::string& message, std::string data) : PropertyTreeError(message), data_(std::move(data)) {}

  /** The data that did not convert. */
  const std::string& data() const { return data_; }

 private:
  std::string data_;
};

namespace property_tree_internal {

/**
 * Extracts a T from a stream exactly as boost::property_tree::customize_stream does: '>>', then any trailing
 * whitespace is consumed. Specialized below for the types Boost specializes.
 */
template <typename T>
struct StreamExtractor {
  static void extract(std::istream& stream, T& value) {
    stream >> value;
    if (!stream.eof()) {
      stream >> std::ws;
    }
  }
};

/** A single character is read without skipping whitespace, so " " is a space and " a" does not convert. */
template <>
struct StreamExtractor<char> {
  static void extract(std::istream& stream, char& value) {
    stream.unsetf(std::ios_base::skipws);
    stream >> value;
  }
};

/** A bool is read as a number first ("0" or "1"), then, when that fails, as the words "true" or "false". */
template <>
struct StreamExtractor<bool> {
  static void extract(std::istream& stream, bool& value) {
    stream >> value;
    if (stream.fail()) {
      stream.clear();
      stream.setf(std::ios_base::boolalpha);
      stream >> value;
    }
    if (!stream.eof()) {
      stream >> std::ws;
    }
  }
};

/** A signed char is read as a number, and a number out of its range does not convert. */
template <>
struct StreamExtractor<signed char> {
  static void extract(std::istream& stream, signed char& value) {
    int wide = 0;
    stream >> wide;
    if (wide > std::numeric_limits<signed char>::max() || wide < std::numeric_limits<signed char>::min()) {
      stream.clear();
      value = 0;
      stream.setstate(std::ios_base::badbit);
      return;
    }
    value = static_cast<signed char>(wide);
    if (!stream.eof()) {
      stream >> std::ws;
    }
  }
};

/** An unsigned char is read as a number, and a number out of its range does not convert. */
template <>
struct StreamExtractor<unsigned char> {
  static void extract(std::istream& stream, unsigned char& value) {
    unsigned wide = 0;
    stream >> wide;
    if (wide > std::numeric_limits<unsigned char>::max()) {
      stream.clear();
      value = 0;
      stream.setstate(std::ios_base::badbit);
      return;
    }
    value = static_cast<unsigned char>(wide);
    if (!stream.eof()) {
      stream >> std::ws;
    }
  }
};

/**
 * Converts the data of a node to a T the way boost::property_tree's stream_translator does: a std::string is the data
 * verbatim; anything else is extracted from an istringstream imbued with the global locale, and the conversion fails
 * unless the whole data, but for trailing whitespace, was consumed.
 */
template <typename T>
std::optional<T> fromData(const std::string& data) {
  if constexpr (std::is_same_v<T, std::string>) {
    return data;
  } else {
    std::istringstream stream(data);
    stream.imbue(std::locale());
    T value{};
    StreamExtractor<T>::extract(stream, value);
    if (stream.fail() || stream.bad() || stream.get() != std::char_traits<char>::eof()) {
      return std::nullopt;
    }
    return value;
  }
}

/** A readable name of T for the messages of PropertyTreeBadData. */
template <typename T>
std::string typeName() {
  if constexpr (std::is_same_v<T, bool>) {
    return "bool";
  } else if constexpr (std::is_same_v<T, double>) {
    return "double";
  } else if constexpr (std::is_same_v<T, float>) {
    return "float";
  } else if constexpr (std::is_same_v<T, int>) {
    return "int";
  } else if constexpr (std::is_same_v<T, long>) {
    return "long";
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return "unsigned long";
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return "unsigned int";
  } else if constexpr (std::is_same_v<T, std::string>) {
    return "std::string";
  } else {
    return typeid(T).name();
  }
}

/** The message of a PropertyTreeBadData: `conversion of data "<data>" to type "<type>" failed`, then ` (<path>)`. */
std::string badDataMessage(const std::string& data, const std::string& type, const std::string* path);

}  // namespace property_tree_internal

/**
 * An ordered tree of string-keyed children, each node carrying a string of data: the structure a YAML task file is read
 * into (ocs2::loadData::readPropertyTree), and a Boost-free replacement for boost::property_tree::ptree with the same
 * semantics for everything this repository does with one.
 *
 * - Children keep the order they were added in, and keys need not be unique; a lookup finds the first child with the
 *   key. A YAML sequence is stored as children keyed "[0]", "[1]", ...; a matrix entry in a task file is keyed "(i,j)".
 * - A path is a sequence of keys joined by '.': "a.b.[0]" is the first item of the sequence `b` of the map `a`. The empty
 *   path names the node itself. A key that itself contains a '.' cannot be reached by path.
 * - Values convert from the data exactly as ptree's stream_translator converts them (see property_tree_internal::
 *   fromData): numbers by '>>' with trailing whitespace allowed, a bool from "1"/"0" or "true"/"false" only (not YAML's
 *   yes/on), a std::string verbatim.
 */
class PropertyTree {
 public:
  using key_type = std::string;
  using data_type = std::string;
  /** A child: its key, then the subtree. The same layout as ptree::value_type. */
  using value_type = std::pair<const std::string, PropertyTree>;
  using iterator = std::list<value_type>::iterator;
  using const_iterator = std::list<value_type>::const_iterator;
  using size_type = std::size_t;

  /** An empty node: no data, no children. */
  PropertyTree() = default;

  /** A leaf carrying `data`. */
  explicit PropertyTree(std::string data) : data_(std::move(data)) {}

  /** A deep copy. */
  PropertyTree(const PropertyTree& other) = default;
  PropertyTree(PropertyTree&& other) = default;
  ~PropertyTree() = default;

  /**
   * Replaces this tree with `other` (a deep copy, or the moved-from tree), as ptree's copy-and-swap assignment does.
   * Written out because the implicit operator would assign the children element by element, which a child's const key
   * forbids. `other` is taken first, so it may be a node of this very tree.
   */
  PropertyTree& operator=(PropertyTree other) noexcept {
    swap(other);
    return *this;
  }

  /** Exchanges the data and the children of the two trees; references to children follow them. */
  void swap(PropertyTree& other) noexcept {
    data_.swap(other.data_);
    children_.swap(other.children_);
  }

  /** The data of this node; empty for a map, a sequence or a YAML null. */
  const std::string& data() const { return data_; }

  /** Replaces the data of this node. */
  void setData(std::string data) { data_ = std::move(data); }

  /** The number of immediate children. */
  size_type size() const { return children_.size(); }

  /** Whether this node has no children. Like ptree::empty(), it says nothing about the data. */
  bool empty() const { return children_.empty(); }

  /** The immediate children, in order. */
  iterator begin() { return children_.begin(); }
  const_iterator begin() const { return children_.begin(); }
  iterator end() { return children_.end(); }
  const_iterator end() const { return children_.end(); }

  /** Appends a child, even when one with its key exists already, and returns an iterator to it. */
  iterator push_back(value_type child);

  /** Appends a child keyed `key` and returns it. References to children stay valid when more are added. */
  PropertyTree& addChild(std::string key, PropertyTree child = PropertyTree());

  /** Removes the data and every child. */
  void clear();

  /**
   * The first IMMEDIATE child keyed `key`, or end() when there is none. Not a path: "a.b" is the key "a.b". Replaces
   * ptree::find, whose miss compared equal to ptree::not_found().
   */
  iterator find(absl::string_view key);
  const_iterator find(absl::string_view key) const;

  /** The number of IMMEDIATE children keyed `key`. Not a path. */
  size_type count(absl::string_view key) const;

  /** The node at `path`, or nullptr when there is none. Replaces ptree::get_child_optional. */
  const PropertyTree* findChild(absl::string_view path) const;
  PropertyTree* findChild(absl::string_view path);

  /**
   * The node at `path`. Replaces ptree::get_child.
   * @throws PropertyTreeBadPath when there is no node at `path`.
   */
  const PropertyTree& getChild(absl::string_view path) const;
  PropertyTree& getChild(absl::string_view path);

  /** The data of this node as a T, or nullopt when it does not convert. Replaces ptree::get_value_optional. */
  template <typename T>
  std::optional<T> getValueOptional() const {
    return property_tree_internal::fromData<T>(data_);
  }

  /**
   * The data of this node as a T. Replaces ptree::get_value.
   * @throws PropertyTreeBadData when the data does not convert.
   */
  template <typename T>
  T getValue() const {
    std::optional<T> value = getValueOptional<T>();
    if (!value.has_value()) {
      throw PropertyTreeBadData(property_tree_internal::badDataMessage(data_, property_tree_internal::typeName<T>(), /*path=*/nullptr),
                                data_);
    }
    return *std::move(value);
  }

  /**
   * The data of the node at `path` as a T. Replaces ptree::get<T>(path).
   * @throws PropertyTreeBadPath when there is no node at `path`.
   * @throws PropertyTreeBadData when there is one but its data does not convert.
   */
  template <typename T>
  T get(absl::string_view path) const {
    const PropertyTree& child = getChild(path);
    std::optional<T> value = child.getValueOptional<T>();
    if (!value.has_value()) {
      const std::string pathString(path);
      throw PropertyTreeBadData(property_tree_internal::badDataMessage(child.data_, property_tree_internal::typeName<T>(), &pathString),
                                child.data_);
    }
    return *std::move(value);
  }

  /**
   * The data of the node at `path` as a T, or `defaultValue` when there is no node at `path` OR its data does not
   * convert. This is ptree::get(path, default) exactly, which never throws: a mistyped value silently becomes the
   * default. To refuse a value that is present but does not convert, use findChild() and getValueOptional().
   */
  template <typename T>
  T get(absl::string_view path, const T& defaultValue) const {
    std::optional<T> value = getOptional<T>(path);
    return value.has_value() ? *std::move(value) : defaultValue;
  }

  /** As get(path, defaultValue), for a string literal default, as ptree::get(path, const char*). */
  std::string get(absl::string_view path, const char* defaultValue) const { return get<std::string>(path, std::string(defaultValue)); }

  /**
   * The data of the node at `path` as a T, or nullopt when there is no node at `path` or its data does not convert.
   * Replaces ptree::get_optional.
   */
  template <typename T>
  std::optional<T> getOptional(absl::string_view path) const {
    const PropertyTree* child = findChild(path);
    if (child == nullptr) {
      return std::nullopt;
    }
    return child->getValueOptional<T>();
  }

  /** Two trees are equal when their data, and their children's keys, order and subtrees, are. As ptree::operator==. */
  bool operator==(const PropertyTree& other) const;
  bool operator!=(const PropertyTree& other) const { return !(*this == other); }

 private:
  std::string data_;
  std::list<value_type> children_;
};

}  // namespace ocs2
