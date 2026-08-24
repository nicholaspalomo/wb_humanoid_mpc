/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include <yaml-cpp/yaml.h>

namespace ocs2 {
namespace loadData {

class PropertyTreeBadPath : public std::runtime_error {
 public:
  explicit PropertyTreeBadPath(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * Lightweight, native C++ replacement for boost::property_tree::ptree.
 * Supports dot-notation paths ("a.b.c"), vector indexing ("topic.[0]"),
 * matrix indexing ("(0,0)"), iteration, and direct YAML loading.
 */
class PropertyTree {
 public:
  using value_type = std::pair<std::string, PropertyTree>;
  using iterator = typename std::vector<value_type>::iterator;
  using const_iterator = typename std::vector<value_type>::const_iterator;

  PropertyTree() = default;
  explicit PropertyTree(std::string value) : value_(std::move(value)) {}

  bool empty() const { return children_.empty() && value_.empty(); }

  void put_value(std::string val) { value_ = std::move(val); }
  const std::string& get_value() const { return value_; }

  void push_back(value_type child) { children_.push_back(std::move(child)); }

  iterator begin() { return children_.begin(); }
  iterator end() { return children_.end(); }
  const_iterator begin() const { return children_.begin(); }
  const_iterator end() const { return children_.end(); }
  size_t size() const { return children_.size(); }

  const PropertyTree* find_child(absl::string_view path) const {
    if (path.empty()) {
      return this;
    }
    size_t dotPos = path.find('.');
    absl::string_view head = (dotPos == absl::string_view::npos) ? path : path.substr(0, dotPos);
    absl::string_view tail = (dotPos == absl::string_view::npos) ? absl::string_view() : path.substr(dotPos + 1);

    for (const auto& kv : children_) {
      if (kv.first == head) {
        return kv.second.find_child(tail);
      }
    }
    return nullptr;
  }

  PropertyTree* find_child(absl::string_view path) {
    return const_cast<PropertyTree*>(const_cast<const PropertyTree*>(this)->find_child(path));
  }

  const PropertyTree& get_child(absl::string_view path) const {
    const PropertyTree* child = find_child(path);
    if (!child) {
      throw PropertyTreeBadPath(absl::StrCat("PropertyTree path not found: ", path));
    }
    return *child;
  }

  std::optional<std::reference_wrapper<const PropertyTree>> get_child_optional(absl::string_view path) const {
    const PropertyTree* child = find_child(path);
    if (!child) {
      return std::nullopt;
    }
    return std::cref(*child);
  }

  const_iterator find(absl::string_view key) const {
    for (auto it = children_.begin(); it != children_.end(); ++it) {
      if (it->first == key) {
        return it;
      }
    }
    return children_.end();
  }

  const_iterator not_found() const { return children_.end(); }

  size_t count(absl::string_view key) const {
    return (find_child(key) != nullptr) ? 1 : 0;
  }

  template <typename T>
  T get(absl::string_view path) const {
    const PropertyTree* child = find_child(path);
    if (!child || child->value_.empty()) {
      throw PropertyTreeBadPath(absl::StrCat("PropertyTree value not found: ", path));
    }
    return convertValue<T>(child->value_);
  }

  template <typename T>
  T get(absl::string_view path, const T& defaultValue) const {
    const PropertyTree* child = find_child(path);
    if (!child || child->value_.empty()) {
      return defaultValue;
    }
    try {
      return convertValue<T>(child->value_);
    } catch (...) {
      return defaultValue;
    }
  }

 private:
  template <typename T>
  static T convertValue(const std::string& str) {
    if constexpr (std::is_same_v<T, std::string>) {
      return str;
    } else if constexpr (std::is_same_v<T, bool>) {
      if (str == "true" || str == "True" || str == "TRUE" || str == "1") return true;
      if (str == "false" || str == "False" || str == "FALSE" || str == "0") return false;
      return std::stoi(str) != 0;
    } else if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_unsigned_v<T>) {
        return static_cast<T>(std::stoull(str));
      } else {
        return static_cast<T>(std::stoll(str));
      }
    } else if constexpr (std::is_floating_point_v<T>) {
      return static_cast<T>(std::stod(str));
    } else {
      std::istringstream iss(str);
      T val;
      iss >> val;
      return val;
    }
  }

  std::string value_;
  std::vector<value_type> children_;
};

using ptree = PropertyTree;

/**
 * Helper: recursively convert a YAML::Node into a PropertyTree.
 * - Scalar nodes become leaf values.
 * - Map nodes become named children.
 * - Sequence nodes become children keyed by "[0]", "[1]", etc.
 */
inline void yamlToPropertyTree(const YAML::Node& node, PropertyTree& pt) {
  if (node.IsScalar()) {
    pt.put_value(node.as<std::string>());
  } else if (node.IsMap()) {
    for (const auto& kv : node) {
      PropertyTree child;
      yamlToPropertyTree(kv.second, child);
      pt.push_back({kv.first.as<std::string>(), std::move(child)});
    }
  } else if (node.IsSequence()) {
    for (size_t i = 0; i < node.size(); ++i) {
      PropertyTree child;
      yamlToPropertyTree(node[i], child);
      pt.push_back({absl::StrCat("[", i, "]"), std::move(child)});
    }
  }
}

/**
 * Helper: recursive parser for Boost INFO format files.
 */
inline void parseInfoStream(std::istream& is, PropertyTree& pt) {
  std::string line;
  std::string pendingKey;

  while (std::getline(is, line)) {
    // Strip comments (; // #)
    auto commentPos = line.find_first_of(";#");
    if (commentPos != std::string::npos) line = line.substr(0, commentPos);
    commentPos = line.find("//");
    if (commentPos != std::string::npos) line = line.substr(0, commentPos);

    line = std::string(absl::StripAsciiWhitespace(line));
    if (line.empty()) continue;

    if (line == "{") {
      if (!pendingKey.empty()) {
        PropertyTree child;
        parseInfoStream(is, child);
        pt.push_back({pendingKey, std::move(child)});
        pendingKey.clear();
      }
      continue;
    }

    if (line == "}") {
      return;
    }

    if (!pendingKey.empty()) {
      pt.push_back({pendingKey, PropertyTree()});
      pendingKey.clear();
    }

    if (line.back() == '{') {
      std::string key = std::string(absl::StripAsciiWhitespace(line.substr(0, line.size() - 1)));
      PropertyTree child;
      parseInfoStream(is, child);
      pt.push_back({key, std::move(child)});
    } else {
      auto spacePos = line.find_first_of(" \t");
      if (spacePos == std::string::npos) {
        pendingKey = line;
      } else {
        std::string key = line.substr(0, spacePos);
        std::string val = std::string(absl::StripAsciiWhitespace(line.substr(spacePos + 1)));
        if (val == "{") {
          PropertyTree child;
          parseInfoStream(is, child);
          pt.push_back({key, std::move(child)});
        } else {
          if (val.size() >= 2 && ((val.front() == '"' && val.back() == '"') || (val.front() == '\'' && val.back() == '\''))) {
            val = val.substr(1, val.size() - 2);
          }
          PropertyTree child(val);
          pt.push_back({key, std::move(child)});
        }
      }
    }
  }
  if (!pendingKey.empty()) {
    pt.push_back({pendingKey, PropertyTree()});
  }
}

/**
 * Reads a property tree from a config file, auto-detecting the format by extension.
 * - .yaml / .yml  → parsed via yaml-cpp
 * - anything else  → parsed via INFO parser
 */
inline void readPropertyTree(const std::string& filename, PropertyTree& pt) {
  const auto dot = filename.rfind('.');
  if (dot != std::string::npos) {
    const std::string ext = filename.substr(dot);
    if (ext == ".yaml" || ext == ".yml") {
      YAML::Node root = YAML::LoadFile(filename);
      yamlToPropertyTree(root, pt);
      return;
    }
  }
  std::ifstream is(filename);
  if (!is.is_open()) {
    throw std::runtime_error(absl::StrCat("Failed to open file: ", filename));
  }
  parseInfoStream(is, pt);
}

/**
 * Print settings option
 */
template <typename T>
static inline void printValue(std::ostream& stream, const T& value, const std::string& name, bool updated = true, long printWidth = 80) {
  const std::string nameString = " #### '" + name + "'";
  stream << nameString;

  printWidth = std::max<long>(printWidth, nameString.size() + 15);
  stream.width(printWidth - nameString.size());
  const char fill = stream.fill('.');

  if (updated) {
    stream << value << '\n';
  } else {
    stream << value << " (default)\n";
  }

  stream.fill(fill);
}

/**
 * An auxiliary function to help loading OCS2 settings from a property tree
 */
template <typename T>
inline void loadPtreeValue(const PropertyTree& pt, T& value, const std::string& name, bool verbose, long printWidth = 80) {
  bool updated = true;

  try {
    value = pt.get<T>(name);
  } catch (const PropertyTreeBadPath&) {
    updated = false;
  }

  if (verbose) {
    const std::string nameString = name.substr(name.find_last_of('.') + 1);
    printValue(std::cerr, value, nameString, updated, printWidth);
  }
}

/**
 * An auxiliary function to check if the property tree has a certain value
 */
inline bool containsPtreeValueFind(const PropertyTree& pt, const std::string& name) {
  return pt.find(name) != pt.not_found();
}

/**
 * An auxiliary function which loads value of the c++ data types from a file.
 */
template <typename cpp_data_t>
inline void loadCppDataType(const std::string& filename, const std::string& dataName, cpp_data_t& value) {
  PropertyTree pt;
  readPropertyTree(filename, pt);
  value = pt.get<cpp_data_t>(dataName);
}

/**
 * An auxiliary function which loads an Eigen matrix from a file.
 */
template <typename Derived>
inline void loadEigenMatrix(const std::string& filename, const std::string& matrixName, Eigen::MatrixBase<Derived>& matrix) {
  using scalar_t = typename Eigen::MatrixBase<Derived>::Scalar;

  size_t rows = matrix.rows();
  size_t cols = matrix.cols();

  if (rows == 0 || cols == 0) {
    throw std::runtime_error("[loadEigenMatrix] Loading empty matrix \"" + matrixName + "\" is not allowed.");
  }

  PropertyTree pt;
  readPropertyTree(filename, pt);

  const scalar_t scaling = pt.get<scalar_t>(matrixName + ".scaling", 1.0);
  const scalar_t defaultValue = pt.get<scalar_t>(matrixName + ".default", 0.0);

  size_t numFailed = 0;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      scalar_t aij;
      try {
        aij = pt.get<scalar_t>(matrixName + "." + "(" + std::to_string(i) + "," + std::to_string(j) + ")");
      } catch (const std::exception&) {
        aij = defaultValue;
        numFailed++;
      }
      matrix(i, j) = scaling * aij;
    }
  }

  if (numFailed == matrix.size()) {
    throw std::runtime_error("[loadEigenMatrix] Could not load matrix \"" + matrixName + "\" from file \"" + filename + "\".");
  } else if (numFailed > 0) {
    std::cerr << "WARNING: Loaded at least one default value in matrix: \"" + matrixName + "\"\n";
  }
}

template <typename T>
inline void loadStdVector(const std::string& filename, const std::string& topicName, std::vector<T>& loadVector, bool verbose = true) {
  PropertyTree pt;
  readPropertyTree(filename, pt);

  std::vector<T> backup;
  backup.swap(loadVector);
  loadVector.clear();

  size_t vectorSize = 0;
  while (true) {
    try {
      loadVector.push_back(pt.get<T>(topicName + ".[" + std::to_string(vectorSize) + "]"));
      vectorSize++;
    } catch (const std::exception&) {
      if (vectorSize == 0) {
        loadVector.swap(backup);
      }
      break;
    }
  }

  if (verbose) {
    std::cerr << " #### '" << topicName << "': {";
    if (vectorSize == 0) {
      std::cerr << " }\n";
    } else {
      for (size_t i = 0; i < vectorSize; i++) {
        std::cerr << loadVector[i] << ", ";
      }
      std::cerr << "\b\b}\n";
    }
  }
}

}  // namespace loadData
}  // namespace ocs2
