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

#include "ocs2_core/misc/PropertyTree.h"

#include <algorithm>
#include <iterator>

#include "absl/strings/str_cat.h"

namespace ocs2 {

namespace property_tree_internal {

std::string badDataMessage(const std::string& data, const std::string& type, const std::string* path) {
  std::string message = absl::StrCat("conversion of data \"", data, "\" to type \"", type, "\" failed");
  if (path != nullptr) {
    absl::StrAppend(&message, " (", *path, ")");
  }
  return message;
}

}  // namespace property_tree_internal

namespace {

/**
 * Splits off the first key of `path` at `position`, as boost::property_tree::string_path::reduce does: the key runs to
 * the next '.', or to the end of the path when no '.' is left, and a '.' that ends the key is always consumed with it,
 * even when it is the last character of the path. Advances `position` past what it consumed. A trailing '.' therefore
 * adds no empty last key: "a." names the same node as "a".
 */
absl::string_view nextKey(absl::string_view path, size_t& position) {
  const size_t separator = path.find('.', position);
  if (separator == absl::string_view::npos) {
    const absl::string_view key = path.substr(position);
    position = path.size();
    return key;
  }
  const absl::string_view key = path.substr(position, separator - position);
  position = separator + 1;
  return key;
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PropertyTree::iterator PropertyTree::push_back(value_type child) {
  children_.push_back(std::move(child));
  return std::prev(children_.end());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PropertyTree& PropertyTree::addChild(std::string key, PropertyTree child) {
  return push_back(value_type(std::move(key), std::move(child)))->second;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void PropertyTree::clear() {
  data_.clear();
  children_.clear();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PropertyTree::iterator PropertyTree::find(absl::string_view key) {
  return std::find_if(children_.begin(), children_.end(), [key](const value_type& child) { return child.first == key; });
}

PropertyTree::const_iterator PropertyTree::find(absl::string_view key) const {
  return std::find_if(children_.begin(), children_.end(), [key](const value_type& child) { return child.first == key; });
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PropertyTree::size_type PropertyTree::count(absl::string_view key) const {
  return static_cast<size_type>(
      std::count_if(children_.begin(), children_.end(), [key](const value_type& child) { return child.first == key; }));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const PropertyTree* PropertyTree::findChild(absl::string_view path) const {
  const PropertyTree* node = this;
  size_t position = 0;
  while (position != path.size()) {
    const absl::string_view key = nextKey(path, position);
    const const_iterator child = node->find(key);
    if (child == node->end()) {
      return nullptr;
    }
    node = &child->second;
  }
  return node;
}

PropertyTree* PropertyTree::findChild(absl::string_view path) {
  return const_cast<PropertyTree*>(static_cast<const PropertyTree&>(*this).findChild(path));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const PropertyTree& PropertyTree::getChild(absl::string_view path) const {
  const PropertyTree* child = findChild(path);
  if (child == nullptr) {
    throw PropertyTreeBadPath(absl::StrCat("No such node (", path, ")"), std::string(path));
  }
  return *child;
}

PropertyTree& PropertyTree::getChild(absl::string_view path) {
  return const_cast<PropertyTree&>(static_cast<const PropertyTree&>(*this).getChild(path));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool PropertyTree::operator==(const PropertyTree& other) const {
  return data_ == other.data_ && children_ == other.children_;
}

}  // namespace ocs2
