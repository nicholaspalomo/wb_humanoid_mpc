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

#include "ocs2_core/misc/LoadData.h"

#include <cstddef>

namespace ocs2 {
namespace loadData {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void yamlToPropertyTree(const YAML::Node& node, PropertyTree& pt) {
  if (node.IsScalar()) {
    pt.setData(node.as<std::string>());
  } else if (node.IsMap()) {
    for (YAML::const_iterator entry = node.begin(); entry != node.end(); ++entry) {
      PropertyTree child;
      yamlToPropertyTree(entry->second, child);
      pt.addChild(entry->first.as<std::string>(), std::move(child));
    }
  } else if (node.IsSequence()) {
    for (std::size_t i = 0; i < node.size(); ++i) {
      PropertyTree child;
      yamlToPropertyTree(node[i], child);
      pt.addChild("[" + std::to_string(i) + "]", std::move(child));
    }
  }
  // Null nodes are ignored (produce an empty node).
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void readPropertyTree(const std::string& filename, PropertyTree& pt) {
  const std::string::size_type dot = filename.rfind('.');
  if (dot != std::string::npos) {
    const std::string extension = filename.substr(dot);
    if (extension == ".yaml" || extension == ".yml") {
      const YAML::Node root = YAML::LoadFile(filename);
      yamlToPropertyTree(root, pt);
      return;
    }
  }
  throw std::invalid_argument("[loadData::readPropertyTree] \"" + filename +
                              "\" is not a YAML file: a configuration file must have the extension .yaml or .yml.");
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void readPropertyTreeFromString(absl::string_view yamlText, PropertyTree& pt) {
  const YAML::Node root = YAML::Load(std::string(yamlText));
  yamlToPropertyTree(root, pt);
}

}  // namespace loadData
}  // namespace ocs2
