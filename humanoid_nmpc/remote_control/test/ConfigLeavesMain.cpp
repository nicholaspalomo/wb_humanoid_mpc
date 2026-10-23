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

#include <initializer_list>

#include "absl/base/nullability.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"

#include "humanoid_common_mpc/config/ConfigReload.h"
#include "humanoid_mpc_config/contact_planning_file.pb.h"
#include "humanoid_mpc_config/joint_pd_gains_file.pb.h"
#include "humanoid_mpc_config/reference_file.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"

/*
 * Prints every value field of the configuration files as the C++ side reads its tuning options (configLeaves(),
 * humanoid_common_mpc/config/ConfigReload.h), one line per field:
 *
 *   <file message>\t<path>\t<reload>\t<consumer>\t<formulations, comma-separated>\t<tunable: 0 or 1>
 *
 * The tuning GUI reads the same options with a walk of its own (remote_control/config_schema.py);
 * test_tuning_inheritance_parity.py compares the two.
 */
namespace {

absl::string_view reloadName(ocs2::humanoid::ConfigReload reload) {
  switch (reload) {
    case ocs2::humanoid::ConfigReload::kUnspecified:
      return "RELOAD_UNSPECIFIED";
    case ocs2::humanoid::ConfigReload::kHot:
      return "RELOAD_HOT";
    case ocs2::humanoid::ConfigReload::kStartUp:
      return "RELOAD_START_UP";
  }
  return "RELOAD_UNKNOWN";
}

}  // namespace

int main() {
  // LINT.IfChange(file_messages)
  const std::initializer_list<const google::protobuf::Descriptor* absl_nonnull> files = {
      humanoid_mpc_config::TaskFile::descriptor(), humanoid_mpc_config::ReferenceFile::descriptor(),
      humanoid_mpc_config::ContactPlanningFile::descriptor(), humanoid_mpc_config::JointPdGainsFile::descriptor()};
  // LINT.ThenChange(//humanoid_nmpc/remote_control/test/test_tuning_inheritance_parity.py:file_messages)
  for (const google::protobuf::Descriptor* absl_nonnull file : files) {
    for (const ocs2::humanoid::ConfigLeaf& leaf : ocs2::humanoid::configLeaves(*file)) {
      absl::PrintF("%s\t%s\t%s\t%s\t%s\t%d\n", file->full_name(), leaf.path, reloadName(leaf.reload), leaf.consumer,
                   absl::StrJoin(leaf.formulations, ","), leaf.tunable ? 1 : 0);
    }
  }
  return 0;
}
