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

#include "humanoid_multi_contact/plan/PlanFile.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/contact_set/contact_set.nproto.pb.h"
#include "humanoid_multi_contact/contact_set/contact_set.pb.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.pb.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.pb.h"
#include "humanoid_multi_contact/terrain/terrain_scene.nproto.pb.h"
#include "humanoid_multi_contact/terrain/terrain_scene.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// FNV-1a, 64 bits.
constexpr uint64_t kFnvOffsetBasis = 14'695'981'039'346'656'037ULL;
constexpr uint64_t kFnvPrime = 1'099'511'628'211ULL;

uint64_t fnv1a(absl::string_view bytes) {
  uint64_t hash = kFnvOffsetBasis;
  for (const char byte : bytes) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= kFnvPrime;
  }
  return hash;
}

std::string deterministicBytes(const google::protobuf::Message& message) {
  std::string bytes;
  {
    google::protobuf::io::StringOutputStream stream(&bytes);
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.SetSerializationDeterministic(true);
    // Fails only for a message of more than 2 GiB, which no plan, contact set
    // or scene comes near.
    const bool serialized = message.SerializeToCodedStream(&coded);
    ABSL_CHECK(serialized) << "cannot serialize a " << message.GetTypeName();
  }
  return bytes;
}

// The schema of a plan, from the repository root.
constexpr char kPlanProtoFile[] =
    "humanoid_nmpc/humanoid_multi_contact/plan/multi_contact_plan.proto";

absl::Status writeFile(const std::string& path, absl::string_view contents) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    return absl::InternalError(absl::StrCat("cannot write '", path, "'"));
  }
  file.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  file.close();
  if (!file) {
    return absl::InternalError(absl::StrCat("writing '", path, "' failed"));
  }
  return absl::OkStatus();
}

absl::StatusOr<config::MultiContactPlan> readPlan(absl::string_view path) {
  if (!absl::EndsWith(path, kPlanBinarySuffix)) {
    return nproto::LoadTextprotoFile<config::MultiContactPlan,
                                     humanoid_multi_contact::MultiContactPlan>(
        path);
  }
  std::ifstream file{std::string(path), std::ios::binary};
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("cannot read the plan '", path, "'"));
  }
  humanoid_multi_contact::MultiContactPlan message;
  if (!message.ParseFromIstream(&file)) {
    return absl::InvalidArgumentError(
        absl::StrCat("'", path, "' is not a binary MultiContactPlan"));
  }
  config::MultiContactPlan plan;
  RETURN_IF_ERROR(FromProto(message, &plan));
  return plan;
}

}  // namespace

uint64_t contactSetHash(const config::ContactSet& contactSet) {
  humanoid_multi_contact::ContactSet message;
  ToProto(contactSet, &message);
  return fnv1a(deterministicBytes(message));
}

uint64_t terrainSceneHash(const config::TerrainScene& scene) {
  humanoid_multi_contact::TerrainScene message;
  ToProto(scene, &message);
  return fnv1a(deterministicBytes(message));
}

void stampPlanHashes(const config::ContactSet& contactSet,
                     const config::TerrainScene& scene,
                     config::MultiContactPlan& plan) {
  plan.contact_set_hash = contactSetHash(contactSet);
  plan.terrain_scene_hash = terrainSceneHash(scene);
}

absl::Status checkPlanHashes(const config::MultiContactPlan& plan,
                             const config::ContactSet& contactSet,
                             const config::TerrainScene& scene) {
  const uint64_t contactHash = contactSetHash(contactSet);
  if (plan.contact_set_hash != contactHash) {
    return absl::FailedPreconditionError(absl::StrCat(
        "plan '", plan.scenario, "' was made for the contact set of hash ",
        plan.contact_set_hash, ", not for the loaded one of '",
        contactSet.robot, "' (hash ", contactHash, "): plan it again"));
  }
  const uint64_t sceneHash = terrainSceneHash(scene);
  if (plan.terrain_scene_hash != sceneHash) {
    return absl::FailedPreconditionError(absl::StrCat(
        "plan '", plan.scenario, "' was made for the terrain scene of hash ",
        plan.terrain_scene_hash, ", not for the loaded scene '", scene.name,
        "' (hash ", sceneHash, "): plan it again"));
  }
  return absl::OkStatus();
}

absl::StatusOr<config::MultiContactPlan> loadPlanFile(
    absl::string_view path,
    const config::ContactSet& contactSet,
    const config::TerrainScene& scene) {
  ASSIGN_OR_RETURN(config::MultiContactPlan plan, readPlan(path));
  RETURN_IF_ERROR(checkPlanHashes(plan, contactSet, scene));
  return plan;
}

std::string planSummaryPath(absl::string_view binaryPath) {
  absl::string_view stem = binaryPath;
  absl::ConsumeSuffix(&stem, kPlanBinarySuffix);
  return absl::StrCat(stem, kPlanSummarySuffix);
}

absl::Status savePlanFile(const config::MultiContactPlan& plan,
                          absl::string_view binaryPath) {
  if (!absl::EndsWith(binaryPath, kPlanBinarySuffix)) {
    return absl::InvalidArgumentError(
        absl::StrCat("a plan's binary file ends in ", kPlanBinarySuffix, ": '",
                     binaryPath, "'"));
  }
  const std::filesystem::path parent =
      std::filesystem::path(std::string(binaryPath)).parent_path();
  if (!parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
      return absl::InternalError(absl::StrCat(
          "cannot create '", parent.string(), "': ", error.message()));
    }
  }
  humanoid_multi_contact::MultiContactPlan message;
  ToProto(plan, &message);
  RETURN_IF_ERROR(
      writeFile(std::string(binaryPath), deterministicBytes(message)));
  message.clear_knots();
  // The summary names its schema as every textproto of the repository does.
  return writeFile(planSummaryPath(binaryPath),
                   absl::StrCat("# proto-file: ", kPlanProtoFile, "\n",
                                nproto::WriteTextprotoFile(message)));
}

}  // namespace ocs2::humanoid::multi_contact
