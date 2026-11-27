#include "tools/clang_tidy/testdata/DiscardedStatus.h"

#include "absl/status/status.h"

namespace clang_tidy_testdata {

absl::Status validate() {
  return absl::OkStatus();
}

void validateAndIgnore() {
  validate();
}

}  // namespace clang_tidy_testdata
