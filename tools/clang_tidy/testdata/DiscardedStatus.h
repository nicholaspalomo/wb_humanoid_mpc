#pragma once

#include "absl/status/status.h"

namespace clang_tidy_testdata {

absl::Status validate();
void validateAndIgnore();

}  // namespace clang_tidy_testdata
