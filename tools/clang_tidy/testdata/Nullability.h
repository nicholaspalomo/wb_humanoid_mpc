#pragma once

#include "absl/base/nullability.h"

namespace clang_tidy_testdata {

int take(const int* absl_nonnull value);

int redeclared(const int* absl_nonnull value);

int passNull();

}  // namespace clang_tidy_testdata
