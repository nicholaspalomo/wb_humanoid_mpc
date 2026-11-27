#include "tools/clang_tidy/testdata/Nullability.h"

#include "absl/base/nullability.h"

namespace clang_tidy_testdata {

int take(const int* absl_nonnull value) { return *value; }

int redeclared(const int* absl_nullable value) { return value == nullptr ? 0 : *value; }

int passNull() { return take(nullptr); }

}  // namespace clang_tidy_testdata
