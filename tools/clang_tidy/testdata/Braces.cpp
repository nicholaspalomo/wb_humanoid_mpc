#include "tools/clang_tidy/testdata/Braces.h"

namespace clang_tidy_testdata {

int clampOnTwoLines(int value) {
  if (value < 0)
    return 0;
  return value;
}

int clampOnOneLine(int value) {
  if (value < 0) return 0;
  return value;
}

}  // namespace clang_tidy_testdata
