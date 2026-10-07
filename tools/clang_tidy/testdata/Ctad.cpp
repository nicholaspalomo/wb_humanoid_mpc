#include "tools/clang_tidy/testdata/Ctad.h"

namespace clang_tidy_testdata {

int boxedValue() {
  const Box box(3);
  return box.value();
}

}  // namespace clang_tidy_testdata
