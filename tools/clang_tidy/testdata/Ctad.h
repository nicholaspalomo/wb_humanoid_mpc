#pragma once

namespace clang_tidy_testdata {

template <typename T>
class Box {
 public:
  explicit Box(T value) : value_(value) {}
  T value() const { return value_; }

 private:
  T value_;
};

int boxedValue();

}  // namespace clang_tidy_testdata
