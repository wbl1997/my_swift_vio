#include <iostream>
#include <random>

#include <gtest/gtest.h>
class Foo {
 public:
  explicit Foo(int a) : bar_(a) { }
  void plus(int delta, Foo& out) const {
    out.bar_ = bar_ + delta;
  }
  int bar_;
};

TEST(StandardC, ModifyInConstMethod) {
  int origin = std::rand() % 47;
  int delta = std::rand() % 47;
  Foo foo(origin);
  foo.plus(delta, foo);
  EXPECT_EQ(foo.bar_, origin + delta);
}
