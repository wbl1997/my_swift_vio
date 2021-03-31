// This test shows that it is possible to call member functions at creating an
// object.

#include <gtest/gtest.h>

class Bar {
 public:
  Bar() { foo = 0; };
  ~Bar() {}
  Bar& createBar(int a) {
    foo = a;
    return *this;
  }

  Bar& operator=(const Bar& bar) {
    if (this == &bar) {
      return *this;
    } else {
      foo = bar.foo;
      return *this;
    }
  }

  int foo;
};

TEST(StandardC, CallMemberFunctionAtConstruction) {
  Bar b = b.createBar(50);
  EXPECT_EQ(b.foo, 50);
}
