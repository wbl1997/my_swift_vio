#include <okvis/Time.hpp>
#include <gtest/gtest.h>

okvis::Time minusSafe(okvis::Time right, okvis::Duration dura) {
  if (right > okvis::Time(dura.sec, dura.nsec)) {
    return right - dura;
  } else {
    return okvis::Time(0, 0);
  }
}

TEST(OkvisTime, smallMinusLarge) {
    okvis::Time right(0, 30034);
    okvis::Duration dura(2, 230);

    okvis::Time left = minusSafe(right, dura);
    EXPECT_EQ(left.sec, 0);
    EXPECT_EQ(left.nsec, 0);
}
