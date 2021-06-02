#include <okvis/Time.hpp>
#include <gtest/gtest.h>

TEST(OkvisTime, smallMinusLarge) {
    okvis::Time right(0, 30034);
    okvis::Duration dura(2, 230);

    okvis::Time left = okvis::minusSafe(right, dura);
    EXPECT_EQ(left.sec, 0);
    EXPECT_EQ(left.nsec, 0);
}
