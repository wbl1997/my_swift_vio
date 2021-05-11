#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
using namespace std;
typedef unsigned char uchar;
TEST(StandardC, BinaryOperators)
{
    uchar b= 0x01;
    uchar c= 0x02;
    uchar d= b|c;
    d &=b;
    uchar f= d&c;
    uchar g= b&c;
    ASSERT_EQ(d, 0x01);
    ASSERT_EQ(f, 0x00);
    ASSERT_EQ(g, 0x00);
}

TEST(StandardC, Bool)
{
    bool a=true, b=false;
    ASSERT_EQ(a, 1);
    ASSERT_EQ(b, 0);
}
