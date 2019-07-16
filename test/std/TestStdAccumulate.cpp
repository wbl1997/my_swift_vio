
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>

TEST(StandardC,StdAccumulate)
{
    std::vector<bool> vInliers;
    int myres=0;
    for (int jack=0; jack<100;++jack){
        int cur=std::rand()%2;
        vInliers.push_back((bool)(cur));
        myres+=cur;
    }
    int res=std::accumulate(vInliers.begin(),vInliers.end(), 0);
    ASSERT_EQ(res, myres);
}
