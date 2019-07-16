//#include <iostream>
//#include <iterator>
#include <vector>
#include <map>

#include <gtest/gtest.h>


TEST(StandardC, ReverseIterator)
{
    std::map<int, int> v;
    v[0]= 0;
    v[1]= 2;
    v[2]= 4;
    v[3]= 6;
    v[4]= 8;
    v[5]= 10;

    std::vector<int> r={1,3,5};

    for (auto it = v.rbegin(); it != v.rend(); ) {
        if(it->first%2 ==0)
        {
            std::advance(it, 1);
            v.erase(it.base());
        }
        else{
            ++it;
        }
    }
    std::vector<int>::const_iterator iter = r.begin();
    for (auto it = v.begin(); it != v.end(); ++it, ++iter) {
        ASSERT_EQ(*iter, it->first);
    }
}
