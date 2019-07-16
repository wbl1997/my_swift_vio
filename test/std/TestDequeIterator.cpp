#include <deque>
#include <vector>
#include <iostream>

// Finding: (1) to remove entries based on their stored iterators are fragile, does not work for large deque

void testDequeIterator()
{
    std::deque<int> q;
    size_t totalLength = 10;
    for(size_t jack =0; jack< totalLength; ++jack)
        q.push_back(jack);

    std::vector<std::deque<int>::iterator> vToRemove;
    for(size_t jack =0; jack<totalLength; jack+=2)
        vToRemove.push_back(q.begin() + jack);
    for(size_t jack =0; jack<vToRemove.size(); ++jack)
    {
        q.erase(vToRemove[jack]);
    }

    size_t counter =0;
    std::cout <<"max value in deque "<< q.back()<<std::endl;
    try{
        for(auto it =q.begin(); it!=q.end(); ++it, ++counter){
            std::cout <<*it << " "<< (counter*2+1) << std::endl;
            if(*it != (int)(counter*2+1))
                throw(*it);
        }
    }
    catch(int e)
    {
        std::cout <<"even number in deque is not properly deleted at "<<e<<std::endl;
    }

    q.clear();
    totalLength = 2e6;
    int checkOddNumber = 1e4+1; //must be smaller than totalLength
    for(size_t jack =0; jack< totalLength; ++jack)
        q.push_back(jack);
    for(auto it=q.begin(); it!=q.end();){
        if((*it)%2 ==0)
        {
            it = q.erase(it);
        }
        else
            ++it;
        if(*it>= checkOddNumber)
        {
            std::deque<int>::iterator idPos = std::find(q.begin(), q.end(), checkOddNumber);
            assert(idPos - q.begin() == checkOddNumber/2);
        }
    }
}
