
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <fstream>
#include <memory> //shared_ptr
#include <vector>

using namespace std;

class Trump
{
public:
    Trump(cv::Mat initValue, int i=0):myse3(initValue.clone()), id(i)
    {
        cout<<"Trump() "<<i<<endl;
    }
    Trump(const Trump & rhs): myse3(rhs.myse3), id(rhs.id){cout<<"Trump(const Trump) "<<id<<endl;}
    Mat myse3;
    int id;
};
class Frame
{
public:
    Frame(int ain):a(ain){cout<<"Frame(int):"<<a<<endl;}
    ~Frame(){cout<<"~Frame:"<<a<<endl;}
    int a;

    virtual inline void change(int b=30)
    {
        a+=b;
        cout<<"Frame:a:"<<a<<endl;
    }
    virtual void change(string b)
    {
        a+=atoi(b.c_str())*2;
        cout<<"Frame:a:"<<a<<endl;
    }
    virtual bool isKeyFrame(){return false;}
    std::vector<Trump> vTrump;
    std::vector<Trump>& GetTrumps(){
        return vTrump;
    }
};
class KeyFrame:public Frame
{
public:
    KeyFrame(int ain):Frame(ain){}
    bool isKeyFrame(){return true;}
    void change(int b=30);

    void change(string b)
    {
        a+=atoi(b.c_str())*2;
        cout<<"KeyFrame:a:"<<a<<endl;
    }
    void isBad(){
        cout<<"a>100:"<< (bool)(a>100)<<endl;
    }
};
void KeyFrame::change(int b)
{
    a+=b*2;
    cout<<"KeyFrame:a:"<<a<<endl;
}
void TestVirtualFunc()
{
    Frame a(100);
    a.vTrump.push_back(Trump(cv::Mat(), 1));
    a.vTrump.push_back(Trump(cv::Mat(), 2));
    a.vTrump.push_back(Trump(cv::Mat(), 3));

    std::vector<Trump>& me=a.GetTrumps();
    cout<<me[1].id<<endl;
    me[1].id=100;
    cout<< me[1].id<<endl;
    cout<<a.vTrump[1].id<<endl;
    vector<std::shared_ptr<Frame> > vtr;
    std::shared_ptr<Frame> pF(new Frame(100));
    std::shared_ptr<KeyFrame> pKF(new KeyFrame(200));
    std::shared_ptr<Frame> pF2= pKF;
    vtr.push_back(pF);
    vtr.push_back(pKF);
    cout<<vtr.size()<<endl;
    vtr[0]->change("45");
    vtr[1]->change("35");
    pF2->change("20");
//    Frame * pF=new Frame(100);
//    Frame * pKF=new KeyFrame(200);
//    KeyFrame * pKF2=new KeyFrame(300);
//    pF->change();
//    pKF->change(30);
//    pKF->change("20");
//    pF->change("40");

    //    deque<Frame*> ptrs;
    //    ptrs.push_back(pF);
    //    ptrs.push_back(pKF);
    //    ptrs.push_back(pKF2);
    //    Frame obj(30);
    //    ptrs.push_back(&obj);
    //    for(deque<Frame*>::iterator it=ptrs.begin(); it!=ptrs.end();++it)
    //        cout<< (*it)->isKeyFrame()<<endl;
    //    cout<< ptrs.back()->isKeyFrame()<<endl;
    //    int i=0;
    //    for(deque<Frame*>::iterator it=ptrs.begin(); it!=ptrs.end();++it, ++i)
    //    {
    //        cout<<"deleting "<< i<<endl;
    //        delete (*it);

    //    }
    //    cout<<"finished"<<endl;
}

