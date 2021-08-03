#include <gtest/gtest.h>
#include <iostream>
#include <memory>

using namespace std;

class Base {
 public:
  Base(int a = 20) : a_(a) {}
  virtual ~Base() {}
  virtual void print() const { std::cout << "val " << a_ << std::endl; }
  int a_;

  static const int id;
};

const int Base::id = 10;

class Derived : public Base {
 public:
  Derived() : Base(30) {}
  virtual ~Derived() {}
  static const int id;
};

const int Derived::id = 11;

std::shared_ptr<Base> createObject(int typeId) {
  std::shared_ptr<Base> object;
  switch (typeId) {
    case Base::id:
      object.reset(new Base());
      break;
    case Derived::id:
      object.reset(new Derived());
      break;
  }
  return object;
}

TEST(StandardC, constStaticUnderivable) {
  int typeId = 10;
  std::shared_ptr<Base> object = createObject(typeId);
  EXPECT_EQ(object->a_, 20);
  EXPECT_EQ(object->id, 10);
  typeId = 11;
  object = createObject(typeId);
  EXPECT_EQ(object->a_, 30);
  EXPECT_EQ(object->id, 10);
}
