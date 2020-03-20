#include <gtest/gtest.h>

class Base {
 protected:
  std::string id;
  std::vector<int> vals;

 public:
  Base() : id("base"), vals{1, 2, 3, 4, 5} {}
  Base(const std::string name, const std::vector<int>& vec)
      : id(name), vals(vec) {}
  int valAt(int index) const { return vals[index]; }
  void setValAt(int index, int val) { vals[index] = val; }
  void print() const {
    std::cout << "id " << id;
    for (auto val : vals) {
      std::cout << " " << val;
    }
    std::cout << std::endl;
  }
};

class Derived : public Base {
  std::vector<int> vals2;

 public:
  Derived() : Base("derived", {0, 6, 7, 8, 9}), vals2{10, 11, 12, 13, 14} {}
  int valAt(int index) const { return vals[index + 1]; }
  int valAt2(int index) const { return vals2[index]; }
  void setValAt(int index, int val) { vals[index + 1] = val; }
  void print() const {
    std::cout << "id " << id;
    for (auto val : vals) {
      std::cout << " " << val;
    }
    for (auto val : vals2) {
      std::cout << " " << val;
    }
    std::cout << std::endl;
  }
};

class Foo {
 public:
  Foo() {}
  void get(const Base& base) {
    std::cout << "val at index 1 " << base.valAt(1) << std::endl;
  }
  void set(Base& base) {
    base.setValAt(2, 100);
    std::cout << "val at index 2 set to 100 " << std::endl;
  }
};

TEST(StandardC, DerivedClass) {
  Foo foo;
  Base b;
  Derived d;

  foo.get(b);
  foo.set(b);
  b.print();
  EXPECT_EQ(b.valAt(0), 1);
  EXPECT_EQ(b.valAt(2), 100);
  EXPECT_EQ(b.valAt(4), 5);

  foo.get(d);
  // set() will use the base::set for the derived object,
  // but it will not slice the derived instance.
  foo.set(d);
  d.print();
  EXPECT_EQ(d.valAt(0), 6);
  EXPECT_EQ(d.valAt(1), 100);
  EXPECT_EQ(d.valAt(2), 8);
  EXPECT_EQ(d.valAt(3), 9);
  EXPECT_EQ(d.valAt2(0), 10);
  EXPECT_EQ(d.valAt2(4), 14);
}
