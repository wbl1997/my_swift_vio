#include <gtest/gtest.h>

#include <iostream>
#include <memory>

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

struct A {
  explicit A(int f) : foo(f) {}
  virtual ~A() {}
  virtual void plus() { ++foo; }
  virtual void print() const { std::cout << "foo " << foo << "\n"; }
  virtual bool equals(const A& rhs) const { return rhs.foo == foo; }
  int foo;
};

struct B : public A {
  explicit B(int b) : A(b / 2), bar(b) {}
  virtual ~B() {}
  void plus() override {
    A::plus();
    bar += 2;
  }
  void print() const override {
    A::print();
    std::cout << "bar: " << bar << std::endl;
  }

  bool equals(const A& obj) const override {
    const auto& rhs = static_cast<const B&>(obj);

    return A::equals(obj) && bar == rhs.bar;
  }
  int bar;
};

TEST(StandardC, AssignToBaseClass) {
  // This test shows that assigning to base class object causes object slicing.
  std::shared_ptr<A> b, backup;
  b.reset(new B(100));
  backup.reset(new B(*std::static_pointer_cast<B>(b)));

  b->plus();

//  std::cout << "Before assignment\nb\n";
//  b->print();
//  std::cout << "backup\n";
//  backup->print();
  *b = *backup;

//  std::cout << "After assignment\n";
//  b->print();

  bool res = b->equals(*backup);
  EXPECT_FALSE(res);
}

struct S {
  virtual std::string f() const { return "B::f" + g(); }
  std::string g() const { return "B::g"; }  // not virtual
};

struct E : S {
  std::string f() const override { return "D::f" + g(); }  // overrides B::f
  std::string g() const { return "D::g"; }
};

TEST(StandardC, VirtualMemberCallOverride) {
  // This test shows that the virtual member for a derived class will call the
  // member function of the derived class rather than the counterpart in the
  // Base class.
  std::shared_ptr<S> dptr(new E());
  std::shared_ptr<E> d(new E());
  EXPECT_TRUE(dptr->f() == d->f());
}
