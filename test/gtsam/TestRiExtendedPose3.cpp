
/**
 * @file   TestRiExtendedPose3.cpp
 * @brief  Unit tests for ExtendedPose3 class
 */

#include <gtsam/RiExtendedPose3.h>
#include <gtest/gtest.h>

#include <gtsam/base/numericalDerivative.h>
//#include <gtsam/base/testLie.h>
//#include <gtsam/base/lieProxies.h>

#include <boost/assign/std/vector.hpp> // for operator +=
using namespace boost::assign;

#include <cmath>

using namespace std;
using namespace gtsam;

static const Point3 V(3,0.4,-2.2);
static const Point3 P(0.2,0.7,-2);
static const Rot3 R = Rot3::Rodrigues(0.3,0,0);
static const Point3 V2(-6.5,3.5,6.2);
static const Point3 P2(3.5,-8.2,4.2);
static const ExtendedPose3 T(R,V2,P2);
static const ExtendedPose3 T2(Rot3::Rodrigues(0.3,0.2,0.1),V2,P2);
static const ExtendedPose3 T3(Rot3::Rodrigues(-90, 0, 0), Point3(5,6,7), Point3(1, 2, 3));

/* ************************************************************************* */
TEST( ExtendedPose3, equals)
{
  ExtendedPose3 pose2 = T3;
  EXPECT_TRUE(T3.equals(pose2));
  ExtendedPose3 origin;
  EXPECT_FALSE(T3.equals(origin));
}

/* ************************************************************************* */
#ifndef GTSAM_POSE3_EXPMAP
TEST( ExtendedPose3, retract_first_order)
{
  ExtendedPose3 id;
  Vector xi = Z_9x1;
  xi(0) = 0.3;
  EXPECT_TRUE(assert_equal(ExtendedPose3(R, Vector3(0,0,0), Point3(0,0,0)), id.retract(xi),1e-2));
  xi(3)=3;xi(4)=0.4;xi(5)=-2.2;
  xi(6)=0.2;xi(7)=0.7;xi(8)=-2;
  EXPECT_TRUE(assert_equal(ExtendedPose3(R, V, P),id.retract(v),1e-2));
}
#endif
/* ************************************************************************* */
TEST( ExtendedPose3, retract_expmap)
{
  Vector xi = Z_9x1; xi(0) = 0.3;
  ExtendedPose3 pose = ExtendedPose3::Expmap(xi);
  EXPECT_TRUE(assert_equal(ExtendedPose3(R, Point3(0,0,0), Point3(0,0,0)), pose, 1e-2));
  EXPECT_TRUE(assert_equal(xi,ExtendedPose3::Logmap(pose),1e-2));
}

/* ************************************************************************* */
TEST( ExtendedPose3, expmap_a_full)
{
  ExtendedPose3 id;
  Vector xi = Z_9x1;
  xi(0) = 0.3;
  EXPECT_TRUE(assert_equal(expmap_default<ExtendedPose3>(id, xi), ExtendedPose3(R, Vector3(0,0,0), Point3(0,0,0))));
  xi(3)=-0.2;xi(4)=-0.394742;xi(5)=2.08998;
  xi(6)=0.2;xi(7)=0.394742;xi(8)=-2.08998;
  EXPECT_TRUE(assert_equal(ExtendedPose3(R, -P, P),expmap_default<ExtendedPose3>(id, xi),1e-5));
}

/* ************************************************************************* */
TEST( ExtendedPose3, expmap_a_full2)
{
  ExtendedPose3 id;
  Vector xi = Z_9x1;
  xi(0) = 0.3;
  EXPECT_TRUE(assert_equal(expmap_default<ExtendedPose3>(id, xi), ExtendedPose3(R, Point3(0,0,0), Point3(0,0,0))));
  xi(3)=-0.2;xi(4)=-0.394742;xi(5)=2.08998;
  xi(6)=0.2;xi(7)=0.394742;xi(8)=-2.08998;
  EXPECT_TRUE(assert_equal(ExtendedPose3(R, -P, P),expmap_default<ExtendedPose3>(id, xi),1e-5));
}

/* ************************************************************************* */
TEST(ExtendedPose3, expmap_b)
{
  ExtendedPose3 p1(Rot3(), Vector3(-100, 0, 0), Point3(100, 0, 0));
  ExtendedPose3 p2 = p1.retract((Vector(9) << 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished());
  Rot3 R = Rot3::Rodrigues(0.0, 0.0, 0.1);
  ExtendedPose3 expected(R, R * Point3(-100.0, 0.0, 0.0), R * Point3(100.0, 0.0, 0.0));
  EXPECT_TRUE(assert_equal(expected, p2,1e-2));
}

/* ************************************************************************* */
// test case for screw motion in the plane
namespace screwExtendedPose3 {
  double a=0.3, c=cos(a), s=sin(a), w=0.3;
  Vector xi = (Vector(9) << 0.0, 0.0, w, w, 0.0, 1.0, w, 0.0, 1.0).finished();
  Rot3 expectedR(c, -s, 0, s, c, 0, 0, 0, 1);
  Point3 expectedV(0.29552, 0.0446635, 1);
  Point3 expectedP(0.29552, 0.0446635, 1);
  ExtendedPose3 expected(expectedR, expectedV, expectedP);
}

/* ************************************************************************* */
// Checks correct exponential map (Expmap) with brute force matrix exponential
TEST(ExtendedPose3, expmap_c_full)
{
  EXPECT_TRUE(assert_equal(screwExtendedPose3::expected, ExtendedPose3::Expmap(screwExtendedPose3::xi),1e-6));
}

/* ************************************************************************* */
// assert that T*exp(xi)*T^-1 is equal to exp(Ad_T(xi))
TEST(ExtendedPose3, Adjoint_full)
{
  ExtendedPose3 expected = T * ExtendedPose3::Expmap( screwExtendedPose3::xi) * T.inverse();
  Vector xiprime = T.Adjoint( screwExtendedPose3::xi);
  EXPECT_TRUE(assert_equal(expected, ExtendedPose3::Expmap(xiprime), 1e-6));

  ExtendedPose3 expected2 = T2 * ExtendedPose3::Expmap( screwExtendedPose3::xi) * T2.inverse();
  Vector xiprime2 = T2.Adjoint( screwExtendedPose3::xi);
  EXPECT_TRUE(assert_equal(expected2, ExtendedPose3::Expmap(xiprime2), 1e-6));

  ExtendedPose3 expected3 = T3 * ExtendedPose3::Expmap( screwExtendedPose3::xi) * T3.inverse();
  Vector xiprime3 = T3.Adjoint( screwExtendedPose3::xi);
  EXPECT_TRUE(assert_equal(expected3, ExtendedPose3::Expmap(xiprime3), 1e-6));
}

/* ************************************************************************* */
// assert that T*wedge(xi)*T^-1 is equal to wedge(Ad_T(xi))
TEST(ExtendedPose3, Adjoint_hat)
{
  auto hat = [](const Vector& xi) { return ExtendedPose3::wedge(xi); };
  Matrix5 expected = T.matrix() * hat( screwExtendedPose3::xi) * T.matrix().inverse();
  Matrix5 xiprime = hat(T.Adjoint( screwExtendedPose3::xi));

  EXPECT_TRUE(assert_equal(expected, xiprime, 1e-6));

  Matrix5 expected2 = T2.matrix() * hat( screwExtendedPose3::xi) * T2.matrix().inverse();
  Matrix5 xiprime2 = hat(T2.Adjoint( screwExtendedPose3::xi));
  EXPECT_TRUE(assert_equal(expected2, xiprime2, 1e-6));

  Matrix5 expected3 = T3.matrix() * hat( screwExtendedPose3::xi) * T3.matrix().inverse();

  Matrix5 xiprime3 = hat(T3.Adjoint( screwExtendedPose3::xi));
  EXPECT_TRUE(assert_equal(expected3, xiprime3, 1e-6));
}

/* ************************************************************************* */
TEST(ExtendedPose3, expmaps_galore_full)
{
  Vector xi; ExtendedPose3 actual;
  xi = (Vector(9) << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9).finished();
  actual = ExtendedPose3::Expmap(xi);
  EXPECT_TRUE(assert_equal(xi, ExtendedPose3::Logmap(actual),1e-6));

  xi = (Vector(9) << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -0.7, -0.8, -0.9).finished();
  for (double theta=1.0;0.3*theta<=M_PI;theta*=2) {
    Vector txi = xi*theta;
    actual = ExtendedPose3::Expmap(txi);
    Vector log = ExtendedPose3::Logmap(actual);
    EXPECT_TRUE(assert_equal(actual, ExtendedPose3::Expmap(log),1e-6));
    EXPECT_TRUE(assert_equal(txi,log,1e-6)); // not true once wraps
  }

  // Works with large v as well, but expm needs 10 iterations!
  xi = (Vector(9) << 0.2, 0.3, -0.8, 100.0, 120.0, -60.0, 12, 14, 45).finished();
  actual = ExtendedPose3::Expmap(xi);
  EXPECT_TRUE(assert_equal(xi, ExtendedPose3::Logmap(actual),1e-9));
}


/* ************************************************************************* */
TEST(ExtendedPose3, Adjoint_compose_full)
{
  // To debug derivatives of compose, assert that
  // T1*T2*exp(Adjoint(inv(T2),x) = T1*exp(x)*T2
  const ExtendedPose3& T1 = T;
  Vector x = (Vector(9) << 0.1, 0.1, 0.1, 0.4, 0.2, 0.8, 0.4, 0.2, 0.8).finished();
  ExtendedPose3 expected = T1 * ExtendedPose3::Expmap(x) * T2;
  Vector y = T2.inverse().Adjoint(x);
  ExtendedPose3 actual = T1 * T2 * ExtendedPose3::Expmap(y);
  EXPECT_TRUE(assert_equal(expected, actual, 1e-6));
}

/* ************************************************************************* */
TEST( ExtendedPose3, compose_inverse)
{
  Matrix actual = (T*T.inverse()).matrix();
  Matrix expected = I_5x5;
  EXPECT_TRUE(assert_equal(actual,expected,1e-8));
}

/* ************************************************************************* */
TEST(ExtendedPose3, retract_localCoordinates)
{
  Vector9 d12;
  d12 << 1,2,3,4,5,6,7,8,9; d12/=10;
  ExtendedPose3 t1 = T, t2 = t1.retract(d12);
  EXPECT_TRUE(assert_equal(d12, t1.localCoordinates(t2)));
}
/* ************************************************************************* */
TEST(ExtendedPose3, expmap_logmap)
{
  Vector d12 = Vector9::Constant(0.1);
  ExtendedPose3 t1 = T, t2 = t1.expmap(d12);
  EXPECT_TRUE(assert_equal(d12, t1.logmap(t2)));
}

/* ************************************************************************* */
TEST(ExtendedPose3, retract_localCoordinates2)
{
  ExtendedPose3 t1 = T;
  ExtendedPose3 t2 = T3;
  ExtendedPose3 origin;
  Vector d12 = t1.localCoordinates(t2);
  EXPECT_TRUE(assert_equal(t2, t1.retract(d12)));
  Vector d21 = t2.localCoordinates(t1);
  EXPECT_TRUE(assert_equal(t1, t2.retract(d21)));
  EXPECT_TRUE(assert_equal(d12, -d21));
}
/* ************************************************************************* */
TEST(ExtendedPose3, manifold_expmap)
{
  ExtendedPose3 t1 = T;
  ExtendedPose3 t2 = T3;
  ExtendedPose3 origin;
  Vector d12 = t1.logmap(t2);
  EXPECT_TRUE(assert_equal(t2, t1.expmap(d12)));
  Vector d21 = t2.logmap(t1);
  EXPECT_TRUE(assert_equal(t1, t2.expmap(d21)));

  // Check that log(t1,t2)=-log(t2,t1)
  EXPECT_TRUE(assert_equal(d12,-d21));
}

/* ************************************************************************* */
TEST(ExtendedPose3, subgroups)
{
  // Frank - Below only works for correct "Agrawal06iros style expmap
  // lines in canonical coordinates correspond to Abelian subgroups in SE(3)
   Vector d = (Vector(9) << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9).finished();
  // exp(-d)=inverse(exp(d))
   EXPECT_TRUE(assert_equal(ExtendedPose3::Expmap(-d),ExtendedPose3::Expmap(d).inverse()));
  // exp(5d)=exp(2*d+3*d)=exp(2*d)exp(3*d)=exp(3*d)exp(2*d)
   ExtendedPose3 T2 = ExtendedPose3::Expmap(2*d);
   ExtendedPose3 T3 = ExtendedPose3::Expmap(3*d);
   ExtendedPose3 T5 = ExtendedPose3::Expmap(5*d);
   EXPECT_TRUE(assert_equal(T5,T2*T3));
   EXPECT_TRUE(assert_equal(T5,T3*T2));
}


/* ************************************************************************* */
TEST( ExtendedPose3, adjointMap) {
  Matrix res = ExtendedPose3::adjointMap( screwExtendedPose3::xi);
  Matrix wh = skewSymmetric( screwExtendedPose3::xi(0),  screwExtendedPose3::xi(1),  screwExtendedPose3::xi(2));
  Matrix vh = skewSymmetric( screwExtendedPose3::xi(3),  screwExtendedPose3::xi(4),  screwExtendedPose3::xi(5));
  Matrix rh = skewSymmetric( screwExtendedPose3::xi(6),  screwExtendedPose3::xi(7),  screwExtendedPose3::xi(8));
  Matrix9 expected;
  expected << wh, Z_3x3, Z_3x3, vh, wh, Z_3x3, rh, Z_3x3, wh;
  EXPECT_TRUE(assert_equal(expected,res,1e-5));
}

/* ************************************************************************* */
// This will not pass because numericalDerivative21 takes Retract like Xexp(\xi) rather than exp(\xi)X.
//TEST( ExtendedPose3, ExpmapDerivative1) {
//  Matrix9 actualH;
//  Vector9 w; w << 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
//  ExtendedPose3::Expmap(w,actualH);
//  Matrix expectedH = numericalDerivative21<ExtendedPose3, Vector9,
//      OptionalJacobian<9, 9> >(&ExtendedPose3::Expmap, w, boost::none);
//  EXPECT_TRUE(assert_equal(expectedH, actualH));
//}

/* ************************************************************************* */
TEST( ExtendedPose3, LogmapDerivative) {
  Matrix9 actualH;
  Vector9 w; w << 0.1, 0.2, 0.3, 4.0, 5.0, 6.0,7.0,8.0,9.0;
  ExtendedPose3 p = ExtendedPose3::Expmap(w);
  EXPECT_TRUE(assert_equal(w, ExtendedPose3::Logmap(p,actualH), 1e-5));
  Matrix expectedH = numericalDerivative21<Vector9, ExtendedPose3,
      OptionalJacobian<9, 9> >(&ExtendedPose3::Logmap, p, boost::none);
//  EXPECT_TRUE(assert_equal(expectedH, actualH));
}

/* ************************************************************************* */
TEST( ExtendedPose3, stream)
{
  ExtendedPose3 T;
  std::ostringstream os;
  os << T;
  EXPECT_TRUE(os.str() == "\n|1, 0, 0|\n|0, 1, 0|\n|0, 0, 1|\nv:[0, 0, 0];\np:[0, 0, 0];\n");
}

/* ************************************************************************* */
// This will not pass because numericalDerivative21 takes Retract like Xexp(\xi) rather than exp(\xi)X.
//TEST(ExtendedPose3, Create) {
//  Matrix93 actualH1, actualH2, actualH3;
//  ExtendedPose3 actual = ExtendedPose3::Create(R, V2, P2, actualH1, actualH2, actualH3);
//  EXPECT_TRUE(assert_equal(T, actual));
//  boost::function<ExtendedPose3(Rot3,Point3,Point3)> create = boost::bind(ExtendedPose3::Create,_1,_2,_3,boost::none,boost::none,boost::none);
//  EXPECT_TRUE(assert_equal(numericalDerivative31<ExtendedPose3,Rot3,Point3,Point3>(create, R, V2, P2), actualH1, 1e-9));
//  EXPECT_TRUE(assert_equal(numericalDerivative32<ExtendedPose3,Rot3,Point3,Point3>(create, R, V2, P2), actualH2, 1e-9));
//  EXPECT_TRUE(assert_equal(numericalDerivative33<ExtendedPose3,Rot3,Point3,Point3>(create, R, V2, P2), actualH3, 1e-9));
//}

/* ************************************************************************* */
TEST(ExtendedPose3, print) {
  std::stringstream redirectStream;
  std::streambuf* ssbuf = redirectStream.rdbuf();
  std::streambuf* oldbuf  = std::cout.rdbuf();
  // redirect cout to redirectStream
  std::cout.rdbuf(ssbuf);

  ExtendedPose3 pose(Rot3::identity(), Vector3(1, 2, 3), Point3(1, 2, 3));
  // output is captured to redirectStream
  pose.print();

  // Generate the expected output
  std::stringstream expected;
  Vector3 velocity(1, 2, 3);
  Point3 position(1, 2, 3);

#ifdef GTSAM_TYPEDEF_POINTS_TO_VECTORS
  expected << "1\n"
              "2\n"
              "3;\n";
#else
  expected << "v:[" << velocity.x() << ", " << velocity.y() << ", " << velocity.z() << "]';\np:[" << position.x() << ", " << position.y() << ", " << position.z() << "]';\n";
#endif

  // reset cout to the original stream
  std::cout.rdbuf(oldbuf);

  // Get substring corresponding to position part
  std::string actual = redirectStream.str().substr(41);
  EXPECT_EQ(expected.str(), actual);
}
