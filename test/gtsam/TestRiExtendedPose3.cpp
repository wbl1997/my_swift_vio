
/**
 * @file   TestRiExtendedPose3.cpp
 * @brief  Unit tests for RiExtendedPose3 class
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
static const RiExtendedPose3 T(R,V2,P2);
static const RiExtendedPose3 T2(Rot3::Rodrigues(0.3,0.2,0.1),V2,P2);
static const RiExtendedPose3 T3(Rot3::Rodrigues(-90, 0, 0), Point3(5,6,7), Point3(1, 2, 3));

/* ************************************************************************* */
TEST( RiExtendedPose3, equals)
{
  RiExtendedPose3 pose2 = T3;
  EXPECT_TRUE(T3.equals(pose2));
  RiExtendedPose3 origin;
  EXPECT_FALSE(T3.equals(origin));
}

/* ************************************************************************* */
#ifndef GTSAM_POSE3_EXPMAP
TEST( RiExtendedPose3, retract_first_order)
{
  RiExtendedPose3 id;
  Vector xi = Z_9x1;
  xi(0) = 0.3;
  EXPECT_TRUE(assert_equal(RiExtendedPose3(R, Vector3(0,0,0), Point3(0,0,0)), id.retract(xi),1e-2));
  xi(3)=3;xi(4)=0.4;xi(5)=-2.2;
  xi(6)=0.2;xi(7)=0.7;xi(8)=-2;
  EXPECT_TRUE(assert_equal(RiExtendedPose3(R, V, P),id.retract(v),1e-2));
}
#endif
/* ************************************************************************* */
TEST( RiExtendedPose3, retract_expmap)
{
  Vector xi = Z_9x1; xi(0) = 0.3;
  RiExtendedPose3 pose = RiExtendedPose3::Expmap(xi);
  EXPECT_TRUE(assert_equal(RiExtendedPose3(R, Point3(0,0,0), Point3(0,0,0)), pose, 1e-2));
  EXPECT_TRUE(assert_equal(xi,RiExtendedPose3::Logmap(pose),1e-2));
}

/* ************************************************************************* */
TEST( RiExtendedPose3, expmap_a_full)
{
  RiExtendedPose3 id;
  Vector xi = Z_9x1;
  xi(0) = 0.3;
  EXPECT_TRUE(assert_equal(expmap_default<RiExtendedPose3>(id, xi), RiExtendedPose3(R, Vector3(0,0,0), Point3(0,0,0))));
  xi(3)=-0.2;xi(4)=-0.394742;xi(5)=2.08998;
  xi(6)=0.2;xi(7)=0.394742;xi(8)=-2.08998;
  EXPECT_TRUE(assert_equal(RiExtendedPose3(R, -P, P),expmap_default<RiExtendedPose3>(id, xi),1e-5));
}

/* ************************************************************************* */
TEST( RiExtendedPose3, expmap_a_full2)
{
  RiExtendedPose3 id;
  Vector xi = Z_9x1;
  xi(0) = 0.3;
  EXPECT_TRUE(assert_equal(expmap_default<RiExtendedPose3>(id, xi), RiExtendedPose3(R, Point3(0,0,0), Point3(0,0,0))));
  xi(3)=-0.2;xi(4)=-0.394742;xi(5)=2.08998;
  xi(6)=0.2;xi(7)=0.394742;xi(8)=-2.08998;
  EXPECT_TRUE(assert_equal(RiExtendedPose3(R, -P, P),expmap_default<RiExtendedPose3>(id, xi),1e-5));
}

/* ************************************************************************* */
TEST(RiExtendedPose3, expmap_b)
{
  RiExtendedPose3 p1(Rot3(), Vector3(-100, 0, 0), Point3(100, 0, 0));
  RiExtendedPose3 p2 = p1.retract((Vector(9) << 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished());
  Rot3 R = Rot3::Rodrigues(0.0, 0.0, 0.1);
  RiExtendedPose3 expected(R, R * Point3(-100.0, 0.0, 0.0), R * Point3(100.0, 0.0, 0.0));
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
  RiExtendedPose3 expected(expectedR, expectedV, expectedP);
}

/* ************************************************************************* */
// Checks correct exponential map (Expmap) with brute force matrix exponential
TEST(RiExtendedPose3, expmap_c_full)
{
  EXPECT_TRUE(assert_equal(screwExtendedPose3::expected, RiExtendedPose3::Expmap(screwExtendedPose3::xi),1e-6));
}

/* ************************************************************************* */
// assert that T*exp(xi)*T^-1 is equal to exp(Ad_T(xi))
TEST(RiExtendedPose3, Adjoint_full)
{
  RiExtendedPose3 expected = T * RiExtendedPose3::Expmap( screwExtendedPose3::xi) * T.inverse();
  Vector xiprime = T.Adjoint( screwExtendedPose3::xi);
  EXPECT_TRUE(assert_equal(expected, RiExtendedPose3::Expmap(xiprime), 1e-6));

  RiExtendedPose3 expected2 = T2 * RiExtendedPose3::Expmap( screwExtendedPose3::xi) * T2.inverse();
  Vector xiprime2 = T2.Adjoint( screwExtendedPose3::xi);
  EXPECT_TRUE(assert_equal(expected2, RiExtendedPose3::Expmap(xiprime2), 1e-6));

  RiExtendedPose3 expected3 = T3 * RiExtendedPose3::Expmap( screwExtendedPose3::xi) * T3.inverse();
  Vector xiprime3 = T3.Adjoint( screwExtendedPose3::xi);
  EXPECT_TRUE(assert_equal(expected3, RiExtendedPose3::Expmap(xiprime3), 1e-6));
}

/* ************************************************************************* */
// assert that T*wedge(xi)*T^-1 is equal to wedge(Ad_T(xi))
TEST(RiExtendedPose3, Adjoint_hat)
{
  auto hat = [](const Vector& xi) { return RiExtendedPose3::wedge(xi); };
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
TEST(RiExtendedPose3, expmaps_galore_full)
{
  Vector xi; RiExtendedPose3 actual;
  xi = (Vector(9) << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9).finished();
  actual = RiExtendedPose3::Expmap(xi);
  EXPECT_TRUE(assert_equal(xi, RiExtendedPose3::Logmap(actual),1e-6));

  xi = (Vector(9) << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, -0.7, -0.8, -0.9).finished();
  for (double theta=1.0;0.3*theta<=M_PI;theta*=2) {
    Vector txi = xi*theta;
    actual = RiExtendedPose3::Expmap(txi);
    Vector log = RiExtendedPose3::Logmap(actual);
    EXPECT_TRUE(assert_equal(actual, RiExtendedPose3::Expmap(log),1e-6));
    EXPECT_TRUE(assert_equal(txi,log,1e-6)); // not true once wraps
  }

  // Works with large v as well, but expm needs 10 iterations!
  xi = (Vector(9) << 0.2, 0.3, -0.8, 100.0, 120.0, -60.0, 12, 14, 45).finished();
  actual = RiExtendedPose3::Expmap(xi);
  EXPECT_TRUE(assert_equal(xi, RiExtendedPose3::Logmap(actual),1e-9));
}


/* ************************************************************************* */
TEST(RiExtendedPose3, Adjoint_compose_full)
{
  // To debug derivatives of compose, assert that
  // T1*T2*exp(Adjoint(inv(T2),x) = T1*exp(x)*T2
  const RiExtendedPose3& T1 = T;
  Vector x = (Vector(9) << 0.1, 0.1, 0.1, 0.4, 0.2, 0.8, 0.4, 0.2, 0.8).finished();
  RiExtendedPose3 expected = T1 * RiExtendedPose3::Expmap(x) * T2;
  Vector y = T2.inverse().Adjoint(x);
  RiExtendedPose3 actual = T1 * T2 * RiExtendedPose3::Expmap(y);
  EXPECT_TRUE(assert_equal(expected, actual, 1e-6));
}

/* ************************************************************************* */
TEST( RiExtendedPose3, compose_inverse)
{
  Matrix actual = (T*T.inverse()).matrix();
  Matrix expected = I_5x5;
  EXPECT_TRUE(assert_equal(actual,expected,1e-8));
}

/* ************************************************************************* */
TEST(RiExtendedPose3, retract_localCoordinates)
{
  Vector9 d12;
  d12 << 1,2,3,4,5,6,7,8,9; d12/=10;
  RiExtendedPose3 t1 = T, t2 = t1.retract(d12);
  EXPECT_TRUE(assert_equal(d12, t1.localCoordinates(t2)));
}
/* ************************************************************************* */
TEST(RiExtendedPose3, expmap_logmap)
{
  Vector d12 = Vector9::Constant(0.1);
  RiExtendedPose3 t1 = T, t2 = t1.expmap(d12);
  EXPECT_TRUE(assert_equal(d12, t1.logmap(t2)));
}

/* ************************************************************************* */
TEST(RiExtendedPose3, retract_localCoordinates2)
{
  RiExtendedPose3 t1 = T;
  RiExtendedPose3 t2 = T3;
  RiExtendedPose3 origin;
  Vector d12 = t1.localCoordinates(t2);
  EXPECT_TRUE(assert_equal(t2, t1.retract(d12)));
  Vector d21 = t2.localCoordinates(t1);
  EXPECT_TRUE(assert_equal(t1, t2.retract(d21)));
  EXPECT_TRUE(assert_equal(d12, -d21));
}
/* ************************************************************************* */
TEST(RiExtendedPose3, manifold_expmap)
{
  RiExtendedPose3 t1 = T;
  RiExtendedPose3 t2 = T3;
  RiExtendedPose3 origin;
  Vector d12 = t1.logmap(t2);
  EXPECT_TRUE(assert_equal(t2, t1.expmap(d12)));
  Vector d21 = t2.logmap(t1);
  EXPECT_TRUE(assert_equal(t1, t2.expmap(d21)));

  // Check that log(t1,t2)=-log(t2,t1)
  EXPECT_TRUE(assert_equal(d12,-d21));
}

/* ************************************************************************* */
TEST(RiExtendedPose3, subgroups)
{
  // Frank - Below only works for correct "Agrawal06iros style expmap
  // lines in canonical coordinates correspond to Abelian subgroups in SE(3)
   Vector d = (Vector(9) << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9).finished();
  // exp(-d)=inverse(exp(d))
   EXPECT_TRUE(assert_equal(RiExtendedPose3::Expmap(-d),RiExtendedPose3::Expmap(d).inverse()));
  // exp(5d)=exp(2*d+3*d)=exp(2*d)exp(3*d)=exp(3*d)exp(2*d)
   RiExtendedPose3 T2 = RiExtendedPose3::Expmap(2*d);
   RiExtendedPose3 T3 = RiExtendedPose3::Expmap(3*d);
   RiExtendedPose3 T5 = RiExtendedPose3::Expmap(5*d);
   EXPECT_TRUE(assert_equal(T5,T2*T3));
   EXPECT_TRUE(assert_equal(T5,T3*T2));
}


/* ************************************************************************* */
TEST( RiExtendedPose3, adjointMap) {
  Matrix res = RiExtendedPose3::adjointMap( screwExtendedPose3::xi);
  Matrix wh = skewSymmetric( screwExtendedPose3::xi(0),  screwExtendedPose3::xi(1),  screwExtendedPose3::xi(2));
  Matrix vh = skewSymmetric( screwExtendedPose3::xi(3),  screwExtendedPose3::xi(4),  screwExtendedPose3::xi(5));
  Matrix rh = skewSymmetric( screwExtendedPose3::xi(6),  screwExtendedPose3::xi(7),  screwExtendedPose3::xi(8));
  Matrix9 expected;
  expected << wh, Z_3x3, Z_3x3, vh, wh, Z_3x3, rh, Z_3x3, wh;
  EXPECT_TRUE(assert_equal(expected,res,1e-5));
}

/* ************************************************************************* */
// This will not pass because Expmap computes Jacobians for the left invariant
// error which differs from the right invariant error in retract().
//TEST( RiExtendedPose3, ExpmapDerivative1) {
//  Matrix9 actualH;
//  Vector9 w; w << 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
//  RiExtendedPose3::Expmap(w,actualH);
//  Matrix expectedH = numericalDerivative21<RiExtendedPose3, Vector9,
//      OptionalJacobian<9, 9> >(&RiExtendedPose3::Expmap, w, boost::none);
//  EXPECT_TRUE(assert_equal(expectedH, actualH));
//}

/* ************************************************************************* */
//TEST( RiExtendedPose3, LogmapDerivative) {
//  Matrix9 actualH;
//  Vector9 w; w << 0.1, 0.2, 0.3, 4.0, 5.0, 6.0,7.0,8.0,9.0;
//  RiExtendedPose3 p = RiExtendedPose3::Expmap(w);
//  EXPECT_TRUE(assert_equal(w, RiExtendedPose3::Logmap(p,actualH), 1e-5));
//  Matrix expectedH = numericalDerivative21<Vector9, RiExtendedPose3,
//      OptionalJacobian<9, 9> >(&RiExtendedPose3::Logmap, p, boost::none);
//  EXPECT_TRUE(assert_equal(expectedH, actualH));
//}

/* ************************************************************************* */
TEST( RiExtendedPose3, stream)
{
  RiExtendedPose3 T;
  std::ostringstream os;
  os << T;
  EXPECT_TRUE(os.str() == "\n|1, 0, 0|\n|0, 1, 0|\n|0, 0, 1|\nv:[0, 0, 0];\np:[0, 0, 0];\n");
}

/* ************************************************************************* */
// This will not pass because Create computes Jacobians for the left invariant
// error which differs from the right invariant error in retract().

//TEST(RiExtendedPose3, Create) {
//  Matrix93 actualH1, actualH2, actualH3;
//  RiExtendedPose3 actual = RiExtendedPose3::Create(R, V2, P2, actualH1, actualH2, actualH3);
//  EXPECT_TRUE(assert_equal(T, actual));
//  boost::function<RiExtendedPose3(Rot3,Point3,Point3)> create = boost::bind(RiExtendedPose3::Create,_1,_2,_3,boost::none,boost::none,boost::none);
//  EXPECT_TRUE(assert_equal(numericalDerivative31<RiExtendedPose3,Rot3,Point3,Point3>(create, R, V2, P2), actualH1, 1e-9));
//  EXPECT_TRUE(assert_equal(numericalDerivative32<RiExtendedPose3,Rot3,Point3,Point3>(create, R, V2, P2), actualH2, 1e-9));
//  EXPECT_TRUE(assert_equal(numericalDerivative33<RiExtendedPose3,Rot3,Point3,Point3>(create, R, V2, P2), actualH3, 1e-9));
//}

/* ************************************************************************* */
TEST(RiExtendedPose3, print) {
  std::stringstream redirectStream;
  std::streambuf* ssbuf = redirectStream.rdbuf();
  std::streambuf* oldbuf  = std::cout.rdbuf();
  // redirect cout to redirectStream
  std::cout.rdbuf(ssbuf);

  RiExtendedPose3 pose(Rot3::identity(), Vector3(1, 2, 3), Point3(1, 2, 3));
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
