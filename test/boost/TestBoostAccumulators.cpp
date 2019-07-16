
#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/moment.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>


TEST(Boost, Accumulators)
{
    using namespace std;
    using namespace boost::accumulators;

    // Define an accumulator set for calculating the mean and the
    // 2nd moment ...

    boost::accumulators::accumulator_set<
          double,
          boost::accumulators::features<
        boost::accumulators::tag::sum,
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::rolling_mean,
        boost::accumulators::tag::mean
        >
          > acc(boost::accumulators::tag::rolling_window::window_size = 7);

    // push in some data ...
    acc(1.2);
    acc(2.3);
    acc(3.4);
    acc(4.5);

    // Display the results ...
    ASSERT_EQ(boost::accumulators::mean(acc), 2.85);
}
