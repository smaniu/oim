/*
 Copyright (c) 2015-2017 Paul Lagrée, Siyu Lei, Silviu Maniu, Luyi Mo

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#ifndef __oim__BetaInfluence__
#define __oim__BetaInfluence__

#include "common.hpp"
#include "InfluenceDistribution.hpp"

#include <boost/math/distributions.hpp>
#include <random>
#include <sys/time.h>
#include <math.h>

class BetaInfluence: public InfluenceDistribution {
 private:
  double alpha_prior_, beta_prior_;
  double alpha_, beta_;
  double quartile_med_;
  double quartile_upper_;
  double quartile_stdev_;
  double original_mean_;
  std::default_random_engine gen_;

 public:
  BetaInfluence(double alpha, double beta, double orig)
      : alpha_prior_(alpha), beta_prior_(beta), alpha_(alpha), beta_(beta),
        original_mean_(orig), gen_(seed_ns()) {
    update_quartiles();
  }

  void update(unode_int hit, unode_int miss) {
    alpha_ += (double)hit;
    beta_ += (double)miss;
    hits_ += hit;
    misses_ += miss;
    update_quartiles();
  };

  void update_prior(double new_alpha, double new_beta) {
    alpha_prior_ = (new_alpha) > 0 ? new_alpha : 1.0;
    beta_prior_ = (new_beta) > 0 ? new_beta : 1.0;
    alpha_ = alpha_prior_ + (double)hits_;
    beta_ = beta_prior_ + (double)misses_;
    update_quartiles();
  }

  double mean() { return (double)alpha_ / (double)(alpha_ + beta_); }

  double sample(unsigned int interval) {
    if (interval == INFLUENCE_MED) {
      return quartile_med_;
    } else if (interval == INFLUENCE_UPPER) {
      return quartile_upper_;
    } else if(interval == INFLUENCE_UCB) {
      double val = quartile_med_ + sqrt(3.0 * log(round_) /
          (2.0 * (alpha_ + beta_)));
      return (val < 1) ? val : 1.0;
    } else if (interval == INFLUENCE_THOMPSON) {
      std::gamma_distribution<double> a(alpha_, 1.0);
      std::gamma_distribution<double> b(beta_, 1.0);
      double x = a(gen_);
      double y = b(gen_);
      return x / (x + y);
    } else { // Case where we shift the distributions by theta stdev (EG)
      double val = quartile_med_ + (interval - (double)THETA_OFFSET - 1.0)
          * quartile_stdev_;
      val = val < 1 ? val : 1.0;
      return val > 0 ? val : 0.0;
    }
    return quartile_med_;
  }

  double sq_error() {
    return (quartile_med_ - original_mean_) * (quartile_med_ - original_mean_);
  }

 private:
  void update_quartiles() {
    boost::math::beta_distribution<> dist(alpha_, beta_);
    quartile_med_ = alpha_ / (alpha_ + beta_);
    quartile_stdev_ = sqrt(alpha_ * beta_ / (alpha_ + beta_ + 1.0))
        / (alpha_ + beta_);
    quartile_upper_ = quantile(dist, 0.75);
  }
};

#endif /* defined(__oim__BetaInfluence__) */
