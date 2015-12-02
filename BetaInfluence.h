/*
 Copyright (c) 2015 Siyu Lei, Silviu Maniu, Luyi Mo (University of Hong Kong)
 
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

#include "common.h"
#include "InfluenceDistribution.h"

#include <boost/math/distributions.hpp>
#include <random>
#include <sys/time.h>
#include <math.h>

class BetaInfluence: public InfluenceDistribution{
private:
  double alpha_prior, beta_prior;
  double alpha,beta;
  double quartile_med;
  double quartile_upper;
  double quartile_stdev;
  double original_mean;
  std::default_random_engine gen;
  
public:
  BetaInfluence(double alpha, double beta, double orig){
    this->alpha = alpha;
    this->beta = beta;
    alpha_prior = alpha;
    beta_prior = beta;
    original_mean = orig;
    update_quartiles();
  };
  
  void update(unsigned long hit, unsigned long miss){
    alpha += (double) hit;
    beta += (double) miss;
    hits += hit;
    misses += miss;
    update_quartiles();
  };
  
  void update_prior(double new_alpha, double new_beta){
    alpha_prior = new_alpha>0?new_alpha:1.0;
    beta_prior = new_beta>0?new_beta:1.0;
    alpha = alpha_prior + (double) hits;
    beta = beta_prior + (double) misses;
    update_quartiles();
  }
  
  double mean() {return (double)alpha/(double)(alpha+beta);}
  
  double sample(unsigned int interval){
    if(interval==INFLUENCE_MED) return quartile_med;
    else if(interval==INFLUENCE_UPPER) return quartile_upper;
    else if(interval==INFLUENCE_UCB){
      double val = quartile_med+sqrt(3.0*log(round)/(2.0*(alpha+beta)));
      return val<1?val:1.0;
    }
    else if(interval==INFLUENCE_THOMPSON){
      gen.seed(time(0));
      std::gamma_distribution<double> a(alpha,1.0);
      std::gamma_distribution<double> b(beta,1.0);
      double x = a(gen);
      double y = b(gen);
      return x/(x+y);
    }
    return quartile_med;
  }
  
  double sq_error(){
    return (quartile_med-original_mean)*(quartile_med-original_mean);
  }
  
private:
  void update_quartiles(){
    boost::math::beta_distribution<> dist(alpha,beta);
    quartile_med = alpha/(alpha+beta);
    quartile_stdev = sqrt(alpha*beta/(alpha+beta+1.0))/(alpha+beta);
    quartile_upper = quantile(dist, 0.75);
  }
};

#endif /* defined(__oim__BetaInfluence__) */
