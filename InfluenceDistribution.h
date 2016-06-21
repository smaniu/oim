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

#ifndef __oim__InfluenceDistribution__
#define __oim__InfluenceDistribution__

#define INFLUENCE_MED  0
#define INFLUENCE_UPPER  1
#define INFLUENCE_ADAPTIVE 2
#define INFLUENCE_UCB 3
#define INFLUENCE_THOMPSON 4

class InfluenceDistribution {
 protected:
  unsigned long hits = 0;
  unsigned long misses = 0;
  double round = 0;

 public:
  virtual void update(unsigned long, unsigned long) {};

  virtual void update_prior(double, double) {};

  virtual double mean() = 0;

  virtual double sample(unsigned int) = 0;

  virtual double sq_error() {return 0.0;}

  void set_round(double new_round) { round += new_round; }

  unsigned long get_hits() { return hits; }

  unsigned long get_misses() { return misses; }
};

#endif /* defined(__oim__InfluenceDistribution__) */
