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

#ifndef __oim__Sampler__
#define __oim__Sampler__

#include "common.hpp"
#include "Graph.hpp"


/**
  Pseudorandom number generator from PMC implementation (`Fast and Accurate
  Influence Maximization on Large Networks with Pruned Monte-Carlo Simulations`
  by Naoto Ohsaka et al., AAAI 2014.)
*/
class Xorshift {
 public:
	Xorshift(unsigned int seed) {
		x_ = _(seed, 0);
		y_ = _(x_, 1);
		z_ = _(y_, 2);
		w_ = _(z_, 3);
	}

	int _(int s, int i) {
		return 1812433253 * (s ^ (s >> 30)) + i + 1;
	}

	inline int gen_int() {
		unsigned int t = x_ ^ (x_ << 11);
		x_ = y_;
		y_ = z_;
		z_ = w_;
		return w_ = w_ ^ (w_ >> 19) ^ t ^ (t >> 8);
	}

	inline int gen_int(int n) {
		return (int) (n * gen_double());
	}

	inline double gen_double() {
		unsigned int a = ((unsigned int) gen_int()) >> 5, b =
				((unsigned int) gen_int()) >> 6;
		return (a * 67108864.0 + b) * (1.0 / (1LL << 53));
	}

 private:
	unsigned int x_, y_, z_, w_;
};

/**
  Abstract class giving methods that a sampler needs to provide to Evaluators.
  Two implementations are given so far, PathSampler -- which actually does not
  sample but uses paths on graph to estimate node values -- and SpreadSampler.
  SpreadSampler is now fast enough to be used for everything. PathSampler is now
  depreciated.
*/
class Sampler {
 protected:
  unsigned int type_;
  std::vector<TrialType> trials_;
  int model_;  // 0 for linear threshold, 1 for cascade model

 public:
  Sampler(unsigned int type, int model) : type_(type), model_(model) {}

  /**
    Method to estimate the standard deviation of TODO
  */
  virtual double sample(const Graph& graph,
                        const std::unordered_set<unode_int>& activated,
                        const std::unordered_set<unode_int>& seeds,
                        unode_int samples) = 0;

  virtual double trial(const Graph& graph,
                       const std::unordered_set<unode_int>& activated,
                       const std::unordered_set<unode_int>& seeds,
                       bool inv=false) = 0;

  virtual std::shared_ptr<std::vector<unode_int>> perform_unique_sample(
      const Graph& graph, std::vector<unode_int>& nodes_activated,
      std::vector<bool>& bool_activated, const unode_int source,
      const std::unordered_set<unode_int>& activated, bool inv=false) = 0;

  virtual std::unordered_set<unode_int> perform_diffusion(
      const Graph& graph, const std::unordered_set<unode_int>& seeds) = 0;

  std::vector<TrialType>& get_trials() { return trials_; }

  unsigned int get_type() { return type_; }

};

#endif /* defined(__oim__Sampler__) */
