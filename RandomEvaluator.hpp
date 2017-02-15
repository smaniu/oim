/*
 Copyright (c) 2017 Paul Lagrée

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

#ifndef __oim__RandomEvaluator__
#define __oim__RandomEvaluator__

#include "common.hpp"
#include "Evaluator.hpp"

#include <sys/time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

class RandomEvaluator : public Evaluator {
 private:
  std::mt19937 gen_;
  std::uniform_int_distribution<unode_int> dst_;

 public:
  RandomEvaluator() : gen_(seed_ns()) {};

  std::unordered_set<unode_int> select(
      const Graph& graph, Sampler&,
      const std::unordered_set<unode_int>&, unsigned int k) {
    dst_ = std::uniform_int_distribution<unode_int>(
        0, graph.get_number_nodes() - 1);
    std::unordered_set<unode_int> seeds;
    while (seeds.size() < k) {
      unode_int seed = dst_(gen_);
      seeds.insert(seed);
    }
    return seeds;
  }
};

#endif /* defined(__oim__RandomEvaluator__) */
