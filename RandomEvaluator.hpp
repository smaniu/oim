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

#ifndef __oim__RandomEvaluator__
#define __oim__RandomEvaluator__

#include "common.hpp"
#include "Evaluator.hpp"

#include <sys/time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

class RandomEvaluator : public Evaluator {
 public:
  std::unordered_set<unode_int> select(
      const Graph& graph, Sampler&,
      const std::unordered_set<unode_int>& activated, unsigned int k) {
    boost::mt19937 gen((int)time(0));
    std::vector<unode_int> reservoir;
    unsigned int index = 0;
    for (unode_int node : graph.get_nodes()) {
      if (activated.find(node) == activated.end()) {
        if (index < k) {
          reservoir.push_back(node);
        } else {
          boost::random::uniform_int_distribution<> dist(0, index);
          unsigned int dice = dist(gen);
          if (dice < k) reservoir[dice] = node;
        }
        index++;
      }
    }
    std::unordered_set<unode_int> set;
    for (unode_int node : reservoir) set.insert(node);
    return set;
  }
};

#endif /* defined(__oim__RandomEvaluator__) */
