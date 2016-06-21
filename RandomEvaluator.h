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

#include "common.h"
#include "Evaluator.h"

#include <sys/time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

class RandomEvaluator : public Evaluator {
public:
  std::unordered_set<unsigned long> select(
      const Graph& graph, Sampler& sampler,
      const std::unordered_set<unsigned long>& activated, unsigned int k,
      unsigned long samples) {
    boost::mt19937 gen((int)time(0));
    std::vector<unsigned long> reservoir;
    int index = 0;
    for (unsigned long node : graph.get_nodes()) {
      if (activated.find(node) == activated.end()) {
        if (index < k) {
          reservoir.push_back(node);
        }
        else{
          boost::random::uniform_int_distribution<> dist(0, index);
          int dice = dist(gen);
          if(dice < k) reservoir[dice] = node;
        }
        index++;
      }
    }
    std::unordered_set<unsigned long> set;
    for (unsigned long node : reservoir) set.insert(node);
    return set;
  }
};

#endif /* defined(__oim__RandomEvaluator__) */
