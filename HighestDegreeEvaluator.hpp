/*
 Copyright (c) 2015 Siyu Lei, Silviu Maniu, Luyi Mo

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

#ifndef __oim__HighestDegreeEvaluator__
#define __oim__HighestDegreeEvaluator__

#include "common.hpp"
#include "Evaluator.hpp"

#include <boost/heap/fibonacci_heap.hpp>

class HighestDegreeEvaluator : public Evaluator {
 private:
  std::unordered_set<unode_int> seed_sets_;

  std::unordered_set<unode_int> select(
        const Graph& graph, Sampler&,
        const std::unordered_set<unode_int>&, unsigned int k) {
    std::unordered_set<unode_int> set;
    boost::heap::fibonacci_heap<NodeType> queue;
    for (unode_int node : graph.get_nodes()) {
      NodeType nstruct;
      nstruct.id = node;
      nstruct.deg = 0;
      if(graph.has_neighbours(node))
        nstruct.deg = graph.get_neighbours(node).size();
      queue.push(nstruct);
    }
    while (set.size() < k && !queue.empty()) {
      NodeType nstruct = queue.top();
      if (seed_sets_.find(nstruct.id) == seed_sets_.end()) {
        // Guarantee no duplicate nodes in the seed set for all trials
        set.insert(nstruct.id);
        seed_sets_.insert(nstruct.id);
      }
      queue.pop();
    }
    return set;
  }
};

#endif /* defined(__oim__HighestDegreeEvaluator__) */
