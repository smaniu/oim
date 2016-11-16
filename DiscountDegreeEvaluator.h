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

#ifndef __oim__DiscountDegreeEvaluator__
#define __oim__DiscountDegreeEvaluator__

#include "common.h"
#include "Evaluator.h"

#include <boost/heap/fibonacci_heap.hpp>

class DiscountDegreeEvaluator : public Evaluator {
 public:
  std::unordered_set<unsigned long> select(
        const Graph& graph, Sampler& sampler,
        const std::unordered_set<unsigned long>& activated,
        unsigned int k, unsigned long samples) {
    std::unordered_set<unsigned long> set;
    unsigned int type = sampler.get_type();
    boost::heap::fibonacci_heap<NodeType> queue;
    std::unordered_map<unsigned long,
        boost::heap::fibonacci_heap<NodeType>::handle_type> queue_nodes;
    for (unsigned long node : graph.get_nodes()) {
      NodeType nstruct;
        nstruct.id = node;
        nstruct.deg = activated.find(node) == activated.end() ? 1.0f : 0.0f;
        if (graph.has_neighbours(node)) {
          for (auto edge : graph.get_neighbours(node)) {
            if(activated.find(edge.target) == activated.end()) {
              nstruct.deg += edge.dist->sample(type);
            }
          }
        }
        queue_nodes[node] = queue.push(nstruct);
    }
    while (set.size() < k && (!queue.empty())) {
      NodeType nstruct = queue.top();
      set.insert(nstruct.id);
      for (auto edge : graph.get_neighbours(nstruct.id)) {
        if (activated.find(edge.target) == activated.end() &&
            set.find(edge.target) == set.end()) {
          NodeType newnstruct = *queue_nodes[edge.target];
          newnstruct.id = edge.target;
          newnstruct.deg = newnstruct.deg*(1.0f - edge.dist->sample(type));
          queue.update(queue_nodes[edge.target], newnstruct);
        }
      }
      queue.pop();
    }
    queue_nodes.clear();
    return set;
  }
};

#endif /* defined(__oim__DiscountDegreeEvaluator__) */
