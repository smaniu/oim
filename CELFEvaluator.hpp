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

#ifndef __oim__CELFEvaluator__
#define __oim__CELFEvaluator__

#include <boost/heap/fibonacci_heap.hpp>

#include "common.hpp"
#include "Evaluator.hpp"

class CELFEvaluator : public Evaluator {
 private:
  struct celf_node_type {
    unsigned long id;
    double spr;
    bool operator<(const celf_node_type &a) const {
      return (spr < a.spr) ? true : ((spr > a.spr) ? false : id > a.id);
    }
  };
  unsigned int samples_;

 public:
  CELFEvaluator(unsigned int samples) : samples_(samples) {}

  std::unordered_set<unsigned long> select(
      const Graph& graph, Sampler& sampler,
      const std::unordered_set<unsigned long>& activated, unsigned int k) {
    boost::heap::fibonacci_heap<celf_node_type> queue;
    std::unordered_map<unsigned long,
      boost::heap::fibonacci_heap<celf_node_type>::handle_type> queue_nodes;
    std::unordered_set<unsigned long> set;

    //initial loop
    for (unsigned long node : graph.get_nodes()) {
      celf_node_type u;
      u.id = node;
      std::unordered_set<unsigned long> seeds;
      seeds.insert(node);
      u.spr = sampler.sample(graph, activated, seeds, samples_);
      queue_nodes[node] = queue.push(u);
    }

    //main loop
    set.insert(queue.top().id);
    queue.pop();
    while ((set.size() < k) && (queue.size() > 0)) {
      bool found = false;
      while (!found) {
        celf_node_type u = queue.top();
        queue.pop();
        std::unordered_set<unsigned long> seeds;
        for (unsigned long node : set) seeds.insert(node);
        seeds.insert(u.id);
        double prev_val = u.spr;
        u.spr = sampler.sample(graph, activated, seeds, samples_) - prev_val;
        if (u.spr >= queue.top().spr) {
          set.insert(u.id);
          found = true;
        } else {
          queue_nodes[u.id] = queue.push(u);
        }
      }
    }
    return set;
  }
};

#endif /* defined(__oim__CELFEvaluator__) */
