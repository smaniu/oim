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

#ifndef __oim__PathSampler__
#define __oim__PathSampler__

#include <queue>
#include <iostream>
#include <unordered_set>
#include <random>
#include <sys/time.h>
#include <boost/heap/fibonacci_heap.hpp>

#include "common.h"
#include "Graph.h"
#include "Sampler.h"

class PathSampler : public Sampler {
 private:
  struct node_type {
    unsigned long id;
    double prob;
    bool operator<(const node_type &a) const {
      return (prob < a.prob) ? true : ((prob > a.prob) ? false : id > a.id);
    }
  };

 public:
  PathSampler(unsigned int type)
      : Sampler(type) {};

  double sample(const Graph& graph,
                const std::unordered_set<unsigned long>& activated,
                const std::unordered_set<unsigned long>& seeds,
                unsigned long samples) {
    return perform_sample(graph, activated, seeds, samples, false);
  }

  double trial(const Graph& graph,
               const std::unordered_set<unsigned long>& activated,
               const std::unordered_set<unsigned long>& seeds, bool inv) {
    return perform_sample(graph, activated, seeds, 1, true, inv);
  }

private:
  double perform_sample(const Graph& graph,
                        const std::unordered_set<unsigned long>& activated,
                        const std::unordered_set<unsigned long>& seeds,
                        unsigned long samples, bool trial, bool inv=false) {
    trials_.clear();
    boost::heap::fibonacci_heap<node_type> queue;
    std::unordered_map<unsigned long,
        boost::heap::fibonacci_heap<node_type>::handle_type> queue_nodes;
    std::unordered_set<unsigned long> visited;
    double spread = 0.0;
    for (unsigned long seed : seeds) {
      node_type node;
      node.id = seed;
      node.prob = 1.0;
      queue_nodes[seed] = queue.push(node);
      if (trial) {
        trial_type tt;
        tt.source = node.id;
        tt.target = node.id;
        tt.trial = 1;
        trials_.push_back(tt);
      }
    }
    while (queue.size() > 0) {
      node_type node = queue.top();
      queue.pop();
      if(trial) {
        trial_type tt;
        tt.source = node.id;
        tt.target = node.id;
        tt.trial = 1;
        trials_.push_back(tt);
      }
      if (activated.find(node.id) == activated.end()) spread += node.prob;
      if (node.prob < 0.001) break;
      visited.insert(node.id);
      sample_outgoing_edges(graph, node.id, queue, visited, queue_nodes, inv);
    }
    return spread;
  }

  void sample_outgoing_edges(
      const Graph& graph, unsigned long node,
      boost::heap::fibonacci_heap<node_type>& queue,
      std::unordered_set<unsigned long>& visited,
      std::unordered_map<unsigned long,
          boost::heap::fibonacci_heap<node_type>::handle_type>& queue_nodes,
      bool inv=false) {

    if (graph.has_neighbours(node, inv)) {
      for (auto edge : graph.get_neighbours(node, inv)) {
        if (visited.find(edge.target) == visited.end()) {
          double dst_prob = edge.dist->sample(quantile_);
          relax(node, edge.target, dst_prob, queue, queue_nodes);
        }
      }
    }
  }

  void relax(
      unsigned long src, unsigned long tgt, double dst,
      boost::heap::fibonacci_heap<node_type>& queue,
      std::unordered_map<unsigned long,
          boost::heap::fibonacci_heap<node_type>::handle_type>& queue_nodes) {

    double new_prob = (*queue_nodes[src]).prob * dst;
    if (queue_nodes.find(tgt) == queue_nodes.end()) {
      node_type node;
      node.id = tgt;
      node.prob = new_prob;
      queue_nodes[tgt] = queue.push(node);
    } else {
      double prev_prob = (*queue_nodes[tgt]).prob;
      if (new_prob > prev_prob) {
        node_type node;
        node.id = tgt;
        node.prob = new_prob;
        auto handle = queue_nodes[tgt];
        queue.update(handle, node);
      }
    }
  }
};

#endif /* defined(__oim__PathSampler__) */
