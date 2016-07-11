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

#ifndef __oim__SpreadSampler__
#define __oim__SpreadSampler__

#include <queue>
#include <unordered_set>
#include <random>
#include <sys/time.h>
#include <math.h>

#include "common.h"
#include "Graph.h"
#include "Sampler.h"

using namespace std;

class SpreadSampler : public Sampler {
 private:
  boost::mt19937 gen_;
  boost::uniform_01<boost::mt19937> dist_;
  double stdev_;

 public:
  SpreadSampler(unsigned int type)
      : Sampler(type), gen_((int)time(0)), dist_(gen_) {};

  double sample(const Graph& graph,
                const std::unordered_set<unsigned long>& activated,
                const std::unordered_set<unsigned long>& seeds,
                unsigned long samples) {
    return perform_sample(graph, activated, seeds, samples, false);
  }

  double trial(const Graph& graph,
               const std::unordered_set<unsigned long>& activated,
               const std::unordered_set<unsigned long>& seeds,
               bool inv=false) {
    return perform_sample(graph, activated, seeds, 1, true, inv);
  }

  double get_stdev() { return stdev_; }

 private:
  double perform_sample(const Graph& graph,
                        const std::unordered_set<unsigned long>& activated,
                        const std::unordered_set<unsigned long>& seeds,
                        unsigned long samples, bool trial, bool inv=false) {
    trials_.clear();
    double spread = 0;
    stdev_ = 0;

    for (unsigned long sample = 1; sample <= samples; sample++) {
      double reached_round = 0;
      std::queue<unsigned long> queue;
      std::unordered_set<unsigned long> visited;
      for (unsigned long source : seeds) {
        queue.push(source);
        visited.insert(source);
      }
      while (queue.size() > 0) {
        unsigned long node_id = queue.front();
        sample_outgoing_edges(graph, node_id, queue, visited, trial, inv);
        queue.pop();
        if (activated.find(node_id) == activated.end())
          reached_round++;
      }
      double os = spread;
      spread += (reached_round - os) / (double)sample;
      stdev_ += (reached_round - os) * (reached_round - spread);
    }
    stdev_ = sqrt(stdev_ / (double)(samples - 1));
    return spread;
  }

  void sample_outgoing_edges(const Graph& graph, const unsigned long node,
                             std::queue<unsigned long>& queue,
                             std::unordered_set<unsigned long>& visited,
                             bool trial, bool inv=false) {
    if (graph.has_neighbours(node, inv)) {
      for (auto edge : graph.get_neighbours(node, inv)) {
        if (visited.find(edge.target) == visited.end()) {
          double dice_dst = edge.dist->sample(quantile_);
          unsigned int act = 0;
          double dice = dist_();
          if (dice < dice_dst) {
            visited.insert(edge.target);
            queue.push(edge.target);
            act = 1;
          }
          if (trial) {
            trial_type tt;
            tt.source = node;
            tt.target = edge.target;
            tt.trial = act;
            trials_.push_back(tt);
          }
        }
      }
    }
  }
};

#endif /* defined(__oim__SpreadSampler__) */
