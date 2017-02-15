/*
 Copyright (c) 2015-2017 Paul Lagrée, Siyu Lei, Silviu Maniu, Luyi Mo

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
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <sys/time.h>
#include <math.h>

#include "common.hpp"
#include "Graph.hpp"
#include "Sampler.hpp"

using namespace std;


/**
  LT or Independent Cascade Model Sampler of the graph (does *real* samples).
*/
class SpreadSampler : public Sampler {
 private:
  boost::mt19937 gen_;
  Xorshift dist_;
  double stdev_;

 public:
  SpreadSampler(unsigned int type, int model)
      : Sampler(type, model), gen_(seed_ns()), dist_(Xorshift(seed_ns())) {};

  /**
    Samples `n_samples` from seeds.
  */
  double sample(const Graph& graph,
                const std::unordered_set<unode_int>& activated,
                const std::unordered_set<unode_int>& seeds,
                unode_int n_samples) {
    return perform_sample(graph, activated, seeds, n_samples, false);
  }

  /**
    Performs the *real* sample, that is, diffuse the influence from selected
    seeds. Compared to the sample method, it saves sampled edges in `trials_`.
  */
  double trial(const Graph& graph,
               const std::unordered_set<unode_int>& activated,
               const std::unordered_set<unode_int>& seeds,
               bool inv=false) {
    return perform_sample(graph, activated, seeds, 1, true, inv);
  }

  /**
    Performs a unique sample from `source`. This method is used for sampling
    RR sets in SSAEvaluator. It implements both LT and IC models.

    @param nodes_activated Reserved vector of size the number of nodes. It is
        used as queue while performing the sample
    @param bool_activated When a node is activated, mark its corresponding index
    @return Vector containing activated nodes in this sample.
  */
  std::shared_ptr<vector<unode_int>> perform_unique_sample(
        const Graph& graph, std::vector<unode_int>& nodes_activated,
        std::vector<bool>& bool_activated, unode_int source,
        const std::unordered_set<unode_int>&, bool inv=false) {
    unode_int cur = source;
    unode_int num_marked = 1, cur_pos = 0;
    bool_activated[cur] = true;
    nodes_activated[0] = cur;
    while (cur_pos < num_marked) {
      cur = nodes_activated[cur_pos];
      cur_pos++;
      if (model_ == 0) { // Linear threshold model
        int index = graph.sample_living_edge(cur, gen_);
        if (index == -1)  // Unconnected node or sample with weights summing to less than 1
          continue;
        unode_int living_node = graph.get_neighbours(cur, true)[index].target;
        if (!bool_activated[living_node]) {
          bool_activated[living_node] = true;
          nodes_activated[num_marked] = living_node;
          num_marked++;
        }
      } else if (model_ == 1) { // Independent Cascade model
        if (graph.has_neighbours(cur, inv)) {
          for (auto& neighbour : graph.get_neighbours(cur, inv)) {
            if (dist_.gen_double() < neighbour.dist->sample(type_)) {
              if (!bool_activated[neighbour.target]) {
                bool_activated[neighbour.target] = true;
                nodes_activated[num_marked] = neighbour.target;
                num_marked++;
              }
            }
          }
        }
      }
    }
    std::vector<unode_int> result;
    for (unode_int i = 0; i < num_marked; i++) {
      result.push_back(nodes_activated[i]);
    }
    std::shared_ptr<vector<unode_int>> rr_sample =
        std::make_shared<vector<unode_int>>(result);
    for (unsigned int i = 0; i < num_marked; i++) {
      bool_activated[nodes_activated[i]] = false;
    }
    return rr_sample;
  }

  /**
    Performs the real diffusion from selected seeds.
    Returns the set of activated users.
  */
  std::unordered_set<unode_int> perform_diffusion(const Graph& graph,
        const std::unordered_set<unode_int>& seeds) {
    std::unordered_set<unode_int> visited;
    std::queue<unode_int> queue;
    if (model_ == 0) {  // LT model
      std::unordered_map<unode_int, std::vector<unode_int>> live_edges;
      for (unode_int u = 0; u < graph.get_number_nodes(); u++) {
        int index = graph.sample_living_edge(u, gen_);
        if (index == -1)  // Unconnected node or sample with weights summing to less than 1
          continue;
        unode_int living_node = graph.get_neighbours(u, true)[index].target;
        if (live_edges.find(living_node) == live_edges.end())
          live_edges[living_node] = std::vector<unode_int>();
        live_edges[living_node].push_back(u);
      }
      for (auto source : seeds) {
        queue.push(source);
        visited.insert(source);
      }
      while (queue.size() > 0) {
        auto node_id = queue.front();
        visited.insert(node_id);
        if (live_edges.find(node_id) != live_edges.end()) {
          for (auto& neighbour : live_edges[node_id]) {
            if (visited.find(neighbour) == visited.end())
              queue.push(neighbour);
          }
        }
        queue.pop();
      }
    } else if (model_ == 1) { // IC model
      for (auto source : seeds) {
        queue.push(source);
        visited.insert(source);
      }
      while (queue.size() > 0) {
        auto node_id = queue.front();
        sample_outgoing_edges(graph, node_id, queue, visited, false, false);
        queue.pop();
      }
    }
    return visited; // Potentially a copy, depending on compiler's optimzations
  }

 private:
  /**
    [depreciated] Performs `n_samples` samples starting from `seeds`.
  */
  double perform_sample(const Graph& graph,
                        const std::unordered_set<unode_int>& activated,
                        const std::unordered_set<unode_int>& seeds,
                        unode_int n_samples, bool trial, bool inv=false) {
    trials_.clear();
    double spread = 0;
    double outspread = 0;
    stdev_ = 0;
    for (unode_int sample = 1; sample <= n_samples; sample++) {
      double reached_round = 0; // Number of nodes activated
      std::queue<unode_int> queue;
      std::unordered_set<unode_int> visited;
      for (unode_int source : seeds) {
        queue.push(source);
        visited.insert(source);
      }
      while (queue.size() > 0) {
        unode_int node_id = queue.front();
        sample_outgoing_edges(graph, node_id, queue, visited, trial, inv);
        queue.pop();
        if (activated.find(node_id) == activated.end())
          reached_round++;
      }
      double os = spread;
      spread += (reached_round - os) / (double)sample;
      outspread += reached_round;
      stdev_ += (reached_round - os) * (reached_round - spread);
    }
    stdev_ = sqrt(stdev_ / (double)(n_samples - 1));
    return outspread / n_samples;
  }

  /**
    Samples outgoing edges from `node`. New activated nodes are added to
    `visited`. If `trial` is true, we add sampled edges in the vector `trials_`.
    This method is implemented for both linear threshold and independent cascade
    models.
  */
  void sample_outgoing_edges(const Graph& graph, unode_int node,
                             std::queue<unode_int>& queue,
                             std::unordered_set<unode_int>& visited,
                             bool trial, bool inv=false) {
    if (model_ == 0) { // Linear threshold model, this method isn't implemented for LT
      std::cerr << "Error: this part is only run by IC model." << std::endl;
      exit(1);
    } else if (model_ == 1) { // Independent Cascade model
      if (graph.has_neighbours(node, inv)) {
        for (auto edge : graph.get_neighbours(node, inv)) {
          if (visited.find(edge.target) == visited.end()) {
            double dice_dst = edge.dist->sample(type_);
            unsigned int act = 0;
            double dice = dist_.gen_double();
            if (dice < dice_dst) {
              visited.insert(edge.target);
              queue.push(edge.target);
              act = 1;
            }
            if (trial) {  // If trial, we want to save the generated RR set sample
              TrialType tt;
              tt.source = node;
              tt.target = edge.target;
              tt.trial = act;
              trials_.push_back(tt);
            }
          }
        }
      }
    }
  }
};

#endif /* defined(__oim__SpreadSampler__) */
