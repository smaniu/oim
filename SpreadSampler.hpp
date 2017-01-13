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
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <sys/time.h>
#include <math.h>

#include "common.hpp"
#include "Graph.hpp"
#include "Sampler.hpp"

using namespace std;

/**
  LT or Independent Cascade Model Sampler of the graph (does a *real* sample).
*/
class SpreadSampler : public Sampler {
 private:
  boost::mt19937 gen_;
  boost::uniform_01<boost::mt19937> dist_;
  double stdev_;

 public:
  SpreadSampler(unsigned int type, int model)
      : Sampler(type, model), gen_(seed_ns()), dist_(gen_) {};

  /**
    Samples `n_samples` from seeds.
  */
  double sample(const Graph& graph,
                const std::unordered_set<unsigned long>& activated,
                const std::unordered_set<unsigned long>& seeds,
                unsigned long n_samples) {
    return perform_sample(graph, activated, seeds, n_samples, false);
  }

  /**
    Performs the *real* sample, that is, diffuse the influence from selected
    seeds. Compared to the sample method, it saves sampled edges in `trials_`.
  */
  double trial(const Graph& graph,
               const std::unordered_set<unsigned long>& activated,
               const std::unordered_set<unsigned long>& seeds,
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
  std::shared_ptr<vector<unsigned long>> perform_unique_sample(
        const Graph& graph, std::vector<unsigned long>& nodes_activated,
        std::vector<bool>& bool_activated, unsigned long source,
        const std::unordered_set<unsigned long> activated, bool inv=false) {
    unsigned long cur = source;
    unsigned long num_marked = 1, cur_pos = 0;
    bool_activated[cur] = true;
    nodes_activated[0] = cur;
    while (cur_pos < num_marked) {
      cur = nodes_activated[cur_pos];
      cur_pos++;
      if (model_ == 0) { // Linear threshold model
        int index = graph.sample_living_edge(cur, gen_);
        if (index == -1)  // Unconnected node or sample with weights summing to less than 1
          continue;
        unsigned long living_node = graph.get_neighbours(cur, true)[index].target;
        if (!bool_activated[living_node]) {
          bool_activated[living_node] = true;
          nodes_activated[num_marked] = living_node;
          num_marked++;
        }
      } else if (model_ == 1) { // Independent Cascade model
        if (graph.has_neighbours(cur, inv)) {
          for (auto& neighbour : graph.get_neighbours(cur, inv)) {
            if (dist_() < neighbour.dist->sample(type_)) {
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
    std::vector<unsigned long> result;
    for (unsigned long i = 0; i < num_marked; i++) {
      if (activated.find(nodes_activated[i]) == activated.end())
        result.push_back(nodes_activated[i]);
    }
    std::shared_ptr<vector<unsigned long>> rr_sample =
        std::make_shared<vector<unsigned long>>(result);
        // std::make_shared<vector<unsigned long>>(vector<unsigned long>(
        // nodes_activated.begin(), nodes_activated.begin() + num_marked));
    for (unsigned int i = 0; i < num_marked; i++) {
      bool_activated[nodes_activated[i]] = false;
    }
    return rr_sample;
  }

  /**
    Performs the real diffusion from selected seeds.
    Returns the set of activated users.
  */
  std::unordered_set<unsigned long> perform_diffusion(const Graph& graph,
        const std::unordered_set<unsigned long>& seeds) {
    std::unordered_set<unsigned long> visited;
    std::queue<unsigned long> queue;
    if (model_ == 0) {
      std::unordered_map<unsigned long, std::vector<unsigned long>> live_edges;
      for (unsigned long u = 0; u < graph.get_number_nodes(); u++) {
        int index = graph.sample_living_edge(u, gen_);
        if (index == -1)  // Unconnected node or sample with weights summing to less than 1
          continue;
        live_edges[graph.get_neighbours(u, true)[index].target].push_back(u);
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
    } else if (model_ == 1) {
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

  //double get_stdev() { return stdev_; }

 private:
  /**
    Performs `n_samples` samples starting from `seeds`. // TODO depreciated, remove its use everywhere
  */
  double perform_sample(const Graph& graph,
                        const std::unordered_set<unsigned long>& activated,
                        const std::unordered_set<unsigned long>& seeds,
                        unsigned long n_samples, bool trial, bool inv=false) {
    trials_.clear();
    double spread = 0;
    double outspread = 0;
    stdev_ = 0;
    for (unsigned long sample = 1; sample <= n_samples; sample++) {
      double reached_round = 0; // Number of nodes activated
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
      //exit(1);
      double os = spread;
      spread += (reached_round - os) / (double)sample;  // TODO Don't understand this line ERROR
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
  void sample_outgoing_edges(const Graph& graph, unsigned long node,
                             std::queue<unsigned long>& queue,
                             std::unordered_set<unsigned long>& visited,
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
            double dice = dist_();
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
