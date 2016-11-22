/*
 Copyright (c) 2016 Paul Lagrée (Université Paris Sud)

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

#ifndef __oim__SSAEvaluator__
#define __oim__SSAEvaluator__

#include "common.hpp"
#include "Graph.hpp"
#include "Evaluator.hpp"
#include "Sampler.hpp"

#include <math.h>

using namespace std;

/**
  Implementation of SSA algorithm introduced in `Stop-and-Stare: Optimal
  Sampling Algorithms for Viral Marketing in Billion-scale Networks` by H. T.
  Nguyen et al., SIGMOD 2017.
*/
class SSAEvaluator : public Evaluator {
 private:
  std::unordered_set<unsigned long> seed_set_;  // Set of k selected nodes
  vector<shared_ptr<vector<unsigned long>>> rr_samples_;  // List of RR samples
  vector<vector<unsigned int>> hyper_graph_;    // RR samples where appear each node
  std::random_device rd_;
  std::mt19937 gen_;
  double epsilon_;
  double delta_;
  std::uniform_int_distribution<unsigned long> dst_;

 public:
  SSAEvaluator(double epsilon)
      : gen_(rd_()), epsilon_(epsilon) {};

  /**
    Selects `k` nodes from the graph using the sampler given in parameter.
    -> `samples` isn't used and should be removed.
    -> `activated` is used to avoid sampling already activated nodes, I think I
       made a mistake and it should be used to count only unactivated nodes in
       the diffusion process. TODO
  */
  std::unordered_set<unsigned long> select(
        const Graph& graph, Sampler& sampler,
        const std::unordered_set<unsigned long>& activated,
        unsigned int k, unsigned long samples) {
    hyper_graph_.clear();
    rr_samples_.clear();
    delta_ = 1. / graph.get_number_nodes();
    dst_ = uniform_int_distribution<unsigned long>(0, graph.get_number_nodes() - 1);

    for (unsigned int i = 0; i < graph.get_number_nodes(); i++) {
      hyper_graph_.push_back(vector<unsigned int>());
    }
    double epsilon_1 = epsilon_ / 6, epsilon_2 = epsilon_ / 2;
    double epsilon_3 = (epsilon_ - epsilon_1 - epsilon_2 -
        epsilon_1 * epsilon_2) / (1 - 1 / exp(1));
    unsigned long lambda_1 = (unsigned long)((1 + epsilon_1) * (1 + epsilon_2) *
        (2 + 2 / 3 * epsilon_3) * log(3 / delta_) / (epsilon_3 * epsilon_3));
    unsigned long n_samples = 2 * lambda_1;
    // Algorithm here
    while(true) {
      unsigned long n_new_samples = n_samples - rr_samples_.size();
      buildSamples(n_new_samples, graph, sampler, activated);
      n_samples *= 2;
      double biased_estimator = buildSeedSet(graph, k);
      if (biased_estimator * rr_samples_.size() / graph.get_number_nodes() >= lambda_1) {
        unsigned int T_max = (unsigned int)(2 * rr_samples_.size() *
              (1 + epsilon_2) / (1 - epsilon_2) * epsilon_3 * epsilon_3 /
              (/*k * */epsilon_2 * epsilon_2));   // k dropped like in the paper
        double unbiased_estimator = estimateInf(graph, sampler, epsilon_2,
                                                k, T_max);
        if (biased_estimator <= (1 + epsilon_1) * unbiased_estimator) {
          return seed_set_;
        }
      }
    }
    return seed_set_;
  }

 private:
  /**
    Influence estimation of a given seed set seed_set_
  */
  double estimateInf(const Graph &graph, Sampler &sampler, double epsilon_2,
                     unsigned int k, unsigned int T_max) {  // delta_2 = delta_3
    vector<unsigned long> nodes_activated(graph.get_number_nodes(), 0);
    vector<bool> bool_activated(graph.get_number_nodes(), false);
    unsigned long n = graph.get_number_nodes();
    // This is copied from the original code, not clear yet
    double f = (log(2 / delta_) + lgamma(n + 1) - lgamma(k + 1) -
          lgamma(n - k + 1)) / (k * log(2 / delta_));
    double lambda_2 = 1 + (2 + 2 * epsilon_2 / 3) * (1 + epsilon_2) *
          (log(3 / delta_) + log(f)) / (epsilon_2 * epsilon_2);
    double cov = 0;
    for (unsigned int i = 0; i < T_max; i++) {
      unsigned long source = dst_(gen_);
      // We sample a new RR set
      shared_ptr<vector<unsigned long>> rr_sample = sampler.perform_unique_sample(
          graph, nodes_activated, bool_activated, source, true);  // TODO can be improved because if we found a node from seed_set, we can stop diffusion
      for (unsigned long sampled_node : *rr_sample) {
        if (seed_set_.find(sampled_node) != seed_set_.end()) {
          cov += 1;
          break;
        }
      }
      if (cov >= lambda_2) {
        return (double)n * cov / (double)i;
      }
    }
    return -1;
  }

  /**
    Samples n_samples new RR sets and add them to set of RR samples rr_samples_
    TODO Pretty sure I don't use `activated` properly.
  */
  void buildSamples(unsigned long n_samples, const Graph& graph, Sampler& sampler,
                    const unordered_set<unsigned long>& activated) {
    vector<unsigned long> nodes_activated(graph.get_number_nodes(), 0);
    vector<bool> bool_activated(graph.get_number_nodes(), false);
    unsigned int nb_rr_samples = rr_samples_.size();
    for (unsigned int i = 0; i < n_samples; i++) {
      unsigned long source = dst_(gen_);
      while (activated.find(source) != activated.end()) { // While the randomly sampled node was already activated
        source = dst_(gen_);
      }
      shared_ptr<vector<unsigned long>> rr_sample = sampler.perform_unique_sample(
            graph, nodes_activated, bool_activated, source, true);
      rr_samples_.push_back(rr_sample);
      for (unsigned long node : *rr_sample) {
        hyper_graph_[node].push_back(nb_rr_samples);
      }
      nb_rr_samples += 1;
    }
  }

  /**
  * Greedy algorithm computing the maximum coverage
  */
  double buildSeedSet(const Graph &graph, unsigned int k) {
    seed_set_.clear();
    vector<unsigned int> degree(graph.get_number_nodes(), 0);  // Number of covered sets
    vector<bool> visited_samples(rr_samples_.size(), false);
    for (unsigned int i = 0; i < hyper_graph_.size(); i++) {
      degree[i] = hyper_graph_[i].size();
    }
    for (unsigned int i = 0; i < k; i++) {
      unsigned long max_node = max_element(degree.begin(), degree.end()) - degree.begin();
      seed_set_.insert(max_node);
      for (unsigned int rr_sample_id : hyper_graph_[max_node]) {
        if (!visited_samples[rr_sample_id]) {
          visited_samples[rr_sample_id] = true;
          for (unsigned long node : *rr_samples_[rr_sample_id]) {
            degree[node]--;
          }
        }
      }
    }
    double cov = 0;
    for (bool visited : visited_samples) {
      if (visited)
        cov += 1;
    }
    return cov * graph.get_number_nodes() / rr_samples_.size();
  }
};

#endif /* defined(__oim__SSAEvaluator__) */
