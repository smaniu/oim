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

#ifndef __oim__SSAEvaluator__
#define __oim__SSAEvaluator__

#include "common.h"
#include "Graph.h"
#include "Evaluator.h"
#include "Sampler.h"
#include "SpreadSampler.h"
#include "PathSampler.h"
#include "SampleManager.h"

#include <math.h>

using namespace std;

class SSAEvaluator : public Evaluator {
 private:
  std::unordered_set<unsigned long> seed_set_;
  vector<unsigned long> mapping_index_to_node_;
  unordered_map<unsigned long, unsigned long> mapping_node_to_index_;
  vector<shared_ptr<vector<unsigned long>>> rr_samples_;
  vector<vector<unsigned int>> hyper_graph_;

  std::random_device rd_;
  std::mt19937 gen_;
  double epsilon_;
  double delta_;
  std::uniform_int_distribution<unsigned long> dst_;
  bool incremental_;

 public:
  SSAEvaluator(double epsilon)
      : gen_(rd_()), epsilon_(epsilon) {};

  void setIncremental(bool inc) { incremental_ = inc; }

  std::unordered_set<unsigned long> select(
        const Graph& graph, Sampler& sampler,
        const std::unordered_set<unsigned long>& activated,
        unsigned int k, unsigned long samples) {

    hyper_graph_.clear();
    rr_samples_.clear();
    delta_ = 1. / graph.get_number_nodes();
    dst_ = uniform_int_distribution<unsigned long>(0, graph.get_number_nodes() - 1);
    mapping_index_to_node_ = vector<unsigned long>(graph.get_number_nodes(), 0);
    unsigned long index = 0;
    for (auto node : graph.get_nodes()) {
      mapping_index_to_node_[index] = node;
      mapping_node_to_index_[node] = index;
      index++;
    }

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
    cerr << "Lambda_1 = " << lambda_1 << endl;
    while(true) {
      unsigned long n_new_samples = n_samples - rr_samples_.size();
      clock_t begin = clock();
      buildSamples(n_new_samples, graph, sampler, activated);
      double timee = (double)(clock() - begin) / CLOCKS_PER_SEC;
      cerr << "Time to sample " << n_new_samples << " RR sets = " << timee << "s." << endl;
      n_samples *= 2;
      double biased_estimator = buildSeedSet(graph, k);
      cerr << "biased_estimator = " << biased_estimator << ", cov = " << biased_estimator * rr_samples_.size() / graph.get_number_nodes() << endl;
      if (biased_estimator * rr_samples_.size() / graph.get_number_nodes() >= lambda_1) {
        unsigned int T_max = (unsigned int)(2 * rr_samples_.size() *
              (1 + epsilon_2) / (1 - epsilon_2) * epsilon_3 * epsilon_3 /
              (k * epsilon_2 * epsilon_2));
        double unbiased_estimator = estimateInf(graph, sampler, epsilon_2,
                                                delta_ / 3, T_max);
        cerr << "6 : unbiased_estimator = " << unbiased_estimator << ", prod = " << (1 + epsilon_1) * unbiased_estimator << endl;
        if (biased_estimator <= (1 + epsilon_1) * unbiased_estimator) {
          return seed_set_;
        }
      }
    }
    return seed_set_;
  }

 private:
  /**
  * Influence estimation of a given seed set seed_set_
  */
  double estimateInf(const Graph &graph, Sampler &sampler, double epsilon_2,
                     double delta_2, unsigned int T_max) {
    vector<unsigned long> nodes_activated(graph.get_number_nodes(), 0);
    vector<bool> bool_activated(graph.get_number_nodes(), false);
    double lambda_2 = 1 + (2 + 2 * epsilon_2 / 3) * (1 + epsilon_2) *
          (log(1 / delta_2) + log(2 * log2(graph.get_number_nodes()))) /
          (epsilon_2 * epsilon_2);
    cerr << "Lambda_2 " << lambda_2 << endl;
    cerr << "T_max = " << T_max << endl;
    double cov = 0;
    for (unsigned int i = 0; i < T_max; i++) {
      unsigned long source = dst_(gen_);
      // We sample a new RR set
      shared_ptr<vector<unsigned long>> rr_sample = sampler.perform_unique_sample(
          graph, nodes_activated, bool_activated, source, true);  // can be improved because if we found a node from seed_set, we can stop diffusion
      for (unsigned long sampled_node : *rr_sample) {
        if (seed_set_.find(sampled_node) != seed_set_.end()) {
          cov += 1;
          break;
        }
      }
      if (cov >= lambda_2)
        return graph.get_number_nodes() * cov / i;
    }
    cerr << "T_max = " << T_max << ", cov = " << cov << ", ratio =" << (double)T_max / cov << endl;
    cerr << "estimation = " << cov*graph.get_number_nodes() / T_max << endl;
    return -1;
  }

  /**
  * Samples n_samples new RR sets and add them to set of RR samples rr_samples_
  */
  void buildSamples(const unsigned long n_samples, const Graph& graph,
                    Sampler& sampler, const unordered_set<unsigned long> &activated) {

    vector<unsigned long> nodes_activated(graph.get_number_nodes(), 0);
    vector<bool> bool_activated(graph.get_number_nodes(), false);

    for (unsigned int i = 0; i < n_samples; i++) {
      unsigned long source = dst_(gen_);
      while (activated.find(source) != activated.end()) { // While the random nodes was already activated
        source = dst_(gen_);
      }
      shared_ptr<vector<unsigned long>> rr_sample = sampler.perform_unique_sample(
            graph, nodes_activated, bool_activated, source, true);
      rr_samples_.push_back(rr_sample);
      for (unsigned long node : *rr_sample) {
        hyper_graph_[mapping_node_to_index_[node]].push_back(i);
      }
    }
  }

  /**
  * Greedy algorithm computing the maximum coverage
  */
  double buildSeedSet(const Graph &graph, unsigned int k) {
    seed_set_.clear();
    vector<unsigned int> degree = vector<unsigned int>(graph.get_number_nodes(), 0);  // Number of covered sets
    vector<bool> visited_samples = vector<bool>(rr_samples_.size(), false);
    for (unsigned int i = 0; i < hyper_graph_.size(); i++) {
      degree[i] = hyper_graph_[i].size();
    }
    for (unsigned int i = 0; i < k; i++) {
      int index = max_element(degree.begin(), degree.end()) - degree.begin();
      seed_set_.insert(mapping_index_to_node_[index]);
      degree[index] = 0;
      for (unsigned int rr_sample_id : hyper_graph_[index]) {
        if (!visited_samples[rr_sample_id]) {
          visited_samples[rr_sample_id] = true;
          for (unsigned long node : *rr_samples_[rr_sample_id]) {
            degree[mapping_node_to_index_[node]]--;
          }
        }
      }
    }
    double cov = 0;
    for (bool visited : visited_samples) {
      if (visited)
        cov += 1;
    }
    cerr << "Cov = " << cov << ", #RR samples = " << rr_samples_.size() << endl;
    return cov * graph.get_number_nodes() / rr_samples_.size();
  }

};

#endif /* defined(__oim__SSAEvaluator__) */
