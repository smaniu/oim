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

#ifndef __oim__TIMEvaluator__
#define __oim__TIMEvaluator__

#include "common.h"
#include "Graph.h"
#include "Evaluator.h"
#include "Sampler.h"
#include "SpreadSampler.h"
#include "PathSampler.h"
#include "SampleManager.h"

#include <math.h>

using namespace std;

class TIMEvaluator : public Evaluator {
 private:
  std::unordered_set<unsigned long> activated_;
  unsigned int k_;
  unsigned long n_;
  unsigned long n_max_;
  unsigned long m_;
  double epsilon_;

  std::unordered_set<unsigned long> seed_set_;
  std::vector<std::shared_ptr<std::vector<unsigned long>>> rr_sets_;
  std::vector<unsigned long> graph_nodes_;
  std::vector<std::shared_ptr<std::vector<unsigned long>>> hyper_g_;
  int64 hyper_id_;
  int64 total_r_;
  std::random_device rd_;
  std::mt19937 gen_;

 public:
  TIMEvaluator() : gen_(rd_()) {};

  std::unordered_set<unsigned long> select(
      const Graph& graph, Sampler& sampler,
      const std::unordered_set<unsigned long>& activated,
      unsigned int k, unsigned long samples) {

    timestamp_t t0, t1, t2;

    for (auto node : activated) {
      activated_.insert(node);
    }
    k_ = k;

    t0 = get_timestamp();
    n_ = graph.get_number_nodes();
    m_ = graph.get_number_edges();
    hyper_id_ = 0;
    total_r_ = 0;
    epsilon_ = 0.1;

    graph_nodes_.clear(); // Nodes not yet activated
    n_max_ = 0;
    for (auto src : graph.get_nodes()) {
      if (activated.find(src) == activated.end()) {
        if (src > n_max_) n_max_ = src;
        graph_nodes_.push_back(src);
      }
    }

    PathSampler sampler_s(sampler.get_type());
    std::uniform_int_distribution<int> dst(0, (int)graph_nodes_.size() - 1);
    double ep_step2, ep_step3;
    ep_step2 = ep_step3 = epsilon_;
    ep_step2 = 5 * pow(sqr(ep_step3) / k, 1.0 / 3.0);
    double ept;

    ept = EstimateEPT(graph, sampler_s, dst);
    BuildSeedSet();
    BuildHyperGraph2(ep_step2, ept, graph, sampler_s, dst);
    ept = InfluenceHyperGraph();
    ept /= 1 + ep_step2;
    BuildHyperGraph3(ep_step3, ept, graph, sampler_s, dst);
    t1 = get_timestamp();
    BuildSeedSet();
    t2 = get_timestamp();
    sampling_time = (double)(t1 - t0) / 1000000;
    choosing_time = (double)(t2 - t1) / 1000000;

    return seed_set_;
  }

 private:
  double EstimateEPT(const Graph& graph, Sampler& sampler,
                     std::uniform_int_distribution<int>& dst) {
    double ept = EstimateKPT(graph, sampler, dst);
    ept /= 2;
    return ept;
  }

  double EstimateKPT(const Graph& graph, Sampler& sampler,
                     std::uniform_int_distribution<int>& dst) {
    double lb = 1 / 2.0;
    double c = 0;
    int64 last_r = 0;

    double return_value = 1;
    int steps = 1;  // added for algorithm 2 line 1
    while (steps <= log(n_) / log(2) - 1) {
      int loop = (6 * log(n_) + 6 * log(log(n_)/ log(2))) / lb;
      c = 0;
      last_r = loop;

      for (int i = 0; i < loop; i++) {
        std::shared_ptr<std::vector<unsigned long>>
            rr(new std::vector<unsigned long>());
        if (!incremental_) {
          std::unordered_set<unsigned long> seeds;
          unsigned long u = graph_nodes_[dst(gen_)];
          seeds.insert(u);
          rr->push_back(u);
          sampler.trial(graph, activated_, seeds, true);
          for (TrialType tt : sampler.get_trials()) {
            if (tt.trial == 1) {
              rr->push_back(tt.target);
            }
          }
        } else {
          rr = SampleManager::getInstance()->getSample(
              graph_nodes_, sampler, activated_, dst);
        }
        double mg_tu = 0;
        for (auto node : (*rr)) {
          if (graph.has_neighbours(node,true)) {
            mg_tu += graph.get_neighbours(node,true).size();
          }
        }
        double pu = mg_tu / m_;
        c += 1 - pow(1 - pu, k_);
      }
      c /= loop;
      if (c > lb) { return_value = c * n_; break; }
      lb /= 2;
      steps++;
    }
    buildSamples(last_r, graph, sampler, dst);
    return return_value;
  }

  void buildSamples(int64& R, const Graph& graph, Sampler& sampler,
                    std::uniform_int_distribution<int>& dst) {
    total_r_ += R;

    if (R > MAX_R)
      R = MAX_R;
    hyper_id_ = R;

    hyper_g_.clear();
    hyper_g_.reserve(n_);
    for (unsigned int i = 0; i < n_; ++i) {
      hyper_g_.push_back(std::shared_ptr<std::vector<unsigned long>>(
          new std::vector<unsigned long>()));
    }

    rr_sets_.clear();
    rr_sets_.reserve(R);

    double totTime = 0.0;
    double totInDegree = 0;

    for (int i = 0; i < R; i++) {
      if (!incremental_) {
        std::shared_ptr<std::vector<unsigned long>> rr(
            new std::vector<unsigned long>());
        std::unordered_set<unsigned long> seeds;
        unsigned long nd = graph_nodes_[dst(gen_)]; // Only RR set samples from unreached nodes
        seeds.insert(nd);
        rr->push_back(nd);

        timestamp_t t0, t1;
        t0 = get_timestamp();
        sampler.trial(graph, activated_, seeds, true);
        t1 = get_timestamp();
        totTime += (double)(t1 - t0) / 1000000;

        totInDegree += sampler.get_trials().size();
        for (TrialType tt : sampler.get_trials()) {
          if (tt.trial == 1) {
            //deg[tt.target] += 1;
            rr->push_back(tt.target);
          }
        }
        rr_sets_.push_back(rr);
      } else {
        rr_sets_.push_back(SampleManager::getInstance()->getSample(
            graph_nodes_, sampler, activated_, dst));
      }
    }

    for (int i = 0; i < R; i++) {
      for (unsigned long t : (*rr_sets_[i])) {
        hyper_g_[t]->push_back(i);
      }
    }
  }

  vector<bool> visit_local;
  void BuildSeedSet() {
    seed_set_.clear();
    vector<int> deg = vector<int>(n_, 0);
    visit_local = vector<bool>(rr_sets_.size(), false);

    for (unsigned int i = 0; i < graph_nodes_.size(); ++i) {
      deg[graph_nodes_[i]] = hyper_g_[graph_nodes_[i]]->size();
    }

    for (unsigned int i = 0; i < k_; ++i) {
      auto t = max_element(deg.begin(), deg.end());
      int id = t - deg.begin();
      seed_set_.insert(id);
      deg[id] = 0;
      for (int t : (*hyper_g_[id])) {
        if (!visit_local[t]) {
          visit_local[t] = true;
          for (int item : (*rr_sets_[t])) {
            deg[item]--;
          }
        }
      }
    }
  }

  double InfluenceHyperGraph() {
    unordered_set<unsigned long> s;
    for(auto t : seed_set_) {
      for(auto tt : (*hyper_g_[t])) {
        s.insert(tt);
      }
    }
    double inf = (double)n_ * s.size() / hyper_id_;
    return inf;
  }

  void BuildHyperGraph2(double epsilon_, double ept, const Graph& graph,
                        Sampler& sampler,
                        std::uniform_int_distribution<int>& dst) {
    int64 R = (8 + 2 * epsilon_) * (n_ * log(n_) + n_ * log(2)) /
        (epsilon_ * epsilon_ * ept) / 4;
    buildSamples(R, graph, sampler, dst);
  }

  void BuildHyperGraph3(double epsilon_, double opt, const Graph& graph,
                        Sampler& sampler,
                        std::uniform_int_distribution<int>& dst) {
    double logCnk = 0.0;
    for (unsigned long i = n_, j = 1; j <= k_; --i, ++j) {
      logCnk += log10(i) - log10(j);
    }
    int64 R = (8 + 2 * epsilon_) * (n_ * log(n_) + n_ * log(2) + n_ * logCnk) /
        (epsilon_ * epsilon_ * opt);
    buildSamples(R, graph, sampler, dst);
  }

};

#endif /* defined(__oim__TIMEvaluator__) */
