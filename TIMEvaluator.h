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
  std::unordered_set<unsigned long> activated;
  unsigned int k;
  unsigned long n;
  unsigned long n_max;
  unsigned long m;
  double epsilon;

  std::unordered_set<unsigned long> seedSet;
  std::vector<std::shared_ptr<std::vector<unsigned long>>> rr_sets;
  std::vector<unsigned long> graph_nodes;
  std::vector<std::shared_ptr<std::vector<unsigned long>>> hyperG;
  int64 hyperId;
  int64 totalR;
  //random devices
  std::random_device rd;
  std::mt19937 gen;
  bool INCREMENTAL;

 public:
  TIMEvaluator() : gen(rd()) {};

  void setIncremental(bool inc) { INCREMENTAL = inc; }

  std::unordered_set<unsigned long> select(
      const Graph& graph, Sampler& sampler,
      const std::unordered_set<unsigned long>& activated,
      unsigned int k, unsigned long samples) {

    timestamp_t t0, t1, t2;

    for (auto node : activated) {
      (this->activated).insert(node);
    }
    this->k = k;

    t0 = get_timestamp();
    n = graph.get_number_nodes();
    m = graph.get_number_edges();
    hyperId = 0;
    totalR = 0;
    epsilon = 0.1;

    graph_nodes.clear();
    n_max = 0;
    for (auto src : graph.get_nodes()) {
      if (activated.find(src) == activated.end()) {
        if (src > n_max) n_max = src;
        graph_nodes.push_back(src);
      }
    }

    PathSampler sampler_s(sampler.get_quantile());

    std::uniform_int_distribution<int> dst(0, (int)graph_nodes.size() - 1);

    double ep_step2, ep_step3;
    ep_step2 = ep_step3 = epsilon;
    ep_step2 = 5 * pow(sqr(ep_step3) / k, 1.0 / 3.0);
    double ept;

    ept = EstimateEPT(graph, sampler_s, dst);
    std::cerr << "Profiling 1: " << (get_timestamp() - t0) / 1000000 << "sec" << std::endl;

    BuildSeedSet();
    std::cerr << "Profiling 2: " << (get_timestamp() - t0) / 1000000 << "sec" << std::endl;

    BuildHyperGraph2(ep_step2, ept, graph, sampler_s, dst);
    std::cerr << "Profiling 3: " << (get_timestamp() - t0) / 1000000 << "sec" << std::endl;
    ept = InfluenceHyperGraph();
    std::cerr << "Profiling 4: " << (get_timestamp() - t0) / 1000000 << "sec" << std::endl;

    ept /= 1 + ep_step2;

    BuildHyperGraph3(ep_step3, ept, graph, sampler_s, dst);

    t1 = get_timestamp();
    std::cerr << "Sampling time: " << (t1 - t0) / 1000000 << "sec" << std::endl;

    BuildSeedSet();

    t2 = get_timestamp();
    std::cerr << "Choice time: " << (t2 - t1) / 1000000 << "sec" << std::endl;

    sampling_time = (t1 - t0) / 60000000.0L;
    choosing_time = (t2 - t1) / 60000000.0L;

    return seedSet;
  }

 private:
  double EstimateEPT(const Graph& graph, Sampler& sampler,
                     std::uniform_int_distribution<int>& dst) {
    double ept = Estimate_KPT(graph, sampler, dst);
    ept /= 2;
    return ept;
  }

  double Estimate_KPT(const Graph& graph, Sampler& sampler,
                      std::uniform_int_distribution<int>& dst) {
    double lb = 1 / 2.0;
    double c = 0;
    int64 lastR = 0;

    double return_value = 1;
    int steps = 1;  // added for algorithm 2 line 1
    while (steps <= log(n) / log(2) - 1) {
      int loop = (6 * log(n) + 6 * log(log(n)/ log(2))) / lb;
      c = 0;
      lastR = loop;

      for (int i = 0; i < loop; i++) {
        std::shared_ptr<std::vector<unsigned long>>
          rr(new std::vector<unsigned long>());
        if (!INCREMENTAL) {
          std::unordered_set<unsigned long> seeds;
          unsigned long u = graph_nodes[dst(gen)];
          seeds.insert(u);
          rr->push_back(u);

          sampler.trial(graph, activated, seeds, true);

          for (trial_type tt : sampler.get_trials()) {
            if (tt.trial == 1) {
              rr->push_back(tt.target);
            }
          }
        } else {
          rr = SampleManager::getInstance()->getSample(
              graph_nodes, sampler, activated, dst);
        }
        double MgTu = 0;
        for (auto node : (*rr)) {
          if (graph.has_neighbours(node,true)) {
            MgTu += graph.get_neighbours(node,true).size();
          }
        }
        double pu = MgTu / m;
        c += 1 - pow(1 - pu, k);
      }
      c /= loop;
      if (c > lb) { return_value = c * n; break; }
      lb /= 2;
      steps++;
    }
    buildSamples(lastR, graph, sampler, dst);
    return return_value;
  }

  void buildSamples(int64 &R, const Graph& graph, Sampler& sampler,
                    std::uniform_int_distribution<int>& dst) {
    totalR += R;

    if (R > MAX_R)
      R = MAX_R;
    hyperId = R;

    hyperG.clear();
    hyperG.reserve(n);
    for (unsigned int i = 0; i < n; ++i) {
      hyperG.push_back(std::shared_ptr<std::vector<unsigned long>>(
          new std::vector<unsigned long>()));
    }

    rr_sets.clear();
    rr_sets.reserve(R);

    double totTime = 0.0;
    double totInDegree = 0;

    for (int i = 0; i < R; i++) {
      if (!INCREMENTAL) {
        std::shared_ptr<std::vector<unsigned long>> rr(
            new std::vector<unsigned long>());
        std::unordered_set<unsigned long> seeds;
        unsigned long nd = graph_nodes[dst(gen)];
        seeds.insert(nd);
        rr->push_back(nd);

        timestamp_t t0, t1;
        t0 = get_timestamp();
        sampler.trial(graph, activated, seeds, true);
        t1 = get_timestamp();
        totTime += (t1 - t0) / 60000000.0L;

        totInDegree += sampler.get_trials().size();
        for (trial_type tt : sampler.get_trials()) {
          if (tt.trial == 1) {
            //deg[tt.target] += 1;
            rr->push_back(tt.target);
          }
        }
        rr_sets.push_back(rr);
      } else {
        rr_sets.push_back(SampleManager::getInstance()->getSample(
              graph_nodes, sampler, activated, dst));
      }
    }

    for (int i = 0; i < R; i++) {
      for (unsigned long t : (*rr_sets[i])) {
        hyperG[t]->push_back(i);
      }
    }
  }

  vector<bool> visit_local;
  void BuildSeedSet() {
    seedSet.clear();
    vector<int> deg = vector<int>(n, 0);
    visit_local = vector<bool>(rr_sets.size(), false);

    for (unsigned int i = 0; i < graph_nodes.size(); ++i) {
      deg[graph_nodes[i]] = hyperG[graph_nodes[i]]->size();
    }

    for (unsigned int i = 0; i < k; ++i) {
      auto t = max_element(deg.begin(), deg.end());
      int id = t - deg.begin();
      seedSet.insert(id);
      deg[id] = 0;
      for (int t : (*hyperG[id])) {
        if (!visit_local[t]) {
          visit_local[t] = true;
          for (int item : (*rr_sets[t])) {
            deg[item]--;
          }
        }
      }
    }
  }

  double InfluenceHyperGraph() {
    unordered_set<unsigned long> s;
    for(auto t : seedSet) {
      for(auto tt : (*hyperG[t])) {
        s.insert(tt);
      }
    }
    double inf = (double)n * s.size() / hyperId;
    return inf;
  }

  void BuildHyperGraph2(double epsilon, double ept, const Graph& graph,
                        Sampler& sampler,
                        std::uniform_int_distribution<int>& dst) {
    int64 R = (8 + 2 * epsilon) * (n * log(n) + n * log(2)) /
        (epsilon * epsilon * ept) / 4;
    buildSamples(R, graph, sampler, dst);
  }

  void BuildHyperGraph3(double epsilon, double opt, const Graph& graph,
                        Sampler& sampler,
                        std::uniform_int_distribution<int>& dst) {
    double logCnk = 0.0;
    for (unsigned long i = n, j = 1; j <= k; --i, ++j) {
      logCnk += log10(i) - log10(j);
    }
    int64 R = (8 + 2 * epsilon) * (n * log(n) + n * log(2) + n * logCnk) /
        (epsilon * epsilon * opt);
    buildSamples(R, graph, sampler, dst);
  }

};

#endif /* defined(__oim__TIMEvaluator__) */
