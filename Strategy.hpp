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

#ifndef __oim__Strategy__
#define __oim__Strategy__

#include "common.hpp"
#include "Evaluator.hpp"
#include "SpreadSampler.hpp"
#include "PathSampler.hpp"
#include "Graph.hpp"
#include "GraphReduction.hpp"
#include "Policy.hpp"
#include "LogDiffusion.hpp"

#include <iostream>
#include <unordered_set>
#include <sys/time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <limits>


double sampling_time = 0;
double choosing_time = 0;
double reused_ratio = 0;

struct TrialData {
  std::unordered_set<unode_int> seeds;
  double spread;
  std::vector<TrialType> trials;
};

/**
  Abstract class which is implemented by all strategies.
*/
class Strategy {
 protected:
  Graph& original_graph_; // The real graph (unknown from the decision maker)
  int model_;    // 0 for linear threshold, 1 for cascade model
  boost::mt19937 gen_;
  boost::uniform_01<boost::mt19937> dist_;
  std::shared_ptr<LogDiffusion> log_diffusion_; // Pointer to the structure handling cascades (nullptr if we don't use logs)

 public:
  Strategy(Graph& original_graph, int model,
           std::shared_ptr<LogDiffusion> diffusion)
      : original_graph_(original_graph), model_(model),
        gen_(seed_ns()), dist_(gen_), log_diffusion_(diffusion) {}

  virtual void perform(unsigned int budget, unsigned int k) = 0;
};

/**
  Strategy using the influence maximization algorithm (Evaluator) on the *known*
  graph. This strategy isn't an online version that doesn't know the original
  graph, but only a (oracle) baseline.

  Note, we also use this strategy for Random and HighestDegree evaluators (but
  it does not use the extra knowledge then)
*/
class OriginalGraphStrategy : public Strategy {
 private:
  Evaluator& evaluator_;
  unode_int samples_;

 public:
  OriginalGraphStrategy(Graph& original_graph, Evaluator& evaluator,
                        double n_samples, int model=1,
                        std::shared_ptr<LogDiffusion> diffusion=nullptr)
      : Strategy(original_graph, model, diffusion),
        evaluator_(evaluator), samples_(n_samples) {}

  void perform(unsigned int budget, unsigned int k) {
    SpreadSampler sampler(INFLUENCE_MED, model_);
    std::unordered_set<unode_int> activated;
    double expected = 0, real = 0, roundtime = 0, timetotal = 0;
    for (unsigned int stage = 0; stage < budget; stage++) {
      timestamp_t t0, t1;
      t0 = get_timestamp();

      // Select seeds using explore or exploit
      std::unordered_set<unode_int> seeds =
          evaluator_.select(original_graph_, sampler, activated, k);

      // Evaluating the expected and real spread on the seeds
      double new_expected = 0;
      for (unsigned int i = 0; i < samples_; i++) {
        std::unordered_set<unode_int> spread;
        if (log_diffusion_ == nullptr)  // We sample a diffusion according to a model
          spread = sampler.perform_diffusion(original_graph_, seeds);
        else    // We sample a cascade from the seeds at random (cascdes from the LOGS)
          spread = log_diffusion_->perform_diffusion(seeds);
        for (auto& elt : spread)
          if (activated.find(elt) == activated.end())
            new_expected++;
      }
      expected += new_expected / samples_;

      // Perform real diffusion
      std::unordered_set<unode_int> diffusion;
      if (log_diffusion_ == nullptr)  // We sample a diffusion according to a model
        diffusion = sampler.perform_diffusion(original_graph_, seeds);
      else    // We sample a cascade from the seeds at random (cascdes from the LOGS)
        diffusion = log_diffusion_->perform_diffusion(seeds);
      for (auto& node : diffusion)
        activated.insert(node);
      real = activated.size();

      t1 = get_timestamp();
      // Printing results
      timetotal += (double)(t1 - t0) / 1000000;
      roundtime = (double)(t1 - t0) / 1000000;
      std::cout << stage << "\t" << real << "\t" << expected << "\t"
                << roundtime << "\t" << timetotal << "\t" << k << "\t"
                << model_ << "\t";
      for (auto seed : seeds)
        std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

/**
  Strategy using the missing mass estimator to sequentially select the best k
  experts. See our paper for further details.

  Output: on standard output
    stage <TAB> realspread <TAB> totaltime <TAB> roundtime <TAB> selectingtime
          <TAB> updatingtime <TAB> memory <TAB> seed1.seed2.....seedN
*/
class MissingMassStrategy : public Strategy {
 private:
  GraphReduction& g_reduction_;
  int n_experts_;
  unsigned int n_policy_;
  int n_graph_reduction_;

 public:
  /**
    Give the graph, the reduction method with the number of experts.
    The last argument refers to the Policy employed (must be chosen among
    {RandomPolicy, GoodUCBPolicy}).
  */
  MissingMassStrategy(Graph& original_graph, GraphReduction& g_reduction,
                      int n_experts, unsigned int n_policy=1, int model=1,
                      std::shared_ptr<LogDiffusion> diffusion=nullptr)
      : Strategy(original_graph, model, diffusion), g_reduction_(g_reduction),
        n_experts_(n_experts), n_policy_(n_policy) {}

  /**
    Performs the experiment with the missing mass strategy (good-UCB estimator).
    (Be careful, if `n_policy`=0, the policy selects randomly the experts at
    each round: the strategy doesn't rely on missing mass anymore).

    Output: stage <TAB> totalspread" <TAB> reductiontime <TAB> totaltime <TAB>
            roundtime <TAB> selectingtime" <TAB> updatingtime <TAB>
            memory <TAB> experts.
  */
  void perform(unsigned int budget, unsigned int k) {
    SpreadSampler exploit_spread(INFLUENCE_MED, model_);
    double totaltime = 0, roundtime = 0, memory = 0,
           updatingtime = 0, selectingtime = 0, reductiontime = 0;
    std::unordered_set<unode_int> total_spread;

    // 1. (a) Extract experts from graph
    timestamp_t t0, t1;
    t0 = get_timestamp();
    std::vector<unode_int> experts = g_reduction_.extractExperts(
        original_graph_, n_experts_); // So far, we do not give children of experts
    t1 = get_timestamp();
    reductiontime = (double)(t1 - t0) / 1000000;

    // 1. (b) Create the right policy object
    vector<unode_int> nb_neighbours(
        n_experts_, original_graph_.get_number_nodes());
    std::unique_ptr<Policy> policy;
    if (n_policy_ == 0) {
      policy = std::unique_ptr<Policy>(new RandomPolicy(n_experts_));
    } else if (n_policy_ == 1) {
      policy = std::unique_ptr<Policy>(
          new GoodUcbPolicy(n_experts_, nb_neighbours));
    }
    policy->init();

    // 2. Sequentially select the best k nodes from missing mass estimator ucb
    std::unordered_set<unode_int> spread;
    for (unsigned int stage = 0; stage < budget; stage++) {
      // 2. (a) Select k experts for this round
      timestamp_t t2;
      t0 = get_timestamp();
      std::unordered_set<unsigned int> chosen_experts = policy->selectExpert(k);
      t1 = get_timestamp();
      selectingtime = (double)(t1 - t0) / 1000000;

      // 2. (b) Apply diffusion
      std::unordered_set<unode_int> seeds;
      for (unsigned int chosen_expert : chosen_experts) {
        seeds.insert(experts[chosen_expert]); // We add the associated node
      }
      if (log_diffusion_ == nullptr) {  // We sample a diffusion according to a model
        spread = exploit_spread.perform_diffusion(original_graph_, seeds);
      }
      else    // We sample a cascade from the seeds at random (cascdes from the LOGS)
        spread = log_diffusion_->perform_diffusion(seeds);
      total_spread.insert(spread.begin(), spread.end());

      // 3. (c) Update statistics of experts
      std::vector<unode_int> expert_nodes;  // For each expert, the associated node in the graph
      for (unsigned int chosen_expert : chosen_experts)
        expert_nodes.push_back(experts[chosen_expert]);
      auto expert_spreads = extract_expert_spreads(
          spread, expert_nodes, k);  // Spread associated to each expert
      int n_expert = 0;
      for (unsigned int expert : chosen_experts) {
        policy->updateState(expert, expert_spreads[n_expert]);
        n_expert++;
      }
      t2 = get_timestamp();
      updatingtime = (double)(t2 - t1) / 1000000.;
      roundtime = (double)(t2 - t0) / 1000000;
      totaltime += roundtime;
      memory = disp_mem_usage();

      // 4. Printing results
      std::cout << stage << "\t" << total_spread.size() << '\t'
                << reductiontime << "\t" << selectingtime << "\t"
                << updatingtime << "\t" << roundtime << "\t"
                << totaltime << "\t" << memory << "\t" << k << "\t"
                << n_experts_ << "\t" << n_policy_ << "\t"
                << n_graph_reduction_ << "\t" << model_ << "\t";
      for (auto seed : seeds)
        std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }

  /**
    Set the Graph reduction method number in order to write it on the standard
    output (useful for experiments).
  */
  void set_graph_reduction(int n_reduction) {
    n_graph_reduction_ = n_reduction;
  }

 private:
  /**
    Returns the spread associated to each selected expert.

    We assign each activated node to the closest expert using the geodesic
    distance (graph distance == shortest path).
  */
  std::vector<std::unordered_set<unode_int>> extract_expert_spreads(
      std::unordered_set<unode_int>& stage_spread,
      const std::vector<unode_int>& expert_nodes, unsigned int k) {
    std::vector<std::unordered_set<unode_int>> res;
    std::vector<std::queue<unode_int>> queues;
    // Initialization
    for (unsigned int i = 0; i < k; i++) {
      queues.push_back(std::queue<unode_int>());
      queues[i].push(expert_nodes[i]);
      res.push_back(std::unordered_set<unode_int>());
      res[i].insert(expert_nodes[i]);
      stage_spread.erase(expert_nodes[i]);
    }
    // While we haven't assigned each activated node to an expert
    while (stage_spread.size() > 0) {
      std::unordered_set<unode_int> seen_round;
      for (unsigned int i = 0; i < k; i++) {
        unsigned int queue_size = queues[i].size(); // Number of elements to inspect at this round
        for (unsigned j = 0; j < queue_size; j++) {
          auto node = queues[i].front();
          queues[i].pop();
          if (!original_graph_.has_neighbours(node))
            continue;
          for (auto& neighbour : original_graph_.get_neighbours(node)) {
            if (stage_spread.find(neighbour.target) == stage_spread.end())  // This node has not been activated
              continue;
            seen_round.insert(neighbour.target);
            queues[i].push(neighbour.target);
            res[i].insert(neighbour.target);
          }
        }
      }
      for (auto elt : seen_round)
        stage_spread.erase(elt);
    }
    return res;
  }
};

/**
  Confidence bound strategy that dynamically updates the factor of exploration
  theta using exponentiated gradients.
*/
class ExponentiatedGradientStrategy : public Strategy {
 private:
  Graph& model_graph_;
  Evaluator& evaluator_;
  bool update_;
  unsigned int learn_;  // Corresponds to update_type

 public:
  ExponentiatedGradientStrategy(Graph& model_graph, Graph& original_graph,
                                Evaluator& evaluator, bool update=true,
                                unsigned int learn=0, int model=1,
                                std::shared_ptr<LogDiffusion> diffusion=nullptr)
      : Strategy(original_graph, model, diffusion), model_graph_(model_graph),
        evaluator_(evaluator), update_(update), learn_(learn) {}

  void perform(unsigned int budget, unsigned int k) {
    std::vector<double> p(3, 0.333);
    double w[3] = {1.0, 1.0, 1.0};
    unsigned int cur_theta = THETA_OFFSET;
    double mu = log(300.0) / (3 * budget);
    double tau = 12 * mu / (3.0 + mu);
    double lambda = tau / 6.0;
    SpreadSampler exploit_sampler(INFLUENCE_MED, model_); // Sampler for *real* graph
    std::unordered_set<unode_int> activated;
    double expected = 0, real = 0, selectingtime = 0, updatingtime = 0,
           totaltime = 0, roundtime = 0, memory = 0;
    double alpha = 1, beta = 1;
    std::vector<TrialData> results;
    std::unordered_map<long long, int> edge_hit, edge_miss;

    for (unsigned int stage = 0; stage < budget; stage++) {
      timestamp_t t0, t1, t2;
      t0 = get_timestamp();
      // Sampling the distribution
      std::discrete_distribution<int> prob(p.begin(), p.end());
      cur_theta = prob(gen_) + THETA_OFFSET;

      // PathSampler path_sampler(cur_theta); (version with path sampler, not used anymore)
      SpreadSampler explore_sampler(cur_theta, 1);  // For expgr, only model cascade can be handled

      // Selecting seeds using explore or exploit
      std::unordered_set<unode_int> seeds;
      seeds = evaluator_.select(model_graph_, explore_sampler, activated, k);
      // Evaluating the expected and real spread on the seeds
      double cur_expected = 0.1;
      if (update_) {  // We don't compute for Random and HighestDegree because it's useless
        cur_expected = explore_sampler.sample(model_graph_, activated, seeds, 100);
      }
      expected += cur_expected;
      double cur_real = exploit_sampler.trial(original_graph_, activated, seeds);
      double cur_gain = 1.0 - std::abs(cur_real - cur_expected) / cur_expected;
      real += cur_real;
      // Recalibrating
      for (unsigned int i = 0; i < 3; i++) {
        if (i == cur_theta - THETA_OFFSET)
          w[i] = w[i] * exp(lambda * (cur_gain + mu) / p[i]);
        else
          w[i] = w[i] * exp(lambda * mu / p[i]);
      }
      double sum_w = 0;
      for (int i = 0; i < 3; i++)
        sum_w += w[i];
      for (int i = 0; i < 3; i++)
        p[i] = (1 - tau) * w[i] / sum_w + tau / 3.0;

      t1 = get_timestamp();
      selectingtime = (double)(t1 - t0) / 1000000;

      std::unordered_set<unode_int> nodes_to_update;
      for (unode_int node : seeds) {
        activated.insert(node);
        nodes_to_update.insert(node);
      }
      unsigned int hits = 0, misses = 0;
      for (TrialType tt : exploit_sampler.get_trials()) {
        nodes_to_update.insert(tt.target);
        if (tt.trial == 1) {
          hits++;
          activated.insert(tt.target);
        } else {
          misses++;
        }
        if (update_)
          model_graph_.update_edge(tt.source, tt.target, tt.trial);
      }

      if (learn_ > 0) {
        TrialData result;
        for (unode_int seed : seeds)
          result.seeds.insert(seed);
        result.spread = cur_real;
        for (TrialType tt : exploit_sampler.get_trials())
          result.trials.push_back(tt);
        results.push_back(result);
        // Linear regression learning
        if (learn_ == 1) {
          double total_spread = 0, total_seeds = 0;
          for(TrialData res : results) {
            total_spread += res.spread;
            total_seeds += (double) res.seeds.size();
          }
          double avg_spread = total_spread / total_seeds;
          double xy = 0, xx = 0;
          for (TrialData res : results) {
            double x = res.spread - 1;
            double y = 0;
            for (unode_int seed : res.seeds) {
              double o = 0, t = 0, h = 0;
              for (auto node : model_graph_.get_neighbours(seed)) {
                o += 1;
                t += (double)(node.dist->get_hits() + node.dist->get_misses());
                h += (double)node.dist->get_hits();
              }
              y += -(t + 1) * x + (o + h) * avg_spread;
            }
            xy += x * y;
            xx += x * x;
          }
          beta = xy / xx;
          beta = (beta > 0) ? beta : -beta;
        } else if (learn_ == 3) { // MLE learning
          double t = 0.0, a = 0.0;
          for (TrialData res : results) {
            t += (double)res.trials.size();
            for (TrialType tt : res.trials) a += (double)tt.trial;
          }
          alpha += a;
          beta += t - a;
        } else if (learn_ == 2) { // MLE with alpha = 1
          for (TrialType tt : result.trials) {
            long long edge = tt.source * 100000000LL + tt.target;
            if (tt.trial == 0) {
              auto iter = edge_miss.find(edge);
              if (iter == edge_miss.end()) edge_miss[edge] = 1;
              else ++(iter->second);
            } else {
              auto iter = edge_hit.find(edge);
              if (iter == edge_hit.end()) edge_hit[edge] = 1;
              else ++(iter->second);
            }
          }
          alpha = 1;
          double a = 0.0;
          for (auto item = edge_hit.begin(); item != edge_hit.end(); ++item) {
            a -= 1 / (alpha + item->second);
          }
          auto fbeta = [&](double beta, double a) {
            double ret = 0.0;
            for (auto item = edge_miss.begin(); item != edge_miss.end(); ++item)
              ret += 1.0 / (beta + item->second);
            return ret + a;
          };
          double beta_L = 1, beta_R = 1;
          while (fbeta(beta_R, a) > -1e-9 && beta_R < 10000) {
            beta_R *= 2;
          }
          beta = beta_R;
          while (beta_L < beta_R - 1) {
            beta = (beta_L + beta_R) / 2;
            double calc = fbeta(beta, a);
            if (-1e-9 <= calc && calc <= 1e-9) break;
            if (calc < 0) beta_R = beta;
            else beta_L = beta;
          }
        }
        model_graph_.update_edge_priors(alpha, beta);
      }
      model_graph_.update_rounds((double)(stage + 1));

      t2 = get_timestamp();
      updatingtime = (double)(t2 - t1) / 1000000;
      roundtime = (double)(t2 - t0) / 1000000;
      totaltime += roundtime;
      memory = disp_mem_usage();

      // Printing results
      std::cout << stage << "\t" << real << "\t" << expected << "\t"
          << selectingtime << "\t" << updatingtime << "\t" << roundtime << "\t"
          << totaltime  << "\t" << (int)cur_theta - THETA_OFFSET - 1 << "\t"
          << memory << "\t" << k << "\t" << model_ << "\t";
      for (auto seed : seeds)
        std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

#endif /* defined(__oim__Strategy__) */
