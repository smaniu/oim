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

#ifndef __oim__Strategy__
#define __oim__Strategy__

#include "common.h"
#include "Evaluator.h"
#include "SpreadSampler.h"
#include "PathSampler.h"
#include "Graph.h"
#include "SampleManager.h"
#include "GraphReduction.h"

#include <iostream>
#include <unordered_set>
#include <sys/time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <cmath>
#include <limits>


double sampling_time = 0;
double choosing_time = 0;
double selecting_time = 0;
double updating_time = 0;
double reused_ratio = 0;

struct TrialData {
  std::unordered_set<unsigned long> seeds;
  double spread;
  std::vector<trial_type> trials;
};

/**
* Abstract class which is implemented by all strategies.
*/
class Strategy {
 protected:
  Graph &original_graph_;  // The real graph (unknown from the decision maker)
  unsigned long samples_;
  boost::mt19937 gen_;
  boost::uniform_01<boost::mt19937> dist_;

 public:
  Strategy(Graph &original_graph, unsigned long samples)
      : original_graph_(original_graph), samples_(samples),
        gen_(time(NULL)), dist_(gen_) {}

  virtual void perform(unsigned int budget, unsigned int k) = 0;
};

/**
* Strategy using the influence maximization algorithm (Evaluator) on the *known*
* graph. This strategy isn't an online version that doesn't know the original
* graph, but only a baseline.
*/
class OriginalGraphStrategy {
 private:
  Graph& original_g;
  Evaluator& exploit_e;
  unsigned long samples;
  bool INCREMENTAL;

 public:
  OriginalGraphStrategy(Graph& original_graph, Evaluator& eval_exploit,
                        double number_samples, bool INCREMENTAL=0)
      : original_g(original_graph), exploit_e(eval_exploit),
        samples(number_samples), INCREMENTAL(INCREMENTAL) {}

  void perform(unsigned int budget, unsigned int k) {
    SpreadSampler exploit_s(INFLUENCE_MED);
    std::unordered_set<unsigned long> activated;
    boost::mt19937 gen((int)time(0));
    boost::uniform_01<boost::mt19937> dst(gen);
    double expected = 0;
    double real = 0;
    double time_min = 0;
    for (unsigned int stage = 0; stage < budget; stage++) {
      timestamp_t t0, t1;
      t0 = get_timestamp();

      if (INCREMENTAL)
        SampleManager::reset(stage);

      //selecting seeds using explore or exploit
      std::unordered_set<unsigned long> seeds =
          exploit_e.select(original_g, exploit_s, activated, k, samples);
      //evaluating the expected and real spread on the seeds
      expected += exploit_s.sample(original_g, activated, seeds, samples);
      real += exploit_s.trial(original_g, activated, seeds);

      //updating the model graph
      std::unordered_set<unsigned long> nodes_to_update;

      for (unsigned long node : seeds) {
        activated.insert(node);
        nodes_to_update.insert(node);
      }
      unsigned int hits = 0, misses = 0;
      for (trial_type tt : exploit_s.get_trials()) {
        nodes_to_update.insert(tt.target);
        if (tt.trial == 1) {
          hits++;
          activated.insert(tt.target);
        } else {
          misses++;
        }
      }

      if (INCREMENTAL)
        SampleManager::update_node_age(nodes_to_update);

      t1 = get_timestamp();
      //printing results
      time_min += (t1 - t0) / 60000000.0L;
      std::cout << stage << "\t" << real << "\t" << expected << "\t" <<
          hits << "\t" << misses << "\t" << time_min << "\t";
      for (auto seed:seeds) std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

/**
* Good-UCB policy for our problem. Each experts maintains a missing mass
* estimator which is used to select the next expert to play. See paper for
* details.
*/
class GoodUcbPolicy {
 private:
  unsigned int nb_experts_;                   // Number of experts
  std::vector<unsigned long> &nb_neighbours_; // Number of reachable nodes for each expert (useless so far)
  unsigned int t_;                            // Number of rounds played
  std::vector<float> n_plays_;                // Number of times experts were played
  // For each expert, hashmap {node : #activations}
  std::vector<std::unordered_map<unsigned long, unsigned int>> n_rewards_;
  float SIGMA = 0.1;  // TODO change that (should be an estimator or ??)

  /**
  * Get the `k` largest elements of a vector and returns them as unordered_set.
  * Trick with negative weights to get the lowest element of the priority_queue.
  */
  template<typename T>
  std::unordered_set<T> get_k_largest_arguments(
        std::vector<float>& vec, unsigned int k) {
    std::priority_queue<std::pair<float, T>> q;
    for (T i = 0; i < k; ++i) {
      q.push(std::pair<float, T>(-vec[i], i));
    }
    for (T i = k; i < vec.size(); ++i) {
      if (q.top().first > -vec[i]) {
        q.pop();
        q.push(std::pair<float, T>(-vec[i], i));
      }
    }
    std::unordered_set<T> result;
    while (!q.empty()) {
      result.insert(q.top().second);
      q.pop();
    }
    return result;
  }

 public:
  GoodUcbPolicy(unsigned int nb_experts, std::vector<unsigned long>& nb_neighbours)
      : nb_experts_(nb_experts), nb_neighbours_(nb_neighbours) { init(); }

  /**
  * Selects `k` experts whose Good-UCB indices are the largest.
  */
  std::unordered_set<unsigned int> selectExpert(unsigned int k) {
    std::vector<float> ucbs(nb_experts_, 0);
    for (unsigned int i = 0; i < nb_experts_; i++) {
      float missing_mass_i = (float)std::count_if(
          n_rewards_[i].begin(), n_rewards_[i].end(), [](auto &elt) {
            return elt.second == 1; // Count hapaxes
          }) / n_plays_[i];
      ucbs[i] = missing_mass_i + (1 + sqrt(2)) * sqrt(SIGMA * log(4 * t_) /
          n_plays_[i]) + log(4 * t_) / (3 * n_plays_[i]);
    }
    return get_k_largest_arguments<unsigned int>(ucbs, k);
  }

  /**
  * Update statistics on the chosen expert (k == 1).
  */
  void updateState(unsigned int expert,
                   const std::unordered_set<unsigned long>& stage_spread) {
    t_++;
    for (auto& activated_node : stage_spread) {
      if (n_rewards_[expert].count(activated_node) == 0)
        n_rewards_[expert][activated_node] = 0;
      n_rewards_[expert][activated_node]++;
    }
    n_plays_[expert]++;
  }

  /**
  * Initialize datastructures required for computing UCB bounds. It is useful
  * when restarting the algorithm (to reset).
  */
  void init() {
    t_ = 0;
    n_plays_ = std::vector<float>(nb_experts_, 0);
    for (unsigned int k = 0; k < nb_experts_; k++) {
      n_rewards_[k] = std::unordered_map<unsigned long, unsigned int>();
    }
  }
};

/**
* Strategy using the missing mass estimator to sequentially select the best k
* experts. See paper : TODO description
*/
class MissingMassStrategy : public Strategy {
 private:
  GraphReduction& g_reduction_;
  int n_experts_;

 public:
  MissingMassStrategy(Graph& original_graph, unsigned long samples,
                      GraphReduction& g_reduction, int n_experts)
      : Strategy(original_graph, samples), g_reduction_(g_reduction),
        n_experts_(n_experts) {}

  void perform(unsigned int budget, unsigned int k) {
    SpreadSampler exploit_spread(INFLUENCE_MED);
    double expected = 0, real = 0, totaltime = 0, round_time = 0, memory = 0;
    std::unordered_map<long long, int> edge_hit, edge_miss;
    std::unordered_set<unsigned long> total_spread;

    // 1. Extract experts from graph
    std::vector<unsigned long> experts = g_reduction_.extractExperts(
        original_graph_, n_experts_); // So far, we do not give children of experts
    vector<unsigned long> nb_neighbours(
        n_experts_, original_graph_.get_number_nodes());
    GoodUcbPolicy policy(n_experts_, nb_neighbours);

    // 2. Sequentially select the best k nodes from missing mass estimator ucb
    std::unordered_set<unsigned long> spread;
    for (unsigned int stage = 0; stage < budget; stage++) {
      // 2. (a) Select k experts for this round
      timestamp_t t0, t1, t2;
      t0 = get_timestamp();
      std::unordered_set<unsigned int> chosen_experts = policy.selectExpert(k);
      t1 = get_timestamp();
      selecting_time = (t1 - t0) / 60000000.0L;

      // 2. (b) Apply diffusion
      std::unordered_set<unsigned long> seeds;
      for (unsigned int chosen_expert : chosen_experts) {
        seeds.insert(experts[chosen_expert]); // We add the associated node
      }
      auto stage_spread = exploit_spread.perform_diffusion(
          original_graph_, seeds);
      total_spread.insert(stage_spread.begin(), stage_spread.end());

      // 3. (c) Update statistics of experts
      for (unsigned long expert : seeds) {
        policy.updateState(expert, stage_spread);
      }
      t2 = get_timestamp();
      updating_time = (t2 - t1) / 60000000.0L;
      round_time = (t2 - t0) / 60000000.0L;
      totaltime += round_time;
      memory = disp_mem_usage();

      // Printing results
      std::cout << stage << "\t" << real << "\t" << expected << "\t" <<
          totaltime << "\t" << round_time << "\t" << sampling_time << "\t" <<
          choosing_time << "\t" << selecting_time << "\t" << updating_time <<
          "\t" << "\t" << reused_ratio << "\t" << memory << "\t";
      for (auto seed : seeds) std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

class EpsilonGreedyStrategy {
 private:
  Graph& model_g;
  Graph& original_g;
  Evaluator& explore_e;
  Evaluator& exploit_e;
  unsigned long samples;
  double epsilon;
  bool INCREMENTAL;

 public:
  EpsilonGreedyStrategy(Graph& model_graph, Graph& original_graph,
                        Evaluator& eval_explore, Evaluator& eval_exploit,
                        double number_samples, double eps, bool INCREMENTAL)
      : model_g(model_graph), original_g(original_graph),
        explore_e(eval_explore), exploit_e(eval_exploit),
        samples(number_samples), epsilon(eps), INCREMENTAL(INCREMENTAL)  {}

  void perform(unsigned int budget, unsigned int k, bool update=true,
               unsigned int learn=0,
               unsigned int interval_exploit=INFLUENCE_MED,
               unsigned int interval_explore=INFLUENCE_MED) {
    SpreadSampler exploit_s(INFLUENCE_MED);
    PathSampler exploit_p(interval_exploit);
    PathSampler explore_p(interval_explore);
    std::unordered_set<unsigned long> activated;
    boost::mt19937 gen((int)time(0));
    boost::uniform_01<boost::mt19937> dst(gen);
    double expected = 0;
    double real = 0;
    double time_min = 0;
    double alpha = model_g.alpha_prior;
    double beta = model_g.beta_prior;
    std::vector<TrialData> results;
    std::unordered_map<long long, int> edge_hit, edge_miss;

    for (unsigned int stage = 0; stage < budget; stage++) {
      timestamp_t t0, t1;
      t0 = get_timestamp();
      //selecting seeds using explore or exploit
      std::unordered_set<unsigned long> seeds;
      double dice = dst();

      if (INCREMENTAL)
        SampleManager::reset(stage, (bool)(dice < epsilon));

      if (dice < epsilon) {
        explore_e.setIncremental(INCREMENTAL);
        seeds = explore_e.select(model_g, explore_p, activated, k, samples);
      } else {
        exploit_e.setIncremental(INCREMENTAL);
        seeds = exploit_e.select(model_g, exploit_p, activated, k, samples);
      }
      //evaluating the expected and real spread on the seeds
      expected += exploit_s.sample(original_g, activated, seeds, samples);
      double cur_real = exploit_s.trial(original_g, activated, seeds);
      real += cur_real;

      //updating the model graph
      std::unordered_set<unsigned long> nodes_to_update;

      for (unsigned long node : seeds) {
        activated.insert(node);
        nodes_to_update.insert(node);
      }
      unsigned int hits = 0, misses = 0;
      for (trial_type tt : exploit_s.get_trials()) {
        nodes_to_update.insert(tt.target);
        if (tt.trial ==1 ) {
          hits++;
          activated.insert(tt.target);
        } else {
          misses++;
        }
        if (update) model_g.update_edge(tt.source, tt.target, tt.trial);
      }

      if (INCREMENTAL) {
        SampleManager::update_node_age(nodes_to_update);
      }

      //TODO learning the graph
      if (learn > 0) {
        TrialData result;
        for (unsigned long seed : seeds) result.seeds.insert(seed);
        result.spread = cur_real;
        for(trial_type tt:exploit_s.get_trials()) result.trials.push_back(tt);
        results.push_back(result);
        //linear regression learning
        if (learn == 1) {
          double total_spread = 0.0, total_seeds = 0.0;
          for (TrialData res : results) {
            total_spread += res.spread;
            total_seeds += (double) res.seeds.size();
          }
          double avg_spread = total_spread / total_seeds;
          double xy = 0.0, xx = 0.0;
          for (TrialData res : results){
            double x = res.spread - 1;
            double y = 0.0;
            for (unsigned long seed : res.seeds) {
              double o = 0.0, t = 0.0, h = 0.0;
              for (auto node : model_g.get_neighbours(seed)) {
                o += 1.0;
                t += (double)node.dist->get_hits() +
                    (double)node.dist->get_misses();
                h += (double) node.dist->get_hits();
              }
              y += -(t + 1) * x + (o + h) * avg_spread;
            }
            xy += x * y;
            xx += x * x;
          }
          beta = xy / xx;
          beta = beta > 0 ? beta : -beta;
        } else if (learn == 3) { //MLE learning
          double t = 0.0, a = 0.0;
          for (TrialData res : results) {
            t += (double) res.trials.size();
            for (trial_type tt : res.trials)
              a += (double)tt.trial;
          }
          alpha += a;
          beta += t - a;
          //beta = a>0?(t-a)/a:t;
        } else if (learn == 2) { // MLE with alpha = 1
          //cerr<< "Learning..."<<endl;
          for (trial_type tt:result.trials) {
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
        model_g.update_edge_priors(alpha, beta);
      }
      model_g.update_rounds(1.0);
      t1 = get_timestamp();
      //printing results
      time_min += (t1 - t0) / 60000000.0L;
      std::cout << stage << "\t" << real << "\t" << expected << "\t" <<
          hits << "\t" << misses << "\t" << time_min << "\t" << alpha <<
          "\t" << beta << "\t" << model_g.get_mse() << "\t";
      for (auto seed : seeds) std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

class ExponentiatedGradientStrategy {
 private:
  Graph& model_g_;
  Graph& original_g_;
  Evaluator& eval_;
  bool incremental_;

 public:
  ExponentiatedGradientStrategy(Graph& model_graph, Graph& original_graph,
                                Evaluator& eval_explore, bool incremental)
      : model_g_(model_graph), original_g_(original_graph),
        eval_(eval_explore), incremental_(incremental) {}

  void perform(unsigned int budget, unsigned int k, bool update=true,
               unsigned int learn=0) {
    double p[3] = {0.333, 0.333, 0.333};
    double w[3] = {1.0, 1.0, 1.0};
    unsigned int cur_theta = THETA_OFFSET;
    double mu = log(300.0) / (3 * budget);
    double tau = 12.0 * mu / (3.0 + mu);
    double lambda = tau / 6.0;
    SpreadSampler exploit_s(INFLUENCE_MED);

    std::unordered_set<unsigned long> activated;
    boost::mt19937 gen((int)time(0));
    boost::uniform_01<boost::mt19937> dst(gen);
    double expected = 0;
    double real = 0;
    double totaltime = 0;
    double round_time = 0;
    double alpha = 1;
    double beta = 1;
    double memory = 0;
    std::vector<TrialData> results;
    std::unordered_map<long long, int> edge_hit, edge_miss;

    for (unsigned int stage = 0; stage < budget; stage++) {
      if (incremental_)
        SampleManager::reset(stage);

      timestamp_t t0, t1, t2;
      t0 = get_timestamp();
      //sampling the distribution
      std::default_random_engine generator((int)time(0));
      // trick for stupid GCC which expects std::discrete_distribution to be
      // templated by an integral type (clang doesn't)
      int p0 = (int)(p[0] * 1000.0);
      int p1 = (int)(p[1] * 1000.0);
      int p2 = (int)(p[2] * 1000.0);
      std::discrete_distribution<int> prob {static_cast<double>(p0),
          static_cast<double>(p1), static_cast<double>(p2)};
      cur_theta = prob(generator) + THETA_OFFSET;

      PathSampler exploit_p(cur_theta);

      SpreadSampler explore_s(cur_theta);
      eval_.setIncremental(incremental_);

      //selecting seeds using explore or exploit
      std::unordered_set<unsigned long> seeds;
      seeds = eval_.select(model_g_, exploit_p, activated, k, 100);
      //evaluating the expected and real spread on the seeds
      double cur_expected = explore_s.sample(model_g_, activated, seeds, 100);
      expected += cur_expected;
      double cur_real = exploit_s.trial(original_g_, activated, seeds);
      double cur_gain = 1.0 - std::abs(cur_real-cur_expected)/cur_expected;
      real += cur_real;
      //recalibrating
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
      selecting_time = (t1 - t0) / 60000000.0L;

      std::unordered_set<unsigned long> nodes_to_update;

      for (unsigned long node : seeds) {
        activated.insert(node);
        nodes_to_update.insert(node);
      }
      unsigned int hits = 0, misses = 0;
      for (trial_type tt : exploit_s.get_trials()) {
        nodes_to_update.insert(tt.target);
        if (tt.trial == 1) {
          hits++;
          activated.insert(tt.target);
        } else {
          misses++;
        }
        if(update) model_g_.update_edge(tt.source, tt.target, tt.trial);
      }

      if (incremental_)
        SampleManager::update_node_age(nodes_to_update);

      //TODO learning the graph
      if (learn > 0) {
        TrialData result;
        for (unsigned long seed : seeds)
          result.seeds.insert(seed);
        result.spread = cur_real;
        for (trial_type tt : exploit_s.get_trials())
          result.trials.push_back(tt);
        results.push_back(result);
        //linear regression learning
        if (learn == 1) {
          double total_spread = 0.0, total_seeds = 0.0;
          for(TrialData res : results) {
            total_spread += res.spread;
            total_seeds += (double) res.seeds.size();
          }
          double avg_spread = total_spread / total_seeds;
          double xy = 0.0, xx = 0.0;
          for (TrialData res : results) {
            double x = res.spread - 1;
            double y = 0.0;
            for(unsigned long seed : res.seeds) {
              double o = 0.0, t = 0.0, h = 0.0;
              for (auto node : model_g_.get_neighbours(seed)) {
                o += 1.0;
                t += (double)node.dist->get_hits() +
                    (double)node.dist->get_misses();
                h += (double)node.dist->get_hits();
              }
              y += -(t + 1) * x + (o + h) * avg_spread;
            }
            xy += x * y;
            xx += x * x;
          }
          beta = xy / xx;
          beta = (beta > 0) ? beta : -beta;
        } else if (learn == 3) { //MLE learning
          double t = 0.0, a = 0.0;
          for (TrialData res : results) {
            t += (double)res.trials.size();
            for (trial_type tt : res.trials) a += (double)tt.trial;
          }
          alpha += a;
          beta += t - a;
        }
        else if (learn == 2) { // MLE with alpha = 1
          for (trial_type tt : result.trials) {
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
        model_g_.update_edge_priors(alpha, beta);
      }
      model_g_.update_rounds((double)(stage + 1));

      t2 = get_timestamp();
      updating_time = (t2 - t1) / 60000000.0L;
      round_time = (t2 - t0) / 60000000.0L;
      totaltime += round_time;
      memory = disp_mem_usage();
      double mse = model_g_.get_mse();

      //printing results
      std::cout << stage << "\t" << real << "\t" << expected << "\t" <<
          hits << "\t" << misses << "\t" << totaltime << "\t" << round_time <<
          "\t" << sampling_time << "\t" << choosing_time << "\t" <<
          selecting_time << "\t" << updating_time << "\t" << alpha << "\t" <<
          beta << "\t" << mse << "\t" << (int)cur_theta - THETA_OFFSET - 1 <<
          "\t" << reused_ratio << "\t" << memory << "\t";
      for (auto seed : seeds) std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

class ZScoresStrategy {
 private:
  Graph& model_g;
  Graph& original_g;
  Evaluator& eval;

 public:
  ZScoresStrategy(Graph& model_graph, Graph& original_graph,
                  Evaluator& eval_explore)
      : model_g(model_graph), original_g(original_graph), eval(eval_explore) {}

  void perform(unsigned int budget, unsigned int k, bool update=true,
               unsigned int learn=0) {
    unsigned int cur_theta = THETA_OFFSET;
    SpreadSampler exploit_s(INFLUENCE_MED);
    SpreadSampler test_s(INFLUENCE_MED);
    std::unordered_set<unsigned long> activated;
    boost::mt19937 gen((int)time(0));
    boost::uniform_01<boost::mt19937> dst(gen);
    double expected = 0;
    double real = 0;
    double time_min = 0;
    double beta = 1;
    std::vector<TrialData> results;
    for (unsigned int stage = 0; stage < budget; stage++) {
      timestamp_t t0, t1;
      t0 = get_timestamp();
      PathSampler exploit_p(cur_theta);
      SpreadSampler explore_s(cur_theta);
      //selecting seeds using explore or exploit
      std::unordered_set<unsigned long> seeds;
      seeds = eval.select(model_g, exploit_p, activated, k, 100);
      //evaluating the expected and real spread on the seeds
      double cur_expected = test_s.sample(model_g, activated, seeds, 100);
      expected += cur_expected;
      double cur_real = exploit_s.trial(original_g, activated, seeds);
      real += cur_real;
      //recalibrating
      double err = test_s.get_stdev();
      double stat = (cur_real - cur_expected) / err;
      int theta_est = THETA_OFFSET + (int)(stat/2);
      if (theta_est < THETA_OFFSET - 2) theta_est = THETA_OFFSET - 2;
      if (theta_est > THETA_OFFSET) theta_est = THETA_OFFSET;
      cur_theta = theta_est;
      //updating the model graph
      for (unsigned long node : seeds) activated.insert(node);
      unsigned int hits = 0, misses = 0;
      for (trial_type tt : exploit_s.get_trials()) {
        if (tt.trial == 1) {
          hits++;
          activated.insert(tt.target);
        } else {
          misses++;
        }
        if(update) model_g.update_edge(tt.source, tt.target, tt.trial);
      }
      //TODO learning the graph
      if (learn > 0) {
        TrialData result;
        for (unsigned long seed : seeds) result.seeds.insert(seed);
        result.spread = cur_real;
        for (trial_type tt : exploit_s.get_trials())
          result.trials.push_back(tt);
        results.push_back(result);
        //linear regression learning
        if (learn == 1) {
          double total_spread = 0.0, total_seeds = 0.0;
          for (TrialData res : results) {
            total_spread += res.spread;
            total_seeds += (double) res.seeds.size();
          }
          double avg_spread = total_spread / total_seeds;
          double xy = 0.0, xx = 0.0;
          for (TrialData res : results) {
            double x = res.spread - 1;
            double y = 0.0;
            for (unsigned long seed : res.seeds) {
              double o = 0.0, t = 0.0, h = 0.0;
              for (auto node : model_g.get_neighbours(seed)) {
                o += 1.0;
                t += (double)node.dist->get_hits() +
                    (double) node.dist->get_misses();
                h += (double) node.dist->get_hits();
              }
              y += (t + 1) * x + (o + h) * avg_spread;
            }
            xy += x * y;
            xx += x * x;
          }
          beta = xy / xx;
        } else if (learn == 2) { //MLE learning
          double t = 0.0, a = 0.0;
          for (TrialData res : results) {
            t += (double)res.trials.size();
            for (trial_type tt:res.trials) a += (double)tt.trial;
          }
          beta = (a > 0) ? (t - a) / a : t;
        }
        model_g.update_edge_priors(1.0, beta);
      }
      model_g.update_rounds((double)(stage + 1));
      t1 = get_timestamp();
      //printing results
      time_min += (t1 - t0) / 60000000.0L;
      std::cout << stage << "\t" << real << "\t" << expected <<
          "\t" << hits << "\t" << misses << "\t" << time_min <<
          "\t" << beta << "\t" << model_g.get_mse() << "\t" <<
          (int)cur_theta - THETA_OFFSET - 1 << std::endl << std::flush;
    }
  }
};

#endif /* defined(__oim__Strategy__) */
