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

#include "common.hpp"
#include "Evaluator.hpp"
#include "SpreadSampler.hpp"
#include "PathSampler.hpp"
#include "Graph.hpp"
#include "SampleManager.hpp"
#include "GraphReduction.hpp"
#include "Policy.hpp"

#include <iostream>
#include <unordered_set>
#include <sys/time.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <limits>


double sampling_time = 0;
double choosing_time = 0;
double selecting_time = 0;
double updating_time = 0;
double reused_ratio = 0;

struct TrialData {
  std::unordered_set<unsigned long> seeds;
  double spread;
  std::vector<TrialType> trials;
};

/**
  Abstract class which is implemented by all strategies.
*/
class Strategy {
 protected:
  Graph& original_graph_;  // The real graph (unknown from the decision maker)
  boost::mt19937 gen_;
  boost::uniform_01<boost::mt19937> dist_;

 public:
  Strategy(Graph &original_graph)
      : original_graph_(original_graph), gen_(seed_ns()), dist_(gen_) {}

  virtual void perform(unsigned int budget, unsigned int k) = 0;
};

/**
  Strategy using the influence maximization algorithm (Evaluator) on the *known*
  graph. This strategy isn't an online version that doesn't know the original
  graph, but only a baseline.
*/
class OriginalGraphStrategy : public Strategy {
 private:
  Evaluator& evaluator_;
  unsigned long samples_;
  bool incremental_;

 public:
  OriginalGraphStrategy(Graph& original_graph, Evaluator& evaluator,
                        double n_samples, bool incremental=0)
      : Strategy(original_graph), evaluator_(evaluator),
        samples_(n_samples), incremental_(incremental) {}

  void perform(unsigned int budget, unsigned int k) {
    SpreadSampler sampler(INFLUENCE_MED);
    std::unordered_set<unsigned long> activated;
    double expected = 0, real = 0, time_min = 0;
    for (unsigned int stage = 0; stage < budget; stage++) {
      timestamp_t t0, t1;
      t0 = get_timestamp();

      if (incremental_)
        SampleManager::reset(stage);

      // Select seeds using explore or exploit
      std::unordered_set<unsigned long> seeds =
          evaluator_.select(original_graph_, sampler, activated, k);
      //evaluating the expected and real spread on the seeds
      expected += sampler.sample(original_graph_, activated, seeds, samples_);
      real += sampler.trial(original_graph_, activated, seeds);

      //updating the model graph
      std::unordered_set<unsigned long> nodes_to_update;
      for (unsigned long node : seeds) {
        activated.insert(node);
        nodes_to_update.insert(node);
      }
      unsigned int hits = 0, misses = 0;
      for (TrialType tt : sampler.get_trials()) {
        nodes_to_update.insert(tt.target);
        if (tt.trial == 1) {
          hits++;
          activated.insert(tt.target);
        } else {
          misses++;
        }
      }

      if (incremental_)
        SampleManager::update_node_age(nodes_to_update);

      t1 = get_timestamp();
      // Printing results
      time_min += (double)(t1 - t0) / 1000000;
      std::cout << stage << "\t" << real << "\t" << expected << "\t" << hits
                << "\t" << misses << "\t" << time_min << "\t";
      for (auto seed : seeds)
        std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

/**
  Strategy using the missing mass estimator to sequentially select the best k
  experts. See paper : TODO description

  Output: on standard output
    stage <TAB> realspread <TAB> totaltime <TAB> roundtime <TAB> selectingtime
          <TAB> updatingtime <TAB> memory <TAB> seed1.seed2.....seedN
*/
class MissingMassStrategy : public Strategy {
 private:
  GraphReduction& g_reduction_;
  int n_experts_;
  unsigned int n_policy_;

 public:
  /**
    Give the graph, the reduction method with the number of experts.
    The last argument refers to the Policy employed (must be chosen among
    {RandomPolicy, GoodUCBPolicy}).
  */
  MissingMassStrategy(Graph& original_graph, GraphReduction& g_reduction,
                      int n_experts, unsigned int n_policy=1)
      : Strategy(original_graph), g_reduction_(g_reduction),
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
    SpreadSampler exploit_spread(INFLUENCE_MED);
    double totaltime = 0, roundtime = 0, memory = 0,
        updatingtime = 0, selectingtime = 0, reductiontime;
    std::unordered_set<unsigned long> total_spread;

    // 1. (a) Extract experts from graph
    timestamp_t t0, t1;
    t0 = get_timestamp();
    std::vector<unsigned long> experts = g_reduction_.extractExperts(
        original_graph_, n_experts_); // So far, we do not give children of experts
    t1 = get_timestamp();
    reductiontime = (double)(t1 - t0) / 1000000;

    // 1. (b) Create the right policy object
    vector<unsigned long> nb_neighbours(
        n_experts_, original_graph_.get_number_nodes());
    std::unique_ptr<Policy> policy;
    if (n_policy_ == 0)
      policy = std::unique_ptr<Policy>(new RandomPolicy(n_experts_));
    else if (n_policy_ == 1) {
      policy = std::unique_ptr<Policy>(
          new GoodUcbPolicy(n_experts_, nb_neighbours));
    }

    // 2. Sequentially select the best k nodes from missing mass estimator ucb
    std::unordered_set<unsigned long> spread;
    for (unsigned int stage = 0; stage < budget; stage++) {
      policy->init();
      // 2. (a) Select k experts for this round
      timestamp_t t2;
      t0 = get_timestamp();
      std::unordered_set<unsigned int> chosen_experts = policy->selectExpert(k);
      t1 = get_timestamp();
      selectingtime = (double)(t1 - t0) / 1000000;

      // 2. (b) Apply diffusion
      std::unordered_set<unsigned long> seeds;
      for (unsigned int chosen_expert : chosen_experts) {
        seeds.insert(experts[chosen_expert]); // We add the associated node
      }
      auto stage_spread = exploit_spread.perform_diffusion(
          original_graph_, seeds);
      total_spread.insert(stage_spread.begin(), stage_spread.end());

      // 3. (c) Update statistics of experts
      for (unsigned long expert : chosen_experts) {
        policy->updateState(expert, stage_spread);
      }
      t2 = get_timestamp();
      updatingtime = (double)(t2 - t1) / 1000000.;
      roundtime = (double)(t2 - t0) / 1000000;
      totaltime += roundtime;
      memory = disp_mem_usage();

      // 4. Printing results
      std::cout << stage << "\t" << total_spread.size() << '\t'
                << reductiontime << "\t" << totaltime << "\t"
                << roundtime << "\t" << selectingtime << "\t"
                << updatingtime << "\t" << memory << "\t";
      for (auto seed : seeds)
        std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

/**
  TODO description + migrate as Strategy implementation
*/
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
        seeds = explore_e.select(model_g, explore_p, activated, k);
      } else {
        exploit_e.setIncremental(INCREMENTAL);
        seeds = exploit_e.select(model_g, exploit_p, activated, k);
      }
      // evaluating the expected and real spread on the seeds
      expected += exploit_s.sample(original_g, activated, seeds, samples);
      double cur_real = exploit_s.trial(original_g, activated, seeds);
      real += cur_real;

      // updating the model graph
      std::unordered_set<unsigned long> nodes_to_update;

      for (unsigned long node : seeds) {
        activated.insert(node);
        nodes_to_update.insert(node);
      }
      unsigned int hits = 0, misses = 0;
      for (TrialType tt : exploit_s.get_trials()) {
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
        for (TrialType tt:exploit_s.get_trials()) result.trials.push_back(tt);
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
            for (TrialType tt : res.trials)
              a += (double)tt.trial;
          }
          alpha += a;
          beta += t - a;
          //beta = a>0?(t-a)/a:t;
        } else if (learn == 2) { // MLE with alpha = 1
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
        model_g.update_edge_priors(alpha, beta);
      }
      model_g.update_rounds(1.0);
      t1 = get_timestamp();
      // Printing results
      time_min += (t1 - t0) / 60000000.0L;
      std::cout << stage << "\t" << real << "\t" << expected << "\t" <<
          hits << "\t" << misses << "\t" << time_min << "\t" << alpha <<
          "\t" << beta << "\t" << model_g.get_mse() << "\t";
      for (auto seed : seeds) std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
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
  bool incremental_;
  bool update_;
  unsigned int learn_;

 public:
  ExponentiatedGradientStrategy(Graph& model_graph, Graph& original_graph,
                                Evaluator& evaluator, bool incremental,
                                bool update=true, unsigned int learn=0)
      : Strategy(original_graph), model_graph_(model_graph),
        evaluator_(evaluator), incremental_(incremental), update_(update),
        learn_(learn) {}

  void perform(unsigned int budget, unsigned int k) {
    std::vector<double> p(3, 0.333);
    double w[3] = {1.0, 1.0, 1.0};
    unsigned int cur_theta = THETA_OFFSET;
    double mu = log(300.0) / (3 * budget);
    double tau = 12 * mu / (3.0 + mu);
    double lambda = tau / 6.0;
    SpreadSampler exploit_sampler(INFLUENCE_MED); // Sampler for *real* graph
    std::unordered_set<unsigned long> activated;
    double expected = 0, real = 0, totaltime = 0, roundtime = 0, memory = 0;
    double alpha = 1, beta = 1;
    std::vector<TrialData> results;
    std::unordered_map<long long, int> edge_hit, edge_miss;

    for (unsigned int stage = 0; stage < budget; stage++) {
      if (incremental_)
        SampleManager::reset(stage);

      timestamp_t t0, t1, t2;
      t0 = get_timestamp();
      // sampling the distribution
      std::discrete_distribution<int> prob(p.begin(), p.end());
      cur_theta = prob(gen_) + THETA_OFFSET;
      std::cerr << "type_ == " << cur_theta << std::endl;

      // PathSampler path_sampler(cur_theta); (version with path sampler, not used anymore)
      SpreadSampler explore_sampler(cur_theta);
      evaluator_.setIncremental(incremental_);

      // Selecting seeds using explore or exploit
      std::unordered_set<unsigned long> seeds;
      // seeds = evaluator_.select(model_graph_, path_sampler, activated, k, 100); (version with path sampler, not used anymore)
      seeds = evaluator_.select(model_graph_, explore_sampler, activated, k);
      // Evaluating the expected and real spread on the seeds
      double cur_expected = explore_sampler.sample(
            model_graph_, activated, seeds, 100);
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
      selecting_time = (double)(t1 - t0) / 1000000;

      std::unordered_set<unsigned long> nodes_to_update;
      for (unsigned long node : seeds) {
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
      if (incremental_)
        SampleManager::update_node_age(nodes_to_update);

      // TODO learning the graph
      if (learn_ > 0) {
        TrialData result;
        for (unsigned long seed : seeds)
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
            for (unsigned long seed : res.seeds) {
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
      updating_time = (double)(t2 - t1) / 1000000;
      roundtime = (double)(t2 - t0) / 1000000;
      totaltime += roundtime;
      memory = disp_mem_usage();
      //double mse = model_graph_.get_mse();

      // Printing results
      std::cout << stage << "\t" << real << "\t" << expected << "\t" <<
          /*hits << "\t" << misses << "\t" <<*/ totaltime << "\t" << roundtime <</*
          "\t" << sampling_time << "\t" << choosing_time <<*/ "\t" <<
          selecting_time << "\t" << updating_time <</* "\t" << alpha << "\t" <<
          beta << "\t" << mse <<*/ "\t" << (int)cur_theta - THETA_OFFSET - 1 <<
          "\t" << /*reused_ratio << "\t" <<*/ memory << "\t";
      for (auto seed : seeds)
        std::cout << seed << ".";
      std::cout << std::endl << std::flush;
    }
  }
};

#endif /* defined(__oim__Strategy__) */
