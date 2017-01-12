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

#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include "common.hpp"

class Policy {
 protected:
  unsigned int n_experts_;                   // Number of experts

  /**
    Get the `k` largest elements of a vector and returns them as unordered_set.
    Trick with negative weights to get the lowest element of the priority_queue.
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
  Policy(unsigned int n_experts) : n_experts_(n_experts) {}

  virtual std::unordered_set<unsigned int> selectExpert(unsigned int k) = 0;

  /**
    This method does not necessarly need to be overloaded by classes inheriting
    Policy (e.g. RandomPolicy).
  */
  virtual void updateState(unsigned int expert,
                   const std::unordered_set<unsigned long>& stage_spread) {}

  /**
    Reinitialize the object to start parameters.
  */
  virtual void init() {}

  virtual void printdebug() {}
};

/**
  Selects randomly the expert to play at each round.
*/
class RandomPolicy : public Policy {
 private:
  std::mt19937 gen_;
  std::uniform_int_distribution<unsigned int> dst_;
 public:
  RandomPolicy(unsigned int n_experts)
      : Policy(n_experts), gen_(seed_ns()), dst_(0, n_experts_) {}

  std::unordered_set<unsigned int> selectExpert(unsigned int k) {
    std::unordered_set<unsigned int> result;
    while (result.size() < k)
      result.insert(dst_(gen_));
    return result;
  }
};

/**
  Type of spread estimation.
*/
enum Sigma {
  MEAN,       // Replace sigma by the mean of observed spreads
  SAMPLE_STD, // Mean + sampled standard deviation of observed spreads
  INTERSECTING_SUPPORT, // Simple heuristic when intersecting support isn't null
};

/**
  Good-UCB policy for our problem. Each experts maintains a missing mass
  estimator which is used to select the next expert to play. See paper for
  details.
*/
class GoodUcbPolicy : public Policy {
 private:
  std::vector<unsigned long>& nb_neighbours_; // Number of reachable nodes for each expert
  unsigned int t_;                            // Number of rounds played
  std::vector<float> n_plays_;                // Number of times experts were played
  // For each expert, hashmap {node : #activations}
  std::vector<std::unordered_map<unsigned long, unsigned int>> n_rewards_;
  std::vector<std::vector<double>> spreads_;  // List of sampled spreads for each experts
  Sigma sigma_type_;        // Type of estimation of expert expected diffusion

 public:
  GoodUcbPolicy(unsigned int n_experts, std::vector<unsigned long>& nb_neighbours,
                Sigma type=MEAN)
      : Policy(n_experts), nb_neighbours_(nb_neighbours),
        sigma_type_(type) { init(); }

  /**
    TODO Handle this remaining stuff I don't remember why I did that.
  */
  void printdebug() {
    std::unordered_map<unsigned long, std::vector<int>> activations;  // {user: [expert 1, expert 2]}
    for (unsigned int i = 0; i < n_experts_; i++) {
      for (auto& elt : n_rewards_[i]) {
        if (activations.count(elt.first) == 0)
          activations[elt.first] = std::vector<int>();
        for (unsigned int a = 0; a < elt.second; a++)
          activations[elt.first].push_back(i);
      }
    }
    std::cerr << "Number of plays\n===============" << std::endl;
    for (unsigned long i = 0; i < n_experts_; i++)
      std::cerr << i << "\t" << n_plays_[i] << std::endl;
    std::cerr << "\nNumber of activations\n===============" << std::endl;
    for (auto& elt : activations) {
      std::cerr << elt.first << "\t";
      for (auto item : elt.second)
        std::cerr << item << " ";
      std::cerr << std::endl;
    }
  }

  /**
    Selects `k` experts whose Good-UCB indices are the largest.
  */
  std::unordered_set<unsigned int> selectExpert(unsigned int k) {
    // 1. Test if all experts were played at least once
    std::unordered_set<unsigned int> chosen_experts;
    unsigned int n_selected_experts = 0;
    for (unsigned int i = 0; i < n_experts_; i++) {
      if (n_plays_[i] == 0) {
        chosen_experts.insert(i);
        n_selected_experts++;
        if (n_selected_experts == k)
          break;
      }
    }
    if (chosen_experts.size() > 0) {
      for (unsigned int i = 0; i < n_experts_ && chosen_experts.size() < k;
           i++) {
        chosen_experts.insert(i);
      }
      return chosen_experts;
    }
    // 2. If all experts were played once, use missing mass estimator
    std::vector<float> ucbs(n_experts_, 0);
    for (unsigned int i = 0; i < n_experts_; i++) {
      // 2. (a) Compute missing mass estimator
      float missing_mass_i = (float)std::count_if(
          n_rewards_[i].begin(), n_rewards_[i].end(), [](auto& elt) {
            return elt.second == 1; // Count hapaxes
          }) / n_plays_[i];
      // 2. (b) Compute estimator of expected diffusion from this expert
      float sigma = 0;
      for (auto& elt : n_rewards_[i])
        sigma += elt.second;
      sigma /= n_plays_[i];
      if (sigma_type_ == SAMPLE_STD) {  // If we estimate sum of p(x) by the sample mean + std
        double empirical_std = 0;
        for (auto elt : spreads_[i])
          empirical_std += (elt - sigma) * (elt - sigma);
        if (n_plays_[i] == 1)
          empirical_std = sigma;
        else
          empirical_std = sqrt(empirical_std / (n_plays_[i] - 1));
        sigma += empirical_std;
      } else if (sigma_type_ == INTERSECTING_SUPPORT) {
        missing_mass_i = 0;
        for (auto& elt : n_rewards_[i]) {
          if (elt.second != 1)
            continue;
          float degree_experts = 1; // Number of experts which activated this node
          for (unsigned int j = 0; j < n_experts_; j++) {
            if (i == j)
              continue;
            else if (n_rewards_[j].find(elt.first) == n_rewards_[j].end())
              continue;
            else
              degree_experts += 1;
          }
          missing_mass_i += 1 / degree_experts;
        }
        missing_mass_i /= n_plays_[i];
      }
      ucbs[i] = missing_mass_i + (1 + sqrt(2)) * sqrt(sigma * log(4 * t_) /
          n_plays_[i]) + log(4 * t_) / (3 * n_plays_[i]);
    }
    return get_k_largest_arguments<unsigned int>(ucbs, k);
  }

  /**
    Update statistics on the chosen expert (k == 1).
  */
  void updateState(unsigned int expert,
                   const std::unordered_set<unsigned long>& stage_spread) {
    t_++;
    for (auto& activated_node : stage_spread) {
      if (n_rewards_[expert].count(activated_node) == 0)
        n_rewards_[expert][activated_node] = 0;
      n_rewards_[expert][activated_node]++;
    }
    spreads_[expert].push_back(stage_spread.size());
    n_plays_[expert]++;
  }

  /**
    Initialize datastructures required for computing UCB bounds. It is useful
    when restarting the algorithm (to reset).
  */
  void init() {
    t_ = 0;
    n_plays_ = std::vector<float>(n_experts_, 0);
    spreads_ = std::vector<std::vector<double>>(n_experts_);
    for (unsigned int k = 0; k < n_experts_; k++) {
      n_rewards_.push_back(std::unordered_map<unsigned long, unsigned int>());
    }
  }
};
