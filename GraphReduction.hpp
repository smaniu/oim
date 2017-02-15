/*
 Copyright (c) 2016-2017 Paul Lagrée

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

#ifndef __oim__GraphReduction__
#define __oim__GraphReduction__

#include <unordered_set>
#include "Graph.hpp"
#include "Evaluator.hpp"
#include "SpreadSampler.hpp"
#include "SingleInfluence.hpp"

/**
  Abstract class for reducing graphs to experts. Each method must implement this
  class.
*/
class GraphReduction {
 public:
  virtual std::vector<unode_int> extractExperts(const Graph& graph,
                                                int n_experts) = 0;
};

/**
  This method selects `n_experts` nodes that maximize the influence on the graph
  for uniform weights.
*/
class EvaluatorReduction : public GraphReduction {
 private:
  double p_;  // transmission probability (same for all edges)
  Evaluator& evaluator_;
  int model_;

 public:
  EvaluatorReduction(double p, Evaluator& evaluator, int model=1)
      : p_(p), evaluator_(evaluator), model_(model) {}

  std::vector<unode_int> extractExperts(
      const Graph& graph, int n_experts) {
    // 1. Copy graph assigning probability `p_` on every edge
    Graph model_graph;
    for (unode_int i = 0; i < graph.get_number_nodes(); i++) {
      if (!graph.has_neighbours(i))
        continue;
      for (auto& edge : graph.get_neighbours(i)) {
        std::shared_ptr<InfluenceDistribution> dst(new SingleInfluence(p_));
        model_graph.add_edge(edge.source, edge.target, dst);
      }
    }
    // 2. Select experts using the evaluator
    SpreadSampler sampler(INFLUENCE_MED, model_);
    std::unordered_set<unode_int> activated;
    auto experts = evaluator_.select(model_graph, sampler, activated, n_experts);
    std::vector<unode_int> result;
    for (auto expert : experts) {
      result.push_back(expert);
    }
    return result;
  }
};

/**
  This method selects `n_experts` nodes with the highest degrees as experts.
*/
class HighestDegreeReduction : public GraphReduction {
 public:
  std::vector<unode_int> extractExperts(const Graph& graph, int n_experts) {
    std::vector<std::pair<unode_int, int>> users(graph.get_number_nodes());
    for (auto& node : graph.get_nodes()) {
      users[node].first = node;
      if (graph.has_neighbours(node))
        users[node].second = graph.get_neighbours(node).size();
      else
        users[node].second = 0;
    }
    std::sort(users.begin(), users.end(), [](auto& v1, auto& v2) -> bool {
        return v1.second > v2.second; // Inversed sort
      });
    std::vector<unode_int> result(n_experts, 0);
    for (int i = 0; i < n_experts; i++)
      result[i] = users[i].first;
    return result;
  }
};

/**
  This method greedily selects n_experts nodes to maximize the cover of the
  model graph. Specifically, the algorithm is as follows:
    1. Pick node with highest degree
    2. Remove all neighbours of selected node (to avoid intersecting support)
    3. Restart from 1.
*/
class GreedyMaxCoveringReduction : public GraphReduction {
 public:
  std::vector<unode_int> extractExperts(const Graph& graph, int n_experts) {
    std::vector<unode_int> result(n_experts, 0);
    Graph copy_graph(graph); // Copy the graph
    for (int i = 0; i < n_experts; i++) {
      // 1. Pick the node with highest degree
      unode_int current_node = 0; // Current picked node
      unsigned int current_value = 0; // Number of neighbours for current node
      for (auto& node : copy_graph.get_nodes()) {
        if (!copy_graph.has_neighbours(node))
          continue;
        unsigned int value = copy_graph.get_neighbours(node).size();
        if (value > current_value) {
          current_value = value;
          current_node = node;
        }
      }
      // Add the node the the result
      result[i] = current_node;
      // 2. Remove all neighbours of chosen node
      copy_graph.remove_node(current_node);
    }
    return result;
  }
};

/**
  DivRank graph reduction. Implementation of `DivRank: the Interplay of Prestige
  and Diversity in Information Networks` by Q. Mei, J. Guo and D. Radev, SIGKDD
  2010.
*/
class DivRankReduction : public GraphReduction {
 private:
  double alpha_;
  double p_;
  int n_iter_;
  double d_ = 0.85;
  double node_error_ = 1e-6;

  /**
    Get the `k` largest elements of a vector and returns them as a vector.
    Trick with negative weights to get the lowest element of the priority_queue.
  */
  template<typename T>
  std::vector<T> get_k_largest_arguments(
        std::vector<double>& vec, unsigned int k) {
    std::priority_queue<std::pair<double, T>> q;
    for (T i = 0; i < k; ++i) {
      q.push(std::pair<double, T>(-vec[i], i));
    }
    for (T i = k; i < vec.size(); ++i) {
      if (q.top().first > -vec[i]) {
        q.pop();
        q.push(std::pair<double, T>(-vec[i], i));
      }
    }
    std::vector<T> result(k, 0);
    for (int i = k - 1; i >= 0; i--) {
      result[i] = q.top().second;
      q.pop();
    }
    return result;
  }

 public:
  DivRankReduction(double alpha, double p=0.05, int n_iter=100)
      : alpha_(alpha), p_(p), n_iter_(n_iter) {}

  std::vector<unode_int> extractExperts(const Graph& graph, int n_experts) {
    // 1. Copy graph assigning probability `p_` on every edge
    Graph model_graph;
    for (unode_int i = 0; i < graph.get_number_nodes(); i++) {
      if (!graph.has_neighbours(i))
        continue;
      for (auto& edge : graph.get_neighbours(i)) {
        std::shared_ptr<InfluenceDistribution> dst(new SingleInfluence(p_));
        model_graph.add_edge(edge.source, edge.target, dst);
      }
    }
    // 2. DivRank on the model graph.
    unode_int n = model_graph.get_number_nodes();
    std::vector<double> pi(n, 1. / n);
    std::vector<double> p_star(n, 1. / n);
    // W is the transition matrix: for a node u it gives the list of pairs
    // (neighbour, weight)
    std::vector<std::vector<std::pair<unode_int, double>>> W(n);
    for (unode_int i = 0; i < n; i++) {
      W[i] = std::vector<std::pair<unode_int, double>>();
      if (!model_graph.has_neighbours(i, true)) {
        W[i].push_back(std::make_pair(i, 1.));
        continue;
      }
      float n_neighbours = (double)model_graph.get_neighbours(i, true).size();
      for (auto& edge : model_graph.get_neighbours(i, true))
        W[i].push_back(std::make_pair(edge.target, alpha_ / n_neighbours));
      W[i].push_back(std::make_pair(i, 1 - alpha_));
    }
    for (int i = 0; i < n_iter_; i++) {
      std::vector<double> last_pi(pi);
      std::fill(pi.begin(), pi.end(), 0);
      for (unode_int u = 0; u < n; u++) {
        // Normalization D_t
        double D_t = 0;
        for (auto& p : W[u])  // p = pair (neighbour, weight)
          D_t += p.second * last_pi[p.first]; // weight * last_pi[v]
        for (auto& p : W[u]) {
          pi[p.first] += (d_ * p.second * last_pi[p.first] / D_t) * last_pi[u];
        }
        pi[u] += (1 - d_) * p_star[u];
      }
    }
    return get_k_largest_arguments<unode_int>(pi, n_experts);
  }
};

#endif /* defined(__oim__GraphReduction__) */
