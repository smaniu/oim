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

#ifndef __oim__GraphReduction__
#define __oim__GraphReduction__

#include <unordered_set>
#include "Graph.hpp"
#include "Evaluator.hpp"
#include "SpreadSampler.hpp"

/**
  Abstract class for reducing graphs to experts. Each method must implement this
  class.
*/
class GraphReduction {
 public:
  virtual std::vector<unsigned long> extractExperts(const Graph& graph,
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

 public:
  EvaluatorReduction(double p, Evaluator& evaluator)
      : p_(p), evaluator_(evaluator) {}

  std::vector<unsigned long> extractExperts(const Graph& graph, int n_experts) {
    // 1. Copy graph assigning probability `p_` on every edge
    Graph model_graph;
    for (unsigned long i = 0; i < graph.get_number_nodes(); i++) {
      if (!graph.has_neighbours(i))
        continue;
      for (auto& edge : graph.get_neighbours(i)) {
        std::shared_ptr<InfluenceDistribution> dst(new SingleInfluence(p_));
        model_graph.add_edge(edge.source, edge.target, dst);
      }
    }
    // 2. Select experts using the evaluator
    SpreadSampler sampler(INFLUENCE_MED);
    std::unordered_set<unsigned long> activated;
    auto experts = evaluator_.select(model_graph, sampler, activated, n_experts);
    std::vector<unsigned long> result;
    for (auto expert : experts)
      result.push_back(expert);
    return result;
  }
};

/**
 This method selects `n_experts` nodes with the highest degrees as experts.
*/
class HighestDegreeReduction : public GraphReduction {
 public:
  std::vector<unsigned long> extractExperts(const Graph& graph, int n_experts) {
    std::vector<std::pair<unsigned long, int>> users(graph.get_number_nodes());
    for (auto& node : graph.get_nodes()) {
      users[node].first = node;
      users[node].second = graph.get_neighbours(node).size();
    }
    std::sort(users.begin(), users.end(), [](auto &v1, auto &v2) -> bool {
        return v1.second > v2.second; // Inversed sort
      });
    std::vector<unsigned long> result(n_experts, 0);
    for (int i = 0; i < n_experts; i++)
      result[i] = users[i].first;
    return result;
  }
};

/**
 This method greedily selects n_experts nodes to maximize the cover of the
 graph. Specifically, the algorithm is as follows:
   1. Pick node with highest degree
   2. Remove all neighbours of selected node (to avoid intersecting support)
   3. Restart from 1.
*/
class GreedyMaxCoveringReduction : public GraphReduction {
 public:
  std::vector<unsigned long> extractExperts(const Graph& graph, int n_experts) {
    std::vector<unsigned long> result(n_experts, 0);
    Graph copy_graph(graph); // Copy the graph
    for (int i = 0; i < n_experts; i++) {
      // 1. Pick the node with highest degree
      unsigned long current_node = 0; // Current picked node
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
  int n_iter_;
  double d_ = 0.85;
  double node_error_ = 1e-6;

  /**
    Get the `k` largest elements of a vector and returns them as unordered_set.
    Trick with negative weights to get the lowest element of the priority_queue.
  */
  template<typename T>
  std::vector<T> get_k_largest_arguments(
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
    std::vector<T> result;
    while (!q.empty()) {
      result.push_back(q.top().second);
      q.pop();
    }
    return result;
  }

 public:
  DivRankReduction(double alpha, int n_iter=1000)
      : alpha_(alpha), n_iter_(n_iter) {}

  std::vector<unsigned long> extractExperts(const Graph& graph, int n_experts) {
    unsigned long n = graph.get_number_nodes();
    std::vector<float> pi(n, 1. / n);
    std::vector<float> p_star(n, 1. / n);
    // std::vector<unsigned long> dangling_nodes;
    // W is the transition matrix: for a node u it gives the list of pairs
    // (neighbour, weight)
    std::vector<std::vector<std::pair<unsigned long, float>>> W(n);
    for (unsigned long i = 0; i < n; i++) {
      W[i] = std::vector<std::pair<unsigned long, float>>();
      W[i].push_back(std::make_pair(i, 1 - alpha_));
      if (!graph.has_neighbours(i)) {
        // dangling_nodes.push_back(i);
        continue;
      }
      float n_neighbours = (float)graph.get_neighbours(i).size();
      for (auto& edge : graph.get_neighbours(i))
        W[i].push_back(std::make_pair(edge.target, alpha_ / n_neighbours));
    }
    for (int i = 0; i < n_iter_; i++) {
      std::vector<float> last_pi(pi);
      std::fill(pi.begin(), pi.end(), 0);
      // Dangling nodes last state cumulative probability
      //float cum_dangling = 0;
      //for (auto dn : dangling_nodes)
      //  cum_dangling += last_pi[dn];
      //std::cerr << cum_dangling << std::endl;
      for (unsigned long u = 0; u < n; u++) {
        // Normalization D_t
        float D_t = 0;
        for (auto& p : W[u])  // p = pair (neighbour, weight)
          D_t += p.second * last_pi[p.first]; // weight * last_pi[v]
        for (auto& p : W[u]) {
          pi[p.first] += (d_ * p.second * last_pi[p.first] / D_t) * last_pi[u];
        }
        pi[u] += (/*d_ * cum_dangling +*/ (1 - d_)) * p_star[u];
      }
      // Check convergence
      float err = 0;
      for (unsigned long u = 0; u < n; u++)
        err += abs(pi[u] - last_pi[u]);
      if (err < n * node_error_)
        break;
    }
    return get_k_largest_arguments<unsigned long>(pi, n_experts);
  }
};

#endif /* defined(__oim__GraphReduction__) */
