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
  TODO DivRank graph reduction.
*/
class DivRankReduction : public GraphReduction {
 public:
  std::vector<unsigned long> extractExperts(const Graph& graph, int n_experts) {
    std::vector<unsigned long> result(n_experts, 0);
    return result;
  }
};

#endif /* defined(__oim__GraphReduction__) */
