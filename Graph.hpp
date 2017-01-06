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

#ifndef __oim__Graph__
#define __oim__Graph__

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <algorithm>

#include "common.hpp"
#include "InfluenceDistribution.hpp"
#include <boost/random/mersenne_twister.hpp>


class EdgeType {
 public:
  unsigned long source;
  unsigned long target;
  std::shared_ptr<InfluenceDistribution> dist;
  EdgeType(unsigned long src, unsigned long tgt,
           std::shared_ptr<InfluenceDistribution> dst)
      : source(src), target(tgt), dist(dst) {};
};

/**
  Class representing the graph. Be careful:
    - nodes index starts at 0 up to n - 1
*/
class Graph {
 private:
  std::unordered_map<unsigned long, std::vector<EdgeType>> adj_list_;
  std::unordered_map<unsigned long, std::vector<EdgeType>> inv_adj_list_;
  // For each node, if LT model was activated in graph loading, we have the
  // distribution to sample an incoming edge according to its weight.
  std::unordered_map<
      unsigned long, std::discrete_distribution<>> lt_dist_;
  std::unordered_set<unsigned long> node_set_;
  unsigned long num_edges_ = 0;
  unsigned long num_nodes_ = 0;

 public:
  double alpha_prior, beta_prior;

  Graph() = default;

  Graph(const Graph& g)
      : adj_list_(g.adj_list_), inv_adj_list_(g.inv_adj_list_),
        lt_dist_(g.lt_dist_), node_set_(g.node_set_), num_edges_(g.num_edges_),
        num_nodes_(g.num_nodes_) {}

  Graph(Graph&& g)
      : adj_list_(std::move(g.adj_list_)),
        inv_adj_list_(std::move(g.inv_adj_list_)),
        lt_dist_(std::move(g.lt_dist_)), node_set_(std::move(g.node_set_)),
        num_edges_(std::move(g.num_edges_)), num_nodes_(std::move(g.num_nodes_)) {}

  void set_prior(double alpha, double beta) {
    alpha_prior = alpha;
    beta_prior = beta;
  }

  /**
    Adds an edge and the corresponding inversed edge to the Graph.
  */
  void add_edge(unsigned long source, unsigned long target,
                std::shared_ptr<InfluenceDistribution> dist) {
    add_node(source);
    add_node(target);
    EdgeType edge1(source, target, dist);
    EdgeType edge2(target, source, dist);
    adj_list_[source].push_back(edge1);
    inv_adj_list_[target].push_back(edge2);
    num_edges_++;
  };

  /**
    Adds a node to the Graph.
  */
  void add_node(unsigned long node) {
    node_set_.insert(node);
    num_nodes_ = node_set_.size();
  }

  /**
    Sort edges of the graph such that for each node n, its list of neighbour
    edges is sorted from the lowest to the highest numbered. Used in PMCEvaluator.
  */
  void sort_edges() {
    for (unsigned int i = 0; i < get_number_nodes(); i++) {
      if (!has_neighbours(i))
        continue;
      auto& vec = adj_list_.find(i)->second;
      sort(vec.begin(), vec.end(), [](auto& e1, auto& e2) {
        return (e1.target < e2.target);
      });
    }
  }

  /**
    Remove node from graph, remove all the corresponding data (neighbours,
    appearances in neighours' neighbours).
    TODO Renumber last node to removed node number
  */
  void remove_node(unsigned long node) {
    // 1. Remove node
    node_set_.erase(node);
    num_nodes_ = node_set_.size();
    // 2. Remove real edges from `node`
    if (has_neighbours(node)) {
      std::vector<EdgeType>& neighbours = adj_list_[node];
      for (auto& edge : neighbours) {
        num_edges_--;  // We remove one edge leaving from `node`
        std::vector<EdgeType>& cur_inv_list = inv_adj_list_[edge.target];
        auto it = std::find_if(cur_inv_list.begin(), cur_inv_list.end(),
                               [node](auto& e) { return e.target == node; }); // search for reversed edge
        if (it->target != node) {
          std::cerr << "Corresponding inverted edge node found" << std::endl;
          exit(1);
        }
        cur_inv_list.erase(it);
      }
      adj_list_.erase(node);
    }
    // 3. Remove inversed edges from `node`
    if (has_neighbours(node, true)) {
      std::vector<EdgeType>& inv_neighbours = inv_adj_list_[node];
      for (auto& inv_edge : inv_neighbours) {
        num_edges_--;  // We remove one edge pointing to `node` (reversed points to `target`)
        std::vector<EdgeType>& cur_list = adj_list_[inv_edge.target];
        auto it = std::find_if(cur_list.begin(), cur_list.end(),
                               [node](auto& e) { return e.target == node; });
        if (it->target != node) {
          std::cerr << "Corresponding edge node found" << std::endl;
          exit(1);
        }
        cur_list.erase(it);
      }
      inv_adj_list_.erase(node);
    }
  }

  void update_edge(unsigned long src, unsigned long tgt, unsigned int trial) {
    if (adj_list_.find(src) != adj_list_.end()) {
      for (EdgeType edge : adj_list_[src]) {
        if (edge.target == tgt) {
          edge.dist->update(trial, 1.0 - trial);
          break;
        }
      }
    }
  }

  void update_edge_priors(double alpha, double beta) {
    set_prior(alpha, beta);
    for (auto lst : adj_list_)
      for (auto edge : lst.second) {
        edge.dist->update_prior(alpha, beta);
      }
  }

  double get_mse(){
    double edges = 0.0;
    double tse = 0.0;
    for (auto lst : adj_list_)
      for (auto edge : lst.second) {
        edges += 1.0;
        tse += edge.dist->sq_error();
      }
    return tse / edges;
  }

  void update_rounds(double round) {
    for (auto lst : adj_list_) {
      for (auto edge : lst.second) {
        edge.dist->set_round(round);
      }
    }
  }

  bool has_neighbours(unsigned long node, bool inv=false) const {
    if (!inv)
      return adj_list_.find(node) != adj_list_.end();
    else
      return inv_adj_list_.find(node) != inv_adj_list_.end();
  }

  /**
    Get the list of neighbours for the `node` given in parameter. To obtain the
    reversed neighbors for TIM-like algorithms, set inv to `true`.
  */
  const std::vector<EdgeType>& get_neighbours(
      unsigned long node, bool inv=false) const {
    if (!inv)
      return (adj_list_.find(node))->second;
    else
      return (inv_adj_list_.find(node))->second;
  };

  /**
    Get the set of nodes.
  */
  const std::unordered_set<unsigned long>& get_nodes() const {
    return node_set_;
  }

  /**
    Test if a node is in the graph.
  */
  bool has_node(unsigned long node) {
    return node_set_.find(node) != node_set_.end();
  }

  /**
    Get number of nodes in the graph.
  */
  unsigned long get_number_nodes() const {
    return num_nodes_;
  }

  /**
    Get number of edges in the graph.
  */
  unsigned long get_number_edges() const {
    return num_edges_;
  }

  /**
    Build the distribution on each node to sample incoming living edges for the
    LT model. It requires the graph has been entirely loaded. This method can
    also be called after an update of
  */
  void build_lt_distribution(unsigned int type) {
    for (unsigned long u = 0; u < num_nodes_; u++) {
      if (has_neighbours(u, true)) {  // Only reversed edges are interesting
        auto& neighbours = get_neighbours(u, true);
        std::vector<double> w(neighbours.size() + 1, 0);
        double total = 0;
        for (unsigned int i = 0; i < neighbours.size(); i++) {
          double cur_weight = neighbours[i].dist->sample(type);
          total += cur_weight;
          w[i] = cur_weight;
        }
        if (total < 1) {  // Weights do not sum to 1, we can sample no edge
          w[neighbours.size()] = 1 - total;
        }
        // Note, the UCB-like case is handled automatically by discrete_distribution
        lt_dist_[u] = std::discrete_distribution<>(w.begin(), w.end());
      }
    }
  }

  /**
    Sample a living edge from a given node. This is used for LT algorithms.
    Be careful, it returns the INDEX of the chosen node in the list of inverted
    neighbours of `node`. If weights do not sum to 1, we can return -1 for no
    sample.
  */
  int sample_living_edge(unsigned long node, boost::mt19937& gen) {
    if (has_neighbours(node, true)) {
      int index = lt_dist_[node](gen);
      if (index < (int)get_neighbours(node, true).size())
        return index;
    }
    return -1;
  }

  /**
    Display graph edges for debug purposes.
  */
  void write_err(int type) {
    for (unsigned long i = 0; i < get_number_nodes(); i++) {
      if (!has_neighbours(i))
        continue;
      for (auto& edge : get_neighbours(i))
        std::cerr << edge.source << "\t" << edge.target << "\t" << edge.dist->sample(type) << std::endl;
    }
  }
};

#endif /* defined(__oim__Graph__) */
