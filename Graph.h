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

#include "common.h"
#include "InfluenceDistribution.h"

class Graph {
 private:
  std::unordered_map<unsigned long, std::vector<EdgeType>> adj_list;
  std::unordered_map<unsigned long, std::vector<EdgeType>> inv_adj_list;
  std::unordered_set<unsigned long> node_set;
  unsigned long num_edges = 0;
  unsigned long num_nodes = 0;

 public:
  double alpha_prior, beta_prior;

  void set_prior(double alpha, double beta) {
    alpha_prior = alpha;
    beta_prior = beta;
  }

  void add_edge(unsigned long source, unsigned long target,
                std::shared_ptr<InfluenceDistribution> dist) {
    add_node(source);
    add_node(target);
    EdgeType edge1(source, target, dist);
    EdgeType edge2(target, source, dist);
    adj_list[source].push_back(edge1);
    inv_adj_list[target].push_back(edge2);
    num_edges++;
  };

  void add_node(unsigned long node) {
    node_set.insert(node);
    if (node + 1 > num_nodes) num_nodes = node + 1;
  }

  void remove_node(unsigned long node) {
    node_set.erase(node);
  }

  void update_edge(unsigned long src, unsigned long tgt, unsigned int trial) {
    if (adj_list.find(src) != adj_list.end()) {
      for (EdgeType edge : adj_list[src]) {
        if (edge.target == tgt) {
          edge.dist->update(trial, 1.0 - trial);
          break;
        }
      }
    }
  }

  void update_edge_priors(double alpha, double beta) {
    set_prior(alpha, beta);
    for (auto lst : adj_list)
      for (auto edge : lst.second) {
        edge.dist->update_prior(alpha, beta);
      }
  }

  double get_mse(){
    double edges = 0.0;
    double tse = 0.0;
    for (auto lst : adj_list)
      for (auto edge : lst.second) {
        edges += 1.0;
        tse += edge.dist->sq_error();
      }
    return tse / edges;
  }

  void update_rounds(double round) {
    for (auto lst : adj_list)
      for (auto edge : lst.second) {
        edge.dist->set_round(round);
      }
  }

  bool has_neighbours(unsigned long node, bool inv=false) const {
    if (!inv)
      return adj_list.find(node) != adj_list.end();
    else
      return inv_adj_list.find(node) != inv_adj_list.end();
  }

  const std::vector<EdgeType>& get_neighbours(
      unsigned long node, bool inv=false) const {
    if (!inv)
      return (adj_list.find(node))->second;
    else
      return (inv_adj_list.find(node))->second;
  };

  const std::unordered_set<unsigned long>& get_nodes() const {
    return node_set;
  }

  bool has_node(unsigned long node) {
    return node_set.find(node) != node_set.end();
  }

  unsigned long get_number_nodes() const {
    return num_nodes;
  }

  unsigned long get_number_edges() const {
    return num_edges;
  }
};

#endif /* defined(__oim__Graph__) */
