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

#ifndef __oim__graph_utils__
#define __oim__graph_utils__

#include <iostream>
#include <fstream>

#include "InfluenceDistribution.hpp"
#include "SingleInfluence.hpp"
#include "BetaInfluence.hpp"
#include "Graph.hpp"


// Load the graph from file and returns the number of nodes
unsigned long load_original_graph(
      std::string filename, Graph& graph, int model=1) {
  std::ifstream file(filename);
  unsigned long src, tgt;
  double prob;
  unsigned long edges = 0;
  while (file >> src >> tgt >> prob) {
    std::shared_ptr<InfluenceDistribution> dst_original(
        new SingleInfluence(prob));
    graph.add_edge(src, tgt, dst_original);
    edges++;
  }
  if (model == 0) // If LT model, we need to create distributions for each nodes
    graph.build_lt_distribution(INFLUENCE_MED);
  return edges;
}

/**
  Load the graph from file in two Graph objects: (a) original graph (the *real*
  graph) (b) model graph (graph estimation).
  Returns the number of nodes.
*/
unsigned long load_model_and_original_graph(
      std::string filename, double alpha, double beta,
      Graph& original_graph, Graph& model_graph, int model=1) {
  std::ifstream file(filename);
  unsigned long src, tgt;
  double prob;
  unsigned long edges = 0;
  while (file >> src >> tgt >> prob) {
    std::shared_ptr<InfluenceDistribution> dst_original(
        new SingleInfluence(prob));
    std::shared_ptr<InfluenceDistribution> dst_model(
        new BetaInfluence(alpha, beta, prob));
    original_graph.add_edge(src, tgt, dst_original);
    model_graph.add_edge(src, tgt, dst_model);
    if (model == 0) { // If LT model, we need to create distributions for each nodes
      original_graph.build_lt_distribution(INFLUENCE_MED);
      model_graph.build_lt_distribution(INFLUENCE_MED);
    }
    edges++;
  }
  model_graph.set_prior(alpha, beta);
  return edges;
}

#endif /* defined(__oim__graph_utils__) */
