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

#include "InfluenceDistribution.h"
#include "SingleInfluence.h"
#include "Graph.h"

// Load the graph from file and returns the number of nodes
unsigned long load_graph(std::string filename, Graph& graph) {
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
  return edges;
}

#endif /* defined(__oim__graph_utils__) */
