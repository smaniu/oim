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

#ifndef __oim__HighestDegreeEvaluator__
#define __oim__HighestDegreeEvaluator__

#include "common.hpp"
#include "Evaluator.hpp"

#include <boost/heap/fibonacci_heap.hpp>

class HighestDegreeEvaluator : public Evaluator {
 private:
  /**
   Get the `k` largest elements of a vector and returns them as unordered_set.
   Trick with negative weights to get the lowest element of the priority_queue.
  */
  template<typename T>
  std::unordered_set<T> get_k_largest_arguments(
       std::vector<unsigned int>& vec, unsigned int k) {
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
  std::unordered_set<unode_int> select(
        const Graph& graph, Sampler&,
        const std::unordered_set<unode_int>& activated, unsigned int k) {
    std::vector<unsigned int> current_degree(graph.get_number_nodes(), 0);
    for (unode_int u = 0; u < graph.get_number_nodes(); u++) {
      if (!graph.has_neighbours(u))
        continue;
      unsigned int cur_u_deg = 0;
      for (auto& edge : graph.get_neighbours(u)) {
        if (activated.find(edge.target) == activated.end()) {
          cur_u_deg++;
        }
      }
      current_degree[u] = cur_u_deg;
    }
    return get_k_largest_arguments<unode_int>(current_degree, k);
  }
};

#endif /* defined(__oim__HighestDegreeEvaluator__) */
