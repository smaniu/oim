/*
 Copyright (c) 2017 Paul Lagrée

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

#ifndef __oim__LogDiffusion__
#define __oim__LogDiffusion__

#include <unordered_set>
#include <unordered_map>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <math.h>

#include "common.hpp"


/**
  Selects a random cascade from the logs.
*/
class LogDiffusion {
 private:
  boost::mt19937 gen_;
  // Logs of cascades
  std::unordered_map<unode_int, std::vector<std::vector<unode_int>>> cascades_;

 public:
  LogDiffusion() : gen_(seed_ns()) {};

  /**
    Performs the real diffusion from selected seeds.
    It selects randomly cascades from the logs triggered by seeds given in
    parameter.
  */
  std::unordered_set<unode_int> perform_diffusion(
        const std::unordered_set<unode_int>& seeds) {
    std::unordered_set<unode_int> diffusion;
    for (auto& seed : seeds) {
      if (cascades_.find(seed) == cascades_.end())
        continue;
      auto dst = std::uniform_int_distribution<int>(0, cascades_[seed].size() - 1);
      int index = dst(gen_);
      for (auto& node : cascades_[seed][index])
        diffusion.insert(node);
    }
    return diffusion;
  }

  /**
    Load cascades from logs. The input file must be of the form:
      <seed> <TAB> <u1> <TAB> <u2> <TAB> ... <TAB> <un>
  */
  void load_cascades(std::string filename) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      unode_int source;
      iss >> source;
      std::vector<unode_int> cascade;
      do {
        unode_int activated;
        iss >> activated;
        cascade.push_back(activated);
      } while (iss);
      cascades_[source].push_back(cascade);
    }
  }
};

#endif /* defined(__oim__LogDiffusion__) */
