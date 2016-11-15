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

#ifndef __oim__Sampler__
#define __oim__Sampler__

#include "common.h"
#include "Graph.h"

/**
  TODO Description
*/
class Sampler {
 protected:
  unsigned int type_;
  std::vector<trial_type> trials_;

 public:
  Sampler(unsigned int type) : type_(type) {};

  /**
  Method to estimate the standard deviation of TODO
  */
  virtual double sample(const Graph& graph,
                        const std::unordered_set<unsigned long>& activated,
                        const std::unordered_set<unsigned long>& seeds,
                        unsigned long samples) = 0;

  virtual double trial(const Graph& graph,
                       const std::unordered_set<unsigned long>& activated,
                       const std::unordered_set<unsigned long>& seeds,
                       bool inv=false) = 0;

  virtual std::shared_ptr<std::vector<unsigned long>> perform_unique_sample(
      const Graph& graph, std::vector<unsigned long>& nodes_activated,
      std::vector<bool>& bool_activated, const unsigned long source,
      bool inv=false) = 0;

  virtual std::unordered_set<unsigned long> perform_diffusion(
      const Graph& graph, const std::unordered_set<unsigned long>& seeds) = 0;

  std::vector<trial_type>& get_trials() { return trials_; }

  unsigned int get_type() { return type_; }

};

#endif /* defined(__oim__Sampler__) */
