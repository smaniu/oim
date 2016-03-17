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

#ifndef oim_samplemanager_h
#define oim_samplemanager_h

#include "common.h"
#include "Sampler.h"
#include "SpreadSampler.h"

#include "InfluenceDistribution.h"

#define INCREMENTAL_THRESHOLD 0.02

using namespace std;

struct sample_type{
  shared_ptr<vector<unsigned long> > sample;
  int age; // this sample is generated at trial #age
  int lastUsedTrial;
  double alpha, beta; // this sample is generated under prior (alpha, beta)
};

class SampleManager{
private:
  std::random_device rd;
  std::mt19937 gen;

  const Graph& graph;
  vector<sample_type> sample_pool;
  int pointer;

  SampleManager(const Graph& graph) : graph(graph), gen(rd()) {
    sample_pool.clear();
    sample_pool.reserve(MAX_R);
    pointer = -1;
  }

  ~SampleManager() {
    if (instance) delete instance;
    if (exploreInstance) delete exploreInstance;
  }


  static SampleManager* instance;
  static SampleManager* exploreInstance;
  
  static int currentTrial;
  static bool explore;
  static vector<int> node_age;
  static double hit, miss, case1, case2, case3;

public:

  static void setInstance(const Graph& graph) {
    if (instance) delete instance;
    if (exploreInstance) delete exploreInstance;
    instance = new SampleManager(graph);
    exploreInstance = new SampleManager(graph);

    node_age = vector<int>(graph.get_number_nodes(), -1);
    currentTrial = -1;
    explore = false;
  }

  static SampleManager* getInstance() {
    return explore ? exploreInstance : instance;
  }

  static void reset(int trial, bool _explore = false) {
    hit = 0, miss = 0;
    case1 = 0, case2 = 0, case3 = 0;
/*    cerr << "lastTrial: " << currentTrial << "\t" << "currentTrial: " << trial << endl;
    if (instance->sample_pool.size()) {
      cerr << "lastUsedTrial: " << instance->sample_pool[0].lastUsedTrial << endl;
      cerr << "(a, b) = "  << instance->sample_pool[0].alpha << "\t" << instance->sample_pool[0].beta << endl;
    }
*/
    currentTrial = trial;
    explore = _explore;
    if (explore && exploreInstance->sample_pool.size()) {
      exploreInstance->pointer = -1;
      //exploreInstance->pointer = rand() % ((int)exploreInstance->sample_pool.size());
    } 
    if (!explore && instance->sample_pool.size()) {
      instance->pointer = -1;
      //instance->pointer = rand() % ((int)instance->sample_pool.size());
    }
  }

  static void update_node_age(const unordered_set<unsigned long>& trials) {
    if (hit + miss > 0.1) {
      reused_ratio = (hit) / (hit + miss);
      //cerr << "ratio of reused samples in last trial " << (hit) / (hit + miss) << endl;
    }
    /*
    cerr << "miss profile: " << case1 << "\t" << case2 << "\t" << case3 << endl;

    cerr << "number of nodes to be updated: " << trials.size() << endl;*/

    for (auto node:trials) {
      node_age[(int)node] = currentTrial;
    }
  }

  double ABS(double x) {
    return x < -1e-9 ? -x : x;
  }

  bool isAccepted(double alpha, double beta) {
    return ABS(alpha / (alpha + beta) - graph.alpha_prior / (graph.alpha_prior + graph.beta_prior)) <= INCREMENTAL_THRESHOLD;
  }

  // hardcoded for reverse set
  shared_ptr<vector<unsigned long> > getSample(const vector<unsigned long>& graph_nodes,\
                                   Sampler& sampler,\
                                   const unordered_set<unsigned long>& activated,\
                                   std::uniform_int_distribution<int>& dst) 
  {
    bool goodSampleFlag = true;
    // check whether there is sample that can be reused
    do {
      if (sample_pool.size() == 0) break;
      sample_type& sample = sample_pool[(pointer + 1) % (int)sample_pool.size()];
      if (sample.lastUsedTrial >= currentTrial) {
        case1 += 1;
        break;
      }

      if (!isAccepted(sample.alpha, sample.beta)) {
        goodSampleFlag = false;
        case2 += 1;
        break;
      }
      for (auto node:*(sample.sample)) {
        if (node_age[node] >= sample.age) {
          goodSampleFlag = false;
          case3 += 1;
          break;
        }
      }
      if (!goodSampleFlag) break;

      hit += 1;
      sample.lastUsedTrial = currentTrial;
      pointer++;
      return sample.sample;
    } while (false);

    miss += 1;
    std::unordered_set<unsigned long> seeds;
    unsigned long nd = graph_nodes[dst(gen)];
    seeds.insert(nd);
    sampler.trial(graph, activated, seeds, true);
    
    shared_ptr<vector<unsigned long> >sample (new vector<unsigned long>());
    sample->push_back(nd);
    for(trial_type tt:sampler.get_trials()){
      if(tt.trial==1){
        sample->push_back(tt.target);
      }
    }

    if (!goodSampleFlag || (int)sample_pool.size() >= MAX_R) {
      pointer = (pointer + 1) % (int)sample_pool.size();
    } else {
      pointer = sample_pool.size();
      sample_pool.push_back(sample_type());
    }
    sample_type& new_sample = sample_pool[pointer];
    new_sample.alpha = graph.alpha_prior;
    new_sample.beta = graph.beta_prior;
    new_sample.sample = sample;
    new_sample.age = currentTrial;
    new_sample.lastUsedTrial = currentTrial;
    return new_sample.sample;
  } 
};

SampleManager* SampleManager::instance = NULL;
SampleManager* SampleManager::exploreInstance = NULL;
int SampleManager::currentTrial = -1;
bool SampleManager::explore = false;
vector<int> SampleManager::node_age = vector<int>();
double SampleManager::hit = 0;
double SampleManager::miss = 0;
double SampleManager::case1 = 0;
double SampleManager::case2 = 0;
double SampleManager::case3 = 0;


#endif
