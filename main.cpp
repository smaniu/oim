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

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <memory>
#include <time.h>

#include "common.hpp"
#include "graph_utils.hpp"
#include "Graph.hpp"
#include "InfluenceDistribution.hpp"
#include "SingleInfluence.hpp"
#include "BetaInfluence.hpp"
#include "SpreadSampler.hpp"
#include "CELFEvaluator.hpp"
#include "TIMEvaluator.hpp"
#include "SSAEvaluator.hpp"
#include "RandomEvaluator.hpp"
#include "HighestDegreeEvaluator.hpp"
#include "DiscountDegreeEvaluator.hpp"
#include "PMCEvaluator.hpp"
#include "Strategy.hpp"

using namespace std;

/**
  Function performing diffusion with *known* graph. The seeds are selected with
  one of the Evaluators.

  Ex. usage: ./oim --real graph.txt 5 20 2
*/
void real(int argc, const char * argv[],
          std::vector<std::unique_ptr<Evaluator>>& evaluators) {
  if (argc < 6) {
    std::cerr << "Wrong number of arguments.\n\tUsage ./oim --real <graph> "
              << "<exploit> <budget> <k> [<samples>]" << std::endl;
    exit(1);
  }
  Graph original_graph;
  load_original_graph(argv[2], original_graph);
  SampleManager::setInstance(original_graph);
  original_graph.set_prior(1.0, 1.0); // TODO Why ?
  unsigned int exploit = atoi(argv[3]);
  unsigned int budget = atoi(argv[4]);
  unsigned int k = atoi(argv[5]);
  int samples = 100;
  int inc = 0;
  if (argc > 6)
    inc = atoi(argv[6]);
  if (argc > 7)
    samples = atoi(argv[7]);
  OriginalGraphStrategy strategy(original_graph, *evaluators.at(exploit),
                                 samples, inc);
  strategy.perform(budget, k);
}

void prior(int argc, const char * argv[]) {
  std::string file_name_graph(argv[2]);
  double alpha = atof(argv[3]);
  double beta = atof(argv[4]);
  std::ifstream file(file_name_graph);
  Graph original_graph;
  unsigned long src, tgt;
  double prob;
  unsigned long edges = 0;
  while (file >> src >> tgt >> prob) {
    std::shared_ptr<InfluenceDistribution>\
      dst_model(new BetaInfluence(alpha, beta, prob));
    original_graph.add_edge(src, tgt, dst_model);
    edges++;
  }
  unsigned int exploit = atoi(argv[5]);
  unsigned int budget = atoi(argv[6]);
  unsigned int k = atoi(argv[7]);
  std::vector<std::unique_ptr<Evaluator>> evals;
  evals.push_back(std::unique_ptr<Evaluator>(new CELFEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new RandomEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new DiscountDegreeEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new TIMEvaluator()));
  SpreadSampler s_exploit(INFLUENCE_MED);
  int samples = 100;
  if (argc > 9)
    samples = atoi(argv[9]);
  OriginalGraphStrategy strategy(original_graph, *evals.at(exploit), samples);
  strategy.perform(budget, k);
}

void explore(int argc, const char * argv[]){
  std::string file_name_graph(argv[2]);
  double alpha = atof(argv[3]);
  double beta = atof(argv[4]);
  std::ifstream file(file_name_graph);
  Graph original_graph, model_graph;
  unsigned long src, tgt;
  double prob;
  unsigned long edges = 0;
  while (file >> src >> tgt >> prob) {
    shared_ptr<InfluenceDistribution> dst_original(new SingleInfluence(prob));
    shared_ptr<InfluenceDistribution> dst_model(
        new BetaInfluence(alpha, beta, prob));
    original_graph.add_edge(src, tgt, dst_original);
    model_graph.add_edge(src, tgt, dst_model);
    edges++;
  }

  SampleManager::setInstance(model_graph);
  model_graph.set_prior(alpha, beta);
  model_graph.update_rounds(alpha+beta);

  unsigned int explore = atoi(argv[5]);
  unsigned int budget = atoi(argv[6]);
  unsigned int k = atoi(argv[7]);
  double eps = 1.0;
  std::vector<std::unique_ptr<Evaluator>> evals;
  evals.push_back(std::unique_ptr<Evaluator>(new RandomEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new DiscountDegreeEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new CELFEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new TIMEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new HighestDegreeEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new SSAEvaluator(0.1)));
  int samples = 1000;
  bool update = true;
  unsigned int learn = 0;
  unsigned int int_explore = INFLUENCE_MED;
  int inc = 0;
  if(argc>8){
    int orig_explore = atoi(argv[8]);
    if(orig_explore>0) int_explore = orig_explore+2;
  }
  if(argc>9)
    learn = atoi(argv[9]);
  EpsilonGreedyStrategy strategy(model_graph, original_graph,
                                 *evals.at(explore), *evals.at(explore),
                                 samples, eps, inc);
  strategy.perform(budget, k, update, learn, int_explore, int_explore);
}

void epsgreedy(int argc, const char * argv[]){
  std::string file_name_graph(argv[2]);
  double alpha = atof(argv[3]);
  double beta = atof(argv[4]);
  std::ifstream file(file_name_graph);
  Graph original_graph, model_graph;
  unsigned long src, tgt;
  double prob;
  unsigned long edges = 0;
  while (file >> src >> tgt >> prob) {
    std::shared_ptr<InfluenceDistribution> dst_original(new SingleInfluence(prob));
    std::shared_ptr<InfluenceDistribution> dst_model(new BetaInfluence(alpha, beta, prob));
    original_graph.add_edge(src, tgt, dst_original);
    model_graph.add_edge(src, tgt, dst_model);
    edges++;
  }

  SampleManager::setInstance(model_graph);
  model_graph.set_prior(alpha, beta);

  unsigned int exploit = atoi(argv[5]);
  unsigned int explore = atoi(argv[6]);
  unsigned int budget = atoi(argv[7]);
  unsigned int k = atoi(argv[8]);
  double eps = atof(argv[9]);
  std::vector<std::unique_ptr<Evaluator>> evals;
  evals.push_back(std::unique_ptr<Evaluator>(new CELFEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new RandomEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new DiscountDegreeEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new TIMEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new HighestDegreeEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new SSAEvaluator(0.1)));
  int samples = 1000;
  bool update = true;
  unsigned int learn = 0;
  if (argc > 10) {
    unsigned int upd = atoi(argv[10]);
    update = (upd == 1) ? true : false;
  }
  unsigned int int_explore = INFLUENCE_MED;
  unsigned int int_exploit = INFLUENCE_MED;
  int inc = 0;
  if (argc > 11)
    learn = atoi(argv[11]);
  if (argc > 12)
    int_exploit = atoi(argv[12]);
  if (argc > 13)
    int_explore = atoi(argv[13]);
  if (argc > 14)
    inc = atoi(argv[14]);
  if (argc > 15)
    samples = atoi(argv[15]);
  EpsilonGreedyStrategy strategy(model_graph, original_graph,
                                 *evals.at(explore), *evals.at(exploit),
                                 samples, eps, inc);
  strategy.perform(budget, k, update, learn, int_exploit, int_explore);
}

/**
  TODO description

  Ex. usage: ./oim --eg graph.txt 1 20 5 20 2
*/
void expgr(int argc, const char * argv[],
           std::vector<std::unique_ptr<Evaluator>>& evaluators) {
  if (argc < 8) {
    std::cerr << "Wrong number of arguments.\n\tUsage ./oim --eg <graph> "
              << "<alpha> <beta> <exploit> <trials> <k> [<update> "
              << "<update_type> <samples>]" << std::endl;
    exit(1);
  }
  // Take parameters
  bool update = true;
  unsigned int learn = 0;
  int inc = 0;
  double alpha = atof(argv[3]);
  double beta = atof(argv[4]);
  unsigned int exploit = atoi(argv[5]);
  if (exploit > 6) {
    std::cerr << "Error: <exploit> must be in range 0..6" << std::endl;
    exit(1);
  }
  unsigned int budget = atoi(argv[6]);
  unsigned int k = atoi(argv[7]);
  if (argc > 8)
    update = (atoi(argv[8]) == 1) ? true : false;
  if (argc > 9)
    learn = atoi(argv[9]);
  if (argc > 10)
    inc = atoi(argv[10]);
  // Load model and original graphs
  Graph original_graph, model_graph;
  load_model_and_original_graph(argv[2], alpha, beta,
                                original_graph, model_graph);
  SampleManager::setInstance(model_graph);
  // Run experiment with Exponentiated Gradient strategy
  ExponentiatedGradientStrategy strategy(
      model_graph, original_graph, *evaluators.at(exploit), inc, update, learn);
  strategy.perform(budget, k);
}

void benchmark(int argc, const char * argv[]){
  std::cout<<"loading graph...";
  std::string file_name_graph(argv[2]);
  double alpha = atoi(argv[3]);
  double beta = atoi(argv[4]);
  std::ifstream file(file_name_graph);
  timestamp_t t0, t1;
  double time_msec, time_min;
  Graph graph;
  unsigned long src, tgt;
  double prob;
  unsigned long edges = 0;
  while (file >> src >> tgt >> prob) {
    std::shared_ptr<InfluenceDistribution>
        dst(new BetaInfluence(alpha, beta, prob));
    graph.add_edge(src, tgt, dst);
    edges++;
  }
  std::cout << " done." << std::endl;
  unsigned long nodes = graph.get_nodes().size();
  std::cout << "\t" << nodes << " nodes, " << edges << " edges" << std::endl;
  int samples = 100;
  if (argc > 6)
    samples = atoi(argv[6]);
  std::cout<<"done."<<std::endl;
  SpreadSampler sampler(INFLUENCE_MED);
  std::cout<<"sampling... "<< std::flush;
  std::unordered_set<unsigned long> activated;
  t0 = get_timestamp();
  for (unsigned long node : graph.get_nodes()) {
    std::unordered_set<unsigned long> seeds;
    seeds.insert(node);
    sampler.sample(graph, activated, seeds, samples);
  }
  t1 = get_timestamp();
  std::cout << "done." << std::endl;
  time_min = (t1 - t0) / 60000000.0L;
  time_msec =((t1-t0)/1000.0L)/(double)nodes/(double)samples;
  std::cout << "total time " << time_min << "min" <<std::endl;
  std::cout << "time/sample/node " << time_msec << "ms" << std::endl;
}

void spread(int argc, const char * argv[]) {
  std::string file_name_graph(argv[2]);
  unsigned long alpha = atoi(argv[3]);
  unsigned long beta = atoi(argv[4]);
  unsigned int k = atoi(argv[5]);
  std::ifstream file(file_name_graph);
  timestamp_t t0, t1;
  double time_min_celf, time_min_random;
  Graph graph;
  unsigned long src, tgt;
  double prob;
  unsigned long edges = 0;
  while(file >> src >> tgt >> prob) {
    std::shared_ptr<InfluenceDistribution>
        dst(new BetaInfluence(alpha, beta, prob));
    graph.add_edge(src, tgt, dst);
    edges++;
  }
  int samples = 100;
  if (argc > 6)
    samples = atoi(argv[6]);
  std::unordered_set<unsigned long> activated;
  PathSampler sampler(INFLUENCE_MED);
  CELFEvaluator evaluator_celf;
  t0 = get_timestamp();
  evaluator_celf.select(graph, sampler, activated, k, samples);
  t1 = get_timestamp();
  time_min_celf = (t1 - t0) / 60000000.0L;
  RandomEvaluator evaluator_random;
  t0 = get_timestamp();
  evaluator_random.select(graph, sampler, activated, k, samples);
  t1 = get_timestamp();
  time_min_random = (t1-t0)/60000000.0L;
  std::cout << k << "\t" << time_min_celf << "\t" <<
    time_min_random <<std::endl;
}

/**
  Function performing experiment with MissingMassStrategy.

  Ex. usage: ./oim --missing_mass graph.txt 0 20 2 5
*/
void missing_mass(int argc, const char * argv[],
      std::vector<std::unique_ptr<GraphReduction>>& greduction) {
  if (argc != 7) {
    std::cerr << "Wrong number of arguments.\n\tUsage ./oim --missing_mass "
              << "<graph> <reduction> <budget> <k> <n_experts>" << std::endl;
    exit(1);
  }
  Graph original_graph;
  load_original_graph(argv[2], original_graph);
  // Graph reduction method
  unsigned int reduction = atoi(argv[3]);
  if (reduction >= greduction.size()) {
    std::cerr << "Wrong type of graph reduction." << std::endl;
    exit(1);
  }
  unsigned int budget = atoi(argv[4]);
  unsigned int k = atoi(argv[5]);
  int n_experts = atoi(argv[6]);
  MissingMassStrategy strategy(
      original_graph, *greduction.at(reduction), n_experts);
  strategy.perform(budget, k);
}

int main(int argc, const char * argv[]) {
  // Vector of different GraphReduction implementations
  std::vector<unique_ptr<GraphReduction>> greductions;
  greductions.push_back(std::unique_ptr<GraphReduction>(
      new GreedyMaxCoveringReduction()));
  greductions.push_back(std::unique_ptr<GraphReduction>(
      new HighestDegreeReduction()));

  // Vector of different Evaluator implementations
  std::vector<std::unique_ptr<Evaluator>> evaluators;
  evaluators.push_back(std::unique_ptr<Evaluator>(new RandomEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new DiscountDegreeEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new HighestDegreeEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new CELFEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new TIMEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new SSAEvaluator(0.1)));
  evaluators.push_back(std::unique_ptr<Evaluator>(new PMCEvaluator(200)));

  std::string experiment(argv[1]);
  if (experiment == "--benchmark") benchmark(argc, argv);
  else if (experiment == "--spread") spread(argc, argv);
  else if (experiment == "--egreedy") epsgreedy(argc, argv);
  else if (experiment == "--explore") explore(argc, argv);
  else if (experiment == "--real") real(argc, argv, evaluators);
  else if (experiment == "--prior") prior(argc, argv);
  else if (experiment == "--eg") expgr(argc, argv, evaluators);
  else if (experiment == "--missing_mass") missing_mass(argc, argv, greductions);
}
