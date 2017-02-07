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
#include "CELFEvaluator.hpp"
#include "TIMEvaluator.hpp"
#include "SSAEvaluator.hpp"
#include "RandomEvaluator.hpp"
#include "HighestDegreeEvaluator.hpp"
#include "DiscountDegreeEvaluator.hpp"
#include "PMCEvaluator.hpp"
#include "Strategy.hpp"
#include "LogDiffusion.hpp"

using namespace std;

/**
  Function performing diffusion with *known* graph. The seeds are selected with
  one of the Evaluators.

  Ex. usage: ./oim --real graph.txt 5 20 2 1
*/
void real(int argc, const char * argv[],
          std::vector<std::unique_ptr<Evaluator>>& evaluators) {
  if (argc < 6) {
    std::cerr << "Wrong number of arguments.\n\tUsage ./oim --real <graph> "
              << "<exploit> <budget> <k> [<model> <cascades> <samples>]"
              << std::endl;
    exit(1);
  }
  Graph original_graph;
  unsigned int exploit = atoi(argv[3]);
  unsigned int budget = atoi(argv[4]);
  unsigned int k = atoi(argv[5]);
  int samples = 1; // TODO change that to take the actual parameter given as input
  int model = (argc > 6) ? atoi(argv[6]) : 1;
  load_original_graph(argv[2], original_graph, model);
  // Load cascades from logs if 11th parameter
  std::unique_ptr<LogDiffusion> log_diffusion;
  if (argc > 7) {
    log_diffusion = std::make_unique<LogDiffusion>();
    log_diffusion->load_cascades(argv[7]);
  }
  OriginalGraphStrategy strategy(original_graph, *evaluators.at(exploit),
                                 samples, model, std::move(log_diffusion));
  strategy.perform(budget, k);
}

void epsgreedy(int argc, const char * argv[]) {
  std::string file_name_graph(argv[2]);
  double alpha = atof(argv[3]);
  double beta = atof(argv[4]);
  std::ifstream file(file_name_graph);
  Graph original_graph, model_graph;
  unode_int src, tgt;
  double prob;
  unode_int edges = 0;
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
  bool update = (argc > 10) ? (atoi(argv[10]) == 1) : true;
  unsigned int learn = (argc > 11) ? atoi(argv[11]) : 0;
  unsigned int int_exploit = (argc > 12) ? atoi(argv[12]) : INFLUENCE_MED;
  unsigned int int_explore = (argc > 13) ? atoi(argv[13]) : INFLUENCE_MED;
  int inc = (argc > 14) ? atoi(argv[14]) : 0;
  int samples = (argc > 15) ? atoi(argv[15]) : 1000;

  std::vector<std::unique_ptr<Evaluator>> evals;
  evals.push_back(std::unique_ptr<Evaluator>(new CELFEvaluator(samples)));
  evals.push_back(std::unique_ptr<Evaluator>(new RandomEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new DiscountDegreeEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new TIMEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new HighestDegreeEvaluator()));
  evals.push_back(std::unique_ptr<Evaluator>(new SSAEvaluator(0.1)));
  EpsilonGreedyStrategy strategy(model_graph, original_graph,
                                 *evals.at(explore), *evals.at(exploit),
                                 samples, eps, inc);
  strategy.perform(budget, k, update, learn, int_exploit, int_explore);
}

/**
  Function performing experiment with ExponentiatedGradientStrategy.

  Ex. usage: ./oim --eg graph.txt 1 20 5 20 2
*/
void expgr(int argc, const char * argv[],
           std::vector<std::unique_ptr<Evaluator>>& evaluators) {
  if (argc < 8) {
    std::cerr << "Wrong number of arguments.\n\tUsage ./oim --eg <graph> "
              << "<alpha> <beta> <exploit> <trials> <k> [<model> <update> "
              << "<update_type> <cascades>]" << std::endl;
    exit(1);
  }
  // Take parameters
  double alpha = atof(argv[3]), beta = atof(argv[4]);
  unsigned int exploit = atoi(argv[5]);
  if (exploit > 6) {
    std::cerr << "Error: <exploit> must be in range 0..6" << std::endl;
    exit(1);
  }
  unsigned int budget = atoi(argv[6]);
  unsigned int k = atoi(argv[7]);
  int model = (argc > 8) ? atoi(argv[8]) : 1;
  // TODO Check that the selected Evaluator implemented the LT if chosen
  bool update = (argc > 9) ? (atoi(argv[9]) == 1) : true;
  unsigned int learn = (argc > 10) ? atoi(argv[10]) : 0;

  // Load model and original graphs
  Graph original_graph, model_graph;
  load_model_and_original_graph(argv[2], alpha, beta, original_graph,
                                model_graph, model);
  // Load cascades from logs if 11th parameter
  std::unique_ptr<LogDiffusion> log_diffusion;
  if (argc > 11) {
    log_diffusion = std::make_unique<LogDiffusion>();
    log_diffusion->load_cascades(argv[11]);
  }
  // Run experiment with Exponentiated Gradient strategy
  ExponentiatedGradientStrategy strategy(
      model_graph, original_graph, *evaluators.at(exploit),
      update, learn, model, std::move(log_diffusion));
  strategy.perform(budget, k);
}

/**
  Function performing experiment with MissingMassStrategy.

  Ex. usage: ./oim --missing_mass graph.txt 1 0 20 2 5
*/
void missing_mass(int argc, const char * argv[],
      std::vector<std::unique_ptr<GraphReduction>>& greduction) {
  if (argc < 8) {
    std::cerr << "Wrong number of arguments.\n\tUsage ./oim --missing_mass "
              << "<graph> <policy> <reduction> <budget> <k> <n_experts> "
              << "[<model> <cascades>]" << std::endl;
    exit(1);
  }
  // Policy to choose expert
  unsigned int n_policy = atoi(argv[3]);
  if (n_policy >= 2) {
    std::cerr << "Wrong type of policy." << std::endl;
    exit(1);
  }
  // Graph reduction method
  unsigned int reduction = atoi(argv[4]);
  if (reduction >= greduction.size()) {
    std::cerr << "Wrong type of graph reduction." << std::endl;
    exit(1);
  }
  unsigned int budget = atoi(argv[5]);
  unsigned int k = atoi(argv[6]);
  int n_experts = atoi(argv[7]);
  int model = argc > 8 ? atoi(argv[8]) : 1;

  // Load graph
  Graph original_graph;
  load_original_graph(argv[2], original_graph, model);

  // Load real cascades
  std::unique_ptr<LogDiffusion> log_diffusion;
  if (argc > 9) {
    log_diffusion = std::make_unique<LogDiffusion>();
    log_diffusion->load_cascades(argv[9]);
  }
  MissingMassStrategy strategy(
      original_graph, *greduction.at(reduction), n_experts, n_policy, model,
      std::move(log_diffusion));
  // Give strategy the reduction method for output
  strategy.set_graph_reduction(reduction);
  strategy.perform(budget, k);
}

int main(int argc, const char * argv[]) {
  // Vector of different GraphReduction implementations
  std::vector<unique_ptr<GraphReduction>> greductions;
  greductions.push_back(std::unique_ptr<GraphReduction>(
      new GreedyMaxCoveringReduction()));
  greductions.push_back(std::unique_ptr<GraphReduction>(
      new HighestDegreeReduction()));
  std::unique_ptr<Evaluator> evaluator(new PMCEvaluator(200));
  greductions.push_back(std::unique_ptr<GraphReduction>(
      new EvaluatorReduction(0.01, *evaluator, 1)));  // TODO allow change on the model expected to select experts
  greductions.push_back(std::unique_ptr<GraphReduction>(
      new DivRankReduction(0.25, 0.05, 100)));

  // Vector of different Evaluator implementations
  std::vector<std::unique_ptr<Evaluator>> evaluators;
  evaluators.push_back(std::unique_ptr<Evaluator>(new RandomEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new DiscountDegreeEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new HighestDegreeEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new CELFEvaluator(100)));
  evaluators.push_back(std::unique_ptr<Evaluator>(new TIMEvaluator()));
  evaluators.push_back(std::unique_ptr<Evaluator>(new SSAEvaluator(0.1)));
  evaluators.push_back(std::unique_ptr<Evaluator>(new PMCEvaluator(200)));

  std::string experiment(argv[1]);
  if  (experiment == "--egreedy") epsgreedy(argc, argv);
  else if (experiment == "--real") real(argc, argv, evaluators);
  else if (experiment == "--eg") expgr(argc, argv, evaluators);
  else if (experiment == "--missing_mass") missing_mass(argc, argv, greductions);
}
