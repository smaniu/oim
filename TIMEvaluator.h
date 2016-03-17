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

#ifndef __oim__TIMEvaluator__
#define __oim__TIMEvaluator__

#include "common.h"
#include "Graph.h"
#include "Evaluator.h"
#include "Sampler.h"
#include "SpreadSampler.h"
#include "PathSampler.h"
#include "SampleManager.h"


#include <math.h>

using namespace std;

class TIMEvaluator : public Evaluator{
private:
  std::unordered_set<unsigned long> activated;
  unsigned int k;
  unsigned long n;
  unsigned long n_max;
  unsigned long m;

  double epsilon;

  std::unordered_set<unsigned long> seedSet;
//  std::vector<std::vector<unsigned long> > rr_sets;
  std::vector<std::shared_ptr<std::vector<unsigned long> > > rr_sets;

  //std::unordered_map<unsigned long, unsigned long> deg;

  std::vector<unsigned long> graph_nodes;

//  std::vector<std::vector<unsigned long> > hyperG;
  std::vector<std::shared_ptr<std::vector<unsigned long> > > hyperG;

  int64 hyperId;
  int64 totalR;
  //random devices
  std::random_device rd;
  std::mt19937 gen;

  bool INCREMENTAL;

public:
  
  TIMEvaluator() : gen(rd()) {};

  void setIncremental(bool inc) {INCREMENTAL = inc;}

  std::unordered_set<unsigned long> select(const Graph& graph,
                                           Sampler& sampler,
                                           const std::unordered_set<unsigned long>& activated,
                                           unsigned int k,
                                           unsigned long samples){
    //initializing data structures
    //printf("activated nodes before this trial: ");

    timestamp_t t0, t1, t2;

    for(auto node:activated) {
      (this->activated).insert(node);
      //printf("%ld\t", node);
    }
    //puts("");
    this->k = k;

    t0 = get_timestamp();

    n = graph.get_number_nodes();
    m = graph.get_number_edges();
    hyperId = 0;
    totalR = 0;
    epsilon = 0.1; 

    graph_nodes.clear();
    n_max = 0;
    for(auto src:graph.get_nodes()){
      if(activated.find(src)==activated.end()){
          if(src > n_max) n_max = src;
          graph_nodes.push_back(src);
        }
    }

    //SpreadSampler sampler_s(sampler.get_quantile());
    PathSampler sampler_s(sampler.get_quantile());

    std::uniform_int_distribution<int> dst(0,(int)graph_nodes.size() - 1);

    double ep_step2, ep_step3;
    ep_step2 = ep_step3 = epsilon;
    ep_step2 = 5*pow(sqr(ep_step3)/k, 1.0/3.0);
    double ept;

    ept=EstimateEPT(graph, sampler_s, dst);

    //printf("ept = %.lf \n", ept);

    BuildSeedSet();


    BuildHyperGraph2(ep_step2, ept, graph, sampler_s, dst);
    ept=InfluenceHyperGraph();
    //printf("ept1 = %.lf \n", ept);

    ept/=1+ep_step2;
    //printf("ept2 = %.lf \n", ept);
    

    BuildHyperGraph3(ep_step3, ept, graph, sampler_s, dst);
    
    t1 = get_timestamp();

    //cerr << "start BuildSeedSet()" << endl;
    BuildSeedSet();
    //puts("end BuildSeedSet()");
    
    t2 = get_timestamp();

    //puts("start InfluenceHyperGraph()");
    //ept=InfluenceHyperGraph();
    //puts("end InfluenceHyperGraph()");

    //for(auto item:seedSet)
    //    cerr<< item << " ";
    //cerr<<totalR;
    //cerr<<endl;

    sampling_time = (t1-t0)/60000000.0L;
    choosing_time = (t2-t1)/60000000.0L;

    //printf("Tim Expected = %.lf \n", ept);


    return seedSet;    
  }
  
private:
  
  double EstimateEPT(const Graph& graph, Sampler& sampler, std::uniform_int_distribution<int>& dst){
    double ept=Estimate_KPT(graph, sampler, dst);
    ept/=2;
    return ept;
  }

  double Estimate_KPT(const Graph& graph, Sampler& sampler, std::uniform_int_distribution<int>& dst){

    double lb=1/2.0;
    double c=0;
    int64 lastR=0;

    double return_value = 1;
    int steps = 1;  // added for algorithm 2 line 1
    while(steps <= log(n) / log(2) - 1){
        int loop= (6 * log(n)  +  6 * log(log(n)/ log(2)) )* 1/lb  ;
        c=0;
        lastR=loop;

        for(int i=0; i<loop; i++){
            std::shared_ptr<std::vector<unsigned long> > rr (\
                                            new std::vector<unsigned long>());
            if (!INCREMENTAL) {
              std::unordered_set<unsigned long> seeds;
              unsigned long u = graph_nodes[dst(gen)];
              seeds.insert(u);
              rr->push_back(u);

              sampler.trial(graph, activated, seeds, true);

              //cout<<"Seed and Trial size: "<< u<< " " << sampler.get_trials().size();

              for(trial_type tt:sampler.get_trials()){
                if(tt.trial==1){
                  rr->push_back(tt.target);
                }
              }
            } else { 
              rr = SampleManager::getInstance()->getSample(graph_nodes, sampler, activated, dst);
            }
            double MgTu = 0;
            for(auto node:(*rr)){
              if(graph.has_neighbours(node,true)){
                MgTu += graph.get_neighbours(node,true).size();
              }
            }
            double pu=MgTu/m;
            
            //printf("-- MgTu, pu, c = %lf , %lf, %lf\n", MgTu, pu, c);

            c+=1-pow((1-pu), k);
        }

        c/=loop;
        if(c>lb) {return_value = c * n; break;}
        lb /= 2;
        steps++;
    }

    //printf("Estimate_KPT -- R = %lld\n", lastR);

    buildSamples(lastR, graph, sampler, dst);

    return return_value;
  }

  void buildSamples(int64 &R, const Graph& graph, Sampler& sampler, std::uniform_int_distribution<int>& dst){

    //cerr << "buildSamples R = " << R << endl;
    totalR += R;

    if (R > MAX_R) R = MAX_R;
    hyperId = R;

    hyperG.clear();
    hyperG.reserve(n);
    for (int i = 0; i < n; ++i) {
      hyperG.push_back(std::shared_ptr<std::vector<unsigned long> >(new std::vector<unsigned long>()));
    }
//    hyperG = vector<vector<unsigned long> >(n, vector<unsigned long>());

    rr_sets.clear();
    rr_sets.reserve(R);    
//    rr_sets = vector<vector<unsigned long> >(R, vector<unsigned long>());

//    deg.clear();
//    for(auto src:graph.get_nodes()){
//      deg[src] = 0;
//    }

    double totTime = 0.0;
    double totInDegree = 0;

    for(int i=0; i<R; i++){
      if (!INCREMENTAL) {
        std::shared_ptr<std::vector<unsigned long> > rr (new std::vector<unsigned long>());
        std::unordered_set<unsigned long> seeds;
        unsigned long nd = graph_nodes[dst(gen)];
        seeds.insert(nd);
        rr->push_back(nd);

        timestamp_t t0, t1;
        t0 = get_timestamp();
        sampler.trial(graph, activated, seeds, true);
        t1 = get_timestamp();
        totTime += (t1 - t0) / 60000000.0L;

        totInDegree += sampler.get_trials().size();
        for(trial_type tt:sampler.get_trials()){
          if(tt.trial==1){
            //deg[tt.target] += 1;
            rr->push_back(tt.target);
          }
        }
        rr_sets.push_back(rr);
      } else { 
        rr_sets.push_back(SampleManager::getInstance()->getSample(graph_nodes, sampler, activated, dst));
      }
    }

    for(int i=0; i<R; i++){
      for(unsigned long t:(*rr_sets[i])){
        hyperG[t]->push_back(i);
      }
    }

    //printf("total time for sampler: %lf minutes \n", totTime);
    //printf("average width for samples: %lf\n", totInDegree / R);
  }

  vector<bool> visit_local;
  void BuildSeedSet(){

    seedSet.clear();
    vector<int> deg= vector<int>(n, 0);
    visit_local = vector<bool>(rr_sets.size(), false);

    for (int i = 0; i < graph_nodes.size(); ++i) {
      deg[graph_nodes[i]] = hyperG[graph_nodes[i]]->size();
    }

    for (int i = 0; i < k; ++i) {
      auto t = max_element(deg.begin(), deg.end());
      int id = t - deg.begin();
      seedSet.insert(id);
      deg[id] = 0;
      for (int t:(*hyperG[id])) {
        if (!visit_local[t]) {
          visit_local[t] = true;
          for (int item:(*rr_sets[t])) {
            deg[item]--;
          }
        }
      }
    }
  }
/*
  void remove_node(unsigned long node,\
                   std::vector<std::unordered_set<unsigned long>>& rr_sets,\
                   std::unordered_map<unsigned long, unsigned long>& deg){
    std::vector<std::unordered_set<unsigned long>> new_rr_sets;
    for(auto rr_set:rr_sets){
      if(rr_set.find(node)!=rr_set.end()){
        for(auto other_node:rr_set)
          if(deg[other_node]>0) deg[other_node] -= 1;
      }
      else
        new_rr_sets.push_back(rr_set);
    }
    rr_sets.clear();
    for(auto rr_set:new_rr_sets) rr_sets.push_back(rr_set);
  }
*/  
  double InfluenceHyperGraph(){
    unordered_set<unsigned long> s;
    for(auto t:seedSet){
        for(auto tt:(*hyperG[t])){
          s.insert(tt);
        }
    }
    double inf=(double)n*s.size()/hyperId;
    return inf;
  }

  void BuildHyperGraph2(double epsilon, double ept, const Graph& graph, Sampler& sampler, std::uniform_int_distribution<int>& dst){

    int64 R = (8+2 * epsilon) * ( n * log(n) +  n * log(2)  ) / ( epsilon * epsilon * ept)/4;

    //printf("BuildHyperGraph2 -- R = %lld\n", R);

    buildSamples(R, graph, sampler, dst);
  }


  void BuildHyperGraph3(double epsilon, double opt, const Graph& graph, Sampler& sampler, std::uniform_int_distribution<int>& dst){

    //int64 R = 16.0 * k * n * log(n) / epsilon / epsilon / opt;

    //cout<<"OPT: " << opt <<endl;
    
    double logCnk = 0.0;
    for (unsigned long i = n, j = 1; j <= k; --i, ++j) {
      logCnk += log10(i) - log10(j);
    }

    int64 R = (8+2 * epsilon) * ( n * log(n) + n * log(2) +  n * logCnk ) / ( epsilon * epsilon * opt);

    //printf("BuildHyperGraph3 -- R = %lld\n", R);

    buildSamples(R, graph, sampler, dst);
  }

};



#endif /* defined(__oim__TIMEvaluator__) */
