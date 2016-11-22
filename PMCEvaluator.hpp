/*
 Copyright (c) 2016 Paul Lagrée (Université Paris Sud)

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

#ifndef __oim__PMCEvaluator__
#define __oim__PMCEvaluator__

#include <omp.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>

#include "common.hpp"
#include "Graph.hpp"
#include "Evaluator.hpp"
#include "Sampler.hpp"

using namespace std;


class PrunedEstimator {
 private:
	int n, n1;
	std::vector<int> weight, comp, sigmas;
	std::vector<int> pmoc;
	std::vector<int> at_p;
	std::vector<int> up;
	std::vector<bool> memo, removed;
	std::vector<int> es, rs;
	std::vector<int> at_e, at_r;
	std::vector<bool> visited;
	int hub;
	vector<bool> descendant, ancestor;
	bool flag;

  void first() {
  	hub = 0;
  	for (int i = 0; i < n; i++) {
  		if ((at_e[i + 1] - at_e[i]) + (at_r[i + 1] - at_r[i])
  				> (at_e[hub + 1] - at_e[hub]) + (at_r[hub + 1] - at_r[hub])) {
  			hub = i;
  		}
  	}

  	descendant.resize(n);
  	queue<int> Q;
  	Q.push(hub);
  	for (; !Q.empty();) {
  		// forall v, !remove[v]
  		const int v = Q.front();
  		Q.pop();
  		descendant[v] = true;
  		for (int i = at_e[v]; i < at_e[v + 1]; i++) {
  			const int u = es[i];
  			if (!descendant[u]) {
  				descendant[u] = true;
  				Q.push(u);
  			}
  		}
  	}

  	ancestor.resize(n);
  	Q.push(hub);
  	for (; !Q.empty();) {
  		const int v = Q.front();
  		Q.pop();
  		ancestor[v] = true;
  		for (int i = at_r[v]; i < at_r[v + 1]; i++) {
  			const int u = rs[i];
  			if (!ancestor[u]) {
  				ancestor[u] = true;
  				Q.push(u);
  			}
  		}
  	}
  	ancestor[hub] = false;

  	for (int i = 0; i < n; i++) {
  		sigma(i);
  	}
  	ancestor.assign(n, false);
  	descendant.assign(n, false);

  	for (int i = 0; i < n1; i++) {
  		up.push_back(i);
  	}
  }

  int sigma(const int v0) {
  	if (memo[v0]) {
  		return sigmas[v0];
  	}
  	memo[v0] = true;
  	if (removed[v0]) {
  		return sigmas[v0] = 0;
  	} else {
  		int child = unique_child(v0);
  		if (child == -1) {
  			return sigmas[v0] = weight[v0];
  		} else if (child >= 0) {
  			return sigmas[v0] = sigma(child) + weight[v0];
  		} else {
  			int delta = 0;
  			vector<int> vec;
  			visited[v0] = true;
  			vec.push_back(v0);
  			queue<int> Q;
  			Q.push(v0);
  			bool prune = ancestor[v0];

  			if (prune) {
  				delta += sigma(hub);
  			}

  			for (; !Q.empty();) {
  				const int v = Q.front();
  				Q.pop();
  				if (removed[v]) {
  					continue;
  				}
  				if (prune && descendant[v]) {
  					continue;
  				}
  				delta += weight[v];
  				for (int i = at_e[v]; i < at_e[v + 1]; i++) {
  					const int u = es[i];
  					if (removed[u]) {
  						continue;
  					}
  					if (!visited[u]) {
  						visited[u] = true;
  						vec.push_back(u);
  						Q.push(u);
  					}
  				}
  			}
  			for (int i = 0; i < (int) vec.size(); i++) {
  				visited[vec[i]] = false;
  			}
  			return sigmas[v0] = delta;
  		}
  	}
  }

  inline int unique_child(const int v) {
  	int outdeg = 0, child = -1;
  	for (int i = at_e[v]; i < at_e[v + 1]; i++) {
  		const int u = es[i];
  		if (!removed[u]) {
  			outdeg++;
  			child = u;
  		}
  	}
  	if (outdeg == 0) {
  		return -1;
  	} else if (outdeg == 1) {
  		return child;
  	} else {
  		return -2;
  	}
  }

 public:
  void init(const int _n, vector<pair<int, int>>& _es, vector<int>& _comp) {
   	flag = true;
   	n = _n;
   	n1 = _comp.size();

   	visited.resize(n, false);

   	int m = _es.size();
   	vector<int> outdeg(n), indeg(n);

   	for (int i = 0; i < m; i++) {
   		int a = _es[i].first, b = _es[i].second;
   		outdeg[a]++;
   		indeg[b]++;
   	}
   	es.resize(m, -1);
   	rs.resize(m, -1);

   	at_e.resize(n + 1, 0);
   	at_r.resize(n + 1, 0);

   	at_e[0] = at_r[0] = 0;
   	for (int i = 1; i <= n; i++) {
   		at_e[i] = at_e[i - 1] + outdeg[i - 1];
   		at_r[i] = at_r[i - 1] + indeg[i - 1];
   	}

   	for (int i = 0; i < m; i++) {
   		int a = _es[i].first, b = _es[i].second;
   		es[at_e[a]++] = b;
   		rs[at_r[b]++] = a;
   	}

   	at_e[0] = at_r[0] = 0;
   	for (int i = 1; i <= n; i++) {
   		at_e[i] = at_e[i - 1] + outdeg[i - 1];
   		at_r[i] = at_r[i - 1] + indeg[i - 1];
   	}

   	sigmas.resize(n);
   	comp = _comp;
   	vector<pair<int, int> > ps;
   	for (int i = 0; i < n1; i++) {
   		ps.push_back(make_pair(comp[i], i));
   	}
   	sort(ps.begin(), ps.end());
   	at_p.resize(n + 1);
   	for (int i = 0; i < n1; i++) {
   		pmoc.push_back(ps[i].second);
   		at_p[ps[i].first + 1]++;
   	}
   	for (int i = 1; i <= n; i++) {
   		at_p[i] += at_p[i - 1];
   	}

   	memo.resize(n);
   	removed.resize(n);

   	weight.resize(n1, 0);
   	for (int i = 0; i < n1; i++) {
   		weight[comp[i]]++;
   	}

   	first();
  }

  int sigma1(const int v) {
  	return sigma(comp[v]);
  }

  void add(int v0) {
  	v0 = comp[v0];
  	queue<int> Q;
  	Q.push(v0);
  	removed[v0] = true;
  	vector<int> rm;
  	for (; !Q.empty();) {
  		const int v = Q.front();
  		Q.pop();
  		rm.push_back(v);
  		for (int i = at_e[v]; i < at_e[v + 1]; i++) {
  			const int u = es[i];
  			if (!removed[u]) {
  				Q.push(u);
  				removed[u] = true;
  			}
  		}
  	}

  	up.clear();

  	vector<int> vec;
  	for (int i = 0; i < (int) rm.size(); i++) {
  		const int v = rm[i];
  		memo[v] = false; // for update()
  		for (int j = at_p[v]; j < at_p[v + 1]; j++) {
  			up.push_back(pmoc[j]);
  		}
  		for (int j = at_r[v]; j < at_r[v + 1]; j++) {
  			const int u = rs[j];
  			if (!removed[u] && !visited[u]) {
  				visited[u] = true;
  				vec.push_back(u);
  				Q.push(u);
  			}
  		}
  	}
  	// reachable to removed node
  	for (; !Q.empty();) {
  		const int v = Q.front();
  		Q.pop();
  		memo[v] = false;
  		for (int j = at_p[v]; j < at_p[v + 1]; j++) {
  			up.push_back(pmoc[j]);
  		}
  		for (int i = at_r[v]; i < at_r[v + 1]; i++) {
  			const int u = rs[i];
  			if (!visited[u]) {
  				visited[u] = true;
  				vec.push_back(u);
  				Q.push(u);
  			}
  		}
  	}
  	for (int i = 0; i < (int) vec.size(); i++) {
  		visited[vec[i]] = false;
  	}
  }

  void update(vector<long long> &sums) {
  	for (int i = 0; i < (int) up.size(); i++) {
  		int v = up[i];
  		if (!flag) {
  			sums[v] -= sigmas[comp[v]];
  		}
  	}
  	for (int i = 0; i < (int) up.size(); i++) {
  		int v = up[i];
  		sums[v] += sigma1(v);
  	}
  	flag = false;
  }
};

/**
  Random Number Generator
*/
class Xorshift {
 public:
	Xorshift(unsigned int seed) {
		x_ = _(seed, 0);
		y_ = _(x_, 1);
		z_ = _(y_, 2);
		w_ = _(z_, 3);
	}

	int _(int s, int i) {
		return 1812433253 * (s ^ (s >> 30)) + i + 1;
	}

	inline int gen_int() {
		unsigned int t = x_ ^ (x_ << 11);
		x_ = y_;
		y_ = z_;
		z_ = w_;
		return w_ = w_ ^ (w_ >> 19) ^ t ^ (t >> 8);
	}

	inline int gen_int(int n) {
		return (int) (n * gen_double());
	}

	inline double gen_double() {
		unsigned int a = ((unsigned int) gen_int()) >> 5, b =
				((unsigned int) gen_int()) >> 6;
		return (a * 67108864.0 + b) * (1.0 / (1LL << 53));
	}

 private:
	unsigned int x_, y_, z_, w_;
};

/**
  Implementation of PMC algorithm introduced in `Fast and Accurate Influence
  Maximization on Large Networks with Pruned Monte-Carlo Simulations` by Naoto
  Ohsaka et al., AAAI 2014.
*/
class PMCEvaluator : public Evaluator {
 private:
  std::unordered_set<unsigned long> seed_set_;  // Set of k selected nodes
  std::vector<unsigned long> es1_;
  std::vector<unsigned long> rs1_;
  std::vector<unsigned long> at_e_;
  std::vector<unsigned long> at_r_;
  std::random_device rd_;
  unsigned int R_;  // Number of DAGs (Directed Acyclic Graphs)
  unsigned int type_;
  unsigned long n_; // Number of vertices
  unsigned long m_; // Number of edges

  int scc(vector<int>& comp) {
  	std::vector<bool> vis(n_);
  	std::stack<pair<int, int>> S;
  	std::vector<int> lis;
  	int k = 0;
  	for (unsigned long i = 0; i < n_; i++) {
  		S.push(make_pair(i, 0));
  	}
  	for (; !S.empty();) {
  		int v = S.top().first, state = S.top().second;
  		S.pop();
  		if (state == 0) {
  			if (vis[v]) {
  				continue;
  			}
  			vis[v] = true;
  			S.push(make_pair(v, 1));
  			for (unsigned long i = at_e_[v]; i < at_e_[v + 1]; i++) {
  				unsigned long u = es1_[i];
  				S.push(make_pair(u, 0));
  			}
  		} else {
  			lis.push_back(v);
  		}
  	}
  	for (unsigned long i = 0; i < n_; i++) {
  		S.push(make_pair(lis[i], -1));
  	}
  	vis.assign(n_, false);
  	for (; !S.empty();) {
  		int v = S.top().first, arg = S.top().second;
  		S.pop();
  		if (vis[v]) {
  			continue;
  		}
  		vis[v] = true;
  		comp[v] = arg == -1 ? k++ : arg;
  		for (unsigned long i = at_r_[v]; i < at_r_[v + 1]; i++) {
  			unsigned long u = rs1_[i];
  			S.push(make_pair(u, comp[v]));
  		}
  	}
  	return k;
  }

 public:
  PMCEvaluator(unsigned int R)
      : R_(R) {};

  std::unordered_set<unsigned long> select(
        const Graph& graph, Sampler& sampler,
        const std::unordered_set<unsigned long>& activated, unsigned int k) {
  	n_ = graph.get_number_nodes();
  	m_ = graph.get_number_edges();
    type_ = sampler.get_type();

  	// sort(es.begin(), es.end()); TODO sort the graph edges

  	es1_.resize(m_);
  	rs1_.resize(m_);
  	at_e_.resize(n_ + 1);
  	at_r_.resize(n_ + 1);

  	std::vector<PrunedEstimator> infs(R_);

  	for (unsigned int t = 0; t < R_; t++) {
  		Xorshift xs = Xorshift(t/* + time(NULL)*/);
  		unsigned int mp = 0;
  		at_e_.assign(n_ + 1, 0);
  		at_r_.assign(n_ + 1, 0);
  		std::vector<pair<unsigned long, unsigned long>> ps;

  		for (unsigned long i = 0; i < n_; i++) {
        if (!graph.has_neighbours(i))
          continue;
        for (auto& edge : graph.get_neighbours(i)) {
    			if (xs.gen_double() < edge.dist->sample(type_)) {
    				es1_[mp++] = edge.target;
    				at_e_[edge.source + 1]++;
    				ps.push_back(make_pair(edge.target, edge.source));
    			}
        }
  		}
  		at_e_[0] = 0;

  		sort(ps.begin(), ps.end());

  		for (unsigned int i = 0; i < mp; i++) {
  			rs1_[i] = ps[i].second;
  			at_r_[ps[i].first + 1]++;
  		}
  		for (unsigned long i = 1; i <= n_; i++) {
  			at_e_[i] += at_e_[i - 1];
  			at_r_[i] += at_r_[i - 1];
  		}
      /**
        Here at_e is the sum of the number of activations from each node.
        Ex. [0, 2, 5, 8] if node 0 activated two edges, 1 three edges and 2 three
        edges.
        Similar thing for at_r;
      */
  		std::vector<int> comp(n_);

  		int nscc = scc(comp);

  		vector<pair<int, int>> es2;
  		for (unsigned long u = 0; u < n_; u++) {
  			unsigned int a = comp[u];
  			for (unsigned long i = at_e_[u]; i < at_e_[u + 1]; i++) {
  				unsigned long b = comp[es1_[i]];
  				if (a != b) {
  					es2.push_back(make_pair(a, b));
  				}
  			}
  		}

  		sort(es2.begin(), es2.end());
  		es2.erase(unique(es2.begin(), es2.end()), es2.end());

  		infs[t].init(nscc, es2, comp);
  	}

  	vector<long long> gain(n_);
  	vector<int> S;

  	for (unsigned int t = 0; t < k; t++) {
  		for (unsigned int j = 0; j < R_; j++) {
  			infs[j].update(gain);
  		}
  		int next = 0;
  		for (unsigned int i = 0; i < n_; i++) {
  			if (gain[i] > gain[next]) {
  				next = i;
  			}
  		}

  		S.push_back(next);
  		for (unsigned int j = 0; j < R_; j++) {
  			infs[j].add(next);
  		}
  		seed_set_.insert(next);
  	}
  	return seed_set_;
  }
};

#endif /* defined(__oim__PMCEvaluator__) */
