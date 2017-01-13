#!/usr/bin/env python
#-*-coding: utf-8 -*-

"""
Script to clean graph in the format:
    node1 <TAB> node2

Specifically, it
    1. Renumbers nodes from 0 to n - 1
    2. Removes multiple edges (when several edges from u to v)
    3. Selects the largest component
"""

import sys
import networkx as nx


def load_graph(graph, directed=True):
    """
    If the graph is undirected, we assign each edge to both sides.
    """
    mapping, G = dict(), dict()
    n = 0
    with open(graph, 'r') as fin:
        for l in fin:
            u1, u2 = map(int, l.rstrip().split()[:2])
            if u1 not in mapping:
                mapping[u1] = n
                n += 1
            if u2 not in mapping:
                mapping[u2] = n
                n += 1
            u1 = mapping[u1]
            u2 = mapping[u2]
            if u1 not in G:
                G[u1] = set()
            G[u1].add(u2)
            if not directed:
                if u2 not in G:
                    G[u2] = set()
                G[u2].add(u1)
    # Translation to networkx format
    G2 = nx.DiGraph()
    for u1 in G:
        for u2 in G[u1]:
            G2.add_edge(u1, u2)
    return G2

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python clean_graph.py <graph> [<directed>]'
        sys.exit(1)
    if len(sys.argv) >= 3:
        directed = (int(sys.argv[2]) == 1)
    G = load_graph(sys.argv[1], directed)
    giant = max(nx.connected_component_subgraphs(G), key=len)
    for u1 in giant:
        for u2 in giant[u1]:
            print '%d\t%d' % (u1, u2)
