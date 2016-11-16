#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include "../Graph.h"
#include "../graph_utils.hpp"
#include "../GraphReduction.h"

// Test the graph structure and the loading of a graph
TEST_CASE( "GRAPH LOADED", "[graph loading]" ) {
  Graph graph;
  unsigned long n_edges = load_original_graph("datasets/graph_test.csv", graph);
  REQUIRE(graph.get_number_nodes() == 8);
  REQUIRE(n_edges == 14 );
  REQUIRE(graph.get_number_edges() == 14);
  REQUIRE(graph.has_neighbours(0) == true);
  REQUIRE(graph.has_neighbours(7) == false);      // Node 7 is only pointed
  REQUIRE(graph.has_neighbours(7, true) == true); // edge (2, 7)
  REQUIRE(graph.get_neighbours(0).size() == 2);
  REQUIRE(graph.get_neighbours(0, true).size() == 1);
}

// Test the removal of a node in the graph (and checks we also delete the
// desired edges
TEST_CASE( "REMOVE NODE", "[remove node]" ) {
  Graph graph;
  // Load graph
  unsigned long n_edges = load_original_graph("datasets/graph_test.csv", graph);
  // Copy graph
  Graph copy_graph(graph);
  // Remove one node in the copy_graph
  copy_graph.remove_node(3);
  // Check graph didn't change
  REQUIRE(graph.get_number_nodes() == 8);
  REQUIRE(graph.get_number_edges() == 14);
  REQUIRE(graph.has_neighbours(3) == true);
  REQUIRE(graph.get_neighbours(2).size() == 4);
  REQUIRE(graph.get_neighbours(2, true).size() == 2);
  REQUIRE(graph.get_neighbours(4).size() == 1);
  REQUIRE(graph.get_neighbours(4, true).size() == 2);
  REQUIRE(graph.get_neighbours(6).size() == 2);
  REQUIRE(graph.get_neighbours(6, true).size() == 2);
  // Check new structure of copy_graph
  REQUIRE(copy_graph.get_number_nodes() == 7);
  REQUIRE(copy_graph.get_number_edges() == 10);
  REQUIRE(copy_graph.has_neighbours(3) == false);
  REQUIRE(copy_graph.get_neighbours(2).size() == 3);
  REQUIRE(copy_graph.get_neighbours(2, true).size() == 1);
  REQUIRE(copy_graph.get_neighbours(4).size() == 1);
  REQUIRE(copy_graph.get_neighbours(4, true).size() == 1);
  REQUIRE(copy_graph.get_neighbours(6).size() == 1);
  REQUIRE(copy_graph.get_neighbours(6, true).size() == 2);
}

// Test that the reduction with greedy algorithm works
TEST_CASE( "GREEDY MAX COVERING REDUCTION", "[greedy max cover]" ) {
  Graph graph;
  load_original_graph("datasets/graph_test.csv", graph);
  GreedyMaxCoveringReduction g_reduction = GreedyMaxCoveringReduction();
  std::vector<unsigned long> experts = g_reduction.extractExperts(graph, 1);
  REQUIRE(experts.size() == 1);
  REQUIRE(experts[0] == 2);
  experts = g_reduction.extractExperts(graph, 2);
  REQUIRE(experts.size() == 2);
  REQUIRE(experts[0] == 2);
  REQUIRE(experts[1] == 5);
}
