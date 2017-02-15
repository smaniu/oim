This folder contains Python scripts to clean graphs and generate weights in
the desired format.

# Clean graph

The script can be run as follows:

    python clean_graph.py <graph> [<directed>]

## Parameters

The parameters are set as follows:

* *graph* is the name of the graph file. The file must be of the following
  format:

    node1 <TAB> node2

  where *node1* and *node2* are the endpoints of a graph edge.

* *directed* is **1** if the input graph is supposed to be directed and **0**
  otherwise. In any case, the script will output a directed graph from the
  largest component, possibly doubling edges when needed.

## Output

The cleaned graph is written on the standard output following this format

    node1 <TAB> node2

# Generate edge weights according various models

The script can be run as follows:

    python edge_weights.py <graph> <model> [<p1> <p2> ...]

## Parameters

The parameters are set as follows:

* *graph* is the name of the graph file. The file must be of the following
  format:

    node1 <TAB> node2

  where *node1* and *node2* are the endpoints of a graph edge.

* *model* can take one of the following values: **0** IC model with constant
  probabilities, **1** Weighted Cascade **2** Tri-valency Model, **3** Uniform,
  **4** Random.
* *p_i* probability for constant weights (for model **0** and **2**).

For details about *model*, see [this paper][1] in page 3.

## Output

The graph is written on the standard output following this format

    node1 <TAB> node2 <TAB> weight

[1]: <http://people.cs.umass.edu/~sainyam/papers/SIGMOD17_im_benchmarking.pdf> "A. Arora, S. Galhotra, S. Ranu. Debunking the Myths of Influence Maximization: An In-Depth Benchmarking study. SIGMOD 2017"
