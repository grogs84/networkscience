# Chapter 2: Graph Theory

Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grogs84/networkscience/blob/main/notebooks/Chapter2.ipynb)

---

## Topics Covered

- **2.3** Degree, Average Degree, Degree Distribution
- **2.4** Adjacency Matrix
- **2.7** Bipartite Graphs and Projections
- **2.8** Paths and Distances
- **2.9** Clustering Coefficient

---

## 2.3 Degree, Average Degree, Degree Distribution

### Key Definitions

| Term | Formula | Description |
|------|---------|-------------|
| **Degree** ($k_i$) | — | Number of edges connected to node $i$ |
| **Total Edges** | $L = \frac{1}{2}\sum_{i=1}^{N} k_i$ | Sum of all degrees divided by 2 |
| **Average Degree** | $\langle k \rangle = \frac{1}{N}\sum_{i=1}^{N} k_i = \frac{2L}{N}$ | Mean degree across all nodes |
| **Degree Distribution** | $p_k = \frac{N_k}{N}$ | Fraction of nodes with degree $k$ |

### Code Example

```python
import rustworkx as rx

# Create a simple graph
G = rx.PyGraph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from_no_data([(0, 1), (0, 2), (0, 3), (1, 2)])

# Calculate degree metrics
degrees = [G.degree(n) for n in G.node_indexes()]
avg_degree = sum(degrees) / len(degrees)

print(f"Degrees: {degrees}")
print(f"Average degree ⟨k⟩ = {avg_degree}")
```

### Degree Distribution

The degree distribution $p_k$ gives the probability that a randomly selected node has degree $k$:

```python
import numpy as np

def degree_distribution(G):
    """Calculate the degree distribution of a graph."""
    degrees = np.array([G.degree(n) for n in G.node_indexes()])
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    N = len(degrees)
    return {int(k): count / N for k, count in zip(unique_degrees, counts)}

p_k = degree_distribution(G)
print(f"Degree distribution: {p_k}")
```

---

## 2.4 Adjacency Matrix

For mathematical purposes we often represent a network through its **adjacency matrix**. The adjacency matrix of a network with $N$ nodes has $N$ rows and $N$ columns, with elements:

- $A_{ij} = 1$ if there is a link pointing from node $j$ to node $i$
- $A_{ij} = 0$ if nodes $i$ and $j$ are not connected

### Creating a Graph from an Adjacency Matrix

```python
import numpy as np
import rustworkx as rx

# Define adjacency matrix
A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
], dtype=float)

# Create graph from adjacency matrix
G = rx.PyGraph.from_adjacency_matrix(A)
```

### Degree from Adjacency Matrix

**Undirected graphs**: The degree of node $i$ can be computed by summing row or column $i$:

$$k_i = \sum_{j=1}^{N} A_{ij} = \sum_{j=1}^{N} A_{ji}$$

```python
# Degree for each node (sum along rows or columns)
ki = np.sum(A, axis=1)
print(f"Degree for nodes: {ki}")
```

**Directed graphs**: In-degree and out-degree are computed separately:

$$k_i^{in} = \sum_{j=1}^{N} A_{ij} \qquad k_i^{out} = \sum_{j=1}^{N} A_{ji}$$

```python
# For directed graphs
in_degrees = np.sum(A, axis=0)   # sum columns
out_degrees = np.sum(A, axis=1)  # sum rows
```

### Trace Tricks

Matrix powers of the adjacency matrix reveal important network properties:

| Expression | Meaning |
|------------|--------|
| $A^2_{ii}$ | Number of walks of length 2 starting and ending at node $i$ |
| $A^N_{ii}$ | Number of walks of length $N$ starting and ending at node $i$ |
| $\text{Tr}(A^3) / 6$ | Number of triangles in an undirected graph |

```python
# Get adjacency matrix
A = rx.adjacency_matrix(G)

# Walks of length 2 for each node (diagonal of A²)
walks_2 = np.diag(A @ A)
print(f"Walks of length 2: {walks_2}")

# Count triangles in undirected graph
A3 = np.linalg.matrix_power(A, 3)
num_triangles = np.trace(A3) // 6
print(f"Number of triangles: {num_triangles}")
```

```python
# Walks of length N for each node
length = 3
for node, walks in zip(G.node_indexes(), np.diagonal(np.linalg.matrix_power(A, length))):
    print(f"Node {node} has {walks} walks of length {length}")
```

---

## 2.7 Bipartite Graphs

### Key Definitions

| Term | Definition |
|------|------------|
| **Bipartite Graph** | A graph whose nodes can be divided into two disjoint sets $U$ and $V$ such that every edge connects a node in $U$ to a node in $V$ |
| **Two-Coloring** | A valid assignment of two colors to nodes where no adjacent nodes share a color |
| **Projection** | A new graph containing only nodes from one set, with edges connecting nodes that share a common neighbor |

### Bipartite Coloring

```python
import rustworkx as rx

def bipartite_colors(G):
    """
    Return the two-coloring of a bipartite graph.
    Returns (U, V) where U and V are sets of node indices.
    """
    colors = rx.graph_two_color(G)
    U = {n for n, c in colors.items() if c == 0}
    V = {n for n, c in colors.items() if c == 1}
    return U, V

# Check if graph is bipartite
try:
    U, V = bipartite_colors(G)
    print(f"Graph is bipartite: U={U}, V={V}")
except rx.GraphNotBipartite:
    print("Graph is not bipartite")
```

### Bipartite Projection

Two target nodes are connected in the projection if they share a common neighbor in the original graph:

```python
from itertools import combinations

def bipartite_project(B, target_nodes):
    """
    Project a bipartite graph onto a set of target nodes.
    """
    other_nodes = set(B.node_indexes()).difference(target_nodes)
    
    # Create subgraph with only target nodes
    G, n_map = B.subgraph_with_nodemap(list(target_nodes))
    ix_map = {v: k for k, v in n_map.items()}
    
    # Find projected edges through shared neighbors
    projected_edges = set()
    for k in other_nodes:
        neighbors = list(B.neighbors(k))
        edges = set(combinations(neighbors, 2))
        projected_edges.update(edges)
    
    # Add edges to new graph
    new_edges = [(ix_map[i], ix_map[j]) for i, j in projected_edges 
                 if i in ix_map and j in ix_map]
    G.add_edges_from_no_data(new_edges)
    
    return G, n_map
```

---

## 2.8 Paths and Distances

### Key Definitions

| Term | Definition |
|------|------------|
| **Path** | A sequence of nodes where each consecutive pair is connected by an edge |
| **Shortest Path (Geodesic)** | The path with minimum number of edges between two nodes |
| **Distance** ($d_{ij}$) | The length of the shortest path between nodes $i$ and $j$ |
| **Diameter** ($d_{max}$) | The longest shortest path in the network |
| **Average Path Length** ($\langle d \rangle$) | The mean distance between all pairs of nodes |
| **Cycle** | A closed path that starts and ends at the same node |
| **Eulerian Path** | A path that traverses each edge exactly once |
| **Hamiltonian Path** | A path that visits each node exactly once |

### Shortest Paths

```python
# Find all shortest paths between two nodes
source, target = 0, 4
all_paths = rx.all_shortest_paths(G, source, target)

for path in all_paths:
    print(f"Path: {list(path)}, Length: {len(path) - 1}")
```

### Diameter

The diameter is the maximum distance between any pair of nodes:

```python
# Compute diameter using all-pairs shortest paths
apsp = rx.all_pairs_bellman_ford_shortest_paths(G, edge_cost_fn=lambda x: 1)

max_length = 0
for node, mapping in apsp.items():
    for target, path in mapping.items():
        path_len = len(path) - 1
        max_length = max(max_length, path_len)

print(f"Diameter: {max_length}")
```

### Eulerian Paths

An undirected graph has an Eulerian path if and only if:
1. Either 0 or 2 vertices have odd degree
2. All vertices with non-zero degree are in one connected component

```python
def has_eulerian_path(G):
    """Check if graph has an Eulerian path."""
    degrees = [G.degree(n) for n in G.node_indexes()]
    odd_count = sum(d % 2 != 0 for d in degrees)
    components = rx.connected_components(G)
    
    return odd_count in [0, 2] and len(components) == 1
```

### The Königsberg Bridge Problem

Euler proved that it's impossible to walk through Königsberg crossing each bridge exactly once because the graph representing the city has 4 vertices with odd degree.

### Average Path Length

The average path length characterizes how efficiently information spreads through a network:

$$\langle d \rangle = \frac{1}{N(N-1)} \sum_{\substack{i,j = 1 \\ i \ne j}}^{N} d_{i,j}$$

```python
import numpy as np

def average_path_length(G):
    """Calculate the average path length of a graph."""
    dm = rx.distance_matrix(G, null_value=np.inf)
    finite = dm[np.isfinite(dm)]
    finite = finite[finite > 0]
    return finite.mean() if len(finite) > 0 else 0

avg_d = average_path_length(G)
print(f"Average path length ⟨d⟩ = {avg_d:.3f}")
```

---

## 2.9 Clustering Coefficient

The clustering coefficient captures the degree to which the neighbors of a given node link to each other.

### Key Definitions

| Term | Formula | Description |
|------|---------|-------------|
| **Local Clustering Coefficient** | $C_i = \frac{2 L_i}{k_i (k_i - 1)}$ | Fraction of possible edges between node $i$'s neighbors that actually exist |
| **$L_i$** | — | Number of edges between neighbors of node $i$ |
| **$k_i$** | — | Degree of node $i$ |
| **Average Clustering Coefficient** | $\langle C \rangle = \frac{1}{N} \sum_{i=1}^{N} C_i$ | Mean clustering coefficient across all nodes |

### Local Clustering Coefficient

For a node $i$ with degree $k_i$, the local clustering coefficient measures how close its neighbors are to forming a complete graph (clique):

```python
from itertools import combinations
import math

def clustering_coefficient(G, n):
    """Calculate the local clustering coefficient for node n."""
    neighbors = G.neighbors(n)
    k = len(neighbors)
    
    if k < 2:
        return 0.0
    
    # Count edges between neighbors
    edges_between_neighbors = sum(
        G.has_edge(u, v) for u, v in combinations(neighbors, 2)
    )
    
    # Maximum possible edges between neighbors
    possible_edges = math.comb(k, 2)
    
    return edges_between_neighbors / possible_edges
```

### Average Clustering Coefficient

The average clustering coefficient characterizes the overall tendency of nodes to cluster together:

```python
def average_clustering_coefficient(G):
    """Calculate the average clustering coefficient of the graph."""
    N = G.num_nodes()
    return sum(clustering_coefficient(G, i) for i in G.node_indexes()) / N

avg_C = average_clustering_coefficient(G)
print(f"Average clustering coefficient ⟨C⟩ = {avg_C:.3f}")
```

### Interpretation

- $C_i = 0$: None of node $i$'s neighbors are connected to each other
- $C_i = 1$: All of node $i$'s neighbors are connected (form a clique)
- High $\langle C \rangle$: Network has strong local clustering (common in social networks)
- Low $\langle C \rangle$: Neighbors rarely connect (common in random networks)

---

## Summary

| Section | Key Concepts |
|---------|--------------|
| **2.3** | Degree, average degree, degree distribution |
| **2.4** | Adjacency matrix, degree from matrix, trace tricks, counting triangles |
| **2.7** | Bipartite graphs and projections |
| **2.8** | Paths, shortest paths, diameter, cycles, Eulerian/Hamiltonian paths, average path length |
| **2.9** | Local clustering coefficient, average clustering coefficient |

---

## Exercises

1. **Degree Analysis**: Create a random graph with `rx.undirected_gnp_random_graph(20, 0.3)` and compute its degree distribution. Plot the distribution as a histogram.

2. **Bipartite Verification**: Given a graph, write a function that determines if it's bipartite and returns the two-coloring if it exists.

3. **Path Finding**: For a given graph, find all pairs of nodes that are at the maximum distance (diameter) from each other.

4. **Königsberg Variant**: Add one bridge to the Königsberg graph to make an Eulerian path possible. Which nodes should it connect?

5. **Small World**: Generate random graphs of increasing size (N = 10, 50, 100, 500) with edge probability p = 0.1. Plot how the average path length grows with N.

---

*Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási*
