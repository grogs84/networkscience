# Chapter 1: Graph Theory Fundamentals

This is a placeholder for Chapter 1, which will cover the fundamentals of graph theory.

## Topics to be covered:

- Basic graph definitions
- Types of graphs (directed, undirected, weighted)
- Graph representations (adjacency matrix, adjacency list)
- Basic graph algorithms
- Path finding algorithms

## Example

Here's a simple example of representing a graph:

```python
# Example: Creating a simple graph using NetworkX
import networkx as nx

# Create an empty graph
G = nx.Graph()

# Add nodes
G.add_nodes_from([1, 2, 3, 4])

# Add edges
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

# Display graph information
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
```

## Exercises

1. Create a simple graph with 5 nodes
2. Calculate the degree of each node
3. Find the shortest path between two nodes

*More content to be added...*
