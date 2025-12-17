# Introduction

Companion notes and Python examples for [Network Science](http://networksciencebook.com/) by Albert-László Barabási.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grogs84/networkscience/blob/main/notebooks/intro.ipynb)

---

## About This Book

This is a hands-on companion to Barabási's *Network Science* textbook. While the original book provides the theoretical foundations, these notes focus on **practical implementation** using Python.

Each chapter includes:
- Key concepts and definitions from the textbook
- Working code examples you can run and modify
- Visualizations to build intuition
- Exercises to test your understanding

## Python Libraries

We primarily use these graph libraries:

| Library | Use Case |
|---------|----------|
| **[rustworkx](https://www.rustworkx.org/)** | Primary library — fast, Rust-based graph algorithms |
| **[NetworkX](https://networkx.org/)** | Secondary — extensive algorithms, great documentation |
| **[igraph](https://python.igraph.org/)** | Occasionally — specialized algorithms, C-based performance |

### Why rustworkx?

`rustworkx` offers excellent performance for large graphs while maintaining a Pythonic API. It's particularly well-suited for computationally intensive network analysis.

```python
import rustworkx as rx

# Create a graph
G = rx.PyGraph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from_no_data([(0, 1), (1, 2), (2, 3)])

# Compute shortest paths
paths = rx.all_pairs_shortest_path_lengths(G)
```

## Getting Started

Install the required packages:

```bash
pip install rustworkx networkx matplotlib numpy
```

Optional (for specific chapters):

```bash
pip install python-igraph scipy
```

## Chapters

1. **Introduction** — You are here
2. **Graph Theory** — Degree, paths, distances, bipartite graphs
3. **Random Networks** — Erdős-Rényi model, giant component
4. **Scale-Free Networks** — Power laws, preferential attachment
5. **The Barabási-Albert Model** — Growing networks
6. **Evolving Networks** — Network dynamics
7. **Degree Correlations** — Assortativity
8. **Network Robustness** — Failures and attacks
9. **Communities** — Detection algorithms
10. **Spreading Phenomena** — Epidemics, information diffusion

---

*These notes accompany [Network Science](http://networksciencebook.com/) by Albert-László Barabási*
