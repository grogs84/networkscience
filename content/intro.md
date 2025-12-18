# Network Science: A Practical Companion

Companion notes and Python examples for [Network Science](http://networksciencebook.com/) by Albert-László Barabási.

---

## Overview

This is a hands-on companion to Barabási's *Network Science* textbook. While the original book provides the theoretical foundations, these notes focus on **practical implementation** using Python.

Each chapter includes:
- Key concepts and definitions from the textbook
- Working Python code examples you can run and modify
- Visualizations to build intuition
- Exercises to test your understanding

---

## Prerequisites

To get the most out of these materials, you should have:

- **Basic Python knowledge** — variables, functions, loops, classes
- **Familiarity with NumPy** — array operations, basic linear algebra
- **Some probability/statistics** — distributions, expected values
- **Curiosity about networks!**

No prior graph theory experience is required — we'll build up from the basics.

---

## Installation

### Required packages

```bash
pip install rustworkx networkx matplotlib numpy scipy
```

### Optional packages

```bash
pip install python-igraph pandas seaborn
```

### Running the notebooks

Each chapter has an accompanying Jupyter notebook. You can:

1. **Run locally** after installing packages above
2. **Use Google Colab** — click the "Open in Colab" badge on any chapter

---

## Tools & Libraries

We primarily use these graph libraries:

| Library | Use Case |
|---------|----------|
| **[rustworkx](https://www.rustworkx.org/)** | Primary library — fast, Rust-based graph algorithms |
| **[networkX](https://networkx.org/)** | Secondary — extensive algorithms, great documentation |
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

---

## Table of Contents

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 1 | Introduction | What are networks? Why study them? |
| 2 | Graph Theory | Degree, paths, distances, bipartite graphs |
| 3 | Random Networks | Erdős-Rényi model, phase transitions |
| 4 | Scale-Free Networks | Power laws, hubs |
| 5 | The Barabási-Albert Model | Preferential attachment, growth |
| 6 | Evolving Networks | Network dynamics over time |
| 7 | Degree Correlations | Assortativity, mixing patterns |
| 8 | Network Robustness | Failures, attacks, resilience |
| 9 | Communities | Detection algorithms, modularity |
| 10 | Spreading Phenomena | Epidemics, information diffusion |

---

## How to Use This Book

1. **Read the textbook chapter first** — Barabási's explanations provide essential context
2. **Work through the companion notes** — Reinforce concepts with code
3. **Run the notebooks** — Experiment with parameters, modify examples
4. **Try the exercises** — Apply what you've learned

---

## Acknowledgments

- **Albert-László Barabási** for the excellent [Network Science](http://networksciencebook.com/) textbook
- The **rustworkx**, **networkX**, and **igraph** development teams
- The broader network science community

---

## License & Citation

These companion notes are provided for educational purposes. If you find them useful, please cite the original textbook:

> Barabási, A.-L. (2016). *Network Science*. Cambridge University Press.

---

*Happy exploring!*
