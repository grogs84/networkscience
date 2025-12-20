# Chapter 3: Random Networks

Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grogs84/networkscience/blob/main/notebooks/Chapter3.ipynb)

---

## Topics Covered

- **3.2** The Random Network Model
- **3.3** Number of Links
- **3.4** Degree Distribution (Binomial & Poisson)
- **3.5** The Evolution of a Random Network
- **Box 3.5** Network Evolution in Graph Theory
- **3.7** Real Networks are Supercritical
- **3.8** Small Worlds

---

## 3.2 The Random Network Model

A **random network** (Erdős–Rényi model) consists of $N$ nodes where each pair of nodes is connected with probability $p$.

### Key Definitions

| Term | Description |
|------|-------------|
| $G(N, p)$ | Each pair of $N$ labeled nodes is connected independently with probability $p$ |
| $G(N, L)$ | $N$ labeled nodes are connected by $L$ randomly placed links |
| $N$ | Number of nodes in the network |
| $p$ | Probability of connection between any two nodes |

### G(N, p) Model

```python
import random
from itertools import combinations

N = 5
p = 0.1

nodes = list(range(N))
possible_edges = [(u, v) for u, v in combinations(nodes, 2)]
edges = [e for e in possible_edges if random.random() < p]

print(f"Number of nodes: {N}")
print(f"Possible edges: {len(possible_edges)}")
print(f"Generated edges: {len(edges)}")
```

### G(N, L) Model

```python
import math
import numpy as np
from itertools import combinations

N = 5
L = 2

nodes = list(range(N))
n_possible_edges = math.comb(N, 2)

# Randomly select L edges
selected_indices = np.random.choice(n_possible_edges, L, replace=False)
edges = [e for i, e in enumerate(combinations(nodes, 2)) if i in selected_indices]

print(f"Generated {len(edges)} edges: {edges}")
```

### Visualizing Random Networks

```python
import rustworkx as rx
from rustworkx.visualization import mpl_draw

N, p = 10, 0.3
G = rx.undirected_gnp_random_graph(N, p)
mpl_draw(G, node_color='#8e44ad', edge_color='#2ecc71')
```

---

## 3.3 Number of Links

The probability that a random network has exactly $L$ links follows a binomial distribution.

### Probability of Exactly L Links

$$p_L = \binom{\frac{N(N-1)}{2}}{L} p^L (1 - p)^{\frac{N(N-1)}{2} - L}$$

Where:
- $\binom{\frac{N(N-1)}{2}}{L}$ is the number of ways to place $L$ links among all possible pairs
- $p^L$ is the probability that $L$ attempts result in links
- $(1-p)^{\frac{N(N-1)}{2} - L}$ is the probability that the remaining attempts do not result in links

```python
import math
import numpy as np

def prob_L_links(N, p, L):
    """Probability that a random network has exactly L links."""
    n_possible = math.comb(N, 2)
    n_ways = math.comb(n_possible, L)
    return n_ways * (p ** L) * ((1 - p) ** (n_possible - L))

# Example
N, p, L = 10, 0.1, 5
print(f"P(L={L}) = {prob_L_links(N, p, L):.4f}")
```

### Expected Number of Links

$$\langle L \rangle = p \cdot \frac{N(N-1)}{2}$$

```python
N, p = 10, 0.1
n_possible = math.comb(N, 2)
expected_links = p * n_possible
print(f"Expected number of links: {expected_links}")
```

### Average Degree

$$\langle k \rangle = \frac{2 \langle L \rangle}{N} = p(N - 1)$$

```python
N, p = 10, 0.25
avg_degree = p * (N - 1)
print(f"Average degree: {avg_degree}")
```

---

## 3.4 Degree Distribution

The degree distribution $p_k$ is the probability that a randomly chosen node has degree $k$.

### 3.4.1 Binomial Distribution

The exact degree distribution of a random network follows the binomial distribution:

$$p_k = \binom{N-1}{k} p^k (1 - p)^{N-1-k}$$

```python
from scipy.stats import binom
import numpy as np

N, p = 20, 0.25
n = N - 1  # Each node can connect to N-1 others
ks = np.arange(0, N)

pk = binom.pmf(ks, n=n, p=p)
```

### 3.4.2 Poisson Approximation

For sparse networks where $\langle k \rangle \ll N$, the degree distribution is well approximated by the Poisson distribution:

$$p_k = e^{-\langle k \rangle} \frac{\langle k \rangle^k}{k!}$$

```python
from scipy.stats import poisson

avg_degree = 5
ks = np.arange(0, 15)

pk_poisson = poisson.pmf(ks, mu=avg_degree)
```

### When to Use Poisson vs Binomial

| Condition | Use |
|-----------|-----|
| $\langle k \rangle \ll N$ | Poisson approximation works well |
| $\langle k \rangle$ close to $N$ | Use exact Binomial distribution |

The Poisson approximation becomes exact in the limit $N \to \infty$ with $\langle k \rangle$ fixed.

---

## 3.5 The Evolution of a Random Network

One would expect that the largest component grows gradually from $N_G = 1$ to $N_G = N$ as $\langle k \rangle$ increases from $0$ to $N-1$. Yet, this is not the case: $N_G/N$ remains zero for small $\langle k \rangle$, indicating the lack of a large cluster. Once $\langle k \rangle$ exceeds a critical value, $N_G/N$ increases, signaling the rapid emergence of a large cluster called the **giant component**.

### ER Graph Regimes

The relationship $\langle k \rangle = p(N-1) \approx pN$ for large $N$ gives us key thresholds:

| Regime | Condition | Description |
|--------|-----------|-------------|
| **Subcritical** | $\langle k \rangle < 1$ ($p < 1/N$) | No giant component |
| **Critical** | $\langle k \rangle = 1$ ($p = 1/N$) | Critical point |
| **Supercritical** | $\langle k \rangle > 1$ ($p > 1/N$) | Giant component emerges |
| **Connected** | $\langle k \rangle > \ln N$ ($p > \ln N / N$) | Graph becomes connected w.h.p. |

```python
import rustworkx as rx

def largest_component_fraction(G):
    """Return the fraction of nodes in the largest connected component."""
    largest = max(rx.connected_components(G), key=len)
    return len(largest) / G.num_nodes()

N = 100
regimes = [
    ("subcritical",   0.5,  "p < 1/N"),
    ("critical",      1.0,  "p = 1/N"),
    ("supercritical", 3.75, "p > 1/N"),
    ("connected",     5.25, "p > ln(N)/N"),
]

for label, avg_degree, threshold in regimes:
    p = avg_degree / (N - 1)
    G = rx.undirected_gnp_random_graph(N, p, seed=42)
    gcc = largest_component_fraction(G)
    print(f"{label:>13} | ⟨k⟩={avg_degree:<4} | GCC={gcc:.2f}")
```

---

## Box 3.5: Network Evolution in Graph Theory

When referring to *threshold probabilities*, we mean the emergence of **specific finite subgraphs** (motifs) as the network grows.

In the Erdős–Rényi model with $p(N) \sim N^{z}$ where $z < 0$, a *threshold* corresponds to the value of $z$ at which a given subgraph appears with high probability as $N \to \infty$.

### The Organizing Principle

A subgraph $H$ appears when the expected number of copies of $H$ transitions from going to zero to diverging as $N \to \infty$.

### Threshold for Paths of Length 2

Consider a **path of length 2** (3 nodes, 2 edges):

- Number of ways to choose 3 nodes: $\sim N^3$
- Probability the two required edges exist: $p^2 \sim N^{2z}$

Expected count scales as:

$$\mathbb{E}[\text{paths of length 2}] \sim N^{3 + 2z}$$

Setting $3 + 2z = 0$ gives $z = -3/2$.

### Threshold Summary

| $z$ value | What emerges |
|-----------|--------------|
| $z < -2$ | No edges |
| $z = -2$ | Isolated edges appear |
| $-2 < z < -3/2$ | Isolated edges dominate |
| $z = -3/2$ | Paths of length 2 appear |
| $-3/2 < z < -1$ | Small trees emerge |
| $z = -1$ | Trees of all orders and cycles appear |

### Trees of Order k

A tree of order $k$ has $k$ nodes and $k-1$ edges. The threshold exponent is:

$$z = -\frac{k}{k-1}$$

Examples:
- $k = 3 \Rightarrow z = -3/2$
- $k = 4 \Rightarrow z = -4/3$

### Motif Detection Functions

```python
from itertools import combinations

def exists_edge(G):
    """Check if any edge exists."""
    return G.num_edges() > 0

def exists_wedge_len2(G):
    """Check if a path of length 2 exists (node with degree >= 2)."""
    return any(G.degree(v) >= 2 for v in G.node_indices())

def exists_3star(G):
    """Check if a 3-star (K_{1,3}) exists."""
    return any(G.degree(v) >= 3 for v in G.node_indices())

def exists_4star(G):
    """Check if a 4-star (K_{1,4}) exists."""
    return any(G.degree(v) >= 4 for v in G.node_indices())

def exists_triangle(G):
    """Check if a triangle exists."""
    nbrs = [set(G.neighbors(i)) for i in G.node_indices()]
    for u, v in G.edge_list():
        if nbrs[u].intersection(nbrs[v]):
            return True
    return False

def exists_K4(G):
    """Check if K4 (complete graph on 4 nodes) exists."""
    nbrs = [set(G.neighbors(i)) for i in G.node_indices()]
    edge_set = {tuple(sorted(e)) for e in G.edge_list()}
    
    for u, v in G.edge_list():
        common = nbrs[u].intersection(nbrs[v])
        if len(common) < 2:
            continue
        for a, b in combinations(common, 2):
            if (min(a, b), max(a, b)) in edge_set:
                return True
    return False
```

### Testing Motif Emergence

```python
import numpy as np
import rustworkx as rx

motif_tests = [
    ("edge", exists_edge),
    ("wedge", exists_wedge_len2),
    ("3-star", exists_3star),
    ("4-star", exists_4star),
    ("triangle", exists_triangle),
    ("K4", exists_K4),
]

zs = -np.array([2, 3/2, 4/3, 5/4, 1, 2/3, 1/2], dtype=float)
Ns = [10**i for i in range(1, 3)]
trials = 20

for z in zs:
    print(f"\n=== z = {z:.3f} ===")
    for N in Ns:
        p = float(N**z)
        
        indicators = np.zeros((trials, len(motif_tests)), dtype=int)
        for t in range(trials):
            G = rx.undirected_gnp_random_graph(N, p, seed=t)
            indicators[t, :] = [int(fn(G)) for _, fn in motif_tests]
        
        mean_met = indicators.sum(axis=1).mean()
        print(f"N={N:>6}, p={p:.2e} | motifs present = {mean_met:.1f}/{len(motif_tests)}")
```

---

## 3.7 Real Networks are Supercritical

Two predictions of random network theory are of direct importance for real networks:

1. Once the average degree exceeds $\langle k \rangle = 1$, a **giant component** should emerge that contains a finite fraction of all nodes. Hence only for $\langle k \rangle > 1$ do nodes organize themselves into a recognizable network.

2. For $\langle k \rangle > \ln N$, all components are absorbed by the giant component, resulting in a **single connected network**.

### Testing with Real Data

```python
import itertools
import rustworkx as rx
from pathlib import Path
import numpy as np

# Load Internet topology data
DATADIR = Path('data')
file = 'internet.edgelist.txt'
dataset = DATADIR / file

with dataset.open() as f:
    data = [tuple(l.strip().split('\t')) for l in f.readlines()]

edges = [tuple(map(int, x)) for x in data]
nodes = list(set(itertools.chain.from_iterable(data)))

G = rx.PyGraph()
G.add_nodes_from(nodes)
G.add_edges_from_no_data(edges)

N = len(nodes)
L = len(edges)
avg_degree = 2 * L / N
lnN = np.log(N)

print('Internet Data')
print(f"Nodes (N):     {N}")
print(f"Edges (L):     {L}")
print(f"⟨k⟩:           {avg_degree:.2f}")
print(f"ln(N):         {lnN:.2f}")
print(f"⟨k⟩ > 1?       {avg_degree > 1}")
print(f"⟨k⟩ > ln(N)?   {avg_degree > lnN}")
```

Real networks consistently satisfy both conditions, confirming they are in the **supercritical regime**.

---

## 3.8 Small Worlds

The "small world" phenomenon means that the average path length depends **logarithmically** on the system size. That is, the average distance $\langle d \rangle$ between two nodes scales as $\ln N$ rather than $N$.

### Counting Nodes at Distance d

The expected number of nodes within distance $d$ from a starting node (in a tree-like approximation):

$$N(d) \approx 1 + \langle k \rangle + \langle k \rangle^2 + \cdots + \langle k \rangle^d = \frac{\langle k \rangle^{d+1} - 1}{\langle k \rangle - 1}$$

```python
def nodes_within_distance(k: float, d: int) -> float:
    """
    Expected number of nodes within distance d in the branching approximation:
        1 + k + k^2 + ... + k^d
    """
    if np.isclose(k, 1.0):
        return d + 1
    return (k**(d + 1) - 1) / (k - 1)
```

### Deriving Average Distance

Since $N(d)$ cannot exceed $N$, we set $N(d_{max}) \approx N$. For $\langle k \rangle \gg 1$:

$$\langle k \rangle^{d_{max}} \approx N$$

Taking logarithms:

$$d_{max} \approx \frac{\ln N}{\ln \langle k \rangle}$$

This better approximates the **average distance** $\langle d \rangle$ than the diameter $d_{max}$, because the diameter is dominated by a few extreme paths while $\langle d \rangle$ averages over all node pairs:

$$\langle d \rangle \approx \frac{\ln N}{\ln \langle k \rangle}$$

### Verifying the Small World Formula

```python
import rustworkx as rx
import numpy as np

def avg_degree_fn(G):
    return 2 * G.num_edges() / G.num_nodes()

def graph_diameter(G):
    """Compute diameter using all-pairs shortest paths."""
    apsp = rx.all_pairs_dijkstra_path_lengths(G, edge_cost_fn=lambda _: 1)
    return max(dist for mapping in apsp.values() for dist in mapping.values())

N, p = 1000, 0.01
G = rx.undirected_gnp_random_graph(N, p, seed=67)

avg_degree = avg_degree_fn(G)
diameter = graph_diameter(G)
empirical_avg_d = rx.unweighted_average_shortest_path_length(G)
predicted_avg_d = np.log(G.num_nodes()) / np.log(avg_degree)

print(f"Number of nodes: {G.num_nodes():,}")
print(f"Average degree: {avg_degree:.3f}")
print(f"Diameter (max shortest path): {diameter}")
print(f"Measured ⟨d⟩: {empirical_avg_d:.3f}")
print(f"Predicted ⟨d⟩ ≈ ln(N)/ln⟨k⟩: {predicted_avg_d:.3f}")
```

### Practical Implications

The small world formula allows us to estimate path lengths in large networks without computing all pairwise distances. This is the mathematical basis for phenomena like "six degrees of separation" — using knowledge of the average degree and network size, we can predict that social networks have short paths connecting any two people.

---

## Summary

| Section | Key Concepts |
|---------|--------------|
| **3.2** | Random network models: $G(N,p)$ and $G(N,L)$ |
| **3.3** | Number of links follows binomial distribution; $\langle L \rangle = p \cdot N(N-1)/2$ |
| **3.4** | Degree distribution: Binomial (exact) or Poisson (sparse networks) |
| **3.5** | Giant component emerges at $\langle k \rangle = 1$; connected at $\langle k \rangle > \ln N$ |
| **Box 3.5** | Subgraph thresholds: $p(N) \sim N^z$ determines which motifs appear |
| **3.7** | Real networks are supercritical: $\langle k \rangle > 1$ and often $\langle k \rangle > \ln N$ |
| **3.8** | Small worlds: $\langle d \rangle \approx \ln N / \ln \langle k \rangle$ |

---

## Exercises

1. **Link Distribution**: Generate 1000 random networks with $N=20$ and $p=0.3$. Plot the histogram of the number of links and compare to the theoretical binomial distribution.

2. **Degree Verification**: Create a random network with $N=100$ and $p=0.1$. Compute the empirical degree distribution and compare it to both the binomial and Poisson predictions.

3. **Sparse vs Dense**: For $\langle k \rangle = 10$, plot the binomial and Poisson degree distributions for $N = 50, 100, 500, 1000$. At what $N$ does the Poisson become a good approximation?

4. **Average Degree**: Derive the relationship $\langle k \rangle = p(N-1)$ from $\langle L \rangle = p \cdot N(N-1)/2$.

5. **Network Realizations**: Generate three random networks with the same parameters ($N=15$, $p=0.2$). How much do they differ in number of edges and average degree?

---

*Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási*
