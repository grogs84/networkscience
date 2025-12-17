# Chapter 3: Random Networks

Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grogs84/networkscience/blob/main/notebooks/Chapter3.ipynb)

---

## Topics Covered

- **3.2** The Random Network Model
- **3.3** Number of Links
- **3.4** Degree Distribution (Binomial & Poisson)

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

## Summary

| Section | Key Concepts |
|---------|--------------|
| **3.2** | Random network models: $G(N,p)$ and $G(N,L)$ |
| **3.3** | Number of links follows binomial distribution; $\langle L \rangle = p \cdot N(N-1)/2$ |
| **3.4** | Degree distribution: Binomial (exact) or Poisson (sparse networks) |

---

## Exercises

1. **Link Distribution**: Generate 1000 random networks with $N=20$ and $p=0.3$. Plot the histogram of the number of links and compare to the theoretical binomial distribution.

2. **Degree Verification**: Create a random network with $N=100$ and $p=0.1$. Compute the empirical degree distribution and compare it to both the binomial and Poisson predictions.

3. **Sparse vs Dense**: For $\langle k \rangle = 10$, plot the binomial and Poisson degree distributions for $N = 50, 100, 500, 1000$. At what $N$ does the Poisson become a good approximation?

4. **Average Degree**: Derive the relationship $\langle k \rangle = p(N-1)$ from $\langle L \rangle = p \cdot N(N-1)/2$.

5. **Network Realizations**: Generate three random networks with the same parameters ($N=15$, $p=0.2$). How much do they differ in number of edges and average degree?

---

*Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási*
