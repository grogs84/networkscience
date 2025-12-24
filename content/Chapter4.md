# Chapter 4: The Scale-Free Property

Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grogs84/networkscience/blob/main/notebooks/Chapter4.ipynb)

---

## Topics Covered

- **4.2** Power Laws and Scale-Free Networks
- **4.3** Hubs

---

## 4.2 Power Laws and Scale-Free Networks

Many network quantities (like degree and centrality) exhibit **fat-tailed behavior** consistent with a power-law, meaning there is no characteristic "typical" scale. As a result, large events aren't anomalies—they're a predictable consequence of the distribution's *tail*.

### Discrete Formalism

For discrete degree values $k \in \{1, 2, 3, \ldots\}$, the power-law degree distribution is:

$$p_k = C k^{-\gamma}$$

where $\gamma$ is the **degree exponent** (typically $2 < \gamma < 3$ for real networks).

### Normalization Constant

The constant $C$ is determined by the normalization condition:

$$C \sum_{k=1}^{\infty} k^{-\gamma} = 1$$

This can be expressed in closed form using the **Riemann zeta function**:

$$C = \frac{1}{\zeta(\gamma)}$$

where $\zeta(\gamma) = \sum_{k=1}^{\infty} k^{-\gamma}$.

Thus the discrete power-law (zeta) distribution has the form:

$$p_k = \frac{k^{-\gamma}}{\zeta(\gamma)}$$

```python
import numpy as np
from scipy import special as sp

def discrete_power_pmf(k, gamma=2):
    """Discrete power-law (zeta) probability mass function."""
    return (k ** -gamma) / sp.zeta(gamma, 1)

# Example: probability of degree k=3 with gamma=2
gamma = 2
k = 3
p_k = discrete_power_pmf(k, gamma)
print(f"P(k={k}) = {p_k:.4f}")
```

### Computing the Normalization Constant

```python
gamma = 2
ks = range(1, 1000)

# Sum k^(-gamma) for k = 1 to K
k_gamma_sum = sum(k**-gamma for k in ks)
C = 1.0 / k_gamma_sum

print(f"C (truncated sum) = {C:.6f}")
print(f"1/zeta({gamma}) = {1/sp.zeta(gamma, 1):.6f}")
```

---

## 4.3 Hubs

The main difference between a random and a scale-free network comes in the **tail of the degree distribution**, representing the high-$k$ region of $p_k$.

### Random Networks vs Scale-Free Networks

| Property | Random Network | Scale-Free Network |
|----------|----------------|-------------------|
| Degree distribution | Poisson | Power-law |
| Tail behavior | Exponential decay | Fat tail |
| Hubs | Rare/absent | Common |
| Characteristic scale | Yes ($\langle k \rangle$) | No |

### Comparing Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy import special as sp

def discrete_power_pmf(k, gamma=2.1):
    return (k ** -gamma) / sp.zeta(gamma, 1)

ks = np.arange(1, 50, dtype=float)
poisson_pks = poisson.pmf(ks, mu=11)
power_pks = discrete_power_pmf(ks, gamma=2.1)

plt.figure(figsize=(6, 4))
plt.plot(ks, power_pks, linewidth=2, color='#8e44ad', label='Power-law (γ=2.1)')
plt.plot(ks, poisson_pks, linewidth=2, color='#2ecc71', label='Poisson (⟨k⟩=11)')
plt.xlim(0, 50)
plt.ylim(0, 0.15)
plt.xlabel("k")
plt.ylabel(r"$p_k$")
plt.legend()
plt.tight_layout()
plt.show()
```

The **power-law distribution** has a much heavier tail than the Poisson distribution. This means:

- In random networks, most nodes have degree close to $\langle k \rangle$
- In scale-free networks, there is significant probability of finding nodes with very high degree (**hubs**)

### Why "Scale-Free"?

The term "scale-free" refers to the absence of a characteristic scale in the degree distribution. While Poisson-distributed networks have a well-defined average degree that characterizes most nodes, power-law networks lack this—the distribution looks similar at all scales (self-similar).

---

## Summary

| Section | Key Concepts |
|---------|--------------|
| **4.2** | Power-law degree distribution: $p_k = k^{-\gamma}/\zeta(\gamma)$; no characteristic scale |
| **4.3** | Hubs are common in scale-free networks; fat-tailed distributions vs exponential decay |

---

## Exercises

1. **Normalization**: Verify that the discrete power-law pmf sums to 1 by computing $\sum_{k=1}^{K} p_k$ for increasing values of $K$.

2. **Tail Comparison**: For $k = 100$, compare $p_k$ for a Poisson distribution with $\langle k \rangle = 10$ versus a power-law with $\gamma = 2.5$. How many orders of magnitude do they differ?

3. **Real Networks**: Load a real network dataset and plot its degree distribution on log-log axes. Does it appear to follow a power law?

---

*Companion notes for [Network Science](http://networksciencebook.com/) by Albert-László Barabási*
