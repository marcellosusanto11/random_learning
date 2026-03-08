# Hidden Markov Model (HMM) Fundamentals

A comprehensive educational resource for understanding Hidden Markov Models through practical examples.

## Overview

This project provides two Jupyter notebooks that explain HMM concepts from basics to advanced applications:

| Notebook | Depth | Example | Best For |
|----------|-------|---------|----------|
| `01_dishonest_casino.ipynb` | Medium 📘 | Classic casino problem | Quick intuition |
| `02_agoda_customer_journey.ipynb` | Deep 📚 | Marketing customer intent | Full understanding |

## What You'll Learn

### Theory
- Markov chains and transition matrices
- Stable state via eigenvalue decomposition
- HMM structure: π (initial), A (transition), B (emission)

### Algorithms
| Algorithm | Purpose | Complexity |
|-----------|---------|------------|
| **Forward** | P(observations \| model) | O(N²T) |
| **Viterbi** | Best state sequence | O(N²T) |
| **Baum-Welch** | Learn parameters | O(N²T) × iterations |

### Practical Skills
- Implementing algorithms from scratch
- Numerical stability (log-space)
- Using `hmmlearn` library
- Real-world applications

## Applications

### Marketing (Agoda example)
- **Multi-touch Attribution**: Which touchpoints drive conversions?
- **Campaign Targeting**: Target users by inferred intent
- **Funnel Optimization**: Find where users get stuck
- **Personalization**: Tailor UX to user's stage

### Other Domains
- Speech recognition (audio → text)
- NLP / POS tagging (words → grammar)
- Finance (returns → market regime)
- Biology (DNA → gene structure)

## Installation

```bash
pip install numpy scipy matplotlib seaborn hmmlearn
```

## Quick Start

```python
from hmm_utils import forward_log, viterbi, baum_welch

# Define HMM
pi = np.array([0.7, 0.2, 0.1])  # Initial distribution
A = np.array([[0.7, 0.2, 0.1], ...])  # Transition matrix
B = np.array([[0.5, 0.3, 0.2], ...])  # Emission matrix

# Compute likelihood
_, log_lik = forward_log(observations, pi, A, B)

# Decode hidden states
best_path, prob = viterbi(observations, pi, A, B)

# Learn parameters
pi, A, B, history = baum_welch(observations, n_states=3, n_obs=5)
```

## File Structure

```
hidden_markov_model_fundamental/
├── 01_dishonest_casino.ipynb      # Medium depth tutorial
├── 02_agoda_customer_journey.ipynb # Deep dive tutorial
├── hmm_utils.py                    # Reusable functions
├── data/                           # Sample data
└── README.md
```

## References

- [Jonathan Hui's HMM Article](https://jonathan-hui.medium.com/machine-learning-hidden-markov-model-hmm-31660d217a61)
- [Rabiner's HMM Tutorial (1989)](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf)
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)

## Author

Created for learning HMM concepts with practical applications in marketing and data science.

---

*Part of the `random_learning` collection.*
