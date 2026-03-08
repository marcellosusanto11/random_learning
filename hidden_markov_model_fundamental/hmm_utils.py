"""
HMM Utility Functions
=====================

Reusable implementations of HMM algorithms for educational purposes.
"""

import numpy as np
from scipy.special import logsumexp


def forward_naive(observations, pi, A, B):
    """
    Forward algorithm (naive implementation - may underflow).
    
    Args:
        observations: Sequence of observation indices
        pi: Initial state distribution
        A: Transition matrix
        B: Emission matrix
    
    Returns:
        alpha: Forward probabilities (T x N)
        likelihood: P(observations | model)
    """
    T = len(observations)
    N = len(pi)
    
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, observations[0]]
    
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, observations[t]]
    
    return alpha, np.sum(alpha[-1])


def forward_log(observations, pi, A, B):
    """
    Forward algorithm in log-space (numerically stable).
    
    Args:
        observations: Sequence of observation indices
        pi: Initial state distribution
        A: Transition matrix
        B: Emission matrix
    
    Returns:
        log_alpha: Log forward probabilities (T x N)
        log_likelihood: log P(observations | model)
    """
    T = len(observations)
    N = len(pi)
    
    log_pi = np.log(pi + 1e-300)
    log_A = np.log(A + 1e-300)
    log_B = np.log(B + 1e-300)
    
    log_alpha = np.zeros((T, N))
    log_alpha[0] = log_pi + log_B[:, observations[0]]
    
    for t in range(1, T):
        for j in range(N):
            log_alpha[t, j] = logsumexp(log_alpha[t-1] + log_A[:, j]) + log_B[j, observations[t]]
    
    return log_alpha, logsumexp(log_alpha[-1])


def backward(observations, A, B):
    """
    Backward algorithm.
    
    Returns:
        beta: Backward probabilities (T x N)
    """
    T = len(observations)
    N = A.shape[0]
    
    beta = np.zeros((T, N))
    beta[-1] = 1.0
    
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, observations[t+1]] * beta[t+1])
    
    return beta


def viterbi(observations, pi, A, B):
    """
    Viterbi algorithm - find most likely state sequence.
    
    Returns:
        best_path: Most likely state sequence
        best_prob: Probability of best path
    """
    T = len(observations)
    N = len(pi)
    
    v = np.zeros((T, N))
    backpointers = np.zeros((T, N), dtype=int)
    
    v[0] = pi * B[:, observations[0]]
    
    for t in range(1, T):
        for j in range(N):
            probs = v[t-1] * A[:, j]
            best_prev = np.argmax(probs)
            v[t, j] = probs[best_prev] * B[j, observations[t]]
            backpointers[t, j] = best_prev
    
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = np.argmax(v[-1])
    
    for t in range(T-2, -1, -1):
        best_path[t] = backpointers[t+1, best_path[t+1]]
    
    return best_path, v[-1, best_path[-1]]


def baum_welch(observations, n_states, n_obs, n_iter=100, tol=1e-6):
    """
    Baum-Welch algorithm to learn HMM parameters.
    
    Returns:
        pi, A, B: Learned parameters
        log_likelihoods: Training history
    """
    T = len(observations)
    
    # Random initialization
    pi = np.random.dirichlet(np.ones(n_states) * 2)
    A = np.random.dirichlet(np.ones(n_states) * 2, size=n_states)
    B = np.random.dirichlet(np.ones(n_obs) * 2, size=n_states)
    
    log_likelihoods = []
    
    for iteration in range(n_iter):
        alpha, likelihood = forward_naive(observations, pi, A, B)
        beta = backward(observations, A, B)
        
        log_lik = np.log(likelihood + 1e-300)
        log_likelihoods.append(log_lik)
        
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
        
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-300)
        
        xi = np.zeros((T-1, n_states, n_states))
        for t in range(T-1):
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, observations[t+1]] * beta[t+1, j]
            xi[t] = xi[t] / (xi[t].sum() + 1e-300)
        
        pi = gamma[0]
        
        for i in range(n_states):
            for j in range(n_states):
                A[i, j] = (xi[:, i, j].sum() + 1e-300) / (gamma[:-1, i].sum() + 1e-300)
        
        for j in range(n_states):
            for k in range(n_obs):
                mask = observations == k
                B[j, k] = (gamma[mask, j].sum() + 1e-300) / (gamma[:, j].sum() + 1e-300)
    
    return pi, A, B, log_likelihoods


def generate_sequence(pi, A, B, T):
    """Generate a sequence of states and observations from an HMM."""
    N, M = B.shape
    
    states = np.zeros(T, dtype=int)
    observations = np.zeros(T, dtype=int)
    
    states[0] = np.random.choice(N, p=pi)
    observations[0] = np.random.choice(M, p=B[states[0]])
    
    for t in range(1, T):
        states[t] = np.random.choice(N, p=A[states[t-1]])
        observations[t] = np.random.choice(M, p=B[states[t]])
    
    return states, observations
