# HMM Fundamentals - Progress

## Completed (Mar 8, 2026)

### Notebooks Created
- [x] `01_dishonest_casino.ipynb` - Medium depth, classic example
- [x] `02_agoda_customer_journey.ipynb` - Deep dive, marketing application

### Features
- [x] Markov chain basics with eigenvalue analysis
- [x] HMM structure (π, A, B) with visualizations
- [x] Forward algorithm (naive + log-space)
- [x] Viterbi algorithm with backtracking
- [x] Baum-Welch EM algorithm
- [x] Step-by-step manual calculations
- [x] Numerical stability (log-space)
- [x] Comparison with hmmlearn library
- [x] Real-time intent detection example
- [x] Visualization of customer journey

### Files
- `hmm_utils.py` - Reusable algorithm implementations
- `requirements.txt` - numpy, scipy, matplotlib, seaborn, hmmlearn
- `README.md` - Documentation

### Tests
- [x] Both notebooks execute without errors
- [x] Log-likelihood matches hmmlearn
- [x] Viterbi decoding matches hmmlearn

## Applications Covered

### Marketing (Agoda)
- Multi-touch attribution
- Campaign targeting by intent
- Funnel optimization
- Personalization

### Personal Projects
- Spending tracker: infer spending "mood"
- Stock analysis: detect market regime
- Travel agent: infer booking intent
