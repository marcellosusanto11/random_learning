# Progress — random_learning

## 20 Feb 2026

### prognostic_score_matching

**Status:** Active development — metric framework & causal diagnostics (PR #1)

**Today's work:**
- Replaced log-loss with multi-metric evaluation (MAE on probability, Brier Score, Risk Difference) for classification
- Implemented Austin's Caliper Rule for data-driven threshold selection (regression + classification)
- Added bootstrap 95% confidence intervals for all improvement metrics
- Added prognostic score quality diagnostics (CatBoost vs LGBM)
- Added smart decision variable grouping (auto-detect categorical vs continuous via dtype/nunique)
- Fixed all ruff lint issues and pushed via feature branch PR #1

**Project state:**
- Notebook (`prognostic_score_matching.ipynb`) — 44 cells covering regression (house prices) and classification (adult income)
- Regression: CatBoost + LGBM with threshold sweep, bootstrap CIs, summary interpretation
- Classification: Multi-metric evaluation (MAE + Brier + Risk Difference), threshold sweep, quality diagnostics, decision variable grouping
- All ruff checks passing

**Next steps:**
- Test with different decision variables (age, occupation) to validate framework
- Consider adding cross-validation for more robust threshold selection
- Explore sensitivity analysis for `n_decision_groups` parameter

---

## 18 Feb 2026

### shapley_value_break

**Status:** Active development — code quality & documentation phase

**Today's work:**
- Refactored code structure (PR #2)
- Fixed ruff lint errors across notebook and `greedy_iterative.py` module (PR #3)
- Added descriptive import comment to notebook (PR #4)
- Added function comment to notebook cell (PR #5)

**Project state:**
- Notebook (`shapley_values_break.ipynb`) — 47 cells covering Parts 1-6: toy model, breaking Shapley with duplicated features, Grouped Shapley, Greedy Iterative Selection, healthcare application, and summary
- Standalone module (`greedy_iterative.py`) — reusable `greedy_iterative_shapley()` function with auto-detection for tree-based models
- All ruff checks passing
- README and PLAN_AND_SUMMARY.md fully documented

**Next steps:**
- Consider adding unit tests for `greedy_iterative.py`
- Explore additional datasets or model types for validation
- Potentially extract `GroupedShapley` class into its own module
