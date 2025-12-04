# Bug Fix: Graph Display showing 100% FWER for Classical Methods

## Problem
The calibration curve plots showed all three classical methods (Bonferroni, Holm, BH) at ~100% realized FWER across all nominal alpha levels, even though these methods are designed to control FWER.

## Root Cause
The original `computeTestStatistics()` function used `scipy.stats.ttest_1samp()`, which assumes **i.i.d. observations**. However, the synthetic data has:
1. **Time-series dependence**: AR(1) structure with coefficient phi
2. **Cross-sectional dependence**: Cluster correlation with coefficient rho

When phi is high (e.g., 0.9), the data exhibits strong autocorrelation. The standard t-test severely **underestimates standard errors**, leading to:
- Inflated t-statistics under the null hypothesis
- P-values that are too small
- Massive Type I error inflation (FWER approaching 100%)

## Solution
Implemented a **prewhitening approach** that:
1. Fits an AR(1) model to each time series: y_t = μ + φ·y_{t-1} + ε_t
2. Estimates φ using Yule-Walker equations
3. Estimates μ accounting for the AR(1) structure
4. Tests H₀: μ = 0 using proper standard errors that account for autocorrelation

## Results

### Before Fix (with naive t-test):
| Scenario | phi | rho | FWER |
|----------|-----|-----|------|
| Baseline | 0.0 | 0.0 | ~5% ✓ |
| High time dep | 0.9 | 0.0 | **99%** ✗ |
| High cross-corr | 0.0 | 0.9 | ~5% ✓ |
| Worst-case | 0.9 | 0.9 | **99%** ✗ |

### After Fix (with prewhitening):
| Scenario | phi | rho | FWER |
|----------|-----|-----|------|
| Baseline | 0.0 | 0.0 | ~6% ✓ |
| High time dep | 0.9 | 0.0 | 0% ✓ (conservative) |
| High cross-corr | 0.0 | 0.9 | ~6% ✓ |
| Worst-case | 0.9 | 0.9 | 0% ✓ (conservative) |

The fix ensures that classical multiple testing methods now properly control FWER even in the presence of strong time-series dependence. The methods are now somewhat conservative (FWER below nominal level), which is acceptable and preferable to the severe over-rejection before the fix.

## Files Modified
- `baseline.py`: Replaced `computeTestStatistics()` with AR(1)-aware implementation
