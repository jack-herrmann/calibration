# Changes from Main Branch

This branch consolidates two bug fixes from the active development branches:
- `claude/fix-graph-display-bug-01BfZbMNHuGeJQwc8HLdz4V1`
- `claude/review-code-plots-01LkRTMFtQ8fpvBbKrRGNyZG`

---

## Fix 1: T-Statistic Computation (baseline.py)

**Problem:** Classical multiple testing methods (Bonferroni, Holm, BH) showed ~100% FWER across all nominal alpha levels in calibration plots.

**Root Cause:** The original `computeTestStatistics()` function used `scipy.stats.ttest_1samp()`, which assumes i.i.d. observations. However, the synthetic data has:
- Time-series dependence (AR(1) structure with coefficient phi)
- Cross-sectional dependence (cluster correlation with coefficient rho)

When phi is high (e.g., 0.9), the standard t-test severely underestimates standard errors, causing:
- Inflated t-statistics under the null hypothesis
- P-values that are too small
- Type I error inflation (FWER approaching 100%)

**Solution:** Implemented AR(1) prewhitening approach:
1. Fit AR(1) model to each time series using Yule-Walker equations
2. Estimate the mean (mu) accounting for the AR(1) structure
3. Compute prewhitened residuals
4. Test H0: mu = 0 using proper standard errors that account for autocorrelation

**Files Modified:**
- `baseline.py` - Replaced naive t-test with AR(1)-aware implementation

---

## Fix 2: Stability Analysis Signal Preservation (stability_analysis.py)

**Problem:** Stability analysis was centering data before resampling, which destroyed the planted signal structure and made all hypotheses appear unstable.

**Root Cause:** The code was doing:
```python
centeredData = data - np.mean(data, axis=0, keepdims=True)
bootstrapSamples = movingBlockClusterBootstrap(centeredData, ...)
```

This removed the signal means before resampling, so true discoveries couldn't persist.

**Solution:** Resample the ORIGINAL data (not centered) to preserve signal structure:
```python
bootstrapSamples = movingBlockClusterBootstrap(data, clusterLabels, blockLength, numberBootstrap)
```

**Key Concept:** Stability analysis asks "Do rejections from original data persist in resamples?" True signals should have high survivor rates; false positives should be fragile. This requires preserving the original signal structure.

**Files Modified:**
- `stability_analysis.py` - Removed centering, consolidated bootstrap/classical method handling

---

## Summary of Changes

| File | Lines Changed | Description |
|------|---------------|-------------|
| `baseline.py` | +72/-8 | AR(1) prewhitening for t-statistic computation |
| `stability_analysis.py` | +37/-39 | Signal-preserving stability analysis |
| `BUG_FIX_SUMMARY.md` | +44 (new) | Detailed explanation of t-statistic fix |

## Verification

After these fixes:
- Classical methods properly control FWER even with strong time-series dependence
- Stability analysis correctly distinguishes between true discoveries (high survivor rate) and false positives (low survivor rate)
