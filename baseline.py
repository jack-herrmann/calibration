from scipy import stats
import pandas as pd
from generateSyntheticData import *
from constants import (
    TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH,
    BASEPHI, BASERHO, ALPHA, NUMBERREPS, PHI_LEVELS, RHO_LEVELS
)

#compute test statistics using prewhitening to handle autocorrelation
def computeTestStatistics(data):
    T, signals = data.shape
    tStats = np.zeros(signals)
    pVals = np.zeros(signals)

    for i in range(signals):
        y = data[:, i]
        y_mean = np.mean(y)
        y_centered = y - y_mean

        # lag-1 autocovariance and variance
        gamma_0 = np.mean(y_centered ** 2)
        gamma_1 = np.mean(y_centered[1:] * y_centered[:-1])

        # AR(1) coefficient estimate
        if gamma_0 > 0:
            phi_hat = gamma_1 / gamma_0
            # stabilize
            phi_hat = max(-0.99, min(0.99, phi_hat))
        else:
            phi_hat = 0.0

        # estimate mu accounting for AR(1) structure
        mu_hat = y_mean * (1 - phi_hat)

        # compute prewhitened residuals
        epsilon = np.zeros(T)
        epsilon[0] = y[0] - mu_hat / (1 - phi_hat) if abs(phi_hat) < 0.99 else y[0] - y_mean
        for t in range(1, T):
            epsilon[t] = y[t] - mu_hat - phi_hat * y[t-1]

        # standard error of mu under AR(1):
        sigma_sq = np.var(epsilon, ddof=1)

        # effective sample size adjustment for AR(1)
        if abs(phi_hat) < 0.99:
            se_mu = np.sqrt(sigma_sq * (1 + phi_hat) / (T * (1 - phi_hat)))
        else:
            se_mu = np.sqrt(sigma_sq / T) * np.sqrt(T / 2)  # very conservative

        # t-statistic
        if se_mu > 0:
            tStat = mu_hat / se_mu
        else:
            tStat = 0.0

        # two-tailed p-value
        df = max(1, T - 2)
        pVal = 2 * (1 - stats.t.cdf(np.abs(tStat), df))

        tStats[i] = tStat
        pVals[i] = pVal

    return tStats, pVals

# control FWER
def bonferroni(pVals, alpha):
    K = len(pVals)
    threshold = alpha / K
    rejected = pVals < threshold
    return rejected

# strictly more powerful extension of bonferroni
def holm(pVals, alpha):
    K = len(pVals)

    sortedIndices = np.argsort(pVals)
    sortedPvals = pVals[sortedIndices]

    rejected = np.zeros(K, dtype=bool)

    for k in range(K):
        threshold = alpha / (K - k)
        if sortedPvals[k] <= threshold:
            rejected[sortedIndices[k]] = True
        else:
            break

    return rejected

# control FDR
def benjaminiHochberg(pVals, alpha):
    K = len(pVals)

    sortedIndices = np.argsort(pVals)
    sortedPvals = pVals[sortedIndices]

    # find largest k where p_(k) <= k/K * alpha
    rejected = np.zeros(K, dtype=bool)

    for k in range(K, 0, -1):
        threshold = (k / K) * alpha
        if sortedPvals[k-1] <= threshold:
            rejected[sortedIndices[:k]] = True
            break

    return rejected

# just for this one dataset
def measurePerformance(rejected, isTrue):
    truePositives = np.sum(rejected & isTrue)
    falsePositives = np.sum(rejected & ~isTrue)
    totalDiscoveries = np.sum(rejected)
    totalTrue = np.sum(isTrue)

    fwer = 1.0 if falsePositives > 0 else 0.0

    if totalDiscoveries > 0:
        fdr = falsePositives / totalDiscoveries
    else:
        fdr = 0.0

    if totalTrue > 0:
        power = truePositives / totalTrue
    else:
        power = np.nan

    return {
        'fwer': fwer,
        'fdr': fdr,
        'power': power,
        'truePositives': truePositives,
        'falsePositives': falsePositives,
        'totalDiscoveries': totalDiscoveries
    }

def monteCarloMultipleMethods(phi, rho, alpha=ALPHA, numReps=NUMBERREPS):
    methods = {
        'Bonferroni': bonferroni,
        'Holm': holm,
        'BH': benjaminiHochberg
    }

    results = {method: {'fwer': [], 'fdr': [], 'power': []} for method in methods}

    for rep in range(numReps):
        data, clusterLabels, isTrue = generateClusteredPanelWithPlantedSignals(
            TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH, phi, rho
        )

        _, pVals = computeTestStatistics(data)

        for methodName, methodFunc in methods.items():
            rejected = methodFunc(pVals, alpha)
            perf = measurePerformance(rejected, isTrue)

            results[methodName]['fwer'].append(perf['fwer'])
            results[methodName]['fdr'].append(perf['fdr'])
            results[methodName]['power'].append(perf['power'])

    summary = {}
    for methodName in methods:
        summary[methodName] = {
            'fwer_mean': np.mean(results[methodName]['fwer']),
            'fdr_mean': np.mean(results[methodName]['fdr']),
            'power_mean': np.nanmean(results[methodName]['power']),
            'fwer_ci': np.percentile(results[methodName]['fwer'], [2.5, 97.5]),
            'fdr_ci': np.percentile(results[methodName]['fdr'], [2.5, 97.5]),
            'power_ci': np.nanpercentile(results[methodName]['power'], [2.5, 97.5])
        }

    return summary

def runFullGrid():
    allResults = []

    # row 0: varying phi
    for phi in PHI_LEVELS:
        summary = monteCarloMultipleMethods(phi=phi, rho=BASERHO)

        for method in summary:
            summary[method]['scenario'] = f'phi={phi}'
            summary[method]['varied_param'] = 'phi'
            summary[method]['phi'] = phi
            summary[method]['rho'] = BASERHO

        allResults.append(summary)

    # row 1: varying rho
    for rho in RHO_LEVELS:
        summary = monteCarloMultipleMethods(phi=BASEPHI, rho=rho)

        for method in summary:
            summary[method]['scenario'] = f'rho={rho}'
            summary[method]['varied_param'] = 'rho'
            summary[method]['phi'] = BASEPHI
            summary[method]['rho'] = rho

        allResults.append(summary)

    return allResults

def createSummaryTable(allResults, savePath=None):
    rows = []

    for scenarioResults in allResults:
        for method, stats in scenarioResults.items():

            row = {
                'Method': method,
                'Scenario': stats['scenario'],
                'phi': stats['phi'],
                'rho': stats['rho'],
                'alpha': ALPHA,
                'FWER': f"{stats['fwer_mean']:.1%}",
                'FWER_CI': f"[{stats['fwer_ci'][0]:.1%}, {stats['fwer_ci'][1]:.1%}]",
                'FDR': f"{stats['fdr_mean']:.1%}",
                'FDR_CI': f"[{stats['fdr_ci'][0]:.1%}, {stats['fdr_ci'][1]:.1%}]",
                'Power': f"{stats['power_mean']:.1%}",
                'Power_CI': f"[{stats['power_ci'][0]:.1%}, {stats['power_ci'][1]:.1%}]",
            }
            rows.append(row)

    dataframe = pd.DataFrame(rows)

    if savePath:
        dataframe.to_csv(savePath, index=False)
        print(f"Table saved: {savePath}")

    return dataframe
