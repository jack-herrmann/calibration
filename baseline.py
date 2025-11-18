from scipy import stats
import pandas as pd
from generateSyntheticData import *
from constants import (
    TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH,
    BASEPHI, BASERHO, ALPHA, NUMBERREPS
)

def computeTestStatistics(data):
    _, signals = data.shape
    tStats = np.zeros(signals)
    pVals = np.zeros(signals)
    
    for i in range(signals):
        signal = data[:, i]
        
        # deviation from mean, 0 in this null hypothesis case
        tStat, pVal = stats.ttest_1samp(signal, 0)
        
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
    phiLevels = [0.0, 0.3, 0.6, 0.9]
    rhoLevels = [0.0, 0.3, 0.6, 0.9]

    allResults = []

    # row 0: varying phi
    for phi in phiLevels:
        summary = monteCarloMultipleMethods(phi=phi, rho=BASERHO)

        for method in summary:
            summary[method]['scenario'] = f'phi={phi}'
            summary[method]['varied_param'] = 'phi'
            summary[method]['phi'] = phi
            summary[method]['rho'] = BASERHO

        allResults.append(summary)

    # row 1: varying rho
    for rho in rhoLevels:
        summary = monteCarloMultipleMethods(phi=BASEPHI, rho=rho)

        for method in summary:
            summary[method]['scenario'] = f'rho={rho}'
            summary[method]['varied_param'] = 'rho'
            summary[method]['phi'] = BASEPHI
            summary[method]['rho'] = rho

        allResults.append(summary)

    return allResults

def createSummaryTable(allResults):
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
    return dataframe
