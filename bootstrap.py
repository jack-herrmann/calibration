import numpy as np
from baseline import *
from constants import (
    TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH,
    BASEPHI, BASERHO, ALPHA, NUMBERREPS, NUMBERBOOTSTRAP, BLOCKLENGTH
)

def movingBlockBootstrap(data, blockLength, numberBootstrap):
    time, _ = data.shape
    numBlocks = int(np.ceil(time / blockLength))
    maxStart = time - blockLength
    bootstrapSamples = []
    
    for b in range(numberBootstrap):
        blockStarts = np.random.choice(maxStart + 1, size=numBlocks, replace=True)
        
        bootstrapData = []
        for start in blockStarts:
            block = data[start:start + blockLength, :]
            bootstrapData.append(block)
        
        bootstrapData = np.vstack(bootstrapData)
        bootstrapData = bootstrapData[:time, :]
        bootstrapSamples.append(bootstrapData)
    
    return bootstrapSamples

def clusterBootstrap(data, clusterLabels, numberBootstrap):
    uniqueClusters = np.unique(clusterLabels)
    numberClusters = len(uniqueClusters)

    bootstrapSamples = []

    for b in range(numberBootstrap):
        selectedClusters = np.random.choice(uniqueClusters, size=numberClusters, replace=True)

        bootstrapData = []
        for clusterID in selectedClusters:
            firmIndices = np.where(clusterLabels == clusterID)[0]
            clusterData = data[:, firmIndices]
            bootstrapData.append(clusterData)

        bootstrapData = np.hstack(bootstrapData)
        bootstrapSamples.append(bootstrapData)

    return bootstrapSamples

def movingBlockClusterBootstrap(data, clusterLabels, blockLength, numberBootstrap):
    time, _ = data.shape
    uniqueClusters = np.unique(clusterLabels)
    numberClusters = len(uniqueClusters)

    numBlocks = int(np.ceil(time / blockLength))
    maxStart = time - blockLength

    bootstrapSamples = []

    for b in range(numberBootstrap):

        blockStarts = np.random.choice(maxStart + 1, size=numBlocks, replace=True)
        timeBootstrapData = []
        for start in blockStarts:
            block = data[start:start + blockLength, :]
            timeBootstrapData.append(block)

        timeBootstrapData = np.vstack(timeBootstrapData)[:time, :]

        selectedClusters = np.random.choice(uniqueClusters, size=numberClusters, replace=True)
        bootstrapData = []
        for clusterID in selectedClusters:
            firmIndices = np.where(clusterLabels == clusterID)[0]
            clusterData = timeBootstrapData[:, firmIndices]
            bootstrapData.append(clusterData)

        bootstrapData = np.hstack(bootstrapData)
        bootstrapSamples.append(bootstrapData)

    return bootstrapSamples

def computeBootstrapMaxStats(data, clusterLabels, blockLength, numberBootstrap):
    centeredData = data - np.mean(data, axis=0, keepdims=True)

    bootstrapSamples = movingBlockClusterBootstrap(centeredData, clusterLabels, blockLength, numberBootstrap)
    maxStats = []

    for bootstrapData in bootstrapSamples:
        tStats, _ = computeTestStatistics(bootstrapData)
        maxStat = np.max(np.abs(tStats))
        maxStats.append(maxStat)

    return np.array(maxStats)

# KEFF IS STILL A WORK IN PROGRESS
# compute the effective number of independent tests
def effectiveNumberTests(maxStats, alpha, K, df):
    tStar = np.percentile(maxStats, 100 * (1 - alpha))

    pTwoTail = 2 * (1 - stats.t.cdf(tStar, df=df))
    p_clipped = np.clip(pTwoTail, 1e-12, 1 - 1e-12)
    kEff = np.log1p(-alpha) / np.log1p(-p_clipped)
    kEff = np.clip(kEff, 1.0, float(K))

    return kEff

def applyBootstrapCalibration(data, clusterLabels, isTrue, alpha=ALPHA, blockLength=BLOCKLENGTH, numberBootstrap=NUMBERBOOTSTRAP):
    maxStats = computeBootstrapMaxStats(data, clusterLabels, blockLength, numberBootstrap)
    tStar = np.percentile(maxStats, 100 * (1 - alpha))

    tStats, _ = computeTestStatistics(data) #  test statistics on original data
    K = len(tStats)
    degreesFreedom = data.shape[0] - 1

    kEff = effectiveNumberTests(maxStats, alpha, K, degreesFreedom)

    rejected = np.abs(tStats) > tStar
    performance = measurePerformance(rejected, isTrue)
    performance['tStar'] = tStar
    performance['kEff'] = kEff

    return performance, tStar, rejected

def applyRomanoWolfBootstrapCalibration(data, clusterLabels, isTrue, alpha=ALPHA, blockLength=BLOCKLENGTH, numberBootstrap=NUMBERBOOTSTRAP):
    tStats, _ = computeTestStatistics(data)
    K = len(tStats)

    sortedIndices = np.argsort(np.abs(tStats))[::-1]
    sortedStats = np.abs(tStats)[sortedIndices]

    rejected = np.zeros(K, dtype=bool)

    centeredData = data - np.mean(data, axis=0, keepdims=True)
    bootstrapSamples = movingBlockClusterBootstrap(centeredData, clusterLabels, blockLength, numberBootstrap)

    bootstrapTStats = []
    for bootData in bootstrapSamples:
        bootT, _ = computeTestStatistics(bootData)
        bootstrapTStats.append(np.abs(bootT))
    bootstrapTStats = np.array(bootstrapTStats)

    maxStats = np.max(bootstrapTStats, axis=1)
    degreesFreedom = data.shape[0] - 1
    kEff = effectiveNumberTests(maxStats, alpha, K, degreesFreedom)

    for k in range(K):
        remainingIndices = sortedIndices[k:]

        maxStatsRemaining = np.max(bootstrapTStats[:, remainingIndices], axis=1)
        pAdj = np.mean(maxStatsRemaining >= sortedStats[k])

        if pAdj < alpha:
            rejected[sortedIndices[k]] = True
        else:
            break

    performance = measurePerformance(rejected, isTrue)
    performance['kEff'] = kEff

    return performance, rejected, kEff

def monteCarloWithBootstrap(phi, rho, alpha=ALPHA, numReps=NUMBERREPS, numberBootstrap=NUMBERBOOTSTRAP, blockLength=BLOCKLENGTH):
    methods = {
        'Bonferroni': bonferroni,
        'Holm': holm,
        'BH': benjaminiHochberg,
        'Bootstrap-Single': None,
        'Bootstrap-RomanoWolf': None
    }

    results = {method: {'fwer': [], 'fdr': [], 'power': [], 'kEff': []} for method in methods}
    results['Bootstrap-Single']['tStars'] = []

    for rep in range(numReps):
        data, clusterLabels, isTrue = generateClusteredPanelWithPlantedSignals(
            TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH, phi, rho
        )
        tStats, pVals = computeTestStatistics(data)

        for methodName, methodFunc in methods.items():
            if 'Bootstrap' in methodName:
                continue # needs special handling below

            rejected = methodFunc(pVals, alpha)
            performance = measurePerformance(rejected, isTrue)

            results[methodName]['fwer'].append(performance['fwer'])
            results[methodName]['fdr'].append(performance['fdr'])
            results[methodName]['power'].append(performance['power'])
            results[methodName]['kEff'].append(np.nan)

        performanceBoot, tStar, _ = applyBootstrapCalibration(data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap)
        results['Bootstrap-Single']['fwer'].append(performanceBoot['fwer'])
        results['Bootstrap-Single']['fdr'].append(performanceBoot['fdr'])
        results['Bootstrap-Single']['power'].append(performanceBoot['power'])
        results['Bootstrap-Single']['tStars'].append(tStar)
        results['Bootstrap-Single']['kEff'].append(performanceBoot['kEff'])


        performanceRW, _, _ = applyRomanoWolfBootstrapCalibration(
            data,
            clusterLabels,
            isTrue,
            alpha,
            blockLength,
            numberBootstrap,
        )
        results['Bootstrap-RomanoWolf']['fwer'].append(performanceRW['fwer'])
        results['Bootstrap-RomanoWolf']['fdr'].append(performanceRW['fdr'])
        results['Bootstrap-RomanoWolf']['power'].append(performanceRW['power'])
        results['Bootstrap-RomanoWolf']['kEff'].append(performanceRW['kEff'])


    summary = {}
    for methodName in methods:
        kEffValues = np.array(results[methodName]['kEff'])
        hasKeff = 'Bootstrap' in methodName and kEffValues.size > 0 and np.any(~np.isnan(kEffValues))

        kEffMean = np.nanmean(kEffValues) if hasKeff else np.nan
        kEffCI = np.nanpercentile(kEffValues, [2.5, 97.5]) if hasKeff else np.array([np.nan, np.nan])

        summary[methodName] = {
            'fwer_mean': np.mean(results[methodName]['fwer']),
            'fdr_mean': np.mean(results[methodName]['fdr']),
            'power_mean': np.nanmean(results[methodName]['power']),
            'kEff_mean': kEffMean,
            'fwer_ci': np.percentile(results[methodName]['fwer'], [2.5, 97.5]),
            'fdr_ci': np.percentile(results[methodName]['fdr'], [2.5, 97.5]),
            'power_ci': np.nanpercentile(results[methodName]['power'], [2.5, 97.5]),
            'kEff_ci': kEffCI
        }

        if methodName == 'Bootstrap-Single':
            summary[methodName]['tStar_mean'] = np.mean(results[methodName]['tStars'])
            summary[methodName]['tStar_std'] = np.std(results[methodName]['tStars'])

    return summary

def runFullGridWithBootstrap():
    phiLevels = [0.0, 0.3, 0.6, 0.9]
    rhoLevels = [0.0, 0.3, 0.6, 0.9]

    allResults = []

    # row 0: varying phi
    for phi in phiLevels:
        summary = monteCarloWithBootstrap(phi=phi, rho=BASERHO)

        for method in summary:
            summary[method]['scenario'] = f'phi={phi}'
            summary[method]['varied_param'] = 'phi'
            summary[method]['phi'] = phi
            summary[method]['rho'] = BASERHO

        allResults.append(summary)

    # row 1: varying rho
    for rho in rhoLevels:
        summary = monteCarloWithBootstrap(phi=BASEPHI, rho=rho)

        for method in summary:
            summary[method]['scenario'] = f'rho={rho}'
            summary[method]['varied_param'] = 'rho'
            summary[method]['phi'] = BASEPHI
            summary[method]['rho'] = rho

        allResults.append(summary)

    return allResults

def createSummaryTableWithBootstrap(allResults):

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

            if 'Bootstrap' in method:
                row['Keff'] = f"{stats['kEff_mean']:.2f}"
                row['Keff_CI'] = f"[{stats['kEff_ci'][0]:.2f}, {stats['kEff_ci'][1]:.2f}]"
            else:
                row['Keff'] = '-'
                row['Keff_CI'] = '-'

            if method == 'Bootstrap-Single':
                row['Avg_tStar'] = f"{stats['tStar_mean']:.2f}"
                row['Threshold'] = f"{stats['tStar_mean']:.2f}"
            elif method == 'Bootstrap-RomanoWolf':
                row['Avg_tStar'] = '-'
                row['Threshold'] = 'Adaptive'
            else:
                row['Avg_tStar'] = '-'
                row['Threshold'] = '-'

            rows.append(row)

    dataframe = pd.DataFrame(rows)
    return dataframe
