import numpy as np
import pandas as pd
from baseline import *
from bootstrap import *
from generateSyntheticData import *
from constants import (
    TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH,
    ALPHALEVELS, NUMBERREPS_STABILITY, NUMBERBOOTSTRAP, computeBlockLength
)

def analyzeDiscoveryStability(data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap=NUMBERBOOTSTRAP, method='Bootstrap-RomanoWolf'):
    K = data.shape[1]

    # original data rejections
    tStats, pVals = computeTestStatistics(data)

    if method == 'Bonferroni':
        originalRejected = bonferroni(pVals, alpha)
    elif method == 'Holm':
        originalRejected = holm(pVals, alpha)
    elif method == 'BH':
        originalRejected = benjaminiHochberg(pVals, alpha)
    elif method == 'Bootstrap-Single':
        _, _, originalRejected = applyBootstrapCalibration(data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap)
    elif method == 'Bootstrap-RomanoWolf':
        _, originalRejected, _ = applyRomanoWolfBootstrapCalibration(data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap)
    else:
        raise ValueError(f"Unknown method: {method}")

    if 'Bootstrap' in method:
        centeredData = data - np.mean(data, axis=0, keepdims=True)
        bootstrapSamples = movingBlockClusterBootstrap(centeredData, clusterLabels, blockLength, numberBootstrap)

        rejectionMatrix = np.zeros((numberBootstrap, K), dtype=bool)
        pvalMatrix = np.zeros((numberBootstrap, K))

        for b, bootData in enumerate(bootstrapSamples):
            bootTStats, bootPVals = computeTestStatistics(bootData)
            pvalMatrix[b, :] = bootPVals

            if method == 'Bootstrap-Single':
                _, tStar, _ = applyBootstrapCalibration(data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap)
                rejectionMatrix[b, :] = np.abs(bootTStats) > tStar
            elif method == 'Bootstrap-RomanoWolf':
                maxStat = np.max(np.abs(bootTStats))
                threshold = np.percentile(np.abs(bootTStats), 100 * (1 - alpha))
                rejectionMatrix[b, :] = np.abs(bootTStats) > threshold
    else:
        centeredData = data - np.mean(data, axis=0, keepdims=True)
        bootstrapSamples = movingBlockClusterBootstrap(centeredData, clusterLabels, blockLength, numberBootstrap)

        rejectionMatrix = np.zeros((numberBootstrap, K), dtype=bool)
        pvalMatrix = np.zeros((numberBootstrap, K))

        for b, bootData in enumerate(bootstrapSamples):
            _, bootPVals = computeTestStatistics(bootData)
            pvalMatrix[b, :] = bootPVals

            if method == 'Bonferroni':
                rejectionMatrix[b, :] = bonferroni(bootPVals, alpha)
            elif method == 'Holm':
                rejectionMatrix[b, :] = holm(bootPVals, alpha)
            elif method == 'BH':
                rejectionMatrix[b, :] = benjaminiHochberg(bootPVals, alpha)

    # compute stability metrics
    survivorRate = np.mean(rejectionMatrix, axis=0)
    iqrBootP = np.percentile(pvalMatrix, 75, axis=0) - np.percentile(pvalMatrix, 25, axis=0)

    trueAltSurvivor = survivorRate[isTrue]
    nullSurvivor = survivorRate[~isTrue]
    trueAltIQR = iqrBootP[isTrue]
    nullIQR = iqrBootP[~isTrue]

    return {
        'originalRejected': originalRejected,
        'survivorRate': survivorRate,
        'iqrBootP': iqrBootP,
        'trueAltSurvivorMean': np.mean(trueAltSurvivor) if len(trueAltSurvivor) > 0 else np.nan,
        'nullSurvivorMean': np.mean(nullSurvivor) if len(nullSurvivor) > 0 else np.nan,
        'trueAltIQRMean': np.mean(trueAltIQR) if len(trueAltIQR) > 0 else np.nan,
        'nullIQRMean': np.mean(nullIQR) if len(nullIQR) > 0 else np.nan,
        'trueAltSurvivorMedian': np.median(trueAltSurvivor) if len(trueAltSurvivor) > 0 else np.nan,
        'nullSurvivorMedian': np.median(nullSurvivor) if len(nullSurvivor) > 0 else np.nan,
    }

def createStabilityTable(stabilityResults, savePath=None):
    rows = []

    for (alpha, method), stats in stabilityResults.items():
        row = {
            'Method': method,
            'Alpha': f"{alpha:.3f}",
            'TrueAlt_Survivor%': f"{stats['trueAltSurvivorMean']:.1%}",
            'TrueAlt_Survivor_SE': f"{stats['trueAltSurvivorSE']:.3f}",
            'Null_Survivor%': f"{stats['nullSurvivorMean']:.1%}",
            'Null_Survivor_SE': f"{stats['nullSurvivorSE']:.3f}",
            'TrueAlt_IQR': f"{stats['trueAltIQRMean']:.3f}",
            'Null_IQR': f"{stats['nullIQRMean']:.3f}",
            'Stability_Ratio': f"{stats['trueAltSurvivorMean'] / max(stats['nullSurvivorMean'], 0.001):.2f}"
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    df['Alpha_num'] = df['Alpha'].astype(float)
    method_order = {'Bonferroni': 0, 'Holm': 1, 'BH': 2, 'Bootstrap-Single': 3, 'Bootstrap-RomanoWolf': 4}
    df['Method_order'] = df['Method'].map(method_order)
    df = df.sort_values(['Alpha_num', 'Method_order']).drop(['Alpha_num', 'Method_order'], axis=1)

    if savePath:
        df.to_csv(savePath, index=False)
        print(f"✓ Table saved: {savePath}")

    return df

def runStabilityExperiment(phi, rho, alphaLevels=ALPHALEVELS, numReps=NUMBERREPS_STABILITY, numberBootstrap=NUMBERBOOTSTRAP, blockLength=None):
    # Compute optimal block length based on phi if not provided
    if blockLength is None:
        blockLength = computeBlockLength(phi)
        print(f"Using block length = {blockLength} (computed from phi={phi})")

    methods = ['Bonferroni', 'Holm', 'BH', 'Bootstrap-Single', 'Bootstrap-RomanoWolf']

    allResults = {}

    for alpha in alphaLevels:
        print(f"Stability analysis for alpha = {alpha:.3f}...")

        for method in methods:
            results = {
                'trueAltSurvivorMean': [],
                'nullSurvivorMean': [],
                'trueAltIQRMean': [],
                'nullIQRMean': [],
                'trueAltSurvivorMedian': [],
                'nullSurvivorMedian': []
            }

            for rep in range(numReps):
                data, clusterLabels, isTrue = generateClusteredPanelWithPlantedSignals(
                    TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH, phi, rho
                )

                stability = analyzeDiscoveryStability(data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap, method)

                results['trueAltSurvivorMean'].append(stability['trueAltSurvivorMean'])
                results['nullSurvivorMean'].append(stability['nullSurvivorMean'])
                results['trueAltIQRMean'].append(stability['trueAltIQRMean'])
                results['nullIQRMean'].append(stability['nullIQRMean'])
                results['trueAltSurvivorMedian'].append(stability['trueAltSurvivorMedian'])
                results['nullSurvivorMedian'].append(stability['nullSurvivorMedian'])

            summary = {
                'trueAltSurvivorMean': np.nanmean(results['trueAltSurvivorMean']),
                'nullSurvivorMean': np.nanmean(results['nullSurvivorMean']),
                'trueAltIQRMean': np.nanmean(results['trueAltIQRMean']),
                'nullIQRMean': np.nanmean(results['nullIQRMean']),
                'trueAltSurvivorMedian': np.nanmean(results['trueAltSurvivorMedian']),
                'nullSurvivorMedian': np.nanmean(results['nullSurvivorMedian']),
                'trueAltSurvivorSE': np.nanstd(results['trueAltSurvivorMean']) / np.sqrt(numReps),
                'nullSurvivorSE': np.nanstd(results['nullSurvivorMean']) / np.sqrt(numReps),
            }

            allResults[(alpha, method)] = summary
    return allResults
