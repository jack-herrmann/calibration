import numpy as np
import matplotlib.pyplot as plt
from baseline import *
from bootstrap import *
from generateSyntheticData import *
from plots import plotCalibrationCurves
from constants import (
    TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH,
    ALPHALEVELS, NUMBERREPS, NUMBERBOOTSTRAP, computeBlockLength
)

def runCalibrationCurveExperiment(phi, rho, alphaLevels=ALPHALEVELS, numReps=NUMBERREPS, numberBootstrap=NUMBERBOOTSTRAP, blockLength=None):
    if blockLength is None:
        blockLength = computeBlockLength(phi)
        print(f"Using block length = {blockLength} (computed from phi={phi})")

    methods = {
        'Bonferroni': bonferroni,
        'Holm': holm,
        'BH': benjaminiHochberg,
        'Bootstrap-Single': None,
        'Bootstrap-RomanoWolf': None
    }

    allResults = {}

    for alpha in alphaLevels:
        print(f"calibration for alpha = {alpha:.3f}...")

        results = {method: {'fwer': [], 'fdr': [], 'power': []} for method in methods}

        for rep in range(numReps):
            data, clusterLabels, isTrue = generateClusteredPanelWithPlantedSignals(
                TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH, phi, rho
            )
            tStats, pVals = computeTestStatistics(data)

            # classical methods
            for methodName, methodFunc in methods.items():
                if 'Bootstrap' in methodName:
                    continue

                rejected = methodFunc(pVals, alpha)
                performance = measurePerformance(rejected, isTrue)

                results[methodName]['fwer'].append(performance['fwer'])
                results[methodName]['fdr'].append(performance['fdr'])
                results[methodName]['power'].append(performance['power'])

            # bootstrap single
            performanceBoot, _, _ = applyBootstrapCalibration(
                data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap
            )
            results['Bootstrap-Single']['fwer'].append(performanceBoot['fwer'])
            results['Bootstrap-Single']['fdr'].append(performanceBoot['fdr'])
            results['Bootstrap-Single']['power'].append(performanceBoot['power'])

            # bootstrap RomanoWolf
            performanceRW, _, _ = applyRomanoWolfBootstrapCalibration(
                data, clusterLabels, isTrue, alpha, blockLength, numberBootstrap
            )
            results['Bootstrap-RomanoWolf']['fwer'].append(performanceRW['fwer'])
            results['Bootstrap-RomanoWolf']['fdr'].append(performanceRW['fdr'])
            results['Bootstrap-RomanoWolf']['power'].append(performanceRW['power'])

        summary = {}
        for methodName in methods:
            summary[methodName] = {
                'fwer_mean': np.mean(results[methodName]['fwer']),
                'fdr_mean': np.mean(results[methodName]['fdr']),
                'power_mean': np.nanmean(results[methodName]['power']),
                'fwer_se': np.std(results[methodName]['fwer']) / np.sqrt(numReps),
                'fdr_se': np.std(results[methodName]['fdr']) / np.sqrt(numReps)
            }

        allResults[alpha] = summary

    return allResults


def createCalibrationTable(calibrationResults, savePath=None):
    rows = []

    for alpha in sorted(calibrationResults.keys()):
        for method, stats in calibrationResults[alpha].items():
            row = {
                'Method': method,
                'Nominal_Alpha': f"{alpha:.3f}",
                'Realized_FWER': f"{stats['fwer_mean']:.3f}",
                'FWER_SE': f"{stats['fwer_se']:.3f}",
                'Realized_FDR': f"{stats['fdr_mean']:.3f}",
                'FDR_SE': f"{stats['fdr_se']:.3f}",
                'Power': f"{stats['power_mean']:.3f}",
                'Calibration_Error': f"{abs(stats['fwer_mean'] - alpha):.3f}"
            }
            rows.append(row)

    dataframe = pd.DataFrame(rows)

    if savePath:
        dataframe.to_csv(savePath, index=False)
        print(f"Table saved: {savePath}")

    return dataframe