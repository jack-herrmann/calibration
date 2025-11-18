import numpy as np
import matplotlib.pyplot as plt
from constants import PERIOD

# Plot function from generateSyntheticData.py
def plotDatasets(datasets, periods=PERIOD):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    phiLevels = [0.0, 0.3, 0.6, 0.9]
    rhoLevels = [0.0, 0.3, 0.6, 0.9]

    colors = plt.cm.tab10(range(10))  # up to 10 different cluster colors

    # row 0: varying phi
    for col, (data, clusterLabels, isTrue) in enumerate(datasets[0]):
        ax = axes[0, col]

        numberClusters = len(np.unique(clusterLabels))

        for clusterIdx in range(numberClusters):
            clusterFirms = np.where(clusterLabels == clusterIdx)[0]

            trueFirmsInCluster = [f for f in clusterFirms if isTrue[f]]
            nullFirmsInCluster = [f for f in clusterFirms if not isTrue[f]]

            if len(trueFirmsInCluster) > 0:
                firmIdx = trueFirmsInCluster[0]
                ax.plot(data[:periods, firmIdx],
                       color=colors[clusterIdx],
                       linestyle='-',
                       linewidth=1.5,
                       alpha=0.8,
                       label=f'Cluster {clusterIdx} (true)' if col == 0 else '')

            if len(nullFirmsInCluster) > 0:
                firmIdx = nullFirmsInCluster[0]
                ax.plot(data[:periods, firmIdx],
                       color=colors[clusterIdx],
                       linestyle='--',
                       linewidth=1.5,
                       alpha=0.6,
                       label=f'Cluster {clusterIdx} (null)' if col == 0 else '')

        ax.set_title(f'φ={phiLevels[col]}', fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        if col == 0:
            ax.set_ylabel('Vary Time Dep', fontweight='bold')
            ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)

    # row 1: varying rho
    for col, (data, clusterLabels, isTrue) in enumerate(datasets[1]):
        ax = axes[1, col]

        numberClusters = len(np.unique(clusterLabels))

        for clusterIdx in range(numberClusters):
            clusterFirms = np.where(clusterLabels == clusterIdx)[0]

            trueFirmsInCluster = [f for f in clusterFirms if isTrue[f]]
            nullFirmsInCluster = [f for f in clusterFirms if not isTrue[f]]

            if len(trueFirmsInCluster) > 0:
                firmIdx = trueFirmsInCluster[0]
                ax.plot(data[:periods, firmIdx],
                       color=colors[clusterIdx],
                       linestyle='-',
                       linewidth=1.5,
                       alpha=0.8,
                       label=f'Cluster {clusterIdx} (true)' if col == 0 else '')

            if len(nullFirmsInCluster) > 0:
                firmIdx = nullFirmsInCluster[0]
                ax.plot(data[:periods, firmIdx],
                       color=colors[clusterIdx],
                       linestyle='--',
                       linewidth=1.5,
                       alpha=0.6,
                       label=f'Cluster {clusterIdx} (null)' if col == 0 else '')

        ax.set_title(f'ρ={rhoLevels[col]}', fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        if col == 0:
            ax.set_ylabel('Vary Cluster Corr', fontweight='bold')
            ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.show()

# Plot functions from baseline.py
def plotFWERvsDependence(allResults):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['Bonferroni', 'Holm', 'BH']
    colors = {'Bonferroni': 'blue', 'Holm': 'green', 'BH': 'orange'}

    # plot 1: varying phi
    phiResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'phi']

    for method in methods:
        phis = [list(r.values())[0]['phi'] for r in phiResults]
        fwers = [r[method]['fwer_mean'] for r in phiResults]

        ax1.plot(phis, fwers, marker='o', label=method, color=colors[method], linewidth=2)

    ax1.axhline(y=0.05, color='red', linestyle='--', label='Nominal 5%', linewidth=2)
    ax1.set_xlabel('Time Dependence (φ)', fontsize=12)
    ax1.set_ylabel('FWER', fontsize=12)
    ax1.set_title('FWER vs Time Dependence', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])

    # plot 2: varying rho
    rhoResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'rho']

    for method in methods:
        rhos = [list(r.values())[0]['rho'] for r in rhoResults]
        fwers = [r[method]['fwer_mean'] for r in rhoResults]

        ax2.plot(rhos, fwers, marker='o', label=method, color=colors[method], linewidth=2)

    ax2.axhline(y=0.05, color='red', linestyle='--', label='Nominal 5%', linewidth=2)
    ax2.set_xlabel('Cross-Sectional Correlation (ρ)', fontsize=12)
    ax2.set_ylabel('FWER', fontsize=12)
    ax2.set_title('FWER vs Correlation', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.show()

def plotPowerVsDependence(allResults):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['Bonferroni', 'Holm', 'BH']
    colors = {'Bonferroni': 'blue', 'Holm': 'green', 'BH': 'orange'}

    # plot 1: varying phi
    phiResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'phi']

    for method in methods:
        phis = [list(r.values())[0]['phi'] for r in phiResults]
        powers = [r[method]['power_mean'] for r in phiResults]

        ax1.plot(phis, powers, marker='o', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax1.set_xlabel('Time Dependence (φ)', fontsize=12)
    ax1.set_ylabel('Power', fontsize=12)
    ax1.set_title('Power vs Time Dependence', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])

    # plot 2: varying rho
    rhoResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'rho']

    for method in methods:
        rhos = [list(r.values())[0]['rho'] for r in rhoResults]
        powers = [r[method]['power_mean'] for r in rhoResults]

        ax2.plot(rhos, powers, marker='o', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax2.set_xlabel('Cross-Sectional Correlation (ρ)', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_title('Power vs Correlation', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.show()

def plotFWERvsPowerDetailed(allResults):
    fig, ax = plt.subplots(figsize=(10, 8))

    methods = ['Bonferroni', 'Holm', 'BH']
    colors = {'Bonferroni': 'blue', 'Holm': 'green', 'BH': 'orange'}

    jitterX = 0.003
    jitterY = 0.015

    for method in methods:
        phiFWER, phiPower = [], []
        rhoFWER, rhoPower = [], []

        for scenarioResults in allResults:
            variedParam = scenarioResults[method]['varied_param']
            fwer = scenarioResults[method]['fwer_mean']
            power = scenarioResults[method]['power_mean']

            if variedParam == 'phi':
                phiFWER.append(fwer)
                phiPower.append(power)
            else:
                rhoFWER.append(fwer)
                rhoPower.append(power)

        phiFWER_jittered = np.array(phiFWER) + np.random.uniform(-jitterX, jitterX, len(phiFWER))
        phiPower_jittered = np.array(phiPower) + np.random.uniform(-jitterY, jitterY, len(phiPower))
        rhoFWER_jittered = np.array(rhoFWER) + np.random.uniform(-jitterX, jitterX, len(rhoFWER))
        rhoPower_jittered = np.array(rhoPower) + np.random.uniform(-jitterY, jitterY, len(rhoPower))

        ax.scatter(phiFWER_jittered, phiPower_jittered, label=f'{method} (vary φ)',
                  color=colors[method], marker='o', s=100,
                  alpha=0.7, edgecolors='black', linewidth=1.5)

        ax.scatter(rhoFWER_jittered, rhoPower_jittered, label=f'{method} (vary ρ)',
                  color=colors[method], marker='s', s=100,
                  alpha=0.7, edgecolors='black', linewidth=1.5)

    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2,
              label='Target FWER = 5%')

    ax.axvspan(0, 0.05, alpha=0.15, color='green', label='Acceptable FWER region')

    ax.set_xlabel('FWER (Family-Wise Error Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (True Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Error Control vs Power Trade-off\n(circles=vary φ, squares=vary ρ)',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.show()



# Plot functions from bootstrap.py
def plotFWERvsDependenceWithBootstrap(allResults):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['Bonferroni', 'Holm', 'BH', 'Bootstrap-Single', 'Bootstrap-RomanoWolf']
    colors = {
        'Bonferroni': 'blue',
        'Holm': 'green',
        'BH': 'orange',
        'Bootstrap-Single': 'purple',
        'Bootstrap-RomanoWolf': 'red'
    }
    linewidths = {
        'Bonferroni': 2,
        'Holm': 2,
        'BH': 2,
        'Bootstrap-Single': 2,
        'Bootstrap-RomanoWolf': 2
    }

    phiLevels = [0.0, 0.3, 0.6, 0.9]
    rhoLevels = [0.0, 0.3, 0.6, 0.9]

    # plot 1: varying phi
    phiResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'phi']

    for method in methods:
        phis = [list(r.values())[0]['phi'] for r in phiResults]
        fwers = [r[method]['fwer_mean'] for r in phiResults]

        ax1.plot(phis, fwers, marker='o', label=method,
                color=colors[method], linewidth=linewidths[method])

    ax1.axhline(y=0.05, color='red', linestyle='--', label='Nominal 5%', linewidth=2)
    ax1.set_xlabel('Time Dependence (φ)', fontsize=12)
    ax1.set_ylabel('FWER', fontsize=12)
    ax1.set_title('FWER vs Time Dependence', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])

    # plot 2: varying rho
    rhoResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'rho']

    for method in methods:
        rhos = [list(r.values())[0]['rho'] for r in rhoResults]
        fwers = [r[method]['fwer_mean'] for r in rhoResults]

        ax2.plot(rhos, fwers, marker='o', label=method,
                color=colors[method], linewidth=linewidths[method])

    ax2.axhline(y=0.05, color='red', linestyle='--', label='Nominal 5%', linewidth=2)
    ax2.set_xlabel('Cross-Sectional Correlation (ρ)', fontsize=12)
    ax2.set_ylabel('FWER', fontsize=12)
    ax2.set_title('FWER vs Correlation', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.show()

def plotPowerVsDependenceWithBootstrap(allResults):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = ['Bonferroni', 'Holm', 'BH', 'Bootstrap-Single', 'Bootstrap-RomanoWolf']
    colors = {
        'Bonferroni': 'blue',
        'Holm': 'green',
        'BH': 'orange',
        'Bootstrap-Single': 'purple',
        'Bootstrap-RomanoWolf': 'red'
    }

    # plot 1: varying phi
    phiResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'phi']

    for method in methods:
        phis = [list(r.values())[0]['phi'] for r in phiResults]
        powers = [r[method]['power_mean'] for r in phiResults]

        ax1.plot(phis, powers, marker='o', label=method, color=colors[method], linewidth=2)

    ax1.set_xlabel('Time Dependence (φ)', fontsize=12)
    ax1.set_ylabel('Power', fontsize=12)
    ax1.set_title('Power vs Time Dependence', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])

    # plot 2: varying rho
    rhoResults = [r for r in allResults if list(r.values())[0]['varied_param'] == 'rho']

    for method in methods:
        rhos = [list(r.values())[0]['rho'] for r in rhoResults]
        powers = [r[method]['power_mean'] for r in rhoResults]

        ax2.plot(rhos, powers, marker='o', label=method, color=colors[method], linewidth=2)

    ax2.set_xlabel('Cross-Sectional Correlation (ρ)', fontsize=12)
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_title('Power vs Correlation', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.show()

def plotFWERvsPowerWithBootstrap(allResults):
    fig, ax = plt.subplots(figsize=(10, 8))

    methods = ['Bonferroni', 'Holm', 'BH', 'Bootstrap-Single', 'Bootstrap-RomanoWolf']
    colors = {
        'Bonferroni': 'blue',
        'Holm': 'green',
        'BH': 'orange',
        'Bootstrap-Single': 'purple',
        'Bootstrap-RomanoWolf': 'red'
    }

    jitterX = 0.003
    jitterY = 0.015

    for method in methods:
        phiFWER, phiPower = [], []
        rhoFWER, rhoPower = [], []

        for scenarioResults in allResults:
            variedParam = scenarioResults[method]['varied_param']
            fwer = scenarioResults[method]['fwer_mean']
            power = scenarioResults[method]['power_mean']

            if variedParam == 'phi':
                phiFWER.append(fwer)
                phiPower.append(power)
            else:
                rhoFWER.append(fwer)
                rhoPower.append(power)

        phiFWER_j = np.array(phiFWER) + np.random.uniform(-jitterX, jitterX, len(phiFWER))
        phiPower_j = np.array(phiPower) + np.random.uniform(-jitterY, jitterY, len(phiPower))
        rhoFWER_j = np.array(rhoFWER) + np.random.uniform(-jitterX, jitterX, len(rhoFWER))
        rhoPower_j = np.array(rhoPower) + np.random.uniform(-jitterY, jitterY, len(rhoPower))

        ax.scatter(phiFWER_j, phiPower_j, label=f'{method} (vary φ)',
                  color=colors[method], marker='o', s=100,
                  alpha=0.7, edgecolors='black', linewidth=1.5)

        ax.scatter(rhoFWER_j, rhoPower_j, label=f'{method} (vary ρ)',
                  color=colors[method], marker='s', s=100,
                  alpha=0.7, edgecolors='black', linewidth=1.5)

    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2,
              label='Target FWER = 5%')
    ax.axvspan(0, 0.05, alpha=0.15, color='green', label='Acceptable FWER region')

    ax.set_xlabel('FWER (Family-Wise Error Rate)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (True Positive Rate)', fontsize=12, fontweight='bold')
    ax.set_title('Error Control vs Power Trade-off\n(circles=vary φ, squares=vary ρ)',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.show()

# Plot functions from bootstrap.py
def plotCalibrationCurves(calibrationResults, savePath=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    alphaLevels = sorted(calibrationResults.keys())

    methods = ['Bonferroni', 'Holm', 'BH', 'Bootstrap-Single', 'Bootstrap-RomanoWolf']
    colors = {
        'Bonferroni': 'blue',
        'Holm': 'green',
        'BH': 'orange',
        'Bootstrap-Single': 'purple',
        'Bootstrap-RomanoWolf': 'red'
    }

    # fwer calibration
    for method in methods:
        fwers = [calibrationResults[alpha][method]['fwer_mean'] for alpha in alphaLevels]
        fwer_ses = [calibrationResults[alpha][method]['fwer_se'] for alpha in alphaLevels]

        ax1.plot(alphaLevels, fwers, marker='o', label=method,
                color=colors[method], linewidth=2, markersize=8)
        ax1.errorbar(alphaLevels, fwers, yerr=fwer_ses, fmt='none',
                    color=colors[method], alpha=0.3, capsize=4)

    # perfect calibration line
    ax1.plot([0, max(alphaLevels)], [0, max(alphaLevels)],
            'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)

    ax1.set_xlabel('Nominal FWER (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Realized FWER', fontsize=12, fontweight='bold')
    ax1.set_title('FWER Calibration Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, max(alphaLevels) * 1.1])
    ax1.set_ylim([0, 1.05])

    # fdr calibration
    for method in methods:
        fdrs = [calibrationResults[alpha][method]['fdr_mean'] for alpha in alphaLevels]
        fdr_ses = [calibrationResults[alpha][method]['fdr_se'] for alpha in alphaLevels]

        ax2.plot(alphaLevels, fdrs, marker='o', label=method,
                color=colors[method], linewidth=2, markersize=8)
        ax2.errorbar(alphaLevels, fdrs, yerr=fdr_ses, fmt='none',
                    color=colors[method], alpha=0.3, capsize=4)

    # perfect calibration line
    ax2.plot([0, max(alphaLevels)], [0, max(alphaLevels)],
            'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)

    ax2.set_xlabel('Nominal FDR (α)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Realized FDR', fontsize=12, fontweight='bold')
    ax2.set_title('FDR Calibration Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, max(alphaLevels) * 1.1])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()

    if savePath:
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savePath}")

    plt.show()