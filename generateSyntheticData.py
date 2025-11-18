import numpy as np
import matplotlib.pyplot as plt
from constants import (
    TIME, PERIOD, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH,
    BASEPHI, BASERHO
)

np.random.seed(73)

# generate one independent AR1 series
def generateTimeDependentSeries(time, phi):
    x = np.zeros(time)
    x[0] = np.random.randn()

    for t in range(1, time):
        x[t] = phi * x[t-1] + np.random.randn()

    return x

# multiple industries
def generateClusteredPanelWithTimeDependence(time, numberClusters, firmsPerCluster, phi, rho):
    totalFirms = numberClusters * firmsPerCluster
    data = np.zeros((time, totalFirms))

    clusterFactors = []
    for _ in range(numberClusters):
        factor = generateTimeDependentSeries(time, phi)
        clusterFactors.append(factor)

    clusterLabels = []
    firmIdx = 0

    for clusterIdx in range(numberClusters):
        for _ in range(firmsPerCluster):
            epsilon = generateTimeDependentSeries(time, phi)
            data[:, firmIdx] = (np.sqrt(rho) * clusterFactors[clusterIdx] + np.sqrt(1 - rho) * epsilon)
            clusterLabels.append(clusterIdx)
            firmIdx += 1

    clusterLabels = np.array(clusterLabels)

    return data, clusterLabels

def generateClusteredPanelWithPlantedSignals(time, numberClusters, firmsPerCluster, numberTrue, strength, phi, rho):
    if numberTrue >= firmsPerCluster:
        raise ValueError(f"numberTrue ({numberTrue}) must be less than firmsPerCluster ({firmsPerCluster})")

    data, clusterLabels = generateClusteredPanelWithTimeDependence(time, numberClusters, firmsPerCluster, phi, rho)

    totalFirms = numberClusters * firmsPerCluster
    isTrue = np.zeros(totalFirms, dtype=bool)

    for clusterIdx in range(numberClusters):
        clusterFirmIndices = np.where(clusterLabels == clusterIdx)[0]

        for i in range(numberTrue):
            firmIdx = clusterFirmIndices[i]
            data[:, firmIdx] += strength
            isTrue[firmIdx] = True

    return data, clusterLabels, isTrue

def generateClusteredDatasets():
    phiLevels = [0.0, 0.3, 0.6, 0.9]
    rhoLevels = [0.0, 0.3, 0.6, 0.9]

    datasets = []

    phiRow = []
    for phi in phiLevels:
        data, clusterLabels, isTrue = generateClusteredPanelWithPlantedSignals(
            TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH, phi, BASERHO
        )
        phiRow.append((data, clusterLabels, isTrue))
    datasets.append(phiRow)

    rhoRow = []
    for rho in rhoLevels:
        data, clusterLabels, isTrue = generateClusteredPanelWithPlantedSignals(
            TIME, NUMBERCLUSTERS, FIRMSPERCLUSTER, NUMBERTRUE, STRENGTH, BASEPHI, rho
        )
        rhoRow.append((data, clusterLabels, isTrue))
    datasets.append(rhoRow)

    return datasets
